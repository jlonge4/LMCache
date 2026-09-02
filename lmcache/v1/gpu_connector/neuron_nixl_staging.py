# SPDX-License-Identifier: Apache-2.0
"""Neuron NIXL staging: per-request descriptors with two-agent transfers.

Registers whole tensor memory regions once at init. Builds small descriptor
lists per-request for only the selected blocks. Uses two NIXL agents
(device + CPU) with add_remote_agent for cross-agent DMA.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import torch

from lmcache import device_ops
from lmcache.logging import init_logger
import lmcache.lmcache_native as lmcache_native

logger = init_logger(__name__)


def _parse_backends(backends=None):
    import os

    if backends is None:
        env = os.environ.get("LMCACHE_NEURON_NIXL_BACKENDS", "")
        if env:
            return [x.strip() for x in env.split(",") if x.strip()]
        return ["LIBFABRIC"]
    if isinstance(backends, str):
        return [x.strip() for x in backends.split(",") if x.strip()]
    return [str(x).strip() for x in backends if str(x).strip()]


def _view_nbytes(t):
    if t.numel() == 0:
        return 0
    mo = sum((int(d) - 1) * int(s) for d, s in zip(t.shape, t.stride(), strict=False))
    return (mo + 1) * t.element_size()


def _tensor_device_id(t):
    if t.device.type == "cpu":
        return 0
    idx = getattr(t.device, "index", None)
    return int(idx) if idx is not None else 0


def _slice_addr(t, indices):
    off = sum(int(i) * int(s) for i, s in zip(indices, t.stride(), strict=False))
    return int(t.data_ptr()) + off * t.element_size()


class NeuronNixlBlockStager:
    """Stage KV blocks between Neuron HBM and CPU via NIXL."""

    def __init__(self, backends=None):
        self.backends = _parse_backends(backends)
        self._dev_wrapper: Any = None
        self._cpu_wrapper: Any = None
        self._remote_name: str | None = None
        self._mem_registered = False
        self._block_byte_size = 0
        self._kv_split = 1
        self._cpu_buffer: torch.Tensor | None = None

    def _create_wrapper(self):
        from vllm.distributed.nixl_utils import (
            NixlWrapper,
            nixl_agent_config,
        )

        return NixlWrapper(
            str(uuid.uuid4()),
            nixl_agent_config(backends=self.backends),
        )

    def _register_memory(self, layer_tensors, engine_kv_format, block_size):
        """One-time: register whole tensors as memory regions, connect agents."""
        if self._mem_registered:
            return

        self._dev_wrapper = self._create_wrapper()
        self._cpu_wrapper = self._create_wrapper()

        fmt = int(engine_kv_format)
        sample = layer_tensors[0]
        elt_size = sample.element_size()

        if fmt == int(lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS):
            nh, hs = int(sample.shape[2]), int(sample.shape[4])
            self._block_byte_size = nh * block_size * hs * elt_size
            self._kv_split = 2
        elif fmt == int(
            lmcache_native.EngineKVFormat.NL_X_NB_TWO_NH_BS_HS
        ):
            nh, hs = int(sample.shape[2]), int(sample.shape[4])
            self._block_byte_size = 2 * nh * block_size * hs * elt_size
            self._kv_split = 1
        elif fmt == int(
            lmcache_native.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS
        ):
            hidden = int(sample.shape[3])
            self._block_byte_size = 2 * block_size * hidden * elt_size
            self._kv_split = 1
        else:
            hidden = int(sample.shape[-1])
            self._block_byte_size = block_size * hidden * elt_size
            self._kv_split = 1

        # Register each layer tensor on dev_wrapper
        for lt in layer_tensors:
            dev_reg = self._dev_wrapper.get_reg_descs(
                [(
                    int(lt.data_ptr()),
                    _view_nbytes(lt),
                    _tensor_device_id(lt),
                    "",
                )],
                "VRAM",
            )
            self._dev_wrapper.register_memory(
                dev_reg, backends=self.backends
            )

        # CPU buffer (sized for max 1024 blocks * num_layers)
        max_blocks = 1024
        descs_per_block = 2 if self._kv_split == 2 else 1
        buf_bytes = (
            len(layer_tensors)
            * max_blocks
            * descs_per_block
            * self._block_byte_size
        )
        self._cpu_buffer = torch.empty(
            buf_bytes // elt_size, dtype=sample.dtype, device="cpu"
        )

        cpu_reg = self._cpu_wrapper.get_reg_descs(
            [(int(self._cpu_buffer.data_ptr()), buf_bytes, 0, "")],
            "DRAM",
        )
        self._cpu_wrapper.register_memory(
            cpu_reg, backends=self.backends
        )

        # Cross-agent link: dev reads from/writes to cpu
        self._remote_name = self._dev_wrapper.add_remote_agent(
            self._cpu_wrapper.get_agent_metadata()
        )

        self._mem_registered = True
        logger.info(
            "NIXL memory registered: %d layers, cpu_buf=%d MB",
            len(layer_tensors),
            buf_bytes // (1024 * 1024),
        )

    def _do_xfer(self, op, layer_tensors, selected_blocks, fmt, bs):
        """Per-request: build descriptors for selected blocks, transfer."""
        elt_size = layer_tensors[0].element_size()
        cpu_base = int(self._cpu_buffer.data_ptr())
        cpu_off = 0

        handles = []
        for lt in layer_tensors:
            dev_data = []
            cpu_data = []
            did = _tensor_device_id(lt)
            f = int(fmt)

            for bid in selected_blocks:
                if f == int(
                    lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS
                ):
                    for kv in range(2):
                        addr = _slice_addr(lt, (kv, bid, 0, 0, 0))
                        dev_data.append(
                            (addr, self._block_byte_size, did)
                        )
                        cpu_data.append(
                            (cpu_base + cpu_off, self._block_byte_size, 0)
                        )
                        cpu_off += self._block_byte_size
                elif f == int(
                    lmcache_native.EngineKVFormat.NL_X_NB_TWO_NH_BS_HS
                ):
                    addr = _slice_addr(lt, (bid, 0, 0, 0, 0))
                    dev_data.append(
                        (addr, self._block_byte_size, did)
                    )
                    cpu_data.append(
                        (cpu_base + cpu_off, self._block_byte_size, 0)
                    )
                    cpu_off += self._block_byte_size
                elif f == int(
                    lmcache_native.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS
                ):
                    addr = _slice_addr(lt, (bid, 0, 0, 0))
                    dev_data.append(
                        (addr, self._block_byte_size, did)
                    )
                    cpu_data.append(
                        (cpu_base + cpu_off, self._block_byte_size, 0)
                    )
                    cpu_off += self._block_byte_size
                else:
                    addr = _slice_addr(lt, (bid * bs, 0))
                    dev_data.append(
                        (addr, self._block_byte_size, did)
                    )
                    cpu_data.append(
                        (cpu_base + cpu_off, self._block_byte_size, 0)
                    )
                    cpu_off += self._block_byte_size

            # Build per-layer descriptor lists
            dev_descs = self._dev_wrapper.get_xfer_descs(
                dev_data, "VRAM"
            )
            cpu_descs = self._dev_wrapper.get_xfer_descs(
                cpu_data, "DRAM"
            )

            dev_h = self._dev_wrapper.prep_xfer_dlist(
                "NIXL_INIT_AGENT", dev_descs
            )
            cpu_h = self._dev_wrapper.prep_xfer_dlist(
                self._remote_name, cpu_descs
            )

            ids = list(range(len(dev_data)))
            xh = self._dev_wrapper.make_prepped_xfer(
                op, cpu_h, ids, dev_h, ids, b"lmc",
            )
            self._dev_wrapper.transfer(xh)
            handles.append((xh, dev_h, cpu_h))

        # Poll all layers
        deadline = time.monotonic() + 60.0
        remaining = set(range(len(handles)))
        while remaining:
            if time.monotonic() > deadline:
                for i in remaining:
                    self._dev_wrapper.release_xfer_handle(handles[i][0])
                raise RuntimeError(f"NIXL {op} timed out")
            done = set()
            for i in remaining:
                st = self._dev_wrapper.check_xfer_state(handles[i][0])
                if st == "DONE":
                    done.add(i)
                elif st == "ERR":
                    raise RuntimeError(f"NIXL {op} layer {i} failed")
            remaining -= done

        for xh, dh, ch in handles:
            self._dev_wrapper.release_xfer_handle(xh)
            self._dev_wrapper.release_dlist_handle(dh)
            self._dev_wrapper.release_dlist_handle(ch)

    def transfer_into_key_value(
        self, key_value, layer_tensors, slot_mapping,
        engine_kv_format, block_size, head_size,
    ):
        """D2H: Neuron paged KV -> CPU key_value buffer."""
        if not layer_tensors or slot_mapping.numel() == 0:
            return
        self._register_memory(
            layer_tensors, engine_kv_format, block_size
        )
        selected, compact_sm = self._compact_slot_mapping(
            slot_mapping, block_size
        )
        if not selected:
            return

        t0 = time.perf_counter()
        self._do_xfer(
            "READ", layer_tensors, selected, engine_kv_format, block_size
        )
        logger.info(
            "[PERF] D2H: %.2fms, %d blocks, %d layers",
            (time.perf_counter() - t0) * 1000,
            len(selected),
            len(layer_tensors),
        )

        staged = self._cpu_staged_views(
            layer_tensors, len(selected), engine_kv_format, block_size
        )
        device_ops.multi_layer_kv_transfer(
            key_value, staged, compact_sm, torch.device("cpu"),
            len(selected) * block_size,
            lmcache_native.TransferDirection.D2H,
            engine_kv_format,
            block_size=block_size, head_size=head_size,
        )

    def transfer_from_key_value(
        self, key_value, layer_tensors, slot_mapping,
        engine_kv_format, block_size, head_size,
        skip_prefix_n_tokens=0,
    ):
        """H2D: CPU key_value buffer -> Neuron paged KV."""
        if not layer_tensors or slot_mapping.numel() == 0:
            return
        self._register_memory(
            layer_tensors, engine_kv_format, block_size
        )
        selected, compact_sm = self._compact_slot_mapping(
            slot_mapping, block_size
        )
        if not selected:
            return

        staged = self._cpu_staged_views(
            layer_tensors, len(selected), engine_kv_format, block_size
        )
        device_ops.multi_layer_kv_transfer(
            key_value, staged, compact_sm, torch.device("cpu"),
            len(selected) * block_size,
            lmcache_native.TransferDirection.H2D,
            engine_kv_format,
            block_size=block_size, head_size=head_size,
            skip_prefix_n_tokens=skip_prefix_n_tokens,
        )

        t0 = time.perf_counter()
        self._do_xfer(
            "WRITE", layer_tensors, selected,
            engine_kv_format, block_size,
        )
        logger.info(
            "[PERF] H2D: %.2fms, %d blocks, %d layers",
            (time.perf_counter() - t0) * 1000,
            len(selected),
            len(layer_tensors),
        )

    def _cpu_staged_views(self, layer_tensors, n, fmt, bs):
        """Views into pre-allocated CPU buffer."""
        assert self._cpu_buffer is not None
        views = []
        off = 0
        for lt in layer_tensors:
            shape = list(lt.shape)
            f = int(fmt)
            if f == int(lmcache_native.EngineKVFormat.NL_X_NB_BS_HS):
                shape[0] = n * bs
            elif f in (
                int(lmcache_native.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS),
                int(lmcache_native.EngineKVFormat.NL_X_NB_TWO_NH_BS_HS),
            ):
                shape[0] = n
            elif f == int(
                lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS
            ):
                shape[1] = n
            else:
                shape[1] = n * bs
            numel = 1
            for s in shape:
                numel *= s
            views.append(self._cpu_buffer[off:off + numel].view(shape))
            off += numel
        return views

    def _compact_slot_mapping(self, slot_mapping, block_size):
        sc = slot_mapping.to(dtype=torch.long, device="cpu")
        valid = [int(v) for v in sc.tolist() if int(v) >= 0]
        if not valid:
            return [], sc
        selected = sorted({s // block_size for s in valid})
        bmap = {b: i for i, b in enumerate(selected)}
        compact = sc.clone()
        for idx, slot in enumerate(compact.tolist()):
            if slot < 0:
                continue
            compact[idx] = (
                bmap[int(slot) // block_size] * block_size
                + int(slot) % block_size
            )
        return selected, compact
