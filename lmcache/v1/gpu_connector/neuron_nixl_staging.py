# SPDX-License-Identifier: Apache-2.0
"""Neuron NIXL staging: pre-registered per-layer async transfers.

Registers block descriptors per-layer at init (like NeuronNixlConnector),
uses make_prepped_xfer + async transfer per layer, polls all in parallel.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Sequence
from typing import Any, Optional

import torch

from lmcache import device_ops
from lmcache.logging import init_logger
import lmcache.lmcache_native as lmcache_native

logger = init_logger(__name__)


def _parse_backends(backends: Optional[Sequence[str] | str]) -> list[str]:
    import os

    if backends is None:
        env = os.environ.get("LMCACHE_NEURON_NIXL_BACKENDS", "")
        if env:
            return [item.strip() for item in env.split(",") if item.strip()]
        return ["LIBFABRIC"]
    if isinstance(backends, str):
        return [item.strip() for item in backends.split(",") if item.strip()]
    return [str(item).strip() for item in backends if str(item).strip()]


def _view_nbytes(tensor: torch.Tensor) -> int:
    if tensor.numel() == 0:
        return 0
    max_offset = 0
    for dim, stride in zip(tensor.shape, tensor.stride(), strict=True):
        max_offset += (int(dim) - 1) * int(stride)
    return (max_offset + 1) * tensor.element_size()


def _tensor_device_id(tensor: torch.Tensor) -> int:
    device = tensor.device
    if device.type == "cpu":
        return 0
    idx = getattr(device, "index", None)
    if idx is not None:
        return int(idx)
    return 0


def _slice_addr(tensor: torch.Tensor, indices: tuple[int, ...]) -> int:
    offset_elems = 0
    for idx, stride in zip(indices, tensor.stride(), strict=True):
        offset_elems += int(idx) * int(stride)
    return int(tensor.data_ptr()) + offset_elems * tensor.element_size()


class _LayerHandles:
    """Pre-registered NIXL handles for one layer."""

    __slots__ = ("dev_handle", "cpu_handle", "blocks_per_layer", "kv_split")

    def __init__(self, dev_handle, cpu_handle, blocks_per_layer, kv_split):
        self.dev_handle = dev_handle
        self.cpu_handle = cpu_handle
        self.blocks_per_layer = blocks_per_layer
        self.kv_split = kv_split


class NeuronNixlBlockStager:
    """Stage KV blocks between Neuron HBM and CPU via NIXL."""

    def __init__(self, backends: Optional[Sequence[str] | str] = None):
        self.backends = _parse_backends(backends)
        self._wrapper: Any = None
        self._cpu_wrapper: Any = None
        self._remote_name: str | None = None
        self._layer_handles: list[_LayerHandles] = []
        self._registered = False
        self._block_byte_size: int = 0
        self._kv_split: int = 1
        self._cpu_buffer: torch.Tensor | None = None

    def _create_wrapper(self):
        from vllm.distributed.nixl_utils import (
            NixlWrapper,
            nixl_agent_config,
        )

        return NixlWrapper(str(uuid.uuid4()), nixl_agent_config(
            backends=self.backends
        ))

    def _register_kv_caches(
        self,
        layer_tensors: list[torch.Tensor],
        engine_kv_format: lmcache_native.EngineKVFormat,
        block_size: int,
    ) -> None:
        if self._registered:
            return

        self._wrapper = self._create_wrapper()
        self._cpu_wrapper = self._create_wrapper()

        fmt = int(engine_kv_format)
        sample = layer_tensors[0]
        elt_size = sample.element_size()
        num_layers = len(layer_tensors)

        # Determine block geometry
        if fmt == int(lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS):
            num_blocks = int(sample.shape[1])
            nh, hs = int(sample.shape[2]), int(sample.shape[4])
            self._block_byte_size = nh * block_size * hs * elt_size
            self._kv_split = 2
        elif fmt == int(lmcache_native.EngineKVFormat.NL_X_NB_TWO_NH_BS_HS):
            num_blocks = int(sample.shape[0])
            nh, hs = int(sample.shape[2]), int(sample.shape[4])
            self._block_byte_size = 2 * nh * block_size * hs * elt_size
            self._kv_split = 1
        elif fmt == int(lmcache_native.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS):
            num_blocks = int(sample.shape[0])
            hidden = int(sample.shape[3])
            self._block_byte_size = 2 * block_size * hidden * elt_size
            self._kv_split = 1
        else:
            num_blocks = int(sample.shape[0]) // block_size
            hidden = int(sample.shape[1])
            self._block_byte_size = block_size * hidden * elt_size
            self._kv_split = 1

        descs_per_layer = (
            num_blocks * self._kv_split if self._kv_split == 2
            else num_blocks
        )

        # Allocate CPU staging buffer for all layers
        total_cpu_descs = num_layers * descs_per_layer
        total_cpu_bytes = total_cpu_descs * self._block_byte_size
        self._cpu_buffer = torch.empty(
            total_cpu_bytes // elt_size,
            dtype=sample.dtype,
            device="cpu",
        )

        # Register CPU buffer
        cpu_reg = self._cpu_wrapper.get_reg_descs(
            [(int(self._cpu_buffer.data_ptr()), total_cpu_bytes, 0, "")],
            "DRAM",
        )
        self._cpu_wrapper.register_memory(cpu_reg, backends=self.backends)

        # Connect wrappers
        self._remote_name = self._wrapper.add_remote_agent(
            self._cpu_wrapper.get_agent_metadata()
        )

        # Register per-layer
        cpu_offset_bytes = 0
        cpu_base = int(self._cpu_buffer.data_ptr())

        for li, layer_tensor in enumerate(layer_tensors):
            # Register device memory for this layer
            dev_reg = self._wrapper.get_reg_descs(
                [(
                    int(layer_tensor.data_ptr()),
                    _view_nbytes(layer_tensor),
                    _tensor_device_id(layer_tensor),
                    "",
                )],
                "VRAM",
            )
            self._wrapper.register_memory(dev_reg, backends=self.backends)

            # Build device block descriptors
            dev_data = self._block_descs_for_layer(
                layer_tensor, engine_kv_format, block_size
            )
            dev_descs = self._wrapper.get_xfer_descs(dev_data, "VRAM")
            dev_handle = self._wrapper.prep_xfer_dlist(
                "NIXL_INIT_AGENT", dev_descs
            )

            # Build CPU block descriptors for this layer's slice
            cpu_data = []
            for j in range(descs_per_layer):
                addr = cpu_base + cpu_offset_bytes
                cpu_data.append((addr, self._block_byte_size, 0))
                cpu_offset_bytes += self._block_byte_size

            cpu_descs = self._wrapper.get_xfer_descs(cpu_data, "DRAM")
            cpu_handle = self._wrapper.prep_xfer_dlist(
                self._remote_name, cpu_descs
            )

            self._layer_handles.append(_LayerHandles(
                dev_handle=dev_handle,
                cpu_handle=cpu_handle,
                blocks_per_layer=descs_per_layer,
                kv_split=self._kv_split,
            ))

        self._registered = True
        logger.info(
            "NIXL staging registered: %d layers, %d descs/layer, "
            "%d bytes/block",
            num_layers, descs_per_layer, self._block_byte_size,
        )

    def _block_descs_for_layer(
        self,
        tensor: torch.Tensor,
        engine_kv_format: lmcache_native.EngineKVFormat,
        block_size: int,
    ) -> list[tuple[int, int, int]]:
        fmt = int(engine_kv_format)
        elt_size = tensor.element_size()
        did = _tensor_device_id(tensor)
        data: list[tuple[int, int, int]] = []

        if fmt == int(lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS):
            nb = int(tensor.shape[1])
            nh, hs = int(tensor.shape[2]), int(tensor.shape[4])
            bsz = nh * block_size * hs * elt_size
            for bid in range(nb):
                for kv in range(2):
                    data.append((
                        _slice_addr(tensor, (kv, bid, 0, 0, 0)), bsz, did
                    ))
        elif fmt == int(lmcache_native.EngineKVFormat.NL_X_NB_TWO_NH_BS_HS):
            nb = int(tensor.shape[0])
            nh, hs = int(tensor.shape[2]), int(tensor.shape[4])
            bsz = 2 * nh * block_size * hs * elt_size
            for bid in range(nb):
                data.append((
                    _slice_addr(tensor, (bid, 0, 0, 0, 0)), bsz, did
                ))
        elif fmt == int(lmcache_native.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS):
            nb = int(tensor.shape[0])
            hidden = int(tensor.shape[3])
            bsz = 2 * block_size * hidden * elt_size
            for bid in range(nb):
                data.append((
                    _slice_addr(tensor, (bid, 0, 0, 0)), bsz, did
                ))
        else:
            total = int(tensor.shape[0])
            nb = total // block_size
            hidden = int(tensor.shape[1])
            bsz = block_size * hidden * elt_size
            for bid in range(nb):
                data.append((
                    _slice_addr(tensor, (bid * block_size, 0)), bsz, did
                ))

        return data

    def _do_transfers(
        self,
        op: str,
        selected_blocks: list[int],
        num_layers: int,
    ) -> None:
        """Issue per-layer async transfers and poll all."""
        handles = []
        for li in range(num_layers):
            lh = self._layer_handles[li]
            dev_ids = []
            cpu_ids = []
            for bid in selected_blocks:
                if lh.kv_split == 2:
                    dev_ids.extend([bid * 2, bid * 2 + 1])
                    cpu_ids.extend([bid * 2, bid * 2 + 1])
                else:
                    dev_ids.append(bid)
                    cpu_ids.append(bid)

            if op == "READ":
                xh = self._wrapper.make_prepped_xfer(
                    "READ", lh.cpu_handle, cpu_ids,
                    lh.dev_handle, dev_ids, b"d2h",
                )
            else:
                xh = self._wrapper.make_prepped_xfer(
                    "WRITE", lh.cpu_handle, cpu_ids,
                    lh.dev_handle, dev_ids, b"h2d",
                )
            self._wrapper.transfer(xh)
            handles.append(xh)

        # Poll all
        deadline = time.monotonic() + 60.0
        remaining = set(range(len(handles)))
        while remaining:
            if time.monotonic() > deadline:
                for idx in remaining:
                    self._wrapper.release_xfer_handle(handles[idx])
                raise RuntimeError(
                    f"NIXL {op} timed out: {len(remaining)} layers pending"
                )
            done = set()
            for idx in remaining:
                state = self._wrapper.check_xfer_state(handles[idx])
                if state == "DONE":
                    done.add(idx)
                elif state == "ERR":
                    for j in remaining:
                        self._wrapper.release_xfer_handle(handles[j])
                    raise RuntimeError(f"NIXL {op} layer {idx} failed")
            remaining -= done

        for xh in handles:
            self._wrapper.release_xfer_handle(xh)

    def transfer_into_key_value(
        self,
        key_value: torch.Tensor,
        layer_tensors: list[torch.Tensor],
        slot_mapping: torch.Tensor,
        engine_kv_format: lmcache_native.EngineKVFormat,
        block_size: int,
        head_size: int,
    ) -> None:
        """D2H: Neuron paged KV -> CPU key_value buffer."""
        if key_value.device.type != "cpu":
            raise ValueError("Requires CPU destination")
        if not layer_tensors or slot_mapping.numel() == 0:
            return

        self._register_kv_caches(
            layer_tensors, engine_kv_format, block_size
        )

        selected_blocks, compact_slot_mapping = self._compact_slot_mapping(
            slot_mapping, block_size
        )
        if not selected_blocks:
            return

        t0 = time.perf_counter()
        self._do_transfers("READ", selected_blocks, len(layer_tensors))
        d2h_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "[PERF] D2H prepped xfer: %.2fms, %d blocks, %d layers",
            d2h_ms, len(selected_blocks), len(layer_tensors),
        )

        staged_layers = self._cpu_staged_views(
            layer_tensors, len(selected_blocks),
            engine_kv_format, block_size,
        )

        device_ops.multi_layer_kv_transfer(
            key_value, staged_layers, compact_slot_mapping,
            torch.device("cpu"),
            len(selected_blocks) * block_size,
            lmcache_native.TransferDirection.D2H,
            engine_kv_format,
            block_size=block_size, head_size=head_size,
        )

    def transfer_from_key_value(
        self,
        key_value: torch.Tensor,
        layer_tensors: list[torch.Tensor],
        slot_mapping: torch.Tensor,
        engine_kv_format: lmcache_native.EngineKVFormat,
        block_size: int,
        head_size: int,
        skip_prefix_n_tokens: int = 0,
    ) -> None:
        """H2D: CPU key_value buffer -> Neuron paged KV."""
        if key_value.device.type != "cpu":
            raise ValueError("Requires CPU source")
        if not layer_tensors or slot_mapping.numel() == 0:
            return

        self._register_kv_caches(
            layer_tensors, engine_kv_format, block_size
        )

        selected_blocks, compact_slot_mapping = self._compact_slot_mapping(
            slot_mapping, block_size
        )
        if not selected_blocks:
            return

        staged_layers = self._cpu_staged_views(
            layer_tensors, len(selected_blocks),
            engine_kv_format, block_size,
        )

        device_ops.multi_layer_kv_transfer(
            key_value, staged_layers, compact_slot_mapping,
            torch.device("cpu"),
            len(selected_blocks) * block_size,
            lmcache_native.TransferDirection.H2D,
            engine_kv_format,
            block_size=block_size, head_size=head_size,
            skip_prefix_n_tokens=skip_prefix_n_tokens,
        )

        t0 = time.perf_counter()
        self._do_transfers("WRITE", selected_blocks, len(layer_tensors))
        h2d_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "[PERF] H2D prepped xfer: %.2fms, %d blocks, %d layers",
            h2d_ms, len(selected_blocks), len(layer_tensors),
        )

    def _cpu_staged_views(
        self,
        layer_tensors: list[torch.Tensor],
        compact_num_blocks: int,
        engine_kv_format: lmcache_native.EngineKVFormat,
        block_size: int,
    ) -> list[torch.Tensor]:
        """Views into pre-allocated CPU buffer as compact paged tensors."""
        assert self._cpu_buffer is not None
        views = []
        offset = 0
        for lt in layer_tensors:
            shape = list(lt.shape)
            fmt = int(engine_kv_format)
            if fmt == int(lmcache_native.EngineKVFormat.NL_X_NB_BS_HS):
                shape[0] = compact_num_blocks * block_size
            elif fmt in (
                int(lmcache_native.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS),
                int(lmcache_native.EngineKVFormat.NL_X_NB_TWO_NH_BS_HS),
            ):
                shape[0] = compact_num_blocks
            elif fmt == int(
                lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS
            ):
                shape[1] = compact_num_blocks
            else:
                shape[1] = compact_num_blocks * block_size
            numel = 1
            for s in shape:
                numel *= s
            views.append(self._cpu_buffer[offset:offset + numel].view(shape))
            offset += numel
        return views

    def _compact_slot_mapping(
        self, slot_mapping: torch.Tensor, block_size: int
    ) -> tuple[list[int], torch.Tensor]:
        slots_cpu = slot_mapping.to(dtype=torch.long, device="cpu")
        valid = [int(v) for v in slots_cpu.tolist() if int(v) >= 0]
        if not valid:
            return [], slots_cpu
        selected = sorted({s // block_size for s in valid})
        bmap = {bid: i for i, bid in enumerate(selected)}
        compact = slots_cpu.clone()
        for idx, slot in enumerate(compact.tolist()):
            if slot < 0:
                continue
            compact[idx] = (
                bmap[int(slot) // block_size] * block_size
                + int(slot) % block_size
            )
        return selected, compact
