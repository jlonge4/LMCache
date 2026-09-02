# SPDX-License-Identifier: Apache-2.0
"""Neuron-specific NIXL staging for LMCache GPU connectors.

Pre-registers all KV cache block descriptors at init time (like
NeuronNixlConnector does for P->D transfers) and uses
make_prepped_xfer + async transfer for D2H and H2D copies.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
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
    get_device = getattr(tensor, "get_device", None)
    if callable(get_device):
        try:
            return max(int(get_device()), 0)
        except Exception:
            return 0
    return 0


def _slice_addr(
    tensor: torch.Tensor,
    indices: tuple[int, ...],
) -> int:
    offset_elems = 0
    for idx, stride in zip(indices, tensor.stride(), strict=True):
        offset_elems += int(idx) * int(stride)
    return int(tensor.data_ptr()) + offset_elems * tensor.element_size()


@dataclass(frozen=True)
class _BlockDesc:
    addr: int
    size_bytes: int
    device_id: int


class NeuronNixlBlockStager:
    """Stage KV blocks between Neuron HBM and CPU via NIXL."""

    def __init__(self, backends: Optional[Sequence[str] | str] = None):
        self.backends = _parse_backends(backends)
        self._dev_wrapper: Any = None
        self._cpu_wrapper: Any = None
        self._remote_agent_name: str | None = None
        self._registered = False
        self._dev_handle: Any = None
        self._cpu_handle: Any = None
        self._blocks_per_layer: int = 0
        self._block_byte_size: int = 0
        self._num_layers: int = 0
        self._kv_split: int = 1

    def _create_wrapper(self):
        from vllm.distributed.nixl_utils import NixlWrapper, nixl_agent_config

        config = nixl_agent_config(backends=self.backends)
        return NixlWrapper(str(uuid.uuid4()), config)

    def _register_kv_caches(
        self,
        layer_tensors: list[torch.Tensor],
        engine_kv_format: lmcache_native.EngineKVFormat,
        block_size: int,
    ) -> None:
        """One-time registration of all KV cache blocks as NIXL descriptors."""
        if self._registered:
            return

        self._dev_wrapper = self._create_wrapper()
        self._cpu_wrapper = self._create_wrapper()

        fmt = int(engine_kv_format)
        sample = layer_tensors[0]
        elt_size = sample.element_size()
        self._num_layers = len(layer_tensors)

        if fmt == int(lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS):
            num_blocks = int(sample.shape[1])
            num_heads = int(sample.shape[2])
            head_size = int(sample.shape[4])
            self._block_byte_size = num_heads * block_size * head_size * elt_size
            self._kv_split = 2
            self._blocks_per_layer = num_blocks * 2
        elif fmt == int(lmcache_native.EngineKVFormat.NL_X_NB_TWO_NH_BS_HS):
            num_blocks = int(sample.shape[0])
            num_heads = int(sample.shape[2])
            head_size = int(sample.shape[4])
            self._block_byte_size = 2 * num_heads * block_size * head_size * elt_size
            self._kv_split = 1
            self._blocks_per_layer = num_blocks
        elif fmt == int(lmcache_native.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS):
            num_blocks = int(sample.shape[0])
            hidden = int(sample.shape[3])
            self._block_byte_size = 2 * block_size * hidden * elt_size
            self._kv_split = 1
            self._blocks_per_layer = num_blocks
        elif fmt == int(lmcache_native.EngineKVFormat.NL_X_NB_BS_HS):
            total_slots = int(sample.shape[0])
            num_blocks = total_slots // block_size
            hidden = int(sample.shape[1])
            self._block_byte_size = block_size * hidden * elt_size
            self._kv_split = 1
            self._blocks_per_layer = num_blocks
        else:
            total_slots = int(sample.shape[1])
            num_blocks = total_slots // block_size
            hidden = int(sample.shape[2])
            self._block_byte_size = block_size * hidden * elt_size
            self._kv_split = 2
            self._blocks_per_layer = num_blocks * 2

        dev_blocks_data: list[tuple[int, int, int]] = []
        for layer_tensor in layer_tensors:
            descs = self._all_block_descs(
                layer_tensor, engine_kv_format, block_size
            )
            for d in descs:
                dev_blocks_data.append((d.addr, d.size_bytes, d.device_id))

        dev_reg_tuples = [
            (int(t.data_ptr()), _view_nbytes(t), _tensor_device_id(t), "")
            for t in layer_tensors
        ]
        dev_reg_descs = self._dev_wrapper.get_reg_descs(dev_reg_tuples, "VRAM")
        self._dev_wrapper.register_memory(dev_reg_descs, backends=self.backends)

        dev_xfer_descs = self._dev_wrapper.get_xfer_descs(dev_blocks_data, "VRAM")
        self._dev_handle = self._dev_wrapper.prep_xfer_dlist(
            "NIXL_INIT_AGENT", dev_xfer_descs
        )

        total_cpu_bytes = (
            self._num_layers * self._blocks_per_layer * self._block_byte_size
        )
        self._cpu_buffer = torch.empty(
            total_cpu_bytes // elt_size, dtype=sample.dtype, device="cpu"
        )

        cpu_reg_tuples = [
            (int(self._cpu_buffer.data_ptr()), total_cpu_bytes, 0, "")
        ]
        cpu_reg_descs = self._cpu_wrapper.get_reg_descs(cpu_reg_tuples, "DRAM")
        self._cpu_wrapper.register_memory(cpu_reg_descs, backends=self.backends)

        cpu_blocks_data: list[tuple[int, int, int]] = []
        base = int(self._cpu_buffer.data_ptr())
        for i in range(self._num_layers * self._blocks_per_layer):
            cpu_blocks_data.append(
                (base + i * self._block_byte_size, self._block_byte_size, 0)
            )

        cpu_xfer_descs = self._cpu_wrapper.get_xfer_descs(cpu_blocks_data, "DRAM")

        self._remote_agent_name = self._dev_wrapper.add_remote_agent(
            self._cpu_wrapper.get_agent_metadata()
        )
        self._cpu_handle = self._dev_wrapper.prep_xfer_dlist(
            self._remote_agent_name, cpu_xfer_descs
        )

        self._registered = True
        logger.info(
            "NIXL staging registered: %d layers x %d blocks, "
            "block_size=%d bytes, total=%d descriptors",
            self._num_layers,
            self._blocks_per_layer,
            self._block_byte_size,
            len(dev_blocks_data),
        )

    def _all_block_descs(
        self,
        tensor: torch.Tensor,
        engine_kv_format: lmcache_native.EngineKVFormat,
        block_size: int,
    ) -> list[_BlockDesc]:
        fmt = int(engine_kv_format)
        elt_size = tensor.element_size()
        device_id = _tensor_device_id(tensor)
        descs: list[_BlockDesc] = []

        if fmt == int(lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS):
            num_blocks = int(tensor.shape[1])
            num_heads = int(tensor.shape[2])
            head_size = int(tensor.shape[4])
            block_bytes = num_heads * block_size * head_size * elt_size
            for block_id in range(num_blocks):
                for kv_idx in range(2):
                    descs.append(_BlockDesc(
                        addr=_slice_addr(tensor, (kv_idx, block_id, 0, 0, 0)),
                        size_bytes=block_bytes,
                        device_id=device_id,
                    ))
        elif fmt == int(lmcache_native.EngineKVFormat.NL_X_NB_TWO_NH_BS_HS):
            num_blocks = int(tensor.shape[0])
            num_heads = int(tensor.shape[2])
            head_size = int(tensor.shape[4])
            block_bytes = 2 * num_heads * block_size * head_size * elt_size
            for block_id in range(num_blocks):
                descs.append(_BlockDesc(
                    addr=_slice_addr(tensor, (block_id, 0, 0, 0, 0)),
                    size_bytes=block_bytes,
                    device_id=device_id,
                ))
        elif fmt == int(lmcache_native.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS):
            num_blocks = int(tensor.shape[0])
            hidden = int(tensor.shape[3])
            block_bytes = 2 * block_size * hidden * elt_size
            for block_id in range(num_blocks):
                descs.append(_BlockDesc(
                    addr=_slice_addr(tensor, (block_id, 0, 0, 0)),
                    size_bytes=block_bytes,
                    device_id=device_id,
                ))
        elif fmt == int(lmcache_native.EngineKVFormat.NL_X_NB_BS_HS):
            total_slots = int(tensor.shape[0])
            num_blocks = total_slots // block_size
            hidden = int(tensor.shape[1])
            block_bytes = block_size * hidden * elt_size
            for block_id in range(num_blocks):
                descs.append(_BlockDesc(
                    addr=_slice_addr(tensor, (block_id * block_size, 0)),
                    size_bytes=block_bytes,
                    device_id=device_id,
                ))
        else:
            total_slots = int(tensor.shape[1])
            num_blocks = total_slots // block_size
            hidden = int(tensor.shape[2])
            block_bytes = block_size * hidden * elt_size
            for block_id in range(num_blocks):
                for kv_idx in range(2):
                    descs.append(_BlockDesc(
                        addr=_slice_addr(
                            tensor, (kv_idx, block_id * block_size, 0)
                        ),
                        size_bytes=block_bytes,
                        device_id=device_id,
                    ))

        return descs

    def _block_indices(
        self,
        selected_blocks: list[int],
        num_layers: int,
    ) -> tuple[list[int], list[int]]:
        """Compute dev and cpu descriptor indices for selected blocks."""
        dev_ids: list[int] = []
        cpu_ids: list[int] = []
        cpu_offset = 0
        for layer_idx in range(num_layers):
            layer_base = layer_idx * self._blocks_per_layer
            for block_id in selected_blocks:
                if self._kv_split == 2:
                    dev_ids.append(layer_base + block_id * 2)
                    dev_ids.append(layer_base + block_id * 2 + 1)
                    cpu_ids.append(cpu_offset)
                    cpu_ids.append(cpu_offset + 1)
                    cpu_offset += 2
                else:
                    dev_ids.append(layer_base + block_id)
                    cpu_ids.append(cpu_offset)
                    cpu_offset += 1
        return dev_ids, cpu_ids

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
            raise ValueError("Neuron NIXL staging requires a CPU destination tensor")
        if not layer_tensors or slot_mapping.numel() == 0:
            return

        self._register_kv_caches(layer_tensors, engine_kv_format, block_size)

        selected_blocks, compact_slot_mapping = self._compact_slot_mapping(
            slot_mapping, block_size
        )
        if not selected_blocks:
            return

        dev_ids, cpu_ids = self._block_indices(selected_blocks, len(layer_tensors))

        t0 = time.perf_counter()
        xfer_handle = self._dev_wrapper.make_prepped_xfer(
            "READ",
            self._cpu_handle, cpu_ids,
            self._dev_handle, dev_ids,
            b"lmcache-d2h",
        )
        self._dev_wrapper.transfer(xfer_handle)

        deadline = time.monotonic() + 60.0
        while True:
            state = self._dev_wrapper.check_xfer_state(xfer_handle)
            if state == "DONE":
                break
            if state == "ERR":
                self._dev_wrapper.release_xfer_handle(xfer_handle)
                raise RuntimeError("NIXL D2H transfer failed")
            if time.monotonic() > deadline:
                self._dev_wrapper.release_xfer_handle(xfer_handle)
                raise RuntimeError("NIXL D2H transfer timed out")
        self._dev_wrapper.release_xfer_handle(xfer_handle)

        d2h_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "[PERF] D2H prepped xfer: %.2fms, %d blocks, %d layers",
            d2h_ms, len(selected_blocks), len(layer_tensors),
        )

        compact_num_blocks = len(selected_blocks)
        staged_layers = self._cpu_staged_views(
            layer_tensors, compact_num_blocks, engine_kv_format, block_size
        )

        compact_page_buffer_size = compact_num_blocks * block_size
        device_ops.multi_layer_kv_transfer(
            key_value,
            staged_layers,
            compact_slot_mapping,
            torch.device("cpu"),
            compact_page_buffer_size,
            lmcache_native.TransferDirection.D2H,
            engine_kv_format,
            block_size=block_size,
            head_size=head_size,
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
            raise ValueError("Neuron NIXL staging requires a CPU source tensor")
        if not layer_tensors or slot_mapping.numel() == 0:
            return

        self._register_kv_caches(layer_tensors, engine_kv_format, block_size)

        selected_blocks, compact_slot_mapping = self._compact_slot_mapping(
            slot_mapping, block_size
        )
        if not selected_blocks:
            return

        compact_num_blocks = len(selected_blocks)
        staged_layers = self._cpu_staged_views(
            layer_tensors, compact_num_blocks, engine_kv_format, block_size
        )

        compact_page_buffer_size = compact_num_blocks * block_size
        device_ops.multi_layer_kv_transfer(
            key_value,
            staged_layers,
            compact_slot_mapping,
            torch.device("cpu"),
            compact_page_buffer_size,
            lmcache_native.TransferDirection.H2D,
            engine_kv_format,
            block_size=block_size,
            head_size=head_size,
            skip_prefix_n_tokens=skip_prefix_n_tokens,
        )

        dev_ids, cpu_ids = self._block_indices(selected_blocks, len(layer_tensors))

        t0 = time.perf_counter()
        xfer_handle = self._dev_wrapper.make_prepped_xfer(
            "WRITE",
            self._cpu_handle, cpu_ids,
            self._dev_handle, dev_ids,
            b"lmcache-h2d",
        )
        self._dev_wrapper.transfer(xfer_handle)

        deadline = time.monotonic() + 60.0
        while True:
            state = self._dev_wrapper.check_xfer_state(xfer_handle)
            if state == "DONE":
                break
            if state == "ERR":
                self._dev_wrapper.release_xfer_handle(xfer_handle)
                raise RuntimeError("NIXL H2D transfer failed")
            if time.monotonic() > deadline:
                self._dev_wrapper.release_xfer_handle(xfer_handle)
                raise RuntimeError("NIXL H2D transfer timed out")
        self._dev_wrapper.release_xfer_handle(xfer_handle)

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
        views = []
        offset = 0
        for layer_tensor in layer_tensors:
            shape = list(layer_tensor.shape)
            fmt = int(engine_kv_format)
            if fmt == int(lmcache_native.EngineKVFormat.NL_X_NB_BS_HS):
                shape[0] = compact_num_blocks * block_size
            elif fmt in (
                int(lmcache_native.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS),
                int(lmcache_native.EngineKVFormat.NL_X_NB_TWO_NH_BS_HS),
            ):
                shape[0] = compact_num_blocks
            elif fmt == int(lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS):
                shape[1] = compact_num_blocks
            else:
                shape[1] = compact_num_blocks * block_size

            numel = 1
            for s in shape:
                numel *= s
            view = self._cpu_buffer[offset : offset + numel].view(shape)
            views.append(view)
            offset += numel
        return views

    def _compact_slot_mapping(
        self, slot_mapping: torch.Tensor, block_size: int
    ) -> tuple[list[int], torch.Tensor]:
        slots_cpu = slot_mapping.to(dtype=torch.long, device="cpu")
        valid_slots = [int(v) for v in slots_cpu.tolist() if int(v) >= 0]
        if not valid_slots:
            return [], slots_cpu

        selected_blocks = sorted({slot // block_size for slot in valid_slots})
        block_map = {block_id: i for i, block_id in enumerate(selected_blocks)}
        compact = slots_cpu.clone()
        for idx, slot in enumerate(compact.tolist()):
            if slot < 0:
                continue
            old_block = int(slot) // block_size
            offset_in_block = int(slot) % block_size
            compact[idx] = block_map[old_block] * block_size + offset_in_block
        return selected_blocks, compact
