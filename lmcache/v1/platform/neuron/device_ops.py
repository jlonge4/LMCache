# SPDX-License-Identifier: Apache-2.0
"""Neuron device operations backed by NKI kernels."""

# Future
from __future__ import annotations

# Standard
from typing import ClassVar

# Third Party
import torch

# First Party
from lmcache.lmcache_native import EngineKVFormat, TransferDirection
from lmcache.v1.platform.base.device_ops import DeviceOps
from lmcache.v1.platform.neuron import nki_kv_transfer


class NeuronDeviceOps(DeviceOps):
    """Device operation strategy for the native TorchNeuron backend."""

    device_type: ClassVar[str] = "neuron"

    def multi_layer_kv_transfer(
        self,
        key_value: torch.Tensor,
        key_value_ptrs: torch.Tensor | list[torch.Tensor],
        slot_mapping: torch.Tensor,
        paged_memory_device: torch.device,
        page_buffer_size: int,
        direction: TransferDirection,
        engine_kv_format: EngineKVFormat,
        block_size: int = 0,
        head_size: int = 0,
        skip_prefix_n_tokens: int = 0,
        block_stride_elems: int = 0,
    ) -> None:
        """Transfer KV rows through the Neuron NKI implementation.

        Args:
            key_value: LMCache's dense KV tensor.
            key_value_ptrs: Normalized per-layer Neuron tensors.
            slot_mapping: Token-to-vLLM-slot mapping.
            paged_memory_device: Neuron device containing the paged tensors.
            page_buffer_size: Total token capacity of one paged layer.
            direction: Transfer direction enum.
            engine_kv_format: Paged KV layout enum.
            block_size: Number of slots per paged block.
            head_size: Width of one KV head.
            skip_prefix_n_tokens: Leading H2D tokens that must not be written.
            block_stride_elems: Physical block stride; unused for tight HND.

        Raises:
            TypeError: If pointer-form operands are supplied.
            ValueError: If the declared page geometry is inconsistent.
            RuntimeError: If NKI integration is unavailable.
        """
        del block_stride_elems
        if not isinstance(key_value_ptrs, list):
            raise TypeError("Neuron NKI transfer requires per-layer tensors")
        if not key_value_ptrs:
            raise ValueError("Neuron NKI transfer requires at least one layer")
        if page_buffer_size != key_value_ptrs[0].shape[1] * block_size:
            raise ValueError("page_buffer_size does not match the paged layer")

        nki_kv_transfer.multi_layer_kv_transfer(
            key_value=key_value,
            layer_tensors=key_value_ptrs,
            slot_mapping=slot_mapping,
            direction=direction,
            engine_kv_format=engine_kv_format,
            block_size=block_size,
            head_size=head_size,
            skip_prefix_n_tokens=skip_prefix_n_tokens,
        )
