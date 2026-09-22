# SPDX-License-Identifier: Apache-2.0
"""NKI implementation of slot-mapped Neuron KV-cache transfers.

The public transfer function matches ``DeviceOps.multi_layer_kv_transfer``.
Callers provide the normalized per-layer paged tensors and do not need to know
about HND row addressing, NKI launch grids, or the device staging tensors.
"""

# Future
from __future__ import annotations

# Standard
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any, Callable
import os

# Third Party
import torch

# First Party
from lmcache.lmcache_native import EngineKVFormat, TransferDirection

if TYPE_CHECKING:
    from collections.abc import Sequence

_HAS_NKI = find_spec("nki") is not None

if _HAS_NKI:
    # Third Party
    import nki
    import nki.isa as nisa
    import nki.language as nl

    @nki.jit
    def gather_hnd_rows(
        paged_rows: torch.Tensor,
        row_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Gather indirect HBM rows into a compact tensor.

        Args:
            paged_rows: Flattened HND cache rows shaped ``[rows, head_size]``.
            row_indices: Indirect row indices shaped ``[selected_rows, 1]``.

        Returns:
            The selected rows in index order.
        """
        num_rows = row_indices.shape[0]
        row_width = paged_rows.shape[1]
        tile_rows = nl.tile_size.pmax
        num_full = num_rows // tile_rows
        remainder = num_rows % tile_rows

        output = nl.ndarray(
            (num_rows, row_width),
            dtype=paged_rows.dtype,
            buffer=nl.shared_hbm,
        )

        # NKI does not permit direct calls to nested helper functions, so the
        # full-tile and tail bodies intentionally remain inlined.
        for tile_id in nl.affine_range(num_full):
            start = tile_id * tile_rows
            offsets = nl.ndarray(
                (tile_rows, 1),
                dtype=nl.int32,
                buffer=nl.sbuf,
            )
            nisa.dma_copy(
                dst=offsets,
                src=row_indices[start : start + tile_rows, :],
            )
            row_tile = nl.ndarray(
                (tile_rows, row_width),
                dtype=paged_rows.dtype,
                buffer=nl.sbuf,
            )
            nisa.dma_copy(
                dst=row_tile,
                src=paged_rows.vector_select(0, offsets),  # type: ignore[attr-defined]
            )
            nisa.dma_copy(
                dst=output[start : start + tile_rows, :],
                src=row_tile,
            )

        if remainder:
            tail = num_full * tile_rows
            tail_offsets = nl.ndarray(
                (remainder, 1),
                dtype=nl.int32,
                buffer=nl.sbuf,
            )
            nisa.dma_copy(
                dst=tail_offsets,
                src=row_indices[tail : tail + remainder, :],
            )
            tail_tile = nl.ndarray(
                (remainder, row_width),
                dtype=paged_rows.dtype,
                buffer=nl.sbuf,
            )
            nisa.dma_copy(
                dst=tail_tile,
                src=paged_rows.vector_select(  # type: ignore[attr-defined]
                    0, tail_offsets
                ),
            )
            nisa.dma_copy(
                dst=output[tail : tail + remainder, :],
                src=tail_tile,
            )
        return output

    @nki.jit
    def scatter_hnd_rows(
        paged_rows: torch.Tensor,
        row_indices: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """Scatter compact rows into an existing HBM paged-cache tensor.

        Args:
            paged_rows: Flattened HND cache rows shaped ``[rows, head_size]``.
            row_indices: Indirect destination rows shaped ``[selected_rows, 1]``.
            values: Rows to write, shaped ``[selected_rows, head_size]``.

        Returns:
            The updated ``paged_rows`` tensor.
        """
        num_rows = row_indices.shape[0]
        tile_rows = nl.tile_size.pmax
        num_full = num_rows // tile_rows
        remainder = num_rows % tile_rows

        for tile_id in nl.affine_range(num_full):
            start = tile_id * tile_rows
            offsets = nl.ndarray(
                (tile_rows, 1),
                dtype=nl.int32,
                buffer=nl.sbuf,
            )
            value_tile = nl.ndarray(
                (tile_rows, values.shape[1]),
                dtype=values.dtype,
                buffer=nl.sbuf,
            )
            nisa.dma_copy(
                dst=offsets,
                src=row_indices[start : start + tile_rows, :],
            )
            nisa.dma_copy(
                dst=value_tile,
                src=values[start : start + tile_rows, :],
            )
            nisa.dma_copy(
                dst=paged_rows.vector_select(  # type: ignore[attr-defined]
                    0, offsets
                ),
                src=value_tile,
            )

        if remainder:
            tail = num_full * tile_rows
            tail_offsets = nl.ndarray(
                (remainder, 1),
                dtype=nl.int32,
                buffer=nl.sbuf,
            )
            tail_values = nl.ndarray(
                (remainder, values.shape[1]),
                dtype=values.dtype,
                buffer=nl.sbuf,
            )
            nisa.dma_copy(
                dst=tail_offsets,
                src=row_indices[tail : tail + remainder, :],
            )
            nisa.dma_copy(
                dst=tail_values,
                src=values[tail : tail + remainder, :],
            )
            nisa.dma_copy(
                dst=paged_rows.vector_select(  # type: ignore[attr-defined]
                    0, tail_offsets
                ),
                src=tail_values,
            )
        return paged_rows


def is_available() -> bool:
    """Return whether NKI and a Torch-compatible kernel wrapper are installed."""
    if not _HAS_NKI:
        return False
    return _load_wrap_nki() is not None


def build_hnd_row_indices(
    slot_mapping: torch.Tensor,
    num_blocks: int,
    num_heads: int,
    block_size: int,
) -> torch.Tensor:
    """Build flattened HND row indices for every K/V, token, and head.

    A cache with shape ``[2, NB, NH, BS, HS]`` can be viewed as
    ``[2 * NB * NH * BS, HS]`` without changing storage. Returned rows are
    ordered as ``[K/V, token, head]``, matching a compact tensor with shape
    ``[2, num_tokens, NH, HS]``.

    Args:
        slot_mapping: Non-negative physical token slots.
        num_blocks: Number of blocks in the paged cache.
        num_heads: Number of KV heads.
        block_size: Number of token slots in each block.

    Returns:
        An ``int32`` tensor shaped ``[2 * num_tokens * num_heads, 1]``.

    Raises:
        ValueError: If a slot is negative or outside the paged cache.
    """
    slots = slot_mapping.to(dtype=torch.int64, device="cpu")
    max_slot = num_blocks * block_size
    if bool(((slots < 0) | (slots >= max_slot)).any()):
        raise ValueError(f"slot_mapping must contain values in [0, {max_slot})")

    block_ids = slots // block_size
    block_offsets = slots % block_size
    kv_ids = torch.arange(2).reshape(2, 1, 1)
    head_ids = torch.arange(num_heads).reshape(1, 1, -1)
    rows = (
        (kv_ids * num_blocks + block_ids.reshape(1, -1, 1)) * num_heads + head_ids
    ) * block_size + block_offsets.reshape(1, -1, 1)
    return rows.reshape(-1, 1).to(dtype=torch.int32)


def multi_layer_kv_transfer(
    key_value: torch.Tensor,
    layer_tensors: Sequence[torch.Tensor],
    slot_mapping: torch.Tensor,
    direction: TransferDirection,
    engine_kv_format: EngineKVFormat,
    block_size: int,
    head_size: int,
    skip_prefix_n_tokens: int = 0,
) -> None:
    """Transfer selected HND KV rows between LMCache and Neuron.

    Args:
        key_value: LMCache tensor shaped ``[2, layers, tokens, hidden_size]``.
        layer_tensors: Normalized Neuron paged tensors, one per layer.
        slot_mapping: Physical vLLM slot for each token in ``key_value``.
        direction: ``D2H`` gathers from vLLM; ``H2D`` scatters into vLLM.
        engine_kv_format: Physical layout of each paged tensor.
        block_size: Number of token slots per block.
        head_size: Width of one KV head.
        skip_prefix_n_tokens: Leading tokens that H2D must not overwrite.

    Raises:
        RuntimeError: If NKI integration is unavailable.
        ValueError: If the tensors or layout do not match the supported HND
            transfer contract.
    """
    if not is_available():
        raise RuntimeError(
            "Neuron NKI KV transfer requires nki and torch-neuronx nki_hop"
        )
    _validate_transfer(
        key_value,
        layer_tensors,
        slot_mapping,
        direction,
        engine_kv_format,
        block_size,
        head_size,
        skip_prefix_n_tokens,
    )
    if not layer_tensors or slot_mapping.numel() == 0:
        return

    slots_cpu = slot_mapping.to(dtype=torch.int64, device="cpu")
    token_indices = torch.arange(slots_cpu.numel(), dtype=torch.int64)
    valid_mask = slots_cpu >= 0
    if skip_prefix_n_tokens:
        valid_mask[:skip_prefix_n_tokens] = False
    valid_token_indices = token_indices[valid_mask]
    if valid_token_indices.numel() == 0:
        return

    first_layer = layer_tensors[0]
    _, num_blocks, num_heads, _, _ = first_layer.shape
    row_indices_cpu = build_hnd_row_indices(
        slots_cpu[valid_mask],
        num_blocks=num_blocks,
        num_heads=num_heads,
        block_size=block_size,
    )
    row_indices = row_indices_cpu.to(first_layer.device)
    lnc = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2"))
    wrap_nki = _require_wrap_nki()

    if int(direction) == int(TransferDirection.D2H):
        gather = wrap_nki(gather_hnd_rows)
        for layer_id, layer_tensor in enumerate(layer_tensors):
            paged_rows = layer_tensor.reshape(-1, head_size)
            gathered = gather[lnc](
                paged_rows=paged_rows,
                row_indices=row_indices,
            )
            compact = gathered.reshape(
                2,
                valid_token_indices.numel(),
                num_heads * head_size,
            )
            key_value[:, layer_id].index_copy_(
                1,
                valid_token_indices,
                compact.cpu(),
            )
        return

    scatter = wrap_nki(scatter_hnd_rows)
    for layer_id, layer_tensor in enumerate(layer_tensors):
        values = (
            key_value[:, layer_id]
            .index_select(1, valid_token_indices)
            .reshape(-1, head_size)
            .to(first_layer.device)
        )
        updated = scatter[lnc](
            paged_rows=layer_tensor.reshape(-1, head_size),
            row_indices=row_indices,
            values=values,
        )
        # The model path has an FX aliasing pass, but connector calls execute
        # outside it. Copy the returned same-shaped view back explicitly so the
        # paged cache is updated whether nki_hop aliases the input or not.
        layer_tensor.reshape(-1, head_size).copy_(updated)


def _validate_transfer(
    key_value: torch.Tensor,
    layer_tensors: Sequence[torch.Tensor],
    slot_mapping: torch.Tensor,
    direction: TransferDirection,
    engine_kv_format: EngineKVFormat,
    block_size: int,
    head_size: int,
    skip_prefix_n_tokens: int,
) -> None:
    if int(engine_kv_format) != int(EngineKVFormat.NL_X_TWO_NB_NH_BS_HS):
        raise ValueError(
            "Neuron NKI KV transfer currently supports only NL_X_TWO_NB_NH_BS_HS"
        )
    if key_value.device.type != "cpu":
        raise ValueError("Neuron NKI KV transfer requires a CPU LMCache tensor")
    if key_value.ndim != 4 or key_value.shape[0] != 2:
        raise ValueError("key_value must have shape [2, layers, tokens, hidden]")
    if len(layer_tensors) != key_value.shape[1]:
        raise ValueError("layer count does not match key_value")
    if slot_mapping.numel() != key_value.shape[2]:
        raise ValueError("slot_mapping length does not match key_value tokens")
    if skip_prefix_n_tokens < 0 or skip_prefix_n_tokens > slot_mapping.numel():
        raise ValueError("skip_prefix_n_tokens is outside the token range")
    if int(direction) not in (
        int(TransferDirection.H2D),
        int(TransferDirection.D2H),
    ):
        raise ValueError(f"unsupported transfer direction: {direction!r}")
    if block_size <= 0 or head_size <= 0:
        raise ValueError("block_size and head_size must be positive")
    expected_shape = layer_tensors[0].shape if layer_tensors else None
    for layer_tensor in layer_tensors:
        if layer_tensor.device.type != "neuron":
            raise ValueError("all paged layer tensors must be on Neuron")
        if layer_tensor.shape != expected_shape:
            raise ValueError("all paged layer tensors must have the same shape")
        if layer_tensor.ndim != 5 or layer_tensor.shape[0] != 2:
            raise ValueError("paged layers must have shape [2, NB, NH, BS, HS]")
        if layer_tensor.shape[3] != block_size:
            raise ValueError("paged layer block size does not match block_size")
        if layer_tensor.shape[4] != head_size:
            raise ValueError("paged layer head size does not match head_size")
        if layer_tensor.shape[2] * head_size != key_value.shape[3]:
            raise ValueError("paged layer hidden size does not match key_value")


def _load_wrap_nki() -> Callable[[Callable[..., Any]], Any] | None:
    try:
        # Third Party
        from torch_neuronx.nki_hop import wrap_nki
    except ImportError:
        try:
            # Third Party
            from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
        except ImportError:
            return None
    return wrap_nki


def _require_wrap_nki() -> Callable[[Callable[..., Any]], Any]:
    wrap_nki = _load_wrap_nki()
    if wrap_nki is None:
        raise RuntimeError("No Torch-compatible NKI wrapper is installed")
    return wrap_nki
