# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the Neuron NKI DeviceOps integration."""

# Standard
from typing import Any

# Third Party
import pytest
import torch

# First Party
from lmcache.lmcache_native import EngineKVFormat, TransferDirection
from lmcache.v1.platform.neuron import nki_kv_transfer
from lmcache.v1.platform.neuron.device_ops import NeuronDeviceOps


class _FakeWrappedKernel:
    """Torch implementation of the two NKI row kernels."""

    def __init__(self, kernel: object, gather_marker: object) -> None:
        self._kernel = kernel
        self._gather_marker = gather_marker

    def __getitem__(self, grid: int) -> Any:
        assert grid == 2

        def launch(**kwargs: torch.Tensor) -> torch.Tensor:
            rows = kwargs["row_indices"].flatten().to(torch.int64)
            if self._kernel is self._gather_marker:
                return kwargs["paged_rows"].index_select(0, rows)
            result = kwargs["paged_rows"].clone()
            result.index_copy_(0, rows, kwargs["values"])
            return result

        return launch


def _install_fake_nki(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install Torch stand-ins at the internal NKI seam."""
    gather_marker = object()
    scatter_marker = object()
    monkeypatch.setattr(
        nki_kv_transfer, "gather_hnd_rows", gather_marker, raising=False
    )
    monkeypatch.setattr(
        nki_kv_transfer, "scatter_hnd_rows", scatter_marker, raising=False
    )
    monkeypatch.setattr(nki_kv_transfer, "is_available", lambda: True)
    monkeypatch.setattr(
        nki_kv_transfer,
        "_require_wrap_nki",
        lambda: lambda kernel: _FakeWrappedKernel(kernel, gather_marker),
    )
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    monkeypatch.setattr(nki_kv_transfer, "_validate_transfer", lambda *args: None)


def _make_transfer_inputs() -> tuple[
    torch.Tensor,
    list[torch.Tensor],
    torch.Tensor,
]:
    num_layers = 2
    num_blocks = 4
    num_heads = 2
    block_size = 4
    head_size = 3
    layers = [
        (
            torch.arange(
                2 * num_blocks * num_heads * block_size * head_size,
                dtype=torch.float32,
            ).reshape(2, num_blocks, num_heads, block_size, head_size)
            + layer_id * 10_000
        )
        for layer_id in range(num_layers)
    ]
    key_value = torch.full(
        (2, num_layers, 5, num_heads * head_size),
        -1.0,
    )
    slots = torch.tensor([0, -1, 7, 11, 15], dtype=torch.int64)
    return key_value, layers, slots


def test_nki_transfer_gathers_valid_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D2H fills valid tokens and leaves negative slots untouched."""
    _install_fake_nki(monkeypatch)
    key_value, layers, slots = _make_transfer_inputs()

    nki_kv_transfer.multi_layer_kv_transfer(
        key_value=key_value,
        layer_tensors=layers,
        slot_mapping=slots,
        direction=TransferDirection.D2H,
        engine_kv_format=EngineKVFormat.NL_X_TWO_NB_NH_BS_HS,
        block_size=4,
        head_size=3,
    )

    for layer_id, layer in enumerate(layers):
        for token_id, slot in enumerate(slots.tolist()):
            if slot < 0:
                torch.testing.assert_close(
                    key_value[:, layer_id, token_id],
                    torch.full((2, 6), -1.0),
                )
                continue
            block_id, block_offset = divmod(slot, 4)
            expected = layer[:, block_id, :, block_offset].reshape(2, 6)
            torch.testing.assert_close(key_value[:, layer_id, token_id], expected)


def test_nki_transfer_scatters_and_skips_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """H2D updates selected rows while preserving skipped and invalid tokens."""
    _install_fake_nki(monkeypatch)
    key_value, layers, slots = _make_transfer_inputs()
    original = [layer.clone() for layer in layers]
    for layer_id in range(key_value.shape[1]):
        key_value[:, layer_id] = layer_id * 1_000 + torch.arange(
            key_value[:, layer_id].numel(),
            dtype=key_value.dtype,
        ).reshape_as(key_value[:, layer_id])

    nki_kv_transfer.multi_layer_kv_transfer(
        key_value=key_value,
        layer_tensors=layers,
        slot_mapping=slots,
        direction=TransferDirection.H2D,
        engine_kv_format=EngineKVFormat.NL_X_TWO_NB_NH_BS_HS,
        block_size=4,
        head_size=3,
        skip_prefix_n_tokens=2,
    )

    for layer_id, layer in enumerate(layers):
        for token_id, slot in enumerate(slots.tolist()):
            if token_id < 2 or slot < 0:
                continue
            block_id, block_offset = divmod(slot, 4)
            expected = key_value[:, layer_id, token_id].reshape(2, 2, 3)
            torch.testing.assert_close(
                layer[:, block_id, :, block_offset],
                expected,
            )

        prefix_block, prefix_offset = divmod(int(slots[0]), 4)
        torch.testing.assert_close(
            layer[:, prefix_block, :, prefix_offset],
            original[layer_id][:, prefix_block, :, prefix_offset],
        )


def test_neuron_device_ops_delegates_tensor_lists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NeuronDeviceOps preserves the shared multi-layer transfer interface."""
    recorded: dict[str, Any] = {}

    def record(**kwargs: Any) -> None:
        recorded.update(kwargs)

    monkeypatch.setattr(nki_kv_transfer, "multi_layer_kv_transfer", record)
    key_value = torch.empty((2, 1, 2, 6))
    layers = [torch.empty((2, 4, 2, 4, 3))]
    slots = torch.tensor([0, 1])

    NeuronDeviceOps().multi_layer_kv_transfer(
        key_value,
        layers,
        slots,
        torch.device("cpu"),
        page_buffer_size=16,
        direction=TransferDirection.D2H,
        engine_kv_format=EngineKVFormat.NL_X_TWO_NB_NH_BS_HS,
        block_size=4,
        head_size=3,
    )

    assert recorded["key_value"] is key_value
    assert recorded["layer_tensors"] is layers
    assert recorded["slot_mapping"] is slots
