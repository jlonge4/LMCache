# SPDX-License-Identifier: Apache-2.0
"""Prototype NKI gather/scatter tests for Neuron paged KV caches.

This intentionally lives in tests rather than the runtime connector. It proves
the primitive needed to replace ``NeuronKVBlockStager``: indirect row movement
between a vLLM HND cache and a compact token-major Neuron tensor.

The simulator test runs when the Neuron SDK's ``nki`` package is installed.
The hardware probe is opt-in because it compiles kernels and requires a Neuron
device plus vllm-neuron's ``wrap_nki`` integration::

    LMCACHE_RUN_NKI_HARDWARE_PROBE=1 \
      NEURON_LOGICAL_NC_CONFIG=1 NEURON_CC_FLAGS=" --logical-nc-config=1 " \
      pytest -xvs tests/v1/platform/neuron/test_nki_kv_transfer_probe.py

The LNC variables are required, not optional: the kernels are launched with
grid 1 (``gather[1]``), and neuronx-cc rejects the compile with NCC_EVRF066 if
its ``--logical-nc-config`` does not match that grid. Add
``NEURON_RT_ULTRASERVER_MODE=4`` on trn3.
"""

# Standard
from importlib.util import find_spec
import os

# Third Party
import numpy as np
import pytest
import torch

# First Party
from lmcache.v1.platform.neuron.nki_kv_transfer import build_hnd_row_indices

_HAS_NKI = find_spec("nki") is not None

if _HAS_NKI:
    # Third Party
    import nki

    # First Party
    from lmcache.v1.platform.neuron.nki_kv_transfer import (
        gather_hnd_rows,
        scatter_hnd_rows,
    )


def _make_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a one-tile HND cache, slot mapping, and flattened row indices."""
    num_blocks = 8
    num_heads = 4
    block_size = 16
    head_size = 8
    slot_mapping = torch.tensor(
        [0, 17, 34, 51, 68, 85, 102, 119, 8, 25, 42, 59, 76, 93, 110, 127],
        dtype=torch.int64,
    )
    paged = torch.arange(
        2 * num_blocks * num_heads * block_size * head_size,
        dtype=torch.float32,
    ).reshape(2, num_blocks, num_heads, block_size, head_size)
    rows = build_hnd_row_indices(
        slot_mapping,
        num_blocks=num_blocks,
        num_heads=num_heads,
        block_size=block_size,
    )
    return paged, slot_mapping, rows


def test_hnd_row_indices_match_paged_layout() -> None:
    """Flattened rows select the same values as direct HND indexing."""
    paged, slot_mapping, rows = _make_inputs()
    expected_parts = []
    for kv_index in range(2):
        for slot in slot_mapping.tolist():
            block_index, block_offset = divmod(slot, paged.shape[3])
            for head_index in range(paged.shape[2]):
                expected_parts.append(
                    paged[kv_index, block_index, head_index, block_offset]
                )

    expected = torch.stack(expected_parts)
    actual = paged.reshape(-1, paged.shape[-1]).index_select(
        0, rows.flatten().to(torch.int64)
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not _HAS_NKI, reason="Neuron SDK NKI package is not installed")
def test_nki_simulator_gathers_and_scatters_hnd_rows() -> None:
    """NKI indirect DMA round-trips selected HND rows in the CPU simulator."""
    paged, _, rows = _make_inputs()
    paged_rows = paged.reshape(-1, paged.shape[-1]).numpy()
    row_indices = rows.numpy()

    gathered = np.asarray(nki.simulate(gather_hnd_rows)(paged_rows, row_indices))
    expected_gather = paged_rows[row_indices[:, 0]]
    np.testing.assert_array_equal(gathered, expected_gather)

    replacement = gathered + 100_000.0
    scattered = np.asarray(
        nki.simulate(scatter_hnd_rows)(
            paged_rows.copy(),
            row_indices,
            replacement,
        )
    )
    expected_scatter = paged_rows.copy()
    expected_scatter[row_indices[:, 0]] = replacement
    np.testing.assert_array_equal(scattered, expected_scatter)


@pytest.mark.neuron
@pytest.mark.skipif(
    os.environ.get("LMCACHE_RUN_NKI_HARDWARE_PROBE") != "1",
    reason="set LMCACHE_RUN_NKI_HARDWARE_PROBE=1 to compile and run NKI kernels",
)
def test_nki_hardware_gathers_and_scatters_hnd_rows() -> None:
    """Run the prototype through vllm-neuron's Torch-compatible NKI wrapper."""
    if not _HAS_NKI:
        pytest.skip("Neuron SDK NKI package is not installed")

    # Third Party
    # ToT torch-neuronx exposes nki_hop directly; the released vllm-neuron
    # container ships the same entry point under libtorch_neuronx_lite.
    try:
        from torch_neuronx.nki_hop import wrap_nki
    except ImportError:
        from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    if not hasattr(torch, "neuron") or not torch.neuron.is_available():  # type: ignore[attr-defined]
        pytest.skip("Neuron device is not available")

    paged, _, rows = _make_inputs()
    device = torch.device("neuron:0")
    paged_rows = paged.reshape(-1, paged.shape[-1]).to(device)
    row_indices = rows.to(device)

    # The launch grid must equal the LNC the compiler is using, or neuronx-cc
    # rejects the kernel with NCC_EVRF066. Derive it from the same environment
    # variable so one setting covers the framework, the compiler and the grid.
    lnc = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2"))

    gather = wrap_nki(gather_hnd_rows)
    gathered = gather[lnc](paged_rows=paged_rows, row_indices=row_indices)
    expected_gather = paged.reshape(-1, paged.shape[-1]).index_select(
        0, rows.flatten().to(torch.int64)
    )
    torch.testing.assert_close(gathered.cpu(), expected_gather)

    replacement = gathered + 100_000.0
    scatter = wrap_nki(scatter_hnd_rows)
    scattered = scatter[lnc](
        paged_rows=paged_rows,
        row_indices=row_indices,
        values=replacement,
    )
    expected_scatter = paged.reshape(-1, paged.shape[-1]).clone()
    expected_scatter.index_copy_(
        0,
        rows.flatten().to(torch.int64),
        replacement.cpu(),
    )
    torch.testing.assert_close(scattered.cpu(), expected_scatter)
