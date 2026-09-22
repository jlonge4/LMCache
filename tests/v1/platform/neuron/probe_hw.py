#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Standalone hardware run of the NKI gather/scatter probe, no pytest.

Imports the production prototype kernels and the test input fixture. Reads
NEURON_LOGICAL_NC_CONFIG for the launch grid, exactly like the pytest path.

    NEURON_LOGICAL_NC_CONFIG=1 NEURON_CC_FLAGS=" --logical-nc-config=1 " \
      python tests/v1/platform/neuron/probe_hw.py

Exits 0 on success, 1 on a value mismatch, 2 if the device or SDK is missing.
"""

# Standard
import os
import sys

# Third Party
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# First Party
from test_nki_kv_transfer_probe import (  # noqa: E402
    _make_inputs,
)
from lmcache.v1.platform.neuron.nki_kv_transfer import (  # noqa: E402
    gather_hnd_rows,
    scatter_hnd_rows,
)


def main() -> int:
    try:
        from torch_neuronx.nki_hop import wrap_nki
    except ImportError:
        try:
            from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
        except ImportError:
            print("no nki_hop: install torch-neuronx or vllm-neuron")
            return 2

    if not hasattr(torch, "neuron") or not torch.neuron.is_available():
        print("no Neuron device visible")
        return 2

    lnc = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2"))
    print(f"LNC={lnc} grid={lnc}")

    paged, _, rows = _make_inputs()
    device = torch.device("neuron:0")
    paged_rows = paged.reshape(-1, paged.shape[-1]).to(device)
    row_indices = rows.to(device)

    gathered = wrap_nki(gather_hnd_rows)[lnc](
        paged_rows=paged_rows, row_indices=row_indices
    )
    expected_gather = paged.reshape(-1, paged.shape[-1]).index_select(
        0, rows.flatten().to(torch.int64)
    )
    gather_ok = torch.equal(gathered.cpu(), expected_gather)
    print(f"gather: {'OK' if gather_ok else 'MISMATCH'} shape={tuple(gathered.shape)}")

    replacement = gathered + 100_000.0
    scattered = wrap_nki(scatter_hnd_rows)[lnc](
        paged_rows=paged_rows,
        row_indices=row_indices,
        values=replacement,
    )
    expected_scatter = paged.reshape(-1, paged.shape[-1]).clone()
    expected_scatter.index_copy_(0, rows.flatten().to(torch.int64), replacement.cpu())
    scatter_ok = torch.equal(scattered.cpu(), expected_scatter)
    print(f"scatter: {'OK' if scatter_ok else 'MISMATCH'}")

    if not (gather_ok and scatter_ok):
        bad = (gathered.cpu() != expected_gather).nonzero()[:5]
        print(f"first mismatched gather positions: {bad.tolist()}")
        return 1

    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
