#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Run the NKI gather/scatter probe at production KV-cache shapes.

Same kernels as test_nki_kv_transfer_probe.py, but the cache geometry, dtype and
chunk size are parameters so the indirect-DMA primitive can be checked at the
sizes a real MoE serving stack would hand it.

    NEURON_LOGICAL_NC_CONFIG=1 NEURON_CC_FLAGS=" --logical-nc-config=1 " \
      python tests/v1/platform/neuron/probe_hw_shapes.py

Values are generated in the target dtype so an exact comparison is valid: the
kernels move rows without arithmetic, so any bit change is a real defect.
"""

# Standard
import os
import sys
import time

# Third Party
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# First Party
from lmcache.v1.platform.neuron.nki_kv_transfer import (  # noqa: E402
    build_hnd_row_indices,
    gather_hnd_rows,
    scatter_hnd_rows,
)

# name, num_blocks, num_heads, block_size, head_size, num_tokens, dtype
CASES = [
    # The existing test case, as a baseline.
    ("toy fp32", 8, 4, 16, 8, 16, torch.float32),
    # Llama-3.1-70B / Mixtral 8x22B class, one layer, 8 KV heads on this rank,
    # one 256-token LMCache chunk. 4096 rows = 32 pmax tiles, 256-byte rows.
    ("llama70b-class bf16", 256, 8, 16, 128, 256, torch.bfloat16),
    # Same but block_size 128, which vllm-neuron commonly uses.
    ("bs128 bf16", 32, 8, 128, 128, 256, torch.bfloat16),
    # One KV head per rank at high TP, still a 256-token chunk: 512 rows.
    ("tp-sharded 1 head bf16", 256, 1, 16, 128, 256, torch.bfloat16),
    # DeepSeek-V3 MLA latent width, 512 kv_lora + 64 rope, treated as one
    # "head" of 576. Row width is not a power of two.
    ("mla 576 bf16", 256, 1, 16, 576, 256, torch.bfloat16),
    # 128 partitions x 2048 elements x 2 B = 512 KiB per tile, the ideal DMA
    # transfer size from the DMA bundle.
    ("wide 2048 bf16", 256, 1, 16, 2048, 256, torch.bfloat16),
    # Ragged row counts, which is what production actually hands over: a chunk
    # is however many tokens are left, not a multiple of 128.
    ("ragged 250 tok bf16", 256, 8, 16, 128, 250, torch.bfloat16),
    ("ragged 300 tok mla", 256, 1, 16, 576, 300, torch.bfloat16),
    ("ragged 65 rows fp32", 8, 1, 16, 8, 33, torch.float32),
    ("ragged 1 tok bf16", 256, 8, 16, 128, 1, torch.bfloat16),
]


def run_case(
    name, num_blocks, num_heads, block_size, head_size, num_tokens, dtype, wrap_nki, lnc
):
    total_slots = num_blocks * block_size
    if num_tokens > total_slots:
        print(f"{name}: SKIP, {num_tokens} tokens > {total_slots} slots")
        return None

    # Distinct, non-contiguous slots, the way a real slot_mapping looks.
    stride = max(1, total_slots // num_tokens)
    slots = torch.arange(num_tokens, dtype=torch.int64) * stride
    slots = slots % total_slots

    paged = torch.randn(
        2, num_blocks, num_heads, block_size, head_size, dtype=torch.float32
    ).to(dtype)
    rows = build_hnd_row_indices(
        slots, num_blocks=num_blocks, num_heads=num_heads, block_size=block_size
    )
    num_rows = rows.shape[0]
    row_bytes = head_size * paged.element_size()
    cache_mb = paged.numel() * paged.element_size() / 1e6

    tile = 128
    full, rem = divmod(num_rows, tile)

    device = torch.device("neuron:0")
    flat = paged.reshape(-1, head_size)
    paged_rows = flat.to(device)
    row_indices = rows.to(device)

    print(f"\n== {name}")
    print(
        f"   cache [2,{num_blocks},{num_heads},{block_size},{head_size}] "
        f"{str(dtype).replace('torch.', '')}  {cache_mb:.1f} MB"
    )
    print(
        f"   rows={num_rows} ({full} full tiles of {tile}"
        f"{f' + tail of {rem}' if rem else ''})  "
        f"row_bytes={row_bytes}  per_tile={tile * row_bytes / 1024:.0f} KiB  "
        f"gathered={num_rows * row_bytes / 1e6:.2f} MB"
    )

    t0 = time.time()
    try:
        gathered = wrap_nki(gather_hnd_rows)[lnc](
            paged_rows=paged_rows, row_indices=row_indices
        )
        got = gathered.cpu()
    except Exception as e:  # compile or runtime failure
        print(f"   gather: FAIL after {time.time() - t0:.1f}s")
        print(f"   {type(e).__name__}: {str(e)[:1500]}")
        return False
    want = flat.index_select(0, rows.flatten().to(torch.int64))
    gather_ok = torch.equal(got, want)
    print(f"   gather: {'OK' if gather_ok else 'MISMATCH'}  {time.time() - t0:.1f}s")
    if not gather_ok:
        diff = (got != want).nonzero()
        print(f"   {diff.shape[0]} differing elements, first: {diff[:3].tolist()}")

    replacement = torch.randn(num_rows, head_size, dtype=torch.float32).to(dtype)
    t1 = time.time()
    try:
        scattered = wrap_nki(scatter_hnd_rows)[lnc](
            paged_rows=paged_rows,
            row_indices=row_indices,
            values=replacement.to(device),
        )
        got_s = scattered.cpu()
    except Exception as e:
        print(f"   scatter: FAIL after {time.time() - t1:.1f}s")
        print(f"   {type(e).__name__}: {str(e)[:1500]}")
        return False
    want_s = flat.clone()
    want_s.index_copy_(0, rows.flatten().to(torch.int64), replacement)
    scatter_ok = torch.equal(got_s, want_s)
    print(f"   scatter: {'OK' if scatter_ok else 'MISMATCH'}  {time.time() - t1:.1f}s")
    return gather_ok and scatter_ok


def main() -> int:
    try:
        from torch_neuronx.nki_hop import wrap_nki
    except ImportError:
        from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    if not hasattr(torch, "neuron") or not torch.neuron.is_available():
        print("no Neuron device visible")
        return 2

    lnc = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2"))
    print(f"LNC={lnc} grid={lnc}")

    results = {}
    for case in CASES:
        results[case[0]] = run_case(*case, wrap_nki=wrap_nki, lnc=lnc)

    print("\n=== summary ===")
    for name, ok in results.items():
        print(f"{name:26s} {'PASS' if ok else 'SKIP' if ok is None else 'FAIL'}")
    return 0 if all(v for v in results.values() if v is not None) else 1


if __name__ == "__main__":
    sys.exit(main())
