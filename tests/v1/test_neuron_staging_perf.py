# SPDX-License-Identifier: Apache-2.0
"""Micro-benchmark: round-trip KV staging D2H then H2D.

Compares the current PR path (NIXL D2H + torch copy H2D) against
the H2D staging path (NIXL D2H + NIXL WRITE H2D).

Run on a Neuron host:
    python -m pytest tests/v1/test_neuron_staging_perf.py -v -s
"""
import time
import pytest
import torch

try:
    import lmcache.lmcache_native as lmcache_native

    HAS_NATIVE = True
except ImportError:
    HAS_NATIVE = False

try:
    from lmcache.v1.gpu_connector.neuron_nixl_staging import NeuronNixlBlockStager

    HAS_STAGER = True
except ImportError:
    HAS_STAGER = False

from lmcache import device_ops

NUM_LAYERS = 16
NUM_HEADS = 8
HEAD_SIZE = 64
BLOCK_SIZE = 16
NUM_BLOCKS = 32
NUM_TOKENS = 128
DTYPE = torch.bfloat16
ROUNDS = 10


def _make_paged_kv(device: str) -> list[torch.Tensor]:
    """Create per-layer paged KV tensors in NL_X_TWO_NB_NH_BS_HS format."""
    return [
        torch.randn(
            2, NUM_BLOCKS, NUM_HEADS, BLOCK_SIZE, HEAD_SIZE,
            dtype=DTYPE, device=device,
        )
        for _ in range(NUM_LAYERS)
    ]


def _make_slot_mapping(num_tokens: int) -> torch.Tensor:
    return torch.arange(num_tokens, dtype=torch.int64, device="cpu")


def _make_kv_buffer(num_tokens: int) -> torch.Tensor:
    """CPU buffer matching LMCache chunk format: [num_layers, 2, num_tokens, hidden]."""
    hidden = NUM_HEADS * HEAD_SIZE
    return torch.zeros(NUM_LAYERS, 2, num_tokens, hidden, dtype=DTYPE, device="cpu")


@pytest.mark.skipif(not HAS_NATIVE, reason="lmcache_native not available")
class TestStagingPerf:
    """Measure round-trip: paged KV -> CPU buffer -> paged KV."""

    def _d2h_via_transfer(
        self, kv_buffer, layer_tensors, slot_mapping, fmt
    ):
        """D2H using multi_layer_kv_transfer (works on CPU tensors too)."""
        device_ops.multi_layer_kv_transfer(
            kv_buffer,
            layer_tensors,
            slot_mapping,
            torch.device("cpu"),
            NUM_BLOCKS * BLOCK_SIZE,
            lmcache_native.TransferDirection.D2H,
            fmt,
            block_size=BLOCK_SIZE,
            head_size=HEAD_SIZE,
        )

    def _h2d_via_transfer(
        self, kv_buffer, layer_tensors, slot_mapping, fmt
    ):
        """H2D using multi_layer_kv_transfer (current PR path)."""
        device_ops.multi_layer_kv_transfer(
            kv_buffer,
            layer_tensors,
            slot_mapping,
            torch.device("cpu"),
            NUM_BLOCKS * BLOCK_SIZE,
            lmcache_native.TransferDirection.H2D,
            fmt,
            block_size=BLOCK_SIZE,
            head_size=HEAD_SIZE,
        )

    def test_roundtrip_cpu_baseline(self):
        """Round-trip via multi_layer_kv_transfer only (CPU tensors)."""
        fmt = lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS
        layers = _make_paged_kv("cpu")
        slots = _make_slot_mapping(NUM_TOKENS)
        buf = _make_kv_buffer(NUM_TOKENS)

        # Warmup
        self._d2h_via_transfer(buf, layers, slots, fmt)
        self._h2d_via_transfer(buf, layers, slots, fmt)

        # Timed
        times = []
        for _ in range(ROUNDS):
            t0 = time.perf_counter()
            self._d2h_via_transfer(buf, layers, slots, fmt)
            self._h2d_via_transfer(buf, layers, slots, fmt)
            times.append((time.perf_counter() - t0) * 1000)

        p50 = sorted(times)[len(times) // 2]
        print(
            f"\nCPU round-trip (multi_layer_kv_transfer): "
            f"p50={p50:.2f}ms over {ROUNDS} rounds "
            f"({NUM_LAYERS} layers, {NUM_TOKENS} tokens)"
        )

    @pytest.mark.skipif(not HAS_STAGER, reason="NeuronNixlBlockStager not available")
    def test_roundtrip_nixl_d2h_only(self):
        """D2H via NIXL stager, H2D via multi_layer_kv_transfer."""
        fmt = lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS
        layers = _make_paged_kv("cpu")
        slots = _make_slot_mapping(NUM_TOKENS)
        buf = _make_kv_buffer(NUM_TOKENS)

        stager = NeuronNixlBlockStager()

        # Warmup
        stager.transfer_into_key_value(
            buf, layers, slots, fmt, BLOCK_SIZE, HEAD_SIZE
        )
        self._h2d_via_transfer(buf, layers, slots, fmt)

        times = []
        for _ in range(ROUNDS):
            t0 = time.perf_counter()
            stager.transfer_into_key_value(
                buf, layers, slots, fmt, BLOCK_SIZE, HEAD_SIZE
            )
            self._h2d_via_transfer(buf, layers, slots, fmt)
            times.append((time.perf_counter() - t0) * 1000)

        p50 = sorted(times)[len(times) // 2]
        print(
            f"\nNIXL D2H + torch H2D round-trip: "
            f"p50={p50:.2f}ms over {ROUNDS} rounds"
        )

    @pytest.mark.skipif(not HAS_STAGER, reason="NeuronNixlBlockStager not available")
    def test_roundtrip_nixl_both(self):
        """D2H + H2D both via NIXL stager (H2D staging branch)."""
        fmt = lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS
        layers = _make_paged_kv("cpu")
        slots = _make_slot_mapping(NUM_TOKENS)
        buf = _make_kv_buffer(NUM_TOKENS)

        stager = NeuronNixlBlockStager()

        if not hasattr(stager, "transfer_from_key_value"):
            pytest.skip("transfer_from_key_value not on this branch")

        # Warmup
        stager.transfer_into_key_value(
            buf, layers, slots, fmt, BLOCK_SIZE, HEAD_SIZE
        )
        stager.transfer_from_key_value(
            buf, layers, slots, fmt, BLOCK_SIZE, HEAD_SIZE
        )

        times = []
        for _ in range(ROUNDS):
            t0 = time.perf_counter()
            stager.transfer_into_key_value(
                buf, layers, slots, fmt, BLOCK_SIZE, HEAD_SIZE
            )
            stager.transfer_from_key_value(
                buf, layers, slots, fmt, BLOCK_SIZE, HEAD_SIZE
            )
            times.append((time.perf_counter() - t0) * 1000)

        p50 = sorted(times)[len(times) // 2]
        print(
            f"\nNIXL D2H + NIXL H2D round-trip: "
            f"p50={p50:.2f}ms over {ROUNDS} rounds"
        )
