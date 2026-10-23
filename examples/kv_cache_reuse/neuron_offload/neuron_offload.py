#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Show LMCache offloading vLLM KV cache from Trainium to host RAM and reading it back.

One LMCache multiprocess (MP) server keeps KV chunks in CPU memory (its L1 tier).
Two vLLM engines then run the same prompt one after the other, each in a new
process. Engine 1 computes the prompt, and LMCache writes its KV to the server.
Engine 2 starts with an empty vLLM prefix cache, because that cache lived in
engine 1's process. So any KV that engine 2 reuses was read back from LMCache.

The run passes only if the server counters match that exactly and both engines
generate the same tokens. Setup and expected output are in README.md.

    python neuron_offload.py [--model ID_OR_PATH] [--log-dir DIR]
"""

# Future
from __future__ import annotations

# Standard
from pathlib import Path
from typing import TypedDict
import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

CHUNK_SIZE = 256  # tokens per LMCache chunk
PREFIX_CHUNKS = 4
# vLLM always computes the last prompt token itself to get logits. One token
# past the chunk boundary lets all PREFIX_CHUNKS chunks be served from cache.
PROMPT_TOKENS = PREFIX_CHUNKS * CHUNK_SIZE + 1
PREFIX_TOKENS = PREFIX_CHUNKS * CHUNK_SIZE
MAX_TOKENS = 16

MP_PORT = 5555  # ZMQ port the vLLM connector sends KV requests to
HTTP_PORT = 8080  # server HTTP API: /healthcheck and /metrics
SERVER_START_TIMEOUT_S = 120
COUNTER_SETTLE_TIMEOUT_S = 30

# Prometheus names of the server counters this run checks.
COUNTERS = {
    "l1_write_chunks": "lmcache_mp_l1_write_chunks_total",
    "l1_read_chunks": "lmcache_mp_l1_read_chunks_total",
    "lookup_requested_tokens": "lmcache_mp_lookup_requested_tokens_total",
    "lookup_hit_tokens": "lmcache_mp_lookup_hit_tokens_total",
    "lookup_hit_l1_tokens": "lmcache_mp_lookup_hit_l1_tokens_total",
}

# What each engine must add to each counter.
EXPECTED_ENGINE_1 = {  # cold cache: nothing to read, the prefix is written
    "l1_write_chunks": PREFIX_CHUNKS,
    "l1_read_chunks": 0,
    "lookup_requested_tokens": PREFIX_TOKENS,
    "lookup_hit_tokens": 0,
    "lookup_hit_l1_tokens": 0,
}
EXPECTED_ENGINE_2 = {  # warm cache: the whole prefix is read back, nothing new written
    "l1_write_chunks": 0,
    "l1_read_chunks": PREFIX_CHUNKS,
    "lookup_requested_tokens": PREFIX_TOKENS,
    "lookup_hit_tokens": PREFIX_TOKENS,
    "lookup_hit_l1_tokens": PREFIX_TOKENS,
}
# Prompt tokens each engine reports it did not compute. vLLM counts both its
# own prefix-cache hits and KV loaded through the connector. Its own cache
# starts empty in each new process, so for engine 2 this is KV from LMCache.
EXPECTED_CACHED_TOKENS = (0, PREFIX_TOKENS)


class EngineResult(TypedDict):
    """What an engine process writes to its result file."""

    token_ids: list[int]
    text: str
    num_cached_tokens: int


def run_engine(model: str, result_path: Path) -> None:
    """Generate once through the LMCache MP connector and save the output.

    Runs in a child process so that each engine starts with an empty vLLM
    prefix cache.

    Args:
        model: Model path or Hugging Face id.
        result_path: File that receives the generated token ids and the
            number of cached prompt tokens as JSON.

    Raises:
        ValueError: If the prompt text tokenizes to fewer than PROMPT_TOKENS.
    """
    # Third Party
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    from vllm.inputs import TokensPrompt

    llm = LLM(
        model=model,
        max_model_len=2048,
        # The connector identifies KV by prefix-cache block hashes.
        enable_prefix_caching=True,
        # On Neuron, prefix caching needs segmented prefill, which turns on
        # only for a supported size strictly below max_model_len.
        max_num_batched_tokens=1024,
        # One request at a time, so only batch size 1 is compiled.
        max_num_seqs=1,
        kv_transfer_config=KVTransferConfig(
            kv_connector="LMCacheMPConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "lmcache.mp.host": "tcp://localhost",
                "lmcache.mp.port": MP_PORT,
            },
        ),
    )
    text = (
        "Trainium keeps the KV cache in HBM, and LMCache offloads it to host RAM. "
        * 100
    )
    token_ids = llm.get_tokenizer().encode(text)[:PROMPT_TOKENS]
    if len(token_ids) != PROMPT_TOKENS:
        raise ValueError(f"prompt has {len(token_ids)} tokens, need {PROMPT_TOKENS}")
    output = llm.generate(
        TokensPrompt(prompt_token_ids=token_ids),
        SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS),
    )
    completion = output[0].outputs[0]
    result: EngineResult = {
        "token_ids": list(completion.token_ids),
        "text": completion.text,
        "num_cached_tokens": output[0].num_cached_tokens,
    }
    result_path.write_text(json.dumps(result))


def read_counters() -> dict[str, float]:
    """Return the checked server counters, summed over their label sets.

    An OTel counter appears on /metrics only after its first increment, so a
    missing counter reads as 0.
    """
    with urllib.request.urlopen(f"http://localhost:{HTTP_PORT}/metrics") as response:
        text = response.read().decode()
    totals = {key: 0.0 for key in COUNTERS}
    for line in text.splitlines():
        for key, name in COUNTERS.items():
            if line.startswith((name + "{", name + " ")):
                totals[key] += float(line.rsplit(" ", 1)[1])
    return totals


def wait_for_counters(
    baseline: dict[str, float], expected: dict[str, int]
) -> dict[str, float]:
    """Wait for the counters to reach ``baseline + expected``.

    The server applies stores and publishes metric events asynchronously, so
    counters can lag the engine exit briefly.

    Returns:
        The counter deltas observed when they matched or when the wait ran out.
    """
    deadline = time.monotonic() + COUNTER_SETTLE_TIMEOUT_S
    while True:
        current = read_counters()
        delta = {key: current[key] - baseline[key] for key in COUNTERS}
        if delta == expected or time.monotonic() > deadline:
            return delta
        time.sleep(1)


def start_server(log_path: Path) -> subprocess.Popen[bytes]:
    """Start the LMCache MP server and wait until /healthcheck reports healthy.

    Raises:
        RuntimeError: If the ports are already served, or the server exits or
            does not become healthy in time.
    """
    if _healthy():
        raise RuntimeError(
            f"something already serves localhost:{HTTP_PORT}; stop it first"
        )
    command = [
        sys.executable, "-m", "lmcache.v1.multiprocess.http_server",
        "--host", "localhost",
        "--port", str(MP_PORT),
        "--http-host", "localhost",
        "--http-port", str(HTTP_PORT),
        "--chunk-size", str(CHUNK_SIZE),
        # Host RAM for KV chunks (L1). Required: there is no default.
        "--l1-size-gb", "4",
        # Required: there is no default.
        "--eviction-policy", "LRU",
        # Neuron supports only engine-driven transfers. The default,
        # lmcache_driven, rejects the Neuron connector's registration.
        "--supported-transfer-mode", "engine_driven",
    ]  # fmt: skip
    server = subprocess.Popen(
        command, stdout=log_path.open("w"), stderr=subprocess.STDOUT
    )
    deadline = time.monotonic() + SERVER_START_TIMEOUT_S
    while not _healthy():
        if server.poll() is not None:
            raise RuntimeError(
                f"server exited with code {server.returncode}; see {log_path}"
            )
        if time.monotonic() > deadline:
            server.kill()
            raise RuntimeError(
                f"server not healthy after {SERVER_START_TIMEOUT_S}s; see {log_path}"
            )
        time.sleep(1)
    return server


def _healthy() -> bool:
    """Return True if the server's /healthcheck answers 200."""
    try:
        with urllib.request.urlopen(
            f"http://localhost:{HTTP_PORT}/healthcheck", timeout=2
        ):
            return True
    except (urllib.error.URLError, ConnectionError):
        return False


def run_engine_process(model: str, log_dir: Path, index: int) -> EngineResult:
    """Run one engine in a new process and return what it wrote.

    Raises:
        RuntimeError: If the engine fails, or LMCache in it did not detect Neuron.
    """
    log_path = log_dir / f"engine{index}.log"
    result_path = log_dir / f"engine{index}.json"
    result_path.unlink(missing_ok=True)
    with log_path.open("w") as log:
        returncode = subprocess.run(
            [
                sys.executable,
                __file__,
                "--model",
                model,
                "--engine-result",
                str(result_path),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        ).returncode
    log_text = log_path.read_text()
    if not result_path.exists():
        raise RuntimeError(
            f"engine {index} produced no result (exit {returncode}); see {log_path}"
        )
    if returncode != 0:
        raise RuntimeError(f"engine {index} exited with {returncode}; see {log_path}")
    # LMCache logs its device choice when the engine process first imports it.
    # vLLM forks the EngineCore worker that moves the KV (its default
    # VLLM_WORKER_MULTIPROC_METHOD is fork), so the worker inherits this choice.
    if (
        "torch_device_type=neuron" not in log_text
        or "torch_device_type=cpu" in log_text
    ):
        raise RuntimeError(
            f"LMCache in engine {index} did not detect Neuron; see {log_path}"
        )
    result: EngineResult = json.loads(result_path.read_text())
    print(f"engine {index}: generated {result['text']!r}")
    return result


def report(rows: dict[str, tuple[tuple[float, float], tuple[int, int]]]) -> bool:
    """Print observed and expected values per engine; return True if all match.

    Args:
        rows: Maps a row label to ``((engine 1, engine 2) observed,
            (engine 1, engine 2) expected)``.
    """
    print(f"\n{'per-engine value':<30}{'engine 1':>10}{'engine 2':>10}{'expected':>12}")
    ok = True
    for label, (observed, expected) in rows.items():
        row_ok = observed == expected
        ok &= row_ok
        values = "".join(f"{value:>10.0f}" for value in observed)
        wanted = " / ".join(str(value) for value in expected)
        print(f"{label:<30}{values}{wanted:>12}{'' if row_ok else '   MISMATCH'}")
    return ok


def main() -> int:
    """Run the server and both engines, then check counters and outputs."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default=DEFAULT_MODEL, help="model path or HF id")
    parser.add_argument("--log-dir", type=Path, default=Path("lmcache_repro_logs"))
    parser.add_argument("--engine-result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.engine_result is not None:  # child process: one engine run
        run_engine(args.model, args.engine_result)
        return 0

    args.log_dir.mkdir(parents=True, exist_ok=True)
    server = start_server(args.log_dir / "server.log")
    try:
        baseline = read_counters()
        result_1 = run_engine_process(args.model, args.log_dir, 1)
        delta_1 = wait_for_counters(baseline, EXPECTED_ENGINE_1)
        after_1 = {key: baseline[key] + delta_1[key] for key in COUNTERS}
        result_2 = run_engine_process(args.model, args.log_dir, 2)
        delta_2 = wait_for_counters(after_1, EXPECTED_ENGINE_2)
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()

    rows = {
        f"server {key}": (
            (delta_1[key], delta_2[key]),
            (EXPECTED_ENGINE_1[key], EXPECTED_ENGINE_2[key]),
        )
        for key in COUNTERS
    }
    rows["vllm cached prompt tokens"] = (
        (result_1["num_cached_tokens"], result_2["num_cached_tokens"]),
        EXPECTED_CACHED_TOKENS,
    )
    values_ok = report(rows)
    outputs_ok = result_1["token_ids"] == result_2["token_ids"]
    print(f"\nengine outputs identical: {outputs_ok}")
    passed = values_ok and outputs_ok
    print(f"RESULT: {'PASS' if passed else 'FAIL'}  (logs: {args.log_dir})")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
