# KV cache offload on AWS Trainium

`neuron_offload.py` shows LMCache moving vLLM KV cache from Trainium HBM to host RAM and back, using the multiprocess (MP) server.

It starts one LMCache MP server, then runs the same 1025-token prompt in two vLLM engines, one after the other, each in a new process. Engine 1 writes the prompt KV to the server. Engine 2's own prefix cache starts empty, so the prompt KV it reuses must come from LMCache. The script passes only if the server counters and vLLM's cached-token count match exactly and both engines generate the same tokens.

## Requirements

- A trn2 instance running `public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04`.
- Access to `meta-llama/Llama-3.2-1B-Instruct`, or a local copy passed with `--model PATH`.

## Install

On vllm-neuron 0.29, LMCache needs the fix in commit "[Bugfix][Neuron] Treat vllm-neuron's resolved LBNHC layout as HND". Without it, the first store fails with `Neuron block transfer requires a per-layer, two-major HND KV layout`. Until that fix is on `dev`, install from a branch that has it.

```bash
pip install --no-deps --no-build-isolation --ignore-requires-python "lmcache @ git+https://github.com/LMCache/LMCache.git@dev"
pip install --no-deps sortedcontainers "opentelemetry-exporter-prometheus==0.66b1"
```

`--no-deps` keeps the image's torch 2.13.0, which the Neuron backend is built against. A normal install replaces it. `--ignore-requires-python` is needed because the image ships Python 3.14 and LMCache declares `<3.14`. The second line adds the only runtime packages the image lacks, at the version that matches the image's OpenTelemetry SDK.

## Run

```bash
python neuron_offload.py
```

Expected output (exit code 0):

```
per-engine value                engine 1  engine 2    expected
server l1_write_chunks                 4         0       4 / 0
server l1_read_chunks                  0         4       0 / 4
server lookup_requested_tokens      1024      1024 1024 / 1024
server lookup_hit_tokens               0      1024    0 / 1024
server lookup_hit_l1_tokens            0      1024    0 / 1024
vllm cached prompt tokens              0      1024    0 / 1024

engine outputs identical: True
RESULT: PASS
```

The server logs each transfer in `server.log`: `Stored 1024 tokens` for engine 1, then `Retrieved 1024 tokens` for engine 2.

`sample_run/` is a passing run on trn2.48xlarge with the image above.
