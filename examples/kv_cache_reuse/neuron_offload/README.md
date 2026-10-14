# KV cache offload on AWS Trainium

`neuron_offload.py` shows LMCache moving vLLM KV cache from Trainium HBM to host RAM and back, using the multiprocess (MP) server.

It starts one LMCache MP server, then runs the same 1025-token prompt in two vLLM engines, one after the other, each in a new process. Engine 1 writes the prompt KV to the server. Engine 2's own prefix cache starts empty, so the prompt KV it reuses must come from LMCache. The script passes only if the server counters and vLLM's cached-token count match exactly and both engines generate the same tokens.

## Requirements

- A trn2 instance running `public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.24.0.1.1.0-neuronx-py313-sdk2.32.0-ubuntu24.04`.
- Access to `meta-llama/Llama-3.2-1B-Instruct`, or a local copy passed with `--model PATH`.

## Install

```bash
pip install "grpcio-tools==$(python -c 'import grpc; print(grpc.__version__)')"
pip install --no-deps --no-build-isolation "lmcache @ git+https://github.com/LMCache/LMCache.git@dev"
pip install --no-deps sortedcontainers "opentelemetry-exporter-prometheus==0.65b0"
```

`--no-deps` keeps the image's torch 2.11.0, which the Neuron backend is built against. A normal install replaces it. The first and last lines add the only build and runtime packages the image lacks, at versions that match the image's grpcio and OpenTelemetry SDK.

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

Each engine log ends with `RuntimeError: ... Allocator for neuron is not a DeviceAllocator`. vLLM's shutdown raises it after the engine's work is done. The script accepts it and no other engine failure.

`sample_run/` is a passing run on trn2.48xlarge with the image above.
