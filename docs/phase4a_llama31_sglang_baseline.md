# Phase 4A: Llama 3.1 8B Standalone SGLang Baseline

Validated: 2026-10-06

Phase 4 marks a deliberate model transition. Phases 1-3 used `Qwen/Qwen3-0.6B` because it was small enough for fast controlled learning experiments around standalone SGLang serving, mixed prefill/decode interference, and chunked-prefill behavior. Phase 4+ uses `meta-llama/Llama-3.1-8B-Instruct` as a more representative inference workload for the later Dynamo aggregated and P/D experiments.

Do not numerically compare Phase 4A absolute latency or throughput values against the earlier Qwen3-0.6B measurements. The model, tokenizer, parameter count, memory footprint, and runtime behavior changed.

## Research Role

Phase 4A answers only one functional question:

Can standalone aggregated SGLang 0.5.21 load and serve Llama 3.1 8B Instruct on the H100 NVL using the Phase 4 configuration, while preserving the request/streaming/measurement path needed for later experiments?

This is not a performance study. The single-request and concurrency-1 measurements below must not be generalized as H100/Llama performance.

## Architecture

```text
Client
  |
  v
SGLang 0.5.21 OpenAI-compatible API
  |
  v
meta-llama/Llama-3.1-8B-Instruct
  |
  v
1x NVIDIA H100 NVL
  |
  +-- Prefill
  +-- Decode
```

Prefill and decode are colocated in one SGLang worker on one GPU. KV cache remains local to that worker/GPU. Dynamo is not installed or running, and P/D disaggregation is not configured.

## Exact Configuration

- GPU: `NVIDIA H100 NVL`, `95830 MiB`
- Driver: `580.159.04`
- Python: `3.12.3`
- SGLang: `0.5.21`
- Torch: `2.13.0+cu130`
- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Served model: `llama3.1-8b-instruct`
- Tensor parallelism: `--tp-size 1`
- Weight/activation dtype: `--dtype bfloat16`
- Configured per-request context length: `--context-length 16384`
- Static memory fraction: `--mem-fraction-static 0.704`
- Maximum prefill tokens: `--max-prefill-tokens 16384`
- Scheduler policy: `--schedule-policy fcfs`
- Chunked prefill: `--chunked-prefill-size 4096`
- Streaming interval: `--stream-interval 1`
- Observed attention backend: `fa3`

`chunked_prefill_size=4096` is held fixed for continuity with the current aggregated baseline. It is not claimed to be optimal for Llama 3.1 8B.

Observed server configuration is preserved in `outputs/phase4a_llama31_8b_aggregated/server_args_observed.txt`.

## CUDA Runtime Repair

SGLang 0.5.21 installed an isolated CUDA 13 user-space stack inside `.venv`. The pod's default `/usr/local/cuda/bin/nvcc` was CUDA 12.8, but SGLang's DeepGEMM/JIT path required CUDA NVCC >= 12.9. No NVIDIA driver change was required.

The successful launch used the venv-local CUDA compiler:

```text
.venv/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc
Cuda compilation tools, release 13.4, V13.4.92
```

The Python CUDA package provided `cu13/lib`, while SGLang's JIT link command searched `cu13/lib64`. A venv-local compatibility link was required:

```text
.venv/lib/python3.12/site-packages/nvidia/cu13/lib64 -> lib
.venv/lib/python3.12/site-packages/nvidia/cu13/lib/libcudart.so -> libcudart.so.13
```

This repair is local to the virtual environment and does not modify the host NVIDIA driver.

## Startup And Memory Observations

- Startup-to-ready: `70.26 s`
- Weight load from local Hugging Face snapshot: `2.97 s`
- SGLang-reported model weight memory: `15.02 GB`
- KV cache pool capacity: `409738` tokens
- K cache allocation: `25.01 GB`
- V cache allocation: `25.01 GB`
- GPU memory after load: about `68832 MiB` used / `26488 MiB` free

The `409738` token value is SGLang's allocated KV pool capacity across serving. It is not a single-request context limit. The configured per-request context length for this Phase 4A baseline is `16384`.

## Functional Validation

`/v1/models` returned:

- model id: `llama3.1-8b-instruct`
- max model length: `16384`

Non-streaming request:

- prompt tokens: `50`
- completion tokens: `38`
- total tokens: `88`
- finish reason: `stop`
- client-observed latency: `0.502 s`

Streaming request with usage enabled:

- prompt tokens: `54`
- completion tokens: `96`
- total tokens: `150`
- finish reason: `length`
- TTFT: `0.0550 s`
- TPOT: `0.00688 s`
- end-to-end latency: `0.709 s`
- client-observed inter-stream-event p50: `0.00690 s`
- client-observed inter-stream-event p95: `0.00706 s`
- client-observed inter-stream-event max: `0.00749 s`

The inter-stream-event timings are client-observed streaming event gaps. They are useful for continuity with prior instrumentation, but they are not guaranteed to be GPU-level per-token ITL.

## Benchmark Instrumentation Validation

The existing `benchmarks/benchmark_sglang.py` client was run for one minimal baseline request at concurrency 1. This validates that the Phase 2/3 instrumentation still captures the required fields with the Llama model.

Minimal benchmark result:

- prompt tokens: `226`
- completion tokens: `128`
- total tokens: `354`
- TTFT: `0.0685 s`
- request TPOT: `0.00696 s`
- latency: `0.953 s`
- inter-chunk p50: `0.00699 s`
- inter-chunk p95: `0.00707 s`
- aggregate output throughput: `133.87 tok/s`
- requests/s: `1.046`

Because this was one request at concurrency 1, it should be treated only as instrumentation validation. It does not establish throughput, saturation, tail latency, or representative Llama 3.1 8B serving performance.

## Preserved Artifacts

Intentional Phase 4A artifacts are under `outputs/phase4a_llama31_8b_aggregated/`:

- `environment.txt`
- `gpu_after_load.csv`
- `model_config_metadata.json`
- `models.json`
- `non_stream_response.json`
- `server_args_observed.txt`
- `stream_events.json`
- `stream_summary.json`
- `benchmark_minimal/`

Runtime logs and PID files are intentionally not committed because repository policy ignores `*.log` and `*.pid`. Hugging Face cache contents, model weights, `.venv`, credentials, and tokens are not committed.

## What Phase 4A Proved / Did Not Prove

Phase 4A proved:

- the H100 NVL environment can load Llama 3.1 8B Instruct through standalone SGLang 0.5.21;
- the OpenAI-compatible non-streaming request path works;
- the OpenAI-compatible streaming request path works;
- prompt/completion token counts are returned;
- TTFT, TPOT, client-observed stream-event gaps, and end-to-end latency can be captured;
- the existing benchmark client can run against the Llama baseline.

Phase 4A did not prove:

- representative Llama 3.1 8B latency or throughput;
- concurrency scaling;
- GPU saturation;
- memory-bandwidth saturation;
- compute-bound or memory-bound behavior;
- that `chunked_prefill_size=4096` is optimal;
- any benefit or overhead from Dynamo;
- any benefit or overhead from P/D disaggregation.

## Next Step

Phase 4B should introduce Dynamo around SGLang while keeping the architecture aggregated and one-GPU. The purpose is to measure what Dynamo adds to an otherwise standalone aggregated SGLang serving path before introducing P/D disaggregation.
