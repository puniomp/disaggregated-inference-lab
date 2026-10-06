# Disaggregated Inference Lab

Hands-on lab for learning modern LLM serving architecture through incremental experiments.

Current scope: Phase 0 through Phase 5.

- Phase 0 validates the architecture vocabulary and upstream documentation assumptions.
- Phase 1 runs an aggregated SGLang baseline on one NVIDIA GPU.
- Phase 2 benchmarks that aggregated SGLang baseline under different workloads and concurrency levels.
- Phase 3 measures mixed prefill/decode interference and chunked-prefill tuning. The RTX 4090 run is preserved as historical evidence, and the H100 NVL chunk-size sweep is the controlled baseline for later distributed-serving work.
- Phase 4A transitions to `meta-llama/Llama-3.1-8B-Instruct` and validates a standalone one-GPU aggregated SGLang baseline.
- Phase 4B introduces Dynamo in front of SGLang while keeping serving aggregated on one H100.
- Phase 5 validates the first two-GPU Dynamo + SGLang P/D request with prefill on GPU 0, decode on GPU 1, and the configured NIXL/UCX KV handoff path.

Next step is Phase 6, a controlled aggregated-vs-P/D comparison.

This repository intentionally does not yet include P/D performance results or Kubernetes implementations.

## Validated Environment

Phase 1 and Phase 2 were run on RunPod with:

- 1x NVIDIA GeForce RTX 4090
- 24 GB VRAM
- SGLang 0.5.21
- Model: `Qwen/Qwen3-0.6B`
- Served model name: `qwen3-0.6b`

Phase 3 also includes a controlled H100 NVL chunk-size sweep for the later aggregated-vs-disaggregated comparison:

- 1x NVIDIA H100 NVL
- 95830 MiB VRAM
- Driver `580.159.04`, CUDA `13.0`
- SGLang `0.5.21`
- Model: `Qwen/Qwen3-0.6B`

Phase 4A establishes the new model baseline for distributed-serving work:

- 1x NVIDIA H100 NVL
- 95830 MiB VRAM
- Driver `580.159.04`
- SGLang `0.5.21`
- Torch `2.13.0+cu130`
- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Served model name: `llama3.1-8b-instruct`

The validated architecture is aggregated inference:

```text
Client
  |
  v
SGLang OpenAI-compatible API
  |
  v
One GPU
  |
  +-- Prefill
  +-- Decode
```

In this setup, prefill and decode run in the same SGLang worker on the same GPU. KV cache remains local to that worker/GPU, so no cross-GPU KV transfer is required.

## Repository Layout

```text
benchmarks/                 Phase 2 benchmark client, Phase 3 interference client, and chart generator
configs/                    Workload profile definitions
docs/                       Architecture and phase documentation
outputs/                    Preserved Phase 2 benchmark artifacts
scripts/                    Environment, server, smoke test, and benchmark scripts
```

Important documents:

- `docs/architecture.md`: Phase 0 architecture validation.
- `docs/phase1_sglang_baseline.md`: Phase 1 setup and validated run notes.
- `docs/phase2_sglang_benchmarking.md`: Phase 2 methodology, measured results, interpretations, and limitations.
- `docs/phase3_chunked_prefill_interference.md`: Phase 3 prefill/decode interference experiment and matched chunked-prefill A/B.
- `docs/phase4a_llama31_sglang_baseline.md`: Phase 4A standalone Llama 3.1 8B aggregated SGLang baseline.
- `docs/phase4b_dynamo_sglang_aggregated.md`: Phase 4B Dynamo + SGLang aggregated functional baseline.

## Preserved Phase 2 Outputs

The committed benchmark artifacts are intentionally small enough for normal GitHub source control.

- `outputs/phase2_quick`: initial quick benchmark.
- `outputs/phase2_baseline_probe`: baseline concurrency probe.
- `outputs/phase2_prefill_probe`: prefill-heavy concurrency probe.
- `outputs/phase2_decode_probe`: decode-heavy probe.
- `outputs/repro`: RunPod reproducibility metadata.

Each benchmark output directory contains raw request records, grouped summaries, run metadata, and SVG charts.

## Reproducing The Current Setup

On a fresh RunPod Linux instance with one NVIDIA GPU:

```bash
git clone https://github.com/puniomp/disaggregated-inference-lab.git
cd disaggregated-inference-lab

bash scripts/check_env.sh

uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install --prerelease=allow sglang

bash scripts/start_sglang.sh
```

Wait for SGLang to report that the server is ready, then run:

```bash
bash scripts/smoke_test.sh
```

For Phase 2 benchmarks, use:

```bash
bash scripts/phase2_quick_benchmark.sh
```

or a targeted run such as:

```bash
PROFILES=baseline \
CONCURRENCY=1,2,4,8 \
REQUESTS_PER_CONCURRENCY=16 \
OUT_DIR=outputs/phase2_baseline_probe \
bash scripts/run_phase2_benchmarks.sh
```

## Phase 2 Status

Completed probes:

- baseline
- prefill-heavy
- decode-heavy

The original decode-heavy probe naturally stopped at roughly 273-283 completion tokens, so a revised sustained-decode workload was added and validated at 2000 completion tokens.

## Phase 3 Status

Phase 3 is complete. The original RTX 4090 matched A/B remains historical evidence:

- No-chunk baseline: `--chunked-prefill-size -1`
- Chunked condition: `--chunked-prefill-size 4096`
- Workload: four active sustained-decode requests, then one 10285-token long-prefill request injected at about 1.508 seconds

Main result: chunked prefill reduced during-prefill p95/p99/max inter-stream-event gaps by about 34-35%, but it did not improve the worst synchronized stall. This suggests chunking mitigated part of the interference but did not fully isolate decode token delivery in the aggregated worker.

The new H100 NVL baseline sweep reran all chunked-prefill configurations fresh: `-1`, `8192`, `4096`, `2048`, and `1024`. Throughput stayed roughly `1527-1538` output tokens/sec. The relationship was non-monotonic: `4096` produced the lowest injected-request TTFT/latency, `2048` produced the lowest during-prefill p99/max among chunked configurations, and smaller chunks were not always better.

Do not compare RTX 4090 and H100 absolute latency numbers as a controlled hardware comparison.

Do not infer GPU saturation, memory-bandwidth saturation, compute-bound behavior, or P/D disaggregation benefit from the current results alone.


## Phase 4A Status

Phase 4A is complete. It marks the deliberate transition from `Qwen/Qwen3-0.6B` to `meta-llama/Llama-3.1-8B-Instruct` for the remaining distributed-serving experiments.

Validated architecture:

```text
Client -> SGLang 0.5.21 -> Llama 3.1 8B Instruct -> 1x H100 NVL
```

Prefill and decode remain colocated in one SGLang worker on one GPU. Dynamo is not installed or running, and P/D disaggregation is not configured.

This is a functional baseline only. The single-request/concurrency-1 validation proves model load, OpenAI-compatible request handling, streaming, usage accounting, and benchmark instrumentation continuity. It is not a performance study, and its absolute values should not be compared against the earlier Qwen3-0.6B runs.

## Phase 4B Status

Phase 4B is complete. It adds Dynamo as the OpenAI-compatible frontend, discovery layer, and router in front of a single aggregated SGLang worker:

```text
Client -> Dynamo frontend -> SGLang worker -> Llama 3.1 8B Instruct -> 1x H100 NVL
```

Prefill and decode still run in the same SGLang worker on the same GPU. NIXL is installed in the Dynamo environment, but it is not part of this aggregated request path and no cross-GPU KV transfer occurs. The key validation signal is that Dynamo reported the same backend worker ID for prefill and decode on the functional request.

This is a functional serving checkpoint, not a performance comparison with Phase 4A.

## Phase 5 Status

Phase 5 is functionally validated. It separates prefill and decode across two H100 NVL GPUs:

```text
Client -> Dynamo frontend -> prefill router -> Prefill Worker on GPU 0 -> NIXL/UCX KV handoff -> Decode Worker on GPU 1 -> streamed response
```

The first request succeeded with distinct worker IDs:

- `prefill_worker_id`: `7537786647078909697`
- `decode_worker_id`: `7403880299791686667`

NIXL KV managers initialized on both workers with backend `UCX`, and the request returned HTTP 200 with generated output. The captured evidence does not include an explicit per-request `NIXL transfer completed` event or byte-count-level transfer telemetry, so the KV handoff claim is intentionally framed as functional evidence rather than byte-level transfer proof.

Phase 5 proves the lab can execute one real request with P/D physically separated across two H100 GPUs. It does not prove that P/D is faster than aggregated serving; that is the Phase 6 question.

See `docs/phase5_dynamo_sglang_pd_functional.md` for the full checkpoint.
