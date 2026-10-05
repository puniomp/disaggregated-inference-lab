# Disaggregated Inference Lab

Hands-on lab for learning modern LLM serving architecture through incremental experiments.

Current scope: Phase 0 through Phase 3.

- Phase 0 validates the architecture vocabulary and upstream documentation assumptions.
- Phase 1 runs an aggregated SGLang baseline on one NVIDIA GPU.
- Phase 2 benchmarks that aggregated SGLang baseline under different workloads and concurrency levels.
- Phase 3 measures prefill/decode interference and compares no chunking against chunked prefill. The RTX 4090 run is preserved as historical evidence, and the H100 NVL chunk-size sweep is the new controlled baseline for later distributed-serving work.

This repository intentionally does not yet include Dynamo, NIXL, prefill/decode disaggregation, multi-GPU serving, or Kubernetes.

## Validated Environment

Phase 1 and Phase 2 were run on RunPod with:

- 1x NVIDIA GeForce RTX 4090
- 24 GB VRAM
- SGLang 0.5.21
- Model: `Qwen/Qwen3-0.6B`
- Served model name: `qwen3-0.6b`

Phase 3 now also includes a controlled H100 NVL baseline sweep for the later aggregated-vs-disaggregated comparison:

- 1x NVIDIA H100 NVL
- 95830 MiB VRAM
- Driver `580.159.04`, CUDA `13.0`
- SGLang `0.5.21`
- Model: `Qwen/Qwen3-0.6B`

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
