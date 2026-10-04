# Disaggregated Inference Lab

Hands-on lab for learning modern LLM serving architecture through incremental experiments.

Current scope: Phase 0 through Phase 2 only.

- Phase 0 validates the architecture vocabulary and upstream documentation assumptions.
- Phase 1 runs an aggregated SGLang baseline on one NVIDIA GPU.
- Phase 2 benchmarks that aggregated SGLang baseline under different workloads and concurrency levels.

This repository intentionally does not yet include Phase 3, Dynamo, NIXL, prefill/decode disaggregation, multi-GPU serving, or Kubernetes.

## Validated Environment

Phase 1 and Phase 2 were run on RunPod with:

- 1x NVIDIA GeForce RTX 4090
- 24 GB VRAM
- SGLang 0.5.21
- Model: `Qwen/Qwen3-0.6B`
- Served model name: `qwen3-0.6b`

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
benchmarks/                 Phase 2 benchmark client and chart generator
configs/                    Workload profile definitions
docs/                       Architecture and phase documentation
outputs/                    Preserved Phase 2 benchmark artifacts
scripts/                    Environment, server, smoke test, and benchmark scripts
```

Important documents:

- `docs/architecture.md`: Phase 0 architecture validation.
- `docs/phase1_sglang_baseline.md`: Phase 1 setup and validated run notes.
- `docs/phase2_sglang_benchmarking.md`: Phase 2 methodology, measured results, interpretations, and limitations.

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

Important limitation: the decode-heavy profile used `max_tokens=2000`, but the model naturally stopped at roughly 273-283 completion tokens. That result is preserved as a valid short-to-medium generation experiment, but it does not test sustained 2000-token decode behavior.

Next Phase 2 TODO:

- Redesign the decode-heavy workload so it reliably produces a much longer generation before rerunning it.

Do not infer GPU saturation, memory-bandwidth saturation, compute-bound behavior, chunked-prefill benefit, or P/D disaggregation benefit from the current Phase 2 results alone.
