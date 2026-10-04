# Phase 2: SGLang Benchmarking

Phase 2 benchmarks the Phase 1 architecture only:

```
Client benchmark
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

Do not use Dynamo, NIXL, P/D disaggregation, Kubernetes, or Phase 3 chunked-prefill experiments in this phase.

## What Was Added

- `configs/workloads.json`: configurable workload profiles.
- `benchmarks/benchmark_sglang.py`: standard-library streaming benchmark client.
- `benchmarks/plot_phase2.py`: standard-library SVG chart generator.
- `scripts/run_phase2_benchmarks.sh`: full Phase 2 benchmark run.
- `scripts/phase2_quick_benchmark.sh`: small smoke benchmark before committing to the full run.

## Workload Profiles

The default profiles are:

- `baseline`: approximately 128 input tokens, 128 max output tokens.
- `prefill-heavy`: approximately 4096 input tokens, 128 max output tokens.
- `very-prefill-heavy`: approximately 8192 input tokens, 500 max output tokens.
- `decode-heavy`: approximately 128 input tokens, 2000 max output tokens.

Prompt length is approximate because the benchmark creates synthetic text and lets the model tokenizer decide the true token count. Use the server-returned `prompt_tokens`, when available, as the measured value.

## Metrics

`latency_s`

- Measured from immediately before the HTTP request is submitted until the streaming response completes.
- This is end-to-end client-observed request latency.

`ttft_s`

- Measured from immediately before request submission until the first streamed content or reasoning chunk arrives.
- This includes client-side HTTP overhead, server queueing, tokenization, prefill, and the first decode step.

`inter_chunk_latency_*`

- Measured from timestamps between streamed response chunks.
- Limitation: streamed chunks are not guaranteed to equal model tokens. A chunk can contain zero, one, or multiple tokens. Treat this as an approximation of ITL behavior, not a true per-token ITL distribution.

`request_tpot_s`

- Request-level average over the streamed generation interval.
- If `completion_tokens` is returned, it is calculated as `(last_chunk_time - first_chunk_time) / (completion_tokens - 1)`.
- If usage is unavailable, it falls back to emitted stream chunks.
- Limitation: request-averaged TPOT hides per-token jitter and is not a substitute for a true ITL distribution.

`aggregate_output_tokens_per_s`

- Sum of completion tokens for a profile/concurrency group divided by wall-clock time for that group.
- This is aggregate output throughput, not a proof of GPU saturation.

`requests_per_s`

- Successful requests divided by wall-clock time for a profile/concurrency group.

## What Each Experiment Teaches

`baseline`

- Establishes a sanity baseline for request latency, TTFT, TPOT, and throughput before stressing prefill or decode heavily.
- Useful for confirming the benchmark harness is working.

`prefill-heavy`

- Increases prompt-processing work and KV-cache creation while keeping generation short.
- Teaches how long prompts affect TTFT and queueing under concurrency.

`very-prefill-heavy`

- Pushes prompt length further and adds medium generation.
- Teaches how long prefills interact with batching, cache allocation, and request contention in the aggregated worker.

`decode-heavy`

- Keeps prompt short but requests long output.
- Teaches how sustained autoregressive decode affects TPOT, end-to-end latency, and aggregate output tokens/sec.

Concurrency levels:

- `1` shows single-request behavior with little queueing.
- `8`, `16`, and `32` introduce batching and contention.
- Higher throughput at higher concurrency does not by itself prove memory-bandwidth saturation; it only shows the observed end-to-end serving behavior for this workload and configuration.

## Reproduce on Fresh RunPod

From a clean RunPod shell:

```bash
cd /workspace/inference-disaggregation-lab

bash scripts/check_env.sh

uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install --prerelease=allow sglang

bash scripts/start_sglang.sh
tail -f logs/sglang.log
```

Wait until:

```text
The server is fired up and ready to roll!
```

In another terminal, run a quick benchmark first:

```bash
cd /workspace/inference-disaggregation-lab
source .venv/bin/activate
bash scripts/phase2_quick_benchmark.sh
```

Then run the full benchmark:

```bash
cd /workspace/inference-disaggregation-lab
source .venv/bin/activate
bash scripts/run_phase2_benchmarks.sh
```

Outputs:

- `outputs/phase2/raw_all.jsonl`: raw per-request records.
- `outputs/phase2/raw_all.csv`: raw per-request CSV.
- `outputs/phase2/summary.csv`: grouped summary table.
- `outputs/phase2/summary.json`: grouped summary JSON.
- `outputs/phase2/run_metadata.json`: benchmark settings and metric notes.
- `outputs/phase2/charts/*.svg`: comparison charts.

## Custom Runs

Run fewer requests:

```bash
REQUESTS_PER_CONCURRENCY=8 bash scripts/run_phase2_benchmarks.sh
```

Run selected profiles:

```bash
PROFILES=baseline,decode-heavy bash scripts/run_phase2_benchmarks.sh
```

Run selected concurrency levels:

```bash
CONCURRENCY=1,4,8 bash scripts/run_phase2_benchmarks.sh
```

Write to a different output directory:

```bash
OUT_DIR=outputs/phase2_experiment_a bash scripts/run_phase2_benchmarks.sh
```

## Stopping Point

Stop after Phase 2 results are generated. Review the raw records, summary table, and charts before deciding whether to proceed to Phase 3.
