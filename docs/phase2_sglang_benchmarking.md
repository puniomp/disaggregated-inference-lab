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

- Keeps prompt short but requests long structured output.
- Teaches how sustained autoregressive decode affects TPOT, end-to-end latency, and aggregate output tokens/sec.
- Uses a numbered-list prompt to encourage substantially longer generation. `max_tokens` remains an upper bound, not a guarantee that the model will generate that many tokens.

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

## Validated Phase 2 Runs

### Baseline Probe

Measured on the earlier Phase 2 RunPod RTX 4090 run with `Qwen/Qwen3-0.6B`, `16` requests per concurrency, and output capped at `128` tokens.

| Concurrency | TTFT p50 (s) | TTFT p95 (s) | TPOT p50 (s) | TPOT p95 (s) | E2E p50 (s) | E2E p95 (s) | Output tok/s | Req/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0201 | 0.0409 | 0.00187 | 0.00187 | 0.2578 | 0.2791 | 485.9 | 3.80 |
| 2 | 0.0543 | 0.0556 | 0.00217 | 0.00218 | 0.3301 | 0.3318 | 774.0 | 6.05 |
| 4 | 0.0458 | 0.0473 | 0.00221 | 0.00223 | 0.3274 | 0.3288 | 1558.6 | 12.18 |
| 8 | 0.0505 | 0.0518 | 0.00233 | 0.00236 | 0.3472 | 0.3485 | 2923.2 | 22.84 |

Measured facts: all baseline requests succeeded, completion tokens averaged `128`, and aggregate output throughput increased from concurrency `1` through `8`.

Reasonable interpretation: this workload is short enough that SGLang can improve aggregate throughput substantially with concurrency while keeping request latency low. The small TPOT increase suggests additional batching/scheduling work per request as concurrency rises.

Cannot conclude: these measurements do not prove GPU saturation, memory-bandwidth saturation, compute-bound behavior, or the cause of any latency variation.

### Prefill-Heavy Probe

Measured on the earlier Phase 2 RunPod RTX 4090 run with `Qwen/Qwen3-0.6B`, approximately `5162` input tokens observed in the raw records, `128` output tokens, and `16` requests per concurrency.

| Concurrency | TTFT p50 (s) | TTFT p95 (s) | TPOT p50 (s) | TPOT p95 (s) | E2E p50 (s) | E2E p95 (s) | Output tok/s | Req/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.1040 | 0.2046 | 0.00244 | 0.00245 | 0.4141 | 0.5149 | 298.3 | 2.33 |
| 2 | 0.0797 | 0.0834 | 0.00337 | 0.00338 | 0.5085 | 0.5123 | 502.4 | 3.93 |
| 4 | 0.1132 | 0.1274 | 0.00482 | 0.00505 | 0.7248 | 0.7291 | 704.0 | 5.50 |
| 8 | 0.1325 | 0.1488 | 0.00760 | 0.00829 | 1.0976 | 1.1710 | 900.5 | 7.03 |

Measured facts: compared with baseline, the prefill-heavy workload had higher TTFT, higher TPOT, higher end-to-end latency, and lower request/output throughput at the same concurrency levels.

Reasonable interpretation: long prompts increased prefill work and KV-cache creation, and under concurrency this likely increased contention inside the single aggregated SGLang worker.

Cannot conclude: this does not isolate whether the limiting factor was attention compute, memory bandwidth, scheduler policy, tokenization overhead, KV allocation, or another subsystem.

### Initial Decode-Heavy Probe Limitation

The original decode-heavy probe used `max_tokens=2000`, but the prompt did not reliably produce long generations. The model naturally stopped around `273` to `283` completion tokens.

Measured facts: the saved `outputs/phase2_decode_probe` directory remains a valid experiment for that prompt, but it is not a sustained long-decode workload.

Reasonable interpretation: `max_tokens` is only an upper bound. It does not force the model to continue generating if the model emits a stop condition first.

Cannot conclude: the original decode-heavy results should not be used to reason about approximately `2000`-token sustained decode behavior.

### Revised Decode-Heavy Validation

The decode-heavy workload was changed to keep the prompt short while requesting long structured numbered output. The benchmark methodology was otherwise kept the same: streaming OpenAI-compatible requests, client-observed TTFT, request-averaged TPOT, end-to-end latency, output throughput, and request throughput.

Generation settings note: this change uses prompt design rather than `ignore_eos` or `min_new_tokens`. That avoids suppressing natural stop behavior or forcing continuation through a sampling control, but it also means the prompt content itself is now part of the workload definition. The revised prompt successfully reached the `max_tokens=2000` cap in validation and in the concurrency probe.

Validation run: `outputs/phase2_decode_long_validation`, concurrency `1`, `2` requests.

| Request | Prompt tokens | Completion tokens | Finish reason | TTFT (s) | TPOT (s) | E2E latency (s) |
|---:|---:|---:|---|---:|---:|---:|
| 0 | 123 | 2000 | length | 0.073 | 0.00197 | 4.005 |
| 1 | 123 | 2000 | length | 8.323 | 0.00197 | 12.268 |

Measured facts: both validation requests generated `2000` completion tokens and stopped because of the length cap. The second request had a high TTFT outlier.

Reasonable interpretation: the revised prompt is reliable enough to create a sustained decode workload, but outliers still need to be preserved and inspected rather than explained away.

Cannot conclude: the high validation TTFT outlier is not by itself enough to identify a root cause.

### Revised Decode-Heavy Probe

Measured on the new Phase 2 RunPod RTX 4090 run with `Qwen/Qwen3-0.6B`, `8` requests per concurrency, short prompt, and output capped at `2000` tokens. Raw results and charts are preserved in `outputs/phase2_decode_long_probe`.

| Concurrency | TTFT p50 (s) | TTFT p95 (s) | TPOT p50 (s) | TPOT p95 (s) | E2E p50 (s) | E2E p95 (s) | Output tok/s | Req/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0258 | 0.0727 | 0.00197 | 0.00197 | 3.9554 | 4.0044 | 504.2 | 0.25 |
| 2 | 0.0698 | 0.1615 | 0.00234 | 0.00238 | 4.7442 | 4.8782 | 837.1 | 0.42 |
| 4 | 0.0673 | 0.0733 | 0.00263 | 0.00263 | 5.3246 | 5.3282 | 1501.2 | 0.75 |
| 8 | 0.0648 | 0.0669 | 0.00323 | 0.00323 | 6.5147 | 6.5218 | 2450.5 | 1.23 |

Raw-row checks: `32` successful requests, `0` errors, prompt tokens averaged `123`, every request generated exactly `2000` completion tokens, and every request finished with `finish_reason=length`.

Measured facts: aggregate output throughput increased with concurrency, request throughput increased with concurrency, TPOT increased with concurrency, and end-to-end latency increased with concurrency. TTFT stayed below `0.162` seconds p95 in the probe and the earlier approximately `8` second validation outlier did not repeat during the 32-request probe.

Reasonable interpretation: batching more concurrent long-decode requests improved aggregate output throughput but increased per-request decode time and end-to-end latency. The rising TPOT suggests each request receives tokens less frequently as more requests share the worker/GPU decode loop.

Cannot conclude: this does not prove GPU saturation, memory-bandwidth saturation, compute-bound behavior, exact scheduler causality, or that the system has reached maximum throughput. Those claims require profiling and additional experiments.

### Cross-Workload Observation

Measured facts: at concurrency `8`, baseline reached about `2923` output tokens/sec with `128` output tokens/request; prefill-heavy reached about `900` output tokens/sec with long prompts and `128` output tokens/request; revised decode-heavy reached about `2451` output tokens/sec with short prompts and `2000` output tokens/request.

Reasonable interpretation: the single-worker aggregated setup benefits from concurrency for all measured workloads, but long prefill and long decode stress different parts of the request lifecycle. Prefill-heavy mainly inflated TTFT and overall latency for short generations; revised decode-heavy mainly inflated end-to-end latency through sustained token-by-token generation while keeping TTFT comparatively low in the final probe.

Cannot conclude: these comparisons do not yet tell us when P/D disaggregation is worth its KV-transfer and distributed-systems overhead. They only establish the aggregated single-GPU baseline behavior needed before asking that question.

### TODO For Next Session

- Consider whether to run `very-prefill-heavy` after reviewing the current baseline, prefill-heavy, and revised decode-heavy results.
- Preserve the current decode-heavy prompt as the first sustained-decode baseline before trying alternative generation controls such as `min_new_tokens` or `ignore_eos`.
- If investigating outliers, collect server-side timing/profiling data rather than inferring root cause from client metrics alone.

## Stopping Point

Stop after Phase 2 results are generated. Review the raw records, summary table, and charts before deciding whether to proceed to Phase 3.
