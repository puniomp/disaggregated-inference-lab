# Phase 3: Chunked-Prefill Interference

## Experimental Question

Phase 3 asks:

> When long-prefill work arrives while other requests are actively decoding on the same aggregated SGLang worker/GPU, does it interfere with their token delivery, and can chunked prefill mitigate that interference?

This phase stays within the single-worker, single-GPU SGLang architecture. It does not use Dynamo, NIXL, P/D disaggregation, multi-GPU serving, or Kubernetes.

## Environment

### Historical RTX 4090 A/B

- SGLang version: `0.5.21`
- Model: `Qwen/Qwen3-0.6B`
- Served model name: `qwen3-0.6b`
- GPU: `NVIDIA GeForce RTX 4090`, `24564 MiB`, driver `580.65.06`
- Architecture: one SGLang worker on one GPU; prefill and decode colocated

The RTX 4090 experiment remains historical Phase 3 evidence. It demonstrated the prefill/decode interference phenomenon and the initial matched no-chunk vs chunked-prefill A/B result.

### Controlled H100 NVL Baseline

The later distributed-serving experiments will require H100-class hardware, so the chunk-size sweep was rerun as a new controlled baseline on H100 NVL. Do not numerically combine these H100 measurements with the earlier RTX 4090 measurements or present their absolute latencies as a controlled hardware comparison.

- SGLang version: `0.5.21`
- Model: `Qwen/Qwen3-0.6B`
- Served model name: `qwen3-0.6b`
- GPU: `NVIDIA H100 NVL`, `95830 MiB`, driver `580.159.04`
- CUDA shown by `nvidia-smi`: `13.0`
- PyTorch: `2.13.0+cu130`
- PyTorch CUDA: `13.0`
- `torch.cuda.is_available()`: `True`
- NVCC used for SGLang JIT: Python-installed CUDA `13.4`, `V13.4.92`
- Architecture: one SGLang worker on one GPU; prefill and decode colocated

Environment note: the pod's system `nvcc` was CUDA `12.8`, while SGLang `0.5.21` used CUDA 13 packages. The H100 rerun therefore used the Python-installed CUDA 13 `nvcc` and local virtualenv CUDA library paths for SGLang JIT linking. This environment repair is recorded in `outputs/phase3_chunk_sweep_h100_baseline_rerun/environment.txt`.

## SGLang Chunked-Prefill Configuration

The installed SGLang CLI help says `--chunked-prefill-size` controls the maximum number of tokens in a prefill chunk, and `-1` disables chunked prefill. The current official SGLang server-arguments documentation states the same behavior.

The Phase 2 server, when launched without an explicit chunked-prefill flag, logged:

- `chunked_prefill_size: 4096`
- `max_prefill_tokens: 16384`
- `enable_mixed_chunk: False`
- `schedule_policy: fcfs`
- `schedule_conservativeness: 1.0`
- `max_running_requests: None`
- `max_total_tokens: None`
- `mem_fraction_static: 0.704`

The matched A/B therefore used:

- No-chunk baseline: `--chunked-prefill-size -1`
- Chunked condition: `--chunked-prefill-size 4096`

All other relevant settings were held fixed:

- `--mem-fraction-static 0.704`
- `--max-prefill-tokens 16384`
- `--schedule-policy fcfs`
- `--schedule-conservativeness 1.0`
- `enable_mixed_chunk: False`
- same model, GPU, prompts, output limits, injection delay, and request counts

## Controlled Workload

The interference benchmark intentionally does not start all requests simultaneously.

Timeline:

1. Start `4` short-prompt, long-output background requests.
2. Let them enter sustained decode.
3. At about `1.508s` after experiment start, inject one long-prompt, short-output request.
4. Measure the background decode streams before, during, and after the injected request's TTFT window.

Measured workload details:

- Background decode requests: `4`
- Background prompt tokens: `115` per request
- Background completion tokens: `2000` per request
- Background finish reasons: all `length`
- Injected request prompt tokens: `10285`
- Injected request completion tokens: `64`
- Injected request finish reason: `length`
- Injection delay target: `1.5s`
- Observed injection time: `1.508s` in both primary runs

## Measurement Methodology

`benchmarks/interference_sglang.py` records each streamed content/reasoning event with a client-observed arrival timestamp.

Preserved outputs for each run:

- `requests.jsonl` and `requests.csv`: per-request TTFT, latency, request-level TPOT, token counts, finish reason.
- `stream_events.jsonl` and `stream_events.csv`: per-stream-event arrival time and gap from the previous event in the same request.
- `phase_summary.csv`: background decode inter-event gap summaries before, during, and after the injected request's TTFT window.
- `itl_aligned_to_injection.svg`: visualization of background decode inter-event gaps aligned to injection time.
- `repro.txt`: configuration and environment metadata.

Measurement limitation: ITL is a client-observed inter-stream-event proxy. The OpenAI-compatible stream does not provide a formal GPU-side per-token trace. A streaming event can contain a token-like piece, partial text, or multiple characters/pieces. SGLang was using `stream_interval=1`, so this is close to token cadence for this setup, but the directly measured unit is still the streaming event gap.

## Preserved Output Directories

- `outputs/phase3_interference_baseline_no_chunked_controlled`: primary no-chunk baseline.
- `outputs/phase3_interference_chunked_4096`: matched chunked-prefill run.
- `outputs/phase3_interference_ab_comparison`: computed comparison metrics and aligned A/B chart.
- `outputs/phase3_chunk_sweep_h100_baseline_rerun`: controlled H100 NVL chunk-size sweep used as the new baseline for later distributed-serving work.

## Reproducibility Commands

No-chunk baseline:

```bash
SGLANG_EXTRA_ARGS="--chunked-prefill-size -1 --mem-fraction-static 0.704 --max-prefill-tokens 16384 --schedule-policy fcfs --schedule-conservativeness 1.0"   LOG_FILE="logs/sglang_phase3_no_chunked.log"   PID_FILE="logs/sglang_phase3_no_chunked.pid"   bash scripts/start_sglang.sh

python3 benchmarks/interference_sglang.py   --base-url http://localhost:30000   --out-dir outputs/phase3_interference_baseline_no_chunked_controlled   --server-config-label chunked_prefill_disabled_chunked_prefill_size_-1   --background-decode-requests 4   --background-max-output-tokens 2000   --injection-delay-s 1.5   --injected-prefill-terms 8192   --injected-max-output-tokens 64
```

Chunked-prefill `4096` run:

```bash
SGLANG_EXTRA_ARGS="--chunked-prefill-size 4096 --mem-fraction-static 0.704 --max-prefill-tokens 16384 --schedule-policy fcfs --schedule-conservativeness 1.0"   LOG_FILE="logs/sglang_phase3_chunked_4096.log"   PID_FILE="logs/sglang_phase3_chunked_4096.pid"   bash scripts/start_sglang.sh

python3 benchmarks/interference_sglang.py   --base-url http://localhost:30000   --out-dir outputs/phase3_interference_chunked_4096   --server-config-label chunked_prefill_size_4096   --background-decode-requests 4   --background-max-output-tokens 2000   --injection-delay-s 1.5   --injected-prefill-terms 8192   --injected-max-output-tokens 64
```

## Matched A/B Results

These results are from the historical RTX 4090 matched A/B run. They should not be numerically combined with the H100 NVL chunk-size sweep.

Background decode inter-stream-event gap proxy:

| Phase | Metric | No Chunking | Chunked 4096 | Change |
|---|---:|---:|---:|---:|
| Before | p50 | 2.270 ms | 2.277 ms | +0.3% |
| Before | p95 | 2.425 ms | 2.423 ms | -0.1% |
| Before | p99 | 2.608 ms | 2.627 ms | +0.7% |
| Before | max | 12.711 ms | 12.917 ms | +1.6% |
| During prefill proxy | p50 | 0.575 ms | 0.566 ms | -1.5% |
| During prefill proxy | p95 | 48.284 ms | 31.328 ms | -35.1% |
| During prefill proxy | p99 | 80.379 ms | 52.598 ms | -34.6% |
| During prefill proxy | max | 80.559 ms | 53.105 ms | -34.1% |
| After | p50 | 2.793 ms | 2.803 ms | +0.3% |
| After | p95 | 3.492 ms | 3.400 ms | -2.6% |
| After | p99 | 3.896 ms | 3.903 ms | +0.2% |
| After | max | 145.569 ms | 172.563 ms | +18.5% |

Key result: chunked prefill at `4096` reduced during-prefill p95/p99/max inter-stream-event gaps by about `34-35%`.

Important limitation: the worst synchronized stall was **not** improved. The largest aligned stall increased from `145.6 ms` in the no-chunk baseline to `172.6 ms` in the chunked run, occurring just after the injected request's first streamed token.

Request-level TPOT for background decode requests:

- No chunking: `2.780`, `2.780`, `2.783`, `2.779` ms
- Chunked `4096`: `2.763`, `2.763`, `2.763`, `2.764` ms

Injected long-prefill request:

| Metric | No Chunking | Chunked 4096 | Change |
|---|---:|---:|---:|
| TTFT | 283.790 ms | 266.222 ms | -6.2% |
| Latency | 524.530 ms | 508.064 ms | -3.1% |
| TPOT | 3.812 ms | 3.832 ms | +0.5% |

Aggregate output throughput:

- No chunking: `1395.2` output tokens/sec
- Chunked `4096`: `1406.8` output tokens/sec
- Change: `+0.8%`

The TTFT and throughput differences are small single-run differences. They are recorded, but should not be overinterpreted without repetitions.

Largest synchronized stalls:

- No chunking: around `+50 ms` after injection, gaps near `48 ms`; around `+142 ms`, gaps near `80 ms`; around `+287 ms`, gaps near `145 ms` just after the injected first token.
- Chunked `4096`: around `+55 ms`, gaps near `52-53 ms`; around `+97 ms`, gaps near `31 ms`; around `+270 ms`, gaps near `172 ms` just after the injected first token.

## H100 NVL Chunk-Size Sweep

### Experimental Matrix

The H100 NVL sweep reran all configurations fresh on the same pod and treats `chunked-prefill-size` as the independent variable:

| Configuration | Launch argument |
|---|---|
| Disabled | `--chunked-prefill-size -1` |
| Chunked 8192 | `--chunked-prefill-size 8192` |
| Chunked 4096 | `--chunked-prefill-size 4096` |
| Chunked 2048 | `--chunked-prefill-size 2048` |
| Chunked 1024 | `--chunked-prefill-size 1024` |

All rows were restarted from a fresh SGLang process. The active `chunked_prefill_size` value was verified from the SGLang startup log before each measurement.

### Controlled Variables

- SGLang version: `0.5.21`
- Model: `Qwen/Qwen3-0.6B`
- Served model name: `qwen3-0.6b`
- GPU/runtime: same H100 NVL pod for all rows
- Background decode requests: `4`
- Background prompt tokens: about `115` per request
- Background max output tokens: `2000`
- Injected request prompt tokens: `10285`
- Injected request max output tokens: `64`
- Injection delay target: `1.5s`
- Scheduling: `--schedule-policy fcfs`
- Memory/scheduler settings: `--mem-fraction-static 0.704`, `--max-prefill-tokens 16384`, `--schedule-conservativeness 1.0`
- `enable_mixed_chunk: False`
- Same client-observed streaming event timing instrumentation

### H100 NVL Results

All five rows completed with `5` successful requests and `0` errors. All four background decode requests generated `2000` completion tokens with finish reason `length`; the injected request generated `64` completion tokens with finish reason `length`.

Background timing values are client-observed inter-stream-event latency proxies in milliseconds.

| Chunk size | Before p50 | Before p95 | Before p99 | Before max | During p50 | During p95 | During p99 | During max | After p50 | After p95 | After p99 | After max | Injected TTFT | Injected latency | Background TPOT mean | Output tok/s | Samples before/during/after |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `-1` | 2.283 | 2.982 | 3.371 | 6.353 | 0.157 | 47.652 | 48.270 | 48.531 | 2.475 | 2.784 | 2.974 | 49.700 | 155.769 | 336.211 | 2.470 | 1528.573 | 2108 / 100 / 5788 |
| `8192` | 2.300 | 2.373 | 2.447 | 9.959 | 0.393 | 56.611 | 79.598 | 79.634 | 2.481 | 2.826 | 3.017 | 64.862 | 210.358 | 390.536 | 2.504 | 1537.695 | 2273 / 108 / 5615 |
| `4096` | 2.288 | 2.387 | 2.754 | 97.406 | 0.393 | 2.549 | 56.359 | 56.657 | 2.477 | 2.712 | 3.005 | 67.860 | 135.740 | 312.884 | 2.463 | 1528.873 | 2068 / 108 / 5820 |
| `2048` | 2.305 | 2.564 | 3.953 | 12.986 | 0.390 | 4.469 | 55.265 | 55.270 | 2.481 | 2.711 | 2.930 | 73.763 | 143.735 | 321.126 | 2.480 | 1534.050 | 2122 / 112 / 5762 |
| `1024` | 2.286 | 2.413 | 3.045 | 11.887 | 0.328 | 4.941 | 67.932 | 68.018 | 2.477 | 2.783 | 2.965 | 87.339 | 168.997 | 346.747 | 2.475 | 1526.805 | 2107 / 128 / 5761 |

Aggregate throughput remained roughly `1527-1538` output tokens/sec across the sweep.

### H100 NVL Observations

- `4096` produced the lowest injected-request TTFT and total latency in this sweep: `135.740 ms` TTFT and `312.884 ms` latency.
- `2048` produced the lowest during-prefill p99 and max among chunked configurations: `55.265 ms` p99 and `55.270 ms` max.
- `4096`, `2048`, and `1024` dramatically improved during-prefill p95 relative to disabled chunking and `8192`.
- The relationship was non-monotonic. Smaller chunks were not always better.
- Rare p99/max stalls remained even when p95 improved substantially. For example, `4096` reduced during-prefill p95 to `2.549 ms`, while its during-prefill p99 remained `56.359 ms`.
- `8192` had the worst injected TTFT and the worst during-prefill p99/max among the chunked configurations in this run.

### H100 NVL Caveats

- ITL is still a client-observed OpenAI-compatible inter-stream-event gap proxy, not a GPU-side per-token ITL trace.
- A streaming event can contain a token-like piece, partial text, or multiple characters/pieces.
- The injected request's TTFT is a submission-to-first-streamed-token proxy. It includes queueing, prefill, first decode, and stream delivery effects; it is not pure prefill time.
- These H100 results are the baseline for later distributed-serving work, but they do not replace the RTX 4090 historical result as if hardware were controlled.
- Do not infer GPU saturation, memory-bandwidth saturation, compute-bound behavior, memory-bound behavior, or scheduler causality from this client-side benchmark alone.

## Interpretation

### MEASURED

- The no-chunk baseline produced visible prefill/decode interference.
- The chunked `4096` condition used the same workload and same scheduler/memory settings except for chunked-prefill size.
- During the injected request's TTFT window, chunking reduced background decode p95/p99/max gap by about `34-35%`.
- The worst synchronized stall was not improved; it was larger in the chunked run.
- The long request's TTFT was slightly lower in the chunked run, but this is a single-run result.
- Aggregate throughput was nearly unchanged, with a small measured increase in the chunked run.
- The H100 NVL sweep completed all five chunk-size configurations with the same workload shape and no request errors.
- In the H100 NVL sweep, throughput stayed roughly `1527-1538` output tokens/sec.
- In the H100 NVL sweep, `4096` had the lowest injected TTFT/latency, and `2048` had the lowest during-prefill p99/max among chunked configurations.

### REASONABLE INFERENCE

- Chunked prefill at `4096` mitigated part of the transient decode disturbance during the long-prefill window.
- Chunking did not fully isolate decode token delivery for this workload.
- The remaining synchronized stall suggests an aggregated worker can still produce decode jitter when long-prefill work arrives, even with chunking enabled.
- This is a useful scheduler-level mitigation signal, not a full solution claim.
- The H100 NVL sweep suggests chunk-size tradeoffs are non-monotonic for this workload; middle chunk sizes looked better than both very large and very small chunk sizes on different metrics.

### CANNOT CONCLUDE

- We cannot claim scheduler causality from client-side timings alone.
- We cannot claim GPU saturation, memory-bandwidth saturation, compute-bound behavior, or memory-bound behavior.
- We cannot claim `4096` is the best chunk size.
- We cannot claim chunked prefill is generally sufficient or insufficient from one workload and one run.
- We cannot claim P/D disaggregation benefit yet because no disaggregated system has been measured.
- We cannot compare RTX 4090 and H100 absolute latency values as a controlled hardware experiment.

## Why This Motivates Phase 4

Phase 3 shows that scheduler-level chunking can reduce part of the prefill/decode interference signal, but it did not eliminate the largest synchronized decode stall. That motivates the next P/D disaggregation experiment:

> If prefill and decode are separated onto different workers/GPUs, can the decode worker maintain steadier token delivery when long-prefill requests arrive, and is that benefit large enough to justify KV-transfer and distributed-systems overhead?

Do not proceed to Phase 4 until the Phase 3 artifacts are reviewed.
