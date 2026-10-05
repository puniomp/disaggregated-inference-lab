# Phase 3: Chunked-Prefill Interference

## Experimental Question

Phase 3 asks:

> When long-prefill work arrives while other requests are actively decoding on the same aggregated SGLang worker/GPU, does it interfere with their token delivery, and can chunked prefill mitigate that interference?

This phase stays within the single-worker, single-GPU SGLang architecture. It does not use Dynamo, NIXL, P/D disaggregation, multi-GPU serving, or Kubernetes.

## Environment

- SGLang version: `0.5.21`
- Model: `Qwen/Qwen3-0.6B`
- Served model name: `qwen3-0.6b`
- GPU: `NVIDIA GeForce RTX 4090`, `24564 MiB`, driver `580.65.06`
- Architecture: one SGLang worker on one GPU; prefill and decode colocated

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

## Interpretation

### MEASURED

- The no-chunk baseline produced visible prefill/decode interference.
- The chunked `4096` condition used the same workload and same scheduler/memory settings except for chunked-prefill size.
- During the injected request's TTFT window, chunking reduced background decode p95/p99/max gap by about `34-35%`.
- The worst synchronized stall was not improved; it was larger in the chunked run.
- The long request's TTFT was slightly lower in the chunked run, but this is a single-run result.
- Aggregate throughput was nearly unchanged, with a small measured increase in the chunked run.

### REASONABLE INFERENCE

- Chunked prefill at `4096` mitigated part of the transient decode disturbance during the long-prefill window.
- Chunking did not fully isolate decode token delivery for this workload.
- The remaining synchronized stall suggests an aggregated worker can still produce decode jitter when long-prefill work arrives, even with chunking enabled.
- This is a useful scheduler-level mitigation signal, not a full solution claim.

### CANNOT CONCLUDE

- We cannot claim scheduler causality from client-side timings alone.
- We cannot claim GPU saturation, memory-bandwidth saturation, compute-bound behavior, or memory-bound behavior.
- We cannot claim `4096` is the best chunk size.
- We cannot claim chunked prefill is generally sufficient or insufficient from one workload and one run.
- We cannot claim P/D disaggregation benefit yet because no disaggregated system has been measured.

## Why This Motivates Phase 4

Phase 3 shows that scheduler-level chunking can reduce part of the prefill/decode interference signal, but it did not eliminate the largest synchronized decode stall. That motivates the next P/D disaggregation experiment:

> If prefill and decode are separated onto different workers/GPUs, can the decode worker maintain steadier token delivery when long-prefill requests arrive, and is that benefit large enough to justify KV-transfer and distributed-systems overhead?

Do not proceed to Phase 4 until the Phase 3 artifacts are reviewed.
