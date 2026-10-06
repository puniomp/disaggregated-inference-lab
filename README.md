# Disaggregated Inference Lab

A hands-on inference-systems lab that measures how separating prefill and decode changes client-observed decode responsiveness during mixed LLM serving traffic.

## Key Result

The final matched Phase 6 experiment compared two architectures on the same 2x H100 NVL pod, software stack, model snapshot, workload generator, scheduler policy, and chunked-prefill configuration.

| Metric, median of 3 valid trials | Aggregated SGLang | Dynamo + SGLang P/D |
| --- | ---: | ---: |
| During-window p50 stream-event gap | 5.76 ms | 6.03 ms |
| During-window p95 stream-event gap | 372.99 ms | 6.50 ms |
| During-window p99 stream-event gap | 373.05 ms | 8.07 ms |
| During-window max stream-event gap | 373.08 ms | 14.15 ms |
| Injected long-prefill TTFT | 420.47 ms | 450.45 ms |
| Injected long-prefill E2E latency | 861.37 ms | 879.87 ms |

`DURING` means the interval from injected long-prefill request submission to that request's first content token. The gap metrics are client-observed OpenAI streaming event gaps during that TTFT/prefill proxy window. They are not GPU kernel timings or direct per-token scheduler measurements.

Measured result: under this workload, P/D disaggregation dramatically reduced the decode-stream tail gaps observed while an approximately 8K-token prefill request was being processed. It did not make every metric lower: the injected request's TTFT and end-to-end latency were slightly higher in the P/D runs.

## Why This Matters

LLM inference has two phases with different system behavior:

- **Prefill** processes the prompt and creates KV cache. Long prompts can produce large, bursty work.
- **Decode** generates one token at a time and is latency-sensitive for streaming users.

In an aggregated worker, long prefill and ongoing decode share the same GPU and scheduler. Chunked prefill can reduce interference, but the phases still contend inside one worker. P/D disaggregation asks a different question: can the system isolate prefill from decode by placing them on different workers and moving KV state between them?

This lab does not try to prove that P/D is universally faster. It shows one controlled case where P/D improved decode responsiveness under mixed traffic, while adding a second active GPU, distributed orchestration, and KV-transfer complexity.

## Architectures

### Aggregated Baseline

```text
Client
  |
  v
Dynamo frontend
  |
  v
SGLang worker on GPU0
  |-- Prefill
  |-- Decode
  |-- KV cache local to worker/GPU0

GPU1 idle
```

In the final aggregated run, prefill and decode were colocated in one SGLang worker on GPU0. Chunked prefill was enabled with `chunked_prefill_size=4096`. No P/D split or cross-GPU KV handoff was used.

### P/D Disaggregated

```text
Client
  |
  v
Dynamo frontend
  |
  v
Prefill routing / coordination
  |
  v
Prefill SGLang worker on GPU0
  |
  |  KV handoff via configured NIXL/UCX path
  v
Decode SGLang worker on GPU1
  |
  v
Streamed response
```

In the final P/D run, the prefill worker ran on GPU0 and the decode worker ran on GPU1. Dynamo handled frontend and worker coordination. SGLang executed model work. NIXL/UCX was the configured KV-transfer path. The experiment observed successful generation with distinct prefill and decode worker IDs, but it did not capture byte-level per-request KV-transfer telemetry.

## Experimental Method

Final Phase 6 environment:

| Component | Value |
| --- | --- |
| Model | `meta-llama/Llama-3.1-8B-Instruct` |
| Served model | `llama3.1-8b-instruct` |
| Hardware | 2x NVIDIA H100 NVL, same node |
| GPU topology | GPU0 `<->` GPU1 = `NODE` |
| Python | 3.12.3 |
| Dynamo | `1.6.0.dev20261006` |
| SGLang | `0.5.21` |
| Torch | `2.13.0+cu130` |
| NIXL | `1.4.0` |
| `nixl-cu13` | `1.4.0` |
| Tensor parallelism | `TP=1` |
| dtype | BF16 |
| Context length | 16384 |
| `mem_fraction_static` | 0.704 |
| `max_prefill_tokens` | 16384 |
| Scheduler | FCFS |
| Chunked prefill | 4096 |
| Stream interval | 1 |
| Page size | 16 |

Final workload:

- Start 4 concurrent sustained-decode background streams.
- Each background request uses approximately 87 prompt tokens and targets 768 output tokens.
- Wait until all 4 background streams have begun producing tokens.
- Inject one fresh, unique long-prefill request with approximately 8K prompt tokens and 64 output tokens.
- Require all 4 background streams to remain active from injected request submission through injected first content token.
- Run exactly 3 valid trials per architecture.

Measurement notes:

- Background interference is measured as client-observed gaps between streaming events.
- The `DURING` window is injected request submission to injected first content token.
- Injected-request TTFT is used as the closest available client-side prefill/TTFT proxy.
- Token counts come from server usage metadata where available.
- Unique long-prefill prompts are generated to minimize meaningful prefix-cache reuse.

## Results

### Final Phase 6 Mixed-Traffic Comparison

| Metric, median of 3 valid trials | Aggregated SGLang | Dynamo + SGLang P/D |
| --- | ---: | ---: |
| During p50 | 5.76 ms | 6.03 ms |
| During p95 | 372.99 ms | 6.50 ms |
| During p99 | 373.05 ms | 8.07 ms |
| During max | 373.08 ms | 14.15 ms |
| Injected TTFT | 420.47 ms | 450.45 ms |
| Injected E2E | 861.37 ms | 879.87 ms |

The aggregated runs produced repeatable during-window p95 gaps around 373 ms while the long-prefill request was processed on the same GPU as the sustained decode streams. The P/D runs, using the same workload and current environment, kept during-window p95 near normal decode cadence while the long prefill ran on the separate prefill GPU.

This comparison is about resource isolation under mixed traffic. It is not an equal-cost throughput comparison because aggregated used one active GPU while P/D used two active GPUs.

### Supporting Historical Work

Earlier phases built up the result incrementally:

- **Phase 1:** Standalone SGLang aggregated serving with `Qwen/Qwen3-0.6B`.
- **Phase 2:** SGLang benchmarking harness for TTFT, TPOT/ITL proxy, end-to-end latency, and throughput.
- **Phase 3:** Chunked-prefill interference experiments with Qwen on H100 NVL. This showed that chunk size affects decode responsiveness non-monotonically, and that smaller chunks are not always better.
- **Phase 4A:** Standalone SGLang aggregated baseline with Llama 3.1 8B on H100 NVL.
- **Phase 4B:** Dynamo + SGLang aggregated serving, still one GPU and no P/D split.
- **Phase 5:** First functional two-GPU P/D request with Dynamo, SGLang, and configured NIXL/UCX KV handoff.

Historical RTX 4090 and earlier H100 experiments are preserved as learning evidence, but their absolute latency numbers are not quantitatively mixed with the final Phase 6 comparison because hardware, topology, and software versions differ.

## What I Learned

- Prefill and decode stress different parts of an inference server, so mixed traffic can expose issues that single-request smoke tests miss.
- Chunked prefill is a scheduler-level mitigation, not magic. It can reduce tail gaps, but its effect depends on workload and chunk size.
- P/D disaggregation changes the isolation boundary: long prefill can run on one worker/GPU while decode continues on another.
- The benefit observed here is lower client-observed decode tail gaps during long-prefill overlap, not universal lower latency.
- Dynamo adds distributed serving coordination around SGLang workers. SGLang still executes prefill and decode. NIXL/UCX is the configured KV data path when prefill and decode are split.
- Worker IDs and successful output prove that the request traversed distinct prefill/decode workers, but they do not by themselves provide byte-level KV-transfer timing.

## Limitations

- Stream-event gaps are a client-observed ITL proxy, not GPU-level per-token timing.
- The injected request TTFT window is a prefill proxy, not a direct measurement of prefill kernels.
- The final result uses 3 valid trials per architecture, enough for a controlled lab comparison but not production-scale statistical characterization.
- The experiment does not prove GPU saturation, memory-bandwidth saturation, compute-bound behavior, or exact scheduler causality.
- The P/D setup uses two active GPUs while the aggregated setup uses one active GPU. The result should not be read as equal-cost superiority.
- The experiment does not isolate NIXL transfer latency or prove the exact cost of KV movement.
- Results are specific to this model, software stack, topology, prompt construction, and workload shape.

## Reproduction

This repository is designed to preserve the experiment and make it auditable. Reproducing the final Phase 6 comparison requires a 2x H100 NVL node with Hugging Face access to `meta-llama/Llama-3.1-8B-Instruct`.

High-level steps:

1. Clone the repository.
2. Create the isolated Dynamo/SGLang environment matching the Phase 6 versions above.
3. Authenticate to Hugging Face without storing tokens in the repository.
4. Pre-stage the Llama 3.1 8B model snapshot into the Hugging Face cache.
5. Run the aggregated Dynamo + SGLang configuration with one worker on GPU0.
6. Validate tiny streaming, unique approximately 8K prefill, and sustained decode requests.
7. Run three valid mixed trials with `benchmarks/phase6_client.py` and preserve outputs.
8. Stop aggregated serving and start the P/D configuration with prefill on GPU0 and decode on GPU1.
9. Repeat the same validation and three valid mixed trials.

The final artifacts are preserved under:

- [`outputs/phase6_aggregated_validate`](outputs/phase6_aggregated_validate)
- [`outputs/phase6_aggregated_mixed_final`](outputs/phase6_aggregated_mixed_final)
- [`outputs/phase6_pd_final`](outputs/phase6_pd_final)
- [`outputs/phase6_pd_mixed_final`](outputs/phase6_pd_mixed_final)

The benchmark client is [`benchmarks/phase6_client.py`](benchmarks/phase6_client.py). Some deployment paths are environment-specific, so reproduction should pass explicit frontend/log/output arguments rather than relying on local pod paths.

## Repository Layout

```text
benchmarks/
  phase6_client.py                 Final workload generator and measurement client

docs/
  architecture.md                  Architecture notes and source-grounded assumptions
  phase1_sglang_baseline.md        Initial SGLang aggregated baseline
  phase2_sglang_benchmarking.md    Benchmark harness and Phase 2 results
  phase3_chunked_prefill.md        Chunked-prefill interference work
  phase4a_llama31_sglang_baseline.md
  phase4b_dynamo_aggregated.md
  phase5_disaggregated_validation.md

outputs/
  phase6_aggregated_validate/      Aggregated validation artifacts
  phase6_aggregated_mixed_final/   Final aggregated mixed trials
  phase6_pd_final/                 P/D validation artifacts
  phase6_pd_mixed_final/           Final P/D mixed trials

scripts/
  check_env.sh
  start_sglang.sh
  stop_sglang.sh
  smoke_test.sh
```
