# Phase 5: Dynamo + SGLang P/D Functional Validation

Phase 5 validates the first two-GPU prefill/decode-disaggregated request in this lab. This is a functional checkpoint only. It does not evaluate whether P/D disaggregation is faster than aggregated serving; that controlled comparison belongs to Phase 6.

## Experimental Question

Can the lab execute a real OpenAI-compatible streaming request with prefill and decode physically separated across two H100 GPUs using Dynamo orchestration, SGLang execution, and the configured NIXL/UCX KV-transfer path?

## Architecture

```text
Client
  |
  v
Dynamo frontend
  |
  v
Prefill router
  |
  v
Prefill Worker A on GPU 0
  |
  v
NIXL/UCX KV handoff
  |
  v
Decode Worker B on GPU 1
  |
  v
Streamed response
```

The validated control path separated prefill and decode at the worker level:

- Prefill worker ID: `7537786647078909697`
- Decode worker ID: `7403880299791686667`
- `prefill_worker_id != decode_worker_id`

The validated physical placement was:

- GPU 0: prefill SGLang scheduler
- GPU 1: decode SGLang scheduler

## Environment

Hardware:

- 2x NVIDIA H100 NVL
- 95830 MiB VRAM each
- Driver `580.159.04`
- CUDA reported by driver: `13.0`

Topology:

- GPU0 <-> GPU1 topology: `SYS`
- GPU0 NUMA node: `0`
- GPU1 NUMA node: `1`

This topology is recorded as an experimental condition. It does not by itself prove any transfer bottleneck.

Software:

- Python `3.12.3`
- `ai-dynamo 1.6.0.dev20261004`
- `ai-dynamo-runtime 1.6.0.dev20261004`
- SGLang `0.5.19`
- Torch `2.13.0+cu130`
- NIXL `1.4.0`
- `nixl-cu13 1.4.0`

Model and serving configuration:

- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Served model name: `llama3.1-8b-instruct`
- Context length: `16384`
- Chunked prefill size: `4096`
- Transfer backend: `nixl`
- NIXL backend observed: `UCX`

## Launch Configuration

The functional validation used one Dynamo frontend, one prefill SGLang worker, and one decode SGLang worker.

The prefill worker was launched with:

- `CUDA_VISIBLE_DEVICES=0`
- `--disaggregation-mode prefill`
- `--disaggregation-transfer-backend nixl`

The decode worker was launched with:

- `CUDA_VISIBLE_DEVICES=1`
- `--disaggregation-mode decode`
- `--disaggregation-transfer-backend nixl`

Both workers used the same model, context length, chunked-prefill size, and SGLang runtime policy established for the Phase 5 launch checkpoint.

## Registration Evidence

Before inference, the stack reached a healthy registration state:

- `/v1/models` returned `llama3.1-8b-instruct` with context window `16384`.
- Dynamo detected the prefill worker and activated the prefill router.
- Dynamo registered a prefill endpoint namespace: `dynamo.prefill.generate`.
- Dynamo registered a decode/backend endpoint namespace: `dynamo.backend.generate`.
- The frontend activated a prefill router target for `dynamo/prefill/generate`.

Worker startup logs showed:

- Prefill: `NIXL KVManager initialized with backend: UCX`
- Prefill: `Prefill worker handler initialized`
- Prefill: `Registered endpoint 'dynamo.prefill.generate'`
- Decode: `NIXL KVManager initialized with backend: UCX`
- Decode: `Decode worker handler initialized (disaggregated decode mode)`
- Decode: `Registered endpoint 'dynamo.backend.generate'`

## First Request

One small streaming request was sent through the Dynamo frontend. This was a functional validation request, not a benchmark.

Observed result:

- HTTP status: `200`
- Prompt tokens: `50`
- Completion tokens: `41`
- Total tokens: `91`
- Finish reason: `stop`
- Client-observed TTFT: `0.4206 s`
- Client-observed end-to-end latency: `6.4836 s`
- Dynamo-reported TTFT: `396.88 ms`
- Dynamo-reported elapsed time: `6460 ms`
- `prefill_worker_id`: `7537786647078909697`
- `decode_worker_id`: `7403880299791686667`

These timings are preserved only to show the instrumentation worked for the first request. They must not be interpreted as Phase 5 performance results.

## NIXL Evidence

### Directly Observed

- NIXL KVManager initialized on both workers.
- The observed NIXL backend was `UCX`.
- Dynamo routed one request to distinct prefill and decode workers.
- The prefill request executed on Worker A / GPU 0.
- The decode request executed on Worker B / GPU 1.
- The same request completed successfully.
- The frontend returned HTTP `200`.
- The model generated output tokens.

### Strongly Inferred

The configured KV handoff path successfully supplied the decode worker with the state required to continue generation. This follows from successful generation in disaggregated mode with separate prefill/decode worker IDs and initialized NIXL/UCX KV managers on both sides.

### Not Directly Proven

The logs captured for this validation do not include an explicit per-request `NIXL transfer completed` event or byte-count-level transfer telemetry. Therefore this checkpoint should not claim direct byte-level proof of KV transfer completion.

## Request Flow Explanation

Control path:

```text
Client
  |
  v
Dynamo frontend
  |
  v
Decode/backend routing
  |
  v
Prefill routing
  |
  +--> Prefill Worker A: 7537786647078909697
  |
  +--> Decode Worker B: 7403880299791686667
  |
  v
Streamed response
```

Data path:

```text
Prompt
  |
  v
Prefill GPU 0
  |
  v
KV created during prefill
  |
  v
NIXL/UCX handoff path
  |
  v
Decode GPU 1
  |
  v
KV consumed by decode
  |
  v
Autoregressive decode
  |
  v
Output tokens
```

Dynamo accepted the OpenAI-compatible request, selected backend workers, activated the prefill router path, and recorded the prefill/decode worker IDs.

SGLang loaded and served Llama 3.1 8B on both workers. The prefill worker performed the prompt-side computation; the decode worker performed autoregressive generation.

NIXL provided the configured KV-transfer interface, and UCX was the initialized backend transport. The current instrumentation confirms initialization and successful disaggregated request behavior, but not a byte-count-level transfer-completion event.

## Deployment And Debugging Observations

Two model-acquisition issues occurred before the successful launch:

1. Hugging Face credential visibility caused an initial `401 Unauthorized` while fetching gated model files.
2. Explicit credential propagation to the worker environment resolved the credential issue.
3. Starting both workers before the full model snapshot was staged caused a Hugging Face cache lock collision.
4. Pre-staging the complete `meta-llama/Llama-3.1-8B-Instruct` snapshot in the shared cache resolved the cache-lock issue.
5. Both workers subsequently loaded from the completed shared cache without redownloading the model.

These are deployment/debugging observations, not inference performance findings.

## Warnings

Two warnings were observed and should be preserved for later interpretation:

- The Dynamo frontend CPU affinity spanned both NUMA nodes.
- MM-aware routing was unsupported for this text model, so Dynamo fell back to text-prefix routing behavior.

Neither warning blocked the functional validation.

## What Phase 5 Establishes

Phase 5 establishes that the lab can execute a real request with prefill and decode physically separated across two H100 GPUs using Dynamo orchestration, SGLang execution, and the configured NIXL/UCX KV-transfer path.

Phase 5 does not establish that P/D disaggregation is faster than aggregated serving. That question belongs to Phase 6.
