# Phase 4B: Dynamo + SGLang Aggregated Serving Baseline

Phase 4B introduces NVIDIA Dynamo in front of SGLang while keeping the inference architecture aggregated. The goal is to understand what Dynamo adds before enabling prefill/decode disaggregation.

This phase does not configure P/D disaggregation, does not use a second GPU, and does not benchmark or tune performance.

## Research Question

What does NVIDIA Dynamo add to an otherwise aggregated SGLang serving architecture?

## Validated Architecture

```text
Client
  |
  v
Dynamo frontend / OpenAI-compatible ingress
  |
  v
Dynamo discovery + request routing
  |
  v
Single SGLang worker
  |
  v
meta-llama/Llama-3.1-8B-Instruct
  |
  v
1x NVIDIA H100 NVL
```

Prefill and decode remained colocated inside the same SGLang worker on the same H100. The KV cache remained local to that worker/GPU. No cross-GPU KV transfer was required.

## Environment

Phase 4B ran on a fresh RunPod H100 NVL pod.

- GPU: NVIDIA H100 NVL, 95830 MiB
- Driver: 580.159.04
- Host CUDA reported by driver: 13.0
- Python: 3.12.3
- Dynamo source checkout: `/workspace/dynamo-phase4b`
- Dynamo source commit: `5c8b22b` (`chore(sglang): bump to 0.5.21 (#15538)`)
- Isolated environment: `/workspace/dynamo-phase4b/.venv`
- ai-dynamo: 1.6.0.dev20261004
- ai-dynamo-runtime: 1.6.0.dev20261004
- SGLang dependency installed by Dynamo environment: 0.5.19
- Torch: 2.13.0+cu130
- torch CUDA runtime: 13.0
- NIXL package: installed (`nixl`/`nixl-cu13` 1.4.0)
- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Served model name: `llama3.1-8b-instruct`

Phase 4B intentionally used an isolated Dynamo environment instead of modifying the Phase 4A SGLang environment.

## Launch Configuration

Validated running processes:

```text
python3 -m dynamo.frontend --model-name llama3.1-8b-instruct

python3 -m dynamo.sglang   --model-path meta-llama/Llama-3.1-8B-Instruct   --served-model-name llama3.1-8b-instruct   --page-size 16   --tp-size 1   --dtype bfloat16   --context-length 16384   --mem-fraction-static 0.704   --max-prefill-tokens 16384   --schedule-policy fcfs   --chunked-prefill-size 4096   --stream-interval 1   --trust-remote-code   --enable-metrics   --cuda-graph-backend-prefill=disabled
```

File-based Dynamo discovery was used because Docker/etcd was not available in the pod:

```text
DYN_DISCOVERY_BACKEND=file
DYN_FILE_KV=/tmp/dynamo_file_kv
```

The previous Phase 4A venv-local CUDA repair was also needed in the isolated Dynamo environment:

- add `cu13/lib64 -> lib` symlink for JIT linking
- add `libcudart.so -> libcudart.so.13` symlink
- no NVIDIA driver change

A RunPod port conflict on `DYN_SYSTEM_PORT=8081` appeared in worker logs. Functional request validation still succeeded because Dynamo frontend ingress and worker serving remained healthy.

## Functional Validation

`GET /v1/models` returned:

```json
{"id":"llama3.1-8b-instruct","context_window":16384}
```

Non-streaming request:

- HTTP status: 200
- Latency: 6.4345 s
- Prompt tokens: 52
- Completion tokens: 33
- Total tokens: 85
- Finish reason: `stop`

Streaming request:

- HTTP status: 200
- Latency: 0.5592 s
- TTFT proxy: 0.0288 s
- Event count: 93
- Inter-stream-event p50: 0.00584 s
- Inter-stream-event p95: 0.00607 s
- Inter-stream-event max: 0.00848 s
- Prompt tokens: 52
- Completion tokens: 92
- Total tokens: 144
- Cached tokens reported in usage details: 16

These are functional instrumentation checks, not a performance study and not a comparison against Phase 4A.

GPU state after model load:

- GPU memory used: about 68076 MiB
- GPU memory free: about 27244 MiB
- SGLang worker log reported model weight memory: 15.02 GB
- SGLang worker log reported KV cache allocation: 409472 tokens, K cache 24.99 GB, V cache 24.99 GB

The KV cache capacity is SGLang's allocated serving pool capacity, not the per-request context length. The configured request context length remained 16384.

## What Dynamo Added In Aggregated Mode

Dynamo added an OpenAI-compatible frontend/ingress, service discovery, model/worker registration, request routing to the discovered worker, and request-path metadata/observability such as worker IDs. It did not execute model forward passes. SGLang still loaded the model, scheduled prefill/decode, allocated and owned KV cache, and performed GPU execution.

Observed logs showed the frontend discovering one worker set for `llama3.1-8b-instruct` with one member and selecting the single worker using round-robin routing. For the validated request, Dynamo reported the same worker ID for prefill and decode:

```text
prefill_worker_id=5099387232919803540
decode_worker_id=5099387232919803540
```

That is the key Phase 4B evidence that this was aggregated serving, not P/D disaggregation.

## Source Trace: Aggregated Request Path

The following call graph is based on the Dynamo checkout at `/workspace/dynamo-phase4b` and the installed package files in `/workspace/dynamo-phase4b/.venv`.

1. `dynamo.frontend.main.main()` -> `async_main()`
   - File: `.venv/lib/python3.12/site-packages/dynamo/frontend/main.py`
   - Builds `DistributedRuntime`, router config, frontend entrypoint args, then runs HTTP input.

2. Dynamo HTTP/OpenAI service receives the client request.
   - Files: `lib/llm/src/http/service/openai.rs`, `lib/llm/src/http/service/generate.rs`, `lib/llm/src/http/service/metrics.rs`
   - Handles OpenAI-compatible request processing and records worker metadata on request spans/metrics.

3. Dynamo discovery constructs a model group and serving engines from registered workers.
   - `Model::add_worker_set()` in `lib/llm/src/discovery/model.rs`
   - `DiscoveryWatcher` in `lib/llm/src/discovery/watcher.rs`
   - Logs: `Adding worker set to model`, `Chat completions is ready`, `Committed discovered model group`.

4. Dynamo router selects the backend worker.
   - `PushRouter::round_robin_prepared()` and related `generate_with_fault_detection*()` paths in `lib/runtime/src/pipeline/network/egress/push_router.rs`
   - Log: `Selected worker router_mode="round-robin" worker_id=5099387232919803540`.

5. Dynamo SGLang worker process starts and registers a generate endpoint.
   - `worker()` in `.venv/lib/python3.12/site-packages/dynamo/sglang/main.py`
   - `init_decode()` in `.venv/lib/python3.12/site-packages/dynamo/sglang/init_llm.py`
   - `Worker.run()` in `.venv/lib/python3.12/site-packages/dynamo/common/backend/worker.py`
   - Because `server_args.disaggregation_mode == "null"`, `DynamoConfig.serving_mode` maps to `DisaggregationMode.AGGREGATED`.

6. Dynamo SGLang request handler calls SGLang.
   - `DecodeWorkerHandler.generate()` in `.venv/lib/python3.12/site-packages/dynamo/sglang/request_handlers/llm/decode_handler.py`
   - In aggregated mode it takes the non-`DECODE` branch and calls `self.engine.async_generate(..., stream=True, ...)` without `bootstrap_host`, `bootstrap_port`, or `bootstrap_room`.

7. SGLang executes prefill and decode internally.
   - `Scheduler.init_disaggregation()` in `.venv/lib/python3.12/site-packages/sglang/srt/managers/scheduler.py` reads disaggregation mode and transfer backend.
   - With mode `NULL`, SGLang uses the normal aggregated scheduler path.
   - SGLang allocates/owns KV memory through memory pool classes in `.venv/lib/python3.12/site-packages/sglang/srt/mem_cache/memory_pool.py`.

8. Dynamo frontend streams results back to the client.
   - Worker response chunks flow back through Dynamo's request pipeline and are serialized through the OpenAI-compatible frontend.

## Worker IDs

The numeric worker ID is Dynamo's runtime/discovery instance identifier for a registered backend endpoint instance. It is not a CUDA device ID and not a model-specific ID.

In this run, exactly one SGLang worker registered a `generate` endpoint for `llama3.1-8b-instruct`. Because there was one candidate, round-robin selection always selected that instance. The OpenAI service metrics/timing path records prefill and decode worker IDs; for aggregated requests the same backend instance performs both phases, so the logged IDs match.

## Where P/D Diverges Later

Phase 4B stayed aggregated. The future P/D path diverges at both the Dynamo worker role and the SGLang serving mode.

Dynamo-side role split:

- `WorkerConfig.disaggregation_mode` in `.venv/lib/python3.12/site-packages/dynamo/common/backend/worker.py` maps `AGGREGATED`, `PREFILL`, `DECODE`, and `ENCODE` into Rust worker roles.
- The worker registration changes the worker type advertised through discovery.
- `dynamo.common.backend.disagg` defines helper behavior for P/D, including clamping prefill work to one output token and requiring `prefill_result` for decode-mode requests.

SGLang adapter split:

- `.venv/lib/python3.12/site-packages/dynamo/sglang/args.py` maps SGLang `--disaggregation-mode null|prefill|decode` into Dynamo serving mode.
- `.venv/lib/python3.12/site-packages/dynamo/sglang/main.py` starts decode and/or prefill initialization based on serving mode.
- `PrefillWorkerHandler.generate()` returns `disaggregated_params` containing bootstrap information for decode.
- `DecodeWorkerHandler.generate()` in decode mode requires `bootstrap_info` and calls SGLang with `bootstrap_host`, `bootstrap_port`, and `bootstrap_room`.

SGLang internal split:

- `Scheduler.init_disaggregation()` in `.venv/lib/python3.12/site-packages/sglang/srt/managers/scheduler.py` initializes prefill/decode-specific paths when disaggregation mode is not `NULL`.
- `.venv/lib/python3.12/site-packages/sglang/srt/disaggregation/prefill.py` creates KV senders.
- `.venv/lib/python3.12/site-packages/sglang/srt/disaggregation/decode.py` creates KV receivers and sends decode-side metadata for transfer.

## NIXL And KV Data Path

NIXL was installed in the Dynamo environment, but it was not part of the Phase 4B request data path.

Evidence:

- Only one SGLang worker process was used.
- `--tp-size 1` was used.
- No `--disaggregation-mode prefill` or `--disaggregation-mode decode` was used.
- No `--disaggregation-transfer-backend nixl` was used.
- Worker logs showed `disaggregation_mode: null`.
- The same worker ID appeared as both prefill and decode worker.
- KV cache allocation remained inside the SGLang worker's GPU memory pool.

When P/D is enabled later, NIXL would be part of the KV transfer data path only if SGLang is configured with the NIXL transfer backend. The relevant SGLang code path is:

```text
sglang/srt/disaggregation/utils.py
  TransferBackend.NIXL -> NixlKVManager / NixlKVSender / NixlKVReceiver

sglang/srt/disaggregation/prefill.py
  DisaggPrefillBootstrapQueue._init_kv_manager()
  DisaggPrefillBootstrapQueue.create_sender()

sglang/srt/disaggregation/decode.py
  decode-side receiver creation and send_metadata(...)

sglang/srt/disaggregation/nixl/conn.py
  NIXL agent setup, memory registration, transfer descriptors, agent.transfer(...)
```

Conceptually, Dynamo routes/control-plane messages decide which worker receives each request stage, while SGLang owns the device KV memory and transfer mechanism. In P/D mode with NIXL, the KV tensors move between SGLang prefill and decode workers through SGLang's disaggregation transfer backend. Dynamo does not become the owner of the KV cache.

## What Phase 4B Proved

Measured/observed:

- Dynamo frontend can receive OpenAI-compatible requests.
- Dynamo can discover one SGLang worker and route requests to it.
- SGLang can serve Llama 3.1 8B behind Dynamo on one H100 NVL.
- Non-streaming and streaming requests succeeded.
- Prompt/completion token counts, TTFT proxy, inter-stream-event timings, and end-to-end latency can still be captured.
- Prefill and decode remained colocated in one worker/GPU.

Reasonable interpretation:

- In aggregated mode, Dynamo adds ingress, discovery, routing, and observability around SGLang without changing who executes the model.
- Worker IDs are useful for verifying whether prefill and decode are colocated or split.
- This is the right checkpoint before P/D because it isolates the effect of adding Dynamo itself.

Cannot conclude:

- Whether Dynamo improves or hurts throughput or latency.
- Whether the GPU is saturated.
- Whether execution is compute-bound or memory-bandwidth-bound.
- Whether P/D disaggregation will improve this workload.
- Whether NIXL behavior is correct or beneficial, because NIXL was installed but not exercised.

## Preserved Artifacts

Intentional Phase 4B artifacts are under `outputs/phase4b_dynamo_aggregated/`.

Committed artifacts include the model response JSON, streaming event JSON, summaries, environment/runtime evidence, software versions, and an artifact manifest. Raw `.log` and `.pid` files remain ignored by `.gitignore`; they are local runtime artifacts and are not needed to reproduce the documented checkpoint.
