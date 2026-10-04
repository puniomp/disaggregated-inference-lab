# Phase 0 Architecture Validation

Validated: 2026-10-02

This document is the architectural contract for the lab. It separates durable inference-system concepts from current SGLang, NVIDIA Dynamo, and NIXL implementation details, and from assumptions that must be tested on the target machine before Phase 1 and later phases.

Claim labels:

- **[Stable concept]**: a general LLM serving concept that should remain true across serving stacks.
- **[Current implementation detail]**: true of the current upstream documentation or repository behavior found during Phase 0; re-check before relying on exact flags, process topology, or routing behavior.
- **[Experimental assumption]**: plausible and useful for the lab, but must be measured or smoke-tested locally.

## Source Map

- SGLang PD disaggregation docs: https://docs.sglang.ai/backend/pd_disaggregation.html
- SGLang server arguments docs: https://docs.sglang.ai/advanced_features/server_arguments.html
- SGLang current release page: https://www.sglang.io/
- Dynamo system architecture: https://docs.dynamo.nvidia.com/dynamo/dev/knowledge-base/concepts/architecture
- Dynamo disaggregated serving concept page: https://docs.nvidia.com/dynamo/dev/knowledge-base/concepts/system-architecture/disaggregated-serving
- Dynamo SGLang backend disaggregation page: https://docs.dynamo.nvidia.com/dynamo/knowledge-base/modular-components/backends/sg-lang/disaggregation
- Dynamo local disaggregated serving guide: https://docs.dynamo.nvidia.com/dynamo/dev/cli/disaggregated-serving/overview
- Dynamo KV-aware routing overview: https://docs.dynamo.nvidia.com/dynamo/cli/kv-aware-routing/overview
- Dynamo KV-aware routing concepts: https://docs.nvidia.com/dynamo/dev/knowledge-base/concepts/system-architecture/kv-aware-routing
- Dynamo Frontend KV routing guide: https://docs.dynamo.nvidia.com/dynamo/kubernetes/kv-aware-routing/using-the-dynamo-frontend
- NIXL repository README: https://github.com/ai-dynamo/nixl/blob/main/README.md
- NIXL backend/plugin guide: https://github.com/ai-dynamo/nixl/blob/main/docs/BackendGuide.md

## Component Responsibilities

### SGLang

**[Stable concept]** SGLang is the model-serving engine in this lab: it loads the model weights, accepts generation requests, performs tokenization/request handling as configured, schedules prefill and decode work, executes model forward passes on GPU, manages KV cache memory, and streams generated tokens.

**[Current implementation detail]** SGLang supports PD disaggregation, where one SGLang instance can run in `--disaggregation-mode prefill` and another in `--disaggregation-mode decode`; current docs show NIXL and Mooncake as supported transfer engines, with `--disaggregation-transfer-backend nixl` for NIXL. Source: SGLang PD disaggregation docs.

**[Current implementation detail]** SGLang exposes chunked prefill through `--chunked-prefill-size`; its server-argument docs say a smaller chunked prefill size can help avoid OOM during long prefills. Source: SGLang server arguments docs.

**[Current implementation detail]** SGLang exposes `--disable-radix-cache`, which disables RadixAttention/prefix caching. Source: SGLang server arguments docs.

**[Experimental assumption]** For this lab, Qwen3-0.6B or another small Hugging Face model supported by the installed SGLang version will fit on one GPU and produce enough tokens to expose scheduling effects. Validate by launching SGLang and running a smoke request before using it as the benchmark model.

### NVIDIA Dynamo

**[Stable concept]** Dynamo is the distributed serving layer in this lab: it supplies the frontend/request path, worker discovery, routing, and orchestration around one or more backend engines such as SGLang.

**[Current implementation detail]** Dynamo's architecture separates request execution, service discovery, and event delivery. The documented request path is HTTP client -> Frontend -> preprocessing/tokenization/validation -> routing to prefill/decode workers when disaggregation is enabled. Source: Dynamo system architecture.

**[Current implementation detail]** Dynamo can run aggregated serving with workers behind a frontend, and disaggregated serving with separate prefill and decode workers. Its local guide describes disaggregated serving as frontend plus a prefill worker that computes KV cache and a decode worker that receives KV cache over NIXL and generates tokens. Source: Dynamo local disaggregated serving guide.

**[Current implementation detail]** Dynamo's current disaggregated design can use KV-aware routing or load balancing to choose workers. Dynamo docs describe a `PrefillRouter` that selects a prefill worker, sends prefill to it, injects transfer metadata into the decode request, then routes to a decode worker. Source: Dynamo disaggregated serving concept page.

**[Current implementation detail]** Dynamo's SGLang backend integration differs from SGLang standalone PD: Dynamo routes to a decode worker first, chooses a prefill worker by round-robin or KV-aware selection, sends the request to both workers, and uses SGLang's bootstrap server plus NIXL or Mooncake for KV transfer. Source: Dynamo SGLang backend disaggregation page.

**[Experimental assumption]** The smallest local Dynamo + SGLang aggregated setup will add routing/discovery overhead relative to SGLang alone. The sign and size of that overhead must be measured; do not assume Dynamo aggregated is faster or slower without data.

### NIXL

**[Stable concept]** NIXL is the data movement layer in this lab: it is responsible for moving KV cache data between memory locations used by inference workers, especially GPU memory in prefill/decode disaggregation.

**[Current implementation detail]** NIXL is an inference transfer library intended to accelerate point-to-point communication in AI inference frameworks such as Dynamo, abstracting CPU/GPU memory and storage through a modular plugin architecture. Source: NIXL README.

**[Current implementation detail]** NIXL's backend guide describes asynchronous, non-blocking transfer requests, descriptor lists over memory spaces including DRAM and VRAM, and pluggable backends such as UCX and GPUDirect Storage. Source: NIXL backend/plugin guide.

**[Current implementation detail]** Dynamo's disaggregated serving docs state that Dynamo uses NIXL to transfer KV cache directly from prefill-engine VRAM to decode-engine VRAM, and that NIXL chooses an available transport such as NVLink or InfiniBand/UCX. Source: Dynamo disaggregated serving concept page.

**[Experimental assumption]** On a same-node, two-GPU machine, the actual NIXL path may depend on topology, drivers, UCX configuration, and whether the GPUs support peer access/NVLink. Validate with logs, NIXL/UCX diagnostics, and benchmark deltas rather than assuming the fastest path is active.

## Where the Model Executes

**[Stable concept]** The transformer forward pass executes inside the backend engine process, not inside the client. In this lab, that backend engine is SGLang.

**[Stable concept]** In aggregated inference, the same SGLang worker performs both prefill and decode for a request on the GPU(s) assigned to that worker.

**[Current implementation detail]** In SGLang PD mode, prefill and decode are separate SGLang server roles. Current examples start a prefill server on one GPU and a decode server on another GPU for a single-node Llama deployment. Source: SGLang PD disaggregation docs.

**[Current implementation detail]** In Dynamo P/D disaggregation with SGLang, SGLang remains the backend executing the model; Dynamo orchestrates frontend, routing, discovery, and transfer metadata flow. Sources: Dynamo system architecture and Dynamo SGLang backend disaggregation page.

## Where KV Cache Lives

**[Stable concept]** KV cache is generated during prefill and then extended/read during decode. It stores transformer attention keys and values for prior tokens so decode can generate new tokens without recomputing the full prompt history each step.

**[Stable concept]** In aggregated serving, the KV cache for a request remains local to the worker/GPU memory that performs both prefill and decode, unless the engine uses additional offload or prefix-cache mechanisms outside this lab's initial scope.

**[Current implementation detail]** Dynamo's disaggregated serving docs describe the prefill engine computing KV cache, transferring it to the decode engine, and the decode engine then computing the decode phase. Source: Dynamo disaggregated serving concept page.

**[Current implementation detail]** Dynamo's SGLang integration docs describe decode allocating GPU memory pages for incoming KV cache and prefill using RDMA to write KV cache directly into decode GPU memory. Source: Dynamo SGLang backend disaggregation page.

**[Experimental assumption]** For the first lab runs, keep prefix caching/radix behavior explicit because prefix reuse can change prefill work and hide the cost of P/D transfer. Use fixed synthetic prompts and consider `--disable-radix-cache` when isolating raw prefill/decode behavior.

## Aggregated Inference Request Flow

```
Client
  |
  v
SGLang worker
  |
  v
GPU
  |
  +--> Prefill: process prompt tokens, create KV cache
  |
  +--> Decode: generate one or more output tokens, reading/extending KV cache
  |
  v
Stream/return tokens to client
```

**[Stable concept]** Prefill processes the input prompt in parallel over prompt tokens and creates KV cache. Decode then generates output autoregressively, usually one token per sequence per decode step.

**[Stable concept]** TTFT is strongly affected by queueing plus prefill work before the first output token can be emitted. ITL/TPOT is strongly affected by decode scheduling and the cost of each subsequent token step.

**[Current implementation detail]** SGLang is responsible for the local scheduling policy, chunked prefill behavior, KV cache allocation, and token streaming in this aggregated setup. Source: SGLang server arguments docs and SGLang PD docs for the contrast with separated roles.

**[Experimental assumption]** Aggregated SGLang should be the baseline because it removes Dynamo routing and remote KV movement from the path. It may still include chunked prefill and prefix-cache behavior depending on launch flags.

## Dynamo + SGLang Aggregated Request Flow

```
Client
  |
  v
Dynamo Frontend / Router
  |
  |  discovery + worker selection
  v
SGLang worker
  |
  v
GPU
  |
  +--> Prefill
  +--> Decode
  |
  v
Tokens back through Dynamo path
```

**[Stable concept]** Adding Dynamo in aggregated mode should not split prefill and decode; it adds a distributed serving control plane and request-routing layer around SGLang workers.

**[Current implementation detail]** Dynamo docs describe the frontend as receiving requests and selecting workers, including KV-aware mode where workers publish KV events and the router scores cache overlap plus active load. Sources: Dynamo KV-aware routing overview and concepts.

**[Current implementation detail]** KV-aware routing is useful only when there are multiple candidate workers and repeated/shared prefixes that can create cache locality. If workers do not publish KV events, Dynamo docs say KV mode may use load-only or predicted state depending on configuration. Sources: Dynamo KV-aware routing overview and Dynamo Frontend KV guide.

**[Experimental assumption]** In the two-GPU lab, Dynamo aggregated may be useful mainly to understand routing and overhead. It may not beat a single SGLang worker for simple workloads unless there are multiple workers, useful cache locality, or enough concurrency.

## P/D Disaggregated Request Flow

```
                         Dynamo
                           |
                +----------+----------+
                |                     |
                v                     v
         Prefill selection      Decode selection
                |                     |
                v                     v
        SGLang PREFILL          SGLang DECODE
        worker on GPU 0         worker on GPU 1
                |                     ^
                |                     |
                +---- KV cache -------+
                    via NIXL / backend transfer
```

More detailed SGLang/Dynamo flow:

```
Client
  |
  v
Dynamo Frontend
  |
  v
PrefillRouter / routing policy
  |
  +--> choose decode worker
  |
  +--> choose prefill worker
         |
         v
    SGLang prefill computes prompt KV
         |
         v
    transfer metadata / bootstrap coordination
         |
         v
    NIXL moves KV cache to decode worker GPU memory
         |
         v
    SGLang decode generates output tokens
         |
         v
Client receives streamed tokens
```

**[Stable concept]** Disaggregation separates the prompt-processing phase from the token-generation phase so each can be scaled and tuned independently.

**[Current implementation detail]** Dynamo docs describe three main disaggregated steps: prefill computes KV cache, prefill transfers KV cache to decode, decode computes decode. Source: Dynamo disaggregated serving concept page.

**[Current implementation detail]** Dynamo docs say the router selects a prefill worker using KV-aware routing or load balancing, the prefill worker returns backend-specific transfer metadata, the router injects that result into the decode request, and the decode worker coordinates transfer. Source: Dynamo disaggregated serving concept page.

**[Current implementation detail]** For SGLang specifically, transfer metadata is `bootstrap_info` containing host, port, and room ID; SGLang prefill workers publish their bootstrap endpoint to discovery. Source: Dynamo disaggregated serving concept page.

**[Current implementation detail]** Dynamo's SGLang disaggregation page describes setup where decode workers register RDMA connection information with prefill workers, including base GPU memory pointers for direct memory access. Source: Dynamo SGLang backend disaggregation page.

**[Current implementation detail]** Dynamo's local disaggregated serving guide says the launch presets start frontend plus both workers pinned to separate GPUs and require 2 GPUs. Source: Dynamo local disaggregated serving guide.

**[Experimental assumption]** The lab's two-GPU version will use one prefill GPU and one decode GPU. This is the smallest topology that demonstrates P/D mechanics, but it may underperform aggregated serving for small models because transfer and orchestration overhead can dominate.

## Why KV Cache Must Move

**[Stable concept]** Decode needs the prompt's KV cache because each generated token attends over prior context. If prefill and decode run on different workers, the decode worker either needs the KV cache transferred to it or must recompute the prompt itself.

**[Stable concept]** Recomputing the prompt on decode would erase the main benefit of disaggregation for long prompts, so practical P/D disaggregation includes a KV cache transfer path.

**[Current implementation detail]** Dynamo explicitly treats efficient KV transfer as key to high-performance disaggregation and uses NIXL for direct prefill-VRAM to decode-VRAM transfer. Source: Dynamo disaggregated serving concept page.

**[Current implementation detail]** SGLang's current PD docs list Mooncake and NIXL as transfer engines, and the NIXL example starts prefill and decode SGLang servers with `--disaggregation-transfer-backend nixl`. Source: SGLang PD disaggregation docs.

## Transport for KV Movement

**[Stable concept]** Same-node GPU-to-GPU KV movement and cross-node GPU-to-GPU KV movement are different physical problems. Same-node transfer can use local GPU interconnects or host-mediated paths; cross-node transfer requires a network path.

**[Current implementation detail]** Dynamo docs say NIXL handles direct GPU-to-GPU transfer using the optimal available transport, listing NVLink and InfiniBand/UCX as examples. Source: Dynamo disaggregated serving concept page.

**[Current implementation detail]** NIXL's backend guide describes UCX as a backend that can move data between system and/or GPU memories and support local and remote transfers; the guide also describes GDS for storage-to-GPU use cases, which is not the initial KV-transfer focus. Source: NIXL backend/plugin guide.

**[Current implementation detail]** NIXL transfer requests are posted asynchronously and status is checked later; the backend guide says `postXfer` should start the transfer without waiting for completion and `checkXfer` checks status. Source: NIXL backend/plugin guide.

**[Experimental assumption]** For the local two-GPU lab, expected same-node transports may include NVLink, PCIe peer-to-peer, CUDA IPC, or UCX paths depending on hardware/software. For different nodes, expect UCX over InfiniBand/RDMA when properly configured, with possible TCP fallback or failure depending on UCX/NIXL settings. Validate using logs and a NIXL-specific benchmark if available.

## Why Disaggregation Can Improve Performance

**[Stable concept]** Prefill and decode stress the GPU differently: prefill is prompt-token-heavy and often compute intensive; decode is autoregressive and commonly memory-bandwidth/KV-capacity sensitive.

**[Stable concept]** Separating prefill and decode can reduce interference: large incoming prompts no longer have to share the exact same scheduler iteration and GPU with active decode requests.

**[Stable concept]** Independent pools make it possible to allocate different numbers or shapes of GPUs to prefill and decode as traffic changes.

**[Current implementation detail]** Dynamo docs state that disaggregated serving allows better hardware allocation and scalability, and that long-context prefill on dedicated prefill engines can avoid blocking ongoing decode requests. Source: Dynamo disaggregated serving concept page.

**[Current implementation detail]** Dynamo docs describe runtime-reconfigurable xPyD, where workers register through discovery and the router incorporates added workers into routing decisions. Source: Dynamo disaggregated serving concept page.

**[Experimental assumption]** The lab should expect disaggregation to help most on prefill-heavy or very-prefill-heavy workloads, especially when active decode streams would otherwise suffer from large prompt arrivals.

## Why Disaggregation Can Make Performance Worse

**[Stable concept]** Disaggregation adds work that aggregated serving does not need: cross-worker coordination, transfer metadata, KV movement, possible synchronization/polling, and more routing decisions.

**[Stable concept]** If prompts are short, output is small, the model is tiny, or the interconnect is slow, the cost of moving KV cache can exceed the scheduling benefit.

**[Stable concept]** If either pool is underprovisioned, disaggregation can move the bottleneck rather than remove it. A fast prefill pool can queue behind decode; a fast decode pool can sit idle waiting for prefill and KV transfer.

**[Current implementation detail]** Dynamo docs warn that disaggregation changes where compute and memory are consumed, does not make either phase free, and that optimal boundaries depend on model, quantization, GPU, backend, and request-length distribution. Source: Dynamo disaggregated serving concept page.

**[Current implementation detail]** Dynamo docs also describe mixed prefill/decode iterations in aggregated or decode engines and note that chunk-size policy affects ITL even when total prefill work is unchanged. Source: Dynamo disaggregated serving concept page.

**[Experimental assumption]** With Qwen3-0.6B or another small model, P/D disaggregation may be slower than aggregated SGLang. Treat that as a useful result, not a failed experiment.

## Chunked Prefill Positioning

**[Stable concept]** Chunked prefill splits a large prompt prefill into smaller pieces so the scheduler can interleave prefill chunks with decode work instead of letting one huge prefill monopolize scheduling for a long stretch.

**[Stable concept]** Chunked prefill mitigates interference between long-prefill requests and active decodes, but it does not independently scale prefill and decode across separate GPUs by itself. It is still an aggregated-worker scheduling strategy unless combined with P/D disaggregation.

**[Current implementation detail]** SGLang exposes `--chunked-prefill-size` as the relevant launch control. Source: SGLang server arguments docs.

**[Current implementation detail]** Dynamo docs describe the tradeoff: larger chunks create fewer but longer decode interruptions, while smaller chunks create more frequent but shorter interruptions; exact controls are backend-specific. Source: Dynamo disaggregated serving concept page.

**[Experimental assumption]** The lab should demonstrate chunked prefill before P/D disaggregation so you can distinguish scheduler interleaving benefits from true resource-pool separation.

## Metrics and Interpretation

**[Stable concept]** TTFT measures time from request start to first output token. It includes queueing, preprocessing, prefill, and any transfer/setup needed before first decode output.

**[Stable concept]** ITL measures gaps between streamed output tokens for a single request. TPOT is often used as a per-output-token latency summary. Request latency measures end-to-end completion.

**[Stable concept]** Throughput can be measured as requests/sec and output tokens/sec; both are needed because architectures can trade per-request latency against aggregate token production.

**[Experimental assumption]** Streaming timestamps are required for true per-token ITL distributions. Do not label request-level output-token averages as a true ITL distribution.

## Phase 0 Proceed/No-Proceed Checklist

Before Phase 1, the following claims are considered validated from current docs but not from local execution:

- SGLang is the backend engine that executes the model in this lab.
- Dynamo adds frontend, discovery, routing, and orchestration around backend workers.
- NIXL is the current Dynamo/SGLang KV-transfer path to test first for P/D disaggregation.
- SGLang currently documents PD roles and a NIXL transfer backend.
- Dynamo currently documents SGLang-specific bootstrap metadata and decode/prefill coordination.
- The two-GPU P/D experiment is architecturally valid but performance-neutral until measured.

Do not proceed to Phase 1 until you can explain the following without looking.

1. In one sentence each, what is SGLang responsible for, what is Dynamo responsible for, and what is NIXL responsible for?
2. In aggregated SGLang serving, where do prefill, decode, model weights, and KV cache live?
3. Why does decode need the KV cache produced during prefill?
4. In P/D disaggregation, why is transferring KV cache better than recomputing the prompt on the decode worker?
5. What extra work does disaggregation add compared with aggregated serving?
6. What is the difference between chunked prefill and prefill/decode disaggregation?
7. Why can a long prompt hurt ITL for already-running decode requests in aggregated serving?
8. What current Dynamo component or logic chooses prefill/decode workers, and what signals may it use?
9. What hardware/software facts determine whether same-node or cross-node KV transfer is fast enough?
10. Which workload shapes do you expect to favor aggregated serving, and which might favor P/D disaggregation?
