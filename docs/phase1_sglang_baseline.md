# Phase 1: SGLang Aggregated Baseline

Validated against current SGLang docs on 2026-10-03.

Scope:

```
Client
  |
  v
SGLang
  |
  v
GPU
  |
  +-- Prefill
  +-- Decode
```

This phase intentionally does not use Dynamo, NIXL, P/D disaggregation, or Kubernetes.

## Current Official SGLang Setup Path

The current SGLang installation docs support both Docker and a Python environment installed with `uv`. The Docker path is cleanest on a Docker-capable GPU host. This RunPod shell is already inside a GPU container and does not expose `docker`, so use the official `uv` path inside `/workspace` instead.

For this lab, use:

- Model: `Qwen/Qwen3-0.6B`
- Served model name: `qwen3-0.6b`
- Port: `30000`
- Python: `3.12`
- Install command: `uv pip install --prerelease=allow sglang`

## Recommended RunPod GPU

Recommended: NVIDIA L4 24GB, RTX 4090 24GB, A10 24GB, or A40 48GB.

Minimum practical VRAM for this small model: 8GB should be enough for a basic smoke test of `Qwen/Qwen3-0.6B`, but choose 16GB or 24GB if possible so later long-context and chunked-prefill experiments are not immediately constrained by KV-cache memory.

Do not change CUDA drivers manually unless SGLang fails with a clear driver/runtime compatibility error. On the checked RunPod, `nvidia-smi` reported driver `610.43.02` and CUDA UMD `13.3`, which is compatible with the current SGLang CUDA 13 direction. The host/container also has CUDA toolkit `12.8`; that is not a reason by itself to change drivers.

## Environment Checks

Run this first:

```bash
bash scripts/check_env.sh
```

It prints:

- `nvidia-smi`
- GPU model, total VRAM, free VRAM, driver version, and compute capability
- host CUDA compiler version if `nvcc` exists
- Python version
- `uv` / `pip` availability
- whether SGLang is already importable
- disk space
- Hugging Face cache location

## Launch Arguments

The important SGLang command is intentionally visible in `scripts/start_sglang.sh`:

```bash
sglang serve "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" \
  --port "$HOST_PORT"
```

`MODEL_PATH`

- Controls: Hugging Face model ID or local model directory.
- Why here: `Qwen/Qwen3-0.6B` is small enough for a one-GPU baseline and is used in current SGLang docs as an example model.
- If changed: SGLang downloads/loads a different model; VRAM, disk, startup time, tokenizer behavior, chat template, and output quality may change.

`--served-model-name`

- Controls: the model name clients send in OpenAI-compatible API requests.
- Why here: the smoke test can use a short stable name, `qwen3-0.6b`, rather than the full Hugging Face ID.
- If changed: client requests must use the new model name.

`--host 0.0.0.0`

- Controls: which network interface the HTTP server binds to.
- Why here: RunPod exposes services from the container/host; binding all interfaces is the normal server behavior.
- If changed to `127.0.0.1`: access is limited to localhost from inside the same network namespace, which can make remote access harder.

`--port 30000`

- Controls: SGLang's HTTP server port inside the container.
- Why here: `30000` is the port used in current SGLang quickstart examples.
- If changed: update `HOST_PORT` for the smoke test.

Environment variables used by the script:

`HF_HOME`

- Controls where Hugging Face model files are cached.
- Why here: defaults to `/workspace/.cache/huggingface` on RunPod so model files persist in the normal workspace area.
- If changed: model downloads and cache reuse move to a different directory.

`VENV_DIR`

- Controls which virtual environment contains `sglang`.
- Why here: defaults to `.venv` in the project directory.
- If changed: install SGLang into that environment or ensure `sglang` is otherwise on `PATH`.

## One Request Lifecycle

1. Model weights are loaded by the SGLang server process and placed in GPU memory as needed for inference.
2. Your prompt is sent by `curl` to SGLang's OpenAI-compatible `/v1/chat/completions` endpoint.
3. SGLang tokenizes the prompt and schedules the request on the single local worker.
4. During prefill, the GPU runs the model over the prompt tokens and creates KV cache entries for the prompt.
5. The KV cache lives in the same SGLang worker's GPU memory.
6. The first token can be returned only after request admission, tokenization, prefill, and the first decode step are complete.
7. During each decode step, SGLang runs the model for the next generated token.
8. Decode reads the existing KV cache and appends new KV entries for generated tokens.
9. No KV transfer between GPUs is required because prefill and decode are colocated in one SGLang worker on one GPU.

## RunPod Commands

From the project root:

```bash
chmod +x scripts/*.sh
bash scripts/check_env.sh

uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install --prerelease=allow sglang

bash scripts/start_sglang.sh
```

Watch the startup logs:

```bash
tail -f logs/sglang.log
```

Wait until SGLang logs that the server is ready. In a second terminal:

```bash
bash scripts/smoke_test.sh
```

To stop:

```bash
bash scripts/stop_sglang.sh
```

## Validated Run

Validated on RunPod after Phase 1 execution.

Measured or observed facts:

- GPU: `NVIDIA GeForce RTX 4090`
- User-reported VRAM: `24 GB`
- Model served: `Qwen/Qwen3-0.6B`
- Served model name in response: `qwen3-0.6b`
- Endpoint shape: OpenAI-compatible chat completions endpoint
- Smoke test result: request succeeded
- Smoke-test token counts:
  - `prompt_tokens`: `23`
  - `completion_tokens`: `64`
  - `total_tokens`: `87`
- Smoke-test finish reason: `length`

Architectural interpretation:

- This was aggregated inference.
- There was one SGLang worker.
- There was one GPU.
- Prefill and decode were colocated in the same SGLang worker on the same GPU.
- KV cache remained local to that worker/GPU.
- No cross-GPU KV transfer was required.
- No Dynamo, NIXL, P/D disaggregation, or Kubernetes component was involved.

The `length` finish reason means the smoke test hit its configured output-token cap. It is not, by itself, a serving failure or a performance measurement.

## What Phase 1 Proved / Did Not Prove

Phase 1 proved:

- The RunPod environment works for this baseline.
- SGLang can serve the selected model.
- The OpenAI-compatible request path works.
- The model can execute end-to-end on the GPU.

Phase 1 did not prove:

- TTFT performance.
- ITL performance.
- Throughput.
- GPU saturation.
- Compute-bound vs memory-bandwidth-bound behavior.
- Chunked-prefill benefits.
- P/D disaggregation benefits.

## Reproducibility Capture Before Terminating RunPod

Run these before terminating the pod and save the output with your experiment notes:

```bash
cd /workspace/inference-disaggregation-lab

date -u
hostname
pwd

nvidia-smi
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free,compute_cap --format=csv
nvcc --version || true

python3 --version
uv --version || true
source .venv/bin/activate
python -m pip freeze | tee phase1_pip_freeze.txt
python -c 'import sglang; print("sglang", getattr(sglang, "__version__", "version unknown"))'

df -h .
du -sh /workspace/.cache/huggingface 2>/dev/null || true

curl -s http://localhost:30000/model_info || true
tail -n 200 logs/sglang.log > phase1_sglang_startup_tail.log
```
