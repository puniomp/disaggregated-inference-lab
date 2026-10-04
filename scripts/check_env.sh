#!/usr/bin/env bash
set -euo pipefail

echo "== NVIDIA GPU / driver =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
  echo
  echo "== GPU model / VRAM summary =="
  nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free,compute_cap --format=csv
else
  echo "nvidia-smi was not found. Use a RunPod image with NVIDIA drivers available on the host."
fi

echo
echo "== CUDA compiler, if installed on host =="
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version
else
  echo "nvcc is not installed on the host. That is OK for the Docker path; SGLang's container provides its CUDA runtime."
fi

echo
echo "== Python =="
if command -v python3 >/dev/null 2>&1; then
  python3 --version
else
  echo "python3 was not found on the host. That is OK for the Docker path, but useful for local helper scripts later."
fi

echo
echo "== uv / pip =="
if command -v uv >/dev/null 2>&1; then
  uv --version
else
  echo "uv was not found. Install it with: python3 -m pip install uv"
fi
python3 -m pip --version 2>/dev/null || echo "pip was not found for python3."

echo
echo "== SGLang import check =="
python3 -c 'import sglang; print("sglang", getattr(sglang, "__version__", "version unknown"))' 2>/dev/null || echo "sglang is not installed in the current Python environment yet."

echo
echo "== Disk space =="
df -h .

echo
echo "== Hugging Face cache location =="
echo "HF_HOME=${HF_HOME:-${HF_CACHE_DIR:-$HOME/.cache/huggingface}}"
