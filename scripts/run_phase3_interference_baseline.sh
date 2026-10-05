#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${OUT_DIR:-outputs/phase3_interference_baseline_no_chunked_controlled}"
HOST_PORT="${HOST_PORT:-30000}"
SGLANG_BASE_URL="${SGLANG_BASE_URL:-http://localhost:$HOST_PORT}"

mkdir -p "$OUT_DIR"

bash scripts/stop_sglang.sh || true

SGLANG_EXTRA_ARGS="--chunked-prefill-size -1 --mem-fraction-static 0.704 --max-prefill-tokens 16384 --schedule-policy fcfs --schedule-conservativeness 1.0" \
  LOG_FILE="logs/sglang_phase3_no_chunked.log" \
  PID_FILE="logs/sglang_phase3_no_chunked.pid" \
  bash scripts/start_sglang.sh

for i in $(seq 1 180); do
  if curl -fsS "$SGLANG_BASE_URL/model_info" > "$OUT_DIR/model_info.json" 2>/dev/null; then
    break
  fi
  if [ "$i" -eq 180 ]; then
    echo "SGLang did not become ready" >&2
    tail -120 logs/sglang_phase3_no_chunked.log >&2 || true
    exit 1
  fi
  sleep 2
done

python3 benchmarks/interference_sglang.py \
  --base-url "$SGLANG_BASE_URL" \
  --out-dir "$OUT_DIR" \
  --server-config-label "chunked_prefill_disabled_chunked_prefill_size_-1" \
  --background-decode-requests "${BACKGROUND_DECODE_REQUESTS:-4}" \
  --background-max-output-tokens "${BACKGROUND_MAX_OUTPUT_TOKENS:-2000}" \
  --injection-delay-s "${INJECTION_DELAY_S:-1.5}" \
  --injected-prefill-terms "${INJECTED_PREFILL_TERMS:-8192}" \
  --injected-max-output-tokens "${INJECTED_MAX_OUTPUT_TOKENS:-64}"

SGLANG_VERSION=$(python3 -c 'import sglang; print(getattr(sglang, "__version__", "unknown"))')
{
  echo "# Phase 3 baseline no-chunked configuration"
  echo
  echo "Captured at: $(date -Is)"
  echo "Git HEAD: $(git rev-parse HEAD)"
  echo "SGLang version: $SGLANG_VERSION"
  echo "Launch extra args: --chunked-prefill-size -1 --mem-fraction-static 0.704 --max-prefill-tokens 16384 --schedule-policy fcfs --schedule-conservativeness 1.0"
  echo "Background decode requests: ${BACKGROUND_DECODE_REQUESTS:-4}"
  echo "Background max output tokens: ${BACKGROUND_MAX_OUTPUT_TOKENS:-2000}"
  echo "Injection delay seconds: ${INJECTION_DELAY_S:-1.5}"
  echo "Injected prefill terms target: ${INJECTED_PREFILL_TERMS:-8192}"
  echo "Injected max output tokens: ${INJECTED_MAX_OUTPUT_TOKENS:-64}"
  echo
  echo "GPU:"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
  echo
  echo "Server args subset from log:"
  grep -o "'chunked_prefill_size': [^,}]*\|'max_prefill_tokens': [^,}]*\|'enable_mixed_chunk': [^,}]*\|'schedule_policy': [^,}]*\|'schedule_conservativeness': [^,}]*\|'max_running_requests': [^,}]*\|'max_total_tokens': [^,}]*\|'mem_fraction_static': [^,}]*" logs/sglang_phase3_no_chunked.log | tail -20 || true
} > "$OUT_DIR/repro.txt"
