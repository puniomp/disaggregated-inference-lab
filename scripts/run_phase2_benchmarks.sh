#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:30000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-0.6b}"
PROFILES="${PROFILES:-baseline,prefill-heavy,very-prefill-heavy,decode-heavy}"
CONCURRENCY="${CONCURRENCY:-1,8,16,32}"
REQUESTS_PER_CONCURRENCY="${REQUESTS_PER_CONCURRENCY:-32}"
OUT_DIR="${OUT_DIR:-outputs/phase2}"

python3 benchmarks/benchmark_sglang.py \
  --base-url "$BASE_URL" \
  --model "$SERVED_MODEL_NAME" \
  --profiles "$PROFILES" \
  --concurrency "$CONCURRENCY" \
  --requests-per-concurrency "$REQUESTS_PER_CONCURRENCY" \
  --out-dir "$OUT_DIR"

python3 benchmarks/plot_phase2.py \
  --summary "$OUT_DIR/summary.csv" \
  --out-dir "$OUT_DIR/charts"
