#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-0.6B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-0.6b}"
HOST_PORT="${HOST_PORT:-30000}"
HOST="${HOST:-0.0.0.0}"
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs}"
PID_FILE="${PID_FILE:-$LOG_DIR/sglang.pid}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/sglang.log}"

if [ -d /workspace ]; then
  export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
else
  export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
fi

mkdir -p "$HF_HOME" "$LOG_DIR"

if [ -x "$VENV_DIR/bin/sglang" ]; then
  SGLANG_BIN="$VENV_DIR/bin/sglang"
elif command -v sglang >/dev/null 2>&1; then
  SGLANG_BIN="$(command -v sglang)"
else
  echo "SGLang is not installed."
  echo "Run:"
  echo "  uv venv --python 3.12 .venv"
  echo "  source .venv/bin/activate"
  echo "  uv pip install --prerelease=allow sglang"
  exit 1
fi

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "SGLang already appears to be running with PID $(cat "$PID_FILE")."
  echo "Log: $LOG_FILE"
  exit 0
fi

echo "Starting SGLang aggregated baseline"
echo "  binary:             $SGLANG_BIN"
echo "  model path:         $MODEL_PATH"
echo "  served model name:  $SERVED_MODEL_NAME"
echo "  URL:                http://localhost:$HOST_PORT"
echo "  HF cache:           $HF_HOME"
echo "  log file:           $LOG_FILE"
echo

nohup "$SGLANG_BIN" serve "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" \
  --port "$HOST_PORT" \
  > "$LOG_FILE" 2>&1 &

echo "$!" > "$PID_FILE"
echo "Started SGLang with PID $(cat "$PID_FILE")."
echo "Watch startup logs with:"
echo "  tail -f $LOG_FILE"
