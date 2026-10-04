#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs}"
PID_FILE="${PID_FILE:-$LOG_DIR/sglang.pid}"

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  PID="$(cat "$PID_FILE")"
  kill "$PID"
  rm -f "$PID_FILE"
  echo "Stopped SGLang process $PID"
else
  echo "No running SGLang process found via $PID_FILE"
fi
