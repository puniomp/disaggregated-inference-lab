#!/usr/bin/env bash
set -euo pipefail

HOST_PORT="${HOST_PORT:-30000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-0.6b}"

curl "http://localhost:${HOST_PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${SERVED_MODEL_NAME}\",
    \"messages\": [
      {\"role\": \"user\", \"content\": \"In one short sentence, explain what prefill does in LLM inference.\"}
    ],
    \"max_tokens\": 64,
    \"temperature\": 0
  }"
