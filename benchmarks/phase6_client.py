#!/usr/bin/env python3
"""Phase 6 benchmark client for matched aggregated-vs-P/D experiments.

This file reconstructs the validated Phase 6 workload generators after an
ephemeral pod loss. It does not launch servers; it only sends OpenAI-compatible
streaming requests to a caller-provided endpoint.

Topology note for interpreting artifacts:
- Previous P/D validation pod: GPU0 <-> GPU1 = SYS, GPUs on different NUMA nodes.
- Current experimental pod: GPU0 <-> GPU1 = NODE, both GPUs NUMA affinity 0.
Cross-pod latency numbers are historical/reference results, not topology-identical
controls.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODEL = "llama3.1-8b-instruct"
DEFAULT_BASE_URL = "http://localhost:8000"
KNOWN_GOOD_DECODE_SEED = "decode-validation-001"
PREFILL_TARGET = 8000
PREFILL_MIN = 7500
PREFILL_MAX = 8500
DEFAULT_FRONTEND_LOG = "/workspace/dynamo-phase5/phase5_launch_logs/frontend.log"
DEFAULT_PREFILL_LOG = "/workspace/dynamo-phase5/phase5_launch_logs/prefill.log"

log_lock = threading.Lock()


@dataclass(frozen=True)
class RequestSpec:
    label: str
    request_index: int
    messages: list[dict[str, str]]
    max_tokens: int
    client_estimated_prompt_tokens: int | None = None


def perf_now() -> float:
    return time.perf_counter()


def wall_now() -> float:
    return time.time()


def utc_run_id(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{prefix}-{stamp}"


def seed_for(run_id: str, request_index: int) -> str:
    """Stable seed derived from (run_id, request_index)."""
    return hashlib.sha256(f"{run_id}:{request_index}".encode("utf-8")).hexdigest()


def load_tokenizer(tokenizer_path: str | None):
    if not tokenizer_path:
        raise SystemExit("--tokenizer-path is required for token-aware prefill/mixed workloads")
    path = Path(tokenizer_path)
    if not path.exists():
        raise SystemExit(f"Tokenizer path does not exist; refusing to download silently: {path}")
    from transformers import AutoTokenizer  # Imported lazily so inspect/help works without it.
    return AutoTokenizer.from_pretrained(str(path), local_files_only=True)


def apply_chat_token_count(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if isinstance(encoded, dict):
        return len(encoded["input_ids"])
    try:
        return len(encoded["input_ids"])
    except Exception:
        return len(encoded)


def unique_record(seed: str, idx: int) -> str:
    h1 = hashlib.sha256(f"{seed}:{idx}".encode("utf-8")).hexdigest()
    h2 = hashlib.sha256(f"{idx}:{seed}:payload".encode("utf-8")).hexdigest()
    return (
        f"record_{idx:04d} trace={h1[:16]} shard={h1[16:24]} "
        f"route={h1[24:32]} value={h2[:20]} checksum={h2[20:32]}. "
        f"Observation {idx:04d} contains unique deterministic fields for cache-resistant prefill sizing."
    )


def prefill_messages_from_records(seed: str, records: int) -> list[dict[str, str]]:
    # Unique record content intentionally begins at the start of the user message.
    body = "\n".join(unique_record(seed, i) for i in range(records))
    user = body + "\n\nSummarize the records briefly."
    return [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": user},
    ]


def build_prefill_spec(tokenizer: Any, run_id: str, request_index: int, label: str) -> RequestSpec:
    seed = seed_for(run_id, request_index)
    lo, hi = 1, 1
    while apply_chat_token_count(tokenizer, prefill_messages_from_records(seed, hi)) < PREFILL_MIN:
        hi *= 2
        if hi > 5000:
            raise RuntimeError("Unable to size prefill prompt into target range")

    best: tuple[int, int] | None = None
    while lo <= hi:
        mid = (lo + hi) // 2
        messages = prefill_messages_from_records(seed, mid)
        count = apply_chat_token_count(tokenizer, messages)
        if best is None or abs(count - PREFILL_TARGET) < abs(best[1] - PREFILL_TARGET):
            best = (mid, count)
        if count < PREFILL_MIN:
            lo = mid + 1
        elif count > PREFILL_MAX:
            hi = mid - 1
        elif count < PREFILL_TARGET:
            lo = mid + 1
        else:
            hi = mid - 1

    if best is None or not (PREFILL_MIN <= best[1] <= PREFILL_MAX):
        raise RuntimeError(f"No prefill candidate in {PREFILL_MIN}-{PREFILL_MAX}; best={best}")
    messages = prefill_messages_from_records(seed, best[0])
    return RequestSpec(label=label, request_index=request_index, messages=messages, max_tokens=64, client_estimated_prompt_tokens=best[1])


def sustained_decode_messages() -> list[dict[str, str]]:
    # This exact semantic construction produced the validated 87-token prompt and
    # 768-token finish_reason=length standalone decode request. Do not substitute
    # per-request hash-derived seed text here; that caused early natural stops.
    prompt = (
        f"Seed {KNOWN_GOOD_DECODE_SEED}. Write a numbered technical checklist with exactly 120 items. "
        "Each item must be one complete sentence of at least twelve words about LLM serving, "
        "prefill, decode, batching, queues, or cache transfer. Do not stop early."
    )
    return [{"role": "user", "content": prompt}]


def build_decode_spec(request_index: int, label: str) -> RequestSpec:
    return RequestSpec(label=label, request_index=request_index, messages=sustained_decode_messages(), max_tokens=768)


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * quantile
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def summarize(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "p50_s": percentile(values, 0.50),
        "p95_s": percentile(values, 0.95),
        "p99_s": percentile(values, 0.99),
        "max_s": max(values) if values else None,
    }


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def read_new_log(path: str, offset: int) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    with p.open("r", errors="replace") as f:
        f.seek(offset)
        return f.read()


def parse_worker_line(line: str) -> dict[str, str] | None:
    clean = strip_ansi(line)
    if "request completed" not in clean or "prefill_worker_id" not in clean:
        return None
    out = {"raw_line": clean}
    for key in ["prefill_worker_id", "decode_worker_id", "ttft_ms", "elapsed_ms", "input_tokens", "output_tokens"]:
        m = re.search(key + r"=\"?([0-9.]+)\"?", clean)
        if m:
            out[key] = m.group(1)
    ids = re.findall(r"request_id=([0-9a-f-]{36})", clean)
    if ids:
        out["dynamo_request_id"] = ids[0]
    return out


def parse_new_worker_lines(log_text: str) -> list[dict[str, str]]:
    rows = []
    for line in log_text.splitlines():
        parsed = parse_worker_line(line)
        if parsed:
            rows.append(parsed)
    return rows


def find_matching_worker_line(rows: list[dict[str, str]], prompt_tokens: int | None, completion_tokens: int | None) -> dict[str, str]:
    if prompt_tokens is not None and completion_tokens is not None:
        matches = [
            r for r in rows
            if r.get("input_tokens") == str(prompt_tokens) and r.get("output_tokens") == str(completion_tokens)
        ]
        if matches:
            return matches[-1]
    return rows[-1] if rows else {}


def parse_sse_line(line: str) -> dict[str, Any] | str | None:
    line = line.strip()
    if not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if payload == "[DONE]":
        return "DONE"
    return json.loads(payload)


def send_streaming_request(
    *,
    base_url: str,
    model: str,
    spec: RequestSpec,
    architecture: str,
    run_id: str,
    frontend_log: str,
    first_content_event: threading.Event | None = None,
) -> dict[str, Any]:
    with log_lock:
        log_offset = Path(frontend_log).stat().st_size if Path(frontend_log).exists() else 0

    payload = {
        "model": model,
        "messages": spec.messages,
        "max_tokens": spec.max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start_s = perf_now()
    start_wall_time = wall_now()
    first_content_s = None
    finish_s = None
    all_event_times_s: list[float] = []
    content_event_times_s: list[float] = []
    usage = None
    finish_reason = None
    response_id = None
    http_status = None
    error = None
    error_body = None
    text_preview: list[str] = []

    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            http_status = response.status
            for raw in response:
                event_s = perf_now()
                parsed = parse_sse_line(raw.decode("utf-8", errors="replace"))
                if parsed is None:
                    continue
                if parsed == "DONE":
                    break
                all_event_times_s.append(event_s)
                response_id = response_id or parsed.get("id")
                if parsed.get("usage"):
                    usage = parsed["usage"]
                emitted_content = False
                for choice in parsed.get("choices", []):
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        emitted_content = True
                        if len("".join(text_preview)) < 300:
                            text_preview.append(delta["content"])
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice.get("finish_reason")
                if emitted_content:
                    if first_content_s is None:
                        first_content_s = event_s
                        if first_content_event is not None:
                            first_content_event.set()
                    content_event_times_s.append(event_s)
    except urllib.error.HTTPError as exc:
        http_status = exc.code
        error = repr(exc)
        error_body = exc.read().decode("utf-8", errors="replace")
    except Exception as exc:  # Keep raw failure record; caller decides validity.
        error = repr(exc)
    finally:
        finish_s = perf_now()

    time.sleep(0.05)
    new_log = read_new_log(frontend_log, log_offset)
    worker_rows = parse_new_worker_lines(new_log)
    prompt_tokens = (usage or {}).get("prompt_tokens")
    completion_tokens = (usage or {}).get("completion_tokens")
    worker = find_matching_worker_line(worker_rows, prompt_tokens, completion_tokens)
    gaps = [b - a for a, b in zip(content_event_times_s, content_event_times_s[1:])]
    tpot_s = None
    if completion_tokens and completion_tokens > 1 and first_content_s is not None:
        tpot_s = (finish_s - first_content_s) / (completion_tokens - 1)

    return {
        "architecture": architecture,
        "run_id": run_id,
        "request_index": spec.request_index,
        "request_label": spec.label,
        "response_id": response_id,
        "http_status": http_status,
        "error": error,
        "error_body": error_body,
        "start_wall_time": start_wall_time,
        "start_s": start_s,
        "first_content_token_s": first_content_s,
        "finish_s": finish_s,
        "ttft_s": None if first_content_s is None else first_content_s - start_s,
        "e2e_latency_s": finish_s - start_s,
        "tpot_s": tpot_s,
        "all_stream_event_times_s": all_event_times_s,
        "content_event_times_s": content_event_times_s,
        "inter_content_event_gaps_s": gaps,
        "inter_content_event_summary_s": summarize(gaps),
        "client_estimated_prompt_tokens": spec.client_estimated_prompt_tokens,
        "usage": usage,
        "server_prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": (usage or {}).get("total_tokens"),
        "cached_tokens": ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
        "finish_reason": finish_reason,
        "prefill_worker_id": worker.get("prefill_worker_id"),
        "decode_worker_id": worker.get("decode_worker_id"),
        "worker_ids_differ": bool(worker.get("prefill_worker_id") and worker.get("decode_worker_id") and worker.get("prefill_worker_id") != worker.get("decode_worker_id")),
        "frontend_worker_line": worker,
        "frontend_log_excerpt": new_log[-4000:],
        "text_preview": "".join(text_preview)[:300],
    }


def write_raw_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())


def classify_mixed_gaps(backgrounds: list[dict[str, Any]], injected: dict[str, Any]) -> tuple[list[float], list[float], list[float], list[dict[str, Any]]]:
    before: list[float] = []
    during: list[float] = []
    after: list[float] = []
    per_stream = []
    inj_start = injected["start_s"]
    inj_first = injected["first_content_token_s"]
    for row in backgrounds:
        local = {"before": [], "during": [], "after": []}
        events = row.get("content_event_times_s") or []
        for a, b in zip(events, events[1:]):
            gap = b - a
            midpoint = (a + b) / 2.0
            if midpoint < inj_start:
                before.append(gap)
                local["before"].append(gap)
            elif inj_first is not None and midpoint <= inj_first:
                during.append(gap)
                local["during"].append(gap)
            else:
                after.append(gap)
                local["after"].append(gap)
        per_stream.append({
            "request_label": row.get("request_label"),
            "before": summarize(local["before"]),
            "during": summarize(local["during"]),
            "after": summarize(local["after"]),
        })
    return before, during, after, per_stream


def prefill_chunk_pairs(prefill_log_text: str) -> list[dict[str, Any]]:
    pairs = []
    for line in prefill_log_text.splitlines():
        clean = strip_ansi(line)
        if "Prefill batch" not in clean or "#new-token" not in clean or "#cached-token" not in clean:
            continue
        m = re.search(r"#new-token:\s*(\d+).*?#cached-token:\s*(\d+)", clean)
        if m:
            pairs.append({"new_token": int(m.group(1)), "cached_token": int(m.group(2)), "line": clean})
    return pairs


def run_prefill(args: argparse.Namespace, tokenizer: Any, run_id: str) -> dict[str, Any]:
    spec = build_prefill_spec(tokenizer, run_id, 0, "prefill_heavy_0")
    row = send_streaming_request(base_url=args.base_url, model=args.model, spec=spec, architecture=args.architecture, run_id=run_id, frontend_log=args.frontend_log)
    return {"raw_rows": [row], "summary": {"workload": "prefill", "request": row}}


def run_decode(args: argparse.Namespace, run_id: str) -> dict[str, Any]:
    spec = build_decode_spec(0, "decode_heavy_0")
    row = send_streaming_request(base_url=args.base_url, model=args.model, spec=spec, architecture=args.architecture, run_id=run_id, frontend_log=args.frontend_log)
    return {"raw_rows": [row], "summary": {"workload": "decode", "request": row}}


def run_mixed(args: argparse.Namespace, tokenizer: Any, run_id: str) -> dict[str, Any]:
    prefill_offset = Path(args.prefill_log).stat().st_size if Path(args.prefill_log).exists() else 0
    bg_events = [threading.Event() for _ in range(4)]
    bg_specs = [build_decode_spec(i, f"mixed_background_decode_{i}") for i in range(4)]
    injected_spec = build_prefill_spec(tokenizer, run_id, 1000, "mixed_injected_prefill")

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        bg_futs = [
            pool.submit(send_streaming_request, base_url=args.base_url, model=args.model, spec=spec,
                        architecture=args.architecture, run_id=run_id, frontend_log=args.frontend_log,
                        first_content_event=bg_events[i])
            for i, spec in enumerate(bg_specs)
        ]
        for event in bg_events:
            if not event.wait(timeout=args.background_start_timeout_s):
                raise RuntimeError("Not all background streams produced a first content token before injection")
        if any(f.done() for f in bg_futs):
            raise RuntimeError("A background stream finished before injection")
        inj_fut = pool.submit(send_streaming_request, base_url=args.base_url, model=args.model, spec=injected_spec,
                              architecture=args.architecture, run_id=run_id, frontend_log=args.frontend_log)
        backgrounds = [f.result() for f in bg_futs]
        injected = inj_fut.result()

    prefill_new = read_new_log(args.prefill_log, prefill_offset)
    before, during, after, per_stream = classify_mixed_gaps(backgrounds, injected)
    active_at_injection = [
        b["first_content_token_s"] is not None and b["first_content_token_s"] <= injected["start_s"] and b["finish_s"] >= injected["start_s"]
        for b in backgrounds
    ]
    active_full_ttft = [
        b["first_content_token_s"] is not None and injected["first_content_token_s"] is not None
        and b["first_content_token_s"] <= injected["start_s"] and b["finish_s"] >= injected["first_content_token_s"]
        for b in backgrounds
    ]
    summary_obj = {
        "workload": "mixed",
        "during_definition": "client-observed prefill/TTFT proxy: injected request submission to injected first content token; not GPU kernel timing",
        "background_completion_tokens": [b.get("completion_tokens") for b in backgrounds],
        "background_finish_reasons": [b.get("finish_reason") for b in backgrounds],
        "active_at_injection": active_at_injection,
        "active_for_entire_injected_ttft_proxy_window": active_full_ttft,
        "injected": injected,
        "background_gap_stats": {
            "before": summarize(before),
            "during": summarize(during),
            "after": summarize(after),
            "per_stream": per_stream,
        },
        "prefill_chunk_pairs": prefill_chunk_pairs(prefill_new),
        "success_criteria": {
            "all_http_200": all(b.get("http_status") == 200 for b in backgrounds) and injected.get("http_status") == 200,
            "each_background_ge_500_completion_tokens": all((b.get("completion_tokens") or 0) >= 500 for b in backgrounds),
            "all_4_background_streams_active_when_injection_occurs": all(active_at_injection),
            "background_decode_active_through_injected_ttft_proxy": all(active_full_ttft),
            "injected_prompt_7500_8500_tokens": injected.get("server_prompt_tokens") is not None and PREFILL_MIN <= injected.get("server_prompt_tokens") <= PREFILL_MAX,
            "injected_cached_tokens_small_relative_to_prompt": (injected.get("cached_tokens") or 0) <= 64,
            "sufficient_before_during_after_observations": len(before) > 0 and len(during) > 0 and len(after) > 0,
        },
    }
    return {"raw_rows": backgrounds + [injected], "summary": summary_obj, "mixed_events": {"backgrounds": backgrounds, "injected": injected, **summary_obj["background_gap_stats"]}}


def write_outputs(out_dir: Path, raw_rows: list[dict[str, Any]], summary_obj: dict[str, Any], mixed_events: dict[str, Any] | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_requests.jsonl"
    summary_path = out_dir / "summary.json"
    mixed_path = out_dir / "mixed_events.json"
    write_raw_jsonl(raw_path, raw_rows)  # Raw observations are written before summaries.
    summary_path.write_text(json.dumps(summary_obj, indent=2, sort_keys=True), encoding="utf-8")
    if mixed_events is not None:
        mixed_path.write_text(json.dumps(mixed_events, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 6 matched workload client")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--architecture", choices=["pd", "aggregated"], required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--workloads", required=True, help="Comma-separated: prefill,decode,mixed")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--frontend-log", default=DEFAULT_FRONTEND_LOG)
    parser.add_argument("--prefill-log", default=DEFAULT_PREFILL_LOG)
    parser.add_argument("--background-start-timeout-s", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workloads = [w.strip() for w in args.workloads.split(",") if w.strip()]
    invalid = sorted(set(workloads) - {"prefill", "decode", "mixed"})
    if invalid:
        raise SystemExit(f"Unsupported workloads: {invalid}")
    run_id = args.run_id or utc_run_id(f"phase6-{args.architecture}")
    tokenizer = None
    if any(w in {"prefill", "mixed"} for w in workloads):
        tokenizer = load_tokenizer(args.tokenizer_path)

    all_raw: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {
        "run_id": run_id,
        "architecture": args.architecture,
        "model": args.model,
        "workloads": workloads,
        "topology_note": {
            "previous_pd_validation_pod": "GPU0 <-> GPU1 = SYS; GPUs on different NUMA nodes",
            "current_experimental_pod": "GPU0 <-> GPU1 = NODE; both GPUs NUMA affinity 0",
            "interpretation": "Cross-pod latency numbers are historical/reference results, not topology-identical controls.",
        },
        "results": {},
    }
    mixed_events = None

    for workload in workloads:
        if workload == "prefill":
            result = run_prefill(args, tokenizer, run_id)
        elif workload == "decode":
            result = run_decode(args, run_id)
        elif workload == "mixed":
            result = run_mixed(args, tokenizer, run_id)
            mixed_events = result.get("mixed_events")
        else:
            raise AssertionError(workload)
        all_raw.extend(result["raw_rows"])
        summaries["results"][workload] = result["summary"]

    write_outputs(Path(args.out_dir), all_raw, summaries, mixed_events)
    print(json.dumps({"run_id": run_id, "out_dir": args.out_dir, "workloads": workloads}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
