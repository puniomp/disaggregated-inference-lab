#!/usr/bin/env python3
"""Staged prefill/decode interference benchmark for Phase 3.

The benchmark starts sustained decode requests first, waits a controlled delay,
then injects one long-prefill request. It records streaming event arrival times.
A streaming event is not guaranteed to equal one model token, so the ITL fields
below are inter-stream-event gaps unless the server emits one token per event.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TERMS = [
    "prefill", "decode", "kv-cache", "scheduler", "latency", "throughput",
    "batching", "request", "attention", "memory", "queueing", "worker",
]

@dataclass(frozen=True)
class RequestSpec:
    role: str
    request_index: int
    prompt: str
    max_output_tokens: int

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[int(rank)]
    weight = rank - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight

def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None

def synthetic_context(target_terms: int) -> str:
    return " ".join(TERMS[i % len(TERMS)] for i in range(max(1, target_terms)))

def long_decode_prompt(request_index: int) -> str:
    return (
        f"Phase 3 background decode request {request_index}. "
        "Write exactly 180 numbered entries about LLM inference serving. "
        "Each entry must be a standalone sentence of 12 to 18 words. "
        "Start at 001 and continue sequentially. Do not write an introduction, "
        "summary, conclusion, apology, or meta commentary. Do not stop early. "
        "Continue until entry 180.\n\n"
        "Topic anchors: prefill, decode, KV cache, batching, scheduler, latency, "
        "throughput, GPU memory, queueing, request concurrency."
    )

def long_prefill_prompt(target_terms: int) -> str:
    return (
        "Phase 3 injected long-prefill request. Use the following synthetic "
        "context for an inference-serving benchmark. After reading it, respond "
        "with one concise paragraph explaining what the context is about.\n\n"
        f"{synthetic_context(target_terms)}"
    )

def build_payload(model: str, spec: RequestSpec, include_usage: bool, disable_thinking: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": spec.prompt}],
        "max_tokens": spec.max_output_tokens,
        "temperature": 0,
        "stream": True,
    }
    if include_usage:
        payload["stream_options"] = {"include_usage": True}
    if disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    return payload

def iter_sse_lines(response: Any):
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line or line.startswith(":"):
            continue
        if line.startswith("data:"):
            yield line[len("data:") :].strip()

def run_streaming_request(
    base_url: str,
    model: str,
    spec: RequestSpec,
    timeout_s: float,
    include_usage: bool,
    disable_thinking: bool,
    experiment_t0: float,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(build_payload(model, spec, include_usage, disable_thinking)).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    first_chunk_time: float | None = None
    last_chunk_time: float | None = None
    chunk_events: list[dict[str, Any]] = []
    output_parts: list[str] = []
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    status = "ok"
    error: str | None = None
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as response:
            for data in iter_sse_lines(response):
                if data == "[DONE]":
                    break
                event_time = time.perf_counter()
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                choices = obj.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                finish_reason = choice.get("finish_reason") or finish_reason
                delta = choice.get("delta") or {}
                piece = delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning") or ""
                if not piece:
                    continue
                output_parts.append(piece)
                if first_chunk_time is None:
                    first_chunk_time = event_time
                gap_s = None if last_chunk_time is None else event_time - last_chunk_time
                last_chunk_time = event_time
                chunk_events.append({
                    "role": spec.role,
                    "request_index": spec.request_index,
                    "chunk_index": len(chunk_events),
                    "arrival_s": event_time - experiment_t0,
                    "gap_s": gap_s,
                    "chars": len(piece),
                    "preview": piece[:80],
                })
    except urllib.error.HTTPError as exc:
        status = "error"
        error = f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:500]}"
    except Exception as exc:
        status = "error"
        error = repr(exc)
    end = time.perf_counter()
    completion_tokens = prompt_tokens = total_tokens = None
    if usage:
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
    request_tpot_s = None
    if first_chunk_time is not None and last_chunk_time is not None:
        denom = None
        if isinstance(completion_tokens, int) and completion_tokens > 1:
            denom = completion_tokens - 1
        elif len(chunk_events) > 1:
            denom = len(chunk_events) - 1
        if denom:
            request_tpot_s = (last_chunk_time - first_chunk_time) / denom
    gaps = [e["gap_s"] for e in chunk_events if e["gap_s"] is not None]
    return {
        "schema_version": 1,
        "timestamp_utc": now_iso(),
        "status": status,
        "error": error,
        "role": spec.role,
        "request_index": spec.request_index,
        "max_output_tokens": spec.max_output_tokens,
        "start_s": start - experiment_t0,
        "first_chunk_s": None if first_chunk_time is None else first_chunk_time - experiment_t0,
        "end_s": end - experiment_t0,
        "latency_s": end - start,
        "ttft_s": None if first_chunk_time is None else first_chunk_time - start,
        "request_tpot_s": request_tpot_s,
        "inter_chunk_gap_p50_s": percentile(gaps, 50),
        "inter_chunk_gap_p95_s": percentile(gaps, 95),
        "inter_chunk_gap_p99_s": percentile(gaps, 99),
        "inter_chunk_gap_max_s": max(gaps) if gaps else None,
        "emitted_stream_chunks": len(chunk_events),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "finish_reason": finish_reason,
        "output_chars": len("".join(output_parts)),
        "output_preview": "".join(output_parts)[:200],
        "chunks": chunk_events,
    }

def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row if key != "chunks"})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: v for k, v in row.items() if k != "chunks"})

def phase_for_gap(gap_end_s: float, injection_s: float, injected_first_chunk_s: float | None, injected_end_s: float | None) -> str:
    if gap_end_s < injection_s:
        return "before"
    boundary = injected_first_chunk_s if injected_first_chunk_s is not None else injected_end_s
    if boundary is not None and gap_end_s <= boundary:
        return "during_prefill_proxy"
    return "after"

def summarize_gaps(chunk_rows: list[dict[str, Any]], requests: list[dict[str, Any]], injection_s: float) -> list[dict[str, Any]]:
    injected = next((r for r in requests if r["role"] == "injected_prefill"), None)
    injected_first = injected.get("first_chunk_s") if injected else None
    injected_end = injected.get("end_s") if injected else None
    for row in chunk_rows:
        if row["role"] == "background_decode" and row.get("gap_s") is not None:
            row["phase"] = phase_for_gap(row["arrival_s"], injection_s, injected_first, injected_end)
        else:
            row["phase"] = None
    summaries = []
    for phase in ["before", "during_prefill_proxy", "after"]:
        vals = [r["gap_s"] for r in chunk_rows if r.get("role") == "background_decode" and r.get("phase") == phase and r.get("gap_s") is not None]
        summaries.append({
            "phase": phase,
            "samples": len(vals),
            "itl_p50_s": percentile(vals, 50),
            "itl_p95_s": percentile(vals, 95),
            "itl_p99_s": percentile(vals, 99),
            "itl_max_s": max(vals) if vals else None,
        })
    return summaries

def svg_plot(path: Path, chunk_rows: list[dict[str, Any]], injection_s: float, injected_first_s: float | None) -> None:
    pts = [r for r in chunk_rows if r.get("role") == "background_decode" and r.get("gap_s") is not None]
    if not pts:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg' width='900' height='360'></svg>\n")
        return
    max_x = max(r["arrival_s"] for r in pts)
    max_y = max(max(r["gap_s"] for r in pts), 0.01)
    w, h = 900, 360
    left, right, top, bottom = 70, 20, 20, 55
    plot_w, plot_h = w - left - right, h - top - bottom
    def x(v: float) -> float:
        return left + (v / max_x) * plot_w if max_x else left
    def y(v: float) -> float:
        return top + plot_h - (v / max_y) * plot_h
    elems = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' viewBox='0 0 {w} {h}'>"]
    elems.append("<rect width='100%' height='100%' fill='white'/>")
    if injected_first_s is not None:
        elems.append(f"<rect x='{x(injection_s):.1f}' y='{top}' width='{max(1, x(injected_first_s)-x(injection_s)):.1f}' height='{plot_h}' fill='#fee2e2'/>")
    elems.append(f"<line x1='{left}' y1='{top + plot_h}' x2='{left + plot_w}' y2='{top + plot_h}' stroke='#111'/>")
    elems.append(f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top + plot_h}' stroke='#111'/>")
    elems.append(f"<line x1='{x(injection_s):.1f}' y1='{top}' x2='{x(injection_s):.1f}' y2='{top + plot_h}' stroke='#dc2626' stroke-dasharray='5 4'/>")
    elems.append(f"<text x='{x(injection_s)+6:.1f}' y='{top+14}' font-size='12' fill='#991b1b'>prefill injected</text>")
    elems.append(f"<text x='{left}' y='18' font-size='14' font-family='sans-serif'>Background decode inter-stream-event gaps aligned to injection</text>")
    elems.append(f"<text x='{left + plot_w/2 - 40}' y='{h-12}' font-size='12' font-family='sans-serif'>seconds since experiment start</text>")
    elems.append(f"<text x='10' y='{top + plot_h/2}' font-size='12' font-family='sans-serif' transform='rotate(-90 10 {top + plot_h/2})'>gap seconds</text>")
    for r in pts:
        elems.append(f"<circle cx='{x(r['arrival_s']):.1f}' cy='{y(r['gap_s']):.1f}' r='2.2' fill='#2563eb' opacity='0.65'/>")
    elems.append("</svg>\n")
    path.write_text("\n".join(elems), encoding="utf-8")

def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3 staged prefill/decode interference benchmark.")
    parser.add_argument("--base-url", default=os.environ.get("SGLANG_BASE_URL", "http://localhost:30000"))
    parser.add_argument("--model", default=os.environ.get("SERVED_MODEL_NAME", "qwen3-0.6b"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/phase3_interference_baseline"))
    parser.add_argument("--background-decode-requests", type=int, default=4)
    parser.add_argument("--background-max-output-tokens", type=int, default=2000)
    parser.add_argument("--injection-delay-s", type=float, default=1.5)
    parser.add_argument("--injected-prefill-terms", type=int, default=8192)
    parser.add_argument("--injected-max-output-tokens", type=int, default=64)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--include-usage", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--server-config-label", default=os.environ.get("SERVER_CONFIG_LABEL", "unknown"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    experiment_t0 = time.perf_counter()
    background_specs = [RequestSpec("background_decode", i, long_decode_prompt(i), args.background_max_output_tokens) for i in range(args.background_decode_requests)]
    injected_spec = RequestSpec("injected_prefill", 0, long_prefill_prompt(args.injected_prefill_terms), args.injected_max_output_tokens)
    request_rows: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.background_decode_requests + 1) as executor:
        futures = [executor.submit(run_streaming_request, args.base_url, args.model, spec, args.timeout_s, args.include_usage, args.disable_thinking, experiment_t0) for spec in background_specs]
        time.sleep(args.injection_delay_s)
        injection_s = time.perf_counter() - experiment_t0
        futures.append(executor.submit(run_streaming_request, args.base_url, args.model, injected_spec, args.timeout_s, args.include_usage, args.disable_thinking, experiment_t0))
        for future in concurrent.futures.as_completed(futures):
            request_rows.append(future.result())
    chunk_rows = [chunk for row in request_rows for chunk in row.pop("chunks")]
    gap_summaries = summarize_gaps(chunk_rows, request_rows, injection_s)
    injected = next((r for r in request_rows if r["role"] == "injected_prefill"), None)
    background = [r for r in request_rows if r["role"] == "background_decode"]
    total_completion = sum(r.get("completion_tokens") or 0 for r in request_rows if r["status"] == "ok")
    wall_s = max(r["end_s"] for r in request_rows) if request_rows else None
    summary = {
        "created_utc": now_iso(),
        "server_config_label": args.server_config_label,
        "model": args.model,
        "background_decode_requests": args.background_decode_requests,
        "background_max_output_tokens": args.background_max_output_tokens,
        "injection_delay_target_s": args.injection_delay_s,
        "injection_s": injection_s,
        "injected_prefill_terms_target": args.injected_prefill_terms,
        "injected_max_output_tokens": args.injected_max_output_tokens,
        "ok_requests": sum(r["status"] == "ok" for r in request_rows),
        "error_requests": sum(r["status"] != "ok" for r in request_rows),
        "wall_time_s": wall_s,
        "aggregate_output_tokens_per_s": total_completion / wall_s if wall_s else None,
        "background_decode_completion_tokens": [r.get("completion_tokens") for r in sorted(background, key=lambda x: x["request_index"])],
        "background_decode_finish_reasons": [r.get("finish_reason") for r in sorted(background, key=lambda x: x["request_index"])],
        "injected_prompt_tokens": None if injected is None else injected.get("prompt_tokens"),
        "injected_completion_tokens": None if injected is None else injected.get("completion_tokens"),
        "injected_finish_reason": None if injected is None else injected.get("finish_reason"),
        "injected_ttft_s": None if injected is None else injected.get("ttft_s"),
        "injected_latency_s": None if injected is None else injected.get("latency_s"),
        "background_itl_by_phase": gap_summaries,
        "metric_notes": {
            "itl": "Measured as inter-stream-event gap for background decode chunks. SGLang stream_interval=1 makes this close to per-token timing, but the OpenAI stream event is still the unit directly measured here.",
            "during_prefill_proxy": "Gaps whose arrival time is between injected request submission and that request's first streamed content. This approximates the injected prefill/TTFT window from client-observed timing.",
        },
    }
    write_jsonl(args.out_dir / "requests.jsonl", request_rows)
    write_jsonl(args.out_dir / "stream_events.jsonl", chunk_rows)
    write_csv(args.out_dir / "requests.csv", request_rows)
    write_csv(args.out_dir / "stream_events.csv", chunk_rows)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(args.out_dir / "phase_summary.csv", gap_summaries)
    svg_plot(args.out_dir / "itl_aligned_to_injection.svg", chunk_rows, injection_s, None if injected is None else injected.get("first_chunk_s"))
    print(json.dumps(summary, indent=2), flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
