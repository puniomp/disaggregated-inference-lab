#!/usr/bin/env python3
"""Streaming benchmark client for the Phase 2 SGLang baseline.

This intentionally uses only the Python standard library so the benchmark
client is independent of SGLang, vLLM, and third-party HTTP packages.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_WORKLOADS = Path("configs/workloads.json")


@dataclass(frozen=True)
class RequestSpec:
    profile: str
    request_index: int
    approx_input_tokens: int
    max_output_tokens: int
    prompt: str


def long_decode_instruction(request_index: int) -> str:
    return (
        f"Profile=decode-heavy. Request={request_index}. "
        "This is a sustained decode benchmark. Write exactly 180 numbered "
        "entries about LLM inference serving. Each entry must be a standalone "
        "sentence of 12 to 18 words. Start at 001 and continue sequentially. "
        "Do not write an introduction, summary, conclusion, apology, or any "
        "meta commentary. Do not stop early. Continue until entry 180.\n\n"
        "Topic anchors: prefill, decode, KV cache, batching, scheduler, "
        "latency, throughput, GPU memory, queueing, request concurrency."
    )


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


def make_prompt(
    profile: str,
    approx_tokens: int,
    request_index: int,
    prompt_style: str | None = None,
) -> str:
    if prompt_style == "long_numbered_generation":
        return long_decode_instruction(request_index)

    # This is an approximation. The benchmark records prompt_tokens from the
    # server when available; use that value for analysis rather than assuming
    # the generator hit the target exactly.
    terms = [
        "prefill",
        "decode",
        "kv-cache",
        "scheduler",
        "latency",
        "throughput",
        "batching",
        "request",
    ]
    body = " ".join(terms[i % len(terms)] for i in range(max(1, approx_tokens)))
    return (
        f"Profile={profile}. Request={request_index}. "
        "Use the following synthetic context for an inference benchmark. "
        "At the end, summarize the main point in one concise paragraph.\n\n"
        f"{body}"
    )


def load_workloads(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "profiles" not in data or not isinstance(data["profiles"], dict):
        raise ValueError(f"{path} must contain a top-level 'profiles' object")
    return data["profiles"]


def build_payload(model: str, spec: RequestSpec, include_usage: bool, disable_thinking: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": spec.prompt,
            }
        ],
        "max_tokens": spec.max_output_tokens,
        "temperature": 0,
        "stream": True,
    }
    if include_usage:
        payload["stream_options"] = {"include_usage": True}
    if disable_thinking:
        # SGLang/Qwen3 may honor this through its chat-template kwargs. If the
        # active server ignores it, the raw output will still be measured.
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    return payload


def iter_sse_lines(response: Any):
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line or line.startswith(":"):
            continue
        if line.startswith("data:"):
            yield line[len("data:") :].strip()


def run_one_request(
    base_url: str,
    model: str,
    spec: RequestSpec,
    timeout_s: float,
    include_usage: bool,
    disable_thinking: bool,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = build_payload(model, spec, include_usage, disable_thinking)
    encoded = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=encoded,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()
    first_chunk_time: float | None = None
    last_chunk_time: float | None = None
    chunk_times: list[float] = []
    emitted_chunks = 0
    output_parts: list[str] = []
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    error: str | None = None
    status = "ok"

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
                piece = (
                    delta.get("content")
                    or delta.get("reasoning_content")
                    or delta.get("reasoning")
                    or ""
                )
                if piece:
                    output_parts.append(piece)
                    emitted_chunks += 1
                    chunk_times.append(event_time)
                    if first_chunk_time is None:
                        first_chunk_time = event_time
                    last_chunk_time = event_time
    except urllib.error.HTTPError as exc:
        status = "error"
        error = f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:500]}"
    except Exception as exc:  # noqa: BLE001 - benchmark should record failures.
        status = "error"
        error = repr(exc)

    end = time.perf_counter()
    latency_s = end - start
    ttft_s = (first_chunk_time - start) if first_chunk_time is not None else None
    inter_chunk_latencies = [
        chunk_times[i] - chunk_times[i - 1] for i in range(1, len(chunk_times))
    ]

    completion_tokens = None
    prompt_tokens = None
    total_tokens = None
    if usage:
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")

    request_tpot_s = None
    if first_chunk_time is not None and last_chunk_time is not None:
        denom = None
        if isinstance(completion_tokens, int) and completion_tokens > 1:
            denom = completion_tokens - 1
        elif emitted_chunks > 1:
            denom = emitted_chunks - 1
        if denom:
            request_tpot_s = (last_chunk_time - first_chunk_time) / denom

    return {
        "schema_version": 1,
        "timestamp_utc": now_iso(),
        "status": status,
        "error": error,
        "profile": spec.profile,
        "request_index": spec.request_index,
        "approx_input_tokens_target": spec.approx_input_tokens,
        "max_output_tokens": spec.max_output_tokens,
        "model": model,
        "latency_s": latency_s,
        "ttft_s": ttft_s,
        "request_tpot_s": request_tpot_s,
        "inter_chunk_latency_avg_s": mean(inter_chunk_latencies),
        "inter_chunk_latency_p50_s": percentile(inter_chunk_latencies, 50),
        "inter_chunk_latency_p95_s": percentile(inter_chunk_latencies, 95),
        "emitted_stream_chunks": emitted_chunks,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "finish_reason": finish_reason,
        "output_chars": len("".join(output_parts)),
        "output_preview": "".join(output_parts)[:200],
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_group(rows: list[dict[str, Any]], profile: str, concurrency: int, started: float, ended: float) -> dict[str, Any]:
    ok_rows = [r for r in rows if r["status"] == "ok"]
    latencies = [r["latency_s"] for r in ok_rows if r["latency_s"] is not None]
    ttfts = [r["ttft_s"] for r in ok_rows if r["ttft_s"] is not None]
    tpots = [r["request_tpot_s"] for r in ok_rows if r["request_tpot_s"] is not None]
    completion_tokens = [
        r["completion_tokens"] for r in ok_rows if isinstance(r.get("completion_tokens"), int)
    ]
    elapsed = ended - started
    output_tokens = sum(completion_tokens)
    return {
        "profile": profile,
        "concurrency": concurrency,
        "requests": len(rows),
        "ok_requests": len(ok_rows),
        "error_requests": len(rows) - len(ok_rows),
        "wall_time_s": elapsed,
        "requests_per_s": len(ok_rows) / elapsed if elapsed > 0 else None,
        "aggregate_output_tokens_per_s": output_tokens / elapsed if elapsed > 0 else None,
        "latency_p50_s": percentile(latencies, 50),
        "latency_p95_s": percentile(latencies, 95),
        "ttft_p50_s": percentile(ttfts, 50),
        "ttft_p95_s": percentile(ttfts, 95),
        "request_tpot_p50_s": percentile(tpots, 50),
        "request_tpot_p95_s": percentile(tpots, 95),
        "completion_tokens_total": output_tokens,
        "completion_tokens_avg": mean([float(x) for x in completion_tokens]),
    }


def run_group(
    args: argparse.Namespace,
    profile: str,
    profile_cfg: dict[str, Any],
    concurrency: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    total = max(args.requests_per_concurrency, concurrency)
    specs = [
        RequestSpec(
            profile=profile,
            request_index=i,
            approx_input_tokens=int(profile_cfg["approx_input_tokens"]),
            max_output_tokens=int(profile_cfg["max_output_tokens"]),
            prompt=make_prompt(
                profile,
                int(profile_cfg["approx_input_tokens"]),
                i,
                profile_cfg.get("prompt_style"),
            ),
        )
        for i in range(total)
    ]

    print(f"Running profile={profile} concurrency={concurrency} requests={total}", flush=True)
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                run_one_request,
                args.base_url,
                args.model,
                spec,
                args.timeout_s,
                args.include_usage,
                args.disable_thinking,
            )
            for spec in specs
        ]
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            row["concurrency"] = concurrency
            rows.append(row)
            if row["status"] != "ok":
                print(f"  error request={row['request_index']}: {row['error']}", flush=True)
    ended = time.perf_counter()
    summary = summarize_group(rows, profile, concurrency, started, ended)
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark SGLang's OpenAI-compatible streaming endpoint.")
    parser.add_argument("--base-url", default=os.environ.get("SGLANG_BASE_URL", "http://localhost:30000"))
    parser.add_argument("--model", default=os.environ.get("SERVED_MODEL_NAME", "qwen3-0.6b"))
    parser.add_argument("--workloads", type=Path, default=DEFAULT_WORKLOADS)
    parser.add_argument("--profiles", default="baseline,prefill-heavy,very-prefill-heavy,decode-heavy")
    parser.add_argument("--concurrency", default="1,8,16,32")
    parser.add_argument("--requests-per-concurrency", type=int, default=32)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/phase2"))
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--include-usage", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-thinking", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    workloads = load_workloads(args.workloads)
    selected_profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    concurrencies = [int(c.strip()) for c in args.concurrency.split(",") if c.strip()]
    for profile in selected_profiles:
        if profile not in workloads:
            raise SystemExit(f"Unknown profile {profile!r}. Available: {', '.join(workloads)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "created_utc": now_iso(),
        "base_url": args.base_url,
        "model": args.model,
        "workloads": str(args.workloads),
        "profiles": selected_profiles,
        "concurrency": concurrencies,
        "requests_per_concurrency": args.requests_per_concurrency,
        "include_usage": args.include_usage,
        "disable_thinking": args.disable_thinking,
        "metric_notes": {
            "ttft_s": "Time from immediately before HTTP request submission until first streamed content/reasoning chunk arrives.",
            "request_tpot_s": "Request-level average over streamed generation interval. Uses completion_tokens when usage is returned; otherwise falls back to emitted stream chunk count.",
            "inter_chunk_latency": "Measured between streamed chunks, not guaranteed to be a true per-token ITL distribution because a server chunk may contain zero, one, or multiple tokens.",
            "throughput": "Aggregate output tokens per wall-clock second for a profile/concurrency group when completion token counts are available.",
        },
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for profile in selected_profiles:
        for concurrency in concurrencies:
            rows, summary = run_group(args, profile, workloads[profile], concurrency)
            all_rows.extend(rows)
            summaries.append(summary)
            write_jsonl(args.out_dir / f"raw_{profile}_c{concurrency}.jsonl", rows)

    write_jsonl(args.out_dir / "raw_all.jsonl", all_rows)
    write_csv(args.out_dir / "raw_all.csv", all_rows)
    write_csv(args.out_dir / "summary.csv", summaries)
    (args.out_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"Wrote results to {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
