#!/usr/bin/env python3
"""Generate simple SVG comparison charts from Phase 2 summary.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_float(value: str) -> float | None:
    if value in ("", "None", None):
        return None
    return float(value)


def load_rows(summary_csv: Path):
    with summary_csv.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def color_for(profile: str) -> str:
    palette = {
        "baseline": "#2563eb",
        "prefill-heavy": "#dc2626",
        "very-prefill-heavy": "#9333ea",
        "decode-heavy": "#059669",
    }
    return palette.get(profile, "#374151")


def write_bar_chart(rows, metric: str, title: str, out_path: Path, unit: str) -> None:
    data = []
    for row in rows:
        value = parse_float(row.get(metric, ""))
        if value is None:
            continue
        label = f"{row['profile']} c{row['concurrency']}"
        data.append((label, row["profile"], value))

    width = 1200
    row_h = 28
    left = 260
    top = 70
    height = max(180, top + len(data) * row_h + 60)
    max_value = max([v for _, _, v in data], default=1.0)
    scale = (width - left - 80) / max_value if max_value else 1.0

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="36" font-family="Arial" font-size="22" font-weight="700" fill="#111827">{title}</text>',
        f'<text x="24" y="58" font-family="Arial" font-size="13" fill="#4b5563">Unit: {unit}. Compare shapes, not just winners.</text>',
    ]

    for i, (label, profile, value) in enumerate(data):
        y = top + i * row_h
        bar_w = max(1, value * scale)
        lines.append(f'<text x="24" y="{y + 17}" font-family="Arial" font-size="12" fill="#111827">{label}</text>')
        lines.append(f'<rect x="{left}" y="{y}" width="{bar_w:.2f}" height="18" rx="2" fill="{color_for(profile)}"/>')
        lines.append(f'<text x="{left + bar_w + 8:.2f}" y="{y + 14}" font-family="Arial" font-size="12" fill="#111827">{value:.4g}</text>')

    lines.append("</svg>")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=Path("outputs/phase2/summary.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/phase2/charts"))
    args = parser.parse_args()

    rows = load_rows(args.summary)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    charts = [
        ("ttft_p50_s", "TTFT p50 by Workload and Concurrency", "seconds"),
        ("ttft_p95_s", "TTFT p95 by Workload and Concurrency", "seconds"),
        ("request_tpot_p50_s", "Request-Averaged TPOT p50", "seconds/token or seconds/chunk fallback"),
        ("latency_p95_s", "End-to-End Latency p95", "seconds"),
        ("aggregate_output_tokens_per_s", "Aggregate Output Throughput", "output tokens/second"),
        ("requests_per_s", "Request Throughput", "requests/second"),
    ]
    for metric, title, unit in charts:
        write_bar_chart(rows, metric, title, args.out_dir / f"{metric}.svg", unit)
    print(f"Wrote charts to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
