#!/usr/bin/env python3
"""Create dependency-free SVG figures for the literature route benchmark."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path


BASE = PROJECT / "literature_benchmark/results"
MATCH = BASE / "literature_route_match_top5.csv"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def svg_text(x: float, y: float, text: str, size: int = 13, anchor: str = "start", weight: str = "normal") -> str:
    safe = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" font-family="Arial" font-weight="{weight}">{safe}</text>'


def bar(x: float, y: float, w: float, h: float, color: str) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{color}" rx="3" />'


def make_svg(rows: list[dict[str, str]], path: Path) -> None:
    total = len(rows)
    covered = [r for r in rows if r["matched_rank"]]
    exact = [r for r in covered if r["match_type"] == "exact_precursor_match"]
    partial = [r for r in covered if r["match_type"] in {"exact_precursor_match", "literature_precursors_contained", "partial_precursor_match"}]
    same_element = [r for r in covered if float(r["element_jaccard"]) >= 0.999]
    exergonic = [r for r in covered if r["pet_deltaE_per_target"] and float(r["pet_deltaE_per_target"]) < 0]
    metrics = [
        ("Covered in 120-target workflow", len(covered) / total),
        ("Exact precursor match in top-5", len(exact) / len(covered)),
        ("Partial-or-better precursor match", len(partial) / len(covered)),
        ("Same precursor element pool", len(same_element) / len(covered)),
        ("Matched route exergonic by PET", len(exergonic) / len(covered)),
    ]
    category_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        category = row["category"] or "not in workflow"
        category_counts[category][row["match_type"]] += 1

    width = 980
    height = 640
    left = 315
    top = 80
    bar_h = 32
    max_w = 520
    colors = {
        "main": "#3b6fb6",
        "exact_precursor_match": "#2b8a3e",
        "literature_precursors_contained": "#66a61e",
        "partial_precursor_match": "#f2a93b",
        "same_element_pool": "#4c78a8",
        "partial_element_match": "#9ecae1",
        "weak_or_no_match": "#d95f02",
        "not_in_recommendation_table": "#999999",
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(30, 35, "Literature-route agreement benchmark", 22, weight="bold"),
        svg_text(30, 58, "Top-5 PET practical routes compared with representative literature precursor sets", 13),
    ]
    for idx, (label, value) in enumerate(metrics):
        y = top + idx * 58
        parts.append(svg_text(30, y + 22, label, 14))
        parts.append(bar(left, y, max_w, bar_h, "#e8eef7"))
        parts.append(bar(left, y, max_w * value, bar_h, colors["main"]))
        parts.append(svg_text(left + max_w + 15, y + 22, f"{value:.3f}", 14, weight="bold"))
    parts.append(svg_text(30, 400, "Match-type distribution by category", 17, weight="bold"))
    y0 = 430
    x0 = 315
    stack_w = 520
    for idx, (category, counter) in enumerate(sorted(category_counts.items())):
        y = y0 + idx * 33
        count_total = sum(counter.values())
        parts.append(svg_text(30, y + 18, category, 12))
        x = x0
        for match_type, count in counter.most_common():
            w = stack_w * count / count_total
            parts.append(bar(x, y, w, 20, colors.get(match_type, "#777777")))
            x += w
        parts.append(svg_text(x0 + stack_w + 15, y + 16, str(count_total), 12))
    legend_items = [
        ("exact_precursor_match", "Exact precursor"),
        ("same_element_pool", "Same element pool"),
        ("partial_element_match", "Partial element"),
        ("weak_or_no_match", "Weak/no match"),
        ("not_in_recommendation_table", "Not covered"),
    ]
    lx, ly = 30, 595
    for idx, (key, label) in enumerate(legend_items):
        x = lx + idx * 180
        parts.append(bar(x, ly, 16, 16, colors[key]))
        parts.append(svg_text(x + 23, ly + 13, label, 11))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    rows = read_rows(MATCH)
    out = BASE / "literature_route_benchmark_summary.svg"
    make_svg(rows, out)
    print(out)


if __name__ == "__main__":
    main()
