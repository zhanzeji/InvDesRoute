#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from pymatgen.core import Composition


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
CORPUS = PROJECT / "literature_benchmark" / "ceder_text_mined_routes.csv"
BENCHMARK = PROJECT / "multi_route_benchmark" / "multi_route_literature_dataset.csv"
OUT = ROOT / "data"


def normalize(value: str) -> str:
    return Composition(str(value).strip().split("·", 1)[0]).reduced_formula


def formulas(value: str) -> tuple[str, ...]:
    result = []
    for token in str(value or "").split(";"):
        token = token.strip()
        if token:
            result.append(normalize(token))
    return tuple(sorted(set(result)))


def valid_route(target: str, precursors: tuple[str, ...]) -> bool:
    if not precursors or len(precursors) > 8:
        return False
    target_elements = {str(element) for element in Composition(target).elements}
    precursor_elements = {
        str(element)
        for precursor in precursors
        for element in Composition(precursor).elements
    }
    return target_elements.issubset(precursor_elements)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with BENCHMARK.open(encoding="utf-8", newline="") as handle:
        benchmark_rows = list(csv.DictReader(handle))
    held_out_targets = sorted({normalize(row["target"]) for row in benchmark_rows})

    cleaned = []
    rejected = Counter()
    with CORPUS.open(encoding="utf-8", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle), start=1):
            try:
                target = normalize(row["target"])
                precursors = formulas(row["literature_precursors"])
            except Exception:
                rejected["formula_parse"] += 1
                continue
            if target in held_out_targets:
                rejected["held_out_target"] += 1
                continue
            if not valid_route(target, precursors):
                rejected["invalid_or_uncovered"] += 1
                continue
            cleaned.append(
                {
                    "record_id": f"train_{index:06d}",
                    "target": target,
                    "precursors": ";".join(precursors),
                    "source_note": row.get("source_note", ""),
                }
            )

    unique = {}
    for row in cleaned:
        unique[(row["target"], row["precursors"])] = row
    cleaned = list(unique.values())
    cleaned.sort(key=lambda row: (row["target"], row["precursors"]))

    benchmark = []
    for row in benchmark_rows:
        target = normalize(row["target"])
        precursor_set = formulas(row["literature_precursors"])
        benchmark.append(
            {
                "route_id": row["route_id"],
                "target": target,
                "precursors": ";".join(precursor_set),
                "material_category": row["material_category"],
                "doi": row["doi"],
            }
        )

    for name, rows in (("train_target_disjoint.csv", cleaned), ("benchmark_24x5.csv", benchmark)):
        with (OUT / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    metadata = {
        "source_corpus": str(CORPUS),
        "benchmark_source": str(BENCHMARK),
        "held_out_targets": held_out_targets,
        "held_out_target_count": len(held_out_targets),
        "benchmark_route_count": len(benchmark),
        "training_unique_target_route_pairs": len(cleaned),
        "training_unique_targets": len({row["target"] for row in cleaned}),
        "rejected": dict(rejected),
        "leakage_rule": "Every record whose normalized target equals one of the 24 benchmark targets is excluded before model fitting.",
    }
    (OUT / "split_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
