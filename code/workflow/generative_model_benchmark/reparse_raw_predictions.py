#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from run_official_llm_baselines import parse_candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    predictions = []
    with args.raw_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            candidates, parse_status = parse_candidates(row["raw_generation"], row["target"], args.limit)
            for rank, formulas in enumerate(candidates, start=1):
                predictions.append(
                    {
                        "model": row["model"],
                        "target": row["target"],
                        "rank": rank,
                        "precursors": ";".join(formulas),
                        "byproducts": "",
                        "balanced_reaction": "",
                        "notes": f"parse={parse_status}; deterministic_zero_shot; reparsed_from_frozen_raw",
                    }
                )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["model", "target", "rank", "precursors", "byproducts", "balanced_reaction", "notes"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)


if __name__ == "__main__":
    main()
