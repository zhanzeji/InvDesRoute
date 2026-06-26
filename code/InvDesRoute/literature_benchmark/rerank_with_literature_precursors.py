#!/usr/bin/env python3
"""Rerank PET routes with a literature/common-precursor prior.

This is a lightweight bridge before full precursor-pool reruns. It keeps PET
energies from the existing route table but promotes routes whose reactants match
known literature/common precursor sets.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from compare_literature_routes import element_set, jaccard, split_formulas


DEFAULT_ROUTES = PROJECT / "results/mp_pet_benchmark_120/recommended_routes_top5_all.csv"
DEFAULT_LIT = PROJECT / "literature_benchmark/literature_routes_seed.csv"
DEFAULT_OUT = PROJECT / "literature_benchmark/results/literature_prior_reranked_routes.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
    parser.add_argument("--literature", type=Path, default=DEFAULT_LIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--energy-weight", type=float, default=0.65)
    parser.add_argument("--precursor-weight", type=float, default=0.25)
    parser.add_argument("--element-weight", type=float, default=0.10)
    args = parser.parse_args()

    literature_by_target = {}
    for row in read_csv(args.literature):
        literature_by_target[row["target"]] = split_formulas(row.get("literature_precursors"))

    route_rows = read_csv(args.routes)
    by_target: dict[str, list[dict[str, str]]] = {}
    for row in route_rows:
        by_target.setdefault(row["target"], []).append(row)

    output = []
    for target, rows in by_target.items():
        if target not in literature_by_target:
            continue
        lit_precursors = literature_by_target[target]
        lit_elements = element_set(lit_precursors)
        energies = []
        for row in rows:
            try:
                energies.append(float(row["pet_deltaE_per_target"]))
            except Exception:
                pass
        if not energies:
            continue
        min_energy = min(energies)
        max_energy = max(energies)
        span = max(max_energy - min_energy, 1e-9)
        scored = []
        for row in rows:
            reactants = split_formulas(row.get("reactant_formulas"))
            try:
                energy = float(row["pet_deltaE_per_target"])
            except Exception:
                continue
            energy_score = (max_energy - energy) / span
            precursor_score = jaccard(lit_precursors, reactants)
            element_score = jaccard(lit_elements, element_set(reactants))
            combined = (
                args.energy_weight * energy_score
                + args.precursor_weight * precursor_score
                + args.element_weight * element_score
            )
            scored.append((combined, energy, precursor_score, element_score, row))
        scored.sort(key=lambda item: (-item[0], item[1]))
        for new_rank, (combined, energy, precursor_score, element_score, row) in enumerate(scored, 1):
            output.append(
                {
                    "target": target,
                    "new_rank": new_rank,
                    "old_rank": row["rank"],
                    "reaction": row["reaction"],
                    "pet_deltaE_per_target": energy,
                    "literature_precursors": ";".join(sorted(lit_precursors)),
                    "recommended_reactants": row.get("reactant_formulas", ""),
                    "precursor_jaccard": precursor_score,
                    "element_jaccard": element_score,
                    "combined_score": combined,
                    "category": row.get("category", ""),
                }
            )
    fieldnames = [
        "target",
        "new_rank",
        "old_rank",
        "reaction",
        "pet_deltaE_per_target",
        "literature_precursors",
        "recommended_reactants",
        "precursor_jaccard",
        "element_jaccard",
        "combined_score",
        "category",
    ]
    write_csv(args.output, output, fieldnames)
    top1 = [row for row in output if row["new_rank"] == 1]
    exact = [row for row in top1 if float(row["precursor_jaccard"]) >= 0.999]
    same_element = [row for row in top1 if float(row["element_jaccard"]) >= 0.999]
    print(f"targets={len(top1)}")
    print(f"reranked_top1_exact_precursor_rate={len(exact)/len(top1) if top1 else 0:.3f}")
    print(f"reranked_top1_same_element_rate={len(same_element)/len(top1) if top1 else 0:.3f}")
    print(args.output)


if __name__ == "__main__":
    main()
