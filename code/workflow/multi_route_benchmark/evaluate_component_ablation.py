#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
LITERATURE_DIR = ROOT.parent / "literature_benchmark"
sys.path.insert(0, str(LITERATURE_DIR))
sys.path.insert(0, str(ROOT))

from evaluate_multi_route_benchmark import MODERATE_LABELS, pet_energy, route_family_key  # noqa: E402
from evaluate_route_similarity import best_similarity_in_topk, normalize, split_formulas  # noqa: E402


BASE_WEIGHTS = {
    "energy_score": 0.18,
    "precursor_prior_score": 0.25,
    "route_template_score": 0.25,
    "practical_filter_score": 0.12,
    "route_complexity_score": 0.08,
    "phase_availability_score": 0.07,
    "condition_simplicity_score": 0.05,
}
TOP_K_VALUES = [1, 5, 10, 20]


def score(row: dict[str, object], removed: str | None = None, condition_heavy: bool = False) -> float:
    weights = dict(BASE_WEIGHTS)
    if condition_heavy:
        weights = {
            "energy_score": 0.10,
            "precursor_prior_score": 0.25,
            "route_template_score": 0.25,
            "practical_filter_score": 0.10,
            "route_complexity_score": 0.07,
            "phase_availability_score": 0.08,
            "condition_simplicity_score": 0.15,
        }
    if removed:
        weights[removed] = 0.0
    value = sum(float(row.get(component) or 0.0) * weight for component, weight in weights.items())
    template = float(row.get("route_template_score") or 0.0)
    if removed != "route_template_score":
        if template >= 0.95:
            value += 0.06
        elif template >= 0.90:
            value += 0.03
    return min(1.0, value)


def ranked_routes(routes: list[dict[str, object]], mode: str) -> list[dict[str, object]]:
    removed = mode.removeprefix("without_") if mode.startswith("without_") else None
    condition_heavy = mode == "condition_heavy"
    ranked = sorted(
        routes,
        key=lambda row: (-score(row, removed=removed, condition_heavy=condition_heavy), pet_energy(row), str(row.get("reaction", ""))),
    )
    for rank, route in enumerate(ranked, start=1):
        route["synth_rank"] = rank
    return ranked


def diverse_rank(routes: list[dict[str, object]], penalty: float) -> list[dict[str, object]]:
    base = ranked_routes(routes, "current")
    occurrence: Counter[tuple[str, str]] = Counter()
    adjusted = []
    for route in base:
        key = route_family_key(route)
        duplicate_index = occurrence[key]
        occurrence[key] += 1
        adjusted.append((score(route) - penalty * duplicate_index, pet_energy(route), str(route.get("reaction", "")), route))
    adjusted.sort(key=lambda item: (-item[0], item[1], item[2]))
    ranked = [item[-1] for item in adjusted]
    for rank, route in enumerate(ranked, start=1):
        route["synth_rank"] = rank
    return ranked


def evaluate(dataset: pd.DataFrame, ranked_by_target: dict[str, list[dict[str, object]]], label: str, parameter: object) -> list[dict[str, object]]:
    details = []
    for _, literature in dataset.iterrows():
        target = normalize(literature["target"])
        for top_k in TOP_K_VALUES:
            best = best_similarity_in_topk(
                target,
                split_formulas(literature["literature_precursors"]),
                split_formulas(literature["literature_byproducts"]),
                ranked_by_target[target],
                top_k,
            )
            details.append(
                {
                    "experiment": label,
                    "parameter": parameter,
                    "top_k": top_k,
                    "route_id": literature["route_id"],
                    "target": target,
                    "moderate_or_better": best.get("route_similarity_label") in MODERATE_LABELS,
                    "similarity_score": float(best.get("route_similarity_score", 0.0)),
                }
            )
    return details


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate InvDesRoute score-component ablations.")
    parser.add_argument("--case-root", type=Path, default=ROOT / "cases_blind")
    parser.add_argument("--outdir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    dataset = pd.read_csv(ROOT / "multi_route_literature_dataset.csv").fillna("")
    routes = {
        target: pd.read_csv(args.case_root / target / "synthesizability_ranked_routes.csv").fillna("").to_dict("records")
        for target in sorted(dataset["target"].map(normalize).unique())
    }
    modes = ["current", *[f"without_{component}" for component in BASE_WEIGHTS], "condition_heavy"]
    detail_rows = []
    for mode in modes:
        ranked = {target: ranked_routes([dict(row) for row in target_routes], mode) for target, target_routes in routes.items()}
        detail_rows.extend(evaluate(dataset, ranked, "component_ablation", mode))
    for penalty in [0.0, 0.005, 0.01, 0.02, 0.04, 0.08]:
        ranked = {target: diverse_rank([dict(row) for row in target_routes], penalty) for target, target_routes in routes.items()}
        detail_rows.extend(evaluate(dataset, ranked, "diversity_penalty", penalty))
    detail = pd.DataFrame(detail_rows)
    summary = (
        detail.groupby(["experiment", "parameter", "top_k"], sort=False)
        .agg(
            records=("route_id", "size"),
            moderate_or_better=("moderate_or_better", "sum"),
            mean_similarity=("similarity_score", "mean"),
        )
        .reset_index()
    )
    outdir = args.outdir
    detail.to_csv(outdir / "component_ablation_detail.csv", index=False)
    summary.to_csv(outdir / "component_ablation_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
