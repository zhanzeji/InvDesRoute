#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
LITERATURE = PROJECT / "literature_benchmark"
sys.path.insert(0, str(LITERATURE))

from evaluate_route_similarity import parse_side, route_similarity, split_formulas  # noqa: E402


def best_score(reference: pd.Series, candidates: list[dict[str, object]]) -> tuple[float, bool]:
    target = reference["target"]
    precursors = split_formulas(reference["literature_precursors"])
    byproducts = split_formulas(reference["literature_byproducts"])
    best = 0.0
    strict = False
    for route in candidates:
        score = float(route_similarity(target, precursors, byproducts, route)["route_similarity_score"])
        best = max(best, score)
        candidate_precursors = parse_side(route.get("reactants"))
        candidate_byproducts = parse_side(route.get("products")) - {target}
        strict = strict or (candidate_precursors == precursors and candidate_byproducts == byproducts)
    return best, strict


def main() -> None:
    parser = argparse.ArgumentParser(description="Target-swapped controls for the 2022 temporal route-family metric.")
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--displayed-routes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()

    references = pd.read_csv(args.references).fillna("")
    displayed = pd.read_csv(args.displayed_routes).fillna("")
    displayed = displayed.loc[displayed["configuration"].eq("selected_with_equiformer_energy")].copy()
    by_target = {
        target: group.sort_values("display_rank").to_dict("records")
        for target, group in displayed.groupby("target")
    }
    targets = sorted(by_target)

    actual_rows = []
    for _, reference in references.iterrows():
        score, strict = best_score(reference, by_target[reference["target"]])
        actual_rows.append({"route_id": reference["route_id"], "target": reference["target"], "score": score, "strict_exact": strict})
    actual = pd.DataFrame(actual_rows)

    rng = np.random.default_rng(20220912)
    decoy_rows = []
    for iteration in range(args.iterations):
        for _, reference in references.iterrows():
            alternatives = [target for target in targets if target != reference["target"]]
            sampled_targets = rng.choice(alternatives, size=5, replace=False)
            candidates = [by_target[target][0] for target in sampled_targets]
            score, strict = best_score(reference, candidates)
            decoy_rows.append(
                {
                    "iteration": iteration,
                    "route_id": reference["route_id"],
                    "target": reference["target"],
                    "score": score,
                    "strict_exact": strict,
                }
            )
    decoys = pd.DataFrame(decoy_rows)

    rows = []
    thresholds = sorted({*[round(float(value), 2) for value in np.arange(0.55, 0.851, 0.05)], 0.62, 0.78})
    for threshold in thresholds:
        iteration_rates = decoys.groupby("iteration")["score"].apply(lambda values: float((values >= threshold).mean()))
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "actual_recovered": int(actual["score"].ge(threshold).sum()),
                "actual_fraction": float(actual["score"].ge(threshold).mean()),
                "decoy_false_positive_rate": float(decoys["score"].ge(threshold).mean()),
                "decoy_iteration_ci95_low": float(iteration_rates.quantile(0.025)),
                "decoy_iteration_ci95_high": float(iteration_rates.quantile(0.975)),
            }
        )
    summary = {
        "future_records": len(actual),
        "iterations": args.iterations,
        "strict_exact_actual": int(actual["strict_exact"].sum()),
        "strict_exact_decoy_rate": float(decoys["strict_exact"].mean()),
        "thresholds": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    actual.to_csv(args.output_dir / "future_similarity_actual.csv", index=False)
    decoys.to_csv(args.output_dir / "future_similarity_target_swapped_decoys.csv", index=False)
    pd.DataFrame(rows).to_csv(args.output_dir / "future_similarity_threshold_controls.csv", index=False)
    (args.output_dir / "future_similarity_control_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
