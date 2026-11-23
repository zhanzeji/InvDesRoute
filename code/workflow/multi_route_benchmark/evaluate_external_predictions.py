#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
LITERATURE_DIR = ROOT.parent / "literature_benchmark"
sys.path.insert(0, str(LITERATURE_DIR))

from evaluate_route_similarity import normalize, route_similarity, split_formulas  # noqa: E402


MODERATE = {"exact_route_recovered", "strong_route_family_match", "moderate_route_family_match"}
STRONG = {"exact_route_recovered", "strong_route_family_match"}


def candidate_row(target: str, row: pd.Series, precursor_only: bool) -> dict[str, object]:
    reactants = split_formulas(row.get("precursors"))
    byproducts = set() if precursor_only else split_formulas(row.get("byproducts"))
    products = {normalize(target), *byproducts}
    return {
        "reactants": " | ".join(f"1*{formula}" for formula in sorted(reactants)),
        "products": " | ".join(f"1*{formula}" for formula in sorted(products)),
        "reaction": row.get("balanced_reaction", ""),
        "synth_rank": int(row["rank"]),
        "route_template_reason": "external_model_prediction",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate external top-k precursor or route predictions.")
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--task", choices=["precursor_set", "balanced_route"], default="precursor_set")
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5, 10, 20])
    parser.add_argument("--outdir", type=Path, default=ROOT / "results" / "external_models")
    args = parser.parse_args()

    literature = pd.read_csv(ROOT / "multi_route_literature_dataset.csv").fillna("")
    predictions = pd.read_csv(args.predictions).fillna("")
    required = {"model", "target", "rank", "precursors"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"missing prediction columns: {sorted(missing)}")
    details = []
    precursor_only = args.task == "precursor_set"
    for model, model_rows in predictions.groupby("model"):
        for _, reference in literature.iterrows():
            target = normalize(reference["target"])
            candidates = model_rows[model_rows["target"].map(normalize) == target].sort_values("rank")
            lit_reactants = split_formulas(reference["literature_precursors"])
            lit_byproducts = set() if precursor_only else split_formulas(reference["literature_byproducts"])
            for top_k in args.top_k:
                best = None
                for _, prediction in candidates[candidates["rank"].astype(int) <= top_k].iterrows():
                    current = route_similarity(target, lit_reactants, lit_byproducts, candidate_row(target, prediction, precursor_only))
                    if best is None or float(current["route_similarity_score"]) > float(best["route_similarity_score"]):
                        best = current
                best = best or {"route_similarity_score": 0.0, "route_similarity_label": "weak_or_missing_route_match"}
                details.append(
                    {
                        "model": model,
                        "task": args.task,
                        "top_k": top_k,
                        "route_id": reference["route_id"],
                        "target": target,
                        "similarity_score": best["route_similarity_score"],
                        "similarity_label": best["route_similarity_label"],
                        "moderate_or_better": best["route_similarity_label"] in MODERATE,
                        "strong_or_exact": best["route_similarity_label"] in STRONG,
                        "exact": best["route_similarity_label"] == "exact_route_recovered",
                    }
                )
    detail = pd.DataFrame(details)
    summary_rows = []
    for keys, group in detail.groupby(["model", "task", "top_k"]):
        model, task, top_k = keys
        per_target = group.groupby("target")["moderate_or_better"].agg(["sum", "count"])
        summary_rows.append(
            {
                "model": model,
                "task": task,
                "top_k": top_k,
                "literature_records": len(group),
                "moderate_or_better": int(group["moderate_or_better"].sum()),
                "strong_or_exact": int(group["strong_or_exact"].sum()),
                "exact": int(group["exact"].sum()),
                "targets_at_least_three_of_five": int((per_target["sum"] >= 3).sum()),
                "targets_all_five": int((per_target["sum"] == per_target["count"]).sum()),
                "mean_similarity": float(group["similarity_score"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    args.outdir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.outdir / f"{args.predictions.stem}_{args.task}_detail.csv", index=False)
    summary.to_csv(args.outdir / f"{args.predictions.stem}_{args.task}_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
