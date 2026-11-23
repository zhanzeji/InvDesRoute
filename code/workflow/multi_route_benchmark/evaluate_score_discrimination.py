#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parent
LITERATURE = ROOT.parent / "literature_benchmark"
sys.path.insert(0, str(LITERATURE))

from evaluate_route_similarity import normalize, parse_side, route_similarity, split_formulas  # noqa: E402


def safe_metric(function, labels: pd.Series, scores: pd.Series) -> float:
    return float(function(labels, scores)) if labels.nunique() == 2 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate route-score discrimination.")
    parser.add_argument("--case-root", type=Path, default=ROOT / "cases_blind")
    parser.add_argument("--outdir", type=Path, default=ROOT / "results" / "robustness")
    args = parser.parse_args()
    references = pd.read_csv(ROOT / "multi_route_literature_dataset.csv").fillna("")
    by_target = {
        target: group.to_dict("records")
        for target, group in references.groupby(references["target"].map(normalize))
    }
    rows = []
    for target, target_refs in by_target.items():
        routes = pd.read_csv(args.case_root / target / "synthesizability_ranked_routes.csv").fillna("")
        ref_precursors = [split_formulas(ref["literature_precursors"]) for ref in target_refs]
        ref_byproducts = [split_formulas(ref["literature_byproducts"]) for ref in target_refs]
        for _, route in routes.iterrows():
            route_dict = route.to_dict()
            reactants = parse_side(route_dict.get("reactants"))
            products = parse_side(route_dict.get("products"))
            byproducts = products - {target}
            similarities = [
                route_similarity(target, precursors, products_ref, route_dict)
                for precursors, products_ref in zip(ref_precursors, ref_byproducts)
            ]
            best_score = max(float(value["route_similarity_score"]) for value in similarities)
            rows.append(
                {
                    "target": target,
                    "reaction": route_dict["reaction"],
                    "synth_score": float(route_dict["synth_score"]),
                    "pet_energy_predictor": -pd.to_numeric(route_dict["synth_rank_deltaE_per_target"], errors="coerce"),
                    "precursor_prior_score": float(route_dict["precursor_prior_score"]),
                    "route_template_score": float(route_dict["route_template_score"]),
                    "best_route_similarity": best_score,
                    "moderate_or_better": best_score >= 0.62,
                    "exact_precursor_set": any(reactants == value for value in ref_precursors),
                    "exact_full_route": any(
                        reactants == precursor_set and byproducts == byproduct_set
                        for precursor_set, byproduct_set in zip(ref_precursors, ref_byproducts)
                    ),
                }
            )
    detail = pd.DataFrame(rows)
    predictors = ["synth_score", "pet_energy_predictor", "precursor_prior_score", "route_template_score"]
    endpoints = ["moderate_or_better", "exact_precursor_set", "exact_full_route"]
    summary = []
    for endpoint in endpoints:
        for predictor in predictors:
            available = detail[detail[predictor].notna()].copy()
            labels = available[endpoint].astype(int)
            scores = available[predictor].astype(float)
            target_aucs = []
            for _, group in available.groupby("target"):
                if group[endpoint].nunique() == 2:
                    target_aucs.append(roc_auc_score(group[endpoint].astype(int), group[predictor].astype(float)))
            cutoff = scores.quantile(0.90)
            top = labels[scores >= cutoff]
            summary.append(
                {
                    "endpoint": endpoint,
                    "predictor": predictor,
                    "candidates": len(available),
                    "positives": int(labels.sum()),
                    "prevalence": float(labels.mean()),
                    "roc_auc": safe_metric(roc_auc_score, labels, scores),
                    "average_precision": safe_metric(average_precision_score, labels, scores),
                    "macro_target_roc_auc": float(np.mean(target_aucs)) if target_aucs else float("nan"),
                    "top_decile_precision": float(top.mean()),
                    "top_decile_enrichment": float(top.mean() / labels.mean()) if labels.mean() else float("nan"),
                }
            )

    detail["score_decile"] = pd.qcut(detail["synth_score"], 10, duplicates="drop")
    calibration = (
        detail.groupby("score_decile", observed=True)
        .agg(
            candidates=("reaction", "size"),
            mean_synth_score=("synth_score", "mean"),
            moderate_fraction=("moderate_or_better", "mean"),
            exact_precursor_fraction=("exact_precursor_set", "mean"),
            exact_route_fraction=("exact_full_route", "mean"),
        )
        .reset_index()
    )
    calibration["score_decile"] = calibration["score_decile"].astype(str)
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(outdir / "candidate_score_labels.csv", index=False)
    pd.DataFrame(summary).to_csv(outdir / "score_discrimination_summary.csv", index=False)
    calibration.to_csv(outdir / "score_decile_calibration.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))
    print("\nScore deciles\n", calibration.to_string(index=False))


if __name__ == "__main__":
    main()
