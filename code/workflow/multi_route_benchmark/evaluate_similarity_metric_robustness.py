#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
LITERATURE = ROOT.parent / "literature_benchmark"
sys.path.insert(0, str(LITERATURE))

from evaluate_route_similarity import normalize, route_similarity, split_formulas  # noqa: E402


SIMILARITY_FILES = [
    LITERATURE / "results/ceder_seed80_route_similarity_expanded/route_similarity_eval.csv",
    LITERATURE / "results/literature_seeded_extension_cached_53_similarity/route_similarity_eval.csv",
]
ABLATION_FILES = [
    LITERATURE / "results/recommendation_ablation_expanded/recommendation_ablation_detail.csv",
    LITERATURE / "results/literature_seeded_extension_cached_53_ablation/recommendation_ablation_detail.csv",
]
THRESHOLDS = [0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.75, 0.78, 0.80]


def parse_reaction(reaction: str) -> tuple[set[str], set[str]]:
    if "->" not in str(reaction):
        return set(), set()
    left, right = str(reaction).split("->", 1)

    def formulas(side: str) -> set[str]:
        result = set()
        for token in side.split(" + "):
            formula = re.sub(r"^\s*[0-9.]+\s+", "", token).strip()
            try:
                result.add(normalize(formula))
            except Exception:
                continue
        return result

    return formulas(left), formulas(right)


def candidate_row(reaction: str, template: str) -> dict[str, object]:
    reactants, products = parse_reaction(reaction)
    return {
        "reactant_set": reactants,
        "product_set": products,
        "reactants": " | ".join(f"1*{value}" for value in sorted(reactants)),
        "products": " | ".join(f"1*{value}" for value in sorted(products)),
        "reaction": reaction,
        "route_template_reason": template,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--similarity-files", nargs="+", type=Path, default=SIMILARITY_FILES)
    parser.add_argument("--ablation-files", nargs="+", type=Path, default=ABLATION_FILES)
    parser.add_argument("--outdir", type=Path, default=ROOT / "results" / "robustness")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    positives = pd.concat([pd.read_csv(path).fillna("") for path in args.similarity_files], ignore_index=True)
    ablation = pd.concat([pd.read_csv(path).fillna("") for path in args.ablation_files], ignore_index=True)
    top5 = ablation[(ablation["mode"] == "synth_score") & (ablation["top_k"] == 5)].reset_index(drop=True)
    assert len(positives) == 145 and len(top5) == 145

    threshold_rows = []
    for threshold in THRESHOLDS:
        for window in ("best_all", "best_top5", "best_top20"):
            scores = pd.to_numeric(positives[f"{window}_similarity_score"], errors="coerce").fillna(0.0)
            threshold_rows.append(
                {
                    "threshold": threshold,
                    "window": window,
                    "moderate_or_better": int((scores >= threshold).sum()),
                    "records": len(scores),
                    "recovery": float((scores >= threshold).mean()),
                }
            )

    candidates = [
        {
            "target": normalize(row["target"]),
            "route": candidate_row(row["recommended_reaction"], row["recommended_template"]),
        }
        for _, row in top5.iterrows()
    ]
    rng = np.random.default_rng(20260722)
    iterations = 1000
    decoy_rates = {threshold: [] for threshold in THRESHOLDS}
    decoy_detail = []
    for iteration in range(iterations):
        scores = []
        for _, reference in positives.iterrows():
            target = normalize(reference["target"])
            eligible = [candidate for candidate in candidates if candidate["target"] != target]
            selected = rng.choice(len(eligible), size=5, replace=False)
            best = 0.0
            for index in selected:
                decoy = eligible[int(index)]
                decoy_route = dict(decoy["route"])
                byproducts = set(decoy_route.pop("product_set")) - {decoy["target"]}
                decoy_route.pop("reactant_set")
                decoy_route["products"] = " | ".join(f"1*{value}" for value in sorted({target, *byproducts}))
                result = route_similarity(
                    target,
                    split_formulas(reference["literature_precursors"]),
                    split_formulas(reference["literature_byproducts"]),
                    decoy_route,
                )
                best = max(best, float(result["route_similarity_score"]))
            scores.append(best)
        for threshold in THRESHOLDS:
            rate = float((np.asarray(scores) >= threshold).mean())
            decoy_rates[threshold].append(rate)
            decoy_detail.append({"iteration": iteration, "threshold": threshold, "false_positive_rate": rate})

    decoy_summary = []
    for threshold, values in decoy_rates.items():
        array = np.asarray(values)
        decoy_summary.append(
            {
                "threshold": threshold,
                "mean_false_positive_rate": float(array.mean()),
                "ci95_low": float(np.quantile(array, 0.025)),
                "ci95_high": float(np.quantile(array, 0.975)),
            }
        )

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(threshold_rows).to_csv(outdir / "similarity_threshold_sensitivity.csv", index=False)
    pd.DataFrame(decoy_detail).to_csv(outdir / "similarity_decoy_iterations.csv", index=False)
    pd.DataFrame(decoy_summary).to_csv(outdir / "similarity_decoy_summary.csv", index=False)
    print(pd.DataFrame(threshold_rows).to_string(index=False))
    print("\nTarget-mismatched top-5 decoys\n", pd.DataFrame(decoy_summary).to_string(index=False))


if __name__ == "__main__":
    main()
