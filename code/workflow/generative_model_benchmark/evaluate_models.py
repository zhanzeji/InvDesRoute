#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from pymatgen.core import Composition


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / "literature_benchmark"))
from evaluate_route_similarity import route_similarity  # noqa: E402

from common import normalize, split_formulas


BENCHMARK = ROOT / "data" / "benchmark_24x5.csv"
OUT = ROOT / "results"
EXISTING = PROJECT / "multi_route_benchmark" / "external_model_interface"
MODERATE = {"exact_route_recovered", "strong_route_family_match", "moderate_route_family_match"}
STRONG = {"exact_route_recovered", "strong_route_family_match"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_prediction(row: dict[str, str]) -> tuple[str, ...] | None:
    try:
        return split_formulas(row.get("precursors", ""))
    except Exception:
        return None


def candidate(target: str, precursors: tuple[str, ...], rank: int) -> dict[str, object]:
    return {
        "reactants": " | ".join(f"1*{formula}" for formula in precursors),
        "products": f"1*{normalize(target)}",
        "reaction": "",
        "synth_rank": rank,
        "route_template_reason": "external_precursor_prediction",
    }


def target_element_coverage(target: str, precursors: tuple[str, ...]) -> float:
    target_elements = {str(element) for element in Composition(target).elements}
    predicted_elements = {
        str(element)
        for formula in precursors
        for element in Composition(formula).elements
    }
    return len(target_elements & predicted_elements) / len(target_elements)


def evaluate_file(path: Path, top_ks: list[int]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    references = read_rows(BENCHMARK)
    raw_predictions = read_rows(path)
    model_names = sorted({row["model"] for row in raw_predictions})
    details = []
    diagnostics = []
    for model in model_names:
        model_rows = [row for row in raw_predictions if row["model"] == model]
        by_target: dict[str, list[tuple[int, tuple[str, ...]]]] = defaultdict(list)
        invalid = 0
        for row in model_rows:
            precursor_set = normalize_prediction(row)
            if precursor_set is None or not precursor_set:
                invalid += 1
                continue
            by_target[normalize(row["target"])].append((int(float(row["rank"])), precursor_set))
        for values in by_target.values():
            values.sort(key=lambda item: item[0])
        diagnostics.append(
            {
                "model": model,
                "prediction_rows": len(model_rows),
                "valid_prediction_rows": len(model_rows) - invalid,
                "invalid_prediction_rows": invalid,
                "targets_with_predictions": len(by_target),
            }
        )
        for reference in references:
            target = normalize(reference["target"])
            reference_set = split_formulas(reference["precursors"])
            for top_k in top_ks:
                best = None
                strict_exact = False
                best_jaccard = 0.0
                best_recall = 0.0
                best_element_coverage = 0.0
                for rank, predicted_set in by_target.get(target, []):
                    if rank > top_k:
                        continue
                    overlap = len(set(reference_set) & set(predicted_set))
                    union = len(set(reference_set) | set(predicted_set))
                    jaccard = overlap / union if union else 1.0
                    recall = overlap / len(reference_set) if reference_set else 1.0
                    similarity = route_similarity(target, set(reference_set), set(), candidate(target, predicted_set, rank))
                    if best is None or float(similarity["route_similarity_score"]) > float(best["route_similarity_score"]):
                        best = similarity
                    strict_exact = strict_exact or set(reference_set) == set(predicted_set)
                    best_jaccard = max(best_jaccard, jaccard)
                    best_recall = max(best_recall, recall)
                    best_element_coverage = max(best_element_coverage, target_element_coverage(target, predicted_set))
                best = best or {"route_similarity_score": 0.0, "route_similarity_label": "weak_or_missing_route_match"}
                details.append(
                    {
                        "model": model,
                        "route_id": reference["route_id"],
                        "target": target,
                        "material_category": reference["material_category"],
                        "top_k": top_k,
                        "strict_exact_precursor_set": strict_exact,
                        "best_precursor_jaccard": best_jaccard,
                        "best_precursor_recall": best_recall,
                        "best_target_element_coverage": best_element_coverage,
                        "family_similarity": float(best["route_similarity_score"]),
                        "family_label": best["route_similarity_label"],
                        "strong_or_better": best["route_similarity_label"] in STRONG,
                        "moderate_or_better": best["route_similarity_label"] in MODERATE,
                    }
                )
    return details, diagnostics


def bootstrap(details: list[dict[str, object]], top_k: int, metric: str, iterations: int = 5000) -> list[dict[str, object]]:
    rows = [row for row in details if row["top_k"] == top_k]
    models = sorted({str(row["model"]) for row in rows})
    targets = sorted({str(row["target"]) for row in rows})
    values = {
        model: {
            target: np.mean([float(row[metric]) for row in rows if row["model"] == model and row["target"] == target])
            for target in targets
        }
        for model in models
    }
    rng = np.random.default_rng(20260724)
    output = []
    for model in models:
        observed = float(np.mean(list(values[model].values())))
        samples = []
        for _ in range(iterations):
            selected = rng.choice(targets, size=len(targets), replace=True)
            samples.append(float(np.mean([values[model][target] for target in selected])))
        output.append(
            {
                "model": model,
                "metric": metric,
                "top_k": top_k,
                "target_cluster_mean": observed,
                "ci95_low": float(np.quantile(samples, 0.025)),
                "ci95_high": float(np.quantile(samples, 0.975)),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", nargs="*", type=Path)
    args = parser.parse_args()
    if args.predictions:
        prediction_files = list(dict.fromkeys(args.predictions))
    else:
        prediction_files = sorted(
            path
            for path in (ROOT / "predictions").glob("*.csv")
            if not path.stem.endswith("_raw")
        )
        existing_files = [
            EXISTING / "csllm_greedy_top1.csv",
            EXISTING / "csllm_beam5_top5.csv",
            EXISTING / "csllm_precursor_top5.csv",
            EXISTING / "internal_network_precursor_top20.csv",
        ]
        prediction_files.extend(path for path in existing_files if path.exists())
    all_details = []
    all_diagnostics = []
    for path in prediction_files:
        details, diagnostics = evaluate_file(path, [1, 5, 10, 20])
        all_details.extend(details)
        all_diagnostics.extend(diagnostics)
    summary = []
    for model in sorted({str(row["model"]) for row in all_details}):
        for top_k in [1, 5, 10, 20]:
            rows = [row for row in all_details if row["model"] == model and row["top_k"] == top_k]
            summary.append(
                {
                    "model": model,
                    "top_k": top_k,
                    "n_literature_routes": len(rows),
                    "strict_exact": int(sum(bool(row["strict_exact_precursor_set"]) for row in rows)),
                    "strong_or_better": int(sum(bool(row["strong_or_better"]) for row in rows)),
                    "moderate_or_better": int(sum(bool(row["moderate_or_better"]) for row in rows)),
                    "mean_precursor_jaccard": float(np.mean([row["best_precursor_jaccard"] for row in rows])),
                    "mean_precursor_recall": float(np.mean([row["best_precursor_recall"] for row in rows])),
                    "mean_family_similarity": float(np.mean([row["family_similarity"] for row in rows])),
                    "mean_target_element_coverage": float(np.mean([row["best_target_element_coverage"] for row in rows])),
                }
            )
    bootstrap_rows = []
    for metric in ["strict_exact_precursor_set", "moderate_or_better", "best_precursor_recall"]:
        bootstrap_rows.extend(bootstrap(all_details, 5, metric))
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "model_comparison_detail.csv", all_details)
    write_csv(OUT / "model_comparison_summary.csv", summary)
    write_csv(OUT / "prediction_diagnostics.csv", all_diagnostics)
    write_csv(OUT / "top5_target_cluster_bootstrap.csv", bootstrap_rows)
    metadata = {
        "prediction_files": [str(path) for path in prediction_files],
        "strict_exact_definition": "The normalized predicted precursor set equals the normalized literature precursor set; both missing and extra precursors make the prediction non-exact.",
        "family_metric": "Existing InvDesRoute route-family similarity applied to precursor sets only; byproducts are excluded to keep external precursor generators comparable.",
        "bootstrap": "5000 target-cluster bootstrap resamples over the 24 retrospective benchmark targets.",
    }
    (OUT / "evaluation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\n".join(
        f"{row['model']:45s} top-{row['top_k']:2d} exact={row['strict_exact']:3d} moderate+={row['moderate_or_better']:3d} recall={row['mean_precursor_recall']:.3f}"
        for row in summary if row["top_k"] in {1, 5}
    ))


if __name__ == "__main__":
    main()
