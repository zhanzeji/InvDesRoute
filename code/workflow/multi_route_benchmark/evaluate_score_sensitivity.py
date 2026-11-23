#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = PROJECT / "multi_route_benchmark"
LITERATURE_DIR = PROJECT / "literature_benchmark"
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(LITERATURE_DIR))

from evaluate_multi_route_benchmark import MODERATE_LABELS, pet_energy  # noqa: E402
from evaluate_route_similarity import best_similarity_in_topk, normalize, split_formulas  # noqa: E402


PROFILES = {
    "current": (0.18, 0.25, 0.25, 0.12, 0.08, 0.07, 0.05),
    "reduced_energy": (0.08, 0.30, 0.30, 0.12, 0.08, 0.07, 0.05),
    "no_energy": (0.00, 0.34, 0.34, 0.12, 0.08, 0.07, 0.05),
    "energy_heavy": (0.30, 0.21, 0.21, 0.10, 0.07, 0.06, 0.05),
    "practical_heavy": (0.12, 0.25, 0.25, 0.22, 0.06, 0.06, 0.04),
    "condition_heavy": (0.10, 0.25, 0.25, 0.10, 0.07, 0.08, 0.15),
}
COMPONENTS = (
    "energy_score",
    "precursor_prior_score",
    "route_template_score",
    "practical_filter_score",
    "route_complexity_score",
    "phase_availability_score",
    "condition_simplicity_score",
)


def score(row: dict[str, object], weights: tuple[float, ...]) -> float:
    value = sum(float(row.get(component) or 0.0) * weight for component, weight in zip(COMPONENTS, weights))
    template = float(row.get("route_template_score") or 0.0)
    if template >= 0.95:
        value += 0.06
    elif template >= 0.90:
        value += 0.03
    return min(1.0, value)


def evaluate_profile(dataset: pd.DataFrame, case_root: Path, profile: str, weights: tuple[float, ...]) -> pd.DataFrame:
    rows = []
    for target, literature_rows in dataset.groupby("target", sort=False):
        target = normalize(target)
        routes = pd.read_csv(case_root / target / "synthesizability_ranked_routes.csv").fillna("").to_dict("records")
        routes.sort(key=lambda row: (-score(row, weights), pet_energy(row), str(row.get("reaction", ""))))
        for rank, route in enumerate(routes, start=1):
            route["synth_rank"] = rank
        for _, literature in literature_rows.iterrows():
            best = best_similarity_in_topk(
                target,
                split_formulas(literature.get("literature_precursors")),
                split_formulas(literature.get("literature_byproducts")),
                routes,
                5,
            )
            rows.append(
                {
                    "profile": profile,
                    "target": target,
                    "material_category": literature["material_category"],
                    "route_id": literature["route_id"],
                    "moderate_or_better": best.get("route_similarity_label") in MODERATE_LABELS,
                    "similarity_score": float(best.get("route_similarity_score", 0.0)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate synthesizability-score weight sensitivity.")
    parser.add_argument("--case-root", type=Path, default=BENCHMARK_DIR / "cases_blind")
    parser.add_argument("--outdir", type=Path, default=BENCHMARK_DIR / "results")
    args = parser.parse_args()
    dataset = pd.read_csv(BENCHMARK_DIR / "multi_route_literature_dataset.csv").fillna("")
    case_root = args.case_root
    detail = pd.concat(
        [evaluate_profile(dataset, case_root, profile, weights) for profile, weights in PROFILES.items()],
        ignore_index=True,
    )
    summary = (
        detail.groupby("profile", sort=False)
        .agg(
            top5_moderate_or_better=("moderate_or_better", "sum"),
            mean_similarity=("similarity_score", "mean"),
        )
        .reset_index()
        .sort_values(["top5_moderate_or_better", "mean_similarity"], ascending=False)
    )
    category = (
        detail.groupby(["profile", "material_category"], sort=False)
        .agg(
            routes=("route_id", "size"),
            top5_moderate_or_better=("moderate_or_better", "sum"),
            mean_similarity=("similarity_score", "mean"),
        )
        .reset_index()
    )

    held_out_rows = []
    categories = sorted(detail["material_category"].unique())
    for held_out in categories:
        training = detail[detail["material_category"] != held_out]
        training_scores = (
            training.groupby("profile")
            .agg(recovered=("moderate_or_better", "sum"), similarity=("similarity_score", "mean"))
            .sort_values(["recovered", "similarity"], ascending=False)
        )
        selected_profile = training_scores.index[0]
        test = detail[(detail["material_category"] == held_out) & (detail["profile"] == selected_profile)]
        current = detail[(detail["material_category"] == held_out) & (detail["profile"] == "current")]
        held_out_rows.append(
            {
                "held_out_category": held_out,
                "selected_profile_on_training_categories": selected_profile,
                "test_routes": len(test),
                "selected_profile_test_recovered": int(test["moderate_or_better"].sum()),
                "current_test_recovered": int(current["moderate_or_better"].sum()),
                "selected_profile_test_similarity": float(test["similarity_score"].mean()),
                "current_test_similarity": float(current["similarity_score"].mean()),
            }
        )
    held_out = pd.DataFrame(held_out_rows)
    train_targets = []
    test_targets = []
    current_rows = detail[detail["profile"] == "current"]
    for _, group in current_rows.groupby("material_category"):
        targets = sorted(group["target"].unique())
        train_targets.extend(targets[::2])
        test_targets.extend(targets[1::2])
    train = detail[detail["target"].isin(train_targets)]
    train_scores = (
        train.groupby("profile")
        .agg(recovered=("moderate_or_better", "sum"), similarity=("similarity_score", "mean"))
        .sort_values(["recovered", "similarity"], ascending=False)
    )
    selected_profile = train_scores.index[0]
    split_rows = []
    for split_name, targets in (("train", train_targets), ("held_out_targets", test_targets)):
        for profile in ("current", selected_profile):
            selected = detail[(detail["target"].isin(targets)) & (detail["profile"] == profile)]
            split_rows.append(
                {
                    "split": split_name,
                    "profile": profile,
                    "targets": len(set(targets)),
                    "routes": len(selected),
                    "top5_moderate_or_better": int(selected["moderate_or_better"].sum()),
                    "mean_similarity": float(selected["similarity_score"].mean()),
                }
            )
    split_validation = pd.DataFrame(split_rows)
    outdir = args.outdir
    detail.to_csv(outdir / "score_sensitivity_detail.csv", index=False)
    summary.to_csv(outdir / "score_sensitivity_summary.csv", index=False)
    category.to_csv(outdir / "score_sensitivity_by_category.csv", index=False)
    held_out.to_csv(outdir / "score_sensitivity_leave_one_category_out.csv", index=False)
    split_validation.to_csv(outdir / "score_sensitivity_target_holdout.csv", index=False)
    print(summary.to_string(index=False))
    print("\nLeave-one-category-out\n", held_out.to_string(index=False))
    print("\nTarget holdout\n", split_validation.to_string(index=False))


if __name__ == "__main__":
    main()
