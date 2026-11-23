#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SEED = 20260722
ITERATIONS = 1000


def select_profile(training: pd.DataFrame) -> str:
    ranked = (
        training.groupby("profile", sort=False)
        .agg(recovered=("moderate_or_better", "sum"), similarity=("similarity_score", "mean"))
        .reset_index()
        .sort_values(["recovered", "similarity", "profile"], ascending=[False, False, True])
    )
    return str(ranked.iloc[0]["profile"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-validate score-weight profiles.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    args = parser.parse_args()
    detail = pd.read_csv(args.results_dir / "score_sensitivity_detail.csv")
    current = detail[detail["profile"] == "current"]
    target_categories = current[["target", "material_category"]].drop_duplicates()
    grouped = {
        category: sorted(group["target"].unique())
        for category, group in target_categories.groupby("material_category")
    }
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []

    for iteration in range(ITERATIONS):
        test_targets = {rng.choice(targets) for targets in grouped.values()}
        train = detail[~detail["target"].isin(test_targets)]
        test = detail[detail["target"].isin(test_targets)]
        selected_profile = select_profile(train)
        selected = test[test["profile"] == selected_profile]
        baseline = test[test["profile"] == "current"]
        rows.append(
            {
                "iteration": iteration,
                "selected_profile": selected_profile,
                "test_targets": len(test_targets),
                "test_records": len(baseline),
                "selected_recovered": int(selected["moderate_or_better"].sum()),
                "current_recovered": int(baseline["moderate_or_better"].sum()),
                "recovery_difference": int(selected["moderate_or_better"].sum())
                - int(baseline["moderate_or_better"].sum()),
                "selected_similarity": float(selected["similarity_score"].mean()),
                "current_similarity": float(baseline["similarity_score"].mean()),
                "similarity_difference": float(selected["similarity_score"].mean())
                - float(baseline["similarity_score"].mean()),
            }
        )

    iterations = pd.DataFrame(rows)
    frequencies = Counter(iterations["selected_profile"])
    summary = pd.DataFrame(
        [
            {
                "iterations": ITERATIONS,
                "split_design": "one held-out target per material category",
                "mean_test_records": iterations["test_records"].mean(),
                "mean_selected_recovered": iterations["selected_recovered"].mean(),
                "mean_current_recovered": iterations["current_recovered"].mean(),
                "mean_recovery_difference": iterations["recovery_difference"].mean(),
                "recovery_difference_ci_low": iterations["recovery_difference"].quantile(0.025),
                "recovery_difference_ci_high": iterations["recovery_difference"].quantile(0.975),
                "mean_similarity_difference": iterations["similarity_difference"].mean(),
                "similarity_difference_ci_low": iterations["similarity_difference"].quantile(0.025),
                "similarity_difference_ci_high": iterations["similarity_difference"].quantile(0.975),
                "selected_profile_counts": ";".join(
                    f"{profile}:{frequencies[profile]}" for profile in sorted(frequencies)
                ),
                "random_seed": SEED,
            }
        ]
    )
    iterations.to_csv(args.results_dir / "weight_profile_cv_iterations.csv", index=False)
    summary.to_csv(args.results_dir / "weight_profile_cv_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
