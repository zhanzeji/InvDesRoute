#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MODES = ["pet_energy", "practical_pet_energy", "chemistry_prior_generator", "synth_score", "diverse_synth_score"]


def cluster_interval(group: pd.DataFrame, iterations: int = 10000) -> tuple[float, float]:
    rng = np.random.default_rng(20260722)
    targets = sorted(group["target"].unique())
    by_target = {target: group[group["target"] == target]["moderate_or_better"].astype(float).to_numpy() for target in targets}
    values = []
    for _ in range(iterations):
        sampled = rng.choice(targets, size=len(targets), replace=True)
        values.append(float(np.concatenate([by_target[target] for target in sampled]).mean()))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser(description="Stratify top-5 generalization results.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    args = parser.parse_args()
    detail = pd.read_csv(args.results_dir / "multi_route_route_level_detail.csv")
    metadata = pd.read_csv(RESULTS / "multi_route_literature_dataset_with_doi_metadata.csv")
    metadata = metadata[["route_id", "publication_year"]]
    data = detail[(detail["network"] == "blind") & (detail["top_k"].astype(str) == "5") & detail["mode"].isin(MODES)]
    data = data.merge(metadata, on="route_id", how="left")
    data["publication_era"] = pd.cut(
        data["publication_year"], bins=[0, 2008, 2013, 9999], labels=["1998-2008", "2009-2013", "2014-2018"]
    )

    rows = []
    for grouping in ("material_category", "publication_era"):
        for (mode, stratum), group in data.groupby(["mode", grouping], observed=True):
            low, high = cluster_interval(group)
            rows.append(
                {
                    "grouping": grouping,
                    "stratum": str(stratum),
                    "mode": mode,
                    "records": len(group),
                    "targets": group["target"].nunique(),
                    "moderate_or_better": int(group["moderate_or_better"].sum()),
                    "recovery": float(group["moderate_or_better"].mean()),
                    "cluster_ci95_low": low,
                    "cluster_ci95_high": high,
                }
            )
    output = pd.DataFrame(rows)
    outdir = args.results_dir / "robustness"
    outdir.mkdir(parents=True, exist_ok=True)
    output.to_csv(outdir / "stratified_top5_generalization.csv", index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
