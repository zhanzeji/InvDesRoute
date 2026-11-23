#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MODES = ["pet_energy", "practical_pet_energy", "chemistry_prior_generator", "synth_score", "diverse_synth_score"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detail = pd.read_csv(args.results_dir / "multi_route_route_level_detail.csv")
    detail = detail[(detail["network"] == "blind") & (detail["top_k"].astype(str) == "5") & detail["mode"].isin(MODES)]
    rng = np.random.default_rng(20260722)
    iterations = 10000
    summaries = []
    samples = []
    for mode in MODES:
        data = detail[detail["mode"] == mode].copy()
        unique = data.drop_duplicates(["target", "route_pair_signature"])
        per_target = data.groupby("target")["moderate_or_better"].mean()
        sampled_values = []
        for iteration in range(iterations):
            chosen = []
            for _, group in unique.groupby("target"):
                chosen.append(bool(group.iloc[int(rng.integers(0, len(group)))]["moderate_or_better"]))
            value = float(np.mean(chosen))
            sampled_values.append(value)
            samples.append({"iteration": iteration, "mode": mode, "one_unique_route_per_target_recovery": value})
        array = np.asarray(sampled_values)
        summaries.append(
            {
                "mode": mode,
                "record_recovery": float(data["moderate_or_better"].mean()),
                "unique_route_recovery": float(unique["moderate_or_better"].mean()),
                "macro_target_recovery": float(per_target.mean()),
                "one_unique_route_per_target_mean": float(array.mean()),
                "ci95_low": float(np.quantile(array, 0.025)),
                "ci95_high": float(np.quantile(array, 0.975)),
            }
        )
    outdir = args.results_dir / "robustness"
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(outdir / "route_sampling_robustness_summary.csv", index=False)
    pd.DataFrame(samples).to_csv(outdir / "route_sampling_iterations.csv", index=False)
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
