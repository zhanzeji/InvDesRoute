#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "external_models"


def load_detail_files() -> pd.DataFrame:
    frames = []
    for path in sorted(RESULTS.glob("*_precursor_set_detail.csv")):
        frame = pd.read_csv(path)
        frames.append(frame[frame["top_k"] == 5])
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    return combined.drop_duplicates(["model", "route_id", "top_k"], keep="last")


def cluster_bootstrap(detail: pd.DataFrame, iterations: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    route_to_target = pd.read_csv(ROOT / "multi_route_literature_dataset.csv")[["route_id", "target"]]
    detail = detail.merge(route_to_target, on="route_id", how="left", suffixes=("", "_reference"))
    if "target_reference" in detail:
        detail["target"] = detail["target_reference"].fillna(detail.get("target", ""))
    targets = sorted(route_to_target["target"].unique())
    models = sorted(detail["model"].unique())
    model_maps = {
        model: detail[detail["model"] == model].set_index("route_id")["moderate_or_better"].astype(float)
        for model in models
    }
    target_routes = route_to_target.groupby("target")["route_id"].apply(list).to_dict()
    samples = {model: [] for model in models}
    difference_samples = {model: [] for model in models if model != "InvDesRoute"}
    for _ in range(iterations):
        sampled_targets = rng.choice(targets, size=len(targets), replace=True)
        route_ids = [route_id for target in sampled_targets for route_id in target_routes[target]]
        inv_values = model_maps.get("InvDesRoute", pd.Series(dtype=float)).reindex(route_ids).fillna(0.0).to_numpy()
        for model in models:
            values = model_maps[model].reindex(route_ids).fillna(0.0).to_numpy()
            samples[model].append(float(values.mean()))
            if model != "InvDesRoute":
                difference_samples[model].append(float((values - inv_values).mean()))
    summary_rows = []
    for model in models:
        observed = float(model_maps[model].reindex(route_to_target["route_id"]).fillna(0.0).mean())
        values = np.asarray(samples[model])
        summary_rows.append(
            {
                "model": model,
                "top_k": 5,
                "coverage": observed,
                "ci95_low": float(np.quantile(values, 0.025)),
                "ci95_high": float(np.quantile(values, 0.975)),
            }
        )
    difference_rows = []
    for model, values_list in difference_samples.items():
        values = np.asarray(values_list)
        observed = float(
            (
                model_maps[model].reindex(route_to_target["route_id"]).fillna(0.0)
                - model_maps["InvDesRoute"].reindex(route_to_target["route_id"]).fillna(0.0)
            ).mean()
        )
        p_value = 2.0 * min(float((values <= 0).mean()), float((values >= 0).mean()))
        difference_rows.append(
            {
                "model_minus_InvDesRoute": model,
                "observed_difference": observed,
                "ci95_low": float(np.quantile(values, 0.025)),
                "ci95_high": float(np.quantile(values, 0.975)),
                "bootstrap_two_sided_p": min(1.0, p_value),
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(difference_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()
    detail = load_detail_files()
    summary, differences = cluster_bootstrap(detail, args.iterations, args.seed)
    summary.to_csv(RESULTS / "precursor_top5_cluster_bootstrap.csv", index=False)
    differences.to_csv(RESULTS / "precursor_top5_paired_differences.csv", index=False)
    print(summary.sort_values("coverage", ascending=False).to_string(index=False))
    print("\nPaired differences\n", differences.sort_values("observed_difference", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
