#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
ROBUST = RESULTS / "robustness"
EXTERNAL = RESULTS / "external_models"


def main() -> None:
    thresholds = pd.read_csv(ROBUST / "similarity_threshold_sensitivity.csv")
    decoys = pd.read_csv(ROBUST / "similarity_decoy_summary.csv")
    discrimination = pd.read_csv(ROBUST / "score_discrimination_summary.csv")
    strata = pd.read_csv(ROBUST / "stratified_top5_generalization.csv")
    sampling = pd.read_csv(ROBUST / "route_sampling_robustness_summary.csv")
    bootstrap = pd.read_csv(EXTERNAL / "precursor_top5_cluster_bootstrap.csv")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#263746",
            "axes.labelcolor": "#263746",
            "xtick.color": "#435466",
            "ytick.color": "#435466",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.0))
    blue, teal, orange, purple, red = "#3978b8", "#2f9e8f", "#ef8a35", "#7251a1", "#d65f5f"

    top5 = thresholds[thresholds["window"] == "best_top5"]
    axes[0, 0].plot(top5["threshold"], top5["recovery"], marker="o", color=blue, label="DOI routes")
    axes[0, 0].plot(decoys["threshold"], decoys["mean_false_positive_rate"], marker="o", color=red, label="Target-swapped decoys")
    axes[0, 0].fill_between(decoys["threshold"], decoys["ci95_low"], decoys["ci95_high"], color=red, alpha=0.14)
    axes[0, 0].axvline(0.62, color="#777777", linestyle="--", linewidth=1)
    axes[0, 0].axvline(0.78, color="#777777", linestyle=":", linewidth=1)
    axes[0, 0].set_xlabel("Route-similarity threshold")
    axes[0, 0].set_ylabel("Fraction")
    axes[0, 0].set_ylim(0, 0.9)
    axes[0, 0].set_title("a  Similarity threshold and negative controls", loc="left")
    axes[0, 0].legend(frameon=False)

    selected = discrimination[
        (discrimination["endpoint"] == "exact_full_route")
        & discrimination["predictor"].isin(["synth_score", "pet_energy_predictor", "precursor_prior_score", "route_template_score"])
    ].copy()
    labels = {
        "synth_score": "InvDesRoute score",
        "pet_energy_predictor": "PET energy",
        "precursor_prior_score": "Precursor prior",
        "route_template_score": "Route template",
    }
    selected["label"] = selected["predictor"].map(labels)
    selected = selected.sort_values("roc_auc")
    axes[0, 1].barh(selected["label"], selected["roc_auc"], color=[red, purple, teal, blue])
    axes[0, 1].axvline(0.5, color="#777777", linestyle="--", linewidth=1)
    axes[0, 1].set_xlim(0.25, 0.95)
    axes[0, 1].set_xlabel("ROC AUC for exact full-route labels")
    axes[0, 1].set_title("b  Candidate-level score discrimination", loc="left")
    for index, value in enumerate(selected["roc_auc"]):
        axes[0, 1].text(value + 0.012, index, f"{value:.3f}", va="center", fontsize=8)

    era = strata[(strata["grouping"] == "publication_era") & (strata["mode"].isin(["pet_energy", "synth_score"]))]
    era_order = ["1998-2008", "2009-2013", "2014-2018"]
    x = np.arange(len(era_order))
    width = 0.34
    for offset, (mode, label, color) in zip([-width / 2, width / 2], [("pet_energy", "PET energy", red), ("synth_score", "InvDesRoute", blue)]):
        current = era[era["mode"] == mode].set_index("stratum").loc[era_order]
        axes[1, 0].bar(x + offset, current["recovery"], width=width, label=label, color=color)
    axes[1, 0].set_xticks(x, era_order)
    axes[1, 0].set_ylim(0, 0.85)
    axes[1, 0].set_ylabel("Top-5 full-route moderate+ recovery")
    axes[1, 0].set_title("c  Publication-era stratification", loc="left")
    axes[1, 0].legend(frameon=False)

    decode_names = ["CSLLM-greedy", "CSLLM-beam5", "CSLLM-precursor-llama3-release", "InvDesRoute"]
    decode_labels = ["CSLLM greedy", "CSLLM beam-5", "CSLLM beam-10", "InvDesRoute"]
    decode = bootstrap.set_index("model").loc[decode_names]
    values = decode["coverage"].to_numpy()
    errors = np.vstack([values - decode["ci95_low"].to_numpy(), decode["ci95_high"].to_numpy() - values])
    axes[1, 1].barh(range(4), values, color=["#b8a6d5", "#9678bf", purple, blue])
    axes[1, 1].errorbar(values, range(4), xerr=errors, fmt="none", ecolor="#263746", capsize=2, linewidth=0.8)
    axes[1, 1].set_yticks(range(4), decode_labels)
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_xlim(0.55, 1.0)
    axes[1, 1].set_xlabel("Top-5 precursor-set moderate+ recovery")
    axes[1, 1].set_title("d  External decoding sensitivity", loc="left")
    for index, value in enumerate(values):
        axes[1, 1].text(value + 0.008, index, f"{round(value * 120):.0f}/120", va="center", fontsize=8)
    inv_sample = sampling[sampling["mode"] == "synth_score"].iloc[0]
    axes[1, 1].text(
        0.0,
        -0.22,
        f"Unique-route recovery: {inv_sample.unique_route_recovery:.3f}; one-route-per-target mean: {inv_sample.one_unique_route_per_target_mean:.3f}",
        transform=axes[1, 1].transAxes,
        fontsize=8,
    )

    fig.tight_layout(h_pad=2.7, w_pad=3.0)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(ROBUST / f"robustness_and_ablation_summary.{suffix}", dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
