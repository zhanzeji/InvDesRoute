#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
EXTERNAL = RESULTS / "external_models"


def label_bars(axis, values, denominator, offset=0.012) -> None:
    for index, value in enumerate(values):
        axis.text(value / denominator + offset, index, f"{int(value)}/{denominator}", va="center", fontsize=8)


def main() -> None:
    mode = pd.read_csv(RESULTS / "multi_route_mode_summary.csv")
    components = pd.read_csv(RESULTS / "component_ablation_summary.csv")
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
    blue = "#3c78b5"
    teal = "#2f9e8f"
    orange = "#ed8b3a"
    purple = "#6b4c9a"
    pale = "#c9d8e8"

    pool = mode[(mode["mode"] == "full_network") & (mode["top_k"].astype(str) == "all")].set_index("network")
    pool_names = ["MP phases", "+ generic library", "+ target-specific\nprecursors (upper bound)"]
    pool_values = [pool.loc["mp_only", "record_moderate_or_better"], pool.loc["blind", "record_moderate_or_better"], pool.loc["seeded_upper_bound", "record_moderate_or_better"]]
    axes[0, 0].barh(pool_names, np.array(pool_values) / 120, color=[pale, teal, orange], height=0.62)
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_xlim(0, 1.05)
    axes[0, 0].set_xlabel("Full-network moderate+ recovery")
    axes[0, 0].set_title("a  Candidate-space ablation", loc="left")
    label_bars(axes[0, 0], pool_values, 120)
    axes[0, 0].text(0.0, -0.25, "120 DOI records; 87 unique routes", transform=axes[0, 0].transAxes, fontsize=8)

    model_order = [
        "PET-energy-network",
        "Practical-PET-network",
        "Composition-retrieval",
        "CSLLM-precursor-llama3-release",
        "Chemistry-prior-network",
        "InvDesRoute",
        "InvDesRoute-condition-heavy",
        "Elementwise-frequency",
    ]
    model_labels = {
        "PET-energy-network": "PET energy",
        "Practical-PET-network": "Practical + PET",
        "Composition-retrieval": "Composition retrieval",
        "CSLLM-precursor-llama3-release": "CSLLM",
        "Chemistry-prior-network": "Chemistry-prior network",
        "InvDesRoute": "InvDesRoute",
        "InvDesRoute-condition-heavy": "InvDesRoute, condition-heavy",
        "Elementwise-frequency": "Elementwise frequency",
    }
    comparison = bootstrap[bootstrap["model"].isin(model_order)].set_index("model").loc[model_order]
    values = comparison["coverage"].to_numpy()
    low = values - comparison["ci95_low"].to_numpy()
    high = comparison["ci95_high"].to_numpy() - values
    colors = [pale, "#a8bfd8", "#78a6ca", purple, "#4da99b", blue, orange, "#d65f5f"]
    y = np.arange(len(model_order))
    axes[0, 1].barh(y, values, color=colors, height=0.62)
    axes[0, 1].errorbar(values, y, xerr=np.vstack([low, high]), fmt="none", ecolor="#263746", elinewidth=0.8, capsize=2)
    axes[0, 1].set_yticks(y, [model_labels[name] for name in model_order])
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_xlim(0, 1.04)
    axes[0, 1].set_xlabel("Top-5 precursor-set moderate+ recovery")
    axes[0, 1].set_title("b  Common precursor-set comparison", loc="left")
    for index, value in enumerate(values):
        axes[0, 1].text(value + 0.012, index, f"{round(value * 120):.0f}/120", va="center", fontsize=7.5)

    component_order = [
        "without_route_template_score",
        "without_practical_filter_score",
        "without_precursor_prior_score",
        "without_energy_score",
        "without_route_complexity_score",
        "without_phase_availability_score",
        "current",
        "without_condition_simplicity_score",
        "condition_heavy",
    ]
    component_labels = {
        "without_route_template_score": "No route template",
        "without_practical_filter_score": "No practical-route term",
        "without_precursor_prior_score": "No precursor prior",
        "without_energy_score": "No energy term",
        "without_route_complexity_score": "No complexity term",
        "without_phase_availability_score": "No availability term",
        "current": "Current score",
        "without_condition_simplicity_score": "No condition term",
        "condition_heavy": "Condition-heavy candidate",
    }
    top5 = components[(components["top_k"] == 5) & (components["parameter"].isin(component_order))].set_index("parameter").loc[component_order]
    comp_values = top5["moderate_or_better"].to_numpy()
    comp_colors = ["#d86c68" if name.startswith("without_") else blue if name == "current" else orange for name in component_order]
    axes[1, 0].barh(range(len(component_order)), comp_values / 120, color=comp_colors, height=0.62)
    axes[1, 0].set_yticks(range(len(component_order)), [component_labels[name] for name in component_order])
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlim(0.50, 0.82)
    axes[1, 0].set_xlabel("Top-5 full-route moderate+ recovery")
    axes[1, 0].set_title("c  Score-component ablation", loc="left")
    label_bars(axes[1, 0], comp_values, 120, offset=0.004)

    topks = [5, 10, 20]
    current = mode[(mode["network"] == "blind") & (mode["mode"] == "synth_score")].set_index("top_k")
    diverse = mode[(mode["network"] == "blind") & (mode["mode"] == "diverse_synth_score")].set_index("top_k")
    current_values = [current.loc[str(k), "record_moderate_or_better"] for k in topks]
    diverse_values = [diverse.loc[str(k), "record_moderate_or_better"] for k in topks]
    axes[1, 1].plot(topks, np.array(current_values) / 120, marker="o", linewidth=2.0, color=blue, label="Score only")
    axes[1, 1].plot(topks, np.array(diverse_values) / 120, marker="o", linewidth=2.0, color=orange, label="Diversity-aware")
    axes[1, 1].set_xticks(topks)
    axes[1, 1].set_ylim(0.64, 0.98)
    axes[1, 1].set_xlabel("Recommendation window, top-k")
    axes[1, 1].set_ylabel("Full-route moderate+ recovery")
    axes[1, 1].set_title("d  Diversity-display ablation", loc="left")
    axes[1, 1].legend(frameon=False, loc="lower right")
    for x, value in zip(topks, current_values):
        axes[1, 1].annotate(f"{int(value)}/120", (x, value / 120), xytext=(0, -15), textcoords="offset points", ha="center", color=blue, fontsize=8)
    for x, value in zip(topks, diverse_values):
        axes[1, 1].annotate(f"{int(value)}/120", (x, value / 120), xytext=(0, 7), textcoords="offset points", ha="center", color=orange, fontsize=8)

    fig.tight_layout(h_pad=2.4, w_pad=2.6)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(RESULTS / f"multi_route_benchmark_summary.{suffix}", dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
