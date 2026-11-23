#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def family_name(model: str) -> str:
    model = re.sub(r"-seed\d+$", "", model)
    names = {
        "ElemwiseRetro-architecture-reproduction": "ElemwiseRetro-style",
        "Retrieval-Retro-architecture-reproduction": "Retrieval-Retro-style",
        "CVAE-precursor-architecture-reproduction": "Precursor CVAE",
        "CSLLM-precursor-llama3-release": "CSLLM (released)",
    }
    return names.get(model, model)


def main() -> None:
    summary = pd.read_csv(RESULTS / "model_comparison_summary.csv")
    summary["family"] = summary["model"].map(family_name)
    stochastic = {"ElemwiseRetro-style", "Retrieval-Retro-style", "Precursor CVAE"}
    selected = summary[
        summary["family"].isin(stochastic | {"CSLLM (released)", "InvDesRoute"})
        & ~summary["model"].isin({"CSLLM-beam5", "CSLLM-greedy"})
    ].copy()
    metrics = [
        "strict_exact",
        "moderate_or_better",
        "mean_precursor_jaccard",
        "mean_precursor_recall",
        "mean_family_similarity",
        "mean_target_element_coverage",
    ]
    aggregates = []
    for (family, top_k), group in selected.groupby(["family", "top_k"]):
        for metric in metrics:
            values = group[metric].astype(float)
            aggregates.append(
                {
                    "model_family": family,
                    "top_k": top_k,
                    "metric": metric,
                    "mean": values.mean(),
                    "sd_across_seeds": values.std(ddof=1) if len(values) > 1 else 0.0,
                    "n_runs": len(values),
                }
            )
    aggregate = pd.DataFrame(aggregates)
    aggregate.to_csv(RESULTS / "model_family_aggregate.csv", index=False)

    capability = pd.DataFrame(
        [
            ["CSLLM (released)", "formula/text", True, False, False, False, False, False],
            ["ElemwiseRetro-style", "formula", True, False, False, False, False, False],
            ["Retrieval-Retro-style", "formula", True, False, False, False, False, False],
            ["Precursor CVAE", "formula", True, False, False, False, False, False],
            ["InvDesRoute", "formula or CIF", True, True, True, True, True, True],
        ],
        columns=[
            "model",
            "input",
            "precursor_sets",
            "balanced_reactions",
            "structure_energy",
            "Ehull_context",
            "decomposition_competition",
            "risk_and_refinement",
        ],
    )
    capability.to_csv(RESULTS / "model_capability_matrix.csv", index=False)

    colors = {
        "CSLLM (released)": "#7B5AA6",
        "ElemwiseRetro-style": "#2A9D8F",
        "Retrieval-Retro-style": "#3A78B8",
        "Precursor CVAE": "#E9A23B",
        "InvDesRoute": "#D64B5F",
    }
    order = ["CSLLM (released)", "Precursor CVAE", "Retrieval-Retro-style", "ElemwiseRetro-style", "InvDesRoute"]
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.3))
    ax = axes[0, 0]
    x = np.arange(len(order))
    width = 0.36
    for offset, (metric, label) in zip([-width / 2, width / 2], [("strict_exact", "Strict exact"), ("moderate_or_better", "Moderate+ family")]):
        means, errors = [], []
        for family in order:
            rows = aggregate[(aggregate.model_family == family) & (aggregate.top_k == 5) & (aggregate.metric == metric)]
            means.append(float(rows.iloc[0]["mean"]) / 120)
            errors.append(float(rows.iloc[0]["sd_across_seeds"]) / 120)
        ax.bar(x + offset, means, width, yerr=errors, capsize=3, label=label, alpha=0.9)
    ax.set_xticks(x, order, rotation=24, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Top-5 literature-route recovery")
    ax.legend(frameon=False, ncol=2)
    ax.set_title("a  Precursor-set prediction on 120 held-out routes", loc="left", fontweight="bold")

    ax = axes[0, 1]
    for family in order:
        curve = aggregate[(aggregate.model_family == family) & (aggregate.metric == "moderate_or_better")].sort_values("top_k")
        ax.plot(curve.top_k, curve["mean"] / 120, marker="o", lw=2.2, color=colors[family], label=family)
    ax.set_xticks([1, 5, 10, 20])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Recommendation window, top-k")
    ax.set_ylabel("Moderate+ family recovery")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("b  Route-family coverage versus recommendation depth", loc="left", fontweight="bold")

    ax = axes[1, 0]
    ablation_names = ["PET-energy-network", "Practical-PET-network", "Chemistry-prior-network", "InvDesRoute"]
    ablation_labels = ["PET energy", "Practical + PET", "Chemistry prior", "Full InvDesRoute"]
    values = []
    for name in ablation_names:
        row = summary[(summary.model == name) & (summary.top_k == 5)].iloc[0]
        values.append(row.moderate_or_better / 120)
    bars = ax.barh(ablation_labels, values, color=["#CDD7E1", "#9BB8D3", "#70A6C8", "#D64B5F"])
    for bar, value in zip(bars, values):
        ax.text(value + 0.015, bar.get_y() + bar.get_height() / 2, f"{round(value * 120):.0f}/120", va="center")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Top-5 moderate+ family recovery")
    ax.set_title("c  Ranking ablation under the strict evaluator", loc="left", fontweight="bold")

    ax = axes[1, 1]
    columns = ["precursor_sets", "balanced_reactions", "structure_energy", "Ehull_context", "decomposition_competition", "risk_and_refinement"]
    matrix = capability.set_index("model").loc[order, columns].astype(int).to_numpy()
    ax.imshow(matrix, cmap=plt.matplotlib.colors.ListedColormap(["#F0F2F4", "#2A9D8F"]), vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(order)), order)
    ax.set_xticks(range(len(columns)), ["Precursors", "Balanced\nroutes", "PET\nenergy", "Ehull", "Decomp.", "Risk /\nrefine"], rotation=0)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, "✓" if matrix[row, column] else "–", ha="center", va="center", color="white" if matrix[row, column] else "#65707C", fontweight="bold")
    ax.set_title("d  Output capability is broader than precursor recall", loc="left", fontweight="bold")

    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    for suffix in ["png", "pdf", "svg"]:
        figure.savefig(RESULTS / f"generative_model_comparison.{suffix}", dpi=400, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
