#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def family_name(model: str) -> str:
    model = re.sub(r"-seed\d+$", "", model)
    return {
        "ElemwiseRetro-architecture-reproduction": "ElemwiseRetro-style",
        "Retrieval-Retro-architecture-reproduction": "Retrieval-Retro-style",
        "CVAE-precursor-architecture-reproduction": "Precursor CVAE",
        "CSLLM-precursor-llama3-release": "CSLLM (released)",
    }.get(model, model)


def main() -> None:
    detail = pd.read_csv(RESULTS / "model_comparison_detail.csv")
    detail["family"] = detail.model.map(family_name)
    detail = detail[(detail.top_k == 5) & detail.family.isin({
        "CSLLM (released)", "Precursor CVAE", "Retrieval-Retro-style", "ElemwiseRetro-style", "InvDesRoute"
    })]
    rng = np.random.default_rng(20260724)
    output = []
    for metric in ["strict_exact_precursor_set", "moderate_or_better", "best_precursor_recall"]:
        route_family = detail.groupby(["family", "route_id", "target"])[metric].mean().reset_index()
        reference = route_family[route_family.family == "InvDesRoute"].set_index("route_id")[metric]
        for family in ["CSLLM (released)", "Precursor CVAE", "Retrieval-Retro-style", "ElemwiseRetro-style"]:
            current = route_family[route_family.family == family].set_index("route_id")
            common = current.index.intersection(reference.index)
            paired = pd.DataFrame({
                "target": current.loc[common, "target"],
                "difference": current.loc[common, metric].astype(float) - reference.loc[common].astype(float),
            })
            target_differences = paired.groupby("target").difference.mean().to_numpy()
            observed = float(target_differences.mean())
            bootstrap = np.array([
                rng.choice(target_differences, size=len(target_differences), replace=True).mean()
                for _ in range(10000)
            ])
            signs = rng.choice([-1.0, 1.0], size=(100000, len(target_differences)))
            null = (signs * target_differences).mean(axis=1)
            p_value = float((np.abs(null) >= abs(observed)).mean())
            output.append({
                "comparison": f"{family} minus InvDesRoute",
                "metric": metric,
                "top_k": 5,
                "mean_paired_difference": observed,
                "target_cluster_ci95_low": float(np.quantile(bootstrap, 0.025)),
                "target_cluster_ci95_high": float(np.quantile(bootstrap, 0.975)),
                "target_sign_permutation_p": p_value,
                "n_target_clusters": len(target_differences),
            })
    pd.DataFrame(output).to_csv(RESULTS / "paired_target_cluster_statistics.csv", index=False)


if __name__ == "__main__":
    main()
