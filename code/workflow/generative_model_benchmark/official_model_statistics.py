#!/usr/bin/env python3
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OFFICIAL_MODELS = {
    "InvDesRoute-corpus-informed",
    "CSLLM-precursor-llama3-release",
    "ChemDFM-v1.5-8B-official-zero-shot",
    "ChemLLM-7B-Chat-official-zero-shot",
    "Mistral-7B-Instruct-v0.3-official-zero-shot",
    "LLM4SYN-TGT2CEQ-v1.2.1-official",
}
DISPLAY_NAMES = {
    "InvDesRoute-corpus-informed": "InvDesRoute",
    "CSLLM-precursor-llama3-release": "CSLLM (released checkpoint)",
    "ChemDFM-v1.5-8B-official-zero-shot": "ChemDFM-v1.5-8B",
    "ChemLLM-7B-Chat-official-zero-shot": "ChemLLM-7B-Chat",
    "Mistral-7B-Instruct-v0.3-official-zero-shot": "Mistral-7B-Instruct-v0.3",
    "LLM4SYN-TGT2CEQ-v1.2.1-official": "LLM4SYN-TGT2CEQ-v1.2.1",
}


def main() -> None:
    detail = pd.read_csv(RESULTS / "model_comparison_detail.csv")
    detail = detail[(detail.top_k == 5) & detail.model.isin(OFFICIAL_MODELS)].copy()
    detail["model"] = detail.model.map(lambda value: DISPLAY_NAMES.get(value, value))
    metrics = ["strict_exact_precursor_set", "strong_or_better", "moderate_or_better", "best_precursor_recall"]
    rng = np.random.default_rng(20260729)
    rows = []
    for model_a, model_b in itertools.combinations(sorted(detail.model.unique()), 2):
        for metric in metrics:
            values = (
                detail.groupby(["model", "target"])[metric]
                .mean()
                .unstack("model")
                .dropna(subset=[model_a, model_b])
            )
            paired = (values[model_a] - values[model_b]).to_numpy(dtype=float)
            observed = float(paired.mean())
            bootstrap = np.array([
                rng.choice(paired, size=len(paired), replace=True).mean()
                for _ in range(10_000)
            ])
            signs = rng.choice([-1.0, 1.0], size=(100_000, len(paired)))
            null = (signs * paired).mean(axis=1)
            rows.append({
                "comparison": f"{model_a} minus {model_b}",
                "metric": metric,
                "top_k": 5,
                "mean_target_level_difference": observed,
                "target_cluster_ci95_low": float(np.quantile(bootstrap, 0.025)),
                "target_cluster_ci95_high": float(np.quantile(bootstrap, 0.975)),
                "target_sign_permutation_p": float((np.abs(null) >= abs(observed)).mean()),
                "n_target_clusters": len(paired),
            })
    pd.DataFrame(rows).to_csv(RESULTS / "official_model_paired_statistics.csv", index=False)


if __name__ == "__main__":
    main()
