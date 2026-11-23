#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PROJECT = ROOT.parent
MODELS = [
    "EquiformerV3-DeNS-OMat24-MPtrj-sAlex",
    "TECE-OAM-RRA-1.0",
    "EquFlashV2-45M-OAM",
    "GRACE-3L-OAM-L",
    "SevenNet-Omni-i12",
    "ORB-v3-conservative-inf-MPA",
    "PET-OAM-XL-protocol-check",
]


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = scores[labels]
    negatives = scores[~labels]
    if not len(positives) or not len(negatives):
        return float("nan")
    comparisons = (positives[:, None] > negatives[None, :]).mean()
    ties = (positives[:, None] == negatives[None, :]).mean()
    return float(comparisons + 0.5 * ties)


def main() -> None:
    merged = None
    for model in MODELS:
        frame = pd.read_csv(RESULTS / f"{model}_reaction_comparison.csv")
        frame = frame[["reaction", "predicted_delta_e_eV", "mp_dft_delta_e_eV"]].rename(
            columns={"predicted_delta_e_eV": model}
        )
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame[["reaction", model]], on="reaction", how="inner")
    assert merged is not None and len(merged) == 416

    predictions = merged[MODELS].to_numpy(float)
    reference = merged["mp_dft_delta_e_eV"].to_numpy(float)
    equiformer = merged[MODELS[0]].to_numpy(float)
    merged["ensemble_mean_eV"] = predictions.mean(axis=1)
    merged["ensemble_std_eV"] = predictions.std(axis=1, ddof=1)
    merged["ensemble_range_eV"] = predictions.max(axis=1) - predictions.min(axis=1)
    merged["equiformer_absolute_error_eV"] = np.abs(equiformer - reference)
    merged["equiformer_same_sign"] = np.sign(equiformer) == np.sign(reference)
    merged["model_sign_vote_fraction"] = np.maximum(
        (predictions < 0).mean(axis=1), (predictions >= 0).mean(axis=1)
    )

    high_error_threshold = 0.10
    labels = merged["equiformer_absolute_error_eV"].to_numpy() > high_error_threshold
    pearson = pearsonr(merged["ensemble_std_eV"], merged["equiformer_absolute_error_eV"])
    spearman = spearmanr(merged["ensemble_std_eV"], merged["equiformer_absolute_error_eV"])

    merged["uncertainty_quintile"] = pd.qcut(
        merged["ensemble_std_eV"], 5, labels=["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"]
    )
    bins = (
        merged.groupby("uncertainty_quintile", observed=True, as_index=False)
        .agg(
            reactions=("reaction", "size"),
            mean_ensemble_std_eV=("ensemble_std_eV", "mean"),
            equiformer_mae_eV=("equiformer_absolute_error_eV", "mean"),
            high_error_fraction=("equiformer_absolute_error_eV", lambda values: float((values > high_error_threshold).mean())),
            sign_consistency=("equiformer_same_sign", "mean"),
        )
    )

    coverage_rows = []
    ordered = merged.sort_values("ensemble_std_eV")
    for coverage in (0.50, 0.70, 0.80, 0.90, 1.00):
        retained = ordered.iloc[: int(np.ceil(coverage * len(ordered)))]
        coverage_rows.append(
            {
                "retained_fraction": coverage,
                "reactions": len(retained),
                "max_ensemble_std_eV": float(retained["ensemble_std_eV"].max()),
                "equiformer_mae_eV": float(retained["equiformer_absolute_error_eV"].mean()),
                "sign_consistency": float(retained["equiformer_same_sign"].mean()),
            }
        )
    coverage = pd.DataFrame(coverage_rows)

    merged.to_csv(RESULTS / "route_ensemble_uncertainty_detail.csv", index=False)
    bins.to_csv(RESULTS / "route_ensemble_uncertainty_quintiles.csv", index=False)
    coverage.to_csv(RESULTS / "route_ensemble_uncertainty_coverage.csv", index=False)
    summary = {
        "models": MODELS,
        "reactions": len(merged),
        "high_error_definition_eV": high_error_threshold,
        "high_error_reactions": int(labels.sum()),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "high_error_detection_roc_auc": auc(labels, merged["ensemble_std_eV"].to_numpy()),
        "lowest_80pct_mae_eV": float(coverage.loc[coverage["retained_fraction"].eq(0.80), "equiformer_mae_eV"].iloc[0]),
        "highest_20pct_mae_eV": float(ordered.iloc[int(np.ceil(0.80 * len(ordered))) :]["equiformer_absolute_error_eV"].mean()),
        "interpretation": "Cross-model reaction-energy spread is evaluated as a route-level refinement signal, not as a calibrated uncertainty probability.",
    }

    shared_path = PROJECT / "results" / "production_model_comparison" / "shared_120_reaction_details.csv"
    target_path = PROJECT / "results" / "production_model_comparison" / "shared_120_target_metrics.csv"
    if shared_path.exists() and target_path.exists():
        shared = pd.read_csv(shared_path)
        shared_models = [
            "DeltaE_PET-OAM-XI_per_target",
            "DeltaE_GRACE-3L-OAM-L_per_target",
            "DeltaE_EquiformerV3+DeNS-OAM2_per_target",
        ]
        shared["three_model_std_eV"] = shared[shared_models].std(axis=1, ddof=1)
        shared["equiformer_mp_form_absolute_error_eV"] = shared["EquiformerV3_error"].abs()
        broad_rho = spearmanr(
            shared["three_model_std_eV"], shared["equiformer_mp_form_absolute_error_eV"]
        )
        target_rows = []
        target_metrics = pd.read_csv(target_path)[["target", "EquiformerV3_top1"]]
        for target, group in shared.groupby("target"):
            top = group.nsmallest(min(5, len(group)), "DeltaE_EquiformerV3+DeNS-OAM2_per_target")
            target_rows.append(
                {
                    "target": target,
                    "top5_mean_three_model_std_eV": float(top["three_model_std_eV"].mean()),
                    "top5_max_three_model_std_eV": float(top["three_model_std_eV"].max()),
                    "all_route_mean_three_model_std_eV": float(group["three_model_std_eV"].mean()),
                }
            )
        target_uncertainty = pd.DataFrame(target_rows).merge(target_metrics, on="target")
        mismatch = ~target_uncertainty["EquiformerV3_top1"].astype(bool)
        target_uncertainty.to_csv(RESULTS / "route_ensemble_uncertainty_120_targets.csv", index=False)
        summary["broad_120_shared_reactions"] = len(shared)
        summary["broad_120_route_spread_error_spearman_rho"] = float(broad_rho.statistic)
        summary["broad_120_route_spread_error_spearman_p"] = float(broad_rho.pvalue)
        summary["broad_120_top1_mismatch_targets"] = int(mismatch.sum())
        summary["top5_spread_mean_matching_targets_eV"] = float(
            target_uncertainty.loc[~mismatch, "top5_mean_three_model_std_eV"].mean()
        )
        summary["top5_spread_mean_mismatching_targets_eV"] = float(
            target_uncertainty.loc[mismatch, "top5_mean_three_model_std_eV"].mean()
        )
        summary["top5_spread_mismatch_roc_auc"] = auc(
            mismatch.to_numpy(), target_uncertainty["top5_mean_three_model_std_eV"].to_numpy()
        )
    (RESULTS / "route_ensemble_uncertainty_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
