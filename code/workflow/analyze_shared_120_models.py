#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT / "results" / "production_model_comparison"
REFERENCE_COLUMN = "DeltaE_MP_FORM_per_target"
MODEL_SPECS = {
    "PET": {
        "root": PROJECT / "results" / "mp_pet_benchmark_120",
        "column": "DeltaE_PET-OAM-XI_per_target",
    },
    "GRACE": {
        "root": PROJECT / "results" / "grace_benchmark_120",
        "column": "DeltaE_GRACE-3L-OAM-L_per_target",
    },
    "EquiformerV3": {
        "root": PROJECT / "results" / "equiformer_v3_benchmark_120",
        "column": "DeltaE_EquiformerV3+DeNS-OAM2_per_target",
    },
}


def top_overlap(frame: pd.DataFrame, left: str, right: str, n: int = 5) -> int:
    left_routes = set(frame.nsmallest(min(n, len(frame)), left)["reaction"])
    right_routes = set(frame.nsmallest(min(n, len(frame)), right)["reaction"])
    return len(left_routes & right_routes)


def bootstrap_mean_ci(values: np.ndarray, seed: int = 20260729) -> list[float]:
    rng = np.random.default_rng(seed)
    boot = np.empty(20000)
    for index in range(len(boot)):
        sample = rng.integers(0, len(values), len(values))
        boot[index] = values[sample].mean()
    return [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]


def main() -> None:
    available = {
        name: spec
        for name, spec in MODEL_SPECS.items()
        if Path(spec["root"]).exists()
    }
    if "PET" not in available:
        raise SystemExit("PET reference workflow is required.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target_rows: list[dict[str, object]] = []
    paired_rows: list[pd.DataFrame] = []

    pet_root = Path(available["PET"]["root"])
    for pet_path in sorted(pet_root.glob("*/formation_reactions.csv")):
        target = pet_path.parent.name
        pet_column = str(available["PET"]["column"])
        paired = pd.read_csv(pet_path)[["reaction", pet_column, REFERENCE_COLUMN]].copy()
        complete = True
        for model, spec in available.items():
            if model == "PET":
                continue
            reaction_path = Path(spec["root"]) / target / "formation_reactions.csv"
            if not reaction_path.exists():
                complete = False
                break
            column = str(spec["column"])
            model_frame = pd.read_csv(reaction_path)[["reaction", column]].copy()
            paired = paired.merge(model_frame, on="reaction", how="inner")
        if not complete:
            continue
        model_columns = [str(spec["column"]) for spec in available.values()]
        paired = paired.dropna(subset=model_columns + [REFERENCE_COLUMN]).copy()
        if paired.empty:
            continue
        paired["target"] = target
        for model, spec in available.items():
            column = str(spec["column"])
            paired[f"{model}_error"] = paired[column] - paired[REFERENCE_COLUMN]
        paired_rows.append(paired)

        row: dict[str, object] = {"target": target, "n_shared_reactions": len(paired)}
        for model, spec in available.items():
            column = str(spec["column"])
            error = paired[column] - paired[REFERENCE_COLUMN]
            row[f"{model}_mae"] = float(error.abs().mean())
            row[f"{model}_rmse"] = float(np.sqrt(np.mean(error**2)))
            row[f"{model}_bias"] = float(error.mean())
            row[f"{model}_sign_consistency"] = float(
                np.mean(np.sign(paired[column]) == np.sign(paired[REFERENCE_COLUMN]))
            )
            row[f"{model}_top1"] = bool(
                paired.nsmallest(1, column).iloc[0]["reaction"]
                == paired.nsmallest(1, REFERENCE_COLUMN).iloc[0]["reaction"]
            )
            row[f"{model}_top5_overlap"] = top_overlap(paired, column, REFERENCE_COLUMN)
        target_rows.append(row)

    if not target_rows:
        raise SystemExit("No complete shared target set was found.")
    targets = pd.DataFrame(target_rows).sort_values("target").reset_index(drop=True)
    all_pairs = pd.concat(paired_rows, ignore_index=True)
    targets.to_csv(OUTPUT_DIR / "shared_120_target_metrics.csv", index=False)
    all_pairs.to_csv(OUTPUT_DIR / "shared_120_reaction_details.csv", index=False)

    aggregate: dict[str, object] = {
        "n_targets": int(len(targets)),
        "n_shared_reactions": int(len(all_pairs)),
        "strict_intersection_models": list(available),
    }
    for model, spec in available.items():
        column = str(spec["column"])
        error = all_pairs[f"{model}_error"]
        aggregate[model] = {
            "macro_target_mae_eV_per_target": float(targets[f"{model}_mae"].mean()),
            "median_target_mae_eV_per_target": float(targets[f"{model}_mae"].median()),
            "micro_reaction_mae_eV_per_target": float(error.abs().mean()),
            "micro_reaction_rmse_eV_per_target": float(np.sqrt(np.mean(error**2))),
            "macro_sign_consistency": float(targets[f"{model}_sign_consistency"].mean()),
            "micro_sign_consistency": float(
                np.mean(np.sign(all_pairs[column]) == np.sign(all_pairs[REFERENCE_COLUMN]))
            ),
            "top1_agreement_count": int(targets[f"{model}_top1"].sum()),
            "mean_top5_overlap": float(targets[f"{model}_top5_overlap"].mean()),
        }

    for model in available:
        if model == "PET":
            continue
        differences = targets[f"{model}_mae"].to_numpy() - targets["PET_mae"].to_numpy()
        aggregate[f"{model}_minus_PET_target_MAE"] = {
            "mean_difference_eV_per_target": float(differences.mean()),
            "bootstrap_95pct_CI": bootstrap_mean_ci(differences),
            f"targets_{model}_lower": int(np.sum(differences < 0)),
            "targets_PET_lower": int(np.sum(differences > 0)),
            "ties": int(np.sum(differences == 0)),
        }
    (OUTPUT_DIR / "shared_120_aggregate.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
