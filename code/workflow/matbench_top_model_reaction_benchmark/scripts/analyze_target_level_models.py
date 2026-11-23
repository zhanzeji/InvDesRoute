from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_common import normalize_formula, reaction_energy


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DEFAULT_TARGET_ROUTES = Path(
    "/home/zhzhan/final_paper/support_tables/model_evidence/batch_top5_dft_paths.csv"
)


def load_model_energies(path: Path) -> tuple[str, dict[str, float]]:
    frame = pd.read_csv(path)
    valid = frame.loc[frame["status"] == "ok"].copy()
    valid["formula"] = valid["formula"].map(normalize_formula)
    valid = valid.drop_duplicates("formula", keep="last")
    model = str(valid.iloc[0]["model"])
    return model, dict(zip(valid["formula"], valid["energy_per_formula_eV"].astype(float)))


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, n_bootstrap: int) -> tuple[float, float]:
    samples = rng.choice(values, size=(n_bootstrap, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def evaluate_target_routes(
    target_routes: pd.DataFrame,
    model: str,
    energy_map: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = target_routes[["target", "reaction", "DeltaE_DFT"]].copy()
    frame["DeltaE_model"] = frame["reaction"].map(lambda value: reaction_energy(value, energy_map))
    frame["error_eV"] = frame["DeltaE_model"] - frame["DeltaE_DFT"]
    rows = []
    for target, group in frame.groupby("target", sort=True):
        dft_order = list(group.sort_values("DeltaE_DFT")["reaction"])
        model_order = list(group.sort_values("DeltaE_model")["reaction"])
        rows.append(
            {
                "model": model,
                "target": target,
                "n_routes": int(len(group)),
                "target_mae_eV": float(group["error_eV"].abs().mean()),
                "target_rmse_eV": float(np.sqrt((group["error_eV"] ** 2).mean())),
                "top1_match": bool(dft_order[0] == model_order[0]),
                "top5_overlap": int(len(set(dft_order[:5]) & set(model_order[:5]))),
            }
        )
    target_frame = pd.DataFrame(rows)
    summary = {
        "model": model,
        "targets": int(len(target_frame)),
        "mean_target_mae_eV": float(target_frame["target_mae_eV"].mean()),
        "median_target_mae_eV": float(target_frame["target_mae_eV"].median()),
        "top1_match_rate": float(target_frame["top1_match"].mean()),
        "top1_match_count": int(target_frame["top1_match"].sum()),
        "mean_top5_overlap": float(target_frame["top5_overlap"].mean()),
    }
    return target_frame, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-routes", type=Path, default=DEFAULT_TARGET_ROUTES)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()

    target_routes = pd.read_csv(args.target_routes)
    energy_paths = sorted(RESULTS.glob("*_phase_energies.csv"))
    target_frames = []
    summaries = []
    for path in energy_paths:
        model, energy_map = load_model_energies(path)
        try:
            target_frame, summary = evaluate_target_routes(target_routes, model, energy_map)
        except KeyError as error:
            print(f"skip {model}: missing {error}", flush=True)
            continue
        target_frames.append(target_frame)
        summaries.append(summary)

    all_targets = pd.concat(target_frames, ignore_index=True)
    summary_frame = pd.DataFrame(summaries).sort_values("mean_target_mae_eV").reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    intervals = []
    for model, group in all_targets.groupby("model", sort=False):
        low, high = bootstrap_mean(group["target_mae_eV"].to_numpy(), rng, args.bootstrap)
        intervals.append({"model": model, "target_mae_ci95_low_eV": low, "target_mae_ci95_high_eV": high})
    summary_frame = summary_frame.merge(pd.DataFrame(intervals), on="model", how="left")

    pet_names = [name for name in all_targets["model"].unique() if name.startswith("PET-OAM-XL")]
    pairwise_rows = []
    if pet_names:
        pet = all_targets.loc[all_targets["model"] == pet_names[0], ["target", "target_mae_eV"]].rename(
            columns={"target_mae_eV": "pet_target_mae_eV"}
        )
        for model, group in all_targets.groupby("model", sort=False):
            if model == pet_names[0]:
                continue
            paired = group[["target", "target_mae_eV"]].merge(pet, on="target", how="inner")
            differences = paired["target_mae_eV"].to_numpy() - paired["pet_target_mae_eV"].to_numpy()
            low, high = bootstrap_mean(differences, rng, args.bootstrap)
            pairwise_rows.append(
                {
                    "model": model,
                    "reference_model": pet_names[0],
                    "n_targets": int(len(paired)),
                    "mean_target_mae_difference_eV": float(differences.mean()),
                    "difference_ci95_low_eV": low,
                    "difference_ci95_high_eV": high,
                    "targets_lower_mae_than_pet": int((differences < 0).sum()),
                }
            )

    all_targets.to_csv(RESULTS / "current_model_target_level_details.csv", index=False)
    summary_frame.to_csv(RESULTS / "current_model_target_level_summary.csv", index=False)
    pd.DataFrame(pairwise_rows).to_csv(RESULTS / "current_model_target_level_paired_vs_pet.csv", index=False)
    protocol = {
        "target_route_source": str(args.target_routes),
        "n_targets": int(target_routes["target"].nunique()),
        "n_routes": int(len(target_routes)),
        "route_scope": "DFT top-5 routes retained for each target",
        "bootstrap_unit": "target",
        "bootstrap_replicates": args.bootstrap,
        "random_seed": args.seed,
    }
    (RESULTS / "current_model_target_level_protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    print(summary_frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
