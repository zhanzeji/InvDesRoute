#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def top_overlap(rows: pd.DataFrame, model_column: str, k: int = 5) -> int:
    reference = set(rows.nsmallest(k, "DeltaE_MP_FORM_per_target")["reaction"])
    predicted = set(rows.nsmallest(k, model_column)["reaction"])
    return len(reference & predicted)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a production model on a frozen MP-comparable reaction set."
    )
    parser.add_argument("--frozen-routes", type=Path, required=True)
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-column",
        default="DeltaE_PRODUCTION_FORM_ALIGNED_per_target",
    )
    parser.add_argument("--model-name", default="production_model")
    args = parser.parse_args()

    frozen = pd.read_csv(args.frozen_routes)
    frozen = frozen[["target", "reaction", "DeltaE_MP_FORM_per_target"]].drop_duplicates()
    collected: list[pd.DataFrame] = []
    missing: list[dict[str, str]] = []
    for target, expected in frozen.groupby("target", sort=False):
        path = args.case_root / str(target) / "formation_reactions.csv"
        if not path.exists():
            missing.extend(
                {"target": str(target), "reaction": str(reaction), "reason": "missing_target_file"}
                for reaction in expected["reaction"]
            )
            continue
        current = pd.read_csv(path)
        if args.model_column not in current.columns:
            raise KeyError(f"{args.model_column} is absent from {path}")
        merged = expected.merge(
            current[["reaction", args.model_column]].drop_duplicates("reaction"),
            on="reaction",
            how="left",
        )
        unavailable = merged[args.model_column].isna()
        missing.extend(
            {
                "target": str(target),
                "reaction": str(reaction),
                "reason": "missing_model_energy_or_reaction",
            }
            for reaction in merged.loc[unavailable, "reaction"]
        )
        collected.append(merged.loc[~unavailable].copy())

    routes = pd.concat(collected, ignore_index=True) if collected else pd.DataFrame()
    routes["error"] = routes[args.model_column] - routes["DeltaE_MP_FORM_per_target"]
    target_rows = []
    for target, group in routes.groupby("target", sort=False):
        reference_best = group.nsmallest(1, "DeltaE_MP_FORM_per_target").iloc[0]
        model_best = group.nsmallest(1, args.model_column).iloc[0]
        target_rows.append(
            {
                "target": target,
                "n_frozen_routes": len(frozen[frozen["target"] == target]),
                "n_comparable_routes": len(group),
                "mae_eV_per_target": float(group["error"].abs().mean()),
                "rmse_eV_per_target": float(np.sqrt(np.mean(group["error"] ** 2))),
                "bias_eV_per_target": float(group["error"].mean()),
                "sign_consistency": float(
                    np.mean(
                        np.sign(group[args.model_column])
                        == np.sign(group["DeltaE_MP_FORM_per_target"])
                    )
                ),
                "top1_agreement": bool(reference_best["reaction"] == model_best["reaction"]),
                "top5_overlap": top_overlap(group, args.model_column),
            }
        )
    target_metrics = pd.DataFrame(target_rows)
    aggregate = {
        "model": args.model_name,
        "n_targets_expected": int(frozen["target"].nunique()),
        "n_targets_comparable": int(routes["target"].nunique()),
        "n_frozen_routes": int(len(frozen)),
        "n_comparable_routes": int(len(routes)),
        "mean_target_mae_eV_per_target": float(target_metrics["mae_eV_per_target"].mean()),
        "median_target_mae_eV_per_target": float(target_metrics["mae_eV_per_target"].median()),
        "micro_route_mae_eV_per_target": float(routes["error"].abs().mean()),
        "micro_route_rmse_eV_per_target": float(np.sqrt(np.mean(routes["error"] ** 2))),
        "mean_target_sign_consistency": float(target_metrics["sign_consistency"].mean()),
        "micro_route_sign_consistency": float(
            np.mean(
                np.sign(routes[args.model_column])
                == np.sign(routes["DeltaE_MP_FORM_per_target"])
            )
        ),
        "top1_agreement_count": int(target_metrics["top1_agreement"].sum()),
        "mean_top5_overlap": float(target_metrics["top5_overlap"].mean()),
        "missing_route_count": len(missing),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    routes.to_csv(args.output_dir / "frozen_route_details.csv", index=False)
    target_metrics.to_csv(args.output_dir / "frozen_target_metrics.csv", index=False)
    pd.DataFrame(missing).to_csv(args.output_dir / "missing_routes.csv", index=False)
    (args.output_dir / "frozen_route_aggregate.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
