#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the pre-ranking production-energy completeness gate.")
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--production-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    production = pd.read_csv(args.production_summary)
    eligible_targets = set(
        production.loc[production["n_routes_with_production_energy"].gt(0), "target"]
    )
    manifest = pd.read_csv(args.split_dir / "target_manifest.csv")
    manifest = manifest.merge(
        production[["target", "n_routes", "n_routes_with_production_energy", "n_phases_model_aligned"]],
        on="target",
        how="left",
    )
    manifest["energy_complete"] = manifest["target"].isin(eligible_targets)
    exclusions = manifest.loc[~manifest["energy_complete"]].copy()
    exclusions["exclusion_reason"] = "no formation route with complete EquiformerV3 production energy"
    retained_manifest = manifest.loc[manifest["energy_complete"]].copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename in (
        "historical_prior_routes_through_2020.csv",
        "all_auditable_routes_with_year.csv",
    ):
        pd.read_csv(args.split_dir / filename).to_csv(args.output_dir / filename, index=False)
    development = pd.read_csv(args.split_dir / "development_routes_2021_2022.csv")
    future = pd.read_csv(args.split_dir / "future_routes_2023_2024.csv")
    development = development.loc[development["target"].isin(eligible_targets)].copy()
    future = future.loc[future["target"].isin(eligible_targets)].copy()
    development.to_csv(args.output_dir / "development_routes_2021_2022.csv", index=False)
    future.to_csv(args.output_dir / "future_routes_2023_2024.csv", index=False)
    retained_manifest.to_csv(args.output_dir / "target_manifest.csv", index=False)
    exclusions.to_csv(args.output_dir / "energy_completeness_exclusions.csv", index=False)

    source_summary = json.loads((args.split_dir / "split_summary.json").read_text())
    source_summary.update(
        {
            "development_records": len(development),
            "development_targets": int(development["target"].nunique()),
            "future_records": len(future),
            "future_targets": int(future["target"].nunique()),
            "energy_incomplete_exclusions": int(len(exclusions)),
            "energy_incomplete_targets": exclusions["target"].tolist(),
            "energy_gate_timing": "Applied before ranking-parameter selection and future-route evaluation.",
        }
    )
    (args.output_dir / "split_summary.json").write_text(json.dumps(source_summary, indent=2), encoding="utf-8")
    print(json.dumps(source_summary, indent=2))


if __name__ == "__main__":
    main()
