#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from evaluate_2022_temporal_benchmark import (  # noqa: E402
    CorpusPrior,
    load_cases,
    prepare_features,
    rank_routes,
)


def summarize_routes(rows: list[dict[str, object]]) -> dict[str, object]:
    frame = pd.DataFrame(rows)
    energy = pd.to_numeric(frame["synth_rank_deltaE_per_target"], errors="coerce")
    mp_energy = pd.to_numeric(frame["DeltaE_MP_FORM_per_target"], errors="coerce")
    comparable = energy.notna() & mp_energy.notna()
    return {
        "displayed_routes": len(frame),
        "energy_complete": int(energy.notna().sum()),
        "exergonic_routes": int(energy.lt(-0.05).sum()),
        "near_neutral_routes": int(energy.between(-0.05, 0.25, inclusive="both").sum()),
        "uphill_routes": int(energy.gt(0.25).sum()),
        "strongly_uphill_routes": int(energy.gt(1.0).sum()),
        "median_equiformer_delta_e_eV_per_target": float(energy.median()),
        "equiformer_mp_sign_agreement": float(((energy[comparable] * mp_energy[comparable]) > 0).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantify the physical contribution of the production MLIP to displayed routes.")
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--benchmark-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    development = pd.read_csv(args.split_dir / "development_routes_2021_2022.csv").fillna("")
    future = pd.read_csv(args.split_dir / "future_routes_2023_2024.csv").fillna("")
    prior_frame = pd.read_csv(args.split_dir / "historical_prior_routes_through_2020.csv").fillna("")
    summary = json.loads(args.benchmark_summary.read_text())
    profile = summary["selected_profile"]
    alpha = float(summary["selected_historical_corpus_weight"])
    penalty = float(summary["selected_diversity_penalty"])
    targets = set(development["target"]) | set(future["target"])
    prior = CorpusPrior.fit(prior_frame, targets)
    cases = prepare_features(load_cases(args.case_root, targets), prior)

    configurations = [
        ("production_energy_only", "production", True),
        ("selected_without_energy", "no_energy", False),
        ("selected_with_mp_form_energy", "mp_form", False),
        ("selected_with_equiformer_energy", "production", False),
    ]
    summary_rows = []
    route_rows = []
    selected_sets: dict[str, dict[str, set[str]]] = {}
    for label, energy_mode, energy_only in configurations:
        selected_sets[label] = {}
        current = []
        for target in sorted(future["target"].unique()):
            selected = rank_routes(
                target,
                cases[target],
                profile,
                alpha,
                penalty,
                5,
                energy_mode=energy_mode,
                energy_only=energy_only,
            )
            selected_sets[label][target] = {str(route.get("reaction", "")) for route in selected}
            for rank, route in enumerate(selected, start=1):
                record = dict(route)
                record.update({"configuration": label, "target": target, "display_rank": rank})
                current.append(record)
                route_rows.append(record)
        metrics = summarize_routes(current)
        metrics["configuration"] = label
        current_frame = pd.DataFrame(current)
        current_frame["numeric_delta_e"] = pd.to_numeric(
            current_frame["synth_rank_deltaE_per_target"], errors="coerce"
        )
        metrics["targets_with_exergonic_top1"] = int(
            current_frame.loc[current_frame["display_rank"].eq(1), "numeric_delta_e"].lt(-0.05).sum()
        )
        metrics["targets_with_exergonic_top5"] = int(
            current_frame.groupby("target")["numeric_delta_e"].min().lt(-0.05).sum()
        )
        summary_rows.append(metrics)

    baseline = selected_sets["selected_without_energy"]
    production = selected_sets["selected_with_equiformer_energy"]
    overlaps = []
    for target in sorted(future["target"].unique()):
        intersection = len(baseline[target] & production[target])
        union = len(baseline[target] | production[target])
        overlaps.append(
            {
                "target": target,
                "shared_top5_routes": intersection,
                "top5_jaccard": intersection / union if union else 1.0,
                "energy_changes_display": baseline[target] != production[target],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(args.output_dir / "mlip_display_physics_summary.csv", index=False)
    pd.DataFrame(route_rows).to_csv(args.output_dir / "mlip_displayed_routes.csv", index=False)
    pd.DataFrame(overlaps).to_csv(args.output_dir / "mlip_route_set_changes.csv", index=False)
    output = {
        "selected_profile": profile,
        "summary": summary_rows,
        "targets_whose_top5_changes_when_energy_is_added": int(sum(row["energy_changes_display"] for row in overlaps)),
        "mean_top5_jaccard_without_vs_equiformer": float(np.mean([row["top5_jaccard"] for row in overlaps])),
        "interpretation": "Literature recovery and thermodynamic auditing are orthogonal endpoints; the MLIP contribution is evaluated through physical route evidence rather than formula imitation alone.",
    }
    (args.output_dir / "mlip_contribution_summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
