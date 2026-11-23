#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODEL = "EquiformerV3-DeNS-OMat24-MPtrj-sAlex"


def chemistry_family(formula: str) -> str:
    elements = {str(element) for element in Composition(formula).elements}
    if elements & {"Cu", "Pd", "Pt"}:
        return "late-transition-metal hydrides"
    if elements & {"Al", "Ga"}:
        return "alanate/gallate hydrides"
    if elements & {"Mg", "Ca"}:
        return "alkaline-earth-containing hydrides"
    return "other complex hydrides"


def main() -> None:
    target_path = RESULTS / "current_model_target_level_details.csv"
    reaction_path = RESULTS / f"{MODEL}_reaction_comparison.csv"
    targets = pd.read_csv(target_path)
    targets = targets.loc[targets["model"].eq(MODEL)].copy()
    targets["chemistry_family"] = targets["target"].map(chemistry_family)
    family = (
        targets.groupby("chemistry_family", as_index=False)
        .agg(
            targets=("target", "nunique"),
            mean_target_mae_eV=("target_mae_eV", "mean"),
            median_target_mae_eV=("target_mae_eV", "median"),
            top1_agreement=("top1_match", "mean"),
            mean_top5_overlap=("top5_overlap", "mean"),
        )
        .sort_values("mean_target_mae_eV")
    )

    reactions = pd.read_csv(reaction_path)
    absolute_reference = reactions["mp_dft_delta_e_eV"].abs()
    reactions["driving_force_bin"] = pd.cut(
        absolute_reference,
        bins=[-np.inf, 0.10, 0.50, 1.00, np.inf],
        labels=["|DFT deltaE| <= 0.10", "0.10 < |DFT deltaE| <= 0.50", "0.50 < |DFT deltaE| <= 1.00", "|DFT deltaE| > 1.00"],
    )
    bins = (
        reactions.groupby("driving_force_bin", observed=True, as_index=False)
        .agg(
            reactions=("reaction", "size"),
            mae_eV=("absolute_error_eV", "mean"),
            median_ae_eV=("absolute_error_eV", "median"),
            sign_consistency=("same_sign", "mean"),
        )
    )

    family.to_csv(RESULTS / "equiformer_chemistry_family_metrics.csv", index=False)
    bins.to_csv(RESULTS / "equiformer_driving_force_bin_metrics.csv", index=False)
    summary = {
        "model": MODEL,
        "targets": int(targets["target"].nunique()),
        "reactions": int(reactions.shape[0]),
        "chemistry_families": family.to_dict("records"),
        "driving_force_bins": bins.to_dict("records"),
        "interpretation": "Family stratification and near-zero driving-force bins test where the aggregate reaction MAE and sign metric remain reliable.",
    }
    (RESULTS / "equiformer_boundary_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
