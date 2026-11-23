#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent


def main() -> None:
    dataset = pd.read_csv(ROOT / "multi_route_literature_dataset.csv")
    targets = dataset[["target", "material_category"]].drop_duplicates().sort_values("target")
    rows = []
    for _, row in targets.iterrows():
        target = row["target"]
        phase_path = ROOT / "cases_blind" / target / "phase_stability.csv"
        phases = pd.read_csv(phase_path)
        target_rows = phases[phases["formula"].astype(str) == str(target)]
        cif_path = ""
        if len(target_rows) and "cif_path" in target_rows:
            values = target_rows["cif_path"].dropna().astype(str)
            cif_path = values.iloc[0] if len(values) else ""
        rows.append(
            {
                "target": target,
                "material_category": row["material_category"],
                "cif_path": cif_path,
                "evaluation_split": "held_out_multi_route_test",
            }
        )
    outdir = ROOT / "external_model_interface"
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(outdir / "external_model_targets.csv", index=False)
    pd.DataFrame(
        columns=["model", "target", "rank", "precursors", "byproducts", "balanced_reaction", "notes"]
    ).to_csv(outdir / "external_predictions_template.csv", index=False)


if __name__ == "__main__":
    main()
