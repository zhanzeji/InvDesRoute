#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from pymatgen.core import Composition


PROJECT = Path(__file__).resolve().parents[2]
OLD_TEMPORAL = PROJECT / "multi_route_benchmark" / "temporal_validation" / "split_2012"


def material_category(formula: str) -> str:
    elements = {str(element) for element in Composition(formula).elements}
    if elements & {"F", "Cl", "Br", "I"}:
        return "halide"
    if "O" in elements:
        return "oxide_or_polyanion"
    if elements & {"S", "Se", "Te"}:
        return "chalcogenide"
    if elements & {"C", "N", "B"}:
        return "carbide_nitride_boride"
    return "intermetallic_or_elemental"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a 2022-cutoff literature-knowledge split.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cutoff-year", type=int, default=2022)
    args = parser.parse_args()

    routes = pd.read_csv(args.data_dir / "lee2025_valid_balanced_routes.csv", low_memory=False).fillna("")
    metadata = pd.read_csv(args.data_dir / "publication_metadata.csv", low_memory=False)
    metadata["doi"] = metadata["doi"].astype(str).str.lower()
    year_map = dict(zip(metadata["doi"], metadata["publication_year"]))
    title_map = dict(zip(metadata["doi"], metadata["title"]))
    routes["publication_year"] = routes["doi"].astype(str).str.lower().map(year_map)
    routes["source_title"] = routes["doi"].astype(str).str.lower().map(title_map)
    dated = routes.loc[routes["publication_year"].notna()].copy()
    dated["publication_year"] = dated["publication_year"].astype(int)
    first_year = dated.groupby("target")["publication_year"].min().rename("target_first_year")
    dated["target_first_year"] = dated["target"].map(first_year).astype(int)
    dated["material_category"] = dated["target"].map(material_category)

    eligible = dated.loc[
        dated["target_mp_id"].astype(str).ne("")
        & dated["phase_pure_report"].astype(str).str.lower().isin({"true", "1"})
        & dated["n_target_elements"].between(2, 4)
        & dated["n_precursors"].between(1, 4)
        & dated["n_byproducts"].le(3)
    ].copy()
    future = eligible.loc[eligible["target_first_year"] > args.cutoff_year].copy()
    future = (
        future.sort_values(["target", "publication_year", "doi", "route_id"])
        .drop_duplicates("target", keep="first")
        .reset_index(drop=True)
    )

    development = pd.read_csv(OLD_TEMPORAL / "historical_development_routes.csv").fillna("")
    development["material_category"] = development["target"].map(material_category)
    future_targets = set(future["target"])
    development = development.loc[~development["target"].isin(future_targets)].copy()
    development_targets = set(development["target"])

    prior = eligible.loc[
        (eligible["publication_year"] <= args.cutoff_year)
        & ~eligible["target"].isin(future_targets | development_targets)
    ].copy()
    if development_targets & future_targets:
        raise RuntimeError("Development/future target overlap")
    if (future["target_first_year"] <= args.cutoff_year).any():
        raise RuntimeError("Future target has a pre-cutoff record")
    if (prior["publication_year"] > args.cutoff_year).any():
        raise RuntimeError("Post-cutoff record leaked into prior")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dated.to_csv(args.output_dir / "lee2025_routes_with_year.csv", index=False)
    eligible.to_csv(args.output_dir / "eligible_routes_with_year.csv", index=False)
    prior.to_csv(args.output_dir / "pre2023_prior_routes.csv", index=False)
    development.to_csv(args.output_dir / "parameter_development_routes.csv", index=False)
    future.to_csv(args.output_dir / "post2022_future_test_routes.csv", index=False)
    future[["target", "target_mp_id", "material_category", "publication_year", "doi"]].to_csv(
        args.output_dir / "future_target_manifest.csv", index=False
    )

    summary = {
        "source_dataset": "Lee et al. 2025 solid-state synthesis reactions with impurity phases",
        "source_records": 80806,
        "independently_valid_balanced_routes": len(routes),
        "dated_valid_routes": len(dated),
        "eligible_phase_pure_routes": len(eligible),
        "cutoff_year": args.cutoff_year,
        "prior_routes": len(prior),
        "prior_targets": int(prior["target"].nunique()),
        "parameter_development_routes": len(development),
        "parameter_development_targets": int(development["target"].nunique()),
        "future_test_routes": len(future),
        "future_test_targets": int(future["target"].nunique()),
        "future_test_years": sorted(int(value) for value in future["publication_year"].unique()),
        "future_category_counts": future["material_category"].value_counts().to_dict(),
        "development_future_target_overlap": 0,
        "time_boundary": "Literature route prior and parameter labels only; Materials Project and MLIP training are not time frozen.",
    }
    (args.output_dir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
