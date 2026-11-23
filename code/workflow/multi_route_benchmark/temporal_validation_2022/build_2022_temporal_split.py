#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition


def material_category(formula: str) -> str:
    elements = {str(element) for element in Composition(formula).elements}
    if "O" in elements:
        return "oxide_or_oxyanion"
    if elements & {"F", "Cl", "Br", "I"}:
        return "halide"
    if elements & {"S", "Se", "Te"}:
        return "chalcogenide"
    if elements & {"N", "C", "B", "P", "As", "Sb"}:
        return "pnictide_carbide_boride"
    return "intermetallic_other"


def add_route_ids(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    frame = frame.sort_values(["target", "publication_year", "doi", "literature_route_text"]).copy()
    frame["route_index"] = frame.groupby("target").cumcount() + 1
    frame["route_id"] = [
        f"{target}__{prefix}{index:02d}" for target, index in zip(frame["target"], frame["route_index"])
    ]
    return frame.drop(columns="route_index")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a 2022-cutoff target-disjoint synthesis-route split.")
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--legacy-routes", type=Path, help="Earlier route corpus used to exclude pre-cutoff target appearances")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cutoff-year", type=int, default=2022)
    parser.add_argument("--prior-end-year", type=int, default=2020)
    parser.add_argument("--seed", type=int, default=20220912)
    args = parser.parse_args()

    routes = pd.read_csv(args.routes).fillna("")
    metadata = pd.read_csv(args.metadata)
    year_map = dict(zip(metadata["doi"].str.lower(), metadata["publication_year"]))
    routes["publication_year"] = routes["doi"].str.lower().map(year_map)
    routes = routes.loc[routes["publication_year"].notna()].copy()
    routes["publication_year"] = routes["publication_year"].astype(int)
    first_year = routes.groupby("target")["publication_year"].min()
    routes["target_first_publication_year"] = routes["target"].map(first_year).astype(int)
    routes["n_elements"] = routes["target"].map(lambda value: len(Composition(value).elements))
    routes["n_target_atoms"] = routes["target"].map(lambda value: float(Composition(value).num_atoms))
    routes["material_category"] = routes["target"].map(material_category)

    eligible = routes.loc[
        routes["target_mp_id"].astype(str).str.len().gt(0)
        & routes["n_elements"].le(4)
        & routes["n_target_atoms"].le(50)
        & routes["n_precursors"].le(4)
    ].copy()

    legacy_targets: set[str] = set()
    if args.legacy_routes:
        legacy_targets = set(pd.read_csv(args.legacy_routes, usecols=["target"])["target"].astype(str))
    future_before_legacy_check = eligible.loc[
        eligible["target_first_publication_year"].gt(args.cutoff_year)
    ].copy()
    future = future_before_legacy_check.loc[
        ~future_before_legacy_check["target"].isin(legacy_targets)
    ].copy()
    future_targets = sorted(future["target"].unique())
    future_manifest = future.drop_duplicates("target")[["target", "material_category", "n_elements"]]

    development_pool = eligible.loc[
        eligible["target_first_publication_year"].between(args.prior_end_year + 1, args.cutoff_year)
    ].drop_duplicates("target").copy()
    rng = np.random.default_rng(args.seed)
    development_pool["random_order"] = rng.random(len(development_pool))

    selected_targets: list[str] = []
    used: set[str] = set()
    target_counts = future_manifest.groupby(["material_category", "n_elements"]).size()
    for (category, n_elements), count in target_counts.items():
        candidates = development_pool.loc[
            development_pool["material_category"].eq(category)
            & development_pool["n_elements"].eq(n_elements)
            & ~development_pool["target"].isin(used)
        ].sort_values("random_order")
        chosen = candidates.head(int(count))["target"].tolist()
        selected_targets.extend(chosen)
        used.update(chosen)

    missing = len(future_targets) - len(selected_targets)
    if missing:
        remainder = development_pool.loc[~development_pool["target"].isin(used)].copy()
        future_category_counts = future_manifest["material_category"].value_counts().to_dict()
        remainder["category_priority"] = remainder["material_category"].map(
            lambda value: -future_category_counts.get(value, 0)
        )
        chosen = remainder.sort_values(["category_priority", "random_order"]).head(missing)["target"].tolist()
        selected_targets.extend(chosen)
        used.update(chosen)

    if len(selected_targets) != len(future_targets):
        raise RuntimeError("Unable to construct a size-matched development target set")

    development = eligible.loc[
        eligible["target"].isin(selected_targets)
        & eligible["publication_year"].le(args.cutoff_year)
    ].copy()
    development = development.sort_values(["target", "publication_year", "doi"]).groupby("target").head(3)
    future = future.sort_values(["target", "publication_year", "doi"]).groupby("target").head(3)
    development = add_route_ids(development, "D")
    future = add_route_ids(future, "F")

    heldout_targets = set(selected_targets) | set(future_targets)
    prior = routes.loc[
        routes["publication_year"].le(args.prior_end_year)
        & ~routes["target"].isin(heldout_targets)
    ].copy()

    if set(development["target"]) & set(future["target"]):
        raise RuntimeError("Development/future target overlap")
    if (development["publication_year"] > args.cutoff_year).any():
        raise RuntimeError("Post-cutoff route leaked into development")
    if (future["target_first_publication_year"] <= args.cutoff_year).any():
        raise RuntimeError("Future target has a pre-cutoff record")
    if set(prior["target"]) & heldout_targets:
        raise RuntimeError("Held-out target leaked into historical prior")

    manifests = []
    for split_name, frame in (("development_2021_2022", development), ("future_2023_2024", future)):
        manifest = frame.groupby("target", as_index=False).agg(
            material_category=("material_category", "first"),
            n_elements=("n_elements", "first"),
            target_mp_id=("target_mp_id", "first"),
            target_first_publication_year=("target_first_publication_year", "first"),
            route_records=("route_id", "size"),
        )
        manifest["split"] = split_name
        manifests.append(manifest)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    routes.to_csv(args.output_dir / "all_auditable_routes_with_year.csv", index=False)
    prior.to_csv(args.output_dir / "historical_prior_routes_through_2020.csv", index=False)
    development.to_csv(args.output_dir / "development_routes_2021_2022.csv", index=False)
    future.to_csv(args.output_dir / "future_routes_2023_2024.csv", index=False)
    pd.concat(manifests, ignore_index=True).to_csv(args.output_dir / "target_manifest.csv", index=False)
    pd.DataFrame(
        {
            "split": ["historical_prior", "development", "future_test"],
            "year_definition": ["publication <= 2020", "target first appears in 2021-2022", "target first appears after 2022"],
            "route_records": [len(prior), len(development), len(future)],
            "unique_targets": [prior["target"].nunique(), development["target"].nunique(), future["target"].nunique()],
            "unique_dois": [prior["doi"].nunique(), development["doi"].nunique(), future["doi"].nunique()],
        }
    ).to_csv(args.output_dir / "split_counts.csv", index=False)

    summary = {
        "source_dataset": "Lee et al., Scientific Data 12, 1969 (2025)",
        "source_doi": "10.1038/s41597-025-06222-y",
        "source_data_doi": "10.6084/m9.figshare.30423274",
        "resolved_route_records": len(routes),
        "cutoff_year": args.cutoff_year,
        "historical_prior_end_year": args.prior_end_year,
        "historical_prior_records": len(prior),
        "historical_prior_targets": int(prior["target"].nunique()),
        "development_records": len(development),
        "development_targets": int(development["target"].nunique()),
        "future_records": len(future),
        "future_targets": int(future["target"].nunique()),
        "development_future_target_overlap": 0,
        "future_target_pre_cutoff_records": 0,
        "future_targets_excluded_by_legacy_corpus": int(
            future_before_legacy_check["target"].nunique() - future["target"].nunique()
        ),
        "selection": "All eligible post-2022 targets with MP identifiers; size- and chemistry-stratified development sample from targets first appearing in 2021-2022.",
        "knowledge_boundary": "Literature prior and parameter labels are time frozen; current MP phase data and EquiformerV3 pretraining are not time frozen.",
    }
    (args.output_dir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
