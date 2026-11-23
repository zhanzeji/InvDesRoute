#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = PROJECT / "multi_route_benchmark"
LITERATURE_DIR = PROJECT / "literature_benchmark"
sys.path.insert(0, str(LITERATURE_DIR))

from run_cached_literature_extension import (  # noqa: E402
    DEFAULT_PET_ROOTS,
    load_global_pet_map,
    normalize,
    run_target,
)


def split_formulas(value: str | None) -> set[str]:
    return {normalize(item) for item in str(value or "").split(";") if item.strip()}


def run_case_set(
    targets: list[str],
    precursor_map: dict[str, set[str]],
    byproduct_map: dict[str, set[str]],
    output_root: Path,
    pet_map: dict[str, float],
    force: bool,
    include_common_library: bool = True,
    include_reservoirs: bool = True,
) -> list[dict[str, object]]:
    manifest = []
    for target in targets:
        target_dir = output_root / target
        output_file = target_dir / "synthesizability_ranked_routes.csv"
        if force or not output_file.exists():
            run_target(
                target,
                output_root,
                pet_map,
                precursor_map.get(target, set()),
                byproduct_map.get(target, set()),
                max_phases=160,
                targeted_max_reactants=3,
                targeted_max_byproducts=4,
                targeted_max_total_coeff=180,
                include_common_library=include_common_library,
                include_reservoirs=include_reservoirs,
            )
        routes = pd.read_csv(output_file)
        manifest.append(
            {
                "target": target,
                "case_dir": str(target_dir),
                "n_routes": len(routes),
                "n_injected_precursors": len(precursor_map.get(target, set())),
                "n_injected_byproducts": len(byproduct_map.get(target, set())),
            }
        )
        print(f"{output_root.name}: {target} -> {len(routes)} routes")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-controlled and seeded-upper-bound multi-route cases.")
    parser.add_argument("--dataset", type=Path, default=BENCHMARK_DIR / "multi_route_literature_dataset.csv")
    parser.add_argument("--blind-root", type=Path, default=BENCHMARK_DIR / "cases_blind")
    parser.add_argument("--mp-only-root", type=Path, default=BENCHMARK_DIR / "cases_mp_only")
    parser.add_argument("--seeded-root", type=Path, default=BENCHMARK_DIR / "cases_seeded_upper_bound")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    dataset = pd.read_csv(args.dataset).fillna("")
    targets = sorted({normalize(value) for value in dataset["target"]})
    literature_precursors: dict[str, set[str]] = defaultdict(set)
    literature_byproducts: dict[str, set[str]] = defaultdict(set)
    for _, row in dataset.iterrows():
        target = normalize(row["target"])
        literature_precursors[target].update(split_formulas(row.get("literature_precursors")))
        literature_byproducts[target].update(split_formulas(row.get("literature_byproducts")))

    pet_roots = [*DEFAULT_PET_ROOTS, BENCHMARK_DIR / "cases_generated"]
    pet_map = load_global_pet_map(pet_roots)
    blind_manifest = run_case_set(targets, {}, {}, args.blind_root, pet_map, args.force)
    mp_only_manifest = run_case_set(
        targets,
        {},
        {},
        args.mp_only_root,
        pet_map,
        args.force,
        include_common_library=False,
        include_reservoirs=False,
    )
    seeded_manifest = run_case_set(
        targets,
        literature_precursors,
        literature_byproducts,
        args.seeded_root,
        pet_map,
        args.force,
    )
    pd.DataFrame(blind_manifest).to_csv(BENCHMARK_DIR / "blind_case_manifest.csv", index=False)
    pd.DataFrame(mp_only_manifest).to_csv(BENCHMARK_DIR / "mp_only_case_manifest.csv", index=False)
    pd.DataFrame(seeded_manifest).to_csv(BENCHMARK_DIR / "seeded_upper_bound_case_manifest.csv", index=False)


if __name__ == "__main__":
    main()
