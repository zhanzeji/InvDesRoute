#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import subprocess
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
    cached_systems,
    chemical_system,
    load_global_pet_map,
    normalize,
    run_target,
)


EXISTING_ROOTS = [
    PROJECT / "results/seed80_synthesizability_cases",
    PROJECT / "results/literature_seeded_extension_cached_53",
]


def split_formulas(value: str | None) -> set[str]:
    return {normalize(item) for item in str(value or "").split(";") if item.strip()}


def write_manual_precursors(path: Path, formulas: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["formula", "source_id", "source"])
        writer.writeheader()
        for formula in sorted(formulas):
            writer.writerow({"formula": formula, "source_id": "literature_benchmark", "source": "manual_precursor"})


def existing_case(target: str) -> Path | None:
    for root in EXISTING_ROOTS:
        target_dir = root / target
        if (target_dir / "synthesizability_ranked_routes.csv").exists():
            return target_dir
    return None


def run_external_case(target: str, output_root: Path, manual_table: Path, api_key: str, force: bool) -> Path:
    target_dir = output_root / target
    done = target_dir / "synthesizability_ranked_routes.csv"
    if done.exists() and not force:
        return target_dir
    command = [
        str(PROJECT / "run_screening.py"),
        "--target-formula",
        target,
        "--database",
        "mp",
        "--mp-api-key",
        api_key,
        "--mp-max-phases",
        "160",
        "--include-common-precursors",
        "--include-gas-reservoirs",
        "--include-virtual-common-precursors",
        "--manual-precursor-table",
        str(manual_table),
        "--compute-pet",
        "--pet-python",
        "/home/zhzhan/miniconda3/envs/upet_env/bin/python",
        "--pet-device",
        "auto",
        "--max-reactants",
        "2",
        "--max-byproducts",
        "1",
        "--byproducts-reservoir-only",
        "--max-products",
        "2",
        "--max-total-coeff",
        "100",
        "--include-targeted-precursor-routes",
        "--targeted-max-reactants",
        "3",
        "--targeted-max-byproducts",
        "4",
        "--targeted-max-total-coeff",
        "180",
        "--practical-max-atoms",
        "24",
        "--practical-max-coeff-sum-per-target",
        "100",
        "--output-dir",
        str(target_dir),
    ]
    target_dir.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(command, text=True, capture_output=True)
    (target_dir / "multi_route_run_stdout.log").write_text(process.stdout, encoding="utf-8")
    (target_dir / "multi_route_run_stderr.log").write_text(process.stderr, encoding="utf-8")
    if process.returncode != 0:
        raise RuntimeError(f"{target}: run_screening failed with return code {process.returncode}")
    return target_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare InvDesRoute outputs for the 24-target multi-route benchmark.")
    parser.add_argument("--dataset", type=Path, default=BENCHMARK_DIR / "multi_route_literature_dataset.csv")
    parser.add_argument("--output-root", type=Path, default=BENCHMARK_DIR / "cases_generated")
    parser.add_argument("--mp-api-key", default="")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    api_key = args.mp_api_key or os.environ.get("MP_API_KEY") or os.environ.get("PMG_MAPI_KEY") or ""
    dataset = pd.read_csv(args.dataset).fillna("")
    target_precursors: dict[str, set[str]] = defaultdict(set)
    target_byproducts: dict[str, set[str]] = defaultdict(set)
    for _, row in dataset.iterrows():
        target = normalize(row["target"])
        target_precursors[target].update(split_formulas(row.get("literature_precursors")))
        target_byproducts[target].update(split_formulas(row.get("literature_byproducts")))

    pet_map = load_global_pet_map(DEFAULT_PET_ROOTS)
    systems = cached_systems()
    manifest = []
    args.output_root.mkdir(parents=True, exist_ok=True)
    manual_root = BENCHMARK_DIR / "manual_precursors"
    for target in sorted(target_precursors):
        manual_table = manual_root / f"{target}.csv"
        write_manual_precursors(manual_table, target_precursors[target] | target_byproducts[target])
        case_dir = existing_case(target)
        mode = "existing"
        if case_dir is None and target in pet_map and chemical_system(target) in systems:
            mode = "cached"
            case_dir = args.output_root / target
            if args.force or not (case_dir / "synthesizability_ranked_routes.csv").exists():
                run_target(
                    target,
                    args.output_root,
                    pet_map,
                    target_precursors[target],
                    target_byproducts[target],
                    max_phases=160,
                    targeted_max_reactants=3,
                    targeted_max_byproducts=4,
                    targeted_max_total_coeff=180,
                )
        elif case_dir is None:
            if not api_key:
                raise RuntimeError(f"{target}: MP_API_KEY is required because no reusable target cache exists")
            mode = "fresh_mp_pet"
            case_dir = run_external_case(target, args.output_root, manual_table, api_key, args.force)
        manifest.append(
            {
                "target": target,
                "case_mode": mode,
                "case_dir": str(case_dir),
                "manual_precursor_table": str(manual_table),
                "n_literature_precursors": len(target_precursors[target]),
                "n_literature_byproducts": len(target_byproducts[target]),
            }
        )
        print(f"{target}: {mode} -> {case_dir}")
    pd.DataFrame(manifest).to_csv(BENCHMARK_DIR / "multi_route_case_manifest.csv", index=False)


if __name__ == "__main__":
    main()
