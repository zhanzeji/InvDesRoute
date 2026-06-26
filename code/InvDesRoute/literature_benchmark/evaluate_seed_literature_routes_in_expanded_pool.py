#!/usr/bin/env python3
"""Evaluate seed literature routes inside precursor-expanded phase pools.

This script checks whether literature precursor/byproduct sets can be balanced
with the target and whether the required species exist in a generated
`phase_stability.csv`. It is a targeted complement to exhaustive reaction
enumeration, which can become expensive after adding experimental precursors.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analysis import mp_formation_energy_map  # noqa: E402
from common import BalancedReaction, normalize_formula  # noqa: E402
from reactions import balance_species, reaction_energy, target_coefficient  # noqa: E402
from compare_literature_routes import split_formulas  # noqa: E402


DEFAULT_LIT = PROJECT / "literature_benchmark/literature_routes_seed.csv"
DEFAULT_CASE_DIR = PROJECT / "results/precursor_expanded_cases_fast"
DEFAULT_OUT = PROJECT / "literature_benchmark/results/expanded_pool_seed_literature_route_eval.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_phase_rows(path: Path) -> list[dict[str, str]]:
    rows = read_csv(path)
    for row in rows:
        row["formula"] = normalize_formula(row["formula"])
    return rows


def normalized_formula_set(value: str | None) -> set[str]:
    out = set()
    for item in split_formulas(value):
        try:
            out.add(normalize_formula(item))
        except Exception:
            out.add(item)
    return out


def mp_form_energy_map_from_rows(rows: list[dict[str, str]]) -> dict[str, float]:
    out = {}
    for row in rows:
        try:
            fe = float(row["formation_energy_per_atom"])
            natoms = float(row["n_atoms"])
        except Exception:
            continue
        out[normalize_formula(row["formula"])] = fe * natoms
    return out


def build_reaction(target: str, precursors: set[str], byproducts: set[str]) -> tuple[BalancedReaction | None, str]:
    target = normalize_formula(target)
    reactants = tuple(sorted(normalize_formula(item) for item in precursors))
    products = tuple(sorted({target, *(normalize_formula(item) for item in byproducts)}))
    balanced = balance_species(reactants, products)
    if balanced is None:
        return None, "not_balanceable"
    reactant_coeffs, product_coeffs = balanced
    return BalancedReaction(tuple(zip(reactants, reactant_coeffs)), tuple(zip(products, product_coeffs))), "balanced"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--literature", type=Path, default=DEFAULT_LIT)
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    rows = []
    for lit in read_csv(args.literature):
        target = normalize_formula(lit["target"])
        phase_path = args.case_dir / target / "phase_stability.csv"
        precursors = normalized_formula_set(lit.get("literature_precursors"))
        byproducts = normalized_formula_set(lit.get("literature_byproducts"))
        out = {
            "target": target,
            "literature_precursors": ";".join(sorted(precursors)),
            "literature_byproducts": ";".join(sorted(byproducts)),
            "literature_route_text": lit.get("literature_route_text", ""),
            "case_available": phase_path.exists(),
            "all_species_in_expanded_pool": False,
            "missing_species": "",
            "balanced_reaction": "",
            "balance_status": "",
            "mp_form_deltaE_per_target": "",
            "route_energy_note": "",
        }
        if not phase_path.exists():
            out["balance_status"] = "case_not_run"
            rows.append(out)
            continue
        phases = load_phase_rows(phase_path)
        formulas = {row["formula"] for row in phases}
        needed = {target, *precursors, *byproducts}
        missing = sorted(needed - formulas)
        out["all_species_in_expanded_pool"] = not missing
        out["missing_species"] = ";".join(missing)
        reaction, status = build_reaction(target, precursors, byproducts)
        out["balance_status"] = status
        if reaction is not None:
            out["balanced_reaction"] = reaction.reaction
            energy_map = mp_form_energy_map_from_rows(phases)
            energy = reaction_energy(reaction, energy_map)
            coeff = target_coefficient(reaction, target, "formation")
            if energy is not None and coeff:
                out["mp_form_deltaE_per_target"] = energy / coeff
                if byproducts & {"CO2", "H2O", "H2", "O2", "N2", "H3N"}:
                    out["route_energy_note"] = "contains gas/reservoir species; 0 K formation-energy ranking is qualitative"
        rows.append(out)

    fieldnames = [
        "target",
        "literature_precursors",
        "literature_byproducts",
        "literature_route_text",
        "case_available",
        "all_species_in_expanded_pool",
        "missing_species",
        "balanced_reaction",
        "balance_status",
        "mp_form_deltaE_per_target",
        "route_energy_note",
    ]
    write_csv(args.output, rows, fieldnames)
    available = [row for row in rows if row["case_available"]]
    present = [row for row in available if row["all_species_in_expanded_pool"]]
    balanced = [row for row in present if row["balance_status"] == "balanced"]
    print(f"cases={len(available)}")
    print(f"all_species_present={len(present)}/{len(available) if available else 0}")
    print(f"balanced={len(balanced)}/{len(available) if available else 0}")
    print(args.output)


if __name__ == "__main__":
    main()
