#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from pymatgen.core import Composition

DEFAULT_CASE_DIR = PROJECT / 'results/precursor_expanded_targeted_final'
DEFAULT_OUT = PROJECT / 'literature_benchmark/results/literature_synth_rank_eval.csv'


def normalize(formula: str) -> str:
    return Composition(formula).reduced_formula


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def parse_side(value: str) -> set[str]:
    formulas = set()
    for item in str(value).split(' | '):
        if '*' in item:
            formulas.add(normalize(item.split('*', 1)[1]))
    return formulas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--case-dir', type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    cases = [
        ('BaTiO3', {'BaCO3', 'TiO2'}, {'CO2'}, 'carbonate ceramic'),
        ('SrTiO3', {'SrCO3', 'TiO2'}, {'CO2'}, 'carbonate ceramic'),
        ('CaTiO3', {'CaCO3', 'TiO2'}, {'CO2'}, 'carbonate ceramic'),
        ('GaN', {'Ga', 'NH3'}, {'H2'}, 'ammonolysis'),
        ('InN', {'In', 'NH3'}, {'H2'}, 'ammonolysis'),
        ('CaCO3', {'CaO', 'CO2'}, set(), 'carbonation'),
        ('LiCoO2', {'Li2CO3', 'Co3O4', 'O2'}, {'CO2'}, 'battery carbonate'),
        ('LiMn2O4', {'Li2CO3', 'MnO2'}, {'CO2', 'O2'}, 'battery carbonate'),
        ('LiFePO4', {'Li2CO3', 'FeC2O4', 'NH4H2PO4'}, {'CO2', 'CO', 'H2O', 'NH3'}, 'phosphate oxalate'),
    ]
    rows = []
    for target, reactants_raw, byproducts_raw, route_class in cases:
        target = normalize(target)
        reactants = {normalize(formula) for formula in reactants_raw}
        byproducts = {normalize(formula) for formula in byproducts_raw}
        path = args.case_dir / target / 'synthesizability_ranked_routes.csv'
        all_rows = read_csv(path) if path.exists() else []
        hits = []
        for row in all_rows:
            if reactants.issubset(parse_side(row.get('reactants', ''))) and byproducts.issubset(parse_side(row.get('products', ''))):
                hits.append(row)
        top = hits[0] if hits else {}
        rows.append(
            {
                'target': target,
                'route_class': route_class,
                'found': bool(hits),
                'synth_rank': top.get('synth_rank', ''),
                'synth_score': top.get('synth_score', ''),
                'reaction': top.get('reaction', ''),
                'deltaE_per_target': top.get('synth_rank_deltaE_per_target', ''),
                'energy_source': top.get('synth_rank_energy_source', ''),
                'template_reason': top.get('route_template_reason', ''),
                'risk_flags': top.get('synthesis_risk_flags', ''),
                'n_routes': len(all_rows),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row['target'], row['found'], row['synth_rank'], row['synth_score'], row['reaction'])
    print(args.output)


if __name__ == '__main__':
    main()
