#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from pymatgen.core import Composition

DEFAULT_CASE_DIR = PROJECT / 'results/precursor_expanded_targeted_final'
DEFAULT_OUT = PROJECT / 'literature_benchmark/results/targeted_literature_route_ranking_v2.csv'


def normalize(formula: str) -> str:
    return Composition(formula).reduced_formula


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def parse_side(value: str) -> set[str]:
    formulas = set()
    if not value:
        return formulas
    for item in value.split(' | '):
        if '*' not in item:
            continue
        _, formula = item.split('*', 1)
        formulas.add(normalize(formula))
    return formulas


def rank_literature_hits(case_dir: Path, reactants: set[str], byproducts: set[str]) -> tuple[list[dict[str, str]], int]:
    path = case_dir / 'formation_reactions.csv'
    if not path.exists():
        return [], 0
    rows = read_csv(path)
    ranked = [row for row in rows if row.get('DeltaE_MP_FORM_per_target') not in ('', None)]
    ranked.sort(key=lambda row: float(row['DeltaE_MP_FORM_per_target']))
    rank_by_reaction = {row['reaction']: i for i, row in enumerate(ranked, 1)}
    hits = []
    for row in rows:
        row_reactants = parse_side(row.get('reactants', ''))
        row_products = parse_side(row.get('products', ''))
        if reactants.issubset(row_reactants) and byproducts.issubset(row_products):
            hit = dict(row)
            hit['mp_form_rank'] = str(rank_by_reaction.get(row['reaction'], 'unranked'))
            hit['n_ranked_routes'] = str(len(ranked))
            hits.append(hit)
    return hits, len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--case-dir', type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    cases = [
        ('BaTiO3', {'BaCO3', 'TiO2'}, {'CO2'}, 'carbonate + oxide ceramic route'),
        ('SrTiO3', {'SrCO3', 'TiO2'}, {'CO2'}, 'carbonate + oxide ceramic route'),
        ('CaTiO3', {'CaCO3', 'TiO2'}, {'CO2'}, 'carbonate + oxide ceramic route'),
        ('GaN', {'Ga', 'NH3'}, {'H2'}, 'ammonolysis/nitridation route'),
        ('InN', {'In', 'NH3'}, {'H2'}, 'ammonolysis/nitridation route'),
        ('CaCO3', {'CaO', 'CO2'}, set(), 'carbonation route'),
        ('LiCoO2', {'Li2CO3', 'Co3O4', 'O2'}, {'CO2'}, 'carbonate cathode route'),
        ('LiMn2O4', {'Li2CO3', 'MnO2'}, {'CO2', 'O2'}, 'carbonate cathode route'),
        ('LiFePO4', {'Li2CO3', 'FeC2O4', 'NH4H2PO4'}, {'CO2', 'CO', 'H2O', 'NH3'}, 'virtual oxalate/ammonium-phosphate route'),
    ]
    out_rows = []
    for target, reactants_raw, byproducts_raw, route_class in cases:
        case_dir = args.case_dir / normalize(target)
        phase_sources = {}
        phase_path = case_dir / 'phase_stability.csv'
        if phase_path.exists():
            for row in read_csv(phase_path):
                phase_sources[normalize(row['formula'])] = row.get('source', '')
        reactants = {normalize(formula) for formula in reactants_raw}
        byproducts = {normalize(formula) for formula in byproducts_raw}
        hits, total = rank_literature_hits(case_dir, reactants, byproducts)
        ranked_hits = [hit for hit in hits if hit['mp_form_rank'] != 'unranked']
        if ranked_hits:
            best = sorted(ranked_hits, key=lambda row: int(row['mp_form_rank']))[0]
        elif hits:
            best = hits[0]
        else:
            best = None
        species = sorted(reactants | byproducts)
        missing = [formula for formula in species if formula not in phase_sources]
        has_virtual = any(phase_sources.get(formula) == 'virtual_common_precursor' for formula in species)
        has_gas = bool(byproducts | ({formula for formula in reactants if formula in {normalize(x) for x in ['O2', 'N2', 'H2', 'NH3', 'CO2', 'CO', 'H2O']}}))
        out_rows.append({
            'target': normalize(target),
            'route_class': route_class,
            'literature_reactants_normalized': ';'.join(sorted(reactants)),
            'literature_byproducts_normalized': ';'.join(sorted(byproducts)),
            'key_species_sources': ';'.join(f'{formula}:{phase_sources.get(formula, "missing")}' for formula in species),
            'missing_key_species': ';'.join(missing),
            'literature_type_route_found': bool(hits),
            'n_literature_type_hits': len(hits),
            'n_formation_reactions': total,
            'best_literature_type_reaction': best.get('reaction', '') if best else '',
            'best_literature_type_mp_form_rank': best.get('mp_form_rank', '') if best else '',
            'n_ranked_routes': best.get('n_ranked_routes', '') if best else '',
            'best_literature_type_mp_form_deltaE_per_target': best.get('DeltaE_MP_FORM_per_target', '') if best else '',
            'best_literature_type_practical': best.get('practical_route', '') if best else '',
            'contains_gas_reservoir': has_gas,
            'contains_virtual_precursor': has_virtual,
            'note': 'energy unranked because at least one virtual precursor lacks formation energy' if has_virtual and (not best or best.get('mp_form_rank') == 'unranked') else 'gas/reservoir route; static MP_FORM ranking is qualitative' if has_gas else 'solid-state route with MP_FORM ranking available',
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)
    print(args.output)
    for row in out_rows:
        print(row['target'], row['literature_type_route_found'], row['n_literature_type_hits'], row['best_literature_type_mp_form_rank'], row['best_literature_type_mp_form_deltaE_per_target'])


if __name__ == '__main__':
    main()
