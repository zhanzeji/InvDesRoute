#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from pymatgen.core import Composition

DEFAULT_CASE_DIR = Path('/home/zhzhan/synthesis_pathway_screening/results/precursor_expanded_targeted_final')
DEFAULT_OUT = Path('/home/zhzhan/synthesis_pathway_screening/literature_benchmark/results/literature_route_family_match.csv')


def normalize(formula: str) -> str:
    return Composition(formula).reduced_formula


def elements(formula: str) -> set[str]:
    return set(Composition(formula).as_dict())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def parse_side(value: str) -> set[str]:
    out = set()
    for item in str(value).split(' | '):
        if '*' in item:
            out.add(normalize(item.split('*', 1)[1]))
    return out


def has_carbonate(formulas: set[str]) -> bool:
    return any({'C', 'O'}.issubset(elements(formula)) and bool(elements(formula) - {'C', 'O'}) for formula in formulas)


def has_oxide(formulas: set[str], element: str | None = None) -> bool:
    for formula in formulas:
        elems = elements(formula)
        if 'O' in elems and len(elems) == 2 and (element is None or element in elems):
            return True
    return False


def has_formula(formulas: set[str], formula: str) -> bool:
    return normalize(formula) in formulas


def family_match(target: str, row: dict[str, str]) -> bool:
    reactants = parse_side(row.get('reactants', ''))
    products = parse_side(row.get('products', ''))
    target = normalize(target)
    if target in {normalize(x) for x in ['BaTiO3', 'SrTiO3', 'CaTiO3']}:
        metal = next(iter(elements(target) - {'Ti', 'O'}))
        return has_carbonate(reactants) and has_oxide(reactants, 'Ti') and has_formula(products, 'CO2')
    if target == normalize('GaN'):
        return has_formula(reactants, 'Ga') and has_formula(reactants, 'NH3') and has_formula(products, 'H2')
    if target == normalize('InN'):
        return has_formula(reactants, 'In') and has_formula(reactants, 'NH3') and has_formula(products, 'H2')
    if target == normalize('CaCO3'):
        return has_formula(reactants, 'CaO') and has_formula(reactants, 'CO2')
    if target == normalize('LiCoO2'):
        return has_formula(reactants, 'Li2CO3') and (has_oxide(reactants, 'Co') or has_carbonate({f for f in reactants if 'Co' in elements(f)})) and has_formula(products, 'CO2')
    if target == normalize('LiMn2O4'):
        return has_formula(reactants, 'Li2CO3') and (has_oxide(reactants, 'Mn') or has_carbonate({f for f in reactants if 'Mn' in elements(f)})) and has_formula(products, 'CO2')
    if target == normalize('LiFePO4'):
        return has_formula(reactants, 'Li2CO3') and any('Fe' in elements(f) and {'C', 'O'}.issubset(elements(f)) for f in reactants) and any({'P', 'O'}.issubset(elements(f)) for f in reactants)
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--case-dir', type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    targets = ['BaTiO3', 'SrTiO3', 'CaTiO3', 'GaN', 'InN', 'CaCO3', 'LiCoO2', 'LiMn2O4', 'LiFePO4']
    rows = []
    for target in targets:
        target_norm = normalize(target)
        path = args.case_dir / target_norm / 'synthesizability_ranked_routes.csv'
        routes = read_csv(path)
        matches = [row for row in routes if family_match(target_norm, row)]
        top = matches[0] if matches else {}
        rows.append({
            'target': target_norm,
            'family_found': bool(matches),
            'best_family_synth_rank': top.get('synth_rank', ''),
            'best_family_score': top.get('synth_score', ''),
            'best_family_reaction': top.get('reaction', ''),
            'best_family_deltaE_per_target': top.get('synth_rank_deltaE_per_target', ''),
            'best_family_template': top.get('route_template_reason', ''),
            'best_family_risk_flags': top.get('synthesis_risk_flags', ''),
            'n_family_matches': len(matches),
            'n_routes': len(routes),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row['target'], row['family_found'], row['best_family_synth_rank'], row['best_family_reaction'])
    print(args.output)


if __name__ == '__main__':
    main()
