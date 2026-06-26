#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

from pymatgen.core import Composition


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_CEDER = PROJECT / 'literature_benchmark/ceder_text_mined_routes.csv'
DEFAULT_CASE_DIR = PROJECT / 'results/seed80_synthesizability_cases'
DEFAULT_OUTPUT = PROJECT / 'literature_benchmark/ceder_seed80_real_literature_routes.csv'
FORMULA_TOKEN = re.compile(r'[A-Z][A-Za-z0-9().]*')
ALLOWED_EXTRA_REACTANT_ELEMENTS = {'C', 'H', 'N', 'O'}


def normalize(formula: str) -> str:
    return Composition(str(formula).strip()).reduced_formula


def valid_formula(formula: str) -> bool:
    try:
        comp = Composition(formula)
    except Exception:
        return False
    if not comp.as_dict() or len(comp.elements) > 5:
        return False
    return all(amount > 0 for amount in comp.as_dict().values())


def split_formulas(value: str | None) -> list[str]:
    formulas = []
    for item in str(value or '').split(';'):
        item = item.strip()
        if not item:
            continue
        try:
            formulas.append(normalize(item))
        except Exception:
            continue
    return formulas


def formula_elements(formula: str) -> set[str]:
    return {str(element) for element in Composition(formula).elements}


def formulas_in_text(text: str) -> set[str]:
    formulas = set()
    for token in FORMULA_TOKEN.findall(text or ''):
        try:
            formulas.add(normalize(token))
        except Exception:
            continue
    return formulas


def product_side_contains_target(route_text: str, target: str) -> bool:
    text = route_text or ''
    if '==' in text:
        right = text.split('==', 1)[1]
    elif '->' in text:
        right = text.split('->', 1)[1]
    else:
        right = text
    return target in formulas_in_text(right)


def reactants_are_target_relevant(target: str, precursors: list[str]) -> bool:
    target_elements = formula_elements(target)
    reactant_elements = set().union(*(formula_elements(formula) for formula in precursors)) if precursors else set()
    if not target_elements.issubset(reactant_elements):
        return False
    extra = reactant_elements - target_elements
    return extra.issubset(ALLOWED_EXTRA_REACTANT_ELEMENTS)


def source_doi(note: str | None) -> str:
    match = re.search(r'DOI=([^;]+)', note or '')
    return match.group(1).strip() if match else ''


def is_auditable_route(row: dict[str, str], target: str, precursors: list[str], byproducts: list[str]) -> bool:
    if not precursors:
        return False
    if not valid_formula(target):
        return False
    if not all(valid_formula(formula) for formula in [*precursors, *byproducts]):
        return False
    if not product_side_contains_target(row.get('literature_route_text', ''), target):
        return False
    if not reactants_are_target_relevant(target, precursors):
        return False
    if len(precursors) > 4 or len(byproducts) > 4:
        return False
    text = row.get('literature_route_text', '')
    if any(token in text for token in ['x ', '-x', '+x', '−x', '[', ']']):
        return False
    return bool(source_doi(row.get('source_note')))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description='Select DOI-backed Ceder routes for targets already screened by the workflow.')
    parser.add_argument('--ceder-routes', type=Path, default=DEFAULT_CEDER)
    parser.add_argument('--case-dir', type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--max-per-target', type=int, default=3)
    parser.add_argument('--max-routes', type=int, default=160)
    args = parser.parse_args()

    completed = {path.parent.name for path in args.case_dir.glob('*/synthesizability_ranked_routes.csv')}
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(args.ceder_routes):
        try:
            target = normalize(row['target'])
        except Exception:
            continue
        if target not in completed:
            continue
        precursors = split_formulas(row.get('literature_precursors'))
        byproducts = split_formulas(row.get('literature_byproducts'))
        if not is_auditable_route(row, target, precursors, byproducts):
            continue
        selected = dict(row)
        selected['target'] = target
        selected['literature_precursors'] = ';'.join(sorted(set(precursors)))
        selected['literature_byproducts'] = ';'.join(sorted(set(byproducts)))
        selected['source_note'] = selected.get('source_note', '') + '; selected_for_real_literature_validation'
        grouped[target].append(selected)

    selected_rows = []
    for target, rows in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        seen = set()
        kept = 0
        for row in rows:
            key = (row['literature_precursors'], row['literature_byproducts'])
            if key in seen:
                continue
            seen.add(key)
            selected_rows.append(row)
            kept += 1
            if kept >= args.max_per_target:
                break
        if len(selected_rows) >= args.max_routes:
            break
    selected_rows = selected_rows[: args.max_routes]

    fields = [
        'target',
        'literature_precursors',
        'literature_byproducts',
        'literature_route_text',
        'synthesis_type',
        'condition_summary',
        'source_title',
        'source_note',
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, '') for field in fields} for row in selected_rows])

    counts = Counter(row['target'] for row in selected_rows)
    print({'routes': len(selected_rows), 'targets': len(counts), 'completed_targets': len(completed)})
    print(counts.most_common(20))
    print(args.output)


if __name__ == '__main__':
    main()
