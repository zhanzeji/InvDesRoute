#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

from pymatgen.core import Composition

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_LIT = PROJECT / 'literature_benchmark/literature_routes_seed.csv'
DEFAULT_OUT = PROJECT / 'results/seed40_synthesizability_cases'
RUN = PROJECT / 'run_screening.py'


def normalize(formula: str) -> str:
    return Composition(formula).reduced_formula


def read_seed(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def split_formulas(value: str | None) -> set[str]:
    if not value:
        return set()
    return {normalize(item.strip()) for item in value.split(';') if item.strip()}


def needs_targeted(row: dict[str, str]) -> bool:
    formulas = split_formulas(row.get('literature_precursors')) | split_formulas(row.get('literature_byproducts'))
    elems = set().union(*(set(Composition(formula).as_dict()) for formula in formulas)) if formulas else set()
    target_elems = set(Composition(row['target']).as_dict())
    return bool(elems - target_elems) or len(split_formulas(row.get('literature_precursors'))) >= 3


def needs_virtual(row: dict[str, str]) -> bool:
    formulas = split_formulas(row.get('literature_precursors'))
    return any({'C', 'H', 'N'}.issubset(set(Composition(formula).as_dict())) or 'Fe(CO2)2' == formula for formula in formulas)


def command_for(row: dict[str, str], outdir: Path, api_key: str) -> list[str]:
    target = normalize(row['target'])
    target_dir = outdir / target
    target_dir.mkdir(parents=True, exist_ok=True)
    targeted = needs_targeted(row)
    virtual = needs_virtual(row)
    cmd = [
        str(RUN),
        '--target-formula', target,
        '--database', 'mp',
        '--mp-api-key', api_key,
        '--include-common-precursors',
        '--include-gas-reservoirs',
        '--byproducts-reservoir-only',
        '--output-dir', str(target_dir),
        '--max-reactants', '2',
        '--max-byproducts', '1',
        '--max-products', '2',
        '--max-total-coeff', '80',
        '--practical-max-atoms', '20',
        '--practical-max-coeff-sum-per-target', '80',
        '--mp-max-phases', '100',
    ]
    if targeted:
        cmd.extend([
            '--include-targeted-precursor-routes',
            '--targeted-max-reactants', '3',
            '--targeted-max-byproducts', '4',
            '--targeted-max-total-coeff', '140',
        ])
    if virtual:
        cmd.append('--include-virtual-common-precursors')
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--literature', type=Path, default=DEFAULT_LIT)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--mp-api-key', default='')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--targets', default='')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    api_key = args.mp_api_key
    if not api_key:
        import os
        api_key = os.environ.get('MP_API_KEY') or os.environ.get('PMG_MAPI_KEY') or ''
    if not api_key:
        raise SystemExit('MP API key required via --mp-api-key or MP_API_KEY.')
    rows = read_seed(args.literature)
    wanted = {normalize(item) for item in args.targets.split(',') if item.strip()}
    if wanted:
        rows = [row for row in rows if normalize(row['target']) in wanted]
    if args.limit:
        rows = rows[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_rows = []
    for idx, row in enumerate(rows, start=1):
        target = normalize(row['target'])
        target_dir = args.output_dir / target
        done = target_dir / 'synthesizability_ranked_routes.csv'
        if done.exists() and not args.force:
            print(f'[{idx}/{len(rows)}] skip {target}')
            log_rows.append({'target': target, 'status': 'skipped_existing', 'returncode': 0})
            continue
        cmd = command_for(row, args.output_dir, api_key)
        print(f'[{idx}/{len(rows)}] run {target}: targeted={needs_targeted(row)} virtual={needs_virtual(row)}')
        proc = subprocess.run(cmd, text=True, capture_output=True)
        (target_dir / 'seed_run_stdout.log').write_text(proc.stdout, encoding='utf-8')
        (target_dir / 'seed_run_stderr.log').write_text(proc.stderr, encoding='utf-8')
        status = 'ok' if proc.returncode == 0 else 'failed'
        print(f'  {status} rc={proc.returncode}')
        log_rows.append({'target': target, 'status': status, 'returncode': proc.returncode, 'targeted': needs_targeted(row), 'virtual': needs_virtual(row)})
    with (args.output_dir / 'seed40_run_log.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['target', 'status', 'returncode', 'targeted', 'virtual'])
        writer.writeheader()
        writer.writerows(log_rows)
    print(args.output_dir / 'seed40_run_log.csv')


if __name__ == '__main__':
    main()
