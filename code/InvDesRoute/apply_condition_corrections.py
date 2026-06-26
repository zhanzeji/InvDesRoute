#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from common import PROJECT, normalize_formula  # noqa: E402
from thermo_conditions import add_condition_corrections  # noqa: E402


def parse_pressures(items: list[str]) -> dict[str, float]:
    out = {}
    for item in items:
        if '=' not in item:
            raise ValueError(f'Pressure must be FORMULA=bar, got {item}')
        formula, value = item.split('=', 1)
        out[normalize_formula(formula)] = float(value)
    return out


def apply_case(target_dir: Path, temperatures: list[float], pressures: dict[str, float], base_source: str) -> dict[str, object] | None:
    path = target_dir / 'formation_reactions.csv'
    if not path.exists() or path.stat().st_size == 0:
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    for temperature in temperatures:
        df = add_condition_corrections(df, temperature, pressures, base_source=base_source)
    df.to_csv(path, index=False)
    summary = {'target': normalize_formula(target_dir.name), 'n_routes': len(df)}
    for temperature in temperatures:
        label = f'T{int(round(temperature))}K'
        cols = [col for col in df.columns if col.startswith('DeltaG_') and col.endswith(f'{label}_gas_per_target')]
        if cols:
            col = cols[0]
            valid = df[df[col].notna()].copy()
            if len(valid):
                best = valid.nsmallest(1, col).iloc[0]
                summary[f'best_{label}_reaction'] = best.get('reaction')
                summary[f'best_{label}_deltaG_per_target'] = float(best[col])
                summary[f'n_exergonic_{label}'] = int((valid[col] <= -0.05).sum())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Apply simplified finite-temperature gas chemical-potential corrections.')
    parser.add_argument('--case-dir', type=Path, default=PROJECT / 'results/seed80_synthesizability_cases')
    parser.add_argument('--temperature', type=float, nargs='+', default=[700.0, 1000.0, 1300.0])
    parser.add_argument('--pressure', action='append', default=[], help='Gas pressure in bar, e.g. CO2=0.01')
    parser.add_argument('--base-source', default='PET_FORM_ALIGNED')
    args = parser.parse_args()
    pressures = parse_pressures(args.pressure)
    rows = []
    for target_dir in sorted(path for path in args.case_dir.iterdir() if path.is_dir()):
        row = apply_case(target_dir, args.temperature, pressures, args.base_source)
        if row:
            rows.append(row)
            print(row['target'], row)
    out = args.case_dir / 'finite_temperature_gas_correction_summary.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    print(out)


if __name__ == '__main__':
    main()
