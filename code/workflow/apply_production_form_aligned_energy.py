#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition


PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / 'src'))

from analysis import annotate_reaction_practicality  # noqa: E402
from common import normalize_formula  # noqa: E402
from precursor_library import GAS_RESERVOIR_FORMULAS  # noqa: E402
from synthesizability import write_synthesizability_outputs  # noqa: E402


def load_model_energies(path: Path) -> tuple[dict[str, float], dict[str, float]]:
    frame = pd.read_csv(path)
    valid = frame.loc[frame['status'].eq('ok') & frame['energy_per_formula_eV'].notna()].copy()
    by_path = {
        str(Path(row['cif_path']).resolve()): float(row['energy_per_formula_eV'])
        for row in valid.to_dict('records')
    }
    by_formula = {}
    for row in valid.to_dict('records'):
        by_formula.setdefault(normalize_formula(row['formula']), float(row['energy_per_formula_eV']))
    return by_path, by_formula


def mp_form_energy(row: pd.Series) -> float | None:
    if pd.notna(row.get('mp_formation_energy_per_formula')):
        return float(row['mp_formation_energy_per_formula'])
    if pd.notna(row.get('formation_energy_per_atom')) and pd.notna(row.get('n_atoms')):
        return float(row['formation_energy_per_atom']) * float(row['n_atoms'])
    return None


def raw_model_energy(row: pd.Series, by_path: dict[str, float], by_formula: dict[str, float]) -> float | None:
    cif = row.get('cif_path')
    if isinstance(cif, str) and cif:
        value = by_path.get(str(Path(cif).resolve()))
        if value is not None:
            return value
    return by_formula.get(normalize_formula(row['formula']))


def fit_element_references(
    phases: pd.DataFrame,
    by_path: dict[str, float],
    by_formula: dict[str, float],
) -> tuple[dict[str, float] | None, int, int, float | None]:
    elements = sorted({str(element) for formula in phases['formula'].dropna().map(normalize_formula) for element in Composition(formula).elements})
    records = []
    for _, row in phases.iterrows():
        source = str(row.get('source', ''))
        if source == 'virtual_common_precursor':
            continue
        raw = raw_model_energy(row, by_path, by_formula)
        reference = mp_form_energy(row)
        if raw is None or reference is None:
            continue
        formula = normalize_formula(row['formula'])
        comp = Composition(formula).as_dict()
        records.append((comp, raw, reference))
    if not records:
        return None, 0, 0, None

    refs: dict[str, float] = {}
    for comp, raw, reference in records:
        if len(comp) != 1:
            continue
        element, amount = next(iter(comp.items()))
        refs[element] = (raw - reference) / float(amount)

    missing = [element for element in elements if element not in refs]
    if missing:
        matrix, targets = [], []
        for comp, raw, reference in records:
            row = [float(comp.get(element, 0.0)) for element in missing]
            if not any(row):
                continue
            known = sum(float(amount) * refs[element] for element, amount in comp.items() if element in refs)
            matrix.append(row)
            targets.append(raw - reference - known)
        if matrix:
            solution, *_ = np.linalg.lstsq(np.asarray(matrix, dtype=float), np.asarray(targets, dtype=float), rcond=None)
            refs.update(dict(zip(missing, map(float, solution))))

    design = np.asarray([[float(comp.get(element, 0.0)) for element in elements] for comp, _, _ in records], dtype=float)
    residuals = [abs(raw - sum(float(amount) * refs[element] for element, amount in comp.items()) - reference) for comp, raw, reference in records]
    return refs, len(records), int(np.linalg.matrix_rank(design)), float(np.mean(residuals))


def aligned_phase_energies(
    phases: pd.DataFrame,
    by_path: dict[str, float],
    by_formula: dict[str, float],
    refs: dict[str, float] | None,
) -> tuple[dict[str, float], dict[str, str]]:
    energies, sources = {}, {}
    for _, row in phases.iterrows():
        formula = normalize_formula(row['formula'])
        source = str(row.get('source', ''))
        reference = mp_form_energy(row)
        if source == 'virtual_common_precursor':
            sources[formula] = 'missing_virtual_precursor_energy'
            continue
        if formula in GAS_RESERVOIR_FORMULAS or source == 'manual_reservoir':
            if reference is not None:
                energies[formula] = reference
                sources[formula] = 'manual_gas_or_reservoir_formation_energy'
            continue
        raw = raw_model_energy(row, by_path, by_formula)
        comp = Composition(formula).as_dict()
        if raw is not None and refs is not None and all(element in refs for element in comp):
            energies[formula] = raw - sum(float(amount) * refs[element] for element, amount in comp.items())
            sources[formula] = 'production_model_with_fitted_element_references'
        elif reference is not None:
            energies[formula] = reference
            sources[formula] = 'fallback_MP_FORM_energy'
        else:
            sources[formula] = 'missing_energy'
    return energies, sources


def parse_side(value: object) -> list[tuple[str, float]]:
    out = []
    for part in str(value or '').split('|'):
        if '*' not in part:
            continue
        coefficient, formula = part.split('*', 1)
        out.append((normalize_formula(formula.strip()), float(coefficient)))
    return out


def apply_reaction_energies(
    frame: pd.DataFrame,
    energy_map: dict[str, float],
    source_map: dict[str, str],
    model_name: str,
) -> pd.DataFrame:
    totals, normalized, missing_values = [], [], []
    for _, row in frame.iterrows():
        reactants = parse_side(row.get('reactants'))
        products = parse_side(row.get('products'))
        missing = sorted({formula for formula, _ in reactants + products if formula not in energy_map})
        if missing:
            totals.append(np.nan)
            normalized.append(np.nan)
            missing_values.append(';'.join(f'{formula}:{source_map.get(formula, "missing_energy")}' for formula in missing))
            continue
        total = sum(coefficient * energy_map[formula] for formula, coefficient in products) - sum(
            coefficient * energy_map[formula] for formula, coefficient in reactants
        )
        target_coefficient = float(row.get('target_coefficient', 0.0) or 0.0)
        totals.append(total)
        normalized.append(total / target_coefficient if target_coefficient else np.nan)
        missing_values.append('')
    frame = frame.copy()
    frame['DeltaE_EquiformerV3_FORM_ALIGNED'] = totals
    frame['DeltaE_EquiformerV3_FORM_ALIGNED_per_target'] = normalized
    frame['DeltaE_PRODUCTION_FORM_ALIGNED'] = totals
    frame['DeltaE_PRODUCTION_FORM_ALIGNED_per_target'] = normalized
    frame['production_energy_model'] = model_name
    frame['PRODUCTION_FORM_ALIGNED_missing_species'] = missing_values
    return frame


def process_target(
    target_dir: Path,
    by_path: dict[str, float],
    by_formula: dict[str, float],
    model_name: str,
    top_n: int,
) -> dict[str, object] | None:
    phase_path = target_dir / 'phase_stability.csv'
    formation_path = target_dir / 'formation_reactions.csv'
    if not phase_path.exists() or not formation_path.exists():
        return None
    phases = pd.read_csv(phase_path)
    formation = pd.read_csv(formation_path)
    target = normalize_formula(target_dir.name)
    refs, n_fit, fit_rank, fit_mae = fit_element_references(phases, by_path, by_formula)
    energy_map, source_map = aligned_phase_energies(phases, by_path, by_formula, refs)
    source_frame = pd.DataFrame(
        [
            {
                'formula': formula,
                'production_energy_model': model_name,
                'PRODUCTION_FORM_ALIGNED_source': source,
                'PRODUCTION_FORM_ALIGNED_energy_per_formula': energy_map.get(formula),
            }
            for formula, source in sorted(source_map.items())
        ]
    )
    source_frame.to_csv(target_dir / 'production_form_aligned_phase_energy_sources.csv', index=False)
    formation = apply_reaction_energies(formation, energy_map, source_map, model_name)
    practical_columns = {'max_non_target_atoms', 'max_non_target_stoich_ratio', 'max_non_target_ehull', 'coeff_sum_per_target', 'practical_route'}
    if not practical_columns.issubset(formation.columns):
        formation = annotate_reaction_practicality(formation, phases, target)
    formation.to_csv(formation_path, index=False)
    decomposition_path = target_dir / 'decomposition_reactions.csv'
    if decomposition_path.exists() and decomposition_path.stat().st_size:
        decomposition = apply_reaction_energies(pd.read_csv(decomposition_path), energy_map, source_map, model_name)
        decomposition.to_csv(decomposition_path, index=False)
    scored = write_synthesizability_outputs(target_dir, target, phases, formation, top_n=top_n)
    counts = pd.Series(list(source_map.values())).value_counts().to_dict()
    return {
        'target': target,
        'production_energy_model': model_name,
        'n_fit_phases': n_fit,
        'n_elements': len(refs or {}),
        'fit_rank': fit_rank,
        'fit_mae_eV_per_formula': fit_mae,
        'n_phases_model_aligned': int(counts.get('production_model_with_fitted_element_references', 0)),
        'n_phases_fallback_mp_form': int(counts.get('fallback_MP_FORM_energy', 0)),
        'n_phases_manual_reservoir': int(counts.get('manual_gas_or_reservoir_formation_energy', 0)),
        'n_routes': int(len(formation)),
        'n_routes_with_production_energy': int(formation['DeltaE_PRODUCTION_FORM_ALIGNED_per_target'].notna().sum()),
        'top_reaction': scored.iloc[0]['reaction'] if len(scored) else '',
        'top_deltaE_per_target': scored.iloc[0]['synth_rank_deltaE_per_target'] if len(scored) else np.nan,
        'top_synth_score': scored.iloc[0]['synth_score'] if len(scored) else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--case-dir', type=Path, required=True)
    parser.add_argument('--energy-cache', type=Path, required=True)
    parser.add_argument('--model-name', default='EquiformerV3+DeNS-OAM2')
    parser.add_argument('--top-n', type=int, default=20)
    args = parser.parse_args()
    by_path, by_formula = load_model_energies(args.energy_cache)
    rows = []
    for target_dir in sorted(path for path in args.case_dir.iterdir() if path.is_dir()):
        result = process_target(target_dir, by_path, by_formula, args.model_name, args.top_n)
        if result:
            rows.append(result)
            print(result['target'], result['n_routes_with_production_energy'], '/', result['n_routes'], result['top_reaction'], flush=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.case_dir / 'production_form_aligned_energy_summary.csv', index=False)
    metadata = {
        'model': args.model_name,
        'energy_cache': str(args.energy_cache.resolve()),
        'n_targets': len(summary),
        'n_routes': int(summary['n_routes'].sum()) if len(summary) else 0,
        'n_routes_with_production_energy': int(summary['n_routes_with_production_energy'].sum()) if len(summary) else 0,
    }
    (args.case_dir / 'production_energy_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
