#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from pymatgen.core import Composition

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))

from analysis import annotate_reaction_practicality, mp_formation_energy_map  # noqa: E402
from common import normalize_formula  # noqa: E402
from precursor_library import GAS_RESERVOIR_FORMULAS  # noqa: E402
from reactions import reactions_to_frame  # noqa: E402
from synthesizability import write_synthesizability_outputs  # noqa: E402

DEFAULT_CASE_DIR = PROJECT / 'results/seed40_synthesizability_cases'
DEFAULT_PET_ROOTS = [
    PROJECT / 'results/mp_pet_benchmark_120',
    PROJECT / 'results/mp_pet_benchmark',
    PROJECT / 'results/mp_pet_smoke',
]


def load_global_pet_map(roots: list[Path]) -> dict[str, float]:
    values: dict[str, float] = {}
    for root in roots:
        for cache in root.glob('**/pet_energy_cache.csv'):
            try:
                df = pd.read_csv(cache)
            except Exception:
                continue
            if 'formula' not in df or 'PET_energy_per_formula' not in df:
                continue
            for _, row in df.iterrows():
                if pd.isna(row.get('PET_energy_per_formula')):
                    continue
                try:
                    formula = normalize_formula(row['formula'])
                    values[formula] = float(row['PET_energy_per_formula'])
                except Exception:
                    continue
    return values


def fit_element_references(phase_df: pd.DataFrame, pet_map: dict[str, float]) -> tuple[dict[str, float] | None, int, int, float | None]:
    elements = sorted({str(element) for formula in phase_df['formula'].dropna().map(normalize_formula) for element in Composition(formula).elements})
    matrix = []
    target = []
    fit_records = []
    for _, row in phase_df.iterrows():
        try:
            formula = normalize_formula(row['formula'])
        except Exception:
            continue
        pet = pet_map.get(formula)
        mp = None
        if pd.notna(row.get('mp_formation_energy_per_formula')):
            mp = float(row['mp_formation_energy_per_formula'])
        elif pd.notna(row.get('formation_energy_per_atom')) and pd.notna(row.get('n_atoms')):
            mp = float(row['formation_energy_per_atom']) * float(row['n_atoms'])
        if pet is None or mp is None:
            continue
        source = str(row.get('source', ''))
        if source in {'manual_reservoir', 'virtual_common_precursor'}:
            continue
        comp = Composition(formula).as_dict()
        matrix.append([float(comp.get(element, 0.0)) for element in elements])
        target.append(float(pet) - float(mp))
        fit_records.append((formula, float(pet), float(mp)))
    if not matrix:
        return None, 0, 0, None
    a = np.array(matrix, dtype=float)
    y = np.array(target, dtype=float)
    solution, *_ = np.linalg.lstsq(a, y, rcond=None)
    rank = int(np.linalg.matrix_rank(a))
    refs = {element: float(value) for element, value in zip(elements, solution)}
    residuals = []
    for formula, pet, mp in fit_records:
        comp = Composition(formula).as_dict()
        fitted = pet - sum(float(comp.get(element, 0.0)) * refs[element] for element in elements)
        residuals.append(abs(fitted - mp))
    mae = float(np.mean(residuals)) if residuals else None
    return refs, len(fit_records), rank, mae


def pet_form_aligned_map(phase_df: pd.DataFrame, pet_map: dict[str, float], element_refs: dict[str, float] | None) -> tuple[dict[str, float], dict[str, str]]:
    out: dict[str, float] = {}
    sources: dict[str, str] = {}
    for _, row in phase_df.iterrows():
        try:
            formula = normalize_formula(row['formula'])
        except Exception:
            continue
        source = str(row.get('source', ''))
        if source == 'virtual_common_precursor':
            sources[formula] = 'missing_virtual_precursor_energy'
            continue
        mp_value = None
        if pd.notna(row.get('mp_formation_energy_per_formula')):
            mp_value = float(row['mp_formation_energy_per_formula'])
        elif pd.notna(row.get('formation_energy_per_atom')) and pd.notna(row.get('n_atoms')):
            mp_value = float(row['formation_energy_per_atom']) * float(row['n_atoms'])
        if formula in GAS_RESERVOIR_FORMULAS or source == 'manual_reservoir':
            if mp_value is not None:
                out[formula] = mp_value
                sources[formula] = 'manual_gas_or_reservoir_formation_energy'
            continue
        if formula in pet_map and element_refs is not None:
            comp = Composition(formula).as_dict()
            missing_elements = [element for element in comp if element not in element_refs]
            if missing_elements:
                sources[formula] = 'missing_element_reference'
                continue
            out[formula] = pet_map[formula] - sum(float(amount) * element_refs[element] for element, amount in comp.items())
            sources[formula] = 'PET_energy_with_fitted_element_references'
        elif mp_value is not None:
            out[formula] = mp_value
            sources[formula] = 'fallback_MP_FORM_energy'
        else:
            sources[formula] = 'missing_energy'
    return out, sources


def rebuild_reactions_with_pet_form_aligned(target_dir: Path, pet_map: dict[str, float], top_n: int = 10) -> dict[str, object] | None:
    phase_path = target_dir / 'phase_stability.csv'
    formation_path = target_dir / 'formation_reactions.csv'
    if not phase_path.exists() or not formation_path.exists():
        return None
    phase_df = pd.read_csv(phase_path)
    formation_df = pd.read_csv(formation_path)
    target = normalize_formula(target_dir.name)
    element_refs, n_fit, fit_rank, fit_mae = fit_element_references(phase_df, pet_map)
    aligned_map, source_map = pet_form_aligned_map(phase_df, pet_map, element_refs)
    source_counts = pd.Series(list(source_map.values())).value_counts().to_dict()
    phase_sources = [
        {
            'formula': formula,
            'PET_FORM_ALIGNED_source': source,
            'PET_FORM_ALIGNED_energy_per_formula': aligned_map.get(formula),
        }
        for formula, source in sorted(source_map.items())
    ]
    pd.DataFrame(phase_sources).to_csv(target_dir / 'pet_form_aligned_phase_energy_sources.csv', index=False)
    formation_df = formation_df.copy()
    values = []
    values_per_target = []
    missing_reasons = []
    for _, row in formation_df.iterrows():
        species = []
        for side in ['reactants', 'products']:
            for part in str(row.get(side, '')).split(' | '):
                if '*' not in part:
                    continue
                coeff, formula = part.split('*', 1)
                species.append((normalize_formula(formula), int(coeff), side))
        missing = sorted({formula for formula, _, _ in species if formula not in aligned_map})
        if missing:
            values.append(None)
            values_per_target.append(None)
            missing_reasons.append(';'.join(f'{formula}:{source_map.get(formula, "missing_energy")}' for formula in missing))
            continue
        left = sum(coeff * aligned_map[formula] for formula, coeff, side in species if side == 'reactants')
        right = sum(coeff * aligned_map[formula] for formula, coeff, side in species if side == 'products')
        value = right - left
        target_coeff = row.get('target_coefficient')
        values.append(value)
        values_per_target.append(value / float(target_coeff) if pd.notna(target_coeff) and float(target_coeff) else None)
        missing_reasons.append('')
    formation_df['DeltaE_PET_FORM_ALIGNED'] = values
    formation_df['DeltaE_PET_FORM_ALIGNED_per_target'] = values_per_target
    formation_df['PET_FORM_ALIGNED_missing_species'] = missing_reasons
    practical_cols = {'max_non_target_atoms', 'max_non_target_stoich_ratio', 'max_non_target_ehull', 'coeff_sum_per_target', 'practical_route'}
    if not practical_cols.issubset(formation_df.columns):
        formation_df = annotate_reaction_practicality(formation_df, phase_df, target)
    formation_df.to_csv(target_dir / 'formation_reactions.csv', index=False)
    scored = write_synthesizability_outputs(target_dir, target, phase_df, formation_df, top_n=top_n)
    top = scored.iloc[0] if len(scored) else {}
    return {
        'target': target,
        'n_fit_phases': n_fit,
        'n_elements': len(element_refs or {}),
        'fit_rank': fit_rank,
        'fit_method': 'least_squares_element_references',
        'element_references_eV': ';'.join(f'{element}:{value:.8g}' for element, value in sorted((element_refs or {}).items())),
        'fit_mae_eV_per_formula': fit_mae,
        'n_phases_pet_aligned': int(source_counts.get('PET_energy_with_fitted_element_references', 0)),
        'n_phases_fallback_mp_form': int(source_counts.get('fallback_MP_FORM_energy', 0)),
        'n_phases_manual_reservoir': int(source_counts.get('manual_gas_or_reservoir_formation_energy', 0)),
        'n_phases_missing_or_virtual': int(
            source_counts.get('missing_virtual_precursor_energy', 0)
            + source_counts.get('missing_energy', 0)
            + source_counts.get('missing_element_reference', 0)
        ),
        'n_routes': int(len(formation_df)),
        'n_routes_with_pet_form_aligned': int(pd.Series(values_per_target).notna().sum()),
        'n_routes_missing_pet_form_aligned': int(pd.Series(values_per_target).isna().sum()),
        'top_reaction': top.get('reaction', ''),
        'top_energy_source': top.get('synth_rank_energy_source', ''),
        'top_deltaE_per_target': top.get('synth_rank_deltaE_per_target', ''),
        'top_synth_score': top.get('synth_score', ''),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--case-dir', type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument('--pet-roots', nargs='*', type=Path, default=DEFAULT_PET_ROOTS)
    parser.add_argument('--top-n', type=int, default=10)
    args = parser.parse_args()
    pet_map = load_global_pet_map(args.pet_roots)
    if not pet_map:
        raise SystemExit('No PET cache entries found.')
    rows = []
    for target_dir in sorted(path for path in args.case_dir.iterdir() if path.is_dir()):
        row = rebuild_reactions_with_pet_form_aligned(target_dir, pet_map, top_n=args.top_n)
        if row:
            rows.append(row)
            print(row['target'], row['n_routes_with_pet_form_aligned'], '/', row['n_routes'], row['top_energy_source'], row['top_reaction'])
    out = args.case_dir / 'pet_form_aligned_energy_summary.csv'
    with out.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(out)


if __name__ == '__main__':
    main()
