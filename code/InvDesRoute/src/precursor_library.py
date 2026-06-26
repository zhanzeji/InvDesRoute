from __future__ import annotations

from pathlib import Path

import pandas as pd
from pymatgen.core import Composition

from common import PhaseEntry, PROJECT, formula_num_atoms, normalize_formula, safe_float


RAW_GAS_RESERVOIR_FORMULAS = {
    'O2': 0.0,
    'N2': 0.0,
    'H2': 0.0,
    'Cl2': 0.0,
    'F2': 0.0,
    'Br2': 0.0,
    'I2': 0.0,
    'CO': -1.145,
    'CO2': -4.078,
    'H2O': -2.506,
    'NH3': -0.477,
}

GAS_RESERVOIR_FORMULAS = {normalize_formula(formula): value for formula, value in RAW_GAS_RESERVOIR_FORMULAS.items()}


def oxidation_states_for_symbol(symbol: str) -> list[int]:
    try:
        element = Composition(symbol).elements[0]
        states = [int(value) for value in element.common_oxidation_states if value > 0]
    except Exception:
        states = []
    if not states:
        states = [1]
    return sorted(set(states))


def charge_balanced_binary(cation: str, cation_charge: int, anion: str, anion_charge: int) -> str:
    from math import gcd

    c = abs(cation_charge)
    a = abs(anion_charge)
    divisor = gcd(c, a)
    cation_count = a // divisor
    anion_count = c // divisor
    return normalize_formula(f'{cation}{cation_count if cation_count > 1 else ""}{anion}{anion_count if anion_count > 1 else ""}')


def precursor_templates_for_element(symbol: str) -> set[str]:
    formulas: set[str] = set()
    for ox in oxidation_states_for_symbol(symbol):
        try:
            formulas.add(charge_balanced_binary(symbol, ox, 'O', -2))
            formulas.add(charge_balanced_binary(symbol, ox, 'F', -1))
            formulas.add(charge_balanced_binary(symbol, ox, 'Cl', -1))
            formulas.add(charge_balanced_binary(symbol, ox, 'Br', -1))
            formulas.add(charge_balanced_binary(symbol, ox, 'I', -1))
            formulas.add(normalize_formula(f'{symbol}{"2" if ox == 1 else ""}CO3' if ox in {1, 2} else f'{symbol}2(CO3){ox}'))
            formulas.add(normalize_formula(f'{symbol}(NO3){ox if ox > 1 else ""}'))
            formulas.add(normalize_formula(f'{symbol}(OH){ox if ox > 1 else ""}'))
        except Exception:
            continue
    return formulas


def common_precursor_formulas(target_formula: str, include_reservoirs: bool = True) -> set[str]:
    target = normalize_formula(target_formula)
    elements = [str(element) for element in Composition(target).elements]
    formulas: set[str] = set(elements)
    for element in elements:
        formulas.update(precursor_templates_for_element(element))
    # Common mixed-anion / polyanion precursors for battery and oxide synthesis.
    if 'Li' in elements:
        formulas.update({'Li2CO3', 'LiOH', 'Li2O'})
    if 'Na' in elements:
        formulas.update({'Na2CO3', 'NaOH', 'Na2O'})
    if 'K' in elements:
        formulas.update({'K2CO3', 'KOH', 'K2O'})
    if 'P' in elements:
        formulas.update({'NH4H2PO4', 'P2O5', 'Li3PO4', 'FePO4'})
    if 'Ti' in elements:
        formulas.update({'TiO2'})
    if 'Co' in elements:
        formulas.update({'Co3O4', 'CoO'})
    if 'Mn' in elements:
        formulas.update({'MnO2', 'Mn2O3', 'Mn3O4'})
    if 'Fe' in elements:
        formulas.update({'Fe2O3', 'Fe3O4', 'FeC2O4'})
    if include_reservoirs:
        formulas.update(GAS_RESERVOIR_FORMULAS)
    out = set()
    for formula in formulas:
        try:
            out.add(normalize_formula(formula))
        except Exception:
            continue
    return out


def reservoir_phase_entries(formulas: set[str] | None = None) -> pd.DataFrame:
    selected = formulas or set(GAS_RESERVOIR_FORMULAS)
    rows = []
    for formula in sorted(selected):
        normalized = normalize_formula(formula)
        if normalized not in GAS_RESERVOIR_FORMULAS:
            continue
        energy_per_formula = GAS_RESERVOIR_FORMULAS[normalized]
        rows.append(
            PhaseEntry(
                formula=normalized,
                source_id='manual_reservoir',
                source='manual_reservoir',
                formation_energy_per_atom=energy_per_formula / formula_num_atoms(normalized),
                energy_above_hull=0.0,
                cif_path=None,
            ).as_dict()
        )
    return pd.DataFrame(rows)


def load_manual_precursor_table(path: str | Path | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    table = Path(path)
    if not table.exists():
        raise FileNotFoundError(table)
    df = pd.read_csv(table)
    if 'formula' not in df:
        raise ValueError(f'{table} must contain a formula column.')
    df = df.copy()
    df['formula'] = df['formula'].map(normalize_formula)
    rows = []
    for _, row in df.iterrows():
        formula = row['formula']
        fe = safe_float(row.get('formation_energy_per_atom'))
        epa = safe_float(row.get('dft_energy_per_atom'))
        rows.append(
            PhaseEntry(
                formula=formula,
                source_id=str(row.get('source_id', 'manual_precursor')),
                source=str(row.get('source', 'manual_precursor')),
                dft_energy_per_atom=epa,
                dft_energy_per_formula=epa * formula_num_atoms(formula) if epa is not None else None,
                formation_energy_per_atom=fe,
                energy_above_hull=safe_float(row.get('energy_above_hull')),
                cif_path=row.get('cif_path') if pd.notna(row.get('cif_path')) else None,
            ).as_dict()
        )
    return pd.DataFrame(rows)


def virtual_common_precursor_entries(formulas: set[str], existing_formulas: set[str] | None = None) -> pd.DataFrame:
    existing = {normalize_formula(formula) for formula in (existing_formulas or set())}
    rows = []
    for formula in sorted({normalize_formula(formula) for formula in formulas} - existing - set(GAS_RESERVOIR_FORMULAS)):
        rows.append(
            PhaseEntry(
                formula=formula,
                source_id='virtual_common_precursor',
                source='virtual_common_precursor',
                dft_energy_per_atom=None,
                dft_energy_per_formula=None,
                formation_energy_per_atom=None,
                energy_above_hull=None,
                cif_path=None,
            ).as_dict()
        )
    return pd.DataFrame(rows)


def fetch_mp_precursor_formulas(
    formulas: set[str],
    api_key: str | None,
    cache_dir: str | Path | None = None,
    force_refresh: bool = False,
    include_structures: bool = True,
) -> pd.DataFrame:
    cache_dir = Path(cache_dir or PROJECT / 'cache' / 'mp_precursors')
    table_path = cache_dir / 'mp_precursor_phases.csv'
    cif_dir = cache_dir / 'cifs'
    formulas = {normalize_formula(formula) for formula in formulas}
    cached = pd.DataFrame()
    if table_path.exists() and not force_refresh:
        try:
            cached = pd.read_csv(table_path)
        except pd.errors.EmptyDataError:
            cached = pd.DataFrame()
        if len(cached):
            cached['formula'] = cached['formula'].map(normalize_formula)
            cached = cached.sort_values(['energy_above_hull', 'formula'], na_position='last').drop_duplicates('formula', keep='first')
    if not formulas:
        return pd.DataFrame()
    cached_formulas = set(cached['formula'].dropna().map(normalize_formula)) if len(cached) else set()
    missing_formulas = formulas - cached_formulas - set(GAS_RESERVOIR_FORMULAS)
    if not missing_formulas:
        return cached[cached['formula'].isin(formulas)].reset_index(drop=True) if len(cached) else pd.DataFrame()
    if not api_key:
        import os

        api_key = os.environ.get('MP_API_KEY') or os.environ.get('PMG_MAPI_KEY')
    if not api_key:
        raise RuntimeError('MP API key is required for precursor expansion.')
    try:
        from mp_api.client import MPRester
    except Exception as exc:
        raise RuntimeError('mp-api is not available in this environment.') from exc

    fields = ['material_id', 'formula_pretty', 'energy_per_atom', 'formation_energy_per_atom', 'energy_above_hull']
    if include_structures:
        fields.append('structure')
    rows = []
    with MPRester(api_key) as mpr:
        for formula in sorted(missing_formulas):
            if formula in GAS_RESERVOIR_FORMULAS:
                continue
            try:
                docs = mpr.materials.summary.search(formula=formula, fields=fields)
            except TypeError:
                docs = mpr.materials.summary.search(formula=[formula], fields=fields)
            if not docs:
                continue
            candidates = []
            for doc in docs:
                formula_raw = getattr(doc, 'formula_pretty', None) if not isinstance(doc, dict) else doc.get('formula_pretty')
                if not formula_raw:
                    continue
                normalized = normalize_formula(formula_raw)
                if normalized != formula:
                    continue
                ehull = safe_float(getattr(doc, 'energy_above_hull', None) if not isinstance(doc, dict) else doc.get('energy_above_hull'))
                candidates.append((ehull if ehull is not None else float('inf'), doc))
            if not candidates:
                continue
            _, doc = sorted(candidates, key=lambda item: item[0])[0]
            material_id = str(getattr(doc, 'material_id', None) if not isinstance(doc, dict) else doc.get('material_id'))
            epa = safe_float(getattr(doc, 'energy_per_atom', None) if not isinstance(doc, dict) else doc.get('energy_per_atom'))
            fe = safe_float(getattr(doc, 'formation_energy_per_atom', None) if not isinstance(doc, dict) else doc.get('formation_energy_per_atom'))
            ehull = safe_float(getattr(doc, 'energy_above_hull', None) if not isinstance(doc, dict) else doc.get('energy_above_hull'))
            cif_path = None
            if include_structures:
                structure = getattr(doc, 'structure', None) if not isinstance(doc, dict) else doc.get('structure')
                if structure is not None:
                    cif_dir.mkdir(parents=True, exist_ok=True)
                    cif_path = str(cif_dir / f'{formula}-{material_id}.cif')
                    structure.to(filename=cif_path)
            rows.append(
                PhaseEntry(
                    formula=formula,
                    source_id=material_id,
                    source='materials_project_precursor',
                    dft_energy_per_atom=epa,
                    dft_energy_per_formula=epa * formula_num_atoms(formula) if epa is not None else None,
                    formation_energy_per_atom=fe,
                    energy_above_hull=ehull,
                    cif_path=cif_path,
                ).as_dict()
            )
    out = pd.DataFrame(rows)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if len(out):
        out = out.sort_values(['energy_above_hull', 'formula'], na_position='last').drop_duplicates('formula', keep='first')
    combined = pd.concat([cached, out], ignore_index=True) if len(cached) else out
    if len(combined):
        combined['formula'] = combined['formula'].map(normalize_formula)
        combined = combined.sort_values(['energy_above_hull', 'formula'], na_position='last').drop_duplicates('formula', keep='first')
    if not len(combined):
        combined.to_csv(table_path, index=False)
        return pd.DataFrame()
    combined.to_csv(table_path, index=False)
    return combined[combined['formula'].isin(formulas)].reset_index(drop=True)
