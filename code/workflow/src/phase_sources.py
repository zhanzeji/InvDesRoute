from __future__ import annotations

from itertools import combinations
import os
from pathlib import Path

import pandas as pd
from pymatgen.core import Structure

from common import (
    LOCAL_MP_CIFS,
    LOCAL_MP_ENERGIES,
    PROJECT,
    PhaseEntry,
    chemical_system,
    formula_elements,
    formula_from_cif,
    formula_num_atoms,
    normalize_formula,
    safe_float,
)


def chemical_subsystems(elements: list[str]) -> list[str]:
    ordered = sorted(set(elements))
    systems = []
    for size in range(1, len(ordered) + 1):
        for subset in combinations(ordered, size):
            systems.append('-'.join(subset))
    return systems


def find_local_cif(formula: str, source_id: str | None = None) -> str | None:
    formula = normalize_formula(formula)
    if source_id:
        matches = sorted(LOCAL_MP_CIFS.glob(f'*{source_id}*.cif'))
        if matches:
            return str(matches[0])
    matches = sorted(LOCAL_MP_CIFS.glob(f'{formula}-*.cif'))
    if matches:
        return str(matches[0])
    return None


def load_local_phase_library() -> pd.DataFrame:
    rows = []
    if not LOCAL_MP_ENERGIES.exists():
        return pd.DataFrame()
    df = pd.read_csv(LOCAL_MP_ENERGIES)
    for _, row in df.iterrows():
        formula = normalize_formula(row['formula'])
        epa = safe_float(row.get('energy_per_atom (eV)'))
        entry = PhaseEntry(
            formula=formula,
            source_id=str(row.get('mpid')),
            source='local_mp_table',
            dft_energy_per_atom=epa,
            dft_energy_per_formula=epa * formula_num_atoms(formula) if epa is not None else None,
            cif_path=find_local_cif(formula, str(row.get('mpid'))),
        )
        rows.append(entry.as_dict())
    out = pd.DataFrame(rows)
    return out.drop_duplicates('formula', keep='first').reset_index(drop=True)


def external_target_entry(cif_path: str | Path, formula: str | None = None) -> dict:
    formula = normalize_formula(formula or formula_from_cif(cif_path))
    return PhaseEntry(
        formula=formula,
        source_id='external_cif',
        source='external_cif',
        cif_path=str(Path(cif_path).resolve()),
    ).as_dict()


def fetch_mp_phases(
    formula: str,
    api_key: str | None = None,
    cache_dir: str | Path | None = None,
    max_phases: int | None = None,
    include_structures: bool = True,
    force_refresh: bool = False,
) -> pd.DataFrame:
    target = normalize_formula(formula)
    chemsys = chemical_system(target)
    cache_dir = Path(cache_dir or PROJECT / 'cache' / 'mp')
    system_dir = cache_dir / chemsys
    table_path = system_dir / 'mp_phases.csv'
    cif_dir = system_dir / 'cifs'
    if table_path.exists() and not force_refresh:
        return pd.read_csv(table_path)

    api_key = api_key or os.environ.get('MP_API_KEY') or os.environ.get('PMG_MAPI_KEY')
    if not api_key:
        raise RuntimeError('MP API key is required for database expansion. Set MP_API_KEY or pass --mp-api-key.')
    try:
        from mp_api.client import MPRester
    except Exception as exc:
        raise RuntimeError('mp-api is not available in this environment.') from exc

    fields = ['material_id', 'formula_pretty', 'energy_per_atom', 'formation_energy_per_atom', 'energy_above_hull']
    if include_structures:
        fields.append('structure')
    systems = chemical_subsystems(formula_elements(target))
    docs = []
    with MPRester(api_key) as mpr:
        for subsystem in systems:
            docs.extend(mpr.materials.summary.search(chemsys=subsystem, fields=fields))

    rows = []
    for doc in docs:
        formula_raw = getattr(doc, 'formula_pretty', None) if not isinstance(doc, dict) else doc.get('formula_pretty')
        if not formula_raw:
            continue
        f = normalize_formula(formula_raw)
        material_id = str(getattr(doc, 'material_id', None) if not isinstance(doc, dict) else doc.get('material_id'))
        epa = safe_float(getattr(doc, 'energy_per_atom', None) if not isinstance(doc, dict) else doc.get('energy_per_atom'))
        fe = safe_float(getattr(doc, 'formation_energy_per_atom', None) if not isinstance(doc, dict) else doc.get('formation_energy_per_atom'))
        ehull = safe_float(getattr(doc, 'energy_above_hull', None) if not isinstance(doc, dict) else doc.get('energy_above_hull'))
        cif_path = None
        if include_structures:
            structure = getattr(doc, 'structure', None) if not isinstance(doc, dict) else doc.get('structure')
            if structure is not None:
                cif_dir.mkdir(parents=True, exist_ok=True)
                cif_path = str(cif_dir / f'{f}-{material_id}.cif')
                structure.to(filename=cif_path)
        rows.append(
            PhaseEntry(
                formula=f,
                source_id=material_id,
                source='materials_project',
                dft_energy_per_atom=epa,
                dft_energy_per_formula=epa * formula_num_atoms(f) if epa is not None else None,
                formation_energy_per_atom=fe,
                energy_above_hull=ehull,
                cif_path=cif_path,
            ).as_dict()
        )
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(['energy_above_hull', 'formula'], na_position='last').drop_duplicates('formula', keep='first')
        if max_phases:
            out['_n_elements'] = out['formula'].map(lambda f: len(formula_elements(f)))
            mandatory = out[(out['formula'] == target) | (out['_n_elements'] == 1)]
            optional = out.drop(mandatory.index).sort_values(['energy_above_hull', '_n_elements', 'formula'], ascending=[True, False, True], na_position='last')
            out = pd.concat([mandatory, optional.head(max(0, max_phases - len(mandatory)))], ignore_index=True)
            out = out.drop(columns=['_n_elements'])
    system_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(table_path, index=False)
    return out.reset_index(drop=True)



def load_extra_cif_phase_library(cif_dirs: list[str | Path] | None = None, recursive: bool = False) -> pd.DataFrame:
    """Load user-provided CIF files as additional phase-pool entries.

    The entries do not carry MP formation energies or Ehull values. They are
    intended for user-provided local phase pools, e.g. pressure-specific CIFs
    or metastable phases not represented by the default database query.
    """
    rows = []
    for cif_dir in cif_dirs or []:
        root = Path(cif_dir).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f'Extra CIF directory not found: {root}')
        pattern = '**/*.cif' if recursive else '*.cif'
        for cif_path in sorted(root.glob(pattern)):
            try:
                formula = normalize_formula(Structure.from_file(str(cif_path)).composition.reduced_formula)
            except Exception as exc:
                rows.append({
                    'formula': None,
                    'source_id': cif_path.stem,
                    'source': 'extra_cif_failed',
                    'dft_energy_per_atom': None,
                    'dft_energy_per_formula': None,
                    'formation_energy_per_atom': None,
                    'energy_above_hull': None,
                    'n_atoms': None,
                    'elements': None,
                    'cif_path': str(cif_path),
                    'load_error': str(exc),
                })
                continue
            entry = PhaseEntry(
                formula=formula,
                source_id=cif_path.stem,
                source='extra_cif',
                cif_path=str(cif_path),
            ).as_dict()
            entry['load_error'] = None
            rows.append(entry)
    out = pd.DataFrame(rows)
    if len(out) and 'formula' in out:
        out = out[out['formula'].notna()].drop_duplicates('formula', keep='first').reset_index(drop=True)
    return out

def build_phase_pool(
    target_formula: str,
    include_local: bool = True,
    mp_df: pd.DataFrame | None = None,
    external_cif: str | Path | None = None,
    extra_cif_dirs: list[str | Path] | None = None,
    extra_cif_recursive: bool = False,
) -> pd.DataFrame:
    target = normalize_formula(target_formula)
    target_elements = set(formula_elements(target))
    frames = []
    if mp_df is not None and len(mp_df):
        frames.append(mp_df)
    if include_local:
        frames.append(load_local_phase_library())
    extra_df = load_extra_cif_phase_library(extra_cif_dirs, recursive=extra_cif_recursive)
    if len(extra_df):
        frames.append(extra_df)
    if frames:
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.DataFrame()
    if len(df):
        df['formula'] = df['formula'].map(normalize_formula)
        df = df[df['formula'].map(lambda f: set(formula_elements(f)).issubset(target_elements))]
        df = df.drop_duplicates('formula', keep='first')
    if external_cif:
        df = pd.concat([df, pd.DataFrame([external_target_entry(external_cif, target)])], ignore_index=True)
        df = df.drop_duplicates('formula', keep='last')
    return df.sort_values('formula').reset_index(drop=True) if len(df) else df
