from __future__ import annotations

import json
import os
import subprocess
import tempfile
import textwrap
from pathlib import Path

import pandas as pd

from common import PET_CHECKPOINT, UPET_PYTHON, UPET_VESIN_CUDA_MAX_PAIRS_PER_POINT, normalize_formula


def load_pet_cache(cache_csv: str | Path) -> dict[str, float]:
    path = Path(cache_csv)
    if not path.exists() or path.stat().st_size == 0:
        return {}
    df = pd.read_csv(path)
    if 'formula' not in df or 'PET_energy_per_formula' not in df:
        return {}
    return {normalize_formula(r['formula']): float(r['PET_energy_per_formula']) for _, r in df.iterrows() if pd.notna(r['PET_energy_per_formula'])}


def _run_pet_batch_subprocess(
    input_csv: str,
    output_json: str,
    checkpoint: str,
    device: str,
    timeout: int,
    python_executable: str | Path | None = None,
) -> None:
    python_path = Path(python_executable or UPET_PYTHON)
    checkpoint_path = Path(checkpoint)
    if not python_path.exists():
        raise RuntimeError(f'UPET python executable not found: {python_path}')
    if not checkpoint_path.exists():
        raise RuntimeError(f'PET checkpoint not found: {checkpoint_path}')

    snippet = textwrap.dedent(f'''
        import json, os
        os.environ['VESIN_CUDA_MAX_PAIRS_PER_POINT'] = os.environ.get('VESIN_CUDA_MAX_PAIRS_PER_POINT', {UPET_VESIN_CUDA_MAX_PAIRS_PER_POINT!r})
        import pandas as pd
        import torch
        from ase.io import read
        from pymatgen.core import Structure, Composition
        from upet.calculator import UPETCalculator

        device = {device!r}
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        calc = UPETCalculator(checkpoint_path={checkpoint!r}, device=device)
        df = pd.read_csv({input_csv!r})
        results = []
        for _, row in df.iterrows():
            formula = Composition(str(row['formula'])).reduced_formula
            cif_path = str(row['cif_path'])
            try:
                atoms = read(cif_path)
                atoms.calc = calc
                e_cell = float(atoms.get_potential_energy())
                structure = Structure.from_file(cif_path)
                _, z = structure.composition.get_reduced_composition_and_factor()
                results.append({{
                    'formula': formula,
                    'cif_path': cif_path,
                    'PET_energy_per_formula': e_cell / float(z),
                    'PET_energy_cell': e_cell,
                    'Z': float(z),
                    'calculator': 'UPETCalculator',
                    'checkpoint_path': {checkpoint!r},
                    'device': device,
                    'status': 'ok',
                    'error': None,
                }})
            except Exception as exc:
                results.append({{
                    'formula': formula,
                    'cif_path': cif_path,
                    'PET_energy_per_formula': None,
                    'PET_energy_cell': None,
                    'Z': None,
                    'calculator': 'UPETCalculator',
                    'checkpoint_path': {checkpoint!r},
                    'device': device,
                    'status': 'failed',
                    'error': str(exc),
                }})
        with open({output_json!r}, 'w') as handle:
            json.dump(results, handle)
    ''')
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as handle:
        handle.write(snippet)
        script = handle.name
    try:
        env = os.environ.copy()
        env.setdefault('VESIN_CUDA_MAX_PAIRS_PER_POINT', UPET_VESIN_CUDA_MAX_PAIRS_PER_POINT)
        proc = subprocess.run([str(python_path), script], text=True, capture_output=True, timeout=timeout, env=env)
    finally:
        Path(script).unlink(missing_ok=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())


def predict_pet_for_phase_table(
    phase_df: pd.DataFrame,
    cache_csv: str | Path,
    device: str = 'auto',
    timeout: int = 3600,
    skip_existing: bool = True,
    max_phases: int | None = None,
    python_executable: str | Path | None = None,
) -> pd.DataFrame:
    cache_csv = Path(cache_csv)
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    existing = load_pet_cache(cache_csv)
    existing_df = pd.read_csv(cache_csv) if cache_csv.exists() and cache_csv.stat().st_size > 0 else pd.DataFrame()

    sub = phase_df[phase_df['cif_path'].notna()].copy() if len(phase_df) else pd.DataFrame()
    sub = sub[sub['cif_path'].map(lambda p: Path(str(p)).exists())] if len(sub) else sub
    sub['formula'] = sub['formula'].map(normalize_formula) if len(sub) else []
    sub = sub.drop_duplicates('formula')
    if skip_existing and len(sub):
        sub = sub[~sub['formula'].isin(existing)]
    if max_phases:
        sub = sub.head(max_phases)
    if sub.empty:
        return existing_df if len(existing_df) else pd.DataFrame(columns=['formula', 'cif_path', 'PET_energy_per_formula'])

    with tempfile.NamedTemporaryFile('w', suffix='.csv', delete=False) as in_handle, tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as out_handle:
        input_csv = in_handle.name
        output_json = out_handle.name
    try:
        sub[['formula', 'cif_path']].to_csv(input_csv, index=False)
        _run_pet_batch_subprocess(input_csv, output_json, str(PET_CHECKPOINT), device, timeout, python_executable=python_executable)
        results = pd.DataFrame(json.loads(Path(output_json).read_text()))
    finally:
        Path(input_csv).unlink(missing_ok=True)
        Path(output_json).unlink(missing_ok=True)

    if len(existing_df):
        existing_df['formula'] = existing_df['formula'].map(normalize_formula)
        results['formula'] = results['formula'].map(normalize_formula)
        combined = pd.concat([existing_df, results], ignore_index=True).drop_duplicates('formula', keep='last')
    else:
        combined = results
    combined = combined.sort_values('formula')
    combined.to_csv(cache_csv, index=False)
    return combined
