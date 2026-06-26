from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Structure

PROJECT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get('INVDESROUTE_ROOT', PROJECT.parent))
HECHENGYUCE = Path(os.environ.get('INVDESROUTE_LOCAL_MP_DIR', ROOT / 'hechengyuce'))
LOCAL_MP_ENERGIES = Path(os.environ.get('INVDESROUTE_LOCAL_MP_ENERGIES', HECHENGYUCE / 'mp_energies.csv'))
LOCAL_MP_CIFS = Path(os.environ.get('INVDESROUTE_LOCAL_MP_CIFS', HECHENGYUCE / 'mp_cifs'))
PET_CHECKPOINT = Path(os.environ.get('PET_OAM_XL_CHECKPOINT', ROOT / 'pet-oam-xi' / 'pet-oam-xl-v1.0.0.ckpt'))
UPET_PYTHON = os.environ.get('UPET_PYTHON', 'python')
UPET_VESIN_CUDA_MAX_PAIRS_PER_POINT = os.environ.get('UPET_VESIN_CUDA_MAX_PAIRS_PER_POINT', '4096')


def normalize_formula(formula: str) -> str:
    return Composition(str(formula)).reduced_formula


def formula_elements(formula: str) -> tuple[str, ...]:
    return tuple(sorted(Composition(str(formula)).as_dict()))


def chemical_system(formula: str) -> str:
    return '-'.join(formula_elements(formula))


def formula_num_atoms(formula: str) -> int:
    return int(Composition(str(formula)).num_atoms)


def formula_from_cif(cif_path: str | Path) -> str:
    return Structure.from_file(str(cif_path)).composition.reduced_formula


def safe_float(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


@dataclass(frozen=True)
class PhaseEntry:
    formula: str
    source_id: str | None
    source: str
    dft_energy_per_atom: float | None = None
    dft_energy_per_formula: float | None = None
    formation_energy_per_atom: float | None = None
    energy_above_hull: float | None = None
    cif_path: str | None = None

    @property
    def n_atoms(self) -> int:
        return formula_num_atoms(self.formula)

    @property
    def elements(self) -> tuple[str, ...]:
        return formula_elements(self.formula)

    def as_dict(self) -> dict:
        return {
            'formula': self.formula,
            'source_id': self.source_id,
            'source': self.source,
            'dft_energy_per_atom': self.dft_energy_per_atom,
            'dft_energy_per_formula': self.dft_energy_per_formula,
            'formation_energy_per_atom': self.formation_energy_per_atom,
            'energy_above_hull': self.energy_above_hull,
            'n_atoms': self.n_atoms,
            'elements': '|'.join(self.elements),
            'cif_path': self.cif_path,
        }


@dataclass(frozen=True)
class BalancedReaction:
    reactants: tuple[tuple[str, int], ...]
    products: tuple[tuple[str, int], ...]

    @property
    def reaction(self) -> str:
        def fmt(side: Iterable[tuple[str, int]]) -> str:
            parts = []
            for formula, coeff in side:
                parts.append(f'{coeff} {formula}' if coeff != 1 else formula)
            return ' + '.join(parts)
        return f'{fmt(self.reactants)} -> {fmt(self.products)}'

    def species(self) -> set[str]:
        return {f for f, _ in self.reactants} | {f for f, _ in self.products}


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
