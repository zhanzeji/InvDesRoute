from __future__ import annotations

from dataclasses import dataclass
from math import log

import pandas as pd

from analysis import parse_reaction_side
from common import normalize_formula

KB_EV_PER_K = 8.617333262145e-5
STANDARD_T_K = 298.15


@dataclass(frozen=True)
class GasCorrection:
    entropy_eV_per_formula_1000K: float
    note: str


GAS_CORRECTIONS = {
    normalize_formula('O2'): GasCorrection(-0.72, 'oxygen chemical potential decreases strongly at high T'),
    normalize_formula('N2'): GasCorrection(-0.63, 'nitrogen gas entropy term; N2 activation remains kinetic'),
    normalize_formula('H2'): GasCorrection(-0.52, 'hydrogen gas entropy term'),
    normalize_formula('CO2'): GasCorrection(-0.82, 'CO2 gas-release entropy favors carbonate decomposition routes'),
    normalize_formula('CO'): GasCorrection(-0.74, 'CO gas entropy term; redox chemistry remains approximate'),
    normalize_formula('H2O'): GasCorrection(-0.70, 'water vapor entropy term; humidity/condensation not included'),
    normalize_formula('NH3'): GasCorrection(-0.78, 'ammonia gas entropy term; decomposition chemistry not included'),
    normalize_formula('F2'): GasCorrection(-0.65, 'halogen gas entropy term; safety/activity not included'),
    normalize_formula('Cl2'): GasCorrection(-0.68, 'halogen gas entropy term; safety/activity not included'),
    normalize_formula('Br2'): GasCorrection(-0.70, 'halogen reservoir entropy/activity term; phase state approximate'),
    normalize_formula('I2'): GasCorrection(-0.72, 'halogen reservoir entropy/activity term; phase state approximate'),
}


def gas_mu_correction(formula: str, temperature_k: float, pressure_bar: float = 1.0) -> float:
    formula = normalize_formula(formula)
    if formula not in GAS_CORRECTIONS:
        return 0.0
    scaled_entropy = GAS_CORRECTIONS[formula].entropy_eV_per_formula_1000K * (float(temperature_k) - STANDARD_T_K) / (1000.0 - STANDARD_T_K)
    pressure_term = KB_EV_PER_K * float(temperature_k) * log(max(float(pressure_bar), 1e-30))
    return scaled_entropy + pressure_term


def reaction_gas_correction(row: pd.Series, temperature_k: float, gas_pressures_bar: dict[str, float] | None = None) -> tuple[float, str]:
    gas_pressures_bar = {normalize_formula(k): float(v) for k, v in (gas_pressures_bar or {}).items()}
    correction = 0.0
    notes = []
    for formula, coeff in parse_reaction_side(row.get('products', '')):
        if formula in GAS_CORRECTIONS:
            value = coeff * gas_mu_correction(formula, temperature_k, gas_pressures_bar.get(formula, 1.0))
            correction += value
            notes.append(f'+{coeff}{formula}:{value:.3f}')
    for formula, coeff in parse_reaction_side(row.get('reactants', '')):
        if formula in GAS_CORRECTIONS:
            value = -coeff * gas_mu_correction(formula, temperature_k, gas_pressures_bar.get(formula, 1.0))
            correction += value
            notes.append(f'-{coeff}{formula}:{value:.3f}')
    return correction, ';'.join(notes)


def add_condition_corrections(
    formation_df: pd.DataFrame,
    temperature_k: float,
    gas_pressures_bar: dict[str, float] | None = None,
    base_source: str = 'PET_FORM_ALIGNED',
) -> pd.DataFrame:
    if formation_df.empty:
        return formation_df
    df = formation_df.copy()
    base_col = f'DeltaE_{base_source}_per_target'
    if base_col not in df:
        base_col = 'DeltaE_MP_FORM_per_target' if 'DeltaE_MP_FORM_per_target' in df else ''
    corrections = []
    notes = []
    corrected = []
    for _, row in df.iterrows():
        corr_total, note = reaction_gas_correction(row, temperature_k, gas_pressures_bar)
        target_coeff = row.get('target_coefficient')
        corr_per_target = corr_total / float(target_coeff) if pd.notna(target_coeff) and float(target_coeff) else corr_total
        base = row.get(base_col) if base_col else None
        corrections.append(corr_per_target)
        notes.append(note)
        corrected.append(float(base) + corr_per_target if base is not None and pd.notna(base) else None)
    label = f'T{int(round(temperature_k))}K'
    df[f'DeltaG_{base_source}_{label}_gas_per_target'] = corrected
    df[f'GasCorrection_{label}_per_target'] = corrections
    df[f'GasCorrection_{label}_notes'] = notes
    return df
