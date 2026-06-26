from __future__ import annotations

import itertools
from functools import lru_cache

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from sympy import Matrix, ilcm

from common import BalancedReaction, normalize_formula


@lru_cache(maxsize=500000)
def balance_species(reactants: tuple[str, ...], products: tuple[str, ...]) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    species = list(reactants) + list(products)
    elems = sorted({el for formula in species for el in Composition(formula).as_dict()})
    matrix = []
    for el in elems:
        row = []
        for formula in reactants:
            row.append(int(round(Composition(formula).as_dict().get(el, 0))))
        for formula in products:
            row.append(-int(round(Composition(formula).as_dict().get(el, 0))))
        matrix.append(row)
    ns = Matrix(matrix).nullspace()
    if not ns:
        return None
    vec = ns[0]
    mult = 1
    for value in vec:
        mult = ilcm(mult, value.q)
    coeffs = [int(value * mult) for value in vec]
    if all(x <= 0 for x in coeffs):
        coeffs = [-x for x in coeffs]
    if any(x == 0 for x in coeffs):
        return None
    if not (all(x > 0 for x in coeffs[: len(reactants)]) and all(x > 0 for x in coeffs[len(reactants) :])):
        return None
    vals = [abs(x) for x in coeffs]
    gcd = vals[0]
    for value in vals[1:]:
        gcd = int(np.gcd(gcd, value))
    coeffs = [x // gcd for x in coeffs]
    return tuple(coeffs[: len(reactants)]), tuple(coeffs[len(reactants) :])


def enumerate_formation_reactions(
    target_formula: str,
    phase_formulas: list[str],
    max_reactants: int = 3,
    max_byproducts: int = 2,
    max_coeff: int = 30,
    max_total_coeff: int = 80,
    allowed_byproducts: set[str] | None = None,
) -> list[BalancedReaction]:
    target = normalize_formula(target_formula)
    others = sorted({normalize_formula(f) for f in phase_formulas if normalize_formula(f) != target})
    allowed_byproducts = {normalize_formula(f) for f in allowed_byproducts} if allowed_byproducts else None
    out = []
    seen = set()
    for n_reactants in range(1, min(max_reactants, len(others)) + 1):
        for reactants in itertools.combinations(others, n_reactants):
            remaining = [f for f in others if f not in reactants]
            if allowed_byproducts is not None:
                remaining = [f for f in remaining if f in allowed_byproducts]
            for n_byproducts in range(0, min(max_byproducts, len(remaining)) + 1):
                for byproducts in itertools.combinations(remaining, n_byproducts):
                    products = tuple(sorted((target,) + byproducts))
                    reactants_sorted = tuple(sorted(reactants))
                    if set(products) == set(reactants_sorted):
                        continue
                    balanced = balance_species(reactants_sorted, products)
                    if balanced is None:
                        continue
                    rc, pc = balanced
                    if max(rc + pc) > max_coeff or sum(rc) + sum(pc) > max_total_coeff:
                        continue
                    reaction = BalancedReaction(tuple(zip(reactants_sorted, rc)), tuple(zip(products, pc)))
                    key = reaction.reaction
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(reaction)
    return out


def enumerate_decomposition_reactions(
    target_formula: str,
    phase_formulas: list[str],
    max_products: int = 3,
    max_coeff: int = 30,
    max_total_coeff: int = 80,
) -> list[BalancedReaction]:
    target = normalize_formula(target_formula)
    others = sorted({normalize_formula(f) for f in phase_formulas if normalize_formula(f) != target})
    out = []
    seen = set()
    for n_products in range(1, min(max_products, len(others)) + 1):
        for products in itertools.combinations(others, n_products):
            products_sorted = tuple(sorted(products))
            balanced = balance_species((target,), products_sorted)
            if balanced is None:
                continue
            rc, pc = balanced
            if rc[0] <= 0 or max(rc + pc) > max_coeff or sum(rc) + sum(pc) > max_total_coeff:
                continue
            reaction = BalancedReaction(tuple(zip((target,), rc)), tuple(zip(products_sorted, pc)))
            key = reaction.reaction
            if key in seen:
                continue
            seen.add(key)
            out.append(reaction)
    return out


def reaction_energy(reaction: BalancedReaction, energy_map: dict[str, float]) -> float | None:
    try:
        left = sum(coeff * energy_map[formula] for formula, coeff in reaction.reactants)
        right = sum(coeff * energy_map[formula] for formula, coeff in reaction.products)
    except KeyError:
        return None
    return right - left


def target_coefficient(reaction: BalancedReaction, target_formula: str, direction: str) -> int | None:
    target = normalize_formula(target_formula)
    side = reaction.products if direction == 'formation' else reaction.reactants
    for formula, coeff in side:
        if normalize_formula(formula) == target:
            return coeff
    return None


def reactions_to_frame(
    reactions: list[BalancedReaction],
    energy_maps: dict[str, dict[str, float]],
    direction: str,
    target_formula: str,
) -> pd.DataFrame:
    rows = []
    for reaction in reactions:
        target_coeff = target_coefficient(reaction, target_formula, direction)
        row = {
            'reaction': reaction.reaction,
            'direction': direction,
            'reactants': ' | '.join(f'{c}*{f}' for f, c in reaction.reactants),
            'products': ' | '.join(f'{c}*{f}' for f, c in reaction.products),
            'target_coefficient': target_coeff,
            'num_reactants': len(reaction.reactants),
            'num_products': len(reaction.products),
        }
        for name, energy_map in energy_maps.items():
            energy = reaction_energy(reaction, energy_map)
            row[f'DeltaE_{name}'] = energy
            row[f'DeltaE_{name}_per_target'] = energy / target_coeff if energy is not None and target_coeff else None
        rows.append(row)
    columns = ['reaction', 'direction', 'reactants', 'products', 'target_coefficient', 'num_reactants', 'num_products']
    for name in energy_maps:
        columns.extend([f'DeltaE_{name}', f'DeltaE_{name}_per_target'])
    df = pd.DataFrame(rows, columns=columns)
    sort_col = 'DeltaE_PET-OAM-XI_per_target' if 'DeltaE_PET-OAM-XI_per_target' in df else 'DeltaE_DFT_per_target'
    if len(df) and sort_col in df:
        df = df.sort_values([sort_col, 'reaction'], na_position='last').reset_index(drop=True)
    return df
