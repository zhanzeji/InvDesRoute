from __future__ import annotations

import itertools

import numpy as np
from pymatgen.core import Composition

from common import BalancedReaction, normalize_formula
from reactions import balance_species


COMMON_RESERVOIRS = {normalize_formula(formula) for formula in ['O2', 'N2', 'H2', 'NH3', 'CO2', 'CO', 'H2O']}
COMMON_EXTRA_ELEMENTS = {'C', 'H', 'N', 'O'}


def formula_elements(formula: str) -> set[str]:
    return set(Composition(formula).as_dict())


def likely_byproduct_sets(
    extra_elements: set[str],
    reactant_elements: set[str],
    target_elements: set[str],
    available: set[str],
    max_byproducts: int,
) -> list[tuple[str, ...]]:
    options: list[list[tuple[str, ...]]] = []
    if 'C' in extra_elements:
        options.append([('CO2',), ('CO',), ('CO2', 'CO')])
    if 'H' in extra_elements:
        options.append([('H2',), ('H2O',), ('H2', 'H2O')])
    if 'N' in extra_elements:
        options.append([('N2',), ('H3N',)])
    if not options:
        candidates = [tuple()]
    else:
        candidates = []
        for product in itertools.product(*options):
            merged = tuple(sorted({normalize_formula(formula) for group in product for formula in group}))
            candidates.append(merged)

    allow_optional_o2 = 'O' in reactant_elements or 'O' in target_elements
    expanded = set()
    for candidate in candidates:
        normalized = tuple(sorted(normalize_formula(formula) for formula in candidate))
        expanded.add(normalized)
        if allow_optional_o2:
            expanded.add(tuple(sorted({*normalized, normalize_formula('O2')})))
        if 'H' in reactant_elements:
            expanded.add(tuple(sorted({*normalized, normalize_formula('H2')})))
        if 'N' in reactant_elements and 'N' not in target_elements:
            expanded.add(tuple(sorted({*normalized, normalize_formula('N2')})))
    out = []
    for candidate in sorted(expanded, key=lambda item: (len(item), item)):
        if len(candidate) > max_byproducts:
            continue
        if set(candidate).issubset(available):
            out.append(candidate)
    return out


def enumerate_targeted_formation_reactions(
    target_formula: str,
    phase_formulas: list[str],
    reactant_candidates: set[str],
    byproduct_candidates: set[str],
    max_reactants: int = 3,
    max_byproducts: int = 4,
    max_coeff: int = 60,
    max_total_coeff: int = 120,
) -> list[BalancedReaction]:
    target = normalize_formula(target_formula)
    available = {normalize_formula(formula) for formula in phase_formulas}
    target_elements = formula_elements(target)
    allowed_reactant_elements = target_elements | COMMON_EXTRA_ELEMENTS
    reactants_pool = []
    for formula in sorted({normalize_formula(formula) for formula in reactant_candidates} & available - {target}):
        elements = formula_elements(formula)
        if formula not in COMMON_RESERVOIRS and not elements.issubset(allowed_reactant_elements):
            continue
        reactants_pool.append(formula)
    byproduct_pool = {normalize_formula(formula) for formula in byproduct_candidates} & available - {target}
    out = []
    seen = set()
    for n_reactants in range(1, min(max_reactants, len(reactants_pool)) + 1):
        for reactants in itertools.combinations(reactants_pool, n_reactants):
            reactant_elements = set().union(*(formula_elements(formula) for formula in reactants))
            if not target_elements.issubset(reactant_elements):
                continue
            extra_elements = reactant_elements - target_elements
            if not extra_elements.issubset(COMMON_EXTRA_ELEMENTS):
                continue
            for byproducts in likely_byproduct_sets(extra_elements, reactant_elements, target_elements, byproduct_pool, max_byproducts):
                products = tuple(sorted((target,) + byproducts))
                reactants_sorted = tuple(sorted(reactants))
                balanced = balance_species(reactants_sorted, products)
                if balanced is None:
                    continue
                reactant_coeffs, product_coeffs = balanced
                if max(reactant_coeffs + product_coeffs) > max_coeff:
                    continue
                if sum(reactant_coeffs) + sum(product_coeffs) > max_total_coeff:
                    continue
                coefficients = [*reactant_coeffs, *product_coeffs]
                gcd = coefficients[0]
                for value in coefficients[1:]:
                    gcd = int(np.gcd(gcd, value))
                if gcd > 1:
                    reactant_coeffs = tuple(value // gcd for value in reactant_coeffs)
                    product_coeffs = tuple(value // gcd for value in product_coeffs)
                reaction = BalancedReaction(tuple(zip(reactants_sorted, reactant_coeffs)), tuple(zip(products, product_coeffs)))
                if reaction.reaction in seen:
                    continue
                seen.add(reaction.reaction)
                out.append(reaction)
    return out
