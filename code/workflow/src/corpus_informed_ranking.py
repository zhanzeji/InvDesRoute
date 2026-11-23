from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition

from analysis import parse_reaction_side
from common import normalize_formula
PROCESS_ELEMENTS = {"H", "C", "N", "O"}
GAS_FORMULAS = {
    normalize_formula(formula)
    for formula in ["O2", "N2", "H2", "NH3", "CO2", "CO", "H2O", "Cl2", "F2", "Br2", "I2"]
}


@lru_cache(maxsize=None)
def _elements(formula: str) -> frozenset[str]:
    return frozenset(str(element) for element in Composition(formula).elements)


def _source_elements(target: str) -> set[str]:
    target_elements = _elements(target)
    return target_elements - PROCESS_ELEMENTS or target_elements


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _counter_jaccard(left: Counter[str], right: Counter[str]) -> float:
    keys = set(left) | set(right)
    denominator = sum(max(left[key], right[key]) for key in keys)
    return sum(min(left[key], right[key]) for key in keys) / denominator if denominator else 1.0


@lru_cache(maxsize=None)
def _route_prior_class(formula: str, target: str) -> str:
    formula = normalize_formula(formula)
    formula_elements = _elements(formula)
    target_elements = _elements(target)
    composition = Composition(formula).as_dict()
    if formula in GAS_FORMULAS:
        return "gas_reservoir"
    if len(formula_elements) == 1:
        return "elemental_source"
    if {"C", "O"}.issubset(formula_elements) and formula_elements - {"C", "O"}:
        if composition.get("O", 0.0) >= 2.0 * composition.get("C", 0.0):
            return "carbonate_source"
    if "N" in formula_elements and ({"O", "H"} & formula_elements):
        return "ammonium_phosphate" if {"P", "O"}.issubset(formula_elements) else "nitrate_or_ammonium_salt"
    if {"H", "O"}.issubset(formula_elements) and formula_elements - {"H", "O"}:
        return "hydroxide_or_hydrate"
    if {"P", "O"}.issubset(formula_elements):
        return "phosphate_source"
    if "O" in formula_elements and len(formula_elements) == 2:
        return "binary_oxide_source"
    if len(formula_elements) == 2 and formula_elements & target_elements:
        return "binary_target_element_source"
    if formula_elements <= target_elements:
        return "target_element_compound"
    return "other_precursor"


@dataclass(frozen=True)
class FrozenCorpusPrior:
    formula_counts: dict[str, dict[str, int]]
    class_counts: dict[str, dict[str, int]]
    excluded_targets: tuple[str, ...]
    source: str

    @classmethod
    def load(cls, path: str | Path) -> "FrozenCorpusPrior":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            formula_counts=payload["formula_counts"],
            class_counts=payload["class_counts"],
            excluded_targets=tuple(payload.get("excluded_targets", [])),
            source=payload.get("source", ""),
        )

    def score(self, target: str, reactants: set[str]) -> float:
        values = []
        for element in sorted(_source_elements(target)):
            candidates = [formula for formula in reactants if element in _elements(formula)]
            if not candidates:
                values.append(0.0)
                continue
            formula_counter = self.formula_counts.get(element, {})
            maximum = math.log1p(max(formula_counter.values())) if formula_counter else 1.0
            exact = max(math.log1p(formula_counter.get(formula, 0)) / maximum for formula in candidates)
            class_counter = self.class_counts.get(element, {})
            total = sum(class_counter.values())
            class_support = 0.0
            if total:
                class_support = max(
                    class_counter.get(_route_prior_class(formula, target), 0) / total for formula in candidates
                )
            values.append(0.82 * exact + 0.18 * math.sqrt(class_support))
        represented = set().union(*(_elements(formula) for formula in reactants)) if reactants else set()
        coverage = len(represented & _elements(target)) / len(_elements(target))
        return float(np.clip(0.90 * np.mean(values) + 0.10 * coverage, 0.0, 1.0))


def _redundancy(target: str, left: pd.Series, right: pd.Series) -> float:
    left_formulas = {formula for formula, _ in parse_reaction_side(left.get("reactants", ""))}
    right_formulas = {formula for formula, _ in parse_reaction_side(right.get("reactants", ""))}
    left_classes = Counter(_route_prior_class(formula, target) for formula in left_formulas)
    right_classes = Counter(_route_prior_class(formula, target) for formula in right_formulas)
    return 0.70 * _jaccard(left_formulas, right_formulas) + 0.30 * _counter_jaccard(left_classes, right_classes)


def _route_descriptor(target: str, reactants: object) -> tuple[set[str], Counter[str]]:
    formulas = {formula for formula, _ in parse_reaction_side(reactants)}
    return formulas, Counter(_route_prior_class(formula, target) for formula in formulas)


def _descriptor_redundancy(
    left: tuple[set[str], Counter[str]],
    right: tuple[set[str], Counter[str]],
) -> float:
    return 0.70 * _jaccard(left[0], right[0]) + 0.30 * _counter_jaccard(left[1], right[1])


def _energy_tiebreak(value: object) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return -float(numeric) if pd.notna(numeric) else -1e12


def rerank_routes(
    scored: pd.DataFrame,
    target_formula: str,
    prior: FrozenCorpusPrior,
    prior_weight: float = 0.30,
    diversity_penalty: float = 0.16,
    display_k: int = 20,
) -> pd.DataFrame:
    if scored.empty:
        return scored
    target = normalize_formula(target_formula)
    ranked = scored.copy().reset_index(drop=True)
    ranked["corpus_precursor_prior"] = [
        prior.score(target, {formula for formula, _ in parse_reaction_side(row.get("reactants", ""))})
        for _, row in ranked.iterrows()
    ]
    ranked["optimized_rank_score"] = (
        (1.0 - prior_weight) * ranked["synth_score"].astype(float)
        + prior_weight * ranked["corpus_precursor_prior"].astype(float)
    )
    descriptors = {
        index: _route_descriptor(target, ranked.at[index, "reactants"])
        for index in ranked.index
    }
    selected: list[int] = []
    remaining = list(ranked.index)
    while remaining and len(selected) < display_k:
        best = max(
            remaining,
            key=lambda index: (
                float(ranked.at[index, "optimized_rank_score"])
                - diversity_penalty
                * max(
                    (_descriptor_redundancy(descriptors[index], descriptors[chosen]) for chosen in selected),
                    default=0.0,
                ),
                float(ranked.at[index, "optimized_rank_score"]),
                _energy_tiebreak(ranked.at[index, "synth_rank_deltaE_per_target"]),
                str(ranked.at[index, "reaction"]),
            ),
        )
        selected.append(best)
        remaining.remove(best)
    remaining = sorted(
        remaining,
        key=lambda index: (
            -float(ranked.at[index, "optimized_rank_score"]),
            _energy_tiebreak(ranked.at[index, "synth_rank_deltaE_per_target"]),
            str(ranked.at[index, "reaction"]),
        ),
    )
    ranked = ranked.loc[selected + remaining].reset_index(drop=True)
    ranked["synth_rank"] = np.arange(1, len(ranked) + 1)
    ranked["ranking_policy"] = "corpus_informed_mmr"
    ranked["corpus_prior_weight"] = prior_weight
    ranked["diversity_penalty"] = diversity_penalty
    return ranked
