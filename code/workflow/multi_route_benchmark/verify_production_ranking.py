#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis import parse_reaction_side
from common import normalize_formula
from synthesizability import add_synthesizability_scores


ROOT = Path(__file__).resolve().parent


def precursor_set(value: object, separator: str | None = None) -> frozenset[str]:
    if separator:
        return frozenset(normalize_formula(item) for item in str(value).split(separator) if item)
    return frozenset(normalize_formula(formula) for formula, _ in parse_reaction_side(value))


def main() -> None:
    result_dir = ROOT / 'results_equiformer_v3' / 'optimized_ranking'
    expected = pd.read_csv(result_dir / 'frozen_test_predictions.csv').fillna('')
    prior_path = result_dir / 'frozen_corpus_prior.json'
    mismatches = []
    for target, group in expected.groupby('target', sort=True):
        case_dir = ROOT / 'cases_blind_equiformer_v3' / target
        ranked = add_synthesizability_scores(
            pd.read_csv(case_dir / 'formation_reactions.csv'),
            pd.read_csv(case_dir / 'phase_pool_raw.csv'),
            target,
            corpus_prior_path=str(prior_path),
        ).head(5)
        actual = [precursor_set(value) for value in ranked['reactants']]
        frozen = [precursor_set(value, ';') for value in group.sort_values('rank')['precursors']]
        if actual != frozen:
            mismatches.append(target)
    if mismatches:
        raise RuntimeError(f'Production/frozen top-5 mismatch for: {mismatches}')
    print(f'Production ranking matches frozen output for {expected.target.nunique()}/24 targets.')


if __name__ == '__main__':
    main()
