#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))

from common import normalize_formula  # noqa: E402
from synthesizability import write_synthesizability_outputs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--case-dir', type=Path, default=PROJECT / 'results/precursor_expanded_targeted_final')
    args = parser.parse_args()
    count = 0
    for target_dir in sorted(path for path in args.case_dir.iterdir() if path.is_dir()):
        phase_path = target_dir / 'phase_stability.csv'
        formation_path = target_dir / 'formation_reactions.csv'
        if not phase_path.exists() or not formation_path.exists():
            continue
        phase_df = pd.read_csv(phase_path)
        formation_df = pd.read_csv(formation_path)
        target = normalize_formula(target_dir.name)
        scored = write_synthesizability_outputs(target_dir, target, phase_df, formation_df)
        print(target, len(scored), target_dir / 'synthesizability_ranked_routes.csv')
        count += 1
    print(f'refreshed={count}')


if __name__ == '__main__':
    main()
