#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


STRONG_OR_EXACT = {'exact_route_recovered', 'strong_route_family_match'}
MODERATE_OR_BETTER = STRONG_OR_EXACT | {'moderate_route_family_match'}


def main() -> None:
    parser = argparse.ArgumentParser(description='Combine recommendation-ablation detail tables.')
    parser.add_argument('--inputs', type=Path, nargs='+', required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    args = parser.parse_args()

    frames = []
    for path in args.inputs:
        frame = pd.read_csv(path)
        frame['source_file'] = str(path)
        frames.append(frame)
    detail = pd.concat(frames, ignore_index=True)
    detail['similarity_score'] = pd.to_numeric(detail['similarity_score'], errors='coerce').fillna(0.0)

    rows = []
    for (mode, top_k), group in detail.groupby(['mode', 'top_k'], sort=True):
        labels = group['similarity_label'].fillna('').astype(str)
        rows.append(
            {
                'mode': mode,
                'top_k': int(top_k),
                'n_cases': len(group),
                'strong_or_exact': int(labels.isin(STRONG_OR_EXACT).sum()),
                'moderate_or_better': int(labels.isin(MODERATE_OR_BETTER).sum()),
                'same_element_or_better': int((group['similarity_strength'].fillna('weak') != 'weak').sum()),
                'mean_similarity': round(group['similarity_score'].mean(), 6),
                'median_similarity': round(group['similarity_score'].median(), 6),
            }
        )
    summary = pd.DataFrame(rows)
    args.outdir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.outdir / 'combined_recommendation_ablation_detail.csv', index=False)
    summary.to_csv(args.outdir / 'combined_recommendation_ablation_summary.csv', index=False)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
