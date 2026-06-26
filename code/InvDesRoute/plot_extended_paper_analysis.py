#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CATEGORY_ORDER = [
    'Halides',
    'Binary oxides',
    'Nitrides/carbides/chalcogenides',
    'Ternary oxides',
    'Battery/salts/silicates',
]

CATEGORY_COLORS = {
    'Halides': '#4C78A8',
    'Binary oxides': '#F58518',
    'Nitrides/carbides/chalcogenides': '#54A24B',
    'Ternary oxides': '#B279A2',
    'Battery/salts/silicates': '#E45756',
}

COMPLEXITY_COLORS = {
    'elemental_precursors': '#009E73',
    'simple_compound_precursors': '#56B4E9',
    'complex_precursors': '#D55E00',
}


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({'true', '1', 'yes'})


def load_tables(root: Path) -> dict[str, pd.DataFrame]:
    return {
        'summary': pd.read_csv(root / 'mp_pet_benchmark_summary_with_categories.csv'),
        'pooled': pd.read_csv(root / 'mp_pet_reaction_data.csv'),
        'recommendations': pd.read_csv(root / 'synthesis_recommendation_summary_all.csv'),
        'routes': pd.read_csv(root / 'recommended_routes_top5_all.csv'),
        'queue': pd.read_csv(root / 'dft_refinement_queue.csv'),
        'category': pd.read_csv(root / 'mp_pet_benchmark_category_summary.csv'),
    }


def plot_uncertainty_and_legibility(tables: dict[str, pd.DataFrame], output_prefix: Path) -> None:
    summary = tables['summary'].copy()
    routes = tables['routes'].copy()
    queue = tables['queue'].copy()
    summary['top1_bool'] = bool_series(summary['top1_match'])
    summary['practical_fraction'] = summary['n_practical_formation_reactions'] / summary['n_formation_reactions']

    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.4), constrained_layout=True)

    ax = axes[0, 0]
    for category in CATEGORY_ORDER:
        sub = summary[summary['category'] == category]
        matched = sub[sub['top1_bool']]
        mismatched = sub[~sub['top1_bool']]
        ax.scatter(
            matched['n_formation_reactions'],
            matched['mae_pet_vs_mp_form'],
            s=42,
            color=CATEGORY_COLORS[category],
            alpha=0.78,
            label=category,
            edgecolor='white',
            linewidth=0.4,
        )
        ax.scatter(
            mismatched['n_formation_reactions'],
            mismatched['mae_pet_vs_mp_form'],
            s=86,
            color=CATEGORY_COLORS[category],
            marker='X',
            edgecolor='black',
            linewidth=0.6,
        )
    ax.set_xscale('log')
    ax.set_xlabel('Enumerated formation reactions per target')
    ax.set_ylabel('PET vs MP_FORM MAE (eV/target)')
    ax.set_title('Network size and target-level error')
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    ax.text(
        0.98,
        0.96,
        'X: top-1 mismatch',
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=9,
        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.8'),
    )

    ax = axes[0, 1]
    overlap_counts = summary['top5_overlap'].value_counts().sort_index()
    ax.bar(overlap_counts.index.astype(int), overlap_counts.values, color='#4C78A8')
    ax.set_xlabel('PET/MP_FORM top-5 overlap')
    ax.set_ylabel('Targets')
    ax.set_title('Top-5 agreement preserves alternatives')
    for xval, yval in zip(overlap_counts.index.astype(int), overlap_counts.values):
        ax.text(xval, yval + 0.8, str(int(yval)), ha='center', va='bottom', fontsize=9)
    ax.set_xticks(range(0, 6))

    ax = axes[1, 0]
    crosstab = pd.crosstab(routes[routes['rank'] == 1]['category'], routes[routes['rank'] == 1]['precursor_complexity_class'])
    crosstab = crosstab.reindex(CATEGORY_ORDER).fillna(0)
    complexity_order = ['elemental_precursors', 'simple_compound_precursors', 'complex_precursors']
    bottom = np.zeros(len(crosstab))
    x = np.arange(len(crosstab))
    for key in complexity_order:
        values = crosstab[key].to_numpy() if key in crosstab else np.zeros(len(crosstab))
        ax.bar(x, values, bottom=bottom, label=key.replace('_', ' '), color=COMPLEXITY_COLORS[key])
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORY_ORDER, rotation=24, ha='right')
    ax.set_ylabel('Top-1 recommended routes')
    ax.set_title('Precursor complexity by chemistry class')
    ax.legend(frameon=False, fontsize=8, loc='upper left')

    ax = axes[1, 1]
    queue_with_cat = queue.merge(summary[['target', 'category']], on='target', how='left')
    queue_counts = queue_with_cat.groupby('category')['target'].count().reindex(CATEGORY_ORDER, fill_value=0)
    ax.barh(queue_counts.index, queue_counts.values, color=[CATEGORY_COLORS[c] for c in queue_counts.index])
    ax.set_xlabel('Targets in DFT/manual refinement queue')
    ax.set_title('Where follow-up validation is concentrated')
    for idx, value in enumerate(queue_counts.values):
        ax.text(value + 0.15, idx, str(int(value)), va='center', fontsize=9)

    fig.suptitle('Uncertainty and route legibility in PET-guided synthesis screening', fontsize=16, fontweight='bold')
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix('.png'), dpi=300)
    fig.savefig(output_prefix.with_suffix('.pdf'))
    plt.close(fig)


def derived_text_metrics(tables: dict[str, pd.DataFrame]) -> dict[str, object]:
    summary = tables['summary'].copy()
    routes = tables['routes'].copy()
    queue = tables['queue'].copy()
    summary['top1_bool'] = bool_series(summary['top1_match'])
    metrics: dict[str, object] = {}
    metrics['n_mismatch'] = int((~summary['top1_bool']).sum())
    metrics['n_top5_ge3'] = int((summary['top5_overlap'] >= 3).sum())
    metrics['n_top5_full'] = int((summary['top5_overlap'] == 5).sum())
    metrics['mismatch_by_category'] = {
        category: int((~summary.loc[summary['category'] == category, 'top1_bool']).sum())
        for category in CATEGORY_ORDER
    }
    metrics['mae_matched_mean'] = float(summary.loc[summary['top1_bool'], 'mae_pet_vs_mp_form'].mean())
    metrics['mae_mismatch_mean'] = float(summary.loc[~summary['top1_bool'], 'mae_pet_vs_mp_form'].mean())
    metrics['mae_matched_median'] = float(summary.loc[summary['top1_bool'], 'mae_pet_vs_mp_form'].median())
    metrics['mae_mismatch_median'] = float(summary.loc[~summary['top1_bool'], 'mae_pet_vs_mp_form'].median())
    metrics['spearman_reactions_mae'] = float(summary['n_formation_reactions'].corr(summary['mae_pet_vs_mp_form'], method='spearman'))
    metrics['spearman_phases_mae'] = float(summary['n_phases'].corr(summary['mae_pet_vs_mp_form'], method='spearman'))
    top1 = routes[routes['rank'] == 1]
    metrics['complexity_counts'] = top1['precursor_complexity_class'].value_counts().to_dict()
    metrics['queue_by_category'] = queue.merge(summary[['target', 'category']], on='target', how='left').groupby('category')['target'].count().reindex(CATEGORY_ORDER, fill_value=0).astype(int).to_dict()
    reason_counts: Counter[str] = Counter()
    for value in queue['reason'].fillna(''):
        for reason in str(value).split(';'):
            if reason:
                reason_counts[reason] += 1
    metrics['queue_reason_counts'] = dict(reason_counts)
    return metrics


def write_metrics_markdown(metrics: dict[str, object], output_path: Path) -> None:
    lines = [
        '# Extended paper-analysis metrics',
        '',
        f'- Top-1 mismatches: {metrics["n_mismatch"]}.',
        f'- Targets with PET/MP_FORM top-5 overlap >= 3: {metrics["n_top5_ge3"]}.',
        f'- Targets with full top-5 overlap: {metrics["n_top5_full"]}.',
        f'- Mean MAE for top-1 matched targets: {metrics["mae_matched_mean"]:.3f} eV/target.',
        f'- Mean MAE for top-1 mismatched targets: {metrics["mae_mismatch_mean"]:.3f} eV/target.',
        f'- Median MAE for top-1 matched targets: {metrics["mae_matched_median"]:.3f} eV/target.',
        f'- Median MAE for top-1 mismatched targets: {metrics["mae_mismatch_median"]:.3f} eV/target.',
        f'- Spearman correlation between formation-reaction count and MAE: {metrics["spearman_reactions_mae"]:.3f}.',
        f'- Spearman correlation between phase count and MAE: {metrics["spearman_phases_mae"]:.3f}.',
        f'- Mismatch by category: {metrics["mismatch_by_category"]}.',
        f'- Top-1 route precursor complexity: {metrics["complexity_counts"]}.',
        f'- DFT/manual refinement queue by category: {metrics["queue_by_category"]}.',
        f'- DFT/manual refinement queue reasons: {metrics["queue_reason_counts"]}.',
    ]
    output_path.write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate extended paper analysis figures from benchmark tables.')
    parser.add_argument('--benchmark-dir', default='../data/support_tables/synthesis_pathway_screening_mp_pet_benchmark_120')
    parser.add_argument('--figure-prefix', default='figures/Fig6_uncertainty_route_legibility')
    parser.add_argument('--metrics-md', default='../data/support_tables/synthesis_pathway_screening_mp_pet_benchmark_120/EXTENDED_PAPER_ANALYSIS.md')
    args = parser.parse_args()
    tables = load_tables(Path(args.benchmark_dir))
    plot_uncertainty_and_legibility(tables, Path(args.figure_prefix))
    write_metrics_markdown(derived_text_metrics(tables), Path(args.metrics_md))
    print(f'Saved {args.figure_prefix}.png/.pdf')
    print(f'Saved {args.metrics_md}')


if __name__ == '__main__':
    main()
