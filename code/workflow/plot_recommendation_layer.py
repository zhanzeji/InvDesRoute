#!/home/zhzhan/miniconda3/envs/mattersim/bin/python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from common import PROJECT


CONFIDENCE_ORDER = ['high', 'medium', 'low']
CONFIDENCE_COLORS = {'high': '#009E73', 'medium': '#E69F00', 'low': '#D55E00'}


def plot_recommendation_layer(benchmark_dir: Path, output_prefix: Path) -> None:
    summary = pd.read_csv(benchmark_dir / 'synthesis_recommendation_summary_all.csv')
    category = pd.read_csv(benchmark_dir / 'recommendation_category_summary.csv')
    queue = pd.read_csv(benchmark_dir / 'dft_refinement_queue.csv')
    routes = pd.read_csv(benchmark_dir / 'recommended_routes_top5_all.csv')

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)

    ax = axes[0, 0]
    counts = summary['recommendation_confidence'].value_counts().reindex(CONFIDENCE_ORDER, fill_value=0)
    ax.bar(counts.index, counts.values, color=[CONFIDENCE_COLORS[c] for c in counts.index])
    ax.set_ylabel('Targets')
    ax.set_title('Recommendation confidence')
    for idx, value in enumerate(counts.values):
        ax.text(idx, value + 1, str(int(value)), ha='center', va='bottom', fontsize=10)
    ax.set_ylim(0, max(counts.values) * 1.15)

    ax = axes[0, 1]
    cat_plot = category.set_index('category')[['high_confidence', 'medium_confidence', 'low_confidence']]
    cat_plot = cat_plot.loc[sorted(cat_plot.index)]
    bottom = np.zeros(len(cat_plot))
    x = np.arange(len(cat_plot))
    for col, label in [('high_confidence', 'high'), ('medium_confidence', 'medium'), ('low_confidence', 'low')]:
        values = cat_plot[col].to_numpy()
        ax.bar(x, values, bottom=bottom, label=label, color=CONFIDENCE_COLORS[label])
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels(cat_plot.index, rotation=25, ha='right')
    ax.set_ylabel('Targets')
    ax.set_title('Confidence by chemistry class')
    ax.legend(frameon=False, ncol=3, loc='upper right')

    ax = axes[1, 0]
    scatter_colors = summary['recommendation_confidence'].map(CONFIDENCE_COLORS)
    ax.scatter(
        summary['target_ehull'],
        summary['best_practical_ranked_deltaE_per_target'],
        c=scatter_colors,
        s=45,
        alpha=0.85,
        edgecolor='white',
        linewidth=0.4,
    )
    ax.axvline(0.10, color='0.35', lw=1, ls='--')
    ax.axhline(-0.05, color='0.35', lw=1, ls='--')
    ax.set_xlabel('Target MP Ehull (eV/atom)')
    ax.set_ylabel('Best practical PET ΔE/target (eV)')
    ax.set_title('Decision variables behind the recommendation')

    ax = axes[1, 1]
    if 'precursor_complexity_class' in routes:
        top1 = routes[routes['rank'] == 1].copy()
        complexity_counts = top1['precursor_complexity_class'].value_counts()
        ax.barh(complexity_counts.index, complexity_counts.values, color='#4C78A8')
        for idx, value in enumerate(complexity_counts.values):
            ax.text(value + 0.5, idx, str(int(value)), va='center', fontsize=10)
        ax.set_xlabel('Top-1 recommended routes')
        ax.set_ylabel('Precursor complexity')
        ax.set_title('Chemical legibility of top routes')
    elif len(queue):
        priority_counts = queue['priority'].value_counts().sort_index()
        ax.bar(priority_counts.index.astype(str), priority_counts.values, color='#4C78A8')
        for idx, value in enumerate(priority_counts.values):
            ax.text(idx, value + 0.3, str(int(value)), ha='center', va='bottom', fontsize=10)
        ax.set_xlabel('DFT refinement priority')
        ax.set_ylabel('Targets')
        ax.set_title('Follow-up queue from risk flags')

    fig.suptitle('PET synthesis-pathway recommendation layer', fontsize=15, fontweight='bold')
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix('.png'), dpi=300)
    fig.savefig(output_prefix.with_suffix('.pdf'))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='Plot recommendation-layer summaries.')
    parser.add_argument('--benchmark-dir', default=str(PROJECT / 'results' / 'mp_pet_benchmark_120'))
    parser.add_argument('--output-prefix', default=str(PROJECT / 'results' / 'mp_pet_benchmark_120' / 'recommendation_layer_summary'))
    args = parser.parse_args()
    plot_recommendation_layer(Path(args.benchmark_dir), Path(args.output_prefix))
    print(f'Saved {args.output_prefix}.png/.pdf')


if __name__ == '__main__':
    main()
