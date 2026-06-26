#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_CASE_DIR = PROJECT / 'results/seed80_synthesizability_cases'
DEFAULT_LITERATURE = PROJECT / 'literature_benchmark/ceder_seed80_real_literature_routes.csv'
DEFAULT_OUTDIR = PROJECT / 'literature_benchmark/results/ceder_seed80_diverse_recommendations'

import sys

sys.path.insert(0, str(PROJECT / 'literature_benchmark'))
from evaluate_route_similarity import (  # noqa: E402
    best_similarity,
    best_similarity_in_topk,
    normalize,
    read_csv,
    split_formulas,
)


def load_routes_from_case(target_dir: Path, diverse_top_n: int | None = None) -> list[dict[str, str]]:
    if diverse_top_n is None:
        path = target_dir / 'synthesizability_ranked_routes.csv'
    else:
        path = target_dir / f'synthesizability_diverse_top{diverse_top_n}.csv'
    return read_csv(path)


def label_good(label: object) -> bool:
    return str(label) in {'exact_route_recovered', 'strong_route_family_match', 'moderate_route_family_match'}


def evaluate(case_dir: Path, literature: Path, diverse_top_n: int) -> list[dict[str, object]]:
    rows = []
    for literature_row in read_csv(literature):
        target = normalize(literature_row['target'])
        lit_reactants = split_formulas(literature_row.get('literature_precursors'))
        lit_byproducts = split_formulas(literature_row.get('literature_byproducts'))
        target_dir = case_dir / target
        ranked_routes = load_routes_from_case(target_dir, None)
        diverse_routes = load_routes_from_case(target_dir, diverse_top_n)
        best_original_topn = best_similarity_in_topk(target, lit_reactants, lit_byproducts, ranked_routes, diverse_top_n)
        best_diverse = best_similarity(target, lit_reactants, lit_byproducts, diverse_routes)
        rows.append(
            {
                'target': target,
                'literature_precursors': ';'.join(sorted(lit_reactants)),
                'literature_byproducts': ';'.join(sorted(lit_byproducts)),
                'original_topN_similarity_score': best_original_topn.get('route_similarity_score', ''),
                'original_topN_similarity_label': best_original_topn.get('route_similarity_label', ''),
                'original_topN_synth_rank': best_original_topn.get('recommended_synth_rank', ''),
                'original_topN_reaction': best_original_topn.get('recommended_reaction', ''),
                'diverse_topN_similarity_score': best_diverse.get('route_similarity_score', ''),
                'diverse_topN_similarity_label': best_diverse.get('route_similarity_label', ''),
                'diverse_topN_original_synth_rank': best_diverse.get('recommended_synth_rank', ''),
                'diverse_topN_reaction': best_diverse.get('recommended_reaction', ''),
                'improved_by_diversity': float(best_diverse.get('route_similarity_score') or 0.0) > float(best_original_topn.get('route_similarity_score') or 0.0) + 1e-9,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_svg(summary_rows: list[dict[str, object]], out: Path) -> None:
    width, height = 980, 520

    def esc(value: object) -> str:
        return str(value).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    def text(x: int, y: int, value: object, size: int = 13, weight: str = 'normal') -> str:
        return f'<text x="{x}" y="{y}" font-family="Arial" font-size="{size}" font-weight="{weight}">{esc(value)}</text>'

    def rect(x: int, y: int, w: float, h: int, color: str) -> str:
        return f'<rect x="{x}" y="{y}" width="{w:.1f}" height="{h}" rx="4" fill="{color}"/>'

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        text(36, 42, 'Diversity-aware recommendation improves literature-route visibility', 22, 'bold'),
        text(36, 70, 'Original top-N is score-ranked; diverse top-N keeps one representative from each route family before filling remaining slots.', 12),
    ]
    left, top, maxw = 390, 120, 410
    for idx, row in enumerate(summary_rows):
        n = int(row['n_cases'])
        metrics = [
            (f'top-{row["top_n"]} original moderate+', int(row['original_moderate_or_better']), '#1f78b4'),
            (f'top-{row["top_n"]} diverse moderate+', int(row['diverse_moderate_or_better']), '#2b8a3e'),
            (f'top-{row["top_n"]} improved cases', int(row['improved_cases']), '#d95f02'),
        ]
        y_group = top + idx * 145
        parts.append(text(56, y_group - 18, f'Diverse top-{row["top_n"]}', 15, 'bold'))
        for j, (label, value, color) in enumerate(metrics):
            y = y_group + j * 36
            frac = value / n if n else 0.0
            parts.append(text(72, y + 18, label, 12))
            parts.append(rect(left, y, maxw, 22, '#e9edf5'))
            parts.append(rect(left, y, maxw * frac, 22, color))
            parts.append(text(left + maxw + 15, y + 17, f'{value}/{n} ({frac*100:.1f}%)', 12, 'bold'))
    parts.append(text(40, 470, 'Interpretation: diversity is a display/ranking policy; it does not alter PET energies or reaction enumeration.', 12, 'bold'))
    parts.append('</svg>')
    out.write_text('\n'.join(parts), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate diversity-aware recommendations against DOI-backed literature routes.')
    parser.add_argument('--case-dir', type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument('--literature', type=Path, default=DEFAULT_LITERATURE)
    parser.add_argument('--outdir', type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument('--top-n', type=int, nargs='+', default=[10, 20])
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for top_n in args.top_n:
        rows = evaluate(args.case_dir, args.literature, top_n)
        write_csv(args.outdir / f'diverse_top{top_n}_literature_similarity.csv', rows)
        summary = {
            'top_n': top_n,
            'n_cases': len(rows),
            'original_moderate_or_better': sum(1 for row in rows if label_good(row['original_topN_similarity_label'])),
            'diverse_moderate_or_better': sum(1 for row in rows if label_good(row['diverse_topN_similarity_label'])),
            'original_strong_or_exact': sum(1 for row in rows if str(row['original_topN_similarity_label']) in {'exact_route_recovered', 'strong_route_family_match'}),
            'diverse_strong_or_exact': sum(1 for row in rows if str(row['diverse_topN_similarity_label']) in {'exact_route_recovered', 'strong_route_family_match'}),
            'improved_cases': sum(1 for row in rows if str(row['improved_by_diversity']) == 'True' or row['improved_by_diversity'] is True),
            'mean_original_similarity': round(sum(float(row['original_topN_similarity_score'] or 0.0) for row in rows) / len(rows), 6) if rows else 0.0,
            'mean_diverse_similarity': round(sum(float(row['diverse_topN_similarity_score'] or 0.0) for row in rows) / len(rows), 6) if rows else 0.0,
        }
        summary_rows.append(summary)
    write_csv(args.outdir / 'diverse_recommendation_summary.csv', summary_rows)
    make_svg(summary_rows, args.outdir / 'diverse_recommendation_summary.svg')
    lines = ['# 多样化推荐集对真实文献路线覆盖的影响', '']
    for row in summary_rows:
        lines.append(f'## Top-{row["top_n"]}')
        lines.append('')
        lines.append(f'- 原始 top-{row["top_n"]} moderate-or-better：`{row["original_moderate_or_better"]}/{row["n_cases"]}`。')
        lines.append(f'- 多样化 top-{row["top_n"]} moderate-or-better：`{row["diverse_moderate_or_better"]}/{row["n_cases"]}`。')
        lines.append(f'- 原始 strong/exact：`{row["original_strong_or_exact"]}/{row["n_cases"]}`；多样化 strong/exact：`{row["diverse_strong_or_exact"]}/{row["n_cases"]}`。')
        lines.append(f'- 因 diversity 策略相似度提升的案例：`{row["improved_cases"]}/{row["n_cases"]}`。')
        lines.append(f'- 平均相似度：原始 `{row["mean_original_similarity"]}`，多样化 `{row["mean_diverse_similarity"]}`。')
        lines.append('')
    lines.extend(
        [
            '## 解释',
            '',
            '- 多样化推荐不会改变 PET 能量，也不会新增反应；它改变的是展示和优先检查策略。',
            '- 对真实实验路线而言，研究者通常希望看到多个可解释路线族，而不是 10 条非常相似的碳酸盐/氧化物变体。',
            '- 该策略适合写入 workflow：默认报告给出 score-ranked top routes，同时给出 diversity-ranked actionable routes。',
        ]
    )
    (args.outdir / 'DIVERSE_RECOMMENDATION_ANALYSIS.md').write_text('\n'.join(lines), encoding='utf-8')
    print(summary_rows)
    print(args.outdir / 'DIVERSE_RECOMMENDATION_ANALYSIS.md')


if __name__ == '__main__':
    main()
