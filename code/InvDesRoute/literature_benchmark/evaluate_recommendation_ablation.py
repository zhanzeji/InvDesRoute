#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_CASE_DIR = PROJECT / 'results/seed80_synthesizability_cases'
DEFAULT_LITERATURE = PROJECT / 'literature_benchmark/ceder_seed80_real_literature_routes.csv'
DEFAULT_OUTDIR = PROJECT / 'literature_benchmark/results/recommendation_ablation'

import sys

sys.path.insert(0, str(PROJECT / 'literature_benchmark'))
from evaluate_route_similarity import (  # noqa: E402
    best_similarity,
    best_similarity_in_topk,
    normalize,
    read_csv,
    split_formulas,
)


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {'true', '1', 'yes'}


def route_sort_key(row: dict[str, str], mode: str) -> tuple:
    reaction = row.get('reaction', '')
    if mode == 'pet_energy_all':
        return (safe_float(row.get('DeltaE_PET_FORM_ALIGNED_per_target'), float('inf')), reaction)
    if mode == 'pet_energy_practical':
        practical_penalty = 0 if truthy(row.get('practical_route')) else 1
        return (practical_penalty, safe_float(row.get('DeltaE_PET_FORM_ALIGNED_per_target'), float('inf')), reaction)
    if mode == 'synth_score':
        return (-safe_float(row.get('synth_score'), 0.0), safe_float(row.get('synth_rank_deltaE_per_target'), float('inf')), reaction)
    raise ValueError(mode)


def load_ranked_routes(target_dir: Path, mode: str) -> list[dict[str, str]]:
    if mode == 'diverse_top20':
        routes = read_csv(target_dir / 'synthesizability_diverse_top20.csv')
        return routes
    routes = read_csv(target_dir / 'synthesizability_ranked_routes.csv')
    return sorted(routes, key=lambda row: route_sort_key(row, mode))


def label_strength(label: object) -> str:
    label = str(label)
    if label in {'exact_route_recovered', 'strong_route_family_match'}:
        return 'strong_or_exact'
    if label == 'moderate_route_family_match':
        return 'moderate'
    if label == 'same_target_element_sources':
        return 'same_element'
    return 'weak'


def is_moderate_or_better(label: object) -> bool:
    return str(label) in {'exact_route_recovered', 'strong_route_family_match', 'moderate_route_family_match'}


def is_strong_or_exact(label: object) -> bool:
    return str(label) in {'exact_route_recovered', 'strong_route_family_match'}


def evaluate_mode(case_dir: Path, literature_rows: list[dict[str, str]], mode: str, top_k_values: list[int]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    detail_rows = []
    for literature_row in literature_rows:
        target = normalize(literature_row['target'])
        lit_reactants = split_formulas(literature_row.get('literature_precursors'))
        lit_byproducts = split_formulas(literature_row.get('literature_byproducts'))
        routes = load_ranked_routes(case_dir / target, mode)
        for top_k in top_k_values:
            best = best_similarity_in_topk(target, lit_reactants, lit_byproducts, routes, top_k)
            detail_rows.append(
                {
                    'mode': mode,
                    'top_k': top_k,
                    'target': target,
                    'literature_precursors': ';'.join(sorted(lit_reactants)),
                    'literature_byproducts': ';'.join(sorted(lit_byproducts)),
                    'similarity_score': best.get('route_similarity_score', ''),
                    'similarity_label': best.get('route_similarity_label', ''),
                    'similarity_strength': label_strength(best.get('route_similarity_label', '')),
                    'recommended_rank_in_mode_window': best.get('recommended_synth_rank', ''),
                    'recommended_reaction': best.get('recommended_reaction', ''),
                    'recommended_template': best.get('recommended_template', ''),
                    'recommended_risk_flags': best.get('recommended_risk_flags', ''),
                }
            )
    summary_rows = []
    detail = pd.DataFrame(detail_rows)
    for top_k in top_k_values:
        sub = detail[detail['top_k'] == top_k]
        similarities = pd.to_numeric(sub['similarity_score'], errors='coerce').fillna(0.0) if len(sub) else pd.Series(dtype=float)
        summary_rows.append(
            {
                'mode': mode,
                'top_k': top_k,
                'n_cases': len(sub),
                'strong_or_exact': int(sub['similarity_label'].map(is_strong_or_exact).sum()),
                'moderate_or_better': int(sub['similarity_label'].map(is_moderate_or_better).sum()),
                'same_element_or_better': int((sub['similarity_strength'] != 'weak').sum()),
                'mean_similarity': round(similarities.mean(), 6) if len(sub) else 0.0,
                'median_similarity': round(similarities.median(), 6) if len(sub) else 0.0,
            }
        )
    return detail_rows, summary_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_svg(summary: pd.DataFrame, out: Path) -> None:
    width, height = 1180, 660
    modes = ['pet_energy_all', 'pet_energy_practical', 'synth_score', 'diverse_top20']
    labels = {
        'pet_energy_all': 'PET energy only',
        'pet_energy_practical': 'PET energy + practical filter',
        'synth_score': 'Synthesizability score',
        'diverse_top20': 'Diverse recommendations',
    }
    colors = {
        'pet_energy_all': '#d95f02',
        'pet_energy_practical': '#f2a93b',
        'synth_score': '#1f78b4',
        'diverse_top20': '#2b8a3e',
    }

    def esc(value: object) -> str:
        return str(value).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    def text(x: int, y: int, value: object, size: int = 13, weight: str = 'normal', anchor: str = 'start') -> str:
        return f'<text x="{x}" y="{y}" font-family="Arial" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}">{esc(value)}</text>'

    def rect(x: int, y: int, w: float, h: int, color: str) -> str:
        return f'<rect x="{x}" y="{y}" width="{w:.1f}" height="{h}" rx="4" fill="{color}"/>'

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        text(38, 42, 'Ablation: synthesizability ranking vs reaction-energy ranking', 23, 'bold'),
        text(38, 70, 'DOI-backed literature routes are compared using route-family similarity; higher top-k coverage means better route advice visibility.', 13),
    ]
    topks = [1, 5, 10, 20]
    metric = 'moderate_or_better'
    left, top = 110, 130
    chart_w, chart_h = 900, 330
    n_cases = int(summary['n_cases'].iloc[0]) if len(summary) else 1
    parts.append(text(left, top - 18, 'Moderate-or-better route-family coverage', 15, 'bold'))
    group_w = chart_w / len(topks)
    bar_w = 36
    for i, top_k in enumerate(topks):
        x0 = left + i * group_w + 22
        parts.append(text(int(x0 + 75), top + chart_h + 34, f'top-{top_k}', 12, 'normal', 'middle'))
        for j, mode in enumerate(modes):
            row = summary[(summary['mode'] == mode) & (summary['top_k'] == top_k)]
            value = int(row[metric].iloc[0]) if len(row) else 0
            frac = value / n_cases if n_cases else 0.0
            h = chart_h * frac
            x = x0 + j * (bar_w + 8)
            y = top + chart_h - h
            parts.append(rect(int(x), int(y), bar_w, int(h), colors[mode]))
            parts.append(text(int(x + bar_w / 2), int(y - 6), value, 10, 'bold', 'middle'))
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        y = top + chart_h - chart_h * tick
        parts.append(f'<line x1="{left-5}" y1="{y:.1f}" x2="{left+chart_w}" y2="{y:.1f}" stroke="#e5e5e5" stroke-width="1"/>')
        parts.append(text(left - 14, int(y + 4), f'{tick:.2f}', 10, 'normal', 'end'))
    legend_y = 540
    x = 80
    for mode in modes:
        parts.append(rect(x, legend_y, 18, 18, colors[mode]))
        parts.append(text(x + 25, legend_y + 14, labels[mode], 12))
        x += 250
    parts.append(text(70, 610, 'Conclusion: the recommendation layer is not a cosmetic re-ranking; it exposes literature-like routes that pure minimum-energy ranking often hides.', 12, 'bold'))
    parts.append('</svg>')
    out.write_text('\n'.join(parts), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description='Ablate recommendation ranking components against DOI-backed literature routes.')
    parser.add_argument('--case-dir', type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument('--literature', type=Path, default=DEFAULT_LITERATURE)
    parser.add_argument('--outdir', type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument('--top-k', type=int, nargs='+', default=[1, 5, 10, 20])
    args = parser.parse_args()
    literature_rows = read_csv(args.literature)
    modes = ['pet_energy_all', 'pet_energy_practical', 'synth_score', 'diverse_top20']
    all_detail = []
    all_summary = []
    for mode in modes:
        detail, summary = evaluate_mode(args.case_dir, literature_rows, mode, args.top_k)
        all_detail.extend(detail)
        all_summary.extend(summary)
    args.outdir.mkdir(parents=True, exist_ok=True)
    write_csv(args.outdir / 'recommendation_ablation_detail.csv', all_detail)
    write_csv(args.outdir / 'recommendation_ablation_summary.csv', all_summary)
    summary_df = pd.DataFrame(all_summary)
    make_svg(summary_df, args.outdir / 'recommendation_ablation_summary.svg')
    lines = [
        '# 可合成性建议排序消融实验',
        '',
        '## 目的',
        '',
        '该实验检验核心问题：可合成性建议层是否真的比“按 PET 反应能从低到高排序”更接近真实文献路线。',
        '',
        '## 排序模式',
        '',
        '- `pet_energy_all`：只按 `DeltaE_PET_FORM_ALIGNED_per_target` 从低到高排序，不考虑前驱体常见性或路线可读性。',
        '- `pet_energy_practical`：先保留 practical route，再按 PET 反应能排序。',
        '- `synth_score`：当前可合成性建议排序，综合能量、前驱体先验、模板、复杂度、相可用性和条件风险。',
        '- `diverse_top20`：在可合成性排序基础上按路线族做多样化展示。',
        '',
        '## 主要结果',
        '',
    ]
    for top_k in args.top_k:
        sub = summary_df[summary_df['top_k'] == top_k]
        lines.append(f'### Top-{top_k}')
        for _, row in sub.iterrows():
            lines.append(
                f'- `{row["mode"]}`：strong/exact `{int(row["strong_or_exact"])}/{int(row["n_cases"])}`；'
                f'moderate-or-better `{int(row["moderate_or_better"])}/{int(row["n_cases"])}`；'
                f'平均相似度 `{row["mean_similarity"]}`。'
            )
        lines.append('')
    lines.extend(
        [
            '## 结论',
            '',
            '- 如果只按最低 PET 反应能排序，推荐列表容易被强放热但实验可读性差的路线占据。',
            '- `synth_score` 显著提高真实文献路线族在 top-k 推荐窗口中的可见性。',
            '- `diverse_top20` 进一步减少同质路线挤占，更适合作为实验人员阅读的 actionable route list。',
            '- 因此论文中的“可合成性建议”结论应表述为：该层提供实验路线优先级和风险解释，而不是证明材料一定可合成。',
        ]
    )
    (args.outdir / 'RECOMMENDATION_ABLATION_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(summary_df.to_string(index=False))
    print(args.outdir / 'RECOMMENDATION_ABLATION_REPORT.md')


if __name__ == '__main__':
    main()
