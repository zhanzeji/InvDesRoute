#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from common import PROJECT, normalize_formula
from analysis import write_recommendation


def optional_category_map(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    df = pd.read_csv(path)
    if not {'target', 'category'}.issubset(df.columns):
        return {}
    return {normalize_formula(row['target']): row['category'] for _, row in df.iterrows() if pd.notna(row.get('target'))}


def priority_from_summary(row: pd.Series) -> int:
    risk_flags = str(row.get('risk_flags', ''))
    confidence = str(row.get('recommendation_confidence', ''))
    top1_match = str(row.get('top1_pet_mp_form_match', '')).lower() == 'true'
    if confidence == 'low':
        return 1
    if 'exergonic_decomposition_competition' in risk_flags or 'no_exergonic_practical' in risk_flags:
        return 1
    if confidence == 'medium':
        return 2
    if not top1_match:
        return 2
    return 3


def queue_reason(row: pd.Series) -> str:
    reasons = []
    confidence = str(row.get('recommendation_confidence', ''))
    risk_flags = str(row.get('risk_flags', 'none'))
    top1_match = str(row.get('top1_pet_mp_form_match', '')).lower() == 'true'
    if confidence != 'high':
        reasons.append(f'{confidence}_confidence')
    if risk_flags and risk_flags != 'none':
        reasons.extend(flag for flag in risk_flags.split(';') if flag)
    if not top1_match:
        reasons.append('pet_mp_form_top1_mismatch')
    if not reasons:
        reasons.append('routine_validation')
    return ';'.join(dict.fromkeys(reasons))


def build_refinement_queue(summary: pd.DataFrame, routes: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows = []
    first_routes = routes[routes['rank'] == 1].copy() if len(routes) and 'rank' in routes else pd.DataFrame()
    route_lookup = first_routes.set_index('target').to_dict('index') if len(first_routes) else {}
    for _, row in summary.iterrows():
        priority = priority_from_summary(row)
        reason = queue_reason(row)
        route = route_lookup.get(row['target'], {})
        if priority <= 2 or reason != 'routine_validation':
            rows.append(
                {
                    'target': row['target'],
                    'priority': priority,
                    'recommendation_confidence': row.get('recommendation_confidence'),
                    'classification': row.get('classification'),
                    'reason': reason,
                    'target_ehull': row.get('target_ehull'),
                    'best_practical_ranked_reaction': row.get('best_practical_ranked_reaction'),
                    'best_practical_ranked_deltaE_per_target': row.get('best_practical_ranked_deltaE_per_target'),
                    'top_recommended_reaction': route.get('reaction'),
                    'pet_deltaE_per_target': route.get('pet_deltaE_per_target'),
                    'mp_form_deltaE_per_target': route.get('mp_form_deltaE_per_target'),
                    'abs_pet_mp_form_gap': route.get('abs_pet_mp_form_gap'),
                    'risk_flags': row.get('risk_flags'),
                }
            )
    queue = pd.DataFrame(rows)
    if len(queue):
        queue = queue.sort_values(['priority', 'recommendation_confidence', 'target']).reset_index(drop=True)
    return queue


def category_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty or 'category' not in summary:
        return pd.DataFrame()
    rows = []
    for category, sub in summary.groupby('category', dropna=False):
        top1 = sub['top1_pet_mp_form_match'].astype(str).str.lower().eq('true')
        rows.append(
            {
                'category': category,
                'n_targets': int(len(sub)),
                'high_confidence': int((sub['recommendation_confidence'] == 'high').sum()),
                'medium_confidence': int((sub['recommendation_confidence'] == 'medium').sum()),
                'low_confidence': int((sub['recommendation_confidence'] == 'low').sum()),
                'mean_confidence_score': float(sub['confidence_score'].mean()),
                'top1_pet_mp_form_match_rate': float(top1.mean()),
                'exergonic_best_practical_rate': float((sub['best_practical_ranked_deltaE_per_target'] < -0.05).mean()),
                'low_ehull_rate': float((sub['target_ehull'] <= 0.10).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values('category').reset_index(drop=True)


def top1_route_feature_summary(routes: pd.DataFrame) -> pd.DataFrame:
    if routes.empty or 'precursor_complexity_class' not in routes:
        return pd.DataFrame()
    top1 = routes[routes['rank'] == 1].copy() if 'rank' in routes else routes.copy()
    rows = []
    for complexity, sub in top1.groupby('precursor_complexity_class', dropna=False):
        rows.append(
            {
                'precursor_complexity_class': complexity,
                'n_targets': int(len(sub)),
                'mean_pet_deltaE_per_target': float(sub['pet_deltaE_per_target'].mean()) if 'pet_deltaE_per_target' in sub else np.nan,
                'median_abs_pet_mp_form_gap': float(sub['abs_pet_mp_form_gap'].median()) if 'abs_pet_mp_form_gap' in sub else np.nan,
                'mean_n_reactants': float(sub['n_reactants'].mean()) if 'n_reactants' in sub else np.nan,
                'mean_n_byproducts': float(sub['n_byproducts'].mean()) if 'n_byproducts' in sub else np.nan,
                'elemental_reactants_only_count': int(sub['elemental_reactants_only'].astype(str).str.lower().eq('true').sum()) if 'elemental_reactants_only' in sub else 0,
            }
        )
    return pd.DataFrame(rows).sort_values('precursor_complexity_class').reset_index(drop=True)


def aggregate_recommendations(
    benchmark_dir: Path,
    category_table: Path | None = None,
    top_n: int = 5,
    exergonic_threshold: float = -0.05,
    ehull_threshold: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    category_lookup = optional_category_map(category_table)
    summary_rows = []
    route_rows = []
    for target_dir in sorted(p for p in benchmark_dir.iterdir() if p.is_dir()):
        summary_path = target_dir / 'synthesis_recommendation_summary.csv'
        routes_path = target_dir / 'recommended_routes.csv'
        if summary_path.exists() and summary_path.stat().st_size > 0:
            sub = pd.read_csv(summary_path)
            if len(sub):
                sub['target'] = sub['target'].map(normalize_formula)
                if category_lookup:
                    sub['category'] = sub['target'].map(category_lookup)
                summary_rows.append(sub)
        if routes_path.exists() and routes_path.stat().st_size > 0:
            routes = pd.read_csv(routes_path)
            if len(routes):
                routes['target'] = routes['target'].map(normalize_formula)
                routes = routes.sort_values(['target', 'rank']).groupby('target', as_index=False).head(top_n)
                if category_lookup:
                    routes['category'] = routes['target'].map(category_lookup)
                route_rows.append(routes)
    summary = pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame()
    routes = pd.concat(route_rows, ignore_index=True) if route_rows else pd.DataFrame()
    if len(summary):
        summary = summary.sort_values('target').reset_index(drop=True)
    if len(routes):
        routes = routes.sort_values(['target', 'rank']).reset_index(drop=True)

    summary.to_csv(benchmark_dir / 'synthesis_recommendation_summary_all.csv', index=False)
    route_name = 'recommended_routes_top5_all.csv' if top_n == 5 else f'recommended_routes_top{top_n}_all.csv'
    routes.to_csv(benchmark_dir / route_name, index=False)

    queue = build_refinement_queue(summary, routes)
    queue.to_csv(benchmark_dir / 'dft_refinement_queue.csv', index=False)

    cat = category_summary(summary)
    if len(cat):
        cat.to_csv(benchmark_dir / 'recommendation_category_summary.csv', index=False)

    route_features = top1_route_feature_summary(routes)
    if len(route_features):
        route_features.to_csv(benchmark_dir / 'top1_route_feature_summary.csv', index=False)

    lines = ['# 推荐层汇总', '']
    lines.append(f'- 推荐摘要目标数：{len(summary)}')
    if len(summary):
        lines.append(f'- 分类计数：{summary["classification"].value_counts().to_dict()}')
        lines.append(f'- 置信度计数：{summary["recommendation_confidence"].value_counts().to_dict()}')
        lines.append(f'- 排序能量来源计数：{summary["ranking_energy_source"].value_counts().to_dict()}')
        lines.append(f'- 高置信推荐：{int((summary["recommendation_confidence"] == "high").sum())}')
        lines.append(f'- 中等置信推荐：{int((summary["recommendation_confidence"] == "medium").sum())}')
        lines.append(f'- 低置信推荐：{int((summary["recommendation_confidence"] == "low").sum())}')
        lines.append(f'- PET/MP_FORM top-1 一致：{int(summary["top1_pet_mp_form_match"].sum())}/{len(summary)}')
        lines.append(f'- 最佳 practical route 为放热的目标：{int((summary["best_practical_ranked_deltaE_per_target"] < exergonic_threshold).sum())}/{len(summary)}')
        lines.append(f'- 低 Ehull 目标（<={ehull_threshold:.2f} eV/atom）：{int((summary["target_ehull"] <= ehull_threshold).sum())}/{len(summary)}')
        lines.append(f'- 进入 DFT 精修队列的目标：{len(queue)}')
    if len(route_features):
        lines.append(f'- top-1 推荐路线前驱体复杂度：{route_features.set_index("precursor_complexity_class")["n_targets"].to_dict()}')
    lines.extend(
        [
            '',
            '## 输出文件',
            '',
            '- `synthesis_recommendation_summary_all.csv`：所有目标的一行式推荐摘要。',
            f'- `{route_name}`：每个目标前若干条推荐路线。',
            '- `recommendation_category_summary.csv`：按材料类别统计推荐置信度、top-1 一致率和低 Ehull 覆盖率。',
            '- `top1_route_feature_summary.csv`：按前驱体复杂度统计 top-1 推荐路线。',
            '- `dft_refinement_queue.csv`：建议优先进入后续 DFT 精修或人工核查的目标与原因。',
            '',
            '## 解释',
            '',
            '推荐层不是替代 DFT 或实验，而是把 PET 反应能、MP_FORM 参考、目标 Ehull、practical-route 过滤和分解竞争整合为可执行的优先级。高置信目标可作为自动化候选；中低置信目标应优先检查相空间完整性、PET/MP_FORM 路径不一致、分解竞争和实验前驱体可获得性。',
        ]
    )
    (benchmark_dir / 'RECOMMENDATION_LAYER_ANALYSIS.md').write_text('\n'.join(lines), encoding='utf-8')
    return summary, routes, queue


def refresh_per_target_recommendations(benchmark_dir: Path) -> int:
    refreshed = 0
    for target_dir in sorted(p for p in benchmark_dir.iterdir() if p.is_dir()):
        phase_path = target_dir / 'phase_stability.csv'
        formation_path = target_dir / 'formation_reactions.csv'
        decomposition_path = target_dir / 'decomposition_reactions.csv'
        if not (phase_path.exists() and formation_path.exists() and decomposition_path.exists()):
            continue
        phase_df = pd.read_csv(phase_path)
        formation_df = pd.read_csv(formation_path)
        decomposition_df = pd.read_csv(decomposition_path)
        write_recommendation(target_dir, target_dir.name, phase_df, formation_df, decomposition_df)
        refreshed += 1
    return refreshed


def main() -> None:
    parser = argparse.ArgumentParser(description='Aggregate per-target recommendation files from a benchmark directory.')
    parser.add_argument('--benchmark-dir', default=str(PROJECT / 'results' / 'mp_pet_benchmark_120'))
    parser.add_argument('--category-table', default=None)
    parser.add_argument('--top-n', type=int, default=5)
    parser.add_argument('--exergonic-threshold', type=float, default=-0.05)
    parser.add_argument('--ehull-threshold', type=float, default=0.10)
    parser.add_argument('--refresh-targets', action='store_true')
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir)
    category_table = Path(args.category_table) if args.category_table else benchmark_dir / 'mp_pet_benchmark_summary_with_categories.csv'
    if args.refresh_targets:
        refreshed = refresh_per_target_recommendations(benchmark_dir)
        print(f'Refreshed per-target recommendations: {refreshed}')
    summary, routes, queue = aggregate_recommendations(
        benchmark_dir=benchmark_dir,
        category_table=category_table,
        top_n=args.top_n,
        exergonic_threshold=args.exergonic_threshold,
        ehull_threshold=args.ehull_threshold,
    )
    print(f'Summaries: {len(summary)}')
    print(f'Routes: {len(routes)}')
    print(f'DFT refinement queue: {len(queue)}')
    print(f'Output: {benchmark_dir}')


if __name__ == '__main__':
    main()
