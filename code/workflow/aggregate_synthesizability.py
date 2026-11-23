#!/home/zhzhan/miniconda3/envs/mattersim/bin/python
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from common import PROJECT, normalize_formula  # noqa: E402
from synthesizability import write_synthesizability_outputs  # noqa: E402


def refresh_case(case_dir: Path) -> bool:
    phase_path = case_dir / 'phase_stability.csv'
    formation_path = case_dir / 'formation_reactions.csv'
    if not phase_path.exists() or not formation_path.exists():
        return False
    phase_df = pd.read_csv(phase_path)
    formation_df = pd.read_csv(formation_path)
    write_synthesizability_outputs(case_dir, normalize_formula(case_dir.name), phase_df, formation_df)
    return True


def split_flags(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [flag for flag in str(value).split(';') if flag and flag != 'none']


def aggregate(case_dir: Path, top_n: int = 10, refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if refresh:
        for target_dir in sorted(path for path in case_dir.iterdir() if path.is_dir()):
            refresh_case(target_dir)
    top_rows = []
    route_rows = []
    for target_dir in sorted(path for path in case_dir.iterdir() if path.is_dir()):
        path = target_dir / 'synthesizability_ranked_routes.csv'
        if not path.exists() or path.stat().st_size == 0:
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        df['target'] = normalize_formula(target_dir.name)
        sub = df.head(top_n).copy()
        route_rows.append(sub)
        top = df.iloc[0]
        decision_path = target_dir / 'synthesizability_decision.csv'
        decision = {}
        if decision_path.exists() and decision_path.stat().st_size:
            try:
                decision_df = pd.read_csv(decision_path)
                if len(decision_df):
                    decision = decision_df.iloc[0].to_dict()
            except Exception:
                decision = {}
        row = {
                'target': normalize_formula(target_dir.name),
                'synthesis_confidence': decision.get('synthesis_confidence'),
                'synthesis_decision': decision.get('synthesis_decision'),
                'synthesis_confidence_score': decision.get('synthesis_confidence_score'),
                'target_ehull': decision.get('target_ehull'),
                'primary_support': decision.get('primary_support'),
                'primary_risks': decision.get('primary_risks'),
                'recommended_next_actions': decision.get('recommended_next_actions'),
                'top_synth_reaction': top.get('reaction'),
                'top_synth_score': top.get('synth_score'),
                'top_synth_deltaE_per_target': top.get('synth_rank_deltaE_per_target'),
                'top_synth_energy_source': top.get('synth_rank_energy_source'),
                'top_route_template_reason': top.get('route_template_reason'),
                'top_synthesis_strengths': top.get('synthesis_strengths'),
                'top_synthesis_risk_flags': top.get('synthesis_risk_flags'),
                'n_synth_ranked_routes': len(df),
                'n_high_score_routes_ge_0p90': int((df['synth_score'] >= 0.90).sum()) if 'synth_score' in df else 0,
                'n_routes_with_energy_unavailable': int(df['synthesis_risk_flags'].astype(str).str.contains('energy_unavailable').sum()) if 'synthesis_risk_flags' in df else 0,
                'n_routes_with_virtual_precursor': int(df['synthesis_risk_flags'].astype(str).str.contains('virtual_precursor_energy_missing').sum()) if 'synthesis_risk_flags' in df else 0,
        }
        top_rows.append(row)
    summary = pd.DataFrame(top_rows)
    routes = pd.concat(route_rows, ignore_index=True) if route_rows else pd.DataFrame()
    refinement_rows = []
    if len(routes):
        for _, row in routes.iterrows():
            flags = split_flags(row.get('synthesis_risk_flags'))
            priority = 3
            reasons = []
            if 'virtual_precursor_energy_missing' in flags or 'energy_unavailable' in flags:
                priority = min(priority, 1)
                reasons.append('fill_manual_precursor_thermochemistry')
            if 'uphill_static_energy' in flags and row.get('route_template_score', 0) >= 0.90:
                priority = min(priority, 2)
                reasons.append('finite_temperature_gas_correction_needed')
            if 'manual_reservoir_energy' in flags or 'gas_byproduct_management' in flags:
                priority = min(priority, 2)
                reasons.append('check_gas_reservoir_assumptions')
            if 'unusual_or_hazardous_precursor' in flags:
                priority = min(priority, 2)
                reasons.append('manual_precursor_safety_filter')
            if reasons:
                refinement_rows.append(
                    {
                        'target': row.get('target'),
                        'synth_rank': row.get('synth_rank'),
                        'priority': priority,
                        'reaction': row.get('reaction'),
                        'synth_score': row.get('synth_score'),
                        'deltaE_per_target': row.get('synth_rank_deltaE_per_target'),
                        'route_template_reason': row.get('route_template_reason'),
                        'risk_flags': row.get('synthesis_risk_flags'),
                        'refinement_reason': ';'.join(dict.fromkeys(reasons)),
                    }
                )
    queue = pd.DataFrame(refinement_rows)
    if len(queue):
        queue = queue.sort_values(['priority', 'target', 'synth_rank']).reset_index(drop=True)
    return summary, routes, queue


def write_report(outdir: Path, summary: pd.DataFrame, routes: pd.DataFrame, queue: pd.DataFrame) -> None:
    flag_counter: Counter[str] = Counter()
    template_counter: Counter[str] = Counter()
    if len(summary):
        for value in summary['top_synthesis_risk_flags']:
            flag_counter.update(split_flags(value))
        template_counter.update(summary['top_route_template_reason'].fillna('unknown'))
    lines = ['# 可合成性推荐聚合报告', '']
    lines.append(f'- 目标数：`{len(summary)}`')
    if len(summary):
        lines.append(f'- top 推荐平均 synth_score：`{summary["top_synth_score"].astype(float).mean():.3f}`')
        if 'synthesis_confidence' in summary:
            lines.append(f'- 目标级置信等级统计：`{summary["synthesis_confidence"].fillna("unknown").value_counts().to_dict()}`')
        lines.append(f'- top 推荐路线模板统计：`{dict(template_counter)}`')
        lines.append(f'- top 推荐风险标记统计：`{dict(flag_counter)}`')
        lines.append(f'- 需要补充/精修的 top-N 路线数：`{len(queue)}`')
    lines.extend(['', '## 输出文件', '', '- `synthesizability_summary_all.csv`：每个目标 top-1 可合成性推荐。', '- `synthesizability_routes_topN_all.csv`：每个目标 top-N 可合成性路线。', '- `synthesizability_refinement_queue.csv`：需要补手工热化学、气体校正、安全/可获得性检查的路线。'])
    outdir.joinpath('SYNTHESIZABILITY_AGGREGATE_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description='Aggregate synthesizability-aware recommendations from per-target folders.')
    parser.add_argument('--case-dir', type=Path, default=PROJECT / 'results/precursor_expanded_targeted_final')
    parser.add_argument('--top-n', type=int, default=10)
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    summary, routes, queue = aggregate(args.case_dir, top_n=args.top_n, refresh=args.refresh)
    summary.to_csv(args.case_dir / 'synthesizability_summary_all.csv', index=False)
    routes.to_csv(args.case_dir / f'synthesizability_routes_top{args.top_n}_all.csv', index=False)
    queue.to_csv(args.case_dir / 'synthesizability_refinement_queue.csv', index=False)
    write_report(args.case_dir, summary, routes, queue)
    print(f'Summaries: {len(summary)}')
    print(f'Routes: {len(routes)}')
    print(f'Refinement queue: {len(queue)}')
    print(f'Output: {args.case_dir}')


if __name__ == '__main__':
    main()
