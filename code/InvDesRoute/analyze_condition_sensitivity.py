#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common import PROJECT, normalize_formula


MAJOR_EXCLUSION_FLAGS = [
    'unusual_or_hazardous_precursor',
    'redox_nitrate_reducing_gas_route',
    'missing_precursor_phase',
]


def truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {'true', '1', 'yes'}


def safe_float(value: object, default: float = float('nan')) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def actionable_mask(df: pd.DataFrame, min_score: float) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    if 'practical_route' in df:
        mask &= df['practical_route'].map(truthy)
    if 'synth_score' in df:
        mask &= df['synth_score'].map(lambda value: safe_float(value, 0.0) >= min_score)
    if 'route_template_score' in df:
        mask &= df['route_template_score'].map(lambda value: safe_float(value, 0.0) >= 0.68)
    if 'synthesis_risk_flags' in df:
        risk_text = df['synthesis_risk_flags'].fillna('').astype(str)
        for flag in MAJOR_EXCLUSION_FLAGS:
            mask &= ~risk_text.str.contains(flag, regex=False)
    return mask


def merge_condition_columns(ranked: pd.DataFrame, formation: pd.DataFrame) -> pd.DataFrame:
    condition_cols = [
        col
        for col in formation.columns
        if col.startswith('DeltaG_') or col.startswith('GasCorrection_')
    ]
    base_cols = ['reaction', *condition_cols]
    merged = ranked.merge(formation[base_cols].drop_duplicates('reaction'), on='reaction', how='left')
    return merged


def analyse_target(target_dir: Path, min_score: float) -> dict[str, object] | None:
    ranked_path = target_dir / 'synthesizability_ranked_routes.csv'
    formation_path = target_dir / 'formation_reactions.csv'
    if not ranked_path.exists() or not formation_path.exists():
        return None
    ranked = pd.read_csv(ranked_path)
    formation = pd.read_csv(formation_path)
    if ranked.empty or formation.empty:
        return None
    merged = merge_condition_columns(ranked, formation)
    action = merged[actionable_mask(merged, min_score)].copy()
    top = merged.iloc[0]
    row: dict[str, object] = {
        'target': normalize_formula(target_dir.name),
        'n_routes': int(len(merged)),
        'n_actionable_routes': int(len(action)),
        'top_synth_reaction': top.get('reaction', ''),
        'top_synth_score': top.get('synth_score', ''),
        'top_synth_deltaE_per_target': top.get('synth_rank_deltaE_per_target', ''),
        'top_synth_energy_source': top.get('synth_rank_energy_source', ''),
        'top_synth_template': top.get('route_template_reason', ''),
        'top_synth_risk_flags': top.get('synthesis_risk_flags', ''),
    }
    condition_cols = [col for col in merged.columns if col.startswith('DeltaG_') and col.endswith('_gas_per_target')]
    for col in condition_cols:
        label = col.replace('DeltaG_', '').replace('_gas_per_target', '')
        top_value = top.get(col)
        valid = merged[merged[col].notna()].copy()
        raw_best = valid.nsmallest(1, col).iloc[0] if len(valid) else None
        action_valid = action[action[col].notna()].copy()
        action_best = action_valid.nsmallest(1, col).iloc[0] if len(action_valid) else None
        row[f'top_synth_{label}_deltaG_per_target'] = top_value
        row[f'top_synth_{label}_sign'] = 'exergonic' if pd.notna(top_value) and float(top_value) <= -0.05 else 'not_exergonic_or_missing'
        row[f'best_actionable_{label}_reaction'] = action_best.get('reaction', '') if action_best is not None else ''
        row[f'best_actionable_{label}_deltaG_per_target'] = action_best.get(col, '') if action_best is not None else ''
        row[f'best_actionable_{label}_synth_rank'] = action_best.get('synth_rank', '') if action_best is not None else ''
        row[f'best_actionable_{label}_synth_score'] = action_best.get('synth_score', '') if action_best is not None else ''
        row[f'raw_thermo_best_{label}_reaction'] = raw_best.get('reaction', '') if raw_best is not None else ''
        row[f'raw_thermo_best_{label}_deltaG_per_target'] = raw_best.get(col, '') if raw_best is not None else ''
        if raw_best is not None and action_best is not None and raw_best.get('reaction', '') != action_best.get('reaction', ''):
            row[f'{label}_thermo_practical_disagreement'] = True
        else:
            row[f'{label}_thermo_practical_disagreement'] = False
    return row


def write_report(summary: pd.DataFrame, outdir: Path) -> None:
    condition_labels = sorted(
        {
            col.removeprefix('top_synth_').removesuffix('_sign')
            for col in summary.columns
            if col.startswith('top_synth_') and col.endswith('_sign')
        }
    )
    lines = [
        '# 有限温度/气体化学势敏感性分析',
        '',
        '## 结论',
        '',
        '- 本分析是有限温度与有限气体分压的第一层近似，不是完整热力学数据库替代。',
        '- 原始最低 ΔG 往往会选择卤氧化物、硝酸盐/还原气氛等不适合作为第一实验建议的路线，因此报告同时给出 raw thermodynamic best 和 best actionable route。',
        '- 论文中应使用 best actionable route 或 top synthesizability route 的条件敏感性，而不是直接使用全网络最低 ΔG。',
        '',
        '## 统计',
        '',
        f'- 目标数：`{len(summary)}`。',
        f'- 平均可行动路线数：`{summary["n_actionable_routes"].mean():.1f}`。',
    ]
    for label in condition_labels:
        sign_col = f'top_synth_{label}_sign'
        disagreement_col = f'{label}_thermo_practical_disagreement'
        exergonic = int((summary[sign_col] == 'exergonic').sum())
        disagree = int(summary[disagreement_col].sum()) if disagreement_col in summary else 0
        lines.append(f'- `{label}`：top synthesizability route 放热/有利 `{exergonic}/{len(summary)}`；raw 最低 ΔG 与可行动最佳路线不一致 `{disagree}/{len(summary)}`。')
    lines.extend(
        [
            '',
            '## 使用边界',
            '',
            '- 当前气体校正只包含简化的气体熵和压力项；未包含固体振动自由能、相变、缺陷、真实热容积分或可靠 JANAF/NIST 热化学表。',
            '- 该结果适合用来识别“哪些路线需要考虑 CO2/O2/NH3/H2/H2O 等气体势”，不适合直接给出精确合成温度或压力窗口。',
            '- 后续若要变成强实验预测，需要对重点路线接入实验热化学表、DFT phonon/thermochemistry 或 FactSage/Materials Project finite-temperature 数据源。',
        ]
    )
    (outdir / 'CONDITION_SENSITIVITY_ANALYSIS.md').write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description='Summarize finite-temperature gas corrections on actionable synthesis routes.')
    parser.add_argument('--case-dir', type=Path, default=PROJECT / 'results/seed80_synthesizability_cases')
    parser.add_argument('--outdir', type=Path, default=PROJECT / 'results/seed80_synthesizability_cases')
    parser.add_argument('--min-actionable-score', type=float, default=0.70)
    args = parser.parse_args()
    rows = [
        row
        for target_dir in sorted(path for path in args.case_dir.iterdir() if path.is_dir())
        for row in [analyse_target(target_dir, args.min_actionable_score)]
        if row is not None
    ]
    args.outdir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    out = args.outdir / 'condition_sensitivity_practical_summary.csv'
    summary.to_csv(out, index=False)
    write_report(summary, args.outdir)
    print(out)
    print(args.outdir / 'CONDITION_SENSITIVITY_ANALYSIS.md')


if __name__ == '__main__':
    main()
