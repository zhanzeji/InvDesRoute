#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from analysis import parse_reaction_side  # noqa: E402
from common import PROJECT, normalize_formula  # noqa: E402
from synthesizability import precursor_class  # noqa: E402


def route_signature(row: pd.Series, target: str) -> str:
    reactants = [formula for formula, _ in parse_reaction_side(row.get('reactants', ''))]
    classes = sorted(precursor_class(formula, '') for formula in reactants)
    template = str(row.get('route_template_reason', '')).split(':', 1)[0]
    return f'{template}|{";".join(classes)}'


def choose_diverse(df: pd.DataFrame, target: str, top_n: int) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df['route_family_signature'] = [route_signature(row, target) for _, row in df.iterrows()]
    selected_indices = []
    family_counts: Counter[str] = Counter()
    # First pass: one high-score representative per family.
    for idx, row in df.iterrows():
        sig = row['route_family_signature']
        if family_counts[sig] == 0:
            selected_indices.append(idx)
            family_counts[sig] += 1
        if len(selected_indices) >= top_n:
            break
    # Second pass: fill remaining slots by original synthesizability rank.
    if len(selected_indices) < top_n:
        for idx, row in df.iterrows():
            if idx in selected_indices:
                continue
            selected_indices.append(idx)
            if len(selected_indices) >= top_n:
                break
    out = df.loc[selected_indices].copy().reset_index(drop=True)
    out.insert(0, 'diverse_rank', range(1, len(out) + 1))
    return out


def write_case(target_dir: Path, top_n: int) -> pd.DataFrame | None:
    path = target_dir / 'synthesizability_ranked_routes.csv'
    if not path.exists() or path.stat().st_size == 0:
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    target = normalize_formula(target_dir.name)
    diverse = choose_diverse(df, target, top_n)
    diverse.to_csv(target_dir / f'synthesizability_diverse_top{top_n}.csv', index=False)
    lines = [
        f'# 多样化可合成性推荐：{target}',
        '',
        '该文件在原始 synth_score 排序基础上，按路线模板和前驱体类别做去重，避免 top-N 被同一种路线族占满。',
        '',
    ]
    for _, row in diverse.iterrows():
        delta = row.get('synth_rank_deltaE_per_target')
        delta_text = 'unavailable' if pd.isna(delta) else f'{float(delta):.3f} eV/target'
        lines.append(f'- {int(row["diverse_rank"])}. 原始排名 `{row.get("synth_rank")}`：`{row.get("reaction")}`')
        lines.append(f'  - 分数 `{float(row.get("synth_score", 0.0)):.3f}`；ΔE `{delta_text}`；族 `{row.get("route_family_signature")}`')
        lines.append(f'  - 模板：{row.get("route_template_reason")}')
        lines.append(f'  - 风险：`{row.get("synthesis_risk_flags")}`')
    (target_dir / f'synthesizability_diverse_top{top_n}.md').write_text('\n'.join(lines), encoding='utf-8')
    diverse.insert(0, 'target', target)
    return diverse


def main() -> None:
    parser = argparse.ArgumentParser(description='Build diversity-aware top-N synthesis route recommendations.')
    parser.add_argument('--case-dir', type=Path, default=PROJECT / 'results/seed80_synthesizability_cases')
    parser.add_argument('--top-n', type=int, default=10)
    args = parser.parse_args()
    rows = []
    for target_dir in sorted(path for path in args.case_dir.iterdir() if path.is_dir()):
        diverse = write_case(target_dir, args.top_n)
        if diverse is not None and len(diverse):
            rows.append(diverse)
    all_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out = args.case_dir / f'synthesizability_diverse_top{args.top_n}_all.csv'
    all_df.to_csv(out, index=False)
    print({'targets': int(all_df['target'].nunique()) if len(all_df) else 0, 'routes': len(all_df), 'output': str(out)})


if __name__ == '__main__':
    main()
