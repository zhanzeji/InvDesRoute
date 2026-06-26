#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pymatgen.core import Composition


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_INITIAL = PROJECT / 'literature_benchmark/results/ceder_seed80_route_similarity/route_similarity_eval.csv'
DEFAULT_EXTENSION = PROJECT / 'literature_benchmark/results/literature_seeded_extension_cached_53_similarity/route_similarity_eval.csv'
DEFAULT_OUTDIR = PROJECT / 'literature_benchmark/results/combined_literature_failure_modes_145'

STRONG_OR_EXACT = {'exact_route_recovered', 'strong_route_family_match'}
MODERATE_OR_BETTER = {'exact_route_recovered', 'strong_route_family_match', 'moderate_route_family_match'}

ROUTE_GROUPS = [
    ('binary-oxide source routes', {'binary_oxide_source'}),
    ('carbonate-containing routes', {'carbonate_source'}),
    ('gas-reservoir routes', {'gas_reservoir'}),
    ('hydroxide/hydrate routes', {'hydroxide_or_hydrate'}),
    ('elemental-source routes', {'elemental_source'}),
    ('phosphate-source routes', {'phosphate_source', 'ammonium_phosphate'}),
    ('nitrate/ammonium routes', {'nitrate_or_ammonium_salt', 'ammonium_phosphate'}),
    ('target-element compound routes', {'target_element_compound', 'binary_target_element_source'}),
]


def labels_in(values: pd.Series, allowed: set[str]) -> pd.Series:
    return values.fillna('').astype(str).isin(allowed)


def route_classes(value: object) -> set[str]:
    if pd.isna(value):
        return set()
    return {part.strip() for part in str(value).split(';') if part.strip()}


def class_mask(classes: pd.Series, group_classes: set[str]) -> pd.Series:
    return classes.map(lambda items: bool(items & group_classes))


def safe_mean(values: pd.Series) -> float:
    if len(values) == 0:
        return 0.0
    return round(pd.to_numeric(values, errors='coerce').fillna(0.0).mean(), 6)


def safe_median(values: pd.Series) -> float:
    if len(values) == 0:
        return 0.0
    return round(pd.to_numeric(values, errors='coerce').fillna(0.0).median(), 6)


def target_complexity(target: object) -> tuple[int, str]:
    try:
        n_elements = len(Composition(str(target)).elements)
    except Exception:
        n_elements = 0
    if n_elements == 2:
        label = 'binary targets'
    elif n_elements == 3:
        label = 'ternary targets'
    elif n_elements == 4:
        label = 'quaternary targets'
    elif n_elements > 4:
        label = 'quinary-or-higher targets'
    else:
        label = 'unknown-complexity targets'
    return n_elements, label


def load_similarity_tables(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame['source_file'] = str(path)
        frames.append(frame)
    if not frames:
        raise ValueError('At least one route-similarity CSV is required.')
    df = pd.concat(frames, ignore_index=True)
    df['best_all_classes_set'] = df['best_all_literature_classes'].map(route_classes)
    df['best_all_score_num'] = pd.to_numeric(df['best_all_similarity_score'], errors='coerce').fillna(0.0)
    df['best_top5_score_num'] = pd.to_numeric(df['best_top5_similarity_score'], errors='coerce').fillna(0.0)
    df['best_top20_score_num'] = pd.to_numeric(df['best_top20_similarity_score'], errors='coerce').fillna(0.0)
    df['full_network_strong_or_exact'] = labels_in(df['best_all_similarity_label'], STRONG_OR_EXACT)
    df['full_network_moderate_or_better'] = labels_in(df['best_all_similarity_label'], MODERATE_OR_BETTER)
    df['top5_strong_or_exact'] = labels_in(df['best_top5_similarity_label'], STRONG_OR_EXACT)
    df['top5_moderate_or_better'] = labels_in(df['best_top5_similarity_label'], MODERATE_OR_BETTER)
    df['top20_moderate_or_better'] = labels_in(df['best_top20_similarity_label'], MODERATE_OR_BETTER)
    complexities = df['target'].map(target_complexity)
    df['target_n_elements'] = complexities.map(lambda item: item[0])
    df['target_complexity_group'] = complexities.map(lambda item: item[1])
    return df


def summarize_global(df: pd.DataFrame) -> pd.DataFrame:
    row = {
        'n_routes': len(df),
        'full_network_strong_or_exact': int(df['full_network_strong_or_exact'].sum()),
        'full_network_moderate_or_better': int(df['full_network_moderate_or_better'].sum()),
        'top5_strong_or_exact': int(df['top5_strong_or_exact'].sum()),
        'top5_moderate_or_better': int(df['top5_moderate_or_better'].sum()),
        'top20_moderate_or_better': int(df['top20_moderate_or_better'].sum()),
        'top5_miss_but_full_network_moderate': int((df['full_network_moderate_or_better'] & ~df['top5_moderate_or_better']).sum()),
        'mean_top5_similarity': safe_mean(df['best_top5_score_num']),
    }
    return pd.DataFrame([row])


def summarize_classes(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_name, group_classes in ROUTE_GROUPS:
        sub = df[class_mask(df['best_all_classes_set'], group_classes)]
        rows.append(
            {
                'route_group': group_name,
                'n_routes': len(sub),
                'full_network_strong_or_exact': int(sub['full_network_strong_or_exact'].sum()),
                'full_network_moderate_or_better': int(sub['full_network_moderate_or_better'].sum()),
                'top5_strong_or_exact': int(sub['top5_strong_or_exact'].sum()),
                'top5_moderate_or_better': int(sub['top5_moderate_or_better'].sum()),
                'top20_moderate_or_better': int(sub['top20_moderate_or_better'].sum()),
                'mean_top5_similarity': safe_mean(sub['best_top5_score_num']),
                'median_top5_similarity': safe_median(sub['best_top5_score_num']),
            }
        )
    return pd.DataFrame(rows)


def summarize_rank_buckets(df: pd.DataFrame) -> pd.DataFrame:
    bucket_order = [('top1', 'top1'), ('top5', 'top2_5'), ('top20', 'top6_20'), ('below_top20', 'below_top20')]
    rows = []
    for raw_bucket, report_bucket in bucket_order:
        sub = df[df['best_all_rank_bucket'].fillna('').astype(str) == raw_bucket]
        rows.append(
            {
                'best_analogue_rank_bucket': report_bucket,
                'n_routes': len(sub),
                'strong_or_exact': int(sub['full_network_strong_or_exact'].sum()),
                'moderate_or_better': int(sub['full_network_moderate_or_better'].sum()),
                'mean_best_similarity': safe_mean(sub['best_all_score_num']),
            }
        )
    return pd.DataFrame(rows)


def summarize_target_complexity(df: pd.DataFrame) -> pd.DataFrame:
    order = ['binary targets', 'ternary targets', 'quaternary targets', 'quinary-or-higher targets', 'unknown-complexity targets']
    rows = []
    for group in order:
        sub = df[df['target_complexity_group'] == group]
        if len(sub) == 0:
            continue
        rows.append(
            {
                'target_complexity_group': group,
                'n_routes': len(sub),
                'full_network_moderate_or_better': int(sub['full_network_moderate_or_better'].sum()),
                'top5_moderate_or_better': int(sub['top5_moderate_or_better'].sum()),
                'top20_moderate_or_better': int(sub['top20_moderate_or_better'].sum()),
                'mean_top5_similarity': safe_mean(sub['best_top5_score_num']),
            }
        )
    return pd.DataFrame(rows)


def build_refinement_queue(df: pd.DataFrame) -> pd.DataFrame:
    queue = df[df['full_network_moderate_or_better'] & ~df['top5_moderate_or_better']].copy()
    columns = [
        'target',
        'literature_precursors',
        'literature_byproducts',
        'best_all_similarity_label',
        'best_all_similarity_score',
        'best_all_synth_rank',
        'best_all_rank_bucket',
        'best_all_reaction',
        'best_all_template',
        'best_all_risk_flags',
        'best_all_literature_classes',
        'best_top5_similarity_label',
        'best_top5_similarity_score',
        'best_top5_reaction',
        'best_top20_similarity_label',
        'best_top20_similarity_score',
        'best_top20_reaction',
    ]
    return queue[columns].sort_values(['target', 'best_all_synth_rank', 'best_all_similarity_score'], ascending=[True, True, False])


def write_report(outdir: Path, global_summary: pd.DataFrame, class_summary: pd.DataFrame, rank_summary: pd.DataFrame, complexity_summary: pd.DataFrame, queue: pd.DataFrame) -> None:
    global_row = global_summary.iloc[0].to_dict()
    lines = [
        '# Combined Literature Failure-Mode Diagnostics',
        '',
        '## English summary',
        '',
        f'- Routes audited: `{int(global_row["n_routes"])}` DOI-backed literature routes.',
        f'- Full-network moderate-or-better recovery: `{int(global_row["full_network_moderate_or_better"])}/{int(global_row["n_routes"])}`.',
        f'- Top-5 moderate-or-better recovery: `{int(global_row["top5_moderate_or_better"])}/{int(global_row["n_routes"])}`.',
        f'- Top-20 moderate-or-better recovery: `{int(global_row["top20_moderate_or_better"])}/{int(global_row["n_routes"])}`.',
        f'- Ranking-limited cases: `{int(global_row["top5_miss_but_full_network_moderate"])}` routes have a moderate-or-better analogue somewhere in the network but not in the top-5 window.',
        '',
        '## 中文总结',
        '',
        f'- 共审计 `{int(global_row["n_routes"])}` 条带 DOI 的真实文献合成路线。',
        f'- 完整反应网络中 moderate-or-better 恢复为 `{int(global_row["full_network_moderate_or_better"])}/{int(global_row["n_routes"])}`。',
        f'- 推荐 top-5 中 moderate-or-better 恢复为 `{int(global_row["top5_moderate_or_better"])}/{int(global_row["n_routes"])}`。',
        f'- 推荐 top-20 中 moderate-or-better 恢复为 `{int(global_row["top20_moderate_or_better"])}/{int(global_row["n_routes"])}`。',
        f'- `{int(global_row["top5_miss_but_full_network_moderate"])}` 条路线属于“网络里存在相近路线，但 top-5 没排出来”，主要反映排序和前驱体热化学校正问题，而不是反应枚举完全失败。',
        '',
        '## Precursor-class diagnostics',
        '',
        class_summary.to_markdown(index=False),
        '',
        '## Rank-bucket diagnostics',
        '',
        rank_summary.to_markdown(index=False),
        '',
        '## Target-complexity diagnostics',
        '',
        complexity_summary.to_markdown(index=False),
        '',
        '## Refinement queue',
        '',
        f'- Top-5 miss refinement cases: `{len(queue)}`.',
        '- These cases should be prioritized for explicit precursor decomposition thermochemistry, gas chemical-potential corrections and manual chemistry review.',
        '',
    ]
    (outdir / 'COMBINED_LITERATURE_FAILURE_MODE_DIAGNOSTICS.md').write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description='Summarize precursor-class and ranking failure modes for DOI-backed literature-route validation.')
    parser.add_argument('--inputs', type=Path, nargs='+', default=[DEFAULT_INITIAL, DEFAULT_EXTENSION], help='route_similarity_eval.csv files to combine')
    parser.add_argument('--outdir', type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    df = load_similarity_tables(args.inputs)
    global_summary = summarize_global(df)
    class_summary = summarize_classes(df)
    rank_summary = summarize_rank_buckets(df)
    complexity_summary = summarize_target_complexity(df)
    queue = build_refinement_queue(df)

    global_summary.to_csv(args.outdir / 'combined_literature_diagnostic_summary.csv', index=False)
    class_summary.to_csv(args.outdir / 'combined_literature_precursor_class_diagnostics.csv', index=False)
    rank_summary.to_csv(args.outdir / 'combined_literature_rank_bucket_diagnostics.csv', index=False)
    complexity_summary.to_csv(args.outdir / 'combined_literature_target_complexity_diagnostics.csv', index=False)
    queue.to_csv(args.outdir / 'combined_literature_top5_miss_refinement_queue.csv', index=False)
    write_report(args.outdir, global_summary, class_summary, rank_summary, complexity_summary, queue)

    print(global_summary.to_string(index=False))
    print(args.outdir)


if __name__ == '__main__':
    main()
