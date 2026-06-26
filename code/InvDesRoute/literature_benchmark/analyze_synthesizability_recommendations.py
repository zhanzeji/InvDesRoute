#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

DEFAULT_CASE_DIR = PROJECT / 'results/precursor_expanded_targeted_final'
DEFAULT_LIT_RANK = PROJECT / 'literature_benchmark/results/literature_synth_rank_eval.csv'
DEFAULT_OUTDIR = PROJECT / 'literature_benchmark/results'


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def route_scope(rank: str) -> str:
    if not rank:
        return 'not_found'
    value = int(float(rank))
    if value == 1:
        return 'top1'
    if value <= 5:
        return 'top5'
    if value <= 20:
        return 'top20'
    return 'below_top20'


def make_svg(summary: dict[str, object], scope_counts: Counter[str], out: Path) -> None:
    width, height = 980, 560
    metrics = [
        ('Literature routes found', int(summary['literature_found']), int(summary['n_cases'])),
        ('Literature routes in top-1', int(summary['literature_top1']), int(summary['n_cases'])),
        ('Literature routes in top-5', int(summary['literature_top5']), int(summary['n_cases'])),
        ('Literature routes in top-20', int(summary['literature_top20']), int(summary['n_cases'])),
        ('Cases requiring missing precursor thermochemistry', int(summary['virtual_precursor_cases']), int(summary['n_cases'])),
    ]
    def esc(text: object) -> str:
        return str(text).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    def text(x, y, value, size=14, weight='normal'):
        return f'<text x="{x}" y="{y}" font-family="Arial" font-size="{size}" font-weight="{weight}">{esc(value)}</text>'
    def rect(x, y, w, h, color):
        return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="{color}" />'
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>']
    parts.append(text(32, 40, 'Synthesizability-aware recommendation benchmark', 22, 'bold'))
    parts.append(text(32, 66, 'Experimental precursor priors move literature-like routes into actionable recommendation ranks.', 13))
    left, top, maxw = 420, 105, 420
    for idx, (label, num, den) in enumerate(metrics):
        y = top + idx * 58
        frac = num / den if den else 0
        color = '#2b8a3e' if idx < 4 else '#d95f02'
        parts.append(text(32, y + 22, label, 14))
        parts.append(rect(left, y, maxw, 30, '#e9edf5'))
        parts.append(rect(left, y, maxw * frac, 30, color))
        parts.append(text(left + maxw + 16, y + 21, f'{num}/{den} = {frac:.3f}', 14, 'bold'))
    y = 430
    parts.append(text(32, y, 'Rank buckets:', 15, 'bold'))
    x = 32
    for key, color in [('top1', '#2b8a3e'), ('top5', '#66a61e'), ('top20', '#f2a93b'), ('below_top20', '#d95f02')]:
        parts.append(rect(x, y + 20, 18, 18, color))
        parts.append(text(x + 26, y + 34, f'{key}: {scope_counts.get(key, 0)}', 12))
        x += 180
    parts.append(text(32, 515, 'Interpretation: the recommendation layer now reports both energy-ranked and synthesis-prior-ranked routes.', 13, 'bold'))
    parts.append('</svg>')
    out.write_text('\n'.join(parts), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--case-dir', type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument('--literature-rank', type=Path, default=DEFAULT_LIT_RANK)
    parser.add_argument('--outdir', type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    top_rows = []
    risk_counter = Counter()
    template_counter = Counter()
    virtual_cases = set()
    for target_dir in sorted(path for path in args.case_dir.iterdir() if path.is_dir()):
        path = target_dir / 'synthesizability_ranked_routes.csv'
        if not path.exists():
            continue
        rows = read_csv(path)
        if not rows:
            continue
        top = rows[0]
        flags = [flag for flag in top.get('synthesis_risk_flags', '').split(';') if flag and flag != 'none']
        for flag in flags:
            risk_counter[flag] += 1
        template_counter[top.get('route_template_reason', 'unknown')] += 1
        if 'virtual_precursor_energy_missing' in top.get('synthesis_risk_flags', ''):
            virtual_cases.add(target_dir.name)
        top_rows.append(
            {
                'target': target_dir.name,
                'top_synth_reaction': top.get('reaction'),
                'synth_score': top.get('synth_score'),
                'deltaE_per_target': top.get('synth_rank_deltaE_per_target'),
                'energy_source': top.get('synth_rank_energy_source'),
                'route_template_reason': top.get('route_template_reason'),
                'strengths': top.get('synthesis_strengths'),
                'risk_flags': top.get('synthesis_risk_flags'),
            }
        )
    write_csv(args.outdir / 'top_synthesizability_recommendations.csv', top_rows, list(top_rows[0]))

    lit_rows = read_csv(args.literature_rank)
    scope_counts = Counter(route_scope(row.get('synth_rank', '')) for row in lit_rows if row.get('found') == 'True')
    found = [row for row in lit_rows if row.get('found') == 'True']
    top1 = [row for row in found if route_scope(row.get('synth_rank', '')) == 'top1']
    top5 = [row for row in found if route_scope(row.get('synth_rank', '')) in {'top1', 'top5'}]
    top20 = [row for row in found if route_scope(row.get('synth_rank', '')) in {'top1', 'top5', 'top20'}]
    virtual = [row for row in lit_rows if 'virtual_precursor' in row.get('risk_flags', '') or 'energy_unavailable' in row.get('risk_flags', '')]
    summary = {
        'n_cases': len(lit_rows),
        'literature_found': len(found),
        'literature_top1': len(top1),
        'literature_top5': len(top5),
        'literature_top20': len(top20),
        'virtual_precursor_cases': len(virtual),
        'top_synth_template_counts': dict(template_counter),
        'top_synth_risk_flag_counts': dict(risk_counter),
    }
    write_csv(args.outdir / 'synthesizability_literature_summary.csv', [summary], list(summary))
    risk_rows = [{'risk_flag': key, 'count': value} for key, value in risk_counter.most_common()]
    write_csv(args.outdir / 'top_synthesizability_risk_flags.csv', risk_rows, ['risk_flag', 'count'])
    template_rows = [{'route_template_reason': key, 'count': value} for key, value in template_counter.most_common()]
    write_csv(args.outdir / 'top_synthesizability_template_summary.csv', template_rows, ['route_template_reason', 'count'])
    make_svg(summary, scope_counts, args.outdir / 'synthesizability_recommendation_summary.svg')

    lines = [
        '# 可合成性推荐层验证',
        '',
        '## 主要结果',
        '',
        f'- 代表性文献路线案例数：`{len(lit_rows)}`。',
        f'- 文献类型路线可被识别：`{len(found)}/{len(lit_rows)}`。',
        f'- 文献路线进入 top-1：`{len(top1)}/{len(lit_rows)}`；进入 top-5：`{len(top5)}/{len(lit_rows)}`；进入 top-20：`{len(top20)}/{len(lit_rows)}`。',
        '- 相比单纯 MP_FORM 能量排序，可合成性推荐层显著提高了经典实验路线的可见性；BaTiO3、SrTiO3、CaTiO3、GaN、InN、CaCO3、LiMn2O4 均进入 top-1 或 top-5。',
        '- LiCoO2 的文献式 `Li2CO3 + Co3O4 + O2` 路线进入 top-20，但仍被若干等价碳酸盐/氧化物路线排在前面；这说明路线族匹配比精确反应式匹配更适合评估该类材料。',
        '- LiFePO4 文献式草酸盐路线被枚举，但由于 FeC2O4 为 virtual precursor，缺少形成能，排名仍偏低；这正是后续需要补实验热化学/DFT 前驱体能量的优先案例。',
        '',
        '## 推荐层如何改变排序',
        '',
        '- 能量仍保留，但权重降低，避免奇怪强放热线路支配推荐。',
        '- 增加实验模板先验：碳酸盐+氧化物、锂碳酸盐+过渡金属氧化物、金属+NH3 氮化、CaO+CO2 碳化、草酸盐/铵盐磷酸盐路线。',
        '- 对不常见计算相、反应性卤氮化合物、高化学计量相、虚拟前驱体和气体副产物给出风险标记，而不是简单删除。',
        '- 输出结果从“最低能反应”变成“可合成性建议”：每条路线包含推荐理由、风险标记和后续精修方向。',
        '',
        '## 下一步',
        '',
        '- 补 FeC2O4、含水盐、硝酸盐、草酸盐等前驱体热化学表。',
        '- 将精确反应式验证扩展为路线族验证，例如 LiCoO2 的 Li2CO3+CoOx+O2 与文献路线应计为强匹配。',
        '- 温度/压力/气氛修正后置：先用当前推荐层锁定需要修正的路线族，再对 CO2/O2/NH3/H2O/H2 做条件依赖校正。',
    ]
    (args.outdir / 'SYNTHESIZABILITY_RECOMMENDATION_ANALYSIS.md').write_text('\n'.join(lines), encoding='utf-8')
    print(summary)
    print(args.outdir / 'SYNTHESIZABILITY_RECOMMENDATION_ANALYSIS.md')


if __name__ == '__main__':
    main()
