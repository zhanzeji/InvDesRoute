#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from pymatgen.core import Composition

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_CASE_DIR = PROJECT / 'results/seed40_synthesizability_cases'
DEFAULT_LITERATURE = PROJECT / 'literature_benchmark/literature_routes_seed.csv'
DEFAULT_OUTDIR = PROJECT / 'literature_benchmark/results/seed40_synthesizability'


def normalize(formula: str) -> str:
    return Composition(str(formula).strip()).reduced_formula


def elements(formula: str) -> set[str]:
    return set(Composition(formula).as_dict())


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_formulas(value: str | None) -> set[str]:
    if not value:
        return set()
    return {normalize(item) for item in str(value).split(';') if item.strip()}


def parse_side(value: str | None) -> set[str]:
    formulas = set()
    for item in str(value or '').split(' | '):
        item = item.strip()
        if not item:
            continue
        if '*' in item:
            _, formula = item.split('*', 1)
        else:
            formula = item
        formula = formula.strip()
        if formula:
            formulas.add(normalize(formula))
    return formulas


def has_formula(formulas: set[str], formula: str) -> bool:
    return normalize(formula) in formulas


def has_elemental_reactant(formulas: set[str], element: str) -> bool:
    return any(elements(formula) == {element} for formula in formulas)


def has_binary_oxide(formulas: set[str], element: str | None = None) -> bool:
    for formula in formulas:
        elems = elements(formula)
        if 'O' in elems and len(elems) == 2 and (element is None or element in elems):
            return True
    return False


def has_carbonate_like(formulas: set[str], element: str | None = None) -> bool:
    for formula in formulas:
        elems = elements(formula)
        if {'C', 'O'}.issubset(elems) and bool(elems - {'C', 'O'}) and (element is None or element in elems):
            return True
    return False


def has_oxalate_like(formulas: set[str], element: str | None = None) -> bool:
    for formula in formulas:
        comp = Composition(formula).as_dict()
        elems = set(comp)
        if 'C' in elems and 'O' in elems and comp.get('O', 0) >= 2 * comp.get('C', 0):
            if element is None or element in elems:
                return True
    return False


def has_phosphate_like(formulas: set[str]) -> bool:
    return any({'P', 'O'}.issubset(elements(formula)) for formula in formulas)


def rank_bucket(rank: str | int | None) -> str:
    if rank in (None, ''):
        return 'not_found'
    value = int(float(rank))
    if value == 1:
        return 'top1'
    if value <= 5:
        return 'top5'
    if value <= 20:
        return 'top20'
    return 'below_top20'


def row_rank(row: dict[str, str]) -> str:
    return row.get('synth_rank') or row.get('rank') or ''


def exact_match(lit_reactants: set[str], lit_byproducts: set[str], row: dict[str, str]) -> bool:
    reactants = parse_side(row.get('reactants'))
    products = parse_side(row.get('products'))
    return lit_reactants.issubset(reactants) and lit_byproducts.issubset(products)


def same_binary_direct_family(target: str, lit_reactants: set[str], reactants: set[str]) -> bool:
    target_elems = elements(target)
    if len(target_elems) != 2:
        return False
    non_gases = [formula for formula in lit_reactants if elements(formula) <= target_elems and len(elements(formula)) == 1]
    if len(non_gases) < 1:
        return False
    return all(has_elemental_reactant(reactants, elem) for elem in target_elems)


def family_match(row_seed: dict[str, str], route: dict[str, str]) -> bool:
    target = normalize(row_seed['target'])
    synthesis_type = row_seed.get('synthesis_type', '').lower()
    lit_reactants = split_formulas(row_seed.get('literature_precursors'))
    lit_byproducts = split_formulas(row_seed.get('literature_byproducts'))
    reactants = parse_side(route.get('reactants'))
    products = parse_side(route.get('products'))
    target_elems = elements(target)

    if exact_match(lit_reactants, lit_byproducts, route):
        return True

    if 'ceramic' in synthesis_type and {'Ti', 'O'}.issubset(target_elems):
        a_site = next(iter(target_elems - {'Ti', 'O'}), None)
        return bool(a_site) and has_carbonate_like(reactants, a_site) and has_binary_oxide(reactants, 'Ti') and has_formula(products, 'CO2')

    if target == normalize('LiFePO4') or 'phosphate' in synthesis_type:
        return has_formula(reactants, 'Li2CO3') and has_oxalate_like(reactants, 'Fe') and has_phosphate_like(reactants)

    if 'battery' in row_seed.get('source_note', '').lower() or target in {normalize('LiCoO2'), normalize('LiMn2O4')}:
        transition = next((elem for elem in target_elems if elem not in {'Li', 'O'}), None)
        return (
            has_formula(reactants, 'Li2CO3')
            and bool(transition)
            and (has_binary_oxide(reactants, transition) or has_carbonate_like(reactants, transition))
            and has_formula(products, 'CO2')
        )

    if 'ammonolysis' in synthesis_type:
        metal = next(iter(target_elems - {'N'}), None)
        return bool(metal) and has_elemental_reactant(reactants, metal) and has_formula(reactants, 'NH3') and has_formula(products, 'H2')

    if 'carbonation' in synthesis_type:
        metal = next(iter(target_elems - {'C', 'O'}), None)
        return bool(metal) and has_binary_oxide(reactants, metal) and has_formula(reactants, 'CO2')

    if 'direct oxidation' in synthesis_type or 'controlled oxidation' in synthesis_type:
        metal = next(iter(target_elems - {'O'}), None)
        return bool(metal) and has_elemental_reactant(reactants, metal) and has_formula(reactants, 'O2')

    if 'direct nitridation' in synthesis_type:
        metal = next(iter(target_elems - {'N'}), None)
        return bool(metal) and has_elemental_reactant(reactants, metal) and has_formula(reactants, 'N2')

    if 'carb' in synthesis_type:
        metal = next(iter(target_elems - {'C'}), None)
        return bool(metal) and has_elemental_reactant(reactants, metal) and has_elemental_reactant(reactants, 'C')

    if 'sulfidation' in synthesis_type or 'chalcogenide' in row_seed.get('source_note', '').lower():
        metal = next(iter(target_elems - {'S'}), None)
        return bool(metal) and has_elemental_reactant(reactants, metal) and has_elemental_reactant(reactants, 'S')

    if 'direct combination' in synthesis_type or 'halogen' in synthesis_type:
        return same_binary_direct_family(target, lit_reactants, reactants)

    return False


def best_match(rows: list[dict[str, str]], predicate) -> tuple[dict[str, str] | None, int]:
    for index, row in enumerate(rows, start=1):
        if predicate(row):
            return row, index
    return None, 0


def compact_route(row: dict[str, str] | None) -> dict[str, str]:
    if not row:
        return {
            'rank': '',
            'score': '',
            'reaction': '',
            'deltaE_per_target': '',
            'energy_source': '',
            'template': '',
            'risk_flags': '',
            'reason': '',
        }
    return {
        'rank': row_rank(row),
        'score': row.get('synth_score', ''),
        'reaction': row.get('reaction', ''),
        'deltaE_per_target': row.get('synth_rank_deltaE_per_target', ''),
        'energy_source': row.get('synth_rank_energy_source', ''),
        'template': row.get('route_template_reason', ''),
        'risk_flags': row.get('synthesis_risk_flags', ''),
        'reason': row.get('synthesis_recommendation_reason', ''),
    }


def route_limitation(row_seed: dict[str, str], exact: dict[str, str] | None, family: dict[str, str] | None, top: dict[str, str] | None) -> str:
    if exact and 'virtual_precursor_energy_missing' in exact.get('synthesis_risk_flags', ''):
        return 'literature route is enumerated but uses virtual precursor with missing thermochemistry'
    if exact and rank_bucket(row_rank(exact)) == 'top1':
        return 'exact literature route is top-ranked'
    if exact and rank_bucket(row_rank(exact)) == 'top5':
        return 'exact literature route is recovered in top-5'
    if family and rank_bucket(row_rank(family)) == 'top1':
        return 'exact stoichiometry differs, but literature route family is top-ranked'
    if family and rank_bucket(row_rank(family)) == 'top5':
        return 'exact stoichiometry differs, but literature route family is recovered in top-5'
    if family and rank_bucket(row_rank(family)) == 'top20':
        return 'literature route family is recovered but needs refinement'
    if exact and 'uphill_static_energy' in exact.get('synthesis_risk_flags', ''):
        return 'literature route is enumerated but static 0 K energy is uphill; finite-temperature or gas correction is likely important'
    lit_reactants = split_formulas(row_seed.get('literature_precursors'))
    extra = set().union(*(elements(formula) for formula in lit_reactants), set()) - elements(row_seed['target'])
    if extra & {'C', 'H', 'N'}:
        return 'experimental precursor chemistry includes volatile/decomposition elements not fully represented by static MP formation energies'
    if top and top.get('route_template_reason', '').startswith('direct elemental'):
        return 'direct elemental route is preferred by current synthesis prior'
    return 'no dominant limitation identified'


def make_svg(summary: dict[str, int], out: Path) -> None:
    width, height = 1040, 620
    metrics = [
        ('Exact literature routes found', summary['exact_found'], summary['n_cases'], '#2b8a3e'),
        ('Exact routes in top-1', summary['exact_top1'], summary['n_cases'], '#2b8a3e'),
        ('Exact routes in top-5', summary['exact_top5'], summary['n_cases'], '#66a61e'),
        ('Exact routes in top-20', summary['exact_top20'], summary['n_cases'], '#f2a93b'),
        ('Route-family matches found', summary['family_found'], summary['n_cases'], '#1f78b4'),
        ('Route-family matches in top-5', summary['family_top5'], summary['n_cases'], '#377eb8'),
        ('Cases with virtual/missing thermochemistry', summary['missing_thermo_cases'], summary['n_cases'], '#d95f02'),
    ]

    def esc(text: object) -> str:
        return str(text).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    def text(x: int, y: int, value: object, size: int = 14, weight: str = 'normal') -> str:
        return f'<text x="{x}" y="{y}" font-family="Arial" font-size="{size}" font-weight="{weight}">{esc(value)}</text>'

    def rect(x: int, y: int, w: float, h: int, color: str, rx: int = 5) -> str:
        return f'<rect x="{x}" y="{y}" width="{w:.1f}" height="{h}" rx="{rx}" fill="{color}"/>'

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        text(36, 44, 'Seed literature-route validation of synthesizability recommendations', 22, 'bold'),
        text(36, 72, 'Exact reaction matching is strict; route-family matching counts chemically equivalent precursor families.', 13),
    ]
    left, top, maxw = 470, 110, 390
    for idx, (label, num, den, color) in enumerate(metrics):
        y = top + idx * 58
        frac = num / den if den else 0.0
        parts.append(text(48, y + 22, label, 14))
        parts.append(rect(left, y, maxw, 28, '#e9edf5'))
        parts.append(rect(left, y, maxw * frac, 28, color))
        parts.append(text(left + maxw + 18, y + 20, f'{num}/{den} = {frac:.2f}', 14, 'bold'))
    parts.append(text(48, 545, 'Interpretation: mismatches are mostly caused by missing precursor thermochemistry or strict gas stoichiometry.', 13, 'bold'))
    parts.append(text(48, 570, 'Use exact matching for reproducibility and route-family matching for experimental usefulness.', 13))
    parts.append('</svg>')
    out.write_text('\n'.join(parts), encoding='utf-8')


def output_prefix(literature: Path) -> str:
    stem = literature.stem
    if stem.startswith('literature_routes_'):
        return stem.removeprefix('literature_routes_')
    return stem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--case-dir', type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument('--literature', type=Path, default=DEFAULT_LITERATURE)
    parser.add_argument('--outdir', type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()
    prefix = output_prefix(args.literature)

    seed_rows = read_csv(args.literature)
    results: list[dict[str, object]] = []
    for row_seed in seed_rows:
        target = normalize(row_seed['target'])
        route_path = args.case_dir / target / 'synthesizability_ranked_routes.csv'
        routes = read_csv(route_path)
        lit_reactants = split_formulas(row_seed.get('literature_precursors'))
        lit_byproducts = split_formulas(row_seed.get('literature_byproducts'))
        top = routes[0] if routes else None
        exact, _ = best_match(routes, lambda row: exact_match(lit_reactants, lit_byproducts, row))
        family, _ = best_match(routes, lambda route: family_match(row_seed, route))
        exact_info = compact_route(exact)
        family_info = compact_route(family)
        top_info = compact_route(top)
        missing_flags = ';'.join(
            flag
            for flag in [
                exact_info['risk_flags'],
                family_info['risk_flags'],
                top_info['risk_flags'],
            ]
            if flag
        )
        results.append(
            {
                'target': target,
                'synthesis_type': row_seed.get('synthesis_type', ''),
                'literature_precursors': ';'.join(sorted(lit_reactants)),
                'literature_byproducts': ';'.join(sorted(lit_byproducts)),
                'condition_summary': row_seed.get('condition_summary', ''),
                'n_routes': len(routes),
                'top_recommendation_reaction': top_info['reaction'],
                'top_recommendation_score': top_info['score'],
                'top_recommendation_template': top_info['template'],
                'top_recommendation_risk_flags': top_info['risk_flags'],
                'exact_found': bool(exact),
                'exact_synth_rank': exact_info['rank'],
                'exact_rank_bucket': rank_bucket(exact_info['rank']),
                'exact_reaction': exact_info['reaction'],
                'exact_score': exact_info['score'],
                'exact_deltaE_per_target': exact_info['deltaE_per_target'],
                'exact_energy_source': exact_info['energy_source'],
                'exact_template': exact_info['template'],
                'exact_risk_flags': exact_info['risk_flags'],
                'family_found': bool(family),
                'family_synth_rank': family_info['rank'],
                'family_rank_bucket': rank_bucket(family_info['rank']),
                'family_reaction': family_info['reaction'],
                'family_score': family_info['score'],
                'family_deltaE_per_target': family_info['deltaE_per_target'],
                'family_energy_source': family_info['energy_source'],
                'family_template': family_info['template'],
                'family_risk_flags': family_info['risk_flags'],
                'main_limitation': route_limitation(row_seed, exact, family, top),
                'has_missing_or_virtual_thermo': 'virtual_precursor_energy_missing' in missing_flags or 'energy_unavailable' in missing_flags,
            }
        )

    exact_found = [row for row in results if row['exact_found']]
    family_found = [row for row in results if row['family_found']]
    summary = {
        'n_cases': len(results),
        'completed_cases': sum(1 for row in results if int(row['n_routes']) > 0),
        'exact_found': len(exact_found),
        'exact_top1': sum(1 for row in results if row['exact_rank_bucket'] == 'top1'),
        'exact_top5': sum(1 for row in results if row['exact_rank_bucket'] in {'top1', 'top5'}),
        'exact_top20': sum(1 for row in results if row['exact_rank_bucket'] in {'top1', 'top5', 'top20'}),
        'family_found': len(family_found),
        'family_top1': sum(1 for row in results if row['family_rank_bucket'] == 'top1'),
        'family_top5': sum(1 for row in results if row['family_rank_bucket'] in {'top1', 'top5'}),
        'family_top20': sum(1 for row in results if row['family_rank_bucket'] in {'top1', 'top5', 'top20'}),
        'missing_thermo_cases': sum(1 for row in results if row['has_missing_or_virtual_thermo']),
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    write_csv(args.outdir / f'{prefix}_synthesizability_literature_eval.csv', results)
    write_csv(args.outdir / f'{prefix}_synthesizability_summary.csv', [summary])

    limitation_counts = Counter(str(row['main_limitation']) for row in results)
    write_csv(
        args.outdir / f'{prefix}_limitation_counts.csv',
        [{'limitation': key, 'count': value} for key, value in limitation_counts.most_common()],
        ['limitation', 'count'],
    )
    template_counts = Counter(str(row['top_recommendation_template']) for row in results if row['top_recommendation_template'])
    write_csv(
        args.outdir / f'{prefix}_top_template_counts.csv',
        [{'top_template': key, 'count': value} for key, value in template_counts.most_common()],
        ['top_template', 'count'],
    )
    make_svg(summary, args.outdir / f'{prefix}_synthesizability_validation_summary.svg')

    preferred_examples = ['KI', 'MgO', 'GaN', 'SiC', 'BaTiO3', 'LiCoO2', 'LiMn2O4', 'LiFePO4', 'CaCO3']
    exact_top_examples = [
        row
        for target in preferred_examples
        for row in results
        if row['target'] == target and (row['exact_rank_bucket'] in {'top1', 'top5'} or row['family_rank_bucket'] in {'top1', 'top5'})
    ]
    mismatches = [row for row in results if row['exact_rank_bucket'] not in {'top1', 'top5'}]
    lines = [
        f'# {summary["n_cases"]} 条 seed 文献路线的可合成性推荐验证',
        '',
        '## 结论概览',
        '',
        f'- 完成案例：`{summary["completed_cases"]}/{summary["n_cases"]}`。',
        f'- 严格精确反应式匹配：找到 `{summary["exact_found"]}/{summary["n_cases"]}`；top-1 `{summary["exact_top1"]}/{summary["n_cases"]}`；top-5 `{summary["exact_top5"]}/{summary["n_cases"]}`；top-20 `{summary["exact_top20"]}/{summary["n_cases"]}`。',
        f'- 路线族匹配：找到 `{summary["family_found"]}/{summary["n_cases"]}`；top-1 `{summary["family_top1"]}/{summary["n_cases"]}`；top-5 `{summary["family_top5"]}/{summary["n_cases"]}`；top-20 `{summary["family_top20"]}/{summary["n_cases"]}`。',
        f'- 需要补前驱体热化学或虚拟前驱体能量的案例：`{summary["missing_thermo_cases"]}/{summary["n_cases"]}`。',
        '',
        '## 如何解读',
        '',
        '- 严格匹配要求文献前驱体和气体副产物都出现在同一条配平反应中，因此对 O2、CO2、NH3、H2O、CO 等气体计量非常敏感。',
        '- 路线族匹配更接近实验判断：例如 Li2CO3+Co3O4+O2 和 Li2CO3+CoOx+CO2 都属于锂碳酸盐/钴氧化物固相路线，但精确反应式可能不同。',
        '- 若文献路线被枚举但排名低，通常不是“路线不存在”，而是当前静态能量/MP 相池缺少前驱体分解热、气体化学势或虚拟前驱体能量。',
        '',
        '## 代表性文献路线示例',
        '',
    ]
    if exact_top_examples:
        for row in exact_top_examples:
            lines.append(
                f'- `{row["target"]}`：精确文献式排名 `{row["exact_synth_rank"] or "not found"}`，'
                f'路线族排名 `{row["family_synth_rank"] or "not found"}`；推荐/匹配反应 `{row["family_reaction"] or row["exact_reaction"]}`。'
            )
    else:
        lines.append('- 暂无严格 top-5 示例；请检查批量运行是否完成。')
    lines.extend(['', '## 主要不匹配来源', ''])
    for limitation, count in limitation_counts.most_common():
        lines.append(f'- {limitation}：`{count}` 个案例。')
    if mismatches:
        lines.extend(['', '## 需要优先精修的案例', ''])
        for row in mismatches[:12]:
            exact_rank = row['exact_synth_rank'] or 'not found'
            family_rank = row['family_synth_rank'] or 'not found'
            lines.append(
                f'- `{row["target"]}`：精确排名 `{exact_rank}`，路线族排名 `{family_rank}`；原因：{row["main_limitation"]}。'
            )
    lines.extend(
        [
            '',
            '## 对 workflow 的意义',
            '',
            '- 当前结果验证的是“推荐路线是否接近真实实验路线”，不是温度/压力的最终预测。',
            '- MP 相池足以支持元素相、氧化物、部分碳酸盐和常见二元化合物；但草酸盐、铵盐、含水盐、硝酸盐等实验前驱体需要外部热化学表或 DFT 计算补能量。',
            '- 后续若加入气体化学势和前驱体分解热，排序应更接近真实实验优先级，尤其是碳酸盐放 CO2、铵盐/草酸盐放 NH3/H2O/CO/CO2 的路线。',
        ]
    )
    report_path = args.outdir / f'{prefix.upper()}_SYNTHESIZABILITY_VALIDATION.md'
    report_path.write_text('\n'.join(lines), encoding='utf-8')

    print(summary)
    print(report_path)


if __name__ == '__main__':
    main()
