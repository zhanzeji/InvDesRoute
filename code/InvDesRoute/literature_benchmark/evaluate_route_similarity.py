#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter
from functools import lru_cache
from pathlib import Path

import pandas as pd
from pymatgen.core import Composition


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_CASE_DIR = PROJECT / 'results/seed80_synthesizability_cases'
DEFAULT_LITERATURE = PROJECT / 'literature_benchmark/ceder_seed80_real_literature_routes.csv'
DEFAULT_OUTDIR = PROJECT / 'literature_benchmark/results/ceder_seed80_route_similarity'
VOLATILE_ELEMENTS = {'C', 'H', 'N', 'O'}
GAS_FORMULAS = {'O2', 'N2', 'H2', 'H3N', 'CO2', 'CO', 'H2O', 'F2', 'Cl2', 'Br', 'Br2', 'I', 'I2'}


@lru_cache(maxsize=None)
def normalize(formula: str) -> str:
    return Composition(str(formula).strip()).reduced_formula


@lru_cache(maxsize=None)
def elements(formula: str) -> set[str]:
    return {str(element) for element in Composition(formula).elements}


@lru_cache(maxsize=None)
def comp_dict(formula: str) -> dict[str, float]:
    return {str(k): float(v) for k, v in Composition(formula).as_dict().items()}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def split_formulas(value: str | None) -> set[str]:
    out = set()
    for item in str(value or '').split(';'):
        item = item.strip()
        if not item:
            continue
        try:
            out.add(normalize(item))
        except Exception:
            continue
    return out


@lru_cache(maxsize=None)
def parse_side_cached(value: str) -> frozenset[str]:
    formulas = set()
    for item in str(value or '').split(' | '):
        item = item.strip()
        if not item:
            continue
        formula = item.split('*', 1)[1] if '*' in item else item
        try:
            formulas.add(normalize(formula))
        except Exception:
            continue
    return frozenset(formulas)


def parse_side(value: str | None) -> set[str]:
    return set(parse_side_cached(str(value or '')))


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def recall(reference: set[str], candidate: set[str]) -> float:
    return len(reference & candidate) / len(reference) if reference else 1.0


def target_elements(target: str) -> set[str]:
    return elements(target)


def is_gas(formula: str) -> bool:
    return normalize(formula) in GAS_FORMULAS


def is_elemental(formula: str) -> bool:
    return len(elements(formula)) == 1


def has_subset(formula: str, required: set[str]) -> bool:
    return required.issubset(elements(formula))


def is_oxide(formula: str) -> bool:
    elems = elements(formula)
    return 'O' in elems and len(elems) == 2 and not is_gas(formula)


def is_carbonate(formula: str) -> bool:
    elems = elements(formula)
    return {'C', 'O'}.issubset(elems) and bool(elems - {'C', 'O'})


def is_hydroxide_or_hydrate(formula: str) -> bool:
    elems = elements(formula)
    return {'H', 'O'}.issubset(elems) and bool(elems - {'H', 'O'})


def is_nitrate_or_ammonium(formula: str) -> bool:
    elems = elements(formula)
    return 'N' in elems and ('O' in elems or 'H' in elems)


def is_phosphate(formula: str) -> bool:
    return has_subset(formula, {'P', 'O'})


def is_oxalate_like(formula: str) -> bool:
    comp = comp_dict(formula)
    return 'C' in comp and 'O' in comp and comp.get('O', 0.0) >= 2.0 * comp.get('C', 0.0) and bool(set(comp) - {'C', 'O'})


@lru_cache(maxsize=None)
def precursor_class(formula: str, target: str) -> str:
    formula = normalize(formula)
    target_elems = target_elements(target)
    elems = elements(formula)
    if is_gas(formula):
        return 'gas_reservoir'
    if is_elemental(formula):
        return 'elemental_source'
    if is_carbonate(formula):
        return 'carbonate_source'
    if is_oxalate_like(formula):
        return 'oxalate_or_organic_salt'
    if is_nitrate_or_ammonium(formula):
        if is_phosphate(formula):
            return 'ammonium_phosphate'
        return 'nitrate_or_ammonium_salt'
    if is_hydroxide_or_hydrate(formula):
        return 'hydroxide_or_hydrate'
    if is_phosphate(formula):
        return 'phosphate_source'
    if is_oxide(formula):
        return 'binary_oxide_source'
    if len(elems) == 2 and (elems & target_elems):
        return 'binary_target_element_source'
    if elems <= target_elems:
        return 'target_element_compound'
    return 'other_precursor'


def source_elements(formulas: frozenset[str] | set[str], target: str) -> set[str]:
    target_elems = target_elements(target)
    out = set()
    for formula in formulas:
        out.update(elements(formula) & target_elems)
    return out


@lru_cache(maxsize=None)
def nonvolatile_source_elements_cached(formulas_key: tuple[str, ...], target: str) -> frozenset[str]:
    formulas = set(formulas_key)
    return frozenset(source_elements(formulas, target) - VOLATILE_ELEMENTS)


def nonvolatile_source_elements(formulas: set[str], target: str) -> set[str]:
    return set(nonvolatile_source_elements_cached(tuple(sorted(formulas)), target))


@lru_cache(maxsize=None)
def class_multiset_cached(formulas_key: tuple[str, ...], target: str) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(Counter(precursor_class(formula, target) for formula in formulas_key).items()))


def class_multiset(formulas: set[str], target: str) -> Counter[str]:
    return Counter(dict(class_multiset_cached(tuple(sorted(formulas)), target)))


def _nonvolatile_source_elements_old(formulas: set[str], target: str) -> set[str]:
    return source_elements(formulas, target) - VOLATILE_ELEMENTS


def counter_jaccard(left: Counter[str], right: Counter[str]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 1.0
    inter = sum(min(left[key], right[key]) for key in keys)
    union = sum(max(left[key], right[key]) for key in keys)
    return inter / union if union else 0.0


def route_similarity(target: str, lit_reactants: set[str], lit_byproducts: set[str], route: dict[str, str]) -> dict[str, object]:
    reactants = parse_side(route.get('reactants'))
    products = parse_side(route.get('products'))
    byproducts = products - {normalize(target)}
    exact_reactant_recall = recall(lit_reactants, reactants)
    exact_reactant_jaccard = jaccard(lit_reactants, reactants)
    byproduct_recall = recall(lit_byproducts, byproducts)
    element_recall = recall(nonvolatile_source_elements(lit_reactants, target), nonvolatile_source_elements(reactants, target))
    element_jaccard = jaccard(nonvolatile_source_elements(lit_reactants, target), nonvolatile_source_elements(reactants, target))
    class_score = counter_jaccard(class_multiset(lit_reactants, target), class_multiset(reactants, target))
    gas_class_score = counter_jaccard(class_multiset(lit_byproducts, target), class_multiset(byproducts, target))
    template_bonus = 0.0
    template = str(route.get('route_template_reason', ''))
    lit_classes = set(class_multiset(lit_reactants, target))
    if 'carbonate_source' in lit_classes and 'carbonate' in template:
        template_bonus += 0.08
    if 'hydroxide_or_hydrate' in lit_classes and ('condition-dependent' in template or 'carbonate' in template or 'oxide' in template):
        template_bonus += 0.04
    if 'ammonium_phosphate' in lit_classes and ('phosphate' in template or 'ceramic' in template):
        template_bonus += 0.08
    if 'oxalate_or_organic_salt' in lit_classes and ('phosphate' in template or 'carbonate' in template or 'condition-dependent' in template):
        template_bonus += 0.06
    score = (
        0.28 * exact_reactant_recall
        + 0.18 * exact_reactant_jaccard
        + 0.20 * element_recall
        + 0.16 * class_score
        + 0.08 * byproduct_recall
        + 0.05 * gas_class_score
        + 0.05 * element_jaccard
        + template_bonus
    )
    score = max(0.0, min(1.0, score))
    if exact_reactant_recall >= 0.999 and byproduct_recall >= 0.999:
        label = 'exact_route_recovered'
    elif score >= 0.78:
        label = 'strong_route_family_match'
    elif score >= 0.62:
        label = 'moderate_route_family_match'
    elif element_recall >= 0.999:
        label = 'same_target_element_sources'
    else:
        label = 'weak_or_missing_route_match'
    return {
        'route_similarity_score': round(score, 6),
        'route_similarity_label': label,
        'exact_reactant_recall': round(exact_reactant_recall, 6),
        'exact_reactant_jaccard': round(exact_reactant_jaccard, 6),
        'byproduct_recall': round(byproduct_recall, 6),
        'target_element_recall': round(element_recall, 6),
        'target_element_jaccard': round(element_jaccard, 6),
        'precursor_class_jaccard': round(class_score, 6),
        'byproduct_class_jaccard': round(gas_class_score, 6),
        'literature_precursor_classes': ';'.join(sorted(class_multiset(lit_reactants, target).elements())),
        'recommended_precursor_classes': ';'.join(sorted(class_multiset(reactants, target).elements())),
        'recommended_reactants': ';'.join(sorted(reactants)),
        'recommended_byproducts': ';'.join(sorted(byproducts)),
        'recommended_reaction': route.get('reaction', ''),
        'recommended_synth_rank': route.get('synth_rank') or route.get('rank', ''),
        'recommended_synth_score': route.get('synth_score', ''),
        'recommended_deltaE_per_target': route.get('synth_rank_deltaE_per_target', ''),
        'recommended_template': route.get('route_template_reason', ''),
        'recommended_risk_flags': route.get('synthesis_risk_flags', ''),
    }


def rank_bucket(rank: object) -> str:
    if rank in (None, '') or pd.isna(rank):
        return 'not_found'
    value = int(float(rank))
    if value == 1:
        return 'top1'
    if value <= 5:
        return 'top5'
    if value <= 20:
        return 'top20'
    return 'below_top20'


def best_similarity(target: str, lit_reactants: set[str], lit_byproducts: set[str], routes: list[dict[str, str]]) -> dict[str, object]:
    best = None
    for route in routes:
        current = route_similarity(target, lit_reactants, lit_byproducts, route)
        rank = current.get('recommended_synth_rank')
        rank_value = int(float(rank)) if rank not in (None, '') else 999999
        key = (float(current['route_similarity_score']), -rank_value)
        if best is None or key > best[0]:
            best = (key, current)
    return best[1] if best else {}


def best_similarity_in_topk(target: str, lit_reactants: set[str], lit_byproducts: set[str], routes: list[dict[str, str]], top_k: int) -> dict[str, object]:
    return best_similarity(target, lit_reactants, lit_byproducts, routes[:top_k])


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_svg(summary: dict[str, object], out: Path) -> None:
    width, height = 1120, 620
    metrics = [
        ('strong/moderate match found', summary['top_all_moderate_or_better'], summary['n_cases'], '#1f78b4'),
        ('strong match found', summary['top_all_strong'], summary['n_cases'], '#2b8a3e'),
        ('top-5 moderate or better', summary['top5_moderate_or_better'], summary['n_cases'], '#66a61e'),
        ('top-5 strong', summary['top5_strong'], summary['n_cases'], '#2b8a3e'),
        ('same target-element sources', summary['top_all_same_element_or_better'], summary['n_cases'], '#7570b3'),
    ]

    def esc(value: object) -> str:
        return str(value).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    def text(x: int, y: int, value: object, size: int = 13, weight: str = 'normal') -> str:
        return f'<text x="{x}" y="{y}" font-family="Arial" font-size="{size}" font-weight="{weight}">{esc(value)}</text>'

    def rect(x: int, y: int, w: float, h: int, color: str) -> str:
        return f'<rect x="{x}" y="{y}" width="{w:.1f}" height="{h}" rx="4" fill="{color}"/>'

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        text(36, 42, 'Route-family similarity to DOI-backed literature recipes', 23, 'bold'),
        text(36, 70, 'Similarity combines exact precursors, target-element sources, precursor classes, byproducts and synthesis templates.', 13),
    ]
    left, top, maxw = 430, 115, 470
    for index, (label, numerator, denominator, color) in enumerate(metrics):
        y = top + index * 58
        frac = numerator / denominator if denominator else 0.0
        parts.append(text(54, y + 21, label, 14))
        parts.append(rect(left, y, maxw, 27, '#e9edf5'))
        parts.append(rect(left, y, maxw * frac, 27, color))
        parts.append(text(left + maxw + 18, y + 20, f'{numerator}/{denominator} = {frac:.2f}', 14, 'bold'))
    parts.append(text(54, 500, 'Interpretation', 15, 'bold'))
    parts.append(text(54, 525, 'This metric is less brittle than exact stoichiometry and better reflects experimental route families.', 12))
    parts.append(text(54, 545, 'It still does not prove an experiment will work; it identifies where the recommendation resembles reported precursor chemistry.', 12))
    parts.append('</svg>')
    out.write_text('\n'.join(parts), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate route-family similarity against literature precursor routes.')
    parser.add_argument('--case-dir', type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument('--literature', type=Path, default=DEFAULT_LITERATURE)
    parser.add_argument('--outdir', type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    rows = []
    for literature_row in read_csv(args.literature):
        target = normalize(literature_row['target'])
        route_path = args.case_dir / target / 'synthesizability_ranked_routes.csv'
        routes = read_csv(route_path)
        lit_reactants = split_formulas(literature_row.get('literature_precursors'))
        lit_byproducts = split_formulas(literature_row.get('literature_byproducts'))
        best_all = best_similarity(target, lit_reactants, lit_byproducts, routes)
        best_top5 = best_similarity_in_topk(target, lit_reactants, lit_byproducts, routes, 5)
        best_top20 = best_similarity_in_topk(target, lit_reactants, lit_byproducts, routes, 20)
        rows.append(
            {
                'target': target,
                'literature_precursors': ';'.join(sorted(lit_reactants)),
                'literature_byproducts': ';'.join(sorted(lit_byproducts)),
                'literature_route_text': literature_row.get('literature_route_text', ''),
                'synthesis_type': literature_row.get('synthesis_type', ''),
                'condition_summary': literature_row.get('condition_summary', ''),
                'n_routes': len(routes),
                'best_all_similarity_score': best_all.get('route_similarity_score', ''),
                'best_all_similarity_label': best_all.get('route_similarity_label', ''),
                'best_all_synth_rank': best_all.get('recommended_synth_rank', ''),
                'best_all_rank_bucket': rank_bucket(best_all.get('recommended_synth_rank', '')),
                'best_all_reaction': best_all.get('recommended_reaction', ''),
                'best_all_template': best_all.get('recommended_template', ''),
                'best_all_risk_flags': best_all.get('recommended_risk_flags', ''),
                'best_all_literature_classes': best_all.get('literature_precursor_classes', ''),
                'best_all_recommended_classes': best_all.get('recommended_precursor_classes', ''),
                'best_top5_similarity_score': best_top5.get('route_similarity_score', ''),
                'best_top5_similarity_label': best_top5.get('route_similarity_label', ''),
                'best_top5_synth_rank': best_top5.get('recommended_synth_rank', ''),
                'best_top5_reaction': best_top5.get('recommended_reaction', ''),
                'best_top20_similarity_score': best_top20.get('route_similarity_score', ''),
                'best_top20_similarity_label': best_top20.get('route_similarity_label', ''),
                'best_top20_synth_rank': best_top20.get('recommended_synth_rank', ''),
                'best_top20_reaction': best_top20.get('recommended_reaction', ''),
            }
        )

    labels_strong = {'exact_route_recovered', 'strong_route_family_match'}
    labels_moderate = {'exact_route_recovered', 'strong_route_family_match', 'moderate_route_family_match'}
    labels_same = labels_moderate | {'same_target_element_sources'}
    summary = {
        'n_cases': len(rows),
        'completed_cases': sum(1 for row in rows if int(row['n_routes']) > 0),
        'top_all_strong': sum(1 for row in rows if row['best_all_similarity_label'] in labels_strong),
        'top_all_moderate_or_better': sum(1 for row in rows if row['best_all_similarity_label'] in labels_moderate),
        'top_all_same_element_or_better': sum(1 for row in rows if row['best_all_similarity_label'] in labels_same),
        'top5_strong': sum(1 for row in rows if row['best_top5_similarity_label'] in labels_strong),
        'top5_moderate_or_better': sum(1 for row in rows if row['best_top5_similarity_label'] in labels_moderate),
        'top20_moderate_or_better': sum(1 for row in rows if row['best_top20_similarity_label'] in labels_moderate),
        'mean_best_all_similarity': round(sum(float(row['best_all_similarity_score'] or 0.0) for row in rows) / len(rows), 6) if rows else 0.0,
        'mean_best_top5_similarity': round(sum(float(row['best_top5_similarity_score'] or 0.0) for row in rows) / len(rows), 6) if rows else 0.0,
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    write_csv(args.outdir / 'route_similarity_eval.csv', rows)
    write_csv(args.outdir / 'route_similarity_summary.csv', [summary])
    label_counts = Counter(str(row['best_all_similarity_label']) for row in rows)
    write_csv(args.outdir / 'route_similarity_label_counts.csv', [{'label': key, 'count': value} for key, value in label_counts.most_common()])
    make_svg(summary, args.outdir / 'route_similarity_summary.svg')

    lines = [
        '# 真实文献路线的通用路线族相似度验证',
        '',
        '## 主要结论',
        '',
        f'- 文献路线数：`{summary["n_cases"]}`；完成目标：`{summary["completed_cases"]}`。',
        f'- 全路线表中 strong/exact 路线族匹配：`{summary["top_all_strong"]}/{summary["n_cases"]}`；moderate 或更好：`{summary["top_all_moderate_or_better"]}/{summary["n_cases"]}`。',
        f'- Top-5 推荐中 strong/exact 路线族匹配：`{summary["top5_strong"]}/{summary["n_cases"]}`；moderate 或更好：`{summary["top5_moderate_or_better"]}/{summary["n_cases"]}`。',
        f'- Top-20 推荐中 moderate 或更好：`{summary["top20_moderate_or_better"]}/{summary["n_cases"]}`。',
        f'- 平均 best-all similarity：`{summary["mean_best_all_similarity"]}`；平均 top-5 similarity：`{summary["mean_best_top5_similarity"]}`。',
        '',
        '## 指标含义',
        '',
        '- 该指标同时比较精确前驱体、目标元素来源、前驱体类别、气体副产物和路线模板。',
        '- 它能把 LiOH、Li2CO3、金属氧化物/氢氧化物、草酸盐/铵盐等实验等价路线纳入同一类解释，比 exact reaction match 更接近实验人员理解。',
        '- 该指标仍不是实验成功率，只表示推荐网络是否包含并优先展示了与文献相似的化学路线。',
        '',
        '## 后续改进',
        '',
        '- 对 moderate 但非 strong 的路线，优先补充前驱体分解热和气体化学势。',
        '- 对 weak/missing 的路线，检查是否属于掺杂、固溶体、含水盐或多步过程，必要时扩展目标集和前驱体库。',
    ]
    (args.outdir / 'ROUTE_SIMILARITY_VALIDATION.md').write_text('\n'.join(lines), encoding='utf-8')
    print(summary)
    print(args.outdir / 'ROUTE_SIMILARITY_VALIDATION.md')


if __name__ == '__main__':
    main()
