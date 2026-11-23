#!/usr/bin/env python3
"""Compare PET-recommended synthesis routes with literature precursor routes.

The script intentionally uses only the Python standard library so it can run in
the same lightweight environment as the existing analysis utilities.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_PROJECT = Path("/home/zhzhan/synthesis_pathway_screening")
DEFAULT_ROUTES = DEFAULT_PROJECT / "results/mp_pet_benchmark_120/recommended_routes_top5_all.csv"
DEFAULT_SUMMARY = DEFAULT_PROJECT / "results/mp_pet_benchmark_120/synthesis_recommendation_summary_all.csv"
DEFAULT_LIT = DEFAULT_PROJECT / "literature_benchmark/literature_routes_seed.csv"
DEFAULT_OUT = DEFAULT_PROJECT / "literature_benchmark/results"

FORMULA_RE = re.compile(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)")


@dataclass(frozen=True)
class Route:
    target: str
    rank: int
    reaction: str
    pet_delta: float | None
    mp_delta: float | None
    reactants: set[str]
    byproducts: set[str]
    precursor_class: str
    confidence: str
    category: str


@dataclass(frozen=True)
class LiteratureRoute:
    target: str
    precursors: set[str]
    byproducts: set[str]
    route_text: str
    synthesis_type: str
    condition_summary: str
    source_title: str
    source_note: str


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        value_float = float(value)
    except ValueError:
        return None
    if math.isnan(value_float):
        return None
    return value_float


def split_formulas(value: str | None) -> set[str]:
    if not value:
        return set()
    tokens = []
    for part in re.split(r"[;|,]", value):
        clean = normalize_formula_token(part)
        if clean:
            tokens.append(clean)
    return set(tokens)


def normalize_formula_token(token: str) -> str:
    token = token.strip()
    if not token:
        return ""
    token = re.sub(r"^\s*[0-9./]+\s*", "", token)
    token = token.replace(" ", "")
    token = token.replace("$", "")
    token = token.replace("\\rightarrow", "")
    token = token.replace("->", "")
    token = token.replace("(s)", "").replace("(g)", "").replace("(l)", "")
    token = re.sub(r"[^A-Za-z0-9().]", "", token)
    token = token.strip("()")
    if not token or token.lower() in {"gases", "gas", "none"}:
        return ""
    return token


def formulas_from_reaction_side(side: str) -> set[str]:
    formulas = set()
    for token in side.split("+"):
        formula = normalize_formula_token(token)
        if formula:
            formulas.add(formula)
    return formulas


def reaction_reactants(reaction: str) -> set[str]:
    if "->" not in reaction:
        return set()
    left = reaction.split("->", 1)[0]
    return formulas_from_reaction_side(left)


def formula_elements(formula: str) -> set[str]:
    if not formula:
        return set()
    return {match.group(1) for match in FORMULA_RE.finditer(formula)}


def element_set(formulas: Iterable[str]) -> set[str]:
    elements: set[str] = set()
    for formula in formulas:
        elements.update(formula_elements(formula))
    return elements


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def containment(reference: set[str], candidate: set[str]) -> float:
    if not reference:
        return 1.0
    return len(reference & candidate) / len(reference)


def classify_match(precursor_jaccard: float, precursor_recall: float, element_jaccard: float) -> str:
    if precursor_jaccard >= 0.999:
        return "exact_precursor_match"
    if precursor_recall >= 0.999:
        return "literature_precursors_contained"
    if precursor_jaccard >= 0.5:
        return "partial_precursor_match"
    if element_jaccard >= 0.999:
        return "same_element_pool"
    if element_jaccard >= 0.5:
        return "partial_element_match"
    return "weak_or_no_match"


def load_routes(path: Path) -> dict[str, list[Route]]:
    routes_by_target: dict[str, list[Route]] = defaultdict(list)
    for row in read_csv(path):
        reactants = split_formulas(row.get("reactant_formulas"))
        if not reactants:
            reactants = reaction_reactants(row.get("reaction", ""))
        byproducts = split_formulas(row.get("byproduct_formulas"))
        route = Route(
            target=row["target"],
            rank=int(float(row.get("rank", "999"))),
            reaction=row.get("reaction", ""),
            pet_delta=parse_float(row.get("pet_deltaE_per_target")),
            mp_delta=parse_float(row.get("mp_form_deltaE_per_target")),
            reactants=reactants,
            byproducts=byproducts,
            precursor_class=row.get("precursor_complexity_class", ""),
            confidence=row.get("recommendation_confidence", ""),
            category=row.get("category", ""),
        )
        routes_by_target[route.target].append(route)
    for routes in routes_by_target.values():
        routes.sort(key=lambda item: item.rank)
    return routes_by_target


def load_literature(path: Path) -> list[LiteratureRoute]:
    literature = []
    for row in read_csv(path):
        literature.append(
            LiteratureRoute(
                target=row["target"],
                precursors=split_formulas(row.get("literature_precursors")),
                byproducts=split_formulas(row.get("literature_byproducts")),
                route_text=row.get("literature_route_text", ""),
                synthesis_type=row.get("synthesis_type", ""),
                condition_summary=row.get("condition_summary", ""),
                source_title=row.get("source_title", ""),
                source_note=row.get("source_note", ""),
            )
        )
    return literature


def best_match(lit: LiteratureRoute, routes: list[Route], top_k: int) -> dict[str, object]:
    lit_elements = element_set(lit.precursors)
    best: dict[str, object] | None = None
    for route in routes[:top_k]:
        route_elements = element_set(route.reactants)
        precursor_jaccard = jaccard(lit.precursors, route.reactants)
        precursor_recall = containment(lit.precursors, route.reactants)
        precursor_precision = containment(route.reactants, lit.precursors)
        element_jaccard = jaccard(lit_elements, route_elements)
        byproduct_jaccard = jaccard(lit.byproducts, route.byproducts)
        match_type = classify_match(precursor_jaccard, precursor_recall, element_jaccard)
        score = (
            0.55 * precursor_jaccard
            + 0.20 * precursor_recall
            + 0.15 * element_jaccard
            + 0.05 * byproduct_jaccard
            + 0.05 * max(0.0, (top_k + 1 - route.rank) / top_k)
        )
        candidate = {
            "matched_rank": route.rank,
            "recommended_reaction": route.reaction,
            "recommended_reactants": ";".join(sorted(route.reactants)),
            "recommended_byproducts": ";".join(sorted(route.byproducts)),
            "pet_deltaE_per_target": "" if route.pet_delta is None else route.pet_delta,
            "mp_form_deltaE_per_target": "" if route.mp_delta is None else route.mp_delta,
            "precursor_jaccard": precursor_jaccard,
            "precursor_recall": precursor_recall,
            "precursor_precision": precursor_precision,
            "element_jaccard": element_jaccard,
            "byproduct_jaccard": byproduct_jaccard,
            "match_score": score,
            "match_type": match_type,
            "precursor_complexity_class": route.precursor_class,
            "recommendation_confidence": route.confidence,
            "category": route.category,
        }
        if best is None or float(candidate["match_score"]) > float(best["match_score"]):
            best = candidate
    if best is None:
        return {
            "matched_rank": "",
            "recommended_reaction": "",
            "recommended_reactants": "",
            "recommended_byproducts": "",
            "pet_deltaE_per_target": "",
            "mp_form_deltaE_per_target": "",
            "precursor_jaccard": 0.0,
            "precursor_recall": 0.0,
            "precursor_precision": 0.0,
            "element_jaccard": 0.0,
            "byproduct_jaccard": 0.0,
            "match_score": 0.0,
            "match_type": "not_in_recommendation_table",
            "precursor_complexity_class": "",
            "recommendation_confidence": "",
            "category": "",
        }
    return best


def aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    covered = [row for row in rows if row["matched_rank"] != ""]
    exact = [row for row in rows if row["match_type"] == "exact_precursor_match"]
    contained = [row for row in rows if row["match_type"] in {"exact_precursor_match", "literature_precursors_contained"}]
    partial = [
        row
        for row in rows
        if row["match_type"] in {"exact_precursor_match", "literature_precursors_contained", "partial_precursor_match"}
    ]
    same_element = [row for row in rows if row["element_jaccard"] >= 0.999]
    exergonic = [
        row
        for row in rows
        if isinstance(row["pet_deltaE_per_target"], float) and row["pet_deltaE_per_target"] < 0
    ]
    scores = [float(row["match_score"]) for row in covered]
    return {
        "n_literature_targets": len(rows),
        "n_targets_in_recommendation_table": len(covered),
        "coverage_rate": len(covered) / len(rows) if rows else 0.0,
        "exact_precursor_match_rate": len(exact) / len(covered) if covered else 0.0,
        "literature_precursors_contained_rate": len(contained) / len(covered) if covered else 0.0,
        "partial_or_better_precursor_match_rate": len(partial) / len(covered) if covered else 0.0,
        "same_element_pool_rate": len(same_element) / len(covered) if covered else 0.0,
        "exergonic_matched_route_rate": len(exergonic) / len(covered) if covered else 0.0,
        "mean_match_score": sum(scores) / len(scores) if scores else 0.0,
        "median_match_score": sorted(scores)[len(scores) // 2] if scores else 0.0,
    }


def split_by_precursor_scope(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = "target_element_only_literature_precursors"
        if row.get("extra_literature_precursor_elements"):
            key = "literature_precursors_require_external_reservoir_elements"
        grouped[key].append(row)
    output = []
    for key, group_rows in sorted(grouped.items()):
        summary = aggregate(group_rows)
        summary["precursor_scope"] = key
        output.append(summary)
    return output


def category_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("category") or "unmatched")].append(row)
    output = []
    for category, category_rows in sorted(grouped.items()):
        summary = aggregate(category_rows)
        summary["category"] = category
        output.append(summary)
    return output


def write_report(path: Path, overall: dict[str, object], category_rows: list[dict[str, object]], rows: list[dict[str, object]]) -> None:
    match_counts = Counter(str(row["match_type"]) for row in rows)
    scope_rows = split_by_precursor_scope(rows)
    examples = sorted(rows, key=lambda row: float(row["match_score"]), reverse=True)[:8]
    weak = sorted(rows, key=lambda row: float(row["match_score"]))[:8]
    lines = [
        "# 文献合成路线对比分析",
        "",
        "## 结论摘要",
        "",
        f"- 文献种子路线数：{overall['n_literature_targets']}",
        f"- 出现在 120-target 推荐表中的目标数：{overall['n_targets_in_recommendation_table']}，覆盖率 {overall['coverage_rate']:.3f}",
        f"- Top-5 中完全前驱体匹配率：{overall['exact_precursor_match_rate']:.3f}",
        f"- Top-5 中覆盖全部文献前驱体的比例：{overall['literature_precursors_contained_rate']:.3f}",
        f"- Top-5 中至少部分前驱体匹配的比例：{overall['partial_or_better_precursor_match_rate']:.3f}",
        f"- Top-5 中元素池完全一致的比例：{overall['same_element_pool_rate']:.3f}",
        f"- 匹配到的推荐路线为 PET 放热的比例：{overall['exergonic_matched_route_rate']:.3f}",
        f"- 平均路线匹配分数：{overall['mean_match_score']:.3f}",
        "",
        "## 匹配类型分布",
        "",
    ]
    for key, count in match_counts.most_common():
        lines.append(f"- {key}: {count}")
    lines.extend(["", "## 分类统计", ""])
    for row in category_rows:
        lines.append(
            f"- {row['category']}: n={row['n_literature_targets']}, "
            f"exact={row['exact_precursor_match_rate']:.3f}, "
            f"partial+={row['partial_or_better_precursor_match_rate']:.3f}, "
            f"same-element={row['same_element_pool_rate']:.3f}"
        )
    lines.extend(["", "## 文献前驱体范围诊断", ""])
    for row in scope_rows:
        lines.append(
            f"- {row['precursor_scope']}: n={row['n_literature_targets']}, "
            f"exact={row['exact_precursor_match_rate']:.3f}, "
            f"same-element={row['same_element_pool_rate']:.3f}, "
            f"mean-score={row['mean_match_score']:.3f}"
        )
    lines.extend(["", "## 高匹配例子", ""])
    for row in examples:
        lines.append(
            f"- {row['target']}: 文献 `{row['literature_route_text']}`；推荐 rank {row['matched_rank']} "
            f"`{row['recommended_reaction']}`；类型 {row['match_type']}；score={float(row['match_score']):.3f}"
        )
    lines.extend(["", "## 低匹配或需要扩展前驱体库的例子", ""])
    for row in weak:
        lines.append(
            f"- {row['target']}: 文献前驱体 `{row['literature_precursors']}`；推荐前驱体 "
            f"`{row['recommended_reactants']}`；类型 {row['match_type']}；score={float(row['match_score']):.3f}"
        )
    lines.extend(
        [
            "",
            "## 解释",
            "",
            "该分析不要求推荐路线与文献路线完全相同，因为一个目标材料可以有多种合成路线。",
            "完全前驱体匹配说明 top-5 推荐中直接出现了文献路线前驱体；部分匹配说明推荐路线与实验路线共享关键前驱体；元素池一致说明反应化学空间正确，但具体前驱体选择仍需实验知识、前驱体库或气体化学势修正。",
            "如果文献前驱体含有目标元素之外的 C、H、N 等 reservoir 元素，例如碳酸盐、铵盐、草酸盐或氨气，那么当前只按目标元素展开 MP 子体系的 workflow 天然无法枚举这些路线。这类低匹配不应简单解释为 PET 能量模型失败，而应解释为前驱体库和开放组分 reservoir 尚未纳入。",
            "",
            "当前 seed 数据集用于验证比较流程。正式写入论文前，应继续扩充 DOI 级别文献路线或接入公开 text-mined synthesis 数据集。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--literature", type=Path, default=DEFAULT_LIT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    routes_by_target = load_routes(args.routes)
    literature = load_literature(args.literature)

    rows: list[dict[str, object]] = []
    for lit in literature:
        target_elements = formula_elements(lit.target)
        lit_elements = element_set(lit.precursors)
        extra_lit_elements = lit_elements - target_elements
        match = best_match(lit, routes_by_target.get(lit.target, []), args.top_k)
        row = {
            "target": lit.target,
            "target_elements": ";".join(sorted(target_elements)),
            "literature_precursor_elements": ";".join(sorted(lit_elements)),
            "extra_literature_precursor_elements": ";".join(sorted(extra_lit_elements)),
            "literature_precursors": ";".join(sorted(lit.precursors)),
            "literature_byproducts": ";".join(sorted(lit.byproducts)),
            "literature_route_text": lit.route_text,
            "synthesis_type": lit.synthesis_type,
            "condition_summary": lit.condition_summary,
            "source_title": lit.source_title,
            "source_note": lit.source_note,
            **match,
        }
        rows.append(row)

    fieldnames = [
        "target",
        "category",
        "target_elements",
        "literature_precursor_elements",
        "extra_literature_precursor_elements",
        "literature_precursors",
        "literature_byproducts",
        "literature_route_text",
        "synthesis_type",
        "condition_summary",
        "source_title",
        "source_note",
        "matched_rank",
        "recommended_reaction",
        "recommended_reactants",
        "recommended_byproducts",
        "pet_deltaE_per_target",
        "mp_form_deltaE_per_target",
        "precursor_jaccard",
        "precursor_recall",
        "precursor_precision",
        "element_jaccard",
        "byproduct_jaccard",
        "match_score",
        "match_type",
        "precursor_complexity_class",
        "recommendation_confidence",
    ]
    write_csv(args.outdir / "literature_route_match_top5.csv", rows, fieldnames)

    overall = aggregate(rows)
    category_rows = category_summary(rows)
    summary_fields = [
        "n_literature_targets",
        "n_targets_in_recommendation_table",
        "coverage_rate",
        "exact_precursor_match_rate",
        "literature_precursors_contained_rate",
        "partial_or_better_precursor_match_rate",
        "same_element_pool_rate",
        "exergonic_matched_route_rate",
        "mean_match_score",
        "median_match_score",
    ]
    write_csv(args.outdir / "literature_route_match_summary.csv", [overall], summary_fields)
    write_csv(args.outdir / "literature_route_match_category_summary.csv", category_rows, ["category", *summary_fields])
    write_csv(
        args.outdir / "literature_route_match_precursor_scope_summary.csv",
        split_by_precursor_scope(rows),
        ["precursor_scope", *summary_fields],
    )
    write_report(args.outdir / "LITERATURE_ROUTE_BENCHMARK_ANALYSIS.md", overall, category_rows, rows)

    print(f"Wrote {args.outdir / 'literature_route_match_top5.csv'}")
    print(f"Wrote {args.outdir / 'literature_route_match_summary.csv'}")
    print(f"Wrote {args.outdir / 'LITERATURE_ROUTE_BENCHMARK_ANALYSIS.md'}")
    print(overall)


if __name__ == "__main__":
    main()
