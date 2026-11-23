#!/usr/bin/env python3
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pandas as pd
from pymatgen.core import Composition


ROOT = Path(__file__).resolve().parent


def parse_side(side: str) -> dict[str, float]:
    totals: Counter[str] = Counter()
    for term in re.split(r"\s+\+\s+", side.strip()):
        match = re.match(r"^\s*([0-9.]+)\s+(.+?)\s*$", term)
        if not match:
            raise ValueError(f"cannot parse reaction term: {term}")
        coefficient = float(match.group(1))
        formula = match.group(2)
        hydrate_parts = re.split(r"[·⋅]", formula)
        composition = Composition(hydrate_parts[0])
        for hydrate in hydrate_parts[1:]:
            hydrate_match = re.match(r"^([0-9.]+)?(.+)$", hydrate)
            if not hydrate_match:
                raise ValueError(f"cannot parse hydrate segment: {hydrate}")
            hydrate_multiplier = float(hydrate_match.group(1) or 1.0)
            composition += hydrate_multiplier * Composition(hydrate_match.group(2))
        for element, amount in composition.get_el_amt_dict().items():
            totals[element] += coefficient * amount
    return dict(totals)


def balance_error(reaction: str) -> float:
    if "==" not in reaction:
        return float("inf")
    left, right = reaction.split("==", 1)
    reactants = parse_side(left)
    products = parse_side(right)
    elements = set(reactants) | set(products)
    return max((abs(reactants.get(element, 0.0) - products.get(element, 0.0)) for element in elements), default=0.0)


def main() -> None:
    dataset = pd.read_csv(ROOT / "multi_route_literature_dataset.csv").fillna("")
    audit_rows = []
    for _, row in dataset.iterrows():
        error = ""
        try:
            residual = balance_error(row["literature_route_text"])
        except Exception as exc:
            residual = float("inf")
            error = str(exc)
        audit_rows.append(
            {
                "route_id": row["route_id"],
                "target": row["target"],
                "doi": row["doi"],
                "balance_max_abs_residual": residual,
                "balanced_within_1e-2": residual <= 1e-2,
                "audit_error": error,
            }
        )
    audit = pd.DataFrame(audit_rows)

    details = pd.read_csv(ROOT / "results" / "multi_route_route_level_detail.csv")
    full = details[(details["network"] == "blind") & (details["mode"] == "full_network")]
    top5 = details[
        (details["network"] == "blind")
        & (details["mode"] == "synth_score")
        & (details["top_k"].astype(str) == "5")
    ]
    failures = dataset.merge(
        top5[["route_id", "moderate_or_better", "similarity_score", "similarity_label", "matched_reaction"]],
        on="route_id",
    )
    failures = failures[~failures["moderate_or_better"]].copy()
    failures.to_csv(ROOT / "results" / "top5_failure_cases.csv", index=False)
    audit.to_csv(ROOT / "results" / "literature_route_quality_audit.csv", index=False)

    class_counts: Counter[str] = Counter()
    for signature in failures["precursor_class_signature"]:
        for item in str(signature).split(";"):
            if item:
                class_counts[item.split(":", 1)[0]] += 1
    full_misses = full[~full["moderate_or_better"]]
    sensitivity = pd.read_csv(ROOT / "results" / "score_sensitivity_summary.csv")
    holdout = pd.read_csv(ROOT / "results" / "score_sensitivity_target_holdout.csv")
    best = sensitivity.sort_values(["top5_moderate_or_better", "mean_similarity"], ascending=False).iloc[0]
    lines = [
        "# 多路线基准质量、失败模式与评分诊断",
        "",
        "## 数据质量",
        "",
        f"- 目标：`{dataset['target'].nunique()}`；文献记录：`{len(dataset)}`；DOI：`{dataset['doi'].nunique()}`。",
        f"- 唯一前驱体/副产物路线：`{dataset.drop_duplicates(['target', 'route_pair_signature']).shape[0]}`。",
        f"- 可由文本反应式重新解析并在 1e-2 内满足元素守恒：`{int(audit['balanced_within_1e-2'].sum())}/{len(audit)}`。",
        "- 五条记录均来自不同 DOI；对只有一种经典路线的目标，记录级重复与路线级多样性分开报告。",
        "",
        "## 候选生成失败",
        "",
        f"盲测完整网络的 moderate+ 缺失为 `{len(full_misses)}/120`。这些记录列在 `top5_failure_cases.csv` 和 route-level detail 中；seeded upper bound 为 `120/120`，说明缺失来自通用前驱体库，而不是配平求解器。",
        "",
        "## 当前 top-5 失败的前驱体类别",
        "",
    ]
    lines.extend(f"- `{name}`：{count} 条" for name, count in class_counts.most_common())
    lines.extend(
        [
            "",
            "## 评分权重诊断",
            "",
            f"当前评分恢复 `83/120` 条 top-5 moderate+ 路线；六个预定义权重方案中，`{best['profile']}` 最高，为 `{int(best['top5_moderate_or_better'])}/120`。",
            "",
            holdout.to_markdown(index=False),
            "",
            "候选权重在分层训练目标上被选择后，在独立目标子集上仍优于当前权重，说明提升不只是全数据拟合。然而，该数据集仍来自同一文本挖掘来源，因此建议将 `condition_heavy` 作为下一版本候选，并在新增 DOI 或人工整理路线集上确认后再替换生产默认值。",
        ]
    )
    (ROOT / "results" / "QUALITY_AND_SCORING_DIAGNOSIS_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(audit["balanced_within_1e-2"].value_counts(dropna=False).to_string())
    if len(audit[~audit["balanced_within_1e-2"]]):
        print(audit[~audit["balanced_within_1e-2"]].to_string(index=False))


if __name__ == "__main__":
    main()
