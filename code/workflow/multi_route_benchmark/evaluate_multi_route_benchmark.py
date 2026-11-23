#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = PROJECT / "multi_route_benchmark"
LITERATURE_DIR = PROJECT / "literature_benchmark"
sys.path.insert(0, str(LITERATURE_DIR))

from evaluate_route_similarity import (  # noqa: E402
    best_similarity_in_topk,
    normalize,
    parse_side,
    precursor_class,
    route_similarity,
    split_formulas,
)


TOP_K_VALUES = [1, 3, 5, 10, 20]
MODERATE_LABELS = {
    "exact_route_recovered",
    "strong_route_family_match",
    "moderate_route_family_match",
}
STRONG_LABELS = {"exact_route_recovered", "strong_route_family_match"}


def safe_float(value: object, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def pet_energy(row: dict[str, object]) -> float:
    for column in (
        "DeltaE_PRODUCTION_FORM_ALIGNED_per_target",
        "DeltaE_EquiformerV3_FORM_ALIGNED_per_target",
        "DeltaE_PET-OAM-XI_per_target",
        "DeltaE_PET_FORM_ALIGNED_per_target",
    ):
        value = safe_float(row.get(column), math.inf)
        if math.isfinite(value):
            return value
    return math.inf


def route_family_key(row: dict[str, object]) -> tuple[str, str]:
    return str(row.get("route_template_reason", "")), str(row.get("reactant_classes", ""))


def class_signature(target: str, formulas: set[str]) -> str:
    classes = Counter(precursor_class(formula, target) for formula in formulas)
    return ";".join(f"{name}:{count}" for name, count in sorted(classes.items()))


def build_leave_target_out_frequency(dataset: pd.DataFrame, ceder_path: Path) -> Counter[str]:
    held_out_targets = {normalize(value) for value in dataset["target"]}
    corpus = pd.read_csv(ceder_path).fillna("")
    counts: Counter[str] = Counter()
    for _, row in corpus.iterrows():
        try:
            target = normalize(row["target"])
        except Exception:
            continue
        if target in held_out_targets:
            continue
        precursors = split_formulas(row.get("literature_precursors"))
        if not precursors:
            continue
        counts[class_signature(target, precursors)] += 1
    return counts


def annotate_frequency(routes: list[dict[str, object]], target: str, frequencies: Counter[str]) -> None:
    for route in routes:
        signature = class_signature(target, parse_side(route.get("reactants")))
        route["leave_target_out_class_frequency"] = frequencies.get(signature, 0)


def chemistry_prior_score(row: dict[str, object]) -> float:
    return (
        0.34 * safe_float(row.get("precursor_prior_score"), 0.0)
        + 0.34 * safe_float(row.get("route_template_score"), 0.0)
        + 0.12 * safe_float(row.get("practical_filter_score"), 0.0)
        + 0.08 * safe_float(row.get("route_complexity_score"), 0.0)
        + 0.07 * safe_float(row.get("phase_availability_score"), 0.0)
        + 0.05 * safe_float(row.get("condition_simplicity_score"), 0.0)
    )


def diversity_order(routes: list[dict[str, object]]) -> list[dict[str, object]]:
    occurrence: Counter[tuple[str, str]] = Counter()
    annotated = []
    for route in routes:
        key = route_family_key(route)
        duplicate_index = occurrence[key]
        occurrence[key] += 1
        adjusted_score = safe_float(route.get("synth_score"), 0.0) - 0.005 * duplicate_index
        annotated.append((adjusted_score, pet_energy(route), str(route.get("reaction", "")), route))
    annotated.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [item[-1] for item in annotated]


def rank_routes(routes: list[dict[str, object]], mode: str) -> list[dict[str, object]]:
    if mode == "full_network":
        ranked = list(routes)
    elif mode == "pet_energy":
        ranked = sorted(routes, key=lambda row: (pet_energy(row), str(row.get("reaction", ""))))
    elif mode == "practical_pet_energy":
        ranked = sorted(
            routes,
            key=lambda row: (0 if truthy(row.get("practical_route")) else 1, pet_energy(row), str(row.get("reaction", ""))),
        )
    elif mode == "chemistry_prior_generator":
        ranked = sorted(
            routes,
            key=lambda row: (-chemistry_prior_score(row), pet_energy(row), str(row.get("reaction", ""))),
        )
    elif mode == "leave_target_out_frequency":
        ranked = sorted(
            routes,
            key=lambda row: (
                -safe_float(row.get("leave_target_out_class_frequency"), 0.0),
                -chemistry_prior_score(row),
                pet_energy(row),
                str(row.get("reaction", "")),
            ),
        )
    elif mode in {"synth_score", "diverse_synth_score"}:
        ranked = sorted(
            routes,
            key=lambda row: (-safe_float(row.get("synth_score"), 0.0), pet_energy(row), str(row.get("reaction", ""))),
        )
        if mode == "diverse_synth_score":
            ranked = diversity_order(ranked)
    else:
        raise ValueError(mode)
    for rank, route in enumerate(ranked, start=1):
        route["evaluation_rank"] = rank
        route["synth_rank"] = rank
    return ranked


def is_moderate(label: object) -> bool:
    return str(label) in MODERATE_LABELS


def is_strong(label: object) -> bool:
    return str(label) in STRONG_LABELS


def load_routes(case_root: Path, target: str, frequencies: Counter[str]) -> list[dict[str, object]]:
    path = case_root / target / "synthesizability_ranked_routes.csv"
    routes = pd.read_csv(path).fillna("").to_dict("records")
    annotate_frequency(routes, target, frequencies)
    return routes


def evaluate_network(
    dataset: pd.DataFrame,
    case_root: Path,
    network_name: str,
    frequencies: Counter[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    modes = [
        "full_network",
        "pet_energy",
        "practical_pet_energy",
        "chemistry_prior_generator",
        "leave_target_out_frequency",
        "synth_score",
        "diverse_synth_score",
    ]
    target_routes = {
        target: load_routes(case_root, target, frequencies)
        for target in sorted(dataset["target"].map(normalize).unique())
    }
    ranked_cache = {
        (target, mode): rank_routes([dict(route) for route in routes], mode)
        for target, routes in target_routes.items()
        for mode in modes
    }
    details = []
    for _, literature in dataset.iterrows():
        target = normalize(literature["target"])
        lit_reactants = split_formulas(literature.get("literature_precursors"))
        lit_byproducts = split_formulas(literature.get("literature_byproducts"))
        for mode in modes:
            ranked = ranked_cache[(target, mode)]
            top_values = [len(ranked)] if mode == "full_network" else TOP_K_VALUES
            for top_k in top_values:
                best = best_similarity_in_topk(target, lit_reactants, lit_byproducts, ranked, top_k)
                details.append(
                    {
                        "network": network_name,
                        "mode": mode,
                        "top_k": "all" if mode == "full_network" else top_k,
                        "route_id": literature["route_id"],
                        "target": target,
                        "material_category": literature["material_category"],
                        "doi": literature["doi"],
                        "route_pair_signature": literature["route_pair_signature"],
                        "literature_precursors": literature["literature_precursors"],
                        "literature_byproducts": literature["literature_byproducts"],
                        "similarity_score": best.get("route_similarity_score", 0.0),
                        "similarity_label": best.get("route_similarity_label", "weak_or_missing_route_match"),
                        "moderate_or_better": is_moderate(best.get("route_similarity_label")),
                        "strong_or_exact": is_strong(best.get("route_similarity_label")),
                        "exact": best.get("route_similarity_label") == "exact_route_recovered",
                        "matched_reaction": best.get("recommended_reaction", ""),
                        "matched_rank": best.get("recommended_synth_rank", ""),
                        "matched_template": best.get("recommended_template", ""),
                    }
                )
    detail = pd.DataFrame(details)
    target_rows = []
    for keys, group in detail.groupby(["network", "mode", "top_k", "target"], sort=False):
        network, mode, top_k, target = keys
        unique_pairs = group.drop_duplicates("route_pair_signature")
        target_rows.append(
            {
                "network": network,
                "mode": mode,
                "top_k": top_k,
                "target": target,
                "material_category": group["material_category"].iloc[0],
                "records_recovered": int(group["moderate_or_better"].sum()),
                "records_total": len(group),
                "unique_pairs_recovered": int(unique_pairs["moderate_or_better"].sum()),
                "unique_pairs_total": len(unique_pairs),
                "all_five_records_recovered": bool(group["moderate_or_better"].all()),
                "at_least_three_records_recovered": int(group["moderate_or_better"].sum()) >= 3,
                "mean_similarity": float(pd.to_numeric(group["similarity_score"], errors="coerce").fillna(0).mean()),
            }
        )
    return detail, pd.DataFrame(target_rows)


def summarize(detail: pd.DataFrame, target_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in detail.groupby(["network", "mode", "top_k"], sort=False):
        network, mode, top_k = keys
        unique_pairs = group.drop_duplicates(["target", "route_pair_signature"])
        targets = target_metrics[
            (target_metrics["network"] == network)
            & (target_metrics["mode"] == mode)
            & (target_metrics["top_k"].astype(str) == str(top_k))
        ]
        rows.append(
            {
                "network": network,
                "mode": mode,
                "top_k": top_k,
                "record_count": len(group),
                "record_moderate_or_better": int(group["moderate_or_better"].sum()),
                "record_strong_or_exact": int(group["strong_or_exact"].sum()),
                "record_exact": int(group["exact"].sum()),
                "record_mean_similarity": float(pd.to_numeric(group["similarity_score"], errors="coerce").fillna(0).mean()),
                "unique_pair_count": len(unique_pairs),
                "unique_pair_moderate_or_better": int(unique_pairs["moderate_or_better"].sum()),
                "targets_all_five_recovered": int(targets["all_five_records_recovered"].sum()),
                "targets_at_least_three_recovered": int(targets["at_least_three_records_recovered"].sum()),
                "mean_records_recovered_per_target": float(targets["records_recovered"].mean()),
            }
        )
    return pd.DataFrame(rows)


def make_report(dataset: pd.DataFrame, summary: pd.DataFrame, target_metrics: pd.DataFrame, out: Path) -> None:
    def row(network: str, mode: str, top_k: object) -> pd.Series:
        selected = summary[
            (summary["network"] == network)
            & (summary["mode"] == mode)
            & (summary["top_k"].astype(str) == str(top_k))
        ]
        return selected.iloc[0]

    blind_full = row("blind", "full_network", "all")
    mp_only_full = row("mp_only", "full_network", "all")
    seeded_full = row("seeded_upper_bound", "full_network", "all")
    lines = [
        "# 24 目标多文献路线基准",
        "",
        "## 基准设计",
        "",
        f"- 目标数：`{dataset['target'].nunique()}`；DOI 文献记录：`{len(dataset)}`；每个目标固定 `5` 条记录。",
        f"- 去除重复前驱体/副产物写法后，共 `{dataset.drop_duplicates(['target', 'route_pair_signature']).shape[0]}` 条唯一文献路线。",
        "- `blind` 网络只使用 MP 相、预先定义的通用前驱体库和气体库；生成阶段看不到目标对应的五条文献路线。",
        "- `seeded_upper_bound` 网络额外加入文献中出现的目标特异前驱体，用于测量候选空间的理论上限，不作为无泄漏性能。",
        "- 所有覆盖率同时按 120 条文献记录、去重路线和每目标五条记录三个层级统计。",
        "",
        "## 候选空间覆盖",
        "",
        f"仅使用 MP 相时，完整网络恢复 `{int(mp_only_full.record_moderate_or_better)}/120` 条记录和 `{int(mp_only_full.unique_pair_moderate_or_better)}/{int(mp_only_full.unique_pair_count)}` 条唯一文献路线。加入预先定义的通用前驱体与气体库后，盲测覆盖提高到 `{int(blind_full.record_moderate_or_better)}/120` 和 `{int(blind_full.unique_pair_moderate_or_better)}/{int(blind_full.unique_pair_count)}`；目标特异前驱体注入上限为 `{int(seeded_full.record_moderate_or_better)}/120` 和 `{int(seeded_full.unique_pair_moderate_or_better)}/{int(seeded_full.unique_pair_count)}`。",
        "",
        "## 排序结果",
        "",
        "下表均为盲测网络。`moderate+` 表示 exact、strong 或 moderate 路线族匹配。",
        "",
        "| 排序策略 | top-k | 记录 moderate+ | 唯一路线 moderate+ | 至少恢复 3/5 的目标 | 平均每目标恢复 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    display_modes = [
        ("pet_energy", "生产模型能量"),
        ("practical_pet_energy", "practical + 生产模型能量"),
        ("chemistry_prior_generator", "化学先验生成基线"),
        ("leave_target_out_frequency", "留目标文献频率基线"),
        ("synth_score", "InvDesRoute 评分"),
        ("diverse_synth_score", "InvDesRoute + 多样性"),
    ]
    for mode, label in display_modes:
        for top_k in (5, 10):
            current = row("blind", mode, top_k)
            lines.append(
                f"| {label} | {top_k} | {int(current.record_moderate_or_better)}/120 | "
                f"{int(current.unique_pair_moderate_or_better)}/{int(current.unique_pair_count)} | "
                f"{int(current.targets_at_least_three_recovered)}/24 | {current.mean_records_recovered_per_target:.2f}/5 |"
            )
    lines.extend(
        [
            "",
            "## 解释原则",
            "",
            "- `full_network` 回答候选反应是否被枚举出来；它不是推荐性能。",
            "- `生产模型能量` 回答仅按当前生产模型的 0 K 反应能排序能否接近实验路线。",
            "- `化学先验生成基线` 不使用能量，只使用通用前驱体、模板、复杂度和条件先验。",
            "- `留目标文献频率基线` 的统计语料排除了全部 24 个测试目标，避免直接记忆测试路线。",
            "- `InvDesRoute + 多样性` 在评分后限制近重复路线占满推荐窗口，适合评价一个目标的多条可选路线是否同时可见。",
            "",
            "## 数据质量边界",
            "",
            "五条记录来自独立 DOI，但并不保证五种不同化学路线。对于文献高度集中于一种经典合成的目标（例如 Bi2Te3），记录级结果反映跨论文重复性，唯一反应路线指标才反映化学多样性。文本挖掘得到的反应已经过目标侧、化学式和明显解析伪影过滤，但仍应在投稿前对最终引用的逐条案例进行人工核对。",
        ]
    )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a leakage-controlled multi-route synthesis benchmark.")
    parser.add_argument("--dataset", type=Path, default=BENCHMARK_DIR / "multi_route_literature_dataset.csv")
    parser.add_argument("--blind-root", type=Path, default=BENCHMARK_DIR / "cases_blind")
    parser.add_argument("--mp-only-root", type=Path, default=BENCHMARK_DIR / "cases_mp_only")
    parser.add_argument("--seeded-root", type=Path, default=BENCHMARK_DIR / "cases_seeded_upper_bound")
    parser.add_argument("--ceder", type=Path, default=LITERATURE_DIR / "ceder_text_mined_routes.csv")
    parser.add_argument("--outdir", type=Path, default=BENCHMARK_DIR / "results")
    args = parser.parse_args()

    dataset = pd.read_csv(args.dataset).fillna("")
    frequencies = build_leave_target_out_frequency(dataset, args.ceder)
    all_details = []
    all_targets = []
    for name, root in (
        ("mp_only", args.mp_only_root),
        ("blind", args.blind_root),
        ("seeded_upper_bound", args.seeded_root),
    ):
        detail, target_metrics = evaluate_network(dataset, root, name, frequencies)
        all_details.append(detail)
        all_targets.append(target_metrics)
    detail = pd.concat(all_details, ignore_index=True)
    target_metrics = pd.concat(all_targets, ignore_index=True)
    summary = summarize(detail, target_metrics)
    args.outdir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.outdir / "multi_route_route_level_detail.csv", index=False)
    target_metrics.to_csv(args.outdir / "multi_route_target_level_metrics.csv", index=False)
    summary.to_csv(args.outdir / "multi_route_mode_summary.csv", index=False)
    (args.outdir / "leave_target_out_class_frequencies.json").write_text(
        json.dumps(dict(frequencies.most_common()), indent=2, ensure_ascii=True), encoding="utf-8"
    )
    make_report(dataset, summary, target_metrics, args.outdir / "CORE_NETWORK_BENCHMARK_REPORT_CN.md")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
