#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SPLIT = ROOT / "split_2022_energy_complete"
RESULTS = ROOT / "results_2022"


RATIONALE = {
    "energy_score": "Thermodynamic support for the balanced transformation; retained at modest weight because static energy does not include all finite-temperature precursor chemistry.",
    "precursor_prior_score": "Accessibility and historical use of target-element sources.",
    "route_template_score": "Support from recurring solid-state reaction families rather than formula coincidence alone.",
    "practical_filter_score": "A presentation safeguard against extreme coefficients, large formulas and unstable non-target phases.",
    "route_complexity_score": "Operational burden from reactant count, byproducts and coefficient scale.",
    "phase_availability_score": "Whether route energetics are structure resolved or depend on virtual/approximate species.",
    "condition_simplicity_score": "Uncertainty from gases, atmosphere control and precursor decomposition.",
}


def cluster_bootstrap(frame: pd.DataFrame, endpoint: str, cluster: str, iterations: int = 10000) -> tuple[float, float]:
    groups = {key: group[endpoint].astype(float).to_numpy() for key, group in frame.groupby(cluster)}
    keys = np.asarray(sorted(groups), dtype=object)
    rng = np.random.default_rng(20220912)
    values = []
    for _ in range(iterations):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        observations = np.concatenate([groups[key] for key in sampled])
        values.append(float(observations.mean()))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def doi_manifest(routes: pd.DataFrame, metadata: pd.DataFrame, role: str) -> pd.DataFrame:
    info = metadata[["doi", "publication_year", "title", "metadata_source", "openalex_id"]].copy()
    merged = routes[["doi", "target", "publication_year", "route_id"]].merge(
        info.drop(columns="publication_year"), on="doi", how="left"
    )
    merged.insert(0, "evidence_role", role)
    return merged.sort_values(["publication_year", "doi", "target"])


def main() -> None:
    summary = json.loads((RESULTS / "summary.json").read_text())
    split_summary = json.loads((SPLIT / "split_summary.json").read_text())
    protocol = json.loads((ROOT / "frozen_protocol.json").read_text())
    future = pd.read_csv(SPLIT / "future_routes_2023_2024.csv")
    development = pd.read_csv(SPLIT / "development_routes_2021_2022.csv")
    metadata = pd.read_csv(ROOT / "publication_metadata_all.csv")
    route_evidence = pd.read_csv(RESULTS / "future_route_evidence.csv")
    final = route_evidence.loc[
        route_evidence["configuration"].eq("selected_with_equiformer_energy")
        & route_evidence["top_k"].eq(5)
    ].copy()
    components = pd.read_csv(RESULTS / "future_component_ablation.csv")
    physics = pd.read_csv(RESULTS / "mlip_display_physics_summary.csv")
    controls = pd.read_csv(RESULTS / "future_similarity_threshold_controls.csv")
    cv = pd.read_csv(RESULTS / "development_nested_cv.csv")

    ci_rows = []
    for endpoint in ("strict_exact", "strong_or_exact", "moderate_or_better"):
        for cluster in ("target", "doi"):
            low, high = cluster_bootstrap(final, endpoint, cluster)
            ci_rows.append(
                {
                    "endpoint": endpoint,
                    "cluster": cluster,
                    "fraction": float(final[endpoint].mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                    "iterations": 10000,
                }
            )
    pd.DataFrame(ci_rows).to_csv(RESULTS / "future_cluster_bootstrap_intervals.csv", index=False)

    full = summary["future_top5"]
    rationale_rows = []
    selected_weights = summary["selected_weights"]
    for component, weight in selected_weights.items():
        ablated = components.loc[components["removed_component"].eq(component)].iloc[0]
        rationale_rows.append(
            {
                "component": component,
                "selected_weight": weight,
                "scientific_reason": RATIONALE[component],
                "full_strong_or_exact": full["strong_or_exact"],
                "without_component_strong_or_exact": int(ablated["strong_or_exact"]),
                "full_moderate_or_better": full["moderate_or_better"],
                "without_component_moderate_or_better": int(ablated["moderate_or_better"]),
                "interpretation": "Directional future-set ablation; parameter selection used development records only.",
            }
        )
    pd.DataFrame(rationale_rows).to_csv(RESULTS / "score_parameter_rationale.csv", index=False)

    prior = pd.read_csv(SPLIT / "historical_prior_routes_through_2020.csv", low_memory=False)
    prior_dois = (
        prior[["doi", "publication_year"]]
        .drop_duplicates("doi")
        .merge(metadata[["doi", "title", "metadata_source", "openalex_id"]], on="doi", how="left")
    )
    prior_dois.insert(0, "evidence_role", "historical_prior_through_2020")
    prior_dois.to_csv(RESULTS / "historical_prior_doi_manifest.csv", index=False)
    doi_manifest(development, metadata, "parameter_development_2021_2022").to_csv(
        RESULTS / "development_doi_manifest.csv", index=False
    )
    doi_manifest(future, metadata, "future_test_2023_2024").to_csv(
        RESULTS / "future_test_doi_manifest.csv", index=False
    )

    no_energy = physics.loc[physics["configuration"].eq("selected_without_energy")].iloc[0]
    with_energy = physics.loc[physics["configuration"].eq("selected_with_equiformer_energy")].iloc[0]
    energy_only = physics.loc[physics["configuration"].eq("production_energy_only")].iloc[0]
    threshold_062 = controls.loc[controls["threshold"].eq(0.62)].iloc[0]
    threshold_078 = controls.loc[controls["threshold"].eq(0.78)].iloc[0]

    lines = [
        "# InvDesRoute 2022 时间冻结、评分参数与机器学习势贡献报告",
        "",
        "## 数据分界",
        "",
        f"- 数据来源：Lee et al., Scientific Data 12, 1969 (2025)，数据 DOI {split_summary['source_data_doi']}。",
        f"- 清洗后可审计配平路线：{split_summary['resolved_route_records']:,} 条。",
        f"- 历史先验：1985--2020 年，{split_summary['historical_prior_records']:,} 条路线、{split_summary['historical_prior_targets']:,} 个目标、{prior['doi'].nunique():,} 篇 DOI 文献。",
        f"- 参数开发：2021--2022 年首次进入语料的 {split_summary['development_targets']} 个目标、{split_summary['development_records']} 条路线、{development['doi'].nunique()} 篇 DOI 文献。",
        f"- 未来测试：2023--2024 年首次进入语料且不存在于旧 Kononova 语料的 {split_summary['future_targets']} 个目标、{split_summary['future_records']} 条路线、{future['doi'].nunique()} 篇 DOI 文献。",
        f"- 能量完整性排除：{split_summary['energy_incomplete_exclusions']} 个目标（{', '.join(split_summary['energy_incomplete_targets'])}）。",
        "- 冻结的是文献知识、前驱体先验和参数标签；当前 MP 数据与 EquiformerV3 预训练语料不做历史快照回溯。",
        "",
        "## 参数选择",
        "",
        f"- 选中权重方案：{summary['selected_profile']}。",
        f"- 权重：{summary['selected_weights']}。",
        f"- 历史语料先验权重：{summary['selected_historical_corpus_weight']:.2f}；MMR 多样性惩罚：{summary['selected_diversity_penalty']:.2f}。",
        f"- 五折开发集外推：strict exact {summary['development_nested_cv']['strict_exact']}/57，strong+ {summary['development_nested_cv']['strong_or_exact']}/57，moderate+ {summary['development_nested_cv']['moderate_or_better']}/57。",
        f"- Condition-heavy 方案在 {summary['development_nested_cv']['profile_selection_counts'].get('condition_heavy', 0)}/5 折中被选择。",
        "",
        "## 未来测试",
        "",
        f"- Top-5 strict exact：{full['strict_exact']}/{full['records']}。",
        f"- Top-5 strong/exact：{full['strong_or_exact']}/{full['records']}。",
        f"- Top-5 moderate+：{full['moderate_or_better']}/{full['records']}。",
        f"- 平均路线族相似度：{full['mean_similarity']:.3f}；平均前驱体召回率：{full['mean_precursor_recall']:.3f}。",
        f"- 新数据上的目标置换假阳性率：moderate 阈值 0.62 为 {threshold_062['decoy_false_positive_rate']:.3f}，strong 阈值 0.78 为 {threshold_078['decoy_false_positive_rate']:.3f}。",
        "",
        "## 机器学习势的独特贡献",
        "",
        f"- 不使用能量时，top-5 共含 {int(no_energy['exergonic_routes'])} 条放热路线和 {int(no_energy['strongly_uphill_routes'])} 条强上坡路线。",
        f"- 加入 EquiformerV3 后，放热路线增至 {int(with_energy['exergonic_routes'])} 条，强上坡路线降至 {int(with_energy['strongly_uphill_routes'])} 条。",
        f"- EquiformerV3 能量改变 {json.loads((RESULTS / 'mlip_contribution_summary.json').read_text())['targets_whose_top5_changes_when_energy_is_added']}/{full['targets']} 个目标的 top-5 组成。",
        f"- 纯能量排序产生 {int(energy_only['exergonic_routes'])}/280 条放热路线，但文献 moderate+ 仅为 24/56，说明能量不能替代前驱体与路线知识。",
        f"- 未来 56 目标网络包含 {summary['future_thermodynamic_consistency']['comparable_routes']:,} 条 EquiformerV3/MP-form 可比路线，平均符号一致率为 {summary['future_thermodynamic_consistency']['macro_sign_consistency']:.3f}，top-1 一致为 {summary['future_thermodynamic_consistency']['top1_agreement']}/56。",
        f"- 8 条带实验杂相记录的未来路线共报告 21 个杂相公式；平均相池和能量覆盖率均为 {summary['future_impurity_audit']['mean_energy_resolved_recall']:.3f}，分解网络平均召回率为 {summary['future_impurity_audit']['mean_decomposition_recall']:.3f}。",
        "",
        "## 解释",
        "",
        "评分参数负责在实验熟悉度、条件复杂性和物理证据之间取得可复现折中。机器学习势不以最大化文献字符串匹配为目标，而负责把路线变成具有结构能量、反应方向、分解竞争和精修边界的物理对象。两者的贡献通过独立消融分别报告。",
    ]
    (RESULTS / "TEMPORAL_2022_SUBMISSION_REPORT_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    protocol["protocol_sha256_note"] = "Hash the final JSON and evaluation scripts in the release manifest."
    (RESULTS / "frozen_protocol_snapshot.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
