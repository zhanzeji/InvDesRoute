#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MODEL_INFO = {
    "InvDesRoute-corpus-informed": ("InvDesRoute", "route-network workflow"),
    "CSLLM-precursor-llama3-release": ("CSLLM", "task-specific precursor generator"),
    "LLM4SYN-TGT2CEQ-v1.2.1-official": ("LLM4SYN TGT2CEQ v1.2.1", "task-specific equation generator"),
    "ChemDFM-v1.5-8B-official-zero-shot": ("ChemDFM-v1.5-8B", "chemistry LLM, zero-shot"),
    "ChemLLM-7B-Chat-official-zero-shot": ("ChemLLM-7B-Chat", "chemistry LLM, zero-shot"),
}


def main() -> None:
    summary = pd.read_csv(RESULTS / "model_comparison_summary.csv")
    diagnostics = pd.read_csv(RESULTS / "prediction_diagnostics.csv")
    bootstrap = pd.read_csv(RESULTS / "top5_target_cluster_bootstrap.csv")
    top5 = summary[(summary.top_k == 5) & summary.model.isin(MODEL_INFO)].copy()
    top5["display_name"] = top5.model.map(lambda model: MODEL_INFO[model][0])
    top5["model_role"] = top5.model.map(lambda model: MODEL_INFO[model][1])
    diag = diagnostics[diagnostics.model.isin(MODEL_INFO)].copy()
    diag = diag.drop_duplicates("model", keep="last")
    moderate_ci = bootstrap[
        (bootstrap.top_k == 5)
        & (bootstrap.metric == "moderate_or_better")
        & bootstrap.model.isin(MODEL_INFO)
    ][["model", "target_cluster_mean", "ci95_low", "ci95_high"]]
    output = top5.merge(diag, on="model", how="left").merge(moderate_ci, on="model", how="left")
    columns = [
        "model",
        "display_name",
        "model_role",
        "targets_with_predictions",
        "strict_exact",
        "strong_or_better",
        "moderate_or_better",
        "mean_precursor_recall",
        "mean_family_similarity",
        "mean_target_element_coverage",
        "ci95_low",
        "ci95_high",
    ]
    output = output[columns].sort_values(
        ["moderate_or_better", "strict_exact"], ascending=[False, False]
    )
    output.to_csv(RESULTS / "official_model_top5_summary.csv", index=False)

    lines = [
        "# 官方模型统一前驱体集合比较",
        "",
        "## 评价边界",
        "",
        "24 个目标、每个目标 5 条 DOI 支撑文献记录，共 120 条记录。所有方法仅比较前驱体集合；副产物被忽略，因为 CSLLM 和零样本化学 LLM 不稳定输出完整配平反应。Exact 要求规范化前驱体集合完全相同；strong/moderate 使用冻结的 InvDesRoute 路线族规则。公开 checkpoint 的训练数据可能包含部分文献或目标，因此这是统一回顾性推理比较，不是严格训练集隔离的前瞻测试。",
        "",
        "## Top-5 结果",
        "",
        "| Model | Role | Targets with valid predictions | Exact | Strong+ | Moderate+ | Mean precursor recall |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in output.to_dict("records"):
        targets = int(row["targets_with_predictions"]) if pd.notna(row["targets_with_predictions"]) else 0
        lines.append(
            f"| {row['display_name']} | {row['model_role']} | {targets}/24 | "
            f"{int(row['strict_exact'])}/120 | {int(row['strong_or_better'])}/120 | "
            f"{int(row['moderate_or_better'])}/120 | {row['mean_precursor_recall']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 解读原则",
            "",
            "- CSLLM 和 LLM4SYN 是与无机合成直接相关的任务专用官方模型；ChemDFM 和 ChemLLM 是冻结提示下的化学领域零样本基线。",
            "- InvDesRoute 输出来自同一个通用前驱体/气体库网络和固定评分，不向候选池注入这 120 条路线的具体前驱体标签。",
            "- Exact 指标对水合、分解和同类前驱体替代很敏感；moderate+ 衡量主要前驱体类别和目标元素来源是否被恢复。",
            "- 该比较衡量已发布模型在统一输入和统一后处理下的直接可用性，不等同于各模型原论文任务上的官方成绩。",
        ]
    )
    (RESULTS / "OFFICIAL_MODEL_COMPARISON_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
