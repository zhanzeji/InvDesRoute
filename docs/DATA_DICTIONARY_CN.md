# InvDesRoute 数据字典（中文）

本数据字典解释 `data/` 中各类表格对应论文中的哪些证据。

## 一、模型选择证据：`support_tables/model_evidence/`

用于回答为什么选择 PET-OAM-XL 作为生产能量引擎。

- `compound_model_metrics.csv`：107 个反应网络涉及化合物的单相能量误差。主要指标为 phase MAE，单位 eV atom^-1。
- `reaction_model_metrics.csv`：416 条复杂氢化物配平反应的反应能误差。主要指标包括 reaction MAE、RMSE、sign consistency。
- `batch_model_summary_aggregated.csv`：37 个氢化物目标层面的路径排序统计。
- `batch_target_summary.csv`：每个氢化物目标的逐目标结果。
- `batch_top5_dft_paths.csv`：DFT top 路线及模型 top 路线比较。

论文用途：支撑 PET-OAM-XL 在相能、反应能和路线排序上优于对比模型。

## 二、120 目标主 benchmark：`support_tables/benchmark_120/`

用于支撑公式/CIF 到合成路线报告的主流程可运行性。

- `mp_pet_benchmark_summary.csv`：120 个目标的逐目标 PET/MP-form 对比。
- `mp_pet_reaction_data.csv`：17,085 条 PET/MP-form 可比形成反应。
- `synthesis_recommendation_summary_all.csv`：每个目标的最终推荐、置信标签、风险标记。
- `recommended_routes_top5_all.csv`：每个目标的 top-5 推荐路线。
- `recommendation_category_summary.csv`：按材料类别统计的推荐表现。
- `dft_refinement_queue.csv`：需要 DFT 或专家进一步精修的目标。

论文用途：支撑 120 targets、17,085 comparable routes、93.1% sign consistency、99/120 top-1 agreement、118/120 exergonic practical routes、104/15/1 high/medium/low confidence 等结果。

## 三、文献路线验证：`support_tables/literature_validation_initial_80/` 与 `support_tables/literature_validation_combined_145/`

用于证明推荐路线不仅热力学上合理，而且接近实验文献中真实使用的路线族。

### 初始 80-target 验证

- `literature_routes_seed*.csv`：人工整理或文献种子路线。
- `route_similarity_eval.csv`：计算路线与文献路线的 exact/strong/moderate similarity。
- `recommendation_ablation_summary.csv`：比较 PET energy only、practical+energy、synthesizability score、diverse display。

### 合并 145 DOI-backed 验证

- `selected_literature_routes.csv`：扩展的 DOI-backed 文献路线。
- `combined_literature_similarity_summary.csv`：145 条文献路线的 exact/strong/moderate recovery。
- `combined_recommendation_ablation_summary.csv`：145 条路线上的排序消融。
- `combined_literature_precursor_class_diagnostics.csv`：按前驱体类别统计失败模式。
- `combined_literature_top5_miss_refinement_queue.csv`：完整网络中存在类似路线但未进入 top-5 的案例。

论文用途：支撑 full network 143/145 moderate-or-better、141/145 strong/exact，以及 top-5 recovery 从 13/145 提升到 113/145 的结论。

## 四、排序和风险诊断：`support_tables/ranking_and_risk_diagnostics/`

用于解释为什么需要 synthesizability score，而不只按最低反应能排序。

- `combined_literature_precursor_class_diagnostics.csv`：二元氧化物、碳酸盐、气体库、氢氧化物/水合物、磷酸盐、硝酸盐/铵盐等类别的恢复率。
- `condition_sensitivity_practical_summary.csv`：700/1000/1300 K 简化气体化学势敏感性结果。
- `dft_refinement_queue_120.csv`：120-target 中需要进一步精修的目标与原因。
- `manual_precursor_thermochemistry_template.csv`：后续加入前驱体分解热化学的模板。

## 五、Li 氢化物压力测试：`support_tables/li_hydride_pressure_test/`

用于小规模分布外验证 PET-OAM-XL 对 Li2AgH6/Li2AuH6 压力相关反应能变化的表现。

- `reaction_benchmark_pooled.csv`：总体反应能误差。
- `pressure_shift_benchmark.csv`：压力位移误差，即 10 GPa 和 0 GPa 反应能差值的误差。
- `ag_reaction_error_landscape.csv`、`au_reaction_error_landscape.csv`：Li2AgH6/Li2AuH6 逐反应误差。

## 六、作图源数据：`figure_source_data/`

按论文主图编号组织：

- `01_figure1_workflow/`：工作流和代表案例源数据。
- `02_figure2_energy_engine/`：PET-OAM-XL 模型证据图源数据。
- `03_figure3_120_target_audit/`：120-target 网络与置信结果图源数据。
- `04_figure4_literature_validation/`：文献路线 exact/family recovery 图源数据。
- `05_figure5_ranking_risks/`：排序消融、前驱体类别诊断和风险队列图源数据。
- `06_optional_extended_data/`：可作为扩展数据或补充图的数据。

## 单位说明

- phase MAE：eV atom^-1。
- reaction MAE：通常为 eV per target formula unit，除非表中另有说明。
- sign consistency：PET 与 MP-form 或 DFT 对反应能正负号判断一致的比例。
- top-1 agreement：PET 排名第一的反应式与参考排名第一的反应式完全一致。
- top-5 overlap：PET top-5 与参考 top-5 中共有的反应式数量。
