# 基于 MP 扩展相空间的 PET 合成路径工作流基准测试

## 目的

这个基准测试用于验证重建后的合成路径筛选工作流是否能够从任意无机化合物的化学式出发，自动完成以下闭环：

- 从 Materials Project 拉取目标体系及其所有子化学体系中的相关相。
- 导出或缓存每个相的 CIF 结构文件。
- 调用 PET-OAM-XI 对结构进行批量能量预测。
- 枚举能够生成目标化合物的配平方程，以及目标化合物可能分解到竞争相的反应。
- 按目标化合物的化学式单位归一化反应能，避免大系数反应在排序中被错误放大。
- 将 PET 得到的反应能与 MP 兼容的形成能基准进行对比，评估模型用于路径排序的可行性。

## 数据集

- 目标化合物数量：120 个。
- 化学类型覆盖：碱金属卤化物、碱土/过渡金属氧化物、过渡金属氧化物、氮化物、碳化物、硼化物、硫化物、碲化物、钙钛矿、尖晶石、层状氧化物、电池材料、碳酸盐、硫酸盐和硅酸盐。
- 相数据来源：Materials Project，通过当前版本的 `mp-api` 客户端访问。
- 相空间检索方式：不仅检索完整目标化学体系，也检索全部子化学体系。例如目标为 `KI` 时，同时检索 `K`、`I` 和 `K-I` 体系，从而保证单质和二元竞争相都进入反应网络。
- 能量模型：PET-OAM-XI，输入结构为从 MP 获得的 CIF。

## 汇总结果

- 成功完成目标：120/120。
- 拉取并参与筛选的相：1619 个。
- 完成 PET 能量计算的相：1596 个。
- 枚举形成反应：18063 条。
- 可同时获得 PET 和 `MP_FORM` 基准的形成反应：17085 条。
- 经过低复杂度和低 `Ehull` 过滤后的 practical route 候选：4235 条。
- PET 相对于 `MP_FORM` 的平均反应能 MAE：0.584 eV/目标化学式单位。
- 各目标 MAE 的中位数：0.434 eV/目标化学式单位。
- PET 相对于 `MP_FORM` 的反应能正负号一致率：0.931。
- PET 与 `MP_FORM` 的 top-1 路径一致率：0.825。
- 推荐层分类结果：104 个高置信 promising synthesis candidate，15 个中等置信 plausible but needs DFT/experimental filtering，1 个 low-confidence case。
- 推荐层进一步生成 26 个目标的 `dft_refinement_queue.csv`，用于标记需要优先做 DFT 精修或人工核查的路径。
- top-1 推荐路线的前驱体复杂度：65 个目标使用单质前驱体，50 个目标使用简单化合物前驱体，仅 5 个目标需要复杂前驱体路线。

## 化学类别配比

这次扩展 benchmark 采用 120 个目标，是为了达到更接近子刊级计算论文的覆盖度。目标集合不是随机堆叠，而是按材料类别分层选取：

- 卤化物：20 个目标，81 条可比反应，PET-vs-`MP_FORM` 平均 MAE 为 0.426 eV/目标。
- 二元氧化物：25 个目标，1807 条可比反应，平均 MAE 为 0.538 eV/目标。
- 氮化物/碳化物/硫族化物：25 个目标，433 条可比反应，平均 MAE 为 0.184 eV/目标。
- 三元氧化物：25 个目标，7476 条可比反应，平均 MAE 为 0.871 eV/目标。
- 电池材料/碳酸盐/硫酸盐/硅酸盐：25 个目标，7288 条可比反应，平均 MAE 为 0.870 eV/目标。

这个配比的目的有两个：一是保留简单二元体系用于检查元素参考和基本形成反应；二是加入大量三元/四元复杂体系，用来验证相空间扩展、反应枚举和 PET 批量计算在真实复杂反应网络中的稳定性。

## MP 反应能基准

当前输出只保留一条 MP 反应能路线：

- `MP_FORM`：使用 MP 的 `formation_energy_per_atom` 乘以化学式原子数得到每化学式单位形成能，再对配平反应做产物减反应物。这是当前主参考。对于任意配平反应，形成能中的元素参考项理论上会相互抵消，因此它更适合跨元素体系比较，也能避免单质参考不一致导致的异常。

旧版“直接使用 MP 每个相总能量相加减”的路线已经从代码、结果表和图中删除。原因是它与当前 `MP_FORM` 不是同一个参考口径，直接混用会造成解释错误，尤其在含卤素体系中可能出现明显异常。

## 结果解释

这组结果说明当前工作流已经能够支撑从“用户输入一个化学式或 CIF”到“自动生成候选合成路径并给出 PET 反应能排序”的完整工程闭环。对于 120 个非氢化物为主、化学类型较分散的目标体系，流程都能正常完成 MP 拉取、CIF 缓存、PET 批量计算、反应枚举、`Ehull` 诊断和路径排序。

PET 的路径排序与 `MP_FORM` 基准整体一致性较好。尤其是在把反应能归一化到目标化学式单位后，平均误差、反应能符号一致率和 top-1 路径一致率表明 PET 可以作为快速筛选模型，用于在大量候选反应中识别热力学上更有利的路径。

需要注意的是，“热力学最强放热路径”不一定等同于实验上最现实的合成路线。因为枚举器会尽量使用 MP 数据库中的所有可用相，某些路径可能包含化学计量很极端或实验可获得性较弱的中间相。为避免把这类路径误认为最佳实验方案，当前报告同时给出两类结果：

- 纯热力学最佳路径：只按反应能排序，用于判断反应网络中的最大热力学驱动力。
- practical route：进一步限制前驱体复杂度、化学计量极端程度、反应系数规模和相稳定性，更接近可解释、可讨论的合成候选路径。当前阈值为：非目标相的最简式原子数不超过 10，化学计量最大/最小比不超过 5，非目标相 `Ehull` 不超过 0.10 eV/atom，反应总系数除以目标系数不超过 30。

top-1 路径一致率的定义是：对同一个目标化合物，先在所有同时有 PET 和 `MP_FORM` 反应能的可比形成反应中分别按 PET 反应能和 `MP_FORM` 反应能从低到高排序；如果 PET 排名第一的反应式与 `MP_FORM` 排名第一的反应式完全相同，则该目标记为 top-1 一致。120 个目标中 top-1 一致率为 0.825，即 99/120 个目标的 PET 最优路径与 MP 形成能基准最优路径相同。

新增推荐层把这些统计转化为用户可直接读取的合成建议。每个目标目录现在额外输出 `recommended_routes.csv`、`synthesis_recommendation_summary.csv` 和 `synthesis_recommendation.md`。推荐层默认用 PET 排序；如果用户没有运行 PET，则自动降级为 `MP_FORM` 参考排序，并打上 `pet_energy_unavailable_using_mp_form_reference` 风险标记。置信度由四类证据组成：目标 Ehull 是否低于 0.10 eV/atom、是否存在放热 practical route、PET 与 `MP_FORM` 的 top 路径是否一致或 top-5 是否重叠、是否存在强放热分解竞争。

`aggregate_recommendations.py` 用于从每个目标目录重新汇总推荐结果，避免推荐层只是一次性手工结果。该脚本输出总表、每个目标前 5 条推荐路线、类别级推荐统计，以及 `dft_refinement_queue.csv`。精修队列的逻辑是：中低置信目标、存在强放热分解竞争、没有放热 practical route、或 PET 与 `MP_FORM` top-1 路径不一致的目标，需要优先进入 DFT 复算、相空间补全或人工化学检查。目前 120 个目标中有 26 个进入该队列。

推荐路线还增加了前驱体复杂度标注。每条推荐路线会记录反应物列表、副产物列表、反应物最大原子数、反应物最大元素数、是否全为单质反应物、最大前驱体 Ehull，以及 `elemental_precursors`、`simple_compound_precursors` 或 `complex_precursors` 类别。这一层可以把“能量上最优”与“化学上可读”区分开，避免只根据数据库极端相做合成解释。

因此，这个基准更适合作为“工作流可行性和模型排序一致性”的验证，而不是最终实验路线的唯一判据。后续若要形成更强的子刊级论文证据，还需要进一步加入实验前驱体库、温度/气氛/压力约束、动力学可达性和文献合成路线对照。

## 关键输出文件

- `results/mp_pet_benchmark_120/mp_pet_benchmark_summary.csv`：每个目标化合物的相数、反应数、PET-vs-`MP_FORM` 误差、top-1 是否一致、最佳路径和 practical route。
- `results/mp_pet_benchmark_120/mp_pet_benchmark_summary_with_categories.csv`：在汇总表基础上增加材料类别标签。
- `results/mp_pet_benchmark_120/mp_pet_benchmark_category_summary.csv`：按材料类别聚合的覆盖度、误差和排序一致性统计。
- `results/mp_pet_benchmark_120/mp_pet_reaction_data.csv`：所有可比反应的逐条 PET 和 `MP_FORM` 反应能数据，可用于画 parity plot 或做误差分布分析。
- `results/mp_pet_benchmark_120/synthesis_recommendation_summary_all.csv`：120 个目标的推荐层总表，包含分类、置信度、风险标记、最佳 PET/`MP_FORM` 路线和分解竞争。
- `results/mp_pet_benchmark_120/recommended_routes_top5_all.csv`：每个目标前 5 条推荐路线的汇总表。
- `results/mp_pet_benchmark_120/recommendation_category_summary.csv`：按材料类别统计高/中/低置信推荐比例、top-1 一致率和低 Ehull 覆盖率。
- `results/mp_pet_benchmark_120/top1_route_feature_summary.csv`：按前驱体复杂度统计 top-1 推荐路线。
- `results/mp_pet_benchmark_120/dft_refinement_queue.csv`：建议优先进行 DFT 精修或人工核查的目标及原因。
- `results/mp_pet_benchmark_120/RECOMMENDATION_LAYER_ANALYSIS.md`：推荐层分类统计的中文论文支撑说明。
- `results/mp_pet_benchmark_120/mp_pet_benchmark_summary.png`：120 目标基准测试汇总图。
- `results/mp_pet_benchmark_120/mp_pet_benchmark_summary.pdf`：同一汇总图的 PDF 版本，便于论文插图使用。
- `results/mp_pet_benchmark_120/mp_pet_benchmark_120_category_summary.png`：按材料类别聚合的论文友好型汇总图。
- `results/mp_pet_benchmark_120/mp_pet_benchmark_120_category_summary.pdf`：同一类别汇总图的 PDF 版本。
- `results/mp_pet_benchmark_120/README_summary.md`：基准测试的简短汇总结果。
