# `synthesis_pathway_screening` 项目结构说明

## 总体目标

这个工程用于实现以下工作流：用户输入化学式或 CIF 文件后，程序自动从本地数据和 Materials Project 中拉取相关相结构，生成候选合成/分解反应，调用 PET-OAM-XI 计算结构能量，给出反应能、`Ehull` 稳定性诊断和推荐合成路径。

## 顶层文件

- `README.md`：工程入口说明，包括运行命令、输出文件、benchmark 结果和 PET/UPET 调用方式。
- `run_screening.py`：单个目标化合物的主运行脚本。它负责串联相检索、PET 能量计算、反应枚举、稳定性分析和报告输出。
- `benchmark_mp_pet.py`：批量验证脚本。内置 `core25` 和 `subjournal120` 两个目标集合；当前主结果使用 120 个多类别无机化合物的 MP+PET workflow benchmark。
- `aggregate_recommendations.py`：推荐层聚合脚本。它从每个目标目录收集 `synthesis_recommendation_summary.csv` 和 `recommended_routes.csv`，重建 120 目标总表、类别统计和 DFT 精修队列。

## `src/` 源码模块

- `src/common.py`：全局路径、化学式标准化、元素体系生成、JSON 输出和基础数据结构。这里定义了 PET checkpoint 路径和 UPET Python 环境路径。
- `src/phase_sources.py`：相数据来源模块。负责读取本地相库、通过 `mp-api` 拉取 MP 相、扩展目标的全部子化学体系、缓存 MP CIF，并合并最终相池。
- `src/pet_energy.py`：PET-OAM-XI 能量计算模块。它会调用 `$UPET_PYTHON`，加载 `UPETCalculator` 和 `pet-oam-xl-v1.0.0.ckpt`，对 CIF 批量计算能量。
- `src/reactions.py`：反应枚举和配平模块。负责生成形成反应、分解反应，计算反应能，并把反应能归一化到目标化合物的化学式单位。
- `src/analysis.py`：分析和报告模块。负责 MP-compatible DFT 能量映射、PET/DFT 局部凸包稳定性诊断、practical route 标注和 `report.md` 生成。

## `cache/` 缓存目录

- `cache/mp/<chemsys>/mp_phases.csv`：某个化学体系从 MP 拉取到的相表，例如 `cache/mp/C-Si/mp_phases.csv` 是 C-Si 体系。
- `cache/mp/<chemsys>/cifs/`：对应体系中每个相的 CIF 文件。PET 计算直接读取这些 CIF。
- 缓存的作用是避免重复访问 MP，也方便复现每次 benchmark 使用的结构集合。

## `results/` 结果目录

- `results/mp_pet_smoke/`：小规模烟测结果，例如 KI 和 MgO，用来确认 MP 拉取、PET 调用和反应能计算是否能闭环。
- `results/mp_pet_benchmark/`：早期 25 个目标体系 benchmark 结果。
- `results/mp_pet_benchmark_120/`：当前主 benchmark 结果，覆盖 120 个目标体系，更适合作为论文支撑。
- 每个单目标子目录通常包含：
  - `phase_pool_raw.csv`：原始相池。
  - `phase_stability.csv`：相能量、PET 能量、MP `Ehull` 和局部凸包诊断。
  - `pet_energy_cache.csv`：PET 对每个 CIF 的能量预测结果。
  - `formation_reactions.csv`：形成目标化合物的候选反应。
  - `decomposition_reactions.csv`：目标化合物分解到竞争相的候选反应。
  - `report.md`：该目标的路径推荐和可合成性解释。
  - `recommended_routes.csv`：推荐层输出的路线排序表，包含排序能量来源、PET/`MP_FORM` 反应能、风险标记和置信度。
  - `synthesis_recommendation_summary.csv`：单目标推荐摘要，包含分类、置信度、最佳 practical route、top-1 一致性和分解竞争。
  - `synthesis_recommendation.md`：面向用户的自然语言合成建议报告。
  - `run_metadata.json`：运行参数和统计信息。
- `results/mp_pet_benchmark_120/mp_pet_benchmark_summary.csv`：120 个目标的汇总表，包含 PET-vs-`MP_FORM` 误差、路径排序一致性、最佳路径和 practical route。
- `results/mp_pet_benchmark_120/mp_pet_reaction_data.csv`：所有可比形成反应的逐条 PET/`MP_FORM` 反应能数据。
- `results/mp_pet_benchmark_120/synthesis_recommendation_summary_all.csv`：120 个目标的推荐层汇总表。
- `results/mp_pet_benchmark_120/recommended_routes_top5_all.csv`：每个目标前 5 条推荐路线汇总。
- `results/mp_pet_benchmark_120/recommendation_category_summary.csv`：按材料类别统计推荐置信度、top-1 一致率和低 Ehull 覆盖率。
- `results/mp_pet_benchmark_120/top1_route_feature_summary.csv`：按前驱体复杂度统计 top-1 推荐路线，区分单质前驱体、简单化合物前驱体和复杂前驱体路线。
- `results/mp_pet_benchmark_120/dft_refinement_queue.csv`：建议优先进行 DFT 精修或人工核查的目标队列。
- `results/mp_pet_benchmark_120/RECOMMENDATION_LAYER_ANALYSIS.md`：推荐层分类统计和置信度说明。
- `results/mp_pet_benchmark_120/EXTENDED_PAPER_ANALYSIS.md`：论文扩展分析统计，包含 top-5 重叠、误差来源、前驱体复杂度和 DFT 精修队列类别分布。
- `results/mp_pet_benchmark_120/mp_pet_benchmark_summary.png` 和 `.pdf`：120 目标 benchmark 汇总图。

## `docs/` 文档目录

- `docs/BENCHMARK_ANALYSIS.md`：中文 benchmark 分析，说明数据集、汇总结果、解释和局限性。
- `docs/PROJECT_STRUCTURE.md`：当前文件，说明工程每个部分的用途。

## `examples/` 示例目录

- `examples/KI_mp.sh`：KI 这类普通无机化合物的 MP 检索示例。
- `examples/NaAlH4_local.sh`：本地氢化物相库示例。

## `__pycache__/` 缓存目录

这是 Python 自动生成的字节码缓存，对研究结果没有直接意义，可以忽略。
