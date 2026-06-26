# 文献合成路线对比基准

本文件夹用于检验 PET-OAM-XI 推荐的合成路线是否接近真实文献或教科书中常见的无机合成路线。

## 目标

前面 workflow 已经能为目标材料输出 top practical PET routes、反应能、Ehull、置信度和风险标记。这个扩展基准进一步回答：

- PET 推荐的前驱体是否与真实合成路线使用的前驱体相同或相近？
- 推荐路线是否只是热力学上有利，还是也接近实验可读路线？
- 哪些体系需要引入实验前驱体库、气体化学势或温度/气氛条件？

## 数据文件

- `literature_routes_seed.csv`：人工整理的第一批真实/常见合成路线种子集。
- `raw/`：预留给公开 text-mined synthesis 数据集或手动下载的数据。
- `results/`：脚本生成的路线匹配分析、图和中文报告。

## 当前基准边界

当前 seed 数据集主要用于建立可复现对比流程，不应被表述为完整文献统计。后续可以接入 Ceder 组公开 text-mined solid-state synthesis dataset，或继续人工扩充 DOI 级别的真实论文路线。

## 当前结果

运行：

```bash
python3 literature_benchmark/compare_literature_routes.py
python3 literature_benchmark/plot_literature_benchmark_svg.py
```

当前 40 条 seed 文献路线中，39 个目标出现在 120-target 推荐表中。Top-5 推荐的完全前驱体匹配率为 0.590，元素池完全一致率为 0.795，匹配路线中 PET 放热比例为 0.923。进一步分解后可以看到：若文献路线只使用目标元素内前驱体，完全前驱体匹配率为 0.742、元素池一致率为 1.000；若文献路线使用碳酸盐、铵盐、草酸盐、NH$_3$ 等含目标外元素的 reservoir 前驱体，当前 workflow 难以直接匹配。这说明下一步最重要的工程扩展不是简单增加目标数量，而是加入实验前驱体库和开放组分 reservoir。

## 主要输出

- `results/literature_route_match_top5.csv`：每条文献路线与 PET top-5 推荐的最佳匹配。
- `results/literature_route_match_summary.csv`：总体匹配统计。
- `results/literature_route_match_category_summary.csv`：按材料类别统计。
- `results/literature_route_match_precursor_scope_summary.csv`：按是否需要外部 reservoir 元素统计。
- `results/LITERATURE_ROUTE_BENCHMARK_ANALYSIS.md`：中文解释报告。
- `results/literature_route_benchmark_summary.svg`：无依赖 SVG 汇总图。

## 接入大规模公开 text-mined synthesis 数据

本目录已经预留 CederGroupHub `text-mined-synthesis_public` 数据集接入脚本。该公开数据集包含 solid-state synthesis 文献挖掘反应，可用于后续把 40 条 seed benchmark 扩展到更大规模。

下载 solid-state 2020 数据集：

```bash
cd <InvDesRoute_repo>/literature_benchmark
python3 download_ceder_text_mined_dataset.py \
  --method range \
  --output raw/solid-state_dataset_20200713.json.xz
```

如果 raw 文件流在服务器上卡住，可尝试 GitHub API blob 模式：

```bash
python3 download_ceder_text_mined_dataset.py \
  --method api \
  --output raw/solid-state_dataset_20200713.json.xz
```

解析成与本 benchmark 相同格式的路线表：

```bash
python3 extract_ceder_text_mined_routes.py \
  --input raw/solid-state_dataset_20200713.json.xz \
  --output ceder_text_mined_routes.csv
```

与我们的 PET top-5 推荐进行对比：

```bash
python3 compare_literature_routes.py \
  --literature ceder_text_mined_routes.csv \
  --outdir results_ceder_text_mined
```

注意：自动解析的大规模 text-mined 路线需要质量审核，尤其要检查 precursor 字段、target 字段、溶剂/助熔剂/气氛是否被误当作反应物。
