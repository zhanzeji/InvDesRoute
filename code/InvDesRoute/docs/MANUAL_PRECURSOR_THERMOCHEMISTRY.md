# 手工前驱体热化学表说明

## 目的

MP 相池不能完整覆盖实验常见前驱体，尤其是草酸盐、含水盐、铵盐、硝酸盐、乙酸盐和分子/气体 reservoir。`manual_precursor_table.csv` 用于把经过人工审核的前驱体形成能加入 workflow，避免虚拟前驱体只能参与配平而不能参与能量排序。

## 表格格式

模板位于：`data/manual_precursor_thermochemistry_template.csv`。

必需列：

- `formula`：前驱体化学式，会被 pymatgen 归一化。
- `formation_energy_per_atom`：每原子形成能，单位 eV/atom。推荐优先填这一列。
- `dft_energy_per_atom`：可选，总能/兼容 DFT 能量，单位 eV/atom。
- `energy_above_hull`：可选，若没有可留空。
- `source_id`、`source`、`note`：数据来源和人工审核说明。

## 使用方式

```bash
<InvDesRoute_repo>/run_screening.py \
  --target-formula LiFePO4 \
  --database mp \
  --mp-api-key "$MP_API_KEY" \
  --include-common-precursors \
  --include-gas-reservoirs \
  --include-virtual-common-precursors \
  --manual-precursor-table /path/to/manual_precursor_table.csv \
  --include-targeted-precursor-routes
```

## 当前原则

- 不伪造 FeC2O4 等未知/未审核前驱体能量。
- 没有能量的前驱体仍可参与配平和可合成性推荐，但会被标记为 `virtual_precursor_energy_missing` 或 `energy_unavailable`。
- 后续温度、压力、气氛校正应在路线族锁定之后再做，避免过早引入大量不确定参数。
