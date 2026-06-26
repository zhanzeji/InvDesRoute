# 文献合成路线对比分析

## 结论摘要

- 文献种子路线数：40
- 出现在 120-target 推荐表中的目标数：39，覆盖率 0.975
- Top-5 中完全前驱体匹配率：0.590
- Top-5 中覆盖全部文献前驱体的比例：0.590
- Top-5 中至少部分前驱体匹配的比例：0.590
- Top-5 中元素池完全一致的比例：0.795
- 匹配到的推荐路线为 PET 放热的比例：0.923
- 平均路线匹配分数：0.747

## 匹配类型分布

- exact_precursor_match: 23
- same_element_pool: 8
- partial_element_match: 8
- not_in_recommendation_table: 1

## 分类统计

- Battery/salts/silicates: n=4, exact=0.000, partial+=0.000, same-element=0.250
- Binary oxides: n=11, exact=0.909, partial+=0.909, same-element=1.000
- Halides: n=12, exact=0.500, partial+=0.500, same-element=1.000
- Nitrides/carbides/chalcogenides: n=9, exact=0.778, partial+=0.778, same-element=0.778
- Ternary oxides: n=3, exact=0.000, partial+=0.000, same-element=0.000
- unmatched: n=1, exact=0.000, partial+=0.000, same-element=0.000

## 文献前驱体范围诊断

- literature_precursors_require_external_reservoir_elements: n=8, exact=0.000, same-element=0.000, mean-score=0.283
- target_element_only_literature_precursors: n=32, exact=0.742, same-element=1.000, mean-score=0.867

## 高匹配例子

- LiF: 文献 `Li + 1/2 F2 -> LiF`；推荐 rank 1 `F2 + 2 Li -> 2 LiF`；类型 exact_precursor_match；score=1.000
- NaF: 文献 `Na + 1/2 F2 -> NaF`；推荐 rank 1 `F2 + 2 Na -> 2 NaF`；类型 exact_precursor_match；score=1.000
- KF: 文献 `K + 1/2 F2 -> KF`；推荐 rank 1 `F2 + 2 K -> 2 KF`；类型 exact_precursor_match；score=1.000
- LiCl: 文献 `Li + 1/2 Cl2 -> LiCl`；推荐 rank 1 `Cl2 + 2 Li -> 2 LiCl`；类型 exact_precursor_match；score=1.000
- NaCl: 文献 `Na + 1/2 Cl2 -> NaCl`；推荐 rank 1 `Cl2 + 2 Na -> 2 NaCl`；类型 exact_precursor_match；score=1.000
- KCl: 文献 `K + 1/2 Cl2 -> KCl`；推荐 rank 1 `Cl2 + 2 K -> 2 KCl`；类型 exact_precursor_match；score=1.000
- MgO: 文献 `Mg + 1/2 O2 -> MgO`；推荐 rank 1 `2 Mg + O2 -> 2 MgO`；类型 exact_precursor_match；score=1.000
- CaO: 文献 `Ca + 1/2 O2 -> CaO`；推荐 rank 1 `2 Ca + O2 -> 2 CaO`；类型 exact_precursor_match；score=1.000

## 低匹配或需要扩展前驱体库的例子

- FeS: 文献前驱体 `Fe;S`；推荐前驱体 ``；类型 not_in_recommendation_table；score=0.000
- LiFePO4: 文献前驱体 `FeC2O4;Li2CO3;NH4H2PO4`；推荐前驱体 `LiFeP;O2`；类型 partial_element_match；score=0.136
- BaTiO3: 文献前驱体 `BaCO3;TiO2`；推荐前驱体 `BaO2;Ti2O`；类型 partial_element_match；score=0.162
- SrTiO3: 文献前驱体 `SrCO3;TiO2`；推荐前驱体 `SrO2;Ti3O`；类型 partial_element_match；score=0.162
- CaTiO3: 文献前驱体 `CaCO3;TiO2`；推荐前驱体 `CaO2;Ti`；类型 partial_element_match；score=0.162
- Fe2O3: 文献前驱体 `Fe;O2`；推荐前驱体 `Fe4O5`；类型 same_element_pool；score=0.200
- LiCoO2: 文献前驱体 `Co3O4;Li2CO3;O2`；推荐前驱体 `Co3O4;Li`；类型 partial_element_match；score=0.337
- GaN: 文献前驱体 `Ga;NH3`；推荐前驱体 `Ga;N2`；类型 partial_element_match；score=0.433

## 解释

该分析不要求推荐路线与文献路线完全相同，因为一个目标材料可以有多种合成路线。
完全前驱体匹配说明 top-5 推荐中直接出现了文献路线前驱体；部分匹配说明推荐路线与实验路线共享关键前驱体；元素池一致说明反应化学空间正确，但具体前驱体选择仍需实验知识、前驱体库或气体化学势修正。
如果文献前驱体含有目标元素之外的 C、H、N 等 reservoir 元素，例如碳酸盐、铵盐、草酸盐或氨气，那么当前只按目标元素展开 MP 子体系的 workflow 天然无法枚举这些路线。这类低匹配不应简单解释为 PET 能量模型失败，而应解释为前驱体库和开放组分 reservoir 尚未纳入。

当前 seed 数据集用于验证比较流程。正式写入论文前，应继续扩充 DOI 级别文献路线或接入公开 text-mined synthesis 数据集。
