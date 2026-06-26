# 明早查收重点

## 已完成的新能力

1. **实验前驱体扩展**：common precursor、gas reservoir、virtual precursor、targeted precursor route enumeration 已接入主 workflow。
2. **可合成性推荐层**：新增 `synthesizability_ranked_routes.csv` 和 `synthesizability_recommendation.md`，不再只按最低能量排序，而是综合：
   - 静态反应能；
   - 实验路线模板；
   - 前驱体常见性；
   - 路线复杂度；
   - 相/能量可用性；
   - 气体、副产物、气氛和虚拟前驱体风险。
3. **文献路线验证**：9 个代表性真实/经典文献路线案例均可枚举；精确文献路线进入 top-1/top-5/top-20 分别为 6/9、7/9、8/9；路线族匹配全部找到，且大多数进入前几名。
4. **手工热化学入口**：新增 `data/manual_precursor_thermochemistry_template.csv` 和 `docs/MANUAL_PRECURSOR_THERMOCHEMISTRY.md`，后续可补 FeC2O4 等实验前驱体能量。
5. **聚合脚本**：新增 `aggregate_synthesizability.py`，可对任意 benchmark 目录聚合可合成性 top-N 路线和精修队列。

## 关键文件

- 主工程说明：`<InvDesRoute_repo>/README.md`
- 可合成性评分模块：`<InvDesRoute_repo>/src/synthesizability.py`
- 定向路线枚举：`<InvDesRoute_repo>/src/targeted_routes.py`
- 9 个代表案例结果：`<InvDesRoute_repo>/results/precursor_expanded_targeted_final/`
- 可合成性验证分析：`<InvDesRoute_repo>/literature_benchmark/results/SYNTHESIZABILITY_RECOMMENDATION_ANALYSIS.md`
- 路线族匹配结果：`<InvDesRoute_repo>/literature_benchmark/results/literature_route_family_match.csv`
- final paper 支撑材料：`<local_root>/final_paper/support_tables/literature_route_benchmark/`

## 当前还未做/刻意后置

- 温度、压力、气氛化学势校正后置。
- FeC2O4、含水盐、硝酸盐、草酸盐等前驱体能量没有伪造，等待可靠热化学数据或 DFT/实验数据补齐。
- 论文主体暂未改，等你确认这批工程结果后再写入正文。
