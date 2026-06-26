# 可复现性说明（中文）

## 1. 最短复查路径

如果审稿人只想核查论文数值是否有来源：

1. 打开 `data/figure_source_data/` 查看每张主图的源数据。
2. 打开 `data/support_tables/` 查看完整 benchmark、文献路线验证和风险诊断表。
3. 用 `MANIFEST.txt` 和 `SHA256SUMS.txt` 确认 release 文件未被修改。

## 2. 重新运行工作流需要什么

完整重跑需要：

- Materials Project API key；
- PET-OAM-XL/uPET 运行环境；
- PET-OAM-XL checkpoint；
- Python 材料计算依赖，如 pymatgen、ASE、pandas、numpy、scipy；
- 公开文本挖掘固相合成数据集，用于从头复现文献路线筛选。

## 3. 为什么不包含所有原始数据

- Materials Project 原始结构和数据库条目受其访问条款约束，应由用户自行获取。
- 机器学习势模型 checkpoint 可能受原模型发布协议约束，应从原始发布源获取。
- 本 release 重点提供本文生成的派生数据、审计表、推荐结果和作图源数据。

## 4. 代码状态

`code/InvDesRoute/` 是工作流代码快照，已经过 Python 语法编译检查。完整科学复现仍需用户配置 MP key、PET 模型路径和本地计算环境。
