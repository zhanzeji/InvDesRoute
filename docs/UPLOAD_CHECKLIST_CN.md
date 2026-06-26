# InvDesRoute 上传清单（中文）

本文件说明该 release 包如何上传到 Zenodo、Figshare、OSF 或课题组网站，并对应论文中的 Data availability 与 Code availability。

## 1. 建议上传内容

建议整体上传整个 `InvDesRoute_release/` 文件夹，或上传压缩包 `InvDesRoute_release_20260626.tar.gz`。

上传后应保留以下结构：

- `data/`：论文数值证据、benchmark 表、文献路线验证表、推荐排序表、风险诊断表和作图源数据。
- `code/`：InvDesRoute 工作流代码、环境文件、示例命令和脚本。
- `manuscript/`：正文和补充信息快照，仅用于说明数据与论文版本对应关系。
- `docs/`：数据/代码可用性文本、release notes、上传清单和来源说明。
- `MANIFEST.txt`：文件列表。
- `SHA256SUMS.txt`：文件校验和。

## 2. 上传前必须确认

- 不上传 Materials Project API key。
- 不上传 PET-OAM-XL、MatterSim、SevenNet、MACE 或 CHGNet 模型 checkpoint，除非原许可证允许且你明确希望分发。
- 不把 Materials Project 原始数据库或完整 CIF 缓存作为自己的数据重新分发；本包只包含派生表和结果表。
- 文献路线数据来自公开文本挖掘数据集；本包提供处理后的验证表和审计结果。

## 3. 上传后需要替换的占位符

上传完成后，在论文和 `docs/DATA_AND_CODE_AVAILABILITY_TEXT.md` 中替换：

- `ZENODO_DOI_OR_URL`：替换为数据归档 DOI 或网页链接。
- `GITHUB_OR_ARCHIVE_URL`：替换为代码仓库链接或同一个归档链接。

如果你把 data 和 code 放在同一个 Zenodo 记录中，可以两个位置都写同一个 DOI；如果代码另放 GitHub，则 Data availability 写 Zenodo，Code availability 写 GitHub + Zenodo snapshot。

## 4. 推荐投稿写法

最稳妥做法：

- Data availability：说明 MP 和 Kononova 数据是公开外部来源；本文生成的处理后表格和作图源数据放在 Zenodo。
- Code availability：说明 InvDesRoute 代码放在 GitHub 或 Zenodo；MP API key 和模型 checkpoint 需用户自行获取。

不要写“所有原始结构和模型都已上传”，因为 MP 原始数据和模型 checkpoint 不应默认重新分发。

## 5. 核验命令

在上传前可以运行：

```bash
cd InvDesRoute_release
sha256sum -c SHA256SUMS.txt
```

若输出全部 `OK`，说明文件未被修改。
