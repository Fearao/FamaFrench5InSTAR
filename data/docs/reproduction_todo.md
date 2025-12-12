# 复现 Fama-French 五因子实证的任务清单

> 依据 `data/docs/readme.md` 提炼的第 4、5 章实证结构，梳理完成仓库代码/文档所需的工作项。整体目标是在本地数据基础上复现原论文的所有回归、描述统计与稳健性检验。

## 0. 环境与数据底座
- [ ] **依赖管理**：统一 Python 版本（建议 3.10+），安装 `pandas`, `numpy`, `pyarrow`, `statsmodels`, `linearmodels`, `scipy`, `matplotlib`, `seaborn` 等；准备 `poetry`/`pip-tools` 或 `requirements.txt`。
- [ ] **原始数据检查**：核对 `data/raw/` 是否包含完整的月度财务、日度行情、无风险利率、行业/板块映射；补齐板块标签（主板/中小板/创业板）。
- [ ] **ETL 管线**：在 `code/A_data_prepare.py` 基础上，扩展生成：
  - `monthly_returns.parquet`、`monthly_size.parquet`、`financials.parquet` 等中间表。
  - 板块映射表（证券代码→市场板块），供 4.2 与 5.2 使用。

## 1. 第四章：A 股市场 Fama-French 五因子实证
### 1.1 投资组合描述性统计（4.1）
- [ ] **25 组合构造**：实现 Size-B/M、Size-OP、Size-Inv 三套 5×5 交叉组，频率为月度超额收益；输出 `portfolio_returns_size_bm.parquet` 等。
- [ ] **描述性统计**：计算均值、标准差、t 值、夏普等，生成 `docs/tables/desc_stats_portfolios.csv` 并配套 markdown。

### 1.2 因子特征与相关性
- [ ] **五因子序列**：构造 `MKT-RF`, `SMB`, `HML`, `RMW`, `CMA` 月度序列，确认对齐无风险利率。
- [ ] **相关性/多重共线性检验**：输出因子矩阵相关系数、VIF、互相回归的 R²；保存 `docs/tables/factor_corr.csv`、`docs/figs/factor_corr_heatmap.png`。

### 1.3 交叉分组回归
- [ ] **时间序列回归**：对 25 个组合分别回归五因子模型，记录 α、因子载荷、t 值、调整 R²。
- [ ] **结果存档**：生成 `docs/tables/ff5_cross_section.csv` 与 `docs/memo/ff5_cross_section.md` 图表。

### 1.4 分板块实证（4.2）
- [ ] **板块投资组合**：在主板、创业板、原中小板三大板块内重复 25 组合及描述性统计。
- [ ] **板块因子属性**：分别计算每个板块的因子统计、相关系数及互回归。
- [ ] **板块回归与 GRS**：对各板块组合运行五因子回归，实施 GRS 检验，保存 `docs/tables/grs_board.csv`。

### 1.5 小结（4.3）
- [ ] **文本总结**：撰写 `docs/chapter4_summary.md`，概述主要发现/显著性。

## 2. 第五章：动量修正五因子模型
### 2.1 构造动量修正模型（5.1）
- [ ] **冗余因子检验**：在代码层面支持排除 CMA（投资因子）；实现 `MOM` 动量因子（12-2 或 12-1 方案，与论文一致）。
- [ ] **市值-动量组合**：生成 Size-MOM 5×5 组合及描述性统计。
- [ ] **扩展相关性**：将 MOM 纳入因子矩阵，输出新的相关性、互回归、协方差。
- [ ] **交叉回归**：完成 Size-B/M、Size-OP、Size-MOM 的动量修正回归，并与传统五因子对比。

### 2.2 各板块动量回归（5.2）
- [ ] **板块动量组合**：为主板/中小板/创业板构造 Size-MOM 组合。
- [ ] **回归/显著性**：运行动量修正模型回归，输出表格与图。

### 2.3 稳健性（5.3）
- [ ] **GRS 对比**：在同一脚本中实现 FF5 vs. 动量修正版的 GRS 统计并比较。
- [ ] **中国版三因子回归**：补充 `MKT-RF`, `SMB`, `HML` 的三因子结果，生成对应回归和统计。
- [ ] **额外检验**：视论文需求增加 Newey-West 调整、子样本检验。

### 2.4 小结（5.4）
- [ ] **撰写总结**：输出 `docs/chapter5_summary.md`，描述动量修正模型改进之处。

## 3. 工具化与复现实用性
- [ ] **脚本结构**：补齐 `code/` 下的 `factor_pipeline.py`, `portfolio_analysis.py`, `regression_runner.py`, `grs_test.py` 等模块，CLI 参数化（时间范围、板块、模型类型）。
- [ ] **可重复流程**：提供 `Makefile` 或 `invoke` 任务：`make prepare-data`, `make run-ff5`, `make run-mom`.
- [ ] **可视化**：生成核心图表（收益曲线、α 分布、GRS 柱状图）。
- [ ] **文档更新**：在 `README.md` 与 `data/docs/` 内补充运行步骤、输入输出示例、注意事项。

## 4. 验证与交付
- [ ] **结果复核**：对照论文表格数值（均值、α、t 值）确认误差范围；差异超出容忍度时记录原因。
- [ ] **版本管理**: 编写测试（如 `pytest`）校验数据加载、回归结果维度；提交前运行 `pre-commit`（lint/format/doc 检查）。

> 勾选完上述任务后，即可完整复现论文第 4、5 章的全部实证流程及扩展。
