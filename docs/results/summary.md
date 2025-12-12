# 总结：MOM_EQ 动量基准实验

该部分可直接嵌入论文“实证环节”，全面描述：数据处理 → 因子构建 → 回归设定 → 诊断检验 → 主要发现。

## 流程概览
1. **数据整理**：`code/B_data_prepare.py` 将 `data/raw/` 中的行情、财报、无风险利率 Excel 结构化为 `data/processed/*.parquet`。
2. **五因子构建**：`code/factor_pipeline.py` 基于周频交易数据与滞后 30 天的账面/盈利/投资指标，生成 `weekly_panel.parquet`、`weekly_fundamentals.parquet` 以及五因子序列 `factor_returns_f5.parquet`（市值、规模、价值、盈利、投资因子均采用 5 分位 High-Low 构造）。
3. **动量扩展**：`code/momentum_comparison.py` 计算 4/8/16/26/52/78 周（均跳过近 1~4 周）的 12-2 动量，并对六个动量因子做等权平均得到 `MOM_EQ`；随后对每个动量版本＋等权组合、以及五因子 benchmark 运行 HAC(4) 回归，结果输出至 `docs/mom_w*_s*.csv/.json`、`docs/MOM_EQ_coeffs.csv`、`docs/five_factor_benchmark_coeffs.csv` 等。
4. **诊断检验**：`code/factor_diagnostics.py` 生成因子相关矩阵和 ADF 平稳性检验（`docs/factor_correlations.csv`、`docs/factor_stationarity.csv`），并在 `docs/results/` 下绘制对应图像。

## 数据覆盖与缺失情况
- **行情数据**（2019-07-26 ~ 2025-12-08，13 万行）：仅 8 条记录出现 OHLCV 缺失，缺失率 <0.005%，不会影响截面分位；保存在 `data/processed/market_prices.parquet`。
- **规模因子**（2019-08-02 ~ 2025-12-12，12.9 万行）：`Wretwd` 在 2019 年初期少量缺失（590 行，0.45%），构建周度收益前会直接剔除。
- **账面/盈利因子**（2017-12-31 ~ 2025-09-30）：当前版本无缺失列，滞后 30 天后才与周度面板 merge，确保不泄漏未来信息。
- **投资因子**（2018-12-31 ~ 2025-09-30）：`F080601A/F080602A/F080603A` 缺失 3 列共 12,211 个单元格（7.36%），主要集中在 `Source=1`（母公司/单体口径）记录；若研究只需合并报表，可在构建前过滤 `Source=0` 以提升数据完整性。
- **无风险利率**（2022-09-15 ~ 2025-05-20）：早于首条观测的周度会在 `_build_weekly_rf_series()` 中通过 `merge_asof(..., direction="backward") + bfill()` 自动填入最早可得的 3 个月利率（目前为 1.25% 年化 ≈ 0.00024 周度），保证 2019 年起的回归都有 `RF`；若后续获得 2019-2021 的真实利率，只需补写 `risk_free_rate.parquet` 即可替换该常数。
- 以上统计来自 `data/reports/missing_value_analysis_report.md` 与 `data/reports/missing_value_analysis_report.txt`，建议在更新原始数据后先运行 `python code/C_data_analysis.py` 刷新缺失概览。

## 样本与回归设定
- **时间**：2019-08-09 至 2025-12-05，共 299 个周度观测（剔除动量窗口不足的个别起始周后）。
- **被解释变量**：所有可交易股票的等权周度收益减无风险利率 `RF`，反映“市场平均策略”的表现。
- **解释变量**：五因子（MKT-RF、SMB、HML、RMW、CMA）+ 可选动量因子；所有系数使用 Newey-West/HAC(maxlags=4) 估计，避免周度数据的序列相关与异方差偏误。

## 主要结果
| 模型 | R² | α (周) | 动量系数 | 备注 |
| --- | --- | --- | --- | --- |
| 五因子 benchmark | 0.956 | -6.4e-04 (t≈-1.23) | — | 市场 β≈1.06、SMB≈0.13、HML≈0.096，盈利/投资略弱（t≈1.5/-0.5）。 |
| mom_w16_s2 | 0.960 | -9.2e-04 | -0.080 (t≈-4.59) | 16 周动量显著负值，代表短期领涨股在等权组合中发生回撤。 |
| mom_w26_s4 | 0.959 | -9.6e-04 | -0.065 (t≈-3.49) | 经典 12-2 窗口结果一致。 |
| MOM_EQ | **0.961** | **-1.36e-03** | **-0.126 (t≈-4.07)** | 六窗口等权平均，稳定性最佳，α 进一步下降。 |

文字解读：与五因子 benchmark 相比，任何动量版本都能将 α 拉向零并提高拟合优度；其中 `MOM_EQ` 的 R² 提升约 0.005，α 降低 ~0.0007，意味着动量暴露解释了额外 0.5% 的周度方差、并吸收了残余收益。负系数表明，本样本中“动量高”的股票在随后一周往往产生负收益，与科创板早期多为高估/短炒的特征吻合。

## 诊断与稳健性
- **相关性**：`docs/factor_correlations.csv` 与 `docs/results/correlation/correlation_heatmap.png` 显示，`SMB/HML` 与 `MOM_EQ` 的相关系数落在 -0.36~-0.38，其他因子互相关绝对值均 <0.4，说明动量不会与五因子产生严重共线性。
- **平稳性**：`docs/factor_stationarity.csv` 与 `docs/results/stationarity/adf_stats.png` 表明，所有因子的 ADF 统计量远低于 1% 临界值（p < 0.01），可视为平稳；无需差分或额外变换即可用于线性回归。
- **附加窗口**：`docs/momentum_comparison.md` 中同表列出了 4/8/52/78 周动量的结果，供论文附录或稳健性分析引用。

## 目录指引
- `docs/momentum_comparison.md`：汇总 benchmark + 各动量模型的 R²/α/MOM 系数。
- `docs/MOM_EQ_readme.md`：解读 `MOM_EQ` 与五因子 benchmark 的差异。
- `docs/factor_details.md`：详述因子构建方法，列出相关性与 ADF 表格。
- `docs/results/correlation/`、`docs/results/stationarity/`：分别展示热力图与平稳性图；`docs/results/README.md` 为目录索引。

## 如何复现
```bash
# 1. 准备 raw 数据后
python code/B_data_prepare.py
# 2. 构建五因子 + 周频面板
python code/factor_pipeline.py
# 3. 生成动量因子与回归结果
python code/momentum_comparison.py
# 4. 更新诊断图表（可选）
python code/factor_diagnostics.py
```
执行完毕后，`docs/results/` 即包含本文所述的全部总结、图像与表格。
