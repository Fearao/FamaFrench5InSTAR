# ADF Stationarity Test

- **数据**：`docs/factor_stationarity.csv`，同样基于 2019-08-09 ~ 2025-12-05 的 299 周样本。
- **图像**：`adf_stats.png` 柱状图展示每个因子的 ADF 统计量，并绘制 1%、5%、10% 三条临界线。
- **结论速览**：所有因子（包括 `MOM_EQ`）的 ADF 统计量均低于 1% 临界值（即柱状条位于虚线下方），p 值 < 0.01，因子序列可视为平稳，不需额外差分即可用于回归分析。
- **更新方式**：执行 `python code/factor_diagnostics.py` 自动刷新 CSV 与图像。
