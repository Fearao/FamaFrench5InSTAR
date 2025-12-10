# Factor Correlation Heatmap

- **数据**：来自 `docs/factor_correlations.csv`（2019-08-09 ~ 2025-12-05，299 周样本）。
- **图像**：`correlation_heatmap.png` 显示六个因子（MKT-RF、SMB、HML、RMW、CMA、MOM_EQ）的皮尔逊相关系数。
- **结论速览**：
  - `SMB` 与 `MOM_EQ`、`HML` 与 `MOM_EQ` 的相关系数约为 -0.36~-0.38，说明动量因子对小盘/价值收益拥有显著的补充作用。
  - 其余相关系数绝对值均 <0.4，未出现严重多重共线性，可放心进入回归模型。
- **更新方式**：重新运行 `python code/factor_diagnostics.py` 会自动覆盖 CSV 与 PNG。
