# 总结：MOM_EQ 动量基准实验

## 流程概览
1. **数据整理**：`code/B_data_prepare.py` 将 `data/raw/` 中的行情、财报、无风险利率 Excel 结构化为 `data/processed/*.parquet`。
2. **五因子构建**：`code/factor_pipeline.py` 基于周频交易数据与滞后 30 天的账面/盈利/投资指标，生成 `weekly_panel.parquet`、`weekly_fundamentals.parquet` 以及五因子序列 `factor_returns_f5.parquet`（市值、规模、价值、盈利、投资因子均采用 5 分位 High-Low 构造）。
3. **动量扩展**：`code/momentum_comparison.py` 计算 4/8/16/26/52/78 周（均跳过近 1~4 周）的 12-2 动量，并对六个动量因子做等权平均得到 `MOM_EQ`；随后对每个动量版本＋等权组合、以及五因子 benchmark 运行 HAC(4) 回归，结果输出至 `docs/mom_w*_s*.csv/.json`、`docs/MOM_EQ_coeffs.csv`、`docs/five_factor_benchmark_coeffs.csv` 等。
4. **诊断检验**：`code/factor_diagnostics.py` 生成因子相关矩阵和 ADF 平稳性检验（`docs/factor_correlations.csv`、`docs/factor_stationarity.csv`），并在 `docs/results/` 下绘制对应图像。

## 关键结论
- **五因子 benchmark**：R²≈0.956，α≈-0.00064。组合主要受 `MKT-RF≈1.06`、`SMB≈0.13`、`HML≈0.096` 驱动，盈利/投资因子影响较弱。
- **多窗口动量**：所有动量窗口的系数均为负，16 周 (skip2) 与等权 `MOM_EQ` 最显著（`MOM_EQ` 系数 -0.126、t≈-4.07，R²≈0.961）。
- **解释力增量**：相比 benchmark，加入 `MOM_EQ` 使 R² 提升约 0.005，α 再下降 ~0.0007，显示动量因子能进一步解释等权组合的收益波动。
- **相关性 & 平稳性**：六个因子之间的相关系数绝对值均 <0.4（除 `SMB/HML` 与 `MOM_EQ` ~ -0.37），ADF 检验 p 值<0.01，支持因子在样本内平稳、可直接应用于线性模型。

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
