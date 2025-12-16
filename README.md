# Fama-French 5因子模型与动量因子研究

**科创板市场多窗口动量因子实证分析**

---

## 📊 项目概览

本项目对中国科创板市场实施Fama-French五因子定价模型，并深入研究多窗口动量因子的构造、优化与稳健性。

### 核心发现

- **最优动量窗口**：24个月动量因子(MOM_24M)在全样本R²改善达3.22pp
- **稳健性警告**：时间样本分割显示MOM显著性从39.3% → 14.0%（-25.3pp）
- **类型依赖性**：size_bm（价值维度）完全失效(-40pp），size_inv（投资维度）相对稳定(-4pp)
- **建议策略**：条件化使用动量因子，仅在size_inv组合中应用MOM_24M

---

## 🗂️ 项目结构

```
FamaFrench5InSTAR/
├── code/                           # 核心代码
│   ├── A_data_prep.py             # 原始数据转换
│   ├── B_etl_monthly.py           # 月度数据ETL
│   ├── C_portfolio_construction.py # 投资组合构造(5×5×3=75)
│   ├── D_factor_construction.py    # FF5因子构造
│   ├── L_momentum_multiwindow.py   # 多窗口动量因子(4/8/12/24月+EQ)
│   ├── L_regression_multiwindow.py # 6模型回归对比
│   ├── L_momentum_comparison.py    # 单因子vs加权对比
│   ├── L_momentum_optimization_final.py # 优化决策
│   ├── M_robustness_sample_split.py     # 样本分割检验
│   └── M_robustness_rolling_window.py   # 滚动窗口分析
├── data/                           
│   ├── raw/                       # 原始Excel数据
│   └── processed/                 # Parquet处理数据
├── docs/                          
│   ├── tables/                    # 分析结果表格
│   ├── momentum_optimization_report.md  # 优化决策报告
│   └── robustness_test_summary.md      # 稳健性检验报告
├── run_pipeline.sh                # 自动化执行脚本
└── README.md                      # 本文件
```

---

## ⚡ 快速开始

### 环境要求

- Python 3.8+
- 依赖包：`pandas`, `numpy`, `statsmodels`, `scipy`

### 安装依赖

```bash
pip install -r requirements.txt
```

### 一键运行

```bash
# 端到端自动化执行
bash run_pipeline.sh
```

### 分阶段执行

```bash
# 阶段0-1：数据准备与投资组合构造
python code/B_etl_monthly.py
python code/C_portfolio_construction.py

# 阶段2：FF5因子构造
python code/D_factor_construction.py --summary

# 阶段3：多窗口动量因子
python code/L_momentum_multiwindow.py

# 阶段4：回归分析与优化
python code/L_regression_multiwindow.py
python code/L_momentum_comparison.py
python code/L_momentum_optimization_final.py

# 阶段5：稳健性检验
python code/M_robustness_sample_split.py
python code/M_robustness_rolling_window.py
```

---

## 📈 研究成果

### 阶段1：基础FF5实现

- 构造75个投资组合（5 Size × 5 BM/INV/OP × 3类型）
- 实现市场因子(MKT)、规模因子(SMB)、价值因子(HML)、盈利因子(RMW)、投资因子(CMA)

### 阶段2：多窗口动量优化

| 模型 | ΔR² (pp) | MOM显著性 | 综合分 | 排名 |
|------|---------|----------|--------|------|
| **MOM_24M** | **3.22** | **41.8%** | **82.5** | 🥇 1 |
| MOM_4M | 2.86 | 38.2% | 72.7 | 🥈 2 |
| MOM_EQ | 2.05 | 32.7% | 43.3 | 🥉 3 |
| MOM_8M | 1.89 | 18.2% | 42.8 | 4 |
| MOM_12M | 1.59 | 34.5% | 40.4 | 5 |

**决策权重**：40% R²改善 + 30% MOM显著性 + 20% 简洁性 + 10% 经济可解释性

### 阶段3：稳健性检验（关键发现）

#### 时间稳定性测试

| 指标 | 前期(2021-08~2023-09) | 后期(2023-10~2025-12) | 变化 |
|------|---------------------|---------------------|------|
| MOM显著性 | 39.3% | 14.0% | **-25.3pp** ❌ |
| 平均R² | 79.2% | 89.9% | +10.7pp ✅ |
| Alpha显著性 | 24.6% | 10.5% | -14.1pp |

**矛盾**：R²改善但MOM定价能力衰减

#### 滚动窗口演变

| 窗口结束月份 | MOM显著率 | 趋势 |
|-----------|----------|------|
| 2023-07 | 45.5% | 📈 高位 |
| 2024-01 | 18.4% | 📉 骤降 |
| 2024-07 | 25.0% | ↗️ 反弹 |
| 2025-01 | 44.0% | 📈 恢复 |
| 2025-07 | **16.0%** | 📉 **新低** |

**时间相关性**：-0.3745 → Declining趋势

#### 投资组合类型稳健性

| 类型 | 前期MOM显著 | 后期MOM显著 | 变化 | 判定 |
|------|-----------|-----------|------|------|
| size_inv | 16.0% | 12.0% | -4.0pp | ✅ 最稳定 |
| size_op | 81.8% | 57.1% | -24.7pp | ⚠️ 中度衰退 |
| size_bm | 44.0% | 4.0% | **-40.0pp** | ❌ 完全失效 |

---

## 💡 核心结论

### 主要贡献

1. **首次系统化研究科创板市场的多窗口动量因子**
   - 测试4/8/12/24个月窗口及等权组合
   - 发现24个月窗口在全样本表现最优

2. **揭示新兴市场动量因子的时变性**
   - MOM_24M显著性随时间系统性下降（相关系数-0.37）
   - 投资组合类型对动量敏感度差异巨大（size_bm -40pp vs size_inv -4pp）

3. **提出条件化因子使用策略**
   - 放弃在size_bm（价值维度）中使用动量因子
   - 仅在size_inv（投资维度）中保留MOM_24M
   - 为size_op（运营维度）考虑更短窗口(12M/8M)

### 学术价值

- **时变因子定价**：动量效应在科创板呈现系统性衰减
- **投资组合异质性**：不同排序维度对动量因子敏感性差异
- **因子拥挤效应**：24个月窗口可能过长，后期信号被市场吸收

### 实务建议

**短期**：
- 放弃MOM_24M作为通用因子
- 条件化使用：仅在size_inv组合中应用

**中期**：
- 不同投资组合类型使用不同动量窗口
- 引入时变参数模型，动态调整因子权重

**长期**：
- 调查2023年后科创板市场结构变化
- 探索基于机器学习的自适应动量因子

---

## 📁 关键输出

### 数据文件

- `data/processed/factor_returns_multiwindow.parquet` - FF5 + 5个动量因子（53个月）
- `data/processed/portfolio_returns_size_*.parquet` - 75个投资组合月度收益

### 分析结果

- `docs/tables/optimization_scorecard.csv` - 5个动量因子优化评分
- `docs/tables/robustness_by_portfolio_type.csv` - 类型级别稳健性
- `docs/tables/robustness_rolling_window_stats.csv` - 滚动窗口统计
- `docs/momentum_optimization_report.md` - 优化决策详细报告
- `docs/robustness_test_summary.md` - 稳健性检验综合报告

---

## 🔧 技术栈

- **数据处理**：pandas, numpy
- **统计建模**：statsmodels (OLS + Newey-West HAC)
- **数值计算**：scipy
- **数据存储**：Parquet (Apache Arrow)

---

## 📝 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@misc{famaf rench5_star_2025,
  title={Fama-French 5因子与多窗口动量因子：科创板实证研究},
  author={Your Name},
  year={2025},
  note={GitHub repository},
  url={https://github.com/yourusername/FamaFrench5InSTAR}
}
```

---

## 📧 联系方式

如有疑问或建议，欢迎通过以下方式联系：

- 提交Issue: [GitHub Issues](https://github.com/yourusername/FamaFrench5InSTAR/issues)
- Email: your.email@example.com

---

## 📜 许可证

MIT License

---

**最后更新**：2025-12-16  
**项目状态**：研究完成，代码开源
