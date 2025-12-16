#!/bin/bash
#
# Fama-French Factor Analysis Pipeline
# 端到端自动化执行脚本
#

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 创建时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="pipeline_run_${TIMESTAMP}.log"

log_info "Pipeline启动时间: $(date)"
log_info "日志文件: $LOG_FILE"

# 检查Python环境
if ! command -v python &> /dev/null; then
    log_error "Python未安装"
    exit 1
fi

log_info "Python版本: $(python --version)"

# ===== 阶段0：数据准备 =====
log_info "========================================="
log_info "阶段0：数据准备"
log_info "========================================="

if [ ! -f "data/processed/monthly_returns.parquet" ]; then
    log_info "执行 B_etl_monthly.py"
    python code/B_etl_monthly.py 2>&1 | tee -a $LOG_FILE
    log_success "月度数据ETL完成"
else
    log_warning "月度数据已存在，跳过"
fi

# ===== 阶段1：投资组合构造 =====
log_info "========================================="
log_info "阶段1：投资组合构造"
log_info "========================================="

if [ ! -f "data/processed/portfolio_returns_size_bm.parquet" ]; then
    log_info "执行 C_portfolio_construction.py"
    python code/C_portfolio_construction.py 2>&1 | tee -a $LOG_FILE
    log_success "投资组合构造完成"
else
    log_warning "投资组合数据已存在，跳过"
fi

# ===== 阶段2：FF5因子构造 =====
log_info "========================================="
log_info "阶段2：FF5因子构造"
log_info "========================================="

if [ ! -f "data/processed/factor_returns_ff5.parquet" ]; then
    log_info "执行 D_factor_construction.py"
    python code/D_factor_construction.py --summary 2>&1 | tee -a $LOG_FILE
    log_success "FF5因子构造完成"
else
    log_warning "FF5因子已存在，跳过"
fi

# ===== 阶段3：多窗口动量因子 =====
log_info "========================================="
log_info "阶段3：多窗口动量因子构造"
log_info "========================================="

if [ ! -f "data/processed/factor_returns_multiwindow.parquet" ]; then
    log_info "执行 L_momentum_multiwindow.py"
    python code/L_momentum_multiwindow.py 2>&1 | tee -a $LOG_FILE
    log_success "多窗口动量因子完成"
else
    log_warning "多窗口动量因子已存在，跳过"
fi

# ===== 阶段4：多模型回归分析 =====
log_info "========================================="
log_info "阶段4：6模型回归对比"
log_info "========================================="

if [ ! -f "docs/tables/multiwindow_regression_summary.csv" ]; then
    log_info "执行 L_regression_multiwindow.py"
    python code/L_regression_multiwindow.py 2>&1 | tee -a $LOG_FILE
    log_success "多模型回归完成"
else
    log_warning "回归结果已存在，跳过"
fi

# ===== 阶段5：单因子vs加权对比 =====
log_info "========================================="
log_info "阶段5：单因子vs加权对比"
log_info "========================================="

if [ ! -f "docs/tables/multiwindow_performance_matrix.csv" ]; then
    log_info "执行 L_momentum_comparison.py"
    python code/L_momentum_comparison.py 2>&1 | tee -a $LOG_FILE
    log_success "对比分析完成"
else
    log_warning "对比分析已存在，跳过"
fi

# ===== 阶段6：优化决策 =====
log_info "========================================="
log_info "阶段6：优化决策"
log_info "========================================="

if [ ! -f "docs/tables/optimization_scorecard.csv" ]; then
    log_info "执行 L_momentum_optimization_final.py"
    python code/L_momentum_optimization_final.py 2>&1 | tee -a $LOG_FILE
    log_success "优化决策完成"
else
    log_warning "优化决策已存在，跳过"
fi

# ===== 阶段7：稳健性检验 =====
log_info "========================================="
log_info "阶段7：稳健性检验"
log_info "========================================="

# 样本分割
if [ ! -f "docs/tables/robustness_sample_split_comparison.csv" ]; then
    log_info "执行 M_robustness_sample_split.py"
    python code/M_robustness_sample_split.py 2>&1 | tee -a $LOG_FILE
    log_success "样本分割检验完成"
else
    log_warning "样本分割结果已存在，跳过"
fi

# 滚动窗口
if [ ! -f "docs/tables/robustness_rolling_window_stats.csv" ]; then
    log_info "执行 M_robustness_rolling_window.py"
    python code/M_robustness_rolling_window.py 2>&1 | tee -a $LOG_FILE
    log_success "滚动窗口分析完成"
else
    log_warning "滚动窗口结果已存在，跳过"
fi

# ===== Pipeline完成 =====
log_info "========================================="
log_success "Pipeline执行完成！"
log_info "========================================="

log_info "生成的关键文件："
log_info "  - data/processed/factor_returns_multiwindow.parquet"
log_info "  - docs/tables/optimization_scorecard.csv"
log_info "  - docs/tables/robustness_sample_split_comparison.csv"
log_info "  - docs/momentum_optimization_report.md"
log_info "  - docs/robustness_test_summary.md"

log_info "完成时间: $(date)"
log_info "总日志: $LOG_FILE"

