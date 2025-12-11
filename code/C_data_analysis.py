# -*- coding: utf-8 -*-
"""
数据缺失值分析脚本
专注于分析数据集中的缺失值分布情况
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 配置中文字体
# 尝试使用系统中可用的中文字体
chinese_fonts = [
    '/usr/local/share/fonts/simhei.ttf',  # SimHei
    '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',  # WenQuanYi Zen Hei
]

font_path = None
for font in chinese_fonts:
    if Path(font).exists():
        font_path = font
        logger.info(f"Using Chinese font file: {font}")
        break

selected_font_name = None

if font_path:
    try:
        fm.fontManager.addfont(font_path)
        font_prop = fm.FontProperties(fname=font_path)
        selected_font_name = font_prop.get_name()
        plt.rcParams['font.family'] = [selected_font_name]
        plt.rcParams['font.sans-serif'] = [selected_font_name]
        logger.info(f"已注册中文字体: {selected_font_name}")
    except Exception as exc:
        logger.warning(f"加载中文字体失败，使用默认字体: {exc}")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
else:
    logger.warning("No Chinese font found, using default font")
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['figure.dpi'] = 100

# 路径配置
DATA_PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
REPORTS_DIR = Path(__file__).parent.parent / "data" / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True, parents=True)


class MissingValueAnalyzer:
    """缺失值分析器"""

    def __init__(self, data_dir, figures_dir):
        self.data_dir = Path(data_dir)
        self.figures_dir = Path(figures_dir)
        self.datasets = {}

    def load_all_data(self):
        """加载所有parquet文件"""
        logger.info("=" * 80)
        logger.info("加载数据集")
        logger.info("=" * 80)

        files = {
            '行情数据': 'market_prices.parquet',
            '规模因子': 'size_factor.parquet',
            '账面市值比因子': 'bm_factor.parquet',
            '盈利因子': 'profitability_factor.parquet',
            '投资因子': 'investment_factor.parquet',
            '无风险利率': 'risk_free_rate.parquet',
        }

        for name, filename in files.items():
            filepath = self.data_dir / filename
            if filepath.exists():
                self.datasets[name] = pd.read_parquet(filepath)
                logger.info(f"  已加载 {name}: {self.datasets[name].shape}")
            else:
                logger.warning(f"  文件不存在: {filepath}")

    def analyze_missing_values(self):
        """分析所有数据集的缺失值"""
        logger.info("\n" + "=" * 80)
        logger.info("缺失值分析")
        logger.info("=" * 80)

        missing_stats = []

        for name, df in self.datasets.items():
            total_cells = df.shape[0] * df.shape[1]
            missing_cells = df.isnull().sum().sum()
            missing_pct = (missing_cells / total_cells) * 100 if total_cells > 0 else 0

            logger.info(f"\n【{name}】")
            logger.info(f"  总单元格数: {total_cells:,}")
            logger.info(f"  缺失单元格数: {missing_cells:,}")
            logger.info(f"  缺失率: {missing_pct:.2f}%")

            # 统计每列的缺失值
            col_missing = df.isnull().sum()
            if col_missing.sum() > 0:
                logger.info(f"  列缺失详情:")
                for col, count in col_missing[col_missing > 0].items():
                    pct = (count / len(df)) * 100
                    logger.info(f"    - {col}: {count:,} ({pct:.2f}%)")
            else:
                logger.info(f"  ✅ 无缺失值")

            missing_stats.append({
                'dataset': name,
                'total_cells': total_cells,
                'missing_cells': missing_cells,
                'missing_pct': missing_pct
            })

        return missing_stats

    def plot_missing_overview(self, missing_stats):
        """绘制缺失值总览图"""
        logger.info("\n生成缺失值总览图...")

        df_stats = pd.DataFrame(missing_stats)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 左图：缺失单元格数量
        bars1 = axes[0].barh(df_stats['dataset'], df_stats['missing_cells'], 
                             color='steelblue', edgecolor='black')
        axes[0].set_xlabel('缺失单元格数量', fontsize=13, fontweight='bold')
        axes[0].set_ylabel('数据集', fontsize=13, fontweight='bold')
        axes[0].set_title('各数据集缺失单元格数量', fontsize=15, fontweight='bold', pad=20)
        axes[0].grid(axis='x', alpha=0.3, linestyle='--')

        # 添加数值标签
        for bar in bars1:
            width = bar.get_width()
            if width > 0:
                axes[0].text(width, bar.get_y() + bar.get_height()/2,
                           f'{int(width):,}',
                           ha='left', va='center', fontsize=11, fontweight='bold')

        # 右图：缺失百分比
        bars2 = axes[1].barh(df_stats['dataset'], df_stats['missing_pct'], 
                             color='coral', edgecolor='black')
        axes[1].set_xlabel('缺失率 (%)', fontsize=13, fontweight='bold')
        axes[1].set_ylabel('数据集', fontsize=13, fontweight='bold')
        axes[1].set_title('各数据集缺失率', fontsize=15, fontweight='bold', pad=20)
        axes[1].grid(axis='x', alpha=0.3, linestyle='--')

        # 添加百分比标签
        for bar in bars2:
            width = bar.get_width()
            if width > 0:
                axes[1].text(width + 0.1, bar.get_y() + bar.get_height()/2,
                           f'{width:.2f}%',
                           ha='left', va='center', fontsize=11, fontweight='bold')

        plt.tight_layout()
        output_path = self.figures_dir / "01_missing_values_overview.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"  已保存: {output_path}")
        plt.close()

    def plot_missing_by_column(self):
        """绘制每个数据集的列级缺失值详情"""
        logger.info("\n生成列级缺失值详情图...")

        # 只分析有缺失值的数据集
        datasets_with_missing = {
            name: df for name, df in self.datasets.items()
            if df.isnull().sum().sum() > 0
        }

        if not datasets_with_missing:
            logger.info("  所有数据集均无缺失值")
            return

        saved = 0

        for name, df in datasets_with_missing.items():
            # 计算每列的缺失率并按降序排列
            missing_per_col = df.isnull().sum() / len(df) * 100
            missing_per_col = missing_per_col[missing_per_col > 0].sort_values(ascending=False)

            if missing_per_col.empty:
                continue

            # 根据列数量动态调整画布高度，避免生成超大图像
            dynamic_height = max(4, min(0.45 * len(missing_per_col) + 2, 24))
            fig, ax = plt.subplots(figsize=(14, dynamic_height))
            size_w, size_h = fig.get_size_inches()
            logger.info(f"  - {name}: 缺失列 {len(missing_per_col)} 个, 图像尺寸 {size_w:.2f} x {size_h:.2f} 英寸")

            bars = ax.barh(range(len(missing_per_col)), missing_per_col.values,
                           color='crimson', edgecolor='black', alpha=0.85)
            ax.set_yticks(range(len(missing_per_col)))
            ax.set_yticklabels(missing_per_col.index, fontsize=12)
            ax.set_xlabel('缺失率 (%)', fontsize=12, fontweight='bold')
            ax.set_title(f'{name} - 各列缺失率分布',
                         fontsize=14, fontweight='bold', pad=15)
            ax.grid(axis='x', alpha=0.3, linestyle='--')

            for i, (bar, val) in enumerate(zip(bars, missing_per_col.values)):
                ax.text(val + 0.3, i, f'{val:.2f}%',
                        va='center', fontsize=11, fontweight='bold')
            left_margin = 0.28 if len(missing_per_col) <= 10 else 0.35
            fig.subplots_adjust(left=left_margin, right=0.97, top=0.9, bottom=0.12)
            safe_name = name.replace('/', '_').replace('\\', '_')
            output_path = self.figures_dir / f"02_missing_by_column_{safe_name}.png"
            plt.savefig(output_path, dpi=300)
            logger.info(f"  已保存: {output_path}")
            plt.close(fig)
            saved += 1

        if saved == 0:
            logger.info("  有缺失值的数据集未检测到列级缺失")

    def plot_missing_heatmap(self):
        """绘制缺失值热力图"""
        logger.info("\n生成缺失值热力图...")

        # 只绘制有缺失值的数据集
        datasets_with_missing = {
            name: df for name, df in self.datasets.items()
            if df.isnull().sum().sum() > 0
        }

        if not datasets_with_missing:
            logger.info("  所有数据集均无缺失值")
            return

        for name, df in datasets_with_missing.items():
            # 只显示有缺失值的列
            missing_cols = df.columns[df.isnull().any()].tolist()
            if not missing_cols:
                continue

            # 如果数据太大，采样显示
            sample_size = min(1000, len(df))
            df_sample = df[missing_cols].sample(n=sample_size, random_state=42)

            fig, ax = plt.subplots(figsize=(12, 8))
            
            # 创建缺失值矩阵 (True=缺失, False=存在)
            missing_matrix = df_sample.isnull()
            
            sns.heatmap(missing_matrix, cbar=True, cmap='RdYlGn_r',
                       yticklabels=False, ax=ax, cbar_kws={'label': '缺失(红)/存在(绿)'})
            
            ax.set_title(f'{name} - 缺失值分布热力图\n(抽样 {sample_size} 行)', 
                        fontsize=15, fontweight='bold', pad=20)
            ax.set_xlabel('列名', fontsize=12, fontweight='bold')
            ax.set_ylabel(f'样本行 (共{len(df):,}行)', fontsize=12, fontweight='bold')

            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # 使用安全的文件名
            safe_name = name.replace('/', '_').replace('\\', '_')
            output_path = self.figures_dir / f"03_heatmap_{safe_name}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"  已保存: {output_path}")
            plt.close()

    def plot_missing_matrix(self):
        """绘制缺失值矩阵对比图"""
        logger.info("\n生成缺失值矩阵对比图...")

        # 收集所有数据集的缺失情况
        matrix_data = []
        
        for name, df in self.datasets.items():
            for col in df.columns:
                missing_count = df[col].isnull().sum()
                missing_pct = (missing_count / len(df)) * 100
                
                matrix_data.append({
                    '数据集': name,
                    '列名': col,
                    '缺失数': missing_count,
                    '缺失率': missing_pct
                })
        
        df_matrix = pd.DataFrame(matrix_data)
        
        # 只保留有缺失的行
        df_matrix_missing = df_matrix[df_matrix['缺失数'] > 0]
        
        if len(df_matrix_missing) == 0:
            logger.info("  所有列均无缺失值")
            return
        
        # 绘制气泡图
        fig, ax = plt.subplots(figsize=(14, 8))
        
        datasets = df_matrix_missing['数据集'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(datasets)))
        color_map = dict(zip(datasets, colors))
        
        for dataset in datasets:
            data = df_matrix_missing[df_matrix_missing['数据集'] == dataset]
            ax.scatter(data['缺失率'], data['列名'], 
                      s=data['缺失数']/10, alpha=0.6,
                      c=[color_map[dataset]], label=dataset,
                      edgecolors='black', linewidth=1)
        
        ax.set_xlabel('缺失率 (%)', fontsize=13, fontweight='bold')
        ax.set_ylabel('列名', fontsize=13, fontweight='bold')
        ax.set_title('所有数据集缺失值分布\n(气泡大小 = 缺失数量)', 
                    fontsize=15, fontweight='bold', pad=20)
        ax.legend(title='数据集', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        output_path = self.figures_dir / "04_missing_matrix_all.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"  已保存: {output_path}")
        plt.close()

    def generate_summary_report(self, missing_stats):
        """生成文本报告"""
        logger.info("\n" + "=" * 80)
        logger.info("生成汇总报告")
        logger.info("=" * 80)

        report = []
        report.append("=" * 80)
        report.append("数据缺失值分析报告")
        report.append("=" * 80)
        report.append("")

        # 1. 总体概览
        report.append("一、总体概览")
        report.append("-" * 80)
        total_datasets = len(self.datasets)
        datasets_with_missing = sum(1 for s in missing_stats if s['missing_cells'] > 0)
        
        report.append(f"数据集总数: {total_datasets}")
        report.append(f"有缺失值的数据集: {datasets_with_missing}")
        report.append(f"完整数据集: {total_datasets - datasets_with_missing}")
        report.append("")

        # 2. 各数据集详情
        report.append("二、各数据集缺失详情")
        report.append("-" * 80)
        
        df_stats = pd.DataFrame(missing_stats).sort_values('missing_pct', ascending=False)
        
        for _, row in df_stats.iterrows():
            report.append(f"\n{row['dataset']}:")
            report.append(f"  总单元格: {row['total_cells']:,}")
            report.append(f"  缺失单元格: {row['missing_cells']:,}")
            report.append(f"  缺失率: {row['missing_pct']:.2f}%")
            
            if row['missing_cells'] == 0:
                report.append(f"  状态: ✅ 完整")
            elif row['missing_pct'] < 1:
                report.append(f"  状态: ⚠️  少量缺失")
            elif row['missing_pct'] < 5:
                report.append(f"  状态: ⚠️  中等缺失")
            else:
                report.append(f"  状态: ❌ 严重缺失")

        # 3. 列级缺失详情
        report.append("\n\n三、列级缺失详情")
        report.append("-" * 80)
        
        for name, df in self.datasets.items():
            col_missing = df.isnull().sum()
            if col_missing.sum() > 0:
                report.append(f"\n{name}:")
                for col, count in col_missing[col_missing > 0].items():
                    pct = (count / len(df)) * 100
                    report.append(f"  - {col}: {count:,} / {len(df):,} ({pct:.2f}%)")

        # 4. 生成的图表
        report.append("\n\n四、生成的可视化图表")
        report.append("-" * 80)
        figures = sorted(self.figures_dir.glob("*.png"))
        for fig in figures:
            report.append(f"  - {fig.name}")

        # 5. 建议
        report.append("\n\n五、数据处理建议")
        report.append("-" * 80)
        
        for stat in missing_stats:
            if stat['missing_pct'] > 0:
                report.append(f"\n{stat['dataset']}:")
                if stat['missing_pct'] < 1:
                    report.append(f"  建议: 缺失率较低，可直接删除缺失行")
                elif stat['missing_pct'] < 5:
                    report.append(f"  建议: 根据业务需求选择插补或删除策略")
                else:
                    report.append(f"  建议: 缺失严重，需要评估数据可用性")

        report_text = "\n".join(report)
        
        # 保存报告
        report_path = REPORTS_DIR / "missing_value_analysis_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        logger.info(f"\n报告已保存: {report_path}")
        print("\n" + report_text)


def main():
    logger.info("=" * 80)
    logger.info("开始数据缺失值分析")
    logger.info("=" * 80)

    analyzer = MissingValueAnalyzer(DATA_PROCESSED_DIR, FIGURES_DIR)
    
    # 加载数据
    analyzer.load_all_data()
    
    # 分析缺失值
    missing_stats = analyzer.analyze_missing_values()
    
    # 生成可视化
    analyzer.plot_missing_overview(missing_stats)
    analyzer.plot_missing_by_column()
    analyzer.plot_missing_heatmap()
    analyzer.plot_missing_matrix()
    
    # 生成报告
    analyzer.generate_summary_report(missing_stats)
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ 分析完成！")
    logger.info(f"图表保存位置: {FIGURES_DIR}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
