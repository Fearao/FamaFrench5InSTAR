# -*- coding: utf-8 -*-
"""
数据探索和分析脚本
用于分析 data/raw 目录中的原始数据结构、质量和特征
"""

import pandas as pd
import os
from pathlib import Path
from datetime import datetime
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 数据路径
DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "reports"
OUTPUT_DIR.mkdir(exist_ok=True)


class DataExplorer:
    """数据探索器类"""

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.results = {}

    def explore_excel_files(self):
        """探索所有Excel文件"""
        logger.info("=" * 80)
        logger.info("开始数据探索")
        logger.info("=" * 80)

        excel_files = list(self.data_dir.glob("*.xlsx"))

        if not excel_files:
            logger.warning(f"未找到Excel文件在 {self.data_dir}")
            return

        for file_path in sorted(excel_files):
            logger.info(f"\n处理文件: {file_path.name}")
            self._analyze_excel_file(file_path)

    def _analyze_excel_file(self, file_path):
        """分析单个Excel文件"""
        try:
            xl_file = pd.ExcelFile(file_path)
            file_info = {
                'path': str(file_path),
                'filename': file_path.name,
                'file_size_mb': file_path.stat().st_size / 1024 / 1024,
                'sheets': {}
            }

            logger.info(f"  Sheet列表: {xl_file.sheet_names}")

            for sheet_name in xl_file.sheet_names:
                logger.info(f"\n  【Sheet: {sheet_name}】")
                df = pd.read_excel(file_path, sheet_name=sheet_name)

                sheet_info = self._analyze_sheet(df, sheet_name)
                file_info['sheets'][sheet_name] = sheet_info

                self._log_sheet_info(df, sheet_name)

            self.results[file_path.name] = file_info

        except Exception as e:
            logger.error(f"  ERROR: {str(e)}")

    def _analyze_sheet(self, df, sheet_name):
        """分析单个sheet"""
        return {
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': list(df.columns),
            'dtypes': dict(df.dtypes),
            'missing_values': dict(df.isnull().sum()),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
        }

    def _log_sheet_info(self, df, sheet_name):
        """打印sheet信息"""
        logger.info(f"    Shape: {df.shape} (rows, columns)")
        logger.info(f"    Columns: {list(df.columns)}")

        # Missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            logger.info(f"    Missing values: {dict(missing[missing > 0])}")

        # Memory usage
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        logger.info(f"    Memory usage: {memory_mb:.2f} MB")


def analyze_specific_files():
    """分析具体的数据特征"""
    logger.info("\n" + "=" * 80)
    logger.info("Detailed Analysis")
    logger.info("=" * 80)

    # Analyze market data
    market_file = DATA_DIR / "行情序列后复权.xlsx"
    if market_file.exists():
        logger.info("\n【行情序列后复权.xlsx】")
        xl_file = pd.ExcelFile(market_file)

        all_codes = set()
        total_rows = 0

        for sheet_name in xl_file.sheet_names:
            df = pd.read_excel(market_file, sheet_name=sheet_name)
            codes = df['代码'].dropna().unique()
            all_codes.update(codes)
            total_rows += len(df)

            logger.info(f"  Sheet: {sheet_name}, Rows: {len(df)}, Unique codes: {len(codes)}")

        logger.info(f"  Total: {total_rows} rows, {len(all_codes)} unique stock codes")

    # Analyze basic data
    basic_file = DATA_DIR / "基础数据.xlsx"
    if basic_file.exists():
        logger.info("\n【基础数据.xlsx】")
        xl_file = pd.ExcelFile(basic_file)

        for sheet_name in xl_file.sheet_names:
            df = pd.read_excel(basic_file, sheet_name=sheet_name)
            logger.info(f"  Sheet: {sheet_name}, Shape: {df.shape}")


def generate_summary_report():
    """生成汇总报告"""
    logger.info("\n" + "=" * 80)
    logger.info("Generating Summary Report")
    logger.info("=" * 80)

    report = []
    report.append("Data Exploration Report")
    report.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("=" * 80)

    # File summary
    report.append("\nFile Summary:")
    total_size = 0
    for filename in sorted(Path(DATA_DIR).glob("*.xlsx")):
        size_mb = filename.stat().st_size / 1024 / 1024
        total_size += size_mb
        report.append(f"  - {filename.name}: {size_mb:.2f} MB")

    report.append(f"  - Total size: {total_size:.2f} MB")

    # Save report
    report_path = OUTPUT_DIR / "data_exploration_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report))

    logger.info(f"Report saved to: {report_path}")


if __name__ == "__main__":
    explorer = DataExplorer(DATA_DIR)
    explorer.explore_excel_files()
    analyze_specific_files()
    generate_summary_report()
    logger.info("\nData exploration completed!")
