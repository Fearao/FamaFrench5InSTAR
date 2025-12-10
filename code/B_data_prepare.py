# -*- coding: utf-8 -*-
"""
数据整理脚本
将原始Excel数据整理为结构化的parquet文件
不进行数据清洗，只做结构化整理
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 路径配置
DATA_RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
DATA_PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
DATA_PROCESSED_DIR.mkdir(exist_ok=True)


class DataOrganizer:
    """数据整理器类"""

    def __init__(self, raw_dir, processed_dir):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.schema = {}

    def organize_all_data(self):
        """整理所有数据"""
        logger.info("=" * 80)
        logger.info("Start data organization")
        logger.info("=" * 80)

        # 处理行情数据
        self._organize_market_prices()

        # 处理基础数据（因子数据）
        self._organize_factor_data()

        # 生成数据字典
        self._generate_schema()

        logger.info("\n✅ Data organization completed!")

    def _organize_market_prices(self):
        """整理行情数据：合并8个sheet"""
        logger.info("\n[1/2] Organizing market price data...")

        market_file = self.raw_dir / "行情序列后复权.xlsx"
        if not market_file.exists():
            logger.warning(f"Market price file not found: {market_file}")
            return

        xl_file = pd.ExcelFile(market_file)
        all_data = []

        for sheet_name in xl_file.sheet_names:
            logger.info(f"  Reading sheet: {sheet_name}")
            df = pd.read_excel(market_file, sheet_name=sheet_name)

            # Remove empty rows at the top (if any)
            df = df.dropna(how='all')

            # Add source sheet column
            df['sheet_source'] = sheet_name

            all_data.append(df)

        # Combine all sheets
        market_data = pd.concat(all_data, ignore_index=True)

        # Save to parquet
        output_path = self.processed_dir / "market_prices.parquet"
        market_data.to_parquet(output_path, compression='snappy', index=False)
        logger.info(f"  Saved: {output_path}")
        logger.info(f"  Shape: {market_data.shape}")

        # Record schema
        self.schema['market_prices'] = {
            'shape': list(market_data.shape),
            'columns': list(market_data.columns),
            'dtypes': {k: str(v) for k, v in market_data.dtypes.items()},
            'missing_values': {k: int(v) for k, v in market_data.isnull().sum().items()},
        }

    def _organize_factor_data(self):
        """整理基础数据（因子数据）"""
        logger.info("\n[2/2] Organizing fundamental factor data...")

        basic_file = self.raw_dir / "基础数据.xlsx"
        if not basic_file.exists():
            logger.warning(f"Basic data file not found: {basic_file}")
            return

        xl_file = pd.ExcelFile(basic_file)
        sheet_config = {
            '市场因子': 'risk_free_rate',
            '规模因子': 'size_factor',
            '账面市值比因子': 'bm_factor',
            '盈利因子': 'profitability_factor',
            '投资因子': 'investment_factor',
        }

        for sheet_name, output_name in sheet_config.items():
            logger.info(f"  Processing: {sheet_name}")
            df = pd.read_excel(basic_file, sheet_name=sheet_name)

            # Remove empty rows
            df = df.dropna(how='all')

            # Skip header explanation rows (first 2 rows usually contain descriptions)
            # Check if row contains "没有单位" (no unit) - this is the explanation row
            if len(df) > 2 and '没有单位' in df.iloc[1].values:
                df = df.iloc[2:].reset_index(drop=True)

            # Save to parquet
            output_path = self.processed_dir / f"{output_name}.parquet"
            df.to_parquet(output_path, compression='snappy', index=False)
            logger.info(f"    Saved: {output_path}")
            logger.info(f"    Shape: {df.shape}")

            # Record schema
            self.schema[output_name] = {
                'shape': list(df.shape),
                'columns': list(df.columns),
                'dtypes': {k: str(v) for k, v in df.dtypes.items()},
                'missing_values': {k: int(v) for k, v in df.isnull().sum().items()},
            }

    def _generate_schema(self):
        """生成数据字典"""
        logger.info("\n[3/3] Generating schema.json...")

        schema_path = self.processed_dir / "schema.json"

        schema_output = {
            'generated_at': datetime.now().isoformat(),
            'tables': self.schema,
            'notes': {
                'market_prices': 'Merged from 8 sheets, includes sheet_source column',
                'risk_free_rate': 'Risk-free rate data (no unit explanation rows removed)',
                'size_factor': 'Size factor data',
                'bm_factor': 'Book-to-Market factor data',
                'profitability_factor': 'Profitability factor data',
                'investment_factor': 'Investment factor data',
            }
        }

        with open(schema_path, 'w', encoding='utf-8') as f:
            json.dump(schema_output, f, ensure_ascii=False, indent=2)

        logger.info(f"  Saved: {schema_path}")

        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("Schema Summary:")
        logger.info("=" * 80)
        for table_name, table_info in self.schema.items():
            logger.info(f"\n{table_name}:")
            logger.info(f"  Shape: {table_info['shape']}")
            logger.info(f"  Columns: {len(table_info['columns'])}")
            missing_sum = sum(table_info['missing_values'].values())
            if missing_sum > 0:
                logger.info(f"  Missing values: {missing_sum} cells")


def main():
    organizer = DataOrganizer(DATA_RAW_DIR, DATA_PROCESSED_DIR)
    organizer.organize_all_data()


if __name__ == "__main__":
    main()
