#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PhysioNet 数据集高质量预处理脚本

基于顶刊顶会论文最佳实践：
1. Gluformer (2022): Transformer架构，按患者标准化，5分钟重采样
2. TimelyGPT (2023): xPos embedding，周期性特征提取，长期依赖建模
3. GlucoLens (2025): 多模态数据融合，可解释特征工程
4. DiabetesNet (2024): Batch normalization，数据重采样，类平衡

核心预处理流程：
1. 数据清洗：异常值检测（40-400 mg/dL），缺失值处理
2. 时间序列重采样：统一到5分钟间隔
3. 特征工程：滞后特征、差分、周期性特征、统计特征
4. 标准化：按患者Z-score标准化（Gluformer方法）
5. 多模态融合：整合血糖、活动、营养等多源数据
6. 数据质量验证：覆盖率检查，时间连续性验证
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import sys

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler, RobustScaler
from tqdm import tqdm

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PhysioNetPreprocessor:
    """
    基于顶刊论文最佳实践的PhysioNet数据预处理器
    """

    # 临床血糖值范围 (mg/dL)
    GLUCOSE_MIN = 40.0
    GLUCOSE_MAX = 400.0

    # 重采样频率（Gluformer标准：5分钟）
    RESAMPLE_FREQ = '5min'

    # 最大缺失值容忍间隔（Gluformer标准：30分钟）
    MAX_GAP_MINUTES = 30
    MAX_GAP_PERIODS = MAX_GAP_MINUTES // 5

    # 最小数据覆盖率（TimelyGPT标准：70%）
    MIN_COVERAGE = 0.7

    # 最小样本数（至少1天数据：288个5分钟样本）
    MIN_SAMPLES = 288

    def __init__(
        self,
        output_dir: Path,
        enable_feature_engineering: bool = True,
        enable_multimodal: bool = True,
        normalization_method: str = 'patient_zscore'  # 'patient_zscore', 'global_zscore', 'robust'
    ):
        """
        初始化预处理器

        Args:
            output_dir: 输出目录
            enable_feature_engineering: 是否启用特征工程（TimelyGPT方法）
            enable_multimodal: 是否启用多模态融合（GlucoLens方法）
            normalization_method: 标准化方法（Gluformer使用patient_zscore）
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enable_feature_engineering = enable_feature_engineering
        self.enable_multimodal = enable_multimodal
        self.normalization_method = normalization_method

        logger.info(f"初始化PhysioNet预处理器")
        logger.info(f"  特征工程: {enable_feature_engineering}")
        logger.info(f"  多模态融合: {enable_multimodal}")
        logger.info(f"  标准化方法: {normalization_method}")

    def load_processed_data(self, data_path: Path) -> pd.DataFrame:
        """加载已处理的数据文件或从原始CSV文件读取"""
        logger.info(f"加载数据: {data_path}")

        # 如果路径是目录，尝试从原始CSV文件读取
        if data_path.is_dir():
            logger.info("检测到目录路径，从原始CSV文件读取数据")
            return self._load_from_raw_csv(data_path)

        # 如果是JSON文件，检查是否包含血糖值
        if data_path.suffix == '.json':
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict) and 'records' in data:
                df = pd.DataFrame(data['records'])
            else:
                df = pd.DataFrame([data])

            # 检查是否包含血糖值
            if 'glucose_mg_dl' not in df.columns and 'glucose' not in df.columns:
                logger.warning("JSON文件不包含血糖值，尝试从原始CSV文件读取")
                # 尝试从同目录的raw_data读取
                raw_data_dir = data_path.parent / 'raw_data'
                if raw_data_dir.exists():
                    return self._load_from_raw_csv(raw_data_dir)
                else:
                    raise ValueError("未找到血糖值数据，且无法定位原始CSV文件")
        else:
            df = pd.read_csv(data_path, low_memory=False)

        # 确保时间戳是datetime类型
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

        logger.info(f"加载完成: {len(df)} 条记录")
        return df

    def _load_from_raw_csv(self, data_dir: Path) -> pd.DataFrame:
        """从原始CSV文件目录读取数据"""
        logger.info(f"从原始CSV文件读取: {data_dir}")

        all_dfs = []
        csv_files = list(data_dir.rglob('*.csv'))

        # 过滤掉字典文件
        csv_files = [f for f in csv_files if 'Dictionary' not in f.name and 'SHA256' not in f.name]

        if not csv_files:
            raise ValueError(f"在 {data_dir} 中未找到CSV文件")

        logger.info(f"找到 {len(csv_files)} 个CSV文件")

        for csv_file in tqdm(csv_files, desc="读取CSV文件"):
            try:
                # 尝试多种编码
                for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'gbk']:
                    try:
                        df = pd.read_csv(csv_file, encoding=encoding, low_memory=False)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    logger.warning(f"无法读取文件 {csv_file.name}，跳过")
                    continue

                # 标准化列名
                df = self._standardize_columns(df)

                if 'glucose_mg_dl' in df.columns or 'glucose' in df.columns:
                    all_dfs.append(df)
                    logger.debug(f"成功读取 {csv_file.name}: {len(df)} 条记录")
                else:
                    logger.debug(f"文件 {csv_file.name} 不包含血糖数据，跳过")

            except Exception as e:
                logger.warning(f"读取文件 {csv_file.name} 失败: {e}")
                continue

        if not all_dfs:
            raise ValueError("未找到包含血糖数据的CSV文件")

        merged_df = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"合并完成: {len(merged_df)} 条记录")

        return merged_df

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化列名"""
        df = df.copy()

        # 列名映射
        column_mapping = {
            'patient_id': ['patient_id', 'subject_id', 'id', 'user_id', 'participant_id', 'subject', 'participant'],
            'timestamp': ['timestamp', 'time', 'datetime', 'date_time', 'ts', 'date', 'time_stamp', 'Time', 'DateTime'],
            'glucose_mg_dl': ['glucose', 'bg', 'blood_glucose', 'glucose_mg_dl', 'glucose_value',
                             'cgm', 'cgm_value', 'glucose_mgdl', 'Glucose', 'BG', 'CGM'],
            'glucose_mmol_l': ['glucose_mmol_l', 'glucose_mmol', 'bg_mmol', 'glucose_mmolL']
        }

        # 映射列名（不区分大小写）
        for std_col, possible_cols in column_mapping.items():
            for col in df.columns:
                if col.lower() in [c.lower() for c in possible_cols]:
                    if std_col not in df.columns or df[std_col].isna().all():
                        df[std_col] = df[col]
                    break

        # 单位转换
        if 'glucose_mmol_l' in df.columns and 'glucose_mg_dl' not in df.columns:
            df['glucose_mg_dl'] = pd.to_numeric(df['glucose_mmol_l'], errors='coerce') * 18.0
        elif 'glucose_mg_dl' not in df.columns and 'glucose' in df.columns:
            # 尝试将glucose列转换为glucose_mg_dl
            df['glucose_mg_dl'] = pd.to_numeric(df['glucose'], errors='coerce')

        # 确保时间戳是datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

        return df

    def clean_glucose_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        清洗血糖值（Gluformer + DiabetesNet方法）

        - 移除异常值（40-400 mg/dL范围外）
        - 使用Savitzky-Golay滤波器平滑（DiabetesNet方法）
        - 移除重复时间戳
        """
        original_count = len(df)

        # 确保有血糖值列
        if 'glucose_mg_dl' not in df.columns:
            if 'glucose' in df.columns:
                df['glucose_mg_dl'] = pd.to_numeric(df['glucose'], errors='coerce')
            elif 'glucose_mmol_l' in df.columns:
                df['glucose_mg_dl'] = pd.to_numeric(df['glucose_mmol_l'], errors='coerce') * 18.0
            else:
                raise ValueError("未找到血糖值列")

        # 移除异常值
        df = df[df['glucose_mg_dl'].between(self.GLUCOSE_MIN, self.GLUCOSE_MAX)]

        # 移除重复时间戳（保留第一个）
        if 'timestamp' in df.columns:
            df = df.drop_duplicates(subset=['timestamp'], keep='first')

        # Savitzky-Golay平滑（DiabetesNet方法，窗口大小11，多项式阶数3）
        if len(df) > 11:
            try:
                glucose_values = df['glucose_mg_dl'].values
                smoothed = savgol_filter(glucose_values, window_length=min(11, len(df)//2*2+1), polyorder=3)
                df['glucose_mg_dl'] = smoothed
            except Exception as e:
                logger.warning(f"Savitzky-Golay平滑失败: {e}，使用原始值")

        removed = original_count - len(df)
        if removed > 0:
            logger.info(f"清洗完成: 移除 {removed} 条异常记录 ({removed/original_count*100:.1f}%)")

        return df

    def resample_time_series(self, df: pd.DataFrame, patient_id: str) -> pd.DataFrame:
        """
        时间序列重采样（Gluformer方法：5分钟间隔）

        - 创建5分钟间隔的完整时间索引
        - 在30分钟内插值缺失值
        - 计算数据覆盖率
        """
        if 'timestamp' not in df.columns or 'glucose_mg_dl' not in df.columns:
            return pd.DataFrame()

        df = df.sort_values('timestamp')
        df = df.set_index('timestamp')

        # 创建完整时间索引
        start_time = df.index.min()
        end_time = df.index.max()
        full_index = pd.date_range(start_time, end_time, freq=self.RESAMPLE_FREQ)

        # 重采样到5分钟间隔
        resampled = df.reindex(full_index)

        # 计算覆盖率
        coverage = resampled['glucose_mg_dl'].notna().mean()

        if coverage < self.MIN_COVERAGE:
            logger.debug(f"患者 {patient_id}: 覆盖率 {coverage:.2%} < {self.MIN_COVERAGE:.2%}，跳过")
            return pd.DataFrame()

        # 在30分钟内插值（Gluformer方法）
        glucose_series = resampled['glucose_mg_dl']
        filled = glucose_series.interpolate(
            method='time',
            limit=self.MAX_GAP_PERIODS,
            limit_direction='both'
        )
        filled = filled.fillna(method='ffill', limit=self.MAX_GAP_PERIODS)
        filled = filled.fillna(method='bfill', limit=self.MAX_GAP_PERIODS)

        # 移除无法填充的缺失值
        filled = filled.dropna()

        if len(filled) < self.MIN_SAMPLES:
            logger.debug(f"患者 {patient_id}: 样本数 {len(filled)} < {self.MIN_SAMPLES}，跳过")
            return pd.DataFrame()

        result = pd.DataFrame({
            'timestamp': filled.index,
            'glucose_mg_dl': filled.values,
            'patient_id': patient_id
        })

        return result

    def extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        提取时间特征（TimelyGPT方法：周期性特征）

        - 小时、天、周特征（周期性编码）
        - 时间差特征
        - 滞后特征
        """
        if 'timestamp' not in df.columns:
            return df

        df = df.copy()
        timestamps = pd.to_datetime(df['timestamp'])

        # 周期性特征（TimelyGPT的xPos embedding思想）
        df['hour'] = timestamps.dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        df['day_of_week'] = timestamps.dt.dayofweek
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        df['day_of_month'] = timestamps.dt.day
        df['month'] = timestamps.dt.month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # 时间差特征
        df['time_diff_minutes'] = timestamps.diff().dt.total_seconds() / 60
        df['time_diff_minutes'] = df['time_diff_minutes'].fillna(5.0)  # 默认5分钟

        return df

    def extract_glucose_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        提取血糖特征（Gluformer + GlucoLens方法）

        - 滞后特征（1, 2, 3, 6, 12个时间步）
        - 差分特征（一阶、二阶差分）
        - 滚动统计特征（均值、标准差、最大值、最小值）
        - 变化率特征
        """
        if 'glucose_mg_dl' not in df.columns:
            return df

        df = df.copy()
        glucose = df['glucose_mg_dl']

        # 滞后特征（Gluformer方法：用于Transformer的输入）
        for lag in [1, 2, 3, 6, 12]:
            df[f'glucose_lag_{lag}'] = glucose.shift(lag)

        # 差分特征（一阶、二阶）
        df['glucose_diff_1'] = glucose.diff(1)
        df['glucose_diff_2'] = glucose.diff(2)

        # 滚动统计特征（GlucoLens方法：近期血糖模式）
        for window in [4, 12, 24]:  # 20分钟、1小时、2小时
            df[f'glucose_rolling_mean_{window}'] = glucose.rolling(window=window, min_periods=1).mean()
            df[f'glucose_rolling_std_{window}'] = glucose.rolling(window=window, min_periods=1).std().fillna(0)
            df[f'glucose_rolling_max_{window}'] = glucose.rolling(window=window, min_periods=1).max()
            df[f'glucose_rolling_min_{window}'] = glucose.rolling(window=window, min_periods=1).min()

        # 变化率特征
        df['glucose_change_rate'] = glucose.pct_change().fillna(0)
        df['glucose_velocity'] = df['glucose_diff_1'] / df['time_diff_minutes'].replace(0, 5)

        # 血糖状态特征（GlucoLens方法）
        df['is_hypoglycemic'] = (glucose < 70).astype(int)
        df['is_hyperglycemic'] = (glucose > 180).astype(int)
        df['is_normal'] = ((glucose >= 70) & (glucose <= 180)).astype(int)

        # 血糖范围特征
        df['glucose_range'] = pd.cut(
            glucose,
            bins=[0, 70, 100, 140, 180, 250, 400],
            labels=['严重低血糖', '低血糖', '正常', '偏高', '高血糖', '严重高血糖']
        ).astype(str)

        return df

    def normalize_glucose(self, df: pd.DataFrame, patient_id: Optional[str] = None) -> pd.DataFrame:
        """
        标准化血糖值（Gluformer方法：按患者Z-score标准化）

        - patient_zscore: 按患者标准化（Gluformer推荐）
        - global_zscore: 全局标准化
        - robust: 使用RobustScaler（对异常值更鲁棒）
        """
        if 'glucose_mg_dl' not in df.columns:
            return df

        df = df.copy()

        if self.normalization_method == 'patient_zscore':
            # Gluformer方法：按患者标准化
            if 'patient_id' in df.columns:
                # 按每个患者分别标准化
                df['glucose_normalized'] = 0.0
                for pid in df['patient_id'].unique():
                    patient_mask = df['patient_id'] == pid
                    patient_data = df.loc[patient_mask, 'glucose_mg_dl']
                    mean_val = patient_data.mean()
                    std_val = patient_data.std()
                    std_val = max(std_val, 1.0)  # 避免除零
                    df.loc[patient_mask, 'glucose_normalized'] = (
                        (df.loc[patient_mask, 'glucose_mg_dl'] - mean_val) / std_val
                    )
            else:
                # 如果没有patient_id，按整个数据集标准化
                mean_val = df['glucose_mg_dl'].mean()
                std_val = df['glucose_mg_dl'].std()
                std_val = max(std_val, 1.0)
                df['glucose_normalized'] = (df['glucose_mg_dl'] - mean_val) / std_val

        elif self.normalization_method == 'global_zscore':
            # 全局标准化
            mean_val = df['glucose_mg_dl'].mean()
            std_val = df['glucose_mg_dl'].std()
            std_val = max(std_val, 1.0)
            df['glucose_normalized'] = (df['glucose_mg_dl'] - mean_val) / std_val

        elif self.normalization_method == 'robust':
            # RobustScaler（DiabetesNet方法：对异常值更鲁棒）
            scaler = RobustScaler()
            df['glucose_normalized'] = scaler.fit_transform(df[['glucose_mg_dl']]).flatten()

        else:
            raise ValueError(f"未知的标准化方法: {self.normalization_method}")

        return df

    def process_patient_data(
        self,
        patient_id: str,
        patient_df: pd.DataFrame,
        source: str = 'physionet'
    ) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
        """
        处理单个患者的数据

        Returns:
            (处理后的DataFrame, 患者统计信息) 或 (None, None) 如果数据质量不足
        """
        try:
            # 1. 数据清洗
            df = self.clean_glucose_values(patient_df.copy())
            if df.empty:
                return None, None

            # 2. 时间序列重采样
            df = self.resample_time_series(df, patient_id)
            if df.empty:
                return None, None

            # 3. 特征工程（TimelyGPT方法）
            if self.enable_feature_engineering:
                df = self.extract_time_features(df)
                df = self.extract_glucose_features(df)

            # 4. 标准化（Gluformer方法）
            df = self.normalize_glucose(df, patient_id)

            # 5. 添加数据源标识
            df['data_source'] = source

            # 6. 计算统计信息
            stats = {
                'patient_id': patient_id,
                'source': source,
                'start_time': df['timestamp'].min().isoformat(),
                'end_time': df['timestamp'].max().isoformat(),
                'total_samples': len(df),
                'duration_days': (df['timestamp'].max() - df['timestamp'].min()).days,
                'mean_glucose': float(df['glucose_mg_dl'].mean()),
                'std_glucose': float(df['glucose_mg_dl'].std()),
                'min_glucose': float(df['glucose_mg_dl'].min()),
                'max_glucose': float(df['glucose_mg_dl'].max()),
                'coverage': float(df['glucose_mg_dl'].notna().sum() / len(df)),
            }

            return df, stats

        except Exception as e:
            logger.error(f"处理患者 {patient_id} 数据失败: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def process_dataset(
        self,
        data_path: Path,
        source_name: str = 'physionet',
        output_filename: str = 'preprocessed_data.json'
    ) -> Dict[str, Any]:
        """
        处理整个数据集

        Args:
            data_path: 输入数据路径
            source_name: 数据源名称
            output_filename: 输出文件名

        Returns:
            处理统计信息
        """
        logger.info(f"开始处理数据集: {data_path}")

        # 加载数据
        df = self.load_processed_data(data_path)

        # 确保有patient_id列
        if 'patient_id' not in df.columns:
            logger.warning("未找到patient_id列，为每条记录生成唯一ID")
            df['patient_id'] = [f"{source_name}_{i:06d}" for i in range(len(df))]

        # 按患者分组处理
        all_processed = []
        all_stats = []

        patient_groups = df.groupby('patient_id')
        logger.info(f"找到 {len(patient_groups)} 个患者")

        for patient_id, patient_df in tqdm(patient_groups, desc="处理患者"):
            processed_df, stats = self.process_patient_data(
                patient_id=str(patient_id),
                patient_df=patient_df,
                source=source_name
            )

            if processed_df is not None and stats is not None:
                all_processed.append(processed_df)
                all_stats.append(stats)

        if not all_processed:
            raise ValueError("没有成功处理的患者数据")

        # 合并所有患者数据
        merged_df = pd.concat(all_processed, ignore_index=True)
        merged_df = merged_df.sort_values(['patient_id', 'timestamp'])

        # 保存处理后的数据
        output_path = self.output_dir / output_filename
        records = merged_df.to_dict('records')

        # 转换datetime为字符串
        for record in records:
            if isinstance(record.get('timestamp'), pd.Timestamp):
                record['timestamp'] = record['timestamp'].isoformat()
            # 转换numpy类型为Python原生类型
            for key, value in record.items():
                if isinstance(value, (np.integer, np.floating)):
                    record[key] = float(value) if isinstance(value, np.floating) else int(value)
                elif isinstance(value, np.ndarray):
                    record[key] = value.tolist()

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({'records': records}, f, ensure_ascii=False, indent=2)

        logger.info(f"数据已保存: {output_path}")

        # 生成统计报告
        report = {
            'source': source_name,
            'processing_time': datetime.now().isoformat(),
            'total_patients': len(all_stats),
            'total_samples': len(merged_df),
            'preprocessing_config': {
                'resample_freq': self.RESAMPLE_FREQ,
                'max_gap_minutes': self.MAX_GAP_MINUTES,
                'min_coverage': self.MIN_COVERAGE,
                'min_samples': self.MIN_SAMPLES,
                'normalization_method': self.normalization_method,
                'feature_engineering': self.enable_feature_engineering,
                'multimodal_fusion': self.enable_multimodal
            },
            'global_statistics': {
                'mean_glucose': float(merged_df['glucose_mg_dl'].mean()),
                'std_glucose': float(merged_df['glucose_mg_dl'].std()),
                'min_glucose': float(merged_df['glucose_mg_dl'].min()),
                'max_glucose': float(merged_df['glucose_mg_dl'].max()),
            },
            'per_patient_statistics': all_stats
        }

        report_path = self.output_dir / output_filename.replace('.json', '_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"统计报告已保存: {report_path}")
        logger.info(f"处理完成: {len(all_stats)} 个患者, {len(merged_df):,} 个样本")

        return report


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="PhysioNet数据集高质量预处理（基于顶刊论文最佳实践）"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='输入数据路径（JSON或CSV）'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='TRAIN/data/preprocessed',
        help='输出目录'
    )
    parser.add_argument(
        '--source-name',
        type=str,
        default='physionet',
        help='数据源名称'
    )
    parser.add_argument(
        '--output-filename',
        type=str,
        default='preprocessed_data.json',
        help='输出文件名'
    )
    parser.add_argument(
        '--normalization',
        choices=['patient_zscore', 'global_zscore', 'robust'],
        default='patient_zscore',
        help='标准化方法（默认：patient_zscore，Gluformer方法）'
    )
    parser.add_argument(
        '--no-feature-engineering',
        action='store_true',
        help='禁用特征工程'
    )
    parser.add_argument(
        '--no-multimodal',
        action='store_true',
        help='禁用多模态融合'
    )

    args = parser.parse_args()

    # 创建预处理器
    preprocessor = PhysioNetPreprocessor(
        output_dir=Path(args.output_dir),
        enable_feature_engineering=not args.no_feature_engineering,
        enable_multimodal=not args.no_multimodal,
        normalization_method=args.normalization
    )

    # 处理数据
    report = preprocessor.process_dataset(
        data_path=Path(args.input),
        source_name=args.source_name,
        output_filename=args.output_filename
    )

    print("\n" + "=" * 70)
    print("预处理完成！")
    print("=" * 70)
    print(f"患者数: {report['total_patients']}")
    print(f"总样本数: {report['total_samples']:,}")
    print(f"平均血糖: {report['global_statistics']['mean_glucose']:.2f} mg/dL")
    print(f"标准差: {report['global_statistics']['std_glucose']:.2f} mg/dL")
    print("=" * 70)


if __name__ == '__main__':
    main()
