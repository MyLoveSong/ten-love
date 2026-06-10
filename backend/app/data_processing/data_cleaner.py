

"""
学术级数据清洗与预处理系统
支持缺失值处理、异常值检测、数据标准化、类型转换等
"""

import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import json
from datetime import datetime
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
import asyncio

from backend.app.core.exceptions import CustomException, ValidationError
from app.core.task_queue import async_task, TaskPriority

logger = logging.getLogger(__name__)

class MissingValueStrategy(Enum):
    """缺失值处理策略"""
    DROP = "drop"                    # 删除缺失值
    MEAN = "mean"                    # 均值填充
    MEDIAN = "median"                # 中位数填充
    MODE = "mode"                    # 众数填充
    FORWARD_FILL = "forward_fill"    # 前向填充
    BACKWARD_FILL = "backward_fill"  # 后向填充
    INTERPOLATION = "interpolation"  # 插值填充
    KNN = "knn"                      # KNN填充
    ITERATIVE = "iterative"          # 迭代填充
    PREDICTIVE = "predictive"        # 预测填充

class OutlierDetectionMethod(Enum):
    """异常值检测方法"""
    Z_SCORE = "z_score"              # Z-Score方法
    IQR = "iqr"                      # 四分位数方法
    ISOLATION_FOREST = "isolation_forest"  # 孤立森林
    ONE_CLASS_SVM = "one_class_svm"  # 单类SVM
    LOCAL_OUTLIER_FACTOR = "lof"     # 局部异常因子

class ScalingMethod(Enum):
    """数据标准化方法"""
    STANDARD = "standard"            # 标准化
    MINMAX = "minmax"               # 最小最大缩放
    ROBUST = "robust"               # 鲁棒缩放
    NORMALIZE = "normalize"         # 归一化

class EncodingMethod(Enum):
    """编码方法"""
    LABEL = "label"                  # 标签编码
    ONE_HOT = "one_hot"             # 独热编码
    TARGET = "target"               # 目标编码
    FREQUENCY = "frequency"         # 频率编码

@dataclass
class DataQualityReport:
    """数据质量报告"""
    total_rows: int
    total_columns: int
    missing_values: Dict[str, int]
    missing_percentage: Dict[str, float]
    data_types: Dict[str, str]
    duplicate_rows: int
    outlier_counts: Dict[str, int]
    quality_score: float
    recommendations: List[str]
    timestamp: datetime

@dataclass
class PreprocessingConfig:
    """预处理配置"""
    missing_value_strategy: MissingValueStrategy
    outlier_detection_method: OutlierDetectionMethod
    scaling_method: ScalingMethod
    encoding_method: EncodingMethod
    remove_duplicates: bool = True
    handle_outliers: bool = True
    scale_features: bool = True
    encode_categorical: bool = True
    feature_selection: bool = False
    dimensionality_reduction: bool = False
    custom_transformations: Optional[List[Callable]] = None

def make_default_preprocessing_config() -> "PreprocessingConfig":
    """工厂：统一默认预处理配置（DRY）"""
    return PreprocessingConfig(
        missing_value_strategy=MissingValueStrategy.MEAN,
        outlier_detection_method=OutlierDetectionMethod.Z_SCORE,
        scaling_method=ScalingMethod.STANDARD,
        encoding_method=EncodingMethod.ONE_HOT,
        remove_duplicates=True,
        handle_outliers=True,
        scale_features=True,
        encode_categorical=True,
        feature_selection=False,
        dimensionality_reduction=False,
        custom_transformations=None
    )

class DataCleaner:
    """数据清洗器"""

    def __init__(self):
        self.imputers: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.encoders: Dict[str, Any] = {}
        self.outlier_detectors: Dict[str, Any] = {}
        self.quality_reports: List[DataQualityReport] = []

    def analyze_data_quality(self, df: pd.DataFrame) -> DataQualityReport:
        """分析数据质量"""
        total_rows, total_columns = df.shape

        # 缺失值分析
        missing_values = df.isnull().sum().to_dict()
        missing_percentage = (df.isnull().sum() / len(df) * 100).to_dict()

        # 数据类型分析
        data_types = df.dtypes.astype(str).to_dict()

        # 重复行分析
        duplicate_rows = df.duplicated().sum()

        # 异常值分析
        outlier_counts = self._detect_outliers_all_columns(df)

        # 计算质量分数
        quality_score = self._calculate_quality_score(
            missing_percentage, duplicate_rows, total_rows
        )

        # 生成建议
        recommendations = self._generate_recommendations(
            missing_percentage, duplicate_rows, outlier_counts
        )

        report = DataQualityReport(
            total_rows=total_rows,
            total_columns=total_columns,
            missing_values=missing_values,
            missing_percentage=missing_percentage,
            data_types=data_types,
            duplicate_rows=duplicate_rows,
            outlier_counts=outlier_counts,
            quality_score=quality_score,
            recommendations=recommendations,
            timestamp=datetime.now()
        )

        self.quality_reports.append(report)
        return report

    def _detect_outliers_all_columns(self, df: pd.DataFrame) -> Dict[str, int]:
        """检测所有列的异常值"""
        outlier_counts = {}

        for column in df.select_dtypes(include=[np.number]).columns:
            try:
                outliers = self._detect_outliers_zscore(df[column])
                outlier_counts[column] = len(outliers)
            except Exception as e:
                logger.warning(f"检测异常值失败 {column}: {e}")
                outlier_counts[column] = 0

        return outlier_counts

    def _detect_outliers_zscore(self, series: pd.Series, threshold: float = 3.0) -> List[int]:
        """使用Z-Score检测异常值"""
        z_scores = np.abs(stats.zscore(series.dropna()))
        outlier_indices = np.where(z_scores > threshold)[0]
        return outlier_indices.tolist()

    def _calculate_quality_score(self, missing_percentage: Dict[str, float],
                                duplicate_rows: int, total_rows: int) -> float:
        """计算数据质量分数"""
        # 基于缺失值比例
        avg_missing = np.mean(list(missing_percentage.values()))
        missing_score = max(0, 100 - avg_missing)

        # 基于重复行比例
        duplicate_percentage = (duplicate_rows / total_rows) * 100 if total_rows > 0 else 0
        duplicate_score = max(0, 100 - duplicate_percentage)

        # 综合质量分数
        quality_score = (missing_score + duplicate_score) / 2
        return round(quality_score, 2)

    def _generate_recommendations(self, missing_percentage: Dict[str, float],
                                duplicate_rows: int, outlier_counts: Dict[str, int]) -> List[str]:
        """生成数据质量改进建议"""
        recommendations = []

        # 缺失值建议
        high_missing_cols = [col for col, pct in missing_percentage.items() if pct > 20]
        if high_missing_cols:
            recommendations.append(f"列 {', '.join(high_missing_cols)} 缺失值超过20%，建议检查数据收集过程")

        # 重复行建议
        if duplicate_rows > 0:
            recommendations.append(f"发现 {duplicate_rows} 行重复数据，建议删除重复行")

        # 异常值建议
        high_outlier_cols = [col for col, count in outlier_counts.items() if count > 0]
        if high_outlier_cols:
            recommendations.append(f"列 {', '.join(high_outlier_cols)} 存在异常值，建议进行异常值处理")

        return recommendations

    def handle_missing_values(self, df: pd.DataFrame,
                            strategy: MissingValueStrategy = MissingValueStrategy.MEAN,
                            columns: Optional[List[str]] = None) -> pd.DataFrame:
        """处理缺失值"""
        df_cleaned = df.copy()

        if columns is None:
            columns = df_cleaned.columns.tolist()

        for column in columns:
            if df_cleaned[column].isnull().any():
                if strategy == MissingValueStrategy.DROP:
                    df_cleaned = df_cleaned.dropna(subset=[column])

                elif strategy == MissingValueStrategy.MEAN:
                    if df_cleaned[column].dtype in ['int64', 'float64']:
                        df_cleaned[column].fillna(df_cleaned[column].mean(), inplace=True)

                elif strategy == MissingValueStrategy.MEDIAN:
                    if df_cleaned[column].dtype in ['int64', 'float64']:
                        df_cleaned[column].fillna(df_cleaned[column].median(), inplace=True)

                elif strategy == MissingValueStrategy.MODE:
                    mode_value = df_cleaned[column].mode()
                    if not mode_value.empty:
                        df_cleaned[column].fillna(mode_value[0], inplace=True)

                elif strategy == MissingValueStrategy.FORWARD_FILL:
                    df_cleaned[column].fillna(method='ffill', inplace=True)

                elif strategy == MissingValueStrategy.BACKWARD_FILL:
                    df_cleaned[column].fillna(method='bfill', inplace=True)

                elif strategy == MissingValueStrategy.INTERPOLATION:
                    if df_cleaned[column].dtype in ['int64', 'float64']:
                        df_cleaned[column].interpolate(method='linear', inplace=True)

                elif strategy == MissingValueStrategy.KNN:
                    if column not in self.imputers:
                        self.imputers[column] = KNNImputer(n_neighbors=5)
                    df_cleaned[column] = self.imputers[column].fit_transform(df_cleaned[[column]])

                elif strategy == MissingValueStrategy.ITERATIVE:
                    if column not in self.imputers:
                        self.imputers[column] = IterativeImputer(random_state=42)
                    df_cleaned[column] = self.imputers[column].fit_transform(df_cleaned[[column]])

        logger.info(f"缺失值处理完成，使用策略: {strategy.value}")
        return df_cleaned

    def detect_outliers(self, df: pd.DataFrame,
                      method: OutlierDetectionMethod = OutlierDetectionMethod.Z_SCORE,
                      columns: Optional[List[str]] = None) -> Dict[str, List[int]]:
        """检测异常值"""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        outliers = {}

        for column in columns:
            if method == OutlierDetectionMethod.Z_SCORE:
                outliers[column] = self._detect_outliers_zscore(df[column])

            elif method == OutlierDetectionMethod.IQR:
                outliers[column] = self._detect_outliers_iqr(df[column])

            elif method == OutlierDetectionMethod.ISOLATION_FOREST:
                outliers[column] = self._detect_outliers_isolation_forest(df[column])

            elif method == OutlierDetectionMethod.ONE_CLASS_SVM:
                outliers[column] = self._detect_outliers_one_class_svm(df[column])

        return outliers

    def _detect_outliers_iqr(self, series: pd.Series) -> List[int]:
        """使用IQR方法检测异常值"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = series[(series < lower_bound) | (series > upper_bound)].index.tolist()
        return outliers

    def _detect_outliers_isolation_forest(self, series: pd.Series) -> List[int]:
        """使用孤立森林检测异常值"""
        try:
            if len(series.dropna()) < 10:
                return []

            detector = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = detector.fit_predict(series.dropna().values.reshape(-1, 1))
            outlier_indices = series.dropna().index[outlier_labels == -1].tolist()
            return outlier_indices
        except Exception as e:
            logger.warning(f"孤立森林异常值检测失败: {e}")
            return []

    def _detect_outliers_one_class_svm(self, series: pd.Series) -> List[int]:
        """使用单类SVM检测异常值"""
        try:
            if len(series.dropna()) < 10:
                return []

            detector = OneClassSVM(nu=0.1)
            outlier_labels = detector.fit_predict(series.dropna().values.reshape(-1, 1))
            outlier_indices = series.dropna().index[outlier_labels == -1].tolist()
            return outlier_indices
        except Exception as e:
            logger.warning(f"单类SVM异常值检测失败: {e}")
            return []

    def remove_outliers(self, df: pd.DataFrame, outliers: Dict[str, List[int]]) -> pd.DataFrame:
        """移除异常值"""
        df_cleaned = df.copy()

        for column, outlier_indices in outliers.items():
            if outlier_indices:
                df_cleaned = df_cleaned.drop(outlier_indices)
                logger.info(f"从列 {column} 移除了 {len(outlier_indices)} 个异常值")

        return df_cleaned.reset_index(drop=True)

    def scale_features(self, df: pd.DataFrame,
                      method: ScalingMethod = ScalingMethod.STANDARD,
                      columns: Optional[List[str]] = None) -> pd.DataFrame:
        """特征缩放"""
        df_scaled = df.copy()

        if columns is None:
            columns = df_scaled.select_dtypes(include=[np.number]).columns.tolist()

        for column in columns:
            if column not in self.scalers:
                if method == ScalingMethod.STANDARD:
                    self.scalers[column] = StandardScaler()
                elif method == ScalingMethod.MINMAX:
                    self.scalers[column] = MinMaxScaler()
                elif method == ScalingMethod.ROBUST:
                    self.scalers[column] = RobustScaler()

            scaler = self.scalers[column]
            df_scaled[column] = scaler.fit_transform(df_scaled[[column]])

        logger.info(f"特征缩放完成，使用方法: {method.value}")
        return df_scaled

    def encode_categorical_features(self, df: pd.DataFrame,
                                  method: EncodingMethod = EncodingMethod.ONE_HOT,
                                  columns: Optional[List[str]] = None) -> pd.DataFrame:
        """编码分类特征"""
        df_encoded = df.copy()

        if columns is None:
            columns = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()

        for column in columns:
            if column not in self.encoders:
                if method == EncodingMethod.LABEL:
                    self.encoders[column] = LabelEncoder()
                    df_encoded[column] = self.encoders[column].fit_transform(df_encoded[column])

                elif method == EncodingMethod.ONE_HOT:
                    self.encoders[column] = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    encoded_data = self.encoders[column].fit_transform(df_encoded[[column]])
                    encoded_df = pd.DataFrame(encoded_data, columns=[f"{column}_{i}" for i in range(encoded_data.shape[1])])
                    df_encoded = pd.concat([df_encoded.drop(columns=[column]), encoded_df], axis=1)

        logger.info(f"分类特征编码完成，使用方法: {method.value}")
        return df_encoded

    def remove_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
        """移除重复行"""
        df_cleaned = df.drop_duplicates(subset=subset)
        removed_count = len(df) - len(df_cleaned)

        if removed_count > 0:
            logger.info(f"移除了 {removed_count} 行重复数据")

        return df_cleaned.reset_index(drop=True)

    def convert_data_types(self, df: pd.DataFrame,
                          type_mapping: Dict[str, str]) -> pd.DataFrame:
        """转换数据类型"""
        df_converted = df.copy()

        for column, target_type in type_mapping.items():
            if column in df_converted.columns:
                try:
                    if target_type == 'datetime':
                        df_converted[column] = pd.to_datetime(df_converted[column])
                    elif target_type == 'numeric':
                        df_converted[column] = pd.to_numeric(df_converted[column], errors='coerce')
                    elif target_type == 'category':
                        df_converted[column] = df_converted[column].astype('category')
                    else:
                        df_converted[column] = df_converted[column].astype(target_type)

                    logger.info(f"列 {column} 转换为 {target_type}")
                except Exception as e:
                    logger.warning(f"列 {column} 类型转换失败: {e}")

        return df_converted

class DataPreprocessor:
    """数据预处理器"""

    def __init__(self):
        self.cleaner = DataCleaner()
        self.preprocessing_history: List[Dict[str, Any]] = []
        self.logger = logger

    def _log_step(self, step_name: str, start_time: float, extra: Optional[Dict[str, Any]] = None) -> None:
        """记录单步处理耗时与关键信息"""
        duration = round(time.time() - start_time, 4)
        payload = {"step": step_name, "duration_sec": duration}
        if extra:
            payload.update(extra)
        self.logger.info(f"预处理步骤完成: {json.dumps(payload, ensure_ascii=False)}")

    def preprocess_data(self, df: pd.DataFrame,
                       config: PreprocessingConfig) -> Tuple[pd.DataFrame, DataQualityReport]:
        """完整的数据预处理流程"""
        logger.info("开始数据预处理")

        # 记录预处理开始
        preprocessing_record = {
            "timestamp": datetime.now().isoformat(),
            "original_shape": df.shape,
            "config": asdict(config)
        }

        df_processed = df.copy()

        # 1. 数据类型转换
        if config.custom_transformations:
            _t0 = time.time()
            for transform_func in config.custom_transformations:
                df_processed = transform_func(df_processed)
            self._log_step("custom_transformations", _t0, {"count": len(config.custom_transformations)})

        # 2. 移除重复行
        if config.remove_duplicates:
            _t0 = time.time()
            df_processed = self.cleaner.remove_duplicates(df_processed)
            self._log_step("remove_duplicates", _t0, {"rows_after": len(df_processed)})

        # 3. 处理缺失值
        _t0 = time.time()
        df_processed = self.cleaner.handle_missing_values(
            df_processed, config.missing_value_strategy
        )
        self._log_step("handle_missing_values", _t0, {"strategy": config.missing_value_strategy.value})

        # 4. 处理异常值
        if config.handle_outliers:
            _t0 = time.time()
            outliers = self.cleaner.detect_outliers(
                df_processed, config.outlier_detection_method
            )
            df_processed = self.cleaner.remove_outliers(df_processed, outliers)
            self._log_step("handle_outliers", _t0, {"method": config.outlier_detection_method.value})

        # 5. 特征缩放
        if config.scale_features:
            _t0 = time.time()
            df_processed = self.cleaner.scale_features(
                df_processed, config.scaling_method
            )
            self._log_step("scale_features", _t0, {"method": config.scaling_method.value})

        # 6. 编码分类特征
        if config.encode_categorical:
            _t0 = time.time()
            df_processed = self.cleaner.encode_categorical_features(
                df_processed, config.encoding_method
            )
            self._log_step("encode_categorical", _t0, {"method": config.encoding_method.value})

        # 7. 特征选择
        if config.feature_selection:
            _t0 = time.time()
            df_processed = self._select_features(df_processed)
            self._log_step("feature_selection", _t0)

        # 8. 降维
        if config.dimensionality_reduction:
            _t0 = time.time()
            df_processed = self._reduce_dimensions(df_processed)
            self._log_step("dimensionality_reduction", _t0)

        # 生成质量报告
        _t0 = time.time()
        quality_report = self.cleaner.analyze_data_quality(df_processed)
        self._log_step("analyze_data_quality", _t0, {"quality_score": quality_report.quality_score})

        # 记录预处理结果
        preprocessing_record.update({
            "final_shape": df_processed.shape,
            "quality_score": quality_report.quality_score,
            "processing_steps": len(config.custom_transformations or []) + 6
        })

        self.preprocessing_history.append(preprocessing_record)

        logger.info(f"数据预处理完成，质量分数: {quality_report.quality_score}")
        return df_processed, quality_report

    def _select_features(self, df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
        """特征选择"""
        try:
            # 简单的特征选择实现
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_columns) > k:
                selector = SelectKBest(score_func=f_classif, k=k)
                selected_features = selector.fit_transform(df[numeric_columns], np.zeros(len(df)))
                selected_columns = [numeric_columns[i] for i in selector.get_support(indices=True)]
                return df[selected_columns]
            return df
        except Exception as e:
            logger.warning(f"特征选择失败: {e}")
            return df

    def _reduce_dimensions(self, df: pd.DataFrame, n_components: int = 0.95) -> pd.DataFrame:
        """降维"""
        try:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_columns) > 2:
                pca = PCA(n_components=n_components)
                reduced_data = pca.fit_transform(df[numeric_columns])
                reduced_df = pd.DataFrame(reduced_data, columns=[f"PC_{i+1}" for i in range(reduced_data.shape[1])])
                return reduced_df
            return df
        except Exception as e:
            logger.warning(f"降维失败: {e}")
            return df

    def get_preprocessing_history(self) -> List[Dict[str, Any]]:
        """获取预处理历史"""
        return self.preprocessing_history

    def get_quality_reports(self) -> List[DataQualityReport]:
        """获取质量报告"""
        return self.cleaner.quality_reports

# 全局数据预处理器实例
data_preprocessor = DataPreprocessor()

# 异步任务
@async_task("preprocess_dataset", TaskPriority.HIGH)
def preprocess_dataset_task(data: Dict[str, Any], config_dict: Dict[str, Any]):
    """数据预处理任务"""
    # 将字典转换为DataFrame
    df = pd.DataFrame(data)

    # 创建配置
    config = PreprocessingConfig(**config_dict)

    # 执行预处理
    processed_df, quality_report = data_preprocessor.preprocess_data(df, config)

    return {
        "processed_data": processed_df.to_dict(),
        "quality_report": asdict(quality_report),
        "success": True
    }

# 数据预处理API
def preprocess_data(df: pd.DataFrame, config: PreprocessingConfig) -> Tuple[pd.DataFrame, DataQualityReport]:
    """预处理数据"""
    return data_preprocessor.preprocess_data(df, config)

def analyze_data_quality(df: pd.DataFrame) -> DataQualityReport:
    """分析数据质量"""
    return data_preprocessor.cleaner.analyze_data_quality(df)

def get_preprocessing_history() -> List[Dict[str, Any]]:
    """获取预处理历史"""
    return data_preprocessor.get_preprocessing_history()

def get_quality_reports() -> List[DataQualityReport]:
    """获取质量报告"""
    return data_preprocessor.get_quality_reports()

if __name__ == "__main__":
    # 测试数据预处理器
    import numpy as np

    # 创建测试数据
    np.random.seed(42)
    test_data = {
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.normal(50, 10, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000),
        'target': np.random.randint(0, 2, 1000)
    }

    # 添加一些缺失值和异常值
    test_data['feature1'][:50] = np.nan
    test_data['feature2'][100:110] = 1000  # 异常值

    df = pd.DataFrame(test_data)

    # 创建预处理配置
    config = PreprocessingConfig(
        missing_value_strategy=MissingValueStrategy.MEAN,
        outlier_detection_method=OutlierDetectionMethod.Z_SCORE,
        scaling_method=ScalingMethod.STANDARD,
        encoding_method=EncodingMethod.ONE_HOT,
        remove_duplicates=True,
        handle_outliers=True,
        scale_features=True,
        encode_categorical=True
    )

    # 执行预处理
    processed_df, quality_report = preprocess_data(df, config)

    print("预处理完成:")
    print(f"原始数据形状: {df.shape}")
    print(f"处理后形状: {processed_df.shape}")
    print(f"质量分数: {quality_report.quality_score}")
    print(f"建议: {quality_report.recommendations}")

__all__ = ["'logger'", "'MissingValueStrategy'", "'OutlierDetectionMethod'", "'ScalingMethod'", "'EncodingMethod'", "'DataQualityReport'", "'PreprocessingConfig'", "'make_default_preprocessing_config'", "'DataCleaner'", "'DataPreprocessor'", "'data_preprocessor'", "'preprocess_dataset_task'", "'preprocess_data'", "'analyze_data_quality'", "'get_preprocessing_history'", "'get_quality_reports'"]
