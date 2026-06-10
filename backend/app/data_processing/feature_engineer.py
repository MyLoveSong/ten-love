

"""
学术级特征工程框架
支持特征选择、特征构造、特征转换、特征交互等
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import json
from datetime import datetime
import warnings
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, RFE, RFECV,
    f_classif, f_regression, mutual_info_classif, mutual_info_regression,
    chi2, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mutual_info_score
from scipy import stats
from scipy.stats import skew, kurtosis
import asyncio

from backend.app.core.exceptions import CustomException, ValidationError
from app.core.task_queue import async_task, TaskPriority

logger = logging.getLogger(__name__)

class FeatureSelectionMethod(Enum):
    """特征选择方法"""
    UNIVARIATE = "univariate"        # 单变量选择
    RECURSIVE = "recursive"          # 递归特征消除
    MODEL_BASED = "model_based"      # 基于模型的选择
    CORRELATION = "correlation"      # 相关性选择
    MUTUAL_INFO = "mutual_info"      # 互信息选择
    VARIANCE = "variance"            # 方差选择

class FeatureConstructionMethod(Enum):
    """特征构造方法"""
    POLYNOMIAL = "polynomial"        # 多项式特征
    INTERACTION = "interaction"      # 交互特征
    AGGREGATION = "aggregation"      # 聚合特征
    TIME_SERIES = "time_series"     # 时间序列特征
    TEXT_FEATURES = "text_features"  # 文本特征
    DOMAIN_SPECIFIC = "domain_specific"  # 领域特定特征

class FeatureTransformationMethod(Enum):
    """特征转换方法"""
    LOG = "log"                      # 对数转换
    SQRT = "sqrt"                    # 平方根转换
    BOX_COX = "box_cox"              # Box-Cox转换
    YEO_JOHNSON = "yeo_johnson"      # Yeo-Johnson转换
    QUANTILE = "quantile"            # 分位数转换
    POWER = "power"                  # 幂转换

class DimensionalityReductionMethod(Enum):
    """降维方法"""
    PCA = "pca"                      # 主成分分析
    ICA = "ica"                      # 独立成分分析
    SVD = "svd"                      # 奇异值分解
    TSNE = "tsne"                    # t-SNE
    UMAP = "umap"                    # UMAP

@dataclass
class FeatureImportance:
    """特征重要性"""
    feature_name: str
    importance_score: float
    method: str
    rank: int

@dataclass
class FeatureEngineeringReport:
    """特征工程报告"""
    original_features: int
    selected_features: int
    constructed_features: int
    final_features: int
    feature_importance: List[FeatureImportance]
    selection_method: str
    construction_methods: List[str]
    transformation_methods: List[str]
    performance_metrics: Dict[str, float]
    timestamp: datetime

class FeatureSelector:
    """特征选择器"""

    def __init__(self):
        self.selection_history: List[Dict[str, Any]] = []
        self.feature_importance_cache: Dict[str, List[FeatureImportance]] = {}

    def select_features(self, X: pd.DataFrame, y: pd.Series,
                       method: FeatureSelectionMethod = FeatureSelectionMethod.UNIVARIATE,
                       k: int = 10, threshold: float = 0.01) -> Tuple[pd.DataFrame, List[FeatureImportance]]:
        """特征选择"""
        logger.info(f"开始特征选择，方法: {method.value}")

        if method == FeatureSelectionMethod.UNIVARIATE:
            return self._univariate_selection(X, y, k)
        elif method == FeatureSelectionMethod.RECURSIVE:
            return self._recursive_selection(X, y, k)
        elif method == FeatureSelectionMethod.MODEL_BASED:
            return self._model_based_selection(X, y, threshold)
        elif method == FeatureSelectionMethod.CORRELATION:
            return self._correlation_selection(X, y, threshold)
        elif method == FeatureSelectionMethod.MUTUAL_INFO:
            return self._mutual_info_selection(X, y, k)
        elif method == FeatureSelectionMethod.VARIANCE:
            return self._variance_selection(X, threshold)
        else:
            raise CustomException(f"不支持的特征选择方法: {method}")

    def _univariate_selection(self, X: pd.DataFrame, y: pd.Series, k: int) -> Tuple[pd.DataFrame, List[FeatureImportance]]:
        """单变量特征选择"""
        try:
            # 根据目标变量类型选择评分函数
            if y.dtype == 'object' or len(y.unique()) < 10:
                score_func = f_classif
            else:
                score_func = f_regression

            selector = SelectKBest(score_func=score_func, k=min(k, X.shape[1]))
            X_selected = selector.fit_transform(X, y)

            # 获取选择的特征
            selected_features = X.columns[selector.get_support()].tolist()
            scores = selector.scores_

            # 创建特征重要性列表
            feature_importance = []
            for i, feature in enumerate(selected_features):
                feature_importance.append(FeatureImportance(
                    feature_name=feature,
                    importance_score=scores[i],
                    method="univariate",
                    rank=i+1
                ))

            X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)

            logger.info(f"单变量选择完成，选择了 {len(selected_features)} 个特征")
            return X_selected_df, feature_importance

        except Exception as e:
            logger.error(f"单变量选择失败: {e}")
            return X, []

    def _recursive_selection(self, X: pd.DataFrame, y: pd.Series, k: int) -> Tuple[pd.DataFrame, List[FeatureImportance]]:
        """递归特征消除"""
        try:
            # 根据目标变量类型选择估计器
            if y.dtype == 'object' or len(y.unique()) < 10:
                estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                estimator = RandomForestRegressor(n_estimators=100, random_state=42)

            selector = RFE(estimator=estimator, n_features_to_select=min(k, X.shape[1]))
            X_selected = selector.fit_transform(X, y)

            # 获取选择的特征
            selected_features = X.columns[selector.get_support()].tolist()

            # 创建特征重要性列表
            feature_importance = []
            for i, feature in enumerate(selected_features):
                feature_importance.append(FeatureImportance(
                    feature_name=feature,
                    importance_score=selector.ranking_[i],
                    method="recursive",
                    rank=i+1
                ))

            X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)

            logger.info(f"递归特征消除完成，选择了 {len(selected_features)} 个特征")
            return X_selected_df, feature_importance

        except Exception as e:
            logger.error(f"递归特征消除失败: {e}")
            return X, []

    def _model_based_selection(self, X: pd.DataFrame, y: pd.Series, threshold: float) -> Tuple[pd.DataFrame, List[FeatureImportance]]:
        """基于模型的特征选择"""
        try:
            # 根据目标变量类型选择模型
            if y.dtype == 'object' or len(y.unique()) < 10:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)

            selector = SelectFromModel(model, threshold=threshold)
            X_selected = selector.fit_transform(X, y)

            # 获取选择的特征
            selected_features = X.columns[selector.get_support()].tolist()
            importances = model.feature_importances_

            # 创建特征重要性列表
            feature_importance = []
            for i, feature in enumerate(selected_features):
                feature_importance.append(FeatureImportance(
                    feature_name=feature,
                    importance_score=importances[i],
                    method="model_based",
                    rank=i+1
                ))

            X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)

            logger.info(f"基于模型的选择完成，选择了 {len(selected_features)} 个特征")
            return X_selected_df, feature_importance

        except Exception as e:
            logger.error(f"基于模型的选择失败: {e}")
            return X, []

    def _correlation_selection(self, X: pd.DataFrame, y: pd.Series, threshold: float) -> Tuple[pd.DataFrame, List[FeatureImportance]]:
        """相关性特征选择"""
        try:
            # 计算与目标变量的相关性
            correlations = X.corrwith(y).abs()

            # 选择相关性高于阈值的特征
            selected_features = correlations[correlations > threshold].index.tolist()

            # 创建特征重要性列表
            feature_importance = []
            for i, feature in enumerate(selected_features):
                feature_importance.append(FeatureImportance(
                    feature_name=feature,
                    importance_score=correlations[feature],
                    method="correlation",
                    rank=i+1
                ))

            X_selected = X[selected_features]

            logger.info(f"相关性选择完成，选择了 {len(selected_features)} 个特征")
            return X_selected, feature_importance

        except Exception as e:
            logger.error(f"相关性选择失败: {e}")
            return X, []

    def _mutual_info_selection(self, X: pd.DataFrame, y: pd.Series, k: int) -> Tuple[pd.DataFrame, List[FeatureImportance]]:
        """互信息特征选择"""
        try:
            # 根据目标变量类型选择评分函数
            if y.dtype == 'object' or len(y.unique()) < 10:
                score_func = mutual_info_classif
            else:
                score_func = mutual_info_regression

            selector = SelectKBest(score_func=score_func, k=min(k, X.shape[1]))
            X_selected = selector.fit_transform(X, y)

            # 获取选择的特征
            selected_features = X.columns[selector.get_support()].tolist()
            scores = selector.scores_

            # 创建特征重要性列表
            feature_importance = []
            for i, feature in enumerate(selected_features):
                feature_importance.append(FeatureImportance(
                    feature_name=feature,
                    importance_score=scores[i],
                    method="mutual_info",
                    rank=i+1
                ))

            X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)

            logger.info(f"互信息选择完成，选择了 {len(selected_features)} 个特征")
            return X_selected_df, feature_importance

        except Exception as e:
            logger.error(f"互信息选择失败: {e}")
            return X, []

    def _variance_selection(self, X: pd.DataFrame, threshold: float) -> Tuple[pd.DataFrame, List[FeatureImportance]]:
        """方差特征选择"""
        try:
            # 计算特征方差
            variances = X.var()

            # 选择方差高于阈值的特征
            selected_features = variances[variances > threshold].index.tolist()

            # 创建特征重要性列表
            feature_importance = []
            for i, feature in enumerate(selected_features):
                feature_importance.append(FeatureImportance(
                    feature_name=feature,
                    importance_score=variances[feature],
                    method="variance",
                    rank=i+1
                ))

            X_selected = X[selected_features]

            logger.info(f"方差选择完成，选择了 {len(selected_features)} 个特征")
            return X_selected, feature_importance

        except Exception as e:
            logger.error(f"方差选择失败: {e}")
            return X, []

class FeatureConstructor:
    """特征构造器"""

    def __init__(self):
        self.construction_history: List[Dict[str, Any]] = []

    def construct_features(self, df: pd.DataFrame,
                          methods: List[FeatureConstructionMethod],
                          target_column: Optional[str] = None) -> pd.DataFrame:
        """构造特征"""
        logger.info(f"开始特征构造，方法: {[m.value for m in methods]}")

        df_constructed = df.copy()
        constructed_features = []

        for method in methods:
            if method == FeatureConstructionMethod.POLYNOMIAL:
                df_constructed = self._create_polynomial_features(df_constructed)
                constructed_features.append("polynomial")

            elif method == FeatureConstructionMethod.INTERACTION:
                df_constructed = self._create_interaction_features(df_constructed)
                constructed_features.append("interaction")

            elif method == FeatureConstructionMethod.AGGREGATION:
                df_constructed = self._create_aggregation_features(df_constructed)
                constructed_features.append("aggregation")

            elif method == FeatureConstructionMethod.TIME_SERIES:
                df_constructed = self._create_time_series_features(df_constructed)
                constructed_features.append("time_series")

            elif method == FeatureConstructionMethod.TEXT_FEATURES:
                df_constructed = self._create_text_features(df_constructed)
                constructed_features.append("text_features")

            elif method == FeatureConstructionMethod.DOMAIN_SPECIFIC:
                df_constructed = self._create_domain_specific_features(df_constructed, target_column)
                constructed_features.append("domain_specific")

        # 记录构造历史
        self.construction_history.append({
            "timestamp": datetime.now().isoformat(),
            "original_features": len(df.columns),
            "constructed_features": len(df_constructed.columns) - len(df.columns),
            "methods": constructed_features
        })

        logger.info(f"特征构造完成，新增 {len(df_constructed.columns) - len(df.columns)} 个特征")
        return df_constructed

    def _create_polynomial_features(self, df: pd.DataFrame, degree: int = 2) -> pd.DataFrame:
        """创建多项式特征"""
        try:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_columns) < 2:
                return df

            # 选择前几个数值特征进行多项式扩展
            selected_features = numeric_columns[:min(5, len(numeric_columns))]

            poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=True)
            poly_features = poly.fit_transform(df[selected_features])

            # 创建特征名称
            feature_names = poly.get_feature_names_out(selected_features)

            # 创建新的DataFrame
            poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df.index)

            # 合并到原DataFrame
            df_result = pd.concat([df.drop(columns=selected_features), poly_df], axis=1)

            logger.info(f"多项式特征构造完成，新增 {len(feature_names)} 个特征")
            return df_result

        except Exception as e:
            logger.warning(f"多项式特征构造失败: {e}")
            return df

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建交互特征"""
        try:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_columns) < 2:
                return df

            df_result = df.copy()

            # 创建两两交互特征
            for i, col1 in enumerate(numeric_columns[:5]):  # 限制特征数量
                for col2 in numeric_columns[i+1:6]:
                    interaction_name = f"{col1}_x_{col2}"
                    df_result[interaction_name] = df[col1] * df[col2]

            logger.info("交互特征构造完成")
            return df_result

        except Exception as e:
            logger.warning(f"交互特征构造失败: {e}")
            return df

    def _create_aggregation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建聚合特征"""
        try:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_columns) < 2:
                return df

            df_result = df.copy()

            # 创建聚合特征
            df_result['sum_features'] = df[numeric_columns].sum(axis=1)
            df_result['mean_features'] = df[numeric_columns].mean(axis=1)
            df_result['std_features'] = df[numeric_columns].std(axis=1)
            df_result['max_features'] = df[numeric_columns].max(axis=1)
            df_result['min_features'] = df[numeric_columns].min(axis=1)

            logger.info("聚合特征构造完成")
            return df_result

        except Exception as e:
            logger.warning(f"聚合特征构造失败: {e}")
            return df

    def _create_time_series_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建时间序列特征"""
        try:
            df_result = df.copy()

            # 查找时间列
            time_columns = []
            for col in df.columns:
                if df[col].dtype == 'datetime64[ns]' or 'time' in col.lower() or 'date' in col.lower():
                    time_columns.append(col)

            if not time_columns:
                logger.info("未找到时间列，跳过时间序列特征构造")
                return df

            # 为每个时间列创建特征
            for time_col in time_columns:
                if df[time_col].dtype == 'datetime64[ns]':
                    df_result[f'{time_col}_year'] = df[time_col].dt.year
                    df_result[f'{time_col}_month'] = df[time_col].dt.month
                    df_result[f'{time_col}_day'] = df[time_col].dt.day
                    df_result[f'{time_col}_weekday'] = df[time_col].dt.weekday
                    df_result[f'{time_col}_hour'] = df[time_col].dt.hour
                    df_result[f'{time_col}_quarter'] = df[time_col].dt.quarter

            logger.info("时间序列特征构造完成")
            return df_result

        except Exception as e:
            logger.warning(f"时间序列特征构造失败: {e}")
            return df

    def _create_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建文本特征"""
        try:
            df_result = df.copy()

            # 查找文本列
            text_columns = df.select_dtypes(include=['object']).columns.tolist()

            for text_col in text_columns:
                if df[text_col].dtype == 'object':
                    # 创建文本长度特征
                    df_result[f'{text_col}_length'] = df[text_col].astype(str).str.len()

                    # 创建单词数量特征
                    df_result[f'{text_col}_word_count'] = df[text_col].astype(str).str.split().str.len()

                    # 创建大写字母数量特征
                    df_result[f'{text_col}_upper_count'] = df[text_col].astype(str).str.count(r'[A-Z]')

                    # 创建数字数量特征
                    df_result[f'{text_col}_digit_count'] = df[text_col].astype(str).str.count(r'\d')

            logger.info("文本特征构造完成")
            return df_result

        except Exception as e:
            logger.warning(f"文本特征构造失败: {e}")
            return df

    def _create_domain_specific_features(self, df: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
        """创建领域特定特征"""
        try:
            df_result = df.copy()

            # 血糖预测领域特定特征
            if 'glucose' in df.columns or 'blood_sugar' in df.columns:
                glucose_col = 'glucose' if 'glucose' in df.columns else 'blood_sugar'

                # 血糖变化率
                if len(df) > 1:
                    df_result[f'{glucose_col}_change'] = df[glucose_col].diff()
                    df_result[f'{glucose_col}_change_rate'] = df[glucose_col].pct_change()

                # 血糖统计特征
                if len(df) > 10:
                    df_result[f'{glucose_col}_rolling_mean'] = df[glucose_col].rolling(window=5).mean()
                    df_result[f'{glucose_col}_rolling_std'] = df[glucose_col].rolling(window=5).std()

            # 用户行为领域特定特征
            if 'user_id' in df.columns:
                # 用户行为统计
                user_stats = df.groupby('user_id').agg({
                    col: ['count', 'mean', 'std'] for col in df.select_dtypes(include=[np.number]).columns
                }).reset_index()

                # 扁平化列名
                user_stats.columns = ['_'.join(col).strip() for col in user_stats.columns]
                user_stats = user_stats.rename(columns={'user_id_': 'user_id'})

                # 合并到原DataFrame
                df_result = df_result.merge(user_stats, on='user_id', how='left')

            logger.info("领域特定特征构造完成")
            return df_result

        except Exception as e:
            logger.warning(f"领域特定特征构造失败: {e}")
            return df

class FeatureTransformer:
    """特征转换器"""

    def __init__(self):
        self.transformation_history: List[Dict[str, Any]] = []

    def transform_features(self, df: pd.DataFrame,
                          methods: List[FeatureTransformationMethod],
                          columns: Optional[List[str]] = None) -> pd.DataFrame:
        """转换特征"""
        logger.info(f"开始特征转换，方法: {[m.value for m in methods]}")

        df_transformed = df.copy()

        if columns is None:
            columns = df_transformed.select_dtypes(include=[np.number]).columns.tolist()

        for method in methods:
            if method == FeatureTransformationMethod.LOG:
                df_transformed = self._log_transform(df_transformed, columns)
            elif method == FeatureTransformationMethod.SQRT:
                df_transformed = self._sqrt_transform(df_transformed, columns)
            elif method == FeatureTransformationMethod.BOX_COX:
                df_transformed = self._box_cox_transform(df_transformed, columns)
            elif method == FeatureTransformationMethod.YEO_JOHNSON:
                df_transformed = self._yeo_johnson_transform(df_transformed, columns)
            elif method == FeatureTransformationMethod.QUANTILE:
                df_transformed = self._quantile_transform(df_transformed, columns)
            elif method == FeatureTransformationMethod.POWER:
                df_transformed = self._power_transform(df_transformed, columns)

        # 记录转换历史
        self.transformation_history.append({
            "timestamp": datetime.now().isoformat(),
            "methods": [m.value for m in methods],
            "columns": columns
        })

        logger.info("特征转换完成")
        return df_transformed

    def _log_transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """对数转换"""
        df_result = df.copy()

        for col in columns:
            if col in df_result.columns and df_result[col].min() > 0:
                df_result[f'{col}_log'] = np.log1p(df_result[col])

        return df_result

    def _sqrt_transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """平方根转换"""
        df_result = df.copy()

        for col in columns:
            if col in df_result.columns and df_result[col].min() >= 0:
                df_result[f'{col}_sqrt'] = np.sqrt(df_result[col])

        return df_result

    def _box_cox_transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Box-Cox转换"""
        df_result = df.copy()

        for col in columns:
            if col in df_result.columns and df_result[col].min() > 0:
                try:
                    from scipy.stats import boxcox
                    transformed, _ = boxcox(df_result[col])
                    df_result[f'{col}_boxcox'] = transformed
                except Exception as e:
                    logger.warning(f"Box-Cox转换失败 {col}: {e}")

        return df_result

    def _yeo_johnson_transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Yeo-Johnson转换"""
        df_result = df.copy()

        for col in columns:
            if col in df_result.columns:
                try:
                    from scipy.stats import yeojohnson
                    transformed, _ = yeojohnson(df_result[col])
                    df_result[f'{col}_yeojohnson'] = transformed
                except Exception as e:
                    logger.warning(f"Yeo-Johnson转换失败 {col}: {e}")

        return df_result

    def _quantile_transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """分位数转换"""
        df_result = df.copy()

        for col in columns:
            if col in df_result.columns:
                try:
                    from sklearn.preprocessing import QuantileTransformer
                    transformer = QuantileTransformer(output_distribution='normal')
                    df_result[f'{col}_quantile'] = transformer.fit_transform(df_result[[col]])
                except Exception as e:
                    logger.warning(f"分位数转换失败 {col}: {e}")

        return df_result

    def _power_transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """幂转换"""
        df_result = df.copy()

        for col in columns:
            if col in df_result.columns and df_result[col].min() > 0:
                df_result[f'{col}_power2'] = df_result[col] ** 2
                df_result[f'{col}_power3'] = df_result[col] ** 3

        return df_result

class DimensionalityReducer:
    """降维器"""

    def __init__(self):
        self.reduction_history: List[Dict[str, Any]] = []

    def reduce_dimensions(self, df: pd.DataFrame,
                         method: DimensionalityReductionMethod = DimensionalityReductionMethod.PCA,
                         n_components: Union[int, float] = 0.95) -> pd.DataFrame:
        """降维"""
        logger.info(f"开始降维，方法: {method.value}")

        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_columns) < 2:
            logger.info("数值特征不足，跳过降维")
            return df

        X = df[numeric_columns]

        try:
            if method == DimensionalityReductionMethod.PCA:
                reducer = PCA(n_components=n_components)
                X_reduced = reducer.fit_transform(X)

                # 创建新的特征名称
                feature_names = [f"PC_{i+1}" for i in range(X_reduced.shape[1])]

            elif method == DimensionalityReductionMethod.ICA:
                reducer = FastICA(n_components=n_components, random_state=42)
                X_reduced = reducer.fit_transform(X)
                feature_names = [f"IC_{i+1}" for i in range(X_reduced.shape[1])]

            elif method == DimensionalityReductionMethod.SVD:
                reducer = TruncatedSVD(n_components=n_components, random_state=42)
                X_reduced = reducer.fit_transform(X)
                feature_names = [f"SVD_{i+1}" for i in range(X_reduced.shape[1])]

            elif method == DimensionalityReductionMethod.TSNE:
                reducer = TSNE(n_components=min(2, len(numeric_columns)-1), random_state=42)
                X_reduced = reducer.fit_transform(X)
                feature_names = [f"TSNE_{i+1}" for i in range(X_reduced.shape[1])]

            else:
                raise CustomException(f"不支持的降维方法: {method}")

            # 创建降维后的DataFrame
            reduced_df = pd.DataFrame(X_reduced, columns=feature_names, index=df.index)

            # 合并非数值特征
            non_numeric_df = df.drop(columns=numeric_columns)
            result_df = pd.concat([reduced_df, non_numeric_df], axis=1)

            # 记录降维历史
            self.reduction_history.append({
                "timestamp": datetime.now().isoformat(),
                "method": method.value,
                "original_dimensions": len(numeric_columns),
                "reduced_dimensions": X_reduced.shape[1],
                "explained_variance": getattr(reducer, 'explained_variance_ratio_', None)
            })

            logger.info(f"降维完成，从 {len(numeric_columns)} 维降到 {X_reduced.shape[1]} 维")
            return result_df

        except Exception as e:
            logger.error(f"降维失败: {e}")
            return df

class FeatureEngineer:
    """特征工程主类"""

    def __init__(self):
        self.selector = FeatureSelector()
        self.constructor = FeatureConstructor()
        self.transformer = FeatureTransformer()
        self.reducer = DimensionalityReducer()
        self.engineering_history: List[FeatureEngineeringReport] = []

    def engineer_features(self, df: pd.DataFrame, y: Optional[pd.Series] = None,
                         selection_method: Optional[FeatureSelectionMethod] = None,
                         construction_methods: Optional[List[FeatureConstructionMethod]] = None,
                         transformation_methods: Optional[List[FeatureTransformationMethod]] = None,
                         reduction_method: Optional[DimensionalityReductionMethod] = None,
                         selection_k: int = 10) -> Tuple[pd.DataFrame, FeatureEngineeringReport]:
        """完整的特征工程流程"""
        logger.info("开始特征工程")

        original_features = len(df.columns)
        df_engineered = df.copy()

        # 1. 特征构造
        if construction_methods:
            df_engineered = self.constructor.construct_features(df_engineered, construction_methods)

        constructed_features = len(df_engineered.columns) - original_features

        # 2. 特征转换
        if transformation_methods:
            df_engineered = self.transformer.transform_features(df_engineered, transformation_methods)

        # 3. 特征选择
        feature_importance = []
        if selection_method and y is not None:
            df_engineered, feature_importance = self.selector.select_features(
                df_engineered, y, selection_method, selection_k
            )

        selected_features = len(df_engineered.columns)

        # 4. 降维
        if reduction_method:
            df_engineered = self.reducer.reduce_dimensions(df_engineered, reduction_method)

        final_features = len(df_engineered.columns)

        # 生成特征工程报告
        report = FeatureEngineeringReport(
            original_features=original_features,
            selected_features=selected_features,
            constructed_features=constructed_features,
            final_features=final_features,
            feature_importance=feature_importance,
            selection_method=selection_method.value if selection_method else "none",
            construction_methods=[m.value for m in construction_methods] if construction_methods else [],
            transformation_methods=[m.value for m in transformation_methods] if transformation_methods else [],
            performance_metrics={},  # 可以添加性能指标
            timestamp=datetime.now()
        )

        self.engineering_history.append(report)

        logger.info(f"特征工程完成，最终特征数: {final_features}")
        return df_engineered, report

# 全局特征工程器实例
feature_engineer = FeatureEngineer()

# 异步任务
@async_task("engineer_features", TaskPriority.HIGH)
def engineer_features_task(data: Dict[str, Any], target: Optional[List[Any]] = None, config: Dict[str, Any] = None):
    """特征工程任务"""
    # 将字典转换为DataFrame
    df = pd.DataFrame(data)
    y = pd.Series(target) if target else None

    # 解析配置
    selection_method = FeatureSelectionMethod(config.get('selection_method')) if config.get('selection_method') else None
    construction_methods = [FeatureConstructionMethod(m) for m in config.get('construction_methods', [])]
    transformation_methods = [FeatureTransformationMethod(m) for m in config.get('transformation_methods', [])]
    reduction_method = DimensionalityReductionMethod(config.get('reduction_method')) if config.get('reduction_method') else None

    # 执行特征工程
    engineered_df, report = feature_engineer.engineer_features(
        df, y, selection_method, construction_methods, transformation_methods, reduction_method
    )

    return {
        "engineered_data": engineered_df.to_dict(),
        "report": asdict(report),
        "success": True
    }

# 特征工程API
def engineer_features(df: pd.DataFrame, y: Optional[pd.Series] = None,
                     selection_method: Optional[FeatureSelectionMethod] = None,
                     construction_methods: Optional[List[FeatureConstructionMethod]] = None,
                     transformation_methods: Optional[List[FeatureTransformationMethod]] = None,
                     reduction_method: Optional[DimensionalityReductionMethod] = None) -> Tuple[pd.DataFrame, FeatureEngineeringReport]:
    """特征工程"""
    return feature_engineer.engineer_features(
        df, y, selection_method, construction_methods, transformation_methods, reduction_method
    )

def get_engineering_history() -> List[FeatureEngineeringReport]:
    """获取特征工程历史"""
    return feature_engineer.engineering_history

if __name__ == "__main__":
    # 测试特征工程
    import numpy as np

    # 创建测试数据
    np.random.seed(42)
    test_data = {
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.normal(50, 10, 1000),
        'feature3': np.random.normal(20, 5, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000),
        'timestamp': pd.date_range('2023-01-01', periods=1000, freq='H')
    }

    df = pd.DataFrame(test_data)
    y = pd.Series(np.random.randint(0, 2, 1000))

    # 执行特征工程
    engineered_df, report = engineer_features(
        df, y,
        selection_method=FeatureSelectionMethod.UNIVARIATE,
        construction_methods=[FeatureConstructionMethod.POLYNOMIAL, FeatureConstructionMethod.INTERACTION],
        transformation_methods=[FeatureTransformationMethod.LOG],
        reduction_method=DimensionalityReductionMethod.PCA
    )

    print("特征工程完成:")
    print(f"原始特征数: {report.original_features}")
    print(f"构造特征数: {report.constructed_features}")
    print(f"最终特征数: {report.final_features}")
    print(f"选择方法: {report.selection_method}")
    print(f"构造方法: {report.construction_methods}")
    print(f"转换方法: {report.transformation_methods}")

__all__ = ["'logger'", "'FeatureSelectionMethod'", "'FeatureConstructionMethod'", "'FeatureTransformationMethod'", "'DimensionalityReductionMethod'", "'FeatureImportance'", "'FeatureEngineeringReport'", "'FeatureSelector'", "'FeatureConstructor'", "'FeatureTransformer'", "'DimensionalityReducer'", "'FeatureEngineer'", "'feature_engineer'", "'engineer_features_task'", "'engineer_features'", "'get_engineering_history'"]
