

"""
学术级数据质量监控模块
支持实时数据质量检测、异常监控、质量报告生成
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import json
from datetime import datetime, timedelta
import warnings
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats
import asyncio
import threading
import time

from backend.app.core.exceptions import CustomException, ValidationError
from app.core.task_queue import async_task, TaskPriority

logger = logging.getLogger(__name__)

class QualityMetric(Enum):
    """质量指标"""
    COMPLETENESS = "completeness"      # 完整性
    ACCURACY = "accuracy"              # 准确性
    CONSISTENCY = "consistency"        # 一致性
    VALIDITY = "validity"              # 有效性
    UNIQUENESS = "uniqueness"          # 唯一性
    TIMELINESS = "timeliness"          # 及时性
    RELEVANCE = "relevance"            # 相关性

class AnomalyType(Enum):
    """异常类型"""
    STATISTICAL = "statistical"        # 统计异常
    PATTERN = "pattern"                # 模式异常
    DISTRIBUTION = "distribution"      # 分布异常
    CORRELATION = "correlation"        # 相关性异常
    TEMPORAL = "temporal"              # 时间异常

class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"                      # 信息
    WARNING = "warning"                # 警告
    ERROR = "error"                    # 错误
    CRITICAL = "critical"              # 严重

@dataclass
class QualityThreshold:
    """质量阈值"""
    metric: QualityMetric
    threshold_value: float
    operator: str  # ">", "<", ">=", "<=", "==", "!="
    alert_level: AlertLevel
    description: str

@dataclass
class QualityAlert:
    """质量告警"""
    alert_id: str
    metric: QualityMetric
    current_value: float
    threshold_value: float
    alert_level: AlertLevel
    message: str
    timestamp: datetime
    data_source: str
    resolution_status: str = "open"
    resolved_at: Optional[datetime] = None

@dataclass
class QualityReport:
    """质量报告"""
    report_id: str
    data_source: str
    total_records: int
    quality_score: float
    metric_scores: Dict[str, float]
    alerts: List[QualityAlert]
    recommendations: List[str]
    generated_at: datetime
    time_range: Tuple[datetime, datetime]

class DataQualityMonitor:
    """数据质量监控器"""

    def __init__(self):
        self.quality_thresholds: List[QualityThreshold] = []
        self.active_alerts: List[QualityAlert] = []
        self.quality_history: List[QualityReport] = []
        self.monitoring_enabled: bool = True
        self.monitoring_thread: Optional[threading.Thread] = None
        self.anomaly_detectors: Dict[str, Any] = {}

        # 初始化默认阈值
        self._init_default_thresholds()

        # 启动监控线程
        self._start_monitoring()

    def _init_default_thresholds(self):
        """初始化默认质量阈值"""
        default_thresholds = [
            QualityThreshold(
                metric=QualityMetric.COMPLETENESS,
                threshold_value=0.95,
                operator=">=",
                alert_level=AlertLevel.WARNING,
                description="数据完整性应不低于95%"
            ),
            QualityThreshold(
                metric=QualityMetric.ACCURACY,
                threshold_value=0.90,
                operator=">=",
                alert_level=AlertLevel.WARNING,
                description="数据准确性应不低于90%"
            ),
            QualityThreshold(
                metric=QualityMetric.CONSISTENCY,
                threshold_value=0.85,
                operator=">=",
                alert_level=AlertLevel.WARNING,
                description="数据一致性应不低于85%"
            ),
            QualityThreshold(
                metric=QualityMetric.VALIDITY,
                threshold_value=0.90,
                operator=">=",
                alert_level=AlertLevel.WARNING,
                description="数据有效性应不低于90%"
            ),
            QualityThreshold(
                metric=QualityMetric.UNIQUENESS,
                threshold_value=0.95,
                operator=">=",
                alert_level=AlertLevel.INFO,
                description="数据唯一性应不低于95%"
            )
        ]

        self.quality_thresholds.extend(default_thresholds)

    def _start_monitoring(self):
        """启动监控线程"""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            return

        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("数据质量监控线程已启动")

    def _monitoring_loop(self):
        """监控循环"""
        while self.monitoring_enabled:
            try:
                # 检查活跃告警
                self._check_active_alerts()

                # 清理过期告警
                self._cleanup_expired_alerts()

                # 休眠一段时间
                time.sleep(60)  # 每分钟检查一次

            except Exception as e:
                logger.error(f"监控循环异常: {e}")
                time.sleep(60)

    def _check_active_alerts(self):
        """检查活跃告警"""
        current_time = datetime.now()

        for alert in self.active_alerts:
            if alert.resolution_status == "open":
                # 检查告警是否过期（24小时）
                if (current_time - alert.timestamp).total_seconds() > 86400:
                    alert.resolution_status = "expired"
                    alert.resolved_at = current_time
                    logger.warning(f"告警 {alert.alert_id} 已过期")

    def _cleanup_expired_alerts(self):
        """清理过期告警"""
        current_time = datetime.now()

        # 移除30天前的已解决告警
        self.active_alerts = [
            alert for alert in self.active_alerts
            if not (alert.resolution_status == "resolved" and
                   alert.resolved_at and
                   (current_time - alert.resolved_at).total_seconds() > 2592000)
        ]

    def add_quality_threshold(self, threshold: QualityThreshold):
        """添加质量阈值"""
        self.quality_thresholds.append(threshold)
        logger.info(f"添加质量阈值: {threshold.metric.value}")

    def remove_quality_threshold(self, metric: QualityMetric):
        """移除质量阈值"""
        self.quality_thresholds = [
            t for t in self.quality_thresholds if t.metric != metric
        ]
        logger.info(f"移除质量阈值: {metric.value}")

    def check_data_quality(self, df: pd.DataFrame, data_source: str = "unknown") -> QualityReport:
        """检查数据质量"""
        logger.info(f"开始检查数据质量，数据源: {data_source}")

        # 计算质量指标
        metric_scores = self._calculate_quality_metrics(df)

        # 计算总体质量分数
        quality_score = self._calculate_overall_quality_score(metric_scores)

        # 检查阈值并生成告警
        alerts = self._check_thresholds(metric_scores, data_source)

        # 生成建议
        recommendations = self._generate_recommendations(metric_scores, df)

        # 创建质量报告
        report = QualityReport(
            report_id=f"qr_{int(time.time())}",
            data_source=data_source,
            total_records=len(df),
            quality_score=quality_score,
            metric_scores=metric_scores,
            alerts=alerts,
            recommendations=recommendations,
            generated_at=datetime.now(),
            time_range=(datetime.now() - timedelta(hours=1), datetime.now())
        )

        # 添加到历史记录
        self.quality_history.append(report)

        # 添加新告警到活跃告警列表
        for alert in alerts:
            if alert.resolution_status == "open":
                self.active_alerts.append(alert)

        logger.info(f"数据质量检查完成，质量分数: {quality_score:.4f}")
        return report

    def _calculate_quality_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算质量指标"""
        metrics = {}

        # 完整性
        metrics["completeness"] = self._calculate_completeness(df)

        # 准确性
        metrics["accuracy"] = self._calculate_accuracy(df)

        # 一致性
        metrics["consistency"] = self._calculate_consistency(df)

        # 有效性
        metrics["validity"] = self._calculate_validity(df)

        # 唯一性
        metrics["uniqueness"] = self._calculate_uniqueness(df)

        # 及时性
        metrics["timeliness"] = self._calculate_timeliness(df)

        # 相关性
        metrics["relevance"] = self._calculate_relevance(df)

        return metrics

    def _calculate_completeness(self, df: pd.DataFrame) -> float:
        """计算完整性"""
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        completeness = 1 - (missing_cells / total_cells) if total_cells > 0 else 1.0
        return completeness

    def _calculate_accuracy(self, df: pd.DataFrame) -> float:
        """计算准确性"""
        try:
            # 简单的准确性评估：检查数值列的合理性
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

            if not numeric_columns:
                return 1.0

            accuracy_scores = []

            for col in numeric_columns:
                # 检查异常值比例
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                outlier_ratio = len(outliers) / len(df)

                # 准确性 = 1 - 异常值比例
                accuracy = max(0, 1 - outlier_ratio)
                accuracy_scores.append(accuracy)

            return np.mean(accuracy_scores) if accuracy_scores else 1.0

        except Exception as e:
            logger.warning(f"准确性计算失败: {e}")
            return 0.5

    def _calculate_consistency(self, df: pd.DataFrame) -> float:
        """计算一致性"""
        try:
            # 检查数据类型一致性
            consistency_scores = []

            for col in df.columns:
                # 检查列内数据类型一致性
                if df[col].dtype == 'object':
                    # 对于对象类型，检查是否可以转换为数值
                    try:
                        pd.to_numeric(df[col], errors='raise')
                        consistency_scores.append(1.0)
                    except:
                        # 检查字符串格式一致性
                        unique_formats = df[col].astype(str).str.len().nunique()
                        total_values = len(df[col].dropna())
                        consistency = 1 - (unique_formats / total_values) if total_values > 0 else 1.0
                        consistency_scores.append(consistency)
                else:
                    consistency_scores.append(1.0)

            return np.mean(consistency_scores) if consistency_scores else 1.0

        except Exception as e:
            logger.warning(f"一致性计算失败: {e}")
            return 0.5

    def _calculate_validity(self, df: pd.DataFrame) -> float:
        """计算有效性"""
        try:
            # 检查数据范围有效性
            validity_scores = []

            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    # 数值列：检查是否在合理范围内
                    if col.lower().find('age') != -1:
                        # 年龄列
                        valid_values = df[(df[col] >= 0) & (df[col] <= 150)]
                        validity = len(valid_values) / len(df) if len(df) > 0 else 1.0
                    elif col.lower().find('score') != -1 or col.lower().find('rate') != -1:
                        # 分数或比率列
                        valid_values = df[(df[col] >= 0) & (df[col] <= 1)]
                        validity = len(valid_values) / len(df) if len(df) > 0 else 1.0
                    else:
                        # 其他数值列
                        validity = 1.0

                    validity_scores.append(validity)
                else:
                    # 非数值列
                    validity_scores.append(1.0)

            return np.mean(validity_scores) if validity_scores else 1.0

        except Exception as e:
            logger.warning(f"有效性计算失败: {e}")
            return 0.5

    def _calculate_uniqueness(self, df: pd.DataFrame) -> float:
        """计算唯一性"""
        try:
            # 检查重复行比例
            total_rows = len(df)
            unique_rows = len(df.drop_duplicates())
            uniqueness = unique_rows / total_rows if total_rows > 0 else 1.0

            return uniqueness

        except Exception as e:
            logger.warning(f"唯一性计算失败: {e}")
            return 0.5

    def _calculate_timeliness(self, df: pd.DataFrame) -> float:
        """计算及时性"""
        try:
            # 查找时间列
            time_columns = []
            for col in df.columns:
                if df[col].dtype == 'datetime64[ns]' or 'time' in col.lower() or 'date' in col.lower():
                    time_columns.append(col)

            if not time_columns:
                return 1.0  # 没有时间列，认为及时性为100%

            timeliness_scores = []

            for col in time_columns:
                if df[col].dtype == 'datetime64[ns]':
                    # 计算数据的新鲜度
                    current_time = datetime.now()
                    time_diff = (current_time - df[col].max()).total_seconds()

                    # 假设24小时内的数据是及时的
                    timeliness = max(0, 1 - (time_diff / 86400))
                    timeliness_scores.append(timeliness)

            return np.mean(timeliness_scores) if timeliness_scores else 1.0

        except Exception as e:
            logger.warning(f"及时性计算失败: {e}")
            return 0.5

    def _calculate_relevance(self, df: pd.DataFrame) -> float:
        """计算相关性"""
        try:
            # 简单的相关性评估：检查列之间的相关性
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

            if len(numeric_columns) < 2:
                return 1.0

            # 计算相关性矩阵
            corr_matrix = df[numeric_columns].corr()

            # 计算平均相关性强度
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )

            correlations = upper_triangle.stack().dropna()
            avg_correlation = correlations.abs().mean()

            # 相关性过高或过低都不好
            if avg_correlation < 0.1:
                relevance = 0.5  # 相关性太低
            elif avg_correlation > 0.9:
                relevance = 0.5  # 相关性太高（冗余）
            else:
                relevance = 1.0  # 相关性适中

            return relevance

        except Exception as e:
            logger.warning(f"相关性计算失败: {e}")
            return 0.5

    def _calculate_overall_quality_score(self, metric_scores: Dict[str, float]) -> float:
        """计算总体质量分数"""
        # 加权平均，可以根据业务需求调整权重
        weights = {
            "completeness": 0.25,
            "accuracy": 0.25,
            "consistency": 0.15,
            "validity": 0.15,
            "uniqueness": 0.10,
            "timeliness": 0.05,
            "relevance": 0.05
        }

        weighted_score = sum(
            metric_scores.get(metric.value, 0.5) * weight
            for metric, weight in weights.items()
        )

        return weighted_score

    def _check_thresholds(self, metric_scores: Dict[str, float], data_source: str) -> List[QualityAlert]:
        """检查阈值并生成告警"""
        alerts = []

        for threshold in self.quality_thresholds:
            metric_name = threshold.metric.value
            current_value = metric_scores.get(metric_name, 0.0)

            # 检查是否违反阈值
            violation = False

            if threshold.operator == ">":
                violation = current_value > threshold.threshold_value
            elif threshold.operator == "<":
                violation = current_value < threshold.threshold_value
            elif threshold.operator == ">=":
                violation = current_value >= threshold.threshold_value
            elif threshold.operator == "<=":
                violation = current_value <= threshold.threshold_value
            elif threshold.operator == "==":
                violation = current_value == threshold.threshold_value
            elif threshold.operator == "!=":
                violation = current_value != threshold.threshold_value

            # 如果违反阈值，生成告警
            if not violation:  # 注意：这里检查的是"不违反"，即质量不达标
                alert = QualityAlert(
                    alert_id=f"qa_{int(time.time())}_{metric_name}",
                    metric=threshold.metric,
                    current_value=current_value,
                    threshold_value=threshold.threshold_value,
                    alert_level=threshold.alert_level,
                    message=f"{threshold.description}。当前值: {current_value:.4f}, 阈值: {threshold.threshold_value:.4f}",
                    timestamp=datetime.now(),
                    data_source=data_source
                )
                alerts.append(alert)

        return alerts

    def _generate_recommendations(self, metric_scores: Dict[str, float], df: pd.DataFrame) -> List[str]:
        """生成改进建议"""
        recommendations = []

        # 基于质量指标生成建议
        for metric_name, score in metric_scores.items():
            if score < 0.8:  # 质量分数低于80%
                if metric_name == "completeness":
                    recommendations.append("建议检查数据收集过程，减少缺失值")
                elif metric_name == "accuracy":
                    recommendations.append("建议进行数据清洗，移除异常值")
                elif metric_name == "consistency":
                    recommendations.append("建议统一数据格式和编码标准")
                elif metric_name == "validity":
                    recommendations.append("建议检查数据范围，确保数据有效性")
                elif metric_name == "uniqueness":
                    recommendations.append("建议检查并移除重复数据")
                elif metric_name == "timeliness":
                    recommendations.append("建议优化数据更新频率")
                elif metric_name == "relevance":
                    recommendations.append("建议检查特征相关性，移除冗余特征")

        # 基于数据特征生成建议
        if len(df) < 100:
            recommendations.append("数据量较少，建议增加数据收集")

        missing_ratio = df.isnull().sum().sum() / df.size
        if missing_ratio > 0.1:
            recommendations.append(f"缺失值比例较高({missing_ratio:.2%})，建议进行缺失值处理")

        return recommendations

    def detect_anomalies(self, df: pd.DataFrame, anomaly_type: AnomalyType = AnomalyType.STATISTICAL) -> List[Dict[str, Any]]:
        """检测数据异常"""
        logger.info(f"开始异常检测，类型: {anomaly_type.value}")

        anomalies = []

        try:
            if anomaly_type == AnomalyType.STATISTICAL:
                anomalies = self._detect_statistical_anomalies(df)
            elif anomaly_type == AnomalyType.PATTERN:
                anomalies = self._detect_pattern_anomalies(df)
            elif anomaly_type == AnomalyType.DISTRIBUTION:
                anomalies = self._detect_distribution_anomalies(df)
            elif anomaly_type == AnomalyType.CORRELATION:
                anomalies = self._detect_correlation_anomalies(df)
            elif anomaly_type == AnomalyType.TEMPORAL:
                anomalies = self._detect_temporal_anomalies(df)

            logger.info(f"异常检测完成，发现 {len(anomalies)} 个异常")
            return anomalies

        except Exception as e:
            logger.error(f"异常检测失败: {e}")
            return []

    def _detect_statistical_anomalies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """检测统计异常"""
        anomalies = []

        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in numeric_columns:
            try:
                # 使用孤立森林检测异常
                if col not in self.anomaly_detectors:
                    self.anomaly_detectors[col] = IsolationForest(contamination=0.1, random_state=42)

                detector = self.anomaly_detectors[col]
                anomaly_labels = detector.fit_predict(df[[col]])

                # 记录异常
                anomaly_indices = np.where(anomaly_labels == -1)[0]
                for idx in anomaly_indices:
                    anomalies.append({
                        "type": "statistical",
                        "column": col,
                        "row_index": idx,
                        "value": df.iloc[idx][col],
                        "description": f"列 {col} 第 {idx} 行存在统计异常"
                    })

            except Exception as e:
                logger.warning(f"统计异常检测失败 {col}: {e}")

        return anomalies

    def _detect_pattern_anomalies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """检测模式异常"""
        anomalies = []

        # 简单的模式异常检测：检查重复模式
        for col in df.columns:
            if df[col].dtype == 'object':
                # 检查字符串模式
                value_counts = df[col].value_counts()
                total_values = len(df[col].dropna())

                for value, count in value_counts.items():
                    if count / total_values > 0.5:  # 某个值占比超过50%
                        anomalies.append({
                            "type": "pattern",
                            "column": col,
                            "value": value,
                            "frequency": count,
                            "ratio": count / total_values,
                            "description": f"列 {col} 中值 '{value}' 出现频率异常高"
                        })

        return anomalies

    def _detect_distribution_anomalies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """检测分布异常"""
        anomalies = []

        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in numeric_columns:
            try:
                # 检查分布形状
                data = df[col].dropna()

                # 计算偏度和峰度
                skewness = stats.skew(data)
                kurt = stats.kurtosis(data)

                # 检查是否偏离正态分布
                if abs(skewness) > 2:  # 偏度异常
                    anomalies.append({
                        "type": "distribution",
                        "column": col,
                        "metric": "skewness",
                        "value": skewness,
                        "description": f"列 {col} 偏度异常: {skewness:.4f}"
                    })

                if abs(kurt) > 3:  # 峰度异常
                    anomalies.append({
                        "type": "distribution",
                        "column": col,
                        "metric": "kurtosis",
                        "value": kurt,
                        "description": f"列 {col} 峰度异常: {kurt:.4f}"
                    })

            except Exception as e:
                logger.warning(f"分布异常检测失败 {col}: {e}")

        return anomalies

    def _detect_correlation_anomalies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """检测相关性异常"""
        anomalies = []

        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_columns) < 2:
            return anomalies

        try:
            # 计算相关性矩阵
            corr_matrix = df[numeric_columns].corr()

            # 检查异常高的相关性
            for i in range(len(numeric_columns)):
                for j in range(i+1, len(numeric_columns)):
                    corr_value = corr_matrix.iloc[i, j]

                    if abs(corr_value) > 0.95:  # 相关性过高
                        anomalies.append({
                            "type": "correlation",
                            "columns": [numeric_columns[i], numeric_columns[j]],
                            "correlation": corr_value,
                            "description": f"列 {numeric_columns[i]} 和 {numeric_columns[j]} 相关性异常高: {corr_value:.4f}"
                        })

        except Exception as e:
            logger.warning(f"相关性异常检测失败: {e}")

        return anomalies

    def _detect_temporal_anomalies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """检测时间异常"""
        anomalies = []

        # 查找时间列
        time_columns = []
        for col in df.columns:
            if df[col].dtype == 'datetime64[ns]':
                time_columns.append(col)

        for col in time_columns:
            try:
                # 检查时间序列的连续性
                time_series = df[col].dropna().sort_values()

                if len(time_series) > 1:
                    # 计算时间间隔
                    time_diffs = time_series.diff().dropna()

                    # 检查异常的时间间隔
                    mean_diff = time_diffs.mean()
                    std_diff = time_diffs.std()

                    anomaly_threshold = mean_diff + 3 * std_diff
                    anomaly_indices = time_diffs[time_diffs > anomaly_threshold].index

                    for idx in anomaly_indices:
                        anomalies.append({
                            "type": "temporal",
                            "column": col,
                            "row_index": idx,
                            "time_gap": time_diffs[idx].total_seconds(),
                            "description": f"列 {col} 第 {idx} 行存在时间间隔异常"
                        })

            except Exception as e:
                logger.warning(f"时间异常检测失败 {col}: {e}")

        return anomalies

    def resolve_alert(self, alert_id: str, resolution_notes: str = ""):
        """解决告警"""
        for alert in self.active_alerts:
            if alert.alert_id == alert_id:
                alert.resolution_status = "resolved"
                alert.resolved_at = datetime.now()
                logger.info(f"告警 {alert_id} 已解决")
                break

    def get_active_alerts(self) -> List[QualityAlert]:
        """获取活跃告警"""
        return [alert for alert in self.active_alerts if alert.resolution_status == "open"]

    def get_quality_history(self, limit: int = 100) -> List[QualityReport]:
        """获取质量历史"""
        return self.quality_history[-limit:] if self.quality_history else []

    def get_quality_summary(self) -> Dict[str, Any]:
        """获取质量摘要"""
        if not self.quality_history:
            return {"message": "暂无质量历史数据"}

        latest_report = self.quality_history[-1]
        active_alerts_count = len(self.get_active_alerts())

        return {
            "latest_quality_score": latest_report.quality_score,
            "total_reports": len(self.quality_history),
            "active_alerts": active_alerts_count,
            "data_source": latest_report.data_source,
            "total_records": latest_report.total_records,
            "last_check": latest_report.generated_at.isoformat(),
            "metric_scores": latest_report.metric_scores
        }

    def stop_monitoring(self):
        """停止监控"""
        self.monitoring_enabled = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("数据质量监控已停止")

# 全局数据质量监控器实例
data_quality_monitor = DataQualityMonitor()

# 异步任务
@async_task("check_data_quality", TaskPriority.NORMAL)
def check_data_quality_task(data: Dict[str, Any], data_source: str):
    """数据质量检查任务"""
    df = pd.DataFrame(data)
    report = data_quality_monitor.check_data_quality(df, data_source)

    return {
        "report": asdict(report),
        "success": True
    }

@async_task("detect_anomalies", TaskPriority.NORMAL)
def detect_anomalies_task(data: Dict[str, Any], anomaly_type: str):
    """异常检测任务"""
    df = pd.DataFrame(data)
    anomaly_type_enum = AnomalyType(anomaly_type)
    anomalies = data_quality_monitor.detect_anomalies(df, anomaly_type_enum)

    return {
        "anomalies": anomalies,
        "success": True
    }

# 数据质量监控API
def check_data_quality(df: pd.DataFrame, data_source: str = "unknown") -> QualityReport:
    """检查数据质量"""
    return data_quality_monitor.check_data_quality(df, data_source)

def detect_anomalies(df: pd.DataFrame, anomaly_type: AnomalyType = AnomalyType.STATISTICAL) -> List[Dict[str, Any]]:
    """检测数据异常"""
    return data_quality_monitor.detect_anomalies(df, anomaly_type)

def get_active_alerts() -> List[QualityAlert]:
    """获取活跃告警"""
    return data_quality_monitor.get_active_alerts()

def get_quality_summary() -> Dict[str, Any]:
    """获取质量摘要"""
    return data_quality_monitor.get_quality_summary()

def resolve_alert(alert_id: str, resolution_notes: str = ""):
    """解决告警"""
    data_quality_monitor.resolve_alert(alert_id, resolution_notes)

if __name__ == "__main__":
    # 测试数据质量监控
    import numpy as np

    # 创建测试数据
    np.random.seed(42)
    test_data = {
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.normal(50, 10, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000),
        'timestamp': pd.date_range('2023-01-01', periods=1000, freq='H')
    }

    # 添加一些质量问题
    test_data['feature1'][:50] = np.nan  # 缺失值
    test_data['feature2'][100:110] = 1000  # 异常值

    df = pd.DataFrame(test_data)

    # 检查数据质量
    report = check_data_quality(df, "test_source")

    print("数据质量检查完成:")
    print(f"质量分数: {report.quality_score:.4f}")
    print(f"指标分数: {report.metric_scores}")
    print(f"告警数量: {len(report.alerts)}")
    print(f"建议数量: {len(report.recommendations)}")

    # 检测异常
    anomalies = detect_anomalies(df, AnomalyType.STATISTICAL)
    print(f"检测到 {len(anomalies)} 个异常")

    # 获取质量摘要
    summary = get_quality_summary()
    print("质量摘要:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

__all__ = ["'logger'", "'QualityMetric'", "'AnomalyType'", "'AlertLevel'", "'QualityThreshold'", "'QualityAlert'", "'QualityReport'", "'DataQualityMonitor'", "'data_quality_monitor'", "'check_data_quality_task'", "'detect_anomalies_task'", "'check_data_quality'", "'detect_anomalies'", "'get_active_alerts'", "'get_quality_summary'", "'resolve_alert'"]
