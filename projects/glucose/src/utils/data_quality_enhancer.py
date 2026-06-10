"""
数据质量增强器
实现数据清洗和基本质量评估
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DataQualityMetrics:
    """数据质量指标"""
    completeness: float = 0.0
    uniqueness: float = 0.0
    overall_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)


class DataQualityEnhancer:
    """数据质量增强器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config: Dict[str, Any] = config or {'outlier_threshold': 3.0, 'random_seed': 42}
        self.quality_metrics: Optional[DataQualityMetrics] = None

    def enhance_data(self, data_points: List[Any]) -> Tuple[List[Any], DataQualityMetrics]:
        """增强数据质量"""
        logger.info(f"开始数据质量增强，处理 {len(data_points)} 个数据点")

        cleaned = self._clean_data(data_points)
        unique = self._remove_duplicates(cleaned)
        filtered = self._handle_outliers(unique)

        self.quality_metrics = self._assess_quality(filtered)
        return filtered, self.quality_metrics

    def _clean_data(self, data_points: List[Any]) -> List[Any]:
        """清洗数据"""
        return [dp for dp in data_points if self._validate_data_point(dp)]

    def _validate_data_point(self, dp: Any) -> bool:
        """验证数据点"""
        try:
            if not hasattr(dp, 'acceptance_score'):
                return False
            return 0 <= dp.acceptance_score <= 1
        except Exception:
            return False

    def _remove_duplicates(self, data_points: List[Any]) -> List[Any]:
        """去重——按 Python 对象身份（id）确保同一对象不会被重复添加"""
        seen_ids: set[int] = set()
        unique: List[Any] = []
        for dp in data_points:
            dp_id = id(dp)
            if dp_id not in seen_ids:
                seen_ids.add(dp_id)
                unique.append(dp)
        return unique

    def _handle_outliers(self, data_points: List[Any]) -> List[Any]:
        """处理异常值"""
        if len(data_points) < 10:
            return data_points

        scores = np.array([dp.acceptance_score for dp in data_points])
        mean, std = np.mean(scores), np.std(scores)
        threshold = self.config.get('outlier_threshold', 3.0)

        return [dp for dp in data_points if abs(dp.acceptance_score - mean) <= threshold * std]

    def _assess_quality(self, data_points: List[Any]) -> DataQualityMetrics:
        """评估质量"""
        if not data_points:
            return DataQualityMetrics()

        completeness = sum(1 for dp in data_points if self._validate_data_point(dp)) / len(data_points)
        uniqueness = len(set(id(dp) for dp in data_points)) / len(data_points)
        overall_score = (completeness + uniqueness) / 2

        if overall_score >= 0.9:
            recommendations = ["数据质量优秀"]
        elif overall_score >= 0.7:
            recommendations = ["数据质量良好"]
        else:
            recommendations = ["数据质量需要改进"]

        return DataQualityMetrics(
            completeness=float(completeness),
            uniqueness=float(uniqueness),
            overall_score=float(overall_score),
            recommendations=recommendations
        )

    def get_quality_report(self) -> Dict[str, Any]:
        """获取质量报告"""
        if self.quality_metrics is None:
            return {}
        return {
            'completeness': self.quality_metrics.completeness,
            'uniqueness': self.quality_metrics.uniqueness,
            'overall_score': self.quality_metrics.overall_score,
            'recommendations': self.quality_metrics.recommendations
        }
