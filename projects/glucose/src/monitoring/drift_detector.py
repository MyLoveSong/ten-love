"""
模型漂移检测模块
用于检测生产环境中模型和数据的漂移
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional
from scipy import stats
from collections import deque
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class DriftDetector:
    """检测模型和数据漂移"""

    def __init__(self, reference_data: Optional[torch.Tensor] = None,
                 window_size: int = 1000,
                 drift_threshold: float = 0.05,
                 min_samples: int = 100):
        self.reference_data = reference_data
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.min_samples = min_samples

        self.prediction_window = deque(maxlen=window_size)
        self.input_window = deque(maxlen=window_size)
        self.target_window = deque(maxlen=window_size)

        self.drift_detected = False
        self.last_drift_check = datetime.now()
        self.drift_history: List[Dict[str, Any]] = []

        self.performance_metrics = {
            'mae': deque(maxlen=100),
            'rmse': deque(maxlen=100),
            'r2': deque(maxlen=100)
        }

    def log_prediction(self, input_data: torch.Tensor, prediction: torch.Tensor,
                      actual: Optional[torch.Tensor] = None) -> None:
        """记录预测数据用于漂移监控"""
        try:
            self.input_window.append(input_data.cpu().numpy())
            self.prediction_window.append(prediction.cpu().numpy())

            if actual is not None:
                self.target_window.append(actual.cpu().numpy())

                mae = torch.mean(torch.abs(prediction - actual)).item()
                rmse = torch.sqrt(torch.mean((prediction - actual) ** 2)).item()

                ss_res = torch.sum((actual - prediction) ** 2)
                ss_tot = torch.sum((actual - torch.mean(actual)) ** 2)
                r2 = (1 - ss_res / ss_tot).item() if ss_tot > 0 else 0.0

                self.performance_metrics['mae'].append(mae)
                self.performance_metrics['rmse'].append(rmse)
                self.performance_metrics['r2'].append(r2)

        except Exception as e:
            logger.error(f"记录预测失败: {str(e)}")

    def detect_drift(self) -> Dict[str, Any]:
        """检测漂移"""
        if len(self.input_window) < self.min_samples:
            return {
                'drift_detected': False,
                'reason': '样本不足',
                'samples_available': len(self.input_window)
            }

        try:
            drift_results = {
                'timestamp': datetime.now().isoformat(),
                'drift_detected': False,
                'drift_types': {},
                'recommendations': []
            }

            # 数据漂移检测
            data_drift = self._detect_data_drift()
            drift_results['drift_types']['data_drift'] = data_drift

            # 预测漂移检测
            prediction_drift = self._detect_prediction_drift()
            drift_results['drift_types']['prediction_drift'] = prediction_drift

            # 性能漂移检测
            performance_drift = self._detect_performance_drift()
            drift_results['drift_types']['performance_drift'] = performance_drift

            # 总体漂移评估
            drift_results['drift_detected'] = any([
                data_drift['detected'],
                prediction_drift['detected'],
                performance_drift['detected']
            ])

            if drift_results['drift_detected']:
                drift_results['recommendations'] = self._generate_recommendations(drift_results)

            self.drift_history.append(drift_results)
            self.last_drift_check = datetime.now()

            return drift_results

        except Exception as e:
            logger.error(f"漂移检测失败: {str(e)}")
            return {'drift_detected': False, 'error': str(e)}

    def _detect_data_drift(self) -> Dict[str, Any]:
        """检测输入数据分布漂移"""
        if self.reference_data is None:
            return {'detected': False, 'reason': '无参考数据'}

        try:
            current_data = np.array(list(self.input_window))
            reference_data = self.reference_data.cpu().numpy()

            current_flat = current_data.reshape(-1, current_data.shape[-1])
            reference_flat = reference_data.reshape(-1, reference_data.shape[-1])

            # KS检验
            ks_statistics = []
            p_values = []

            for i in range(current_flat.shape[1]):
                ks_stat, p_val = stats.ks_2samp(reference_flat[:, i], current_flat[:, i])
                ks_statistics.append(ks_stat)
                p_values.append(p_val)

            significant_features = sum(1 for p in p_values if p < self.drift_threshold)

            drift_detected = significant_features > len(p_values) * 0.3

            return {
                'detected': drift_detected,
                'significant_features': significant_features
            }

        except Exception as e:
            return {'detected': False, 'error': str(e)}

    def _detect_prediction_drift(self) -> Dict[str, Any]:
        """检测预测分布漂移"""
        if len(self.prediction_window) < self.min_samples:
            return {'detected': False, 'reason': '预测样本不足'}

        try:
            predictions = np.array(list(self.prediction_window))

            mid_point = len(predictions) // 2
            first_half = predictions[:mid_point]
            second_half = predictions[mid_point:]

            ks_stat, p_val = stats.ks_2samp(
                first_half.flatten(), second_half.flatten()
            )

            mean_diff = np.abs(np.mean(second_half) - np.mean(first_half))
            var_ratio = np.var(second_half) / (np.var(first_half) + 1e-8)

            drift_detected = (p_val < self.drift_threshold or
                            mean_diff > 0.1 or
                            var_ratio > 2.0 or var_ratio < 0.5)

            return {
                'detected': drift_detected,
                'mean_difference': mean_diff,
                'variance_ratio': var_ratio
            }

        except Exception as e:
            return {'detected': False, 'error': str(e)}

    def _detect_performance_drift(self) -> Dict[str, Any]:
        """检测模型性能漂移"""
        if len(self.performance_metrics['mae']) < self.min_samples // 10:
            return {'detected': False, 'reason': '性能样本不足'}

        try:
            mae_values = list(self.performance_metrics['mae'])

            split_point = len(mae_values) // 2
            historical_mae = mae_values[:split_point]
            recent_mae = mae_values[split_point:]

            t_stat, p_val = stats.ttest_ind(historical_mae, recent_mae)

            historical_mean = np.mean(historical_mae)
            recent_mean = np.mean(recent_mae)
            performance_degradation = (recent_mean - historical_mean) / historical_mean

            drift_detected = (performance_degradation > 0.2 or p_val < self.drift_threshold)

            return {
                'detected': drift_detected,
                'performance_degradation': performance_degradation
            }

        except Exception as e:
            return {'detected': False, 'error': str(e)}

    def _generate_recommendations(self, drift_results: Dict[str, Any]) -> List[str]:
        """根据漂移检测结果生成建议"""
        recommendations = []

        if drift_results['drift_types']['data_drift']['detected']:
            recommendations.append("检测到数据漂移 - 考虑使用最新数据重新训练")

        if drift_results['drift_types']['prediction_drift']['detected']:
            recommendations.append("检测到预测漂移 - 模型行为已改变")

        if drift_results['drift_types']['performance_drift']['detected']:
            recommendations.append("检测到性能下降 - 需要立即关注")

        return recommendations

    def get_performance_metrics(self) -> Dict[str, float]:
        """获取当前性能指标"""
        if not self.performance_metrics['mae']:
            return {}

        return {
            'current_mae': list(self.performance_metrics['mae'])[-1],
            'avg_mae': np.mean(list(self.performance_metrics['mae'])),
            'current_rmse': list(self.performance_metrics['rmse'])[-1],
            'avg_rmse': np.mean(list(self.performance_metrics['rmse'])),
            'samples_monitored': len(self.performance_metrics['mae'])
        }

    def reset_monitoring(self) -> None:
        """重置监控状态"""
        self.prediction_window.clear()
        self.input_window.clear()
        self.target_window.clear()

        for metric in self.performance_metrics.values():
            metric.clear()

        self.drift_detected = False
        self.drift_history.clear()
