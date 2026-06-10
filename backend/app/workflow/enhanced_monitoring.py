"""
增强监控模块
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import asyncio
import json

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版工作流监控和智能告警系统
支持预测性告警、自愈机制、实时数据流监控
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd
from collections import deque
import threading
import time

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """告警严重程度"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class DataStreamType(Enum):
    """数据流类型"""
    GLUCOSE = "glucose"
    NUTRITION = "nutrition"
    ACTIVITY = "activity"
    SYSTEM = "system"
    WORKFLOW = "workflow"

class PredictionType(Enum):
    """预测类型"""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DATA_ANOMALY = "data_anomaly"
    WORKFLOW_FAILURE = "workflow_failure"
    USER_BEHAVIOR_CHANGE = "user_behavior_change"

@dataclass
class PredictiveAlertRule:
    """预测性告警规则"""
    rule_id: str
    name: str
    prediction_type: PredictionType
    prediction_window: int  # 预测时间窗口（分钟）
    confidence_threshold: float
    severity: AlertSeverity
    auto_heal_enabled: bool = True
    heal_actions: List[str] = field(default_factory=list)
    enabled: bool = True

@dataclass
class DataStreamConfig:
    """数据流配置"""
    stream_type: DataStreamType
    sampling_rate: float  # 采样率
    buffer_size: int
    anomaly_detection_enabled: bool = True
    prediction_enabled: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=dict)

@dataclass
class HealAction:
    """自愈动作"""
    action_id: str
    name: str
    action_type: str  # restart, scale, reroute, cleanup
    parameters: Dict[str, Any] = field(default_factory=dict)
    success_rate: float = 0.0
    execution_count: int = 0

class RealTimeDataStreamMonitor:
    """实时数据流监控器"""

    def __init__(self):
        self.data_streams: Dict[str, deque] = {}
        self.stream_configs: Dict[str, DataStreamConfig] = {}
        self.anomaly_detectors: Dict[str, IsolationForest] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.prediction_models: Dict[str, Any] = {}

        # 初始化默认配置
        self._initialize_default_configs()

        # 启动监控任务
        self._start_monitoring_tasks()

    def _initialize_default_configs(self):
        """初始化默认配置"""
        default_configs = [
            DataStreamConfig(
                stream_type=DataStreamType.GLUCOSE,
                sampling_rate=1.0,  # 每分钟采样
                buffer_size=1440,  # 24小时数据
                anomaly_detection_enabled=True,
                prediction_enabled=True,
                alert_thresholds={'anomaly_score': 0.8, 'trend_change': 0.7}
            ),
            DataStreamConfig(
                stream_type=DataStreamType.SYSTEM,
                sampling_rate=0.1,  # 每10秒采样
                buffer_size=3600,  # 1小时数据
                anomaly_detection_enabled=True,
                prediction_enabled=True,
                alert_thresholds={'cpu_threshold': 85.0, 'memory_threshold': 90.0}
            ),
            DataStreamConfig(
                stream_type=DataStreamType.WORKFLOW,
                sampling_rate=1.0,  # 每分钟采样
                buffer_size=1440,  # 24小时数据
                anomaly_detection_enabled=True,
                prediction_enabled=True,
                alert_thresholds={'failure_rate': 0.1, 'execution_time': 300.0}
            )
        ]

        for config in default_configs:
            stream_key = config.stream_type.value
            self.stream_configs[stream_key] = config
            self.data_streams[stream_key] = deque(maxlen=config.buffer_size)

            # 初始化异常检测器
            if config.anomaly_detection_enabled:
                self.anomaly_detectors[stream_key] = IsolationForest(
                    contamination=0.1,
                    random_state=42
                )
                self.scalers[stream_key] = StandardScaler()

    def _start_monitoring_tasks(self):
        """启动监控任务"""
        # 延迟启动，避免在初始化时创建异步任务
        self._monitoring_task_started = False

    async def _ensure_monitoring_task(self):
        """确保监控任务正在运行"""
        if not self._monitoring_task_started:
            self._monitoring_task_started = True
            asyncio.create_task(self._monitoring_loop())

    async def _monitoring_loop(self):
        """监控循环"""
        while True:
            try:
                await self._process_data_streams()
                await asyncio.sleep(10)  # 每10秒处理一次
            except Exception as e:
                logger.error(f"数据流监控失败: {e}")
                await asyncio.sleep(5)

    async def feed_data_stream(self, stream_type: DataStreamType, data: Dict[str, Any]):
        """输入数据流"""
        stream_key = stream_type.value

        if stream_key not in self.data_streams:
            logger.warning(f"未知数据流类型: {stream_type}")
            return

        # 添加时间戳
        data_point = {
            'timestamp': datetime.now(),
            'data': data
        }

        self.data_streams[stream_key].append(data_point)

        # 检查是否需要立即处理
        config = self.stream_configs[stream_key]
        if len(self.data_streams[stream_key]) % 10 == 0:  # 每10个数据点处理一次
            await self._analyze_stream(stream_key)

    async def _process_data_streams(self):
        """处理所有数据流"""
        for stream_key in self.data_streams:
            if len(self.data_streams[stream_key]) > 0:
                await self._analyze_stream(stream_key)

    async def _analyze_stream(self, stream_key: str):
        """分析数据流"""
        config = self.stream_configs[stream_key]
        data_points = list(self.data_streams[stream_key])

        if len(data_points) < 10:  # 需要足够的数据点
            return

        # 异常检测
        if config.anomaly_detection_enabled:
            anomalies = await self._detect_anomalies(stream_key, data_points)
            if anomalies:
                await self._handle_anomalies(stream_key, anomalies)

        # 预测分析
        if config.prediction_enabled:
            predictions = await self._make_predictions(stream_key, data_points)
            if predictions:
                await self._handle_predictions(stream_key, predictions)

    async def _detect_anomalies(self, stream_key: str, data_points: List[Dict]) -> List[Dict]:
        """检测异常"""
        try:
            # 提取特征
            features = self._extract_stream_features(data_points)

            if len(features) < 5:
                return []

            # 标准化特征
            scaler = self.scalers[stream_key]
            features_scaled = scaler.fit_transform(features)

            # 异常检测
            detector = self.anomaly_detectors[stream_key]
            anomaly_scores = detector.decision_function(features_scaled)
            anomaly_labels = detector.predict(features_scaled)

            # 识别异常点
            anomalies = []
            threshold = self.stream_configs[stream_key].alert_thresholds.get('anomaly_score', 0.8)

            for i, (score, label) in enumerate(zip(anomaly_scores, anomaly_labels)):
                if score < threshold or label == -1:
                    anomalies.append({
                        'index': i,
                        'score': score,
                        'data_point': data_points[i],
                        'severity': self._calculate_anomaly_severity(score)
                    })

            return anomalies

        except Exception as e:
            logger.error(f"异常检测失败 ({stream_key}): {e}")
            return []

    def _extract_stream_features(self, data_points: List[Dict]) -> np.ndarray:
        """提取数据流特征"""
        features = []

        for point in data_points:
            data = point['data']
            timestamp = point['timestamp']

            # 基础特征
            feature_vector = [
                timestamp.hour,
                timestamp.minute,
                timestamp.weekday()
            ]

            # 数据特征
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, (int, float)):
                        feature_vector.append(value)
                    elif isinstance(value, str):
                        feature_vector.append(len(value))  # 字符串长度

            features.append(feature_vector)

        return np.array(features)

    def _calculate_anomaly_severity(self, score: float) -> AlertSeverity:
        """计算异常严重程度"""
        if score < -0.5:
            return AlertSeverity.CRITICAL
        elif score < -0.3:
            return AlertSeverity.HIGH
        elif score < -0.1:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW

    async def _make_predictions(self, stream_key: str, data_points: List[Dict]) -> List[Dict]:
        """进行预测"""
        try:
            # 简化的预测逻辑
            predictions = []

            # 趋势预测
            trend_prediction = self._predict_trend(data_points)
            if trend_prediction:
                predictions.append(trend_prediction)

            # 性能预测
            performance_prediction = self._predict_performance(stream_key, data_points)
            if performance_prediction:
                predictions.append(performance_prediction)

            return predictions

        except Exception as e:
            logger.error(f"预测失败 ({stream_key}): {e}")
            return []

    def _predict_trend(self, data_points: List[Dict]) -> Optional[Dict]:
        """预测趋势"""
        if len(data_points) < 20:
            return None

        # 简化的趋势分析
        recent_data = data_points[-10:]
        older_data = data_points[-20:-10]

        recent_avg = np.mean([self._extract_numeric_value(p['data']) for p in recent_data])
        older_avg = np.mean([self._extract_numeric_value(p['data']) for p in older_data])

        trend_change = (recent_avg - older_avg) / older_avg if older_avg != 0 else 0

        if abs(trend_change) > 0.1:  # 10%变化
            return {
                'type': 'trend_change',
                'value': trend_change,
                'confidence': min(1.0, abs(trend_change) * 2),
                'prediction_window': 30  # 30分钟
            }

        return None

    def _predict_performance(self, stream_key: str, data_points: List[Dict]) -> Optional[Dict]:
        """预测性能"""
        if stream_key != 'workflow':
            return None

        # 分析执行时间趋势
        execution_times = []
        for point in data_points:
            data = point['data']
            if 'execution_time' in data:
                execution_times.append(data['execution_time'])

        if len(execution_times) < 10:
            return None

        # 计算趋势
        recent_avg = np.mean(execution_times[-5:])
        overall_avg = np.mean(execution_times)

        if recent_avg > overall_avg * 1.2:  # 性能下降20%
            return {
                'type': 'performance_degradation',
                'value': recent_avg / overall_avg,
                'confidence': min(1.0, (recent_avg / overall_avg - 1) * 2),
                'prediction_window': 60  # 60分钟
            }

        return None

    def _extract_numeric_value(self, data: Any) -> float:
        """提取数值"""
        if isinstance(data, (int, float)):
            return float(data)
        elif isinstance(data, dict):
            # 尝试提取第一个数值
            for value in data.values():
                if isinstance(value, (int, float)):
                    return float(value)
        return 0.0

    async def _handle_anomalies(self, stream_key: str, anomalies: List[Dict]):
        """处理异常"""
        for anomaly in anomalies:
            logger.warning(f"检测到异常 ({stream_key}): {anomaly['score']:.3f}")

            # 触发告警
            await self._trigger_predictive_alert(
                stream_key,
                PredictionType.DATA_ANOMALY,
                anomaly['severity'],
                {
                    'anomaly_score': anomaly['score'],
                    'data_point': anomaly['data_point']
                }
            )

    async def _handle_predictions(self, stream_key: str, predictions: List[Dict]):
        """处理预测"""
        for prediction in predictions:
            logger.info(f"预测结果 ({stream_key}): {prediction['type']}")

            # 根据预测类型触发告警
            prediction_type_map = {
                'trend_change': PredictionType.USER_BEHAVIOR_CHANGE,
                'performance_degradation': PredictionType.PERFORMANCE_DEGRADATION
            }

            prediction_type = prediction_type_map.get(prediction['type'])
            if prediction_type:
                await self._trigger_predictive_alert(
                    stream_key,
                    prediction_type,
                    AlertSeverity.MEDIUM,
                    prediction
                )

class EnhancedWorkflowMonitor:
    """增强版工作流监控器"""

    def __init__(self):
        self.data_stream_monitor = RealTimeDataStreamMonitor()
        self.predictive_rules: Dict[str, PredictiveAlertRule] = {}
        self.heal_actions: Dict[str, HealAction] = {}
        self.alert_history: List[Dict[str, Any]] = []
        self.auto_heal_enabled = True

        # 初始化预测性规则
        self._initialize_predictive_rules()

        # 初始化自愈动作
        self._initialize_heal_actions()

    def _initialize_predictive_rules(self):
        """初始化预测性规则"""
        rules = [
            PredictiveAlertRule(
                rule_id="performance_degradation",
                name="性能下降预测",
                prediction_type=PredictionType.PERFORMANCE_DEGRADATION,
                prediction_window=60,
                confidence_threshold=0.7,
                severity=AlertSeverity.HIGH,
                auto_heal_enabled=True,
                heal_actions=["scale_resources", "optimize_workflow"]
            ),
            PredictiveAlertRule(
                rule_id="resource_exhaustion",
                name="资源耗尽预测",
                prediction_type=PredictionType.RESOURCE_EXHAUSTION,
                prediction_window=30,
                confidence_threshold=0.8,
                severity=AlertSeverity.CRITICAL,
                auto_heal_enabled=True,
                heal_actions=["scale_up", "cleanup_resources"]
            ),
            PredictiveAlertRule(
                rule_id="data_anomaly",
                name="数据异常预测",
                prediction_type=PredictionType.DATA_ANOMALY,
                prediction_window=15,
                confidence_threshold=0.6,
                severity=AlertSeverity.MEDIUM,
                auto_heal_enabled=True,
                heal_actions=["data_cleaning", "retry_processing"]
            ),
            PredictiveAlertRule(
                rule_id="workflow_failure",
                name="工作流失败预测",
                prediction_type=PredictionType.WORKFLOW_FAILURE,
                prediction_window=45,
                confidence_threshold=0.75,
                severity=AlertSeverity.HIGH,
                auto_heal_enabled=True,
                heal_actions=["restart_workflow", "fallback_route"]
            )
        ]

        for rule in rules:
            self.predictive_rules[rule.rule_id] = rule

    def _initialize_heal_actions(self):
        """初始化自愈动作"""
        actions = [
            HealAction(
                action_id="scale_resources",
                name="扩展资源",
                action_type="scale",
                parameters={"cpu_multiplier": 1.5, "memory_multiplier": 1.3}
            ),
            HealAction(
                action_id="scale_up",
                name="向上扩展",
                action_type="scale",
                parameters={"instances": 2, "cpu_limit": "2000m"}
            ),
            HealAction(
                action_id="cleanup_resources",
                name="清理资源",
                action_type="cleanup",
                parameters={"cleanup_cache": True, "gc_force": True}
            ),
            HealAction(
                action_id="restart_workflow",
                name="重启工作流",
                action_type="restart",
                parameters={"graceful": True, "timeout": 30}
            ),
            HealAction(
                action_id="fallback_route",
                name="回退路由",
                action_type="reroute",
                parameters={"fallback_workflow": "backup_workflow"}
            ),
            HealAction(
                action_id="data_cleaning",
                name="数据清理",
                action_type="cleanup",
                parameters={"remove_outliers": True, "fill_missing": True}
            ),
            HealAction(
                action_id="retry_processing",
                name="重试处理",
                action_type="restart",
                parameters={"max_retries": 3, "backoff_factor": 2}
            )
        ]

        for action in actions:
            self.heal_actions[action.action_id] = action

    async def _trigger_predictive_alert(self, stream_key: str, prediction_type: PredictionType,
                                       severity: AlertSeverity, details: Dict[str, Any]):
        """触发预测性告警"""
        alert_id = f"pred_{prediction_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        alert = {
            'alert_id': alert_id,
            'timestamp': datetime.now(),
            'stream_key': stream_key,
            'prediction_type': prediction_type.value,
            'severity': severity.value,
            'details': details,
            'status': 'active',
            'auto_heal_attempted': False
        }

        self.alert_history.append(alert)

        # 查找匹配的预测规则
        matching_rules = [
            rule for rule in self.predictive_rules.values()
            if rule.prediction_type == prediction_type and rule.enabled
        ]

        for rule in matching_rules:
            if details.get('confidence', 0) >= rule.confidence_threshold:
                logger.warning(f"预测性告警触发: {rule.name}")

                # 自动自愈
                if rule.auto_heal_enabled and self.auto_heal_enabled:
                    await self._execute_auto_heal(alert, rule)

    async def _execute_auto_heal(self, alert: Dict[str, Any], rule: PredictiveAlertRule):
        """执行自动自愈"""
        alert['auto_heal_attempted'] = True

        for action_id in rule.heal_actions:
            if action_id in self.heal_actions:
                action = self.heal_actions[action_id]

                try:
                    success = await self._execute_heal_action(action, alert)

                    # 更新动作统计
                    action.execution_count += 1
                    if success:
                        action.success_rate = (action.success_rate * (action.execution_count - 1) + 1) / action.execution_count
                    else:
                        action.success_rate = (action.success_rate * (action.execution_count - 1)) / action.execution_count

                    logger.info(f"自愈动作执行: {action.name}, 成功: {success}")

                except Exception as e:
                    logger.error(f"自愈动作失败 ({action.name}): {e}")

    async def _execute_heal_action(self, action: HealAction, alert: Dict[str, Any]) -> bool:
        """执行自愈动作"""
        try:
            if action.action_type == "scale":
                return await self._scale_resources(action.parameters)
            elif action.action_type == "cleanup":
                return await self._cleanup_resources(action.parameters)
            elif action.action_type == "restart":
                return await self._restart_workflow(action.parameters)
            elif action.action_type == "reroute":
                return await self._reroute_workflow(action.parameters)
            else:
                logger.warning(f"未知自愈动作类型: {action.action_type}")
                return False

        except Exception as e:
            logger.error(f"自愈动作执行失败: {e}")
            return False

    async def _scale_resources(self, parameters: Dict[str, Any]) -> bool:
        """扩展资源"""
        # 模拟资源扩展
        logger.info(f"扩展资源: {parameters}")
        await asyncio.sleep(1)  # 模拟执行时间
        return True

    async def _cleanup_resources(self, parameters: Dict[str, Any]) -> bool:
        """清理资源"""
        # 模拟资源清理
        logger.info(f"清理资源: {parameters}")
        await asyncio.sleep(0.5)
        return True

    async def _restart_workflow(self, parameters: Dict[str, Any]) -> bool:
        """重启工作流"""
        # 模拟工作流重启
        logger.info(f"重启工作流: {parameters}")
        await asyncio.sleep(2)
        return True

    async def _reroute_workflow(self, parameters: Dict[str, Any]) -> bool:
        """重路由工作流"""
        # 模拟工作流重路由
        logger.info(f"重路由工作流: {parameters}")
        await asyncio.sleep(1)
        return True

    async def feed_workflow_data(self, workflow_id: str, execution_data: Dict[str, Any]):
        """输入工作流数据"""
        await self.data_stream_monitor.feed_data_stream(
            DataStreamType.WORKFLOW,
            {
                'workflow_id': workflow_id,
                'execution_time': execution_data.get('execution_time', 0),
                'success_rate': execution_data.get('success_rate', 1.0),
                'resource_usage': execution_data.get('resource_usage', {}),
                'error_count': execution_data.get('error_count', 0)
            }
        )

    async def feed_system_data(self, system_data: Dict[str, Any]):
        """输入系统数据"""
        await self.data_stream_monitor.feed_data_stream(
            DataStreamType.SYSTEM,
            {
                'cpu_usage': system_data.get('cpu_usage', 0),
                'memory_usage': system_data.get('memory_usage', 0),
                'disk_usage': system_data.get('disk_usage', 0),
                'network_usage': system_data.get('network_usage', 0)
            }
        )

    async def feed_glucose_data(self, glucose_data: Dict[str, Any]):
        """输入血糖数据"""
        await self.data_stream_monitor.feed_data_stream(
            DataStreamType.GLUCOSE,
            {
                'glucose_level': glucose_data.get('glucose_level', 0),
                'trend': glucose_data.get('trend', 0),
                'variability': glucose_data.get('variability', 0)
            }
        )

    def get_predictive_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """获取预测性告警"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            alert for alert in self.alert_history
            if alert['timestamp'] > cutoff_time
        ]

    def get_heal_action_statistics(self) -> Dict[str, Any]:
        """获取自愈动作统计"""
        stats = {}
        for action_id, action in self.heal_actions.items():
            stats[action_id] = {
                'name': action.name,
                'execution_count': action.execution_count,
                'success_rate': action.success_rate,
                'action_type': action.action_type
            }
        return stats

    def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """获取监控仪表板数据"""
        recent_alerts = self.get_predictive_alerts(1)  # 最近1小时

        return {
            'predictive_alerts': {
                'total': len(recent_alerts),
                'by_severity': self._count_alerts_by_severity(recent_alerts),
                'by_type': self._count_alerts_by_type(recent_alerts)
            },
            'auto_heal': {
                'enabled': self.auto_heal_enabled,
                'actions_executed': sum(action.execution_count for action in self.heal_actions.values()),
                'success_rate': self._calculate_overall_success_rate()
            },
            'data_streams': {
                'active_streams': len(self.data_stream_monitor.data_streams),
                'total_data_points': sum(len(stream) for stream in self.data_stream_monitor.data_streams.values())
            }
        }

    def _count_alerts_by_severity(self, alerts: List[Dict[str, Any]]) -> Dict[str, int]:
        """按严重程度统计告警"""
        counts = {}
        for alert in alerts:
            severity = alert['severity']
            counts[severity] = counts.get(severity, 0) + 1
        return counts

    def _count_alerts_by_type(self, alerts: List[Dict[str, Any]]) -> Dict[str, int]:
        """按类型统计告警"""
        counts = {}
        for alert in alerts:
            alert_type = alert['prediction_type']
            counts[alert_type] = counts.get(alert_type, 0) + 1
        return counts

    def _calculate_overall_success_rate(self) -> float:
        """计算总体成功率"""
        total_executions = sum(action.execution_count for action in self.heal_actions.values())
        if total_executions == 0:
            return 0.0

        total_successes = sum(
            action.success_rate * action.execution_count
            for action in self.heal_actions.values()
        )

        return total_successes / total_executions

# 创建全局增强监控器实例
enhanced_monitor = EnhancedWorkflowMonitor()

logger.info("增强版工作流监控器已创建")

__all__ = ["'logger'", "'AlertSeverity'", "'DataStreamType'", "'PredictionType'", "'PredictiveAlertRule'", "'DataStreamConfig'", "'HealAction'", "'RealTimeDataStreamMonitor'", "'EnhancedWorkflowMonitor'", "'enhanced_monitor'"]
