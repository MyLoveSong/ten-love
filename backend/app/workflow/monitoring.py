

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工作流监控和报警模块
实现实时监控、性能分析和智能报警
"""

import asyncio
import smtplib
from app.email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
import statistics
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class MetricType(Enum):
    """指标类型"""
    EXECUTION_TIME = "execution_time"
    SUCCESS_RATE = "success_rate"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    RESOURCE_USAGE = "resource_usage"
    QUEUE_LENGTH = "queue_length"

@dataclass
class AlertRule:
    """告警规则"""
    rule_id: str
    name: str
    description: str
    metric_type: MetricType
    condition: str  # 如 "> 30", "< 0.9", "== 0"
    threshold: float
    alert_level: AlertLevel
    enabled: bool = True
    cooldown_minutes: int = 10  # 冷却时间，避免重复告警
    notification_channels: List[str] = field(default_factory=list)

@dataclass
class Alert:
    """告警实例"""
    alert_id: str
    rule_id: str
    alert_level: AlertLevel
    message: str
    metric_value: float
    threshold: float
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    workflow_id: Optional[str] = None
    node_id: Optional[str] = None
    additional_context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkflowMetrics:
    """工作流指标"""
    workflow_id: str
    execution_count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_execution_time: float = 0.0
    avg_execution_time: float = 0.0
    max_execution_time: float = 0.0
    min_execution_time: float = float('inf')
    last_execution_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    last_error_time: Optional[datetime] = None
    throughput_per_hour: float = 0.0
    success_rate: float = 0.0
    error_rate: float = 0.0

class WorkflowMonitor:
    """工作流监控器"""

    def __init__(self):
        self.metrics: Dict[str, WorkflowMetrics] = {}
        self.node_metrics: Dict[str, Dict[str, WorkflowMetrics]] = {}
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_handlers: Dict[str, Callable] = {}

        # 性能数据缓存（最近1小时的数据）
        self.execution_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.success_counts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=60))  # 每分钟一个数据点

        # 初始化默认告警规则
        self._initialize_default_alert_rules()

        # 启动后台监控任务
        self._start_monitoring_tasks()

    def _initialize_default_alert_rules(self):
        """初始化默认告警规则"""
        # 执行时间过长告警
        self.alert_rules["execution_time_high"] = AlertRule(
            rule_id="execution_time_high",
            name="执行时间过长",
            description="工作流执行时间超过阈值",
            metric_type=MetricType.EXECUTION_TIME,
            condition="> 300",  # 5分钟
            threshold=300.0,
            alert_level=AlertLevel.WARNING,
            notification_channels=["email", "webhook"]
        )

        # 成功率过低告警
        self.alert_rules["success_rate_low"] = AlertRule(
            rule_id="success_rate_low",
            name="成功率过低",
            description="工作流成功率低于阈值",
            metric_type=MetricType.SUCCESS_RATE,
            condition="< 0.9",
            threshold=0.9,
            alert_level=AlertLevel.ERROR,
            notification_channels=["email", "webhook"]
        )

        # 错误率过高告警
        self.alert_rules["error_rate_high"] = AlertRule(
            rule_id="error_rate_high",
            name="错误率过高",
            description="工作流错误率超过阈值",
            metric_type=MetricType.ERROR_RATE,
            condition="> 0.1",
            threshold=0.1,
            alert_level=AlertLevel.CRITICAL,
            notification_channels=["email", "webhook", "sms"]
        )

        # 吞吐量过低告警
        self.alert_rules["throughput_low"] = AlertRule(
            rule_id="throughput_low",
            name="吞吐量过低",
            description="工作流吞吐量低于预期",
            metric_type=MetricType.THROUGHPUT,
            condition="< 10",  # 每小时少于10次执行
            threshold=10.0,
            alert_level=AlertLevel.WARNING,
            notification_channels=["email"]
        )

        logger.info(f"初始化 {len(self.alert_rules)} 个默认告警规则")

    def _start_monitoring_tasks(self):
        """启动后台监控任务"""
        # 这里应该启动异步任务，但为了简化，我们提供方法供外部调用
        logger.info("监控任务准备就绪")

    def record_workflow_execution(self, workflow_id: str, execution_time: float,
                                success: bool, error_message: Optional[str] = None):
        """记录工作流执行"""
        if workflow_id not in self.metrics:
            self.metrics[workflow_id] = WorkflowMetrics(workflow_id=workflow_id)

        metrics = self.metrics[workflow_id]
        current_time = datetime.now()

        # 更新基础指标
        metrics.execution_count += 1
        metrics.total_execution_time += execution_time
        metrics.last_execution_time = current_time

        if success:
            metrics.success_count += 1
            metrics.last_success_time = current_time
        else:
            metrics.error_count += 1
            metrics.last_error_time = current_time

        # 更新聚合指标
        metrics.avg_execution_time = metrics.total_execution_time / metrics.execution_count
        metrics.max_execution_time = max(metrics.max_execution_time, execution_time)
        metrics.min_execution_time = min(metrics.min_execution_time, execution_time)
        metrics.success_rate = metrics.success_count / metrics.execution_count
        metrics.error_rate = metrics.error_count / metrics.execution_count

        # 计算吞吐量（过去1小时）
        hour_ago = current_time - timedelta(hours=1)
        recent_executions = sum(1 for t in self.execution_times[workflow_id]
                              if datetime.fromtimestamp(t) > hour_ago)
        metrics.throughput_per_hour = recent_executions

        # 记录执行时间用于趋势分析
        self.execution_times[workflow_id].append(current_time.timestamp())

        # 检查告警条件
        self._check_alerts(workflow_id, metrics)

        logger.debug(f"记录工作流执行: {workflow_id}, 耗时: {execution_time:.2f}s, 成功: {success}")

    def record_node_execution(self, workflow_id: str, node_id: str,
                            execution_time: float, success: bool):
        """记录节点执行"""
        if workflow_id not in self.node_metrics:
            self.node_metrics[workflow_id] = {}

        if node_id not in self.node_metrics[workflow_id]:
            self.node_metrics[workflow_id][node_id] = WorkflowMetrics(workflow_id=f"{workflow_id}_{node_id}")

        metrics = self.node_metrics[workflow_id][node_id]
        current_time = datetime.now()

        # 更新节点指标
        metrics.execution_count += 1
        metrics.total_execution_time += execution_time
        metrics.last_execution_time = current_time

        if success:
            metrics.success_count += 1
            metrics.last_success_time = current_time
        else:
            metrics.error_count += 1
            metrics.last_error_time = current_time

        # 更新聚合指标
        metrics.avg_execution_time = metrics.total_execution_time / metrics.execution_count
        metrics.success_rate = metrics.success_count / metrics.execution_count
        metrics.error_rate = metrics.error_count / metrics.execution_count

    def _check_alerts(self, workflow_id: str, metrics: WorkflowMetrics):
        """检查告警条件"""
        current_time = datetime.now()

        for rule_id, rule in self.alert_rules.items():
            if not rule.enabled:
                continue

            # 检查冷却时间
            alert_key = f"{workflow_id}_{rule_id}"
            if alert_key in self.active_alerts:
                last_alert_time = self.active_alerts[alert_key].triggered_at
                if (current_time - last_alert_time).total_seconds() < rule.cooldown_minutes * 60:
                    continue

            # 获取指标值
            metric_value = self._get_metric_value(metrics, rule.metric_type)

            # 检查条件
            if self._evaluate_condition(metric_value, rule.condition, rule.threshold):
                self._trigger_alert(workflow_id, rule, metric_value)

    def _get_metric_value(self, metrics: WorkflowMetrics, metric_type: MetricType) -> float:
        """获取指标值"""
        if metric_type == MetricType.EXECUTION_TIME:
            return metrics.avg_execution_time
        elif metric_type == MetricType.SUCCESS_RATE:
            return metrics.success_rate
        elif metric_type == MetricType.ERROR_RATE:
            return metrics.error_rate
        elif metric_type == MetricType.THROUGHPUT:
            return metrics.throughput_per_hour
        else:
            return 0.0

    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """评估告警条件"""
        if condition.startswith(">"):
            return value > threshold
        elif condition.startswith("<"):
            return value < threshold
        elif condition.startswith("=="):
            return abs(value - threshold) < 0.001
        elif condition.startswith(">="):
            return value >= threshold
        elif condition.startswith("<="):
            return value <= threshold
        else:
            return False

    def _trigger_alert(self, workflow_id: str, rule: AlertRule, metric_value: float):
        """触发告警"""
        import uuid

        alert_id = str(uuid.uuid4())[:8]
        alert = Alert(
            alert_id=alert_id,
            rule_id=rule.rule_id,
            alert_level=rule.alert_level,
            message=f"{rule.name}: {rule.description}，当前值: {metric_value:.3f}，阈值: {rule.threshold}",
            metric_value=metric_value,
            threshold=rule.threshold,
            triggered_at=datetime.now(),
            workflow_id=workflow_id,
            additional_context={
                "metric_type": rule.metric_type.value,
                "condition": rule.condition
            }
        )

        alert_key = f"{workflow_id}_{rule.rule_id}"
        self.active_alerts[alert_key] = alert
        self.alert_history.append(alert)

        # 发送通知
        self._send_notifications(alert, rule.notification_channels)

        logger.warning(f"触发告警: {alert.message}")

    def _send_notifications(self, alert: Alert, channels: List[str]):
        """发送通知"""
        for channel in channels:
            try:
                if channel in self.notification_handlers:
                    self.notification_handlers[channel](alert)
                else:
                    logger.warning(f"未找到通知渠道处理器: {channel}")
            except Exception as e:
                logger.error(f"发送通知失败 ({channel}): {e}")

    def register_notification_handler(self, channel: str, handler: Callable):
        """注册通知处理器"""
        self.notification_handlers[channel] = handler
        logger.info(f"注册通知处理器: {channel}")

    def resolve_alert(self, alert_id: str, resolution_note: str = ""):
        """解决告警"""
        # 从活跃告警中移除
        alert_key_to_remove = None
        for alert_key, alert in self.active_alerts.items():
            if alert.alert_id == alert_id:
                alert.resolved_at = datetime.now()
                alert.additional_context["resolution_note"] = resolution_note
                alert_key_to_remove = alert_key
                break

        if alert_key_to_remove:
            del self.active_alerts[alert_key_to_remove]
            logger.info(f"告警已解决: {alert_id}")
        else:
            logger.warning(f"未找到告警: {alert_id}")

    def get_workflow_metrics(self, workflow_id: str) -> Optional[WorkflowMetrics]:
        """获取工作流指标"""
        return self.metrics.get(workflow_id)

    def get_node_metrics(self, workflow_id: str, node_id: str) -> Optional[WorkflowMetrics]:
        """获取节点指标"""
        if workflow_id in self.node_metrics and node_id in self.node_metrics[workflow_id]:
            return self.node_metrics[workflow_id][node_id]
        return None

    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警"""
        return list(self.active_alerts.values())

    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """获取告警历史"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert.triggered_at > cutoff_time]

    def get_performance_dashboard_data(self) -> Dict[str, Any]:
        """获取性能仪表板数据"""
        dashboard_data = {
            "overview": {
                "total_workflows": len(self.metrics),
                "active_alerts": len(self.active_alerts),
                "total_executions": sum(m.execution_count for m in self.metrics.values()),
                "avg_success_rate": statistics.mean([m.success_rate for m in self.metrics.values()]) if self.metrics else 0
            },
            "workflows": [],
            "alerts": []
        }

        # 工作流性能数据
        for workflow_id, metrics in self.metrics.items():
            workflow_data = {
                "workflow_id": workflow_id,
                "execution_count": metrics.execution_count,
                "success_rate": metrics.success_rate,
                "avg_execution_time": metrics.avg_execution_time,
                "throughput_per_hour": metrics.throughput_per_hour,
                "last_execution": metrics.last_execution_time.isoformat() if metrics.last_execution_time else None
            }
            dashboard_data["workflows"].append(workflow_data)

        # 活跃告警数据
        for alert in self.active_alerts.values():
            alert_data = {
                "alert_id": alert.alert_id,
                "level": alert.alert_level.value,
                "message": alert.message,
                "workflow_id": alert.workflow_id,
                "triggered_at": alert.triggered_at.isoformat()
            }
            dashboard_data["alerts"].append(alert_data)

        return dashboard_data

    def export_metrics_report(self, start_time: datetime, end_time: datetime) -> str:
        """导出指标报告"""
        report_data = {
            "report_period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "summary": {
                "total_workflows_monitored": len(self.metrics),
                "total_alerts_triggered": len([a for a in self.alert_history
                                             if start_time <= a.triggered_at <= end_time])
            },
            "workflow_metrics": {},
            "alert_summary": {}
        }

        # 工作流指标详情
        for workflow_id, metrics in self.metrics.items():
            report_data["workflow_metrics"][workflow_id] = {
                "execution_count": metrics.execution_count,
                "success_rate": metrics.success_rate,
                "error_rate": metrics.error_rate,
                "avg_execution_time": metrics.avg_execution_time,
                "max_execution_time": metrics.max_execution_time,
                "min_execution_time": metrics.min_execution_time if metrics.min_execution_time != float('inf') else 0,
                "throughput_per_hour": metrics.throughput_per_hour
            }

        # 告警汇总
        period_alerts = [a for a in self.alert_history if start_time <= a.triggered_at <= end_time]
        alert_counts = defaultdict(int)
        for alert in period_alerts:
            alert_counts[alert.alert_level.value] += 1

        report_data["alert_summary"] = dict(alert_counts)

        return json.dumps(report_data, indent=2, ensure_ascii=False)

# 默认通知处理器

def email_notification_handler(alert: Alert):
    """邮件通知处理器"""
    # 这里应该配置实际的SMTP服务器
    logger.info(f"邮件通知: {alert.message}")

def webhook_notification_handler(alert: Alert):
    """Webhook通知处理器"""
    # 这里应该发送HTTP请求到配置的webhook URL
    logger.info(f"Webhook通知: {alert.message}")

def sms_notification_handler(alert: Alert):
    """短信通知处理器"""
    # 这里应该调用短信服务API
    logger.info(f"短信通知: {alert.message}")

# 创建全局监控器实例
workflow_monitor = WorkflowMonitor()

# 注册默认通知处理器
workflow_monitor.register_notification_handler("email", email_notification_handler)
workflow_monitor.register_notification_handler("webhook", webhook_notification_handler)
workflow_monitor.register_notification_handler("sms", sms_notification_handler)

logger.info("工作流监控器已创建并注册默认通知处理器")

__all__ = ["'logger'", "'AlertLevel'", "'MetricType'", "'AlertRule'", "'Alert'", "'WorkflowMetrics'", "'WorkflowMonitor'", "'email_notification_handler'", "'webhook_notification_handler'", "'sms_notification_handler'", "'workflow_monitor'"]
