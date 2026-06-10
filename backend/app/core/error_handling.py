

"""
统一错误处理和监控系统
基于MCP架构的错误处理、监控和告警
"""

import logging
import traceback
import asyncio
import threading
from typing import Dict, List, Optional, Any, Union, Callable, Type
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
import json
import sys
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager

from backend.app.core.dependency_injection import injectable, singleton
from backend.app.core.structured_logging import get_logger

logger = get_logger("error_handling")

class ErrorSeverity(Enum):
    """错误严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """错误类别"""
    VALIDATION = "validation"
    BUSINESS_LOGIC = "business_logic"
    DATABASE = "database"
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    EXTERNAL_API = "external_api"
    SYSTEM = "system"
    UNKNOWN = "unknown"

class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ErrorContext:
    """错误上下文"""
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    module: Optional[str] = None
    function: Optional[str] = None
    line_number: Optional[int] = None
    extra_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ErrorRecord:
    """错误记录"""
    error_id: str
    timestamp: datetime
    error_type: str
    error_message: str
    severity: ErrorSeverity
    category: ErrorCategory
    context: ErrorContext
    stack_trace: Optional[str] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AlertRule:
    """告警规则"""
    rule_id: str
    name: str
    description: str
    condition: Callable[[ErrorRecord], bool]
    alert_level: AlertLevel
    enabled: bool = True
    cooldown_period: int = 300  # 5分钟
    last_triggered: Optional[datetime] = None

@dataclass
class Alert:
    """告警"""
    alert_id: str
    rule_id: str
    error_record: ErrorRecord
    alert_level: AlertLevel
    message: str
    timestamp: datetime
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None

class IErrorHandler(ABC):
    """错误处理器接口"""

    @abstractmethod
    async def handle_error(self, error: Exception, context: ErrorContext) -> ErrorRecord:
        """处理错误"""
        pass

    @abstractmethod
    async def resolve_error(self, error_id: str, resolution_notes: str) -> bool:
        """解决错误"""
        pass

class IAlertHandler(ABC):
    """告警处理器接口"""

    @abstractmethod
    async def send_alert(self, alert: Alert) -> None:
        """发送告警"""
        pass

    @abstractmethod
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """确认告警"""
        pass

@singleton(IErrorHandler)
class DatabaseErrorHandler(IErrorHandler):
    """数据库错误处理器"""

    def __init__(self):
        self._error_records: Dict[str, ErrorRecord] = {}
        self._lock = threading.Lock()

    async def handle_error(self, error: Exception, context: ErrorContext) -> ErrorRecord:
        """处理错误"""
        error_id = f"err_{int(datetime.now().timestamp() * 1000)}"

        # 确定错误类别和严重程度
        category, severity = self._classify_error(error)

        error_record = ErrorRecord(
            error_id=error_id,
            timestamp=datetime.now(),
            error_type=type(error).__name__,
            error_message=str(error),
            severity=severity,
            category=category,
            context=context,
            stack_trace=traceback.format_exc()
        )

        with self._lock:
            self._error_records[error_id] = error_record

        logger.error(f"错误已记录: {error_id}", {
            "error_id": error_id,
            "error_type": error_record.error_type,
            "severity": severity.value,
            "category": category.value
        })

        return error_record

    def _classify_error(self, error: Exception) -> tuple[ErrorCategory, ErrorSeverity]:
        """分类错误"""
        error_name = type(error).__name__.lower()

        if "validation" in error_name or "value" in error_name:
            return ErrorCategory.VALIDATION, ErrorSeverity.MEDIUM
        elif "database" in error_name or "sql" in error_name:
            return ErrorCategory.DATABASE, ErrorSeverity.HIGH
        elif "network" in error_name or "connection" in error_name:
            return ErrorCategory.NETWORK, ErrorSeverity.MEDIUM
        elif "auth" in error_name or "permission" in error_name:
            return ErrorCategory.AUTHENTICATION, ErrorSeverity.HIGH
        elif "business" in error_name or "logic" in error_name:
            return ErrorCategory.BUSINESS_LOGIC, ErrorSeverity.MEDIUM
        else:
            return ErrorCategory.UNKNOWN, ErrorSeverity.MEDIUM

    async def resolve_error(self, error_id: str, resolution_notes: str) -> bool:
        """解决错误"""
        with self._lock:
            if error_id in self._error_records:
                error_record = self._error_records[error_id]
                error_record.resolved = True
                error_record.resolved_at = datetime.now()
                error_record.resolution_notes = resolution_notes

                logger.info(f"错误已解决: {error_id}", {
                    "error_id": error_id,
                    "resolution_notes": resolution_notes
                })

                return True

        return False

    def get_error_record(self, error_id: str) -> Optional[ErrorRecord]:
        """获取错误记录"""
        return self._error_records.get(error_id)

    def get_unresolved_errors(self) -> List[ErrorRecord]:
        """获取未解决的错误"""
        return [record for record in self._error_records.values() if not record.resolved]

    def get_errors_by_severity(self, severity: ErrorSeverity) -> List[ErrorRecord]:
        """按严重程度获取错误"""
        return [record for record in self._error_records.values() if record.severity == severity]

@singleton(IAlertHandler)
class ConsoleAlertHandler(IAlertHandler):
    """控制台告警处理器"""

    def __init__(self):
        self._alerts: Dict[str, Alert] = {}
        self._lock = threading.Lock()

    async def send_alert(self, alert: Alert) -> None:
        """发送告警"""
        with self._lock:
            self._alerts[alert.alert_id] = alert

        # 根据告警级别选择颜色
        color_codes = {
            AlertLevel.INFO: "\033[36m",      # 青色
            AlertLevel.WARNING: "\033[33m",  # 黄色
            AlertLevel.ERROR: "\033[31m",     # 红色
            AlertLevel.CRITICAL: "\033[35m"   # 紫色
        }

        color = color_codes.get(alert.alert_level, "")
        reset = "\033[0m"

        print(f"{color}[{alert.alert_level.value.upper()}] {alert.message}{reset}")
        print(f"  错误ID: {alert.error_record.error_id}")
        print(f"  时间: {alert.timestamp.isoformat()}")
        print(f"  规则ID: {alert.rule_id}")
        print()

    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """确认告警"""
        with self._lock:
            if alert_id in self._alerts:
                alert = self._alerts[alert_id]
                alert.acknowledged = True
                alert.acknowledged_at = datetime.now()
                alert.acknowledged_by = acknowledged_by

                logger.info(f"告警已确认: {alert_id}", {
                    "alert_id": alert_id,
                    "acknowledged_by": acknowledged_by
                })

                return True

        return False

    def get_alerts(self) -> List[Alert]:
        """获取所有告警"""
        return list(self._alerts.values())

    def get_unacknowledged_alerts(self) -> List[Alert]:
        """获取未确认的告警"""
        return [alert for alert in self._alerts.values() if not alert.acknowledged]

class ErrorMonitor:
    """错误监控器"""

    def __init__(self):
        self._error_handler = DatabaseErrorHandler()
        self._alert_handler = ConsoleAlertHandler()
        self._alert_rules: Dict[str, AlertRule] = {}
        self._setup_default_rules()
        self._monitoring_enabled = True
        self._monitor_task: Optional[asyncio.Task] = None

    def _setup_default_rules(self):
        """设置默认告警规则"""
        # 严重错误告警
        self.add_alert_rule(AlertRule(
            rule_id="critical_errors",
            name="严重错误告警",
            description="当发生严重错误时触发告警",
            condition=lambda error: error.severity == ErrorSeverity.CRITICAL,
            alert_level=AlertLevel.CRITICAL
        ))

        # 数据库错误告警
        self.add_alert_rule(AlertRule(
            rule_id="database_errors",
            name="数据库错误告警",
            description="当发生数据库错误时触发告警",
            condition=lambda error: error.category == ErrorCategory.DATABASE,
            alert_level=AlertLevel.ERROR
        ))

        # 高频错误告警
        self.add_alert_rule(AlertRule(
            rule_id="frequent_errors",
            name="高频错误告警",
            description="当短时间内发生多个错误时触发告警",
            condition=self._check_frequent_errors,
            alert_level=AlertLevel.WARNING
        ))

    def _check_frequent_errors(self, error: ErrorRecord) -> bool:
        """检查高频错误"""
        # 检查最近5分钟内是否有超过10个错误
        recent_errors = [
            record for record in self._error_handler._error_records.values()
            if datetime.now() - record.timestamp < timedelta(minutes=5)
        ]

        return len(recent_errors) > 10

    def add_alert_rule(self, rule: AlertRule):
        """添加告警规则"""
        self._alert_rules[rule.rule_id] = rule
        logger.info(f"添加告警规则: {rule.name}")

    def remove_alert_rule(self, rule_id: str):
        """移除告警规则"""
        if rule_id in self._alert_rules:
            del self._alert_rules[rule_id]
            logger.info(f"移除告警规则: {rule_id}")

    async def handle_error(self, error: Exception, context: ErrorContext) -> ErrorRecord:
        """处理错误"""
        error_record = await self._error_handler.handle_error(error, context)

        # 检查告警规则
        await self._check_alert_rules(error_record)

        return error_record

    async def _check_alert_rules(self, error_record: ErrorRecord):
        """检查告警规则"""
        for rule in self._alert_rules.values():
            if not rule.enabled:
                continue

            # 检查冷却期
            if rule.last_triggered:
                time_since_last = datetime.now() - rule.last_triggered
                if time_since_last.total_seconds() < rule.cooldown_period:
                    continue

            # 检查规则条件
            try:
                if rule.condition(error_record):
                    await self._trigger_alert(rule, error_record)
                    rule.last_triggered = datetime.now()
            except Exception as e:
                logger.error(f"检查告警规则失败: {rule.rule_id}", exception=e)

    async def _trigger_alert(self, rule: AlertRule, error_record: ErrorRecord):
        """触发告警"""
        alert_id = f"alert_{int(datetime.now().timestamp() * 1000)}"

        alert = Alert(
            alert_id=alert_id,
            rule_id=rule.rule_id,
            error_record=error_record,
            alert_level=rule.alert_level,
            message=f"[{rule.name}] {error_record.error_message}",
            timestamp=datetime.now()
        )

        await self._alert_handler.send_alert(alert)

        logger.warning(f"告警已触发: {rule.name}", {
            "alert_id": alert_id,
            "rule_id": rule.rule_id,
            "error_id": error_record.error_id
        })

    async def resolve_error(self, error_id: str, resolution_notes: str) -> bool:
        """解决错误"""
        return await self._error_handler.resolve_error(error_id, resolution_notes)

    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """确认告警"""
        return await self._alert_handler.acknowledge_alert(alert_id, acknowledged_by)

    def get_error_statistics(self) -> Dict[str, Any]:
        """获取错误统计"""
        all_errors = list(self._error_handler._error_records.values())

        if not all_errors:
            return {"total_errors": 0}

        # 按严重程度统计
        severity_stats = {}
        for severity in ErrorSeverity:
            severity_stats[severity.value] = len([
                error for error in all_errors if error.severity == severity
            ])

        # 按类别统计
        category_stats = {}
        for category in ErrorCategory:
            category_stats[category.value] = len([
                error for error in all_errors if error.category == category
            ])

        # 按时间统计（最近24小时）
        recent_errors = [
            error for error in all_errors
            if datetime.now() - error.timestamp < timedelta(hours=24)
        ]

        return {
            "total_errors": len(all_errors),
            "unresolved_errors": len([e for e in all_errors if not e.resolved]),
            "recent_errors_24h": len(recent_errors),
            "severity_distribution": severity_stats,
            "category_distribution": category_stats,
            "alerts_count": len(self._alert_handler._alerts),
            "unacknowledged_alerts": len([
                a for a in self._alert_handler._alerts.values() if not a.acknowledged
            ])
        }

    async def start_monitoring(self):
        """启动监控"""
        if self._monitor_task:
            return

        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("错误监控已启动")

    async def stop_monitoring(self):
        """停止监控"""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

        logger.info("错误监控已停止")

    async def _monitoring_loop(self):
        """监控循环"""
        while self._monitoring_enabled:
            try:
                # 检查未解决的错误
                unresolved_errors = self._error_handler.get_unresolved_errors()

                # 检查长时间未解决的错误
                for error in unresolved_errors:
                    if datetime.now() - error.timestamp > timedelta(hours=1):
                        # 创建长时间未解决错误的告警
                        alert_id = f"stale_{error.error_id}"
                        alert = Alert(
                            alert_id=alert_id,
                            rule_id="stale_errors",
                            error_record=error,
                            alert_level=AlertLevel.WARNING,
                            message=f"错误长时间未解决: {error.error_message}",
                            timestamp=datetime.now()
                        )
                        await self._alert_handler.send_alert(alert)

                await asyncio.sleep(60)  # 每分钟检查一次

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"监控循环异常: {e}")
                await asyncio.sleep(60)

# 全局错误监控器
_error_monitor = ErrorMonitor()

def get_error_monitor() -> ErrorMonitor:
    """获取错误监控器"""
    return _error_monitor

async def handle_error(error: Exception, context: ErrorContext) -> ErrorRecord:
    """处理错误"""
    return await _error_monitor.handle_error(error, context)

async def resolve_error(error_id: str, resolution_notes: str) -> bool:
    """解决错误"""
    return await _error_monitor.resolve_error(error_id, resolution_notes)

async def acknowledge_alert(alert_id: str, acknowledged_by: str) -> bool:
    """确认告警"""
    return await _error_monitor.acknowledge_alert(alert_id, acknowledged_by)

def get_error_statistics() -> Dict[str, Any]:
    """获取错误统计"""
    return _error_monitor.get_error_statistics()

# 错误处理装饰器
def handle_exceptions(context: Optional[ErrorContext] = None):
    """异常处理装饰器"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error_context = context or ErrorContext(
                    module=func.__module__,
                    function=func.__name__
                )
                await handle_error(e, error_context)
                raise

        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_context = context or ErrorContext(
                    module=func.__module__,
                    function=func.__name__
                )
                asyncio.create_task(handle_error(e, error_context))
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator

@asynccontextmanager
async def error_context(**kwargs):
    """错误上下文管理器"""
    context = ErrorContext(**kwargs)
    try:
        yield context
    except Exception as e:
        await handle_error(e, context)
        raise

if __name__ == "__main__":
    # 测试错误处理系统
    async def test_error_handling():
        # 启动监控
        await _error_monitor.start_monitoring()

        # 测试不同类型的错误
        try:
            raise ValueError("测试验证错误")
        except Exception as e:
            context = ErrorContext(module="test", function="test_error_handling")
            await handle_error(e, context)

        try:
            raise ConnectionError("测试网络错误")
        except Exception as e:
            context = ErrorContext(module="test", function="test_error_handling")
            await handle_error(e, context)

        # 等待一段时间让监控处理
        await asyncio.sleep(2)

        # 获取统计信息
        stats = get_error_statistics()
        print("错误统计:", json.dumps(stats, indent=2, ensure_ascii=False, default=str))

        # 停止监控
        await _error_monitor.stop_monitoring()

    # 运行测试
    asyncio.run(test_error_handling())

__all__ = ["'logger'", "'ErrorSeverity'", "'ErrorCategory'", "'AlertLevel'", "'ErrorContext'", "'ErrorRecord'", "'AlertRule'", "'Alert'", "'IErrorHandler'", "'IAlertHandler'", "'DatabaseErrorHandler'", "'ConsoleAlertHandler'", "'ErrorMonitor'", "'get_error_monitor'", "'get_error_statistics'", "'handle_exceptions'"]
