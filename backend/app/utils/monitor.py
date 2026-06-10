

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统监控模块
支持性能指标收集和监控
"""

import time
import psutil
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from pathlib import Path
import logging

from backend.app.utilslogger import get_logger

logger = get_logger("monitor")

@dataclass
class SystemMetrics:
    """系统指标数据类"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    memory_available: float
    memory_total: float
    disk_usage: float
    disk_free: float
    disk_total: float
    network_sent: float
    network_recv: float
    gpu_usage: Optional[float] = None
    gpu_memory_usage: Optional[float] = None

@dataclass
class ModelMetrics:
    """模型指标数据类"""
    timestamp: datetime
    predictor_type: str
    response_time: float
    accuracy: Optional[float] = None
    confidence: Optional[float] = None
    error_rate: Optional[float] = None
    throughput: Optional[float] = None

class SystemMonitor:
    """系统监控器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化系统监控器

        Args:
            config: 监控配置
        """
        self.config = config or self._get_default_config()
        self.metrics_history: List[SystemMetrics] = []
        self.model_metrics_history: List[ModelMetrics] = []
        self.is_monitoring = False
        self.monitor_thread = None
        self.alert_callbacks: List[Callable] = []

        # 初始化基准值
        self.baseline_metrics = None
        self._init_baseline()

        logger.info("系统监控器初始化完成")

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "enabled": True,
            "collection_interval": 60,  # 秒
            "history_size": 1000,
            "metrics": {
                "cpu_usage": True,
                "memory_usage": True,
                "gpu_usage": True,
                "disk_usage": True,
                "network_usage": True
            },
            "alert_thresholds": {
                "cpu_threshold": 0.8,
                "memory_threshold": 0.8,
                "disk_threshold": 0.9,
                "gpu_threshold": 0.9
            },
            "save_metrics": True,
            "metrics_file": "logs/system_metrics.json"
        }

    def _init_baseline(self) -> None:
        """初始化基准指标"""
        try:
            self.baseline_metrics = self._collect_system_metrics()
            logger.info("系统基准指标初始化完成")
        except Exception as e:
            logger.error(f"初始化基准指标失败: {e}")

    def _collect_system_metrics(self) -> SystemMetrics:
        """收集系统指标"""
        try:
            # CPU使用率
            cpu_usage = psutil.cpu_percent(interval=1)

            # 内存使用情况
            memory = psutil.virtual_memory()
            memory_usage = memory.percent / 100.0
            memory_available = memory.available / (1024 ** 3)  # GB
            memory_total = memory.total / (1024 ** 3)  # GB

            # 磁盘使用情况
            disk = psutil.disk_usage('/')
            disk_usage = disk.percent / 100.0
            disk_free = disk.free / (1024 ** 3)  # GB
            disk_total = disk.total / (1024 ** 3)  # GB

            # 网络使用情况
            network = psutil.net_io_counters()
            network_sent = network.bytes_sent / (1024 ** 2)  # MB
            network_recv = network.bytes_recv / (1024 ** 2)  # MB

            # GPU使用情况（如果可用）
            gpu_usage = None
            gpu_memory_usage = None
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_usage = gpu_util.gpu / 100.0

                gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_memory_usage = gpu_memory.used / gpu_memory.total
            except ImportError:
                logger.debug("pynvml未安装，跳过GPU监控")
            except Exception as e:
                logger.debug(f"GPU监控失败: {e}")

            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage=cpu_usage / 100.0,
                memory_usage=memory_usage,
                memory_available=memory_available,
                memory_total=memory_total,
                disk_usage=disk_usage,
                disk_free=disk_free,
                disk_total=disk_total,
                network_sent=network_sent,
                network_recv=network_recv,
                gpu_usage=gpu_usage,
                gpu_memory_usage=gpu_memory_usage
            )

        except Exception as e:
            logger.error(f"收集系统指标失败: {e}")
            raise

    def start_monitoring(self) -> None:
        """开始监控"""
        if self.is_monitoring:
            logger.warning("监控已在运行中")
            return

        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("系统监控已启动")

    def stop_monitoring(self) -> None:
        """停止监控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("系统监控已停止")

    def _monitor_loop(self) -> None:
        """监控循环"""
        while self.is_monitoring:
            try:
                # 收集指标
                metrics = self._collect_system_metrics()
                self.metrics_history.append(metrics)

                # 限制历史记录大小
                if len(self.metrics_history) > self.config["history_size"]:
                    self.metrics_history.pop(0)

                # 检查告警
                self._check_alerts(metrics)

                # 保存指标
                if self.config.get("save_metrics", True):
                    self._save_metrics()

                # 等待下次收集
                time.sleep(self.config["collection_interval"])

            except Exception as e:
                logger.error(f"监控循环异常: {e}")
                time.sleep(5)  # 出错时短暂等待

    def _check_alerts(self, metrics: SystemMetrics) -> None:
        """检查告警"""
        thresholds = self.config.get("alert_thresholds", {})

        alerts = []

        # CPU告警
        if metrics.cpu_usage > thresholds.get("cpu_threshold", 0.8):
            alerts.append(f"CPU使用率过高: {metrics.cpu_usage:.2%}")

        # 内存告警
        if metrics.memory_usage > thresholds.get("memory_threshold", 0.8):
            alerts.append(f"内存使用率过高: {metrics.memory_usage:.2%}")

        # 磁盘告警
        if metrics.disk_usage > thresholds.get("disk_threshold", 0.9):
            alerts.append(f"磁盘使用率过高: {metrics.disk_usage:.2%}")

        # GPU告警
        if metrics.gpu_usage and metrics.gpu_usage > thresholds.get("gpu_threshold", 0.9):
            alerts.append(f"GPU使用率过高: {metrics.gpu_usage:.2%}")

        # 触发告警回调
        if alerts:
            for alert in alerts:
                logger.warning(f"系统告警: {alert}")
                self._trigger_alert_callbacks(alert, metrics)

    def _trigger_alert_callbacks(self, alert: str, metrics: SystemMetrics) -> None:
        """触发告警回调"""
        for callback in self.alert_callbacks:
            try:
                callback(alert, metrics)
            except Exception as e:
                logger.error(f"告警回调执行失败: {e}")

    def add_alert_callback(self, callback: Callable) -> None:
        """添加告警回调"""
        self.alert_callbacks.append(callback)

    def _save_metrics(self) -> None:
        """保存指标到文件"""
        try:
            metrics_file = Path(self.config.get("metrics_file", "logs/system_metrics.json"))
            metrics_file.parent.mkdir(parents=True, exist_ok=True)

            # 转换为可序列化的格式
            metrics_data = []
            for metrics in self.metrics_history[-100:]:  # 只保存最近100条
                metrics_data.append({
                    "timestamp": metrics.timestamp.isoformat(),
                    "cpu_usage": metrics.cpu_usage,
                    "memory_usage": metrics.memory_usage,
                    "memory_available": metrics.memory_available,
                    "memory_total": metrics.memory_total,
                    "disk_usage": metrics.disk_usage,
                    "disk_free": metrics.disk_free,
                    "disk_total": metrics.disk_total,
                    "network_sent": metrics.network_sent,
                    "network_recv": metrics.network_recv,
                    "gpu_usage": metrics.gpu_usage,
                    "gpu_memory_usage": metrics.gpu_memory_usage
                })

            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"保存指标失败: {e}")

    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """获取当前指标"""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None

    def get_metrics_summary(self, hours: int = 1) -> Dict[str, Any]:
        """获取指标摘要"""
        if not self.metrics_history:
            return {}

        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_time]

        if not recent_metrics:
            return {}

        # 计算统计信息
        cpu_usage = [m.cpu_usage for m in recent_metrics]
        memory_usage = [m.memory_usage for m in recent_metrics]
        disk_usage = [m.disk_usage for m in recent_metrics]

        summary = {
            "period_hours": hours,
            "data_points": len(recent_metrics),
            "cpu": {
                "current": recent_metrics[-1].cpu_usage,
                "average": sum(cpu_usage) / len(cpu_usage),
                "max": max(cpu_usage),
                "min": min(cpu_usage)
            },
            "memory": {
                "current": recent_metrics[-1].memory_usage,
                "average": sum(memory_usage) / len(memory_usage),
                "max": max(memory_usage),
                "min": min(memory_usage),
                "available_gb": recent_metrics[-1].memory_available,
                "total_gb": recent_metrics[-1].memory_total
            },
            "disk": {
                "current": recent_metrics[-1].disk_usage,
                "average": sum(disk_usage) / len(disk_usage),
                "max": max(disk_usage),
                "min": min(disk_usage),
                "free_gb": recent_metrics[-1].disk_free,
                "total_gb": recent_metrics[-1].disk_total
            }
        }

        # 添加GPU信息
        if recent_metrics[-1].gpu_usage is not None:
            gpu_usage = [m.gpu_usage for m in recent_metrics if m.gpu_usage is not None]
            if gpu_usage:
                summary["gpu"] = {
                    "current": recent_metrics[-1].gpu_usage,
                    "average": sum(gpu_usage) / len(gpu_usage),
                    "max": max(gpu_usage),
                    "min": min(gpu_usage)
                }

        return summary

class ModelMonitor:
    """模型监控器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化模型监控器

        Args:
            config: 监控配置
        """
        self.config = config or self._get_default_config()
        self.metrics_history: List[ModelMetrics] = []
        self.performance_stats: Dict[str, Dict[str, Any]] = {}

        logger.info("模型监控器初始化完成")

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "enabled": True,
            "history_size": 1000,
            "alert_thresholds": {
                "accuracy_threshold": 0.8,
                "response_time_threshold": 1000,  # 毫秒
                "error_rate_threshold": 0.05
            },
            "save_metrics": True,
            "metrics_file": "logs/model_metrics.json"
        }

    def record_prediction(self, predictor_type: str, response_time: float,
                         accuracy: Optional[float] = None, confidence: Optional[float] = None,
                         error: bool = False) -> None:
        """
        记录预测指标

        Args:
            predictor_type: 预测器类型
            response_time: 响应时间（毫秒）
            accuracy: 准确率
            confidence: 置信度
            error: 是否出错
        """
        try:
            metrics = ModelMetrics(
                timestamp=datetime.now(),
                predictor_type=predictor_type,
                response_time=response_time,
                accuracy=accuracy,
                confidence=confidence,
                error_rate=1.0 if error else 0.0
            )

            self.metrics_history.append(metrics)

            # 限制历史记录大小
            if len(self.metrics_history) > self.config["history_size"]:
                self.metrics_history.pop(0)

            # 更新性能统计
            self._update_performance_stats(predictor_type, metrics)

            # 检查告警
            self._check_model_alerts(metrics)

            logger.debug(f"记录模型指标: {predictor_type}, 响应时间: {response_time}ms")

        except Exception as e:
            logger.error(f"记录模型指标失败: {e}")

    def _update_performance_stats(self, predictor_type: str, metrics: ModelMetrics) -> None:
        """更新性能统计"""
        if predictor_type not in self.performance_stats:
            self.performance_stats[predictor_type] = {
                "total_predictions": 0,
                "total_response_time": 0,
                "total_accuracy": 0,
                "total_confidence": 0,
                "total_errors": 0,
                "min_response_time": float('inf'),
                "max_response_time": 0,
                "last_update": datetime.now()
            }

        stats = self.performance_stats[predictor_type]
        stats["total_predictions"] += 1
        stats["total_response_time"] += metrics.response_time
        stats["min_response_time"] = min(stats["min_response_time"], metrics.response_time)
        stats["max_response_time"] = max(stats["max_response_time"], metrics.response_time)

        if metrics.accuracy is not None:
            stats["total_accuracy"] += metrics.accuracy

        if metrics.confidence is not None:
            stats["total_confidence"] += metrics.confidence

        if metrics.error_rate > 0:
            stats["total_errors"] += 1

        stats["last_update"] = datetime.now()

    def _check_model_alerts(self, metrics: ModelMetrics) -> None:
        """检查模型告警"""
        thresholds = self.config.get("alert_thresholds", {})

        alerts = []

        # 响应时间告警
        if metrics.response_time > thresholds.get("response_time_threshold", 1000):
            alerts.append(f"响应时间过长: {metrics.response_time}ms")

        # 准确率告警
        if metrics.accuracy is not None and metrics.accuracy < thresholds.get("accuracy_threshold", 0.8):
            alerts.append(f"准确率过低: {metrics.accuracy:.2%}")

        # 错误率告警
        if metrics.error_rate > thresholds.get("error_rate_threshold", 0.05):
            alerts.append(f"错误率过高: {metrics.error_rate:.2%}")

        # 记录告警
        for alert in alerts:
            logger.warning(f"模型告警 [{metrics.predictor_type}]: {alert}")

    def get_performance_summary(self, predictor_type: Optional[str] = None) -> Dict[str, Any]:
        """获取性能摘要"""
        if predictor_type:
            if predictor_type not in self.performance_stats:
                return {}

            stats = self.performance_stats[predictor_type]
            total = stats["total_predictions"]

            if total == 0:
                return {}

            return {
                "predictor_type": predictor_type,
                "total_predictions": total,
                "average_response_time": stats["total_response_time"] / total,
                "min_response_time": stats["min_response_time"],
                "max_response_time": stats["max_response_time"],
                "average_accuracy": stats["total_accuracy"] / total if stats["total_accuracy"] > 0 else None,
                "average_confidence": stats["total_confidence"] / total if stats["total_confidence"] > 0 else None,
                "error_rate": stats["total_errors"] / total,
                "last_update": stats["last_update"].isoformat()
            }
        else:
            # 返回所有预测器的摘要
            summary = {}
            for pred_type in self.performance_stats:
                summary[pred_type] = self.get_performance_summary(pred_type)
            return summary

    def get_recent_metrics(self, hours: int = 1) -> List[ModelMetrics]:
        """获取最近的指标"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [m for m in self.metrics_history if m.timestamp > cutoff_time]

    def save_metrics(self) -> None:
        """保存指标到文件"""
        try:
            metrics_file = Path(self.config.get("metrics_file", "logs/model_metrics.json"))
            metrics_file.parent.mkdir(parents=True, exist_ok=True)

            # 转换为可序列化的格式
            metrics_data = []
            for metrics in self.metrics_history[-100:]:  # 只保存最近100条
                metrics_data.append({
                    "timestamp": metrics.timestamp.isoformat(),
                    "predictor_type": metrics.predictor_type,
                    "response_time": metrics.response_time,
                    "accuracy": metrics.accuracy,
                    "confidence": metrics.confidence,
                    "error_rate": metrics.error_rate,
                    "throughput": metrics.throughput
                })

            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"保存模型指标失败: {e}")

# 全局监控器实例
_system_monitor = None
_model_monitor = None

def get_system_monitor(config: Optional[Dict[str, Any]] = None) -> SystemMonitor:
    """获取系统监控器实例"""
    global _system_monitor

    if _system_monitor is None:
        _system_monitor = SystemMonitor(config)

    return _system_monitor

def get_model_monitor(config: Optional[Dict[str, Any]] = None) -> ModelMonitor:
    """获取模型监控器实例"""
    global _model_monitor

    if _model_monitor is None:
        _model_monitor = ModelMonitor(config)

    return _model_monitor

def start_monitoring(system_config: Optional[Dict[str, Any]] = None,
                    model_config: Optional[Dict[str, Any]] = None) -> None:
    """启动监控"""
    # 启动系统监控
    system_monitor = get_system_monitor(system_config)
    if system_monitor.config.get("enabled", True):
        system_monitor.start_monitoring()

    # 初始化模型监控
    model_monitor = get_model_monitor(model_config)

    logger.info("监控系统已启动")

def stop_monitoring() -> None:
    """停止监控"""
    global _system_monitor

    if _system_monitor:
        _system_monitor.stop_monitoring()

    logger.info("监控系统已停止")

if __name__ == "__main__":
    # 测试监控系统
    start_monitoring()

    # 模拟一些模型预测
    model_monitor = get_model_monitor()
    model_monitor.record_prediction("glucose_predictor", 150.5, 0.85, 0.92)
    model_monitor.record_prediction("image_predictor", 200.3, 0.78, 0.88)

    # 获取性能摘要
    summary = model_monitor.get_performance_summary()
    print("模型性能摘要:", json.dumps(summary, indent=2, ensure_ascii=False))

    # 等待一段时间后停止
    time.sleep(10)
    stop_monitoring()
__all__ = ["'logger'", "'SystemMetrics'", "'ModelMetrics'", "'SystemMonitor'", "'ModelMonitor'", "'get_system_monitor'", "'get_model_monitor'", "'start_monitoring'", "'stop_monitoring'"]
