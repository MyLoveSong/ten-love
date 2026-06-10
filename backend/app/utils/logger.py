

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
结构化日志系统
支持文件轮转、JSON格式和不同日志级别
"""

import logging
import logging.handlers
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime
import traceback
import os

# 自定义JSON格式化器
class JSONFormatter(logging.Formatter):
    """JSON格式日志格式化器"""

    def __init__(self, include_timestamp: bool = True, include_level: bool = True):
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_level = True

    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录为JSON"""
        log_entry = {
            "message": record.getMessage(),
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        if self.include_timestamp:
            log_entry["timestamp"] = datetime.fromtimestamp(record.created).isoformat()

        if self.include_level:
            log_entry["level"] = record.levelname

        # 添加异常信息
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }

        # 添加额外字段
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)

        return json.dumps(log_entry, ensure_ascii=False, default=str)

class StructuredLogger:
    """结构化日志管理器"""

    def __init__(self, name: str = "academic_system", config: Optional[Dict[str, Any]] = None):
        """
        初始化日志管理器

        Args:
            name: 日志器名称
            config: 日志配置
        """
        self.name = name
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(name)
        self._setup_logger()

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": {
                "enabled": True,
                "path": "logs/academic_system.log",
                "max_size": "10MB",
                "backup_count": 5,
                "encoding": "utf-8"
            },
            "console": {
                "enabled": True,
                "level": "INFO"
            },
            "structured": {
                "enabled": True,
                "format": "json"
            }
        }

    def _setup_logger(self) -> None:
        """设置日志器"""
        # 清除现有处理器
        self.logger.handlers.clear()

        # 设置日志级别
        level = getattr(logging, self.config.get("level", "INFO").upper())
        self.logger.setLevel(level)

        # 文件处理器
        if self.config.get("file", {}).get("enabled", True):
            self._setup_file_handler()

        # 控制台处理器
        if self.config.get("console", {}).get("enabled", True):
            self._setup_console_handler()

        # 结构化日志处理器
        if self.config.get("structured", {}).get("enabled", True):
            self._setup_structured_handler()

    def _setup_file_handler(self) -> None:
        """设置文件处理器"""
        try:
            file_config = self.config.get("file", {})
            log_path = Path(file_config.get("path", "logs/academic_system.log"))

            # 创建日志目录
            log_path.parent.mkdir(parents=True, exist_ok=True)

            # 创建轮转文件处理器
            max_bytes = self._parse_size(file_config.get("max_size", "10MB"))
            backup_count = file_config.get("backup_count", 5)
            encoding = file_config.get("encoding", "utf-8")

            file_handler = logging.handlers.RotatingFileHandler(
                filename=log_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding=encoding
            )

            # 设置格式
            formatter = logging.Formatter(self.config.get("format"))
            file_handler.setFormatter(formatter)

            # 设置级别
            file_level = getattr(logging, file_config.get("level", "INFO").upper())
            file_handler.setLevel(file_level)

            self.logger.addHandler(file_handler)

        except Exception as e:
            print(f"设置文件日志处理器失败: {e}")

    def _setup_console_handler(self) -> None:
        """设置控制台处理器"""
        try:
            console_config = self.config.get("console", {})

            console_handler = logging.StreamHandler(sys.stdout)

            # 设置格式
            formatter = logging.Formatter(self.config.get("format"))
            console_handler.setFormatter(formatter)

            # 设置级别
            console_level = getattr(logging, console_config.get("level", "INFO").upper())
            console_handler.setLevel(console_level)

            self.logger.addHandler(console_handler)

        except Exception as e:
            print(f"设置控制台日志处理器失败: {e}")

    def _setup_structured_handler(self) -> None:
        """设置结构化日志处理器"""
        try:
            structured_config = self.config.get("structured", {})

            # 创建结构化日志文件
            structured_path = Path("logs/structured.log")
            structured_path.parent.mkdir(parents=True, exist_ok=True)

            structured_handler = logging.handlers.RotatingFileHandler(
                filename=structured_path,
                maxBytes=self._parse_size("10MB"),
                backupCount=5,
                encoding="utf-8"
            )

            # 使用JSON格式化器
            json_formatter = JSONFormatter()
            structured_handler.setFormatter(json_formatter)

            # 设置级别
            structured_level = getattr(logging, structured_config.get("level", "INFO").upper())
            structured_handler.setLevel(structured_level)

            self.logger.addHandler(structured_handler)

        except Exception as e:
            print(f"设置结构化日志处理器失败: {e}")

    def _parse_size(self, size_str: str) -> int:
        """解析大小字符串"""
        size_str = size_str.upper()
        if size_str.endswith('KB'):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith('MB'):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith('GB'):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            return int(size_str)

    def log_with_context(self, level: str, message: str, **kwargs) -> None:
        """
        记录带上下文的日志

        Args:
            level: 日志级别
            message: 日志消息
            **kwargs: 额外字段
        """
        # 创建自定义日志记录
        record = self.logger.makeRecord(
            name=self.name,
            level=getattr(logging, level.upper()),
            fn="",
            lno=0,
            msg=message,
            args=(),
            exc_info=None
        )

        # 添加额外字段
        record.extra_fields = kwargs

        # 记录日志
        self.logger.handle(record)

    def info(self, message: str, **kwargs) -> None:
        """记录信息日志"""
        self.log_with_context("INFO", message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """记录警告日志"""
        self.log_with_context("WARNING", message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        """记录错误日志"""
        self.log_with_context("ERROR", message, **kwargs)

    def debug(self, message: str, **kwargs) -> None:
        """记录调试日志"""
        self.log_with_context("DEBUG", message, **kwargs)

    def critical(self, message: str, **kwargs) -> None:
        """记录严重错误日志"""
        self.log_with_context("CRITICAL", message, **kwargs)

    def log_prediction(self, predictor_type: str, input_data: Dict[str, Any],
                      output_data: Dict[str, Any], duration: float, **kwargs) -> None:
        """
        记录预测日志

        Args:
            predictor_type: 预测器类型
            input_data: 输入数据
            output_data: 输出数据
            duration: 执行时间
            **kwargs: 额外字段
        """
        self.info(
            f"预测完成: {predictor_type}",
            predictor_type=predictor_type,
            input_data=input_data,
            output_data=output_data,
            duration=duration,
            **kwargs
        )

    def log_experiment(self, experiment_name: str, config: Dict[str, Any],
                      results: Dict[str, Any], **kwargs) -> None:
        """
        记录实验日志

        Args:
            experiment_name: 实验名称
            config: 实验配置
            results: 实验结果
            **kwargs: 额外字段
        """
        self.info(
            f"实验完成: {experiment_name}",
            experiment_name=experiment_name,
            config=config,
            results=results,
            **kwargs
        )

    def log_system_event(self, event_type: str, event_data: Dict[str, Any], **kwargs) -> None:
        """
        记录系统事件

        Args:
            event_type: 事件类型
            event_data: 事件数据
            **kwargs: 额外字段
        """
        self.info(
            f"系统事件: {event_type}",
            event_type=event_type,
            event_data=event_data,
            **kwargs
        )

    def log_performance(self, metric_name: str, value: float, unit: str = "", **kwargs) -> None:
        """
        记录性能指标

        Args:
            metric_name: 指标名称
            value: 指标值
            unit: 单位
            **kwargs: 额外字段
        """
        self.info(
            f"性能指标: {metric_name} = {value}{unit}",
            metric_name=metric_name,
            value=value,
            unit=unit,
            **kwargs
        )

    def get_logger(self) -> logging.Logger:
        """获取原始日志器"""
        return self.logger

# 全局日志管理器实例
_global_logger = None

def get_logger(name: str = "academic_system", config: Optional[Dict[str, Any]] = None) -> StructuredLogger:
    """
    获取日志管理器实例

    Args:
        name: 日志器名称
        config: 日志配置

    Returns:
        StructuredLogger: 日志管理器实例
    """
    global _global_logger

    if _global_logger is None:
        _global_logger = StructuredLogger(name, config)

    return _global_logger

def setup_logging(config: Dict[str, Any]) -> StructuredLogger:
    """
    设置日志系统

    Args:
        config: 日志配置

    Returns:
        StructuredLogger: 日志管理器实例
    """
    return get_logger(config=config)

# 便捷函数
def log_info(message: str, **kwargs) -> None:
    """记录信息日志"""
    get_logger().info(message, **kwargs)

def log_warning(message: str, **kwargs) -> None:
    """记录警告日志"""
    get_logger().warning(message, **kwargs)

def log_error(message: str, **kwargs) -> None:
    """记录错误日志"""
    get_logger().error(message, **kwargs)

def log_debug(message: str, **kwargs) -> None:
    """记录调试日志"""
    get_logger().debug(message, **kwargs)

def log_critical(message: str, **kwargs) -> None:
    """记录严重错误日志"""
    get_logger().critical(message, **kwargs)

if __name__ == "__main__":
    # 测试日志系统
    logger = get_logger("test_logger")

    logger.info("这是一条信息日志")
    logger.warning("这是一条警告日志")
    logger.error("这是一条错误日志")

    # 测试带上下文的日志
    logger.log_prediction(
        predictor_type="glucose_predictor",
        input_data={"glucose": 120, "insulin": 10},
        output_data={"prediction": 125, "confidence": 0.85},
        duration=0.1
    )

    logger.log_experiment(
        experiment_name="comparative_study",
        config={"models": ["model1", "model2"]},
        results={"accuracy": 0.95, "f1_score": 0.92}
    )

    logger.log_performance("response_time", 150.5, "ms")
__all__ = ["'JSONFormatter'", "'StructuredLogger'", "'get_logger'", "'setup_logging'", "'log_info'", "'log_warning'", "'log_error'", "'log_debug'", "'log_critical'"]
