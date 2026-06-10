

"""
统一日志管理系统
基于MCP架构的日志管理和结构化日志
"""

import logging
import logging.handlers
import json
import sys
import os
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
import threading
import traceback
from pathlib import Path
import asyncio
from abc import ABC, abstractmethod

from backend.app.core.dependency_injection import injectable, singleton
from backend.app.core.configuration import get_configuration

logger = logging.getLogger(__name__)

class LogLevel(Enum):
    """日志级别"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class LogFormat(Enum):
    """日志格式"""
    SIMPLE = "simple"
    DETAILED = "detailed"
    JSON = "json"
    STRUCTURED = "structured"

class LogDestination(Enum):
    """日志目标"""
    CONSOLE = "console"
    FILE = "file"
    DATABASE = "database"
    REMOTE = "remote"
    SYSLOG = "syslog"

@dataclass
class LogEntry:
    """日志条目"""
    timestamp: datetime
    level: LogLevel
    logger_name: str
    message: str
    module: str
    function: str
    line_number: int
    thread_id: int
    process_id: int
    extra_data: Dict[str, Any]
    exception_info: Optional[str] = None
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    correlation_id: Optional[str] = None

@dataclass
class LogConfiguration:
    """日志配置"""
    level: LogLevel = LogLevel.INFO
    format: LogFormat = LogFormat.STRUCTURED
    destinations: List[LogDestination] = None
    file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    enable_console: bool = True
    enable_file: bool = True
    enable_remote: bool = False
    remote_endpoint: Optional[str] = None
    buffer_size: int = 1000
    flush_interval: int = 5  # seconds

class ILogHandler(ABC):
    """日志处理器接口"""

    @abstractmethod
    async def handle_log(self, log_entry: LogEntry) -> None:
        """处理日志条目"""
        pass

    @abstractmethod
    async def flush(self) -> None:
        """刷新日志"""
        pass

    @abstractmethod
    async def close(self) -> None:
        """关闭处理器"""
        pass

@singleton(ILogHandler)
class ConsoleLogHandler(ILogHandler):
    """控制台日志处理器"""

    def __init__(self, config: LogConfiguration):
        self.config = config
        self._formatter = self._create_formatter()

    def _create_formatter(self):
        """创建格式化器"""
        if self.config.format == LogFormat.JSON:
            return self._json_formatter
        elif self.config.format == LogFormat.STRUCTURED:
            return self._structured_formatter
        else:
            return self._simple_formatter

    def _simple_formatter(self, log_entry: LogEntry) -> str:
        """简单格式化器"""
        return f"{log_entry.timestamp.isoformat()} [{log_entry.level.value}] {log_entry.logger_name}: {log_entry.message}"

    def _structured_formatter(self, log_entry: LogEntry) -> str:
        """结构化格式化器"""
        return f"{log_entry.timestamp.isoformat()} [{log_entry.level.value}] {log_entry.logger_name} {log_entry.module}:{log_entry.function}:{log_entry.line_number} - {log_entry.message}"

    def _json_formatter(self, log_entry: LogEntry) -> str:
        """JSON格式化器"""
        data = asdict(log_entry)
        data['timestamp'] = log_entry.timestamp.isoformat()
        return json.dumps(data, ensure_ascii=False)

    async def handle_log(self, log_entry: LogEntry) -> None:
        """处理日志条目"""
        formatted_message = self._formatter(log_entry)

        if log_entry.level == LogLevel.DEBUG:
            print(f"\033[36m{formatted_message}\033[0m")  # 青色
        elif log_entry.level == LogLevel.INFO:
            print(f"\033[32m{formatted_message}\033[0m")  # 绿色
        elif log_entry.level == LogLevel.WARNING:
            print(f"\033[33m{formatted_message}\033[0m")  # 黄色
        elif log_entry.level == LogLevel.ERROR:
            print(f"\033[31m{formatted_message}\033[0m")  # 红色
        elif log_entry.level == LogLevel.CRITICAL:
            print(f"\033[35m{formatted_message}\033[0m")  # 紫色

    async def flush(self) -> None:
        """刷新日志"""
        sys.stdout.flush()
        sys.stderr.flush()

    async def close(self) -> None:
        """关闭处理器"""
        await self.flush()

@singleton(ILogHandler)
class FileLogHandler(ILogHandler):
    """文件日志处理器"""

    def __init__(self, config: LogConfiguration):
        self.config = config
        self._file_path = Path(config.file_path or "logs/app.log")
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        self._formatter = self._create_formatter()
        self._file_lock = threading.Lock()

    def _create_formatter(self):
        """创建格式化器"""
        if self.config.format == LogFormat.JSON:
            return self._json_formatter
        elif self.config.format == LogFormat.STRUCTURED:
            return self._structured_formatter
        else:
            return self._simple_formatter

    def _simple_formatter(self, log_entry: LogEntry) -> str:
        """简单格式化器"""
        return f"{log_entry.timestamp.isoformat()} [{log_entry.level.value}] {log_entry.logger_name}: {log_entry.message}\n"

    def _structured_formatter(self, log_entry: LogEntry) -> str:
        """结构化格式化器"""
        return f"{log_entry.timestamp.isoformat()} [{log_entry.level.value}] {log_entry.logger_name} {log_entry.module}:{log_entry.function}:{log_entry.line_number} - {log_entry.message}\n"

    def _json_formatter(self, log_entry: LogEntry) -> str:
        """JSON格式化器"""
        data = asdict(log_entry)
        data['timestamp'] = log_entry.timestamp.isoformat()
        return json.dumps(data, ensure_ascii=False) + "\n"

    async def handle_log(self, log_entry: LogEntry) -> None:
        """处理日志条目"""
        formatted_message = self._formatter(log_entry)

        with self._file_lock:
            try:
                with open(self._file_path, 'a', encoding='utf-8') as f:
                    f.write(formatted_message)
            except Exception as e:
                print(f"写入日志文件失败: {e}")

    async def flush(self) -> None:
        """刷新日志"""
        pass  # 文件写入是同步的

    async def close(self) -> None:
        """关闭处理器"""
        await self.flush()

@singleton(ILogHandler)
class RemoteLogHandler(ILogHandler):
    """远程日志处理器"""

    def __init__(self, config: LogConfiguration):
        self.config = config
        self._endpoint = config.remote_endpoint
        self._buffer: List[LogEntry] = []
        self._buffer_lock = threading.Lock()
        self._flush_task: Optional[asyncio.Task] = None

    async def handle_log(self, log_entry: LogEntry) -> None:
        """处理日志条目"""
        with self._buffer_lock:
            self._buffer.append(log_entry)

            if len(self._buffer) >= self.config.buffer_size:
                await self._flush_buffer()

    async def _flush_buffer(self) -> None:
        """刷新缓冲区"""
        if not self._buffer:
            return

        with self._buffer_lock:
            logs_to_send = self._buffer.copy()
            self._buffer.clear()

        try:
            # 发送到远程端点
            await self._send_logs(logs_to_send)
        except Exception as e:
            logger.error(f"发送远程日志失败: {e}")
            # 重新添加到缓冲区
            with self._buffer_lock:
                self._buffer.extend(logs_to_send)

    async def _send_logs(self, logs: List[LogEntry]) -> None:
        """发送日志"""
        # 这里应该实现实际的远程发送逻辑
        # 例如发送到ELK、Fluentd等
        pass

    async def flush(self) -> None:
        """刷新日志"""
        await self._flush_buffer()

    async def close(self) -> None:
        """关闭处理器"""
        if self._flush_task:
            self._flush_task.cancel()
        await self.flush()

class StructuredLogger:
    """结构化日志记录器"""

    def __init__(self, name: str, config: LogConfiguration):
        self.name = name
        self.config = config
        self._handlers: List[ILogHandler] = []
        self._setup_handlers()

    def _setup_handlers(self):
        """设置处理器"""
        if self.config.enable_console:
            console_handler = ConsoleLogHandler(self.config)
            self._handlers.append(console_handler)

        if self.config.enable_file:
            file_handler = FileLogHandler(self.config)
            self._handlers.append(file_handler)

        if self.config.enable_remote and self.config.remote_endpoint:
            remote_handler = RemoteLogHandler(self.config)
            self._handlers.append(remote_handler)

    def _create_log_entry(self, level: LogLevel, message: str, extra_data: Dict[str, Any] = None,
                         exception: Optional[Exception] = None) -> LogEntry:
        """创建日志条目"""
        frame = sys._getframe(2)  # 获取调用者的帧

        return LogEntry(
            timestamp=datetime.now(),
            level=level,
            logger_name=self.name,
            message=message,
            module=frame.f_globals.get('__name__', 'unknown'),
            function=frame.f_code.co_name,
            line_number=frame.f_lineno,
            thread_id=threading.get_ident(),
            process_id=os.getpid(),
            extra_data=extra_data or {},
            exception_info=traceback.format_exc() if exception else None
        )

    async def _log(self, level: LogLevel, message: str, extra_data: Dict[str, Any] = None,
                  exception: Optional[Exception] = None):
        """记录日志"""
        if level.value not in [l.value for l in LogLevel]:
            return

        # 检查日志级别
        if self._should_log(level):
            log_entry = self._create_log_entry(level, message, extra_data, exception)

            # 异步处理日志
            for handler in self._handlers:
                try:
                    await handler.handle_log(log_entry)
                except Exception as e:
                    # 避免在程序关闭时输出到已关闭的文件
                    try:
                        print(f"日志处理器错误: {e}")
                    except (ValueError, OSError):
                        pass  # 忽略文件已关闭的错误

    async def shutdown(self):
        """优雅关闭日志系统"""
        try:
            # 等待所有待处理的日志任务完成
            await asyncio.sleep(0.1)  # 给异步任务一些时间完成

            # 关闭所有处理器
            for handler in self._handlers:
                if hasattr(handler, 'shutdown'):
                    await handler.shutdown()
        except Exception:
            pass  # 忽略关闭时的错误

    def _should_log(self, level: LogLevel) -> bool:
        """检查是否应该记录日志"""
        level_order = {
            LogLevel.DEBUG: 0,
            LogLevel.INFO: 1,
            LogLevel.WARNING: 2,
            LogLevel.ERROR: 3,
            LogLevel.CRITICAL: 4
        }

        return level_order[level] >= level_order[self.config.level]

    def debug(self, message: str, extra_data: Dict[str, Any] = None):
        """调试日志"""
        asyncio.create_task(self._log(LogLevel.DEBUG, message, extra_data))

    def info(self, message: str, extra_data: Dict[str, Any] = None):
        """信息日志"""
        try:
            # 检查是否有运行中的事件循环
            loop = asyncio.get_running_loop()
            if loop.is_running():
                asyncio.create_task(self._log(LogLevel.INFO, message, extra_data))
            else:
                # 如果没有运行中的循环，同步执行
                asyncio.run(self._log(LogLevel.INFO, message, extra_data))
        except RuntimeError:
            # 没有运行中的事件循环，同步执行
            asyncio.run(self._log(LogLevel.INFO, message, extra_data))

    def warning(self, message: str, extra_data: Dict[str, Any] = None):
        """警告日志"""
        try:
            # 检查是否有运行中的事件循环
            loop = asyncio.get_running_loop()
            if loop.is_running():
                asyncio.create_task(self._log(LogLevel.WARNING, message, extra_data))
            else:
                # 如果没有运行中的循环，同步执行
                asyncio.run(self._log(LogLevel.WARNING, message, extra_data))
        except RuntimeError:
            # 没有运行中的事件循环，同步执行
            asyncio.run(self._log(LogLevel.WARNING, message, extra_data))

    def error(self, message: str, extra_data: Dict[str, Any] = None, exception: Optional[Exception] = None):
        """错误日志"""
        asyncio.create_task(self._log(LogLevel.ERROR, message, extra_data, exception))

    def critical(self, message: str, extra_data: Dict[str, Any] = None, exception: Optional[Exception] = None):
        """严重日志"""
        asyncio.create_task(self._log(LogLevel.CRITICAL, message, extra_data, exception))

    async def flush(self):
        """刷新日志"""
        for handler in self._handlers:
            await handler.flush()

    async def close(self):
        """关闭日志记录器"""
        for handler in self._handlers:
            await handler.close()

class LogManager:
    """日志管理器"""

    def __init__(self):
        self._loggers: Dict[str, StructuredLogger] = {}
        self._config: Optional[LogConfiguration] = None
        self._setup_config()

    def _setup_config(self):
        """设置配置"""
        # 从配置管理器获取配置
        config_manager = get_configuration()

        # 这里应该从配置管理器获取日志配置
        # 为了简化，使用默认配置
        self._config = LogConfiguration(
            level=LogLevel.INFO,
            format=LogFormat.STRUCTURED,
            enable_console=True,
            enable_file=True,
            file_path="logs/app.log"
        )

    def get_logger(self, name: str) -> StructuredLogger:
        """获取日志记录器"""
        if name not in self._loggers:
            self._loggers[name] = StructuredLogger(name, self._config)

        return self._loggers[name]

    async def flush_all(self):
        """刷新所有日志"""
        for logger in self._loggers.values():
            await logger.flush()

    async def close_all(self):
        """关闭所有日志记录器"""
        for logger in self._loggers.values():
            await logger.close()

# 全局日志管理器
_log_manager = LogManager()

def get_logger(name: str) -> StructuredLogger:
    """获取日志记录器"""
    return _log_manager.get_logger(name)

def get_app_logger() -> StructuredLogger:
    """获取应用日志记录器"""
    return _log_manager.get_logger("app")

def get_database_logger() -> StructuredLogger:
    """获取数据库日志记录器"""
    return _log_manager.get_logger("database")

def get_api_logger() -> StructuredLogger:
    """获取API日志记录器"""
    return _log_manager.get_logger("api")

def get_data_processing_logger() -> StructuredLogger:
    """获取数据处理日志记录器"""
    return _log_manager.get_logger("data_processing")

# 日志装饰器
def log_function_call(logger_name: str = "app"):
    """记录函数调用装饰器"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            logger = get_logger(logger_name)
            logger.info(f"调用函数: {func.__name__}", {
                "function": func.__name__,
                "module": func.__module__,
                "args_count": len(args),
                "kwargs_count": len(kwargs)
            })

            try:
                result = await func(*args, **kwargs)
                logger.info(f"函数执行成功: {func.__name__}")
                return result
            except Exception as e:
                logger.error(f"函数执行失败: {func.__name__}", exception=e)
                raise

        def sync_wrapper(*args, **kwargs):
            logger = get_logger(logger_name)
            logger.info(f"调用函数: {func.__name__}", {
                "function": func.__name__,
                "module": func.__module__,
                "args_count": len(args),
                "kwargs_count": len(kwargs)
            })

            try:
                result = func(*args, **kwargs)
                logger.info(f"函数执行成功: {func.__name__}")
                return result
            except Exception as e:
                logger.error(f"函数执行失败: {func.__name__}", exception=e)
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator

def log_performance(logger_name: str = "app"):
    """记录性能日志装饰器"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            logger = get_logger(logger_name)
            start_time = datetime.now()

            try:
                result = await func(*args, **kwargs)
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()

                logger.info(f"函数执行完成: {func.__name__}", {
                    "function": func.__name__,
                    "duration": duration,
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat()
                })

                return result
            except Exception as e:
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()

                logger.error(f"函数执行失败: {func.__name__}", {
                    "function": func.__name__,
                    "duration": duration,
                    "error": str(e)
                }, exception=e)
                raise

        def sync_wrapper(*args, **kwargs):
            logger = get_logger(logger_name)
            start_time = datetime.now()

            try:
                result = func(*args, **kwargs)
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()

                logger.info(f"函数执行完成: {func.__name__}", {
                    "function": func.__name__,
                    "duration": duration,
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat()
                })

                return result
            except Exception as e:
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()

                logger.error(f"函数执行失败: {func.__name__}", {
                    "function": func.__name__,
                    "duration": duration,
                    "error": str(e)
                }, exception=e)
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator

if __name__ == "__main__":
    # 测试日志系统
    async def test_logging():
        logger = get_app_logger()

        logger.info("应用启动", {"version": "1.0.0", "environment": "development"})
        logger.warning("这是一个警告", {"code": "W001"})
        logger.error("这是一个错误", {"code": "E001"})

        try:
            raise ValueError("测试异常")
        except Exception as e:
            logger.error("捕获异常", exception=e)

        await logger.flush()

    # 运行测试
    asyncio.run(test_logging())

__all__ = ["'logger'", "'LogLevel'", "'LogFormat'", "'LogDestination'", "'LogEntry'", "'LogConfiguration'", "'ILogHandler'", "'ConsoleLogHandler'", "'FileLogHandler'", "'RemoteLogHandler'", "'StructuredLogger'", "'LogManager'", "'get_logger'", "'get_app_logger'", "'get_database_logger'", "'get_api_logger'", "'get_data_processing_logger'", "'log_function_call'", "'log_performance'"]
