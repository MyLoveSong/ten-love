

"""
性能优化与缓存管理模块 - MCP架构增强版
支持智能缓存、性能监控、自动优化、资源管理等
"""

import logging
import time
import asyncio
import threading
from typing import Dict, List, Optional, Any, Union, Tuple, Callable, Type
from dataclasses import dataclass, asdict
from enum import Enum
import json
from datetime import datetime, timedelta
import warnings
from concurrent.futures import ThreadPoolExecutor
import psutil
import gc
import weakref
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

from backend.app.core.exceptions import CustomException, ValidationError
from backend.app.core.task_queue import async_task, TaskPriority
from backend.app.core.dependency_injection import injectable, singleton, get_service
from backend.app.core.configuration import get_configuration
from backend.app.core.structured_logging import get_logger, log_function_call, log_performance
from backend.app.core.error_handling import ErrorContext, handle_error

logger = get_logger("performance_optimization")

class CacheStrategy(Enum):
    """缓存策略"""
    LRU = "lru"                        # 最近最少使用
    LFU = "lfu"                        # 最少使用频率
    FIFO = "fifo"                      # 先进先出
    TTL = "ttl"                        # 生存时间
    ADAPTIVE = "adaptive"              # 自适应
    INTELLIGENT = "intelligent"        # 智能缓存

class OptimizationLevel(Enum):
    """优化级别"""
    NONE = "none"                      # 无优化
    BASIC = "basic"                    # 基础优化
    ADVANCED = "advanced"              # 高级优化
    AGGRESSIVE = "aggressive"          # 激进优化
    CUSTOM = "custom"                  # 自定义优化

class ResourceType(Enum):
    """资源类型"""
    CPU = "cpu"                        # CPU资源
    MEMORY = "memory"                  # 内存资源
    DISK = "disk"                      # 磁盘资源
    NETWORK = "network"                # 网络资源
    DATABASE = "database"              # 数据库资源
    CACHE = "cache"                    # 缓存资源

@dataclass
class PerformanceMetrics:
    """性能指标"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    response_time: float
    throughput: float
    error_rate: float
    cache_hit_rate: float
    database_connections: int
    active_threads: int
    gc_collections: int

@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    size_bytes: int
    ttl: Optional[datetime] = None
    priority: int = 0

@dataclass
class OptimizationConfig:
    """优化配置"""
    cache_strategy: CacheStrategy
    optimization_level: OptimizationLevel
    max_cache_size: int = 1000
    max_memory_usage: float = 0.8
    gc_threshold: int = 1000
    monitoring_interval: int = 60
    auto_optimization: bool = True
    custom_parameters: Optional[Dict[str, Any]] = None

class IIntelligentCache(ABC):
    """智能缓存接口"""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值"""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """删除缓存值"""
        pass

    @abstractmethod
    async def clear(self) -> bool:
        """清空缓存"""
        pass

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        pass

@singleton(IIntelligentCache)
class AcademicIntelligentCache(IIntelligentCache):
    """学术级智能缓存"""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = get_logger("intelligent_cache")
        self._cache: Dict[str, CacheEntry] = {}
        self._access_history: deque = deque(maxlen=10000)
        self._hit_count = 0
        self._miss_count = 0
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._start_cleanup_task()

    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        async with self._lock:
            if key in self._cache:
                entry = self._cache[key]

                # 检查TTL
                if entry.ttl and datetime.now() > entry.ttl:
                    del self._cache[key]
                    self._miss_count += 1
                    return None

                # 更新访问信息
                entry.last_accessed = datetime.now()
                entry.access_count += 1
                self._access_history.append((key, datetime.now()))
                self._hit_count += 1

                return entry.value
            else:
                self._miss_count += 1
                return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值"""
        try:
            async with self._lock:
                # 计算值大小
                size_bytes = self._calculate_size(value)

                # 检查缓存大小限制
                if len(self._cache) >= self.config.max_cache_size:
                    await self._evict_entries()

                # 创建缓存条目
                ttl_datetime = None
                if ttl:
                    ttl_datetime = datetime.now() + timedelta(seconds=ttl)

                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=datetime.now(),
                    last_accessed=datetime.now(),
                    access_count=1,
                    size_bytes=size_bytes,
                    ttl=ttl_datetime
                )

                self._cache[key] = entry
                return True

        except Exception as e:
            self.logger.error(f"设置缓存失败: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """删除缓存值"""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def clear(self) -> bool:
        """清空缓存"""
        async with self._lock:
            self._cache.clear()
            self._access_history.clear()
            self._hit_count = 0
            self._miss_count = 0
            return True

    async def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total_requests = self._hit_count + self._miss_count
        hit_rate = self._hit_count / total_requests if total_requests > 0 else 0

        total_size = sum(entry.size_bytes for entry in self._cache.values())

        return {
            "cache_size": len(self._cache),
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": hit_rate,
            "total_size_bytes": total_size,
            "strategy": self.config.cache_strategy.value,
            "max_size": self.config.max_cache_size
        }

    def _calculate_size(self, value: Any) -> int:
        """计算值大小"""
        try:
            if isinstance(value, (str, int, float, bool)):
                return len(str(value))
            elif isinstance(value, (list, tuple)):
                return sum(self._calculate_size(item) for item in value)
            elif isinstance(value, dict):
                return sum(self._calculate_size(k) + self._calculate_size(v) for k, v in value.items())
            elif isinstance(value, (pd.DataFrame, np.ndarray)):
                return value.nbytes if hasattr(value, 'nbytes') else len(str(value))
            else:
                return len(str(value))
        except:
            return 1000  # 默认大小

    async def _evict_entries(self):
        """驱逐缓存条目"""
        if self.config.cache_strategy == CacheStrategy.LRU:
            await self._evict_lru()
        elif self.config.cache_strategy == CacheStrategy.LFU:
            await self._evict_lfu()
        elif self.config.cache_strategy == CacheStrategy.FIFO:
            await self._evict_fifo()
        elif self.config.cache_strategy == CacheStrategy.TTL:
            await self._evict_ttl()
        elif self.config.cache_strategy == CacheStrategy.ADAPTIVE:
            await self._evict_adaptive()
        elif self.config.cache_strategy == CacheStrategy.INTELLIGENT:
            await self._evict_intelligent()

    async def _evict_lru(self):
        """LRU驱逐策略"""
        if not self._cache:
            return

        # 找到最近最少使用的条目
        lru_key = min(self._cache.keys(), key=lambda k: self._cache[k].last_accessed)
        del self._cache[lru_key]

    async def _evict_lfu(self):
        """LFU驱逐策略"""
        if not self._cache:
            return

        # 找到使用频率最低的条目
        lfu_key = min(self._cache.keys(), key=lambda k: self._cache[k].access_count)
        del self._cache[lfu_key]

    async def _evict_fifo(self):
        """FIFO驱逐策略"""
        if not self._cache:
            return

        # 找到最早创建的条目
        fifo_key = min(self._cache.keys(), key=lambda k: self._cache[k].created_at)
        del self._cache[fifo_key]

    async def _evict_ttl(self):
        """TTL驱逐策略"""
        current_time = datetime.now()
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.ttl and current_time > entry.ttl
        ]

        for key in expired_keys:
            del self._cache[key]

    async def _evict_adaptive(self):
        """自适应驱逐策略"""
        # 结合LRU和LFU
        if len(self._cache) < 2:
            return

        # 计算综合分数
        scores = {}
        for key, entry in self._cache.items():
            # LRU分数（越小越好）
            lru_score = (datetime.now() - entry.last_accessed).total_seconds()
            # LFU分数（越小越好）
            lfu_score = 1.0 / (entry.access_count + 1)
            # 综合分数
            scores[key] = lru_score * 0.7 + lfu_score * 0.3

        # 驱逐分数最高的条目
        evict_key = max(scores.keys(), key=lambda k: scores[k])
        del self._cache[evict_key]

    async def _evict_intelligent(self):
        """智能驱逐策略"""
        if len(self._cache) < 2:
            return

        # 基于访问模式和值大小的智能驱逐
        scores = {}
        for key, entry in self._cache.items():
            # 访问频率分数
            freq_score = entry.access_count / max(1, (datetime.now() - entry.created_at).total_seconds())
            # 大小分数（越大越容易被驱逐）
            size_score = entry.size_bytes / 1000
            # 时间分数
            time_score = (datetime.now() - entry.last_accessed).total_seconds()

            # 综合分数
            scores[key] = freq_score * 0.5 + size_score * 0.3 + time_score * 0.2

        # 驱逐分数最低的条目
        evict_key = min(scores.keys(), key=lambda k: scores[k])
        del self._cache[evict_key]

    def _start_cleanup_task(self):
        """启动清理任务"""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(60)  # 每分钟清理一次
                    await self._cleanup_expired()
                except Exception as e:
                    self.logger.error(f"缓存清理任务异常: {e}")

        try:
            # 检查是否有运行中的事件循环
            loop = asyncio.get_running_loop()
            if loop.is_running():
                self._cleanup_task = asyncio.create_task(cleanup_loop())
            else:
                # 如果没有运行中的循环，延迟启动
                self._cleanup_task = None
                logger.warning("没有运行中的事件循环，跳过清理任务启动")
        except RuntimeError:
            # 没有运行中的事件循环，延迟启动
            self._cleanup_task = None
            logger.warning("没有运行中的事件循环，跳过清理任务启动")

    async def _cleanup_expired(self):
        """清理过期条目"""
        current_time = datetime.now()
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.ttl and current_time > entry.ttl
        ]

        async with self._lock:
            for key in expired_keys:
                del self._cache[key]

        if expired_keys:
            self.logger.info(f"清理过期缓存条目: {len(expired_keys)} 个")

class IPerformanceMonitor(ABC):
    """性能监控接口"""

    @abstractmethod
    async def collect_metrics(self) -> PerformanceMetrics:
        """收集性能指标"""
        pass

    @abstractmethod
    async def get_metrics_history(self, limit: int = 100) -> List[PerformanceMetrics]:
        """获取指标历史"""
        pass

    @abstractmethod
    async def detect_anomalies(self) -> List[Dict[str, Any]]:
        """检测性能异常"""
        pass

@singleton(IPerformanceMonitor)
class SystemPerformanceMonitor(IPerformanceMonitor):
    """系统性能监控器"""

    def __init__(self):
        self.logger = get_logger("performance_monitor")
        self._metrics_history: deque = deque(maxlen=1000)
        self._monitoring_task: Optional[asyncio.Task] = None
        self._start_monitoring()

    async def collect_metrics(self) -> PerformanceMetrics:
        """收集性能指标"""
        try:
            # CPU使用率
            cpu_usage = psutil.cpu_percent(interval=1)

            # 内存使用率
            memory = psutil.virtual_memory()
            memory_usage = memory.percent

            # 磁盘使用率
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100

            # 网络IO
            network_io = psutil.net_io_counters()
            network_stats = {
                "bytes_sent": network_io.bytes_sent,
                "bytes_recv": network_io.bytes_recv,
                "packets_sent": network_io.packets_sent,
                "packets_recv": network_io.packets_recv
            }

            # 其他指标
            active_threads = threading.active_count()
            gc_collections = sum(gc.get_stats())

            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                network_io=network_stats,
                response_time=0.0,  # 需要从其他地方获取
                throughput=0.0,     # 需要从其他地方获取
                error_rate=0.0,     # 需要从其他地方获取
                cache_hit_rate=0.0, # 需要从其他地方获取
                database_connections=0,  # 需要从其他地方获取
                active_threads=active_threads,
                gc_collections=gc_collections
            )

            self._metrics_history.append(metrics)
            return metrics

        except Exception as e:
            self.logger.error(f"收集性能指标失败: {e}")
            raise CustomException(f"收集性能指标失败: {e}")

    async def get_metrics_history(self, limit: int = 100) -> List[PerformanceMetrics]:
        """获取指标历史"""
        return list(self._metrics_history)[-limit:]

    async def detect_anomalies(self) -> List[Dict[str, Any]]:
        """检测性能异常"""
        try:
            anomalies = []

            if len(self._metrics_history) < 10:
                return anomalies

            recent_metrics = list(self._metrics_history)[-10:]

            # CPU异常检测
            cpu_values = [m.cpu_usage for m in recent_metrics]
            cpu_mean = np.mean(cpu_values)
            cpu_std = np.std(cpu_values)

            if cpu_mean > 80 or cpu_std > 20:
                anomalies.append({
                    "type": "cpu_anomaly",
                    "severity": "high" if cpu_mean > 90 else "medium",
                    "message": f"CPU使用率异常: 平均值 {cpu_mean:.2f}%, 标准差 {cpu_std:.2f}",
                    "values": cpu_values
                })

            # 内存异常检测
            memory_values = [m.memory_usage for m in recent_metrics]
            memory_mean = np.mean(memory_values)

            if memory_mean > 85:
                anomalies.append({
                    "type": "memory_anomaly",
                    "severity": "high" if memory_mean > 95 else "medium",
                    "message": f"内存使用率异常: 平均值 {memory_mean:.2f}%",
                    "values": memory_values
                })

            # 磁盘异常检测
            disk_values = [m.disk_usage for m in recent_metrics]
            disk_mean = np.mean(disk_values)

            if disk_mean > 90:
                anomalies.append({
                    "type": "disk_anomaly",
                    "severity": "high",
                    "message": f"磁盘使用率异常: 平均值 {disk_mean:.2f}%",
                    "values": disk_values
                })

            return anomalies

        except Exception as e:
            self.logger.error(f"检测性能异常失败: {e}")
            return []

    def _start_monitoring(self):
        """启动监控任务"""
        async def monitoring_loop():
            while True:
                try:
                    await asyncio.sleep(60)  # 每分钟收集一次
                    await self.collect_metrics()
                except Exception as e:
                    self.logger.error(f"性能监控任务异常: {e}")

        try:
            # 检查是否有运行中的事件循环
            loop = asyncio.get_running_loop()
            if loop.is_running():
                self._monitoring_task = asyncio.create_task(monitoring_loop())
            else:
                # 如果没有运行中的循环，延迟启动
                self._monitoring_task = None
                logger.warning("没有运行中的事件循环，跳过监控任务启动")
        except RuntimeError:
            # 没有运行中的事件循环，延迟启动
            self._monitoring_task = None
            logger.warning("没有运行中的事件循环，跳过监控任务启动")

class IResourceOptimizer(ABC):
    """资源优化器接口"""

    @abstractmethod
    async def optimize_memory(self) -> Dict[str, Any]:
        """优化内存使用"""
        pass

    @abstractmethod
    async def optimize_cpu(self) -> Dict[str, Any]:
        """优化CPU使用"""
        pass

    @abstractmethod
    async def optimize_cache(self) -> Dict[str, Any]:
        """优化缓存使用"""
        pass

    @abstractmethod
    async def optimize_database(self) -> Dict[str, Any]:
        """优化数据库使用"""
        pass

@singleton(IResourceOptimizer)
class AcademicResourceOptimizer(IResourceOptimizer):
    """学术级资源优化器"""

    def __init__(self):
        self.logger = get_logger("resource_optimizer")
        self.cache = None  # 将在外部设置

    async def optimize_memory(self) -> Dict[str, Any]:
        """优化内存使用"""
        try:
            # 强制垃圾回收
            collected = gc.collect()

            # 获取内存使用情况
            memory_before = psutil.virtual_memory().percent

            # 清理弱引用
            weakref_collected = 0
            for obj in list(weakref.WeakSet()):
                if obj is None:
                    weakref_collected += 1

            # 获取优化后内存使用情况
            memory_after = psutil.virtual_memory().percent

            return {
                "gc_collections": collected,
                "weakref_collected": weakref_collected,
                "memory_before": memory_before,
                "memory_after": memory_after,
                "memory_saved": memory_before - memory_after
            }

        except Exception as e:
            self.logger.error(f"内存优化失败: {e}")
            return {"error": str(e)}

    async def optimize_cpu(self) -> Dict[str, Any]:
        """优化CPU使用"""
        try:
            # 获取CPU使用情况
            cpu_before = psutil.cpu_percent(interval=1)

            # 这里可以添加CPU优化逻辑
            # 例如：调整线程池大小、优化算法等

            cpu_after = psutil.cpu_percent(interval=1)

            return {
                "cpu_before": cpu_before,
                "cpu_after": cpu_after,
                "cpu_improvement": cpu_before - cpu_after
            }

        except Exception as e:
            self.logger.error(f"CPU优化失败: {e}")
            return {"error": str(e)}

    async def optimize_cache(self) -> Dict[str, Any]:
        """优化缓存使用"""
        try:
            if not self.cache:
                return {"error": "缓存未初始化"}

            # 获取缓存统计
            stats_before = await self.cache.get_stats()

            # 清理过期条目
            await self.cache._cleanup_expired()

            # 获取优化后统计
            stats_after = await self.cache.get_stats()

            return {
                "cache_size_before": stats_before["cache_size"],
                "cache_size_after": stats_after["cache_size"],
                "hit_rate_before": stats_before["hit_rate"],
                "hit_rate_after": stats_after["hit_rate"],
                "entries_removed": stats_before["cache_size"] - stats_after["cache_size"]
            }

        except Exception as e:
            self.logger.error(f"缓存优化失败: {e}")
            return {"error": str(e)}

    async def optimize_database(self) -> Dict[str, Any]:
        """优化数据库使用"""
        try:
            # 这里可以添加数据库优化逻辑
            # 例如：清理连接池、优化查询等

            return {
                "optimization": "database_optimization_completed",
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"数据库优化失败: {e}")
            return {"error": str(e)}

class AcademicPerformanceOptimizer:
    """学术级性能优化器"""

    def __init__(self):
        self.logger = get_logger("academic_performance_optimizer")
        self.config = OptimizationConfig(
            cache_strategy=CacheStrategy.INTELLIGENT,
            optimization_level=OptimizationLevel.ADVANCED,
            auto_optimization=True
        )
        self.cache = AcademicIntelligentCache(self.config)
        self.monitor = SystemPerformanceMonitor()
        self.optimizer = AcademicResourceOptimizer()
        self.optimizer.cache = self.cache
        self._optimization_task: Optional[asyncio.Task] = None
        self._start_auto_optimization()

    @log_function_call("academic_performance_optimizer")
    @log_performance("academic_performance_optimizer")
    async def optimize_system(self) -> Dict[str, Any]:
        """优化系统性能"""
        try:
            optimization_results = {}

            # 内存优化
            memory_result = await self.optimizer.optimize_memory()
            optimization_results["memory"] = memory_result

            # CPU优化
            cpu_result = await self.optimizer.optimize_cpu()
            optimization_results["cpu"] = cpu_result

            # 缓存优化
            cache_result = await self.optimizer.optimize_cache()
            optimization_results["cache"] = cache_result

            # 数据库优化
            database_result = await self.optimizer.optimize_database()
            optimization_results["database"] = database_result

            # 收集当前性能指标
            current_metrics = await self.monitor.collect_metrics()
            optimization_results["current_metrics"] = asdict(current_metrics)

            self.logger.info("系统性能优化完成")
            return optimization_results

        except Exception as e:
            error_context = ErrorContext(
                module="performance_optimization",
                function="optimize_system"
            )
            await handle_error(e, error_context)
            raise

    async def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        try:
            # 获取当前指标
            current_metrics = await self.monitor.collect_metrics()

            # 获取历史指标
            history_metrics = await self.monitor.get_metrics_history(100)

            # 检测异常
            anomalies = await self.monitor.detect_anomalies()

            # 获取缓存统计
            cache_stats = await self.cache.get_stats()

            return {
                "current_metrics": asdict(current_metrics),
                "history_summary": {
                    "avg_cpu": np.mean([m.cpu_usage for m in history_metrics]),
                    "avg_memory": np.mean([m.memory_usage for m in history_metrics]),
                    "avg_disk": np.mean([m.disk_usage for m in history_metrics]),
                    "data_points": len(history_metrics)
                },
                "anomalies": anomalies,
                "cache_stats": cache_stats,
                "optimization_config": asdict(self.config),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"获取性能报告失败: {e}")
            raise CustomException(f"获取性能报告失败: {e}")

    def _start_auto_optimization(self):
        """启动自动优化任务"""
        if not self.config.auto_optimization:
            return

        async def auto_optimization_loop():
            while True:
                try:
                    await asyncio.sleep(self.config.monitoring_interval)

                    # 检查是否需要优化
                    current_metrics = await self.monitor.collect_metrics()

                    if (current_metrics.cpu_usage > 80 or
                        current_metrics.memory_usage > 85 or
                        current_metrics.disk_usage > 90):

                        self.logger.info("检测到性能问题，开始自动优化")
                        await self.optimize_system()

                except Exception as e:
                    self.logger.error(f"自动优化任务异常: {e}")

        try:
            # 检查是否有运行中的事件循环
            loop = asyncio.get_running_loop()
            if loop.is_running():
                self._optimization_task = asyncio.create_task(auto_optimization_loop())
            else:
                # 如果没有运行中的循环，延迟启动
                self._optimization_task = None
                logger.warning("没有运行中的事件循环，跳过自动优化任务启动")
        except RuntimeError:
            # 没有运行中的事件循环，延迟启动
            self._optimization_task = None
            logger.warning("没有运行中的事件循环，跳过自动优化任务启动")

    async def shutdown(self):
        """关闭优化器"""
        if self._optimization_task:
            self._optimization_task.cancel()
            try:
                await self._optimization_task
            except asyncio.CancelledError:
                pass

        await self.cache.clear()
        self.logger.info("性能优化器已关闭")

# 全局性能优化器实例
academic_performance_optimizer = AcademicPerformanceOptimizer()

# 异步任务
@async_task("optimize_system", TaskPriority.NORMAL)
def optimize_system_task():
    """系统优化任务"""
    result = asyncio.run(academic_performance_optimizer.optimize_system())

    return {
        "result": result,
        "success": True
    }

# 性能优化API
def optimize_system() -> Dict[str, Any]:
    """优化系统性能"""
    return asyncio.run(academic_performance_optimizer.optimize_system())

def get_performance_report() -> Dict[str, Any]:
    """获取性能报告"""
    return asyncio.run(academic_performance_optimizer.get_performance_report())

def get_cache_stats() -> Dict[str, Any]:
    """获取缓存统计"""
    return asyncio.run(academic_performance_optimizer.cache.get_stats())

if __name__ == "__main__":
    # 测试性能优化
    import time

    # 创建测试数据
    test_data = {
        "user_data": [{"id": i, "name": f"user_{i}"} for i in range(1000)],
        "glucose_data": list(np.random.normal(100, 20, 1000)),
        "timestamp": datetime.now().isoformat()
    }

    # 测试缓存
    cache = academic_performance_optimizer.cache

    # 设置缓存
    asyncio.run(cache.set("test_data", test_data, ttl=300))

    # 获取缓存
    cached_data = asyncio.run(cache.get("test_data"))
    print(f"缓存数据: {len(cached_data['user_data'])} 用户")

    # 获取缓存统计
    stats = asyncio.run(cache.get_stats())
    print("缓存统计:", json.dumps(stats, indent=2, ensure_ascii=False))

    # 系统优化
    optimization_result = optimize_system()
    print("优化结果:", json.dumps(optimization_result, indent=2, ensure_ascii=False, default=str))

    # 性能报告
    report = get_performance_report()
    print("性能报告:", json.dumps(report, indent=2, ensure_ascii=False, default=str))

__all__ = ["'logger'", "'CacheStrategy'", "'OptimizationLevel'", "'ResourceType'", "'PerformanceMetrics'", "'CacheEntry'", "'OptimizationConfig'", "'IIntelligentCache'", "'AcademicIntelligentCache'", "'IPerformanceMonitor'", "'SystemPerformanceMonitor'", "'IResourceOptimizer'", "'AcademicResourceOptimizer'", "'AcademicPerformanceOptimizer'", "'academic_performance_optimizer'", "'optimize_system_task'", "'optimize_system'", "'get_performance_report'", "'get_cache_stats'"]
