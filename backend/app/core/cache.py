"""
缓存管理模块
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import asyncio
import json

"""
缓存管理模块
企业级缓存系统，支持Redis和内存缓存
"""

import json
import pickle
import hashlib
logger = logging.getLogger(__name__)
from datetime import datetime, timedelta
import logging
from functools import wraps
import asyncio

try:
    import redis
    from redis.asyncio import Redis as AsyncRedis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None
    AsyncRedis = None

from .config import settings

logger = logging.getLogger(__name__)

class CacheManager:
    """缓存管理器"""

    def __init__(self):
        self.redis_client: Optional[AsyncRedis] = None
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0
        }

        # 初始化Redis连接
        if REDIS_AVAILABLE and settings.REDIS_URL:
            self._init_redis()

    def _init_redis(self):
        """初始化Redis连接"""
        try:
            self.redis_client = AsyncRedis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=False,  # 使用二进制模式
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            logger.info("✅ Redis缓存连接初始化成功")
        except Exception as e:
            logger.warning(f"Redis连接失败，使用内存缓存: {e}")
            self.redis_client = None

    async def _redis_ping(self) -> bool:
        """检查Redis连接状态"""
        if not self.redis_client:
            return False
        try:
            await self.redis_client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis ping失败: {e}")
            return False

    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """生成缓存键"""
        key_data = {
            "args": args,
            "kwargs": sorted(kwargs.items())
        }
        key_str = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.md5(key_str.encode()).hexdigest()
        return f"{prefix}:{key_hash}"

    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        # 尝试从Redis获取
        if self.redis_client and await self._redis_ping():
            try:
                value = await self.redis_client.get(key)
                if value:
                    self.cache_stats["hits"] += 1
                    return pickle.loads(value)
            except Exception as e:
                logger.error(f"Redis获取失败: {e}")

        # 从内存缓存获取
        if key in self.memory_cache:
            cache_data = self.memory_cache[key]
            if cache_data["expires_at"] > datetime.now():
                self.cache_stats["hits"] += 1
                return cache_data["value"]
            else:
                # 过期，删除
                del self.memory_cache[key]

        self.cache_stats["misses"] += 1
        return None

    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """设置缓存值"""
        self.cache_stats["sets"] += 1

        # 尝试设置到Redis
        if self.redis_client and await self._redis_ping():
            try:
                serialized_value = pickle.dumps(value)
                await self.redis_client.setex(key, ttl, serialized_value)
                return True
            except Exception as e:
                logger.error(f"Redis设置失败: {e}")

        # 设置到内存缓存
        expires_at = datetime.now() + timedelta(seconds=ttl)
        self.memory_cache[key] = {
            "value": value,
            "expires_at": expires_at
        }

        # 清理过期缓存
        self._cleanup_expired_cache()
        return True

    async def delete(self, key: str) -> bool:
        """删除缓存"""
        self.cache_stats["deletes"] += 1

        # 从Redis删除
        if self.redis_client and await self._redis_ping():
            try:
                await self.redis_client.delete(key)
            except Exception as e:
                logger.error(f"Redis删除失败: {e}")

        # 从内存缓存删除
        if key in self.memory_cache:
            del self.memory_cache[key]

        return True

    async def clear_pattern(self, pattern: str) -> int:
        """清除匹配模式的缓存"""
        deleted_count = 0

        # 清除Redis中的匹配键
        if self.redis_client and await self._redis_ping():
            try:
                keys = await self.redis_client.keys(pattern)
                if keys:
                    await self.redis_client.delete(*keys)
                    deleted_count += len(keys)
            except Exception as e:
                logger.error(f"Redis模式删除失败: {e}")

        # 清除内存缓存中的匹配键
        keys_to_delete = [k for k in self.memory_cache.keys() if pattern.replace("*", "") in k]
        for key in keys_to_delete:
            del self.memory_cache[key]
            deleted_count += 1

        return deleted_count

    def _cleanup_expired_cache(self):
        """清理过期的内存缓存"""
        now = datetime.now()
        expired_keys = [
            key for key, data in self.memory_cache.items()
            if data["expires_at"] <= now
        ]
        for key in expired_keys:
            del self.memory_cache[key]

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = self.cache_stats["hits"] / total_requests if total_requests > 0 else 0

        return {
            **self.cache_stats,
            "hit_rate": hit_rate,
            "memory_cache_size": len(self.memory_cache),
            "redis_available": self.redis_client is not None
        }

    async def health_check(self) -> Dict[str, Any]:
        """缓存健康检查"""
        redis_status = False
        if self.redis_client:
            redis_status = await self._redis_ping()

        return {
            "redis_connected": redis_status,
            "memory_cache_size": len(self.memory_cache),
            "stats": self.get_stats()
        }

# 全局缓存管理器实例
cache_manager = CacheManager()

def cached(prefix: str, ttl: int = 3600, key_func: Optional[callable] = None):
    """缓存装饰器"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # 生成缓存键
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = cache_manager._generate_key(prefix, *args, **kwargs)

            # 尝试从缓存获取
            cached_result = await cache_manager.get(cache_key)
            if cached_result is not None:
                logger.debug(f"缓存命中: {cache_key}")
                return cached_result

            # 执行函数并缓存结果
            logger.debug(f"缓存未命中，执行函数: {func.__name__}")
            result = await func(*args, **kwargs)
            await cache_manager.set(cache_key, result, ttl)

            return result

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # 同步函数的异步缓存包装
            return asyncio.run(async_wrapper(*args, **kwargs))

        # 根据函数类型返回相应的包装器
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator

class CacheKeyBuilder:
    """缓存键构建器"""

    @staticmethod
    def user_data(user_id: str) -> str:
        """用户数据缓存键"""
        return f"user:{user_id}"

    @staticmethod
    def glucose_predictions(user_id: str, days: int = 7) -> str:
        """血糖预测缓存键"""
        return f"glucose:predictions:{user_id}:{days}d"

    @staticmethod
    def cultural_recommendations(user_id: str, meal_type: str) -> str:
        """文化推荐缓存键"""
        return f"cultural:recommendations:{user_id}:{meal_type}"

    @staticmethod
    def food_items(category: str, filters: Dict[str, Any]) -> str:
        """食物项目缓存键"""
        filter_str = json.dumps(filters, sort_keys=True)
        return f"food:items:{category}:{hashlib.md5(filter_str.encode()).hexdigest()}"

    @staticmethod
    def recipes(category: str, filters: Dict[str, Any]) -> str:
        """菜谱缓存键"""
        filter_str = json.dumps(filters, sort_keys=True)
        return f"recipes:{category}:{hashlib.md5(filter_str.encode()).hexdigest()}"

    @staticmethod
    def system_metrics(metric_type: str, time_range: str) -> str:
        """系统指标缓存键"""
        return f"metrics:{metric_type}:{time_range}"

    @staticmethod
    def api_response(endpoint: str, params: Dict[str, Any]) -> str:
        """API响应缓存键"""
        param_str = json.dumps(params, sort_keys=True)
        return f"api:{endpoint}:{hashlib.md5(param_str.encode()).hexdigest()}"

# 缓存键构建器实例
cache_keys = CacheKeyBuilder()

# 预定义的缓存TTL配置
CACHE_TTL = {
    "user_data": 1800,        # 30分钟
    "glucose_predictions": 300,  # 5分钟
    "cultural_recommendations": 600,  # 10分钟
    "food_items": 3600,       # 1小时
    "recipes": 3600,          # 1小时
    "system_metrics": 60,      # 1分钟
    "api_response": 300,      # 5分钟
}

# 缓存预热函数
async def warmup_cache():
    """缓存预热"""
    logger.info("开始缓存预热...")

    try:
        # 预热常用数据
        # 这里可以添加预热逻辑
        logger.info("缓存预热完成")
    except Exception as e:
        logger.error(f"缓存预热失败: {e}")

# 缓存清理函数
async def cleanup_cache():
    """清理缓存"""
    logger.info("开始清理缓存...")

    try:
        # 清理过期缓存
        cache_manager._cleanup_expired_cache()

        # 清理特定模式的缓存
        await cache_manager.clear_pattern("temp:*")

        logger.info("缓存清理完成")
    except Exception as e:
        logger.error(f"缓存清理失败: {e}")

__all__ = ["'logger'", "'CacheManager'", "'cache_manager'", "'cached'", "'CacheKeyBuilder'", "'cache_keys'", "'CACHE_TTL'"]
