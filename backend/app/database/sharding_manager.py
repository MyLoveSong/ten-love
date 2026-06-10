

"""
数据库分库分表管理器
支持水平扩展和分片策略
"""

import hashlib
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import json
import random

from sqlalchemy import create_engine, text, MetaData
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

from backend.app.core.config import settings
from backend.app.core.exceptions import DatabaseError, ValidationError

logger = logging.getLogger(__name__)

class ShardingStrategy(Enum):
    """分片策略枚举"""
    HASH = "hash"  # 哈希分片
    RANGE = "range"  # 范围分片
    ROUND_ROBIN = "round_robin"  # 轮询分片
    CONSISTENT_HASH = "consistent_hash"  # 一致性哈希

@dataclass
class ShardConfig:
    """分片配置"""
    shard_id: str
    database_url: str
    weight: int = 1
    is_active: bool = True
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ShardingRule:
    """分片规则"""
    table_name: str
    shard_key: str
    strategy: ShardingStrategy
    shard_count: int
    shard_configs: List[ShardConfig]
    metadata: Optional[Dict[str, Any]] = None

class ConsistentHashRing:
    """一致性哈希环"""

    def __init__(self, nodes: List[str], replicas: int = 100):
        self.replicas = replicas
        self.ring = {}
        self.sorted_keys = []

        for node in nodes:
            self.add_node(node)

    def add_node(self, node: str):
        """添加节点"""
        for i in range(self.replicas):
            key = self._hash(f"{node}:{i}")
            self.ring[key] = node
            self.sorted_keys.append(key)

        self.sorted_keys.sort()

    def remove_node(self, node: str):
        """移除节点"""
        for i in range(self.replicas):
            key = self._hash(f"{node}:{i}")
            if key in self.ring:
                del self.ring[key]
                self.sorted_keys.remove(key)

    def get_node(self, key: str) -> str:
        """获取节点"""
        if not self.ring:
            raise ValueError("哈希环为空")

        hash_key = self._hash(key)
        idx = self._bisect_left(self.sorted_keys, hash_key)

        if idx == len(self.sorted_keys):
            idx = 0

        return self.ring[self.sorted_keys[idx]]

    def _hash(self, key: str) -> int:
        """计算哈希值"""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def _bisect_left(self, arr: List[int], x: int) -> int:
        """二分查找左边界"""
        left, right = 0, len(arr)
        while left < right:
            mid = (left + right) // 2
            if arr[mid] < x:
                left = mid + 1
            else:
                right = mid
        return left

class DatabaseShardingManager:
    """数据库分片管理器"""

    def __init__(self):
        self.sharding_rules: Dict[str, ShardingRule] = {}
        self.shard_engines: Dict[str, Any] = {}
        self.shard_sessions: Dict[str, Any] = {}
        self.hash_rings: Dict[str, ConsistentHashRing] = {}
        self.round_robin_counters: Dict[str, int] = {}

        # 初始化默认分片规则
        self._init_default_sharding_rules()

    def _init_default_sharding_rules(self):
        """初始化默认分片规则"""
        # 用户表分片规则
        user_shard_configs = [
            ShardConfig("shard_0", settings.DATABASE_URL, 1),
            ShardConfig("shard_1", settings.DATABASE_URL.replace(".db", "_shard1.db"), 1),
        ]

        user_rule = ShardingRule(
            table_name="users",
            shard_key="user_id",
            strategy=ShardingStrategy.HASH,
            shard_count=2,
            shard_configs=user_shard_configs
        )

        self.add_sharding_rule("users", user_rule)

        # 血糖预测表分片规则（按时间分片）
        glucose_shard_configs = [
            ShardConfig("glucose_shard_0", settings.DATABASE_URL.replace(".db", "_glucose0.db"), 1),
            ShardConfig("glucose_shard_1", settings.DATABASE_URL.replace(".db", "_glucose1.db"), 1),
        ]

        glucose_rule = ShardingRule(
            table_name="glucose_predictions",
            shard_key="prediction_time",
            strategy=ShardingStrategy.RANGE,
            shard_count=2,
            shard_configs=glucose_shard_configs
        )

        self.add_sharding_rule("glucose_predictions", glucose_rule)

    def add_sharding_rule(self, table_name: str, rule: ShardingRule):
        """添加分片规则"""
        self.sharding_rules[table_name] = rule

        # 初始化分片引擎
        for config in rule.shard_configs:
            if config.is_active:
                self._init_shard_engine(config)

        # 初始化一致性哈希环
        if rule.strategy == ShardingStrategy.CONSISTENT_HASH:
            nodes = [config.shard_id for config in rule.shard_configs if config.is_active]
            self.hash_rings[table_name] = ConsistentHashRing(nodes)

        # 初始化轮询计数器
        if rule.strategy == ShardingStrategy.ROUND_ROBIN:
            self.round_robin_counters[table_name] = 0

        logger.info(f"添加分片规则: {table_name}, 策略: {rule.strategy.value}")

    def _init_shard_engine(self, config: ShardConfig):
        """初始化分片引擎"""
        try:
            engine = create_engine(
                config.database_url,
                poolclass=QueuePool,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
                pool_recycle=3600
            )

            session_factory = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=engine
            )

            self.shard_engines[config.shard_id] = engine
            self.shard_sessions[config.shard_id] = session_factory

            logger.info(f"初始化分片引擎: {config.shard_id}")

        except Exception as e:
            logger.error(f"初始化分片引擎失败 {config.shard_id}: {e}")
            raise DatabaseError(f"初始化分片引擎失败: {e}")

    def get_shard_for_key(self, table_name: str, shard_key_value: Any) -> str:
        """根据分片键值获取分片"""
        if table_name not in self.sharding_rules:
            raise ValidationError(f"表 {table_name} 没有配置分片规则")

        rule = self.sharding_rules[table_name]
        active_configs = [c for c in rule.shard_configs if c.is_active]

        if not active_configs:
            raise DatabaseError(f"表 {table_name} 没有活跃的分片")

        if rule.strategy == ShardingStrategy.HASH:
            return self._get_hash_shard(table_name, shard_key_value, active_configs)
        elif rule.strategy == ShardingStrategy.RANGE:
            return self._get_range_shard(table_name, shard_key_value, active_configs)
        elif rule.strategy == ShardingStrategy.ROUND_ROBIN:
            return self._get_round_robin_shard(table_name, active_configs)
        elif rule.strategy == ShardingStrategy.CONSISTENT_HASH:
            return self._get_consistent_hash_shard(table_name, shard_key_value)
        else:
            raise ValidationError(f"不支持的分片策略: {rule.strategy}")

    def _get_hash_shard(self, table_name: str, key_value: Any, configs: List[ShardConfig]) -> str:
        """哈希分片"""
        key_str = str(key_value)
        hash_value = int(hashlib.md5(key_str.encode()).hexdigest(), 16)
        shard_index = hash_value % len(configs)
        return configs[shard_index].shard_id

    def _get_range_shard(self, table_name: str, key_value: Any, configs: List[ShardConfig]) -> str:
        """范围分片"""
        if isinstance(key_value, datetime):
            # 按时间分片
            hour = key_value.hour
            shard_index = hour % len(configs)
        elif isinstance(key_value, (int, float)):
            # 按数值范围分片
            shard_index = int(key_value) % len(configs)
        else:
            # 默认哈希分片
            return self._get_hash_shard(table_name, key_value, configs)

        return configs[shard_index].shard_id

    def _get_round_robin_shard(self, table_name: str, configs: List[ShardConfig]) -> str:
        """轮询分片"""
        counter = self.round_robin_counters[table_name]
        shard_index = counter % len(configs)
        self.round_robin_counters[table_name] = counter + 1
        return configs[shard_index].shard_id

    def _get_consistent_hash_shard(self, table_name: str, key_value: Any) -> str:
        """一致性哈希分片"""
        if table_name not in self.hash_rings:
            raise DatabaseError(f"表 {table_name} 的一致性哈希环未初始化")

        key_str = str(key_value)
        return self.hash_rings[table_name].get_node(key_str)

    def get_session(self, table_name: str, shard_key_value: Any):
        """获取分片会话"""
        shard_id = self.get_shard_for_key(table_name, shard_key_value)

        if shard_id not in self.shard_sessions:
            raise DatabaseError(f"分片 {shard_id} 的会话未初始化")

        return self.shard_sessions[shard_id]()

    def get_all_sessions(self, table_name: str):
        """获取所有分片会话"""
        if table_name not in self.sharding_rules:
            raise ValidationError(f"表 {table_name} 没有配置分片规则")

        rule = self.sharding_rules[table_name]
        sessions = []

        for config in rule.shard_configs:
            if config.is_active and config.shard_id in self.shard_sessions:
                sessions.append(self.shard_sessions[config.shard_id]())

        return sessions

    def execute_on_all_shards(self, table_name: str, operation: str, params: Optional[Dict] = None):
        """在所有分片上执行操作"""
        sessions = self.get_all_sessions(table_name)
        results = []

        for session in sessions:
            try:
                result = session.execute(text(operation), params or {})
                session.commit()
                results.append(result)
            except Exception as e:
                session.rollback()
                logger.error(f"分片操作失败: {e}")
                raise DatabaseError(f"分片操作失败: {e}")
            finally:
                session.close()

        return results

    def get_shard_stats(self) -> Dict[str, Any]:
        """获取分片统计信息"""
        stats = {
            "total_rules": len(self.sharding_rules),
            "total_shards": len(self.shard_engines),
            "rules": {}
        }

        for table_name, rule in self.sharding_rules.items():
            stats["rules"][table_name] = {
                "strategy": rule.strategy.value,
                "shard_count": rule.shard_count,
                "shard_key": rule.shard_key,
                "active_shards": len([c for c in rule.shard_configs if c.is_active])
            }

        return stats

    def rebalance_shards(self, table_name: str, new_configs: List[ShardConfig]):
        """重新平衡分片"""
        if table_name not in self.sharding_rules:
            raise ValidationError(f"表 {table_name} 没有配置分片规则")

        old_rule = self.sharding_rules[table_name]

        # 创建新规则
        new_rule = ShardingRule(
            table_name=table_name,
            shard_key=old_rule.shard_key,
            strategy=old_rule.strategy,
            shard_count=len(new_configs),
            shard_configs=new_configs
        )

        # 更新规则
        self.add_sharding_rule(table_name, new_rule)

        logger.info(f"分片重新平衡完成: {table_name}")

    def add_shard(self, table_name: str, config: ShardConfig):
        """添加新分片"""
        if table_name not in self.sharding_rules:
            raise ValidationError(f"表 {table_name} 没有配置分片规则")

        rule = self.sharding_rules[table_name]
        rule.shard_configs.append(config)
        rule.shard_count = len(rule.shard_configs)

        # 初始化新分片
        self._init_shard_engine(config)

        # 更新一致性哈希环
        if rule.strategy == ShardingStrategy.CONSISTENT_HASH:
            self.hash_rings[table_name].add_node(config.shard_id)

        logger.info(f"添加新分片: {table_name} -> {config.shard_id}")

    def remove_shard(self, table_name: str, shard_id: str):
        """移除分片"""
        if table_name not in self.sharding_rules:
            raise ValidationError(f"表 {table_name} 没有配置分片规则")

        rule = self.sharding_rules[table_name]

        # 标记分片为非活跃
        for config in rule.shard_configs:
            if config.shard_id == shard_id:
                config.is_active = False
                break

        # 从一致性哈希环移除
        if rule.strategy == ShardingStrategy.CONSISTENT_HASH and table_name in self.hash_rings:
            self.hash_rings[table_name].remove_node(shard_id)

        logger.info(f"移除分片: {table_name} -> {shard_id}")

# 全局分片管理器实例
sharding_manager = DatabaseShardingManager()

# 分片装饰器
def sharded_operation(table_name: str, shard_key_param: str = "shard_key"):
    """分片操作装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 获取分片键值
            shard_key_value = kwargs.get(shard_key_param)
            if not shard_key_value:
                raise ValidationError(f"缺少分片键参数: {shard_key_param}")

            # 获取分片会话
            session = sharding_manager.get_session(table_name, shard_key_value)

            try:
                # 将会话添加到参数中
                kwargs['session'] = session
                result = func(*args, **kwargs)
                session.commit()
                return result
            except Exception as e:
                session.rollback()
                raise
            finally:
                session.close()

        return wrapper
    return decorator

# 跨分片查询装饰器
def cross_shard_query(table_name: str):
    """跨分片查询装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            sessions = sharding_manager.get_all_sessions(table_name)
            results = []

            for session in sessions:
                try:
                    kwargs['session'] = session
                    result = func(*args, **kwargs)
                    results.extend(result if isinstance(result, list) else [result])
                except Exception as e:
                    logger.error(f"分片查询失败: {e}")
                    continue
                finally:
                    session.close()

            return results

        return wrapper
    return decorator

# 分片管理API
def get_sharding_status() -> Dict[str, Any]:
    """获取分片状态"""
    return sharding_manager.get_shard_stats()

def add_sharding_rule(table_name: str, rule: ShardingRule):
    """添加分片规则"""
    sharding_manager.add_sharding_rule(table_name, rule)

def rebalance_shards(table_name: str, new_configs: List[ShardConfig]):
    """重新平衡分片"""
    sharding_manager.rebalance_shards(table_name, new_configs)

if __name__ == "__main__":
    # 测试分片管理器
    print("分片管理器状态:")
    print(json.dumps(get_sharding_status(), indent=2, ensure_ascii=False))

    # 测试分片选择
    test_user_id = "user_12345"
    shard_id = sharding_manager.get_shard_for_key("users", test_user_id)
    print(f"用户 {test_user_id} 分配到分片: {shard_id}")

    test_time = datetime.now()
    shard_id = sharding_manager.get_shard_for_key("glucose_predictions", test_time)
    print(f"时间 {test_time} 分配到分片: {shard_id}")

__all__ = ["'logger'", "'ShardingStrategy'", "'ShardConfig'", "'ShardingRule'", "'ConsistentHashRing'", "'DatabaseShardingManager'", "'sharding_manager'", "'sharded_operation'", "'cross_shard_query'", "'get_sharding_status'", "'add_sharding_rule'", "'rebalance_shards'"]
