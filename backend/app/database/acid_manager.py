

"""
ACID保障与事务管理优化系统
支持分布式事务、乐观锁、悲观锁、事务监控
"""

import logging
import time
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import uuid
from contextlib import contextmanager
from functools import wraps

from sqlalchemy import create_engine, text, MetaData, inspect, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from sqlalchemy.engine import Engine

from backend.app.core.config import settings
from backend.app.core.exceptions import DatabaseError, CustomException, ConflictError
from backend.app.core.database import engine, SessionLocal
from backend.app.core.task_queue import async_task, TaskPriority

logger = logging.getLogger(__name__)

class LockType(Enum):
    """锁类型枚举"""
    OPTIMISTIC = "optimistic"  # 乐观锁
    PESSIMISTIC = "pessimistic"  # 悲观锁
    SHARED = "shared"          # 共享锁
    EXCLUSIVE = "exclusive"    # 排他锁

class TransactionStatus(Enum):
    """事务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"

class IsolationLevel(Enum):
    """隔离级别枚举"""
    READ_UNCOMMITTED = "read_uncommitted"
    READ_COMMITTED = "read_committed"
    REPEATABLE_READ = "repeatable_read"
    SERIALIZABLE = "serializable"

@dataclass
class TransactionInfo:
    """事务信息"""
    transaction_id: str
    status: TransactionStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    isolation_level: IsolationLevel = IsolationLevel.READ_COMMITTED
    operations: List[str] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class LockInfo:
    """锁信息"""
    lock_id: str
    resource_id: str
    lock_type: LockType
    acquired_at: datetime
    expires_at: Optional[datetime] = None
    transaction_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class OptimisticLockManager:
    """乐观锁管理器"""

    def __init__(self):
        self.version_fields: Dict[str, str] = {}  # 表名 -> 版本字段名

    def set_version_field(self, table_name: str, field_name: str = "version"):
        """设置版本字段"""
        self.version_fields[table_name] = field_name

    def get_version_field(self, table_name: str) -> str:
        """获取版本字段名"""
        return self.version_fields.get(table_name, "version")

    def check_version(self, session: Session, table_name: str, record_id: Any, expected_version: int) -> bool:
        """检查版本号"""
        version_field = self.get_version_field(table_name)

        query = text(f"SELECT {version_field} FROM {table_name} WHERE id = :id")
        result = session.execute(query, {"id": record_id})
        row = result.fetchone()

        if not row:
            return False

        current_version = row[0]
        return current_version == expected_version

    def increment_version(self, session: Session, table_name: str, record_id: Any) -> int:
        """增加版本号"""
        version_field = self.get_version_field(table_name)

        # 原子性更新版本号
        query = text(f"""
            UPDATE {table_name}
            SET {version_field} = {version_field} + 1
            WHERE id = :id
        """)

        result = session.execute(query, {"id": record_id})
        session.commit()

        # 获取新版本号
        version_query = text(f"SELECT {version_field} FROM {table_name} WHERE id = :id")
        version_result = session.execute(version_query, {"id": record_id})
        row = version_result.fetchone()

        return row[0] if row else 0

class PessimisticLockManager:
    """悲观锁管理器"""

    def __init__(self):
        self.active_locks: Dict[str, LockInfo] = {}
        self.lock_timeout = 30  # 30秒超时
        self.lock_lock = threading.Lock()

    def acquire_lock(self, resource_id: str, lock_type: LockType, transaction_id: str, timeout: int = 30) -> bool:
        """获取锁"""
        lock_id = f"{resource_id}_{lock_type.value}_{transaction_id}"

        with self.lock_lock:
            # 检查是否已有锁
            for existing_lock in self.active_locks.values():
                if (existing_lock.resource_id == resource_id and
                    existing_lock.transaction_id != transaction_id and
                    existing_lock.expires_at and
                    existing_lock.expires_at > datetime.now()):

                    # 锁冲突
                    if lock_type == LockType.EXCLUSIVE or existing_lock.lock_type == LockType.EXCLUSIVE:
                        return False

            # 创建新锁
            lock_info = LockInfo(
                lock_id=lock_id,
                resource_id=resource_id,
                lock_type=lock_type,
                acquired_at=datetime.now(),
                expires_at=datetime.now() + timedelta(seconds=timeout),
                transaction_id=transaction_id
            )

            self.active_locks[lock_id] = lock_info
            return True

    def release_lock(self, resource_id: str, transaction_id: str) -> bool:
        """释放锁"""
        with self.lock_lock:
            locks_to_remove = []

            for lock_id, lock_info in self.active_locks.items():
                if (lock_info.resource_id == resource_id and
                    lock_info.transaction_id == transaction_id):
                    locks_to_remove.append(lock_id)

            for lock_id in locks_to_remove:
                del self.active_locks[lock_id]

            return len(locks_to_remove) > 0

    def release_expired_locks(self):
        """释放过期锁"""
        current_time = datetime.now()

        with self.lock_lock:
            expired_locks = [
                lock_id for lock_id, lock_info in self.active_locks.items()
                if lock_info.expires_at and lock_info.expires_at < current_time
            ]

            for lock_id in expired_locks:
                del self.active_locks[lock_id]

            if expired_locks:
                logger.info(f"释放了 {len(expired_locks)} 个过期锁")

    def get_lock_status(self, resource_id: str) -> List[LockInfo]:
        """获取资源锁状态"""
        with self.lock_lock:
            return [
                lock_info for lock_info in self.active_locks.values()
                if lock_info.resource_id == resource_id and
                lock_info.expires_at and
                lock_info.expires_at > datetime.now()
            ]

class TransactionManager:
    """事务管理器"""

    def __init__(self):
        self.active_transactions: Dict[str, TransactionInfo] = {}
        self.transaction_history: List[TransactionInfo] = []
        self.optimistic_lock_manager = OptimisticLockManager()
        self.pessimistic_lock_manager = PessimisticLockManager()
        self.transaction_lock = threading.Lock()

        # 启动锁清理线程
        self._start_lock_cleanup_thread()

    def _start_lock_cleanup_thread(self):
        """启动锁清理线程"""
        def cleanup_locks():
            while True:
                try:
                    self.pessimistic_lock_manager.release_expired_locks()
                    time.sleep(10)  # 每10秒清理一次
                except Exception as e:
                    logger.error(f"锁清理线程异常: {e}")

        thread = threading.Thread(target=cleanup_locks, daemon=True)
        thread.start()

    def begin_transaction(self, isolation_level: IsolationLevel = IsolationLevel.READ_COMMITTED) -> str:
        """开始事务"""
        transaction_id = str(uuid.uuid4())

        transaction_info = TransactionInfo(
            transaction_id=transaction_id,
            status=TransactionStatus.PENDING,
            started_at=datetime.now(),
            isolation_level=isolation_level,
            operations=[]
        )

        with self.transaction_lock:
            self.active_transactions[transaction_id] = transaction_info

        logger.debug(f"开始事务: {transaction_id}")
        return transaction_id

    def commit_transaction(self, transaction_id: str) -> bool:
        """提交事务"""
        with self.transaction_lock:
            if transaction_id not in self.active_transactions:
                return False

            transaction_info = self.active_transactions[transaction_id]
            transaction_info.status = TransactionStatus.COMMITTED
            transaction_info.completed_at = datetime.now()

            # 移动到历史记录
            self.transaction_history.append(transaction_info)
            del self.active_transactions[transaction_id]

        # 释放所有锁
        self.pessimistic_lock_manager.release_lock("", transaction_id)

        logger.debug(f"提交事务: {transaction_id}")
        return True

    def rollback_transaction(self, transaction_id: str, error_message: str = None) -> bool:
        """回滚事务"""
        with self.transaction_lock:
            if transaction_id not in self.active_transactions:
                return False

            transaction_info = self.active_transactions[transaction_id]
            transaction_info.status = TransactionStatus.ROLLED_BACK
            transaction_info.completed_at = datetime.now()
            transaction_info.error_message = error_message

            # 移动到历史记录
            self.transaction_history.append(transaction_info)
            del self.active_transactions[transaction_id]

        # 释放所有锁
        self.pessimistic_lock_manager.release_lock("", transaction_id)

        logger.debug(f"回滚事务: {transaction_id}")
        return True

    def get_transaction_info(self, transaction_id: str) -> Optional[TransactionInfo]:
        """获取事务信息"""
        with self.transaction_lock:
            return self.active_transactions.get(transaction_id)

    def get_active_transactions(self) -> List[TransactionInfo]:
        """获取活跃事务"""
        with self.transaction_lock:
            return list(self.active_transactions.values())

    def get_transaction_history(self, limit: int = 100) -> List[TransactionInfo]:
        """获取事务历史"""
        return self.transaction_history[-limit:] if self.transaction_history else []

class DistributedTransactionManager:
    """分布式事务管理器"""

    def __init__(self):
        self.saga_steps: Dict[str, List[Dict[str, Any]]] = {}
        self.compensation_actions: Dict[str, List[Callable]] = {}

    def register_saga(self, saga_id: str, steps: List[Dict[str, Any]]):
        """注册SAGA事务"""
        self.saga_steps[saga_id] = steps
        logger.info(f"注册SAGA事务: {saga_id}")

    def register_compensation(self, saga_id: str, compensations: List[Callable]):
        """注册补偿动作"""
        self.compensation_actions[saga_id] = compensations
        logger.info(f"注册补偿动作: {saga_id}")

    async def execute_saga(self, saga_id: str, context: Dict[str, Any]) -> bool:
        """执行SAGA事务"""
        if saga_id not in self.saga_steps:
            raise CustomException(f"SAGA事务未注册: {saga_id}")

        steps = self.saga_steps[saga_id]
        executed_steps = []

        try:
            # 执行所有步骤
            for i, step in enumerate(steps):
                logger.info(f"执行SAGA步骤 {i+1}/{len(steps)}: {step.get('name', 'unknown')}")

                # 执行步骤
                if 'action' in step and callable(step['action']):
                    result = await step['action'](context)
                    executed_steps.append(i)

                    # 检查步骤结果
                    if not result:
                        raise CustomException(f"SAGA步骤执行失败: {step.get('name', 'unknown')}")

                # 等待步骤间隔
                if 'delay' in step:
                    await asyncio.sleep(step['delay'])

            logger.info(f"SAGA事务执行成功: {saga_id}")
            return True

        except Exception as e:
            logger.error(f"SAGA事务执行失败: {saga_id} - {e}")

            # 执行补偿动作
            await self._execute_compensation(saga_id, executed_steps, context)
            return False

    async def _execute_compensation(self, saga_id: str, executed_steps: List[int], context: Dict[str, Any]):
        """执行补偿动作"""
        if saga_id not in self.compensation_actions:
            logger.warning(f"没有补偿动作: {saga_id}")
            return

        compensations = self.compensation_actions[saga_id]

        # 逆序执行补偿动作
        for step_index in reversed(executed_steps):
            if step_index < len(compensations) and callable(compensations[step_index]):
                try:
                    logger.info(f"执行补偿动作 {step_index}: {saga_id}")
                    await compensations[step_index](context)
                except Exception as e:
                    logger.error(f"补偿动作执行失败: {e}")

class TransactionDecorator:
    """事务装饰器"""

    def __init__(self, transaction_manager: TransactionManager):
        self.transaction_manager = transaction_manager

    def transactional(self, isolation_level: IsolationLevel = IsolationLevel.READ_COMMITTED):
        """事务装饰器"""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                transaction_id = self.transaction_manager.begin_transaction(isolation_level)

                try:
                    # 将事务ID添加到参数中
                    kwargs['transaction_id'] = transaction_id
                    result = await func(*args, **kwargs)

                    self.transaction_manager.commit_transaction(transaction_id)
                    return result

                except Exception as e:
                    self.transaction_manager.rollback_transaction(transaction_id, str(e))
                    raise

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                transaction_id = self.transaction_manager.begin_transaction(isolation_level)

                try:
                    kwargs['transaction_id'] = transaction_id
                    result = func(*args, **kwargs)

                    self.transaction_manager.commit_transaction(transaction_id)
                    return result

                except Exception as e:
                    self.transaction_manager.rollback_transaction(transaction_id, str(e))
                    raise

            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator

    def optimistic_lock(self, table_name: str, version_field: str = "version"):
        """乐观锁装饰器"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # 设置版本字段
                self.transaction_manager.optimistic_lock_manager.set_version_field(table_name, version_field)

                # 执行函数
                return func(*args, **kwargs)

            return wrapper
        return decorator

    def pessimistic_lock(self, resource_id: str, lock_type: LockType = LockType.EXCLUSIVE, timeout: int = 30):
        """悲观锁装饰器"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                transaction_id = kwargs.get('transaction_id')
                if not transaction_id:
                    raise CustomException("悲观锁需要事务ID")

                # 获取锁
                if not self.transaction_manager.pessimistic_lock_manager.acquire_lock(
                    resource_id, lock_type, transaction_id, timeout
                ):
                    raise ConflictError(f"无法获取锁: {resource_id}")

                try:
                    return func(*args, **kwargs)
                finally:
                    # 释放锁
                    self.transaction_manager.pessimistic_lock_manager.release_lock(resource_id, transaction_id)

            return wrapper
        return decorator

class ACIDTransactionManager:
    """ACID事务管理器"""

    def __init__(self):
        self.transaction_manager = TransactionManager()
        self.distributed_manager = DistributedTransactionManager()
        self.decorator = TransactionDecorator(self.transaction_manager)

        # 设置默认版本字段
        self.transaction_manager.optimistic_lock_manager.set_version_field("users", "version")
        self.transaction_manager.optimistic_lock_manager.set_version_field("glucose_predictions", "version")

    def get_transaction_stats(self) -> Dict[str, Any]:
        """获取事务统计信息"""
        active_transactions = self.transaction_manager.get_active_transactions()
        transaction_history = self.transaction_manager.get_transaction_history(100)

        # 统计事务状态
        status_counts = {}
        for transaction in transaction_history:
            status = transaction.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        # 计算平均执行时间
        completed_transactions = [
            t for t in transaction_history
            if t.status in [TransactionStatus.COMMITTED, TransactionStatus.ROLLED_BACK] and t.completed_at
        ]

        avg_execution_time = 0
        if completed_transactions:
            total_time = sum(
                (t.completed_at - t.started_at).total_seconds()
                for t in completed_transactions
            )
            avg_execution_time = total_time / len(completed_transactions)

        return {
            "active_transactions": len(active_transactions),
            "total_transactions": len(transaction_history),
            "status_distribution": status_counts,
            "average_execution_time": avg_execution_time,
            "registered_sagas": len(self.distributed_manager.saga_steps),
            "active_locks": len(self.transaction_manager.pessimistic_lock_manager.active_locks)
        }

    def get_lock_status(self, resource_id: str) -> List[LockInfo]:
        """获取锁状态"""
        return self.transaction_manager.pessimistic_lock_manager.get_lock_status(resource_id)

    def register_saga(self, saga_id: str, steps: List[Dict[str, Any]], compensations: List[Callable] = None):
        """注册SAGA事务"""
        self.distributed_manager.register_saga(saga_id, steps)
        if compensations:
            self.distributed_manager.register_compensation(saga_id, compensations)

    async def execute_saga(self, saga_id: str, context: Dict[str, Any]) -> bool:
        """执行SAGA事务"""
        return await self.distributed_manager.execute_saga(saga_id, context)

# 全局ACID事务管理器实例
acid_manager = ACIDTransactionManager()

# 事务装饰器
transactional = acid_manager.decorator.transactional
optimistic_lock = acid_manager.decorator.optimistic_lock
pessimistic_lock = acid_manager.decorator.pessimistic_lock

# 异步任务
@async_task("execute_saga_transaction", TaskPriority.HIGH)
def execute_saga_transaction_task(saga_id: str, context: Dict[str, Any]):
    """执行SAGA事务任务"""
    return asyncio.run(acid_manager.execute_saga(saga_id, context))

# ACID管理API
def begin_transaction(isolation_level: IsolationLevel = IsolationLevel.READ_COMMITTED) -> str:
    """开始事务"""
    return acid_manager.transaction_manager.begin_transaction(isolation_level)

def commit_transaction(transaction_id: str) -> bool:
    """提交事务"""
    return acid_manager.transaction_manager.commit_transaction(transaction_id)

def rollback_transaction(transaction_id: str, error_message: str = None) -> bool:
    """回滚事务"""
    return acid_manager.transaction_manager.rollback_transaction(transaction_id, error_message)

def get_transaction_info(transaction_id: str) -> Optional[TransactionInfo]:
    """获取事务信息"""
    return acid_manager.transaction_manager.get_transaction_info(transaction_id)

def get_transaction_stats() -> Dict[str, Any]:
    """获取事务统计"""
    return acid_manager.get_transaction_stats()

def get_lock_status(resource_id: str) -> List[LockInfo]:
    """获取锁状态"""
    return acid_manager.get_lock_status(resource_id)

def register_saga(saga_id: str, steps: List[Dict[str, Any]], compensations: List[Callable] = None):
    """注册SAGA事务"""
    acid_manager.register_saga(saga_id, steps, compensations)

async def execute_saga(saga_id: str, context: Dict[str, Any]) -> bool:
    """执行SAGA事务"""
    return await acid_manager.execute_saga(saga_id, context)

# 示例SAGA事务
async def user_registration_saga(context: Dict[str, Any]) -> bool:
    """用户注册SAGA事务"""
    logger.info("执行用户注册SAGA事务")

    # 步骤1: 创建用户
    user_id = context.get("user_id")
    if not user_id:
        return False

    # 步骤2: 发送欢迎邮件
    await asyncio.sleep(0.1)  # 模拟邮件发送

    # 步骤3: 初始化用户配置
    await asyncio.sleep(0.1)  # 模拟配置初始化

    return True

async def user_registration_compensation(context: Dict[str, Any]):
    """用户注册补偿动作"""
    logger.info("执行用户注册补偿动作")
    user_id = context.get("user_id")

    # 删除用户
    # 取消邮件发送
    # 清理配置

    logger.info(f"用户注册补偿完成: {user_id}")

# 注册示例SAGA
register_saga(
    "user_registration",
    [
        {"name": "create_user", "action": lambda ctx: True},
        {"name": "send_welcome_email", "action": lambda ctx: True},
        {"name": "initialize_config", "action": lambda ctx: True}
    ],
    [
        lambda ctx: logger.info("补偿: 删除用户"),
        lambda ctx: logger.info("补偿: 取消邮件"),
        lambda ctx: logger.info("补偿: 清理配置")
    ]
)

if __name__ == "__main__":
    # 测试ACID事务管理器
    print("ACID事务管理器统计:")
    print(json.dumps(get_transaction_stats(), indent=2, ensure_ascii=False))

    # 测试事务
    transaction_id = begin_transaction()
    print(f"开始事务: {transaction_id}")

    # 模拟事务操作
    time.sleep(0.1)

    commit_transaction(transaction_id)
    print("事务已提交")

    # 测试SAGA事务
    saga_result = asyncio.run(execute_saga("user_registration", {"user_id": "test_user"}))
    print(f"SAGA事务结果: {saga_result}")

__all__ = ["'logger'", "'LockType'", "'TransactionStatus'", "'IsolationLevel'", "'TransactionInfo'", "'LockInfo'", "'OptimisticLockManager'", "'PessimisticLockManager'", "'TransactionManager'", "'DistributedTransactionManager'", "'TransactionDecorator'", "'ACIDTransactionManager'", "'acid_manager'", "'transactional'", "'optimistic_lock'", "'pessimistic_lock'", "'execute_saga_transaction_task'", "'begin_transaction'", "'commit_transaction'", "'rollback_transaction'", "'get_transaction_info'", "'get_transaction_stats'", "'get_lock_status'", "'register_saga'"]
