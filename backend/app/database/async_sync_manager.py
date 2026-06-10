

"""
延迟写与异步数据同步系统
支持批量写入、异步同步、数据一致性保障
"""

import logging
import asyncio
import time
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import queue
import pickle
from contextlib import asynccontextmanager

from sqlalchemy import create_engine, text, MetaData, inspect
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

from backend.app.core.config import settings
from backend.app.core.exceptions import DatabaseError, CustomException
from backend.app.core.database import engine, SessionLocal
from backend.app.core.task_queue import async_task, TaskPriority
from backend.app.core.cache import cache_manager

logger = logging.getLogger(__name__)

class SyncStrategy(Enum):
    """同步策略枚举"""
    IMMEDIATE = "immediate"      # 立即同步
    BATCH = "batch"             # 批量同步
    SCHEDULED = "scheduled"     # 定时同步
    EVENT_DRIVEN = "event_driven"  # 事件驱动同步

class DataType(Enum):
    """数据类型枚举"""
    USER_ACTIVITY = "user_activity"
    GLUCOSE_DATA = "glucose_data"
    SYSTEM_LOG = "system_log"
    ANALYTICS = "analytics"
    AUDIT_LOG = "audit_log"

class SyncStatus(Enum):
    """同步状态枚举"""
    PENDING = "pending"
    SYNCING = "syncing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"

@dataclass
class WriteOperation:
    """写入操作"""
    operation_id: str
    data_type: DataType
    table_name: str
    operation_type: str  # insert, update, delete
    data: Dict[str, Any]
    timestamp: datetime
    priority: int = 5
    retry_count: int = 0
    max_retries: int = 3
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class SyncOperation:
    """同步操作"""
    sync_id: str
    data_type: DataType
    operations: List[WriteOperation]
    status: SyncStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

class DelayedWriteBuffer:
    """延迟写缓冲区"""

    def __init__(self, buffer_size: int = 1000, flush_interval: int = 5):
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.buffers: Dict[DataType, deque] = defaultdict(deque)
        self.buffer_lock = threading.Lock()
        self.is_running = False
        self.flush_thread = None

        # 启动自动刷新线程
        self.start_auto_flush()

    def add_operation(self, operation: WriteOperation):
        """添加写入操作"""
        with self.buffer_lock:
            buffer = self.buffers[operation.data_type]
            buffer.append(operation)

            # 如果缓冲区满了，立即刷新
            if len(buffer) >= self.buffer_size:
                self._flush_buffer(operation.data_type)

    def _flush_buffer(self, data_type: DataType) -> List[WriteOperation]:
        """刷新缓冲区"""
        with self.buffer_lock:
            buffer = self.buffers[data_type]
            operations = list(buffer)
            buffer.clear()
            return operations

    def flush_all_buffers(self) -> Dict[DataType, List[WriteOperation]]:
        """刷新所有缓冲区"""
        flushed_operations = {}

        with self.buffer_lock:
            for data_type in list(self.buffers.keys()):
                flushed_operations[data_type] = self._flush_buffer(data_type)

        return flushed_operations

    def get_buffer_status(self) -> Dict[str, Any]:
        """获取缓冲区状态"""
        with self.buffer_lock:
            status = {
                "total_buffers": len(self.buffers),
                "buffer_sizes": {},
                "total_operations": 0
            }

            for data_type, buffer in self.buffers.items():
                size = len(buffer)
                status["buffer_sizes"][data_type.value] = size
                status["total_operations"] += size

            return status

    def start_auto_flush(self):
        """启动自动刷新"""
        if self.is_running:
            return

        self.is_running = True

        def auto_flush_worker():
            while self.is_running:
                try:
                    time.sleep(self.flush_interval)

                    # 刷新所有缓冲区
                    flushed_operations = self.flush_all_buffers()

                    # 如果有操作需要同步，触发同步
                    for data_type, operations in flushed_operations.items():
                        if operations:
                            asyncio.create_task(self._trigger_sync(data_type, operations))

                except Exception as e:
                    logger.error(f"自动刷新异常: {e}")

        self.flush_thread = threading.Thread(target=auto_flush_worker, daemon=True)
        self.flush_thread.start()

        logger.info("延迟写缓冲区自动刷新已启动")

    def stop_auto_flush(self):
        """停止自动刷新"""
        self.is_running = False
        if self.flush_thread:
            self.flush_thread.join(timeout=5)

        logger.info("延迟写缓冲区自动刷新已停止")

    async def _trigger_sync(self, data_type: DataType, operations: List[WriteOperation]):
        """触发同步"""
        try:
            from app.database.async_sync_manager import sync_manager
            await sync_manager.sync_operations(data_type, operations)
        except Exception as e:
            logger.error(f"触发同步失败: {e}")

class AsyncSyncManager:
    """异步同步管理器"""

    def __init__(self):
        self.sync_strategies: Dict[DataType, SyncStrategy] = {}
        self.sync_queues: Dict[DataType, queue.Queue] = {}
        self.sync_workers: Dict[DataType, List[threading.Thread]] = {}
        self.sync_status: Dict[str, SyncOperation] = {}
        self.sync_history: List[SyncOperation] = []

        # 初始化默认同步策略
        self._init_default_strategies()

        # 启动同步工作线程
        self._start_sync_workers()

    def _init_default_strategies(self):
        """初始化默认同步策略"""
        self.sync_strategies[DataType.USER_ACTIVITY] = SyncStrategy.BATCH
        self.sync_strategies[DataType.GLUCOSE_DATA] = SyncStrategy.IMMEDIATE
        self.sync_strategies[DataType.SYSTEM_LOG] = SyncStrategy.SCHEDULED
        self.sync_strategies[DataType.ANALYTICS] = SyncStrategy.BATCH
        self.sync_strategies[DataType.AUDIT_LOG] = SyncStrategy.IMMEDIATE

    def _start_sync_workers(self):
        """启动同步工作线程"""
        for data_type in DataType:
            # 创建同步队列
            self.sync_queues[data_type] = queue.Queue(maxsize=1000)

            # 创建工作线程
            workers = []
            worker_count = self._get_worker_count(data_type)

            for i in range(worker_count):
                worker = threading.Thread(
                    target=self._sync_worker,
                    args=(data_type, i),
                    daemon=True
                )
                worker.start()
                workers.append(worker)

            self.sync_workers[data_type] = workers

        logger.info("异步同步工作线程已启动")

    def _get_worker_count(self, data_type: DataType) -> int:
        """获取工作线程数量"""
        strategy = self.sync_strategies.get(data_type, SyncStrategy.BATCH)

        if strategy == SyncStrategy.IMMEDIATE:
            return 2  # 立即同步需要更多线程
        elif strategy == SyncStrategy.BATCH:
            return 1  # 批量同步一个线程足够
        elif strategy == SyncStrategy.SCHEDULED:
            return 1  # 定时同步一个线程
        else:
            return 1

    def _sync_worker(self, data_type: DataType, worker_id: int):
        """同步工作线程"""
        sync_queue = self.sync_queues[data_type]

        while True:
            try:
                # 从队列获取同步操作
                sync_operation = sync_queue.get(timeout=1)

                # 执行同步
                asyncio.run(self._execute_sync(sync_operation))

                sync_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"同步工作线程异常 {data_type.value}-{worker_id}: {e}")

    async def sync_operations(self, data_type: DataType, operations: List[WriteOperation]) -> str:
        """同步操作"""
        sync_id = f"sync_{data_type.value}_{int(time.time() * 1000)}"

        sync_operation = SyncOperation(
            sync_id=sync_id,
            data_type=data_type,
            operations=operations,
            status=SyncStatus.PENDING,
            created_at=datetime.now()
        )

        # 根据策略决定同步方式
        strategy = self.sync_strategies.get(data_type, SyncStrategy.BATCH)

        if strategy == SyncStrategy.IMMEDIATE:
            # 立即同步
            await self._execute_sync(sync_operation)
        else:
            # 加入队列
            sync_queue = self.sync_queues[data_type]
            try:
                sync_queue.put_nowait(sync_operation)
                logger.debug(f"同步操作已加入队列: {sync_id}")
            except queue.Full:
                logger.error(f"同步队列已满: {data_type.value}")
                # 队列满时，尝试立即同步
                await self._execute_sync(sync_operation)

        return sync_id

    async def _execute_sync(self, sync_operation: SyncOperation):
        """执行同步"""
        sync_operation.status = SyncStatus.SYNCING
        sync_operation.started_at = datetime.now()

        try:
            # 根据数据类型选择同步方法
            if sync_operation.data_type == DataType.USER_ACTIVITY:
                await self._sync_user_activity(sync_operation)
            elif sync_operation.data_type == DataType.GLUCOSE_DATA:
                await self._sync_glucose_data(sync_operation)
            elif sync_operation.data_type == DataType.SYSTEM_LOG:
                await self._sync_system_log(sync_operation)
            elif sync_operation.data_type == DataType.ANALYTICS:
                await self._sync_analytics(sync_operation)
            elif sync_operation.data_type == DataType.AUDIT_LOG:
                await self._sync_audit_log(sync_operation)
            else:
                await self._sync_generic(sync_operation)

            sync_operation.status = SyncStatus.COMPLETED
            sync_operation.completed_at = datetime.now()

            logger.info(f"同步完成: {sync_operation.sync_id}")

        except Exception as e:
            sync_operation.status = SyncStatus.FAILED
            sync_operation.error_message = str(e)
            sync_operation.retry_count += 1

            logger.error(f"同步失败: {sync_operation.sync_id} - {e}")

            # 重试逻辑
            if sync_operation.retry_count < sync_operation.max_retries:
                sync_operation.status = SyncStatus.RETRYING
                await asyncio.sleep(2 ** sync_operation.retry_count)  # 指数退避
                await self._execute_sync(sync_operation)

        finally:
            # 记录同步历史
            self.sync_history.append(sync_operation)
            if len(self.sync_history) > 1000:
                self.sync_history = self.sync_history[-1000:]

    async def _sync_user_activity(self, sync_operation: SyncOperation):
        """同步用户活动数据"""
        with SessionLocal() as session:
            for operation in sync_operation.operations:
                if operation.operation_type == "insert":
                    # 批量插入用户活动
                    await self._batch_insert_user_activity(session, operation.data)
                elif operation.operation_type == "update":
                    await self._update_user_activity(session, operation.data)
                elif operation.operation_type == "delete":
                    await self._delete_user_activity(session, operation.data)

            session.commit()

    async def _sync_glucose_data(self, sync_operation: SyncOperation):
        """同步血糖数据"""
        with SessionLocal() as session:
            for operation in sync_operation.operations:
                if operation.operation_type == "insert":
                    await self._insert_glucose_data(session, operation.data)
                elif operation.operation_type == "update":
                    await self._update_glucose_data(session, operation.data)

            session.commit()

    async def _sync_system_log(self, sync_operation: SyncOperation):
        """同步系统日志"""
        # 系统日志可以写入文件或外部系统
        for operation in sync_operation.operations:
            await self._write_system_log(operation.data)

    async def _sync_analytics(self, sync_operation: SyncOperation):
        """同步分析数据"""
        # 分析数据可以发送到分析系统
        for operation in sync_operation.operations:
            await self._send_analytics_data(operation.data)

    async def _sync_audit_log(self, sync_operation: SyncOperation):
        """同步审计日志"""
        with SessionLocal() as session:
            for operation in sync_operation.operations:
                await self._insert_audit_log(session, operation.data)

            session.commit()

    async def _sync_generic(self, sync_operation: SyncOperation):
        """通用同步方法"""
        with SessionLocal() as session:
            for operation in sync_operation.operations:
                # 根据操作类型执行相应的SQL
                if operation.operation_type == "insert":
                    await self._generic_insert(session, operation.table_name, operation.data)
                elif operation.operation_type == "update":
                    await self._generic_update(session, operation.table_name, operation.data)
                elif operation.operation_type == "delete":
                    await self._generic_delete(session, operation.table_name, operation.data)

            session.commit()

    # 具体的同步方法实现
    async def _batch_insert_user_activity(self, session: Session, data: Dict[str, Any]):
        """批量插入用户活动"""
        # 实现批量插入逻辑
        pass

    async def _update_user_activity(self, session: Session, data: Dict[str, Any]):
        """更新用户活动"""
        # 实现更新逻辑
        pass

    async def _delete_user_activity(self, session: Session, data: Dict[str, Any]):
        """删除用户活动"""
        # 实现删除逻辑
        pass

    async def _insert_glucose_data(self, session: Session, data: Dict[str, Any]):
        """插入血糖数据"""
        # 实现血糖数据插入逻辑
        pass

    async def _update_glucose_data(self, session: Session, data: Dict[str, Any]):
        """更新血糖数据"""
        # 实现血糖数据更新逻辑
        pass

    async def _write_system_log(self, data: Dict[str, Any]):
        """写入系统日志"""
        # 实现系统日志写入逻辑
        pass

    async def _send_analytics_data(self, data: Dict[str, Any]):
        """发送分析数据"""
        # 实现分析数据发送逻辑
        pass

    async def _insert_audit_log(self, session: Session, data: Dict[str, Any]):
        """插入审计日志"""
        # 实现审计日志插入逻辑
        pass

    async def _generic_insert(self, session: Session, table_name: str, data: Dict[str, Any]):
        """通用插入方法"""
        columns = ", ".join(data.keys())
        placeholders = ", ".join([f":{key}" for key in data.keys()])

        sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        session.execute(text(sql), data)

    async def _generic_update(self, session: Session, table_name: str, data: Dict[str, Any]):
        """通用更新方法"""
        if "id" not in data:
            raise CustomException("更新操作需要ID字段")

        record_id = data.pop("id")
        set_clauses = ", ".join([f"{key} = :{key}" for key in data.keys()])

        sql = f"UPDATE {table_name} SET {set_clauses} WHERE id = :id"
        data["id"] = record_id
        session.execute(text(sql), data)

    async def _generic_delete(self, session: Session, table_name: str, data: Dict[str, Any]):
        """通用删除方法"""
        if "id" not in data:
            raise CustomException("删除操作需要ID字段")

        sql = f"DELETE FROM {table_name} WHERE id = :id"
        session.execute(text(sql), {"id": data["id"]})

    def get_sync_status(self, sync_id: str) -> Optional[SyncOperation]:
        """获取同步状态"""
        return self.sync_status.get(sync_id)

    def get_sync_history(self, limit: int = 100) -> List[SyncOperation]:
        """获取同步历史"""
        return self.sync_history[-limit:] if self.sync_history else []

    def get_sync_stats(self) -> Dict[str, Any]:
        """获取同步统计信息"""
        total_syncs = len(self.sync_history)
        completed_syncs = len([s for s in self.sync_history if s.status == SyncStatus.COMPLETED])
        failed_syncs = len([s for s in self.sync_history if s.status == SyncStatus.FAILED])

        # 按数据类型统计
        data_type_stats = defaultdict(int)
        for sync in self.sync_history:
            data_type_stats[sync.data_type.value] += 1

        return {
            "total_syncs": total_syncs,
            "completed_syncs": completed_syncs,
            "failed_syncs": failed_syncs,
            "success_rate": completed_syncs / total_syncs if total_syncs > 0 else 0,
            "data_type_distribution": dict(data_type_stats),
            "active_workers": sum(len(workers) for workers in self.sync_workers.values()),
            "queue_sizes": {
                data_type.value: queue.qsize()
                for data_type, queue in self.sync_queues.items()
            }
        }

class DelayedWriteManager:
    """延迟写管理器"""

    def __init__(self):
        self.write_buffer = DelayedWriteBuffer()
        self.sync_manager = AsyncSyncManager()
        self.is_enabled = True

    def write_operation(
        self,
        data_type: DataType,
        table_name: str,
        operation_type: str,
        data: Dict[str, Any],
        priority: int = 5,
        immediate: bool = False
    ) -> str:
        """写入操作"""
        operation_id = f"op_{data_type.value}_{int(time.time() * 1000)}"

        operation = WriteOperation(
            operation_id=operation_id,
            data_type=data_type,
            table_name=table_name,
            operation_type=operation_type,
            data=data,
            timestamp=datetime.now(),
            priority=priority
        )

        if immediate or not self.is_enabled:
            # 立即同步
            asyncio.create_task(self.sync_manager.sync_operations(data_type, [operation]))
        else:
            # 加入缓冲区
            self.write_buffer.add_operation(operation)

        return operation_id

    def flush_all(self) -> Dict[DataType, List[WriteOperation]]:
        """刷新所有缓冲区"""
        return self.write_buffer.flush_all_buffers()

    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        return {
            "enabled": self.is_enabled,
            "buffer_status": self.write_buffer.get_buffer_status(),
            "sync_stats": self.sync_manager.get_sync_stats()
        }

    def enable(self):
        """启用延迟写"""
        self.is_enabled = True
        logger.info("延迟写已启用")

    def disable(self):
        """禁用延迟写"""
        self.is_enabled = False
        # 刷新所有缓冲区
        self.flush_all()
        logger.info("延迟写已禁用")

# 全局延迟写管理器实例
delayed_write_manager = DelayedWriteManager()
sync_manager = AsyncSyncManager()

# 延迟写装饰器
def delayed_write(data_type: DataType, table_name: str, operation_type: str = "insert", priority: int = 5):
    """延迟写装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 执行原函数
            result = func(*args, **kwargs)

            # 如果返回数据，进行延迟写
            if result and isinstance(result, dict):
                operation_id = delayed_write_manager.write_operation(
                    data_type=data_type,
                    table_name=table_name,
                    operation_type=operation_type,
                    data=result,
                    priority=priority
                )

                # 将操作ID添加到结果中
                if isinstance(result, dict):
                    result["operation_id"] = operation_id

            return result

        return wrapper
    return decorator

# 异步任务
@async_task("sync_delayed_operations", TaskPriority.NORMAL)
def sync_delayed_operations_task(data_type: str, operations_data: List[Dict[str, Any]]):
    """同步延迟操作任务"""
    data_type_enum = DataType(data_type)
    operations = [
        WriteOperation(
            operation_id=op_data["operation_id"],
            data_type=data_type_enum,
            table_name=op_data["table_name"],
            operation_type=op_data["operation_type"],
            data=op_data["data"],
            timestamp=datetime.fromisoformat(op_data["timestamp"]),
            priority=op_data.get("priority", 5)
        )
        for op_data in operations_data
    ]

    return asyncio.run(sync_manager.sync_operations(data_type_enum, operations))

# 延迟写API
def write_operation(
    data_type: DataType,
    table_name: str,
    operation_type: str,
    data: Dict[str, Any],
    priority: int = 5,
    immediate: bool = False
) -> str:
    """写入操作"""
    return delayed_write_manager.write_operation(
        data_type, table_name, operation_type, data, priority, immediate
    )

def flush_all_operations() -> Dict[DataType, List[WriteOperation]]:
    """刷新所有操作"""
    return delayed_write_manager.flush_all()

def get_delayed_write_status() -> Dict[str, Any]:
    """获取延迟写状态"""
    return delayed_write_manager.get_status()

def enable_delayed_write():
    """启用延迟写"""
    delayed_write_manager.enable()

def disable_delayed_write():
    """禁用延迟写"""
    delayed_write_manager.disable()

def get_sync_status(sync_id: str) -> Optional[SyncOperation]:
    """获取同步状态"""
    return sync_manager.get_sync_status(sync_id)

def get_sync_stats() -> Dict[str, Any]:
    """获取同步统计"""
    return sync_manager.get_sync_stats()

if __name__ == "__main__":
    # 测试延迟写管理器
    print("延迟写管理器状态:")
    print(json.dumps(get_delayed_write_status(), indent=2, ensure_ascii=False))

    # 测试写入操作
    operation_id = write_operation(
        DataType.USER_ACTIVITY,
        "user_activities",
        "insert",
        {"user_id": "test_user", "action": "login", "timestamp": datetime.now().isoformat()}
    )
    print(f"写入操作ID: {operation_id}")

    # 刷新所有操作
    flushed = flush_all_operations()
    print(f"刷新了 {sum(len(ops) for ops in flushed.values())} 个操作")

    # 获取同步统计
    print("同步统计:")
    print(json.dumps(get_sync_stats(), indent=2, ensure_ascii=False))

__all__ = ["'logger'", "'SyncStrategy'", "'DataType'", "'SyncStatus'", "'WriteOperation'", "'SyncOperation'", "'DelayedWriteBuffer'", "'AsyncSyncManager'", "'DelayedWriteManager'", "'delayed_write_manager'", "'sync_manager'", "'delayed_write'", "'sync_delayed_operations_task'", "'write_operation'", "'flush_all_operations'", "'get_delayed_write_status'", "'enable_delayed_write'", "'disable_delayed_write'", "'get_sync_status'", "'get_sync_stats'"]
