

"""
异步任务队列系统
支持Celery和Redis队列
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor

try:
    import redis
    from redis.asyncio import Redis as AsyncRedis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None
    AsyncRedis = None

try:
    from app.celery import Celery
    from celery.result import AsyncResult
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False
    Celery = None
    AsyncResult = None

from backend.app.core.config import settings
from backend.app.core.exceptions import CustomException, DatabaseError
from backend.app.core.cache import cache_manager

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    RETRY = "retry"
    REVOKED = "revoked"

class TaskPriority(Enum):
    """任务优先级枚举"""
    LOW = 1
    NORMAL = 5
    HIGH = 10
    CRITICAL = 20

@dataclass
class TaskInfo:
    """任务信息"""
    task_id: str
    task_name: str
    args: List[Any]
    kwargs: Dict[str, Any]
    priority: TaskPriority
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

class AsyncTaskQueue:
    """异步任务队列"""

    def __init__(self, queue_name: str = "default", max_workers: int = 4):
        self.queue_name = queue_name
        self.max_workers = max_workers
        self.tasks: Dict[str, TaskInfo] = {}
        self.task_handlers: Dict[str, Callable] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.redis_client: Optional[AsyncRedis] = None
        self.is_running = False
        self.worker_threads: List[threading.Thread] = []

        # 初始化Redis连接
        if REDIS_AVAILABLE and settings.REDIS_URL:
            self._init_redis()

    def _init_redis(self):
        """初始化Redis连接"""
        try:
            self.redis_client = AsyncRedis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=False
            )
            logger.info(f"任务队列Redis连接初始化成功: {self.queue_name}")
        except Exception as e:
            logger.warning(f"任务队列Redis连接失败: {e}")
            self.redis_client = None

    def register_task(self, task_name: str, handler: Callable):
        """注册任务处理器"""
        self.task_handlers[task_name] = handler
        logger.info(f"注册任务处理器: {task_name}")

    async def enqueue_task(
        self,
        task_name: str,
        args: List[Any] = None,
        kwargs: Dict[str, Any] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        delay: Optional[int] = None,
        max_retries: int = 3,
        timeout: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """入队任务"""
        task_id = str(uuid.uuid4())

        task_info = TaskInfo(
            task_id=task_id,
            task_name=task_name,
            args=args or [],
            kwargs=kwargs or {},
            priority=priority,
            status=TaskStatus.PENDING,
            created_at=datetime.now(),
            max_retries=max_retries,
            timeout=timeout,
            metadata=metadata or {}
        )

        # 存储任务信息
        self.tasks[task_id] = task_info

        # 如果使用Redis，存储到Redis
        if self.redis_client:
            await self._store_task_to_redis(task_info)

        # 延迟执行
        if delay:
            await asyncio.sleep(delay)

        logger.info(f"任务入队: {task_name} ({task_id})")
        return task_id

    async def _store_task_to_redis(self, task_info: TaskInfo):
        """存储任务到Redis"""
        try:
            task_data = {
                "task_id": task_info.task_id,
                "task_name": task_info.task_name,
                "args": task_info.args,
                "kwargs": task_info.kwargs,
                "priority": task_info.priority.value,
                "status": task_info.status.value,
                "created_at": task_info.created_at.isoformat(),
                "max_retries": task_info.max_retries,
                "timeout": task_info.timeout,
                "metadata": task_info.metadata
            }

            # 序列化任务数据
            serialized_data = pickle.dumps(task_data)

            # 存储到Redis
            await self.redis_client.hset(
                f"task_queue:{self.queue_name}:tasks",
                task_info.task_id,
                serialized_data
            )

            # 添加到优先级队列
            await self.redis_client.zadd(
                f"task_queue:{self.queue_name}:pending",
                {task_info.task_id: task_info.priority.value}
            )

        except Exception as e:
            logger.error(f"存储任务到Redis失败: {e}")

    async def get_task_status(self, task_id: str) -> Optional[TaskInfo]:
        """获取任务状态"""
        if task_id in self.tasks:
            return self.tasks[task_id]

        # 从Redis获取
        if self.redis_client:
            return await self._get_task_from_redis(task_id)

        return None

    async def _get_task_from_redis(self, task_id: str) -> Optional[TaskInfo]:
        """从Redis获取任务"""
        try:
            task_data = await self.redis_client.hget(
                f"task_queue:{self.queue_name}:tasks",
                task_id
            )

            if task_data:
                data = pickle.loads(task_data)
                return TaskInfo(
                    task_id=data["task_id"],
                    task_name=data["task_name"],
                    args=data["args"],
                    kwargs=data["kwargs"],
                    priority=TaskPriority(data["priority"]),
                    status=TaskStatus(data["status"]),
                    created_at=datetime.fromisoformat(data["created_at"]),
                    max_retries=data["max_retries"],
                    timeout=data["timeout"],
                    metadata=data["metadata"]
                )
        except Exception as e:
            logger.error(f"从Redis获取任务失败: {e}")

        return None

    async def start_worker(self):
        """启动工作线程"""
        if self.is_running:
            return

        self.is_running = True

        # 启动多个工作线程
        for i in range(self.max_workers):
            thread = threading.Thread(
                target=self._worker_loop,
                name=f"TaskWorker-{i}",
                daemon=True
            )
            thread.start()
            self.worker_threads.append(thread)

        logger.info(f"任务队列工作线程启动: {self.max_workers}个线程")

    def _worker_loop(self):
        """工作线程循环"""
        while self.is_running:
            try:
                # 获取下一个任务
                task_id = self._get_next_task()
                if task_id:
                    self._process_task(task_id)
                else:
                    time.sleep(0.1)  # 没有任务时短暂休眠
            except Exception as e:
                logger.error(f"工作线程异常: {e}")
                time.sleep(1)

    def _get_next_task(self) -> Optional[str]:
        """获取下一个任务"""
        # 按优先级排序任务
        pending_tasks = [
            (task_id, task_info)
            for task_id, task_info in self.tasks.items()
            if task_info.status == TaskStatus.PENDING
        ]

        if not pending_tasks:
            return None

        # 按优先级排序
        pending_tasks.sort(key=lambda x: x[1].priority.value, reverse=True)

        return pending_tasks[0][0]

    def _process_task(self, task_id: str):
        """处理任务"""
        task_info = self.tasks.get(task_id)
        if not task_info:
            return

        try:
            # 更新任务状态
            task_info.status = TaskStatus.RUNNING
            task_info.started_at = datetime.now()

            # 获取任务处理器
            handler = self.task_handlers.get(task_info.task_name)
            if not handler:
                raise CustomException(f"未找到任务处理器: {task_info.task_name}")

            # 执行任务
            logger.info(f"开始执行任务: {task_info.task_name} ({task_id})")

            # 使用线程池执行任务
            future = self.executor.submit(handler, *task_info.args, **task_info.kwargs)

            # 等待任务完成
            if task_info.timeout:
                result = future.result(timeout=task_info.timeout)
            else:
                result = future.result()

            # 任务成功完成
            task_info.status = TaskStatus.SUCCESS
            task_info.completed_at = datetime.now()
            task_info.result = result

            logger.info(f"任务执行成功: {task_info.task_name} ({task_id})")

        except Exception as e:
            # 任务执行失败
            task_info.status = TaskStatus.FAILURE
            task_info.completed_at = datetime.now()
            task_info.error = str(e)
            task_info.retry_count += 1

            logger.error(f"任务执行失败: {task_info.task_name} ({task_id}) - {e}")

            # 重试逻辑
            if task_info.retry_count < task_info.max_retries:
                task_info.status = TaskStatus.RETRY
                logger.info(f"任务将重试: {task_info.task_name} ({task_id}) - 第{task_info.retry_count}次")

    async def stop_worker(self):
        """停止工作线程"""
        self.is_running = False

        # 等待工作线程结束
        for thread in self.worker_threads:
            thread.join(timeout=5)

        # 关闭线程池
        self.executor.shutdown(wait=True)

        logger.info("任务队列工作线程已停止")

    async def get_queue_stats(self) -> Dict[str, Any]:
        """获取队列统计信息"""
        total_tasks = len(self.tasks)
        pending_tasks = len([t for t in self.tasks.values() if t.status == TaskStatus.PENDING])
        running_tasks = len([t for t in self.tasks.values() if t.status == TaskStatus.RUNNING])
        completed_tasks = len([t for t in self.tasks.values() if t.status == TaskStatus.SUCCESS])
        failed_tasks = len([t for t in self.tasks.values() if t.status == TaskStatus.FAILURE])

        return {
            "queue_name": self.queue_name,
            "max_workers": self.max_workers,
            "is_running": self.is_running,
            "total_tasks": total_tasks,
            "pending_tasks": pending_tasks,
            "running_tasks": running_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "registered_handlers": len(self.task_handlers)
        }

    async def cleanup_completed_tasks(self, older_than_hours: int = 24):
        """清理已完成的任务"""
        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)

        tasks_to_remove = [
            task_id for task_id, task_info in self.tasks.items()
            if task_info.status in [TaskStatus.SUCCESS, TaskStatus.FAILURE] and
            task_info.completed_at and
            task_info.completed_at < cutoff_time
        ]

        for task_id in tasks_to_remove:
            del self.tasks[task_id]

        logger.info(f"清理了 {len(tasks_to_remove)} 个已完成的任务")
        return len(tasks_to_remove)

class CeleryTaskQueue:
    """Celery任务队列"""

    def __init__(self, broker_url: str = None, result_backend: str = None):
        if not CELERY_AVAILABLE:
            raise CustomException("Celery未安装，无法使用Celery任务队列")

        self.broker_url = broker_url or settings.REDIS_URL
        self.result_backend = result_backend or settings.REDIS_URL

        # 创建Celery应用
        self.celery_app = Celery(
            'academic_system',
            broker=self.broker_url,
            backend=self.result_backend
        )

        # 配置Celery
        self.celery_app.conf.update(
            task_serializer='json',
            accept_content=['json'],
            result_serializer='json',
            timezone='UTC',
            enable_utc=True,
            task_track_started=True,
            task_time_limit=30 * 60,  # 30分钟
            task_soft_time_limit=25 * 60,  # 25分钟
            worker_prefetch_multiplier=1,
            task_acks_late=True,
            worker_disable_rate_limits=True
        )

        logger.info("Celery任务队列初始化成功")

    def register_task(self, task_name: str, handler: Callable):
        """注册任务处理器"""
        self.celery_app.task(name=task_name)(handler)
        logger.info(f"注册Celery任务: {task_name}")

    def enqueue_task(
        self,
        task_name: str,
        args: List[Any] = None,
        kwargs: Dict[str, Any] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        delay: Optional[int] = None,
        max_retries: int = 3,
        timeout: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """入队任务"""
        try:
            # 构建任务参数
            task_kwargs = kwargs or {}
            if metadata:
                task_kwargs['metadata'] = metadata

            # 发送任务
            result = self.celery_app.send_task(
                task_name,
                args=args or [],
                kwargs=task_kwargs,
                countdown=delay,
                retry=True,
                max_retries=max_retries,
                time_limit=timeout
            )

            logger.info(f"Celery任务入队: {task_name} ({result.id})")
            return result.id

        except Exception as e:
            logger.error(f"Celery任务入队失败: {e}")
            raise CustomException(f"任务入队失败: {e}")

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        try:
            result = AsyncResult(task_id, app=self.celery_app)

            return {
                "task_id": task_id,
                "status": result.status,
                "result": result.result if result.successful() else None,
                "error": str(result.result) if result.failed() else None,
                "info": result.info
            }
        except Exception as e:
            logger.error(f"获取Celery任务状态失败: {e}")
            return None

# 全局任务队列实例
task_queue = AsyncTaskQueue("academic_system", max_workers=4)

# 任务装饰器
def async_task(
    task_name: str = None,
    priority: TaskPriority = TaskPriority.NORMAL,
    max_retries: int = 3,
    timeout: Optional[int] = None
):
    """异步任务装饰器"""
    def decorator(func):
        name = task_name or func.__name__

        # 注册任务处理器
        task_queue.register_task(name, func)

        async def wrapper(*args, **kwargs):
            # 入队任务
            task_id = await task_queue.enqueue_task(
                task_name=name,
                args=list(args),
                kwargs=kwargs,
                priority=priority,
                max_retries=max_retries,
                timeout=timeout
            )

            return task_id

        return wrapper
    return decorator

# 数据库相关任务
@async_task("process_glucose_data", TaskPriority.HIGH)
def process_glucose_data(user_id: str, data: Dict[str, Any]):
    """处理血糖数据"""
    # 这里实现血糖数据处理逻辑
    logger.info(f"处理血糖数据: 用户{user_id}")
    return {"processed": True, "user_id": user_id}

@async_task("generate_recommendations", TaskPriority.NORMAL)
def generate_recommendations(user_id: str, preferences: Dict[str, Any]):
    """生成推荐"""
    # 这里实现推荐生成逻辑
    logger.info(f"生成推荐: 用户{user_id}")
    return {"recommendations": [], "user_id": user_id}

@async_task("backup_database", TaskPriority.LOW)
def backup_database(backup_path: str):
    """备份数据库"""
    # 这里实现数据库备份逻辑
    logger.info(f"备份数据库到: {backup_path}")
    return {"backup_path": backup_path, "success": True}

@async_task("cleanup_old_data", TaskPriority.LOW)
def cleanup_old_data(days: int = 90):
    """清理旧数据"""
    # 这里实现数据清理逻辑
    logger.info(f"清理{days}天前的旧数据")
    return {"cleaned_count": 0, "days": days}

# 任务队列管理API
async def start_task_queue():
    """启动任务队列"""
    await task_queue.start_worker()
    logger.info("任务队列已启动")

async def stop_task_queue():
    """停止任务队列"""
    await task_queue.stop_worker()
    logger.info("任务队列已停止")

async def get_task_queue_stats() -> Dict[str, Any]:
    """获取任务队列统计"""
    return await task_queue.get_queue_stats()

async def enqueue_task(
    task_name: str,
    args: List[Any] = None,
    kwargs: Dict[str, Any] = None,
    priority: TaskPriority = TaskPriority.NORMAL,
    delay: Optional[int] = None
) -> str:
    """入队任务"""
    return await task_queue.enqueue_task(
        task_name=task_name,
        args=args,
        kwargs=kwargs,
        priority=priority,
        delay=delay
    )

async def get_task_status(task_id: str) -> Optional[TaskInfo]:
    """获取任务状态"""
    return await task_queue.get_task_status(task_id)

if __name__ == "__main__":
    # 测试任务队列
    async def test_task_queue():
        # 启动任务队列
        await start_task_queue()

        # 入队任务
        task_id = await enqueue_task(
            "process_glucose_data",
            args=["user_123"],
            kwargs={"data": {"glucose": 120}},
            priority=TaskPriority.HIGH
        )

        print(f"任务入队: {task_id}")

        # 等待任务完成
        await asyncio.sleep(2)

        # 获取任务状态
        status = await get_task_status(task_id)
        print(f"任务状态: {status}")

        # 获取队列统计
        stats = await get_task_queue_stats()
        print(f"队列统计: {stats}")

        # 停止任务队列
        await stop_task_queue()

    asyncio.run(test_task_queue())

__all__ = ["'logger'", "'TaskStatus'", "'TaskPriority'", "'TaskInfo'", "'AsyncTaskQueue'", "'CeleryTaskQueue'", "'task_queue'", "'async_task'", "'process_glucose_data'", "'generate_recommendations'", "'backup_database'", "'cleanup_old_data'"]
