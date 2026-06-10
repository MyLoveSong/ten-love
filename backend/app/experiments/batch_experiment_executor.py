

"""
批量实验执行与资源管理模块
支持批量实验提交、资源调度、任务队列管理等功能
"""

import asyncio
import uuid
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskPriority(Enum):
    """任务优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

@dataclass
class ResourceRequirement:
    """资源需求"""
    cpu_cores: int = 1
    memory_gb: float = 1.0
    gpu_count: int = 0
    gpu_memory_gb: float = 0.0
    disk_space_gb: float = 1.0
    max_execution_time: float = 3600  # 秒

@dataclass
class BatchTask:
    """批量任务"""
    task_id: str
    experiment_config: Dict[str, Any]
    resource_requirement: ResourceRequirement
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class ResourcePool:
    """资源池"""
    total_cpu_cores: int
    total_memory_gb: float
    total_gpu_count: int
    total_gpu_memory_gb: float
    total_disk_space_gb: float

    # 当前使用情况
    used_cpu_cores: int = 0
    used_memory_gb: float = 0.0
    used_gpu_count: int = 0
    used_gpu_memory_gb: float = 0.0
    used_disk_space_gb: float = 0.0

    def can_allocate(self, requirement: ResourceRequirement) -> bool:
        """检查是否可以分配资源"""
        return (
            self.used_cpu_cores + requirement.cpu_cores <= self.total_cpu_cores and
            self.used_memory_gb + requirement.memory_gb <= self.total_memory_gb and
            self.used_gpu_count + requirement.gpu_count <= self.total_gpu_count and
            self.used_gpu_memory_gb + requirement.gpu_memory_gb <= self.total_gpu_memory_gb and
            self.used_disk_space_gb + requirement.disk_space_gb <= self.total_disk_space_gb
        )

    def allocate(self, requirement: ResourceRequirement):
        """分配资源"""
        self.used_cpu_cores += requirement.cpu_cores
        self.used_memory_gb += requirement.memory_gb
        self.used_gpu_count += requirement.gpu_count
        self.used_gpu_memory_gb += requirement.gpu_memory_gb
        self.used_disk_space_gb += requirement.disk_space_gb

    def deallocate(self, requirement: ResourceRequirement):
        """释放资源"""
        self.used_cpu_cores -= requirement.cpu_cores
        self.used_memory_gb -= requirement.memory_gb
        self.used_gpu_count -= requirement.gpu_count
        self.used_gpu_memory_gb -= requirement.gpu_memory_gb
        self.used_disk_space_gb -= requirement.disk_space_gb

class BatchExperimentExecutor:
    """批量实验执行器"""

    def __init__(
        self,
        resource_pool: ResourcePool,
        max_concurrent_tasks: int = 4,
        task_timeout: float = 3600,
        results_dir: str = "batch_results"
    ):
        self.resource_pool = resource_pool
        self.max_concurrent_tasks = max_concurrent_tasks
        self.task_timeout = task_timeout
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        # 任务管理
        self.tasks: Dict[str, BatchTask] = {}
        self.task_queue: List[BatchTask] = []
        self.running_tasks: Dict[str, BatchTask] = {}

        # 执行器
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_tasks)
        self.running = False
        self.scheduler_thread: Optional[threading.Thread] = None

        # 统计信息
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'cancelled_tasks': 0,
            'total_execution_time': 0.0
        }

        self.logger = logging.getLogger(__name__)

    def submit_batch_experiments(
        self,
        experiment_configs: List[Dict[str, Any]],
        resource_requirements: Optional[List[ResourceRequirement]] = None,
        priorities: Optional[List[TaskPriority]] = None
    ) -> List[str]:
        """提交批量实验"""

        task_ids = []

        for i, config in enumerate(experiment_configs):
            task_id = str(uuid.uuid4())

            # 默认资源需求
            if resource_requirements and i < len(resource_requirements):
                resource_req = resource_requirements[i]
            else:
                resource_req = ResourceRequirement()

            # 默认优先级
            if priorities and i < len(priorities):
                priority = priorities[i]
            else:
                priority = TaskPriority.NORMAL

            task = BatchTask(
                task_id=task_id,
                experiment_config=config,
                resource_requirement=resource_req,
                priority=priority
            )

            self.tasks[task_id] = task
            self.task_queue.append(task)
            task_ids.append(task_id)

            self.stats['total_tasks'] += 1

        # 按优先级排序任务队列
        self.task_queue.sort(key=lambda t: t.priority.value, reverse=True)

        self.logger.info(f"已提交 {len(task_ids)} 个批量实验任务")
        return task_ids

    def start_scheduler(self):
        """启动任务调度器"""
        if self.running:
            return

        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()

        self.logger.info("批量实验调度器已启动")

    def stop_scheduler(self):
        """停止任务调度器"""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)

        # 取消所有待执行的任务
        for task in self.task_queue:
            task.status = TaskStatus.CANCELLED
            self.stats['cancelled_tasks'] += 1

        self.logger.info("批量实验调度器已停止")

    def _scheduler_loop(self):
        """调度器主循环"""
        while self.running:
            try:
                # 检查是否有可执行的任务
                if len(self.running_tasks) < self.max_concurrent_tasks and self.task_queue:
                    # 寻找可分配资源的任务
                    for i, task in enumerate(self.task_queue):
                        if task.status == TaskStatus.PENDING:
                            if self.resource_pool.can_allocate(task.resource_requirement):
                                # 分配资源并启动任务
                                self.resource_pool.allocate(task.resource_requirement)
                                task.status = TaskStatus.RUNNING
                                task.started_at = datetime.now()

                                # 从队列中移除并添加到运行任务
                                self.task_queue.pop(i)
                                self.running_tasks[task.task_id] = task

                                # 提交任务到线程池
                                future = self.executor.submit(self._execute_task, task)

                                self.logger.info(f"任务 {task.task_id} 已开始执行")
                                break

                # 检查超时任务
                self._check_timeout_tasks()

                # 短暂休眠
                time.sleep(1)

            except Exception as e:
                self.logger.error(f"调度器循环错误: {e}")
                time.sleep(5)

    def _execute_task(self, task: BatchTask) -> None:
        """执行单个任务"""
        try:
            # 导入实验运行器
            from app.experiments.experiment_runner import enhanced_kfold_train, enhanced_compare_models

            config = task.experiment_config
            experiment_type = config.get('type', 'kfold')

            start_time = time.time()

            if experiment_type == 'kfold':
                result = enhanced_kfold_train(
                    model_name=config['model_name'],
                    data=config['data'],
                    target=config['target'],
                    k=config.get('k', 5),
                    params=config.get('params'),
                    experiment_config=config.get('experiment_config')
                )
            elif experiment_type == 'compare':
                result = enhanced_compare_models(
                    model_names=config['model_names'],
                    data=config['data'],
                    target=config['target'],
                    runs=config.get('runs', 5),
                    experiment_config=config.get('experiment_config')
                )
            else:
                raise ValueError(f"不支持的实验类型: {experiment_type}")

            execution_time = time.time() - start_time

            # 更新任务状态
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result
            task.result['execution_time'] = execution_time

            # 保存结果
            self._save_task_result(task)

            self.stats['completed_tasks'] += 1
            self.stats['total_execution_time'] += execution_time

            self.logger.info(f"任务 {task.task_id} 执行完成，耗时 {execution_time:.2f} 秒")

        except Exception as e:
            # 任务执行失败
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            task.error = str(e)

            self.stats['failed_tasks'] += 1

            self.logger.error(f"任务 {task.task_id} 执行失败: {e}")

            # 重试逻辑
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.PENDING
                task.started_at = None
                task.completed_at = None
                task.error = None

                # 重新加入队列
                self.task_queue.append(task)
                self.task_queue.sort(key=lambda t: t.priority.value, reverse=True)

                self.logger.info(f"任务 {task.task_id} 将重试 (第 {task.retry_count} 次)")

        finally:
            # 释放资源
            self.resource_pool.deallocate(task.resource_requirement)

            # 从运行任务中移除
            if task.task_id in self.running_tasks:
                del self.running_tasks[task.task_id]

    def _check_timeout_tasks(self):
        """检查超时任务"""
        current_time = datetime.now()

        for task in list(self.running_tasks.values()):
            if task.started_at:
                elapsed_time = (current_time - task.started_at).total_seconds()
                if elapsed_time > task.resource_requirement.max_execution_time:
                    # 任务超时，标记为失败
                    task.status = TaskStatus.FAILED
                    task.completed_at = current_time
                    task.error = f"任务超时 (>{task.resource_requirement.max_execution_time}秒)"

                    # 释放资源
                    self.resource_pool.deallocate(task.resource_requirement)

                    # 从运行任务中移除
                    del self.running_tasks[task.task_id]

                    self.stats['failed_tasks'] += 1

                    self.logger.warning(f"任务 {task.task_id} 超时失败")

    def _save_task_result(self, task: BatchTask):
        """保存任务结果"""
        result_file = self.results_dir / f"{task.task_id}_result.json"

        result_data = {
            'task_id': task.task_id,
            'status': task.status.value,
            'created_at': task.created_at.isoformat(),
            'started_at': task.started_at.isoformat() if task.started_at else None,
            'completed_at': task.completed_at.isoformat() if task.completed_at else None,
            'execution_time': (task.completed_at - task.started_at).total_seconds() if task.started_at and task.completed_at else None,
            'retry_count': task.retry_count,
            'result': task.result,
            'error': task.error
        }

        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        task = self.tasks.get(task_id)
        if not task:
            return None

        return {
            'task_id': task.task_id,
            'status': task.status.value,
            'priority': task.priority.value,
            'created_at': task.created_at.isoformat(),
            'started_at': task.started_at.isoformat() if task.started_at else None,
            'completed_at': task.completed_at.isoformat() if task.completed_at else None,
            'retry_count': task.retry_count,
            'error': task.error
        }

    def get_batch_status(self) -> Dict[str, Any]:
        """获取批量执行状态"""
        return {
            'scheduler_running': self.running,
            'total_tasks': self.stats['total_tasks'],
            'pending_tasks': len(self.task_queue),
            'running_tasks': len(self.running_tasks),
            'completed_tasks': self.stats['completed_tasks'],
            'failed_tasks': self.stats['failed_tasks'],
            'cancelled_tasks': self.stats['cancelled_tasks'],
            'resource_utilization': {
                'cpu_cores': f"{self.resource_pool.used_cpu_cores}/{self.resource_pool.total_cpu_cores}",
                'memory_gb': f"{self.resource_pool.used_memory_gb:.1f}/{self.resource_pool.total_memory_gb:.1f}",
                'gpu_count': f"{self.resource_pool.used_gpu_count}/{self.resource_pool.total_gpu_count}",
                'gpu_memory_gb': f"{self.resource_pool.used_gpu_memory_gb:.1f}/{self.resource_pool.total_gpu_memory_gb:.1f}"
            },
            'average_execution_time': (
                self.stats['total_execution_time'] / self.stats['completed_tasks']
                if self.stats['completed_tasks'] > 0 else 0
            )
        }

    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        task = self.tasks.get(task_id)
        if not task:
            return False

        if task.status == TaskStatus.PENDING:
            task.status = TaskStatus.CANCELLED
            if task in self.task_queue:
                self.task_queue.remove(task)
            self.stats['cancelled_tasks'] += 1
            return True
        elif task.status == TaskStatus.RUNNING:
            # 运行中的任务无法直接取消，只能等待超时
            return False

        return False

    def get_completed_results(self) -> List[Dict[str, Any]]:
        """获取已完成任务的结果"""
        results = []

        for task in self.tasks.values():
            if task.status == TaskStatus.COMPLETED and task.result:
                results.append({
                    'task_id': task.task_id,
                    'experiment_config': task.experiment_config,
                    'result': task.result,
                    'execution_time': (task.completed_at - task.started_at).total_seconds() if task.started_at and task.completed_at else None
                })

        return results

    def generate_batch_summary_report(self) -> Dict[str, Any]:
        """生成批量执行摘要报告"""
        completed_results = self.get_completed_results()

        if not completed_results:
            return {
                'summary': '没有完成的任务',
                'total_tasks': self.stats['total_tasks'],
                'status': self.get_batch_status()
            }

        # 分析结果
        model_performances = {}
        execution_times = []

        for result in completed_results:
            config = result['experiment_config']
            experiment_result = result['result']

            if config.get('type') == 'kfold':
                model_name = config['model_name']
                score = experiment_result.get('mean', 0)
                if model_name not in model_performances:
                    model_performances[model_name] = []
                model_performances[model_name].append(score)

            if result['execution_time']:
                execution_times.append(result['execution_time'])

        # 计算统计信息
        summary = {
            'total_experiments': len(completed_results),
            'success_rate': self.stats['completed_tasks'] / self.stats['total_tasks'] if self.stats['total_tasks'] > 0 else 0,
            'average_execution_time': sum(execution_times) / len(execution_times) if execution_times else 0,
            'model_performances': {
                model: {
                    'count': len(scores),
                    'mean_score': sum(scores) / len(scores),
                    'max_score': max(scores),
                    'min_score': min(scores)
                }
                for model, scores in model_performances.items()
            },
            'best_model': max(model_performances.keys(), key=lambda m: sum(model_performances[m]) / len(model_performances[m])) if model_performances else None,
            'batch_status': self.get_batch_status()
        }

        return summary

# 默认资源池（可根据实际环境调整）
default_resource_pool = ResourcePool(
    total_cpu_cores=8,
    total_memory_gb=16.0,
    total_gpu_count=0,
    total_gpu_memory_gb=0.0,
    total_disk_space_gb=100.0
)

# 全局批量执行器实例
batch_executor = BatchExperimentExecutor(default_resource_pool)

__all__ = ["'logger'", "'TaskStatus'", "'TaskPriority'", "'ResourceRequirement'", "'BatchTask'", "'ResourcePool'", "'BatchExperimentExecutor'", "'default_resource_pool'", "'batch_executor'"]
