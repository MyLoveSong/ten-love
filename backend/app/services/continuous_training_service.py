"""
持续训练服务 - 管理模型的持续训练和更新
遵循SOLID原则的模块化设计
"""

import os
import asyncio
import logging
import subprocess
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
import shutil

# 设置日志
logger = logging.getLogger(__name__)

@dataclass
class TrainingJob:
    """训练任务"""
    job_id: str
    job_type: str  # 'initial', 'incremental', 'online'
    config: Dict[str, Any]
    status: str  # 'pending', 'running', 'completed', 'failed'
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    output_path: Optional[str] = None
    error_message: Optional[str] = None

@dataclass
class TrainingSchedule:
    """训练调度配置"""
    schedule_type: str  # 'interval', 'cron', 'trigger'
    interval_hours: Optional[int] = None
    cron_expression: Optional[str] = None
    trigger_conditions: Optional[Dict[str, Any]] = None
    enabled: bool = True

class ITrainingExecutor(ABC):
    """训练执行器接口"""

    @abstractmethod
    async def execute_training(self, job: TrainingJob) -> bool:
        """执行训练任务"""
        pass

    @abstractmethod
    def get_supported_job_types(self) -> List[str]:
        """获取支持的任务类型"""
        pass

class PythonScriptTrainingExecutor(ITrainingExecutor):
    """Python脚本训练执行器"""

    def __init__(self, train_directory: str):
        """
        初始化Python脚本执行器

        Args:
            train_directory: TRAIN目录路径
        """
        self.train_directory = Path(train_directory)
        self.supported_types = ['initial', 'incremental', 'three_stage']

        logger.info(f"Python脚本训练执行器初始化，TRAIN目录: {train_directory}")

    def get_supported_job_types(self) -> List[str]:
        """获取支持的任务类型"""
        return self.supported_types

    async def execute_training(self, job: TrainingJob) -> bool:
        """执行训练任务"""
        try:
            logger.info(f"开始执行训练任务: {job.job_id} ({job.job_type})")

            # 根据任务类型选择执行脚本
            script_path = self._get_script_path(job.job_type)
            if not script_path.exists():
                raise FileNotFoundError(f"训练脚本不存在: {script_path}")

            # 准备执行参数
            cmd_args = self._prepare_command_args(job, script_path)

            # 执行训练脚本
            result = await self._run_training_script(cmd_args, job)

            if result:
                logger.info(f"✅ 训练任务完成: {job.job_id}")
            else:
                logger.error(f"❌ 训练任务失败: {job.job_id}")

            return result

        except Exception as e:
            logger.error(f"执行训练任务异常: {e}")
            job.error_message = str(e)
            return False

    def _get_script_path(self, job_type: str) -> Path:
        """获取脚本路径"""
        script_mapping = {
            'initial': 'auto_train.py',
            'incremental': 'auto_train.py',
            'three_stage': 'demo_three_stage_training.py',
            'online': 'stage3_online_learning_pipeline.py'
        }

        script_name = script_mapping.get(job_type, 'auto_train.py')
        return self.train_directory / script_name

    def _prepare_command_args(self, job: TrainingJob, script_path: Path) -> List[str]:
        """准备命令参数"""
        args = ['python', str(script_path)]

        # 添加配置参数
        config = job.config

        if 'epochs' in config:
            args.extend(['--epochs', str(config['epochs'])])

        if 'batch_size' in config:
            args.extend(['--batch_size', str(config['batch_size'])])

        if 'learning_rate' in config:
            args.extend(['--lr', str(config['learning_rate'])])

        if 'output_dir' in config:
            args.extend(['--output', config['output_dir']])

        if config.get('use_amp', False):
            args.append('--use-amp')

        if config.get('samples_only', False):
            args.append('--small_sample')

        return args

    async def _run_training_script(self, cmd_args: List[str], job: TrainingJob) -> bool:
        """运行训练脚本"""
        try:
            # 设置工作目录
            cwd = self.train_directory

            # 设置环境变量
            env = os.environ.copy()
            env['PYTHONPATH'] = str(cwd)

            logger.info(f"执行命令: {' '.join(cmd_args)}")
            logger.info(f"工作目录: {cwd}")

            # 异步执行子进程
            process = await asyncio.create_subprocess_exec(
                *cmd_args,
                cwd=cwd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            # 等待进程完成
            stdout, stderr = await process.communicate()

            # 记录输出
            if stdout:
                logger.info(f"训练输出:\n{stdout.decode('utf-8', errors='ignore')}")

            if stderr:
                logger.warning(f"训练错误输出:\n{stderr.decode('utf-8', errors='ignore')}")

            # 检查返回码
            if process.returncode == 0:
                # 查找输出模型
                job.output_path = self._find_output_model(job)
                return True
            else:
                job.error_message = f"训练脚本返回码: {process.returncode}"
                return False

        except Exception as e:
            logger.error(f"运行训练脚本异常: {e}")
            job.error_message = str(e)
            return False

    def _find_output_model(self, job: TrainingJob) -> Optional[str]:
        """查找输出模型"""
        output_dirs = [
            self.train_directory / 'outputs',
            self.train_directory / 'TRAIN' / 'outputs'
        ]

        for output_dir in output_dirs:
            if output_dir.exists():
                # 查找最新的模型文件
                model_files = list(output_dir.rglob('*best*.pt'))
                if model_files:
                    # 按修改时间排序，返回最新的
                    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                    return str(latest_model)

        return None

class OnlineLearningExecutor(ITrainingExecutor):
    """在线学习执行器"""

    def __init__(self, online_service_factory: Callable):
        """
        初始化在线学习执行器

        Args:
            online_service_factory: 在线学习服务工厂函数
        """
        self.online_service_factory = online_service_factory
        self.supported_types = ['online', 'feedback_update']

        logger.info("在线学习执行器初始化完成")

    def get_supported_job_types(self) -> List[str]:
        """获取支持的任务类型"""
        return self.supported_types

    async def execute_training(self, job: TrainingJob) -> bool:
        """执行在线学习任务"""
        try:
            logger.info(f"开始执行在线学习任务: {job.job_id}")

            # 创建在线学习服务
            online_service = self.online_service_factory()

            # 根据任务类型执行不同操作
            if job.job_type == 'online':
                # 启动在线学习
                await online_service.start()
                # 运行一段时间后停止
                await asyncio.sleep(job.config.get('duration', 3600))  # 默认1小时
                await online_service.stop()

            elif job.job_type == 'feedback_update':
                # 处理反馈更新
                feedback_data = job.config.get('feedback_data', [])
                for feedback in feedback_data:
                    online_service.add_feedback(feedback)

            logger.info(f"✅ 在线学习任务完成: {job.job_id}")
            return True

        except Exception as e:
            logger.error(f"在线学习任务异常: {e}")
            job.error_message = str(e)
            return False

class TrainingScheduler:
    """训练调度器"""

    def __init__(self):
        self.schedules: Dict[str, TrainingSchedule] = {}
        self.running = False
        self.scheduler_task: Optional[asyncio.Task] = None

        logger.info("训练调度器初始化完成")

    def add_schedule(self, schedule_id: str, schedule: TrainingSchedule):
        """添加调度"""
        self.schedules[schedule_id] = schedule
        logger.info(f"添加训练调度: {schedule_id} ({schedule.schedule_type})")

    def remove_schedule(self, schedule_id: str):
        """移除调度"""
        if schedule_id in self.schedules:
            del self.schedules[schedule_id]
            logger.info(f"移除训练调度: {schedule_id}")

    async def start(self):
        """启动调度器"""
        if self.running:
            return

        self.running = True
        logger.info("启动训练调度器...")

        self.scheduler_task = asyncio.create_task(self._scheduler_loop())

    async def stop(self):
        """停止调度器"""
        if not self.running:
            return

        self.running = False
        logger.info("停止训练调度器...")

        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass

    async def _scheduler_loop(self):
        """调度循环"""
        while self.running:
            try:
                current_time = datetime.now()

                for schedule_id, schedule in self.schedules.items():
                    if not schedule.enabled:
                        continue

                    if await self._should_trigger(schedule, current_time):
                        await self._trigger_training(schedule_id, schedule)

                await asyncio.sleep(60)  # 每分钟检查一次

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"调度循环异常: {e}")
                await asyncio.sleep(60)

    async def _should_trigger(self, schedule: TrainingSchedule, current_time: datetime) -> bool:
        """判断是否应该触发训练"""
        if schedule.schedule_type == 'interval' and schedule.interval_hours:
            # 简单间隔调度（这里需要存储上次执行时间）
            return True  # 简化实现

        # 其他调度类型的实现...
        return False

    async def _trigger_training(self, schedule_id: str, schedule: TrainingSchedule):
        """触发训练"""
        logger.info(f"调度触发训练: {schedule_id}")
        # 这里需要与ContinuousTrainingService集成
        pass

class ContinuousTrainingService:
    """持续训练服务 - 主服务类"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化持续训练服务

        Args:
            config: 配置字典
        """
        self.config = config or self._get_default_config()

        # 初始化组件
        self.executors: Dict[str, ITrainingExecutor] = {}
        self.scheduler = TrainingScheduler()
        self.job_queue: asyncio.Queue = asyncio.Queue()
        self.active_jobs: Dict[str, TrainingJob] = {}
        self.job_history: List[TrainingJob] = []

        # 初始化执行器
        self._initialize_executors()

        # 运行状态
        self.running = False
        self.worker_task: Optional[asyncio.Task] = None

        logger.info("持续训练服务初始化完成")

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        project_root = Path(__file__).parent.parent.parent.parent

        return {
            'train_directory': str(project_root / 'TRAIN'),
            'max_concurrent_jobs': 1,
            'job_timeout': 7200,  # 2小时
            'auto_cleanup': True,
            'backup_models': True
        }

    def _initialize_executors(self):
        """初始化执行器"""
        # Python脚本执行器
        script_executor = PythonScriptTrainingExecutor(self.config['train_directory'])
        for job_type in script_executor.get_supported_job_types():
            self.executors[job_type] = script_executor

        logger.info(f"初始化了 {len(self.executors)} 个训练执行器")

    async def submit_training_job(
        self,
        job_type: str,
        config: Dict[str, Any],
        job_id: Optional[str] = None
    ) -> str:
        """
        提交训练任务

        Args:
            job_type: 任务类型
            config: 任务配置
            job_id: 任务ID（可选）

        Returns:
            任务ID
        """
        if job_id is None:
            job_id = f"{job_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        job = TrainingJob(
            job_id=job_id,
            job_type=job_type,
            config=config,
            status='pending',
            created_at=datetime.now()
        )

        await self.job_queue.put(job)
        self.active_jobs[job_id] = job

        logger.info(f"提交训练任务: {job_id} ({job_type})")
        return job_id

    async def start(self):
        """启动服务"""
        if self.running:
            return

        self.running = True
        logger.info("启动持续训练服务...")

        # 启动工作线程
        self.worker_task = asyncio.create_task(self._worker_loop())

        # 启动调度器
        await self.scheduler.start()

    async def stop(self):
        """停止服务"""
        if not self.running:
            return

        self.running = False
        logger.info("停止持续训练服务...")

        # 停止调度器
        await self.scheduler.stop()

        # 停止工作线程
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass

    async def _worker_loop(self):
        """工作循环"""
        while self.running:
            try:
                # 获取任务
                job = await asyncio.wait_for(self.job_queue.get(), timeout=10.0)

                # 执行任务
                await self._execute_job(job)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"工作循环异常: {e}")

    async def _execute_job(self, job: TrainingJob):
        """执行任务"""
        try:
            logger.info(f"开始执行任务: {job.job_id}")

            job.status = 'running'
            job.started_at = datetime.now()

            # 获取执行器
            executor = self.executors.get(job.job_type)
            if not executor:
                raise ValueError(f"不支持的任务类型: {job.job_type}")

            # 执行任务
            success = await executor.execute_training(job)

            # 更新状态
            job.status = 'completed' if success else 'failed'
            job.completed_at = datetime.now()

            # 移动到历史记录
            if job.job_id in self.active_jobs:
                del self.active_jobs[job.job_id]

            self.job_history.append(job)

            # 清理旧历史记录
            if len(self.job_history) > 100:
                self.job_history = self.job_history[-50:]

            logger.info(f"任务执行完成: {job.job_id} ({'成功' if success else '失败'})")

        except Exception as e:
            logger.error(f"执行任务异常: {e}")
            job.status = 'failed'
            job.error_message = str(e)
            job.completed_at = datetime.now()

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        # 查找活跃任务
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
        else:
            # 查找历史任务
            job = next((j for j in self.job_history if j.job_id == job_id), None)

        if not job:
            return None

        return {
            'job_id': job.job_id,
            'job_type': job.job_type,
            'status': job.status,
            'created_at': job.created_at.isoformat(),
            'started_at': job.started_at.isoformat() if job.started_at else None,
            'completed_at': job.completed_at.isoformat() if job.completed_at else None,
            'output_path': job.output_path,
            'error_message': job.error_message
        }

    def get_service_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        return {
            'running': self.running,
            'active_jobs': len(self.active_jobs),
            'total_jobs': len(self.job_history),
            'executors': list(self.executors.keys()),
            'config': self.config
        }

# 全局服务实例
_continuous_training_service: Optional[ContinuousTrainingService] = None

def get_continuous_training_service(config: Optional[Dict[str, Any]] = None) -> ContinuousTrainingService:
    """获取持续训练服务实例（单例模式）"""
    global _continuous_training_service

    if _continuous_training_service is None:
        _continuous_training_service = ContinuousTrainingService(config)

    return _continuous_training_service

# 导出所有必要的类和函数
__all__ = [
    'TrainingJob',
    'TrainingSchedule',
    'ITrainingExecutor',
    'PythonScriptTrainingExecutor',
    'OnlineLearningExecutor',
    'TrainingScheduler',
    'ContinuousTrainingService',
    'get_continuous_training_service'
]
