"""
模型监控服务 - 监控TRAIN目录中的模型变化并自动重载
遵循SOLID原则的模块化设计
"""

import os
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
import threading
import time
from dataclasses import dataclass
from abc import ABC, abstractmethod

# 设置日志
logger = logging.getLogger(__name__)

@dataclass
class ModelChangeEvent:
    """模型变化事件"""
    model_path: str
    event_type: str  # 'created', 'modified', 'deleted'
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

class IModelWatcher(ABC):
    """模型监控接口"""

    @abstractmethod
    async def start_watching(self):
        """开始监控"""
        pass

    @abstractmethod
    async def stop_watching(self):
        """停止监控"""
        pass

    @abstractmethod
    def add_callback(self, callback: Callable[[ModelChangeEvent], None]):
        """添加回调函数"""
        pass

class FileSystemModelWatcher(IModelWatcher):
    """基于文件系统的模型监控器"""

    def __init__(
        self,
        watch_directories: List[str],
        file_patterns: List[str] = None,
        poll_interval: float = 5.0
    ):
        """
        初始化文件系统监控器

        Args:
            watch_directories: 监控目录列表
            file_patterns: 文件模式列表（如 ['*.pt', '*.pth']）
            poll_interval: 轮询间隔（秒）
        """
        self.watch_directories = [Path(d) for d in watch_directories]
        self.file_patterns = file_patterns or ['*.pt', '*.pth', '*best*.pt']
        self.poll_interval = poll_interval

        self.callbacks: List[Callable[[ModelChangeEvent], None]] = []
        self.file_states: Dict[str, float] = {}
        self.running = False
        self.watch_task: Optional[asyncio.Task] = None

        logger.info(f"文件系统模型监控器初始化，监控目录: {watch_directories}")

    def add_callback(self, callback: Callable[[ModelChangeEvent], None]):
        """添加回调函数"""
        self.callbacks.append(callback)
        logger.debug(f"添加模型变化回调函数: {callback.__name__}")

    async def start_watching(self):
        """开始监控"""
        if self.running:
            logger.warning("模型监控已在运行")
            return

        self.running = True
        logger.info("开始监控模型文件变化...")

        # 初始化文件状态
        await self._scan_initial_files()

        # 启动监控任务
        self.watch_task = asyncio.create_task(self._watch_loop())

    async def stop_watching(self):
        """停止监控"""
        if not self.running:
            return

        self.running = False
        logger.info("停止模型文件监控...")

        if self.watch_task:
            self.watch_task.cancel()
            try:
                await self.watch_task
            except asyncio.CancelledError:
                pass

    async def _scan_initial_files(self):
        """扫描初始文件状态"""
        logger.info("扫描初始模型文件状态...")

        for directory in self.watch_directories:
            if not directory.exists():
                logger.warning(f"监控目录不存在: {directory}")
                continue

            for pattern in self.file_patterns:
                for file_path in directory.rglob(pattern):
                    if file_path.is_file():
                        self.file_states[str(file_path)] = file_path.stat().st_mtime

        logger.info(f"初始扫描完成，找到 {len(self.file_states)} 个模型文件")

    async def _watch_loop(self):
        """监控循环"""
        while self.running:
            try:
                await self._check_file_changes()
                await asyncio.sleep(self.poll_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"监控循环异常: {e}")
                await asyncio.sleep(self.poll_interval)

    async def _check_file_changes(self):
        """检查文件变化"""
        current_files = {}

        # 扫描当前文件
        for directory in self.watch_directories:
            if not directory.exists():
                continue

            for pattern in self.file_patterns:
                for file_path in directory.rglob(pattern):
                    if file_path.is_file():
                        current_files[str(file_path)] = file_path.stat().st_mtime

        # 检查新文件和修改的文件
        for file_path, mtime in current_files.items():
            if file_path not in self.file_states:
                # 新文件
                await self._handle_file_event(file_path, 'created', mtime)
            elif self.file_states[file_path] != mtime:
                # 修改的文件
                await self._handle_file_event(file_path, 'modified', mtime)

        # 检查删除的文件
        for file_path in list(self.file_states.keys()):
            if file_path not in current_files:
                await self._handle_file_event(file_path, 'deleted', time.time())
                del self.file_states[file_path]

        # 更新文件状态
        self.file_states.update(current_files)

    async def _handle_file_event(self, file_path: str, event_type: str, timestamp: float):
        """处理文件事件"""
        event = ModelChangeEvent(
            model_path=file_path,
            event_type=event_type,
            timestamp=datetime.fromtimestamp(timestamp)
        )

        logger.info(f"检测到模型文件{event_type}: {file_path}")

        # 调用所有回调函数
        for callback in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"回调函数执行失败: {e}")

class ModelReloadService:
    """模型重载服务"""

    def __init__(self):
        self.reload_callbacks: List[Callable[[str], None]] = []
        self.last_reload_time: Optional[datetime] = None
        self.reload_count = 0

        logger.info("模型重载服务初始化完成")

    def add_reload_callback(self, callback: Callable[[str], None]):
        """添加重载回调"""
        self.reload_callbacks.append(callback)
        logger.debug(f"添加模型重载回调: {callback.__name__}")

    async def handle_model_change(self, event: ModelChangeEvent):
        """处理模型变化事件"""
        if event.event_type in ['created', 'modified']:
            # 等待文件写入完成
            await asyncio.sleep(2.0)

            # 验证文件是否有效
            if await self._validate_model_file(event.model_path):
                await self._reload_model(event.model_path)

    async def _validate_model_file(self, model_path: str) -> bool:
        """验证模型文件是否有效"""
        try:
            path = Path(model_path)

            # 检查文件大小（避免部分写入的文件）
            if path.stat().st_size < 1024:  # 小于1KB可能不是完整模型
                return False

            # 检查文件扩展名
            if path.suffix not in ['.pt', '.pth']:
                return False

            # 尝试加载模型头部信息
            import torch
            try:
                torch.load(model_path, map_location='cpu')
                return True
            except Exception:
                return False

        except Exception as e:
            logger.error(f"验证模型文件失败: {e}")
            return False

    async def _reload_model(self, model_path: str):
        """重载模型"""
        logger.info(f"开始重载模型: {model_path}")

        try:
            # 调用所有重载回调
            for callback in self.reload_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(model_path)
                    else:
                        callback(model_path)
                except Exception as e:
                    logger.error(f"模型重载回调失败: {e}")

            self.last_reload_time = datetime.now()
            self.reload_count += 1

            logger.info(f"✅ 模型重载完成: {model_path}")

        except Exception as e:
            logger.error(f"模型重载失败: {e}")

class ModelWatcherService:
    """模型监控服务 - 主服务类"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化模型监控服务

        Args:
            config: 配置字典
        """
        self.config = config or self._get_default_config()

        # 初始化组件
        self.watcher = FileSystemModelWatcher(
            watch_directories=self.config['watch_directories'],
            file_patterns=self.config['file_patterns'],
            poll_interval=self.config['poll_interval']
        )

        self.reload_service = ModelReloadService()

        # 连接监控器和重载服务
        self.watcher.add_callback(self.reload_service.handle_model_change)

        logger.info("模型监控服务初始化完成")

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        project_root = Path(__file__).parent.parent.parent.parent

        return {
            'watch_directories': [
                str(project_root / 'TRAIN' / 'outputs'),
                str(project_root / 'TRAIN' / 'TRAIN' / 'outputs')
            ],
            'file_patterns': ['*best*.pt', '*.pth', 'glucose_head_*.pt'],
            'poll_interval': 5.0,
            'enable_auto_reload': True
        }

    def add_reload_callback(self, callback: Callable[[str], None]):
        """添加模型重载回调"""
        self.reload_service.add_reload_callback(callback)

    async def start(self):
        """启动监控服务"""
        logger.info("启动模型监控服务...")
        await self.watcher.start_watching()

    async def stop(self):
        """停止监控服务"""
        logger.info("停止模型监控服务...")
        await self.watcher.stop_watching()

    def get_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        return {
            'running': self.watcher.running,
            'watched_files': len(self.watcher.file_states),
            'reload_count': self.reload_service.reload_count,
            'last_reload_time': self.reload_service.last_reload_time.isoformat() if self.reload_service.last_reload_time else None,
            'config': self.config
        }

# 全局服务实例
_model_watcher_service: Optional[ModelWatcherService] = None

def get_model_watcher_service(config: Optional[Dict[str, Any]] = None) -> ModelWatcherService:
    """获取模型监控服务实例（单例模式）"""
    global _model_watcher_service

    if _model_watcher_service is None:
        _model_watcher_service = ModelWatcherService(config)

    return _model_watcher_service

# 导出所有必要的类和函数
__all__ = [
    'ModelChangeEvent',
    'IModelWatcher',
    'FileSystemModelWatcher',
    'ModelReloadService',
    'ModelWatcherService',
    'get_model_watcher_service'
]
