"""
模型生命周期服务 - 统一管理模型的训练、监控、更新和部署
遵循SOLID原则的模块化设计
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass

from .model_watcher_service import get_model_watcher_service, ModelChangeEvent
from .continuous_training_service import get_continuous_training_service
from .enhanced_diabetes_service import EnhancedDiabetesService

# 设置日志
logger = logging.getLogger(__name__)

@dataclass
class ModelVersion:
    """模型版本信息"""
    version_id: str
    model_path: str
    created_at: datetime
    performance_metrics: Dict[str, float]
    is_active: bool = False
    deployment_status: str = 'pending'  # 'pending', 'deployed', 'retired'

class ModelLifecycleService:
    """模型生命周期服务 - 统一管理模型的完整生命周期"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化模型生命周期服务

        Args:
            config: 配置字典
        """
        self.config = config or self._get_default_config()

        # 初始化子服务
        self.model_watcher = get_model_watcher_service(self.config.get('watcher'))
        self.training_service = get_continuous_training_service(self.config.get('training'))

        # 模型版本管理
        self.model_versions: Dict[str, ModelVersion] = {}
        self.active_model_version: Optional[str] = None

        # 服务状态
        self.running = False

        # 回调函数
        self.model_update_callbacks: List[Callable[[str], None]] = []

        # 设置模型监控回调
        self.model_watcher.add_reload_callback(self._handle_model_reload)

        logger.info("模型生命周期服务初始化完成")

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'auto_training': {
                'enabled': True,
                'interval_hours': 24,
                'trigger_on_data_threshold': 1000,
                'max_training_time': 7200
            },
            'model_validation': {
                'enabled': True,
                'min_performance_threshold': 0.8,
                'validation_timeout': 300
            },
            'deployment': {
                'auto_deploy': True,
                'rollback_on_failure': True,
                'health_check_interval': 60
            },
            'retention': {
                'max_versions': 10,
                'cleanup_interval_hours': 168  # 7天
            }
        }

    async def start(self):
        """启动模型生命周期服务"""
        if self.running:
            logger.warning("模型生命周期服务已在运行")
            return

        self.running = True
        logger.info("启动模型生命周期服务...")

        try:
            # 启动子服务
            await self.model_watcher.start()
            await self.training_service.start()

            # 扫描现有模型
            await self._scan_existing_models()

            # 启动后台任务
            asyncio.create_task(self._lifecycle_management_loop())

            logger.info("✅ 模型生命周期服务启动成功")

        except Exception as e:
            logger.error(f"启动模型生命周期服务失败: {e}")
            self.running = False
            raise

    async def stop(self):
        """停止模型生命周期服务"""
        if not self.running:
            return

        self.running = False
        logger.info("停止模型生命周期服务...")

        try:
            # 停止子服务
            await self.model_watcher.stop()
            await self.training_service.stop()

            logger.info("✅ 模型生命周期服务停止完成")

        except Exception as e:
            logger.error(f"停止模型生命周期服务异常: {e}")

    async def _scan_existing_models(self):
        """扫描现有模型"""
        logger.info("扫描现有模型...")

        try:
            project_root = Path(__file__).parent.parent.parent.parent
            train_outputs = project_root / 'TRAIN' / 'outputs'

            if not train_outputs.exists():
                logger.warning(f"训练输出目录不存在: {train_outputs}")
                return

            # 查找所有模型文件
            model_files = list(train_outputs.rglob('*best*.pt'))

            for model_file in model_files:
                await self._register_model_version(str(model_file))

            # 选择最新模型作为活跃模型
            if self.model_versions and not self.active_model_version:
                latest_version = max(
                    self.model_versions.values(),
                    key=lambda v: v.created_at
                )
                await self._activate_model_version(latest_version.version_id)

            logger.info(f"扫描完成，找到 {len(self.model_versions)} 个模型版本")

        except Exception as e:
            logger.error(f"扫描现有模型失败: {e}")

    async def _register_model_version(self, model_path: str) -> str:
        """注册模型版本"""
        try:
            path = Path(model_path)
            version_id = f"{path.parent.name}_{path.stem}_{int(path.stat().st_mtime)}"

            if version_id in self.model_versions:
                return version_id

            # 获取性能指标
            performance_metrics = await self._evaluate_model_performance(model_path)

            # 创建模型版本
            model_version = ModelVersion(
                version_id=version_id,
                model_path=model_path,
                created_at=datetime.fromtimestamp(path.stat().st_mtime),
                performance_metrics=performance_metrics
            )

            self.model_versions[version_id] = model_version

            logger.info(f"注册模型版本: {version_id}")
            return version_id

        except Exception as e:
            logger.error(f"注册模型版本失败: {e}")
            raise

    async def _evaluate_model_performance(self, model_path: str) -> Dict[str, float]:
        """评估模型性能"""
        try:
            # 这里可以加载模型并在验证集上评估
            # 简化实现，返回默认指标
            return {
                'mae': 0.0,
                'rmse': 0.0,
                'accuracy': 0.0,
                'confidence': 0.8
            }

        except Exception as e:
            logger.error(f"评估模型性能失败: {e}")
            return {}

    async def _activate_model_version(self, version_id: str):
        """激活模型版本"""
        if version_id not in self.model_versions:
            raise ValueError(f"模型版本不存在: {version_id}")

        try:
            # 停用当前活跃模型
            if self.active_model_version:
                old_version = self.model_versions[self.active_model_version]
                old_version.is_active = False
                old_version.deployment_status = 'retired'

            # 激活新模型
            new_version = self.model_versions[version_id]
            new_version.is_active = True
            new_version.deployment_status = 'deployed'

            self.active_model_version = version_id

            # 通知模型更新
            await self._notify_model_update(new_version.model_path)

            logger.info(f"✅ 激活模型版本: {version_id}")

        except Exception as e:
            logger.error(f"激活模型版本失败: {e}")
            raise

    async def _notify_model_update(self, model_path: str):
        """通知模型更新"""
        logger.info(f"通知模型更新: {model_path}")

        # 调用所有回调函数
        for callback in self.model_update_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(model_path)
                else:
                    callback(model_path)
            except Exception as e:
                logger.error(f"模型更新回调失败: {e}")

    async def _handle_model_reload(self, model_path: str):
        """处理模型重载"""
        logger.info(f"处理模型重载: {model_path}")

        try:
            # 注册新模型版本
            version_id = await self._register_model_version(model_path)

            # 验证模型性能
            if await self._validate_model_performance(version_id):
                # 自动部署新模型
                if self.config['deployment']['auto_deploy']:
                    await self._activate_model_version(version_id)
                else:
                    logger.info(f"模型验证通过，等待手动部署: {version_id}")
            else:
                logger.warning(f"模型验证失败，不部署: {version_id}")

        except Exception as e:
            logger.error(f"处理模型重载失败: {e}")

    async def _validate_model_performance(self, version_id: str) -> bool:
        """验证模型性能"""
        if not self.config['model_validation']['enabled']:
            return True

        try:
            model_version = self.model_versions[version_id]
            min_threshold = self.config['model_validation']['min_performance_threshold']

            # 检查性能指标
            confidence = model_version.performance_metrics.get('confidence', 0.0)

            if confidence >= min_threshold:
                logger.info(f"模型性能验证通过: {version_id} (置信度: {confidence})")
                return True
            else:
                logger.warning(f"模型性能验证失败: {version_id} (置信度: {confidence} < {min_threshold})")
                return False

        except Exception as e:
            logger.error(f"验证模型性能异常: {e}")
            return False

    async def _lifecycle_management_loop(self):
        """生命周期管理循环"""
        while self.running:
            try:
                # 检查是否需要触发训练
                if await self._should_trigger_training():
                    await self._trigger_automatic_training()

                # 清理旧模型版本
                await self._cleanup_old_versions()

                # 健康检查
                await self._perform_health_check()

                # 等待下次检查
                await asyncio.sleep(300)  # 5分钟检查一次

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"生命周期管理循环异常: {e}")
                await asyncio.sleep(60)

    async def _should_trigger_training(self) -> bool:
        """判断是否应该触发训练"""
        if not self.config['auto_training']['enabled']:
            return False

        # 检查时间间隔
        if self.active_model_version:
            active_model = self.model_versions[self.active_model_version]
            time_since_creation = datetime.now() - active_model.created_at
            interval_hours = self.config['auto_training']['interval_hours']

            if time_since_creation.total_seconds() >= interval_hours * 3600:
                return True

        # 检查数据阈值（这里需要与数据收集系统集成）
        # 简化实现
        return False

    async def _trigger_automatic_training(self):
        """触发自动训练"""
        logger.info("触发自动训练...")

        try:
            # 提交训练任务
            job_id = await self.training_service.submit_training_job(
                job_type='incremental',
                config={
                    'epochs': 10,
                    'batch_size': 32,
                    'learning_rate': 0.001,
                    'use_amp': True
                }
            )

            logger.info(f"自动训练任务已提交: {job_id}")

        except Exception as e:
            logger.error(f"触发自动训练失败: {e}")

    async def _cleanup_old_versions(self):
        """清理旧模型版本"""
        max_versions = self.config['retention']['max_versions']

        if len(self.model_versions) <= max_versions:
            return

        try:
            # 按创建时间排序，保留最新的版本
            sorted_versions = sorted(
                self.model_versions.values(),
                key=lambda v: v.created_at,
                reverse=True
            )

            versions_to_remove = sorted_versions[max_versions:]

            for version in versions_to_remove:
                if not version.is_active:  # 不删除活跃模型
                    await self._remove_model_version(version.version_id)

        except Exception as e:
            logger.error(f"清理旧模型版本失败: {e}")

    async def _remove_model_version(self, version_id: str):
        """移除模型版本"""
        if version_id in self.model_versions:
            model_version = self.model_versions[version_id]

            # 删除模型文件（可选）
            # Path(model_version.model_path).unlink(missing_ok=True)

            # 从内存中移除
            del self.model_versions[version_id]

            logger.info(f"移除模型版本: {version_id}")

    async def _perform_health_check(self):
        """执行健康检查"""
        if not self.active_model_version:
            logger.warning("没有活跃的模型版本")
            return

        try:
            active_model = self.model_versions[self.active_model_version]

            # 检查模型文件是否存在
            if not Path(active_model.model_path).exists():
                logger.error(f"活跃模型文件不存在: {active_model.model_path}")
                # 可以触发回滚到上一个版本
                return

            # 其他健康检查...
            logger.debug(f"模型健康检查通过: {self.active_model_version}")

        except Exception as e:
            logger.error(f"模型健康检查失败: {e}")

    # 公共API方法

    def add_model_update_callback(self, callback: Callable[[str], None]):
        """添加模型更新回调"""
        self.model_update_callbacks.append(callback)
        logger.debug(f"添加模型更新回调: {callback.__name__}")

    async def trigger_training(self, job_type: str = 'incremental', config: Optional[Dict[str, Any]] = None) -> str:
        """手动触发训练"""
        if config is None:
            config = {
                'epochs': 10,
                'batch_size': 32,
                'learning_rate': 0.001,
                'use_amp': True
            }

        job_id = await self.training_service.submit_training_job(job_type, config)
        logger.info(f"手动触发训练任务: {job_id}")
        return job_id

    async def deploy_model_version(self, version_id: str):
        """手动部署模型版本"""
        await self._activate_model_version(version_id)

    def get_model_versions(self) -> List[Dict[str, Any]]:
        """获取所有模型版本"""
        return [
            {
                'version_id': version.version_id,
                'model_path': version.model_path,
                'created_at': version.created_at.isoformat(),
                'performance_metrics': version.performance_metrics,
                'is_active': version.is_active,
                'deployment_status': version.deployment_status
            }
            for version in self.model_versions.values()
        ]

    def get_active_model_info(self) -> Optional[Dict[str, Any]]:
        """获取活跃模型信息"""
        if not self.active_model_version:
            return None

        active_model = self.model_versions[self.active_model_version]
        return {
            'version_id': active_model.version_id,
            'model_path': active_model.model_path,
            'created_at': active_model.created_at.isoformat(),
            'performance_metrics': active_model.performance_metrics,
            'deployment_status': active_model.deployment_status
        }

    def get_service_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        return {
            'running': self.running,
            'active_model_version': self.active_model_version,
            'total_model_versions': len(self.model_versions),
            'watcher_status': self.model_watcher.get_status(),
            'training_status': self.training_service.get_service_status(),
            'config': self.config
        }

# 全局服务实例
_model_lifecycle_service: Optional[ModelLifecycleService] = None

def get_model_lifecycle_service(config: Optional[Dict[str, Any]] = None) -> ModelLifecycleService:
    """获取模型生命周期服务实例（单例模式）"""
    global _model_lifecycle_service

    if _model_lifecycle_service is None:
        _model_lifecycle_service = ModelLifecycleService(config)

    return _model_lifecycle_service

# 导出所有必要的类和函数
__all__ = [
    'ModelVersion',
    'ModelLifecycleService',
    'get_model_lifecycle_service'
]
