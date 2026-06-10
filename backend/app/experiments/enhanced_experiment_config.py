

"""
增强实验配置与版本控制模块
支持实验配置参数化、版本控制、环境记录和复现性保证
"""

import json
import hashlib
import uuid
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pathlib import Path
import platform
import psutil
import sys
import logging

logger = logging.getLogger(__name__)

@dataclass
class HardwareInfo:
    """硬件环境信息"""
    cpu_count: int
    cpu_model: str
    memory_total: int  # bytes
    platform: str
    python_version: str
    gpu_available: bool = False
    gpu_count: int = 0
    gpu_models: List[str] = field(default_factory=list)

@dataclass
class SoftwareInfo:
    """软件环境信息"""
    os_name: str
    os_version: str
    python_version: str
    package_versions: Dict[str, str] = field(default_factory=dict)
    environment_variables: Dict[str, str] = field(default_factory=dict)

@dataclass
class ExperimentConfig:
    """实验配置"""
    experiment_id: str
    experiment_name: str
    description: str
    author: str
    timestamp: str
    version: str = "1.0"
    parent_experiment_id: Optional[str] = None  # 用于追踪实验变体

    # 核心配置
    random_seed: int = 42
    model_config: Dict[str, Any] = field(default_factory=dict)
    data_config: Dict[str, Any] = field(default_factory=dict)
    training_config: Dict[str, Any] = field(default_factory=dict)
    evaluation_config: Dict[str, Any] = field(default_factory=dict)

    # 高级配置
    cross_validation_config: Dict[str, Any] = field(default_factory=dict)
    statistical_config: Dict[str, Any] = field(default_factory=dict)
    reproducibility_config: Dict[str, Any] = field(default_factory=dict)

    # 元数据
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    expected_duration: Optional[float] = None  # 预期运行时间（秒）
    resource_requirements: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExperimentSnapshot:
    """实验快照"""
    config: ExperimentConfig
    hardware_info: HardwareInfo
    software_info: SoftwareInfo
    config_hash: str
    timestamp: str

class ExperimentConfigManager:
    """实验配置管理器"""

    def __init__(self, base_dir: str = "experiment_configs"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.configs_dir = self.base_dir / "configs"
        self.snapshots_dir = self.base_dir / "snapshots"
        self.configs_dir.mkdir(exist_ok=True)
        self.snapshots_dir.mkdir(exist_ok=True)

        self.logger = logging.getLogger(__name__)

    def create_experiment_config(
        self,
        experiment_name: str,
        description: str,
        author: str,
        model_config: Dict[str, Any],
        data_config: Dict[str, Any],
        training_config: Dict[str, Any],
        evaluation_config: Dict[str, Any],
        parent_experiment_id: Optional[str] = None,
        **kwargs
    ) -> ExperimentConfig:
        """创建新的实验配置"""

        experiment_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        config = ExperimentConfig(
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            description=description,
            author=author,
            timestamp=timestamp,
            parent_experiment_id=parent_experiment_id,
            model_config=model_config,
            data_config=data_config,
            training_config=training_config,
            evaluation_config=evaluation_config,
            **kwargs
        )

        # 生成配置哈希
        config_hash = self._generate_config_hash(config)

        # 保存配置
        self._save_config(config)

        # 创建快照
        snapshot = self._create_snapshot(config, config_hash)
        self._save_snapshot(snapshot)

        self.logger.info(f"创建实验配置: {experiment_id}")
        return config

    def load_experiment_config(self, experiment_id: str) -> Optional[ExperimentConfig]:
        """加载实验配置"""
        config_file = self.configs_dir / f"{experiment_id}.json"

        if not config_file.exists():
            self.logger.warning(f"实验配置不存在: {experiment_id}")
            return None

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            return ExperimentConfig(**data)
        except Exception as e:
            self.logger.error(f"加载实验配置失败: {e}")
            return None

    def clone_experiment_config(
        self,
        source_experiment_id: str,
        new_name: str,
        modifications: Optional[Dict[str, Any]] = None
    ) -> Optional[ExperimentConfig]:
        """克隆实验配置"""

        source_config = self.load_experiment_config(source_experiment_id)
        if not source_config:
            return None

        # 创建新配置
        new_config = ExperimentConfig(
            experiment_id=str(uuid.uuid4()),
            experiment_name=new_name,
            description=f"克隆自: {source_config.experiment_name}",
            author=source_config.author,
            timestamp=datetime.now().isoformat(),
            parent_experiment_id=source_experiment_id,
            version=self._increment_version(source_config.version),
            model_config=source_config.model_config.copy(),
            data_config=source_config.data_config.copy(),
            training_config=source_config.training_config.copy(),
            evaluation_config=source_config.evaluation_config.copy(),
            cross_validation_config=source_config.cross_validation_config.copy(),
            statistical_config=source_config.statistical_config.copy(),
            reproducibility_config=source_config.reproducibility_config.copy(),
            tags=source_config.tags.copy(),
            notes=source_config.notes,
            expected_duration=source_config.expected_duration,
            resource_requirements=source_config.resource_requirements.copy()
        )

        # 应用修改
        if modifications:
            for key, value in modifications.items():
                if hasattr(new_config, key):
                    setattr(new_config, key, value)

        # 保存新配置
        self._save_config(new_config)

        config_hash = self._generate_config_hash(new_config)
        snapshot = self._create_snapshot(new_config, config_hash)
        self._save_snapshot(snapshot)

        self.logger.info(f"克隆实验配置: {source_experiment_id} -> {new_config.experiment_id}")
        return new_config

    def get_experiment_history(self, experiment_id: str) -> List[ExperimentSnapshot]:
        """获取实验历史版本"""
        history_file = self.snapshots_dir / f"{experiment_id}_history.json"

        if not history_file.exists():
            return []

        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            return [ExperimentSnapshot(**snapshot) for snapshot in data]
        except Exception as e:
            self.logger.error(f"加载实验历史失败: {e}")
            return []

    def validate_reproducibility(self, experiment_id: str) -> Dict[str, Any]:
        """验证实验可复现性"""
        config = self.load_experiment_config(experiment_id)
        if not config:
            return {"valid": False, "error": "配置不存在"}

        # 检查环境一致性
        current_hardware = self._get_hardware_info()
        current_software = self._get_software_info()

        # 获取原始快照
        history = self.get_experiment_history(experiment_id)
        if not history:
            return {"valid": False, "error": "无历史快照"}

        original_snapshot = history[0]

        # 比较环境
        hardware_diff = self._compare_hardware(current_hardware, original_snapshot.hardware_info)
        software_diff = self._compare_software(current_software, original_snapshot.software_info)

        return {
            "valid": len(hardware_diff) == 0 and len(software_diff) == 0,
            "hardware_differences": hardware_diff,
            "software_differences": software_diff,
            "config_hash": original_snapshot.config_hash,
            "current_config_hash": self._generate_config_hash(config)
        }

    def _get_hardware_info(self) -> HardwareInfo:
        """获取硬件信息"""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            gpu_available = len(gpus) > 0
            gpu_count = len(gpus)
            gpu_models = [gpu.name for gpu in gpus]
        except ImportError:
            gpu_available = False
            gpu_count = 0
            gpu_models = []

        return HardwareInfo(
            cpu_count=psutil.cpu_count(),
            cpu_model=platform.processor(),
            memory_total=psutil.virtual_memory().total,
            platform=platform.platform(),
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            gpu_available=gpu_available,
            gpu_count=gpu_count,
            gpu_models=gpu_models
        )

    def _get_software_info(self) -> SoftwareInfo:
        """获取软件信息"""
        import pkg_resources

        # 获取关键包版本
        key_packages = [
            'numpy', 'pandas', 'scikit-learn', 'tensorflow', 'torch',
            'fastapi', 'sqlalchemy', 'pydantic'
        ]

        package_versions = {}
        for package in key_packages:
            try:
                version = pkg_resources.get_distribution(package).version
                package_versions[package] = version
            except pkg_resources.DistributionNotFound:
                package_versions[package] = "not_installed"

        return SoftwareInfo(
            os_name=platform.system(),
            os_version=platform.release(),
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            package_versions=package_versions,
            environment_variables=dict(os.environ)
        )

    def _create_snapshot(self, config: ExperimentConfig, config_hash: str) -> ExperimentSnapshot:
        """创建实验快照"""
        return ExperimentSnapshot(
            config=config,
            hardware_info=self._get_hardware_info(),
            software_info=self._get_software_info(),
            config_hash=config_hash,
            timestamp=datetime.now().isoformat()
        )

    def _generate_config_hash(self, config: ExperimentConfig) -> str:
        """生成配置哈希"""
        config_dict = asdict(config)
        # 排除时间戳和ID等变化字段
        config_dict.pop('timestamp', None)
        config_dict.pop('experiment_id', None)

        config_str = json.dumps(config_dict, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def _save_config(self, config: ExperimentConfig):
        """保存实验配置"""
        config_file = self.configs_dir / f"{config.experiment_id}.json"

        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(config), f, indent=2, ensure_ascii=False)

    def _save_snapshot(self, snapshot: ExperimentSnapshot):
        """保存实验快照"""
        snapshot_file = self.snapshots_dir / f"{snapshot.config.experiment_id}_snapshot.json"

        with open(snapshot_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(snapshot), f, indent=2, ensure_ascii=False)

        # 更新历史记录
        self._update_history(snapshot)

    def _update_history(self, snapshot: ExperimentSnapshot):
        """更新历史记录"""
        history_file = self.snapshots_dir / f"{snapshot.config.experiment_id}_history.json"

        history = []
        if history_file.exists():
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            except Exception:
                pass

        history.append(asdict(snapshot))

        # 只保留最近10个版本
        if len(history) > 10:
            history = history[-10:]

        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    def _increment_version(self, version: str) -> str:
        """递增版本号"""
        try:
            parts = version.split('.')
            if len(parts) >= 2:
                major, minor = int(parts[0]), int(parts[1])
                return f"{major}.{minor + 1}"
            else:
                return f"{version}.1"
        except ValueError:
            return f"{version}.1"

    def _compare_hardware(self, current: HardwareInfo, original: HardwareInfo) -> List[str]:
        """比较硬件差异"""
        differences = []

        if current.cpu_count != original.cpu_count:
            differences.append(f"CPU核心数: {original.cpu_count} -> {current.cpu_count}")

        if current.memory_total != original.memory_total:
            differences.append(f"内存大小: {original.memory_total} -> {current.memory_total}")

        if current.platform != original.platform:
            differences.append(f"平台: {original.platform} -> {current.platform}")

        return differences

    def _compare_software(self, current: SoftwareInfo, original: SoftwareInfo) -> List[str]:
        """比较软件差异"""
        differences = []

        if current.python_version != original.python_version:
            differences.append(f"Python版本: {original.python_version} -> {current.python_version}")

        # 比较关键包版本
        for package in ['numpy', 'pandas', 'scikit-learn']:
            if package in current.package_versions and package in original.package_versions:
                if current.package_versions[package] != original.package_versions[package]:
                    differences.append(f"{package}: {original.package_versions[package]} -> {current.package_versions[package]}")

        return differences

# 全局配置管理器实例
config_manager = ExperimentConfigManager()

__all__ = ["'logger'", "'HardwareInfo'", "'SoftwareInfo'", "'ExperimentConfig'", "'ExperimentSnapshot'", "'ExperimentConfigManager'", "'config_manager'"]
