

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可重现性框架
Reproducibility Framework for Core Journal Standards

本模块实现了满足核心期刊要求的可重现性保证：
1. 环境版本控制
2. 数据版本管理
3. 随机种子管理
4. 实验记录追踪
5. 结果验证机制
6. 代码质量保证
7. 文档自动生成
"""

import os
import sys
import json
import yaml
import hashlib
import platform
import subprocess
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import numpy as np
import pandas as pd
import torch
import random
from dataclasses import dataclass, field, asdict
import logging
import warnings

@dataclass
class EnvironmentInfo:
    """环境信息"""
    python_version: str
    platform_system: str
    platform_release: str
    platform_version: str
    cpu_count: int
    total_memory: str
    gpu_info: List[Dict[str, Any]] = field(default_factory=list)
    package_versions: Dict[str, str] = field(default_factory=dict)

@dataclass
class ExperimentConfig:
    """实验配置"""
    experiment_id: str
    experiment_name: str
    description: str
    author: str
    timestamp: str
    random_seed: int
    model_config: Dict[str, Any] = field(default_factory=dict)
    data_config: Dict[str, Any] = field(default_factory=dict)
    training_config: Dict[str, Any] = field(default_factory=dict)
    evaluation_config: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DatasetInfo:
    """数据集信息"""
    name: str
    version: str
    source: str
    download_date: str
    file_hash: str
    size_bytes: int
    num_samples: int
    num_features: int
    preprocessing_steps: List[str] = field(default_factory=list)
    split_ratios: Dict[str, float] = field(default_factory=dict)

@dataclass
class ModelInfo:
    """模型信息"""
    name: str
    architecture: str
    hyperparameters: Dict[str, Any]
    total_parameters: int
    trainable_parameters: int
    model_hash: str
    checkpoint_path: str

@dataclass
class ExperimentResult:
    """实验结果"""
    metrics: Dict[str, float]
    confusion_matrix: List[List[int]] = None
    feature_importance: Dict[str, float] = field(default_factory=dict)
    training_history: Dict[str, List[float]] = field(default_factory=dict)
    prediction_samples: List[Dict[str, Any]] = field(default_factory=list)
    execution_time: float = 0.0
    memory_usage: float = 0.0

class ReproducibilityFramework:
    """可重现性框架"""

    def __init__(self, base_dir: str = "./reproducibility"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)

        # 创建子目录
        (self.base_dir / "configs").mkdir(exist_ok=True)
        (self.base_dir / "data").mkdir(exist_ok=True)
        (self.base_dir / "models").mkdir(exist_ok=True)
        (self.base_dir / "results").mkdir(exist_ok=True)
        (self.base_dir / "logs").mkdir(exist_ok=True)
        (self.base_dir / "reports").mkdir(exist_ok=True)

        # 初始化日志
        self.logger = self._setup_logging()

        # 实验追踪
        self.experiment_registry = self.base_dir / "experiment_registry.json"
        self.experiments = self._load_experiment_registry()

    def _setup_logging(self) -> logging.Logger:
        """设置日志"""
        log_file = self.base_dir / "logs" / f"reproducibility_{datetime.now().strftime('%Y%m%d')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        return logging.getLogger(__name__)

    def capture_environment(self) -> EnvironmentInfo:
        """捕获环境信息"""
        import psutil

        # 基础系统信息
        env_info = EnvironmentInfo(
            python_version=platform.python_version(),
            platform_system=platform.system(),
            platform_release=platform.release(),
            platform_version=platform.version(),
            cpu_count=psutil.cpu_count(),
            total_memory=f"{psutil.virtual_memory().total / (1024**3):.2f} GB"
        )

        # GPU信息
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_info = {
                    'device_id': i,
                    'name': torch.cuda.get_device_name(i),
                    'memory_total': f"{torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB",
                    'cuda_version': torch.version.cuda
                }
                env_info.gpu_info.append(gpu_info)

        # 包版本信息
        important_packages = [
            'numpy', 'pandas', 'scikit-learn', 'torch', 'torchvision',
            'matplotlib', 'seaborn', 'plotly', 'fastapi', 'uvicorn'
        ]

        for package in important_packages:
            try:
                version = self._get_package_version(package)
                env_info.package_versions[package] = version
            except:
                env_info.package_versions[package] = "Not installed"

        return env_info

    def set_reproducible_environment(self, seed: int = 42):
        """设置可重现环境"""
        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # 确保CUDA算法的确定性
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # 设置环境变量
        os.environ['PYTHONHASHSEED'] = str(seed)

        # 禁用警告以保持输出一致性
        warnings.filterwarnings('ignore')

        self.logger.info(f"可重现环境已设置，随机种子: {seed}")

    def create_experiment_config(self, experiment_name: str, description: str,
                               author: str, **kwargs) -> ExperimentConfig:
        """创建实验配置"""
        experiment_id = self._generate_experiment_id(experiment_name)

        config = ExperimentConfig(
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            description=description,
            author=author,
            timestamp=datetime.now().isoformat(),
            random_seed=kwargs.get('random_seed', 42),
            model_config=kwargs.get('model_config', {}),
            data_config=kwargs.get('data_config', {}),
            training_config=kwargs.get('training_config', {}),
            evaluation_config=kwargs.get('evaluation_config', {})
        )

        return config

    def register_dataset(self, data: Union[pd.DataFrame, np.ndarray],
                        name: str, source: str, **kwargs) -> DatasetInfo:
        """注册数据集"""
        # 计算数据哈希
        if isinstance(data, pd.DataFrame):
            data_bytes = data.to_csv(index=False).encode()
            num_samples, num_features = data.shape
        else:
            data_bytes = data.tobytes()
            num_samples, num_features = data.shape if len(data.shape) == 2 else (len(data), 1)

        file_hash = hashlib.sha256(data_bytes).hexdigest()

        dataset_info = DatasetInfo(
            name=name,
            version=kwargs.get('version', '1.0'),
            source=source,
            download_date=datetime.now().isoformat(),
            file_hash=file_hash,
            size_bytes=len(data_bytes),
            num_samples=num_samples,
            num_features=num_features,
            preprocessing_steps=kwargs.get('preprocessing_steps', []),
            split_ratios=kwargs.get('split_ratios', {})
        )

        # 保存数据集信息
        info_file = self.base_dir / "data" / f"{name}_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(dataset_info), f, indent=2, ensure_ascii=False)

        # 保存数据（如果需要）
        if kwargs.get('save_data', True):
            if isinstance(data, pd.DataFrame):
                data_file = self.base_dir / "data" / f"{name}_data.csv"
                data.to_csv(data_file, index=False)
            else:
                data_file = self.base_dir / "data" / f"{name}_data.npy"
                np.save(data_file, data)

        self.logger.info(f"数据集已注册: {name} (哈希: {file_hash[:8]}...)")
        return dataset_info

    def register_model(self, model, name: str, hyperparameters: Dict[str, Any],
                      checkpoint_path: str = None) -> ModelInfo:
        """注册模型"""
        # 计算模型参数
        if hasattr(model, 'parameters'):
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        else:
            total_params = 0
            trainable_params = 0

        # 计算模型哈希
        if hasattr(model, 'state_dict'):
            model_bytes = pickle.dumps(model.state_dict())
        else:
            model_bytes = pickle.dumps(str(model))
        model_hash = hashlib.sha256(model_bytes).hexdigest()

        # 保存模型检查点
        if checkpoint_path is None:
            checkpoint_path = self.base_dir / "models" / f"{name}_{model_hash[:8]}.pth"

        if hasattr(model, 'state_dict'):
            torch.save(model.state_dict(), checkpoint_path)

        model_info = ModelInfo(
            name=name,
            architecture=str(type(model).__name__),
            hyperparameters=hyperparameters,
            total_parameters=total_params,
            trainable_parameters=trainable_params,
            model_hash=model_hash,
            checkpoint_path=str(checkpoint_path)
        )

        # 保存模型信息
        info_file = self.base_dir / "models" / f"{name}_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(model_info), f, indent=2, ensure_ascii=False)

        self.logger.info(f"模型已注册: {name} (哈希: {model_hash[:8]}...)")
        return model_info

    def track_experiment(self, config: ExperimentConfig,
                        env_info: EnvironmentInfo,
                        dataset_info: DatasetInfo,
                        model_info: ModelInfo,
                        results: ExperimentResult):
        """追踪实验"""
        experiment_record = {
            'config': asdict(config),
            'environment': asdict(env_info),
            'dataset': asdict(dataset_info),
            'model': asdict(model_info),
            'results': asdict(results)
        }

        # 保存完整实验记录
        experiment_file = self.base_dir / "results" / f"{config.experiment_id}.json"
        with open(experiment_file, 'w', encoding='utf-8') as f:
            json.dump(experiment_record, f, indent=2, ensure_ascii=False)

        # 更新实验注册表
        self.experiments[config.experiment_id] = {
            'name': config.experiment_name,
            'timestamp': config.timestamp,
            'author': config.author,
            'file': str(experiment_file)
        }

        self._save_experiment_registry()

        self.logger.info(f"实验已追踪: {config.experiment_id}")

    def validate_reproducibility(self, experiment_id: str,
                               tolerance: float = 1e-6) -> Dict[str, Any]:
        """验证可重现性"""
        # 加载原始实验
        original_experiment = self._load_experiment(experiment_id)
        if not original_experiment:
            return {'error': f'实验 {experiment_id} 未找到'}

        validation_results = {
            'experiment_id': experiment_id,
            'validation_timestamp': datetime.now().isoformat(),
            'reproducible': True,
            'issues': [],
            'metric_differences': {}
        }

        # 验证环境
        current_env = self.capture_environment()
        original_env = EnvironmentInfo(**original_experiment['environment'])

        env_issues = self._compare_environments(current_env, original_env)
        if env_issues:
            validation_results['issues'].extend(env_issues)

        # 验证数据
        try:
            self._validate_dataset_integrity(original_experiment['dataset'])
        except Exception as e:
            validation_results['issues'].append(f"数据验证失败: {str(e)}")
            validation_results['reproducible'] = False

        # 验证模型
        try:
            self._validate_model_integrity(original_experiment['model'])
        except Exception as e:
            validation_results['issues'].append(f"模型验证失败: {str(e)}")
            validation_results['reproducible'] = False

        return validation_results

    def generate_reproducibility_report(self, experiment_id: str) -> str:
        """生成可重现性报告"""
        experiment = self._load_experiment(experiment_id)
        if not experiment:
            return f"实验 {experiment_id} 未找到"

        report_template = """
# 可重现性报告

## 实验信息
- **实验ID**: {experiment_id}
- **实验名称**: {experiment_name}
- **描述**: {description}
- **作者**: {author}
- **时间戳**: {timestamp}
- **随机种子**: {random_seed}

## 环境信息
- **Python版本**: {python_version}
- **操作系统**: {platform_system} {platform_release}
- **CPU核心数**: {cpu_count}
- **总内存**: {total_memory}
- **GPU信息**: {gpu_info}

## 关键包版本
{package_versions}

## 数据集信息
- **名称**: {dataset_name}
- **版本**: {dataset_version}
- **来源**: {dataset_source}
- **样本数**: {num_samples}
- **特征数**: {num_features}
- **数据哈希**: {dataset_hash}

## 模型信息
- **架构**: {model_architecture}
- **总参数数**: {total_parameters:,}
- **可训练参数**: {trainable_parameters:,}
- **模型哈希**: {model_hash}

## 实验结果
{results_summary}

## 重现步骤
1. 确保Python版本为 {python_version}
2. 安装所需包版本（见上方列表）
3. 设置随机种子为 {random_seed}
4. 加载数据集（哈希验证: {dataset_hash}）
5. 初始化模型（哈希验证: {model_hash}）
6. 运行实验脚本

## 验证检查清单
- [ ] 环境版本匹配
- [ ] 数据集哈希验证
- [ ] 模型哈希验证
- [ ] 随机种子设置
- [ ] 结果数值一致性（容差: 1e-6）

---
报告生成时间: {report_timestamp}
"""

        # 格式化包版本信息
        package_versions_str = "\n".join([
            f"- **{pkg}**: {version}"
            for pkg, version in experiment['environment']['package_versions'].items()
        ])

        # 格式化结果摘要
        results_summary = "\n".join([
            f"- **{metric}**: {value}"
            for metric, value in experiment['results']['metrics'].items()
        ])

        # GPU信息格式化
        gpu_info_str = "无GPU" if not experiment['environment']['gpu_info'] else "\n".join([
            f"  - {gpu['name']} ({gpu['memory_total']})"
            for gpu in experiment['environment']['gpu_info']
        ])

        report = report_template.format(
            experiment_id=experiment['config']['experiment_id'],
            experiment_name=experiment['config']['experiment_name'],
            description=experiment['config']['description'],
            author=experiment['config']['author'],
            timestamp=experiment['config']['timestamp'],
            random_seed=experiment['config']['random_seed'],
            python_version=experiment['environment']['python_version'],
            platform_system=experiment['environment']['platform_system'],
            platform_release=experiment['environment']['platform_release'],
            cpu_count=experiment['environment']['cpu_count'],
            total_memory=experiment['environment']['total_memory'],
            gpu_info=gpu_info_str,
            package_versions=package_versions_str,
            dataset_name=experiment['dataset']['name'],
            dataset_version=experiment['dataset']['version'],
            dataset_source=experiment['dataset']['source'],
            num_samples=experiment['dataset']['num_samples'],
            num_features=experiment['dataset']['num_features'],
            dataset_hash=experiment['dataset']['file_hash'][:16],
            model_architecture=experiment['model']['architecture'],
            total_parameters=experiment['model']['total_parameters'],
            trainable_parameters=experiment['model']['trainable_parameters'],
            model_hash=experiment['model']['model_hash'][:16],
            results_summary=results_summary,
            report_timestamp=datetime.now().isoformat()
        )

        # 保存报告
        report_file = self.base_dir / "reports" / f"{experiment_id}_reproducibility_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        self.logger.info(f"可重现性报告已生成: {report_file}")
        return str(report_file)

    def create_requirements_file(self, experiment_id: str = None) -> str:
        """创建requirements.txt文件"""
        if experiment_id:
            experiment = self._load_experiment(experiment_id)
            package_versions = experiment['environment']['package_versions']
        else:
            # 使用当前环境
            env_info = self.capture_environment()
            package_versions = env_info.package_versions

        requirements_content = []
        for package, version in package_versions.items():
            if version != "Not installed":
                requirements_content.append(f"{package}=={version}")

        requirements_file = self.base_dir / "requirements_reproducible.txt"
        with open(requirements_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(requirements_content))

        return str(requirements_file)

    # 辅助方法
    def _generate_experiment_id(self, experiment_name: str) -> str:
        """生成实验ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_hash = hashlib.md5(experiment_name.encode()).hexdigest()[:6]
        return f"{timestamp}_{name_hash}"

    def _get_package_version(self, package_name: str) -> str:
        """获取包版本"""
        try:
            import importlib.metadata
            return importlib.metadata.version(package_name)
        except:
            try:
                exec(f"import {package_name}")
                package = eval(package_name)
                return getattr(package, '__version__', 'Unknown')
            except:
                return "Not installed"

    def _load_experiment_registry(self) -> Dict[str, Any]:
        """加载实验注册表"""
        if self.experiment_registry.exists():
            with open(self.experiment_registry, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_experiment_registry(self):
        """保存实验注册表"""
        with open(self.experiment_registry, 'w', encoding='utf-8') as f:
            json.dump(self.experiments, f, indent=2, ensure_ascii=False)

    def _load_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """加载实验记录"""
        experiment_file = self.base_dir / "results" / f"{experiment_id}.json"
        if experiment_file.exists():
            with open(experiment_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def _compare_environments(self, current: EnvironmentInfo,
                            original: EnvironmentInfo) -> List[str]:
        """比较环境差异"""
        issues = []

        if current.python_version != original.python_version:
            issues.append(f"Python版本不匹配: {current.python_version} vs {original.python_version}")

        if current.platform_system != original.platform_system:
            issues.append(f"操作系统不匹配: {current.platform_system} vs {original.platform_system}")

        # 检查关键包版本
        for package, original_version in original.package_versions.items():
            current_version = current.package_versions.get(package, "Not installed")
            if current_version != original_version:
                issues.append(f"包版本不匹配 {package}: {current_version} vs {original_version}")

        return issues

    def _validate_dataset_integrity(self, dataset_info: Dict[str, Any]):
        """验证数据集完整性"""
        data_file = self.base_dir / "data" / f"{dataset_info['name']}_data.csv"
        if data_file.exists():
            # 重新计算哈希
            with open(data_file, 'rb') as f:
                current_hash = hashlib.sha256(f.read()).hexdigest()

            if current_hash != dataset_info['file_hash']:
                raise ValueError(f"数据集哈希不匹配: {current_hash} vs {dataset_info['file_hash']}")

    def _validate_model_integrity(self, model_info: Dict[str, Any]):
        """验证模型完整性"""
        checkpoint_path = Path(model_info['checkpoint_path'])
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"模型检查点不存在: {checkpoint_path}")

# 使用示例
if __name__ == "__main__":
    # 创建可重现性框架
    framework = ReproducibilityFramework()

    # 设置可重现环境
    framework.set_reproducible_environment(seed=42)

    # 捕获环境信息
    env_info = framework.capture_environment()
    print(f"Python版本: {env_info.python_version}")
    print(f"GPU数量: {len(env_info.gpu_info)}")

    # 创建实验配置
    config = framework.create_experiment_config(
        experiment_name="测试实验",
        description="这是一个测试实验",
        author="研究员",
        random_seed=42
    )

    print(f"实验ID: {config.experiment_id}")

    # 生成requirements文件
    req_file = framework.create_requirements_file()
    print(f"Requirements文件: {req_file}")

__all__ = ["'EnvironmentInfo'", "'ExperimentConfig'", "'DatasetInfo'", "'ModelInfo'", "'ExperimentResult'", "'ReproducibilityFramework'"]
