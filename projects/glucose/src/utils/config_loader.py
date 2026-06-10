#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
配置加载器
支持YAML配置文件和动态配置调整
"""

import os
import sys
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConfigLoader:
    """配置加载器 - 单一职责原则"""

    def __init__(self, config_path: str = "configs/enhanced_training_configs.yaml"):
        self.config_path = Path(config_path)
        self.configs = {}
        self._load_configs()

    def _load_configs(self):
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.configs = yaml.safe_load(f)
            logger.info(f"配置文件加载成功: {self.config_path}")
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
            self.configs = self._get_default_configs()

    def _get_default_configs(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'standard': {
                'data': {
                    'sample_size': 2000,
                    'use_real_data': True,
                    'quality_enhancement': True
                },
                'model': {
                    'cultural_feature_dim': 64,
                    'hidden_dim': 128,
                    'lora_rank': 16,
                    'dropout': 0.1
                },
                'training': {
                    'num_epochs': 50,
                    'batch_size': 64,
                    'learning_rate': 0.001,
                    'patience': 15
                },
                'optimization': {
                    'optimizer': {'type': 'AdamW'},
                    'scheduler': {'type': 'CosineAnnealingWarmRestarts'},
                    'loss': {'type': 'LabelSmoothingLoss'}
                }
            }
        }

    def get_config(self, config_name: str = 'standard') -> Dict[str, Any]:
        """获取指定配置"""
        if config_name not in self.configs:
            logger.warning(f"配置 '{config_name}' 不存在，使用默认配置")
            # 如果standard也不存在，使用默认配置
            if 'standard' not in self.configs:
                return self._get_default_configs()['standard']
            config_name = 'standard'

        config = self.configs.get(config_name)

        # 应用硬件适配
        config = self._apply_hardware_adaptation(config)

        return config

    def _apply_hardware_adaptation(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """应用硬件适配"""
        # 检测硬件配置
        hardware_type = self._detect_hardware()

        if 'hardware_configs' in self.configs and hardware_type in self.configs['hardware_configs']:
            hardware_config = self.configs['hardware_configs'][hardware_type]

            # 调整批次大小
            if 'batch_size_multiplier' in hardware_config:
                multiplier = hardware_config['batch_size_multiplier']
                original_batch_size = config['training']['batch_size']
                config['training']['batch_size'] = max(1, int(original_batch_size * multiplier))
                logger.info(f"硬件适配: 批次大小 {original_batch_size} -> {config['training']['batch_size']}")

            # 调整混合精度
            if 'mixed_precision' in hardware_config:
                if 'optimization' not in config:
                    config['optimization'] = {}
                if 'mixed_precision' not in config['optimization']:
                    config['optimization']['mixed_precision'] = {}
                config['optimization']['mixed_precision']['enabled'] = hardware_config['mixed_precision']

            # 调整梯度累积
            if 'gradient_accumulation' in hardware_config:
                if 'optimization' not in config:
                    config['optimization'] = {}
                config['optimization']['gradient_accumulation'] = hardware_config['gradient_accumulation']

        return config

    def _detect_hardware(self) -> str:
        """检测硬件配置"""
        if torch.cuda.is_available():
            # 获取GPU内存
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

            if gpu_memory_gb >= 8:
                return 'gpu_high_memory'
            else:
                return 'gpu_low_memory'
        else:
            return 'cpu_only'

    def list_available_configs(self) -> List[str]:
        """列出可用配置"""
        return list(self.configs.keys())

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """验证配置有效性"""
        required_sections = ['data', 'model', 'training', 'optimization']

        for section in required_sections:
            if section not in config:
                logger.error(f"配置缺少必要部分: {section}")
                return False

        # 验证数据配置
        data_config = config['data']
        if data_config.get('sample_size', 0) <= 0:
            logger.error("样本数量必须大于0")
            return False

        # 验证训练配置
        training_config = config['training']
        if training_config.get('num_epochs', 0) <= 0:
            logger.error("训练轮数必须大于0")
            return False

        if training_config.get('batch_size', 0) <= 0:
            logger.error("批次大小必须大于0")
            return False

        if training_config.get('learning_rate', 0) <= 0:
            logger.error("学习率必须大于0")
            return False

        logger.info("配置验证通过")
        return True

    def merge_configs(self, base_config: str, override_config: Dict[str, Any]) -> Dict[str, Any]:
        """合并配置"""
        base = self.get_config(base_config)

        def deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
            """深度合并字典"""
            result = dict1.copy()

            for key, value in dict2.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value

            return result

        merged = deep_merge(base, override_config)

        if self.validate_config(merged):
            return merged
        else:
            logger.error("合并后的配置无效，返回基础配置")
            return base

    def save_config(self, config: Dict[str, Any], name: str):
        """保存配置"""
        self.configs[name] = config

        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.configs, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"配置 '{name}' 保存成功")
        except Exception as e:
            logger.error(f"配置保存失败: {e}")

    def get_quality_standards(self) -> Dict[str, Any]:
        """获取数据质量标准"""
        return self.configs.get('quality_standards', {})

    def get_evaluation_standards(self) -> Dict[str, Any]:
        """获取评估标准"""
        return self.configs.get('evaluation_standards', {})

    def create_training_config_object(self, config_name: str = 'standard'):
        """创建TrainingConfig对象"""
        from core.trainer import TrainingConfig

        config = self.get_config(config_name)
        training_config = config['training']

        return TrainingConfig(
            num_epochs=training_config['num_epochs'],
            batch_size=training_config['batch_size'],
            learning_rate=training_config['learning_rate'],
            output_dir=f"outputs/enhanced_cultural_{config_name}",
            use_amp=config.get('optimization', {}).get('mixed_precision', {}).get('enabled', True),
            patience=training_config.get('patience', 15),
            min_delta=training_config.get('min_delta', 1e-4)
        )

    def print_config_summary(self, config_name: str = 'standard'):
        """打印配置摘要"""
        config = self.get_config(config_name)

        logger.info(f"配置摘要 - {config_name}:")
        logger.info(f"  数据: {config['data']['sample_size']} 样本")
        logger.info(f"  模型: {config['model']['hidden_dim']} 隐藏维度")
        logger.info(f"  训练: {config['training']['num_epochs']} 轮次, 批次 {config['training']['batch_size']}")
        logger.info(f"  优化器: {config['optimization']['optimizer']['type']}")
        logger.info(f"  调度器: {config['optimization']['scheduler']['type']}")


def main():
    """测试配置加载器"""
    loader = ConfigLoader()

    # 列出可用配置
    configs = loader.list_available_configs()
    logger.info(f"可用配置: {configs}")

    # 测试各个配置
    for config_name in ['quick_test', 'standard', 'high_performance']:
        if config_name in configs:
            logger.info(f"\n测试配置: {config_name}")
            loader.print_config_summary(config_name)


if __name__ == "__main__":
    main()
