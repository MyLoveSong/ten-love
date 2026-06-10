"""
SSA配置文件加载器
用于加载和管理SSA优化器的配置文件
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class SSAConfigData:
    """SSA配置数据类"""
    # 基础配置
    dim: int = 7
    pop_size: int = 20
    max_iter: int = 30
    lb: List[float] = field(default_factory=lambda: [0.0] * 7)
    ub: List[float] = field(default_factory=lambda: [1.0] * 7)

    # SSA算法参数
    discovery_rate: float = 0.2
    safety_threshold: float = 0.8
    levy_flight: bool = True
    adaptive_chaos: bool = True
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 1e-6

    # 目标函数配置
    objective_function: Dict[str, Any] = field(default_factory=dict)

    # 其他配置
    param_names: List[str] = field(default_factory=list)
    description: str = ""

class SSAConfigLoader:
    """SSA配置文件加载器"""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_cache = {}
        logger.info(f"SSA配置加载器初始化完成，配置目录: {self.config_dir}")

    def load_config(self, config_name: str) -> Dict[str, Any]:
        """
        加载指定名称的配置文件

        Args:
            config_name: 配置文件名（不含扩展名）

        Returns:
            配置字典
        """
        config_file = self.config_dir / f"{config_name}.yaml"

        if not config_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_file}")

        # 检查缓存
        if config_name in self.config_cache:
            logger.debug(f"从缓存加载配置: {config_name}")
            return self.config_cache[config_name]

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # 验证配置
            self._validate_config(config, config_name)

            # 缓存配置
            self.config_cache[config_name] = config

            logger.info(f"成功加载配置文件: {config_name}")
            return config

        except Exception as e:
            logger.error(f"加载配置文件失败: {config_name}, 错误: {e}")
            raise

    def load_ssa_config(self, config_name: str) -> SSAConfigData:
        """
        加载SSA配置并转换为SSAConfigData对象

        Args:
            config_name: 配置文件名

        Returns:
            SSAConfigData对象
        """
        config = self.load_config(config_name)

        # 提取SSA相关配置
        ssa_config = SSAConfigData(
            dim=config.get('dim', 7),
            pop_size=config.get('pop_size', 20),
            max_iter=config.get('max_iter', 30),
            lb=config.get('lb', [0.0] * config.get('dim', 7)),
            ub=config.get('ub', [1.0] * config.get('dim', 7)),
            discovery_rate=config.get('discovery_rate', 0.2),
            safety_threshold=config.get('safety_threshold', 0.8),
            levy_flight=config.get('levy_flight', True),
            adaptive_chaos=config.get('adaptive_chaos', True),
            early_stopping=config.get('early_stopping', True),
            patience=config.get('patience', 10),
            min_delta=config.get('min_delta', 1e-6),
            objective_function=config.get('objective_function', {}),
            param_names=config.get('param_names', []),
            description=config.get('description', "")
        )

        return ssa_config

    def load_moe_integration_config(self, config_name: str = "ssa_moe_integration") -> Dict[str, Any]:
        """
        加载SSA-MoE集成配置文件

        Args:
            config_name: 配置文件名

        Returns:
            集成配置字典
        """
        return self.load_config(config_name)

    def get_target_configs(self, config_name: str = "ssa_moe_integration") -> Dict[str, SSAConfigData]:
        """
        获取所有目标的配置

        Args:
            config_name: 配置文件名

        Returns:
            目标配置字典
        """
        config = self.load_config(config_name)
        target_configs = {}

        if 'target_configs' in config:
            for target_name, target_config in config['target_configs'].items():
                target_configs[target_name] = SSAConfigData(
                    dim=target_config.get('dim', 7),
                    pop_size=config.get('ssa_config', {}).get('pop_size', 20),
                    max_iter=config.get('ssa_config', {}).get('max_iter', 30),
                    lb=target_config.get('lb', [0.0] * target_config.get('dim', 7)),
                    ub=target_config.get('ub', [1.0] * target_config.get('dim', 7)),
                    discovery_rate=config.get('ssa_config', {}).get('discovery_rate', 0.2),
                    safety_threshold=config.get('ssa_config', {}).get('safety_threshold', 0.8),
                    levy_flight=config.get('ssa_config', {}).get('levy_flight', True),
                    adaptive_chaos=config.get('ssa_config', {}).get('adaptive_chaos', True),
                    early_stopping=config.get('ssa_config', {}).get('early_stopping', True),
                    patience=config.get('ssa_config', {}).get('patience', 10),
                    min_delta=config.get('ssa_config', {}).get('min_delta', 1e-6),
                    objective_function=target_config.get('objective_function', {}),
                    param_names=target_config.get('param_names', []),
                    description=target_config.get('description', "")
                )

        return target_configs

    def list_configs(self) -> List[str]:
        """
        列出所有可用的配置文件

        Returns:
            配置文件名称列表
        """
        config_files = []
        for file_path in self.config_dir.glob("*.yaml"):
            config_files.append(file_path.stem)
        return sorted(config_files)

    def _validate_config(self, config: Dict[str, Any], config_name: str):
        """
        验证配置文件的完整性和有效性

        Args:
            config: 配置字典
            config_name: 配置文件名
        """
        # 对于SSA-MoE集成配置，使用不同的验证逻辑
        if config_name == "ssa_moe_integration":
            self._validate_moe_integration_config(config, config_name)
            return

        # 基础SSA配置验证
        required_fields = ['dim', 'pop_size', 'max_iter', 'lb', 'ub']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"配置文件 {config_name} 缺少必需字段: {field}")

        # 类型验证
        if not isinstance(config['dim'], int) or config['dim'] <= 0:
            raise ValueError(f"配置文件 {config_name} 中 'dim' 必须是正整数")

        if not isinstance(config['pop_size'], int) or config['pop_size'] <= 0:
            raise ValueError(f"配置文件 {config_name} 中 'pop_size' 必须是正整数")

        if not isinstance(config['max_iter'], int) or config['max_iter'] <= 0:
            raise ValueError(f"配置文件 {config_name} 中 'max_iter' 必须是正整数")

        # 边界验证
        if len(config['lb']) != config['dim'] or len(config['ub']) != config['dim']:
            raise ValueError(f"配置文件 {config_name} 中边界数组长度与维度不匹配")

        if not all(ub >= lb for ub, lb in zip(config['ub'], config['lb'])):
            raise ValueError(f"配置文件 {config_name} 中上界必须大于等于下界")

        logger.debug(f"配置文件 {config_name} 验证通过")

    def _validate_moe_integration_config(self, config: Dict[str, Any], config_name: str):
        """
        验证SSA-MoE集成配置

        Args:
            config: 配置字典
            config_name: 配置文件名
        """
        # 检查必需的顶级字段
        required_fields = ['optimization_targets', 'ssa_config', 'target_configs']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"配置文件 {config_name} 缺少必需字段: {field}")

        # 验证ssa_config
        ssa_config = config['ssa_config']
        if not isinstance(ssa_config, dict):
            raise ValueError(f"配置文件 {config_name} 中 'ssa_config' 必须是字典")

        # 验证target_configs
        target_configs = config['target_configs']
        if not isinstance(target_configs, dict):
            raise ValueError(f"配置文件 {config_name} 中 'target_configs' 必须是字典")

        # 验证每个目标配置
        for target_name, target_config in target_configs.items():
            if not isinstance(target_config, dict):
                raise ValueError(f"目标配置 {target_name} 必须是字典")

            required_target_fields = ['dim', 'lb', 'ub']
            for field in required_target_fields:
                if field not in target_config:
                    raise ValueError(f"目标配置 {target_name} 缺少必需字段: {field}")

            # 验证维度一致性
            dim = target_config['dim']
            if len(target_config['lb']) != dim or len(target_config['ub']) != dim:
                raise ValueError(f"目标配置 {target_name} 中边界数组长度与维度不匹配")

            if not all(ub >= lb for ub, lb in zip(target_config['ub'], target_config['lb'])):
                raise ValueError(f"目标配置 {target_name} 中上界必须大于等于下界")

        logger.debug(f"SSA-MoE集成配置文件 {config_name} 验证通过")

    def clear_cache(self):
        """清空配置缓存"""
        self.config_cache.clear()
        logger.info("配置缓存已清空")

    def reload_config(self, config_name: str) -> Dict[str, Any]:
        """
        重新加载配置文件（忽略缓存）

        Args:
            config_name: 配置文件名

        Returns:
            配置字典
        """
        if config_name in self.config_cache:
            del self.config_cache[config_name]
        return self.load_config(config_name)

# 全局配置加载器实例
config_loader = SSAConfigLoader()

def get_config_loader() -> SSAConfigLoader:
    """获取全局配置加载器实例"""
    return config_loader

def load_ssa_config(config_name: str) -> SSAConfigData:
    """便捷函数：加载SSA配置"""
    return config_loader.load_ssa_config(config_name)

def load_moe_integration_config(config_name: str = "ssa_moe_integration") -> Dict[str, Any]:
    """便捷函数：加载SSA-MoE集成配置"""
    return config_loader.load_moe_integration_config(config_name)
