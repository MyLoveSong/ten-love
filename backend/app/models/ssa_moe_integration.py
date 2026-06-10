#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SSA-MoE集成模块
将麻雀搜索算法集成到混合专家系统中，实现动态参数优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
import yaml
from pathlib import Path
import sys

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 导入SSA优化器
from app.models.ssa_optimizer.ssa import SSA, SSAConfig, SSACallback
from app.models.ssa_optimizer.objective import (
    GluFormerObjective, MultiTaskObjective, CulturalObjective, ImageNutritionObjective
)
from app.models.ssa_optimizer.utils import SSALogger, ConvergencePlotter, ParameterValidator

# 导入MoE系统
from app.models.mixture_of_experts import (
    MixtureOfExperts, EnhancedMixtureOfExperts, ExpertConfig,
    MultiTaskLearningFramework
)

logger = logging.getLogger(__name__)

@dataclass
class SSAMoEConfig:
    """SSA-MoE集成配置"""
    # 实验配置
    experiment: Optional[Dict[str, Any]] = None

    # SSA优化配置
    ssa_config: Dict[str, Any] = None

    # MoE系统配置
    moe_config: Optional[Dict[str, Any]] = None

    # 优化目标配置
    optimization_targets: List[str] = None  # ['gluformer', 'multitask', 'cultural', 'image_nutrition']

    # 目标详细配置
    target_configs: Optional[Dict[str, Any]] = None

    # 集成策略
    integration_strategy: str = 'sequential'  # 'sequential', 'parallel', 'adaptive'

    # 优化频率
    optimization_frequency: int = 100  # 每100个batch优化一次

    # 早停配置
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 1e-6

    # 日志配置
    log_dir: str = "logs/ssa_moe"
    output_dir: str = "outputs/ssa_moe"

    # 输出配置
    output: Optional[Dict[str, Any]] = None

    # 验证配置
    validation: Optional[Dict[str, Any]] = None

    # 性能配置
    performance: Optional[Dict[str, Any]] = None

    # 日志配置
    logging: Optional[Dict[str, Any]] = None

    # 可视化配置
    visualization: Optional[Dict[str, Any]] = None

    # 实验管理配置
    experiment_management: Optional[Dict[str, Any]] = None

    # 高级配置
    advanced: Optional[Dict[str, Any]] = None

    # 其他配置（可选）
    expert_configs: Optional[Dict[str, Any]] = None
    gating_config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """初始化后处理"""
        if self.ssa_config is None:
            self.ssa_config = {}
        if self.optimization_targets is None:
            self.optimization_targets = ['gluformer', 'multitask', 'cultural', 'image_nutrition']
        if self.moe_config is None:
            self.moe_config = {'input_dim': 10}

class SSAMoEObjective:
    """SSA-MoE集成目标函数"""

    def __init__(self, moe_model: nn.Module, config: SSAMoEConfig, device: str = 'cpu', target: str = None):
        self.moe_model = moe_model
        self.config = config
        self.device = device
        self.target = target
        self.optimization_history = []

        # 初始化目标函数
        self.objectives = {}
        self._initialize_objectives()

        logger.info(f"SSA-MoE目标函数初始化完成，目标: {target}")

    def _initialize_objectives(self):
        """初始化各种目标函数"""
        if self.target == 'gluformer':
            self.objectives[self.target] = GluFormerObjective(device=self.device)
        elif self.target == 'multitask':
            self.objectives[self.target] = MultiTaskObjective()
        elif self.target == 'cultural':
            self.objectives[self.target] = CulturalObjective()
        elif self.target == 'image_nutrition':
            self.objectives[self.target] = ImageNutritionObjective(device=self.device)
        else:
            logger.warning(f"未知的优化目标: {self.target}")

    def evaluate(self, params: np.ndarray) -> float:
        """评估目标函数（SSA优化器接口）"""
        return self.evaluate_moe_performance(params, self.target)

    def evaluate_moe_performance(self, params: np.ndarray, target: str) -> float:
        """评估MoE系统性能"""
        try:
            # 根据目标类型解码参数并更新MoE模型
            if target == 'gluformer':
                # 参数: [lstm_layers, lstm_hidden, dropout, lr, batch_size, gru_hidden, alpha]
                decoded_params = self._decode_gluformer_params(params)
                # 更新MoE模型中的GluFormer相关参数
                self._update_gluformer_params(decoded_params)

            elif target == 'multitask':
                # 参数: [sigma1, sigma2, sigma3]
                decoded_params = self._decode_multitask_params(params)
                # 更新多任务损失权重
                self._update_multitask_weights(decoded_params)

            elif target == 'cultural':
                # 参数: [lambda_health, lambda_culture, lambda_repeat]
                decoded_params = self._decode_cultural_params(params)
                # 更新文化适配参数
                self._update_cultural_params(decoded_params)

            elif target == 'image_nutrition':
                # 参数: [tau, projection_dim]
                decoded_params = self._decode_image_nutrition_params(params)
                # 更新图像-营养对齐参数
                self._update_image_nutrition_params(decoded_params)

            # 使用对应的目标函数评估
            if target in self.objectives:
                # 直接使用原始参数进行评估，目标函数内部会处理参数解码
                result = self.objectives[target].evaluate(params)
                # 如果返回的是OptimizationResult对象，提取fitness
                if hasattr(result, 'fitness'):
                    return result.fitness
                else:
                    return float(result)
            else:
                logger.warning(f"目标函数未找到: {target}")
                return float('inf')

        except Exception as e:
            logger.error(f"MoE性能评估失败: {e}")
            return float('inf')

    def _decode_gluformer_params(self, params: np.ndarray) -> Dict[str, Any]:
        """解码GluFormer参数"""
        return {
            'lstm_layers': int(np.round(params[0])),
            'lstm_hidden': int(np.round(params[1])),
            'dropout': float(params[2]),
            'learning_rate': float(params[3]),
            'batch_size': int(np.round(params[4])),
            'gru_hidden': int(np.round(params[5])),
            'attention_weight': float(params[6])
        }

    def _decode_multitask_params(self, params: np.ndarray) -> Dict[str, float]:
        """解码多任务参数"""
        return {
            'sigma1': max(1e-6, float(params[0])),
            'sigma2': max(1e-6, float(params[1])),
            'sigma3': max(1e-6, float(params[2]))
        }

    def _decode_cultural_params(self, params: np.ndarray) -> Dict[str, float]:
        """解码文化适配参数"""
        return {
            'lambda_health': np.clip(float(params[0]), 0, 1),
            'lambda_culture': np.clip(float(params[1]), 0, 1),
            'lambda_repeat': np.clip(float(params[2]), 0, 1)
        }

    def _decode_image_nutrition_params(self, params: np.ndarray) -> Dict[str, Any]:
        """解码图像-营养对齐参数"""
        return {
            'tau': max(0.01, min(float(params[0]), 1.0)),
            'projection_dim': int(max(64, min(np.round(params[1]), 1024)))
        }

    def _update_gluformer_params(self, params: Dict[str, Any]):
        """更新GluFormer参数"""
        # 这里需要根据实际的MoE模型结构来更新参数
        # 示例：更新门控网络的参数
        if hasattr(self.moe_model, 'gating_network'):
            # 更新门控网络的dropout
            if hasattr(self.moe_model.gating_network, 'dropout'):
                self.moe_model.gating_network.dropout.p = params['dropout']

        # 更新专家网络的参数
        if hasattr(self.moe_model, 'experts'):
            for expert in self.moe_model.experts:
                if hasattr(expert, 'dropout') and hasattr(expert.dropout, 'p'):
                    expert.dropout.p = params['dropout']

    def _update_multitask_weights(self, params: Dict[str, float]):
        """更新多任务损失权重"""
        # 更新多任务学习框架的权重
        if hasattr(self.moe_model, 'multi_task_framework'):
            if hasattr(self.moe_model.multi_task_framework, 'task_weights'):
                # 假设有3个任务
                self.moe_model.multi_task_framework.task_weights = torch.tensor([
                    params['sigma1'], params['sigma2'], params['sigma3']
                ]).to(self.device)

    def _update_cultural_params(self, params: Dict[str, float]):
        """更新文化适配参数"""
        # 更新文化适配相关的参数
        if hasattr(self.moe_model, 'cultural_adapter'):
            for key, value in params.items():
                if hasattr(self.moe_model.cultural_adapter, key):
                    setattr(self.moe_model.cultural_adapter, key, value)

    def _update_image_nutrition_params(self, params: Dict[str, Any]):
        """更新图像-营养对齐参数"""
        # 更新图像-营养对齐模型的参数
        if hasattr(self.moe_model, 'image_nutrition_aligner'):
            if hasattr(self.moe_model.image_nutrition_aligner, 'temperature'):
                self.moe_model.image_nutrition_aligner.temperature = params['tau']
            if hasattr(self.moe_model.image_nutrition_aligner, 'projection_dim'):
                self.moe_model.image_nutrition_aligner.projection_dim = params['projection_dim']

class SSAMoEIntegrator:
    """SSA-MoE集成器"""

    def __init__(self, moe_model: nn.Module, config: SSAMoEConfig):
        self.moe_model = moe_model
        self.config = config
        self.device = next(moe_model.parameters()).device

        # 初始化SSA优化器
        self.ssa_optimizers = {}
        self.objective_functions = {}
        self.optimization_history = {}

        # 初始化日志和绘图工具
        self.logger = SSALogger(log_dir=config.log_dir)
        self.plotter = ConvergencePlotter(output_dir=config.output_dir)
        self.validator = ParameterValidator()

        # 初始化目标函数
        self._initialize_optimization_targets()

        logger.info(f"SSA-MoE集成器初始化完成，设备: {self.device}")

    def _initialize_optimization_targets(self):
        """初始化优化目标"""
        for target in self.config.optimization_targets:
            # 为每个目标创建独立的目标函数实例
            objective = SSAMoEObjective(self.moe_model, self.config, str(self.device), target)
            self.objective_functions[target] = objective

            # 创建SSA优化器
            ssa_config = self._create_ssa_config(target)
            # 创建包装函数，固定target参数
            def objective_wrapper(params):
                return objective.evaluate_moe_performance(params, target)
            ssa_optimizer = SSA(ssa_config, objective_wrapper)
            self.ssa_optimizers[target] = ssa_optimizer

            # 初始化优化历史
            self.optimization_history[target] = []

            logger.info(f"优化目标 '{target}' 初始化完成")

    def _create_ssa_config(self, target: str) -> SSAConfig:
        """为特定目标创建SSA配置"""
        # 从全局ssa_config获取基础配置
        base_ssa_config = self.config.ssa_config.copy()

        # 从target_configs获取特定目标的配置
        target_specific_config = self.config.target_configs.get(target, {})

        # 合并配置，目标特定配置覆盖全局配置
        merged_config = {**base_ssa_config, **target_specific_config}

        # 过滤掉SSAConfig不接受的参数
        ssa_config_params = {
            'dim', 'pop_size', 'max_iter', 'lb', 'ub', 'discovery_rate',
            'safety_threshold', 'levy_flight', 'adaptive_chaos', 'early_stopping',
            'patience', 'min_delta'
        }

        filtered_config = {k: v for k, v in merged_config.items() if k in ssa_config_params}

        # 确保lb和ub是numpy数组
        if 'lb' in filtered_config:
            filtered_config['lb'] = np.array(filtered_config['lb'])
        if 'ub' in filtered_config:
            filtered_config['ub'] = np.array(filtered_config['ub'])

        return SSAConfig(**filtered_config)

    def optimize_target(self, target: str, max_iterations: int = None) -> Dict[str, Any]:
        """优化特定目标"""
        if target not in self.ssa_optimizers:
            raise ValueError(f"未知的优化目标: {target}")

        logger.info(f"开始优化目标: {target}")

        # 获取SSA优化器
        ssa_optimizer = self.ssa_optimizers[target]

        # 设置最大迭代次数
        if max_iterations is not None:
            ssa_optimizer.config.max_iter = max_iterations

        # 执行优化
        best_position, best_fitness, optimization_info = ssa_optimizer.optimize()

        # 记录优化历史
        self.optimization_history[target].append({
            'best_position': best_position,
            'best_fitness': best_fitness,
            'optimization_info': optimization_info
        })

        # 生成可视化
        plot_path = self.plotter.plot_convergence_curve(
            optimization_info['convergence_curve'],
            title=f"SSA-MoE {target} 优化收敛曲线"
        )

        result = {
            'target': target,
            'best_position': best_position,
            'best_fitness': best_fitness,
            'optimization_info': optimization_info,
            'convergence_plot': plot_path
        }

        logger.info(f"目标 '{target}' 优化完成，最佳适应度: {best_fitness:.6f}")
        return result

    def optimize_all_targets(self, max_iterations: int = None) -> Dict[str, Any]:
        """优化所有目标"""
        results = {}

        if self.config.integration_strategy == 'sequential':
            # 顺序优化
            for target in self.config.optimization_targets:
                results[target] = self.optimize_target(target, max_iterations)

        elif self.config.integration_strategy == 'parallel':
            # 并行优化（这里简化为顺序执行，实际可以实现真正的并行）
            for target in self.config.optimization_targets:
                results[target] = self.optimize_target(target, max_iterations)

        elif self.config.integration_strategy == 'adaptive':
            # 自适应优化：根据历史性能动态调整优化顺序
            targets_by_performance = self._rank_targets_by_performance()
            for target in targets_by_performance:
                results[target] = self.optimize_target(target, max_iterations)

        return results

    def _rank_targets_by_performance(self) -> List[str]:
        """根据历史性能对目标进行排序"""
        target_scores = {}

        for target in self.config.optimization_targets:
            if target in self.optimization_history and self.optimization_history[target]:
                # 计算平均性能改进
                recent_improvements = []
                history = self.optimization_history[target]

                for i in range(1, len(history)):
                    improvement = history[i-1]['best_fitness'] - history[i]['best_fitness']
                    recent_improvements.append(improvement)

                if recent_improvements:
                    target_scores[target] = np.mean(recent_improvements)
                else:
                    target_scores[target] = 0.0
            else:
                target_scores[target] = 0.0

        # 按性能改进排序（降序）
        sorted_targets = sorted(target_scores.items(), key=lambda x: x[1], reverse=True)
        return [target for target, _ in sorted_targets]

    def get_optimization_summary(self) -> Dict[str, Any]:
        """获取优化总结"""
        summary = {
            'total_targets': len(self.config.optimization_targets),
            'optimization_history': {},
            'best_performances': {},
            'convergence_analysis': {}
        }

        for target in self.config.optimization_targets:
            if target in self.optimization_history and self.optimization_history[target]:
                history = self.optimization_history[target]

                # 最佳性能
                best_fitness = min([h['best_fitness'] for h in history])
                summary['best_performances'][target] = best_fitness

                # 收敛分析
                if len(history) > 1:
                    initial_fitness = history[0]['best_fitness']
                    final_fitness = history[-1]['best_fitness']
                    improvement = initial_fitness - final_fitness
                    improvement_rate = improvement / initial_fitness if initial_fitness != 0 else 0

                    summary['convergence_analysis'][target] = {
                        'initial_fitness': initial_fitness,
                        'final_fitness': final_fitness,
                        'improvement': improvement,
                        'improvement_rate': improvement_rate
                    }

                summary['optimization_history'][target] = len(history)
            else:
                summary['optimization_history'][target] = 0
                summary['best_performances'][target] = float('inf')

        return summary

    def save_optimization_results(self, output_path: str):
        """保存优化结果"""
        import json

        results = {
            'config': self.config.__dict__,
            'optimization_summary': self.get_optimization_summary(),
            'optimization_history': self.optimization_history
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"优化结果已保存到: {output_path}")

def create_ssa_moe_integrator(moe_model: nn.Module, config_path: str) -> SSAMoEIntegrator:
    """创建SSA-MoE集成器"""
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    config = SSAMoEConfig(**config_dict)

    # 创建集成器
    integrator = SSAMoEIntegrator(moe_model, config)

    return integrator

# 示例使用
if __name__ == "__main__":
    # 创建示例MoE模型
    class ExampleMoEModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.gating_network = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 3),
                nn.Softmax(dim=-1)
            )
            self.experts = nn.ModuleList([
                nn.Linear(128, 64) for _ in range(3)
            ])

        def forward(self, x):
            weights = self.gating_network(x)
            outputs = [expert(x) for expert in self.experts]
            outputs = torch.stack(outputs, dim=1)
            return torch.sum(outputs * weights.unsqueeze(-1), dim=1)

    # 创建示例配置
    config_dict = {
        'ssa_config': {
            'pop_size': 20,
            'max_iter': 30,
            'discovery_rate': 0.2,
            'safety_threshold': 0.8,
            'levy_flight': True,
            'adaptive_chaos': True,
            'early_stopping': True,
            'patience': 10,
            'min_delta': 1e-6
        },
        'moe_config': {
            'input_dim': 128,
            'output_dim': 64,
            'num_experts': 3
        },
        'optimization_targets': ['gluformer', 'multitask'],
        'integration_strategy': 'sequential',
        'optimization_frequency': 100
    }

    # 创建集成器
    moe_model = ExampleMoEModel()
    config = SSAMoEConfig(**config_dict)
    integrator = SSAMoEIntegrator(moe_model, config)

    # 执行优化
    results = integrator.optimize_all_targets(max_iterations=10)

    # 打印结果
    print("SSA-MoE优化结果:")
    for target, result in results.items():
        print(f"{target}: 最佳适应度 = {result['best_fitness']:.6f}")

    # 获取优化总结
    summary = integrator.get_optimization_summary()
    print(f"\n优化总结: {summary}")
