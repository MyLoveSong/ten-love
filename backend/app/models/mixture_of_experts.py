

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合专家系统 (Mixture of Experts, MoE)
实现多任务混合专家微调与强化学习优化
基于项目申请表中的创新点二设计
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)

class ExperienceReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """添加经验"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        """采样经验"""
        import random
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

@dataclass
class ExpertConfig:
    """专家配置"""
    expert_type: str  # 专家类型
    input_dim: int
    hidden_dim: int
    output_dim: int
    num_layers: int = 2
    dropout: float = 0.1

class HealthIndicatorExpert(nn.Module):
    """健康指标专家 - 处理生化数据分析"""

    def __init__(self, config: ExpertConfig):
        super().__init__()
        self.config = config

        # 生化指标处理网络
        self.biochemical_net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.BatchNorm1d(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.BatchNorm1d(config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.output_dim)
        )

        # 医学知识增强层
        self.medical_knowledge_layer = nn.Sequential(
            nn.Linear(config.output_dim, config.output_dim),
            nn.Sigmoid()  # 用于知识门控
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 生化指标分析
        biochemical_features = self.biochemical_net(x)

        # 医学知识增强
        knowledge_gate = self.medical_knowledge_layer(biochemical_features)
        enhanced_features = biochemical_features * knowledge_gate

        return enhanced_features

class MedicalDiagnosisExpert(nn.Module):
    """医学诊断专家 - 处理医学诊断任务"""

    def __init__(self, config: ExpertConfig):
        super().__init__()
        self.config = config

        # 诊断网络
        self.diagnosis_net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.output_dim)
        )

        # 临床规则约束层
        self.clinical_rules = nn.Sequential(
            nn.Linear(config.output_dim, config.output_dim),
            nn.Tanh()  # 用于规则约束
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 诊断分析
        diagnosis_features = self.diagnosis_net(x)

        # 临床规则约束
        rule_constraints = self.clinical_rules(diagnosis_features)
        constrained_features = diagnosis_features * rule_constraints

        return constrained_features

class LifestyleAssessmentExpert(nn.Module):
    """生活习惯评估专家 - 处理生活习惯评估"""

    def __init__(self, config: ExpertConfig):
        super().__init__()
        self.config = config

        # 生活习惯分析网络
        self.lifestyle_net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.BatchNorm1d(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.BatchNorm1d(config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.output_dim)
        )

        # 行为模式识别层
        self.behavior_pattern = nn.Sequential(
            nn.Linear(config.output_dim, config.output_dim),
            nn.Softmax(dim=-1)  # 用于行为模式权重
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 生活习惯分析
        lifestyle_features = self.lifestyle_net(x)

        # 行为模式识别
        pattern_weights = self.behavior_pattern(lifestyle_features)
        weighted_features = lifestyle_features * pattern_weights

        return weighted_features

class GatingNetwork(nn.Module):
    """门控网络 - 动态选择专家组合"""

    def __init__(self, input_dim: int, num_experts: int,
                 hidden_dim: int = 128, temperature: float = 1.0):
        super().__init__()

        self.num_experts = num_experts
        self.temperature = temperature

        # 门控网络
        self.gating_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_experts)
        )

        # 稀疏性控制
        self.sparsity_regularizer = nn.Linear(num_experts, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 计算门控权重
        gate_logits = self.gating_net(x)

        # 温度缩放
        gate_logits = gate_logits / self.temperature

        # 稀疏性正则化
        sparse_weights = self.sparsity_regularizer(gate_logits)

        # Softmax归一化
        gate_weights = F.softmax(sparse_weights, dim=-1)

        return gate_weights

class MixtureOfExperts(nn.Module):
    """混合专家系统"""

    def __init__(self, input_dim: int, output_dim: int,
                 expert_configs: List[ExpertConfig],
                 num_experts_to_activate: int = 2):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = len(expert_configs)
        self.num_experts_to_activate = num_experts_to_activate

        # 创建专家网络
        self.experts = nn.ModuleList()
        for config in expert_configs:
            if config.expert_type == 'health_indicator':
                expert = HealthIndicatorExpert(config)
            elif config.expert_type == 'medical_diagnosis':
                expert = MedicalDiagnosisExpert(config)
            elif config.expert_type == 'lifestyle_assessment':
                expert = LifestyleAssessmentExpert(config)
            else:
                raise ValueError(f"Unknown expert type: {config.expert_type}")

            self.experts.append(expert)

        # 门控网络
        self.gating_network = GatingNetwork(
            input_dim=input_dim,
            num_experts=self.num_experts
        )

        # 输出融合层
        self.output_fusion = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim // 2, output_dim)
        )

        # 专家负载均衡
        self.expert_usage_count = defaultdict(int)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        Args:
            x: 输入特征 [batch_size, input_dim]
        Returns:
            output: 输出特征 [batch_size, output_dim]
            gate_weights: 门控权重 [batch_size, num_experts]
        """
        batch_size = x.size(0)

        # 计算门控权重
        gate_weights = self.gating_network(x)

        # Top-K专家选择
        top_k_weights, top_k_indices = torch.topk(gate_weights, self.num_experts_to_activate, dim=-1)
        top_k_weights = F.softmax(top_k_weights, dim=-1)

        # 专家输出
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_output = expert(x)
            expert_outputs.append(expert_output)

        # 加权融合
        output = torch.zeros_like(expert_outputs[0])
        for i, (weight, idx) in enumerate(zip(top_k_weights, top_k_indices)):
            for j, expert_idx in enumerate(idx):
                output[i] += weight[j] * expert_outputs[expert_idx][i]

        # 输出处理
        final_output = self.output_fusion(output)

        # 更新专家使用统计
        for batch_idx, expert_indices in enumerate(top_k_indices):
            for expert_idx in expert_indices:
                self.expert_usage_count[expert_idx.item()] += 1

        return final_output, gate_weights

    def get_expert_usage_stats(self) -> Dict[str, Any]:
        """获取专家使用统计"""
        total_usage = sum(self.expert_usage_count.values())
        if total_usage == 0:
            return {}

        usage_stats = {}
        for expert_idx, count in self.expert_usage_count.items():
            usage_stats[f'expert_{expert_idx}'] = {
                'usage_count': count,
                'usage_rate': count / total_usage
            }

        return usage_stats

    def reset_usage_stats(self):
        """重置使用统计"""
        self.expert_usage_count.clear()

class MultiTaskLearningFramework(nn.Module):
    """
    多任务学习框架
    基于项目申请表中的创新点四设计
    实现任务相关性图谱和动态任务权重调整
    """

    def __init__(self, input_dim: int, task_configs: List[Dict[str, Any]],
                 shared_dim: int = 128, task_specific_dim: int = 64):
        super().__init__()

        self.input_dim = input_dim
        self.task_configs = task_configs
        self.num_tasks = len(task_configs)
        self.shared_dim = shared_dim
        self.task_specific_dim = task_specific_dim

        # 共享特征提取器
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(shared_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.ReLU()
        )

        # 任务特定编码器
        self.task_encoders = nn.ModuleDict()
        for i, task_config in enumerate(task_configs):
            self.task_encoders[f'task_{i}'] = nn.Sequential(
                nn.Linear(shared_dim, task_specific_dim),
                nn.LayerNorm(task_specific_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )

        # 任务相关性图谱
        self.task_correlation_matrix = nn.Parameter(
            torch.randn(self.num_tasks, self.num_tasks)
        )

        # 动态任务权重网络
        self.task_weight_network = nn.Sequential(
            nn.Linear(shared_dim + self.num_tasks, shared_dim // 2),
            nn.ReLU(),
            nn.Linear(shared_dim // 2, self.num_tasks),
            nn.Softmax(dim=-1)
        )

        # 任务间注意力机制
        self.cross_task_attention = nn.MultiheadAttention(
            embed_dim=task_specific_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )

        # 任务输出头
        self.task_heads = nn.ModuleDict()
        for i, task_config in enumerate(task_configs):
            output_dim = task_config.get('output_dim', 1)
            self.task_heads[f'task_{i}'] = nn.Sequential(
                nn.Linear(task_specific_dim, task_specific_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(task_specific_dim // 2, output_dim)
            )

        # 任务重要性评估
        self.task_importance_estimator = nn.Sequential(
            nn.Linear(shared_dim, shared_dim // 2),
            nn.ReLU(),
            nn.Linear(shared_dim // 2, self.num_tasks),
            nn.Sigmoid()
        )

        # 任务冲突检测
        self.conflict_detector = nn.Sequential(
            nn.Linear(task_specific_dim * self.num_tasks, shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, self.num_tasks),
            nn.Sigmoid()
        )

        # 自适应学习率调整
        self.adaptive_lr_weights = nn.Parameter(torch.ones(self.num_tasks))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        多任务学习前向传播

        Args:
            x: 输入特征 [batch_size, input_dim]

        Returns:
            包含各任务输出和元信息的字典
        """
        batch_size = x.size(0)

        # 共享特征提取
        shared_features = self.shared_encoder(x)

        # 任务特定特征提取
        task_features = {}
        for i in range(self.num_tasks):
            task_key = f'task_{i}'
            task_features[task_key] = self.task_encoders[task_key](shared_features)

        # 任务相关性处理
        task_correlation = F.softmax(self.task_correlation_matrix, dim=-1)

        # 任务间注意力
        task_feature_list = [task_features[f'task_{i}'] for i in range(self.num_tasks)]
        task_feature_stack = torch.stack(task_feature_list, dim=1)  # [batch_size, num_tasks, task_specific_dim]

        attended_features, attention_weights = self.cross_task_attention(
            task_feature_stack, task_feature_stack, task_feature_stack
        )

        # 任务权重计算
        task_importance = self.task_importance_estimator(shared_features)
        task_weights = self.task_weight_network(
            torch.cat([shared_features, task_importance], dim=-1)
        )

        # 任务输出
        task_outputs = {}
        for i in range(self.num_tasks):
            task_key = f'task_{i}'
            # 使用注意力增强的特征
            enhanced_features = task_features[task_key] + attended_features[:, i, :]
            task_outputs[task_key] = self.task_heads[task_key](enhanced_features)

        # 任务冲突检测
        all_task_features = torch.cat(task_feature_list, dim=-1)
        conflict_scores = self.conflict_detector(all_task_features)

        # 加权任务融合
        weighted_outputs = {}
        for i in range(self.num_tasks):
            task_key = f'task_{i}'
            weight = task_weights[:, i:i+1]
            conflict_weight = 1.0 - conflict_scores[:, i:i+1]  # 冲突时降低权重
            final_weight = weight * conflict_weight
            weighted_outputs[task_key] = task_outputs[task_key] * final_weight

        return {
            'task_outputs': task_outputs,
            'weighted_outputs': weighted_outputs,
            'task_weights': task_weights,
            'task_importance': task_importance,
            'task_correlation': task_correlation,
            'conflict_scores': conflict_scores,
            'attention_weights': attention_weights,
            'shared_features': shared_features,
            'task_features': task_features
        }

    def compute_task_loss(self, outputs: Dict[str, torch.Tensor],
                         targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算多任务损失

        Args:
            outputs: 模型输出
            targets: 目标值

        Returns:
            各任务的损失
        """
        losses = {}

        for i in range(self.num_tasks):
            task_key = f'task_{i}'
            if task_key in targets:
                task_config = self.task_configs[i]
                task_type = task_config.get('type', 'regression')

                if task_type == 'regression':
                    loss = F.mse_loss(outputs['task_outputs'][task_key], targets[task_key])
                elif task_type == 'classification':
                    loss = F.cross_entropy(outputs['task_outputs'][task_key], targets[task_key])
                else:
                    loss = F.mse_loss(outputs['task_outputs'][task_key], targets[task_key])

                # 自适应学习率调整
                adaptive_weight = self.adaptive_lr_weights[i]
                losses[task_key] = loss * adaptive_weight

        return losses

    def update_task_correlation(self, task_performance: Dict[str, float]):
        """
        更新任务相关性矩阵

        Args:
            task_performance: 各任务性能指标
        """
        # 基于性能更新相关性
        for i in range(self.num_tasks):
            for j in range(self.num_tasks):
                if i != j:
                    # 如果两个任务性能都高，增加相关性
                    if (task_performance.get(f'task_{i}', 0.5) > 0.7 and
                        task_performance.get(f'task_{j}', 0.5) > 0.7):
                        self.task_correlation_matrix.data[i, j] += 0.01
                    # 如果性能差异大，减少相关性
                    elif abs(task_performance.get(f'task_{i}', 0.5) -
                            task_performance.get(f'task_{j}', 0.5)) > 0.3:
                        self.task_correlation_matrix.data[i, j] -= 0.01

        # 确保相关性矩阵的稳定性
        self.task_correlation_matrix.data = torch.clamp(
            self.task_correlation_matrix.data, min=-1.0, max=1.0
        )

class EnhancedMixtureOfExperts(nn.Module):
    """
    增强的混合专家系统
    集成多任务学习框架和动态专家选择
    """

    def __init__(self, input_dim: int, output_dim: int,
                 expert_configs: List[ExpertConfig],
                 task_configs: List[Dict[str, Any]],
                 num_experts_to_activate: int = 2):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = len(expert_configs)
        self.num_experts_to_activate = num_experts_to_activate

        # 多任务学习框架
        self.multi_task_framework = MultiTaskLearningFramework(
            input_dim=input_dim,
            task_configs=task_configs
        )

        # 创建专家网络
        self.experts = nn.ModuleList()
        for config in expert_configs:
            if config.expert_type == 'health_indicator':
                expert = HealthIndicatorExpert(config)
            elif config.expert_type == 'medical_diagnosis':
                expert = MedicalDiagnosisExpert(config)
            elif config.expert_type == 'lifestyle_assessment':
                expert = LifestyleAssessmentExpert(config)
            else:
                raise ValueError(f"Unknown expert type: {config.expert_type}")

            self.experts.append(expert)

        # 增强的门控网络
        self.gating_network = GatingNetwork(
            input_dim=input_dim,
            num_experts=self.num_experts
        )

        # 任务感知门控网络
        self.task_aware_gating = nn.Sequential(
            nn.Linear(input_dim + len(task_configs), input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, self.num_experts),
            nn.Softmax(dim=-1)
        )

        # 专家任务映射
        self.expert_task_mapping = nn.Parameter(
            torch.randn(self.num_experts, len(task_configs))
        )

        # 动态专家权重调整
        self.dynamic_expert_adjuster = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_experts),
            nn.Sigmoid()
        )

        # 输出融合层
        self.output_fusion = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim // 2, output_dim)
        )

        # 专家负载均衡
        self.expert_usage_count = defaultdict(int)
        self.task_expert_performance = defaultdict(lambda: defaultdict(float))

    def forward(self, x: torch.Tensor, task_context: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """
        增强的前向传播

        Args:
            x: 输入特征 [batch_size, input_dim]
            task_context: 任务上下文信息

        Returns:
            包含专家输出、任务输出等的字典
        """
        batch_size = x.size(0)

        # 多任务学习
        multi_task_output = self.multi_task_framework(x)

        # 任务感知门控
        if task_context is not None:
            task_weights = torch.tensor([
                task_context.get(f'task_{i}', 0.0) for i in range(len(self.multi_task_framework.task_configs))
            ], dtype=torch.float32).unsqueeze(0).expand(batch_size, -1)
            gating_input = torch.cat([x, task_weights], dim=-1)
        else:
            gating_input = x

        # 计算门控权重
        base_gate_weights = self.gating_network(x)
        task_aware_gate_weights = self.task_aware_gating(gating_input)

        # 融合门控权重
        combined_gate_weights = (base_gate_weights + task_aware_gate_weights) / 2

        # Top-K专家选择
        top_k_weights, top_k_indices = torch.topk(combined_gate_weights, self.num_experts_to_activate, dim=-1)
        top_k_weights = F.softmax(top_k_weights, dim=-1)

        # 动态专家权重调整
        dynamic_adjustment = self.dynamic_expert_adjuster(x)

        # 专家输出
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_output = expert(x)
            # 应用动态调整
            adjusted_output = expert_output * dynamic_adjustment[:, i:i+1]
            expert_outputs.append(adjusted_output)

        # 加权融合
        output = torch.zeros_like(expert_outputs[0])
        for i, (weight, idx) in enumerate(zip(top_k_weights, top_k_indices)):
            for j, expert_idx in enumerate(idx):
                output[i] += weight[j] * expert_outputs[expert_idx][i]

        # 输出处理
        final_output = self.output_fusion(output)

        # 更新专家使用统计
        for batch_idx, expert_indices in enumerate(top_k_indices):
            for expert_idx in expert_indices:
                self.expert_usage_count[expert_idx.item()] += 1

        return {
            'expert_output': final_output,
            'gate_weights': combined_gate_weights,
            'top_k_weights': top_k_weights,
            'top_k_indices': top_k_indices,
            'dynamic_adjustment': dynamic_adjustment,
            'multi_task_output': multi_task_output,
            'expert_outputs': expert_outputs
        }

    def get_expert_task_performance(self) -> Dict[str, Any]:
        """获取专家任务性能统计"""
        performance_stats = {}

        for expert_idx in range(self.num_experts):
            expert_performance = {}
            for task_idx in range(len(self.multi_task_framework.task_configs)):
                task_key = f'task_{task_idx}'
                expert_performance[task_key] = self.task_expert_performance[expert_idx][task_key]

            performance_stats[f'expert_{expert_idx}'] = expert_performance

        return performance_stats

    def update_expert_task_performance(self, expert_idx: int, task_idx: int, performance: float):
        """更新专家任务性能"""
        self.task_expert_performance[expert_idx][f'task_{task_idx}'] = performance

class EnhancedMERLSystem(nn.Module):
    """增强的混合专家强化学习系统
    集成时序感知、多任务学习和动态专家选择
    """

    def __init__(self, input_dim: int, action_dim: int,
                 expert_configs: List[ExpertConfig],
                 sequence_length: int = 24, temporal_dim: int = 128):
        super().__init__()

        self.input_dim = input_dim
        self.action_dim = action_dim
        self.sequence_length = sequence_length
        self.temporal_dim = temporal_dim

        # 定义多任务配置
        self.task_configs = [
            {
                'name': 'glucose_control',
                'type': 'regression',
                'output_dim': 1,
                'description': '血糖控制任务'
            },
            {
                'name': 'nutrition_balance',
                'type': 'regression',
                'output_dim': 1,
                'description': '营养均衡任务'
            },
            {
                'name': 'cultural_adaptation',
                'type': 'regression',
                'output_dim': 1,
                'description': '文化适配任务'
            }
        ]

        # 时序感知编码器
        self.temporal_encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=temporal_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=True
        )

        # 时序注意力机制
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=temporal_dim * 2,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # 增强的混合专家系统（集成多任务学习）
        self.moe_system = EnhancedMixtureOfExperts(
            input_dim=temporal_dim * 2,  # 使用时序编码后的维度
            output_dim=128,
            expert_configs=expert_configs,
            task_configs=self.task_configs
        )

        # 动态专家权重调整网络
        self.expert_weight_adjuster = nn.Sequential(
            nn.Linear(temporal_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, len(expert_configs)),
            nn.Sigmoid()
        )

        # 多任务策略网络
        self.policy_networks = nn.ModuleDict({
            'glucose_control': nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, action_dim)
            ),
            'nutrition_balance': nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, action_dim)
            ),
            'cultural_adaptation': nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, action_dim)
            )
        })

        # 任务重要性评估网络
        self.task_importance_network = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, len(self.policy_networks)),
            nn.Softmax(dim=-1)
        )

        # 增强的价值网络（多头）
        self.value_networks = nn.ModuleDict({
            'glucose_value': nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 1)
            ),
            'nutrition_value': nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 1)
            ),
            'cultural_value': nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 1)
            )
        })

        # 动态奖励权重
        self.reward_weights = nn.Parameter(torch.ones(4))  # 血糖控制、营养均衡、患者满意度、文化适配

        # 时序奖励预测网络
        self.temporal_reward_predictor = nn.Sequential(
            nn.Linear(temporal_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # 预测未来4个时间步的奖励
        )

        # 记忆缓冲区（用于经验回放）
        self.memory_buffer = ExperienceReplayBuffer(capacity=10000)

        # 不确定性估计
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()
        )

    def forward(self, x: torch.Tensor, sequence_data: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        增强的前向传播，支持时序感知和多任务处理

        Args:
            x: 当前状态 [batch_size, input_dim]
            sequence_data: 时序数据 [batch_size, seq_len, input_dim] (可选)

        Returns:
            包含策略、价值、专家输出等的字典
        """
        batch_size = x.size(0)

        # 时序特征提取
        if sequence_data is not None:
            # 使用时序数据
            temporal_features, (h_n, c_n) = self.temporal_encoder(sequence_data)

            # 时序注意力
            attended_features, temporal_attention_weights = self.temporal_attention(
                temporal_features, temporal_features, temporal_features
            )

            # 使用最后时间步的特征
            current_temporal_feature = attended_features[:, -1, :]
        else:
            # 如果没有时序数据，创建单步时序
            single_step = x.unsqueeze(1)  # [batch_size, 1, input_dim]
            temporal_features, (h_n, c_n) = self.temporal_encoder(single_step)
            current_temporal_feature = temporal_features.squeeze(1)
            temporal_attention_weights = None

        # 专家系统处理（使用时序特征，集成多任务学习）
        moe_output = self.moe_system(current_temporal_feature)
        expert_output = moe_output['expert_output']
        gate_weights = moe_output['gate_weights']
        multi_task_output = moe_output['multi_task_output']

        # 动态专家权重调整
        expert_weight_adjustment = self.expert_weight_adjuster(current_temporal_feature)
        adjusted_expert_output = expert_output * expert_weight_adjustment.unsqueeze(-1)

        # 多任务策略网络
        task_importance = self.task_importance_network(adjusted_expert_output)
        policy_outputs = {}

        for task_name, policy_net in self.policy_networks.items():
            task_policy_logits = policy_net(adjusted_expert_output)
            task_policy_probs = F.softmax(task_policy_logits, dim=-1)
            policy_outputs[f'{task_name}_policy'] = task_policy_probs

        # 任务重要性加权策略融合
        combined_policy = torch.zeros_like(policy_outputs['glucose_control_policy'])
        for i, (task_name, policy_probs) in enumerate(policy_outputs.items()):
            if 'policy' in task_name:
                task_idx = list(self.policy_networks.keys()).index(task_name.replace('_policy', ''))
                combined_policy += task_importance[:, task_idx:task_idx+1] * policy_probs

        # 多头价值估计
        value_outputs = {}
        for value_name, value_net in self.value_networks.items():
            value_outputs[value_name] = value_net(adjusted_expert_output)

        # 加权价值融合
        combined_value = torch.zeros_like(value_outputs['glucose_value'])
        for i, (value_name, value) in enumerate(value_outputs.items()):
            task_weight = task_importance[:, i:i+1]
            combined_value += task_weight * value

        # 时序奖励预测
        temporal_reward_prediction = self.temporal_reward_predictor(current_temporal_feature)

        # 不确定性估计
        uncertainty = self.uncertainty_estimator(adjusted_expert_output)

        return {
            'combined_policy': combined_policy,
            'task_policies': policy_outputs,
            'combined_value': combined_value,
            'task_values': value_outputs,
            'task_importance': task_importance,
            'expert_output': expert_output,
            'adjusted_expert_output': adjusted_expert_output,
            'gate_weights': gate_weights,
            'expert_weight_adjustment': expert_weight_adjustment,
            'temporal_features': current_temporal_feature,
            'temporal_attention_weights': temporal_attention_weights,
            'temporal_reward_prediction': temporal_reward_prediction,
            'uncertainty': uncertainty,
            'hidden_state': h_n if sequence_data is not None else None,
            # 多任务学习相关信息
            'multi_task_output': multi_task_output,
            'task_correlation': multi_task_output['task_correlation'],
            'task_conflict_scores': multi_task_output['conflict_scores'],
            'expert_task_performance': self.moe_system.get_expert_task_performance()
        }

    def calculate_multi_objective_reward(self,
                                       glucose_control: float,
                                       nutrition_balance: float,
                                       patient_satisfaction: float,
                                       cultural_adaptation: float) -> torch.Tensor:
        """计算多目标奖励"""
        rewards = torch.tensor([
            glucose_control,
            nutrition_balance,
            patient_satisfaction,
            cultural_adaptation
        ])

        # 归一化权重
        normalized_weights = F.softmax(self.reward_weights, dim=0)

        # 加权奖励
        total_reward = torch.sum(rewards * normalized_weights)

        return total_reward

    def update_reward_weights(self, performance_feedback: Dict[str, float]):
        """更新奖励权重"""
        # 基于性能反馈调整权重
        for i, (key, value) in enumerate(performance_feedback.items()):
            if i < len(self.reward_weights):
                # 简单的权重调整策略
                adjustment = (value - 0.5) * 0.1  # 假设0.5为基准
                self.reward_weights.data[i] += adjustment

        # 确保权重为正
        self.reward_weights.data = torch.clamp(self.reward_weights.data, min=0.1)

class PromptTuningModule(nn.Module):
    """提示微调模块"""

    def __init__(self, input_dim: int, prompt_length: int = 10,
                 prompt_dim: int = 64):
        super().__init__()

        self.prompt_length = prompt_length
        self.prompt_dim = prompt_dim

        # 可学习的软提示向量
        self.soft_prompts = nn.Parameter(
            torch.randn(prompt_length, prompt_dim)
        )

        # 提示投影层
        self.prompt_projection = nn.Linear(prompt_dim, input_dim)

        # 提示注意力机制
        self.prompt_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        batch_size = x.size(0)

        # 投影提示向量
        projected_prompts = self.prompt_projection(self.soft_prompts)
        projected_prompts = projected_prompts.unsqueeze(0).expand(batch_size, -1, -1)

        # 输入特征扩展
        x_expanded = x.unsqueeze(1)  # [batch_size, 1, input_dim]

        # 提示注意力
        attended_features, _ = self.prompt_attention(
            x_expanded, projected_prompts, projected_prompts
        )

        # 融合提示信息
        enhanced_features = x_expanded + attended_features

        return enhanced_features.squeeze(1)

# 使用示例
def main():
    """使用示例"""
    # 创建专家配置
    expert_configs = [
        ExpertConfig(
            expert_type='health_indicator',
            input_dim=20,
            hidden_dim=128,
            output_dim=64
        ),
        ExpertConfig(
            expert_type='medical_diagnosis',
            input_dim=20,
            hidden_dim=128,
            output_dim=64
        ),
        ExpertConfig(
            expert_type='lifestyle_assessment',
            input_dim=20,
            hidden_dim=128,
            output_dim=64
        )
    ]

    # 创建MERL系统
    merl_system = MERLSystem(
        input_dim=20,
        action_dim=5,  # 5种推荐动作
        expert_configs=expert_configs
    )

    # 创建提示微调模块
    prompt_module = PromptTuningModule(input_dim=20)

    # 模拟输入
    batch_size = 32
    input_features = torch.randn(batch_size, 20)

    # 提示微调
    enhanced_features = prompt_module(input_features)

    # MERL系统处理
    output = merl_system(enhanced_features)

    print("MERL系统输出:")
    print(f"策略概率形状: {output['policy_probs'].shape}")
    print(f"价值估计形状: {output['value'].shape}")
    print(f"门控权重形状: {output['gate_weights'].shape}")

    # 计算多目标奖励
    reward = merl_system.calculate_multi_objective_reward(
        glucose_control=0.8,
        nutrition_balance=0.7,
        patient_satisfaction=0.9,
        cultural_adaptation=0.6
    )
    print(f"多目标奖励: {reward.item():.3f}")

    # 获取专家使用统计
    usage_stats = merl_system.moe_system.get_expert_usage_stats()
    print("专家使用统计:", usage_stats)

if __name__ == "__main__":
    main()

__all__ = ["'logger'", "'ExperienceReplayBuffer'", "'ExpertConfig'", "'HealthIndicatorExpert'", "'MedicalDiagnosisExpert'", "'LifestyleAssessmentExpert'", "'GatingNetwork'", "'MixtureOfExperts'", "'MultiTaskLearningFramework'", "'EnhancedMixtureOfExperts'", "'EnhancedMERLSystem'", "'PromptTuningModule'", "'main'"]
