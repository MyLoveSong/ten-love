

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨域知识蒸馏框架
实现医学知识增强的可解释混合专家决策系统
基于项目申请表中的创新点二设计
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from dataclasses import dataclass
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class KnowledgeRule:
    """知识规则"""
    rule_id: str
    condition: str
    conclusion: str
    confidence: float
    domain: str  # 医学、营养学、行为学等

class MedicalKnowledgeBase:
    """医学知识库"""

    def __init__(self):
        self.rules: Dict[str, KnowledgeRule] = {}
        self.embeddings: Dict[str, torch.Tensor] = {}
        self.domain_embeddings: Dict[str, torch.Tensor] = {}

        self._initialize_medical_rules()

    def _initialize_medical_rules(self):
        """初始化医学规则"""
        # 糖尿病相关规则
        diabetes_rules = [
            {
                'rule_id': 'diabetes_glucose_high',
                'condition': 'glucose > 7.0',
                'conclusion': 'high_risk_diabetes',
                'confidence': 0.9,
                'domain': '医学'
            },
            {
                'rule_id': 'diabetes_glucose_normal',
                'condition': '3.9 <= glucose <= 6.1',
                'conclusion': 'normal_glucose',
                'confidence': 0.95,
                'domain': '医学'
            },
            {
                'rule_id': 'diabetes_hba1c_high',
                'condition': 'hba1c > 6.5',
                'conclusion': 'diabetes_diagnosis',
                'confidence': 0.85,
                'domain': '医学'
            }
        ]

        # 营养学规则
        nutrition_rules = [
            {
                'rule_id': 'carb_impact_high',
                'condition': 'carbohydrates > 50',
                'conclusion': 'high_glucose_impact',
                'confidence': 0.8,
                'domain': '营养学'
            },
            {
                'rule_id': 'fiber_benefit',
                'condition': 'fiber > 25',
                'conclusion': 'glucose_control_benefit',
                'confidence': 0.75,
                'domain': '营养学'
            },
            {
                'rule_id': 'protein_satiety',
                'condition': 'protein > 20',
                'conclusion': 'increased_satiety',
                'confidence': 0.7,
                'domain': '营养学'
            }
        ]

        # 行为学规则
        behavior_rules = [
            {
                'rule_id': 'exercise_benefit',
                'condition': 'exercise_duration > 30',
                'conclusion': 'glucose_control_improvement',
                'confidence': 0.8,
                'domain': '行为学'
            },
            {
                'rule_id': 'sleep_impact',
                'condition': 'sleep_hours < 6',
                'conclusion': 'glucose_control_worsening',
                'confidence': 0.7,
                'domain': '行为学'
            },
            {
                'rule_id': 'stress_impact',
                'condition': 'stress_level > 4',
                'conclusion': 'glucose_instability',
                'confidence': 0.65,
                'domain': '行为学'
            }
        ]

        # 添加所有规则
        all_rules = diabetes_rules + nutrition_rules + behavior_rules
        for rule_data in all_rules:
            rule = KnowledgeRule(**rule_data)
            self.rules[rule.rule_id] = rule

        logger.info(f"医学知识库初始化完成，包含 {len(self.rules)} 条规则")

    def get_rules_by_domain(self, domain: str) -> List[KnowledgeRule]:
        """根据领域获取规则"""
        return [rule for rule in self.rules.values() if rule.domain == domain]

    def get_applicable_rules(self, features: Dict[str, float]) -> List[KnowledgeRule]:
        """获取适用的规则"""
        applicable_rules = []

        for rule in self.rules.values():
            if self._evaluate_condition(rule.condition, features):
                applicable_rules.append(rule)

        return applicable_rules

    def _evaluate_condition(self, condition: str, features: Dict[str, float]) -> bool:
        """评估条件是否满足"""
        try:
            # 简单的条件评估（实际应用中需要更复杂的解析器）
            if 'glucose > 7.0' in condition:
                return features.get('glucose', 0) > 7.0
            elif 'glucose > 6.1' in condition:
                return features.get('glucose', 0) > 6.1
            elif '3.9 <= glucose <= 6.1' in condition:
                glucose = features.get('glucose', 0)
                return 3.9 <= glucose <= 6.1
            elif 'hba1c > 6.5' in condition:
                return features.get('hba1c', 0) > 6.5
            elif 'carbohydrates > 50' in condition:
                return features.get('carbohydrates', 0) > 50
            elif 'fiber > 25' in condition:
                return features.get('fiber', 0) > 25
            elif 'protein > 20' in condition:
                return features.get('protein', 0) > 20
            elif 'exercise_duration > 30' in condition:
                return features.get('exercise_duration', 0) > 30
            elif 'sleep_hours < 6' in condition:
                return features.get('sleep_hours', 0) < 6
            elif 'stress_level > 4' in condition:
                return features.get('stress_level', 0) > 4

            return False
        except:
            return False

class NeuralSymbolicModule(nn.Module):
    """神经符号联合学习模块"""

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_domains: int = 3):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_domains = num_domains

        # 神经特征提取器
        self.neural_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU()
        )

        # 符号知识编码器
        self.symbolic_encoder = nn.ModuleDict({
            '医学': nn.Linear(hidden_dim // 2, hidden_dim // 4),
            '营养学': nn.Linear(hidden_dim // 2, hidden_dim // 4),
            '行为学': nn.Linear(hidden_dim // 2, hidden_dim // 4)
        })

        # 知识融合层
        self.knowledge_fusion = nn.Sequential(
            nn.Linear(hidden_dim // 4 * num_domains, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )

        # 可解释性层
        self.explainability_layer = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.ReLU(),
            nn.Linear(hidden_dim // 8, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, knowledge_rules: List[KnowledgeRule]) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 神经特征提取
        neural_features = self.neural_extractor(x)

        # 符号知识编码
        domain_features = []
        for domain, encoder in self.symbolic_encoder.items():
            domain_feature = encoder(neural_features)
            domain_features.append(domain_feature)

        # 知识融合
        fused_knowledge = torch.cat(domain_features, dim=-1)
        knowledge_output = self.knowledge_fusion(fused_knowledge)

        # 可解释性分析
        explainability_score = self.explainability_layer(knowledge_output)

        return {
            'neural_features': neural_features,
            'domain_features': domain_features,
            'knowledge_output': knowledge_output,
            'explainability_score': explainability_score
        }

class CrossDomainKnowledgeDistillation(nn.Module):
    """跨域知识蒸馏框架"""

    def __init__(self, teacher_models: Dict[str, nn.Module],
                 student_model: nn.Module, temperature: float = 3.0):
        super().__init__()

        self.teacher_models = nn.ModuleDict(teacher_models)
        self.student_model = student_model
        self.temperature = temperature

        # 知识蒸馏损失权重
        self.distillation_weights = nn.Parameter(torch.ones(len(teacher_models)))

        # 语义对齐层
        self.semantic_alignment = nn.ModuleDict({
            domain: nn.Linear(256, 128) for domain in teacher_models.keys()
        })

        # 跨域注意力机制
        self.cross_domain_attention = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 学生模型输出
        student_output = self.student_model(x)

        # 教师模型输出
        teacher_outputs = {}
        for domain, teacher in self.teacher_models.items():
            teacher_output = teacher(x)
            teacher_outputs[domain] = teacher_output

        return {
            'student_output': student_output,
            'teacher_outputs': teacher_outputs
        }

    def compute_distillation_loss(self, student_output: torch.Tensor,
                                teacher_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算知识蒸馏损失"""
        total_loss = 0.0

        # 归一化权重
        normalized_weights = F.softmax(self.distillation_weights, dim=0)

        for i, (domain, teacher_output) in enumerate(teacher_outputs.items()):
            # 软标签蒸馏
            teacher_soft = F.softmax(teacher_output / self.temperature, dim=-1)
            student_soft = F.log_softmax(student_output / self.temperature, dim=-1)

            # KL散度损失
            kl_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')

            # 加权损失
            weighted_loss = normalized_weights[i] * kl_loss
            total_loss += weighted_loss

        return total_loss * (self.temperature ** 2)

    def compute_semantic_alignment_loss(self, teacher_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算语义对齐损失"""
        aligned_features = []

        for domain, teacher_output in teacher_outputs.items():
            aligned_feature = self.semantic_alignment[domain](teacher_output)
            aligned_features.append(aligned_feature.unsqueeze(1))

        # 堆叠特征
        stacked_features = torch.cat(aligned_features, dim=1)

        # 跨域注意力
        attended_features, attention_weights = self.cross_domain_attention(
            stacked_features, stacked_features, stacked_features
        )

        # 计算对齐损失（特征相似性）
        alignment_loss = F.mse_loss(attended_features, stacked_features)

        return alignment_loss

class AdaLoRA(nn.Module):
    """AdaLoRA参数高效微调模块"""

    def __init__(self, base_model: nn.Module, rank: int = 16,
                 alpha: float = 16, dropout: float = 0.1):
        super().__init__()

        self.base_model = base_model
        self.rank = rank
        self.alpha = alpha

        # 为每个线性层添加LoRA适配器
        self.lora_adapters = nn.ModuleDict()

        for name, module in base_model.named_modules():
            if isinstance(module, nn.Linear):
                adapter_name = name.replace('.', '_')
                self.lora_adapters[adapter_name] = LoRAAdapter(
                    module.in_features, module.out_features, rank, alpha, dropout
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 使用基础模型
        return self.base_model(x)

    def get_lora_output(self, x: torch.Tensor, layer_name: str) -> torch.Tensor:
        """获取LoRA适配器输出"""
        adapter_name = layer_name.replace('.', '_')
        if adapter_name in self.lora_adapters:
            return self.lora_adapters[adapter_name](x)
        return x

class LoRAAdapter(nn.Module):
    """LoRA适配器"""

    def __init__(self, in_features: int, out_features: int,
                 rank: int, alpha: float, dropout: float):
        super().__init__()

        self.rank = rank
        self.alpha = alpha

        # LoRA矩阵
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)

        # 初始化
        nn.init.kaiming_uniform_(self.lora_A.weight, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # LoRA计算
        lora_output = self.lora_B(self.dropout(self.lora_A(x)))

        # 缩放
        scaled_output = lora_output * (self.alpha / self.rank)

        return scaled_output

class ExplainableKnowledgeSystem:
    """可解释知识系统"""

    def __init__(self, knowledge_base: MedicalKnowledgeBase):
        self.knowledge_base = knowledge_base
        self.neural_symbolic = NeuralSymbolicModule(input_dim=20)
        self.counterfactual_generator = CounterfactualGenerator()

    def generate_explanation(self, features: Dict[str, float],
                           prediction: float) -> Dict[str, Any]:
        """生成可解释性分析"""
        # 获取适用规则
        applicable_rules = self.knowledge_base.get_applicable_rules(features)

        # 神经符号分析
        feature_tensor = torch.tensor([list(features.values())], dtype=torch.float32)
        neural_output = self.neural_symbolic(feature_tensor, applicable_rules)

        # 生成反事实解释
        counterfactual = self.counterfactual_generator.generate(
            features, prediction, applicable_rules
        )

        explanation = {
            'prediction': prediction,
            'applicable_rules': [
                {
                    'rule_id': rule.rule_id,
                    'condition': rule.condition,
                    'conclusion': rule.conclusion,
                    'confidence': rule.confidence,
                    'domain': rule.domain
                } for rule in applicable_rules
            ],
            'neural_analysis': {
                'explainability_score': neural_output['explainability_score'].item(),
                'domain_contributions': {
                    '医学': neural_output['domain_features'][0].mean().item(),
                    '营养学': neural_output['domain_features'][1].mean().item(),
                    '行为学': neural_output['domain_features'][2].mean().item()
                }
            },
            'counterfactual_explanation': counterfactual
        }

        return explanation

class CounterfactualGenerator:
    """反事实解释生成器"""

    def generate(self, features: Dict[str, float], prediction: float,
                rules: List[KnowledgeRule]) -> Dict[str, Any]:
        """生成反事实解释"""
        counterfactual = {
            'original_prediction': prediction,
            'alternative_scenarios': []
        }

        # 基于规则生成反事实场景
        for rule in rules:
            if rule.conclusion == 'high_risk_diabetes':
                # 生成降低风险的场景
                scenario = {
                    'description': f"如果{rule.condition.replace('>', '≤')}",
                    'expected_outcome': '降低糖尿病风险',
                    'confidence': rule.confidence,
                    'required_changes': self._generate_required_changes(rule, features)
                }
                counterfactual['alternative_scenarios'].append(scenario)

        return counterfactual

    def _generate_required_changes(self, rule: KnowledgeRule,
                                 features: Dict[str, float]) -> List[str]:
        """生成所需改变"""
        changes = []

        if 'glucose > 7.0' in rule.condition:
            changes.append("降低血糖水平至7.0以下")
            changes.append("控制碳水化合物摄入")
            changes.append("增加运动量")
        elif 'carbohydrates > 50' in rule.condition:
            changes.append("减少碳水化合物摄入至50g以下")
            changes.append("选择低GI食物")

        return changes

# 使用示例
def main():
    """使用示例"""
    # 创建医学知识库
    knowledge_base = MedicalKnowledgeBase()

    # 创建可解释知识系统
    explainable_system = ExplainableKnowledgeSystem(knowledge_base)

    # 模拟患者特征
    patient_features = {
        'glucose': 8.5,
        'hba1c': 7.2,
        'carbohydrates': 60,
        'fiber': 15,
        'protein': 25,
        'exercise_duration': 20,
        'sleep_hours': 5,
        'stress_level': 4
    }

    # 生成解释
    explanation = explainable_system.generate_explanation(patient_features, 0.8)

    print("可解释性分析结果:")
    print(f"预测值: {explanation['prediction']}")
    print(f"可解释性评分: {explanation['neural_analysis']['explainability_score']:.3f}")
    print("适用规则:")
    for rule in explanation['applicable_rules']:
        print(f"  - {rule['rule_id']}: {rule['condition']} -> {rule['conclusion']} (置信度: {rule['confidence']})")

    print("领域贡献:")
    for domain, contribution in explanation['neural_analysis']['domain_contributions'].items():
        print(f"  - {domain}: {contribution:.3f}")

    print("反事实解释:")
    for scenario in explanation['counterfactual_explanation']['alternative_scenarios']:
        print(f"  - {scenario['description']}: {scenario['expected_outcome']}")
        print(f"    所需改变: {', '.join(scenario['required_changes'])}")

if __name__ == "__main__":
    main()

__all__ = ["'logger'", "'KnowledgeRule'", "'MedicalKnowledgeBase'", "'NeuralSymbolicModule'", "'CrossDomainKnowledgeDistillation'", "'AdaLoRA'", "'LoRAAdapter'", "'ExplainableKnowledgeSystem'", "'CounterfactualGenerator'", "'main'"]
