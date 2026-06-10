"""
模型可解释性和临床归因分析功能
实现注意力可视化、特征重要性分析和决策路径追踪
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class FeatureImportance:
    """特征重要性"""
    feature_name: str
    importance_score: float
    contribution_type: str  # 'positive', 'negative', 'neutral'
    clinical_significance: str

@dataclass
class ClinicalAttribution:
    """临床归因"""
    factor_name: str
    attribution_score: float
    clinical_interpretation: str
    evidence_level: str  # 'high', 'medium', 'low'

class GradCAM:
    """Grad-CAM可视化"""

    def __init__(self, model: nn.Module, target_layer: str):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # 注册钩子
        self._register_hooks()

    def _register_hooks(self):
        """注册前向和反向钩子"""
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        # 找到目标层并注册钩子
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """生成类激活图"""
        # 前向传播
        output = self.model(input_tensor)

        # 反向传播
        self.model.zero_grad()
        class_score = output[:, class_idx].sum()
        class_score.backward()

        # 计算权重
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)

        # 生成CAM
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = F.relu(cam)

        # 归一化
        cam = cam - cam.min()
        cam = cam / cam.max()

        return cam.detach().cpu().numpy()

class AttentionVisualizer:
    """注意力权重可视化器"""

    def __init__(self):
        self.attention_maps = {}

    def visualize_attention_weights(
        self,
        attention_weights: torch.Tensor,
        input_tokens: List[str],
        save_path: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        可视化注意力权重
        Args:
            attention_weights: 注意力权重 [num_heads, seq_len, seq_len]
            input_tokens: 输入标记列表
            save_path: 保存路径
        Returns:
            注意力图字典
        """
        num_heads = attention_weights.shape[0]
        attention_maps = {}

        for head in range(num_heads):
            # 提取单个头的注意力权重
            head_attention = attention_weights[head].detach().cpu().numpy()

            # 创建热力图
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                head_attention,
                xticklabels=input_tokens,
                yticklabels=input_tokens,
                cmap='Blues',
                annot=True,
                fmt='.2f'
            )
            plt.title(f'Attention Head {head + 1}')
            plt.xlabel('Key Tokens')
            plt.ylabel('Query Tokens')

            if save_path:
                plt.savefig(f"{save_path}_head_{head}.png", dpi=300, bbox_inches='tight')

            attention_maps[f'head_{head}'] = head_attention
            plt.close()

        return attention_maps

    def visualize_cross_modal_attention(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        attention_weights: torch.Tensor,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """可视化跨模态注意力"""
        attention_map = attention_weights.detach().cpu().numpy()

        plt.figure(figsize=(12, 8))
        sns.heatmap(
            attention_map,
            cmap='Reds',
            annot=True,
            fmt='.3f',
            cbar_kws={'label': 'Attention Weight'}
        )
        plt.title('Cross-Modal Attention: Visual-Text Alignment')
        plt.xlabel('Text Features')
        plt.ylabel('Visual Features')

        if save_path:
            plt.savefig(f"{save_path}_cross_modal.png", dpi=300, bbox_inches='tight')
        plt.close()

        return attention_map

class FeatureImportanceAnalyzer:
    """特征重要性分析器"""

    def __init__(self):
        self.feature_names = [
            "血糖水平", "餐后时间", "碳水化合物摄入", "蛋白质摄入", "脂肪摄入",
            "纤维摄入", "运动强度", "睡眠质量", "压力水平", "药物使用"
        ]

    def calculate_shap_values(
        self,
        model: nn.Module,
        input_data: torch.Tensor,
        baseline: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """计算SHAP值"""
        if baseline is None:
            baseline = torch.zeros_like(input_data)

        # 简化的SHAP计算（集成梯度近似）
        model.eval()

        # 计算梯度
        input_data.requires_grad_(True)
        output = model(input_data)

        # 对输出求梯度
        gradient = torch.autograd.grad(
            outputs=output.sum(),
            inputs=input_data,
            create_graph=False
        )[0]

        # 计算重要性分数
        importance_scores = (input_data - baseline) * gradient
        importance_scores = importance_scores.squeeze().detach().cpu().numpy()

        # 创建特征重要性字典
        feature_importance = {}
        for i, feature_name in enumerate(self.feature_names[:len(importance_scores)]):
            feature_importance[feature_name] = float(importance_scores[i])

        return feature_importance

    def analyze_feature_interactions(
        self,
        model: nn.Module,
        input_data: torch.Tensor
    ) -> Dict[Tuple[str, str], float]:
        """分析特征交互"""
        model.eval()
        interactions = {}

        # 计算所有特征对的交互效应
        for i in range(len(self.feature_names)):
            for j in range(i + 1, len(self.feature_names)):
                if i < input_data.shape[-1] and j < input_data.shape[-1]:
                    # 计算单独效应
                    input_i = input_data.clone()
                    input_i[:, j] = 0
                    effect_i = model(input_i)

                    input_j = input_data.clone()
                    input_j[:, i] = 0
                    effect_j = model(input_j)

                    # 计算联合效应
                    input_ij = input_data.clone()
                    effect_ij = model(input_ij)

                    # 计算交互效应
                    interaction = (effect_ij - effect_i - effect_j).abs().mean().item()
                    interactions[(self.feature_names[i], self.feature_names[j])] = interaction

        return interactions

class ClinicalAttributionAnalyzer:
    """临床归因分析器"""

    def __init__(self):
        self.clinical_factors = {
            "血糖水平": {
                "normal_range": (3.9, 6.1),
                "risk_levels": {
                    "low": (0, 3.9),
                    "normal": (3.9, 6.1),
                    "high": (6.1, 7.8),
                    "very_high": (7.8, float('inf'))
                }
            },
            "碳水化合物摄入": {
                "normal_range": (45, 65),  # 占总热量百分比
                "risk_levels": {
                    "low": (0, 45),
                    "normal": (45, 65),
                    "high": (65, 80),
                    "very_high": (80, 100)
                }
            },
            "BMI": {
                "normal_range": (18.5, 24.9),
                "risk_levels": {
                    "underweight": (0, 18.5),
                    "normal": (18.5, 24.9),
                    "overweight": (25, 29.9),
                    "obese": (30, float('inf'))
                }
            }
        }

    def analyze_clinical_attribution(
        self,
        feature_values: Dict[str, float],
        feature_importance: Dict[str, float],
        prediction: float
    ) -> List[ClinicalAttribution]:
        """分析临床归因"""
        attributions = []

        for factor_name, importance_score in feature_importance.items():
            if factor_name in feature_values and factor_name in self.clinical_factors:
                value = feature_values[factor_name]
                factor_info = self.clinical_factors[factor_name]

                # 确定风险级别
                risk_level = self._get_risk_level(value, factor_info["risk_levels"])

                # 生成临床解释
                interpretation = self._generate_clinical_interpretation(
                    factor_name, value, risk_level, importance_score, prediction
                )

                # 确定证据级别
                evidence_level = self._determine_evidence_level(abs(importance_score))

                attribution = ClinicalAttribution(
                    factor_name=factor_name,
                    attribution_score=importance_score,
                    clinical_interpretation=interpretation,
                    evidence_level=evidence_level
                )

                attributions.append(attribution)

        # 按重要性排序
        attributions.sort(key=lambda x: abs(x.attribution_score), reverse=True)

        return attributions

    def _get_risk_level(self, value: float, risk_levels: Dict[str, Tuple[float, float]]) -> str:
        """确定风险级别"""
        for level, (min_val, max_val) in risk_levels.items():
            if min_val <= value < max_val:
                return level
        return "unknown"

    def _generate_clinical_interpretation(
        self,
        factor_name: str,
        value: float,
        risk_level: str,
        importance_score: float,
        prediction: float
    ) -> str:
        """生成临床解释"""
        interpretations = {
            "血糖水平": {
                "high": f"血糖水平 {value:.1f} mmol/L 超出正常范围，对预测结果有显著影响",
                "normal": f"血糖水平 {value:.1f} mmol/L 在正常范围内",
                "low": f"血糖水平 {value:.1f} mmol/L 偏低，需要注意低血糖风险"
            },
            "碳水化合物摄入": {
                "high": f"碳水化合物摄入 {value:.1f}% 过高，可能导致血糖升高",
                "normal": f"碳水化合物摄入 {value:.1f}% 适中",
                "low": f"碳水化合物摄入 {value:.1f}% 偏低"
            },
            "BMI": {
                "obese": f"BMI {value:.1f} 属于肥胖范围，增加糖尿病并发症风险",
                "overweight": f"BMI {value:.1f} 属于超重范围",
                "normal": f"BMI {value:.1f} 在正常范围内"
            }
        }

        base_interpretation = interpretations.get(factor_name, {}).get(
            risk_level, f"{factor_name}值为 {value:.2f}"
        )

        # 添加重要性说明
        if abs(importance_score) > 0.5:
            importance_desc = "对预测结果有重要影响"
        elif abs(importance_score) > 0.2:
            importance_desc = "对预测结果有中等影响"
        else:
            importance_desc = "对预测结果影响较小"

        return f"{base_interpretation}，{importance_desc}"

    def _determine_evidence_level(self, importance_score: float) -> str:
        """确定证据级别"""
        if importance_score > 0.7:
            return "high"
        elif importance_score > 0.3:
            return "medium"
        else:
            return "low"

class DecisionPathTracker:
    """决策路径追踪器"""

    def __init__(self):
        self.decision_nodes = []

    def track_decision_path(
        self,
        model_outputs: Dict[str, torch.Tensor],
        threshold_values: Dict[str, float]
    ) -> List[Dict[str, Union[str, float, bool]]]:
        """追踪决策路径"""
        decision_path = []

        # 血糖预测决策节点
        if 'glucose_prediction' in model_outputs:
            glucose_pred = model_outputs['glucose_prediction'].item()
            decision_path.append({
                'node_type': 'glucose_prediction',
                'value': glucose_pred,
                'threshold': threshold_values.get('glucose_threshold', 7.0),
                'decision': glucose_pred > threshold_values.get('glucose_threshold', 7.0),
                'description': f"血糖预测值 {glucose_pred:.1f} {'超过' if glucose_pred > 7.0 else '未超过'} 阈值"
            })

        # 营养分析决策节点
        if 'nutrition_analysis' in model_outputs:
            nutrition = model_outputs['nutrition_analysis']
            if isinstance(nutrition, dict):
                for nutrient, value in nutrition.items():
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    decision_path.append({
                        'node_type': 'nutrition_analysis',
                        'nutrient': nutrient,
                        'value': value,
                        'description': f"{nutrient}: {value:.2f}"
                    })

        # 文化适配决策节点
        if 'adaptation_score' in model_outputs:
            adaptation_score = model_outputs['adaptation_score'].item()
            decision_path.append({
                'node_type': 'cultural_adaptation',
                'value': adaptation_score,
                'threshold': 0.5,
                'decision': adaptation_score > 0.5,
                'description': f"文化适配分数 {adaptation_score:.2f}"
            })

        return decision_path

    def generate_decision_summary(self, decision_path: List[Dict]) -> str:
        """生成决策摘要"""
        summary_parts = []

        for node in decision_path:
            if node['node_type'] == 'glucose_prediction':
                if node['decision']:
                    summary_parts.append("血糖预测显示需要干预")
                else:
                    summary_parts.append("血糖预测在正常范围内")

            elif node['node_type'] == 'cultural_adaptation':
                if node['decision']:
                    summary_parts.append("推荐方案符合文化偏好")
                else:
                    summary_parts.append("推荐方案需要文化适配调整")

        return "；".join(summary_parts) if summary_parts else "无明确决策路径"

class ExplainabilityService:
    """可解释性服务"""

    def __init__(self):
        self.attention_visualizer = AttentionVisualizer()
        self.feature_analyzer = FeatureImportanceAnalyzer()
        self.clinical_analyzer = ClinicalAttributionAnalyzer()
        self.path_tracker = DecisionPathTracker()

    def explain_prediction(
        self,
        model: nn.Module,
        input_data: torch.Tensor,
        model_outputs: Dict[str, torch.Tensor],
        feature_values: Dict[str, float],
        attention_weights: Optional[torch.Tensor] = None
    ) -> Dict[str, Union[List, Dict, str]]:
        """解释预测结果"""

        # 1. 计算特征重要性
        feature_importance = self.feature_analyzer.calculate_shap_values(model, input_data)

        # 2. 分析临床归因
        clinical_attributions = self.clinical_analyzer.analyze_clinical_attribution(
            feature_values, feature_importance, model_outputs.get('prediction', 0)
        )

        # 3. 追踪决策路径
        decision_path = self.path_tracker.track_decision_path(
            model_outputs, {'glucose_threshold': 7.0}
        )

        # 4. 生成注意力可视化（如果有注意力权重）
        attention_analysis = {}
        if attention_weights is not None:
            # 简化的注意力分析
            attention_mean = attention_weights.mean(dim=0).detach().cpu().numpy()
            attention_analysis = {
                'attention_distribution': attention_mean.tolist(),
                'max_attention_index': int(np.argmax(attention_mean)),
                'attention_entropy': float(-np.sum(attention_mean * np.log(attention_mean + 1e-8)))
            }

        # 5. 生成解释摘要
        explanation_summary = self._generate_explanation_summary(
            clinical_attributions, decision_path
        )

        return {
            'feature_importance': feature_importance,
            'clinical_attributions': [
                {
                    'factor': attr.factor_name,
                    'score': attr.attribution_score,
                    'interpretation': attr.clinical_interpretation,
                    'evidence_level': attr.evidence_level
                }
                for attr in clinical_attributions
            ],
            'decision_path': decision_path,
            'attention_analysis': attention_analysis,
            'explanation_summary': explanation_summary,
            'confidence_indicators': self._calculate_confidence_indicators(
                feature_importance, clinical_attributions
            )
        }

    def _generate_explanation_summary(
        self,
        clinical_attributions: List[ClinicalAttribution],
        decision_path: List[Dict]
    ) -> str:
        """生成解释摘要"""
        summary_parts = []

        # 主要影响因素
        if clinical_attributions:
            top_factor = clinical_attributions[0]
            summary_parts.append(f"主要影响因素：{top_factor.factor_name}")
            summary_parts.append(top_factor.clinical_interpretation)

        # 决策路径摘要
        decision_summary = self.path_tracker.generate_decision_summary(decision_path)
        if decision_summary:
            summary_parts.append(f"决策分析：{decision_summary}")

        return "。".join(summary_parts)

    def _calculate_confidence_indicators(
        self,
        feature_importance: Dict[str, float],
        clinical_attributions: List[ClinicalAttribution]
    ) -> Dict[str, float]:
        """计算置信度指标"""
        # 特征重要性分布的均匀性
        importance_values = list(feature_importance.values())
        importance_std = np.std(importance_values)

        # 高证据级别归因的比例
        high_evidence_count = sum(
            1 for attr in clinical_attributions
            if attr.evidence_level == 'high'
        )
        high_evidence_ratio = high_evidence_count / max(len(clinical_attributions), 1)

        return {
            'feature_consistency': 1.0 / (1.0 + importance_std),  # 特征一致性
            'clinical_confidence': high_evidence_ratio,  # 临床置信度
            'overall_confidence': (1.0 / (1.0 + importance_std) + high_evidence_ratio) / 2
        }

def create_explainability_service() -> ExplainabilityService:
    """创建可解释性服务"""
    return ExplainabilityService()

if __name__ == "__main__":
    # 创建服务
    service = create_explainability_service()

    # 模拟数据
    input_data = torch.randn(1, 10)
    model_outputs = {
        'glucose_prediction': torch.tensor([7.5]),
        'adaptation_score': torch.tensor([0.8])
    }
    feature_values = {
        '血糖水平': 7.5,
        '碳水化合物摄入': 70.0,
        'BMI': 26.5
    }

    # 创建简单模型用于测试
    test_model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )

    # 生成解释
    explanation = service.explain_prediction(
        test_model, input_data, model_outputs, feature_values
    )

    print("模型解释结果:")
    print(f"解释摘要: {explanation['explanation_summary']}")
    print(f"整体置信度: {explanation['confidence_indicators']['overall_confidence']:.2f}")
    print(f"主要特征重要性: {list(explanation['feature_importance'].items())[:3]}")
    print("模型可解释性系统创建成功！")
