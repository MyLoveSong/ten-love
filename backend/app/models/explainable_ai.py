"""
可解释AI模块 - 为医生和患者提供预测解释
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod
# 可选导入
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import lime
    import lime.tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

try:
    from captum.attr import IntegratedGradients, GradientShap, Saliency
    from captum.attr import visualization as viz
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False

logger = logging.getLogger(__name__)

class ExplainableAI:
    """可解释AI模块"""

    def __init__(self, model: nn.Module, feature_names: List[str] = None):
        self.model = model
        self.feature_names = feature_names or [f"feature_{i}" for i in range(10)]
        self.attribution_methods = {}
        self._initialize_attribution_methods()

        logger.info("可解释AI模块初始化完成")

    def _initialize_attribution_methods(self):
        """初始化归因方法"""
        self.attribution_methods = {}

        if CAPTUM_AVAILABLE:
            self.attribution_methods.update({
                'integrated_gradients': IntegratedGradients(self.model),
                'gradient_shap': GradientShap(self.model),
                'saliency': Saliency(self.model)
            })
        else:
            # 使用简化的归因方法
            logger.warning("Captum不可用，使用简化的归因方法")
            self.attribution_methods = {
                'simple_gradient': self._simple_gradient_attribution
            }

    def generate_explanations(self, model_predictions: Dict[str, Any],
                            input_data: torch.Tensor,
                            target_class: int = None) -> Dict[str, Any]:
        """为模型预测生成解释"""
        explanations = {}

        # 血糖预测解释
        if 'glucose_prediction' in model_predictions:
            explanations['glucose_prediction'] = self._explain_glucose_prediction(
                input_data, model_predictions['glucose_prediction']
            )

        # 膳食影响解释
        if 'meal_impact' in model_predictions:
            explanations['meal_impact'] = self._explain_meal_impact(
                input_data, model_predictions['meal_impact']
            )

        # 运动效果解释
        if 'exercise_effect' in model_predictions:
            explanations['exercise_effect'] = self._explain_exercise_effect(
                input_data, model_predictions['exercise_effect']
            )

        # 综合解释
        explanations['comprehensive'] = self._generate_comprehensive_explanation(
            input_data, model_predictions
        )

        return explanations

    def _explain_glucose_prediction(self, input_data: torch.Tensor,
                                  prediction: Dict[str, Any]) -> Dict[str, Any]:
        """解释血糖预测"""
        # 使用Integrated Gradients
        ig = self.attribution_methods['integrated_gradients']
        attributions = ig.attribute(input_data, target=0)

        # 计算特征重要性
        feature_importance = torch.mean(torch.abs(attributions), dim=0)

        # 生成解释文本
        explanation_text = self._generate_glucose_explanation_text(
            feature_importance, prediction
        )

        return {
            'attributions': attributions.detach().numpy(),
            'feature_importance': feature_importance.detach().numpy(),
            'explanation_text': explanation_text,
            'method': 'integrated_gradients'
        }

    def _explain_meal_impact(self, input_data: torch.Tensor,
                           impact: Dict[str, Any]) -> Dict[str, Any]:
        """解释膳食影响"""
        # 使用Gradient SHAP
        gs = self.attribution_methods['gradient_shap']
        baseline = torch.zeros_like(input_data)
        attributions = gs.attribute(input_data, baselines=baseline, target=0)

        # 分析膳食相关特征
        meal_features = self._extract_meal_features(input_data)
        meal_attributions = attributions[:, meal_features]

        explanation_text = self._generate_meal_explanation_text(
            meal_attributions, impact
        )

        return {
            'attributions': attributions.detach().numpy(),
            'meal_attributions': meal_attributions.detach().numpy(),
            'explanation_text': explanation_text,
            'method': 'gradient_shap'
        }

    def _explain_exercise_effect(self, input_data: torch.Tensor,
                              effect: Dict[str, Any]) -> Dict[str, Any]:
        """解释运动效果"""
        # 使用Saliency
        saliency = self.attribution_methods['saliency']
        attributions = saliency.attribute(input_data, target=0)

        # 分析运动相关特征
        exercise_features = self._extract_exercise_features(input_data)
        exercise_attributions = attributions[:, exercise_features]

        explanation_text = self._generate_exercise_explanation_text(
            exercise_attributions, effect
        )

        return {
            'attributions': attributions.detach().numpy(),
            'exercise_attributions': exercise_attributions.detach().numpy(),
            'explanation_text': explanation_text,
            'method': 'saliency'
        }

    def _generate_comprehensive_explanation(self, input_data: torch.Tensor,
                                          predictions: Dict[str, Any]) -> Dict[str, Any]:
        """生成综合解释"""
        # 整合所有归因结果
        all_attributions = []
        for method_name, method in self.attribution_methods.items():
            if method_name == 'integrated_gradients':
                attr = method.attribute(input_data, target=0)
            elif method_name == 'gradient_shap':
                baseline = torch.zeros_like(input_data)
                attr = method.attribute(input_data, baselines=baseline, target=0)
            else:  # saliency
                attr = method.attribute(input_data, target=0)

            all_attributions.append(attr)

        # 平均归因结果
        avg_attributions = torch.mean(torch.stack(all_attributions), dim=0)

        # 生成综合解释文本
        explanation_text = self._generate_comprehensive_explanation_text(
            avg_attributions, predictions
        )

        return {
            'attributions': avg_attributions.detach().numpy(),
            'explanation_text': explanation_text,
            'confidence_score': self._calculate_explanation_confidence(avg_attributions)
        }

    def _extract_meal_features(self, input_data: torch.Tensor) -> List[int]:
        """提取膳食相关特征索引"""
        # 假设前5个特征是膳食相关
        return list(range(min(5, input_data.size(1))))

    def _extract_exercise_features(self, input_data: torch.Tensor) -> List[int]:
        """提取运动相关特征索引"""
        # 假设第6-8个特征是运动相关
        start_idx = min(5, input_data.size(1))
        end_idx = min(8, input_data.size(1))
        return list(range(start_idx, end_idx))

    def _generate_glucose_explanation_text(self, feature_importance: torch.Tensor,
                                         prediction: Dict[str, Any]) -> str:
        """生成血糖预测解释文本"""
        top_features = torch.topk(feature_importance, k=3)

        explanation = f"血糖预测为 {prediction.get('predicted_glucose', 'N/A'):.1f} mmol/L。"
        explanation += "主要影响因素包括："

        for i, idx in enumerate(top_features.indices):
            feature_name = self.feature_names[idx.item()]
            importance = top_features.values[i].item()
            explanation += f" {feature_name}({importance:.2f})"
            if i < len(top_features.indices) - 1:
                explanation += ","

        explanation += "。"

        return explanation

    def _generate_meal_explanation_text(self, meal_attributions: torch.Tensor,
                                      impact: Dict[str, Any]) -> str:
        """生成膳食影响解释文本"""
        avg_impact = torch.mean(torch.abs(meal_attributions))

        explanation = f"膳食对血糖的影响程度为 {avg_impact.item():.2f}。"

        if impact.get('impact_level') == 'high':
            explanation += "建议减少高碳水化合物食物的摄入。"
        elif impact.get('impact_level') == 'medium':
            explanation += "建议适量摄入，注意血糖监测。"
        else:
            explanation += "膳食选择较为合理。"

        return explanation

    def _generate_exercise_explanation_text(self, exercise_attributions: torch.Tensor,
                                          effect: Dict[str, Any]) -> str:
        """生成运动效果解释文本"""
        avg_effect = torch.mean(torch.abs(exercise_attributions))

        explanation = f"运动对血糖的影响程度为 {avg_effect.item():.2f}。"

        if effect.get('effect_type') == 'hypoglycemic':
            explanation += "运动可能导致血糖下降，建议运动前适当补充碳水化合物。"
        elif effect.get('effect_type') == 'hyperglycemic':
            explanation += "运动有助于血糖控制，建议保持规律运动。"
        else:
            explanation += "运动对血糖影响适中。"

        return explanation

    def _generate_comprehensive_explanation_text(self, attributions: torch.Tensor,
                                               predictions: Dict[str, Any]) -> str:
        """生成综合解释文本"""
        explanation = "综合血糖管理分析：\n"

        # 血糖预测
        if 'glucose_prediction' in predictions:
            pred_value = predictions['glucose_prediction'].get('predicted_glucose', 0)
            explanation += f"• 预测血糖值：{pred_value:.1f} mmol/L\n"

        # 膳食影响
        if 'meal_impact' in predictions:
            impact_level = predictions['meal_impact'].get('impact_level', 'unknown')
            explanation += f"• 膳食影响等级：{impact_level}\n"

        # 运动效果
        if 'exercise_effect' in predictions:
            effect_type = predictions['exercise_effect'].get('effect_type', 'unknown')
            explanation += f"• 运动效果类型：{effect_type}\n"

        # 总体建议
        explanation += "• 建议：保持规律作息，合理饮食，适量运动。"

        return explanation

    def _calculate_explanation_confidence(self, attributions: torch.Tensor) -> float:
        """计算解释置信度"""
        # 基于归因的稳定性计算置信度
        attribution_std = torch.std(attributions)
        confidence = max(0, 1 - attribution_std.item())
        return confidence

    def _simple_gradient_attribution(self, input_data: torch.Tensor) -> torch.Tensor:
        """简化的梯度归因方法"""
        input_data.requires_grad_(True)
        output = self.model(input_data)
        output.backward(torch.ones_like(output))
        return input_data.grad

    def visualize_attributions(self, attributions: np.ndarray,
                             feature_names: List[str] = None,
                             title: str = "Feature Attributions") -> plt.Figure:
        """可视化归因结果"""
        if feature_names is None:
            feature_names = self.feature_names

        fig, ax = plt.subplots(figsize=(10, 6))

        # 计算平均归因
        avg_attributions = np.mean(np.abs(attributions), axis=0)

        # 创建条形图
        bars = ax.bar(range(len(avg_attributions)), avg_attributions)

        # 设置标签
        ax.set_xlabel('Features')
        ax.set_ylabel('Attribution Magnitude')
        ax.set_title(title)
        ax.set_xticks(range(len(feature_names)))
        ax.set_xticklabels(feature_names, rotation=45)

        # 添加颜色映射
        colors = plt.cm.viridis(avg_attributions / np.max(avg_attributions))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        plt.tight_layout()
        return fig

    def generate_clinical_report(self, explanations: Dict[str, Any]) -> str:
        """生成临床报告"""
        report = "=== 糖尿病管理AI解释报告 ===\n\n"

        # 血糖预测解释
        if 'glucose_prediction' in explanations:
            report += "1. 血糖预测分析\n"
            report += f"   {explanations['glucose_prediction']['explanation_text']}\n\n"

        # 膳食影响解释
        if 'meal_impact' in explanations:
            report += "2. 膳食影响分析\n"
            report += f"   {explanations['meal_impact']['explanation_text']}\n\n"

        # 运动效果解释
        if 'exercise_effect' in explanations:
            report += "3. 运动效果分析\n"
            report += f"   {explanations['exercise_effect']['explanation_text']}\n\n"

        # 综合解释
        if 'comprehensive' in explanations:
            report += "4. 综合管理建议\n"
            report += f"   {explanations['comprehensive']['explanation_text']}\n\n"

        # 置信度
        if 'comprehensive' in explanations:
            confidence = explanations['comprehensive'].get('confidence_score', 0)
            report += f"解释置信度：{confidence:.2f}\n"

        return report
