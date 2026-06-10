"""
可解释性热图生成模块 - 临床归因分析
基于特征分解的梯度分析，生成症状-贡献热图
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class InterpretabilityAnalyzer:
    """
    可解释性分析器 - 生成临床归因热图
    """

    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """
        初始化可解释性分析器

        Args:
            model: 训练好的FD增强模型
            device: 设备
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()

        logger.info("可解释性分析器初始化完成")

    def compute_gradient_attribution(self,
                                   x: torch.Tensor,
                                   target_class: int = 0) -> Dict[str, torch.Tensor]:
        """
        计算梯度归因

        Args:
            x: 输入序列 [batch_size, seq_len]
            target_class: 目标类别

        Returns:
            attributions: 各模态的归因分数
        """
        x = x.to(self.device)

        # 确保模型参数需要梯度
        for param in self.model.parameters():
            param.requires_grad_(True)

        # 确保模型在训练模式以启用梯度
        self.model.train()

        # 创建需要梯度的输入
        x_grad = x.clone().detach().requires_grad_(True)

        # 前向传播
        prediction, decomposed_modes = self.model(x_grad)

        # 计算目标类别的梯度
        target_score = prediction[:, target_class]
        target_score.backward(torch.ones_like(target_score))

        # 获取输入梯度
        input_gradients = x_grad.grad

        # 计算各模态的归因
        attributions = {}

        # 趋势模态归因
        trend_grad = torch.abs(input_gradients * decomposed_modes[:, 0, :])
        attributions['trend'] = trend_grad

        # 周期模态归因
        periodic_grad = torch.abs(input_gradients * decomposed_modes[:, 1, :])
        attributions['periodic'] = periodic_grad

        # 残差模态归因
        residual_grad = torch.abs(input_gradients * decomposed_modes[:, 2, :])
        attributions['residual'] = residual_grad

        # 恢复模型到评估模式
        self.model.eval()

        return attributions

    def generate_attribution_heatmap(self,
                                   x: torch.Tensor,
                                   save_path: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        生成归因热图

        Args:
            x: 输入序列 [batch_size, seq_len]
            save_path: 保存路径

        Returns:
            heatmaps: 各模态的热图数据
        """
        # 计算归因（需要梯度）
        attributions = self.compute_gradient_attribution(x)

        # 转换为numpy数组
        heatmaps = {}
        for mode, attr in attributions.items():
            heatmaps[mode] = attr.detach().cpu().numpy()

        # 生成可视化
        if save_path:
            self._plot_attribution_heatmaps(heatmaps, x.detach().cpu().numpy(), save_path)

        return heatmaps

    def _plot_attribution_heatmaps(self,
                                 heatmaps: Dict[str, np.ndarray],
                                 original_data: np.ndarray,
                                 save_path: str):
        """
        绘制归因热图

        Args:
            heatmaps: 热图数据
            original_data: 原始数据
            save_path: 保存路径
        """
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('血糖预测可解释性分析', fontsize=16, fontweight='bold')

        # 原始数据
        axes[0, 0].plot(original_data[0], 'b-', linewidth=2, label='原始血糖数据')
        axes[0, 0].set_title('原始血糖序列', fontsize=12)
        axes[0, 0].set_xlabel('时间点')
        axes[0, 0].set_ylabel('血糖值 (mmol/L)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

        # 趋势模态热图
        im1 = axes[0, 1].imshow(heatmaps['trend'][0:1], aspect='auto', cmap='Reds')
        axes[0, 1].set_title('趋势模态贡献热图', fontsize=12)
        axes[0, 1].set_xlabel('时间点')
        axes[0, 1].set_ylabel('样本')
        plt.colorbar(im1, ax=axes[0, 1], label='归因强度')

        # 周期模态热图
        im2 = axes[1, 0].imshow(heatmaps['periodic'][0:1], aspect='auto', cmap='Blues')
        axes[1, 0].set_title('周期模态贡献热图', fontsize=12)
        axes[1, 0].set_xlabel('时间点')
        axes[1, 0].set_ylabel('样本')
        plt.colorbar(im2, ax=axes[1, 0], label='归因强度')

        # 残差模态热图
        im3 = axes[1, 1].imshow(heatmaps['residual'][0:1], aspect='auto', cmap='Greens')
        axes[1, 1].set_title('残差模态贡献热图', fontsize=12)
        axes[1, 1].set_xlabel('时间点')
        axes[1, 1].set_ylabel('样本')
        plt.colorbar(im3, ax=axes[1, 1], label='归因强度')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"归因热图已保存到: {save_path}")

    def generate_clinical_report(self,
                               x: torch.Tensor,
                               save_path: Optional[str] = None) -> Dict[str, any]:
        """
        生成临床报告

        Args:
            x: 输入序列
            save_path: 保存路径

        Returns:
            report: 临床分析报告
        """
        # 获取预测和分解结果
        prediction, decomposed_modes = self.model(x)

        # 计算归因（需要梯度）
        attributions = self.compute_gradient_attribution(x)

        # 分析各模态贡献
        trend_contribution = torch.mean(attributions['trend']).item()
        periodic_contribution = torch.mean(attributions['periodic']).item()
        residual_contribution = torch.mean(attributions['residual']).item()

        total_contribution = trend_contribution + periodic_contribution + residual_contribution

        # 生成报告
        report = {
            'prediction': prediction.detach().cpu().numpy().tolist(),
            'mode_contributions': {
                'trend': trend_contribution / total_contribution,
                'periodic': periodic_contribution / total_contribution,
                'residual': residual_contribution / total_contribution
            },
            'clinical_insights': self._generate_clinical_insights(
                trend_contribution, periodic_contribution, residual_contribution
            ),
            'risk_assessment': self._assess_risk_level(prediction.item()),
            'recommendations': self._generate_recommendations(
                trend_contribution, periodic_contribution, residual_contribution
            )
        }

        # 保存报告
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

            logger.info(f"临床报告已保存到: {save_path}")

        return report

    def _generate_clinical_insights(self,
                                  trend_contrib: float,
                                  periodic_contrib: float,
                                  residual_contrib: float) -> List[str]:
        """生成临床洞察"""
        insights = []

        if trend_contrib > 0.5:
            insights.append("血糖呈现明显上升趋势，需要关注饮食控制")
        elif trend_contrib < -0.3:
            insights.append("血糖呈现下降趋势，注意低血糖风险")

        if periodic_contrib > 0.4:
            insights.append("血糖存在周期性波动，可能与餐后血糖相关")

        if residual_contrib > 0.3:
            insights.append("血糖存在异常波动，建议增加监测频率")

        if not insights:
            insights.append("血糖变化相对平稳，继续保持当前管理方案")

        return insights

    def _assess_risk_level(self, prediction: float) -> str:
        """评估风险等级"""
        if prediction < 3.9:
            return "低血糖风险"
        elif prediction < 7.8:
            return "正常范围"
        elif prediction < 11.1:
            return "轻度升高"
        else:
            return "高血糖风险"

    def _generate_recommendations(self,
                                trend_contrib: float,
                                periodic_contrib: float,
                                residual_contrib: float) -> List[str]:
        """生成建议"""
        recommendations = []

        if trend_contrib > 0.4:
            recommendations.append("建议调整饮食结构，减少高糖食物摄入")
            recommendations.append("增加餐后运动，帮助控制血糖上升")

        if periodic_contrib > 0.3:
            recommendations.append("注意餐后血糖监测，优化进餐时间")

        if residual_contrib > 0.2:
            recommendations.append("血糖波动较大，建议咨询医生调整治疗方案")

        if not recommendations:
            recommendations.append("血糖控制良好，继续保持当前生活方式")

        return recommendations

class InterpretabilityObjective:
    """
    可解释性目标函数，用于SSA优化可解释性参数
    """

    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device

    def evaluate(self, params: np.ndarray) -> float:
        """
        评估可解释性参数

        Args:
            params: [attention_heads, dropout_rate, interpretability_weight]

        Returns:
            综合评分
        """
        try:
            attention_heads = int(params[0])
            dropout_rate = float(params[1])
            interpretability_weight = float(params[2])

            # 参数合理性检查
            if attention_heads < 4 or attention_heads > 16:
                return float('inf')

            if dropout_rate < 0.0 or dropout_rate > 0.5:
                return float('inf')

            if interpretability_weight < 0.0 or interpretability_weight > 1.0:
                return float('inf')

            # 模拟可解释性评分
            base_score = 0.7

            # 注意力头数影响
            attention_score = 0.1 * (1.0 / (1.0 + abs(attention_heads - 8)))

            # Dropout影响
            dropout_score = 0.05 * (1.0 - abs(dropout_rate - 0.1))

            # 可解释性权重影响
            interpretability_score = 0.15 * interpretability_weight

            total_score = base_score + attention_score + dropout_score + interpretability_score

            return 1.0 - total_score  # 转换为最小化问题

        except Exception as e:
            logger.error(f"可解释性目标函数评估失败: {e}")
            return float('inf')
