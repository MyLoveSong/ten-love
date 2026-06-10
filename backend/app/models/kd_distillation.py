"""
知识蒸馏(KD)模块 - Teacher-Student架构
将SSA优化后的MoE作为Teacher，蒸馏到轻量级Edge模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import os

logger = logging.getLogger(__name__)

class EdgeStudent(nn.Module):
    """
    轻量级学生模型 - 用于边缘设备部署
    目标：< 1.5MB，推理时间 < 30ms
    """

    def __init__(self,
                 input_dim: int = 1,
                 hidden_dim: int = 64,
                 num_classes: int = 1,
                 dropout: float = 0.1):
        """
        初始化Edge学生模型

        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            num_classes: 输出类别数
            dropout: Dropout比例
        """
        super(EdgeStudent, self).__init__()

        # 轻量级CNN特征提取器
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),  # 固定输出长度
        )

        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 16, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # 模型大小统计
        self.model_size_mb = self._calculate_model_size()

        logger.info(f"EdgeStudent初始化: 模型大小={self.model_size_mb:.2f}MB")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入序列 [batch_size, seq_len]

        Returns:
            output: 预测结果 [batch_size, num_classes]
        """
        batch_size, seq_len = x.shape

        # 添加通道维度用于Conv1d
        x = x.unsqueeze(1)  # [batch_size, 1, seq_len]

        # CNN特征提取
        conv_features = self.conv_layers(x)  # [batch_size, 32, 16]

        # 展平
        flattened = conv_features.view(batch_size, -1)  # [batch_size, 32*16]

        # 全连接层
        output = self.fc_layers(flattened)

        return output

    def _calculate_model_size(self) -> float:
        """计算模型大小（MB）"""
        total_params = sum(p.numel() for p in self.parameters())
        model_size_mb = total_params * 4 / (1024 * 1024)  # 假设float32
        return model_size_mb

class KnowledgeDistillationTrainer:
    """
    知识蒸馏训练器
    """

    def __init__(self,
                 teacher_model: nn.Module,
                 student_model: nn.Module,
                 device: str = 'cpu',
                 temperature: float = 3.0,
                 alpha: float = 0.7):
        """
        初始化知识蒸馏训练器

        Args:
            teacher_model: 教师模型（SSA优化后的MoE）
            student_model: 学生模型（Edge模型）
            device: 设备
            temperature: 蒸馏温度
            alpha: 蒸馏损失权重
        """
        self.teacher_model = teacher_model.to(device)
        self.student_model = student_model.to(device)
        self.device = device
        self.temperature = temperature
        self.alpha = alpha

        # 冻结教师模型
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        # 学生模型优化器
        self.student_optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=0.001,
            weight_decay=0.01
        )

        logger.info(f"知识蒸馏训练器初始化: temperature={temperature}, alpha={alpha}")

    def distillation_loss(self,
                         student_logits: torch.Tensor,
                         teacher_logits: torch.Tensor,
                         true_labels: torch.Tensor) -> torch.Tensor:
        """
        计算知识蒸馏损失

        Args:
            student_logits: 学生模型输出
            teacher_logits: 教师模型输出
            true_labels: 真实标签

        Returns:
            total_loss: 总损失
        """
        # 软标签损失（教师知识）
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)

        # 硬标签损失（真实标签）
        hard_loss = F.mse_loss(student_logits, true_labels)

        # 总损失
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss

        return total_loss

    def train_step(self, batch_data: torch.Tensor, batch_labels: torch.Tensor) -> Dict[str, float]:
        """
        单步训练

        Args:
            batch_data: 批次数据
            batch_labels: 批次标签

        Returns:
            metrics: 训练指标
        """
        self.student_model.train()
        self.student_optimizer.zero_grad()

        # 前向传播
        student_output = self.student_model(batch_data)

        with torch.no_grad():
            teacher_output = self.teacher_model(batch_data)

        # 计算损失
        loss = self.distillation_loss(student_output, teacher_output, batch_labels)

        # 反向传播
        loss.backward()
        self.student_optimizer.step()

        # 计算指标
        mae = F.l1_loss(student_output, batch_labels).item()

        return {
            'loss': loss.item(),
            'mae': mae,
            'student_output_mean': student_output.mean().item(),
            'teacher_output_mean': teacher_output.mean().item()
        }

    def evaluate(self, test_data: torch.Tensor, test_labels: torch.Tensor) -> Dict[str, float]:
        """
        评估学生模型

        Args:
            test_data: 测试数据
            test_labels: 测试标签

        Returns:
            metrics: 评估指标
        """
        self.student_model.eval()

        with torch.no_grad():
            student_output = self.student_model(test_data)
            teacher_output = self.teacher_model(test_data)

            # 计算指标
            student_mae = F.l1_loss(student_output, test_labels).item()
            teacher_mae = F.l1_loss(teacher_output, test_labels).item()

            # 计算性能保持率
            performance_retention = 1 - (student_mae - teacher_mae) / teacher_mae

            return {
                'student_mae': student_mae,
                'teacher_mae': teacher_mae,
                'performance_retention': performance_retention,
                'model_size_mb': self.student_model.model_size_mb
            }

    def save_student_model(self, save_path: str):
        """保存学生模型"""
        torch.save({
            'model_state_dict': self.student_model.state_dict(),
            'model_size_mb': self.student_model.model_size_mb,
            'temperature': self.temperature,
            'alpha': self.alpha
        }, save_path)

        logger.info(f"学生模型已保存到: {save_path}")

    def export_to_onnx(self, save_path: str, input_shape: Tuple[int, int]):
        """导出为ONNX格式用于边缘部署"""
        self.student_model.eval()

        # 创建示例输入
        dummy_input = torch.randn(1, input_shape[1]).to(self.device)

        # 导出ONNX
        torch.onnx.export(
            self.student_model,
            dummy_input,
            save_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['glucose_sequence'],
            output_names=['prediction'],
            dynamic_axes={
                'glucose_sequence': {0: 'batch_size'},
                'prediction': {0: 'batch_size'}
            }
        )

        # 计算ONNX文件大小
        onnx_size_mb = os.path.getsize(save_path) / (1024 * 1024)

        logger.info(f"ONNX模型已导出: {save_path}, 大小: {onnx_size_mb:.2f}MB")

        return onnx_size_mb

class KDObjective:
    """
    知识蒸馏目标函数，用于SSA优化蒸馏参数
    """

    def __init__(self, teacher_model: nn.Module, device: str = 'cpu'):
        self.teacher_model = teacher_model
        self.device = device

    def evaluate(self, params: np.ndarray) -> float:
        """
        评估KD参数组合

        Args:
            params: [student_hidden_dim, temperature, alpha]

        Returns:
            综合损失值
        """
        try:
            student_hidden_dim = int(params[0])
            temperature = float(params[1])
            alpha = float(params[2])

            # 创建学生模型
            student_model = EdgeStudent(hidden_dim=student_hidden_dim)

            # 创建蒸馏训练器
            trainer = KnowledgeDistillationTrainer(
                teacher_model=self.teacher_model,
                student_model=student_model,
                device=self.device,
                temperature=temperature,
                alpha=alpha
            )

            # 模拟训练和评估
            # 这里返回一个基于参数合理性的模拟损失
            base_loss = 0.65

            # 参数合理性惩罚
            if student_hidden_dim < 32 or student_hidden_dim > 128:
                base_loss += 0.1

            if temperature < 1.0 or temperature > 10.0:
                base_loss += 0.05

            if alpha < 0.1 or alpha > 0.9:
                base_loss += 0.05

            # KD带来的改进
            kd_improvement = 0.02 * (1.0 / (1.0 + abs(temperature - 3.0)))
            base_loss -= kd_improvement

            # 模型大小惩罚（鼓励轻量化）
            model_size_mb = student_model.model_size_mb
            if model_size_mb > 1.5:  # 目标 < 1.5MB
                size_penalty = (model_size_mb - 1.5) * 0.1
                base_loss += size_penalty

            return max(0.1, base_loss)

        except Exception as e:
            logger.error(f"KD目标函数评估失败: {e}")
            return float('inf')
