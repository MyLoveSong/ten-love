#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分层校准训练器
针对不同健康分数段采用不同的校准策略
- 低分段（0-0.4）：保持现有校准
- 中分段（0.4-0.7）：适度校准
- 高分段（0.7-1.0）：强化校准，使用更大的权重和更精细的局部校正
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class HierarchicalCalibrationHead(nn.Module):
    """分层校准头"""

    def __init__(self, input_dim: int, hidden_dims: list = [64, 32, 16]):
        super().__init__()

        # 为不同分数段创建不同的校准网络
        self.low_calibrator = self._build_calibrator(input_dim, hidden_dims)  # 0-0.4
        self.mid_calibrator = self._build_calibrator(input_dim, hidden_dims)  # 0.4-0.7
        self.high_calibrator = self._build_calibrator(input_dim, hidden_dims)  # 0.7-1.0

        # 分数段边界
        self.low_bound = 0.4
        self.mid_bound = 0.7

    def _build_calibrator(self, input_dim: int, hidden_dims: list) -> nn.Module:
        """构建校准网络"""
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim

        # 输出层：输出校准偏移量
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Tanh())  # 限制在[-1, 1]范围内

        return nn.Sequential(*layers)

    def forward(self, features: torch.Tensor, base_score: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            features: 输入特征 [batch_size, input_dim]
            base_score: 基础预测分数 [batch_size, 1] 或 [batch_size]

        Returns:
            校准后的分数 [batch_size, 1]
        """
        if base_score.dim() == 1:
            base_score = base_score.unsqueeze(1)

        batch_size = base_score.size(0)
        calibrated_scores = torch.zeros_like(base_score)

        # 根据基础分数选择不同的校准器
        low_mask = base_score < self.low_bound
        mid_mask = (base_score >= self.low_bound) & (base_score < self.mid_bound)
        high_mask = base_score >= self.mid_bound

        # 低分段校准（轻微调整）
        if low_mask.any():
            low_calibration = self.low_calibrator(features[low_mask.squeeze()])
            calibrated_scores[low_mask] = base_score[low_mask] + 0.1 * low_calibration

        # 中分段校准（适度调整）
        if mid_mask.any():
            mid_calibration = self.mid_calibrator(features[mid_mask.squeeze()])
            calibrated_scores[mid_mask] = base_score[mid_mask] + 0.2 * mid_calibration

        # 高分段校准（强化调整）
        if high_mask.any():
            high_calibration = self.high_calibrator(features[high_mask.squeeze()])
            # 高分段使用更大的校准幅度，但限制在合理范围内
            calibration_scale = 0.3 + 0.2 * (base_score[high_mask] - self.mid_bound) / (1.0 - self.mid_bound)
            calibrated_scores[high_mask] = base_score[high_mask] + calibration_scale * high_calibration

        # 确保分数在[0, 1]范围内
        calibrated_scores = torch.clamp(calibrated_scores, 0.0, 1.0)

        return calibrated_scores


class HierarchicalCalibrationLoss(nn.Module):
    """分层校准损失函数"""

    def __init__(self,
                 low_weight: float = 1.0,
                 mid_weight: float = 1.5,
                 high_weight: float = 3.0,
                 mse_weight: float = 1.0,
                 smooth_weight: float = 0.1):
        super().__init__()
        self.low_weight = low_weight
        self.mid_weight = mid_weight
        self.high_weight = high_weight
        self.mse_weight = mse_weight
        self.smooth_weight = smooth_weight

        self.mse_loss = nn.MSELoss()
        self.smooth_l1_loss = nn.SmoothL1Loss()

    def forward(self,
                predicted: torch.Tensor,
                target: torch.Tensor,
                base_score: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        计算分层校准损失

        Args:
            predicted: 预测分数 [batch_size, 1] 或 [batch_size]
            target: 目标分数 [batch_size, 1] 或 [batch_size]
            base_score: 基础分数（用于确定分数段）[batch_size, 1] 或 [batch_size]

        Returns:
            损失字典
        """
        if predicted.dim() == 1:
            predicted = predicted.unsqueeze(1)
        if target.dim() == 1:
            target = target.unsqueeze(1)
        if base_score is None:
            base_score = predicted
        if base_score.dim() == 1:
            base_score = base_score.unsqueeze(1)

        # 确定分数段
        low_mask = base_score < 0.4
        mid_mask = (base_score >= 0.4) & (base_score < 0.7)
        high_mask = base_score >= 0.7

        total_loss = 0.0
        loss_dict = {}

        # 低分段损失
        if low_mask.any():
            low_loss = self.smooth_l1_loss(predicted[low_mask], target[low_mask])
            weighted_low_loss = self.low_weight * low_loss
            total_loss += weighted_low_loss
            loss_dict['low_segment_loss'] = low_loss.item()

        # 中分段损失
        if mid_mask.any():
            mid_loss = self.smooth_l1_loss(predicted[mid_mask], target[mid_mask])
            weighted_mid_loss = self.mid_weight * mid_loss
            total_loss += weighted_mid_loss
            loss_dict['mid_segment_loss'] = mid_loss.item()

        # 高分段损失（最重要）
        if high_mask.any():
            high_loss = self.smooth_l1_loss(predicted[high_mask], target[high_mask])
            weighted_high_loss = self.high_weight * high_loss
            total_loss += weighted_high_loss
            loss_dict['high_segment_loss'] = high_loss.item()

        # 整体MSE损失
        mse_loss = self.mse_loss(predicted, target)
        total_loss += self.mse_weight * mse_loss
        loss_dict['mse_loss'] = mse_loss.item()

        # 平滑性损失（鼓励相邻分数段的平滑过渡）
        if base_score is not None and base_score.size(0) > 1:
            # 计算预测分数的平滑性
            sorted_indices = torch.argsort(base_score.squeeze())
            sorted_pred = predicted[sorted_indices]
            smooth_loss = torch.mean((sorted_pred[1:] - sorted_pred[:-1]) ** 2)
            total_loss += self.smooth_weight * smooth_loss
            loss_dict['smooth_loss'] = smooth_loss.item()

        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict


class HierarchicalCalibrationTrainer:
    """分层校准训练器"""

    def __init__(self,
                 model: nn.Module,
                 calibrator: HierarchicalCalibrationHead,
                 device: str = 'cpu'):
        self.model = model
        self.calibrator = calibrator
        self.device = device

        self.model.to(device)
        self.calibrator.to(device)

    def train_step(self,
                   features: torch.Tensor,
                   targets: torch.Tensor,
                   optimizer: torch.optim.Optimizer,
                   loss_fn: HierarchicalCalibrationLoss) -> Dict[str, float]:
        """训练一步"""
        self.model.eval()  # 冻结基础模型
        self.calibrator.train()

        # 获取基础预测
        with torch.no_grad():
            base_score = self.model(features)

        # 校准预测
        calibrated_score = self.calibrator(features, base_score)

        # 计算损失
        loss, loss_dict = loss_fn(calibrated_score, targets, base_score)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss_dict

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """预测"""
        self.model.eval()
        self.calibrator.eval()

        with torch.no_grad():
            base_score = self.model(features)
            calibrated_score = self.calibrator(features, base_score)

        return calibrated_score

    def get_segment_statistics(self,
                               features: torch.Tensor,
                               targets: torch.Tensor) -> Dict[str, Dict]:
        """获取各分数段的统计信息"""
        self.model.eval()
        self.calibrator.eval()

        with torch.no_grad():
            base_score = self.model(features)
            calibrated_score = self.calibrator(features, base_score)

        if base_score.dim() == 1:
            base_score = base_score.unsqueeze(1)
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)

        # 确定分数段
        low_mask = base_score < 0.4
        mid_mask = (base_score >= 0.4) & (base_score < 0.7)
        high_mask = base_score >= 0.7

        stats = {}

        for mask, name in [(low_mask, 'low'), (mid_mask, 'mid'), (high_mask, 'high')]:
            if mask.any():
                pred_seg = calibrated_score[mask]
                target_seg = targets[mask]
                base_seg = base_score[mask]

                mae = torch.mean(torch.abs(pred_seg - target_seg)).item()
                mse = torch.mean((pred_seg - target_seg) ** 2).item()
                base_mae = torch.mean(torch.abs(base_seg - target_seg)).item()

                stats[name] = {
                    'count': mask.sum().item(),
                    'mae': mae,
                    'mse': mse,
                    'base_mae': base_mae,
                    'improvement': base_mae - mae,
                    'improvement_rate': (base_mae - mae) / max(base_mae, 1e-6)
                }

        return stats
