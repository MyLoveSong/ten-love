#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对抗训练增强器
使用FGM（Fast Gradient Method）提升模型对高分菜品的鲁棒性
针对高分菜品样本生成对抗样本，使用对抗损失增强模型泛化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


class FGM:
    """Fast Gradient Method (FGM) 对抗攻击"""

    def __init__(self, model: nn.Module, epsilon: float = 0.1):
        """
        初始化FGM

        Args:
            model: 要攻击的模型
            epsilon: 扰动强度
        """
        self.model = model
        self.epsilon = epsilon
        self.backup = {}

    def attack(self,
               embedding_name: str = 'embedding',
               is_first_attack: bool = False):
        """
        生成对抗样本

        Args:
            embedding_name: 嵌入层的名称
            is_first_attack: 是否是第一次攻击
        """
        # 遍历模型的所有参数
        for name, param in self.model.named_parameters():
            if param.requires_grad and embedding_name in name:
                if is_first_attack:
                    # 保存原始参数
                    self.backup[name] = param.data.clone()

                # 计算梯度
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    # 计算扰动
                    r_at = self.epsilon * param.grad / norm
                    # 添加扰动
                    param.data.add_(r_at)

    def restore(self, embedding_name: str = 'embedding'):
        """
        恢复原始参数

        Args:
            embedding_name: 嵌入层的名称
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and embedding_name in name:
                if name in self.backup:
                    param.data = self.backup[name]
        self.backup = {}


class PGD:
    """Projected Gradient Descent (PGD) 对抗攻击（更强的攻击方法）"""

    def __init__(self, model: nn.Module, epsilon: float = 0.1, alpha: float = 0.01,
                 num_iter: int = 7):
        """
        初始化PGD

        Args:
            model: 要攻击的模型
            epsilon: 扰动强度上限
            alpha: 每次迭代的步长
            num_iter: 迭代次数
        """
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = num_iter
        self.backup = {}

    def attack(self,
               embedding_name: str = 'embedding',
               is_first_attack: bool = False):
        """
        生成对抗样本（多步攻击）

        Args:
            embedding_name: 嵌入层的名称
            is_first_attack: 是否是第一次攻击
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and embedding_name in name:
                if is_first_attack:
                    self.backup[name] = param.data.clone()

                # 多步攻击
                for _ in range(self.num_iter):
                    norm = torch.norm(param.grad)
                    if norm != 0 and not torch.isnan(norm):
                        # 计算扰动
                        r_at = self.alpha * param.grad / norm
                        param.data.add_(r_at)

                        # 投影到epsilon球内
                        delta = param.data - self.backup[name]
                        delta_norm = torch.norm(delta)
                        if delta_norm > self.epsilon:
                            delta = delta / delta_norm * self.epsilon
                            param.data = self.backup[name] + delta

    def restore(self, embedding_name: str = 'embedding'):
        """恢复原始参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and embedding_name in name:
                if name in self.backup:
                    param.data = self.backup[name]
        self.backup = {}


class AdversarialLoss(nn.Module):
    """对抗损失函数"""

    def __init__(self,
                 base_loss: nn.Module = None,
                 adversarial_weight: float = 0.5):
        """
        初始化对抗损失

        Args:
            base_loss: 基础损失函数（如MSE、CrossEntropy等）
            adversarial_weight: 对抗损失的权重
        """
        super().__init__()
        if base_loss is None:
            self.base_loss = nn.MSELoss()
        else:
            self.base_loss = base_loss
        self.adversarial_weight = adversarial_weight

    def forward(self,
                predictions: torch.Tensor,
                targets: torch.Tensor,
                adversarial_predictions: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算对抗损失

        Args:
            predictions: 正常预测 [batch_size, ...]
            targets: 真实标签 [batch_size, ...]
            adversarial_predictions: 对抗样本的预测 [batch_size, ...]

        Returns:
            总损失和损失字典
        """
        # 基础损失
        base_loss_value = self.base_loss(predictions, targets)

        loss_dict = {
            'base_loss': base_loss_value.item()
        }

        # 对抗损失
        if adversarial_predictions is not None:
            adversarial_loss_value = self.base_loss(adversarial_predictions, targets)

            # 总损失 = 基础损失 + 对抗损失
            total_loss = base_loss_value + self.adversarial_weight * adversarial_loss_value

            loss_dict['adversarial_loss'] = adversarial_loss_value.item()
            loss_dict['total_loss'] = total_loss.item()
        else:
            total_loss = base_loss_value
            loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict


class HighHealthDishAdversarialTrainer:
    """针对高分菜品的对抗训练器"""

    def __init__(self,
                 model: nn.Module,
                 attack_method: str = 'fgm',
                 epsilon: float = 0.1,
                 adversarial_weight: float = 0.5,
                 high_health_threshold: float = 0.7,
                 device: str = 'cpu'):
        """
        初始化对抗训练器

        Args:
            model: 要训练的模型
            attack_method: 攻击方法 ('fgm' 或 'pgd')
            epsilon: 扰动强度
            adversarial_weight: 对抗损失权重
            high_health_threshold: 高分菜品阈值
            device: 设备
        """
        self.model = model
        self.device = device
        self.high_health_threshold = high_health_threshold

        # 选择攻击方法
        if attack_method.lower() == 'fgm':
            self.attacker = FGM(model, epsilon=epsilon)
        elif attack_method.lower() == 'pgd':
            self.attacker = PGD(model, epsilon=epsilon)
        else:
            raise ValueError(f"不支持的攻击方法: {attack_method}")

        # 对抗损失
        self.adversarial_loss = AdversarialLoss(adversarial_weight=adversarial_weight)

        self.model.to(device)

    def train_step(self,
                   features: torch.Tensor,
                   targets: torch.Tensor,
                   optimizer: torch.optim.Optimizer,
                   health_scores: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        执行一步对抗训练

        Args:
            features: 输入特征 [batch_size, feature_dim]
            targets: 目标值 [batch_size]
            optimizer: 优化器
            health_scores: 健康分数 [batch_size]，用于识别高分菜品

        Returns:
            损失字典
        """
        self.model.train()

        # 识别高分菜品样本
        if health_scores is not None:
            high_health_mask = health_scores >= self.high_health_threshold
            high_health_indices = torch.where(high_health_mask)[0]
        else:
            # 如果没有提供健康分数，对所有样本进行对抗训练
            high_health_indices = torch.arange(len(features))

        # 正常前向传播
        predictions = self.model(features)
        loss_normal, loss_dict = self.adversarial_loss(predictions, targets, None)

        # 对高分菜品样本进行对抗训练
        if len(high_health_indices) > 0:
            # 提取高分菜品样本
            high_health_features = features[high_health_indices]
            high_health_targets = targets[high_health_indices]

            # 第一次前向传播（计算梯度）
            high_health_predictions = self.model(high_health_features)
            loss_high = self.adversarial_loss.base_loss(high_health_predictions, high_health_targets)

            # 反向传播计算梯度
            loss_high.backward(retain_graph=True)

            # 生成对抗样本
            self.attacker.attack(is_first_attack=True)

            # 对抗样本的前向传播
            adversarial_predictions = self.model(high_health_features)

            # 计算对抗损失
            loss_adversarial, adv_loss_dict = self.adversarial_loss(
                high_health_predictions,
                high_health_targets,
                adversarial_predictions
            )

            # 恢复参数
            self.attacker.restore()

            # 总损失
            total_loss = loss_normal + loss_adversarial

            # 更新损失字典
            loss_dict.update({
                'high_health_loss': loss_high.item(),
                'adversarial_loss': adv_loss_dict.get('adversarial_loss', 0),
                'total_loss': total_loss.item(),
                'high_health_samples': len(high_health_indices)
            })
        else:
            total_loss = loss_normal
            loss_dict['high_health_samples'] = 0

        # 反向传播和优化
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        return loss_dict

    def evaluate_robustness(self,
                           features: torch.Tensor,
                           targets: torch.Tensor,
                           health_scores: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        评估模型鲁棒性

        Args:
            features: 输入特征
            targets: 目标值
            health_scores: 健康分数

        Returns:
            鲁棒性指标字典
        """
        # 正常样本性能（不需要梯度）
        self.model.eval()
        with torch.no_grad():
            predictions_normal = self.model(features)
            mae_normal = F.l1_loss(predictions_normal, targets).item()

        # 生成对抗样本（需要梯度）
        self.model.train()
        self.model.zero_grad(set_to_none=True)
        predictions_temp = self.model(features)
        loss_temp = F.mse_loss(predictions_temp, targets)
        loss_temp.backward()

        self.attacker.attack(is_first_attack=True)
        predictions_adversarial = self.model(features)
        self.attacker.restore()

        # 对抗样本误差在评估模式、禁用梯度下计算
        self.model.eval()
        with torch.no_grad():
            mae_adversarial = F.l1_loss(predictions_adversarial, targets).item()

        # 鲁棒性指标
        robustness_metrics: Dict[str, float] = {
            'mae_normal': mae_normal,
            'mae_adversarial': mae_adversarial,
            'robustness_gap': mae_adversarial - mae_normal,
            'robustness_ratio': mae_adversarial / max(mae_normal, 1e-6),
        }

        # 针对高分菜品的鲁棒性（如提供 health_scores）
        if health_scores is not None:
            high_health_mask = health_scores >= self.high_health_threshold
            if high_health_mask.any():
                high_health_features = features[high_health_mask]
                high_health_targets = targets[high_health_mask]

                # 正常高分样本误差
                self.model.eval()
                with torch.no_grad():
                    pred_normal_high = self.model(high_health_features)
                    mae_normal_high = F.l1_loss(pred_normal_high, high_health_targets).item()

                # 对抗高分样本
                self.model.train()
                self.model.zero_grad(set_to_none=True)
                pred_temp_high = self.model(high_health_features)
                loss_temp_high = F.mse_loss(pred_temp_high, high_health_targets)
                loss_temp_high.backward()

                self.attacker.attack(is_first_attack=True)
                pred_adv_high = self.model(high_health_features)
                self.attacker.restore()

                self.model.eval()
                with torch.no_grad():
                    mae_adversarial_high = F.l1_loss(pred_adv_high, high_health_targets).item()

                robustness_metrics.update({
                    'high_health_mae_normal': mae_normal_high,
                    'high_health_mae_adversarial': mae_adversarial_high,
                    'high_health_robustness_gap': mae_adversarial_high - mae_normal_high,
                    'high_health_robustness_ratio': mae_adversarial_high / max(mae_normal_high, 1e-6),
                })

        return robustness_metrics


class AdaptiveAdversarialTrainer:
    """自适应对抗训练器（根据训练进度调整对抗强度）"""

    def __init__(self,
                 model: nn.Module,
                 initial_epsilon: float = 0.05,
                 max_epsilon: float = 0.2,
                 epsilon_schedule: str = 'linear',
                 device: str = 'cpu'):
        """
        初始化自适应对抗训练器

        Args:
            model: 要训练的模型
            initial_epsilon: 初始扰动强度
            max_epsilon: 最大扰动强度
            epsilon_schedule: 扰动强度调度策略 ('linear', 'cosine', 'step')
            device: 设备
        """
        self.model = model
        self.initial_epsilon = initial_epsilon
        self.max_epsilon = max_epsilon
        self.epsilon_schedule = epsilon_schedule
        self.device = device
        self.current_epoch = 0

        # 创建FGM攻击器（epsilon会在训练过程中更新）
        self.attacker = FGM(model, epsilon=initial_epsilon)

    def update_epsilon(self, epoch: int, total_epochs: int):
        """
        根据训练进度更新扰动强度

        Args:
            epoch: 当前轮次
            total_epochs: 总轮次
        """
        self.current_epoch = epoch

        if self.epsilon_schedule == 'linear':
            # 线性增长
            progress = epoch / total_epochs
            epsilon = self.initial_epsilon + (self.max_epsilon - self.initial_epsilon) * progress
        elif self.epsilon_schedule == 'cosine':
            # 余弦退火
            progress = epoch / total_epochs
            epsilon = self.initial_epsilon + (self.max_epsilon - self.initial_epsilon) * \
                     (1 - np.cos(np.pi * progress)) / 2
        elif self.epsilon_schedule == 'step':
            # 阶梯增长
            steps = total_epochs // 3
            step = epoch // steps
            epsilon = self.initial_epsilon + (self.max_epsilon - self.initial_epsilon) * step / 3
        else:
            epsilon = self.initial_epsilon

        # 更新攻击器的epsilon
        self.attacker.epsilon = epsilon

    def train_step(self,
                   features: torch.Tensor,
                   targets: torch.Tensor,
                   optimizer: torch.optim.Optimizer,
                   health_scores: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """执行一步自适应对抗训练"""
        # 使用HighHealthDishAdversarialTrainer的逻辑
        # 但使用自适应的epsilon
        trainer = HighHealthDishAdversarialTrainer(
            self.model,
            attack_method='fgm',
            epsilon=self.attacker.epsilon,
            device=self.device
        )
        trainer.attacker = self.attacker

        return trainer.train_step(features, targets, optimizer, health_scores)
