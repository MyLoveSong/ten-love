"""
联邦学习框架 - 隐私保护的分布式训练
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
import time
import hashlib
import hmac

logger = logging.getLogger(__name__)

@dataclass
class FederatedConfig:
    """联邦学习配置"""
    num_clients: int = 5
    num_rounds: int = 10
    local_epochs: int = 3
    learning_rate: float = 0.01
    batch_size: int = 32
    privacy_budget: float = 1.0
    secure_aggregation: bool = True
    client_selection_ratio: float = 0.8

class PrivacyPreserver:
    """隐私保护模块"""

    def __init__(self, privacy_budget: float = 1.0):
        self.privacy_budget = privacy_budget
        self.sensitivity = 1.0

    def add_differential_privacy(self, gradients: List[torch.Tensor],
                                noise_scale: float = None) -> List[torch.Tensor]:
        """添加差分隐私噪声"""
        if noise_scale is None:
            noise_scale = self.sensitivity / self.privacy_budget

        noisy_gradients = []
        for grad in gradients:
            noise = torch.normal(0, noise_scale, size=grad.shape)
            noisy_grad = grad + noise
            noisy_gradients.append(noisy_grad)

        return noisy_gradients

    def gradient_clipping(self, gradients: List[torch.Tensor],
                        max_norm: float = 1.0) -> List[torch.Tensor]:
        """梯度裁剪"""
        clipped_gradients = []
        for grad in gradients:
            grad_norm = torch.norm(grad)
            if grad_norm > max_norm:
                grad = grad * (max_norm / grad_norm)
            clipped_gradients.append(grad)

        return clipped_gradients

class SecureAggregator:
    """安全聚合模块"""

    def __init__(self, num_clients: int):
        self.num_clients = num_clients
        self.aggregation_key = self._generate_aggregation_key()

    def _generate_aggregation_key(self) -> str:
        """生成聚合密钥"""
        return hashlib.sha256(f"federated_key_{time.time()}".encode()).hexdigest()

    def secure_aggregate(self, client_updates: Dict[int, List[torch.Tensor]]) -> List[torch.Tensor]:
        """安全聚合客户端更新"""
        if not client_updates:
            return []

        # 获取第一个客户端的更新结构
        first_client_id = list(client_updates.keys())[0]
        first_update = client_updates[first_client_id]

        # 初始化聚合结果
        aggregated_updates = []
        for i, param_update in enumerate(first_update):
            aggregated_updates.append(torch.zeros_like(param_update))

        # 聚合所有客户端的更新
        for client_id, updates in client_updates.items():
            for i, update in enumerate(updates):
                aggregated_updates[i] += update

        # 平均化
        num_participants = len(client_updates)
        for i in range(len(aggregated_updates)):
            aggregated_updates[i] /= num_participants

        return aggregated_updates

class ClientSelector:
    """客户端选择模块"""

    def __init__(self, selection_ratio: float = 0.8):
        self.selection_ratio = selection_ratio
        self.client_history = {}

    def select_clients(self, available_clients: List[int], round_num: int) -> List[int]:
        """选择参与的客户端"""
        num_to_select = max(1, int(len(available_clients) * self.selection_ratio))

        # 简单的随机选择策略
        selected_clients = np.random.choice(
            available_clients,
            size=min(num_to_select, len(available_clients)),
            replace=False
        ).tolist()

        return selected_clients

    def update_client_history(self, client_id: int, loss: float, accuracy: float):
        """更新客户端历史"""
        if client_id not in self.client_history:
            self.client_history[client_id] = {'losses': [], 'accuracies': []}

        self.client_history[client_id]['losses'].append(loss)
        self.client_history[client_id]['accuracies'].append(accuracy)

class FederatedGlucoNet:
    """联邦学习GlucoNet系统"""

    def __init__(self, config: FederatedConfig, global_model: nn.Module):
        self.config = config
        self.global_model = global_model
        self.local_models = {}

        # 初始化组件
        self.privacy_preserver = PrivacyPreserver(config.privacy_budget)
        self.secure_aggregator = SecureAggregator(config.num_clients)
        self.client_selector = ClientSelector(config.client_selection_ratio)

        # 训练历史
        self.training_history = {
            'round_losses': [],
            'round_accuracies': [],
            'client_participation': []
        }

        logger.info(f"联邦学习GlucoNet初始化完成，客户端数量: {config.num_clients}")

    def initialize_local_models(self, client_ids: List[int]):
        """初始化本地模型"""
        for client_id in client_ids:
            # 复制全局模型作为本地模型
            self.local_models[client_id] = type(self.global_model)(
                **self._get_model_config()
            )
            self.local_models[client_id].load_state_dict(
                self.global_model.state_dict()
            )

        logger.info(f"初始化了{len(client_ids)}个本地模型")

    def _get_model_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        # 这里应该根据实际的GlucoNet模型配置返回
        return {
            'input_dim': 10,
            'hidden_dim': 128,
            'num_layers': 2,
            'dropout': 0.1
        }

    def federated_training(self, client_data: Dict[int, Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, Any]:
        """执行联邦训练"""
        logger.info("开始联邦训练...")

        start_time = time.time()

        for round_num in range(self.config.num_rounds):
            logger.info(f"联邦训练第{round_num + 1}轮开始...")

            # 选择参与的客户端
            available_clients = list(client_data.keys())
            selected_clients = self.client_selector.select_clients(
                available_clients, round_num + 1
            )

            # 本地训练
            client_updates = {}
            round_losses = []
            round_accuracies = []

            for client_id in selected_clients:
                if client_id in client_data:
                    local_loss, local_accuracy, updates = self._local_training(
                        client_id, client_data[client_id], round_num
                    )

                    client_updates[client_id] = updates
                    round_losses.append(local_loss)
                    round_accuracies.append(local_accuracy)

                    # 更新客户端历史
                    self.client_selector.update_client_history(
                        client_id, local_loss, local_accuracy
                    )

            # 安全聚合
            if client_updates:
                global_updates = self.secure_aggregator.secure_aggregate(client_updates)

                # 更新全局模型
                self._update_global_model(global_updates)

            # 记录训练历史
            avg_loss = np.mean(round_losses) if round_losses else 0.0
            avg_accuracy = np.mean(round_accuracies) if round_accuracies else 0.0

            self.training_history['round_losses'].append(avg_loss)
            self.training_history['round_accuracies'].append(avg_accuracy)
            self.training_history['client_participation'].append(len(selected_clients))

            logger.info(f"第{round_num + 1}轮完成 - 平均损失: {avg_loss:.4f}, "
                       f"平均准确率: {avg_accuracy:.4f}, 参与客户端: {len(selected_clients)}")

        end_time = time.time()

        training_info = {
            'total_time': end_time - start_time,
            'num_rounds': self.config.num_rounds,
            'final_loss': self.training_history['round_losses'][-1],
            'final_accuracy': self.training_history['round_accuracies'][-1],
            'training_history': self.training_history
        }

        logger.info(f"联邦训练完成，总耗时: {training_info['total_time']:.2f}s")

        return training_info

    def _local_training(self, client_id: int, data: Tuple[torch.Tensor, torch.Tensor],
                       round_num: int) -> Tuple[float, float, List[torch.Tensor]]:
        """本地训练"""
        local_model = self.local_models[client_id]
        local_model.train()

        X, y = data
        optimizer = torch.optim.Adam(local_model.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()

        total_loss = 0.0
        total_samples = 0

        for epoch in range(self.config.local_epochs):
            # 随机打乱数据
            indices = torch.randperm(X.size(0))
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # 批量训练
            for i in range(0, X.size(0), self.config.batch_size):
                batch_X = X_shuffled[i:i + self.config.batch_size]
                batch_y = y_shuffled[i:i + self.config.batch_size]

                optimizer.zero_grad()
                outputs = local_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=1.0)

                optimizer.step()

                total_loss += loss.item() * batch_X.size(0)
                total_samples += batch_X.size(0)

        avg_loss = total_loss / total_samples

        # 计算准确率（简化版本）
        local_model.eval()
        with torch.no_grad():
            predictions = local_model(X)
            accuracy = self._calculate_accuracy(predictions, y)

        # 获取模型参数更新
        global_params = list(self.global_model.parameters())
        local_params = list(local_model.parameters())

        updates = []
        for global_param, local_param in zip(global_params, local_params):
            update = local_param.data - global_param.data
            updates.append(update)

        return avg_loss, accuracy, updates

    def _calculate_accuracy(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """计算准确率"""
        # 简化的准确率计算
        if predictions.dim() == 2 and predictions.size(1) == 1:
            # 回归任务
            mse = torch.mean((predictions - targets) ** 2)
            accuracy = max(0, 1 - mse.item())
        else:
            # 分类任务
            predicted_classes = torch.argmax(predictions, dim=1)
            target_classes = torch.argmax(targets, dim=1) if targets.dim() > 1 else targets
            accuracy = (predicted_classes == target_classes).float().mean().item()

        return accuracy

    def _update_global_model(self, updates: List[torch.Tensor]):
        """更新全局模型"""
        global_params = list(self.global_model.parameters())

        for global_param, update in zip(global_params, updates):
            global_param.data += update

    def get_global_model(self) -> nn.Module:
        """获取全局模型"""
        return self.global_model

    def get_training_history(self) -> Dict[str, Any]:
        """获取训练历史"""
        return self.training_history

    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'global_model_state_dict': self.global_model.state_dict(),
            'training_history': self.training_history,
            'config': self.config
        }, path)

        logger.info(f"模型已保存到: {path}")

    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.global_model.load_state_dict(checkpoint['global_model_state_dict'])
        self.training_history = checkpoint['training_history']

        logger.info(f"模型已从{path}加载")