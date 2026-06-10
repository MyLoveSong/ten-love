#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
温度缩放与校准器
通过温度参数调整模型输出的分布，提升预测的区分度和多样性
使用Platt scaling进行后处理校准
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


class TemperatureScaling(nn.Module):
    """温度缩放模块"""

    def __init__(self, initial_temperature: float = 1.0):
        super().__init__()
        # 温度参数，初始化为1.0（无缩放）
        self.temperature = nn.Parameter(torch.ones(1) * initial_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        应用温度缩放

        Args:
            logits: 模型原始输出 [batch_size, num_classes] 或 [batch_size]

        Returns:
            缩放后的logits
        """
        return logits / self.temperature

    def get_temperature(self) -> float:
        """获取当前温度值"""
        return self.temperature.item()


class AdaptiveTemperatureScaling(nn.Module):
    """自适应温度缩放（针对不同菜品类别使用不同温度）"""

    def __init__(self, num_categories: int = 5, initial_temperature: float = 1.0):
        super().__init__()
        # 为不同类别创建不同的温度参数
        self.temperatures = nn.Parameter(torch.ones(num_categories) * initial_temperature)
        self.num_categories = num_categories

    def forward(self, logits: torch.Tensor, category_ids: torch.Tensor) -> torch.Tensor:
        """
        根据类别应用不同的温度缩放

        Args:
            logits: 模型原始输出 [batch_size]
            category_ids: 类别ID [batch_size]，范围[0, num_categories-1]

        Returns:
            缩放后的logits
        """
        # 确保category_ids在有效范围内
        category_ids = torch.clamp(category_ids, 0, self.num_categories - 1).long()

        # 为每个样本选择对应的温度
        selected_temperatures = self.temperatures[category_ids]

        # 应用温度缩放
        scaled_logits = logits / selected_temperatures

        return scaled_logits

    def get_temperatures(self) -> np.ndarray:
        """获取所有温度值"""
        return self.temperatures.detach().cpu().numpy()


class PlattScaling:
    """Platt Scaling校准器（使用逻辑回归）"""

    def __init__(self):
        self.model = LogisticRegression()
        self.fitted = False

    def fit(self, predictions: np.ndarray, targets: np.ndarray):
        """
        拟合Platt scaling模型

        Args:
            predictions: 模型预测值 [n_samples]
            targets: 真实标签 [n_samples]
        """
        # 将预测值转换为logit空间
        # 避免边界值（0或1）
        predictions = np.clip(predictions, 1e-6, 1 - 1e-6)
        logits = np.log(predictions / (1 - predictions))

        # 训练逻辑回归
        self.model.fit(logits.reshape(-1, 1), targets)
        self.fitted = True

    def predict(self, predictions: np.ndarray) -> np.ndarray:
        """
        校准预测值

        Args:
            predictions: 模型预测值 [n_samples]

        Returns:
            校准后的预测值 [n_samples]
        """
        if not self.fitted:
            raise ValueError("模型尚未拟合，请先调用fit方法")

        # 转换为logit空间
        predictions = np.clip(predictions, 1e-6, 1 - 1e-6)
        logits = np.log(predictions / (1 - predictions))

        # 使用逻辑回归预测
        calibrated = self.model.predict_proba(logits.reshape(-1, 1))[:, 1]

        return calibrated


class IsotonicCalibration:
    """等渗回归校准器"""

    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds='clip')
        self.fitted = False

    def fit(self, predictions: np.ndarray, targets: np.ndarray):
        """
        拟合等渗回归模型

        Args:
            predictions: 模型预测值 [n_samples]
            targets: 真实标签 [n_samples]
        """
        self.model.fit(predictions, targets)
        self.fitted = True

    def predict(self, predictions: np.ndarray) -> np.ndarray:
        """
        校准预测值

        Args:
            predictions: 模型预测值 [n_samples]

        Returns:
            校准后的预测值 [n_samples]
        """
        if not self.fitted:
            raise ValueError("模型尚未拟合，请先调用fit方法")

        return self.model.predict(predictions)


class TemperatureCalibrationTrainer:
    """温度校准训练器"""

    def __init__(self,
                 model: nn.Module,
                 temperature_scaler: TemperatureScaling,
                 device: str = 'cpu'):
        self.model = model
        self.temperature_scaler = temperature_scaler
        self.device = device

        self.model.to(device)
        self.temperature_scaler.to(device)

    def train_temperature(self,
                         features: torch.Tensor,
                         targets: torch.Tensor,
                         optimizer: torch.optim.Optimizer,
                         num_epochs: int = 100) -> Dict[str, list]:
        """
        训练温度参数

        Args:
            features: 输入特征 [batch_size, feature_dim]
            targets: 目标值 [batch_size]
            optimizer: 优化器
            num_epochs: 训练轮数

        Returns:
            训练历史
        """
        self.model.eval()  # 冻结基础模型
        self.temperature_scaler.train()

        history = {'loss': [], 'temperature': []}

        for epoch in range(num_epochs):
            # 获取模型原始输出
            with torch.no_grad():
                logits = self.model(features)

            # 应用温度缩放
            scaled_logits = self.temperature_scaler(logits)

            # 转换为概率（如果logits是原始输出）
            # 假设logits已经是[0,1]范围内的值，直接使用
            if scaled_logits.dim() == 1:
                scaled_logits = scaled_logits.unsqueeze(1)

            # 计算损失（使用负对数似然或MSE）
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)

            loss = nn.MSELoss()(scaled_logits, targets)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 记录历史
            history['loss'].append(loss.item())
            history['temperature'].append(self.temperature_scaler.get_temperature())

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, "
                      f"Temperature: {self.temperature_scaler.get_temperature():.4f}")

        return history

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """预测"""
        self.model.eval()
        self.temperature_scaler.eval()

        with torch.no_grad():
            logits = self.model(features)
            scaled_logits = self.temperature_scaler(logits)

        return scaled_logits


class CombinedCalibrator:
    """组合校准器（温度缩放 + Platt Scaling）"""

    def __init__(self, use_temperature: bool = True, use_platt: bool = True):
        self.use_temperature = use_temperature
        self.use_platt = use_platt

        if use_temperature:
            self.temperature_scaler = TemperatureScaling()
        else:
            self.temperature_scaler = None

        if use_platt:
            self.platt_scaler = PlattScaling()
        else:
            self.platt_scaler = None

    def fit(self,
            predictions: np.ndarray,
            targets: np.ndarray,
            features: Optional[torch.Tensor] = None,
            model: Optional[nn.Module] = None):
        """
        拟合校准器

        Args:
            predictions: 模型预测值 [n_samples]
            targets: 真实标签 [n_samples]
            features: 输入特征（用于温度缩放训练）
            model: 模型（用于温度缩放训练）
        """
        if self.use_temperature and model is not None and features is not None:
            # 训练温度缩放
            optimizer = torch.optim.LBFGS([self.temperature_scaler.temperature],
                                          lr=0.01, max_iter=20)

            def closure():
                optimizer.zero_grad()
                logits = model(features)
                scaled = self.temperature_scaler(logits)
                loss = nn.MSELoss()(scaled, torch.tensor(targets, dtype=torch.float32))
                loss.backward()
                return loss

            optimizer.step(closure)

        if self.use_platt:
            # 如果使用了温度缩放，先用温度缩放处理预测值
            if self.use_temperature and model is not None and features is not None:
                with torch.no_grad():
                    logits = model(features)
                    temp_predictions = self.temperature_scaler(logits).numpy()
            else:
                temp_predictions = predictions

            # 训练Platt scaling
            self.platt_scaler.fit(temp_predictions, targets)

    def predict(self,
                predictions: np.ndarray,
                features: Optional[torch.Tensor] = None,
                model: Optional[nn.Module] = None) -> np.ndarray:
        """
        校准预测值

        Args:
            predictions: 模型预测值 [n_samples]
            features: 输入特征（用于温度缩放）
            model: 模型（用于温度缩放）

        Returns:
            校准后的预测值 [n_samples]
        """
        # 应用温度缩放
        if self.use_temperature and model is not None and features is not None:
            with torch.no_grad():
                logits = model(features)
                temp_predictions = self.temperature_scaler(logits).numpy()
        else:
            temp_predictions = predictions

        # 应用Platt scaling
        if self.use_platt:
            calibrated = self.platt_scaler.predict(temp_predictions)
        else:
            calibrated = temp_predictions

        return calibrated
