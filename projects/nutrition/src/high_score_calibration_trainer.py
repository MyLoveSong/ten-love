#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
高分区间校准头与损失的简化实现
用于避免缺失依赖导致训练/验证流程中断。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HighScoreCalibrationHead(nn.Module):
    """对高分健康/口味输出进行温度缩放的线性校准头。"""

    def __init__(self, input_dim: int = 1, temperature: float = 1.5, hidden_dim: int | None = None):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(float(temperature)))
        # hidden_dim 参数保持兼容，不强制使用
        self.scale = nn.Linear(input_dim, 1)

    def forward(self, fused_features: torch.Tensor, taste_logits: torch.Tensor, health_logits: torch.Tensor):
        """
        Args:
            fused_features: 融合后的特征 [B, hidden]
            taste_logits: 基础口味预测 [B]
            health_logits: 基础健康预测 [B]
        Returns:
            calibrated_taste, calibrated_health, high_score_prob
        """
        temp = torch.clamp(self.temperature, 0.5, 5.0)
        high_prob = torch.sigmoid(self.scale(fused_features) / temp).squeeze(-1)
        calibrated_taste = torch.sigmoid(taste_logits / temp)
        calibrated_health = torch.sigmoid(health_logits / temp)
        return calibrated_taste, calibrated_health, high_prob


class HighScoreCalibrationLoss(nn.Module):
    """结合 BCE 与置信度惩罚的简化损失。"""

    def __init__(self, confidence_penalty: float = 0.05):
        super().__init__()
        self.confidence_penalty = confidence_penalty
        self.bce = nn.BCELoss()

    def forward(self, taste_pred, health_pred, high_score_prob, taste_targets, health_targets, sample_weights=None):
        taste_pred = torch.clamp(taste_pred, 1e-5, 1 - 1e-5)
        health_pred = torch.clamp(health_pred, 1e-5, 1 - 1e-5)
        high_score_prob = torch.clamp(high_score_prob, 1e-5, 1 - 1e-5)

        if sample_weights is None:
            sample_weights = torch.ones_like(taste_pred)

        # 高分标签：口味与健康均达到0.75
        high_score_target = ((taste_targets >= 0.75) & (health_targets >= 0.75)).float()

        taste_loss = (self.bce(taste_pred, taste_targets) * sample_weights).mean()
        health_loss = (self.bce(health_pred, health_targets) * sample_weights).mean()
        high_prob_loss = (self.bce(high_score_prob, high_score_target) * sample_weights).mean()

        entropy = -(high_score_prob * torch.log(high_score_prob) + (1 - high_score_prob) * torch.log(1 - high_score_prob)).mean()
        return taste_loss + health_loss + high_prob_loss + self.confidence_penalty * (-entropy)


# ---------------- 数据生成占位 ----------------
from dataclasses import dataclass
import numpy as np


@dataclass
class HighScoreDataPoint:
    region: str
    cuisine: str
    preferences: np.ndarray
    nutrition_features: np.ndarray
    taste_target: float
    health_target: float
    weight: float


class HighScoreDataGenerator:
    """生成高分与中低分混合样本的占位实现。"""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.regions = ["华南", "华东", "华西"]
        self.cuisines = ["粤菜", "苏菜", "川菜"]

    def _random_pref(self):
        return self.rng.uniform(0, 1, size=6).astype(np.float32)

    def _random_nutrition(self, high: bool):
        base = self.rng.uniform(0.2, 0.6, size=6)
        if high:
            base[0] *= 0.3  # sodium lower
            base[1] *= 0.4  # fat lower
            base[3] *= 1.5  # fiber higher
        return base.astype(np.float32)

    def generate_high_score_dataset(self, high_score_samples: int = 100, medium_low_samples: int = 80):
        data_points = []
        for _ in range(high_score_samples):
            region = self.rng.choice(self.regions)
            cuisine = self.rng.choice(self.cuisines)
            prefs = self._random_pref()
            nutrition = self._random_nutrition(high=True)
            taste = float(np.clip(0.8 + 0.05 * self.rng.standard_normal(), 0.6, 0.95))
            health = float(np.clip(0.8 + 0.05 * self.rng.standard_normal(), 0.65, 0.98))
            data_points.append(HighScoreDataPoint(region, cuisine, prefs, nutrition, taste, health, weight=1.5))

        for _ in range(medium_low_samples):
            region = self.rng.choice(self.regions)
            cuisine = self.rng.choice(self.cuisines)
            prefs = self._random_pref()
            nutrition = self._random_nutrition(high=False)
            taste = float(np.clip(0.55 + 0.1 * self.rng.standard_normal(), 0.3, 0.8))
            health = float(np.clip(0.45 + 0.1 * self.rng.standard_normal(), 0.2, 0.75))
            data_points.append(HighScoreDataPoint(region, cuisine, prefs, nutrition, taste, health, weight=1.0))

        return data_points
