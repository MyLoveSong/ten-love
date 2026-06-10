#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化版营养驱动多任务模型
用于演示脚本的占位实现，避免缺失模块导致流程中断。
"""

from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn as nn


@dataclass
class NutritionDrivenSample:
    features: np.ndarray
    taste: float
    health: float


class NutritionDrivenDataGenerator:
    """基于营养特征的轻量数据生成器。"""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate(self, num_samples: int = 128) -> List[NutritionDrivenSample]:
        samples = []
        for _ in range(num_samples):
            features = self.rng.uniform(0, 1, size=16)
            taste = float(np.clip(0.5 + 0.2 * (features[4] - features[0]), 0, 1))
            health = float(np.clip(0.6 + 0.3 * (features[3] - features[1]), 0, 1))
            samples.append(NutritionDrivenSample(features=features, taste=taste, health=health))
        return samples


class NutritionDrivenMultiTaskModel(nn.Module):
    """简单的营养驱动 MTL 模型，用于演示预测接口。"""

    def __init__(self, input_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.taste_head = nn.Sequential(nn.Linear(hidden_dim // 2, 1), nn.Sigmoid())
        self.health_head = nn.Sequential(nn.Linear(hidden_dim // 2, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.backbone(x)
        return {
            "taste": self.taste_head(feats).squeeze(-1),
            "health": self.health_head(feats).squeeze(-1),
        }
