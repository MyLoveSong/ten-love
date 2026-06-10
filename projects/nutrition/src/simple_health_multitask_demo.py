#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化版健康-口味多任务演示模型
用于演示与验证脚本的轻量依赖，避免缺失模块导致流程中断。
"""

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn


@dataclass
class NutritionProfile:
    sodium_mg: float
    fat_g: float
    sugar_g: float
    fiber_g: float
    protein_g: float
    calories: float

    def calculate_health_score(self) -> float:
        score = 1.0
        score -= 0.4 * min(1.0, self.sodium_mg / 2000.0)
        score -= 0.3 * min(1.0, self.fat_g / 70.0)
        score -= 0.2 * min(1.0, self.sugar_g / 50.0)
        score += 0.3 * min(1.0, self.fiber_g / 25.0)
        score += 0.2 * min(1.0, self.protein_g / 50.0)
        return float(np.clip(score, 0.0, 1.0))


class SimpleNutritionDatabase:
    """简化营养数据库，覆盖关键菜品并提供均值回退。"""

    def __init__(self):
        self.data = {
            "清蒸鲈鱼": NutritionProfile(120, 8, 0, 0, 20, 150),
            "蒸蛋羹": NutritionProfile(200, 6, 1, 0, 12, 120),
            "清炒时蔬": NutritionProfile(300, 5, 3, 8, 4, 80),
            "红烧肉": NutritionProfile(1200, 25, 12, 0, 20, 400),
            "宫保鸡丁": NutritionProfile(800, 15, 8, 2, 18, 250),
            "麻婆豆腐": NutritionProfile(900, 12, 5, 3, 15, 200),
        }

    def get_nutrition_profile(self, dish_name: str) -> NutritionProfile:
        return self.data.get(
            dish_name,
            NutritionProfile(600, 12, 8, 2, 12, 200),
        )


class SimpleHealthTasteModel(nn.Module):
    """
    轻量前馈网络：输入营养特征 -> 预测健康/口味分数。
    仅用于演示和可解释性占位，非生产强度。
    """

    def __init__(self, input_dim: int = 16):
        super().__init__()
        hidden = 64
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
        )
        self.taste_head = nn.Sequential(nn.Linear(hidden // 2, 1), nn.Sigmoid())
        self.health_head = nn.Sequential(nn.Linear(hidden // 2, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.backbone(x)
        taste = self.taste_head(feats)
        health = self.health_head(feats)
        return {"taste": taste.squeeze(-1), "health": health.squeeze(-1)}
