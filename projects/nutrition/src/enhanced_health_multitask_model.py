#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强版健康-口味多任务学习模型
"""

import os
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class NutritionProfile:
    """营养成分档案"""
    sodium_mg: float
    fat_g: float
    sugar_g: float
    fiber_g: float
    protein_g: float
    calories: float

    def health_score(self) -> float:
        score = 1.0
        score -= 0.4 * min(1.0, self.sodium_mg / 2000.0)
        score -= 0.3 * min(1.0, self.fat_g / 70.0)
        score -= 0.2 * min(1.0, self.sugar_g / 50.0)
        score += 0.3 * min(1.0, self.fiber_g / 25.0)
        score += 0.2 * min(1.0, self.protein_g / 50.0)
        return max(0.0, min(1.0, score))


class NutritionDatabase:
    """营养数据库"""

    def __init__(self):
        self.data = {
            "清蒸鲈鱼": NutritionProfile(120, 8, 0, 0, 20, 150),
            "蒸蛋羹": NutritionProfile(150, 10, 2, 0, 12, 120),
            "红烧肉": NutritionProfile(800, 35, 8, 0, 20, 450),
            "宫保鸡丁": NutritionProfile(850, 15, 8, 3, 25, 280),
            "麻婆豆腐": NutritionProfile(750, 14, 4, 4, 15, 220),
            "清炒时蔬": NutritionProfile(200, 5, 3, 5, 3, 80),
        }

    def get(self, name: str) -> NutritionProfile:
        return self.data.get(name, NutritionProfile(400, 10, 5, 2, 10, 200))


class HealthTasteMultiTaskModel(nn.Module):
    """健康-口味多任务模型"""

    def __init__(self, num_regions: int, num_cuisines: int, cultural_dim: int = 64,
                 nutrition_dim: int = 32, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()

        self.region_embed = nn.Embedding(num_regions, cultural_dim)
        self.cuisine_embed = nn.Embedding(num_cuisines, cultural_dim)

        self.nutrition_proj = nn.Sequential(
            nn.Linear(6, nutrition_dim), nn.ReLU(), nn.Dropout(dropout)
        )

        combined_dim = cultural_dim * 2 + cultural_dim + nutrition_dim
        self.encoder = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
        )

        self.taste_head = nn.Linear(hidden_dim, 1)
        self.health_head = nn.Linear(hidden_dim, 1)

    def forward(self, region_ids, cuisine_ids, preferences, nutrition_features):
        r_emb = self.region_embed(region_ids)
        c_emb = self.cuisine_embed(cuisine_ids)
        n_feat = self.nutrition_proj(nutrition_features)

        combined = torch.cat([r_emb, c_emb, preferences, n_feat], dim=-1)
        encoded = self.encoder(combined)

        return self.taste_head(encoded).squeeze(-1), self.health_head(encoded).squeeze(-1)


class MultiTaskLoss(nn.Module):
    """多任务损失"""

    def __init__(self, taste_weight: float = 1.0, health_weight: float = 0.5):
        super().__init__()
        self.taste_weight = taste_weight
        self.health_weight = health_weight

    def forward(self, taste_pred, taste_target, health_pred, health_target):
        taste_loss = F.mse_loss(taste_pred, taste_target)
        health_loss = F.mse_loss(health_pred, health_target)
        return self.taste_weight * taste_loss + self.health_weight * health_loss


class EnhancedHealthTasteTrainer:
    """训练器"""

    def __init__(self, num_regions: int, num_cuisines: int, output_dir: str = "outputs/health_taste"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.region_to_id = {f"region_{i}": i for i in range(num_regions)}
        self.cuisine_to_id = {f"cuisine_{i}": i for i in range(num_cuisines)}

        self.model = HealthTasteMultiTaskModel(num_regions, num_cuisines).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        self.criterion = MultiTaskLoss()

        logger.info(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")

    def train(self, train_loader, val_loader, epochs: int = 100, patience: int = 20):
        best_score = 0.0
        wait = 0
        history = []

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0

            for batch in train_loader:
                r_ids, c_ids, prefs, n_feats, taste_t, health_t = [x.to(self.device) for x in batch]

                taste_p, health_p = self.model(r_ids, c_ids, prefs, n_feats)
                loss = self.criterion(taste_p, taste_t, health_p, health_t)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                train_loss += loss.item()

            val_metrics = self._evaluate(val_loader)
            self.scheduler.step()

            score = val_metrics['overall_score']
            history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss / len(train_loader),
                'val_taste_mae': val_metrics['taste_mae'],
                'val_health_mae': val_metrics['health_mae'],
                'overall_score': score
            })

            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss/len(train_loader):.4f}, "
                       f"Taste MAE: {val_metrics['taste_mae']:.4f}, Health MAE: {val_metrics['health_mae']:.4f}")

            if score > best_score:
                best_score = score
                wait = 0
                self.save()
            else:
                wait += 1
                if wait >= patience:
                    logger.info(f"早停于epoch {epoch+1}")
                    break

        return {'best_score': best_score, 'history': history}

    def _evaluate(self, loader) -> Dict:
        self.model.eval()
        taste_preds, taste_targets = [], []
        health_preds, health_targets = [], []

        with torch.no_grad():
            for batch in loader:
                r_ids, c_ids, prefs, n_feats, taste_t, health_t = [x.to(self.device) for x in batch]
                taste_p, health_p = self.model(r_ids, c_ids, prefs, n_feats)
                taste_preds.extend(taste_p.cpu().numpy())
                health_preds.extend(health_p.cpu().numpy())
                taste_targets.extend(taste_t.cpu().numpy())
                health_targets.extend(health_t.cpu().numpy())

        taste_mae = np.mean(np.abs(np.array(taste_preds) - np.array(taste_targets)))
        health_mae = np.mean(np.abs(np.array(health_preds) - np.array(health_targets)))

        return {
            'taste_mae': taste_mae,
            'health_mae': health_mae,
            'overall_score': 1.0 - (0.8 * taste_mae + 0.2 * health_mae)
        }

    def save(self):
        torch.save({
            'model': self.model.state_dict(),
            'regions': self.region_to_id,
            'cuisines': self.cuisine_to_id,
        }, self.output_dir / "model.pt")

    def predict(self, region_id, cuisine_id, preferences, nutrition_features):
        self.model.eval()
        with torch.no_grad():
            inputs = [torch.tensor(x).to(self.device) for x in [region_id, cuisine_id, preferences, nutrition_features]]
            return self.model(*inputs)
