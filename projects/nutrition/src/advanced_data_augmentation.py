#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级数据增强模块
实现Mixup/CutMix数据增强
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class NutritionSample:
    """营养样本数据结构"""
    calories: float
    protein: float
    carbs: float
    fat: float
    fiber: float
    sodium: float
    sugar: float
    health_score: float
    taste_score: float


class MixupAugmentation:
    """Mixup数据增强"""

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def augment(self, sample1: Dict, sample2: Dict,
                fields: List[str] = None) -> Dict:
        """对两个样本进行Mixup增强"""
        if fields is None:
            fields = ['calories', 'protein', 'carbs', 'fat', 'fiber', 'sodium', 'sugar']

        lam = np.random.beta(self.alpha, self.alpha)
        new_sample = {}

        for field in fields:
            if field in sample1 and field in sample2:
                new_sample[field] = lam * float(sample1[field]) + (1 - lam) * float(sample2[field])

        if 'health_score' in sample1 and 'health_score' in sample2:
            new_sample['health_score'] = lam * sample1['health_score'] + (1 - lam) * sample2['health_score']

        if 'taste_score' in sample1 and 'taste_score' in sample2:
            new_sample['taste_score'] = lam * sample1['taste_score'] + (1 - lam) * sample2['taste_score']

        return new_sample

    def mixup_batch(self, X: torch.Tensor, y: torch.Tensor,
                    alpha: float = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """批量Mixup"""
        if alpha is None:
            alpha = self.alpha

        batch_size = X.size(0)
        lam = torch.from_numpy(np.random.beta(alpha, alpha, batch_size)).float().unsqueeze(1)
        lam = lam.to(X.device)

        index = torch.randperm(batch_size).to(X.device)

        mixed_X = lam * X + (1 - lam) * X[index]
        y_a, y_b = y, y[index]

        return mixed_X, y_a, y_b, lam


class CutMixAugmentation:
    """CutMix数据增强"""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def cutmix(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """CutMix增强"""
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = X.size(0)
        index = torch.randperm(batch_size).to(X.device)

        H, W = X.size(1), X.size(2)
        cut_rat = np.sqrt(1. - lam)
        cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        X_mixed = X.clone()
        X_mixed[:, bby1:bby2, bbx1:bbx2] = X[index, bby1:bby2, bbx1:bbx2]

        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1)) / (W * H)

        return X_mixed, y, y[index], lam


class VAEDataAugmentation:
    """基于VAE的增强"""

    def __init__(self, input_dim: int, latent_dim: int = 16, hidden_dim: int = 64):
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        self.fc_mean = torch.nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = torch.nn.Linear(hidden_dim, latent_dim)
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, input_dim),
        )

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mean, logvar)
        return self.decoder(z), mean, logvar

    def generate(self, num_samples: int, device: str = "cpu") -> torch.Tensor:
        """生成新样本"""
        z = torch.randn(num_samples, self.fc_mean.out_features).to(device)
        return self.decoder(z)


class DataAugmentationPipeline:
    """数据增强管道"""

    def __init__(self, use_mixup: bool = True, use_cutmix: bool = False,
                 mixup_alpha: float = 0.2, cutmix_alpha: float = 1.0):
        self.mixup = MixupAugmentation(mixup_alpha) if use_mixup else None
        self.cutmix = CutMixAugmentation(cutmix_alpha) if use_cutmix else None

    def augment(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """应用增强"""
        if self.mixup is not None:
            return self.mixup.mixup_batch(X, y)
        return X, y, y, torch.ones(X.size(0))
