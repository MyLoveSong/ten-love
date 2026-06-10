#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
最终增强版健康-口味模型
整合LoRA适配、知识蒸馏、多任务学习
"""

import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class LoRALayer(nn.Module):
    """LoRA适配层"""

    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.weight.requires_grad = False
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        lora = (x @ self.lora_A @ self.lora_B) * self.scaling
        return base + lora


class KnowledgeDistillationLoss(nn.Module):
    """知识蒸馏损失"""

    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, teacher_logits, targets):
        soft_target = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_loss = self.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            soft_target
        ) * (self.temperature ** 2)
        hard_loss = F.mse_loss(student_logits, targets)
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss


class MultiHeadAttention(nn.Module):
    """多头注意力"""

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        return self.norm(x + attn_out)


class FusionTransformer(nn.Module):
    """融合Transformer"""

    def __init__(self, input_dim: int = 128, num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, input_dim)
        self.layers = nn.ModuleList([
            MultiHeadAttention(input_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(input_dim, 64)

    def forward(self, x):
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x)
        return self.output_proj(x.mean(dim=1))


class FinalEnhancedHealthTasteModel(nn.Module):
    """最终增强版模型"""

    def __init__(self, num_regions: int = 10, num_cuisines: int = 20,
                 base_dim: int = 128, lora_rank: int = 8):
        super().__init__()

        self.region_embed = nn.Embedding(num_regions, base_dim)
        self.cuisine_embed = nn.Embedding(num_cuisines, base_dim)

        self.preference_proj = nn.Sequential(
            nn.Linear(6, base_dim), nn.ReLU(), nn.Dropout(0.1)
        )

        self.nutrition_proj = nn.Sequential(
            nn.Linear(6, base_dim), nn.ReLU(), nn.Dropout(0.1)
        )

        self.lora_fusion = LoRALayer(base_dim * 4, base_dim, rank=lora_rank)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=base_dim, nhead=4, dropout=0.1, batch_first=True),
            num_layers=2
        )

        self.taste_head = nn.Sequential(
            nn.Linear(base_dim, base_dim // 2), nn.ReLU(), nn.Dropout(0.1), nn.Linear(base_dim // 2, 1)
        )

        self.health_head = nn.Sequential(
            nn.Linear(base_dim, base_dim // 2), nn.ReLU(), nn.Dropout(0.1), nn.Linear(base_dim // 2, 1)
        )

    def forward(self, region_ids, cuisine_ids, preferences, nutrition_features):
        r_emb = self.region_embed(region_ids).unsqueeze(1)
        c_emb = self.cuisine_embed(cuisine_ids).unsqueeze(1)
        p_emb = self.preference_proj(preferences).unsqueeze(1)
        n_emb = self.nutrition_proj(nutrition_features).unsqueeze(1)

        combined = torch.cat([r_emb, c_emb, p_emb, n_emb], dim=1)

        fused = self.lora_fusion(combined.view(combined.size(0), -1)).unsqueeze(1)

        encoded = self.encoder(fused)

        taste = self.taste_head(encoded.squeeze(1))
        health = self.health_head(encoded.squeeze(1))

        return taste.squeeze(-1), health.squeeze(-1)


class EnhancedMultiTaskLoss(nn.Module):
    """增强多任务损失"""

    def __init__(self, taste_weight: float = 1.0, health_weight: float = 0.5):
        super().__init__()
        self.taste_weight = taste_weight
        self.health_weight = health_weight

    def forward(self, taste_pred, taste_target, health_pred, health_target):
        return self.taste_weight * F.mse_loss(taste_pred, taste_target) + \
               self.health_weight * F.mse_loss(health_pred, health_target)


def train_model(train_loader, val_loader, num_epochs: int = 100,
                device: str = "cuda" if torch.cuda.is_available() else "cpu") -> Dict:
    """训练模型"""
    model = FinalEnhancedHealthTasteModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = EnhancedMultiTaskLoss()

    best_score = 0.0
    history = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            r_ids, c_ids, prefs, n_feats, taste_t, health_t = [x.to(device) for x in batch]

            taste_p, health_p = model(r_ids, c_ids, prefs, n_feats)
            loss = criterion(taste_p, taste_t, health_p, health_t)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        score = val_metrics['overall_score']
        history.append({
            'epoch': epoch + 1,
            'loss': train_loss / len(train_loader),
            **val_metrics
        })

        logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss/len(train_loader):.4f}, "
                   f"Taste: {val_metrics['taste_mae']:.4f}, Health: {val_metrics['health_mae']:.4f}")

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), "outputs/best_model.pt")

    return {'best_score': best_score, 'history': history}


def evaluate(model, loader, device: str) -> Dict:
    """评估模型"""
    model.eval()
    taste_preds, health_preds = [], []
    taste_targets, health_targets = [], []

    with torch.no_grad():
        for batch in loader:
            r_ids, c_ids, prefs, n_feats, taste_t, health_t = [x.to(device) for x in batch]
            taste_p, health_p = model(r_ids, c_ids, prefs, n_feats)

            taste_preds.extend(taste_p.cpu().numpy())
            health_preds.extend(health_p.cpu().numpy())
            taste_targets.extend(taste_t.cpu().numpy())
            health_targets.extend(health_t.cpu().numpy())

    taste_mae = np.mean(np.abs(np.array(taste_preds) - np.array(taste_targets)))
    health_mae = np.mean(np.abs(np.array(health_preds) - np.array(health_targets)))

    return {
        'taste_mae': taste_mae,
        'health_mae': health_mae,
        'overall_score': 1.0 - (0.7 * taste_mae + 0.3 * health_mae)
    }
