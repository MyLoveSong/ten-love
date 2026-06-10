#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集成模型训练脚本
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NutritionDataset(Dataset):
    """营养数据集"""

    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        nutrition = s.get('nutrition', {})

        return {
            'nutrition': torch.tensor([
                nutrition.get('calories', 0), nutrition.get('protein', 0),
                nutrition.get('carbs', 0), nutrition.get('fat', 0),
                nutrition.get('fiber', 0), nutrition.get('sodium', 0)
            ], dtype=torch.float32),
            'region_id': torch.tensor(s.get('region_id', 0), dtype=torch.long),
            'cuisine_id': torch.tensor(s.get('cuisine_id', 0), dtype=torch.long),
            'preferences': torch.tensor(s.get('preferences', [0.5]*6), dtype=torch.float32),
            'taste_target': torch.tensor(s.get('target_taste', 0.5), dtype=torch.float32),
            'health_target': torch.tensor(s.get('target_health', 0.5), dtype=torch.float32),
        }


class IntegratedModel(nn.Module):
    """集成模型"""

    def __init__(self, num_regions=10, num_cuisines=20, hidden_dim=128):
        super().__init__()
        self.region_embed = nn.Embedding(num_regions, 64)
        self.cuisine_embed = nn.Embedding(num_cuisines, 64)
        self.nutrition_proj = nn.Sequential(nn.Linear(6, hidden_dim), nn.ReLU(), nn.Dropout(0.1))

        self.encoder = nn.Sequential(
            nn.Linear(64*2+6+6, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )

        self.taste_head = nn.Linear(hidden_dim, 1)
        self.health_head = nn.Linear(hidden_dim, 1)

    def forward(self, region_ids, cuisine_ids, nutrition, preferences):
        r_emb = self.region_embed(region_ids)
        c_emb = self.cuisine_embed(cuisine_ids)
        n_proj = self.nutrition_proj(nutrition)

        combined = torch.cat([r_emb, c_emb, n_proj, preferences], dim=-1)
        encoded = self.encoder(combined)

        return self.taste_head(encoded).squeeze(-1), self.health_head(encoded).squeeze(-1)


def train_model(samples: List[Dict], epochs: int = 100, batch_size: int = 64,
                output_dir: str = "outputs/integrated") -> Dict:
    """训练模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = NutritionDataset(samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = IntegratedModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    history = []
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in loader:
            n = batch['nutrition'].to(device)
            r = batch['region_id'].to(device)
            c = batch['cuisine_id'].to(device)
            p = batch['preferences'].to(device)
            t_taste = batch['taste_target'].to(device)
            t_health = batch['health_target'].to(device)

            taste_p, health_p = model(r, c, n, p)
            loss = criterion(taste_p, t_taste) + 0.5 * criterion(health_p, t_health)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(loader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), Path(output_dir) / "best_model.pt")

        history.append({'epoch': epoch+1, 'loss': avg_loss})

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    return {'best_loss': best_loss, 'history': history}


def load_data(json_path: str) -> List[Dict]:
    """加载数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('samples', [])