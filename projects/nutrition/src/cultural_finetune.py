#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段一：文化适配微调（LoRA）
使用LoRA进行轻量化微调，保留核心训练功能
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class LoRALayer(nn.Module):
    """LoRA适配层"""

    def __init__(self, in_features: int, out_features: int, rank: int = 4, alpha: float = 8.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / max(rank, 1)
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.lora_B @ self.lora_A) * self.scaling


class LoRALinear(nn.Module):
    """LoRA线性层"""

    def __init__(self, linear: nn.Linear, rank: int = 4, alpha: float = 8.0):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank=rank, alpha=alpha)
        self.enable_lora = True

    @property
    def weight(self): return self.linear.weight
    @property
    def bias(self): return self.linear.bias
    @property
    def in_features(self): return self.linear.in_features
    @property
    def out_features(self): return self.linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + (self.lora(x) if self.enable_lora else 0)


def replace_linear_with_lora(module: nn.Module, rank: int, alpha: float) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha))
        else:
            replace_linear_with_lora(child, rank, alpha)


def collect_lora_params(module: nn.Module):
    for child in module.modules():
        if isinstance(child, LoRALinear):
            yield child.lora.lora_A
            yield child.lora.lora_B


def export_lora_weights(module: nn.Module, save_path: Path) -> None:
    state = {}
    for name, child in module.named_modules():
        if isinstance(child, LoRALinear):
            state[name] = {
                'lora_A': child.lora.lora_A.data.cpu(),
                'lora_B': child.lora.lora_B.data.cpu(),
                'alpha': child.lora.alpha,
                'rank': child.lora.rank,
            }
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'lora': state}, save_path)


class CulturalStage1Trainer:
    """阶段一文化适配训练器"""

    def __init__(self, input_dim: int = 256, hidden_dim: int = 128, lora_rank: int = 16,
                 lora_alpha: float = 32.0, device: Optional[str] = None):
        from backend.models.cultural_adaptation import KnowledgeDistillationNetwork

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = KnowledgeDistillationNetwork(input_dim=input_dim, hidden_dim=hidden_dim).to(self.device)
        replace_linear_with_lora(self.model, rank=lora_rank, alpha=lora_alpha)

        self.feature_projector = nn.Sequential(
            nn.Linear(20, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, input_dim)
        ).to(self.device)

        self.output_corrector = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1), nn.Sigmoid()
        ).to(self.device)

        self.target_mean = None
        self.target_std = None
        self.training_history = {'train_loss': [], 'val_loss': [], 'train_r2': [], 'val_r2': []}
        self.calibrator = None
        self.model.to(self.device)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 50,
            lr: float = 1e-4, grad_clip: float = 1.0, patience: int = 30,
            output_dir: Optional[Path] = None) -> Dict[str, Any]:
        lora_params = list(collect_lora_params(self.model))
        param_groups = [
            {'params': lora_params + list(self.feature_projector.parameters()), 'lr': lr},
            {'params': list(self.output_corrector.parameters()), 'lr': lr * 5}
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
        criterion = nn.SmoothL1Loss(beta=0.05)

        best_val_loss, best_epoch, wait = float('inf'), 0, 0
        history = []

        for epoch in range(1, epochs + 1):
            self.model.train()
            self.feature_projector.train()
            train_loss, train_steps = 0.0, 0
            train_preds, train_targets = [], []

            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                x_proj = self.feature_projector(xb)
                out = self.model(x_proj)
                preds = self.output_corrector(out['distilled_output'].mean(dim=1, keepdim=True))
                preds = preds.expand_as(out['distilled_output']).clamp(0, 1)
                loss = criterion(preds, yb)

                optimizer.zero_grad()
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(lora_params + list(self.feature_projector.parameters()) +
                        list(self.output_corrector.parameters()), grad_clip)
                optimizer.step()

                train_loss += loss.item()
                train_steps += 1
                with torch.no_grad():
                    train_preds.extend(preds.mean(dim=1).cpu().numpy())
                    train_targets.extend(yb.mean(dim=1).cpu().numpy())

            self.model.eval()
            val_loss, val_steps = 0.0, 0
            val_preds, val_targets = [], []

            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    x_proj = self.feature_projector(xb)
                    out = self.model(x_proj)
                    preds = self.output_corrector(out['distilled_output'].mean(dim=1, keepdim=True))
                    preds = preds.expand_as(out['distilled_output']).clamp(0, 1)
                    val_loss += criterion(preds, yb).item()
                    val_steps += 1
                    val_preds.extend(preds.mean(dim=1).cpu().numpy())
                    val_targets.extend(yb.mean(dim=1).cpu().numpy())

            avg_train = train_loss / max(train_steps, 1)
            avg_val = val_loss / max(val_steps, 1)
            train_r2 = r2_score(train_targets, train_preds) if train_preds else 0.0
            val_r2 = r2_score(val_targets, val_preds) if val_preds else 0.0

            self.training_history['train_loss'].append(avg_train)
            self.training_history['val_loss'].append(avg_val)
            self.training_history['train_r2'].append(train_r2)
            self.training_history['val_r2'].append(val_r2)

            logger.info(f"Epoch {epoch}/{epochs} - Train Loss: {avg_train:.4f}, Val Loss: {avg_val:.4f}, Val R²: {val_r2:.4f}")

            if avg_val < best_val_loss - 1e-4:
                best_val_loss = avg_val
                best_epoch = epoch
                wait = 0
                self.save_lora(output_dir)
            else:
                wait += 1
                if wait >= patience:
                    logger.info(f"早停于epoch {epoch}")
                    break

            scheduler.step()

        return {'best_val_loss': best_val_loss, 'best_epoch': best_epoch, 'history': history}

    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            x_proj = self.feature_projector(x)
            output = self.model(x_proj)
            raw = output['distilled_output']
            corrected = self.output_corrector(raw.mean(dim=1, keepdim=True)).expand_as(raw).clamp(0, 1)
            if self.target_mean is not None and self.target_std is not None:
                output['distilled_output'] = (corrected * self.target_std + self.target_mean).clamp(0, 1)
            else:
                output['distilled_output'] = corrected
            return output

    def save_lora(self, output_dir: Optional[Path]) -> Optional[Path]:
        if output_dir is None:
            return None
        path = Path(output_dir) / 'lora_weights.pt'
        export_lora_weights(self.model, path)
        return path


def build_feature_vector(item: Dict[str, Any]) -> Optional[List[float]]:
    try:
        calories, protein, carbs, fat = float(item.get('calories', 0)), float(item.get('protein', 0)), \
            float(item.get('carbs', 0)), float(item.get('fat', 0))
        fiber, sugar, sodium = float(item.get('fiber', 0)), float(item.get('sugar', 0)), float(item.get('sodium', 0))

        total = protein + carbs + fat
        features = [
            calories, protein, carbs, fat, fiber, sugar, sodium,
            protein / total if total > 0 else 0,
            carbs / total if total > 0 else 0,
            fat / total if total > 0 else 0,
            (protein * 4 + fiber * 2) / (calories / 100) if calories > 0 else 0,
            max(0, min(1, min(protein / 20, 1) * 0.3 + min(fiber / 10, 1) * 0.2 - min(sugar / 50, 1) * 0.2 - min(sodium / 1000, 1) * 0.1 + (0.2 if 100 <= calories <= 400 else -0.1))),
        ]

        food_name = str(item.get('food_name', '')).lower()
        for kw in ['meat', 'fruit', 'vegetable', 'grain', 'dairy', 'beverage', 'nut', 'chinese']:
            features.append(1 if kw in food_name else 0)
        return features
    except Exception:
        return None


def load_dataset(json_path: Path, batch_size: int = 64) -> Tuple[DataLoader, DataLoader, StandardScaler, float, float]:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    features, targets = [], []
    for item in data:
        fv = build_feature_vector(item)
        if fv:
            features.append(fv)
            targets.append(item.get('acceptance_score', item.get('nutrition_score', 0.5)))

    X = np.array(features, dtype=np.float32)
    y = np.array(targets, dtype=np.float32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    y_mean, y_std = y.mean(), y.std()

    train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(list(zip(X_val, y_val)), batch_size=batch_size)

    return train_loader, val_loader, scaler, y_mean, y_std


def run(output_dir: str = 'outputs/stage1_cultural', epochs: int = 50,
         lora_rank: int = 16, lora_alpha: float = 32.0, lr: float = 1e-4) -> Dict[str, Any]:
    """便捷运行入口"""
    import argparse
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trainer = CulturalStage1Trainer(lora_rank=lora_rank, lora_alpha=lora_alpha)

    json_path = Path('TRAIN/outputs/stage1_data/final_cultural_preferences.json')
    if json_path.exists():
        train_loader, val_loader, scaler, y_mean, y_std = load_dataset(json_path)
        trainer.target_mean = y_mean
        trainer.target_std = y_std
        result = trainer.fit(train_loader, val_loader, epochs=epochs, lr=lr, output_dir=out_dir)
        trainer.save_lora(out_dir)
        return result
    return {'error': 'Dataset not found'}


if __name__ == '__main__':
    run()
