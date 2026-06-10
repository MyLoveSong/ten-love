#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强型文化适配模型架构
实现更深层次的特征提取和更强的表达能力
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from simplified_lora import LoRALinear, LoRAConfig


class FeatureExtractor(nn.Module):
    """增强型特征提取器"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        use_batch_norm: bool = True,
        use_residual: bool = True,
        activation: str = "gelu"
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.use_residual = use_residual

        # 选择激活函数
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.1)
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            self.activation = nn.GELU()

        # 输入层
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim) if use_batch_norm else nn.Identity()
        self.input_dropout = nn.Dropout(dropout)

        # 隐藏层
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if use_batch_norm else nn.Identity(),
                self.activation,
                nn.Dropout(dropout)
            )
            self.layers.append(layer)

        # 输出层
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 输入层
        x = self.input_layer(x)
        x = self.input_norm(x)
        x = self.activation(x)
        x = self.input_dropout(x)

        # 隐藏层 (带残差连接)
        for layer in self.layers:
            if self.use_residual:
                residual = x
                x = layer(x)
                x = x + residual
            else:
                x = layer(x)

        # 输出层
        x = self.output_layer(x)

        return x


class EnhancedCulturalModel(nn.Module):
    """增强型文化适配模型"""

    def __init__(
        self,
        num_regions: int,
        num_cuisines: int,
        feature_dim: int = 192,
        hidden_dim: int = 384,
        lora_rank: int = 48,
        dropout: float = 0.2,
        use_batch_norm: bool = True,
        use_residual: bool = True,
        num_layers: int = 3,
        activation: str = "gelu"
    ):
        super().__init__()

        self.num_regions = num_regions
        self.num_cuisines = num_cuisines
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # 嵌入层
        self.region_embedding = nn.Embedding(num_regions, feature_dim // 2)
        self.cuisine_embedding = nn.Embedding(num_cuisines, feature_dim // 2)

        # 特征提取器
        self.preference_encoder = FeatureExtractor(
            input_dim=10,  # 偏好特征数量
            hidden_dim=hidden_dim,
            output_dim=feature_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
            use_residual=use_residual,
            activation=activation
        )

        # 上下文编码器
        self.context_encoder = FeatureExtractor(
            input_dim=5,  # 上下文特征数量
            hidden_dim=hidden_dim // 2,
            output_dim=feature_dim // 2,
            num_layers=num_layers - 1,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
            use_residual=use_residual,
            activation=activation
        )

        # 特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(feature_dim * 2 + feature_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            getattr(nn, activation.upper())() if hasattr(nn, activation.upper()) else nn.GELU(),
            nn.Dropout(dropout)
        )

        # LoRA配置
        lora_config = LoRAConfig(
            r=lora_rank,
            alpha=lora_rank * 2,
            dropout=dropout,
            use_rslora=True,  # 使用Rank-Stabilized LoRA
            init_lora_weights=True
        )

        # LoRA适配层
        self.lora_layer = LoRALinear(
            in_features=hidden_dim,
            out_features=hidden_dim,
            config=lora_config
        )

        # 预测头
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            getattr(nn, activation.upper())() if hasattr(nn, activation.upper()) else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        region_ids: torch.Tensor,
        cuisine_ids: torch.Tensor,
        preferences: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向传播"""
        # 嵌入
        region_emb = self.region_embedding(region_ids)
        cuisine_emb = self.cuisine_embedding(cuisine_ids)

        # 编码偏好特征
        pref_features = self.preference_encoder(preferences)

        # 合并嵌入
        combined_emb = torch.cat([region_emb, cuisine_emb], dim=1)

        # 编码上下文特征（如果有）
        if context is not None:
            context_features = self.context_encoder(context)
            # 特征融合
            features = torch.cat([combined_emb, pref_features, context_features], dim=1)
        else:
            # 特征融合（无上下文）
            features = torch.cat([combined_emb, pref_features], dim=1)

        # 特征融合
        fused_features = self.fusion_layer(features)

        # LoRA适配
        adapted_features = self.lora_layer(fused_features)

        # 预测接受度
        acceptance = self.prediction_head(adapted_features)

        return acceptance.squeeze(-1)
