#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
改进型文化适配模型架构
实现更深层次的特征提取、注意力机制和更强的表达能力
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


class SelfAttention(nn.Module):
    """自注意力机制"""

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        assert hidden_dim % num_heads == 0, "hidden_dim必须能被num_heads整除"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # 线性变换层
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        batch_size = x.shape[0]

        # 线性变换
        q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 注意力计算
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # 加权求和
        output = torch.matmul(attention, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)

        # 输出投影
        output = self.output(output)

        return output


class FeedForward(nn.Module):
    """前馈神经网络"""

    def __init__(
        self,
        hidden_dim: int,
        ff_dim: int,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()

        self.fc1 = nn.Linear(hidden_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer块"""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        ff_dim: int = None,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()

        if ff_dim is None:
            ff_dim = hidden_dim * 4

        self.attention = SelfAttention(hidden_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = FeedForward(hidden_dim, ff_dim, dropout, activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 自注意力层
        attention_output = self.attention(x)
        x = x + self.dropout(attention_output)
        x = self.norm1(x)

        # 前馈网络
        ff_output = self.ff(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)

        return x


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
        activation: str = "gelu",
        use_attention: bool = True,
        num_heads: int = 4
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.use_residual = use_residual
        self.use_attention = use_attention

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
            if use_attention and i > 0:
                # 使用Transformer块
                layer = TransformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ff_dim=hidden_dim * 4,
                    dropout=dropout,
                    activation=activation
                )
            else:
                # 使用标准层
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
        for i, layer in enumerate(self.layers):
            if self.use_attention and i > 0:
                # Transformer块已经包含残差连接
                x = layer(x.unsqueeze(1)).squeeze(1)
            elif self.use_residual:
                residual = x
                x = layer(x)
                x = x + residual
            else:
                x = layer(x)

        # 输出层
        x = self.output_layer(x)

        return x


class ImprovedCulturalModel(nn.Module):
    """改进型文化适配模型"""

    def __init__(
        self,
        num_regions: int,
        num_cuisines: int,
        feature_dim: int = 256,
        hidden_dim: int = 512,
        lora_rank: int = 64,
        dropout: float = 0.2,
        use_batch_norm: bool = True,
        use_residual: bool = True,
        num_layers: int = 4,
        num_heads: int = 8,
        activation: str = "gelu",
        use_attention: bool = True
    ):
        super().__init__()

        self.num_regions = num_regions
        self.num_cuisines = num_cuisines
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # 嵌入层
        self.region_embedding = nn.Embedding(max(2, num_regions), feature_dim // 2)
        self.cuisine_embedding = nn.Embedding(max(2, num_cuisines), feature_dim // 2)

        # 特征提取器
        self.preference_encoder = FeatureExtractor(
            input_dim=10,  # 偏好特征数量
            hidden_dim=hidden_dim,
            output_dim=feature_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
            use_residual=use_residual,
            activation=activation,
            use_attention=use_attention,
            num_heads=num_heads
        )

        # 上下文编码器
        self.context_encoder = FeatureExtractor(
            input_dim=5,  # 上下文特征数量
            hidden_dim=hidden_dim // 2,
            output_dim=feature_dim // 2,
            num_layers=max(1, num_layers - 1),
            dropout=dropout,
            use_batch_norm=use_batch_norm,
            use_residual=use_residual,
            activation=activation,
            use_attention=use_attention,
            num_heads=num_heads // 2
        )

        # 特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(feature_dim * 2 + feature_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            getattr(nn, activation.upper())() if hasattr(nn, activation.upper()) else nn.GELU(),
            nn.Dropout(dropout)
        )

        # 特征交互层 - 使用Transformer块
        self.interaction_layer = TransformerBlock(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            ff_dim=hidden_dim * 4,
            dropout=dropout,
            activation=activation
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
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            getattr(nn, activation.upper())() if hasattr(nn, activation.upper()) else nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 4, 1)
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

        # 编码上下文特征（如果有）
        if context is not None:
            context_features = self.context_encoder(context)
        else:
            # 如果没有上下文，使用零向量
            context_features = torch.zeros(
                preferences.shape[0],
                self.feature_dim // 2,
                device=preferences.device
            )

        # 特征融合
        combined_features = torch.cat([
            region_emb,
            cuisine_emb,
            pref_features,
            context_features
        ], dim=1)

        fused_features = self.fusion_layer(combined_features)

        # 特征交互 - 通过Transformer块
        fused_features = self.interaction_layer(fused_features.unsqueeze(1)).squeeze(1)

        # LoRA适配
        adapted_features = self.lora_layer(fused_features)

        # 预测
        output = self.prediction_head(adapted_features)

        return output.squeeze(-1)
