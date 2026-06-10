#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多尺度特征融合模块
使用多尺度卷积或注意力机制融合全局和局部特征
引入残差连接保持梯度流
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math


class MultiScaleConv1D(nn.Module):
    """多尺度1D卷积模块"""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_sizes: List[int] = [1, 3, 5, 7],
                 stride: int = 1,
                 padding_mode: str = 'same'):
        super().__init__()

        self.kernel_sizes = kernel_sizes
        self.convs = nn.ModuleList()

        for k in kernel_sizes:
            padding = k // 2 if padding_mode == 'same' else 0
            conv = nn.Conv1d(in_channels, out_channels, kernel_size=k,
                           stride=stride, padding=padding)
            self.convs.append(conv)

        # 融合层
        self.fusion = nn.Conv1d(out_channels * len(kernel_sizes), out_channels,
                               kernel_size=1)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入特征 [batch_size, in_channels, sequence_length]

        Returns:
            融合后的特征 [batch_size, out_channels, sequence_length]
        """
        # 多尺度卷积
        multi_scale_features = []
        for conv in self.convs:
            feat = conv(x)
            multi_scale_features.append(feat)

        # 拼接多尺度特征
        concatenated = torch.cat(multi_scale_features, dim=1)

        # 融合
        fused = self.fusion(concatenated)

        # 转置进行LayerNorm（LayerNorm在最后一个维度）
        fused = fused.transpose(1, 2)  # [batch_size, seq_len, channels]
        fused = self.norm(fused)
        fused = fused.transpose(1, 2)  # [batch_size, channels, seq_len]

        return fused


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim必须能被num_heads整除"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播

        Args:
            query: 查询 [batch_size, seq_len, embed_dim]
            key: 键 [batch_size, seq_len, embed_dim]
            value: 值 [batch_size, seq_len, embed_dim]
            mask: 注意力掩码 [batch_size, seq_len, seq_len]

        Returns:
            注意力输出 [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = query.size()

        # 线性投影
        Q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力权重
        attn_output = torch.matmul(attn_weights, V)

        # 拼接多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embed_dim
        )

        # 输出投影
        output = self.out_proj(attn_output)

        return output


class MultiScaleFeatureFusion(nn.Module):
    """多尺度特征融合模块"""

    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [128, 64, 32],
                 num_attention_heads: int = 8,
                 use_residual: bool = True,
                 dropout: float = 0.1):
        super().__init__()

        self.input_dim = input_dim
        self.use_residual = use_residual

        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])

        # 多尺度卷积（将特征视为序列）
        self.multi_scale_conv = MultiScaleConv1D(
            in_channels=1,  # 将特征维度视为通道
            out_channels=hidden_dims[0] // 4,
            kernel_sizes=[1, 3, 5, 7]
        )

        # 多头注意力
        self.attention = MultiHeadAttention(
            embed_dim=hidden_dims[0],
            num_heads=num_attention_heads,
            dropout=dropout
        )

        # 特征融合层
        layers = []
        prev_dim = hidden_dims[0]
        for hidden_dim in hidden_dims[1:]:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.fusion_layers = nn.Sequential(*layers)

        # 残差连接的投影层
        if use_residual and input_dim != hidden_dims[-1]:
            self.residual_proj = nn.Linear(input_dim, hidden_dims[-1])
        else:
            self.residual_proj = None

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入特征 [batch_size, input_dim] 或 [batch_size, seq_len, input_dim]

        Returns:
            融合后的特征 [batch_size, output_dim]
        """
        # 处理输入维度
        if x.dim() == 2:
            # [batch_size, input_dim] -> [batch_size, 1, input_dim]
            x = x.unsqueeze(1)
            seq_len = 1
        else:
            seq_len = x.size(1)

        batch_size = x.size(0)
        original_x = x

        # 输入投影
        x = self.input_proj(x)  # [batch_size, seq_len, hidden_dim]

        # 多尺度卷积（需要转置为 [batch_size, channels, seq_len]）
        x_conv = x.transpose(1, 2)  # [batch_size, hidden_dim, seq_len]
        x_conv = x_conv.unsqueeze(1)  # [batch_size, 1, hidden_dim, seq_len]
        x_conv = x_conv.squeeze(-1)  # [batch_size, 1, hidden_dim]
        x_conv = self.multi_scale_conv(x_conv)  # [batch_size, out_channels, hidden_dim]
        x_conv = x_conv.squeeze(-1).transpose(1, 2)  # [batch_size, hidden_dim, 1]
        x_conv = x_conv.squeeze(-1)  # [batch_size, hidden_dim]
        x_conv = x_conv.unsqueeze(1)  # [batch_size, 1, hidden_dim]

        # 如果维度不匹配，使用平均池化
        if x_conv.size(1) != x.size(1):
            x_conv = x_conv.mean(dim=1, keepdim=True).expand_as(x)

        # 注意力机制
        x_attn = self.attention(x, x, x)  # [batch_size, seq_len, hidden_dim]

        # 融合多尺度特征
        x_fused = x + x_conv + x_attn  # 残差连接
        x_fused = self.fusion_layers(x_fused)  # [batch_size, seq_len, output_dim]

        # 如果输入是2D，压缩序列维度
        if seq_len == 1:
            x_fused = x_fused.squeeze(1)  # [batch_size, output_dim]
        else:
            # 使用平均池化或最大池化
            x_fused = x_fused.mean(dim=1)  # [batch_size, output_dim]

        # 残差连接
        if self.use_residual:
            if original_x.dim() == 3:
                original_x = original_x.mean(dim=1)  # [batch_size, input_dim]

            if self.residual_proj is not None:
                residual = self.residual_proj(original_x)
            else:
                residual = original_x

            x_fused = x_fused + residual

        x_fused = self.dropout(x_fused)

        return x_fused


class HierarchicalMultiScaleFusion(nn.Module):
    """分层多尺度特征融合（针对不同特征类型）"""

    def __init__(self,
                 nutrition_dim: int,
                 taste_dim: int,
                 context_dim: int,
                 output_dim: int = 64):
        super().__init__()

        # 为不同类型的特征创建不同的融合模块
        self.nutrition_fusion = MultiScaleFeatureFusion(
            input_dim=nutrition_dim,
            hidden_dims=[64, 32],
            num_attention_heads=4
        )

        self.taste_fusion = MultiScaleFeatureFusion(
            input_dim=taste_dim,
            hidden_dims=[64, 32],
            num_attention_heads=4
        )

        self.context_fusion = MultiScaleFeatureFusion(
            input_dim=context_dim,
            hidden_dims=[32, 16],
            num_attention_heads=2
        )

        # 最终融合层
        total_dim = 32 + 32 + 16  # 各融合模块的输出维度之和
        self.final_fusion = nn.Sequential(
            nn.Linear(total_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self,
                nutrition_features: torch.Tensor,
                taste_features: torch.Tensor,
                context_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            nutrition_features: 营养特征 [batch_size, nutrition_dim]
            taste_features: 口味特征 [batch_size, taste_dim]
            context_features: 上下文特征 [batch_size, context_dim]

        Returns:
            融合后的特征 [batch_size, output_dim]
        """
        # 分别融合不同类型的特征
        nutrition_fused = self.nutrition_fusion(nutrition_features)
        taste_fused = self.taste_fusion(taste_features)
        context_fused = self.context_fusion(context_features)

        # 拼接所有特征
        concatenated = torch.cat([nutrition_fused, taste_fused, context_fused], dim=1)

        # 最终融合
        output = self.final_fusion(concatenated)

        return output
