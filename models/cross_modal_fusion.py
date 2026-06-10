#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态融合跨注意力机制 (Cross-Modal Attention Fusion)
创新点2: 分层融合图像和文本特征,实现细粒度语义对齐

参考最新研究:
- MODA (Multimodal Duplex Attention)
- MambaRec (Dilated Refinement Attention)
- MUFASA (Multimodal Fusion Architecture Search)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class DilatedRefinementAttention(nn.Module):
    """
    扩张精炼注意力模块 (DREAM)

    使用多尺度扩张卷积+通道/空间注意力实现细粒度跨模态对齐
    """

    def __init__(self, dim: int, num_scales: int = 3):
        super().__init__()
        self.dim = dim
        self.num_scales = num_scales

        # 多尺度扩张卷积
        self.dilated_convs = nn.ModuleList([
            nn.Conv1d(dim, dim, kernel_size=3, padding=2**i, dilation=2**i)
            for i in range(num_scales)
        ])

        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(dim * num_scales, dim, 1),
            nn.ReLU(),
            nn.Conv1d(dim, dim * num_scales, 1),
            nn.Sigmoid()
        )

        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        # 特征融合
        self.fusion = nn.Conv1d(dim * num_scales, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, dim, seq_len]
        Returns:
            refined: [batch_size, dim, seq_len]
        """
        # 多尺度特征提取
        multi_scale_features = []
        # Compute maximum effective kernel across dilated convs and pad input once
        seq_len = x.size(2)
        max_eff_k = 0
        for conv in self.dilated_convs:
            try:
                k = conv.kernel_size[0] if isinstance(conv.kernel_size, (list, tuple)) else conv.kernel_size
                d = conv.dilation[0] if isinstance(conv.dilation, (list, tuple)) else conv.dilation
            except Exception:
                k = conv.kernel_size
                d = conv.dilation
            eff_k = (k - 1) * d + 1
            if eff_k > max_eff_k:
                max_eff_k = eff_k

        # Only pad if sequence length >1 and less than required effective kernel.
        # If seq_len == 1 (common for image/text single-vector inputs), avoid padding
        # to preserve sequence length = 1 (otherwise downstream code expects squeeze(2) to work).
        if seq_len > 1 and seq_len < max_eff_k:
            pad_right = max_eff_k - seq_len
            x = F.pad(x, (0, pad_right))

        # Ensure contiguous before convs to avoid misaligned address errors
        x = x.contiguous()
        for conv in self.dilated_convs:
            multi_scale_features.append(conv(x))

        # 拼接多尺度特征
        concat_features = torch.cat(multi_scale_features, dim=1)  # [B, dim*num_scales, seq_len]

        # 通道注意力
        # Make contiguous before passing through 1d conv layers
        concat_features = concat_features.contiguous()
        channel_weights = self.channel_attention(concat_features)
        channel_refined = concat_features * channel_weights

        # 空间注意力
        max_pool = torch.max(channel_refined, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(channel_refined, dim=1, keepdim=True)
        spatial_input = torch.cat([max_pool, avg_pool], dim=1).contiguous()  # [B, 2, seq_len]
        spatial_weights = self.spatial_attention(spatial_input)
        spatial_refined = channel_refined * spatial_weights

        # 融合为原始维度
        refined = self.fusion(spatial_refined.contiguous())
        return refined


class CrossModalAttention(nn.Module):
    """
    多模态融合跨注意力机制

    核心创新:
    - 分层融合策略 (hierarchical fusion)
    - 扩张精炼注意力 (DREAM)
    - 模态间交互建模 (inter-modal interaction)

    Args:
        image_dim: 图像特征维度 (default: 2048, ResNet输出)
        text_dim: 文本特征维度 (default: 768, BERT输出)
        cultural_dim: 文化特征维度 (default: 64)
        fusion_dim: 融合后维度 (default: 256)
        fusion_type: 融合类型 ['hierarchical', 'concatenate', 'attention'] (default: 'hierarchical')
        use_dream: 是否使用DREAM模块 (default: True)
    """

    def __init__(
        self,
        image_dim: int = 2048,
        text_dim: int = 768,
        cultural_dim: int = 64,
        fusion_dim: int = 256,
        fusion_type: str = 'hierarchical',
        use_dream: bool = True
    ):
        super().__init__()

        self.image_dim = image_dim
        self.text_dim = text_dim
        self.cultural_dim = cultural_dim
        self.fusion_dim = fusion_dim
        self.fusion_type = fusion_type
        self.use_dream = use_dream

        # 模态投影层 (降维到统一空间)
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.cultural_proj = nn.Sequential(
            nn.Linear(cultural_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # DREAM模块 (可选)
        if use_dream:
            self.dream_image = DilatedRefinementAttention(fusion_dim, num_scales=3)
            self.dream_text = DilatedRefinementAttention(fusion_dim, num_scales=3)

        # 跨模态注意力
        if fusion_type == 'hierarchical':
            # 阶段1: 图像-文本交互
            self.image_text_attention = nn.MultiheadAttention(
                embed_dim=fusion_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )

            # 阶段2: 融合特征-文化交互
            self.fusion_cultural_attention = nn.MultiheadAttention(
                embed_dim=fusion_dim,
                num_heads=4,
                dropout=0.1,
                batch_first=True
            )

            # 分层融合层
            self.hierarchical_fusion = nn.Sequential(
                nn.Linear(fusion_dim * 2, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(fusion_dim, fusion_dim)
            )

        elif fusion_type == 'concatenate':
            self.concat_fusion = nn.Sequential(
                nn.Linear(fusion_dim * 3, fusion_dim * 2),
                nn.LayerNorm(fusion_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(fusion_dim * 2, fusion_dim)
            )

        elif fusion_type == 'attention':
            self.attention_fusion = nn.MultiheadAttention(
                embed_dim=fusion_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )

        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")

        # 模态一致性约束 (MMD - Maximum Mean Discrepancy)
        self.mmd_weight = nn.Parameter(torch.tensor(0.1))

        # 应用Xavier初始化 (基于最佳实践)
        self._init_weights()

        logger.info(
            f"CrossModalAttention initialized: "
            f"image_dim={image_dim}, text_dim={text_dim}, cultural_dim={cultural_dim}, "
            f"fusion_dim={fusion_dim}, fusion_type={fusion_type}, use_dream={use_dream}"
        )

    def _init_weights(self):
        """Xavier/Kaiming初始化提升收敛速度"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.8)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

    def compute_mmd_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        计算最大均值差异损失 (用于全局分布对齐)

        Args:
            x, y: [batch_size, dim]
        Returns:
            mmd_loss: scalar
        """
        # RBF核
        def rbf_kernel(x, y, sigma=1.0):
            dist = torch.cdist(x, y, p=2)
            return torch.exp(-dist ** 2 / (2 * sigma ** 2))

        xx = rbf_kernel(x, x).mean()
        yy = rbf_kernel(y, y).mean()
        xy = rbf_kernel(x, y).mean()

        mmd = xx + yy - 2 * xy
        return mmd

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        cultural_features: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        前向传播

        Args:
            image_features: [batch_size, image_dim] 图像特征
            text_features: [batch_size, text_dim] 文本特征
            cultural_features: [batch_size, cultural_dim] 文化特征 (可选)
            return_attention_weights: 是否返回注意力权重

        Returns:
            fused_features: [batch_size, fusion_dim] 融合后特征
            attention_info: 注意力权重信息 (可选)
        """
        batch_size = image_features.size(0)

        # 模态投影
        image_proj = self.image_proj(image_features)  # [B, fusion_dim]
        text_proj = self.text_proj(text_features)      # [B, fusion_dim]

        if cultural_features is not None:
            cultural_proj = self.cultural_proj(cultural_features)  # [B, fusion_dim]
        else:
            cultural_proj = torch.zeros_like(image_proj)

        # DREAM精炼 (可选)
        if self.use_dream:
            # 转换为序列格式 [B, dim, 1]
            image_seq = image_proj.unsqueeze(2)
            text_seq = text_proj.unsqueeze(2)

            image_refined = self.dream_image(image_seq).squeeze(2)  # [B, fusion_dim]
            text_refined = self.dream_text(text_seq).squeeze(2)     # [B, fusion_dim]
        else:
            image_refined = image_proj
            text_refined = text_proj

        # 融合策略
        attention_info = {}

        if self.fusion_type == 'hierarchical':
            # 阶段1: 图像-文本交互
            image_unsqueezed = image_refined.unsqueeze(1)  # [B, 1, fusion_dim]
            text_unsqueezed = text_refined.unsqueeze(1)    # [B, 1, fusion_dim]

            # 双向注意力
            image_to_text, attn_weights_it = self.image_text_attention(
                image_unsqueezed, text_unsqueezed, text_unsqueezed
            )
            text_to_image, attn_weights_ti = self.image_text_attention(
                text_unsqueezed, image_unsqueezed, image_unsqueezed
            )

            # 融合图像-文本特征
            image_text_fused = (image_to_text + text_to_image).squeeze(1)  # [B, fusion_dim]

            # 阶段2: 融合特征-文化交互
            fused_unsqueezed = image_text_fused.unsqueeze(1)  # [B, 1, fusion_dim]
            cultural_unsqueezed = cultural_proj.unsqueeze(1)   # [B, 1, fusion_dim]

            cultural_aware_fused, attn_weights_fc = self.fusion_cultural_attention(
                fused_unsqueezed, cultural_unsqueezed, cultural_unsqueezed
            )
            cultural_aware_fused = cultural_aware_fused.squeeze(1)  # [B, fusion_dim]

            # 分层融合
            final_fused = self.hierarchical_fusion(
                torch.cat([image_text_fused, cultural_aware_fused], dim=-1)
            )

            if return_attention_weights:
                attention_info = {
                    'image_to_text': attn_weights_it,
                    'text_to_image': attn_weights_ti,
                    'fusion_cultural': attn_weights_fc
                }

        elif self.fusion_type == 'concatenate':
            concat = torch.cat([image_refined, text_refined, cultural_proj], dim=-1)
            final_fused = self.concat_fusion(concat)

        elif self.fusion_type == 'attention':
            # 堆叠为序列
            stacked = torch.stack([image_refined, text_refined, cultural_proj], dim=1)  # [B, 3, fusion_dim]

            attended, attn_weights = self.attention_fusion(stacked, stacked, stacked)
            final_fused = attended.mean(dim=1)  # [B, fusion_dim]

            if return_attention_weights:
                attention_info = {'attention_weights': attn_weights}

        # 计算MMD损失 (用于训练时约束模态一致性)
        mmd_loss = self.compute_mmd_loss(image_refined, text_refined) * self.mmd_weight
        attention_info['mmd_loss'] = mmd_loss

        if return_attention_weights:
            return final_fused, attention_info
        return final_fused, None

    def get_modality_importance(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        cultural_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        获取各模态重要性 (用于消融实验)

        Returns:
            Dict包含各模态的贡献度
        """
        fused, attention_info = self.forward(
            image_features,
            text_features,
            cultural_features,
            return_attention_weights=True
        )

        return {
            'fused_features': fused,
            **attention_info
        }
