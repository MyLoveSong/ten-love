"""
多模态数据融合引擎 - 整合多种生理数据源
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import math

logger = logging.getLogger(__name__)

class FusionStrategy(Enum):
    """融合策略"""
    EARLY_FUSION = "early_fusion"
    LATE_FUSION = "late_fusion"
    HYBRID_FUSION = "hybrid_fusion"
    ATTENTION_FUSION = "attention_fusion"

@dataclass
class MultiModalConfig:
    """多模态配置"""
    modalities: List[str] = None
    fusion_strategy: FusionStrategy = FusionStrategy.ATTENTION_FUSION
    hidden_dim: int = 128
    num_attention_heads: int = 8
    dropout: float = 0.1
    temperature: float = 0.07

    def __post_init__(self):
        if self.modalities is None:
            self.modalities = ['glucose', 'diet', 'exercise', 'sleep', 'stress']

class ModalityEncoder(nn.Module):
    """模态编码器"""

    def __init__(self, input_dim: int, hidden_dim: int, modality_type: str):
        super().__init__()
        self.modality_type = modality_type
        self.hidden_dim = hidden_dim

        # 根据模态类型选择不同的编码器
        if modality_type == 'glucose':
            self.encoder = self._create_glucose_encoder(input_dim, hidden_dim)
        elif modality_type == 'diet':
            self.encoder = self._create_diet_encoder(input_dim, hidden_dim)
        elif modality_type == 'exercise':
            self.encoder = self._create_exercise_encoder(input_dim, hidden_dim)
        elif modality_type == 'sleep':
            self.encoder = self._create_sleep_encoder(input_dim, hidden_dim)
        elif modality_type == 'stress':
            self.encoder = self._create_stress_encoder(input_dim, hidden_dim)
        else:
            self.encoder = self._create_default_encoder(input_dim, hidden_dim)

    def _create_glucose_encoder(self, input_dim: int, hidden_dim: int) -> nn.Module:
        """创建血糖编码器"""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def _create_diet_encoder(self, input_dim: int, hidden_dim: int) -> nn.Module:
        """创建膳食编码器"""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def _create_exercise_encoder(self, input_dim: int, hidden_dim: int) -> nn.Module:
        """创建运动编码器"""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def _create_sleep_encoder(self, input_dim: int, hidden_dim: int) -> nn.Module:
        """创建睡眠编码器"""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def _create_stress_encoder(self, input_dim: int, hidden_dim: int) -> nn.Module:
        """创建压力编码器"""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def _create_default_encoder(self, input_dim: int, hidden_dim: int) -> nn.Module:
        """创建默认编码器"""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

class AttentionFusion(nn.Module):
    """注意力融合模块"""

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, modality_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """注意力融合"""
        modalities = list(modality_features.keys())
        features = torch.stack([modality_features[mod] for mod in modalities])

        batch_size, num_modalities, hidden_dim = features.shape

        # 计算注意力
        Q = self.query(features).view(batch_size, num_modalities, self.num_heads, self.head_dim)
        K = self.key(features).view(batch_size, num_modalities, self.num_heads, self.head_dim)
        V = self.value(features).view(batch_size, num_modalities, self.num_heads, self.head_dim)

        # 转置以便计算注意力
        Q = Q.transpose(1, 2)  # [batch, num_heads, num_modalities, head_dim]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 应用注意力
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, num_modalities, hidden_dim
        )

        # 输出投影
        output = self.out_proj(attended)

        return output

class CrossModalAttention(nn.Module):
    """跨模态注意力"""

    def __init__(self, hidden_dim: int, temperature: float = 0.07):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temperature = temperature

        self.projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, modality_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """跨模态注意力"""
        modalities = list(modality_features.keys())
        features = {mod: self.projection(features) for mod, features in modality_features.items()}

        # 计算模态间的相似度
        enhanced_features = {}

        for mod_i in modalities:
            feature_i = features[mod_i]
            attention_weights = []

            for mod_j in modalities:
                if mod_i != mod_j:
                    feature_j = features[mod_j]
                    # 计算相似度
                    similarity = torch.matmul(feature_i, feature_j.T) / self.temperature
                    attention_weight = F.softmax(similarity, dim=-1)
                    attention_weights.append(attention_weight)

            # 聚合其他模态的信息
            if attention_weights:
                avg_attention = torch.mean(torch.stack(attention_weights), dim=0)
                enhanced_feature = torch.matmul(avg_attention, feature_i)
                enhanced_features[mod_i] = enhanced_feature
            else:
                enhanced_features[mod_i] = feature_i

        return enhanced_features

class MultiModalFusion:
    """多模态数据融合引擎"""

    def __init__(self, config: MultiModalConfig):
        self.config = config
        self.modality_encoders = {}
        self.fusion_module = None

        # 初始化模态编码器
        self._initialize_modality_encoders()

        # 初始化融合模块
        self._initialize_fusion_module()

        logger.info(f"多模态融合引擎初始化完成，模态: {config.modalities}")

    def _initialize_modality_encoders(self):
        """初始化模态编码器"""
        for modality in self.config.modalities:
            # 根据模态类型设置输入维度
            input_dim = self._get_modality_input_dim(modality)

            self.modality_encoders[modality] = ModalityEncoder(
                input_dim=input_dim,
                hidden_dim=self.config.hidden_dim,
                modality_type=modality
            )

    def _get_modality_input_dim(self, modality: str) -> int:
        """获取模态输入维度"""
        dim_map = {
            'glucose': 10,    # 血糖相关特征
            'diet': 15,       # 膳食相关特征
            'exercise': 8,    # 运动相关特征
            'sleep': 6,       # 睡眠相关特征
            'stress': 5       # 压力相关特征
        }
        return dim_map.get(modality, 10)

    def _initialize_fusion_module(self):
        """初始化融合模块"""
        if self.config.fusion_strategy == FusionStrategy.ATTENTION_FUSION:
            self.fusion_module = AttentionFusion(
                hidden_dim=self.config.hidden_dim,
                num_heads=self.config.num_attention_heads,
                dropout=self.config.dropout
            )
        elif self.config.fusion_strategy == FusionStrategy.HYBRID_FUSION:
            self.fusion_module = CrossModalAttention(
                hidden_dim=self.config.hidden_dim,
                temperature=self.config.temperature
            )
        else:
            # 默认使用简单融合
            self.fusion_module = nn.Linear(
                self.config.hidden_dim * len(self.config.modalities),
                self.config.hidden_dim
            )

    def integrate_multimodal_data(self, data_streams: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """整合多模态数据"""
        # 编码各模态数据
        modality_features = {}

        for modality, data in data_streams.items():
            if modality in self.modality_encoders:
                encoder = self.modality_encoders[modality]
                features = encoder(data)
                modality_features[modality] = features

        # 融合特征
        if self.config.fusion_strategy == FusionStrategy.EARLY_FUSION:
            fused_features = self._early_fusion(modality_features)
        elif self.config.fusion_strategy == FusionStrategy.LATE_FUSION:
            fused_features = self._late_fusion(modality_features)
        elif self.config.fusion_strategy == FusionStrategy.ATTENTION_FUSION:
            fused_features = self._attention_fusion(modality_features)
        elif self.config.fusion_strategy == FusionStrategy.HYBRID_FUSION:
            fused_features = self._hybrid_fusion(modality_features)
        else:
            fused_features = self._default_fusion(modality_features)

        # 生成融合结果
        result = {
            'fused_features': fused_features,
            'modality_features': modality_features,
            'fusion_strategy': self.config.fusion_strategy.value,
            'modality_weights': self._calculate_modality_weights(modality_features)
        }

        return result

    def _early_fusion(self, modality_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """早期融合"""
        # 简单拼接
        concatenated = torch.cat(list(modality_features.values()), dim=-1)

        # 通过线性层降维
        if hasattr(self.fusion_module, 'weight'):
            fused = self.fusion_module(concatenated)
        else:
            # 如果没有线性层，直接平均
            fused = torch.mean(torch.stack(list(modality_features.values())), dim=0)

        return fused

    def _late_fusion(self, modality_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """晚期融合"""
        # 对每个模态单独处理，然后融合
        processed_features = []

        for modality, features in modality_features.items():
            # 这里可以添加模态特定的处理
            processed_features.append(features)

        # 平均融合
        fused = torch.mean(torch.stack(processed_features), dim=0)

        return fused

    def _attention_fusion(self, modality_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """注意力融合"""
        if isinstance(self.fusion_module, AttentionFusion):
            attended_features = self.fusion_module(modality_features)
            # 平均所有模态的注意力输出
            fused = torch.mean(attended_features, dim=1)
        else:
            fused = torch.mean(torch.stack(list(modality_features.values())), dim=0)

        return fused

    def _hybrid_fusion(self, modality_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """混合融合"""
        if isinstance(self.fusion_module, CrossModalAttention):
            enhanced_features = self.fusion_module(modality_features)
            # 融合增强后的特征
            fused = torch.mean(torch.stack(list(enhanced_features.values())), dim=0)
        else:
            fused = torch.mean(torch.stack(list(modality_features.values())), dim=0)

        return fused

    def _default_fusion(self, modality_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """默认融合"""
        return torch.mean(torch.stack(list(modality_features.values())), dim=0)

    def _calculate_modality_weights(self, modality_features: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """计算模态权重"""
        weights = {}

        for modality, features in modality_features.items():
            # 基于特征方差计算权重
            variance = torch.var(features).item()
            weights[modality] = variance

        # 归一化权重
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {mod: weight / total_weight for mod, weight in weights.items()}

        return weights

    def get_modality_importance(self, data_streams: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """获取模态重要性"""
        result = self.integrate_multimodal_data(data_streams)
        return result['modality_weights']

    def update_fusion_strategy(self, new_strategy: FusionStrategy):
        """更新融合策略"""
        self.config.fusion_strategy = new_strategy
        self._initialize_fusion_module()

        logger.info(f"融合策略已更新为: {new_strategy.value}")

    def add_modality(self, modality_name: str, input_dim: int):
        """添加新模态"""
        if modality_name not in self.modality_encoders:
            self.modality_encoders[modality_name] = ModalityEncoder(
                input_dim=input_dim,
                hidden_dim=self.config.hidden_dim,
                modality_type=modality_name
            )
            self.config.modalities.append(modality_name)

            logger.info(f"已添加新模态: {modality_name}")

    def remove_modality(self, modality_name: str):
        """移除模态"""
        if modality_name in self.modality_encoders:
            del self.modality_encoders[modality_name]
            self.config.modalities.remove(modality_name)

            logger.info(f"已移除模态: {modality_name}")

    def get_fusion_statistics(self, data_streams: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """获取融合统计信息"""
        result = self.integrate_multimodal_data(data_streams)

        statistics = {
            'num_modalities': len(self.config.modalities),
            'fusion_strategy': self.config.fusion_strategy.value,
            'modality_weights': result['modality_weights'],
            'feature_dimensions': {
                mod: features.shape for mod, features in result['modality_features'].items()
            },
            'fused_feature_shape': result['fused_features'].shape
        }

        return statistics
