#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
优化的模型架构
基于现代工程标准，实现高效、可扩展的模型架构设计
"""

import os
import sys
import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import math
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import MultiheadAttention
import warnings
warnings.filterwarnings("ignore")

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入必要的模块
from simplified_lora import LoRALinear, LoRAConfig


@dataclass
class ModelConfig:
    """模型配置"""
    # 基础参数
    input_dim: int = 10
    hidden_dim: int = 256
    output_dim: int = 1

    # LoRA参数
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    # 架构参数
    num_layers: int = 3
    num_heads: int = 8
    dropout: float = 0.1
    activation: str = "gelu"

    # 优化参数
    use_residual: bool = True
    use_layer_norm: bool = True
    use_attention: bool = True
    use_skip_connections: bool = True

    # 正则化参数
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0


class BaseModel(ABC, nn.Module):
    """基础模型抽象类"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self._initialize_weights()

    @abstractmethod
    def forward(self, *args, **kwargs):
        """前向传播"""
        pass

    def _initialize_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)

    def get_model_size(self) -> Dict[str, int]:
        """获取模型大小信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "non_trainable_params": total_params - trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024)  # 假设float32
        }


class ResidualBlock(nn.Module):
    """残差块"""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.layer_norm1 = nn.LayerNorm(dim)
        self.layer_norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        residual = x

        x = self.layer_norm1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear1(x)

        x = self.layer_norm2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x + residual


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力机制"""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        assert dim % num_heads == 0, "dim必须能被num_heads整除"

        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)
        self.out_linear = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # 计算Q, K, V
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 应用注意力
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)

        # 输出投影
        output = self.out_linear(attended)

        return output, attention_weights


class OptimizedCulturalModel(BaseModel):
    """优化的文化适应模型"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        # 嵌入层
        self.region_embedding = nn.Embedding(9, config.hidden_dim // 4)
        self.cuisine_embedding = nn.Embedding(9, config.hidden_dim // 4)

        # 输入投影
        self.input_projection = nn.Linear(
            config.input_dim + config.hidden_dim // 2,
            config.hidden_dim
        )

        # 特征提取器
        self.feature_extractor = nn.ModuleList([
            ResidualBlock(config.hidden_dim, config.dropout)
            for _ in range(config.num_layers)
        ])

        # 注意力机制
        if config.use_attention:
            self.attention = MultiHeadSelfAttention(
                config.hidden_dim,
                config.num_heads,
                config.dropout
            )

        # LoRA适配器
        lora_config = LoRAConfig(
            r=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
            use_rslora=True,
            init_lora_weights=True
        )

        self.lora_adapter = LoRALinear(
            in_features=config.hidden_dim,
            out_features=config.hidden_dim,
            config=lora_config
        )

        # 输出头
        self.output_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.output_dim)
        )

        # 初始化权重
        self._initialize_weights()

    def forward(self, preferences, nutrition, cultural):
        """前向传播"""
        # 嵌入文化特征
        region_emb = self.region_embedding(cultural[:, 0].long())
        cuisine_emb = self.cuisine_embedding(cultural[:, 1].long())
        cultural_emb = torch.cat([region_emb, cuisine_emb], dim=1)

        # 组合所有特征
        combined_features = torch.cat([preferences, nutrition, cultural_emb], dim=1)

        # 输入投影
        x = self.input_projection(combined_features)

        # 特征提取
        for layer in self.feature_extractor:
            x = layer(x)

        # 注意力机制
        if self.config.use_attention:
            # 为注意力机制添加序列维度
            x_seq = x.unsqueeze(1)  # [batch_size, 1, hidden_dim]
            attended, attention_weights = self.attention(x_seq)
            x = attended.squeeze(1)  # 移除序列维度

        # LoRA适配
        x = self.lora_adapter(x)

        # 输出预测
        output = self.output_head(x)

        return output.squeeze(-1)


class EfficientCulturalModel(BaseModel):
    """高效文化适应模型（轻量级）"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        # 简化的嵌入层
        self.region_embedding = nn.Embedding(9, 16)
        self.cuisine_embedding = nn.Embedding(9, 16)

        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(config.input_dim + 32, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )

        # 简化的LoRA适配器
        lora_config = LoRAConfig(
            r=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
            use_rslora=False,  # 简化配置
            init_lora_weights=True
        )

        self.lora_adapter = LoRALinear(
            in_features=config.hidden_dim // 2,
            out_features=config.hidden_dim // 2,
            config=lora_config
        )

        # 输出头
        self.output_head = nn.Sequential(
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 4),
            nn.LayerNorm(config.hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 4, config.output_dim)
        )

    def forward(self, preferences, nutrition, cultural):
        """前向传播"""
        # 嵌入文化特征
        region_emb = self.region_embedding(cultural[:, 0].long())
        cuisine_emb = self.cuisine_embedding(cultural[:, 1].long())
        cultural_emb = torch.cat([region_emb, cuisine_emb], dim=1)

        # 组合特征
        combined_features = torch.cat([preferences, nutrition, cultural_emb], dim=1)

        # 特征融合
        x = self.feature_fusion(combined_features)

        # LoRA适配
        x = self.lora_adapter(x)

        # 输出预测
        output = self.output_head(x)

        return output.squeeze(-1)


class AdvancedCulturalModel(BaseModel):
    """高级文化适应模型（复杂架构）"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        # 多尺度嵌入
        self.region_embedding = nn.Embedding(9, config.hidden_dim // 2)
        self.cuisine_embedding = nn.Embedding(9, config.hidden_dim // 2)

        # 多分支特征提取器
        self.preference_branch = nn.Sequential(
            nn.Linear(10, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )

        self.nutrition_branch = nn.Sequential(
            nn.Linear(5, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )

        self.cultural_branch = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )

        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(config.hidden_dim // 2 * 3, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )

        # Transformer编码器
        encoder_layer = TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers
        )

        # LoRA适配器
        lora_config = LoRAConfig(
            r=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
            use_rslora=True,
            init_lora_weights=True
        )

        self.lora_adapter = LoRALinear(
            in_features=config.hidden_dim,
            out_features=config.hidden_dim,
            config=lora_config
        )

        # 输出头
        self.output_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.output_dim)
        )

    def forward(self, preferences, nutrition, cultural):
        """前向传播"""
        # 嵌入文化特征
        region_emb = self.region_embedding(cultural[:, 0].long())
        cuisine_emb = self.cuisine_embedding(cultural[:, 1].long())
        cultural_emb = torch.cat([region_emb, cuisine_emb], dim=1)

        # 多分支特征提取
        pref_features = self.preference_branch(preferences)
        nutr_features = self.nutrition_branch(nutrition)
        cult_features = self.cultural_branch(cultural_emb)

        # 特征融合
        fused_features = torch.cat([pref_features, nutr_features, cult_features], dim=1)
        x = self.feature_fusion(fused_features)

        # Transformer编码
        x_seq = x.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        x_encoded = self.transformer_encoder(x_seq)
        x = x_encoded.squeeze(1)  # 移除序列维度

        # LoRA适配
        x = self.lora_adapter(x)

        # 输出预测
        output = self.output_head(x)

        return output.squeeze(-1)


class ModelFactory:
    """模型工厂"""

    @staticmethod
    def create_model(model_type: str, config: ModelConfig) -> BaseModel:
        """创建模型"""
        if model_type == "optimized":
            return OptimizedCulturalModel(config)
        elif model_type == "efficient":
            return EfficientCulturalModel(config)
        elif model_type == "advanced":
            return AdvancedCulturalModel(config)
        else:
            raise ValueError(f"未知的模型类型: {model_type}")

    @staticmethod
    def get_recommended_config(model_type: str) -> ModelConfig:
        """获取推荐配置"""
        if model_type == "optimized":
            return ModelConfig(
                input_dim=10,
                hidden_dim=256,
                output_dim=1,
                lora_rank=16,
                lora_alpha=32,
                num_layers=3,
                num_heads=8,
                dropout=0.1,
                use_attention=True,
                use_residual=True
            )
        elif model_type == "efficient":
            return ModelConfig(
                input_dim=10,
                hidden_dim=128,
                output_dim=1,
                lora_rank=8,
                lora_alpha=16,
                num_layers=2,
                num_heads=4,
                dropout=0.1,
                use_attention=False,
                use_residual=False
            )
        elif model_type == "advanced":
            return ModelConfig(
                input_dim=10,
                hidden_dim=512,
                output_dim=1,
                lora_rank=32,
                lora_alpha=64,
                num_layers=6,
                num_heads=16,
                dropout=0.1,
                use_attention=True,
                use_residual=True
            )
        else:
            raise ValueError(f"未知的模型类型: {model_type}")


class ModelAnalyzer:
    """模型分析器"""

    @staticmethod
    def analyze_model(model: BaseModel) -> Dict[str, Any]:
        """分析模型"""
        # 获取模型大小
        size_info = model.get_model_size()

        # 计算FLOPs（简化估算）
        flops = ModelAnalyzer._estimate_flops(model)

        # 分析层结构
        layer_info = ModelAnalyzer._analyze_layers(model)

        return {
            "size_info": size_info,
            "estimated_flops": flops,
            "layer_info": layer_info,
            "model_type": type(model).__name__
        }

    @staticmethod
    def _estimate_flops(model: BaseModel) -> int:
        """估算FLOPs"""
        # 简化的FLOPs估算
        total_params = sum(p.numel() for p in model.parameters())
        # 假设每个参数对应2个FLOPs（乘加操作）
        return total_params * 2

    @staticmethod
    def _analyze_layers(model: BaseModel) -> Dict[str, int]:
        """分析层结构"""
        layer_counts = {
            "Linear": 0,
            "Embedding": 0,
            "LayerNorm": 0,
            "Dropout": 0,
            "LoRALinear": 0,
            "ResidualBlock": 0,
            "MultiHeadSelfAttention": 0
        }

        for module in model.modules():
            module_type = type(module).__name__
            if module_type in layer_counts:
                layer_counts[module_type] += 1

        return layer_counts


def main():
    """主函数"""
    logger.info("🚀 启动优化模型架构系统")

    # 测试不同模型类型
    model_types = ["efficient", "optimized", "advanced"]

    for model_type in model_types:
        logger.info(f"📊 测试 {model_type} 模型")

        # 获取配置
        config = ModelFactory.get_recommended_config(model_type)

        # 创建模型
        model = ModelFactory.create_model(model_type, config)

        # 分析模型
        analysis = ModelAnalyzer.analyze_model(model)

        # 打印分析结果
        logger.info(f"   模型类型: {analysis['model_type']}")
        logger.info(f"   总参数: {analysis['size_info']['total_params']:,}")
        logger.info(f"   可训练参数: {analysis['size_info']['trainable_params']:,}")
        logger.info(f"   模型大小: {analysis['size_info']['model_size_mb']:.2f} MB")
        logger.info(f"   估算FLOPs: {analysis['estimated_flops']:,}")

        # 测试前向传播
        batch_size = 32
        preferences = torch.randn(batch_size, 10)
        nutrition = torch.randn(batch_size, 5)
        cultural = torch.randint(0, 9, (batch_size, 2))

        with torch.no_grad():
            output = model(preferences, nutrition, cultural)
            logger.info(f"   输出形状: {output.shape}")

        logger.info("")

    logger.info("✅ 模型架构分析完成")


if __name__ == "__main__":
    main()
