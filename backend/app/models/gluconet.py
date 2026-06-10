"""
GlucoNet架构 - 最高级糖尿病管理AI系统
集成多尺度注意力网络(MAN)、策略感知注意力、多模态融合、知识蒸馏等所有先进功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass, field
import math
from enum import Enum

# 导入MAN模块
# from app.models.multi_scale_attention import (
#     MANConfig, MultiScaleAttentionNetwork, create_man_config
# )

logger = logging.getLogger(__name__)

class ModalityType(Enum):
    """模态类型枚举"""
    CGM = "cgm"
    INSULIN = "insulin"
    DIET = "diet"
    EXERCISE = "exercise"
    IMAGE = "image"
    TEXT = "text"
    TEMPORAL = "temporal"

class TaskType(Enum):
    """任务类型枚举"""
    GLUCOSE_PREDICTION = "glucose_prediction"
    INSULIN_SENSITIVITY = "insulin_sensitivity"
    CARB_ESTIMATION = "carb_estimation"
    EXERCISE_EFFECT = "exercise_effect"
    RECIPE_GENERATION = "recipe_generation"
    CULTURAL_CLASSIFICATION = "cultural_classification"

@dataclass
class GlucoNetConfig:
    """GlucoNet模型配置"""
    # 基础配置
    input_dim: int = 10
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.1
    num_heads: int = 8

    # 多模态配置
    cgm_dim: int = 128
    insulin_dim: int = 64
    diet_dim: int = 128
    exercise_dim: int = 64
    image_dim: int = 512
    text_dim: int = 256
    temporal_dim: int = 128

    # 知识图谱配置
    use_knowledge_graph: bool = True
    kg_feature_dim: int = 32  # 知识图谱特征维度
    kg_icr_dim: int = 8       # ICR特征维度
    kg_gi_dim: int = 8        # GI特征维度
    kg_risk_dim: int = 8      # 风险因子特征维度
    kg_drug_dim: int = 8      # 药物相互作用特征维度

    # 特征分解配置
    fd_k: int = 3
    fd_win: int = 32
    fd_stride: int = 16

    # 知识蒸馏配置
    teacher_hidden_dim: int = 512
    student_hidden_dim: int = 128
    temperature: float = 3.0
    alpha: float = 0.7

    # 策略感知配置
    clinical_strategy_dim: int = 64
    patient_profile_dim: int = 128

    # 多任务配置
    task_configs: List[Dict[str, Any]] = None

    # 联邦学习配置
    use_federated_learning: bool = False
    federated_rounds: int = 100

    # ONNX兼容性
    onnx_compatible: bool = True

    # MAN (Multi-scale Attention Network) 配置
    use_man: bool = True  # 是否使用MAN
    man_hidden_dim: int = 128
    man_num_scales: int = 4
    man_scale_factors: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    man_num_heads: int = 8
    man_dropout: float = 0.1

    def __post_init__(self):
        if self.task_configs is None:
            self.task_configs = [
                {
                    'name': 'glucose_prediction',
                    'type': 'regression',
                    'output_dim': 1,
                    'weight': 1.0,
                    'description': '血糖预测主任务'
                },
                {
                    'name': 'insulin_sensitivity',
                    'type': 'regression',
                    'output_dim': 1,
                    'weight': 0.3,
                    'description': '胰岛素敏感性评估'
                },
                {
                    'name': 'carb_estimation',
                    'type': 'regression',
                    'output_dim': 1,
                    'weight': 0.2,
                    'description': '碳水化合物估算'
                },
                {
                    'name': 'exercise_effect',
                    'type': 'regression',
                    'output_dim': 1,
                    'weight': 0.2,
                    'description': '运动效果量化'
                },
                {
                    'name': 'recipe_generation',
                    'type': 'generation',
                    'output_dim': 1000,
                    'weight': 0.1,
                    'description': '食谱生成'
                },
                {
                    'name': 'cultural_classification',
                    'type': 'classification',
                    'output_dim': 10,
                    'weight': 0.1,
                    'description': '文化分类'
                }
            ]

class TemporalFeatureDecomposition(nn.Module):
    """统一的时序特征分解模块"""

    def __init__(self, input_dim: int, k: int = 3, win: int = 32, stride: int = 16, onnx_compatible: bool = True):
        super().__init__()
        self.k = k
        self.win = win
        self.stride = stride
        self.onnx_compatible = onnx_compatible

        # 趋势分量提取器
        self.trend_extractor = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=win, stride=stride, padding=win//2),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # 周期分量提取器
        self.periodic_extractor = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=win//2, stride=stride//2, padding=win//4),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # 残差分量提取器
        self.residual_extractor = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=1),
            nn.ReLU()
        )

        # ONNX兼容的池化层
        if onnx_compatible:
            self.trend_pool = nn.AvgPool1d(kernel_size=2, stride=2)
            self.periodic_pool = nn.AvgPool1d(kernel_size=2, stride=2)
            self.residual_pool = nn.AvgPool1d(kernel_size=2, stride=2)
        else:
            self.trend_pool = None
            self.periodic_pool = None
            self.residual_pool = None

        # 特征融合
        self.fusion_layer = nn.Linear(96, input_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """统一的时序特征分解"""
        batch_size, seq_len, input_dim = x.shape

        # 转换为卷积输入格式
        x_conv = x.transpose(1, 2)  # [batch_size, input_dim, seq_len]

        # 提取各分量
        trend = self.trend_extractor(x_conv)
        periodic = self.periodic_extractor(x_conv)
        residual = self.residual_extractor(x_conv)

        # 确保所有分量长度一致
        if self.onnx_compatible:
            # 使用固定池化
            trend = self.trend_pool(trend)
            periodic = self.periodic_pool(periodic)
            residual = self.residual_pool(residual)
        else:
            # 使用自适应池化
            target_len = seq_len
            trend = F.adaptive_avg_pool1d(trend, target_len)
            periodic = F.adaptive_avg_pool1d(periodic, target_len)
            residual = F.adaptive_avg_pool1d(residual, target_len)

        # 确保所有分量具有相同的长度
        min_len = min(trend.size(-1), periodic.size(-1), residual.size(-1))
        trend = F.adaptive_avg_pool1d(trend, min_len)
        periodic = F.adaptive_avg_pool1d(periodic, min_len)
        residual = F.adaptive_avg_pool1d(residual, min_len)

        # 拼接特征
        decomposed_features = torch.cat([trend, periodic, residual], dim=1)
        decomposed_features = decomposed_features.transpose(1, 2)  # [batch_size, seq_len, 96]

        # 特征融合
        fused_features = self.fusion_layer(decomposed_features)

        # 返回分解结果
        decomposed_modes = {
            'trend': trend.transpose(1, 2),
            'periodic': periodic.transpose(1, 2),
            'residual': residual.transpose(1, 2)
        }

        return fused_features, decomposed_modes

class MultiScaleFeatureFusion(nn.Module):
    """多尺度特征融合模块"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()

        # 多尺度卷积
        self.conv1x1 = nn.Conv1d(input_dim, hidden_dim // 4, kernel_size=1)
        self.conv3x3 = nn.Conv1d(input_dim, hidden_dim // 4, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv1d(input_dim, hidden_dim // 4, kernel_size=5, padding=2)
        self.conv7x7 = nn.Conv1d(input_dim, hidden_dim // 4, kernel_size=7, padding=3)

        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 输出投影
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: [batch_size, seq_len, input_dim]
        Returns:
            output: [batch_size, seq_len, output_dim]
        """
        # 转换为卷积格式 [batch_size, input_dim, seq_len]
        x_conv = x.transpose(1, 2)

        # 多尺度特征提取
        conv1 = self.conv1x1(x_conv)
        conv3 = self.conv3x3(x_conv)
        conv5 = self.conv5x5(x_conv)
        conv7 = self.conv7x7(x_conv)

        # 拼接多尺度特征
        multi_scale_features = torch.cat([conv1, conv3, conv5, conv7], dim=1)

        # 转换回原格式 [batch_size, seq_len, hidden_dim]
        multi_scale_features = multi_scale_features.transpose(1, 2)

        # 特征融合
        fused_features = self.feature_fusion(multi_scale_features)

        # 输出投影
        output = self.output_projection(fused_features)

        return output

class QualityAwareFeatureFusion(nn.Module):
    """质量感知特征融合模块"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()

        # 质量权重网络
        self.quality_weight_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # 特征重要性网络
        self.feature_importance_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, input_dim),
            nn.Softmax(dim=-1)
        )

        # 多头注意力机制
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        # 多尺度特征融合
        self.multi_scale_fusion = MultiScaleFeatureFusion(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            dropout=dropout
        )

        # 特征交互层
        self.feature_interaction = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 残差连接
        self.residual_projection = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()

        # 输出投影
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, quality_scores: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        Args:
            x: [batch_size, seq_len, input_dim]
            quality_scores: [batch_size, seq_len, 1] 可选的质量评分
        Returns:
            output: [batch_size, seq_len, output_dim]
        """
        # 计算质量权重
        if quality_scores is not None:
            quality_weights = self.quality_weight_net(x) * quality_scores
        else:
            quality_weights = self.quality_weight_net(x)

        # 计算特征重要性
        feature_importance = self.feature_importance_net(x)

        # 应用质量权重和特征重要性
        weighted_x = x * quality_weights * feature_importance

        # 多头注意力
        attn_output, _ = self.multi_head_attention(weighted_x, weighted_x, weighted_x)

        # 多尺度特征融合
        multi_scale_output = self.multi_scale_fusion(weighted_x)

        # 特征融合
        fused_features = attn_output + multi_scale_output

        # 残差连接
        residual = self.residual_projection(x)
        x = fused_features + residual

        # 特征交互
        x = self.feature_interaction(x)

        # 输出投影
        output = self.output_projection(x)

        return output

class AdvancedFeatureFusion(nn.Module):
    """增强的高级特征融合模块"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()

        # 质量感知特征融合
        self.quality_aware_fusion = QualityAwareFeatureFusion(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout
        )

        # 特征选择网络
        self.feature_selector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, input_dim),
            nn.Sigmoid()
        )

        # 输出投影
        self.output_projection = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, quality_scores: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        Args:
            x: [batch_size, seq_len, input_dim]
            quality_scores: [batch_size, seq_len, 1] 可选的质量评分
        Returns:
            output: [batch_size, seq_len, output_dim]
        """
        # 特征选择
        feature_weights = self.feature_selector(x)
        selected_x = x * feature_weights

        # 质量感知特征融合
        fused_features = self.quality_aware_fusion(selected_x, quality_scores)

        # 输出投影
        output = self.output_projection(fused_features)

        return output

class KnowledgeGraphEncoder(nn.Module):
    """增强的知识图谱编码器"""

    def __init__(self, config: GlucoNetConfig):
        super().__init__()
        self.config = config

        # 基础特征编码器
        self.icr_encoder = nn.Sequential(
            nn.Linear(1, config.kg_icr_dim),
            nn.LayerNorm(config.kg_icr_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )

        self.gi_encoder = nn.Sequential(
            nn.Linear(1, config.kg_gi_dim),
            nn.LayerNorm(config.kg_gi_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )

        self.risk_encoder = nn.Sequential(
            nn.Linear(1, config.kg_risk_dim),
            nn.LayerNorm(config.kg_risk_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )

        self.drug_encoder = nn.Sequential(
            nn.Linear(1, config.kg_drug_dim),
            nn.LayerNorm(config.kg_drug_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )

        # 新增特征编码器
        self.diet_encoder = nn.Sequential(
            nn.Linear(1, config.kg_icr_dim // 2),
            nn.LayerNorm(config.kg_icr_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )

        self.exercise_encoder = nn.Sequential(
            nn.Linear(1, config.kg_icr_dim // 2),
            nn.LayerNorm(config.kg_icr_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )

        # 新增USDA和WHO特征编码器
        self.usda_encoder = nn.Sequential(
            nn.Linear(1, config.kg_icr_dim // 2),
            nn.LayerNorm(config.kg_icr_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )

        self.who_encoder = nn.Sequential(
            nn.Linear(1, config.kg_icr_dim // 2),
            nn.LayerNorm(config.kg_icr_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )

        # 特征融合维度
        fusion_input_dim = (config.kg_icr_dim + config.kg_gi_dim +
                           config.kg_risk_dim + config.kg_drug_dim +
                           config.kg_icr_dim // 2 + config.kg_icr_dim // 2 +
                           config.kg_icr_dim // 2 + config.kg_icr_dim // 2)

        # 高级特征融合
        self.advanced_fusion = AdvancedFeatureFusion(
            input_dim=fusion_input_dim,
            hidden_dim=config.kg_feature_dim * 2,
            output_dim=config.kg_feature_dim,
            dropout=config.dropout
        )

        # 最终输出层
        self.final_output = nn.Sequential(
            nn.Linear(config.kg_feature_dim, config.kg_feature_dim),
            nn.LayerNorm(config.kg_feature_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )

    def forward(self, kg_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            kg_features: [batch_size, 8] - [icr, avg_gi, risk_count, drug_count, diet_count, exercise_count, usda_count, who_count]
        Returns:
            kg_encoded: [batch_size, kg_feature_dim]
        """
        batch_size = kg_features.size(0)

        # 编码各个特征
        icr_encoded = self.icr_encoder(kg_features[:, 0:1])
        gi_encoded = self.gi_encoder(kg_features[:, 1:2])
        risk_encoded = self.risk_encoder(kg_features[:, 2:3])
        drug_encoded = self.drug_encoder(kg_features[:, 3:4])
        diet_encoded = self.diet_encoder(kg_features[:, 4:5])
        exercise_encoded = self.exercise_encoder(kg_features[:, 5:6])
        usda_encoded = self.usda_encoder(kg_features[:, 6:7])
        who_encoded = self.who_encoder(kg_features[:, 7:8])

        # 拼接所有特征
        kg_concat = torch.cat([
            icr_encoded, gi_encoded, risk_encoded,
            drug_encoded, diet_encoded, exercise_encoded,
            usda_encoded, who_encoded
        ], dim=1)

        # 添加序列维度用于注意力机制
        kg_concat = kg_concat.unsqueeze(1)  # [batch_size, 1, feature_dim]

        # 高级特征融合
        fused_features = self.advanced_fusion(kg_concat)

        # 移除序列维度
        fused_features = fused_features.squeeze(1)  # [batch_size, kg_feature_dim]

        # 最终输出
        kg_encoded = self.final_output(fused_features)

        return kg_encoded

class MultiModalAttention(nn.Module):
    """统一的多模态注意力机制"""

    def __init__(self, config: GlucoNetConfig):
        super().__init__()
        self.config = config

        # 模态编码器 - 动态处理维度
        self.modality_encoders = nn.ModuleDict({
            'cgm': nn.Linear(config.input_dim, config.hidden_dim),
            'insulin': nn.Linear(config.insulin_dim, config.hidden_dim),
            'diet': nn.Linear(config.diet_dim, config.hidden_dim),
            'exercise': nn.Linear(config.exercise_dim, config.hidden_dim),
            'image': nn.Linear(config.image_dim, config.hidden_dim),
            'text': nn.Linear(config.text_dim, config.hidden_dim),
            'temporal': nn.Linear(config.hidden_dim, config.hidden_dim)  # 修复：temporal已经是hidden_dim
        })

        # 知识图谱编码器
        if config.use_knowledge_graph:
            self.kg_encoder = KnowledgeGraphEncoder(config)

        # 自注意力模块
        self.self_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )

        # 交叉注意力模块
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )

        # 模态融合层 - 动态计算输入维度
        self.modality_fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2),  # 默认2个模态
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        )

    def forward(self, modalities: Dict[str, torch.Tensor],
                kg_features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """统一的多模态注意力处理"""
        # 模态编码
        encoded_modalities = {}
        for modality_name, modality_data in modalities.items():
            if modality_name in self.modality_encoders:
                encoded_modalities[modality_name] = self.modality_encoders[modality_name](modality_data)

        # 知识图谱特征编码
        if self.config.use_knowledge_graph and kg_features is not None:
            kg_encoded = self.kg_encoder(kg_features)
            # 将知识图谱特征扩展到序列维度
            seq_len = list(encoded_modalities.values())[0].size(1)
            kg_encoded = kg_encoded.unsqueeze(1).expand(-1, seq_len, -1)
            encoded_modalities['kg'] = kg_encoded

        # 堆叠所有模态
        modality_stack = torch.stack(list(encoded_modalities.values()), dim=1)

        # 自注意力处理
        self_attended, self_attention_weights = self.self_attention(
            modality_stack, modality_stack, modality_stack
        )

        # 交叉注意力处理
        cross_attended, cross_attention_weights = self.cross_attention(
            self_attended, self_attended, self_attended
        )

        # 模态融合 - 动态处理维度
        batch_size = cross_attended.shape[0]
        flattened_features = cross_attended.reshape(batch_size, -1)

        # 动态调整融合层输入维度
        input_dim = flattened_features.shape[1]
        if not hasattr(self, '_fusion_layer') or self._fusion_layer.in_features != input_dim:
            self._fusion_layer = nn.Sequential(
                nn.Linear(input_dim, self.config.hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim)
            ).to(flattened_features.device)

        fused_features = self._fusion_layer(flattened_features)

        # 返回结果
        attention_weights = {
            'self_attention': self_attention_weights,
            'cross_attention': cross_attention_weights
        }

        return fused_features, attention_weights

class StrategyAwareAttention(nn.Module):
    """统一的策略感知注意力模块"""

    def __init__(self, config: GlucoNetConfig):
        super().__init__()
        self.config = config

        # 临床策略嵌入
        self.clinical_strategy_embedding = nn.Embedding(
            num_embeddings=100,  # 假设有100种临床策略
            embedding_dim=config.clinical_strategy_dim
        )

        # 患者个性化偏置网络
        self.patient_bias_network = nn.Sequential(
            nn.Linear(config.patient_profile_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Tanh()
        )

        # 策略感知注意力
        self.strategy_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )

        # 策略融合层
        self.strategy_fusion = nn.Sequential(
            nn.Linear(config.hidden_dim + config.clinical_strategy_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

    def forward(self, features: torch.Tensor,
                clinical_strategy_id: torch.Tensor,
                patient_profile: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        策略感知注意力处理
        Args:
            features: 输入特征 [batch_size, seq_len, hidden_dim]
            clinical_strategy_id: 临床策略ID [batch_size]
            patient_profile: 患者画像 [batch_size, patient_profile_dim]
        Returns:
            策略感知特征和注意力权重
        """
        # 临床策略嵌入
        strategy_embedding = self.clinical_strategy_embedding(clinical_strategy_id)

        # 患者个性化偏置
        patient_bias = self.patient_bias_network(patient_profile)

        # 添加患者偏置到特征
        biased_features = features + patient_bias.unsqueeze(1)

        # 策略感知注意力
        attended_features, attention_weights = self.strategy_attention(
            biased_features, biased_features, biased_features
        )

        # 策略融合
        batch_size, seq_len, hidden_dim = attended_features.shape
        strategy_expanded = strategy_embedding.unsqueeze(1).expand(-1, seq_len, -1)
        fused_features = torch.cat([attended_features, strategy_expanded], dim=-1)
        strategy_aware_features = self.strategy_fusion(fused_features)

        return strategy_aware_features, attention_weights

class MultiTaskHead(nn.Module):
    """统一的多任务学习头"""

    def __init__(self, config: GlucoNetConfig):
        super().__init__()
        self.config = config

        # 任务特定头
        self.task_heads = nn.ModuleDict()
        for task_config in config.task_configs:
            task_name = task_config['name']
            output_dim = task_config['output_dim']
            task_type = task_config['type']

            if task_type == 'regression':
                self.task_heads[task_name] = nn.Sequential(
                    nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.hidden_dim // 2, output_dim)
                )
            elif task_type == 'classification':
                self.task_heads[task_name] = nn.Sequential(
                    nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.hidden_dim // 2, output_dim),
                    nn.Softmax(dim=-1)
                )
            elif task_type == 'generation':
                self.task_heads[task_name] = nn.Sequential(
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.hidden_dim, output_dim),
                    nn.Softmax(dim=-1)
                )

        # 任务权重学习
        self.task_weight_network = nn.Sequential(
            nn.Linear(config.hidden_dim, len(config.task_configs)),
            nn.Softmax(dim=-1)
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """统一的多任务预测"""
        task_outputs = {}
        task_weights = self.task_weight_network(features)

        for i, task_config in enumerate(self.config.task_configs):
            task_name = task_config['name']
            task_output = self.task_heads[task_name](features)
            task_outputs[task_name] = task_output
            task_outputs[f'{task_name}_weight'] = task_weights[:, i:i+1]

        return task_outputs

class KnowledgeDistillationModule(nn.Module):
    """统一的知识蒸馏模块"""

    def __init__(self, config: GlucoNetConfig):
        super().__init__()
        self.config = config

        # 教师模型（复杂）
        self.teacher_model = nn.Sequential(
            nn.Linear(config.hidden_dim, config.teacher_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.teacher_hidden_dim, config.teacher_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.teacher_hidden_dim, config.hidden_dim)
        )

        # 学生模型（轻量）
        self.student_model = nn.Sequential(
            nn.Linear(config.hidden_dim, config.student_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.student_hidden_dim, config.hidden_dim)
        )

        # 蒸馏损失计算
        self.temperature = config.temperature
        self.alpha = config.alpha

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """统一的知识蒸馏前向传播"""
        # 教师模型输出
        teacher_output = self.teacher_model(features)

        # 学生模型输出
        student_output = self.student_model(features)

        # 计算蒸馏损失
        teacher_soft = F.softmax(teacher_output / self.temperature, dim=-1)
        student_log_soft = F.log_softmax(student_output / self.temperature, dim=-1)
        distillation_loss = F.kl_div(student_log_soft, teacher_soft, reduction='batchmean')

        return teacher_output, student_output, distillation_loss

class MultiScaleAttentionModule(nn.Module):
    """多尺度注意力模块 - 直接集成到GlucoNet中"""

    def __init__(self, input_dim: int, hidden_dim: int, num_scales: int,
                 scale_factors: List[int], num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales
        self.scale_factors = scale_factors
        self.num_heads = num_heads

        # 多尺度卷积编码器
        self.scale_encoders = nn.ModuleList()
        for scale_factor in scale_factors:
            encoder = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=scale_factor,
                         padding=scale_factor//2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.scale_encoders.append(encoder)

        # 尺度感知注意力
        self.scale_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 尺度融合层
        self.scale_fusion = nn.Sequential(
            nn.Linear(hidden_dim * num_scales, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 尺度权重学习
        self.scale_weight_net = nn.Sequential(
            nn.Linear(hidden_dim, num_scales),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        多尺度注意力处理
        Args:
            x: 输入序列 [batch_size, seq_len, input_dim]
        Returns:
            包含多尺度特征的字典
        """
        batch_size, seq_len, _ = x.shape

        # 转换为卷积格式 [batch_size, input_dim, seq_len]
        x_conv = x.transpose(1, 2)

        # 多尺度特征提取
        scale_features = []
        for i, encoder in enumerate(self.scale_encoders):
            scale_feat = encoder(x_conv)  # [batch_size, hidden_dim, seq_len]

            # 确保长度一致
            if scale_feat.size(2) != seq_len:
                scale_feat = F.interpolate(scale_feat, size=seq_len, mode='linear', align_corners=False)

            # 转换回序列格式 [batch_size, seq_len, hidden_dim]
            scale_feat = scale_feat.transpose(1, 2)
            scale_features.append(scale_feat)

        # 拼接多尺度特征
        concatenated_features = torch.cat(scale_features, dim=-1)  # [batch_size, seq_len, hidden_dim * num_scales]

        # 尺度融合
        fused_features = self.scale_fusion(concatenated_features)  # [batch_size, seq_len, hidden_dim]

        # 尺度感知注意力
        attended_features, attention_weights = self.scale_attention(
            fused_features, fused_features, fused_features
        )

        # 计算尺度权重
        global_features = attended_features.mean(dim=1)  # [batch_size, hidden_dim]
        scale_weights = self.scale_weight_net(global_features)  # [batch_size, num_scales]

        return {
            'features': attended_features,
            'scale_features': scale_features,
            'attention_weights': attention_weights,
            'scale_weights': scale_weights
        }

class GlucoNet(nn.Module):
    """统一的GlucoNet主模型"""

    def __init__(self, config: GlucoNetConfig):
        super().__init__()
        self.config = config

        # 统一的时序特征分解
        self.temporal_decomposition = TemporalFeatureDecomposition(
            input_dim=config.input_dim,
            k=config.fd_k,
            win=config.fd_win,
            stride=config.fd_stride,
            onnx_compatible=config.onnx_compatible
        )

        # 统一的多模态注意力
        self.multimodal_attention = MultiModalAttention(config)

        # 统一的策略感知注意力
        self.strategy_aware_attention = StrategyAwareAttention(config)

        # 统一的多任务学习头
        self.multi_task_head = MultiTaskHead(config)

        # 统一的知识蒸馏模块
        self.knowledge_distillation = KnowledgeDistillationModule(config)

        # MAN (Multi-scale Attention Network) 模块 - 直接实现
        if config.use_man:
            self.man = MultiScaleAttentionModule(
                input_dim=config.input_dim,
                hidden_dim=config.man_hidden_dim,
                num_scales=config.man_num_scales,
                scale_factors=config.man_scale_factors,
                num_heads=config.man_num_heads,
                dropout=config.man_dropout
            )
            logger.info("MAN模块已集成到GlucoNet")
        else:
            self.man = None

        # 最终融合层
        fusion_input_dim = config.hidden_dim * 3 if config.use_man else config.hidden_dim * 2
        self.final_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

        logger.info("统一GlucoNet模型初始化完成")

    def forward(self,
                cgm_data: torch.Tensor,
                insulin_data: Optional[torch.Tensor] = None,
                diet_data: Optional[torch.Tensor] = None,
                exercise_data: Optional[torch.Tensor] = None,
                image_data: Optional[torch.Tensor] = None,
                text_data: Optional[torch.Tensor] = None,
                clinical_strategy_id: Optional[torch.Tensor] = None,
                patient_profile: Optional[torch.Tensor] = None,
                kg_features: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """统一的GlucoNet前向传播"""

        # 1. 统一的时序特征分解
        temporal_features, temporal_modes = self.temporal_decomposition(cgm_data)

        # 2. MAN多尺度注意力处理（如果启用）
        man_features = None
        man_output = None
        if self.man is not None:
            man_output = self.man(cgm_data)
            man_features = man_output['features'].mean(dim=1)  # [batch_size, man_hidden_dim]

        # 3. 统一的多模态注意力处理
        modalities = {
            'cgm': temporal_features.mean(dim=1),
            'temporal': temporal_features.mean(dim=1)
        }

        if insulin_data is not None:
            modalities['insulin'] = insulin_data
        if diet_data is not None:
            modalities['diet'] = diet_data
        if exercise_data is not None:
            modalities['exercise'] = exercise_data
        if image_data is not None:
            modalities['image'] = image_data
        if text_data is not None:
            modalities['text'] = text_data

        multimodal_features, attention_weights = self.multimodal_attention(modalities, kg_features)

        # 4. 策略感知注意力处理（如果提供了策略信息）
        if clinical_strategy_id is not None and patient_profile is not None:
            # 将multimodal_features转换为序列格式
            seq_features = multimodal_features.unsqueeze(1)  # [batch_size, 1, hidden_dim]
            strategy_aware_features, strategy_weights = self.strategy_aware_attention(
                seq_features, clinical_strategy_id, patient_profile
            )
            strategy_aware_features = strategy_aware_features.squeeze(1)  # [batch_size, hidden_dim]
        else:
            strategy_aware_features = multimodal_features
            strategy_weights = None

        # 5. 统一的知识蒸馏
        teacher_output, student_output, distillation_loss = self.knowledge_distillation(strategy_aware_features)

        # 6. 最终特征融合（包含MAN特征）
        if man_features is not None:
            final_features = torch.cat([strategy_aware_features, student_output, man_features], dim=-1)
        else:
            final_features = torch.cat([strategy_aware_features, student_output], dim=-1)
        final_features = self.final_fusion(final_features)

        # 5. 统一的多任务预测
        task_outputs = self.multi_task_head(final_features)

        # 6. 构建返回结果
        results = {
            'task_outputs': task_outputs,
            'teacher_output': teacher_output,
            'student_output': student_output,
            'distillation_loss': distillation_loss,
            'temporal_modes': temporal_modes,
            'attention_weights': attention_weights,
            'final_features': final_features
        }

        # 添加MAN相关信息
        if man_output is not None:
            results.update({
                'man_features': man_features,
                'man_scale_features': man_output['scale_features'],
                'man_attention_weights': man_output['attention_weights'],
                'man_scale_weights': man_output['scale_weights']
            })

        # 添加策略感知信息
        if strategy_weights is not None:
            results['strategy_weights'] = strategy_weights

        return results

def create_gluconet_model(config: Optional[GlucoNetConfig] = None) -> GlucoNet:
    """创建GlucoNet模型"""
    config = config or GlucoNetConfig()
    return GlucoNet(config)

def create_default_config() -> GlucoNetConfig:
    """创建默认配置"""
    return GlucoNetConfig()

# 导出主要类和函数
__all__ = [
    'GlucoNet',
    'GlucoNetConfig',
    'TemporalFeatureDecomposition',
    'MultiModalAttention',
    'StrategyAwareAttention',
    'MultiTaskHead',
    'KnowledgeDistillationModule',
    'MultiScaleAttentionModule',
    'create_gluconet_model',
    'create_default_config',
    'ModalityType',
    'TaskType'
]
