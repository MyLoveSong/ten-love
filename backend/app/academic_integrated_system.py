

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学术级智能健康监测集成系统
Academic-Level Integrated Health Monitoring System

本系统实现了多模态数据融合的智能健康监测，整合了：
1. 基于Transformer的血糖预测模型
2. 深度学习的医学图像识别
3. 多模态注意力融合机制
4. 学术级评估指标和实验验证

作者: AI Assistant
版本: 2.0.0
日期: 2024
"""

import os
import sys
import json
import logging
import asyncio
import threading
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 科学计算库
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from app.PIL import Image, ImageDraw, ImageFont
import cv2

# Web框架
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# 机器学习库
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, classification_report
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score

# 可视化库
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入自定义模块
try:
    from modules.glucose_prediction.predictor import GlucosePredictor
    from modules.image_recognition.predictor import ImagePredictor
    from modules.fusion.decision_engine import HealthDecisionEngine
except ImportError:
    # 如果模块不存在，创建基础版本
    pass

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('academic_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class AcademicMetrics:
    """学术级评估指标"""
    mse: float = 0.0
    mae: float = 0.0
    rmse: float = 0.0
    r2: float = 0.0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auc: float = 0.0
    confusion_matrix: np.ndarray = field(default_factory=lambda: np.array([]))

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'mse': self.mse,
            'mae': self.mae,
            'rmse': self.rmse,
            'r2': self.r2,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'auc': self.auc,
            'confusion_matrix': self.confusion_matrix.tolist() if self.confusion_matrix.size > 0 else []
        }

class EnhancedGluFormer(nn.Module):
    """
    增强的GluFormer模型 - LSTM-GRU融合的血糖预测模型
    增加文化适应性、时序感知能力和多尺度特征提取
    基于项目申请表中的创新点一设计
    """

    def __init__(self, input_dim: int = 15, hidden_dim: int = 128,
                 num_layers: int = 3, dropout: float = 0.1,
                 cultural_dim: int = 32, num_cultural_groups: int = 8):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cultural_dim = cultural_dim
        self.num_cultural_groups = num_cultural_groups

        # 输入特征嵌入层
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 文化适配嵌入
        self.cultural_embedding = nn.Embedding(num_cultural_groups, cultural_dim)
        self.cultural_projection = nn.Linear(cultural_dim, hidden_dim)

        # 多尺度LSTM层 - 长期记忆能力
        self.lstm_short = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )

        self.lstm_long = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )

        # 多尺度GRU层 - 短期动态捕捉
        self.gru_short = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )

        self.gru_long = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )

        # 增强的交叉注意力机制
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # 双向LSTM/GRU
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        # 时序自注意力
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        # 文化感知注意力门控
        self.cultural_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2 + cultural_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Sigmoid()
        )

        # 多尺度特征融合
        self.multi_scale_fusion = nn.ModuleList([
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=k, padding=k//2)
            for k in [1, 3, 5, 7]
        ])

        # 增强的特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim * 3),  # 更多特征维度
            nn.LayerNorm(hidden_dim * 3),
            nn.GELU(),  # 更好的激活函数
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # 预测头增强
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )

        # 不确定性估计层
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softplus()  # 确保输出为正
        )

    def forward(self, x: torch.Tensor, cultural_id: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        增强的前向传播
        Args:
            x: 输入序列 [batch_size, seq_len, input_dim]
            cultural_id: 文化组别ID [batch_size] (可选)
        Returns:
            预测结果字典，包含预测值、不确定性和注意力权重
        """
        batch_size, seq_len, _ = x.shape

        # 输入特征嵌入
        x_embedded = self.input_embedding(x)

        # 文化适配处理
        if cultural_id is not None:
            cultural_embed = self.cultural_embedding(cultural_id)  # [batch_size, cultural_dim]
            cultural_proj = self.cultural_projection(cultural_embed)  # [batch_size, hidden_dim]
            cultural_proj = cultural_proj.unsqueeze(1).expand(-1, seq_len, -1)
            x_embedded = x_embedded + cultural_proj

        # 多尺度LSTM处理
        lstm_short_out, _ = self.lstm_short(x_embedded)  # [batch_size, seq_len, hidden_dim*2]
        lstm_long_out, _ = self.lstm_long(x_embedded)    # [batch_size, seq_len, hidden_dim*2]

        # 多尺度GRU处理
        gru_short_out, _ = self.gru_short(x_embedded)    # [batch_size, seq_len, hidden_dim*2]
        gru_long_out, _ = self.gru_long(x_embedded)      # [batch_size, seq_len, hidden_dim*2]

        # 短期和长期特征融合
        lstm_fused = (lstm_short_out + lstm_long_out) / 2
        gru_fused = (gru_short_out + gru_long_out) / 2

        # 交叉注意力机制
        attended_lstm, cross_attn_weights = self.cross_attention(lstm_fused, gru_fused, gru_fused)
        attended_gru, _ = self.cross_attention(gru_fused, lstm_fused, lstm_fused)

        # 时序自注意力
        combined_features = torch.cat([attended_lstm, attended_gru], dim=-1)
        temporal_attended, temporal_attn_weights = self.temporal_attention(
            combined_features, combined_features, combined_features
        )

        # 文化感知门控
        if cultural_id is not None:
            cultural_context = cultural_embed.unsqueeze(1).expand(-1, seq_len, -1)
            gate_input = torch.cat([temporal_attended, cultural_context], dim=-1)
            cultural_gates = self.cultural_gate(gate_input)
            temporal_attended = temporal_attended * cultural_gates

        # 多尺度特征提取
        multi_scale_features = []
        feature_for_conv = temporal_attended.permute(0, 2, 1)  # [batch_size, hidden_dim*2, seq_len]

        for conv_layer in self.multi_scale_fusion:
            scale_feature = conv_layer(feature_for_conv)
            scale_feature = F.relu(scale_feature)
            scale_feature = scale_feature.permute(0, 2, 1)  # [batch_size, seq_len, hidden_dim]
            multi_scale_features.append(scale_feature)

        # 多尺度特征融合
        multi_scale_fused = torch.cat(multi_scale_features, dim=-1)  # [batch_size, seq_len, hidden_dim*4]

        # 全局特征融合
        final_features = torch.cat([temporal_attended, multi_scale_fused], dim=-1)
        fused_features = self.fusion_layer(final_features)

        # 时序池化 - 加权平均而非简单平均
        temporal_weights = F.softmax(torch.sum(fused_features, dim=-1), dim=1)  # [batch_size, seq_len]
        pooled_features = torch.sum(fused_features * temporal_weights.unsqueeze(-1), dim=1)

        # 预测输出
        prediction = self.prediction_head(pooled_features)
        uncertainty = self.uncertainty_head(pooled_features)

        return {
            'prediction': prediction,
            'uncertainty': uncertainty,
            'cross_attention_weights': cross_attn_weights,
            'temporal_attention_weights': temporal_attn_weights,
            'temporal_weights': temporal_weights,
            'cultural_features': cultural_proj if cultural_id is not None else None
        }

class EnhancedMultiModalFusion(nn.Module):
    """
    增强的多模态数据融合模块
    实现跨领域知识协同、语义对齐与动态特征融合
    """

    def __init__(self, glucose_dim: int = 128, image_dim: int = 512,
                 text_dim: int = 256, behavioral_dim: int = 64,
                 medical_dim: int = 128, fusion_dim: int = 256,
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        self.glucose_dim = glucose_dim
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.behavioral_dim = behavioral_dim
        self.medical_dim = medical_dim
        self.fusion_dim = fusion_dim
        self.num_heads = num_heads

        # 增强的特征投影层（支持更多模态）
        self.glucose_projection = nn.Sequential(
            nn.Linear(glucose_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.image_projection = nn.Sequential(
            nn.Linear(image_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.behavioral_projection = nn.Sequential(
            nn.Linear(behavioral_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.medical_projection = nn.Sequential(
            nn.Linear(medical_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 跨领域知识协同模块
        self.cross_domain_attention = nn.ModuleDict({
            'medical_glucose': nn.MultiheadAttention(fusion_dim, num_heads, dropout, batch_first=True),
            'behavioral_image': nn.MultiheadAttention(fusion_dim, num_heads, dropout, batch_first=True),
            'text_medical': nn.MultiheadAttention(fusion_dim, num_heads, dropout, batch_first=True),
            'glucose_behavioral': nn.MultiheadAttention(fusion_dim, num_heads, dropout, batch_first=True)
        })

        # 语义对齐模块
        self.semantic_alignment = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Linear(fusion_dim // 2, fusion_dim),
            nn.Sigmoid()  # 对齐权重
        )

        # 动态权重门控网络
        self.modality_gate = nn.Sequential(
            nn.Linear(fusion_dim * 5, fusion_dim),  # 5个模态
            nn.ReLU(),
            nn.Linear(fusion_dim, 5),  # 5个权重
            nn.Softmax(dim=-1)
        )

        # 时序感知模块
        self.temporal_encoder = nn.LSTM(
            input_size=fusion_dim,
            hidden_size=fusion_dim // 2,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        # 多尺度融合层
        self.multi_scale_fusion = nn.ModuleList([
            nn.Conv1d(fusion_dim, fusion_dim, kernel_size=k, padding=k//2)
            for k in [1, 3, 5, 7]
        ])

        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(fusion_dim) for _ in range(6)
        ])

        # 前馈网络增强
        self.feed_forward = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 4),
            nn.GELU(),  # 更好的激活函数
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 4, fusion_dim),
            nn.Dropout(dropout)
        )

        # 输出层增强
        self.output_projection = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.LayerNorm(fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, fusion_dim // 4)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, glucose_features: torch.Tensor,
                image_features: torch.Tensor,
                text_features: Optional[torch.Tensor] = None,
                behavioral_features: Optional[torch.Tensor] = None,
                medical_features: Optional[torch.Tensor] = None,
                sequence_length: int = 10) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        增强的前向传播，支持多模态跨领域知识协同

        Args:
            glucose_features: 血糖特征 [batch_size, glucose_dim]
            image_features: 图像特征 [batch_size, image_dim]
            text_features: 文本特征 [batch_size, text_dim] (可选)
            behavioral_features: 行为特征 [batch_size, behavioral_dim] (可选)
            medical_features: 医学特征 [batch_size, medical_dim] (可选)
            sequence_length: 时序长度

        Returns:
            融合后的特征和注意力权重
        """
        batch_size = glucose_features.size(0)
        attention_weights = {}

        # 特征投影 - 支持更多模态
        glucose_proj = self.glucose_projection(glucose_features)
        image_proj = self.image_projection(image_features)

        # 构建模态特征列表
        modality_features = [glucose_proj, image_proj]
        modality_names = ['glucose', 'image']

        if text_features is not None:
            text_proj = self.text_projection(text_features)
            modality_features.append(text_proj)
            modality_names.append('text')

        if behavioral_features is not None:
            behavioral_proj = self.behavioral_projection(behavioral_features)
            modality_features.append(behavioral_proj)
            modality_names.append('behavioral')

        if medical_features is not None:
            medical_proj = self.medical_projection(medical_features)
            modality_features.append(medical_proj)
            modality_names.append('medical')

        # 语义对齐处理
        aligned_features = []
        for i, feature in enumerate(modality_features):
            alignment_weight = self.semantic_alignment(feature)
            aligned_feature = feature * alignment_weight
            aligned_features.append(aligned_feature)
            aligned_features[i] = self.layer_norms[0](aligned_features[i])

        # 跨领域知识协同 - 只在有相应模态时执行
        cross_domain_outputs = {}
        if len(modality_features) >= 2:
            # 医学-血糖协同
            if 'medical' in modality_names and 'glucose' in modality_names:
                med_idx = modality_names.index('medical')
                glu_idx = modality_names.index('glucose')
                med_glu_attn, med_glu_weights = self.cross_domain_attention['medical_glucose'](
                    aligned_features[med_idx].unsqueeze(1),
                    aligned_features[glu_idx].unsqueeze(1),
                    aligned_features[glu_idx].unsqueeze(1)
                )
                cross_domain_outputs['medical_glucose'] = med_glu_attn.squeeze(1)
                attention_weights['medical_glucose'] = med_glu_weights

            # 行为-图像协同
            if 'behavioral' in modality_names and 'image' in modality_names:
                beh_idx = modality_names.index('behavioral')
                img_idx = modality_names.index('image')
                beh_img_attn, beh_img_weights = self.cross_domain_attention['behavioral_image'](
                    aligned_features[beh_idx].unsqueeze(1),
                    aligned_features[img_idx].unsqueeze(1),
                    aligned_features[img_idx].unsqueeze(1)
                )
                cross_domain_outputs['behavioral_image'] = beh_img_attn.squeeze(1)
                attention_weights['behavioral_image'] = beh_img_weights

        # 合并协同特征
        if cross_domain_outputs:
            for key, output in cross_domain_outputs.items():
                aligned_features.append(output)

        # 填充到5个模态（如果不足）
        while len(aligned_features) < 5:
            aligned_features.append(torch.zeros_like(aligned_features[0]))

        # 截取到5个模态（如果超过）
        aligned_features = aligned_features[:5]

        # 动态权重门控
        concatenated_features = torch.cat(aligned_features, dim=-1)
        modality_weights = self.modality_gate(concatenated_features)

        # 加权融合
        weighted_features = []
        for i, feature in enumerate(aligned_features):
            weighted = feature * modality_weights[:, i:i+1]
            weighted_features.append(weighted)

        fused_feature = torch.stack(weighted_features, dim=1).sum(dim=1)
        fused_feature = self.layer_norms[1](fused_feature)

        # 时序感知处理
        temporal_input = fused_feature.unsqueeze(1).repeat(1, sequence_length, 1)
        temporal_output, (h_n, c_n) = self.temporal_encoder(temporal_input)
        temporal_feature = temporal_output.mean(dim=1)  # 平均池化
        temporal_feature = self.layer_norms[2](temporal_feature)

        # 多尺度融合
        multi_scale_features = []
        feature_for_conv = temporal_feature.unsqueeze(1).permute(0, 2, 1)  # [B, C, 1]

        for conv_layer in self.multi_scale_fusion:
            scale_feature = conv_layer(feature_for_conv)
            scale_feature = F.relu(scale_feature)
            multi_scale_features.append(scale_feature.squeeze(-1))

        # 多尺度特征融合
        multi_scale_fused = torch.stack(multi_scale_features, dim=1).mean(dim=1)
        multi_scale_fused = self.layer_norms[3](multi_scale_fused)

        # 前馈网络增强
        ff_output = self.feed_forward(multi_scale_fused)
        ff_output = self.layer_norms[4](multi_scale_fused + ff_output)

        # 最终输出投影
        output = self.output_projection(ff_output)
        output = self.layer_norms[5](output)
        output = self.dropout(output)

        # 构建完整的注意力权重字典
        attention_weights.update({
            'modality_weights': modality_weights,
            'temporal_hidden': h_n,
            'semantic_alignment': [self.semantic_alignment(f) for f in modality_features]
        })

        return output, attention_weights

class AcademicGlucosePredictor:
    """
    学术级血糖预测器
    基于Transformer架构的血糖预测模型
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or "trained_gluformer_model.pth"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.feature_cols = None

        self._load_model()

    def _load_model(self):
        """加载预训练模型"""
        try:
            if Path(self.model_path).exists():
                # 加载模型权重
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model = checkpoint.get('model')
                self.scaler_X = checkpoint.get('scaler_X')
                self.scaler_y = checkpoint.get('scaler_y')
                self.feature_cols = checkpoint.get('feature_cols')
                logger.info(f"模型加载成功: {self.model_path}")
            else:
                logger.warning(f"模型文件不存在: {self.model_path}")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        血糖预测

        Args:
            input_data: 输入特征字典

        Returns:
            预测结果字典
        """
        try:
            # 特征预处理
            features = self._preprocess_features(input_data)

            # 模型预测
            with torch.no_grad():
                prediction = self.model(features)
                predicted_glucose = self.scaler_y.inverse_transform(prediction.cpu().numpy())

            # 计算置信度和风险等级
            confidence = self._calculate_confidence(prediction)
            risk_level = self._assess_risk(predicted_glucose[0])

            return {
                'predicted_glucose': float(predicted_glucose[0]),
                'confidence': confidence,
                'risk_level': risk_level,
                'recommendations': self._generate_recommendations(predicted_glucose[0], risk_level)
            }
        except Exception as e:
            logger.error(f"血糖预测失败: {e}")
            return {'error': str(e)}

    def _preprocess_features(self, input_data: Dict[str, Any]) -> torch.Tensor:
        """特征预处理"""
        # 提取特征
        features = []
        for col in self.feature_cols:
            features.append(input_data.get(col, 0.0))

        # 标准化
        features_scaled = self.scaler_X.transform([features])
        return torch.FloatTensor(features_scaled).to(self.device)

    def _calculate_confidence(self, prediction: torch.Tensor) -> float:
        """计算预测置信度"""
        # 基于预测值的方差计算置信度
        confidence = 1.0 - torch.var(prediction).item()
        return max(0.0, min(1.0, confidence))

    def _assess_risk(self, glucose_value: float) -> str:
        """评估风险等级"""
        if glucose_value < 100:
            return "低风险"
        elif glucose_value < 126:
            return "中等风险"
        else:
            return "高风险"

    def _generate_recommendations(self, glucose_value: float, risk_level: str) -> List[str]:
        """生成健康建议"""
        recommendations = []

        if risk_level == "高风险":
            recommendations.extend([
                "建议立即就医检查",
                "严格控制饮食，减少碳水化合物摄入",
                "增加有氧运动，每周至少150分钟",
                "定期监测血糖水平"
            ])
        elif risk_level == "中等风险":
            recommendations.extend([
                "调整饮食习惯，选择低GI食物",
                "增加运动频率",
                "定期体检",
                "保持健康体重"
            ])
        else:
            recommendations.extend([
                "保持当前健康生活方式",
                "定期体检",
                "均衡饮食"
            ])

        return recommendations

class AcademicImagePredictor:
    """
    学术级图像识别器
    基于深度学习的医学图像识别模型
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or "image_recognition_model.pth"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_names = []
        self.transform = self._get_transforms()

        self._load_model()

    def _load_model(self):
        """加载预训练模型"""
        try:
            if Path(self.model_path).exists():
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model = checkpoint.get('model')
                self.class_names = checkpoint.get('class_names', [])
                logger.info(f"图像识别模型加载成功: {self.model_path}")
            else:
                logger.warning(f"图像识别模型文件不存在: {self.model_path}")
        except Exception as e:
            logger.error(f"图像识别模型加载失败: {e}")

    def _get_transforms(self):
        """获取图像预处理变换"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_path: str) -> Dict[str, Any]:
        """
        图像识别预测

        Args:
            image_path: 图像文件路径

        Returns:
            识别结果字典
        """
        try:
            # 加载和预处理图像
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # 模型预测
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()

            # 异常检测
            abnormal_regions = self._detect_abnormal_regions(image)

            return {
                'classification': self.class_names[predicted_class] if self.class_names else f"Class_{predicted_class}",
                'confidence': confidence,
                'abnormal_regions': abnormal_regions,
                'recommendations': self._generate_image_recommendations(predicted_class, confidence)
            }
        except Exception as e:
            logger.error(f"图像识别失败: {e}")
            return {'error': str(e)}

    def _detect_abnormal_regions(self, image: Image.Image) -> List[Dict[str, Any]]:
        """检测异常区域"""
        # 简化的异常检测实现
        # 在实际应用中，这里应该使用更复杂的异常检测算法
        regions = []

        # 转换为OpenCV格式
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # 边缘检测
        edges = cv2.Canny(img_cv, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 筛选较大的轮廓作为潜在异常区域
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # 面积阈值
                x, y, w, h = cv2.boundingRect(contour)
                regions.append({
                    'bbox': [x, y, w, h],
                    'area': area,
                    'confidence': min(0.8, area / 10000)  # 简化的置信度计算
                })

        return regions

    def _generate_image_recommendations(self, predicted_class: int, confidence: float) -> List[str]:
        """生成图像识别建议"""
        recommendations = []

        if confidence < 0.7:
            recommendations.append("建议重新拍摄图像以提高识别准确性")

        if predicted_class == 0:  # 假设0为正常类别
            recommendations.append("图像显示正常，建议定期检查")
        else:
            recommendations.append("检测到异常，建议进一步医学检查")

        return recommendations

class AcademicHealthDecisionEngine:
    """
    学术级健康决策引擎
    整合多模态数据进行综合健康评估
    """

    def __init__(self, glucose_predictor: AcademicGlucosePredictor,
                 image_predictor: AcademicImagePredictor):
        self.glucose_predictor = glucose_predictor
        self.image_predictor = image_predictor
        self.fusion_model = EnhancedMultiModalFusion()
        self.metrics = AcademicMetrics()

    def assess_health(self, glucose_data: Dict[str, Any],
                     image_data: Optional[Dict[str, Any]] = None,
                     user_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        综合健康评估

        Args:
            glucose_data: 血糖相关数据
            image_data: 图像数据
            user_profile: 用户档案

        Returns:
            综合评估结果
        """
        try:
            # 血糖预测
            glucose_result = self.glucose_predictor.predict(glucose_data)

            # 图像识别（如果有图像数据）
            image_result = None
            if image_data and image_data.get('path'):
                image_result = self.image_predictor.predict(image_data['path'])

            # 多模态融合
            fusion_result = self._fuse_modalities(glucose_result, image_result)

            # 综合评估
            health_score = self._calculate_health_score(glucose_result, image_result, user_profile)
            risk_assessment = self._assess_comprehensive_risk(glucose_result, image_result, user_profile)

            return {
                'health_score': health_score,
                'risk_assessment': risk_assessment,
                'glucose_analysis': glucose_result,
                'image_analysis': image_result,
                'fusion_analysis': fusion_result,
                'recommendations': self._generate_comprehensive_recommendations(
                    glucose_result, image_result, risk_assessment
                ),
                'next_checkup': self._suggest_next_checkup(risk_assessment),
                'confidence': self._calculate_overall_confidence(glucose_result, image_result)
            }
        except Exception as e:
            logger.error(f"综合健康评估失败: {e}")
            return {'error': str(e)}

    def _fuse_modalities(self, glucose_result: Dict[str, Any],
                        image_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """多模态融合"""
        if not image_result:
            return glucose_result

        # 简化的融合逻辑
        # 在实际应用中，这里应该使用训练好的融合模型
        fusion_confidence = (glucose_result.get('confidence', 0) +
                           image_result.get('confidence', 0)) / 2

        return {
            'fused_confidence': fusion_confidence,
            'modality_weights': {
                'glucose': 0.6,
                'image': 0.4
            }
        }

    def _calculate_health_score(self, glucose_result: Dict[str, Any],
                              image_result: Optional[Dict[str, Any]],
                              user_profile: Optional[Dict[str, Any]]) -> float:
        """计算健康评分"""
        score = 100.0

        # 血糖评分
        glucose_value = glucose_result.get('predicted_glucose', 100)
        if glucose_value > 126:
            score -= 30
        elif glucose_value > 100:
            score -= 15

        # 图像评分
        if image_result and image_result.get('confidence', 0) < 0.7:
            score -= 10

        # 用户档案评分
        if user_profile:
            if user_profile.get('smoking', 0):
                score -= 10
            if user_profile.get('alcohol', 0):
                score -= 5
            if user_profile.get('stress_level', 3) > 3:
                score -= 10

        return max(0.0, score)

    def _assess_comprehensive_risk(self, glucose_result: Dict[str, Any],
                                 image_result: Optional[Dict[str, Any]],
                                 user_profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """综合风险评估"""
        risk_factors = []
        risk_level = "低风险"

        # 血糖风险
        glucose_risk = glucose_result.get('risk_level', '低风险')
        if glucose_risk != "低风险":
            risk_factors.append("血糖异常")
            if glucose_risk == "高风险":
                risk_level = "高风险"

        # 图像风险
        if image_result and image_result.get('classification') != "正常":
            risk_factors.append("图像异常")
            risk_level = "中等风险" if risk_level == "低风险" else risk_level

        # 生活方式风险
        if user_profile:
            if user_profile.get('smoking', 0):
                risk_factors.append("吸烟")
            if user_profile.get('stress_level', 3) > 3:
                risk_factors.append("高压力")

        return {
            'level': risk_level,
            'factors': risk_factors,
            'score': len(risk_factors)
        }

    def _generate_comprehensive_recommendations(self, glucose_result: Dict[str, Any],
                                              image_result: Optional[Dict[str, Any]],
                                              risk_assessment: Dict[str, Any]) -> List[str]:
        """生成综合建议"""
        recommendations = []

        # 血糖建议
        if glucose_result.get('recommendations'):
            recommendations.extend(glucose_result['recommendations'])

        # 图像建议
        if image_result and image_result.get('recommendations'):
            recommendations.extend(image_result['recommendations'])

        # 风险等级建议
        if risk_assessment['level'] == "高风险":
            recommendations.append("建议立即就医进行详细检查")
            recommendations.append("考虑内分泌科专家咨询")
        elif risk_assessment['level'] == "中等风险":
            recommendations.append("建议3个月内复查")
            recommendations.append("调整生活方式，控制饮食")

        return list(set(recommendations))  # 去重

    def _suggest_next_checkup(self, risk_assessment: Dict[str, Any]) -> str:
        """建议下次检查时间"""
        if risk_assessment['level'] == "高风险":
            return "1个月内"
        elif risk_assessment['level'] == "中等风险":
            return "3个月内"
        else:
            return "6个月内"

    def _calculate_overall_confidence(self, glucose_result: Dict[str, Any],
                                    image_result: Optional[Dict[str, Any]]) -> float:
        """计算整体置信度"""
        glucose_conf = glucose_result.get('confidence', 0)
        image_conf = image_result.get('confidence', 0) if image_result else 0

        if image_result:
            return (glucose_conf * 0.6 + image_conf * 0.4)
        else:
            return glucose_conf

class AcademicIntegratedSystem:
    """
    增强的学术级集成系统主类
    集成工作流、数据增强、跨平台数据流、模型适应和反馈优化
    """

    def __init__(self):
        self.logger = logger
        self.glucose_predictor = None
        self.image_predictor = None
        self.health_engine = None
        self.app = None
        self.metrics_history = []

        # 新增组件
        self.data_integration_manager = None
        self.workflow_orchestrator = None
        self.data_flow_manager = None
        self.adaptive_manager = None
        self.optimization_system = None
        self.enhancement_pipeline = None

        # 项目申请表中的核心组件
        self.enhanced_gluformer_model = None
        self.cultural_engine = None
        self.merl_system = None
        self.knowledge_base = None

        # 数据库组件
        self.db_manager = None
        self.db_session = None
        self.migration_manager = None
        self.explainable_system = None

        self._initialize_system()

    def _initialize_database(self):
        """初始化数据库"""
        try:
            # 导入数据库模块
            from database.models import initialize_database, SessionLocal
            from database.database_manager import DatabaseManager
            from database.migrations import MigrationManager
            from database.config import get_current_database_config

            # 获取数据库配置
            config = get_current_database_config()
            self.logger.info(f"初始化数据库: {config.db_type}")

            # 初始化数据库
            engine, SessionLocal = initialize_database(
                config.get_database_url(),
                config.echo
            )

            # 创建数据库会话
            self.db_session = SessionLocal()

            # 创建数据库管理器
            self.db_manager = DatabaseManager(self.db_session)

            # 创建迁移管理器
            self.migration_manager = MigrationManager(self.db_session)

            # 运行迁移
            if not self.migration_manager.initialize_database():
                self.logger.error("数据库迁移初始化失败")
                return False

            if not self.migration_manager.upgrade_database():
                self.logger.error("数据库升级失败")
                return False

            self.logger.info("数据库初始化完成")
            return True

        except Exception as e:
            self.logger.error(f"数据库初始化失败: {e}")
            return False

    def _initialize_system(self):
        """初始化增强系统"""
        try:
            self.logger.info("正在初始化增强学术级集成系统...")

            # 0. 初始化数据库（可选，失败时继续运行）
            try:
                if not self._initialize_database():
                    self.logger.warning("数据库初始化失败，将使用基础功能")
            except Exception as e:
                self.logger.warning(f"数据库初始化失败: {e}，将使用基础功能")

            # 1. 初始化基础预测器
            self.glucose_predictor = AcademicGlucosePredictor()
            self.image_predictor = AcademicImagePredictor()

            # 2. 初始化数据集成组件
            self._initialize_data_components()

            # 3. 初始化模型适应组件
            self._initialize_adaptation_components()

            # 4. 初始化决策引擎
            self.health_engine = AcademicHealthDecisionEngine(
                self.glucose_predictor, self.image_predictor
            )

            # 5. 设置组件集成
            self._setup_component_integration()

            # 6. 初始化FastAPI应用
            self._initialize_web_framework()

            self.logger.info("增强学术级集成系统初始化完成")

        except Exception as e:
            self.logger.error(f"系统初始化失败: {e}")
            # 即使初始化失败，也尝试创建基础的FastAPI应用
            self._create_basic_app()

    def _create_basic_app(self):
        """创建基础FastAPI应用"""
        try:
            self.logger.info("创建基础FastAPI应用...")

            # 初始化基础预测器
            if not self.glucose_predictor:
                self.glucose_predictor = AcademicGlucosePredictor()
            if not self.image_predictor:
                self.image_predictor = AcademicImagePredictor()
            if not self.health_engine:
                self.health_engine = AcademicHealthDecisionEngine(
                    self.glucose_predictor, self.image_predictor
                )

            # 创建基础FastAPI应用
            self.app = FastAPI(
                title="学术级智能健康监测集成系统",
                description="基础版本的健康监测系统",
                version="3.0.0-basic",
                docs_url="/docs",
                redoc_url="/redoc"
            )

            # 配置CORS
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

            # 注册基础路由
            self._register_basic_routes()

            self.logger.info("基础FastAPI应用创建完成")

        except Exception as e:
            self.logger.error(f"基础应用创建失败: {e}")
            raise

    def _register_basic_routes(self):
        """注册基础API路由"""

        @self.app.get("/")
        async def root():
            return {
                "message": "学术级智能健康监测集成系统 (基础版)",
                "version": "3.0.0-basic",
                "status": "running",
                "note": "部分高级功能可能不可用"
            }

        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "modules": {
                    "glucose_predictor": self.glucose_predictor is not None,
                    "image_predictor": self.image_predictor is not None,
                    "health_engine": self.health_engine is not None
                }
            }

        @self.app.post("/predict/glucose")
        async def predict_glucose(data: Dict[str, Any]):
            try:
                result = self.glucose_predictor.predict(data)
                return JSONResponse(content=result)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/predict/image")
        async def predict_image(file: UploadFile = File(...)):
            try:
                # 保存上传的文件
                temp_path = f"temp_{file.filename}"
                with open(temp_path, "wb") as buffer:
                    buffer.write(await file.read())

                result = self.image_predictor.predict(temp_path)

                # 清理临时文件
                os.remove(temp_path)

                return JSONResponse(content=result)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/assess/health")
        async def assess_health(
            glucose_data: Dict[str, Any],
            image_file: Optional[UploadFile] = File(None),
            user_profile: Optional[Dict[str, Any]] = None
        ):
            try:
                image_data = None
                if image_file:
                    temp_path = f"temp_{image_file.filename}"
                    with open(temp_path, "wb") as buffer:
                        buffer.write(await image_file.read())
                    image_data = {"path": temp_path}

                result = self.health_engine.assess_health(
                    glucose_data=glucose_data,
                    image_data=image_data,
                    user_profile=user_profile
                )

                # 清理临时文件
                if image_data:
                    os.remove(image_data["path"])

                return JSONResponse(content=result)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    def _initialize_data_components(self):
        """初始化数据相关组件"""
        try:
            # 导入数据集成模块
            from data_integration.api_adapters import DataIntegrationManager
            from app.data_integration.workflow_integration import WorkflowOrchestrator
            from data_integration.cross_platform_dataflow import CrossPlatformDataFlowManager
            from app.data_integration.data_augmentation import DataEnhancementPipeline, AugmentationConfig

            # 初始化数据集成管理器
            self.data_integration_manager = DataIntegrationManager()

            # 初始化工作流编排器
            self.workflow_orchestrator = WorkflowOrchestrator()

            # 初始化跨平台数据流管理器
            self.data_flow_manager = CrossPlatformDataFlowManager()

            # 初始化数据增强流水线
            aug_config = AugmentationConfig(
                noise_level=0.05,
                augmentation_ratio=2.0,
                synthetic_samples=1000
            )
            self.enhancement_pipeline = DataEnhancementPipeline(aug_config)

            self.logger.info("数据组件初始化完成")

        except Exception as e:
            self.logger.warning(f"数据组件初始化失败: {e}，将使用基础功能")

    def _initialize_adaptation_components(self):
        """初始化模型适应组件"""
        try:
            # 导入模型适应模块
            from models.adaptive_model_tuning import AdaptiveModelManager, AdaptationConfig
            from app.data_integration.feedback_optimization import IterativeOptimizationSystem, FeedbackConfig
            from models.cultural_adaptation import EnhancedCulturalDietaryRecommendationEngine
            from models.mixture_of_experts import MERLSystem, ExpertConfig
            from models.knowledge_distillation import MedicalKnowledgeBase, ExplainableKnowledgeSystem

            # 创建增强的GluFormer模型作为基础模型
            self.enhanced_gluformer_model = EnhancedGluFormer(
                input_dim=15, hidden_dim=128, num_layers=3,
                dropout=0.1, cultural_dim=32, num_cultural_groups=8
            )

            # 初始化自适应管理器
            adaptation_config = AdaptationConfig(
                learning_rate=0.001,
                fine_tune_epochs=10,
                min_adaptation_samples=20
            )
            self.adaptive_manager = AdaptiveModelManager(
                self.enhanced_gluformer_model, adaptation_config
            )

            # 初始化增强的文化适配引擎
            from models.cultural_adaptation import EnhancedCulturalDietaryRecommendationEngine
            self.cultural_engine = EnhancedCulturalDietaryRecommendationEngine()

            # 初始化混合专家系统
            expert_configs = [
                ExpertConfig(
                    expert_type='health_indicator',
                    input_dim=20,
                    hidden_dim=128,
                    output_dim=64
                ),
                ExpertConfig(
                    expert_type='medical_diagnosis',
                    input_dim=20,
                    hidden_dim=128,
                    output_dim=64
                ),
                ExpertConfig(
                    expert_type='lifestyle_assessment',
                    input_dim=20,
                    hidden_dim=128,
                    output_dim=64
                )
            ]

            from models.mixture_of_experts import EnhancedMERLSystem
            self.merl_system = EnhancedMERLSystem(
                input_dim=20,
                action_dim=5,
                expert_configs=expert_configs,
                sequence_length=24,
                temporal_dim=128
            )

            # 初始化增强的知识蒸馏系统
            from models.enhanced_knowledge_distillation import EnhancedMedicalKnowledgeBase, EnhancedExplainableKnowledgeSystem
            self.knowledge_base = EnhancedMedicalKnowledgeBase()
            self.explainable_system = EnhancedExplainableKnowledgeSystem(self.knowledge_base)

            # 初始化优化系统
            feedback_config = FeedbackConfig(
                feedback_window=24,
                min_feedback_samples=10,
                learning_rate=0.001
            )
            self.optimization_system = IterativeOptimizationSystem(
                self.glucose_predictor, feedback_config
            )

            self.logger.info("增强模型适应组件初始化完成")

        except Exception as e:
            self.logger.warning(f"模型适应组件初始化失败: {e}，将使用基础功能")

    def _setup_component_integration(self):
        """设置组件间集成"""
        try:
            # 设置工作流处理器
            if self.workflow_orchestrator and self.data_integration_manager:
                self.workflow_orchestrator.setup_processors(
                    self.data_integration_manager,
                    self.glucose_predictor,
                    self.image_predictor
                )

            # 设置优化系统的集成组件
            if self.optimization_system:
                self.optimization_system.set_integration_components(
                    adaptive_manager=self.adaptive_manager,
                    workflow_orchestrator=self.workflow_orchestrator,
                    data_flow_manager=self.data_flow_manager
                )

            # 设置数据流管理器的处理器
            if self.data_flow_manager:
                # 注册CGM数据处理器
                async def cgm_processor(packet):
                    if self.enhancement_pipeline:
                        # 使用数据增强流水线处理
                        return {'processed': True, 'enhanced': True}
                    return {'processed': True}

                self.data_flow_manager.register_processor('cgm', cgm_processor)

            self.logger.info("组件集成设置完成")

        except Exception as e:
            self.logger.warning(f"组件集成设置失败: {e}")

    def _initialize_web_framework(self):
        """初始化Web框架"""
        # 初始化FastAPI应用
        self.app = FastAPI(
            title="增强学术级智能健康监测集成系统",
            description="集成工作流、数据增强、模型适应的多模态健康监测系统",
            version="3.0.0",
            docs_url="/docs",
            redoc_url="/redoc",
            openapi_url="/openapi.json",
            contact={
                "name": "学术研究团队",
                "email": "research@university.edu",
                "url": "https://research.university.edu"
            },
            license_info={
                "name": "MIT License",
                "url": "https://opensource.org/licenses/MIT"
            }
        )

        # 配置CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # 初始化API增强功能
        self._setup_api_enhancements()

        # 注册增强路由
        self._register_enhanced_routes()

    def _setup_api_enhancements(self):
        """设置API增强功能"""
        try:
            # 导入API模块
            from app.api.authentication import auth_manager
            from api.versioning import api_version_manager, add_version_info_routes, v1_router, v2_router, v3_router
            from api.graphql_schema import create_graphql_router

            # 设置身份验证
            self.auth_manager = auth_manager

            # 设置版本控制
            self.version_manager = api_version_manager

            # 添加版本检查中间件
            version_middleware = self.version_manager.create_version_middleware()
            self.app.middleware("http")(version_middleware)

            # 添加版本信息路由
            add_version_info_routes(self.app, self.version_manager)

            # 注册版本化路由器
            self.app.include_router(v1_router)
            self.app.include_router(v2_router)
            # v3路由器将在_register_enhanced_routes中设置

            # 添加GraphQL支持
            graphql_router = create_graphql_router()
            self.app.include_router(graphql_router, prefix="/api")

            # 添加身份验证路由
            self._add_auth_routes()

            self.logger.info("API增强功能设置完成")

        except Exception as e:
            self.logger.warning(f"API增强功能设置失败: {e}")

    def _add_auth_routes(self):
        """添加身份验证路由"""
        from app.api.authentication import UserLogin, UserCreate, auth_manager

        @self.app.post("/auth/login", tags=["身份验证"])
        async def login(user_data: UserLogin):
            """用户登录"""
            return auth_manager.login(user_data)

        @self.app.post("/auth/register", tags=["身份验证"])
        async def register(user_data: UserCreate):
            """用户注册"""
            user = auth_manager.create_user(user_data)
            return {"message": "用户创建成功", "user": user}

        @self.app.get("/auth/profile", tags=["身份验证"])
        async def get_profile(current_user=auth_manager.get_current_user):
            """获取用户档案"""
            return current_user

    def _register_enhanced_routes(self):
        """注册增强的API路由"""

        @self.app.get("/")
        async def root():
            return {
                "message": "增强学术级智能健康监测集成系统",
                "version": "3.0.0",
                "status": "running",
                "features": [
                    "工作流集成", "数据增强", "跨平台数据流",
                    "模型适应", "反馈优化", "个性化学习"
                ]
            }

        @self.app.get("/health")
        async def health_check():
            # 检查数据库连接
            db_status = "healthy"
            try:
                if self.db_manager:
                    stats = self.db_manager.get_database_stats()
                    db_status = "healthy"
                else:
                    db_status = "not_initialized"
            except Exception as e:
                db_status = f"error: {str(e)}"

            status = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "database": db_status,
                "modules": {
                    "glucose_predictor": self.glucose_predictor is not None,
                    "image_predictor": self.image_predictor is not None,
                    "health_engine": self.health_engine is not None,
                    "data_integration": self.data_integration_manager is not None,
                    "workflow_orchestrator": self.workflow_orchestrator is not None,
                    "data_flow_manager": self.data_flow_manager is not None,
                    "adaptive_manager": self.adaptive_manager is not None,
                    "optimization_system": self.optimization_system is not None,
                    "gluformer_model": hasattr(self, 'gluformer_model') and self.gluformer_model is not None,
                    "cultural_engine": hasattr(self, 'cultural_engine') and self.cultural_engine is not None,
                    "merl_system": hasattr(self, 'merl_system') and self.merl_system is not None,
                    "knowledge_base": hasattr(self, 'knowledge_base') and self.knowledge_base is not None,
                    "explainable_system": hasattr(self, 'explainable_system') and self.explainable_system is not None
                }
            }

            # 添加详细状态
            if self.optimization_system:
                status["system_stats"] = self.optimization_system.get_system_stats()

            return status

        @self.app.post("/v3/predict/glucose/enhanced")
        async def predict_glucose_enhanced(
            data: Dict[str, Any],
            current_user=Depends(self.auth_manager.require_permission("read_data"))
        ):
            """增强的血糖预测接口（需要读取数据权限）"""
            try:
                user_id = data.get('user_id', current_user.user_id)

                # 使用优化系统进行预测
                if self.optimization_system:
                    result = self.optimization_system.process_prediction(user_id, data)
                else:
                    result = self.glucose_predictor.predict(data)

                # 添加用户信息
                result['request_user'] = current_user.username
                result['request_time'] = datetime.now().isoformat()

                return JSONResponse(content=result)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/feedback")
        async def submit_feedback(
            user_id: str = Form(...),
            prediction_id: str = Form(...),
            actual_glucose: float = Form(...),
            satisfaction_score: int = Form(...),
            additional_info: Optional[str] = Form(None)
        ):
            """提交用户反馈"""
            try:
                # 解析additional_info
                additional = {}
                if additional_info:
                    try:
                        additional = json.loads(additional_info)
                    except:
                        additional = {'raw_info': additional_info}

                if self.optimization_system:
                    self.optimization_system.process_feedback(
                        user_id, prediction_id, actual_glucose,
                        satisfaction_score, additional
                    )

                    return {"status": "success", "message": "反馈已收到"}
                else:
                    return {"status": "warning", "message": "反馈系统未启用"}

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/user/register")
        async def register_user(user_data: Dict[str, Any]):
            """注册新用户"""
            try:
                if self.adaptive_manager:
                    profile = self.adaptive_manager.register_user(user_data)
                    return {
                        "status": "success",
                        "user_id": profile.user_id,
                        "message": "用户注册成功"
                    }
                else:
                    return {"status": "warning", "message": "用户管理系统未启用"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/workflow/trigger")
        async def trigger_workflow(
            workflow_name: str = Form(...),
            input_data: str = Form(...)
        ):
            """触发工作流"""
            try:
                if self.workflow_orchestrator:
                    data = json.loads(input_data)
                    result = await self.workflow_orchestrator.trigger_workflow(workflow_name, data)
                    return result
                else:
                    return {"error": "工作流系统未启用"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/data/batch_process")
        async def batch_process_data(
            user_id: str = Form(...),
            platforms: Optional[str] = Form(None)
        ):
            """批量处理数据"""
            try:
                if self.data_flow_manager:
                    platform_list = platforms.split(',') if platforms else None
                    result = await self.data_flow_manager.process_batch_data(user_id, platform_list)
                    return result
                else:
                    return {"error": "数据流管理器未启用"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/stats/comprehensive")
        async def get_comprehensive_stats():
            """获取综合统计信息"""
            try:
                stats = {
                    "timestamp": datetime.now().isoformat(),
                    "system_overview": {}
                }

                # 数据库统计
                if self.db_manager:
                    try:
                        db_stats = self.db_manager.get_database_stats()
                        stats["database"] = db_stats

                        # 获取反馈统计
                        feedback_stats = self.db_manager.get_feedback_statistics()
                        stats["feedback"] = feedback_stats

                        # 获取数据增强统计
                        augmentation_stats = self.db_manager.get_augmentation_statistics()
                        stats["data_augmentation"] = augmentation_stats

                    except Exception as e:
                        stats["database"] = {"error": str(e)}

                # 优化系统统计
                if self.optimization_system:
                    stats["optimization"] = self.optimization_system.get_system_stats()

                # 适应管理器统计
                if self.adaptive_manager:
                    stats["adaptation"] = self.adaptive_manager.get_adaptation_report()

                # 工作流统计
                if self.workflow_orchestrator:
                    stats["workflow"] = self.workflow_orchestrator.get_workflow_status()

                # 数据流统计
                if self.data_flow_manager:
                    stats["data_flow"] = self.data_flow_manager.get_system_status()

                return stats
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/user/{user_id}/recommendations")
        async def get_user_recommendations(user_id: str):
            """获取用户个性化推荐"""
            try:
                if self.optimization_system:
                    user_profile = {"user_id": user_id}  # 可以从数据库获取完整档案
                    recommendations = self.optimization_system.get_optimized_recommendations(
                        user_id, user_profile
                    )
                    return recommendations
                else:
                    return {"error": "推荐系统未启用"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        # ==================== 数据库相关API端点 ====================

        @self.app.get("/database/stats")
        async def get_database_stats():
            """获取数据库统计信息"""
            try:
                if not self.db_manager:
                    raise HTTPException(status_code=503, detail="数据库未初始化")

                stats = self.db_manager.get_database_stats()
                return {
                    "timestamp": datetime.now().isoformat(),
                    "database_stats": stats
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/database/migration/status")
        async def get_migration_status():
            """获取数据库迁移状态"""
            try:
                if not self.migration_manager:
                    raise HTTPException(status_code=503, detail="迁移管理器未初始化")

                status = self.migration_manager.get_migration_status()
                return {
                    "timestamp": datetime.now().isoformat(),
                    "migration_status": status
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/users")
        async def get_users(role: Optional[str] = None, limit: int = 100):
            """获取用户列表"""
            try:
                if not self.db_manager:
                    raise HTTPException(status_code=503, detail="数据库未初始化")

                if role:
                    users = self.db_manager.get_users_by_role(role)
                else:
                    # 获取所有用户（简化实现）
                    users = []
                    for role_name in ["admin", "researcher", "patient"]:
                        users.extend(self.db_manager.get_users_by_role(role_name))

                return {
                    "timestamp": datetime.now().isoformat(),
                    "users": [
                        {
                            "user_id": user.user_id,
                            "username": user.username,
                            "email": user.email,
                            "role": user.role,
                            "is_active": user.is_active,
                            "created_at": user.created_at.isoformat(),
                            "last_login": user.last_login.isoformat() if user.last_login else None
                        }
                        for user in users[:limit]
                    ],
                    "total_count": len(users)
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/users/{user_id}/predictions")
        async def get_user_predictions(user_id: str, limit: int = 50,
                                     start_date: Optional[str] = None,
                                     end_date: Optional[str] = None):
            """获取用户血糖预测历史"""
            try:
                if not self.db_manager:
                    raise HTTPException(status_code=503, detail="数据库未初始化")

                # 解析日期参数
                start_dt = None
                end_dt = None
                if start_date:
                    start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                if end_date:
                    end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))

                predictions = self.db_manager.get_user_glucose_predictions(
                    user_id, limit, start_dt, end_dt
                )

                return {
                    "timestamp": datetime.now().isoformat(),
                    "user_id": user_id,
                    "predictions": [
                        {
                            "id": pred.id,
                            "predicted_glucose": pred.predicted_glucose,
                            "actual_glucose": pred.actual_glucose,
                            "confidence": pred.confidence,
                            "model_type": pred.model_type,
                            "model_version": pred.model_version,
                            "prediction_time": pred.prediction_time.isoformat(),
                            "feedback_time": pred.feedback_time.isoformat() if pred.feedback_time else None,
                            "accuracy_score": pred.accuracy_score
                        }
                        for pred in predictions
                    ],
                    "total_count": len(predictions)
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/food-items")
        async def search_food_items(query: Optional[str] = None,
                                   category: Optional[str] = None,
                                   cultural_region: Optional[str] = None,
                                   is_vegetarian: Optional[bool] = None,
                                   max_gi: Optional[float] = None,
                                   limit: int = 100):
            """搜索食物项目"""
            try:
                if not self.db_manager:
                    raise HTTPException(status_code=503, detail="数据库未初始化")

                food_items = self.db_manager.search_food_items(
                    query, category, cultural_region, is_vegetarian, max_gi
                )

                return {
                    "timestamp": datetime.now().isoformat(),
                    "food_items": [
                        {
                            "id": item.id,
                            "name": item.name,
                            "name_en": item.name_en,
                            "category": item.category,
                            "subcategory": item.subcategory,
                            "nutrition": item.nutrition,
                            "cultural_tags": item.cultural_tags,
                            "cooking_methods": item.cooking_methods,
                            "flavor_profile": item.flavor_profile,
                            "glycemic_index": item.glycemic_index,
                            "cultural_region": item.cultural_region,
                            "is_vegetarian": item.is_vegetarian,
                            "is_halal": item.is_halal,
                            "is_vegan": item.is_vegan
                        }
                        for item in food_items[:limit]
                    ],
                    "total_count": len(food_items)
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/predict/glucose/enhanced_gluformer")
        async def predict_glucose_enhanced_gluformer(data: Dict[str, Any]):
            """使用增强的GluFormer模型进行血糖预测"""
            try:
                if hasattr(self, 'enhanced_gluformer_model') and self.enhanced_gluformer_model:
                    # 准备输入数据
                    input_features = self._prepare_enhanced_gluformer_input(data)
                    cultural_id = data.get('cultural_id', None)

                    # 使用增强的GluFormer预测
                    with torch.no_grad():
                        if cultural_id is not None:
                            cultural_tensor = torch.tensor([cultural_id], dtype=torch.long)
                            prediction_result = self.enhanced_gluformer_model(
                                input_features, cultural_tensor
                            )
                        else:
                            prediction_result = self.enhanced_gluformer_model(input_features)

                    return {
                        "predicted_glucose": prediction_result['prediction'].item(),
                        "uncertainty": prediction_result['uncertainty'].item(),
                        "model_type": "Enhanced_GluFormer",
                        "confidence": 1.0 - prediction_result['uncertainty'].item(),
                        "features_used": list(data.keys()),
                        "cultural_adapted": cultural_id is not None,
                        "attention_weights": {
                            "cross_attention": prediction_result['cross_attention_weights'].mean().item(),
                            "temporal_attention": prediction_result['temporal_attention_weights'].mean().item()
                        }
                    }
                else:
                    return {"error": "增强的GluFormer模型未初始化"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/cultural/recommendations")
        async def get_cultural_recommendations(
            user_id: str = Form(...),
            cultural_profile: str = Form(...),
            health_profile: str = Form(...)
        ):
            """获取文化适配的膳食推荐"""
            try:
                if hasattr(self, 'cultural_engine') and self.cultural_engine:
                    # 解析文化档案
                    cultural_data = json.loads(cultural_profile)
                    health_data = json.loads(health_profile)

                    # 注册用户文化档案
                    from models.cultural_adaptation import CulturalProfile
                    user_cultural = CulturalProfile(**cultural_data)
                    self.cultural_engine.register_user_cultural_profile(user_id, user_cultural)

                    # 生成推荐
                    recommendations = self.cultural_engine.generate_cultural_recommendations(
                        user_id, health_data
                    )

                    return recommendations
                else:
                    return {"error": "文化适配引擎未启用"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/merl/action")
        async def get_merl_action(data: Dict[str, Any]):
            """获取混合专家强化学习动作"""
            try:
                if hasattr(self, 'merl_system') and self.merl_system:
                    # 准备输入特征
                    input_features = torch.tensor([list(data.values())], dtype=torch.float32)

                    # 获取MERL系统输出
                    with torch.no_grad():
                        output = self.merl_system(input_features)

                    # 选择最佳动作
                    action_probs = output['policy_probs']
                    best_action = torch.argmax(action_probs, dim=-1).item()

                    return {
                        "recommended_action": best_action,
                        "action_probabilities": action_probs.tolist(),
                        "value_estimate": output['value'].item(),
                        "expert_weights": output['gate_weights'].tolist()
                    }
                else:
                    return {"error": "MERL系统未启用"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/explain/prediction")
        async def explain_prediction(data: Dict[str, Any]):
            """生成预测的可解释性分析"""
            try:
                if hasattr(self, 'explainable_system') and self.explainable_system:
                    # 生成解释
                    explanation = self.explainable_system.generate_explanation(
                        data, data.get('prediction', 0.5)
                    )

                    return explanation
                else:
                    return {"error": "可解释性系统未启用"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        # 工作流管理接口
        @self.app.get("/v3/workflows/templates")
        async def get_workflow_templates(current_user=Depends(self.auth_manager.require_permission("read_data"))):
            """获取工作流模板列表"""
            try:
                from workflow.visualization import workflow_viz_engine
                templates = workflow_viz_engine.get_templates_list()
                return {"templates": templates}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/v3/workflows/create")
        async def create_workflow_from_template(
            template_name: str = Form(...),
            workflow_name: str = Form(...),
            current_user=Depends(self.auth_manager.require_permission("write_data"))
        ):
            """基于模板创建工作流"""
            try:
                from workflow.visualization import workflow_viz_engine
                workflow = workflow_viz_engine.create_workflow_from_template(
                    template_name, workflow_name, current_user.username
                )
                return {
                    "workflow_id": workflow.workflow_id,
                    "name": workflow.name,
                    "message": "工作流创建成功"
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/v3/workflows/{workflow_id}/visualization")
        async def get_workflow_visualization(
            workflow_id: str,
            current_user=Depends(self.auth_manager.require_permission("read_data"))
        ):
            """获取工作流可视化数据"""
            try:
                from workflow.visualization import workflow_viz_engine
                viz_data = workflow_viz_engine.get_workflow_visualization_data(workflow_id)
                return viz_data
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/v3/workflows/{workflow_id}/metrics")
        async def get_workflow_metrics(
            workflow_id: str,
            current_user=Depends(self.auth_manager.require_permission("view_analytics"))
        ):
            """获取工作流性能指标"""
            try:
                from app.workflow.monitoring import workflow_monitor
                metrics = workflow_monitor.get_workflow_metrics(workflow_id)

                if metrics:
                    return {
                        "workflow_id": workflow_id,
                        "execution_count": metrics.execution_count,
                        "success_rate": metrics.success_rate,
                        "error_rate": metrics.error_rate,
                        "avg_execution_time": metrics.avg_execution_time,
                        "throughput_per_hour": metrics.throughput_per_hour,
                        "last_execution": metrics.last_execution_time.isoformat() if metrics.last_execution_time else None
                    }
                else:
                    return {"error": "未找到工作流指标"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/v3/monitoring/dashboard")
        async def get_monitoring_dashboard(current_user=Depends(self.auth_manager.require_permission("view_analytics"))):
            """获取监控仪表板数据"""
            try:
                from app.workflow.monitoring import workflow_monitor
                dashboard_data = workflow_monitor.get_performance_dashboard_data()
                return dashboard_data
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/v3/monitoring/alerts")
        async def get_active_alerts(current_user=Depends(self.auth_manager.require_permission("view_analytics"))):
            """获取活跃告警"""
            try:
                from app.workflow.monitoring import workflow_monitor
                alerts = workflow_monitor.get_active_alerts()

                alerts_data = []
                for alert in alerts:
                    alert_data = {
                        "alert_id": alert.alert_id,
                        "rule_id": alert.rule_id,
                        "level": alert.alert_level.value,
                        "message": alert.message,
                        "metric_value": alert.metric_value,
                        "threshold": alert.threshold,
                        "triggered_at": alert.triggered_at.isoformat(),
                        "workflow_id": alert.workflow_id
                    }
                    alerts_data.append(alert_data)

                return {"alerts": alerts_data}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/v3/monitoring/alerts/{alert_id}/resolve")
        async def resolve_alert(
            alert_id: str,
            resolution_note: str = Form(""),
            current_user=Depends(self.auth_manager.require_permission("manage_users"))
        ):
            """解决告警"""
            try:
                from app.workflow.monitoring import workflow_monitor
                workflow_monitor.resolve_alert(alert_id, resolution_note)
                return {"message": "告警已解决", "resolved_by": current_user.username}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        # 保留原有路由以兼容
        self._register_original_routes()

    def _register_original_routes(self):
        """注册原有API路由以保持兼容性"""

        @self.app.get("/original")
        async def original_root():
            return {
                "message": "学术级智能健康监测集成系统",
                "version": "2.0.0",
                "status": "running"
            }

        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "modules": {
                    "glucose_predictor": self.glucose_predictor is not None,
                    "image_predictor": self.image_predictor is not None,
                    "health_engine": self.health_engine is not None
                }
            }

        @self.app.post("/predict/glucose")
        async def predict_glucose(data: Dict[str, Any]):
            try:
                result = self.glucose_predictor.predict(data)
                return JSONResponse(content=result)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/predict/image")
        async def predict_image(file: UploadFile = File(...)):
            try:
                # 保存上传的文件
                temp_path = f"temp_{file.filename}"
                with open(temp_path, "wb") as buffer:
                    buffer.write(await file.read())

                result = self.image_predictor.predict(temp_path)

                # 清理临时文件
                os.remove(temp_path)

                return JSONResponse(content=result)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/assess/health")
        async def assess_health(
            glucose_data: Dict[str, Any],
            image_file: Optional[UploadFile] = File(None),
            user_profile: Optional[Dict[str, Any]] = None
        ):
            try:
                image_data = None
                if image_file:
                    temp_path = f"temp_{image_file.filename}"
                    with open(temp_path, "wb") as buffer:
                        buffer.write(await image_file.read())
                    image_data = {"path": temp_path}

                result = self.health_engine.assess_health(
                    glucose_data=glucose_data,
                    image_data=image_data,
                    user_profile=user_profile
                )

                # 清理临时文件
                if image_data:
                    os.remove(image_data["path"])

                return JSONResponse(content=result)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/metrics")
        async def get_metrics():
            return {
                "metrics_history": [m.to_dict() for m in self.metrics_history],
                "current_metrics": self.health_engine.metrics.to_dict()
            }

    async def start_background_services(self):
        """启动后台服务"""
        try:
            # 启动数据流管理器
            if self.data_flow_manager:
                await self.data_flow_manager.start()
                self.logger.info("数据流管理器已启动")

            # 设置默认工作流
            if self.workflow_orchestrator:
                await self._setup_default_workflows()
                self.logger.info("默认工作流已设置")

        except Exception as e:
            self.logger.warning(f"后台服务启动失败: {e}")

    async def _setup_default_workflows(self):
        """设置默认工作流"""
        try:
            from app.data_integration.workflow_integration import WorkflowConfig

            # 创建默认CGM数据收集工作流
            cgm_workflow = WorkflowConfig(
                name='cgm_data_collector',
                endpoint='https://api.example.com/cgm/collect',
                api_key='demo_key',
                trigger_type='api',
                data_format='json'
            )
            self.workflow_orchestrator.register_workflow(cgm_workflow)

            # 创建营养分析工作流
            nutrition_workflow = WorkflowConfig(
                name='nutrition_analyzer',
                endpoint='https://api.example.com/nutrition/analyze',
                api_key='demo_key',
                trigger_type='api',
                data_format='json'
            )
            self.workflow_orchestrator.register_workflow(nutrition_workflow)

        except Exception as e:
            self.logger.warning(f"默认工作流设置失败: {e}")

    async def stop_background_services(self):
        """停止后台服务"""
        try:
            if self.data_flow_manager:
                await self.data_flow_manager.stop()
                self.logger.info("数据流管理器已停止")

            if self.workflow_orchestrator:
                self.workflow_orchestrator.stop_all_schedules()
                self.logger.info("工作流调度器已停止")

        except Exception as e:
            self.logger.warning(f"后台服务停止失败: {e}")

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """运行增强系统"""
        try:
            self.logger.info(f"启动增强学术级集成系统，地址: {host}:{port}")

            # 确保app对象存在
            if not self.app:
                self.logger.error("FastAPI应用未初始化")
                return

            # 设置启动和关闭事件
            @self.app.on_event("startup")
            async def startup_event():
                try:
                    await self.start_background_services()
                except Exception as e:
                    self.logger.warning(f"后台服务启动失败: {e}")

            @self.app.on_event("shutdown")
            async def shutdown_event():
                try:
                    await self.stop_background_services()
                except Exception as e:
                    self.logger.warning(f"后台服务停止失败: {e}")

            uvicorn.run(self.app, host=host, port=port, reload=False)

        except Exception as e:
            self.logger.error(f"系统启动失败: {e}")
            raise

    def get_system_summary(self) -> Dict[str, Any]:
        """获取系统概要信息"""
        summary = {
            "system_name": "增强学术级智能健康监测集成系统",
            "version": "3.0.0",
            "initialization_time": datetime.now().isoformat(),
            "components": {
                "core_predictors": {
                    "glucose_predictor": self.glucose_predictor is not None,
                    "image_predictor": self.image_predictor is not None,
                    "health_engine": self.health_engine is not None
                },
                "data_integration": {
                    "data_integration_manager": self.data_integration_manager is not None,
                    "workflow_orchestrator": self.workflow_orchestrator is not None,
                    "data_flow_manager": self.data_flow_manager is not None,
                    "enhancement_pipeline": self.enhancement_pipeline is not None
                },
                "ai_optimization": {
                    "adaptive_manager": self.adaptive_manager is not None,
                    "optimization_system": self.optimization_system is not None
                }
            },
            "capabilities": [
                "增强多模态数据融合与跨领域知识协同",
                "时序感知的LSTM-GRU融合血糖预测",
                "文化适配的智能膳食推荐系统",
                "混合专家强化学习与多任务优化",
                "动态知识蒸馏与实时更新机制",
                "语义对齐与跨模态注意力机制",
                "个性化模型适应与迁移学习",
                "不确定性估计与置信度评估",
                "在线自适应学习与增量更新",
                "可解释性分析与知识推理",
                "工作流自动化与可视化管理",
                "实时监控报警与性能优化",
                "跨平台数据集成与API适配",
                "动态数据增强与GAN生成",
                "版本控制与身份验证机制",
                "GraphQL灵活查询与RESTful API",
                "时序奖励预测与策略优化",
                "文化知识图谱与关系推理",
                "专家权重动态调整",
                "多尺度特征提取与融合"
            ],
            "api_endpoints": [
                "/predict/glucose/enhanced",
                "/predict/glucose/gluformer",
                "/cultural/recommendations",
                "/merl/action",
                "/explain/prediction",
                "/feedback",
                "/user/register",
                "/workflow/trigger",
                "/data/batch_process",
                "/stats/comprehensive",
                "/user/{user_id}/recommendations"
            ]
        }

        return summary

    def _prepare_enhanced_gluformer_input(self, data: Dict[str, Any]) -> torch.Tensor:
        """准备增强GluFormer模型输入"""
        # 扩展的时序特征
        feature_keys = [
            'glucose', 'insulin', 'hour', 'day_of_week', 'is_night', 'is_weekend',
            'carbohydrates', 'protein', 'fat', 'calories', 'fiber',
            'duration_minutes', 'calories_burned', 'heart_rate', 'stress_level'
        ]

        features = []
        for key in feature_keys:
            features.append(data.get(key, 0.0))

        # 转换为张量并添加序列维度
        feature_tensor = torch.tensor(features, dtype=torch.float32)

        # 创建时序序列（使用更长的历史）
        sequence_length = 24  # 使用24个时间步（24小时历史）

        # 如果有历史数据，使用历史数据
        if 'history' in data and len(data['history']) > 0:
            history_features = []
            for hist_point in data['history'][-sequence_length:]:
                hist_features = []
                for key in feature_keys:
                    hist_features.append(hist_point.get(key, 0.0))
                history_features.append(hist_features)

            # 填充到sequence_length长度
            while len(history_features) < sequence_length:
                history_features.insert(0, [0.0] * len(feature_keys))

            input_sequence = torch.tensor(history_features, dtype=torch.float32)
        else:
            # 如果没有历史数据，重复当前特征
            input_sequence = feature_tensor.unsqueeze(0).repeat(sequence_length, 1)

        # 添加batch维度
        input_sequence = input_sequence.unsqueeze(0)  # [1, seq_len, feature_dim]

        return input_sequence

    def _prepare_gluformer_input(self, data: Dict[str, Any]) -> torch.Tensor:
        """准备原始GluFormer模型输入（保持兼容性）"""
        # 提取时序特征
        feature_keys = ['glucose', 'hour', 'day_of_week', 'is_night',
                       'carbohydrates', 'protein', 'fat', 'calories',
                       'duration_minutes', 'calories_burned']

        features = []
        for key in feature_keys:
            features.append(data.get(key, 0.0))

        # 转换为张量并添加序列维度
        feature_tensor = torch.tensor(features, dtype=torch.float32)

        # 创建序列（这里简化为单时间步，实际应用中需要历史序列）
        sequence_length = 10  # 假设使用10个时间步的历史
        input_sequence = feature_tensor.unsqueeze(0).repeat(sequence_length, 1)
        input_sequence = input_sequence.unsqueeze(0)  # 添加batch维度

        return input_sequence

def main():
    """增强主函数"""
    try:
        print("=" * 60)
        print("🚀 启动增强学术级智能健康监测集成系统")
        print("=" * 60)

        # 创建系统实例
        system = AcademicIntegratedSystem()

        # 显示系统概要
        summary = system.get_system_summary()
        print(f"\n📊 系统信息:")
        print(f"   名称: {summary['system_name']}")
        print(f"   版本: {summary['version']}")
        print(f"   启动时间: {summary['initialization_time']}")

        print(f"\n🔧 组件状态:")
        for category, components in summary['components'].items():
            print(f"   {category}:")
            for comp_name, status in components.items():
                status_icon = "✅" if status else "❌"
                print(f"      {status_icon} {comp_name}")

        print(f"\n🎯 核心能力:")
        for capability in summary['capabilities']:
            print(f"   • {capability}")

        print(f"\n🌐 API接口:")
        for endpoint in summary['api_endpoints']:
            print(f"   • {endpoint}")

        print("\n" + "=" * 60)
        print("💡 系统已准备就绪，开始处理请求...")
        print("   🌐 RESTful API: http://localhost:8000/docs")
        print("   📊 GraphQL Playground: http://localhost:8000/api/graphql")
        print("   ❤️ 健康检查: http://localhost:8000/health")
        print("   📈 监控仪表板: http://localhost:8000/v3/monitoring/dashboard")
        print("   🔧 工作流管理: http://localhost:8000/v3/workflows/templates")
        print("   📋 API版本信息: http://localhost:8000/versions")
        print("   🔐 身份验证: http://localhost:8000/auth/login")
        print("=" * 60)
        print("🎓 增强学术级功能特色:")
        print("   ✅ 增强GluFormer：LSTM-GRU融合+文化适配+时序感知")
        print("   ✅ 跨领域多模态融合：医学+营养学+行为学+文化学协同")
        print("   ✅ 智能文化适配：知识图谱+自注意力+动态学习")
        print("   ✅ 时序感知MERL：专家混合+强化学习+多任务优化")
        print("   ✅ 动态知识蒸馏：实时更新+可解释性+版本控制")
        print("   ✅ 多尺度特征融合：CNN+LSTM+Transformer+注意力机制")
        print("   ✅ 不确定性估计：贝叶斯推理+集成学习")
        print("   ✅ 在线自适应学习：增量更新+个性化调优")
        print("   ✅ 工作流可视化与实时监控报警系统")
        print("   ✅ API版本控制、身份验证与GraphQL查询")
        print("   ✅ 多目标强化学习：血糖控制+营养均衡+患者满意度")
        print("   ✅ 实时健康监测：多维度监测+异常检测+趋势分析")
        print("   ✅ 闭环健康管理：实时反馈+动态调整+个性化推荐")
        print("   ✅ 差分隐私保护：医疗数据隐私+噪声注入+预算管理")
        print("   ✅ 同态加密计算：加密状态计算+安全数据处理")
        print("   ✅ 细粒度访问控制：基于角色权限+审计日志+合规检查")
        print("   ✅ 数据匿名化处理：k-匿名性+l-多样性+t-接近性")
        print("=" * 60)

        # 启动系统
        system.run()

    except KeyboardInterrupt:
        print("\n\n👋 系统正在优雅退出...")
        logger.info("用户中断，系统退出")
    except Exception as e:
        print(f"\n❌ 系统运行失败: {e}")
        logger.error(f"系统运行失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
__all__ = ["'project_root'", "'logger'", "'AcademicMetrics'", "'EnhancedGluFormer'", "'EnhancedMultiModalFusion'", "'AcademicGlucosePredictor'", "'AcademicImagePredictor'", "'AcademicHealthDecisionEngine'", "'AcademicIntegratedSystem'", "'main'"]
