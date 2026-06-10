"""
血糖预测模型
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import asyncio
import json

"""
血糖预测模型
企业级AI模型实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
logger = logging.getLogger(__name__)
import logging
import random
from collections import deque

logger = logging.getLogger(__name__)

class MultiScaleSpatioTemporalAlignmentNetwork(nn.Module):
    """
    多尺度时空对齐网络 (MSTAN)
    基于项目申请表中的创新点六设计
    实现血糖变化的短期和长期趋势捕捉
    """

    def __init__(self,
                 input_dim: int = 15,
                 hidden_dim: int = 128,
                 num_scales: int = 4,
                 temporal_windows: List[int] = [1, 3, 7, 14],  # 1天, 3天, 7天, 14天
                 dropout: float = 0.1):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales
        self.temporal_windows = temporal_windows

        # 多尺度时序编码器
        self.scale_encoders = nn.ModuleList([
            nn.LSTM(input_dim, hidden_dim, batch_first=True, dropout=dropout)
            for _ in range(num_scales)
        ])

        # 空间对齐网络
        self.spatial_alignment = nn.ModuleDict({
            f'scale_{i}': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for i in range(num_scales)
        })

        # 跨尺度注意力机制
        self.cross_scale_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, dropout=dropout, batch_first=True
        )

        # 时序对齐网络
        self.temporal_alignment = nn.Sequential(
            nn.Linear(hidden_dim * num_scales, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # 自适应权重网络
        self.adaptive_weight_network = nn.Sequential(
            nn.Linear(hidden_dim * num_scales, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_scales),
            nn.Softmax(dim=-1)
        )

        # 趋势检测器
        self.trend_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),  # 上升、下降、稳定
            nn.Softmax(dim=-1)
        )

        # 异常检测器
        self.anomaly_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, temporal_context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        x: [batch_size, seq_len, input_dim] 时序数据
        temporal_context: [batch_size, context_dim] 时序上下文
        """
        batch_size, seq_len, _ = x.shape

        # 多尺度时序编码
        scale_outputs = []
        scale_weights = []

        for i, (encoder, window_size) in enumerate(zip(self.scale_encoders, self.temporal_windows)):
            # 根据时间窗口调整输入
            if seq_len >= window_size:
                # 使用滑动窗口
                windowed_x = self._create_sliding_windows(x, window_size)
                scale_output, _ = encoder(windowed_x)
                scale_output = scale_output[:, -1, :]  # 取最后一个时间步
            else:
                # 如果序列长度小于窗口大小，使用整个序列
                scale_output, _ = encoder(x)
                scale_output = scale_output[:, -1, :]

            # 空间对齐
            aligned_output = self.spatial_alignment[f'scale_{i}'](scale_output)
            scale_outputs.append(aligned_output)

        # 堆叠多尺度输出
        multi_scale_features = torch.stack(scale_outputs, dim=1)  # [batch_size, num_scales, hidden_dim]

        # 跨尺度注意力
        attended_features, attention_weights = self.cross_scale_attention(
            multi_scale_features, multi_scale_features, multi_scale_features
        )

        # 自适应权重计算
        adaptive_weights = self.adaptive_weight_network(
            multi_scale_features.view(batch_size, -1)
        )

        # 加权融合
        weighted_features = torch.sum(
            multi_scale_features * adaptive_weights.unsqueeze(-1), dim=1
        )

        # 时序对齐
        aligned_temporal_features = self.temporal_alignment(
            multi_scale_features.view(batch_size, -1)
        )

        # 趋势检测
        trend_probs = self.trend_detector(aligned_temporal_features)

        # 异常检测
        anomaly_score = self.anomaly_detector(aligned_temporal_features)

        return {
            'multi_scale_features': multi_scale_features,
            'attended_features': attended_features,
            'weighted_features': weighted_features,
            'aligned_temporal_features': aligned_temporal_features,
            'adaptive_weights': adaptive_weights,
            'trend_probs': trend_probs,
            'anomaly_score': anomaly_score,
            'attention_weights': attention_weights
        }

    def _create_sliding_windows(self, x: torch.Tensor, window_size: int) -> torch.Tensor:
        """创建滑动窗口"""
        batch_size, seq_len, input_dim = x.shape

        if seq_len <= window_size:
            return x

        # 创建滑动窗口
        windows = []
        for i in range(seq_len - window_size + 1):
            windows.append(x[:, i:i+window_size, :])

        return torch.stack(windows, dim=1)  # [batch_size, num_windows, window_size, input_dim]

class AdaptiveFineTuningModule(nn.Module):
    """
    自适应微调模块
    基于项目申请表中的创新点六设计
    实现根据患者个性化数据的动态参数调整
    """

    def __init__(self,
                 base_model_dim: int = 256,
                 patient_profile_dim: int = 64,
                 adaptation_dim: int = 128,
                 num_adaptation_layers: int = 3):
        super().__init__()

        self.base_model_dim = base_model_dim
        self.patient_profile_dim = patient_profile_dim
        self.adaptation_dim = adaptation_dim

        # 患者画像编码器
        self.patient_encoder = nn.Sequential(
            nn.Linear(patient_profile_dim, adaptation_dim),
            nn.LayerNorm(adaptation_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 自适应参数生成器
        self.adaptation_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(adaptation_dim, adaptation_dim),
                nn.ReLU(),
                nn.Linear(adaptation_dim, base_model_dim),
                nn.Tanh()  # 限制参数范围
            ) for _ in range(num_adaptation_layers)
        ])

        # 参数重要性评估器
        self.importance_evaluator = nn.Sequential(
            nn.Linear(adaptation_dim, adaptation_dim // 2),
            nn.ReLU(),
            nn.Linear(adaptation_dim // 2, num_adaptation_layers),
            nn.Sigmoid()
        )

        # 微调强度控制器
        self.fine_tuning_controller = nn.Sequential(
            nn.Linear(adaptation_dim, adaptation_dim // 2),
            nn.ReLU(),
            nn.Linear(adaptation_dim // 2, 1),
            nn.Sigmoid()
        )

        # 历史适应记录
        self.adaptation_history = deque(maxlen=1000)

    def forward(self,
                base_features: torch.Tensor,
                patient_profile: torch.Tensor,
                adaptation_context: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """
        自适应微调前向传播
        """
        batch_size = base_features.size(0)

        # 编码患者画像
        encoded_profile = self.patient_encoder(patient_profile)

        # 生成自适应参数
        adaptation_params = []
        for generator in self.adaptation_generators:
            param = generator(encoded_profile)
            adaptation_params.append(param)

        # 计算参数重要性
        importance_weights = self.importance_evaluator(encoded_profile)

        # 计算微调强度
        fine_tuning_strength = self.fine_tuning_controller(encoded_profile)

        # 应用自适应参数
        adapted_features = base_features
        for i, (param, importance) in enumerate(zip(adaptation_params, importance_weights.unbind(1))):
            # 加权应用参数
            adaptation_factor = importance.unsqueeze(-1) * fine_tuning_strength
            adapted_features = adapted_features + adaptation_factor * param

        # 记录适应历史
        adaptation_record = {
            'timestamp': torch.tensor([0.0]),  # 简化实现
            'patient_profile': patient_profile.detach(),
            'adaptation_params': [p.detach() for p in adaptation_params],
            'importance_weights': importance_weights.detach(),
            'fine_tuning_strength': fine_tuning_strength.detach()
        }
        self.adaptation_history.append(adaptation_record)

        return {
            'adapted_features': adapted_features,
            'adaptation_params': adaptation_params,
            'importance_weights': importance_weights,
            'fine_tuning_strength': fine_tuning_strength,
            'encoded_profile': encoded_profile
        }

    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """获取适应统计信息"""
        if not self.adaptation_history:
            return {}

        # 计算平均微调强度
        avg_strength = np.mean([
            record['fine_tuning_strength'].item()
            for record in self.adaptation_history
        ])

        # 计算参数重要性分布
        importance_dist = np.mean([
            record['importance_weights'].numpy()
            for record in self.adaptation_history
        ], axis=0)

        return {
            'total_adaptations': len(self.adaptation_history),
            'average_fine_tuning_strength': avg_strength,
            'importance_distribution': importance_dist.tolist(),
            'adaptation_frequency': len(self.adaptation_history) / 1000.0  # 假设1000次记录为基准
        }

class AcademicGlucosePredictor:
    """学术级血糖预测器"""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_trained = False

    def predict(self, input_data: np.ndarray) -> float:
        """预测血糖值"""
        try:
            if not self.is_trained:
                # 使用简单的线性回归作为基础预测
                return self._simple_prediction(input_data)

            # 使用训练好的模型预测
            if self.scaler:
                input_data = self.scaler.transform(input_data)

            prediction = self.model.predict(input_data)
            return float(prediction[0])

        except Exception as e:
            logger.error(f"血糖预测失败: {e}")
            # 返回基于规则的预测
            return self._rule_based_prediction(input_data)

    def _simple_prediction(self, input_data: np.ndarray) -> float:
        """简单预测算法"""
        glucose, hour, carbs, insulin, exercise, stress = input_data[0]

        # 基础血糖值
        base_glucose = glucose

        # 时间因子
        time_factor = 1.0
        if 6 <= hour <= 8:  # 早餐后
            time_factor = 1.1
        elif 12 <= hour <= 14:  # 午餐后
            time_factor = 1.15
        elif 18 <= hour <= 20:  # 晚餐后
            time_factor = 1.2

        # 碳水化合物影响
        carb_factor = 1 + (carbs * 0.02)

        # 胰岛素影响
        insulin_factor = 1 - (insulin * 0.05)

        # 运动影响
        exercise_factor = 1 - (exercise * 0.001)

        # 压力影响
        stress_factor = 1 + (stress * 0.01)

        prediction = base_glucose * time_factor * carb_factor * insulin_factor * exercise_factor * stress_factor

        return max(50, min(400, prediction))  # 限制在合理范围内

    def _rule_based_prediction(self, input_data: np.ndarray) -> float:
        """基于规则的预测"""
        glucose, hour, carbs, insulin, exercise, stress = input_data[0]

        # 简单的规则预测
        if carbs > 50:
            return glucose + 30
        elif exercise > 30:
            return glucose - 20
        elif stress > 7:
            return glucose + 15
        else:
            return glucose + 5

class EnhancedGluFormer(nn.Module):
    """增强GluFormer模型"""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 cultural_dim: int, temporal_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cultural_dim = cultural_dim
        self.temporal_dim = temporal_dim

        # LSTM-GRU融合层
        self.lstm_layer = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.gru_layer = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)

        # 新增：多尺度时空对齐网络
        self.mstan = MultiScaleSpatioTemporalAlignmentNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_scales=4,
            temporal_windows=[1, 3, 7, 14]
        )

        # 新增：自适应微调模块
        self.adaptive_fine_tuning = AdaptiveFineTuningModule(
            base_model_dim=hidden_dim,
            patient_profile_dim=64,  # 患者画像维度
            adaptation_dim=128,
            num_adaptation_layers=3
        )

        # 文化适应性层
        self.cultural_adaptation = nn.Sequential(
            nn.Linear(cultural_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 时序编码器
        self.temporal_encoder = nn.Linear(temporal_dim, hidden_dim)

        # 注意力机制
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)

        # 增强的输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),  # 增加输入维度
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # 趋势预测头
        self.trend_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # 上升、下降、稳定
        )

        # 异常检测头
        self.anomaly_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: Dict[str, torch.Tensor], patient_profile: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 提取特征
        glucose = x['glucose']
        temporal = x['temporal']
        cultural = x['cultural']

        # 文化编码
        cultural_features = self.cultural_adaptation(cultural)

        # 时序编码
        temporal_features = self.temporal_encoder(temporal)

        # 特征融合
        features = torch.cat([glucose, temporal_features, cultural_features], dim=-1)

        # LSTM处理
        lstm_out, _ = self.lstm_layer(features.unsqueeze(1))

        # GRU处理
        gru_out, _ = self.gru_layer(lstm_out)

        # 多尺度时空对齐
        mstan_output = self.mstan(features.unsqueeze(1), temporal_features)
        mstan_features = mstan_output['aligned_temporal_features']

        # 注意力机制
        attn_out, attention_weights = self.attention(gru_out, gru_out, gru_out)

        # 特征融合
        combined_features = torch.cat([gru_out.squeeze(1), attn_out.squeeze(1), mstan_features], dim=-1)

        # 自适应微调
        if patient_profile is not None:
            adaptation_output = self.adaptive_fine_tuning(combined_features, patient_profile)
            adapted_features = adaptation_output['adapted_features']
        else:
            adapted_features = combined_features

        # 血糖预测
        glucose_prediction = self.output_layer(adapted_features)

        # 趋势预测
        trend_prediction = self.trend_head(adapted_features)

        # 异常检测
        anomaly_score = self.anomaly_head(adapted_features)

        return {
            'glucose_prediction': glucose_prediction,
            'trend_prediction': trend_prediction,
            'anomaly_score': anomaly_score,
            'mstan_output': mstan_output,
            'attention_weights': attention_weights,
            'adapted_features': adapted_features,
            'adaptation_stats': self.adaptive_fine_tuning.get_adaptation_statistics() if patient_profile is not None else {}
        }

    def predict(self, input_data: Dict[str, Any]) -> float:
        """预测接口"""
        try:
            # 转换为张量
            x = {}
            for key, value in input_data.items():
                if isinstance(value, np.ndarray):
                    x[key] = torch.FloatTensor(value)
                else:
                    x[key] = torch.FloatTensor([value])

            # 预测
            with torch.no_grad():
                prediction = self.forward(x)
                return float(prediction.item())

        except Exception as e:
            logger.error(f"GluFormer预测失败: {e}")
            return 100.0  # 默认值

class EnhancedMultiModalFusion(nn.Module):
    """
    增强的多模态数据融合模块
    实现跨领域知识协同、语义对齐与动态特征融合
    基于项目申请表中的创新点二设计
    """

    def __init__(self, glucose_dim: int = 128, image_dim: int = 512,
                 text_dim: int = 256, behavioral_dim: int = 64,
                 medical_dim: int = 128, cultural_dim: int = 64,
                 fusion_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        self.glucose_dim = glucose_dim
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.behavioral_dim = behavioral_dim
        self.medical_dim = medical_dim
        self.cultural_dim = cultural_dim
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
        self.cultural_projection = nn.Sequential(
            nn.Linear(cultural_dim, fusion_dim),
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
            nn.Linear(fusion_dim * 6, fusion_dim),  # 6个模态
            nn.ReLU(),
            nn.Linear(fusion_dim, 6),  # 6个权重
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
                cultural_features: Optional[torch.Tensor] = None,
                sequence_length: int = 10) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        增强的前向传播，支持多模态跨领域知识协同

        Args:
            glucose_features: 血糖特征 [batch_size, glucose_dim]
            image_features: 图像特征 [batch_size, image_dim]
            text_features: 文本特征 [batch_size, text_dim] (可选)
            behavioral_features: 行为特征 [batch_size, behavioral_dim] (可选)
            medical_features: 医学特征 [batch_size, medical_dim] (可选)
            cultural_features: 文化特征 [batch_size, cultural_dim] (可选)
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

        if cultural_features is not None:
            cultural_proj = self.cultural_projection(cultural_features)
            modality_features.append(cultural_proj)
            modality_names.append('cultural')

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

        # 填充到6个模态（如果不足）
        while len(aligned_features) < 6:
            aligned_features.append(torch.zeros_like(aligned_features[0]))

        # 截取到6个模态（如果超过）
        aligned_features = aligned_features[:6]

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

class ReinforcementLearningDecisionEngine(nn.Module):
    """
    强化学习决策引擎
    基于项目申请表中的创新点三设计
    实现多目标强化学习与动态策略优化
    """

    def __init__(self, state_dim: int = 256, action_dim: int = 5,
                 hidden_dim: int = 128, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # 多目标价值网络
        self.value_networks = nn.ModuleDict({
            'glucose_control': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            ),
            'nutrition_balance': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            ),
            'patient_satisfaction': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            ),
            'cultural_adaptation': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
        })

        # 策略网络
        self.policy_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

        # 注意力机制用于动态权重调整
        self.attention_layer = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 时序编码器
        self.temporal_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        # 动态权重网络
        self.weight_network = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),  # +4 for objective values
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
            nn.Softmax(dim=-1)
        )

        # 经验回放缓冲区
        self.replay_buffer = deque(maxlen=10000)

        # 优化器
        self.optimizer = None
        self.gamma = 0.99  # 折扣因子
        self.epsilon = 0.1  # 探索率
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def forward(self, state: torch.Tensor,
                temporal_context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            state: 当前状态 [batch_size, state_dim]
            temporal_context: 时序上下文 [batch_size, seq_len, state_dim]

        Returns:
            包含策略、价值和权重的字典
        """
        batch_size = state.size(0)

        # 状态编码
        encoded_state = self.state_encoder(state)

        # 时序处理
        if temporal_context is not None:
            temporal_encoded = self.state_encoder(temporal_context.view(-1, self.state_dim))
            temporal_encoded = temporal_encoded.view(batch_size, -1, self.hidden_dim)

            # LSTM处理
            temporal_output, _ = self.temporal_encoder(temporal_encoded)
            temporal_feature = temporal_output.mean(dim=1)

            # 注意力融合
            attended_state, _ = self.attention_layer(
                encoded_state.unsqueeze(1),
                temporal_feature.unsqueeze(1),
                temporal_feature.unsqueeze(1)
            )
            encoded_state = attended_state.squeeze(1)

        # 多目标价值估计
        values = {}
        for objective, network in self.value_networks.items():
            values[objective] = network(encoded_state)

        # 策略输出
        policy = self.policy_network(encoded_state)

        # 动态权重计算
        objective_values = torch.cat([values[obj] for obj in values.keys()], dim=-1)
        weight_input = torch.cat([encoded_state, objective_values], dim=-1)
        objective_weights = self.weight_network(weight_input)

        # 加权总价值
        weighted_value = sum(values[obj] * objective_weights[:, i:i+1]
                           for i, obj in enumerate(values.keys()))

        return {
            'policy': policy,
            'values': values,
            'weighted_value': weighted_value,
            'objective_weights': objective_weights,
            'encoded_state': encoded_state
        }

    def select_action(self, state: torch.Tensor,
                     temporal_context: Optional[torch.Tensor] = None,
                     training: bool = True) -> Tuple[int, Dict[str, torch.Tensor]]:
        """
        选择动作

        Args:
            state: 当前状态
            temporal_context: 时序上下文
            training: 是否训练模式

        Returns:
            选择的动作和相关信息
        """
        with torch.no_grad():
            output = self.forward(state, temporal_context)
            policy = output['policy']

            if training and random.random() < self.epsilon:
                # 探索
                action = random.randint(0, self.action_dim - 1)
            else:
                # 利用
                action = torch.argmax(policy, dim=-1).item()

            return action, output

    def store_experience(self, state: torch.Tensor, action: int, reward: Dict[str, float],
                        next_state: torch.Tensor, done: bool, temporal_context: Optional[torch.Tensor] = None):
        """
        存储经验

        Args:
            state: 当前状态
            action: 选择的动作
            reward: 多目标奖励
            next_state: 下一状态
            done: 是否结束
            temporal_context: 时序上下文
        """
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'temporal_context': temporal_context
        }
        self.replay_buffer.append(experience)

    def update_policy(self, batch_size: int = 32) -> Dict[str, float]:
        """
        更新策略网络

        Args:
            batch_size: 批次大小

        Returns:
            训练损失
        """
        if len(self.replay_buffer) < batch_size:
            return {}

        # 采样批次
        batch = random.sample(self.replay_buffer, batch_size)

        # 准备数据
        states = torch.stack([exp['state'] for exp in batch])
        actions = torch.tensor([exp['action'] for exp in batch])
        rewards = [exp['reward'] for exp in batch]
        next_states = torch.stack([exp['next_state'] for exp in batch])
        dones = torch.tensor([exp['done'] for exp in batch], dtype=torch.float32)
        temporal_contexts = [exp['temporal_context'] for exp in batch if exp['temporal_context'] is not None]

        # 计算目标价值
        with torch.no_grad():
            next_outputs = self.forward(next_states)
            next_values = next_outputs['weighted_value']
            target_values = next_values * self.gamma * (1 - dones.unsqueeze(-1))

        # 计算当前价值
        current_outputs = self.forward(states)
        current_values = current_outputs['weighted_value']

        # 计算奖励
        batch_rewards = []
        for reward_dict in rewards:
            # 多目标奖励加权
            weighted_reward = sum(reward_dict.get(obj, 0) * current_outputs['objective_weights'][0, i].item()
                               for i, obj in enumerate(['glucose_control', 'nutrition_balance',
                                                       'patient_satisfaction', 'cultural_adaptation']))
            batch_rewards.append(weighted_reward)

        batch_rewards = torch.tensor(batch_rewards, dtype=torch.float32).unsqueeze(-1)
        target_values = target_values + batch_rewards

        # 价值损失
        value_loss = F.mse_loss(current_values, target_values)

        # 策略损失
        policy_loss = -torch.log(current_outputs['policy'].gather(1, actions.unsqueeze(-1))).mean()

        # 总损失
        total_loss = value_loss + policy_loss

        # 反向传播
        if self.optimizer is not None:
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        # 更新探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return {
            'total_loss': total_loss.item(),
            'value_loss': value_loss.item(),
            'policy_loss': policy_loss.item(),
            'epsilon': self.epsilon
        }

    def get_recommendation(self, state: torch.Tensor,
                          temporal_context: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        获取推荐

        Args:
            state: 当前状态
            temporal_context: 时序上下文

        Returns:
            推荐结果
        """
        with torch.no_grad():
            output = self.forward(state, temporal_context)

            # 动作解释
            action_descriptions = {
                0: "增加碳水化合物摄入",
                1: "减少碳水化合物摄入",
                2: "增加运动量",
                3: "调整胰岛素剂量",
                4: "保持当前状态"
            }

            selected_action = torch.argmax(output['policy'], dim=-1).item()

            return {
                'recommended_action': selected_action,
                'action_description': action_descriptions[selected_action],
                'confidence': output['policy'][0, selected_action].item(),
                'objective_values': {obj: val.item() for obj, val in output['values'].items()},
                'objective_weights': output['objective_weights'][0].tolist(),
                'overall_value': output['weighted_value'].item()
            }

__all__ = ["'logger'", "'MultiScaleSpatioTemporalAlignmentNetwork'", "'AdaptiveFineTuningModule'", "'AcademicGlucosePredictor'", "'EnhancedGluFormer'", "'EnhancedMultiModalFusion'", "'ReinforcementLearningDecisionEngine'"]
