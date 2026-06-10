#!/usr/bin/env python3
"""
增强版GluFormer - 基于PREDICT 1研究和临床文献的改进

核心改进：
1. 餐后反应建模 - 区分餐前/餐后状态
2. 个体变异性建模 - 基于肠道微生物和基因因素
3. 临床指标预测 - HOMA-IR、餐后峰值等
4. 营养学整合 - GI/GL指数、碳水化合物质量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class EnhancedGluFormer(nn.Module):
    """
    增强版GluFormer - 基于临床研究的改进

    主要改进：
    1. 餐后时间窗口建模
    2. 个体变异性编码
    3. 临床指标预测头
    4. 营养因素整合
    """

    def __init__(self, input_dim: int = 327, hidden_dim: int = 256,
                 output_dim: int = 6, num_experts: int = 4,
                 individual_features_dim: int = 128,
                 nutritional_features_dim: int = 64):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.individual_features_dim = individual_features_dim
        self.nutritional_features_dim = nutritional_features_dim

        # 个体变异性编码器 (基于PREDICT 1研究的发现)
        self.individual_encoder = nn.Sequential(
            nn.Linear(individual_features_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )

        # 营养因素编码器 (GI/GL等)
        self.nutritional_encoder = nn.Sequential(
            nn.Linear(nutritional_features_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )

        # 时序建模 - LSTM + GRU融合
        self.lstm = nn.LSTM(
            input_dim + hidden_dim // 4 + hidden_dim // 4,  # 输入 + 个体 + 营养特征
            hidden_dim, num_layers=2, batch_first=True, dropout=0.1
        )

        self.gru = nn.GRU(
            input_dim + hidden_dim // 4 + hidden_dim // 4,
            hidden_dim, num_layers=2, batch_first=True, dropout=0.1
        )

        # 餐后状态注意力 (区分餐前/餐后)
        self.postprandial_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, dropout=0.1, batch_first=True
        )

        # 交叉注意力融合
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, dropout=0.1, batch_first=True
        )

        # 临床指标预测头
        self.clinical_heads = nn.ModuleDict({
            'glucose_iauc': nn.Linear(hidden_dim, 1),      # 餐后血糖AUC
            'glucose_peak': nn.Linear(hidden_dim, 1),      # 餐后峰值
            'tir': nn.Linear(hidden_dim, 1),               # Time in Range
            'tbr': nn.Linear(hidden_dim, 1),               # Time Below Range
            'tar': nn.Linear(hidden_dim, 1),               # Time Above Range
            'homa_ir': nn.Linear(hidden_dim, 1),           # 胰岛素抵抗
            'glycemic_variability': nn.Linear(hidden_dim, 1)  # 血糖波动性
        })

        # 多时间步血糖预测头
        self.glucose_prediction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(output_dim)
        ])

        # 个体变异性建模 (基于PREDICT 1研究的发现)
        self.variability_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # 输出变异性因子 0-1
        )

        self.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def forward(self, glucose_history: torch.Tensor,
                individual_features: Optional[torch.Tensor] = None,
                nutritional_features: Optional[torch.Tensor] = None,
                meal_timing: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播 - 增强版GluFormer

        Args:
            glucose_history: 血糖历史序列 [batch_size, seq_len, input_dim]
            individual_features: 个体特征 [batch_size, individual_features_dim]
            nutritional_features: 营养特征 [batch_size, nutritional_features_dim]
            meal_timing: 餐食时间标记 [batch_size, seq_len] (0=餐前, 1=餐后)

        Returns:
            预测结果字典
        """
        batch_size, seq_len, _ = glucose_history.shape

        # 处理个体特征
        if individual_features is None:
            individual_features = torch.zeros(batch_size, self.individual_features_dim).to(glucose_history.device)

        individual_encoded = self.individual_encoder(individual_features)  # [batch, hidden_dim//4]
        individual_expanded = individual_encoded.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, hidden_dim//4]

        # 处理营养特征
        if nutritional_features is None:
            nutritional_features = torch.zeros(batch_size, self.nutritional_features_dim).to(glucose_history.device)

        nutritional_encoded = self.nutritional_encoder(nutritional_features)  # [batch, hidden_dim//4]
        nutritional_expanded = nutritional_encoded.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, hidden_dim//4]

        # 拼接所有输入特征
        combined_input = torch.cat([
            glucose_history,      # [batch, seq_len, input_dim]
            individual_expanded,   # [batch, seq_len, hidden_dim//4]
            nutritional_expanded   # [batch, seq_len, hidden_dim//4]
        ], dim=2)  # [batch, seq_len, input_dim + hidden_dim//2]

        # 双分支时序建模
        lstm_out, _ = self.lstm(combined_input)
        gru_out, _ = self.gru(combined_input)

        # 餐后状态建模
        if meal_timing is not None:
            # 创建餐后mask
            postprandial_mask = meal_timing.unsqueeze(2).expand(-1, -1, self.hidden_dim)  # [batch, seq_len, hidden_dim]

            # 餐后注意力
            lstm_postprandial, _ = self.postprandial_attention(
                lstm_out, lstm_out, lstm_out,
                key_padding_mask=(meal_timing == 0)
            )
            gru_postprandial, _ = self.postprandial_attention(
                gru_out, gru_out, gru_out,
                key_padding_mask=(meal_timing == 0)
            )

            # 融合餐前餐后特征
            lstm_features = lstm_postprandial
            gru_features = gru_postprandial
        else:
            lstm_features = lstm_out
            gru_features = gru_out

        # 交叉注意力融合
        cross_fused, _ = self.cross_attention(
            query=lstm_features,
            key=gru_features,
            value=gru_features
        )

        # 最终特征表示
        final_features = cross_fused[:, -1, :]  # 取最后一个时间步 [batch, hidden_dim]

        # 计算临床指标
        clinical_predictions = {}
        for metric_name, head in self.clinical_heads.items():
            clinical_predictions[metric_name] = head(final_features).squeeze(-1)

        # 多时间步血糖预测
        glucose_predictions = []
        for i, head in enumerate(self.glucose_prediction_heads):
            pred = head(final_features).squeeze(-1)
            glucose_predictions.append(pred)

        glucose_predictions = torch.stack(glucose_predictions, dim=1)  # [batch, output_dim]

        # 个体变异性建模
        variability_factor = self.variability_encoder(final_features).squeeze(-1)

        # 应用个体变异性 (基于PREDICT 1研究的发现)
        glucose_predictions = glucose_predictions * (1 + variability_factor.unsqueeze(1))

        return {
            # 主要预测结果
            'glucose_predictions': glucose_predictions,
            'variability_factor': variability_factor,

            # 临床指标
            'clinical_metrics': clinical_predictions,

            # 中间特征 (用于分析)
            'final_features': final_features,
            'lstm_features': lstm_features,
            'gru_features': gru_features,
            'individual_encoded': individual_encoded,
            'nutritional_encoded': nutritional_encoded
        }


class PersonalizedGluFormerPredictor(nn.Module):
    """
    个性化GluFormer预测器
    基于个体特征的动态模型调整
    """

    def __init__(self, base_model: EnhancedGluFormer,
                 personalization_dim: int = 64):
        super().__init__()

        self.base_model = base_model
        self.personalization_dim = personalization_dim

        # 个性化适配层
        self.personalization_adapter = nn.Sequential(
            nn.Linear(self.base_model.individual_features_dim, personalization_dim),
            nn.ReLU(),
            nn.Linear(personalization_dim, self.base_model.hidden_dim),
            nn.Sigmoid()
        )

        # LoRA风格的个性化微调
        self.lora_rank = 8
        self.lora_alpha = 16.0

        # 为关键层添加LoRA
        self._add_lora_to_model()

    def _add_lora_to_model(self):
        """为模型添加LoRA适配器"""
        # 为LSTM和GRU添加LoRA
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Linear) and 'lstm' in name.lower():
                self._add_lora_adapter(module, name)

    def _add_lora_adapter(self, module: nn.Linear, name: str):
        """添加LoRA适配器到Linear层"""
        in_features, out_features = module.in_features, module.out_features

        # 冻结原始权重
        module.weight.requires_grad = False
        if module.bias is not None:
            module.bias.requires_grad = False

        # 添加LoRA适配器
        self.register_parameter(f'{name}_lora_A',
                               nn.Parameter(torch.randn(in_features, self.lora_rank)))
        self.register_parameter(f'{name}_lora_B',
                               nn.Parameter(torch.randn(self.lora_rank, out_features)))

        # 保存原始forward方法
        original_forward = module.forward

        def lora_forward(x):
            # 原始输出
            original_out = original_forward(x)
            # LoRA输出
            lora_out = (x @ self.get_parameter(f'{name}_lora_A') @
                       self.get_parameter(f'{name}_lora_B'))
            lora_out = lora_out * (self.lora_alpha / self.lora_rank)
            return original_out + lora_out

        module.forward = lora_forward

    def forward(self, *args, **kwargs):
        """前向传播"""
        return self.base_model(*args, **kwargs)


class PostprandialResponseModel(nn.Module):
    """
    餐后反应预测模型
    基于PREDICT 1研究的发现，实现个性化餐后反应预测
    """

    def __init__(self, gluformer: EnhancedGluFormer,
                 microbiome_dim: int = 128,
                 genetic_dim: int = 64):
        super().__init__()

        self.gluformer = gluformer
        self.microbiome_dim = microbiome_dim
        self.genetic_dim = genetic_dim

        # 肠道微生物组编码器 (PREDICT 1研究的关键发现)
        self.microbiome_encoder = nn.Sequential(
            nn.Linear(microbiome_dim, gluformer.hidden_dim // 2),
            nn.LayerNorm(gluformer.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(gluformer.hidden_dim // 2, gluformer.hidden_dim // 4)
        )

        # 基因因素编码器
        self.genetic_encoder = nn.Sequential(
            nn.Linear(genetic_dim, gluformer.hidden_dim // 2),
            nn.LayerNorm(gluformer.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(gluformer.hidden_dim // 2, gluformer.hidden_dim // 4)
        )

        # 多模态融合
        self.multimodal_fusion = nn.Sequential(
            nn.Linear(gluformer.hidden_dim + gluformer.hidden_dim // 4 + gluformer.hidden_dim // 4,
                     gluformer.hidden_dim),
            nn.LayerNorm(gluformer.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, glucose_history: torch.Tensor,
                microbiome_features: Optional[torch.Tensor] = None,
                genetic_features: Optional[torch.Tensor] = None,
                individual_features: Optional[torch.Tensor] = None,
                nutritional_features: Optional[torch.Tensor] = None,
                meal_timing: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        个性化餐后反应预测

        Args:
            glucose_history: 血糖历史
            microbiome_features: 肠道微生物组特征
            genetic_features: 基因因素
            individual_features: 个体特征
            nutritional_features: 营养特征
            meal_timing: 餐食时间标记

        Returns:
            个性化预测结果
        """

        # 基础GluFormer预测
        base_predictions = self.gluformer(
            glucose_history=glucose_history,
            individual_features=individual_features,
            nutritional_features=nutritional_features,
            meal_timing=meal_timing
        )

        # 处理微生物组特征
        if microbiome_features is None:
            microbiome_features = torch.zeros(glucose_history.shape[0], self.microbiome_dim).to(glucose_history.device)

        microbiome_encoded = self.microbiome_encoder(microbiome_features)

        # 处理基因特征
        if genetic_features is None:
            genetic_features = torch.zeros(glucose_history.shape[0], self.genetic_dim).to(glucose_history.device)

        genetic_encoded = self.genetic_encoder(genetic_features)

        # 多模态融合
        final_features = base_predictions['final_features']
        multimodal_features = torch.cat([
            final_features,      # GluFormer特征
            microbiome_encoded,  # 微生物组特征
            genetic_encoded      # 基因特征
        ], dim=1)

        fused_features = self.multimodal_fusion(multimodal_features)

        # 个性化预测调整
        personalization_factor = torch.sigmoid(fused_features.mean(dim=1, keepdim=True))

        # 调整血糖预测
        adjusted_predictions = base_predictions['glucose_predictions'] * personalization_factor

        return {
            **base_predictions,
            'personalized_predictions': adjusted_predictions,
            'personalization_factor': personalization_factor.squeeze(),
            'microbiome_encoded': microbiome_encoded,
            'genetic_encoded': genetic_encoded,
            'multimodal_fused': fused_features
        }


if __name__ == "__main__":
    # 测试增强版GluFormer
    model = EnhancedGluFormer()

    # 测试输入
    batch_size, seq_len = 4, 24
    glucose_history = torch.randn(batch_size, seq_len, 327)
    individual_features = torch.randn(batch_size, 128)
    nutritional_features = torch.randn(batch_size, 64)

    outputs = model(
        glucose_history=glucose_history,
        individual_features=individual_features,
        nutritional_features=nutritional_features
    )

    print("Enhanced GluFormer test:")
    print(f"Glucose predictions shape: {outputs['glucose_predictions'].shape}")
    print(f"Clinical metrics: {list(outputs['clinical_metrics'].keys())}")
    print(f"Variability factor shape: {outputs['variability_factor'].shape}")

    # 测试个性化模型
    personalized_model = PostprandialResponseModel(model)

    microbiome_features = torch.randn(batch_size, 128)
    genetic_features = torch.randn(batch_size, 64)

    personalized_outputs = personalized_model(
        glucose_history=glucose_history,
        microbiome_features=microbiome_features,
        genetic_features=genetic_features,
        individual_features=individual_features,
        nutritional_features=nutritional_features
    )

    print("Personalized model test:")
    print(f"Personalized predictions shape: {personalized_outputs['personalized_predictions'].shape}")
    print(f"Personalization factor shape: {personalized_outputs['personalization_factor'].shape}")
