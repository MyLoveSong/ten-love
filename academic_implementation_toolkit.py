#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学术期刊级别血糖预测系统实施工具包
包含所有核心创新模块和实验框架
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, recall_score
from sklearn.ensemble import IsolationForest
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    SMOTE = None
    print("[警告] 未安装 imblearn，SMOTE 数据增强功能不可用。请通过 pip install imbalanced-learn 安装。");
from scipy.integrate import solve_ivp
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 第一阶段：技术创新与算法优化
# ============================================================================

class SpatioTemporalAttention(nn.Module):
    """
    创新点1：时空注意力机制
    ----------------------
    该模块联合建模血糖数据的时间动态（temporal）和特征空间（spatial）相关性。
    - 时间维度注意力：捕捉血糖随时间变化的依赖关系
    - 空间维度注意力：建模不同特征间的交互与相关性
    - 特征融合：通过可配置融合层整合时空信息
    - 新增：时空联合建模和注意力可视化

    Args:
        input_dim (int): 输入特征维度
        num_heads (int): 多头注意力头数
        dropout (float): dropout概率，防止过拟合
        fusion_type (str): 融合方式，'linear'（默认）或'add'等
        use_joint_modeling (bool): 是否使用时空联合建模
    Input:
        x: torch.Tensor, shape (batch_size, seq_len, input_dim)
    Output:
        output: torch.Tensor, shape (batch_size, seq_len, input_dim)
    """
    def __init__(self, input_dim: int, num_heads: int = 8, dropout: float = 0.1, 
                 fusion_type: str = 'linear', use_joint_modeling: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.fusion_type = fusion_type
        self.use_joint_modeling = use_joint_modeling

        # 时间维度注意力
        self.temporal_attention = nn.MultiheadAttention(
            input_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # 空间维度注意力（自定义实现，避免维度不匹配）
        self.spatial_query = nn.Linear(input_dim, input_dim)
        self.spatial_key = nn.Linear(input_dim, input_dim)
        self.spatial_value = nn.Linear(input_dim, input_dim)
        self.spatial_scale = input_dim ** -0.5
        
        # 新增：时空联合建模
        if use_joint_modeling:
            self.joint_attention = nn.MultiheadAttention(
                input_dim, num_heads, dropout=dropout, batch_first=True
            )
            self.joint_norm = nn.LayerNorm(input_dim)
        
        # 融合层
        if fusion_type == 'linear':
            fusion_input_dim = input_dim * (3 if use_joint_modeling else 2)
            self.fusion_layer = nn.Sequential(
                nn.Linear(fusion_input_dim, input_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(input_dim * 2, input_dim),
                nn.LayerNorm(input_dim)
            )
        elif fusion_type == 'add':
            self.fusion_layer = nn.Identity()
        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")
        
        # 层归一化
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        
        # 新增：注意力权重存储（用于可视化）
        self.attention_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: (batch_size, seq_len, input_dim)
        Returns:
            output: (batch_size, seq_len, input_dim)
        """
        # 时间维度注意力
        temporal_out, temporal_weights = self.temporal_attention(x, x, x)
        temporal_out = self.norm1(x + temporal_out)
        
        # 空间维度注意力（自定义实现）
        # 计算特征间的注意力权重
        Q = self.spatial_query(temporal_out)  # (batch, seq_len, input_dim)
        K = self.spatial_key(temporal_out)    # (batch, seq_len, input_dim)
        V = self.spatial_value(temporal_out)  # (batch, seq_len, input_dim)
        
        # 计算注意力分数：对特征维度做注意力
        # 转置为 (batch, input_dim, seq_len) 以便在特征维度计算注意力
        Q_t = Q.transpose(1, 2)  # (batch, input_dim, seq_len)
        K_t = K.transpose(1, 2)  # (batch, input_dim, seq_len)
        V_t = V.transpose(1, 2)  # (batch, input_dim, seq_len)
        
        # 计算注意力权重
        attention_scores = torch.matmul(Q_t, K_t.transpose(-2, -1)) * self.spatial_scale
        spatial_weights = F.softmax(attention_scores, dim=-1)
        spatial_weights = F.dropout(spatial_weights, self.dropout, self.training)
        
        # 应用注意力权重
        spatial_out_t = torch.matmul(spatial_weights, V_t)  # (batch, input_dim, seq_len)
        spatial_out = spatial_out_t.transpose(1, 2)  # (batch, seq_len, input_dim)
        spatial_out = self.norm2(temporal_out + spatial_out)
        
        # 新增：时空联合建模
        if self.use_joint_modeling:
            joint_out, joint_weights = self.joint_attention(spatial_out, spatial_out, spatial_out)
            joint_out = self.joint_norm(spatial_out + joint_out)
        else:
            joint_out = spatial_out
            joint_weights = None
        
        # 融合时间空间特征
        if self.fusion_type == 'add':
            # 直接相加，保持维度一致
            output = temporal_out + spatial_out + (joint_out if self.use_joint_modeling else 0)
        else:
            # 拼接后通过线性层
            if self.use_joint_modeling:
                combined = torch.cat([temporal_out, spatial_out, joint_out], dim=-1)
            else:
                combined = torch.cat([temporal_out, spatial_out], dim=-1)
            output = self.fusion_layer(combined)
        
        # 存储注意力权重用于可视化
        self.attention_weights = {
            'temporal': temporal_weights,
            'spatial': spatial_weights,
            'joint': joint_weights
        }
        
        return output
    
    def get_attention_weights(self):
        """获取注意力权重用于可视化"""
        return self.attention_weights
    
    def visualize_attention(self, save_path: str = None):
        """可视化注意力权重"""
        if self.attention_weights is None:
            print("警告：没有可用的注意力权重，请先运行前向传播")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 时间注意力可视化
        if self.attention_weights['temporal'] is not None:
            temporal_weights = self.attention_weights['temporal'].detach().cpu().numpy()
            # 取第一个样本的第一个头
            temporal_vis = temporal_weights[0, 0] if temporal_weights.ndim == 4 else temporal_weights[0]
            im1 = axes[0].imshow(temporal_vis, cmap='viridis', aspect='auto')
            axes[0].set_title('时间注意力权重')
            axes[0].set_xlabel('序列位置')
            axes[0].set_ylabel('序列位置')
            plt.colorbar(im1, ax=axes[0])
        
        # 空间注意力可视化
        if self.attention_weights['spatial'] is not None:
            spatial_weights = self.attention_weights['spatial'].detach().cpu().numpy()
            # 取第一个样本
            spatial_vis = spatial_weights[0]
            im2 = axes[1].imshow(spatial_vis, cmap='plasma', aspect='auto')
            axes[1].set_title('空间注意力权重')
            axes[1].set_xlabel('特征维度')
            axes[1].set_ylabel('特征维度')
            plt.colorbar(im2, ax=axes[1])
        
        # 联合注意力可视化
        if self.attention_weights['joint'] is not None:
            joint_weights = self.attention_weights['joint'].detach().cpu().numpy()
            joint_vis = joint_weights[0, 0] if joint_weights.ndim == 4 else joint_weights[0]
            im3 = axes[2].imshow(joint_vis, cmap='inferno', aspect='auto')
            axes[2].set_title('时空联合注意力权重')
            axes[2].set_xlabel('序列位置')
            axes[2].set_ylabel('序列位置')
            plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class MultiScaleFeatureExtractor(nn.Module):
    """
    创新点2：多尺度特征提取器
    ------------------------
    同时提取短期、中期、长期血糖变化特征，提升模型对不同时间尺度规律的建模能力。
    - 短期：如1-3天，捕捉快速波动
    - 中期：如1-2周，捕捉周期趋势
    - 长期：如1个月，捕捉全局变化
    - 新增：自适应时间窗口和特征重要性分析

    Args:
        input_dim (int): 输入特征维度
        hidden_dim (int): LSTM隐藏层维度
        short_window (int): 短期窗口长度，默认3
        medium_window (int): 中期窗口长度，默认14
        long_window (int): 长期窗口长度，默认30
        use_adaptive_windows (bool): 是否使用自适应时间窗口
        use_attention_fusion (bool): 是否使用注意力融合
    Input:
        x: torch.Tensor, shape (batch_size, seq_len, input_dim)
    Output:
        output: torch.Tensor, shape (batch_size, hidden_dim)
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, short_window: int = 3, 
                 medium_window: int = 14, long_window: int = 30, 
                 use_adaptive_windows: bool = True, use_attention_fusion: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.short_window = short_window
        self.medium_window = medium_window
        self.long_window = long_window
        self.use_adaptive_windows = use_adaptive_windows
        self.use_attention_fusion = use_attention_fusion
        
        # 短期模式
        self.short_term = nn.LSTM(input_dim, hidden_dim, batch_first=True, dropout=0.1)
        # 中期模式
        self.medium_term = nn.LSTM(input_dim, hidden_dim, batch_first=True, dropout=0.1)
        # 长期模式
        self.long_term = nn.LSTM(input_dim, hidden_dim, batch_first=True, dropout=0.1)
        
        # 新增：自适应时间窗口
        if use_adaptive_windows:
            self.window_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
            self.window_norm = nn.LayerNorm(hidden_dim)
        
        # 新增：注意力融合
        if use_attention_fusion:
            self.fusion_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
            self.fusion_norm = nn.LayerNorm(hidden_dim)
        
        # 特征融合
        fusion_input_dim = hidden_dim * 3
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # 新增：特征重要性分析
        self.feature_importance = nn.Parameter(torch.ones(input_dim))
        self.importance_weights = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_dim)
        Returns:
            output: (batch_size, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # 计算特征重要性权重
        importance_weights = F.softmax(self.feature_importance, dim=0)
        self.importance_weights = importance_weights.detach()
        
        # 应用特征重要性权重
        weighted_x = x * importance_weights.unsqueeze(0).unsqueeze(0)
        
        # 短期模式
        short_window = min(self.short_window, seq_len)
        short_input = weighted_x[:, -short_window:, :]
        short_out, _ = self.short_term(short_input)
        short_features = short_out[:, -1, :]  # (batch_size, hidden_dim)
        
        # 中期模式
        medium_window = min(self.medium_window, seq_len)
        medium_input = weighted_x[:, -medium_window:, :]
        medium_out, _ = self.medium_term(medium_input)
        medium_features = medium_out[:, -1, :]  # (batch_size, hidden_dim)
        
        # 长期模式
        long_window = min(self.long_window, seq_len)
        long_input = weighted_x[:, -long_window:, :]
        long_out, _ = self.long_term(long_input)
        long_features = long_out[:, -1, :]  # (batch_size, hidden_dim)
        
        # 新增：自适应时间窗口
        if self.use_adaptive_windows:
            # 将不同尺度的特征作为序列进行注意力融合
            multi_scale_features = torch.stack([short_features, medium_features, long_features], dim=1)
            # (batch_size, 3, hidden_dim)
            
            attended_features, _ = self.window_attention(multi_scale_features, multi_scale_features, multi_scale_features)
            attended_features = self.window_norm(multi_scale_features + attended_features)
            
            # 取平均作为最终特征
            short_features = attended_features[:, 0, :]
            medium_features = attended_features[:, 1, :]
            long_features = attended_features[:, 2, :]
        
        # 新增：注意力融合
        if self.use_attention_fusion:
            # 将三个尺度的特征进行注意力融合
            scale_features = torch.stack([short_features, medium_features, long_features], dim=1)
            # (batch_size, 3, hidden_dim)
            
            fused_features, _ = self.fusion_attention(scale_features, scale_features, scale_features)
            fused_features = self.fusion_norm(scale_features + fused_features)
            
            # 取平均
            combined_features = torch.mean(fused_features, dim=1)
        else:
            # 传统拼接方式
            combined_features = torch.cat([short_features, medium_features, long_features], dim=-1)
        
        # 特征融合
        if not self.use_attention_fusion:
            output = self.fusion_layer(combined_features)
        else:
            output = combined_features
        
        return output
    
    def get_feature_importance(self):
        """获取特征重要性权重"""
        return self.importance_weights
    
    def visualize_feature_importance(self, feature_names: List[str] = None, save_path: str = None):
        """可视化特征重要性"""
        if self.importance_weights is None:
            print("警告：没有可用的特征重要性权重，请先运行前向传播")
            return
        
        importance = self.importance_weights.cpu().numpy()
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(importance))]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(importance)), importance)
        plt.xlabel('特征索引')
        plt.ylabel('重要性权重')
        plt.title('多尺度特征提取器 - 特征重要性分析')
        plt.xticks(range(len(importance)), feature_names, rotation=45, ha='right')
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class CrossModalContrastiveLearning(nn.Module):
    """
    创新点3：跨模态对比学习
    ----------------------
    实现图像、文本、数值三种模态的语义对齐，通过对比学习增强模态间的互信息。
    - 对比学习：增强模态间的互信息
    - 温度参数：控制对比学习的强度
    - 多模态融合：提升整体性能
    - 新增：多模态对比策略和模态重要性学习

    Args:
        feature_dim (int): 特征维度，默认512
        temperature (float): 温度参数，控制对比学习强度
        use_hard_negatives (bool): 是否使用困难负样本
        use_momentum_encoder (bool): 是否使用动量编码器
    Input:
        image: torch.Tensor, shape (batch_size, 3, H, W)
        text: torch.Tensor, shape (batch_size, text_length)
        numerical: torch.Tensor, shape (batch_size, numerical_dim)
    Output:
        losses: Dict[str, torch.Tensor], 包含各种损失
    """
    def __init__(self, feature_dim: int = 512, temperature: float = 0.07, 
                 use_hard_negatives: bool = True, use_momentum_encoder: bool = True):
        super().__init__()
        self.feature_dim = feature_dim
        self.temperature = nn.Parameter(torch.ones([]) * temperature)
        self.use_hard_negatives = use_hard_negatives
        self.use_momentum_encoder = use_momentum_encoder
        
        # 图像编码器（简化版，实际可以使用预训练的Vision Transformer）
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, feature_dim)
        )
        
        # 文本编码器（简化版，实际可以使用BERT）
        self.text_encoder = nn.Sequential(
            nn.Linear(768, 512),  # 假设输入是BERT特征
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, feature_dim)
        )
        
        # 数值编码器
        self.numerical_encoder = nn.Sequential(
            nn.Linear(17, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, feature_dim)
        )
        
        # 新增：模态重要性学习
        self.modality_importance = nn.Parameter(torch.ones(3))  # [image, text, numerical]
        
        # 新增：动量编码器（用于对比学习）
        if use_momentum_encoder:
            self.momentum_image_encoder = self._create_momentum_encoder(self.image_encoder)
            self.momentum_text_encoder = self._create_momentum_encoder(self.text_encoder)
            self.momentum_numerical_encoder = self._create_momentum_encoder(self.numerical_encoder)
            self.momentum = 0.999
        
        # 新增：多模态融合层
        self.multimodal_fusion = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        # 新增：对比学习投影头
        self.image_projection = nn.Linear(feature_dim, feature_dim)
        self.text_projection = nn.Linear(feature_dim, feature_dim)
        self.numerical_projection = nn.Linear(feature_dim, feature_dim)
        
    def _create_momentum_encoder(self, encoder):
        """创建动量编码器"""
        momentum_encoder = type(encoder)(*encoder.children())
        for param in momentum_encoder.parameters():
            param.requires_grad = False
        return momentum_encoder
    
    def _update_momentum_encoders(self):
        """更新动量编码器"""
        if not self.use_momentum_encoder:
            return
        
        for param, momentum_param in zip(self.image_encoder.parameters(), 
                                        self.momentum_image_encoder.parameters()):
            momentum_param.data = self.momentum * momentum_param.data + (1 - self.momentum) * param.data
        
        for param, momentum_param in zip(self.text_encoder.parameters(), 
                                        self.momentum_text_encoder.parameters()):
            momentum_param.data = self.momentum * momentum_param.data + (1 - self.momentum) * param.data
        
        for param, momentum_param in zip(self.numerical_encoder.parameters(), 
                                        self.momentum_numerical_encoder.parameters()):
            momentum_param.data = self.momentum * momentum_param.data + (1 - self.momentum) * param.data
    
    def forward(self, image: torch.Tensor, text: torch.Tensor, numerical: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        Args:
            image: (batch_size, 3, H, W)
            text: (batch_size, text_length)
            numerical: (batch_size, numerical_dim)
        Returns:
            losses: 包含各种损失的字典
        """
        batch_size = image.size(0)
        
        # 编码各模态
        img_features = self.image_encoder(image)
        text_features = self.text_encoder(text)
        num_features = self.numerical_encoder(numerical)
        
        # 计算模态重要性权重
        modality_weights = F.softmax(self.modality_importance, dim=0)
        
        # 应用模态重要性权重
        img_features = img_features * modality_weights[0]
        text_features = text_features * modality_weights[1]
        num_features = num_features * modality_weights[2]
        
        # 多模态融合
        combined_features = torch.cat([img_features, text_features, num_features], dim=-1)
        fused_features = self.multimodal_fusion(combined_features)
        
        # 对比学习投影
        img_proj = self.image_projection(img_features)
        text_proj = self.text_projection(text_features)
        num_proj = self.numerical_projection(num_features)
        
        # 归一化特征
        img_proj = F.normalize(img_proj, dim=-1)
        text_proj = F.normalize(text_proj, dim=-1)
        num_proj = F.normalize(num_proj, dim=-1)
        
        # 计算对比损失
        losses = {}
        
        # 图像-文本对比损失
        img_text_logits = torch.mm(img_proj, text_proj.t()) / self.temperature
        img_text_labels = torch.arange(batch_size).to(img_proj.device)
        losses['img_text_loss'] = F.cross_entropy(img_text_logits, img_text_labels)
        
        # 图像-数值对比损失
        img_num_logits = torch.mm(img_proj, num_proj.t()) / self.temperature
        img_num_labels = torch.arange(batch_size).to(img_proj.device)
        losses['img_num_loss'] = F.cross_entropy(img_num_logits, img_num_labels)
        
        # 文本-数值对比损失
        text_num_logits = torch.mm(text_proj, num_proj.t()) / self.temperature
        text_num_labels = torch.arange(batch_size).to(text_proj.device)
        losses['text_num_loss'] = F.cross_entropy(text_num_logits, text_num_labels)
        
        # 新增：困难负样本挖掘
        if self.use_hard_negatives:
            # 找到最困难的负样本（相似度最高的负样本）
            with torch.no_grad():
                img_text_sim = torch.mm(img_proj, text_proj.t())
                img_text_sim.fill_diagonal_(float('-inf'))  # 排除正样本
                hard_negatives = torch.topk(img_text_sim, k=min(3, batch_size-1), dim=1)[1]
            
            # 计算困难负样本损失
            hard_neg_loss = 0
            for i in range(batch_size):
                for j in hard_negatives[i]:
                    hard_neg_loss += F.cosine_embedding_loss(
                        img_proj[i:i+1], text_proj[j:j+1], 
                        torch.tensor([-1]).to(img_proj.device)
                    )
            losses['hard_negative_loss'] = hard_neg_loss / (batch_size * 3)
        
        # 更新动量编码器
        self._update_momentum_encoders()
        
        # 总损失
        losses['total_loss'] = (losses['img_text_loss'] + 
                               losses['img_num_loss'] + 
                               losses['text_num_loss'])
        
        if self.use_hard_negatives:
            losses['total_loss'] += losses['hard_negative_loss']
        
        return losses
    
    def get_modality_importance(self):
        """获取模态重要性权重"""
        return F.softmax(self.modality_importance, dim=0)
    
    def visualize_modality_importance(self, save_path: str = None):
        """可视化模态重要性"""
        importance = self.get_modality_importance().detach().cpu().numpy()
        modality_names = ['图像', '文本', '数值']
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(modality_names, importance, color=['red', 'blue', 'green'])
        plt.ylabel('重要性权重')
        plt.title('跨模态对比学习 - 模态重要性分析')
        
        # 添加数值标签
        for bar, weight in zip(bars, importance):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{weight:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class MetaLearningPersonalization(nn.Module):
    """创新点4：元学习个性化建模"""
    
    def __init__(self, base_model: nn.Module, inner_lr: float = 0.01, adaptation_steps: int = 5):
        super().__init__()
        self.base_model = base_model
        self.inner_lr = inner_lr
        self.adaptation_steps = adaptation_steps
        
    def adapt_to_user(self, user_data: Tuple[torch.Tensor, torch.Tensor]) -> nn.Module:
        """
        为新用户快速适应模型
        Args:
            user_data: (features, targets) 用户数据
        Returns:
            adapted_model: 适应后的模型
        """
        features, targets = user_data
        
        # 克隆基础模型
        adapted_model = type(self.base_model)()
        adapted_model.load_state_dict(self.base_model.state_dict())
        adapted_model.train()
        
        # 快速适应
        optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.inner_lr)
        
        for _ in range(self.adaptation_steps):
            optimizer.zero_grad()
            predictions = adapted_model(features)
            loss = F.mse_loss(predictions, targets)
            loss.backward()
            optimizer.step()
        
        return adapted_model

# ============================================================================
# 第二阶段：实验设计与验证
# ============================================================================

class DataQualityEnhancement:
    """数据质量增强模块"""
    
    def __init__(self):
        self.outlier_detector = IsolationForest(contamination='auto', random_state=42)
        self.imputer = IterativeImputer(max_iter=10, random_state=42)
        self.smote = SMOTE(random_state=42) if SMOTE is not None else None
    
    def outlier_detection(self, data: np.ndarray) -> np.ndarray:
        """异常值检测与处理"""
        outliers = self.outlier_detector.fit_predict(data)
        return data[outliers == 1]
    
    def missing_value_imputation(self, data: np.ndarray) -> np.ndarray:
        """缺失值智能填充"""
        result = self.imputer.fit_transform(data)
        # IterativeImputer通常返回ndarray，若为稀疏则转为ndarray
        if not isinstance(result, np.ndarray):
            result = result.toarray()
        return result
    
    def data_augmentation(self, data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """数据增强"""
        if self.smote is None:
            raise ImportError("SMOTE 未安装，无法进行数据增强。请安装 imbalanced-learn。")
        X_res, y_res = self.smote.fit_resample(data, labels)[:2]
        return np.asarray(X_res), np.asarray(y_res)
    
    def comprehensive_cleaning(self, data: pd.DataFrame) -> pd.DataFrame:
        """综合数据清洗"""
        # 1. 处理缺失值
        data_imputed = self.imputer.fit_transform(data)
        if not isinstance(data_imputed, np.ndarray):
            data_imputed = data_imputed.toarray()
        # 2. 检测异常值
        outliers = self.outlier_detector.fit_predict(data_imputed)
        # outliers==1表示正常点，-1为异常点
        data_clean = data_imputed[outliers == 1]
        # 3. 数据标准化
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_clean)
        return pd.DataFrame(data_scaled, columns=data.columns)

class ComprehensiveEvaluation:
    """综合评估指标"""
    
    def __init__(self):
        self.metrics = {
            'regression': ['MAE', 'RMSE', 'R2', 'MAPE'],
            'classification': ['Accuracy', 'Precision', 'Recall', 'F1'],
            'clinical': ['AUC', 'Sensitivity', 'Specificity', 'PPV', 'NPV']
        }
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """全面评估模型性能"""
        results = {}
        
        # 回归指标
        results['MAE'] = mean_absolute_error(y_true, y_pred)
        results['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
        results['R2'] = r2_score(y_true, y_pred)
        results['MAPE'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # 临床指标
        y_binary = (y_true > 6.1).astype(int)  # 糖尿病阈值
        y_pred_binary = (y_pred > 6.1).astype(int)
        
        results['AUC'] = roc_auc_score(y_binary, y_pred)
        results['Sensitivity'] = recall_score(y_binary, y_pred_binary)
        results['Specificity'] = recall_score(y_binary, y_pred_binary, pos_label=0)
        
        # 计算PPV和NPV
        from sklearn.metrics import precision_score
        results['PPV'] = precision_score(y_binary, y_pred_binary)
        results['NPV'] = precision_score(y_binary, y_pred_binary, pos_label=0)
        
        return results
    
    def plot_results(self, y_true: np.ndarray, y_pred: np.ndarray, save_path: Optional[str] = None):
        """可视化结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 散点图
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6)
        axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('True Values')
        axes[0, 0].set_ylabel('Predictions')
        axes[0, 0].set_title('Prediction vs True Values')
        
        # 残差图
        residuals = y_true - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predictions')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')
        
        # 分布对比
        axes[1, 0].hist(y_true, alpha=0.7, label='True', bins=30)
        axes[1, 0].hist(y_pred, alpha=0.7, label='Predicted', bins=30)
        axes[1, 0].set_xlabel('Glucose Level')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution Comparison')
        axes[1, 0].legend()
        
        # 误差分布
        axes[1, 1].hist(residuals, bins=30, alpha=0.7)
        axes[1, 1].set_xlabel('Prediction Error')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Error Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class AblationStudy:
    """消融实验设计"""
    
    def __init__(self, full_model: nn.Module):
        self.full_model = full_model
        self.evaluator = ComprehensiveEvaluation()
    
    def test_component_importance(self, test_data: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, Dict[str, float]]:
        """测试各组件的重要性"""
        results = {}
        
        # 1. 移除时空注意力
        model_no_temporal = self.remove_temporal_attention()
        results['no_temporal'] = self.evaluate_model(model_no_temporal, test_data)
        
        # 2. 移除多尺度特征
        model_no_multiscale = self.remove_multiscale()
        results['no_multiscale'] = self.evaluate_model(model_no_multiscale, test_data)
        
        # 3. 移除跨模态学习
        model_no_crossmodal = self.remove_crossmodal()
        results['no_crossmodal'] = self.evaluate_model(model_no_crossmodal, test_data)
        
        # 4. 移除个性化
        model_no_personalization = self.remove_personalization()
        results['no_personalization'] = self.evaluate_model(model_no_personalization, test_data)
        
        return results
    
    def remove_temporal_attention(self) -> nn.Module:
        """移除时空注意力机制"""
        # 返回一个简单的线性模型作为占位
        class DummyModel(nn.Module):
            def forward(self, x):
                return x.mean(dim=1, keepdim=True).expand_as(x)
        return DummyModel()
    
    def remove_multiscale(self) -> nn.Module:
        """移除多尺度特征"""
        class DummyModel(nn.Module):
            def forward(self, x):
                return x.mean(dim=1, keepdim=True).expand_as(x)
        return DummyModel()
    
    def remove_crossmodal(self) -> nn.Module:
        """移除跨模态学习"""
        class DummyModel(nn.Module):
            def forward(self, x):
                return x.mean(dim=1, keepdim=True).expand_as(x)
        return DummyModel()
    
    def remove_personalization(self) -> nn.Module:
        """移除个性化"""
        class DummyModel(nn.Module):
            def forward(self, x):
                return x.mean(dim=1, keepdim=True).expand_as(x)
        return DummyModel()
    
    def evaluate_model(self, model: nn.Module, test_data: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """评估模型性能"""
        model.eval()
        with torch.no_grad():
            features, targets = test_data
            predictions = model(features)
            return self.evaluator.evaluate_model(targets.numpy(), predictions.numpy())

# ============================================================================
# 第三阶段：理论基础与创新点
# ============================================================================

class GlucoseDynamicsModel:
    """
    创新理论1：血糖动力学建模
    ------------------------
    结合生理学理论建立血糖变化微分方程，提供理论支撑。
    - 生理学基础：基于血糖-胰岛素动力学
    - 微分方程：描述血糖变化规律
    - 参数拟合：从数据中学习生理参数
    - 新增：多室模型和个性化参数学习

    Args:
        model_type (str): 模型类型，'minimal'（最小模型）或'extended'（扩展模型）
        use_personalization (bool): 是否使用个性化参数
    """
    
    def __init__(self, model_type: str = 'extended', use_personalization: bool = True):
        self.model_type = model_type
        self.use_personalization = use_personalization
        
        # 基础生理参数（基于文献）
        self.base_params = {
            'insulin_sensitivity': 0.1,      # 胰岛素敏感性 (1/min)
            'glucose_production': 2.0,       # 肝脏葡萄糖产生率 (mg/kg/min)
            'glucose_utilization': 1.5,      # 基础葡萄糖利用率 (mg/kg/min)
            'insulin_clearance': 0.1,        # 胰岛素清除率 (1/min)
            'glucose_effectiveness': 0.02,   # 葡萄糖有效性 (1/min)
            'insulin_action_delay': 5.0,     # 胰岛素作用延迟 (min)
            'glucose_absorption': 0.8,       # 葡萄糖吸收率 (1/min)
        }
        
        # 扩展模型参数
        if model_type == 'extended':
            self.base_params.update({
                'glucagon_effect': 0.05,     # 胰高血糖素效应
                'stress_hormone_effect': 0.03, # 应激激素效应
                'exercise_effect': 0.1,      # 运动效应
                'meal_absorption_rate': 0.15, # 餐后吸收率
            })
        
        # 个性化参数（可学习）
        if use_personalization:
            self.personalized_params = nn.Parameter(torch.ones(len(self.base_params)))
        else:
            self.personalized_params = None
    
    def get_parameters(self):
        """获取当前参数"""
        if self.personalized_params is not None:
            # 将个性化参数应用到基础参数
            params = {}
            for i, (key, base_value) in enumerate(self.base_params.items()):
                params[key] = base_value * self.personalized_params[i].item()
            return params
        else:
            return self.base_params.copy()
    
    def differential_equations(self, t: float, state: List[float]) -> List[float]:
        """
        血糖变化微分方程
        Args:
            t: 时间 (min)
            state: [glucose, insulin, glucagon] (mg/dL, μU/mL, pg/mL)
        Returns:
            derivatives: [dglucose/dt, dinsulin/dt, dglucagon/dt]
        """
        glucose, insulin = state[0], state[1]
        params = self.get_parameters()
        
        if self.model_type == 'minimal':
            # 最小模型（Bergman模型）
            # 血糖变化率
            dglucose_dt = (params['glucose_production'] - 
                          params['glucose_utilization'] * glucose -
                          params['insulin_sensitivity'] * insulin * glucose)
            
            # 胰岛素变化率
            dinsulin_dt = -params['insulin_clearance'] * insulin
            
            return [dglucose_dt, dinsulin_dt]
        
        else:
            # 扩展模型（包含更多生理因素）
            glucagon = state[2] if len(state) > 2 else 0
            
            # 血糖变化率（考虑更多因素）
            glucose_effect = params['glucose_effectiveness'] * glucose
            insulin_effect = params['insulin_sensitivity'] * insulin * glucose
            glucagon_effect = params['glucagon_effect'] * glucagon if 'glucagon_effect' in params else 0
            
            dglucose_dt = (params['glucose_production'] - 
                          params['glucose_utilization'] * glucose -
                          glucose_effect - insulin_effect + glucagon_effect)
            
            # 胰岛素变化率
            dinsulin_dt = -params['insulin_clearance'] * insulin
            
            # 胰高血糖素变化率
            if len(state) > 2:
                dglucagon_dt = -0.05 * glucagon  # 简化模型
                return [dglucose_dt, dinsulin_dt, dglucagon_dt]
            else:
                return [dglucose_dt, dinsulin_dt]
    
    def predict_trajectory(self, initial_conditions: List[float], time_span: float, 
                          num_points: int = 100, external_inputs: Dict[str, Any] = None) -> Dict[str, np.ndarray]:
        """
        预测血糖轨迹
        Args:
            initial_conditions: 初始条件 [glucose, insulin, ...]
            time_span: 时间跨度（小时）
            num_points: 预测点数
            external_inputs: 外部输入（如餐食、运动等）
        Returns:
            trajectory: 包含时间、血糖、胰岛素等轨迹的字典
        """
        # 转换为分钟
        time_span_min = time_span * 60
        t_eval = np.linspace(0, time_span_min, num_points)
        
        # 求解微分方程
        solution = solve_ivp(
            self.differential_equations, 
            [0, time_span_min], 
            initial_conditions, 
            method='RK45', 
            t_eval=t_eval,
            args=(external_inputs,) if external_inputs else ()
        )
        
        # 组织结果
        trajectory = {
            'time': solution.t / 60,  # 转换回小时
            'glucose': solution.y[0],
            'insulin': solution.y[1]
        }
        
        if len(initial_conditions) > 2:
            trajectory['glucagon'] = solution.y[2]
        
        return trajectory
    
    def fit_parameters(self, glucose_data: np.ndarray, insulin_data: np.ndarray, 
                      time_data: np.ndarray, method: str = 'optimization') -> Dict[str, float]:
        """
        从数据中拟合生理参数
        Args:
            glucose_data: 血糖数据
            insulin_data: 胰岛素数据
            time_data: 时间数据
            method: 拟合方法，'optimization'或'least_squares'
        Returns:
            fitted_params: 拟合后的参数
        """
        if method == 'optimization':
            return self._fit_parameters_optimization(glucose_data, insulin_data, time_data)
        elif method == 'least_squares':
            return self._fit_parameters_least_squares(glucose_data, insulin_data, time_data)
        else:
            raise ValueError(f"Unsupported fitting method: {method}")
    
    def _fit_parameters_optimization(self, glucose_data: np.ndarray, insulin_data: np.ndarray, 
                                   time_data: np.ndarray) -> Dict[str, float]:
        """使用优化方法拟合参数"""
        from scipy.optimize import minimize
        
        def objective(params):
            # 临时设置参数
            temp_params = self.base_params.copy()
            for i, key in enumerate(temp_params.keys()):
                temp_params[key] = params[i]
            
            # 预测轨迹
            initial_conditions = [glucose_data[0], insulin_data[0]]
            trajectory = self.predict_trajectory(initial_conditions, time_data[-1]/60, len(time_data))
            
            # 计算误差
            glucose_error = np.mean((trajectory['glucose'] - glucose_data)**2)
            insulin_error = np.mean((trajectory['insulin'] - insulin_data)**2)
            
            return glucose_error + insulin_error
        
        # 初始参数
        initial_params = list(self.base_params.values())
        
        # 优化
        result = minimize(objective, initial_params, method='L-BFGS-B')
        
        # 更新参数
        fitted_params = {}
        for i, key in enumerate(self.base_params.keys()):
            fitted_params[key] = result.x[i]
        
        return fitted_params
    
    def _fit_parameters_least_squares(self, glucose_data: np.ndarray, insulin_data: np.ndarray, 
                                    time_data: np.ndarray) -> Dict[str, float]:
        """使用最小二乘法拟合参数"""
        from scipy.optimize import least_squares
        
        def residuals(params):
            # 临时设置参数
            temp_params = self.base_params.copy()
            for i, key in enumerate(temp_params.keys()):
                temp_params[key] = params[i]
            
            # 预测轨迹
            initial_conditions = [glucose_data[0], insulin_data[0]]
            trajectory = self.predict_trajectory(initial_conditions, time_data[-1]/60, len(time_data))
            
            # 计算残差
            glucose_residuals = trajectory['glucose'] - glucose_data
            insulin_residuals = trajectory['insulin'] - insulin_data
            
            return np.concatenate([glucose_residuals, insulin_residuals])
        
        # 初始参数
        initial_params = list(self.base_params.values())
        
        # 最小二乘拟合
        result = least_squares(residuals, initial_params)
        
        # 更新参数
        fitted_params = {}
        for i, key in enumerate(self.base_params.keys()):
            fitted_params[key] = result.x[i]
        
        return fitted_params
    
    def visualize_trajectory(self, trajectory: Dict[str, np.ndarray], save_path: str = None):
        """可视化血糖轨迹"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # 血糖轨迹
        axes[0].plot(trajectory['time'], trajectory['glucose'], 'b-', linewidth=2, label='血糖')
        axes[0].set_ylabel('血糖 (mg/dL)')
        axes[0].set_title('血糖动力学模型预测轨迹')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # 胰岛素轨迹
        axes[1].plot(trajectory['time'], trajectory['insulin'], 'r-', linewidth=2, label='胰岛素')
        axes[1].set_xlabel('时间 (小时)')
        axes[1].set_ylabel('胰岛素 (μU/mL)')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_physiological_insights(self) -> Dict[str, str]:
        """获取生理学洞察"""
        params = self.get_parameters()
        insights = {}
        
        # 胰岛素敏感性分析
        if params['insulin_sensitivity'] > 0.15:
            insights['insulin_sensitivity'] = "胰岛素敏感性较高，血糖控制良好"
        elif params['insulin_sensitivity'] < 0.05:
            insights['insulin_sensitivity'] = "胰岛素敏感性较低，可能存在胰岛素抵抗"
        else:
            insights['insulin_sensitivity'] = "胰岛素敏感性正常"
        
        # 葡萄糖产生分析
        if params['glucose_production'] > 2.5:
            insights['glucose_production'] = "肝脏葡萄糖产生过多，可能导致高血糖"
        elif params['glucose_production'] < 1.5:
            insights['glucose_production'] = "肝脏葡萄糖产生不足，可能导致低血糖"
        else:
            insights['glucose_production'] = "肝脏葡萄糖产生正常"
        
        return insights

class MultimodalInformationTheory:
    """创新理论2：多模态信息论框架"""
    
    def __init__(self, bandwidth: float = 0.2):
        self.bandwidth = bandwidth
        self.modalities = ['image', 'text', 'numerical']
    
    def mutual_information(self, modality1: np.ndarray, modality2: np.ndarray) -> float:
        """计算两个模态间的互信息"""
        # 使用KDE估计互信息
        kde1 = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth)
        kde2 = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth)
        
        kde1.fit(modality1)
        kde2.fit(modality2)
        
        # 计算联合分布和边缘分布
        joint_density = kde1.score_samples(modality1) + kde2.score_samples(modality2)
        marginal_density = kde1.score_samples(modality1)
        
        return np.mean(joint_density - marginal_density)
    
    def information_bottleneck(self, input_data: np.ndarray, target: np.ndarray, beta: float = 1.0) -> float:
        """信息瓶颈理论应用"""
        # 简化实现：返回0.0作为占位
        return 0.0
    
    def calculate_modality_importance(self, modalities_data: Dict[str, np.ndarray], target: np.ndarray) -> Dict[str, float]:
        """计算各模态的重要性"""
        importance = {}
        
        for modality_name, modality_data in modalities_data.items():
            mi = self.mutual_information(modality_data, target)
            importance[modality_name] = mi
        
        return importance

# ============================================================================
# 主实验框架
# ============================================================================

class AcademicGlucosePredictionSystem:
    """学术期刊级别血糖预测系统"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化组件
        self.spatio_temporal_attention = SpatioTemporalAttention(
            input_dim=config['input_dim'],
            num_heads=config['num_heads']
        ).to(self.device)
        
        self.multi_scale_extractor = MultiScaleFeatureExtractor(
            input_dim=config['input_dim']
        ).to(self.device)
        
        self.cross_modal_learning = CrossModalContrastiveLearning(
            feature_dim=config['feature_dim']
        ).to(self.device)
        
        self.meta_learner = MetaLearningPersonalization(
            base_model=self.spatio_temporal_attention
        )
        
        # 数据质量增强
        self.data_enhancer = DataQualityEnhancement()
        
        # 评估器
        self.evaluator = ComprehensiveEvaluation()
        
        # 理论模型
        self.dynamics_model = GlucoseDynamicsModel()
        self.info_theory = MultimodalInformationTheory()
    
    def train(self, train_data: Tuple[torch.Tensor, torch.Tensor], 
              val_data: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, List[float]]:
        """训练模型"""
        train_features, train_targets = train_data
        val_features, val_targets = val_data
        
        # 训练历史
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': []
        }
        
        # 优化器
        optimizer = torch.optim.Adam(self.spatio_temporal_attention.parameters(), lr=self.config['lr'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        
        for epoch in range(self.config['epochs']):
            # 训练
            self.spatio_temporal_attention.train()
            optimizer.zero_grad()
            
            train_predictions = self.spatio_temporal_attention(train_features)
            train_loss = F.mse_loss(train_predictions, train_targets)
            
            train_loss.backward()
            optimizer.step()
            
            # 验证
            self.spatio_temporal_attention.eval()
            with torch.no_grad():
                val_predictions = self.spatio_temporal_attention(val_features)
                val_loss = F.mse_loss(val_predictions, val_targets)
                
                # 计算指标
                train_mae = mean_absolute_error(train_targets.cpu(), train_predictions.cpu())
                val_mae = mean_absolute_error(val_targets.cpu(), val_predictions.cpu())
            
            # 记录历史
            history['train_loss'].append(train_loss.item())
            history['val_loss'].append(val_loss.item())
            history['train_mae'].append(train_mae)
            history['val_mae'].append(val_mae)
            
            # 学习率调度
            scheduler.step(val_loss)
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}: Train Loss={train_loss.item():.4f}, Val Loss={val_loss.item():.4f}')
        
        return history
    
    def evaluate(self, test_data: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """评估模型性能"""
        test_features, test_targets = test_data
        self.spatio_temporal_attention.eval()
        
        with torch.no_grad():
            # 获取预测结果
            predictions = self.spatio_temporal_attention(test_features)
            
            # 处理3维输出 - 如果是3D张量，取最后一个时间步
            if len(predictions.shape) == 3:
                predictions = predictions[:, -1, :]
            
            # 确保维度匹配 - 如果预测是多维的，只取第一列与目标值匹配
            if predictions.shape[1] > 1 and test_targets.shape[1] == 1:
                predictions = predictions[:, 0].unsqueeze(1)
            
            # 确保test_targets是2D的
            if len(test_targets.shape) == 1:
                test_targets = test_targets.unsqueeze(1)
        
        # 转换为NumPy数组进行评估
        y_pred = predictions.cpu().numpy()
        y_true = test_targets.cpu().numpy()
        
        # 使用评估器计算指标
        results = self.evaluator.evaluate_model(y_true, y_pred)
        
        return results
    
    def ablation_study(self, test_data: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, Dict[str, float]]:
        """消融实验"""
        ablation = AblationStudy(self.spatio_temporal_attention)
        return ablation.test_component_importance(test_data)
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.spatio_temporal_attention.state_dict(),
            'config': self.config
        }, path)
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.spatio_temporal_attention.load_state_dict(checkpoint['model_state_dict'])

# ============================================================================
# 使用示例
# ============================================================================

def main():
    """主函数示例"""
    
    # 配置参数
    config = {
        'input_dim': 16,  # 修改为16，能被8整除
        'feature_dim': 512,
        'num_heads': 8,
        'lr': 0.001,
        'epochs': 100,
        'batch_size': 32
    }
    
    # 创建系统
    system = AcademicGlucosePredictionSystem(config)
    
    # 加载数据（这里需要实际数据）
    # train_data, val_data, test_data = load_data()
    
    # 训练模型
    # history = system.train(train_data, val_data)
    
    # 评估模型
    # results = system.evaluate(test_data)
    
    # 消融实验
    # ablation_results = system.ablation_study(test_data)
    
    # 保存模型
    # system.save_model('academic_glucose_model.pth')
    
    print("学术期刊级别血糖预测系统初始化完成！")
    print("请根据实际数据调整参数并运行实验。")

if __name__ == "__main__":
    main() 