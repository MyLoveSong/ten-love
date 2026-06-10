"""
GluFormer血糖预测模型 - 增强版
基于LSTM-GRU融合的个性化血糖趋势预测
集成多模态融合、交叉注意力机制、小波变换和个性化适配
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass
import math

logger = logging.getLogger(__name__)

@dataclass
class GluFormerConfig:
    """GluFormer模型配置"""
    input_dim: int = 10
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    num_heads: int = 8
    prediction_horizon: int = 6
    wavelet_scales: List[int] = None
    use_personalization: bool = True
    personal_feature_dim: int = 5
    use_multimodal: bool = True
    image_feature_dim: int = 512
    text_feature_dim: int = 256

    def __post_init__(self):
        if self.wavelet_scales is None:
            self.wavelet_scales = [1, 2, 4, 8]

class EnhancedWaveletTransform(nn.Module):
    """增强小波变换特征提取模块"""

    def __init__(self, input_dim: int, scales: List[int] = [1, 2, 4, 8]):
        super().__init__()
        self.scales = scales
        self.input_dim = input_dim

        # 多尺度小波滤波器
        self.wavelet_filters = nn.ModuleDict()
        for scale in scales:
            self.wavelet_filters[f'low_{scale}'] = nn.Conv1d(
                input_dim, input_dim, kernel_size=3, padding=1
            )
            self.wavelet_filters[f'high_{scale}'] = nn.Conv1d(
                input_dim, input_dim, kernel_size=3, padding=1
            )

        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(input_dim * len(scales) * 2, input_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        增强小波变换分解
        Args:
            x: 输入血糖序列 [batch_size, seq_len, input_dim]
        Returns:
            多尺度特征字典
        """
        batch_size, seq_len, input_dim = x.shape
        x = x.transpose(1, 2)  # [batch_size, input_dim, seq_len]

        features = {}
        all_features = []

        # 多尺度分解
        for scale in self.scales:
            # 低通滤波 (近似系数)
            low_freq = self.wavelet_filters[f'low_{scale}'](x)
            # 高通滤波 (细节系数)
            high_freq = self.wavelet_filters[f'high_{scale}'](x)

            features[f'low_freq_{scale}'] = low_freq.transpose(1, 2)
            features[f'high_freq_{scale}'] = high_freq.transpose(1, 2)

            # 收集特征用于融合
            all_features.extend([low_freq, high_freq])

            # 下采样
            x = F.avg_pool1d(x, kernel_size=2)

        # 特征融合
        if len(all_features) > 1:
            # 将所有特征调整到相同长度
            min_len = min(f.shape[-1] for f in all_features)
            aligned_features = []
            for f in all_features:
                if f.shape[-1] > min_len:
                    f = F.adaptive_avg_pool1d(f, min_len)
                aligned_features.append(f)

            # 拼接特征
            concatenated = torch.cat(aligned_features, dim=1)  # [batch, input_dim*scales*2, seq_len]
            concatenated = concatenated.transpose(1, 2)  # [batch, seq_len, input_dim*scales*2]

            # 特征融合
            fused_features = self.feature_fusion(concatenated)
            features['fused_wavelet'] = fused_features

        return features

class EnhancedCrossAttention(nn.Module):
    """增强交叉注意力机制"""

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # 投影层
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # 层归一化
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

        # 位置编码
        self.positional_encoding = self._create_positional_encoding(1000, hidden_dim)

    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """创建位置编码"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(
        self,
        lstm_output: torch.Tensor,
        gru_output: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        增强交叉注意力融合
        Args:
            lstm_output: LSTM输出特征 [batch_size, seq_len, hidden_dim]
            gru_output: GRU输出特征 [batch_size, seq_len, hidden_dim]
            mask: 注意力掩码 [batch_size, seq_len, seq_len]
        Returns:
            融合后的特征和注意力权重
        """
        batch_size, seq_len, hidden_dim = lstm_output.shape

        # 添加位置编码
        if seq_len <= self.positional_encoding.size(1):
            pos_encoding = self.positional_encoding[:, :seq_len, :].to(lstm_output.device)
            lstm_output = lstm_output + pos_encoding
            gru_output = gru_output + pos_encoding

        # 残差连接和层归一化
        lstm_output = self.layer_norm1(lstm_output)
        gru_output = self.layer_norm1(gru_output)

        # 投影到查询、键、值
        Q = self.query_proj(lstm_output)  # LSTM作为查询
        K = self.key_proj(gru_output)     # GRU作为键
        V = self.value_proj(gru_output)   # GRU作为值

        # 多头注意力
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 应用掩码
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = F.dropout(attention_weights, p=self.dropout, training=self.training)

        # 加权求和
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, hidden_dim
        )

        # 输出投影和残差连接
        output = self.out_proj(attended)
        output = output + lstm_output  # 残差连接

        # 前馈网络和残差连接
        output = self.layer_norm2(output)
        ff_output = self.feed_forward(output)
        output = output + ff_output

        return output, attention_weights.mean(dim=1)  # 平均多头注意力权重

class MultiModalFusion(nn.Module):
    """多模态融合模块"""

    def __init__(self, glucose_dim: int, image_dim: int, text_dim: int, output_dim: int):
        super().__init__()
        self.glucose_dim = glucose_dim
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.output_dim = output_dim

        # 模态特定的投影层
        self.glucose_proj = nn.Linear(glucose_dim, output_dim)
        self.image_proj = nn.Linear(image_dim, output_dim)
        self.text_proj = nn.Linear(text_dim, output_dim)

        # 跨模态注意力
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(output_dim * 3, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(
        self,
        glucose_features: torch.Tensor,
        image_features: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        多模态特征融合
        Args:
            glucose_features: 血糖特征 [batch_size, seq_len, glucose_dim]
            image_features: 图像特征 [batch_size, image_dim]
            text_features: 文本特征 [batch_size, text_dim]
        Returns:
            融合后的特征 [batch_size, seq_len, output_dim]
        """
        batch_size, seq_len, _ = glucose_features.shape

        # 投影到统一维度
        glucose_proj = self.glucose_proj(glucose_features)

        # 处理图像特征
        if image_features is not None:
            # 将图像特征扩展到序列长度
            image_proj = self.image_proj(image_features).unsqueeze(1).expand(-1, seq_len, -1)
        else:
            image_proj = torch.zeros_like(glucose_proj)

        # 处理文本特征
        if text_features is not None:
            # 将文本特征扩展到序列长度
            text_proj = self.text_proj(text_features).unsqueeze(1).expand(-1, seq_len, -1)
        else:
            text_proj = torch.zeros_like(glucose_proj)

        # 跨模态注意力
        attended_features, _ = self.cross_modal_attention(
            glucose_proj,
            torch.stack([image_proj, text_proj], dim=2).mean(dim=2),
            torch.stack([image_proj, text_proj], dim=2).mean(dim=2)
        )

        # 特征融合
        concatenated = torch.cat([glucose_proj, image_proj, text_proj], dim=-1)
        fused_features = self.fusion_layer(concatenated)

        return fused_features + attended_features  # 残差连接

class PersonalizedAdapter(nn.Module):
    """个性化适配器"""

    def __init__(self, feature_dim: int, personal_dim: int, output_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.personal_dim = personal_dim
        self.output_dim = output_dim

        # 个人特征编码器
        self.personal_encoder = nn.Sequential(
            nn.Linear(personal_dim, personal_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(personal_dim * 2, personal_dim),
            nn.Sigmoid()  # 生成权重
        )

        # 特征适配层
        self.feature_adapter = nn.Sequential(
            nn.Linear(feature_dim + personal_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(
        self,
        features: torch.Tensor,
        personal_features: torch.Tensor
    ) -> torch.Tensor:
        """
        个性化特征适配
        Args:
            features: 输入特征 [batch_size, seq_len, feature_dim]
            personal_features: 个人特征 [batch_size, personal_dim]
        Returns:
            适配后的特征 [batch_size, seq_len, output_dim]
        """
        batch_size, seq_len, _ = features.shape

        # 编码个人特征
        personal_weights = self.personal_encoder(personal_features)  # [batch_size, personal_dim]

        # 扩展个人特征到序列长度
        personal_expanded = personal_weights.unsqueeze(1).expand(-1, seq_len, -1)

        # 特征拼接和适配
        concatenated = torch.cat([features, personal_expanded], dim=-1)
        adapted_features = self.feature_adapter(concatenated)

        return adapted_features

class EnhancedGluFormer(nn.Module):
    """增强版GluFormer血糖预测模型"""

    def __init__(self, config: GluFormerConfig):
        super().__init__()
        self.config = config

        # 增强小波变换特征提取
        self.wavelet_transform = EnhancedWaveletTransform(
            config.input_dim, config.wavelet_scales
        )

        # LSTM网络 (长期记忆)
        self.lstm = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout,
            bidirectional=True
        )

        # GRU网络 (短期动态)
        self.gru = nn.GRU(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout,
            bidirectional=True
        )

        # 增强交叉注意力机制
        self.cross_attention = EnhancedCrossAttention(
            config.hidden_dim * 2, config.num_heads, config.dropout
        )

        # 多模态融合模块
        if config.use_multimodal:
            self.multimodal_fusion = MultiModalFusion(
                glucose_dim=config.hidden_dim * 2,
                image_dim=config.image_feature_dim,
                text_dim=config.text_feature_dim,
                output_dim=config.hidden_dim
            )

        # 个性化适配器
        if config.use_personalization:
            self.personalized_adapter = PersonalizedAdapter(
                feature_dim=config.hidden_dim,
                personal_dim=config.personal_feature_dim,
                output_dim=config.hidden_dim
            )

        # 预测头
        self.prediction_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 4, config.prediction_horizon)
        )

        # 注意力权重存储
        self.attention_weights = None
        self.wavelet_features = None

    def forward(
        self,
        glucose_data: torch.Tensor,
        personal_features: Optional[torch.Tensor] = None,
        image_features: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        增强前向传播
        Args:
            glucose_data: 血糖时序数据 [batch_size, seq_len, input_dim]
            personal_features: 个人特征 [batch_size, personal_feature_dim]
            image_features: 图像特征 [batch_size, image_feature_dim]
            text_features: 文本特征 [batch_size, text_feature_dim]
            mask: 注意力掩码 [batch_size, seq_len, seq_len]
        Returns:
            预测结果和中间特征
        """
        batch_size, seq_len, input_dim = glucose_data.shape

        # 1. 增强小波变换特征提取
        wavelet_features = self.wavelet_transform(glucose_data)
        self.wavelet_features = wavelet_features

        # 2. LSTM处理 (长期依赖)
        lstm_output, _ = self.lstm(glucose_data)

        # 3. GRU处理 (短期动态)
        gru_output, _ = self.gru(glucose_data)

        # 4. 增强交叉注意力融合
        fused_features, attention_weights = self.cross_attention(
            lstm_output, gru_output, mask
        )
        self.attention_weights = attention_weights

        # 5. 多模态融合
        if self.config.use_multimodal and hasattr(self, 'multimodal_fusion'):
            multimodal_features = self.multimodal_fusion(
                fused_features, image_features, text_features
            )
        else:
            multimodal_features = fused_features

        # 6. 个性化适配
        if (self.config.use_personalization and
            personal_features is not None and
            hasattr(self, 'personalized_adapter')):
            personalized_features = self.personalized_adapter(
                multimodal_features, personal_features
            )
        else:
            personalized_features = multimodal_features

        # 7. 血糖预测
        # 取最后一个时间步的输出
        final_features = personalized_features[:, -1, :]
        prediction = self.prediction_head(final_features)

        return prediction, {
            'attention_weights': attention_weights,
            'wavelet_features': wavelet_features,
            'lstm_output': lstm_output,
            'gru_output': gru_output,
            'multimodal_features': multimodal_features,
            'personalized_features': personalized_features,
            'final_features': final_features
        }

class EnhancedPersonalizedGlucosePredictor:
    """增强版个性化血糖预测器"""

    def __init__(self, config: GluFormerConfig):
        self.config = config
        self.model = EnhancedGluFormer(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # 优化器和损失函数
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=0.001,
            weight_decay=0.01
        )
        self.criterion = nn.MSELoss()

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )

        # 训练历史
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'attention_weights': [],
            'learning_rates': []
        }

        # 模型状态
        self.is_trained = False
        self.best_val_loss = float('inf')

    def predict(
        self,
        glucose_data: torch.Tensor,
        personal_features: Optional[torch.Tensor] = None,
        image_features: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        血糖预测
        Args:
            glucose_data: 血糖时序数据
            personal_features: 个人特征
            image_features: 图像特征
            text_features: 文本特征
        Returns:
            预测结果和中间特征
        """
        self.model.eval()
        with torch.no_grad():
            if isinstance(glucose_data, np.ndarray):
                glucose_data = torch.FloatTensor(glucose_data)
            if personal_features is not None and isinstance(personal_features, np.ndarray):
                personal_features = torch.FloatTensor(personal_features)
            if image_features is not None and isinstance(image_features, np.ndarray):
                image_features = torch.FloatTensor(image_features)
            if text_features is not None and isinstance(text_features, np.ndarray):
                text_features = torch.FloatTensor(text_features)

            glucose_data = glucose_data.to(self.device)
            if personal_features is not None:
                personal_features = personal_features.to(self.device)
            if image_features is not None:
                image_features = image_features.to(self.device)
            if text_features is not None:
                text_features = text_features.to(self.device)

            prediction, features = self.model(
                glucose_data, personal_features, image_features, text_features
            )

            return prediction.cpu().detach().numpy(), {
                k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                for k, v in features.items()
            }

    def train(
        self,
        train_data: torch.Tensor,
        train_labels: torch.Tensor,
        personal_features: Optional[torch.Tensor] = None,
        val_data: Optional[torch.Tensor] = None,
        val_labels: Optional[torch.Tensor] = None,
        epochs: int = 100
    ) -> Dict[str, List[float]]:
        """
        训练模型
        Args:
            train_data: 训练数据 [batch_size, seq_len, input_dim]
            train_labels: 训练标签 [batch_size, prediction_horizon]
            personal_features: 个人特征 [batch_size, 5]
            val_data: 验证数据
            val_labels: 验证标签
            epochs: 训练轮数
        """
        self.model.train()

        for epoch in range(epochs):
            # 前向传播
            predictions, outputs = self.model(train_data, personal_features)

            # 计算损失
            loss = self.criterion(predictions, train_labels)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 记录训练历史
            self.training_history['train_loss'].append(loss.item())
            self.training_history['attention_weights'].append(
                outputs['attention_weights'].detach().cpu().numpy()
            )

            # 验证
            if val_data is not None and val_labels is not None:
                val_loss = self.validate(val_data, val_labels, personal_features)
                self.training_history['val_loss'].append(val_loss)

            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        self.is_trained = True
        return self.training_history

    def validate(
        self,
        val_data: torch.Tensor,
        val_labels: torch.Tensor,
        personal_features: Optional[torch.Tensor] = None,
        image_features: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None
    ) -> float:
        """增强验证模型"""
        self.model.eval()
        with torch.no_grad():
            val_data = val_data.to(self.device)
            val_labels = val_labels.to(self.device)
            if personal_features is not None:
                personal_features = personal_features.to(self.device)
            if image_features is not None:
                image_features = image_features.to(self.device)
            if text_features is not None:
                text_features = text_features.to(self.device)

            predictions, _ = self.model(
                val_data, personal_features, image_features, text_features
            )
            val_loss = self.criterion(predictions, val_labels)
        self.model.train()
        return val_loss.item()

    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'training_history': self.training_history,
            'best_val_loss': self.best_val_loss
        }, filepath)
        logger.info(f"模型已保存到: {filepath}")

    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
        self.best_val_loss = checkpoint['best_val_loss']
        logger.info(f"模型已从 {filepath} 加载")

    def get_attention_visualization(self) -> Dict[str, np.ndarray]:
        """获取注意力权重可视化数据"""
        if self.training_history['attention_weights']:
            # 取最后一个epoch的注意力权重
            latest_attention = self.training_history['attention_weights'][-1]
            return {
                'attention_weights': latest_attention,
                'attention_mean': np.mean(latest_attention, axis=0),
                'attention_std': np.std(latest_attention, axis=0)
            }
        return {}

    def get_explainability_analysis(self) -> Dict[str, Any]:
        """获取可解释性分析"""
        if hasattr(self.model, 'attention_weights') and self.model.attention_weights is not None:
            attention_weights = self.model.attention_weights.cpu().numpy()
            return {
                'attention_weights': attention_weights,
                'feature_importance': np.mean(attention_weights, axis=(0, 1)),
                'temporal_importance': np.mean(attention_weights, axis=(0, 2)),
                'explanation': "注意力权重显示了模型对不同时间步和特征的关注程度"
            }
        return {'explanation': '模型尚未训练或注意力权重不可用'}

# 工厂函数
def create_enhanced_gluformer_model(config: Optional[GluFormerConfig] = None) -> EnhancedPersonalizedGlucosePredictor:
    """创建增强版GluFormer模型实例"""
    if config is None:
        config = GluFormerConfig()

    predictor = EnhancedPersonalizedGlucosePredictor(config)
    return predictor

def create_gluformer_model() -> EnhancedPersonalizedGlucosePredictor:
    """创建传统GluFormer模型实例（向后兼容）"""
    config = GluFormerConfig(
        input_dim=10,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1,
        num_heads=8,
        prediction_horizon=6,
        use_multimodal=False,
        use_personalization=True
    )

    predictor = EnhancedPersonalizedGlucosePredictor(config)
    return predictor

if __name__ == "__main__":
    # 创建增强版模型
    config = GluFormerConfig(
        input_dim=10,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1,
        num_heads=8,
        prediction_horizon=6,
        use_multimodal=True,
        use_personalization=True
    )

    predictor = create_enhanced_gluformer_model(config)

    # 模拟数据
    batch_size, seq_len, input_dim = 32, 24, 10
    glucose_data = torch.randn(batch_size, seq_len, input_dim)
    personal_features = torch.randn(batch_size, 5)
    image_features = torch.randn(batch_size, 512)
    text_features = torch.randn(batch_size, 256)
    labels = torch.randn(batch_size, 6)

    # 预测测试
    predictions, outputs = predictor.predict(
        glucose_data, personal_features, image_features, text_features
    )

    print(f"增强版GluFormer模型创建成功！")
    print(f"预测结果形状: {predictions.shape}")
    print(f"可用特征: {list(outputs.keys())}")

    # 可解释性分析
    explainability = predictor.get_explainability_analysis()
    print(f"可解释性分析: {explainability['explanation']}")
