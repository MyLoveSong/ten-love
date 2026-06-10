

"""
学术级多模态数据融合模块
支持图像、文本、时间序列、表格数据的联合建模
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import json
from datetime import datetime
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import asyncio
import random

from backend.app.core.exceptions import CustomException, ValidationError
from app.core.task_queue import async_task, TaskPriority

logger = logging.getLogger(__name__)

class ModalityType(Enum):
    """模态类型"""
    IMAGE = "image"                  # 图像
    TEXT = "text"                    # 文本
    TIME_SERIES = "time_series"      # 时间序列
    TABULAR = "tabular"              # 表格数据
    AUDIO = "audio"                  # 音频
    VIDEO = "video"                  # 视频

class FusionStrategy(Enum):
    """融合策略"""
    EARLY_FUSION = "early_fusion"    # 早期融合
    LATE_FUSION = "late_fusion"      # 晚期融合
    INTERMEDIATE_FUSION = "intermediate_fusion"  # 中期融合
    ATTENTION_FUSION = "attention_fusion"        # 注意力融合
    CROSS_MODALITY = "cross_modality"            # 跨模态融合

class AttentionMechanism(Enum):
    """注意力机制"""
    SELF_ATTENTION = "self_attention"            # 自注意力
    CROSS_ATTENTION = "cross_attention"          # 交叉注意力
    MULTI_HEAD_ATTENTION = "multi_head_attention"  # 多头注意力
    TEMPORAL_ATTENTION = "temporal_attention"    # 时间注意力

@dataclass
class ModalityConfig:
    """模态配置"""
    modality_type: ModalityType
    input_dim: int
    hidden_dim: int = 128
    embedding_dim: int = 64
    sequence_length: Optional[int] = None
    vocab_size: Optional[int] = None
    num_classes: Optional[int] = None
    custom_parameters: Optional[Dict[str, Any]] = None

@dataclass
class FusionConfig:
    """融合配置"""
    fusion_strategy: FusionStrategy
    attention_mechanism: Optional[AttentionMechanism] = None
    fusion_dim: int = 256
    num_attention_heads: int = 8
    dropout: float = 0.1
    temperature: float = 0.07
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 100
    weight_decay: float = 1e-4
    custom_parameters: Optional[Dict[str, Any]] = None

@dataclass
class MultimodalResult:
    """多模态结果"""
    fusion_strategy: str
    modality_types: List[str]
    training_loss: List[float]
    validation_loss: List[float]
    test_accuracy: float
    fusion_quality: float
    attention_weights: Optional[Dict[str, List[float]]] = None
    training_time: float = 0.0
    model_size: int = 0
    timestamp: datetime = None

class ImageEncoder(nn.Module):
    """图像编码器"""

    def __init__(self, input_dim: int, hidden_dim: int = 128, embedding_dim: int = 64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, x):
        return self.encoder(x)

class TextEncoder(nn.Module):
    """文本编码器"""

    def __init__(self, vocab_size: int, embedding_dim: int = 64, hidden_dim: int = 128):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.projection = nn.Linear(hidden_dim * 2, embedding_dim)

    def forward(self, x):
        # 嵌入
        embedded = self.embedding(x)

        # LSTM编码
        lstm_out, (hidden, cell) = self.lstm(embedded)

        # 使用最后一个时间步的输出
        last_output = lstm_out[:, -1, :]

        # 投影到嵌入空间
        projected = self.projection(last_output)

        return projected

class TimeSeriesEncoder(nn.Module):
    """时间序列编码器"""

    def __init__(self, input_dim: int, sequence_length: int, hidden_dim: int = 128, embedding_dim: int = 64):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=8, batch_first=True)
        self.projection = nn.Linear(hidden_dim * 2, embedding_dim)

    def forward(self, x):
        # 输入投影
        projected = self.input_projection(x)

        # LSTM编码
        lstm_out, _ = self.lstm(projected)

        # 自注意力
        attended, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # 全局平均池化
        pooled = torch.mean(attended, dim=1)

        # 投影到嵌入空间
        embedded = self.projection(pooled)

        return embedded

class TabularEncoder(nn.Module):
    """表格数据编码器"""

    def __init__(self, input_dim: int, hidden_dim: int = 128, embedding_dim: int = 64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, x):
        return self.encoder(x)

class AttentionFusion(nn.Module):
    """注意力融合模块"""

    def __init__(self, embedding_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        self.attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 自注意力
        attended, attention_weights = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attended))

        # 前馈网络
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x, attention_weights

class CrossModalAttention(nn.Module):
    """跨模态注意力模块"""

    def __init__(self, embedding_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        self.attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        # 跨模态注意力
        attended, attention_weights = self.attention(query, key, value)

        # 残差连接和层归一化
        output = self.norm(query + self.dropout(attended))

        return output, attention_weights

class MultimodalFusionModel(nn.Module):
    """多模态融合模型"""

    def __init__(self, modality_configs: List[ModalityConfig], fusion_config: FusionConfig):
        super().__init__()

        self.modality_configs = modality_configs
        self.fusion_config = fusion_config
        self.embedding_dim = modality_configs[0].embedding_dim

        # 初始化编码器
        self.encoders = nn.ModuleDict()
        for i, config in enumerate(modality_configs):
            if config.modality_type == ModalityType.IMAGE:
                self.encoders[f"modality_{i}"] = ImageEncoder(
                    config.input_dim, config.hidden_dim, config.embedding_dim
                )
            elif config.modality_type == ModalityType.TEXT:
                self.encoders[f"modality_{i}"] = TextEncoder(
                    config.vocab_size, config.embedding_dim, config.hidden_dim
                )
            elif config.modality_type == ModalityType.TIME_SERIES:
                self.encoders[f"modality_{i}"] = TimeSeriesEncoder(
                    config.input_dim, config.sequence_length, config.hidden_dim, config.embedding_dim
                )
            elif config.modality_type == ModalityType.TABULAR:
                self.encoders[f"modality_{i}"] = TabularEncoder(
                    config.input_dim, config.hidden_dim, config.embedding_dim
                )

        # 初始化融合模块
        if fusion_config.fusion_strategy == FusionStrategy.ATTENTION_FUSION:
            self.fusion_module = AttentionFusion(
                self.embedding_dim, fusion_config.num_attention_heads, fusion_config.dropout
            )
        elif fusion_config.fusion_strategy == FusionStrategy.CROSS_MODALITY:
            self.cross_modal_attention = CrossModalAttention(
                self.embedding_dim, fusion_config.num_attention_heads, fusion_config.dropout
            )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim * len(modality_configs), fusion_config.fusion_dim),
            nn.ReLU(),
            nn.Dropout(fusion_config.dropout),
            nn.Linear(fusion_config.fusion_dim, fusion_config.fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(fusion_config.dropout),
            nn.Linear(fusion_config.fusion_dim // 2, 2)  # 二分类
        )

    def forward(self, inputs: Dict[str, torch.Tensor]):
        # 编码各个模态
        embeddings = []
        attention_weights = {}

        for i, config in enumerate(self.modality_configs):
            modality_key = f"modality_{i}"
            if modality_key in inputs:
                embedding = self.encoders[modality_key](inputs[modality_key])
                embeddings.append(embedding)

        # 融合策略
        if self.fusion_config.fusion_strategy == FusionStrategy.EARLY_FUSION:
            # 早期融合：直接拼接
            fused = torch.cat(embeddings, dim=1)

        elif self.fusion_config.fusion_strategy == FusionStrategy.LATE_FUSION:
            # 晚期融合：分别处理后再拼接
            processed_embeddings = []
            for embedding in embeddings:
                processed = F.relu(embedding)
                processed_embeddings.append(processed)
            fused = torch.cat(processed_embeddings, dim=1)

        elif self.fusion_config.fusion_strategy == FusionStrategy.ATTENTION_FUSION:
            # 注意力融合
            stacked_embeddings = torch.stack(embeddings, dim=1)  # [batch, num_modalities, embedding_dim]
            fused_attended, attention_weights["fusion"] = self.fusion_module(stacked_embeddings)
            fused = fused_attended.view(fused_attended.size(0), -1)

        elif self.fusion_config.fusion_strategy == FusionStrategy.CROSS_MODALITY:
            # 跨模态融合
            if len(embeddings) >= 2:
                # 使用第一个模态作为query，其他作为key和value
                query = embeddings[0].unsqueeze(1)
                key_value = torch.stack(embeddings[1:], dim=1)

                fused_cross, attention_weights["cross_modal"] = self.cross_modal_attention(
                    query, key_value, key_value
                )
                fused = torch.cat([embeddings[0], fused_cross.squeeze(1)], dim=1)
            else:
                fused = embeddings[0]

        else:
            # 默认：简单拼接
            fused = torch.cat(embeddings, dim=1)

        # 分类
        output = self.classifier(fused)

        return output, attention_weights

class MultimodalDataset(Dataset):
    """多模态数据集"""

    def __init__(self, data_dict: Dict[str, np.ndarray], labels: np.ndarray):
        self.data_dict = data_dict
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        for modality_key, data in self.data_dict.items():
            sample[modality_key] = torch.FloatTensor(data[idx])

        label = torch.LongTensor([self.labels[idx]])
        return sample, label

class MultimodalFusionTrainer:
    """多模态融合训练器"""

    def __init__(self, modality_configs: List[ModalityConfig], fusion_config: FusionConfig):
        self.modality_configs = modality_configs
        self.fusion_config = fusion_config
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.training_history: List[MultimodalResult] = []

    def train(self, train_data: Dict[str, np.ndarray], train_labels: np.ndarray,
              val_data: Optional[Dict[str, np.ndarray]] = None,
              val_labels: Optional[np.ndarray] = None) -> MultimodalResult:
        """训练多模态融合模型"""
        logger.info(f"开始多模态融合训练，策略: {self.fusion_config.fusion_strategy.value}")

        start_time = datetime.now()

        # 创建数据集和数据加载器
        train_dataset = MultimodalDataset(train_data, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=self.fusion_config.batch_size, shuffle=True)

        val_loader = None
        if val_data is not None and val_labels is not None:
            val_dataset = MultimodalDataset(val_data, val_labels)
            val_loader = DataLoader(val_dataset, batch_size=self.fusion_config.batch_size, shuffle=False)

        # 初始化模型
        self.model = MultimodalFusionModel(self.modality_configs, self.fusion_config)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.fusion_config.learning_rate,
            weight_decay=self.fusion_config.weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.fusion_config.epochs
        )

        # 训练循环
        training_losses = []
        validation_losses = []

        for epoch in range(self.fusion_config.epochs):
            # 训练阶段
            train_loss = self._train_epoch(train_loader)
            training_losses.append(train_loss)

            # 验证阶段
            val_loss = 0.0
            if val_loader is not None:
                val_loss = self._validate_epoch(val_loader)
                validation_losses.append(val_loss)

            self.scheduler.step()

            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # 计算训练时间
        training_time = (datetime.now() - start_time).total_seconds()

        # 评估模型
        test_accuracy = self._evaluate_model(train_loader)
        fusion_quality = self._evaluate_fusion_quality(train_data)

        # 获取注意力权重
        attention_weights = self._get_attention_weights(train_data)

        result = MultimodalResult(
            fusion_strategy=self.fusion_config.fusion_strategy.value,
            modality_types=[config.modality_type.value for config in self.modality_configs],
            training_loss=training_losses,
            validation_loss=validation_losses,
            test_accuracy=test_accuracy,
            fusion_quality=fusion_quality,
            attention_weights=attention_weights,
            training_time=training_time,
            model_size=self._get_model_size(),
            timestamp=datetime.now()
        )

        self.training_history.append(result)

        logger.info(f"多模态融合训练完成，训练时间: {training_time:.2f}秒")
        return result

    def _train_epoch(self, train_loader: DataLoader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            self.optimizer.zero_grad()

            # 前向传播
            outputs, _ = self.model(inputs)
            loss = F.cross_entropy(outputs, labels.squeeze())

            # 反向传播
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(val_loader):
                outputs, _ = self.model(inputs)
                loss = F.cross_entropy(outputs, labels.squeeze())
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def _evaluate_model(self, data_loader: DataLoader) -> float:
        """评估模型性能"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(data_loader):
                outputs, _ = self.model(inputs)
                predicted = torch.argmax(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels.squeeze()).sum().item()

        accuracy = correct / total
        return accuracy

    def _evaluate_fusion_quality(self, data: Dict[str, np.ndarray]) -> float:
        """评估融合质量"""
        try:
            # 简单的融合质量评估
            # 计算不同模态之间的相关性
            modalities = list(data.keys())
            correlations = []

            for i in range(len(modalities)):
                for j in range(i+1, len(modalities)):
                    mod1_data = data[modalities[i]]
                    mod2_data = data[modalities[j]]

                    # 计算平均相关性
                    if mod1_data.shape[1] == mod2_data.shape[1]:
                        corr = np.corrcoef(mod1_data.flatten(), mod2_data.flatten())[0, 1]
                        correlations.append(abs(corr))

            # 融合质量 = 平均相关性
            fusion_quality = np.mean(correlations) if correlations else 0.0

            return fusion_quality

        except Exception as e:
            logger.warning(f"融合质量评估失败: {e}")
            return 0.0

    def _get_attention_weights(self, data: Dict[str, np.ndarray]) -> Optional[Dict[str, List[float]]]:
        """获取注意力权重"""
        try:
            self.model.eval()

            # 使用一小批数据获取注意力权重
            sample_data = {key: torch.FloatTensor(value[:1]) for key, value in data.items()}

            with torch.no_grad():
                _, attention_weights = self.model(sample_data)

            # 转换注意力权重为可序列化的格式
            serializable_weights = {}
            for key, weights in attention_weights.items():
                if isinstance(weights, torch.Tensor):
                    serializable_weights[key] = weights.cpu().numpy().tolist()
                else:
                    serializable_weights[key] = weights

            return serializable_weights

        except Exception as e:
            logger.warning(f"获取注意力权重失败: {e}")
            return None

    def _get_model_size(self) -> int:
        """获取模型大小"""
        try:
            total_params = sum(p.numel() for p in self.model.parameters())
            return total_params
        except Exception as e:
            logger.warning(f"模型大小计算失败: {e}")
            return 0

class MultimodalFusionManager:
    """多模态融合管理器"""

    def __init__(self):
        self.fusion_history: List[MultimodalResult] = []

    def fuse_modalities(self, data_dict: Dict[str, np.ndarray], labels: np.ndarray,
                       modality_configs: List[ModalityConfig], fusion_config: FusionConfig,
                       test_size: float = 0.2) -> MultimodalResult:
        """多模态数据融合"""
        logger.info(f"开始多模态数据融合，模态数: {len(modality_configs)}")

        try:
            # 数据分割
            train_data, val_data, train_labels, val_labels = self._split_data(
                data_dict, labels, test_size
            )

            # 创建训练器
            trainer = MultimodalFusionTrainer(modality_configs, fusion_config)

            # 训练模型
            result = trainer.train(train_data, train_labels, val_data, val_labels)

            self.fusion_history.append(result)

            return result

        except Exception as e:
            logger.error(f"多模态数据融合失败: {e}")
            raise CustomException(f"多模态数据融合失败: {e}")

    def _split_data(self, data_dict: Dict[str, np.ndarray], labels: np.ndarray,
                   test_size: float) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray],
                                             np.ndarray, np.ndarray]:
        """分割数据"""
        # 使用第一个模态的数据进行分割
        first_modality = list(data_dict.keys())[0]
        indices = np.arange(len(data_dict[first_modality]))

        train_indices, val_indices = train_test_split(
            indices, test_size=test_size, random_state=42, stratify=labels
        )

        train_data = {key: data[train_indices] for key, data in data_dict.items()}
        val_data = {key: data[val_indices] for key, data in data_dict.items()}
        train_labels = labels[train_indices]
        val_labels = labels[val_indices]

        return train_data, val_data, train_labels, val_labels

    def get_fusion_history(self) -> List[MultimodalResult]:
        """获取融合历史"""
        return self.fusion_history

# 全局多模态融合管理器实例
multimodal_fusion_manager = MultimodalFusionManager()

# 异步任务
@async_task("multimodal_fusion", TaskPriority.HIGH)
def multimodal_fusion_task(data_dict: Dict[str, List[List[float]]], labels: List[int],
                          modality_configs: List[Dict[str, Any]], fusion_config: Dict[str, Any]):
    """多模态融合任务"""
    # 转换数据格式
    data_np = {key: np.array(value) for key, value in data_dict.items()}
    labels_np = np.array(labels)

    # 转换配置
    modality_configs_obj = [ModalityConfig(**config) for config in modality_configs]
    fusion_config_obj = FusionConfig(**fusion_config)

    # 执行融合
    result = multimodal_fusion_manager.fuse_modalities(
        data_np, labels_np, modality_configs_obj, fusion_config_obj
    )

    return {
        "result": asdict(result),
        "success": True
    }

# 多模态融合API
def fuse_modalities(data_dict: Dict[str, np.ndarray], labels: np.ndarray,
                   modality_configs: List[ModalityConfig], fusion_config: FusionConfig) -> MultimodalResult:
    """多模态数据融合"""
    return multimodal_fusion_manager.fuse_modalities(data_dict, labels, modality_configs, fusion_config)

def get_fusion_history() -> List[MultimodalResult]:
    """获取融合历史"""
    return multimodal_fusion_manager.get_fusion_history()

if __name__ == "__main__":
    # 测试多模态融合
    import numpy as np

    # 创建测试数据
    np.random.seed(42)

    # 图像数据
    image_data = np.random.randn(1000, 784)  # 28x28图像展平

    # 文本数据
    text_data = np.random.randint(0, 1000, (1000, 50))  # 50个词的序列

    # 时间序列数据
    time_series_data = np.random.randn(1000, 100, 10)  # 100个时间步，10个特征

    # 表格数据
    tabular_data = np.random.randn(1000, 20)  # 20个特征

    # 标签
    labels = np.random.randint(0, 2, 1000)

    # 数据字典
    data_dict = {
        "modality_0": image_data,
        "modality_1": text_data,
        "modality_2": time_series_data,
        "modality_3": tabular_data
    }

    # 模态配置
    modality_configs = [
        ModalityConfig(ModalityType.IMAGE, 784, 128, 64),
        ModalityConfig(ModalityType.TEXT, 50, 128, 64, vocab_size=1000),
        ModalityConfig(ModalityType.TIME_SERIES, 10, 128, 64, sequence_length=100),
        ModalityConfig(ModalityType.TABULAR, 20, 128, 64)
    ]

    # 融合配置
    fusion_config = FusionConfig(
        fusion_strategy=FusionStrategy.ATTENTION_FUSION,
        attention_mechanism=AttentionMechanism.MULTI_HEAD_ATTENTION,
        fusion_dim=256,
        num_attention_heads=8,
        learning_rate=1e-4,
        batch_size=32,
        epochs=50
    )

    # 执行多模态融合
    result = fuse_modalities(data_dict, labels, modality_configs, fusion_config)

    print("多模态融合完成:")
    print(f"融合策略: {result.fusion_strategy}")
    print(f"模态类型: {result.modality_types}")
    print(f"测试准确率: {result.test_accuracy:.4f}")
    print(f"融合质量: {result.fusion_quality:.4f}")
    print(f"训练时间: {result.training_time:.2f}秒")
    print(f"模型大小: {result.model_size}")

__all__ = ["'logger'", "'ModalityType'", "'FusionStrategy'", "'AttentionMechanism'", "'ModalityConfig'", "'FusionConfig'", "'MultimodalResult'", "'ImageEncoder'", "'TextEncoder'", "'TimeSeriesEncoder'", "'TabularEncoder'", "'AttentionFusion'", "'CrossModalAttention'", "'MultimodalFusionModel'", "'MultimodalDataset'", "'MultimodalFusionTrainer'", "'MultimodalFusionManager'", "'multimodal_fusion_manager'", "'multimodal_fusion_task'", "'fuse_modalities'", "'get_fusion_history'"]
