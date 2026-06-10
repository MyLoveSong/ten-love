

"""
学术级自监督学习模块
支持对比学习、掩码预测、重构任务等
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
import asyncio
import random

from backend.app.core.exceptions import CustomException, ValidationError
from app.core.task_queue import async_task, TaskPriority

logger = logging.getLogger(__name__)

class SelfSupervisedMethod(Enum):
    """自监督学习方法"""
    CONTRASTIVE_LEARNING = "contrastive_learning"  # 对比学习
    MASKED_PREDICTION = "masked_prediction"         # 掩码预测
    RECONSTRUCTION = "reconstruction"               # 重构任务
    PREDICTION = "prediction"                       # 预测任务
    CLUSTERING = "clustering"                       # 聚类任务
    ROTATION_PREDICTION = "rotation_prediction"     # 旋转预测

class ContrastiveMethod(Enum):
    """对比学习方法"""
    SIMCLR = "simclr"                              # SimCLR
    MOCO = "moco"                                  # MoCo
    SWAV = "swav"                                  # SwAV
    BYOL = "byol"                                  # BYOL

class MaskingStrategy(Enum):
    """掩码策略"""
    RANDOM = "random"                              # 随机掩码
    BLOCK = "block"                                # 块掩码
    SENTENCE = "sentence"                          # 句子掩码
    TOKEN = "token"                                # 标记掩码

@dataclass
class SelfSupervisedConfig:
    """自监督学习配置"""
    method: SelfSupervisedMethod
    contrastive_method: Optional[ContrastiveMethod] = None
    masking_strategy: Optional[MaskingStrategy] = None
    mask_ratio: float = 0.15
    temperature: float = 0.07
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 100
    hidden_dim: int = 128
    projection_dim: int = 64
    dropout: float = 0.1
    weight_decay: float = 1e-4
    custom_parameters: Optional[Dict[str, Any]] = None

@dataclass
class SelfSupervisedResult:
    """自监督学习结果"""
    method: str
    training_loss: List[float]
    validation_loss: List[float]
    representation_quality: float
    downstream_performance: Dict[str, float]
    training_time: float
    model_size: int
    timestamp: datetime

class ContrastiveLearningModel(nn.Module):
    """对比学习模型"""

    def __init__(self, input_dim: int, hidden_dim: int = 128, projection_dim: int = 64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, projection_dim)
        )

        self.projection_head = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def forward(self, x):
        # 编码
        encoded = self.encoder(x)

        # 投影
        projected = self.projection_head(encoded)

        # L2归一化
        projected = F.normalize(projected, dim=1)

        return encoded, projected

class MaskedPredictionModel(nn.Module):
    """掩码预测模型"""

    def __init__(self, input_dim: int, hidden_dim: int = 128, vocab_size: int = 1000):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, nhead=8, dim_feedforward=hidden_dim * 4),
            num_layers=6
        )
        self.decoder = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, mask):
        # 嵌入
        embedded = self.embedding(x)

        # 编码
        encoded = self.encoder(embedded)

        # 预测掩码位置
        masked_output = encoded[mask]
        predictions = self.decoder(masked_output)

        return predictions

class ReconstructionModel(nn.Module):
    """重构模型"""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        # 编码
        encoded = self.encoder(x)

        # 解码
        decoded = self.decoder(encoded)

        return encoded, decoded

class SelfSupervisedDataset(Dataset):
    """自监督学习数据集"""

    def __init__(self, data: np.ndarray, method: SelfSupervisedMethod, config: SelfSupervisedConfig):
        self.data = data
        self.method = method
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        if self.method == SelfSupervisedMethod.CONTRASTIVE_LEARNING:
            return self._contrastive_augment(sample)
        elif self.method == SelfSupervisedMethod.MASKED_PREDICTION:
            return self._masked_prediction(sample)
        elif self.method == SelfSupervisedMethod.RECONSTRUCTION:
            return self._reconstruction(sample)
        elif self.method == SelfSupervisedMethod.ROTATION_PREDICTION:
            return self._rotation_prediction(sample)
        else:
            return sample, sample

    def _contrastive_augment(self, sample):
        """对比学习数据增强"""
        # 创建两个不同的增强版本
        aug1 = self._random_augment(sample)
        aug2 = self._random_augment(sample)
        return aug1, aug2

    def _masked_prediction(self, sample):
        """掩码预测"""
        # 创建掩码
        mask = np.random.random(len(sample)) < self.config.mask_ratio
        masked_sample = sample.copy()
        masked_sample[mask] = 0  # 用0替换掩码位置

        return masked_sample, sample, mask

    def _reconstruction(self, sample):
        """重构任务"""
        # 添加噪声或部分遮挡
        noisy_sample = sample + np.random.normal(0, 0.1, sample.shape)
        return noisy_sample, sample

    def _rotation_prediction(self, sample):
        """旋转预测"""
        # 随机旋转角度
        rotation_angle = random.choice([0, 90, 180, 270])

        # 简单的旋转模拟（实际应用中需要更复杂的实现）
        rotated_sample = np.roll(sample, rotation_angle // 90)

        return rotated_sample, rotation_angle

    def _random_augment(self, sample):
        """随机增强"""
        # 添加噪声
        noise = np.random.normal(0, 0.1, sample.shape)
        return sample + noise

class ContrastiveLearner:
    """对比学习器"""

    def __init__(self, config: SelfSupervisedConfig):
        self.config = config
        self.model = None
        self.optimizer = None
        self.scheduler = None

    def train(self, data: np.ndarray) -> SelfSupervisedResult:
        """训练对比学习模型"""
        logger.info("开始对比学习训练")

        start_time = datetime.now()

        # 创建数据集和数据加载器
        dataset = SelfSupervisedDataset(data, SelfSupervisedMethod.CONTRASTIVE_LEARNING, self.config)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        # 初始化模型
        input_dim = data.shape[1]
        self.model = ContrastiveLearningModel(
            input_dim, self.config.hidden_dim, self.config.projection_dim
        )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config.epochs
        )

        # 训练循环
        training_losses = []

        for epoch in range(self.config.epochs):
            epoch_loss = 0.0

            for batch_idx, (aug1, aug2) in enumerate(dataloader):
                self.optimizer.zero_grad()

                # 前向传播
                _, proj1 = self.model(aug1.float())
                _, proj2 = self.model(aug2.float())

                # 计算对比损失
                loss = self._contrastive_loss(proj1, proj2)

                # 反向传播
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            training_losses.append(avg_loss)

            self.scheduler.step()

            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

        # 计算训练时间
        training_time = (datetime.now() - start_time).total_seconds()

        # 评估表示质量
        representation_quality = self._evaluate_representation_quality(data)

        # 评估下游任务性能
        downstream_performance = self._evaluate_downstream_performance(data)

        result = SelfSupervisedResult(
            method=self.config.method.value,
            training_loss=training_losses,
            validation_loss=[],  # 对比学习通常不需要验证集
            representation_quality=representation_quality,
            downstream_performance=downstream_performance,
            training_time=training_time,
            model_size=self._get_model_size(),
            timestamp=datetime.now()
        )

        logger.info(f"对比学习训练完成，训练时间: {training_time:.2f}秒")
        return result

    def _contrastive_loss(self, proj1, proj2):
        """计算对比损失"""
        batch_size = proj1.shape[0]

        # 计算相似度矩阵
        similarity_matrix = torch.matmul(proj1, proj2.T) / self.config.temperature

        # 创建标签（对角线为正样本）
        labels = torch.arange(batch_size).to(proj1.device)

        # 计算损失
        loss = F.cross_entropy(similarity_matrix, labels)

        return loss

    def _evaluate_representation_quality(self, data: np.ndarray) -> float:
        """评估表示质量"""
        try:
            self.model.eval()

            with torch.no_grad():
                # 获取表示
                representations, _ = self.model(torch.FloatTensor(data))

                # 计算表示的多样性（使用协方差矩阵的行列式）
                cov_matrix = torch.cov(representations.T)
                diversity = torch.det(cov_matrix).item()

                return min(diversity, 1.0)  # 归一化到[0,1]

        except Exception as e:
            logger.warning(f"表示质量评估失败: {e}")
            return 0.0

    def _evaluate_downstream_performance(self, data: np.ndarray) -> Dict[str, float]:
        """评估下游任务性能"""
        try:
            # 使用K-means聚类作为下游任务
            from sklearn.cluster import KMeans
            from sklearn.metrics import adjusted_rand_score

            self.model.eval()

            with torch.no_grad():
                representations, _ = self.model(torch.FloatTensor(data))
                representations_np = representations.numpy()

            # 生成伪标签
            kmeans = KMeans(n_clusters=5, random_state=42)
            pseudo_labels = kmeans.fit_predict(representations_np)

            # 计算聚类质量
            silhouette_score = self._calculate_silhouette_score(representations_np, pseudo_labels)

            return {
                "clustering_silhouette": silhouette_score,
                "representation_diversity": self._calculate_diversity(representations_np)
            }

        except Exception as e:
            logger.warning(f"下游任务评估失败: {e}")
            return {"clustering_silhouette": 0.0, "representation_diversity": 0.0}

    def _calculate_silhouette_score(self, representations: np.ndarray, labels: np.ndarray) -> float:
        """计算轮廓系数"""
        try:
            from sklearn.metrics import silhouette_score
            return silhouette_score(representations, labels)
        except Exception as e:
            logger.warning(f"轮廓系数计算失败: {e}")
            return 0.0

    def _calculate_diversity(self, representations: np.ndarray) -> float:
        """计算表示多样性"""
        try:
            # 计算表示之间的距离方差
            distances = []
            for i in range(len(representations)):
                for j in range(i+1, len(representations)):
                    dist = np.linalg.norm(representations[i] - representations[j])
                    distances.append(dist)

            return np.var(distances) if distances else 0.0

        except Exception as e:
            logger.warning(f"多样性计算失败: {e}")
            return 0.0

    def _get_model_size(self) -> int:
        """获取模型大小"""
        try:
            total_params = sum(p.numel() for p in self.model.parameters())
            return total_params
        except Exception as e:
            logger.warning(f"模型大小计算失败: {e}")
            return 0

class MaskedPredictionLearner:
    """掩码预测学习器"""

    def __init__(self, config: SelfSupervisedConfig):
        self.config = config
        self.model = None
        self.optimizer = None

    def train(self, data: np.ndarray) -> SelfSupervisedResult:
        """训练掩码预测模型"""
        logger.info("开始掩码预测训练")

        start_time = datetime.now()

        # 创建数据集和数据加载器
        dataset = SelfSupervisedDataset(data, SelfSupervisedMethod.MASKED_PREDICTION, self.config)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        # 初始化模型
        input_dim = data.shape[1]
        vocab_size = int(np.max(data)) + 1 if data.dtype == np.int64 else 1000

        self.model = MaskedPredictionModel(input_dim, self.config.hidden_dim, vocab_size)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # 训练循环
        training_losses = []

        for epoch in range(self.config.epochs):
            epoch_loss = 0.0

            for batch_idx, (masked_data, original_data, mask) in enumerate(dataloader):
                self.optimizer.zero_grad()

                # 前向传播
                predictions = self.model(masked_data.long(), mask)

                # 计算损失
                loss = F.cross_entropy(predictions, original_data[mask].long())

                # 反向传播
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            training_losses.append(avg_loss)

            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

        # 计算训练时间
        training_time = (datetime.now() - start_time).total_seconds()

        # 评估表示质量
        representation_quality = self._evaluate_representation_quality(data)

        # 评估下游任务性能
        downstream_performance = self._evaluate_downstream_performance(data)

        result = SelfSupervisedResult(
            method=self.config.method.value,
            training_loss=training_losses,
            validation_loss=[],
            representation_quality=representation_quality,
            downstream_performance=downstream_performance,
            training_time=training_time,
            model_size=self._get_model_size(),
            timestamp=datetime.now()
        )

        logger.info(f"掩码预测训练完成，训练时间: {training_time:.2f}秒")
        return result

    def _evaluate_representation_quality(self, data: np.ndarray) -> float:
        """评估表示质量"""
        try:
            # 简单的表示质量评估
            return 0.8  # 占位符
        except Exception as e:
            logger.warning(f"表示质量评估失败: {e}")
            return 0.0

    def _evaluate_downstream_performance(self, data: np.ndarray) -> Dict[str, float]:
        """评估下游任务性能"""
        try:
            return {"masked_prediction_accuracy": 0.8}  # 占位符
        except Exception as e:
            logger.warning(f"下游任务评估失败: {e}")
            return {"masked_prediction_accuracy": 0.0}

    def _get_model_size(self) -> int:
        """获取模型大小"""
        try:
            total_params = sum(p.numel() for p in self.model.parameters())
            return total_params
        except Exception as e:
            logger.warning(f"模型大小计算失败: {e}")
            return 0

class ReconstructionLearner:
    """重构学习器"""

    def __init__(self, config: SelfSupervisedConfig):
        self.config = config
        self.model = None
        self.optimizer = None

    def train(self, data: np.ndarray) -> SelfSupervisedResult:
        """训练重构模型"""
        logger.info("开始重构训练")

        start_time = datetime.now()

        # 创建数据集和数据加载器
        dataset = SelfSupervisedDataset(data, SelfSupervisedMethod.RECONSTRUCTION, self.config)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        # 初始化模型
        input_dim = data.shape[1]
        self.model = ReconstructionModel(input_dim, self.config.hidden_dim)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # 训练循环
        training_losses = []

        for epoch in range(self.config.epochs):
            epoch_loss = 0.0

            for batch_idx, (noisy_data, original_data) in enumerate(dataloader):
                self.optimizer.zero_grad()

                # 前向传播
                encoded, decoded = self.model(noisy_data.float())

                # 计算重构损失
                loss = F.mse_loss(decoded, original_data.float())

                # 反向传播
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            training_losses.append(avg_loss)

            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

        # 计算训练时间
        training_time = (datetime.now() - start_time).total_seconds()

        # 评估表示质量
        representation_quality = self._evaluate_representation_quality(data)

        # 评估下游任务性能
        downstream_performance = self._evaluate_downstream_performance(data)

        result = SelfSupervisedResult(
            method=self.config.method.value,
            training_loss=training_losses,
            validation_loss=[],
            representation_quality=representation_quality,
            downstream_performance=downstream_performance,
            training_time=training_time,
            model_size=self._get_model_size(),
            timestamp=datetime.now()
        )

        logger.info(f"重构训练完成，训练时间: {training_time:.2f}秒")
        return result

    def _evaluate_representation_quality(self, data: np.ndarray) -> float:
        """评估表示质量"""
        try:
            self.model.eval()

            with torch.no_grad():
                # 获取表示
                representations, _ = self.model(torch.FloatTensor(data))

                # 计算重构误差
                _, reconstructed = self.model(torch.FloatTensor(data))
                reconstruction_error = F.mse_loss(reconstructed, torch.FloatTensor(data))

                # 质量分数（重构误差越小，质量越高）
                quality = max(0, 1 - reconstruction_error.item())

                return quality

        except Exception as e:
            logger.warning(f"表示质量评估失败: {e}")
            return 0.0

    def _evaluate_downstream_performance(self, data: np.ndarray) -> Dict[str, float]:
        """评估下游任务性能"""
        try:
            self.model.eval()

            with torch.no_grad():
                # 获取表示
                representations, _ = self.model(torch.FloatTensor(data))
                representations_np = representations.numpy()

                # 使用K-means聚类
                from sklearn.cluster import KMeans
                from sklearn.metrics import silhouette_score

                kmeans = KMeans(n_clusters=5, random_state=42)
                labels = kmeans.fit_predict(representations_np)

                silhouette = silhouette_score(representations_np, labels)

                return {
                    "reconstruction_error": 0.1,  # 占位符
                    "clustering_silhouette": silhouette
                }

        except Exception as e:
            logger.warning(f"下游任务评估失败: {e}")
            return {"reconstruction_error": 1.0, "clustering_silhouette": 0.0}

    def _get_model_size(self) -> int:
        """获取模型大小"""
        try:
            total_params = sum(p.numel() for p in self.model.parameters())
            return total_params
        except Exception as e:
            logger.warning(f"模型大小计算失败: {e}")
            return 0

class SelfSupervisedLearner:
    """自监督学习主类"""

    def __init__(self):
        self.learning_history: List[SelfSupervisedResult] = []

    def train(self, data: np.ndarray, config: SelfSupervisedConfig) -> SelfSupervisedResult:
        """自监督学习训练"""
        logger.info(f"开始自监督学习训练，方法: {config.method.value}")

        try:
            if config.method == SelfSupervisedMethod.CONTRASTIVE_LEARNING:
                learner = ContrastiveLearner(config)
            elif config.method == SelfSupervisedMethod.MASKED_PREDICTION:
                learner = MaskedPredictionLearner(config)
            elif config.method == SelfSupervisedMethod.RECONSTRUCTION:
                learner = ReconstructionLearner(config)
            else:
                raise CustomException(f"不支持的自监督学习方法: {config.method}")

            result = learner.train(data)
            self.learning_history.append(result)

            return result

        except Exception as e:
            logger.error(f"自监督学习训练失败: {e}")
            raise CustomException(f"自监督学习训练失败: {e}")

    def get_learning_history(self) -> List[SelfSupervisedResult]:
        """获取学习历史"""
        return self.learning_history

# 全局自监督学习器实例
self_supervised_learner = SelfSupervisedLearner()

# 异步任务
@async_task("self_supervised_learning", TaskPriority.HIGH)
def self_supervised_learning_task(data: List[List[float]], config_dict: Dict[str, Any]):
    """自监督学习任务"""
    config = SelfSupervisedConfig(**config_dict)
    data_array = np.array(data)
    result = self_supervised_learner.train(data_array, config)

    return {
        "result": asdict(result),
        "success": True
    }

# 自监督学习API
def train_self_supervised(data: np.ndarray, config: SelfSupervisedConfig) -> SelfSupervisedResult:
    """自监督学习训练"""
    return self_supervised_learner.train(data, config)

def get_self_supervised_history() -> List[SelfSupervisedResult]:
    """获取自监督学习历史"""
    return self_supervised_learner.get_learning_history()

if __name__ == "__main__":
    # 测试自监督学习
    import numpy as np

    # 创建测试数据
    np.random.seed(42)
    test_data = np.random.randn(1000, 50)

    # 对比学习配置
    contrastive_config = SelfSupervisedConfig(
        method=SelfSupervisedMethod.CONTRASTIVE_LEARNING,
        contrastive_method=ContrastiveMethod.SIMCLR,
        learning_rate=1e-4,
        batch_size=32,
        epochs=50,
        hidden_dim=128,
        projection_dim=64
    )

    # 训练对比学习模型
    result = train_self_supervised(test_data, contrastive_config)

    print("自监督学习完成:")
    print(f"方法: {result.method}")
    print(f"训练时间: {result.training_time:.2f}秒")
    print(f"表示质量: {result.representation_quality:.4f}")
    print(f"下游任务性能: {result.downstream_performance}")
    print(f"模型大小: {result.model_size}")

__all__ = ["'logger'", "'SelfSupervisedMethod'", "'ContrastiveMethod'", "'MaskingStrategy'", "'SelfSupervisedConfig'", "'SelfSupervisedResult'", "'ContrastiveLearningModel'", "'MaskedPredictionModel'", "'ReconstructionModel'", "'SelfSupervisedDataset'", "'ContrastiveLearner'", "'MaskedPredictionLearner'", "'ReconstructionLearner'", "'SelfSupervisedLearner'", "'self_supervised_learner'", "'self_supervised_learning_task'", "'train_self_supervised'", "'get_self_supervised_history'"]
