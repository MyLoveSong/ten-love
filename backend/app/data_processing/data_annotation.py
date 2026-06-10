

"""
学术级数据标注与标签优化模块 - MCP架构增强版
支持自监督学习、对比学习、标签平衡、噪声处理等
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple, Callable, Type
from dataclasses import dataclass, asdict
from enum import Enum
import json
from datetime import datetime
import warnings
import asyncio
from concurrent.futures import ThreadPoolExecutor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
from imblearn.combine import SMOTEENN, SMOTETomek
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod

from backend.app.core.exceptions import CustomException, ValidationError
from backend.app.core.task_queue import async_task, TaskPriority
from backend.app.core.dependency_injection import injectable, singleton, get_service
from backend.app.core.configuration import get_configuration
from backend.app.core.structured_logging import get_logger, log_function_call, log_performance
from backend.app.core.error_handling import ErrorContext, handle_error

logger = get_logger("data_annotation")

class LabelingStrategy(Enum):
    """标注策略"""
    SUPERVISED = "supervised"            # 监督学习
    SEMI_SUPERVISED = "semi_supervised"  # 半监督学习
    SELF_SUPERVISED = "self_supervised"  # 自监督学习
    WEAKLY_SUPERVISED = "weakly_supervised"  # 弱监督学习
    ACTIVE_LEARNING = "active_learning"  # 主动学习

class ContrastiveMethod(Enum):
    """对比学习方法"""
    SIMCLR = "simclr"                    # SimCLR
    MOC = "moco"                         # MoCo
    SWAV = "swav"                        # SwAV
    BYOL = "byol"                        # BYOL
    SUPERVISED_CONTRASTIVE = "supervised_contrastive"  # 监督对比学习

class LabelBalanceMethod(Enum):
    """标签平衡方法"""
    OVERSAMPLING = "oversampling"        # 过采样
    UNDERSAMPLING = "undersampling"      # 欠采样
    COMBINED = "combined"                # 组合采样
    WEIGHTED_LOSS = "weighted_loss"      # 加权损失
    COST_SENSITIVE = "cost_sensitive"    # 代价敏感

class NoiseHandlingMethod(Enum):
    """噪声处理方法"""
    ENSEMBLE = "ensemble"                # 集成学习
    ROBUST_LOSS = "robust_loss"          # 鲁棒损失
    NOISE_CORRECTION = "noise_correction"  # 噪声校正
    CONFIDENCE_WEIGHTING = "confidence_weighting"  # 置信度加权

@dataclass
class LabelingConfig:
    """标注配置"""
    strategy: LabelingStrategy
    contrastive_method: Optional[ContrastiveMethod] = None
    balance_method: Optional[LabelBalanceMethod] = None
    noise_handling: Optional[NoiseHandlingMethod] = None
    augmentation_factor: float = 2.0
    confidence_threshold: float = 0.8
    max_iterations: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    temperature: float = 0.1
    custom_parameters: Optional[Dict[str, Any]] = None

@dataclass
class LabelingResult:
    """标注结果"""
    labeled_data: pd.DataFrame
    unlabeled_data: pd.DataFrame
    pseudo_labels: Optional[pd.DataFrame] = None
    confidence_scores: Optional[np.ndarray] = None
    quality_metrics: Dict[str, float] = None
    processing_time: float = 0.0
    metadata: Dict[str, Any] = None

class IContrastiveLearner(ABC):
    """对比学习接口"""

    @abstractmethod
    async def train(self, data: np.ndarray, config: LabelingConfig) -> Dict[str, Any]:
        """训练对比学习模型"""
        pass

    @abstractmethod
    async def extract_features(self, data: np.ndarray) -> np.ndarray:
        """提取特征"""
        pass

    @abstractmethod
    async def generate_pseudo_labels(self, unlabeled_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """生成伪标签"""
        pass

@singleton(IContrastiveLearner)
class SimCLRLearner(IContrastiveLearner):
    """SimCLR对比学习器"""

    def __init__(self):
        self.model: Optional[nn.Module] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = get_logger("simclr")

    async def train(self, data: np.ndarray, config: LabelingConfig) -> Dict[str, Any]:
        """训练SimCLR模型"""
        self.logger.info("开始训练SimCLR模型")

        # 创建SimCLR模型
        self.model = SimCLRModel(input_dim=data.shape[1], hidden_dim=128, output_dim=64)
        self.model.to(self.device)

        # 创建数据加载器
        dataset = ContrastiveDataset(data)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

        # 训练模型
        optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        for epoch in range(config.max_iterations):
            epoch_loss = 0.0
            for batch in dataloader:
                batch = batch.to(self.device)

                # 生成正负样本对
                pos_pairs, neg_pairs = self._generate_pairs(batch)

                # 前向传播
                optimizer.zero_grad()
                loss = self._contrastive_loss(pos_pairs, neg_pairs, config.temperature)

                # 反向传播
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            total_loss += epoch_loss
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")

        return {
            "total_loss": total_loss,
            "epochs": config.max_iterations,
            "model_size": sum(p.numel() for p in self.model.parameters())
        }

    async def extract_features(self, data: np.ndarray) -> np.ndarray:
        """提取特征"""
        if self.model is None:
            raise CustomException("模型未训练")

        self.model.eval()
        features = []

        with torch.no_grad():
            for i in range(0, len(data), 32):
                batch = torch.FloatTensor(data[i:i+32]).to(self.device)
                batch_features = self.model.encoder(batch)
                features.append(batch_features.cpu().numpy())

        return np.concatenate(features, axis=0)

    async def generate_pseudo_labels(self, unlabeled_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """生成伪标签"""
        features = await self.extract_features(unlabeled_data)

        # 使用K-means聚类生成伪标签
        kmeans = KMeans(n_clusters=2, random_state=42)
        pseudo_labels = kmeans.fit_predict(features)

        # 计算置信度分数
        distances = kmeans.transform(features)
        confidence_scores = 1.0 / (1.0 + distances.min(axis=1))

        return pseudo_labels, confidence_scores

    def _generate_pairs(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成正负样本对"""
        # 简化的正负样本对生成
        batch_size = batch.size(0)

        # 正样本对：同一样本的不同增强版本
        pos_pairs = torch.cat([batch, batch], dim=0)

        # 负样本对：不同样本
        neg_indices = torch.randperm(batch_size)
        neg_pairs = torch.cat([batch, batch[neg_indices]], dim=0)

        return pos_pairs, neg_pairs

    def _contrastive_loss(self, pos_pairs: torch.Tensor, neg_pairs: torch.Tensor, temperature: float) -> torch.Tensor:
        """对比损失"""
        # 简化的对比损失实现
        pos_sim = F.cosine_similarity(pos_pairs[:len(pos_pairs)//2], pos_pairs[len(pos_pairs)//2:])
        neg_sim = F.cosine_similarity(neg_pairs[:len(neg_pairs)//2], neg_pairs[len(neg_pairs)//2:])

        pos_loss = -torch.log(torch.sigmoid(pos_sim / temperature)).mean()
        neg_loss = -torch.log(torch.sigmoid(-neg_sim / temperature)).mean()

        return pos_loss + neg_loss

class SimCLRModel(nn.Module):
    """SimCLR模型"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.projection = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        features = self.encoder(x)
        projection = self.projection(features)
        return features, projection

class ContrastiveDataset(Dataset):
    """对比学习数据集"""

    def __init__(self, data: np.ndarray):
        self.data = torch.FloatTensor(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class LabelBalancer:
    """标签平衡器"""

    def __init__(self):
        self.logger = get_logger("label_balancer")

    async def balance_labels(self, X: np.ndarray, y: np.ndarray, method: LabelBalanceMethod) -> Tuple[np.ndarray, np.ndarray]:
        """平衡标签"""
        self.logger.info(f"使用{method.value}方法平衡标签")

        if method == LabelBalanceMethod.OVERSAMPLING:
            return await self._oversample(X, y)
        elif method == LabelBalanceMethod.UNDERSAMPLING:
            return await self._undersample(X, y)
        elif method == LabelBalanceMethod.COMBINED:
            return await self._combined_sampling(X, y)
        else:
            return X, y

    async def _oversample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """过采样"""
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return X_resampled, y_resampled

    async def _undersample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """欠采样"""
        undersampler = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = undersampler.fit_resample(X, y)
        return X_resampled, y_resampled

    async def _combined_sampling(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """组合采样"""
        smote_enn = SMOTEENN(random_state=42)
        X_resampled, y_resampled = smote_enn.fit_resample(X, y)
        return X_resampled, y_resampled

class NoiseHandler:
    """噪声处理器"""

    def __init__(self):
        self.logger = get_logger("noise_handler")

    async def handle_noise(self, X: np.ndarray, y: np.ndarray, method: NoiseHandlingMethod) -> Tuple[np.ndarray, np.ndarray]:
        """处理标签噪声"""
        self.logger.info(f"使用{method.value}方法处理标签噪声")

        if method == NoiseHandlingMethod.ENSEMBLE:
            return await self._ensemble_method(X, y)
        elif method == NoiseHandlingMethod.CONFIDENCE_WEIGHTING:
            return await self._confidence_weighting(X, y)
        else:
            return X, y

    async def _ensemble_method(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """集成学习方法"""
        # 创建多个基础分类器
        classifiers = [
            RandomForestClassifier(n_estimators=100, random_state=42),
            LogisticRegression(random_state=42),
            SVC(probability=True, random_state=42)
        ]

        # 创建投票分类器
        ensemble = VotingClassifier(
            estimators=[(f"clf_{i}", clf) for i, clf in enumerate(classifiers)],
            voting='soft'
        )

        # 训练集成模型
        ensemble.fit(X, y)

        # 预测并获取置信度
        predictions = ensemble.predict(X)
        probabilities = ensemble.predict_proba(X)
        confidence_scores = np.max(probabilities, axis=1)

        # 过滤低置信度样本
        high_confidence_mask = confidence_scores > 0.8
        X_filtered = X[high_confidence_mask]
        y_filtered = predictions[high_confidence_mask]

        return X_filtered, y_filtered

    async def _confidence_weighting(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """置信度加权方法"""
        # 训练基础模型
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        # 获取预测置信度
        probabilities = model.predict_proba(X)
        confidence_scores = np.max(probabilities, axis=1)

        # 根据置信度加权样本
        weights = confidence_scores / confidence_scores.sum()

        # 重采样
        indices = np.random.choice(len(X), size=len(X), replace=True, p=weights)
        X_weighted = X[indices]
        y_weighted = y[indices]

        return X_weighted, y_weighted

class AcademicDataLabeler:
    """学术级数据标注器"""

    def __init__(self):
        self.logger = get_logger("academic_data_labeler")
        self.contrastive_learner = SimCLRLearner()
        self.label_balancer = LabelBalancer()
        self.noise_handler = NoiseHandler()

    @log_function_call("academic_data_labeler")
    @log_performance("academic_data_labeler")
    async def label_data(self, data: pd.DataFrame, config: LabelingConfig) -> LabelingResult:
        """标注数据"""
        start_time = datetime.now()

        try:
            # 分离有标签和无标签数据
            labeled_data, unlabeled_data = self._split_labeled_unlabeled(data)

            if config.strategy == LabelingStrategy.SELF_SUPERVISED:
                result = await self._self_supervised_labeling(labeled_data, unlabeled_data, config)
            elif config.strategy == LabelingStrategy.SEMI_SUPERVISED:
                result = await self._semi_supervised_labeling(labeled_data, unlabeled_data, config)
            else:
                result = await self._supervised_labeling(labeled_data, config)

            # 计算处理时间
            processing_time = (datetime.now() - start_time).total_seconds()
            result.processing_time = processing_time

            self.logger.info(f"数据标注完成，耗时: {processing_time:.2f}秒")
            return result

        except Exception as e:
            error_context = ErrorContext(
                module="data_annotation",
                function="label_data",
                extra_data={"config": asdict(config)}
            )
            await handle_error(e, error_context)
            raise

    async def _split_labeled_unlabeled(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """分离有标签和无标签数据"""
        # 假设标签列名为'label'，NaN表示无标签
        labeled_mask = data['label'].notna()
        labeled_data = data[labeled_mask].copy()
        unlabeled_data = data[~labeled_mask].copy()

        return labeled_data, unlabeled_data

    async def _self_supervised_labeling(self, labeled_data: pd.DataFrame, unlabeled_data: pd.DataFrame, config: LabelingConfig) -> LabelingResult:
        """自监督学习标注"""
        if not unlabeled_data.empty:
            # 准备数据
            X_unlabeled = unlabeled_data.drop('label', axis=1).values

            # 训练对比学习模型
            training_result = await self.contrastive_learner.train(X_unlabeled, config)

            # 生成伪标签
            pseudo_labels, confidence_scores = await self.contrastive_learner.generate_pseudo_labels(X_unlabeled)

            # 创建伪标签数据
            pseudo_labeled_data = unlabeled_data.copy()
            pseudo_labeled_data['label'] = pseudo_labels
            pseudo_labeled_data['confidence'] = confidence_scores

            # 合并标签数据
            all_labeled_data = pd.concat([labeled_data, pseudo_labeled_data], ignore_index=True)

            # 计算质量指标
            quality_metrics = {
                "pseudo_label_count": len(pseudo_labeled_data),
                "avg_confidence": np.mean(confidence_scores),
                "high_confidence_ratio": np.mean(confidence_scores > config.confidence_threshold)
            }

            return LabelingResult(
                labeled_data=all_labeled_data,
                unlabeled_data=pd.DataFrame(),
                pseudo_labels=pseudo_labeled_data,
                confidence_scores=confidence_scores,
                quality_metrics=quality_metrics,
                metadata={"training_result": training_result}
            )
        else:
            return LabelingResult(
                labeled_data=labeled_data,
                unlabeled_data=unlabeled_data,
                quality_metrics={"pseudo_label_count": 0}
            )

    async def _semi_supervised_labeling(self, labeled_data: pd.DataFrame, unlabeled_data: pd.DataFrame, config: LabelingConfig) -> LabelingResult:
        """半监督学习标注"""
        if not unlabeled_data.empty:
            # 准备数据
            X_labeled = labeled_data.drop('label', axis=1).values
            y_labeled = labeled_data['label'].values
            X_unlabeled = unlabeled_data.drop('label', axis=1).values

            # 训练模型
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_labeled, y_labeled)

            # 预测无标签数据
            predictions = model.predict(X_unlabeled)
            probabilities = model.predict_proba(X_unlabeled)
            confidence_scores = np.max(probabilities, axis=1)

            # 创建伪标签数据
            pseudo_labeled_data = unlabeled_data.copy()
            pseudo_labeled_data['label'] = predictions
            pseudo_labeled_data['confidence'] = confidence_scores

            # 过滤高置信度样本
            high_confidence_mask = confidence_scores > config.confidence_threshold
            high_confidence_data = pseudo_labeled_data[high_confidence_mask]

            # 合并标签数据
            all_labeled_data = pd.concat([labeled_data, high_confidence_data], ignore_index=True)

            # 计算质量指标
            quality_metrics = {
                "pseudo_label_count": len(high_confidence_data),
                "avg_confidence": np.mean(confidence_scores),
                "high_confidence_ratio": np.mean(high_confidence_mask)
            }

            return LabelingResult(
                labeled_data=all_labeled_data,
                unlabeled_data=unlabeled_data[~high_confidence_mask],
                pseudo_labels=high_confidence_data,
                confidence_scores=confidence_scores,
                quality_metrics=quality_metrics
            )
        else:
            return LabelingResult(
                labeled_data=labeled_data,
                unlabeled_data=unlabeled_data,
                quality_metrics={"pseudo_label_count": 0}
            )

    async def _supervised_labeling(self, labeled_data: pd.DataFrame, config: LabelingConfig) -> LabelingResult:
        """监督学习标注"""
        # 准备数据
        X = labeled_data.drop('label', axis=1).values
        y = labeled_data['label'].values

        # 标签平衡
        if config.balance_method:
            X_balanced, y_balanced = await self.label_balancer.balance_labels(X, y, config.balance_method)

            # 创建平衡后的数据
            balanced_data = labeled_data.copy()
            balanced_data = balanced_data.iloc[:len(X_balanced)].copy()
            balanced_data.iloc[:, :-1] = X_balanced
            balanced_data['label'] = y_balanced

            labeled_data = balanced_data

        # 噪声处理
        if config.noise_handling:
            X_clean, y_clean = await self.noise_handler.handle_noise(X, y, config.noise_handling)

            # 创建清理后的数据
            clean_data = labeled_data.iloc[:len(X_clean)].copy()
            clean_data.iloc[:, :-1] = X_clean
            clean_data['label'] = y_clean

            labeled_data = clean_data

        # 计算质量指标
        quality_metrics = {
            "total_samples": len(labeled_data),
            "unique_labels": len(labeled_data['label'].unique()),
            "label_distribution": labeled_data['label'].value_counts().to_dict()
        }

        return LabelingResult(
            labeled_data=labeled_data,
            unlabeled_data=pd.DataFrame(),
            quality_metrics=quality_metrics
        )

# 全局数据标注器实例
academic_data_labeler = AcademicDataLabeler()

# 异步任务
@async_task("label_data", TaskPriority.HIGH)
def label_data_task(data_dict: Dict[str, Any], config_dict: Dict[str, Any]):
    """数据标注任务"""
    data = pd.DataFrame(data_dict)
    config = LabelingConfig(**config_dict)

    result = asyncio.run(academic_data_labeler.label_data(data, config))

    return {
        "result": asdict(result),
        "success": True
    }

# 数据标注API
def label_data(data: pd.DataFrame, config: LabelingConfig) -> LabelingResult:
    """标注数据"""
    return asyncio.run(academic_data_labeler.label_data(data, config))

def balance_labels(X: np.ndarray, y: np.ndarray, method: LabelBalanceMethod) -> Tuple[np.ndarray, np.ndarray]:
    """平衡标签"""
    balancer = LabelBalancer()
    return asyncio.run(balancer.balance_labels(X, y, method))

def handle_label_noise(X: np.ndarray, y: np.ndarray, method: NoiseHandlingMethod) -> Tuple[np.ndarray, np.ndarray]:
    """处理标签噪声"""
    handler = NoiseHandler()
    return asyncio.run(handler.handle_noise(X, y, method))

if __name__ == "__main__":
    # 测试数据标注
    import numpy as np

    # 创建测试数据
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)

    # 创建DataFrame
    data = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    data['label'] = y

    # 随机设置一些标签为NaN（模拟无标签数据）
    mask = np.random.random(n_samples) < 0.3
    data.loc[mask, 'label'] = np.nan

    # 创建配置
    config = LabelingConfig(
        strategy=LabelingStrategy.SELF_SUPERVISED,
        contrastive_method=ContrastiveMethod.SIMCLR,
        confidence_threshold=0.8,
        max_iterations=50
    )

    # 执行标注
    result = label_data(data, config)

    print("标注结果:")
    print(f"标签数据数量: {len(result.labeled_data)}")
    print(f"伪标签数量: {len(result.pseudo_labels) if result.pseudo_labels is not None else 0}")
    print(f"质量指标: {result.quality_metrics}")
    print(f"处理时间: {result.processing_time:.2f}秒")

__all__ = ["'logger'", "'LabelingStrategy'", "'ContrastiveMethod'", "'LabelBalanceMethod'", "'NoiseHandlingMethod'", "'LabelingConfig'", "'LabelingResult'", "'IContrastiveLearner'", "'SimCLRLearner'", "'SimCLRModel'", "'ContrastiveDataset'", "'LabelBalancer'", "'NoiseHandler'", "'AcademicDataLabeler'", "'academic_data_labeler'", "'label_data_task'", "'label_data'", "'balance_labels'", "'handle_label_noise'"]
