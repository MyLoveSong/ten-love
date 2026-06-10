

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据增强和自监督学习模块
解决数据不足问题，提升模型泛化能力
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import random

logger = logging.getLogger(__name__)

@dataclass
class AugmentationConfig:
    """数据增强配置"""
    noise_level: float = 0.05
    time_shift_range: int = 5  # 时间偏移范围（分钟）
    amplitude_scale_range: Tuple[float, float] = (0.9, 1.1)
    missing_data_ratio: float = 0.1
    synthetic_samples: int = 1000
    augmentation_ratio: float = 2.0  # 增强倍数

class GlucoseDataAugmenter:
    """血糖数据增强器"""

    def __init__(self, config: AugmentationConfig):
        self.config = config
        self.scaler = StandardScaler()

    def add_noise(self, data: np.ndarray) -> np.ndarray:
        """添加高斯噪声"""
        noise = np.random.normal(0, self.config.noise_level, data.shape)
        return data + noise

    def time_shift(self, data: pd.DataFrame) -> pd.DataFrame:
        """时间偏移增强"""
        shifted_data = data.copy()

        # 随机时间偏移
        shift_minutes = random.randint(-self.config.time_shift_range, self.config.time_shift_range)
        shifted_data['timestamp'] = shifted_data['timestamp'] + pd.Timedelta(minutes=shift_minutes)

        return shifted_data

    def amplitude_scaling(self, data: np.ndarray) -> np.ndarray:
        """幅度缩放"""
        scale_factor = random.uniform(*self.config.amplitude_scale_range)
        return data * scale_factor

    def missing_data_simulation(self, data: pd.DataFrame) -> pd.DataFrame:
        """模拟缺失数据"""
        augmented_data = data.copy()

        # 随机删除一些数据点
        missing_indices = np.random.choice(
            len(augmented_data),
            size=int(len(augmented_data) * self.config.missing_data_ratio),
            replace=False
        )

        # 标记缺失值
        for idx in missing_indices:
            col = np.random.choice(['glucose', 'carbohydrates', 'exercise'])
            if col in augmented_data.columns:
                augmented_data.loc[idx, col] = np.nan

        return augmented_data

    def interpolate_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """插值填充缺失值"""
        filled_data = data.copy()

        # 线性插值
        numeric_columns = filled_data.select_dtypes(include=[np.number]).columns
        filled_data[numeric_columns] = filled_data[numeric_columns].interpolate(method='linear')

        # 前向填充剩余缺失值
        filled_data[numeric_columns] = filled_data[numeric_columns].fillna(method='ffill')

        return filled_data

    def augment_dataset(self, data: pd.DataFrame) -> List[pd.DataFrame]:
        """数据集增强"""
        augmented_datasets = []

        # 原始数据
        augmented_datasets.append(data.copy())

        # 生成增强数据
        num_augmentations = int(len(data) * self.config.augmentation_ratio)

        for _ in range(num_augmentations):
            augmented_data = data.copy()

            # 应用多种增强技术
            if random.random() < 0.3:  # 30%概率添加噪声
                numeric_cols = augmented_data.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if col != 'timestamp':
                        augmented_data[col] = self.add_noise(augmented_data[col].values)

            if random.random() < 0.2:  # 20%概率时间偏移
                augmented_data = self.time_shift(augmented_data)

            if random.random() < 0.2:  # 20%概率幅度缩放
                numeric_cols = augmented_data.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if col != 'timestamp':
                        augmented_data[col] = self.amplitude_scaling(augmented_data[col].values)

            if random.random() < 0.1:  # 10%概率模拟缺失数据
                augmented_data = self.missing_data_simulation(augmented_data)
                augmented_data = self.interpolate_missing_values(augmented_data)

            augmented_datasets.append(augmented_data)

        return augmented_datasets

class SyntheticDataGenerator:
    """合成数据生成器"""

    def __init__(self, config: AugmentationConfig):
        self.config = config

    def generate_glucose_patterns(self, num_samples: int) -> pd.DataFrame:
        """生成血糖模式数据"""
        patterns = []

        for _ in range(num_samples):
            # 生成基础时间序列
            hours = np.arange(0, 24, 0.25)  # 15分钟间隔

            # 基础血糖模式（正常范围）
            base_glucose = 5.0 + 0.5 * np.sin(2 * np.pi * hours / 24)

            # 添加餐后血糖峰值
            meal_times = [7, 12, 18]  # 早餐、午餐、晚餐
            for meal_time in meal_times:
                meal_peak = np.exp(-((hours - meal_time) ** 2) / 2)
                base_glucose += 2.0 * meal_peak

            # 添加随机波动
            noise = np.random.normal(0, 0.3, len(hours))
            glucose_values = base_glucose + noise

            # 确保血糖值在合理范围内
            glucose_values = np.clip(glucose_values, 3.0, 12.0)

            # 创建数据记录
            for i, hour in enumerate(hours):
                pattern = {
                    'timestamp': pd.Timestamp.now().replace(hour=int(hour), minute=int((hour % 1) * 60)),
                    'glucose': glucose_values[i],
                    'hour': int(hour),
                    'day_of_week': random.randint(0, 6),
                    'carbohydrates': random.uniform(20, 80),
                    'protein': random.uniform(10, 40),
                    'fat': random.uniform(5, 25),
                    'exercise': random.uniform(0, 60),
                    'stress_level': random.uniform(1, 5),
                    'sleep_hours': random.uniform(6, 9)
                }
                patterns.append(pattern)

        return pd.DataFrame(patterns)

    def generate_diabetic_patterns(self, num_samples: int) -> pd.DataFrame:
        """生成糖尿病患者血糖模式"""
        patterns = []

        for _ in range(num_samples):
            hours = np.arange(0, 24, 0.25)

            # 糖尿病患者血糖模式（更高、更不稳定）
            base_glucose = 7.0 + 1.0 * np.sin(2 * np.pi * hours / 24)

            # 更明显的餐后峰值
            meal_times = [7, 12, 18]
            for meal_time in meal_times:
                meal_peak = np.exp(-((hours - meal_time) ** 2) / 1.5)
                base_glucose += 3.0 * meal_peak

            # 更大的随机波动
            noise = np.random.normal(0, 0.8, len(hours))
            glucose_values = base_glucose + noise

            # 血糖值范围更宽
            glucose_values = np.clip(glucose_values, 4.0, 15.0)

            for i, hour in enumerate(hours):
                pattern = {
                    'timestamp': pd.Timestamp.now().replace(hour=int(hour), minute=int((hour % 1) * 60)),
                    'glucose': glucose_values[i],
                    'hour': int(hour),
                    'day_of_week': random.randint(0, 6),
                    'carbohydrates': random.uniform(30, 100),
                    'protein': random.uniform(15, 50),
                    'fat': random.uniform(8, 30),
                    'exercise': random.uniform(0, 45),
                    'stress_level': random.uniform(2, 5),
                    'sleep_hours': random.uniform(5, 8),
                    'diabetes_type': random.choice([1, 2])  # 1型或2型糖尿病
                }
                patterns.append(pattern)

        return pd.DataFrame(patterns)

class SelfSupervisedLearningModule:
    """增强的自监督学习模块"""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 自监督学习模型
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        ).to(self.device)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim)
        ).to(self.device)

        # 时序预测头（用于时序自监督任务）
        self.temporal_head = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.ReLU(),
            nn.Linear(hidden_dim // 8, input_dim)
        ).to(self.device)

        # 掩码预测头（用于掩码自监督任务）
        self.mask_head = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.ReLU(),
            nn.Linear(hidden_dim // 8, input_dim)
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()) +
            list(self.temporal_head.parameters()) + list(self.mask_head.parameters()),
            lr=0.001, weight_decay=1e-5
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )

    def create_positive_pairs(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """创建正样本对（用于对比学习）"""
        # 对同一时间窗口的数据进行轻微变换
        data_augmented = data + torch.randn_like(data) * 0.1
        return data, data_augmented

    def contrastive_loss(self, anchor: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
        """对比损失"""
        # 编码
        anchor_encoded = self.encoder(anchor)
        positive_encoded = self.encoder(positive)

        # 计算相似度
        similarity = torch.cosine_similarity(anchor_encoded, positive_encoded, dim=1)

        # 对比损失（正样本对应该相似）
        loss = 1 - similarity.mean()

        return loss

    def reconstruction_loss(self, original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        """重构损失"""
        return nn.MSELoss()(reconstructed, original)

    def create_masked_data(self, data: torch.Tensor, mask_ratio: float = 0.15) -> Tuple[torch.Tensor, torch.Tensor]:
        """创建掩码数据（用于掩码学习）"""
        masked_data = data.clone()
        mask = torch.rand(data.shape) < mask_ratio
        masked_data[mask] = 0  # 或者用特殊token
        return masked_data, mask

    def create_temporal_pairs(self, data: torch.Tensor, window_size: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """创建时序预测对"""
        if len(data) < window_size + 1:
            return data[:-1], data[1:]

        past_data = []
        future_data = []

        for i in range(len(data) - window_size):
            past_data.append(data[i:i+window_size])
            future_data.append(data[i+window_size])

        return torch.stack(past_data), torch.stack(future_data)

    def temporal_prediction_loss(self, past_data: torch.Tensor, future_data: torch.Tensor) -> torch.Tensor:
        """时序预测损失"""
        # 编码过去的数据
        past_encoded = self.encoder(past_data.mean(dim=1))  # 聚合时序特征
        predicted_future = self.temporal_head(past_encoded)

        return nn.MSELoss()(predicted_future, future_data)

    def masked_prediction_loss(self, masked_data: torch.Tensor,
                             original_data: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """掩码预测损失"""
        encoded = self.encoder(masked_data)
        predicted = self.mask_head(encoded)

        # 只计算被掩码位置的损失
        return nn.MSELoss()(predicted[mask], original_data[mask])

    def train_self_supervised(self, data: torch.Tensor, epochs: int = 100):
        """增强的自监督学习训练"""
        data = data.to(self.device)
        self.encoder.train()
        self.decoder.train()
        self.temporal_head.train()
        self.mask_head.train()

        for epoch in range(epochs):
            total_loss = 0
            contrastive_losses = 0
            reconstruction_losses = 0
            temporal_losses = 0
            mask_losses = 0

            for batch_start in range(0, len(data), 32):
                batch_end = min(batch_start + 32, len(data))
                batch_data = data[batch_start:batch_end]

                if len(batch_data) < 2:
                    continue

                # 1. 对比学习任务
                anchor, positive = self.create_positive_pairs(batch_data)
                contrastive_loss = self.contrastive_loss(anchor, positive)

                # 2. 重构任务
                anchor_encoded = self.encoder(anchor)
                reconstructed = self.decoder(anchor_encoded)
                reconstruction_loss = self.reconstruction_loss(anchor, reconstructed)

                # 3. 时序预测任务
                if len(batch_data) > 5:
                    past_data, future_data = self.create_temporal_pairs(batch_data)
                    if len(past_data) > 0:
                        temporal_loss = self.temporal_prediction_loss(past_data, future_data)
                    else:
                        temporal_loss = torch.tensor(0.0, device=self.device)
                else:
                    temporal_loss = torch.tensor(0.0, device=self.device)

                # 4. 掩码预测任务
                masked_data, mask = self.create_masked_data(batch_data)
                if mask.sum() > 0:
                    mask_loss = self.masked_prediction_loss(masked_data, batch_data, mask)
                else:
                    mask_loss = torch.tensor(0.0, device=self.device)

                # 综合损失
                total_loss_batch = (
                    0.3 * contrastive_loss +
                    0.3 * reconstruction_loss +
                    0.2 * temporal_loss +
                    0.2 * mask_loss
                )

                # 反向传播
                self.optimizer.zero_grad()
                total_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.decoder.parameters()) +
                    list(self.temporal_head.parameters()) + list(self.mask_head.parameters()),
                    max_norm=1.0
                )
                self.optimizer.step()

                total_loss += total_loss_batch.item()
                contrastive_losses += contrastive_loss.item()
                reconstruction_losses += reconstruction_loss.item()
                temporal_losses += temporal_loss.item()
                mask_losses += mask_loss.item()

            self.scheduler.step()

            if epoch % 20 == 0:
                logger.info(f"自监督学习 Epoch {epoch}")
                logger.info(f"  总损失: {total_loss:.4f}")
                logger.info(f"  对比损失: {contrastive_losses:.4f}")
                logger.info(f"  重构损失: {reconstruction_losses:.4f}")
                logger.info(f"  时序损失: {temporal_losses:.4f}")
                logger.info(f"  掩码损失: {mask_losses:.4f}")
                logger.info(f"  学习率: {self.scheduler.get_last_lr()[0]:.6f}")

    def extract_features(self, data: torch.Tensor) -> torch.Tensor:
        """提取特征"""
        self.encoder.eval()
        with torch.no_grad():
            features = self.encoder(data)
        return features

class DataEnhancementPipeline:
    """数据增强流水线"""

    def __init__(self, config: AugmentationConfig):
        self.config = config
        self.augmenter = GlucoseDataAugmenter(config)
        self.synthetic_generator = SyntheticDataGenerator(config)
        self.ssl_module = None

    def enhance_dataset(self, original_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """增强数据集"""
        enhanced_datasets = {}

        # 1. 数据增强
        logger.info("开始数据增强...")
        augmented_datasets = self.augmenter.augment_dataset(original_data)
        enhanced_datasets['augmented'] = pd.concat(augmented_datasets, ignore_index=True)

        # 2. 合成数据生成
        logger.info("生成合成数据...")
        synthetic_normal = self.synthetic_generator.generate_glucose_patterns(
            self.config.synthetic_samples // 2
        )
        synthetic_diabetic = self.synthetic_generator.generate_diabetic_patterns(
            self.config.synthetic_samples // 2
        )
        enhanced_datasets['synthetic'] = pd.concat([synthetic_normal, synthetic_diabetic], ignore_index=True)

        # 3. 合并所有数据
        logger.info("合并增强数据...")
        all_data = [original_data] + augmented_datasets + [synthetic_normal, synthetic_diabetic]
        enhanced_datasets['combined'] = pd.concat(all_data, ignore_index=True)

        # 4. 数据清洗和标准化
        enhanced_datasets['final'] = self._clean_and_normalize(enhanced_datasets['combined'])

        return enhanced_datasets

    def _clean_and_normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        """数据清洗和标准化"""
        cleaned_data = data.copy()

        # 移除异常值
        numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col != 'timestamp':
                Q1 = cleaned_data[col].quantile(0.25)
                Q3 = cleaned_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                cleaned_data = cleaned_data[
                    (cleaned_data[col] >= lower_bound) &
                    (cleaned_data[col] <= upper_bound)
                ]

        # 标准化
        scaler = StandardScaler()
        cleaned_data[numeric_columns] = scaler.fit_transform(cleaned_data[numeric_columns])

        return cleaned_data

    def train_self_supervised_features(self, data: pd.DataFrame):
        """训练自监督学习特征提取器"""
        logger.info("开始自监督学习训练...")

        # 准备数据
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        feature_data = data[numeric_columns].values

        # 转换为张量
        data_tensor = torch.FloatTensor(feature_data)

        # 初始化自监督学习模块
        self.ssl_module = SelfSupervisedLearningModule(
            input_dim=len(numeric_columns)
        )

        # 训练
        self.ssl_module.train_self_supervised(data_tensor)

        logger.info("自监督学习训练完成")

    def extract_enhanced_features(self, data: pd.DataFrame) -> np.ndarray:
        """提取增强特征"""
        if self.ssl_module is None:
            logger.warning("自监督学习模块未训练，使用原始特征")
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            return data[numeric_columns].values

        # 使用训练好的自监督学习模块提取特征
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        feature_data = data[numeric_columns].values
        data_tensor = torch.FloatTensor(feature_data)

        enhanced_features = self.ssl_module.extract_features(data_tensor)
        return enhanced_features.numpy()

# 使用示例
def main():
    """使用示例"""
    # 创建配置
    config = AugmentationConfig(
        noise_level=0.05,
        augmentation_ratio=2.0,
        synthetic_samples=1000
    )

    # 创建增强流水线
    pipeline = DataEnhancementPipeline(config)

    # 模拟原始数据
    original_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='15min'),
        'glucose': np.random.normal(6.0, 1.0, 100),
        'carbohydrates': np.random.uniform(20, 80, 100),
        'exercise': np.random.uniform(0, 60, 100)
    })

    # 增强数据
    enhanced_datasets = pipeline.enhance_dataset(original_data)

    print(f"原始数据大小: {len(original_data)}")
    print(f"增强后数据大小: {len(enhanced_datasets['final'])}")

    # 训练自监督学习特征
    pipeline.train_self_supervised_features(enhanced_datasets['final'])

    # 提取增强特征
    enhanced_features = pipeline.extract_enhanced_features(enhanced_datasets['final'])
    print(f"增强特征维度: {enhanced_features.shape}")

class GANDataGenerator:
    """GAN数据生成器"""

    def __init__(self, latent_dim: int = 100, data_dim: int = 5):
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 生成器网络
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, data_dim),
            nn.Tanh()
        ).to(self.device)

        # 判别器网络
        self.discriminator = nn.Sequential(
            nn.Linear(data_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        ).to(self.device)

        # 优化器
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        # 损失函数
        self.criterion = nn.BCELoss()

        self.is_trained = False
        logger.info("GAN数据生成器初始化完成")

    def train(self, real_data: torch.Tensor, epochs: int = 1000, batch_size: int = 64):
        """训练GAN模型"""
        real_data = real_data.to(self.device)
        dataset = torch.utils.data.TensorDataset(real_data)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.generator.train()
        self.discriminator.train()

        for epoch in range(epochs):
            for i, (real_batch,) in enumerate(dataloader):
                batch_size = real_batch.size(0)

                # 训练判别器
                self.d_optimizer.zero_grad()

                # 真实数据
                real_labels = torch.ones(batch_size, 1).to(self.device)
                real_output = self.discriminator(real_batch)
                real_loss = self.criterion(real_output, real_labels)

                # 生成数据
                noise = torch.randn(batch_size, self.latent_dim).to(self.device)
                fake_data = self.generator(noise)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)
                fake_output = self.discriminator(fake_data.detach())
                fake_loss = self.criterion(fake_output, fake_labels)

                d_loss = real_loss + fake_loss
                d_loss.backward()
                self.d_optimizer.step()

                # 训练生成器
                self.g_optimizer.zero_grad()

                noise = torch.randn(batch_size, self.latent_dim).to(self.device)
                fake_data = self.generator(noise)
                fake_output = self.discriminator(fake_data)
                g_loss = self.criterion(fake_output, real_labels)

                g_loss.backward()
                self.g_optimizer.step()

            if epoch % 100 == 0:
                logger.info(f"GAN训练 Epoch {epoch}, D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")

        self.is_trained = True
        logger.info("GAN训练完成")

    def generate_samples(self, num_samples: int) -> torch.Tensor:
        """生成合成样本"""
        if not self.is_trained:
            logger.warning("GAN模型未训练，将返回随机数据")

        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(num_samples, self.latent_dim).to(self.device)
            synthetic_data = self.generator(noise)

        return synthetic_data.cpu()

class CrossDomainDataSharing:
    """跨领域数据共享模块"""

    def __init__(self):
        self.domain_adapters = {}
        self.shared_knowledge = {}

    def register_domain_adapter(self, domain_name: str, adapter_func):
        """注册领域适配器"""
        self.domain_adapters[domain_name] = adapter_func
        logger.info(f"注册领域适配器: {domain_name}")

    def adapt_data_to_domain(self, source_data: pd.DataFrame,
                           target_domain: str) -> pd.DataFrame:
        """将数据适配到目标领域"""
        if target_domain not in self.domain_adapters:
            logger.warning(f"未找到领域适配器: {target_domain}")
            return source_data

        adapter = self.domain_adapters[target_domain]
        adapted_data = adapter(source_data)

        logger.info(f"数据适配到领域 {target_domain}，样本数: {len(adapted_data)}")
        return adapted_data

    def extract_cross_domain_features(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """提取跨领域通用特征"""
        common_features = []

        # 找到所有数据集的共同列
        all_columns = set()
        for domain, data in datasets.items():
            all_columns.update(data.columns)

        # 提取共同特征
        common_columns = list(all_columns)
        for domain, data in datasets.items():
            available_columns = [col for col in common_columns if col in data.columns]
            if available_columns:
                domain_features = data[available_columns].copy()
                domain_features['domain'] = domain
                common_features.append(domain_features)

        if common_features:
            combined_features = pd.concat(common_features, ignore_index=True)
            logger.info(f"提取跨领域特征，总样本数: {len(combined_features)}")
            return combined_features
        else:
            logger.warning("未找到跨领域共同特征")
            return pd.DataFrame()

# 默认领域适配器

def diabetes_domain_adapter(data: pd.DataFrame) -> pd.DataFrame:
    """糖尿病领域适配器"""
    adapted_data = data.copy()

    # 调整血糖水平模拟糖尿病患者
    if 'glucose' in adapted_data.columns:
        adapted_data['glucose'] = adapted_data['glucose'] * np.random.normal(1.3, 0.2, len(adapted_data))
        adapted_data['glucose'] = adapted_data['glucose'].clip(100, 400)

    # 添加糖尿病特征
    adapted_data['diabetes_risk'] = 'high'
    adapted_data['medication'] = np.random.choice(['insulin', 'metformin', 'none'], len(adapted_data))

    return adapted_data

def healthy_domain_adapter(data: pd.DataFrame) -> pd.DataFrame:
    """健康人群领域适配器"""
    adapted_data = data.copy()

    # 调整血糖水平模拟健康人群
    if 'glucose' in adapted_data.columns:
        adapted_data['glucose'] = adapted_data['glucose'] * np.random.normal(0.85, 0.1, len(adapted_data))
        adapted_data['glucose'] = adapted_data['glucose'].clip(70, 150)

    # 添加健康特征
    adapted_data['diabetes_risk'] = 'low'
    adapted_data['medication'] = 'none'

    return adapted_data

# 创建跨领域数据共享实例
cross_domain_sharing = CrossDomainDataSharing()
cross_domain_sharing.register_domain_adapter('diabetes', diabetes_domain_adapter)
cross_domain_sharing.register_domain_adapter('healthy', healthy_domain_adapter)

if __name__ == "__main__":
    main()

__all__ = ["'logger'", "'AugmentationConfig'", "'GlucoseDataAugmenter'", "'SyntheticDataGenerator'", "'SelfSupervisedLearningModule'", "'DataEnhancementPipeline'", "'main'", "'GANDataGenerator'", "'CrossDomainDataSharing'", "'diabetes_domain_adapter'", "'healthy_domain_adapter'", "'cross_domain_sharing'"]
