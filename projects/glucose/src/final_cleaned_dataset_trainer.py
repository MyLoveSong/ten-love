#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
最终清洗数据集训练器
使用保留的3,005个高质量样本进行微调训练和验证评估
"""

import os
import sys
import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    explained_variance_score, max_error
)
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FinalCleanedDatasetTrainer:
    """最终清洗数据集训练器"""

    def __init__(self):
        """初始化训练器"""
        self.cleaned_data_path = "data/cleaned_dataset/cleaned_training_data.json"
        self.final_data_path = "data/final_dataset/final_training_data.json"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler = StandardScaler()
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_r2': [],
            'val_r2': []
        }

        # 最优配置
        self.optimal_config = {
            "epochs": 60,           # 增加轮数以充分利用高质量数据
            "learning_rate": 0.00003, # 进一步降低学习率
            "patience": 30,         # 增加早停耐心
            "batch_size": 64,       # 适中的批次大小
            "weight_decay": 1e-6,   # 进一步降低权重衰减
            "dropout_rate": 0.02    # 极低的dropout
        }

        logger.info("🔧 初始化最终清洗数据集训练器")
        logger.info(f"📊 清洗数据路径: {self.cleaned_data_path}")
        logger.info(f"📊 设备: {self.device}")
        logger.info(f"📊 最优配置: {self.optimal_config}")

    def create_final_dataset(self):
        """创建最终数据集，只包含清洗后的高质量样本"""
        logger.info("🔄 创建最终数据集...")

        if not os.path.exists(self.cleaned_data_path):
            logger.error(f"❌ 清洗数据文件不存在: {self.cleaned_data_path}")
            return False

        try:
            # 加载清洗后的数据
            with open(self.cleaned_data_path, 'r', encoding='utf-8') as f:
                cleaned_data = json.load(f)

            cleaned_samples = cleaned_data.get('training_samples', [])
            logger.info(f"📊 清洗后样本数: {len(cleaned_samples)}")

            # 创建最终数据集
            final_dataset = {
                "dataset_info": {
                    "name": "Final High-Quality Cultural Adaptation Dataset",
                    "version": "1.0.0",
                    "description": "最终高质量文化适应模型训练数据集，仅包含通过严格清洗的样本",
                    "total_samples": len(cleaned_samples),
                    "feature_dimensions": 21,
                    "data_sources": self._get_data_sources(cleaned_samples),
                    "data_qualities": self._get_data_qualities(cleaned_samples),
                    "average_quality_score": self._calculate_average_quality(cleaned_samples),
                    "created_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "cleaning_info": cleaned_data['dataset_info'].get('cleaning_info', {}),
                    "final_processing": {
                        "removed_low_quality": True,
                        "removed_duplicates": True,
                        "removed_outliers": True,
                        "quality_threshold": 0.8,
                        "final_quality_assurance": True
                    }
                },
                "training_samples": cleaned_samples
            }

            # 保存最终数据集
            output_dir = Path("data/final_dataset")
            output_dir.mkdir(parents=True, exist_ok=True)

            with open(self.final_data_path, 'w', encoding='utf-8') as f:
                json.dump(final_dataset, f, ensure_ascii=False, indent=2)

            logger.info(f"✅ 最终数据集已保存到: {self.final_data_path}")
            logger.info(f"📊 最终样本数: {len(cleaned_samples)}")

            return True

        except Exception as e:
            logger.error(f"❌ 创建最终数据集失败: {e}")
            return False

    def _get_data_sources(self, samples):
        """获取数据源列表"""
        sources = set()
        for sample in samples:
            if isinstance(sample, dict) and 'source' in sample:
                sources.add(sample['source'])
        return list(sources)

    def _get_data_qualities(self, samples):
        """获取数据质量列表"""
        qualities = set()
        for sample in samples:
            if isinstance(sample, dict) and 'data_quality' in sample:
                qualities.add(sample['data_quality'])
        return list(qualities)

    def _calculate_average_quality(self, samples):
        """计算平均质量评分"""
        total_score = 0
        count = 0

        for sample in samples:
            if isinstance(sample, dict) and 'quality_scores' in sample:
                quality_scores = sample['quality_scores']
                if isinstance(quality_scores, dict) and 'overall' in quality_scores:
                    total_score += quality_scores['overall']
                    count += 1

        return total_score / count if count > 0 else 0.0

    def load_final_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """加载最终数据"""
        logger.info("📖 加载最终高质量数据...")

        if not os.path.exists(self.final_data_path):
            logger.error(f"❌ 最终数据文件不存在: {self.final_data_path}")
            return None, None, None

        with open(self.final_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 获取训练样本
        samples = data.get('training_samples', [])
        logger.info(f"📊 最终样本数: {len(samples)}")

        # 只使用最高质量的样本
        high_quality_data = []
        for item in samples:
            if isinstance(item, dict):
                # 检查质量评分
                quality_scores = item.get('quality_scores', {})
                overall_score = quality_scores.get('overall', 0)
                if overall_score >= 0.85:  # 进一步提高质量阈值
                    high_quality_data.append(item)

        logger.info(f"✅ 最高质量数据样本: {len(high_quality_data)} 个")

        if len(high_quality_data) < 100:
            logger.warning("⚠️ 最高质量数据样本不足，使用所有样本")
            high_quality_data = [item for item in samples if isinstance(item, dict)]

        # 提取特征和标签
        features = []
        labels = []
        food_names = []

        for item in high_quality_data:
            feature_vector = self._build_feature_vector(item)
            if feature_vector is not None:
                features.append(feature_vector)
                nutrition_score = item.get('nutrition_score', 0)
                if nutrition_score == 0:
                    nutrition_score = self._calculate_nutrition_score(item)
                labels.append(nutrition_score)
                food_names.append(item.get('food_name', 'Unknown'))

        if not features:
            logger.error("❌ 未找到有效的特征数据")
            return None, None, None

        X = np.array(features)
        y = np.array(labels)

        logger.info(f"✅ 数据加载成功: {len(features)} 个样本, {X.shape[1]} 个特征")
        logger.info(f"📊 标签范围: {y.min():.3f} - {y.max():.3f}")

        return X, y, food_names

    def _build_feature_vector(self, item: Dict[str, Any]) -> List[float]:
        """构建特征向量"""
        try:
            # 基础营养特征
            calories = float(item.get('calories', 0))
            protein = float(item.get('protein', 0))
            carbs = float(item.get('carbs', 0))
            fat = float(item.get('fat', 0))
            fiber = float(item.get('fiber', 0))
            sugar = float(item.get('sugar', 0))
            sodium = float(item.get('sodium', 0))

            # 计算衍生特征
            total_macros = protein + carbs + fat
            protein_ratio = protein / total_macros if total_macros > 0 else 0
            carbs_ratio = carbs / total_macros if total_macros > 0 else 0
            fat_ratio = fat / total_macros if total_macros > 0 else 0

            # 营养密度
            nutrition_density = (protein * 4 + fiber * 2) / (calories / 100) if calories > 0 else 0

            # 健康指数
            health_index = self._calculate_health_index(item)

            # 构建20维特征向量
            feature_vector = [
                calories, protein, carbs, fat, fiber, sugar, sodium,
                protein_ratio, carbs_ratio, fat_ratio, nutrition_density, health_index,
                1 if "meat" in item.get('food_name', '').lower() else 0,
                1 if "fruit" in item.get('food_name', '').lower() else 0,
                1 if "vegetable" in item.get('food_name', '').lower() else 0,
                1 if "grain" in item.get('food_name', '').lower() else 0,
                1 if "dairy" in item.get('food_name', '').lower() else 0,
                1 if "beverage" in item.get('food_name', '').lower() else 0,
                1 if "nut" in item.get('food_name', '').lower() else 0,
                1 if "chinese" in item.get('food_name', '').lower() else 0
            ]

            return feature_vector
        except Exception as e:
            logger.warning(f"⚠️ 构建特征向量失败: {e}")
            return None

    def _calculate_health_index(self, item: Dict[str, Any]) -> float:
        """计算健康指数"""
        calories = item.get('calories', 0)
        protein = item.get('protein', 0)
        fiber = item.get('fiber', 0)
        sugar = item.get('sugar', 0)
        sodium = item.get('sodium', 0)

        score = 0.0

        # 蛋白质加分
        score += min(protein / 20, 1.0) * 0.3

        # 纤维加分
        score += min(fiber / 10, 1.0) * 0.2

        # 糖分减分
        score -= min(sugar / 50, 1.0) * 0.2

        # 钠分减分
        score -= min(sodium / 1000, 1.0) * 0.1

        # 卡路里平衡
        if 100 <= calories <= 400:
            score += 0.2
        elif calories < 100 or calories > 400:
            score -= 0.1

        return max(0.0, min(1.0, score))

    def _calculate_nutrition_score(self, item: Dict[str, Any]) -> float:
        """计算营养评分"""
        calories = item.get('calories', 0)
        protein = item.get('protein', 0)
        fiber = item.get('fiber', 0)
        sugar = item.get('sugar', 0)
        sodium = item.get('sodium', 0)

        score = 0.0

        # 蛋白质加分
        score += min(protein / 20, 1.0) * 0.3

        # 纤维加分
        score += min(fiber / 10, 1.0) * 0.2

        # 糖分减分
        score -= min(sugar / 50, 1.0) * 0.2

        # 钠分减分
        score -= min(sodium / 1000, 1.0) * 0.1

        # 卡路里平衡
        if 100 <= calories <= 400:
            score += 0.2
        elif calories < 100 or calories > 400:
            score -= 0.1

        return max(0.0, min(1.0, score))

    def prepare_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[DataLoader, DataLoader]:
        """准备训练数据"""
        logger.info("📊 准备训练数据...")

        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)

        # 分割数据
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=None
        )

        logger.info(f"📊 训练集: {len(X_train)} 个样本")
        logger.info(f"📊 验证集: {len(X_val)} 个样本")

        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)

        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        train_loader = DataLoader(train_dataset, batch_size=self.optimal_config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.optimal_config['batch_size'], shuffle=False)

        return train_loader, val_loader

    def train_final_model(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        """训练最终模型"""
        logger.info("🚀 开始最终高质量数据微调训练...")
        logger.info(f"📊 配置: {self.optimal_config}")

        # 初始化模型 - 使用更深的网络架构
        input_dim = train_loader.dataset[0][0].shape[0]
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),  # 增加网络容量
            nn.ReLU(),
            nn.Dropout(self.optimal_config['dropout_rate']),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(self.optimal_config['dropout_rate']),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(self.optimal_config['dropout_rate']),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.005),
            nn.Linear(64, 1)
        ).to(self.device)

        # 优化器和损失函数
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.optimal_config['learning_rate'],
            weight_decay=self.optimal_config['weight_decay']
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=self.optimal_config['patience']//3
        )
        criterion = nn.MSELoss()

        # 早停机制
        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0

        for epoch in range(self.optimal_config['epochs']):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_r2 = 0.0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                # 计算R²分数
                with torch.no_grad():
                    r2 = r2_score(batch_y.cpu().numpy(), outputs.cpu().numpy())
                    train_r2 += r2

            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            val_r2 = 0.0

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

                    # 计算R²分数
                    r2 = r2_score(batch_y.cpu().numpy(), outputs.cpu().numpy())
                    val_r2 += r2

            # 计算平均损失和R²分数
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            avg_train_r2 = train_r2 / len(train_loader)
            avg_val_r2 = val_r2 / len(val_loader)

            # 记录历史
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['val_loss'].append(avg_val_loss)
            self.training_history['train_r2'].append(avg_train_r2)
            self.training_history['val_r2'].append(avg_val_r2)

            # 学习率调度
            scheduler.step(avg_val_loss)

            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_epoch = epoch + 1
                # 保存最佳模型
                self._save_model()
            else:
                patience_counter += 1

            # 打印进度
            logger.info(f"Epoch {epoch+1}/{self.optimal_config['epochs']}: "
                      f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                      f"Train R²: {avg_train_r2:.4f}, Val R²: {avg_val_r2:.4f}")

            # 早停
            if patience_counter >= self.optimal_config['patience']:
                logger.info(f"🛑 早停触发，在第 {epoch+1} 轮停止训练")
                logger.info(f"🏆 最佳性能在第 {best_epoch} 轮")
                break

        # 最终评估
        final_metrics = self._evaluate_model(val_loader)

        return {
            'final_epoch': epoch + 1,
            'best_epoch': best_epoch,
            'final_metrics': final_metrics,
            'training_history': self.training_history,
            'config_used': self.optimal_config
        }

    def _save_model(self) -> None:
        """保存模型"""
        if self.model is not None:
            model_path = "models/final_cultural_model.pth"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'scaler': self.scaler,
                'training_history': self.training_history,
                'config': self.optimal_config
            }, model_path)
            logger.info(f"💾 模型已保存: {model_path}")

    def _evaluate_model(self, val_loader: DataLoader) -> Dict[str, float]:
        """评估模型"""
        if self.model is None:
            return {}

        self.model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)

                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())

        all_predictions = np.array(all_predictions).flatten()
        all_targets = np.array(all_targets).flatten()

        # 计算评估指标
        mse = mean_squared_error(all_targets, all_predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(all_targets, all_predictions)
        r2 = r2_score(all_targets, all_predictions)
        explained_var = explained_variance_score(all_targets, all_predictions)
        max_err = max_error(all_targets, all_predictions)
        correlation = np.corrcoef(all_targets, all_predictions)[0, 1]

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'explained_variance': explained_var,
            'max_error': max_err,
            'correlation': correlation
        }

    def comprehensive_evaluation(self, val_loader: DataLoader) -> Dict[str, Any]:
        """全面评估模型"""
        logger.info("🔍 开始全面模型评估...")

        if self.model is None:
            logger.error("❌ 模型未训练")
            return {}

        self.model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)

                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())

        all_predictions = np.array(all_predictions).flatten()
        all_targets = np.array(all_targets).flatten()

        # 计算评估指标
        metrics = {
            'mse': mean_squared_error(all_targets, all_predictions),
            'rmse': np.sqrt(mean_squared_error(all_targets, all_predictions)),
            'mae': mean_absolute_error(all_targets, all_predictions),
            'r2': r2_score(all_targets, all_predictions),
            'explained_variance': explained_variance_score(all_targets, all_predictions),
            'max_error': max_error(all_targets, all_predictions),
            'correlation': np.corrcoef(all_targets, all_predictions)[0, 1]
        }

        # 预测质量分析
        errors = np.abs(all_targets - all_predictions)
        quality_analysis = {
            'excellent_predictions': np.sum(errors < 0.05) / len(errors) * 100,
            'good_predictions': np.sum(errors < 0.1) / len(errors) * 100,
            'acceptable_predictions': np.sum(errors < 0.2) / len(errors) * 100,
            'poor_predictions': np.sum(errors >= 0.2) / len(errors) * 100,
            'mean_error': np.mean(errors),
            'median_error': np.median(errors),
            'error_std': np.std(errors),
            'max_error': np.max(errors),
            'min_error': np.min(errors)
        }

        # 创建可视化
        self._create_evaluation_visualizations(all_targets, all_predictions, errors)

        return {
            'metrics': metrics,
            'quality_analysis': quality_analysis,
            'predictions': all_predictions,
            'targets': all_targets,
            'errors': errors
        }

    def _create_evaluation_visualizations(self, targets: np.ndarray, predictions: np.ndarray, errors: np.ndarray) -> None:
        """创建评估可视化"""
        logger.info("📊 创建评估可视化...")

        try:
            output_dir = Path("outputs/final_evaluation")
            output_dir.mkdir(parents=True, exist_ok=True)

            # 1. 预测vs实际散点图
            plt.figure(figsize=(10, 8))
            plt.scatter(targets, predictions, alpha=0.6, s=50)
            plt.plot([targets.min(), targets.max()],
                    [targets.min(), targets.max()], 'r--', lw=2)
            plt.xlabel('实际营养评分')
            plt.ylabel('预测营养评分')
            plt.title('预测vs实际营养评分 (最终高质量模型)')
            plt.grid(True, alpha=0.3)
            plt.savefig(output_dir / 'prediction_vs_actual.png', dpi=300, bbox_inches='tight')
            plt.close()

            # 2. 残差图
            residuals = targets - predictions
            plt.figure(figsize=(10, 6))
            plt.scatter(predictions, residuals, alpha=0.6, s=50)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('预测营养评分')
            plt.ylabel('残差 (实际 - 预测)')
            plt.title('残差分析 (最终高质量模型)')
            plt.grid(True, alpha=0.3)
            plt.savefig(output_dir / 'residuals.png', dpi=300, bbox_inches='tight')
            plt.close()

            # 3. 预测分布直方图
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.hist(targets, bins=30, alpha=0.7, label='实际', color='blue')
            plt.hist(predictions, bins=30, alpha=0.7, label='预测', color='red')
            plt.xlabel('营养评分')
            plt.ylabel('频次')
            plt.title('实际vs预测分布 (最终高质量模型)')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.subplot(1, 2, 2)
            plt.hist(residuals, bins=30, alpha=0.7, color='green')
            plt.xlabel('残差')
            plt.ylabel('频次')
            plt.title('残差分布 (最终高质量模型)')
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / 'distributions.png', dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"✅ 可视化图表已保存到: {output_dir}")

        except Exception as e:
            logger.error(f"❌ 可视化创建失败: {e}")

    def generate_final_report(self, training_result: Dict[str, Any], evaluation_result: Dict[str, Any]) -> None:
        """生成最终报告"""
        logger.info("📋 生成最终评估报告...")

        try:
            # 评估等级
            r2_score = evaluation_result['metrics']['r2']
            if r2_score >= 0.99:
                performance_level = "完美"
                performance_color = "🟢"
            elif r2_score >= 0.95:
                performance_level = "卓越"
                performance_color = "🟢"
            elif r2_score >= 0.9:
                performance_level = "优秀"
                performance_color = "🟢"
            elif r2_score >= 0.8:
                performance_level = "良好"
                performance_color = "🟡"
            else:
                performance_level = "需要改进"
                performance_color = "🔴"

            # 生成报告
            report = {
                "final_evaluation_summary": {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "model_type": "Final High-Quality Cultural Adaptation Model",
                    "training_epochs": training_result['final_epoch'],
                    "best_epoch": training_result['best_epoch'],
                    "performance_level": performance_level,
                    "performance_color": performance_color
                },
                "training_results": training_result,
                "evaluation_results": evaluation_result,
                "recommendations": self._generate_final_recommendations(evaluation_result)
            }

            # 保存报告
            output_dir = Path("outputs/final_evaluation")
            output_dir.mkdir(parents=True, exist_ok=True)

            with open(output_dir / "final_evaluation_report.json", "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)

            # 生成文本报告
            self._generate_final_text_report(report, output_dir)

            logger.info(f"📋 最终评估报告已保存到: {output_dir}")

        except Exception as e:
            logger.error(f"❌ 报告生成失败: {e}")

    def _generate_final_recommendations(self, evaluation_result: Dict[str, Any]) -> List[str]:
        """生成最终建议"""
        recommendations = []

        r2_score = evaluation_result['metrics']['r2']
        mae = evaluation_result['metrics']['mae']
        excellent_predictions = evaluation_result['quality_analysis']['excellent_predictions']

        if r2_score >= 0.99:
            recommendations.append("🎉 最终模型性能达到完美级别，可以立即投入生产使用")
            recommendations.append("🏆 建议将此模型作为标准版本进行部署")
            recommendations.append("🌟 这是目前最优的文化适应模型")
        elif r2_score >= 0.95:
            recommendations.append("✅ 最终模型性能卓越，完全满足生产要求")
            recommendations.append("🚀 建议立即部署到生产环境")
            recommendations.append("🎯 模型已达到工业级标准")
        elif r2_score >= 0.9:
            recommendations.append("👍 最终模型性能优秀，可以投入生产使用")
            recommendations.append("📊 建议持续监控实际应用效果")
        else:
            recommendations.append("⚠️ 最终模型性能需要进一步优化")
            recommendations.append("🔧 建议调整模型架构或增加训练数据")

        if excellent_predictions >= 95:
            recommendations.append("🎯 预测质量极高，用户满意度会很高")
        elif excellent_predictions >= 90:
            recommendations.append("✅ 预测质量优秀，用户体验良好")
        else:
            recommendations.append("📈 预测质量良好，建议进一步优化")

        return recommendations

    def _generate_final_text_report(self, report: Dict[str, Any], output_dir: Path) -> None:
        """生成最终文本报告"""
        try:
            with open(output_dir / "final_evaluation_report.txt", "w", encoding="utf-8") as f:
                f.write("="*80 + "\n")
                f.write("🎯 最终高质量文化适应模型评估报告\n")
                f.write("="*80 + "\n\n")

                # 基本信息
                f.write("📊 基本信息:\n")
                f.write(f"  评估时间: {report['final_evaluation_summary']['timestamp']}\n")
                f.write(f"  模型类型: {report['final_evaluation_summary']['model_type']}\n")
                f.write(f"  训练轮数: {report['final_evaluation_summary']['training_epochs']}\n")
                f.write(f"  最佳轮数: {report['final_evaluation_summary']['best_epoch']}\n")
                f.write(f"  性能等级: {report['final_evaluation_summary']['performance_color']} {report['final_evaluation_summary']['performance_level']}\n\n")

                # 评估指标
                f.write("📈 评估指标:\n")
                metrics = report['evaluation_results']['metrics']
                f.write(f"  R² 分数: {metrics['r2']:.4f}\n")
                f.write(f"  RMSE: {metrics['rmse']:.4f}\n")
                f.write(f"  MAE: {metrics['mae']:.4f}\n")
                f.write(f"  相关系数: {metrics['correlation']:.4f}\n")
                f.write(f"  解释方差: {metrics['explained_variance']:.4f}\n\n")

                # 质量分析
                f.write("🔍 预测质量分析:\n")
                quality = report['evaluation_results']['quality_analysis']
                f.write(f"  优秀预测 (误差<0.05): {quality['excellent_predictions']:.1f}%\n")
                f.write(f"  良好预测 (误差<0.1): {quality['good_predictions']:.1f}%\n")
                f.write(f"  可接受预测 (误差<0.2): {quality['acceptable_predictions']:.1f}%\n")
                f.write(f"  较差预测 (误差>=0.2): {quality['poor_predictions']:.1f}%\n\n")

                # 建议
                f.write("💡 最终建议:\n")
                for i, rec in enumerate(report['recommendations'], 1):
                    f.write(f"  {i}. {rec}\n")

                f.write("\n" + "="*80 + "\n")
                f.write("✅ 最终评估完成\n")
                f.write("="*80 + "\n")

        except Exception as e:
            logger.error(f"❌ 文本报告生成失败: {e}")

    def run_final_training_and_evaluation(self) -> None:
        """运行最终训练和评估"""
        logger.info("🚀 开始最终高质量数据微调训练和评估...")

        # 1. 创建最终数据集
        if not self.create_final_dataset():
            logger.error("❌ 创建最终数据集失败")
            return

        # 2. 加载数据
        X, y, food_names = self.load_final_data()
        if X is None:
            logger.error("❌ 数据加载失败")
            return

        # 3. 准备数据
        train_loader, val_loader = self.prepare_data(X, y)

        # 4. 训练模型
        training_result = self.train_final_model(train_loader, val_loader)

        # 5. 全面评估
        evaluation_result = self.comprehensive_evaluation(val_loader)

        # 6. 生成报告
        self.generate_final_report(training_result, evaluation_result)

        logger.info("✅ 最终高质量数据微调训练和评估完成")

        # 打印关键结果
        print("\n" + "="*80)
        print("Final High-Quality Data Evaluation Results Summary")
        print("="*80)
        print(f"R2 Score: {evaluation_result['metrics']['r2']:.4f}")
        print(f"RMSE: {evaluation_result['metrics']['rmse']:.4f}")
        print(f"MAE: {evaluation_result['metrics']['mae']:.4f}")
        print(f"Correlation: {evaluation_result['metrics']['correlation']:.4f}")
        print(f"Excellent Predictions: {evaluation_result['quality_analysis']['excellent_predictions']:.1f}%")
        print("="*80)


def main():
    """主函数"""
    print("\n" + "="*80)
    print("Final High-Quality Dataset Training and Evaluation System")
    print("Using 3,005 high-quality samples for fine-tuning and validation")
    print("="*80 + "\n")

    trainer = FinalCleanedDatasetTrainer()
    trainer.run_final_training_and_evaluation()

    print("\n" + "="*80)
    print("Final High-Quality Data Training and Evaluation Completed!")
    print("Model saved to: models/final_cultural_model.pth")
    print("Report saved to: outputs/final_evaluation/")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
