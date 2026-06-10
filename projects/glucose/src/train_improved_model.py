#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用改进型模型架构训练文化适配模型
包含更先进的模型架构、优化策略和数据增强技术
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from tqdm import tqdm
import copy
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 导入自定义模块
from improved_model_architecture import ImprovedCulturalModel
from utils.data_quality_enhancer import DataQualityEnhancer
from utils.enhanced_evaluation import EnhancedEvaluator, EvaluationMetrics
from utils.training_optimizer import TrainingOptimizer

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImprovedModelTrainer:
    """改进型模型训练器"""

    def __init__(self, config: Dict[str, Any]):
        """初始化训练器"""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 创建输出目录
        self.output_dir = Path(config.get("output_dir", "outputs/improved_model"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 数据增强器
        self.data_enhancer = DataQualityEnhancer({
            "augmentation_factor": config["data"].get("augmentation_factor", 3.0),
            "duplicate_threshold": 0.95,
            "missing_threshold": 0.1,
            "outlier_threshold": 3.0,
            "min_samples_per_class": 10,
            "validation_split": 0.2,
            "random_seed": 42
        })

        # 评估器
        self.evaluator = EnhancedEvaluator()

        # 训练优化器
        self.optimizer_manager = TrainingOptimizer(config["optimization"])

        # 模型和数据
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.region_to_id = {}
        self.cuisine_to_id = {}
        self.num_regions = 0
        self.num_cuisines = 0

    def load_data(self, data_path: str) -> List:
        """加载数据"""
        logger.info(f"从 {data_path} 加载数据...")

        with open(data_path, 'r', encoding='utf-8') as f:
            user_samples = json.load(f)

        logger.info(f"已加载 {len(user_samples)} 个用户偏好样本")

        # 转换为CulturalDataPoint格式
        from enhanced_stage1_cultural_adaptation import CulturalDataPoint, CulturalPreference

        data_points = []
        for sample in user_samples:
            preferences = CulturalPreference(
                spice_level=sample.get('spice_tolerance', 0.5) * 10,
                sweet_preference=sample.get('sweet_preference', 0.5) * 10,
                salt_preference=sample.get('salt_preference', 0.5) * 10,
                sour_preference=sample.get('sour_preference', 0.5) * 10,
                bitter_preference=sample.get('bitter_preference', 0.5) * 10,
                umami_preference=sample.get('umami_preference', 0.5) * 10,
                texture_preference=sample.get('preferred_cooking', '脆嫩'),
                cooking_method_preference=sample.get('preferred_cooking', '炒'),
                meal_time_preference=sample.get('meal_time', '午餐'),
                dietary_restrictions=sample.get('dietary_restrictions', [])
            )

            data_point = CulturalDataPoint(
                user_id=sample.get('user_id', f"user_{len(data_points):04d}"),
                region=sample.get('region', '华北'),
                cuisine_type=sample.get('cuisine_type', '川菜'),
                preferences=preferences,
                acceptance_score=sample.get('acceptance_score', 0.7),
                timestamp=sample.get('timestamp', '2025-10-21T12:00:00'),
                context=sample.get('context', {})
            )

            data_points.append(data_point)

        return data_points

    def prepare_data(self, data_points: List) -> None:
        """准备训练数据"""
        logger.info("📊 准备训练数据...")

        # 数据质量增强
        enhanced_data, quality_metrics = self.data_enhancer.enhance_cultural_data(data_points)

        # 输出数据质量报告
        logger.info(f"数据质量评分: {quality_metrics.overall_score:.3f}")
        for rec in quality_metrics.recommendations:
            logger.info(f"建议: {rec}")

        # 创建训练和验证集
        train_data, val_data = self.data_enhancer.create_stratified_splits(enhanced_data)

        # 分析数据特征
        regions = set(dp.region for dp in enhanced_data)
        cuisines = set(dp.cuisine_type for dp in enhanced_data)
        logger.info(f"地区数量: {len(regions)}, 菜系数量: {len(cuisines)}")

        # 创建映射
        self.region_to_id = {region: i for i, region in enumerate(regions)}
        self.cuisine_to_id = {cuisine: i for i, cuisine in enumerate(cuisines)}

        # 创建数据加载器
        self.train_loader, self.val_loader = self._create_data_loaders(
            train_data, val_data, self.region_to_id, self.cuisine_to_id
        )

        logger.info(f"训练集: {len(train_data)}, 验证集: {len(val_data)}")

        # 保存地区和菜系数量
        self.num_regions = len(regions)
        self.num_cuisines = len(cuisines)

    def _create_data_loaders(self, train_data, val_data, region_to_id, cuisine_to_id):
        """创建数据加载器"""
        import torch
        from torch.utils.data import DataLoader, Dataset

        class CulturalDataset(Dataset):
            def __init__(self, data_points, region_map, cuisine_map):
                self.data_points = data_points
                self.region_map = region_map
                self.cuisine_map = cuisine_map

            def __len__(self):
                return len(self.data_points)

            def __getitem__(self, idx):
                dp = self.data_points[idx]

                # 区域和菜系ID
                region_id = self.region_map.get(dp.region, 0)
                cuisine_id = self.cuisine_map.get(dp.cuisine_type, 0)

                # 偏好特征
                preferences = [
                    dp.preferences.spice_level / 10.0,
                    dp.preferences.sweet_preference / 10.0,
                    dp.preferences.salt_preference / 10.0,
                    dp.preferences.sour_preference / 10.0,
                    dp.preferences.bitter_preference / 10.0,
                    dp.preferences.umami_preference / 10.0,
                    # 添加更多特征...
                    0.5,  # 占位符
                    0.5,  # 占位符
                    0.5,  # 占位符
                    0.5,  # 占位符
                ]

                # 上下文特征
                context = [0.5, 0.5, 0.5, 0.5, 0.5]  # 占位符

                # 目标值
                acceptance = dp.acceptance_score

                return {
                    "region_id": torch.tensor(region_id, dtype=torch.long),
                    "cuisine_id": torch.tensor(cuisine_id, dtype=torch.long),
                    "preferences": torch.tensor(preferences, dtype=torch.float),
                    "context": torch.tensor(context, dtype=torch.float),
                    "acceptance": torch.tensor(acceptance, dtype=torch.float)
                }

        # 创建数据集
        train_dataset = CulturalDataset(train_data, region_to_id, cuisine_to_id)
        val_dataset = CulturalDataset(val_data, region_to_id, cuisine_to_id)

        # 创建数据加载器
        batch_size = self.config["training"].get("batch_size", 64)

        # 动态调整批次大小
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # GB
            if gpu_mem < 4:  # 小于4GB的GPU
                batch_size = min(batch_size, 32)
            elif gpu_mem < 8:  # 小于8GB的GPU
                batch_size = min(batch_size, 64)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )

        return train_loader, val_loader

    def build_model(self) -> None:
        """构建改进型模型"""
        logger.info("🧠 构建改进型文化适配模型...")

        model_config = self.config["model"]

        # 创建改进型文化适配模型
        self.model = ImprovedCulturalModel(
            num_regions=self.num_regions,
            num_cuisines=self.num_cuisines,
            feature_dim=model_config.get("cultural_feature_dim", 256),
            hidden_dim=model_config.get("hidden_dim", 512),
            lora_rank=model_config.get("lora_rank", 64),
            dropout=model_config.get("dropout", 0.2),
            use_batch_norm=model_config.get("use_batch_norm", True),
            use_residual=model_config.get("use_residual", True),
            num_layers=model_config.get("num_layers", 4),
            num_heads=model_config.get("num_heads", 8),
            activation=model_config.get("activation", "gelu"),
            use_attention=model_config.get("use_attention", True)
        )

        # 移动模型到设备
        self.model = self.model.to(self.device)

        # 输出模型信息
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")

    def train(self) -> EvaluationMetrics:
        """执行训练过程"""
        logger.info("📈 开始训练模型...")

        # 训练配置
        train_config = self.config["training"]
        num_epochs = train_config.get("num_epochs", 200)
        patience = train_config.get("patience", 30)
        min_delta = train_config.get("min_delta", 1e-6)

        # 自定义早停实现
        best_score = 0.0
        wait = 0
        best_model_state = None

        # 创建优化器和调度器
        optimizer = self.optimizer_manager.create_optimizer(self.model)
        scheduler = self.optimizer_manager.create_scheduler(optimizer, num_epochs)

        # 创建损失函数 - 使用BCEWithLogitsLoss以支持混合精度训练
        criterion = nn.BCEWithLogitsLoss()

        # 梯度裁剪
        gradient_clipper = None
        if self.config["optimization"].get("regularization", {}).get("gradient_clipping", {}).get("enabled", True):
            max_norm = self.config["optimization"].get("regularization", {}).get("gradient_clipping", {}).get("max_norm", 1.0)
            from utils.training_optimizer import GradientClipping
            gradient_clipper = GradientClipping(max_norm=max_norm)

        # 混合精度训练
        scaler = torch.cuda.amp.GradScaler() if self.config["optimization"].get("mixed_precision", {}).get("enabled", True) and torch.cuda.is_available() else None

        # 模型EMA
        model_ema = None
        if self.config["optimization"].get("model_ema", {}).get("enabled", True):
            from utils.training_optimizer import ModelEMA
            decay = self.config["optimization"].get("model_ema", {}).get("decay", 0.999)
            model_ema = ModelEMA(self.model, decay=decay)

        # 学习率预热
        warmup_steps = self.config["optimization"].get("warmup", {}).get("steps", 0)
        warmup_base_lr = self.config["optimization"].get("warmup", {}).get("base_lr", 1e-6)

        logger.info("🏃 开始训练循环...")

        # 训练循环
        for epoch in range(num_epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0

            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                # 获取数据
                region_ids = batch["region_id"].to(self.device)
                cuisine_ids = batch["cuisine_id"].to(self.device)
                preferences = batch["preferences"].to(self.device)
                context = batch["context"].to(self.device)
                targets = batch["acceptance"].to(self.device)

                # 清零梯度
                optimizer.zero_grad()

                # 学习率预热
                if epoch < warmup_steps:
                    lr_scale = min(1.0, (epoch + 1) / warmup_steps)
                    for pg in optimizer.param_groups:
                        pg['lr'] = lr_scale * (pg['initial_lr'] - warmup_base_lr) + warmup_base_lr

                # 混合精度训练
                if scaler:
                    with torch.cuda.amp.autocast():
                        # 前向传播
                        outputs = self.model(region_ids, cuisine_ids, preferences, context)
                        loss = criterion(outputs, targets)

                    # 反向传播
                    scaler.scale(loss).backward()

                    # 梯度裁剪
                    if gradient_clipper:
                        scaler.unscale_(optimizer)
                        gradient_clipper(self.model)

                    # 更新参数
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # 前向传播
                    outputs = self.model(region_ids, cuisine_ids, preferences, context)
                    loss = criterion(outputs, targets)

                    # 反向传播
                    loss.backward()

                    # 梯度裁剪
                    if gradient_clipper:
                        gradient_clipper(self.model)

                    # 更新参数
                    optimizer.step()

                # 更新EMA
                if model_ema:
                    model_ema.update()

                train_loss += loss.item()

            # 计算平均训练损失
            train_loss /= len(self.train_loader)

            # 验证阶段
            self.model.eval()
            val_predictions = []
            val_targets = []

            with torch.no_grad():
                for batch in self.val_loader:
                    # 获取数据
                    region_ids = batch["region_id"].to(self.device)
                    cuisine_ids = batch["cuisine_id"].to(self.device)
                    preferences = batch["preferences"].to(self.device)
                    context = batch["context"].to(self.device)
                    targets = batch["acceptance"].to(self.device)

                    # 前向传播
                    outputs = self.model(region_ids, cuisine_ids, preferences, context)

                    # 收集预测和目标
                    val_predictions.extend(outputs.cpu().numpy())
                    val_targets.extend(targets.cpu().numpy())

            # 评估验证集性能
            val_metrics = EvaluationMetrics()
            val_metrics.mae = mean_absolute_error(val_targets, val_predictions)
            val_metrics.mse = mean_squared_error(val_targets, val_predictions)
            val_metrics.rmse = np.sqrt(val_metrics.mse)
            val_metrics.r2 = r2_score(val_targets, val_predictions)
            val_metrics.cultural_consistency = 0.5  # 占位值
            val_metrics.overall_score = 0.5 - val_metrics.mae  # 简化的评分

            # 获取当前学习率
            current_lr = optimizer.param_groups[0]['lr']

            # 输出训练信息
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_loss:.6f}, "
                f"Val Score: {val_metrics.overall_score:.6f}, "
                f"Val MAE: {val_metrics.mae:.6f}, "
                f"Val R²: {val_metrics.r2:.6f}, "
                f"LR: {current_lr:.2e}"
            )

            # 检查是否为最佳模型
            if val_metrics.overall_score > best_score:
                best_score = val_metrics.overall_score
                logger.info(f"🎯 新的最佳模型! 验证评分: {best_score:.6f}")

                # 保存最佳模型状态
                best_model_state = copy.deepcopy(self.model.state_dict())

                # 保存最佳模型
                self._save_model()

                # 重置早停计数器
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    logger.info(f"模型收敛，在第 {epoch+1} 轮停止训练")
                    # 恢复最佳模型
                    if best_model_state is not None:
                        self.model.load_state_dict(best_model_state)
                    break

            # 更新学习率
            if scheduler:
                scheduler.step()

        # 训练结束，保存最终报告
        self._save_final_report(val_metrics)

        return val_metrics

    def _save_model(self) -> None:
        """保存模型"""
        # 创建输出目录
        output_dir = self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存完整模型
        torch.save(
            self.model.state_dict(),
            output_dir / "improved_cultural_model.pt"
        )

        # 保存LoRA权重
        torch.save(
            self.model.lora_layer.state_dict(),
            output_dir / "improved_cultural_lora_weights.pt"
        )

        logger.info(f"✅ 改进型模型已保存: {output_dir / 'improved_cultural_model.pt'}")
        logger.info(f"✅ LoRA权重已保存: {output_dir / 'improved_cultural_lora_weights.pt'}")

    def _save_final_report(self, metrics: EvaluationMetrics) -> None:
        """保存最终报告"""
        # 创建输出目录
        output_dir = self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存评估报告
        report = {
            "metrics": {
                "mae": float(metrics.mae),
                "mse": float(metrics.mse),
                "rmse": float(metrics.rmse),
                "r2": float(metrics.r2),
                "cultural_consistency": float(metrics.cultural_consistency),
                "overall_score": float(metrics.overall_score)
            },
            "timestamp": datetime.now().isoformat(),
            "config_name": "improved_model_training"
        }

        with open(output_dir / "improved_model_evaluation_report.json", "w") as f:
            json.dump(report, f, indent=2)

        # 创建最终训练报告
        training_report = {
            "model_info": {
                "hidden_dim": self.config["model"].get("hidden_dim", 512),
                "feature_dim": self.config["model"].get("cultural_feature_dim", 256),
                "lora_rank": self.config["model"].get("lora_rank", 64),
                "num_layers": self.config["model"].get("num_layers", 4),
                "num_heads": self.config["model"].get("num_heads", 8),
                "use_attention": self.config["model"].get("use_attention", True)
            },
            "training_info": {
                "epochs": self.config["training"].get("num_epochs", 200),
                "batch_size": self.config["training"].get("batch_size", 64),
                "learning_rate": self.config["training"].get("learning_rate", 0.001)
            },
            "performance": {
                "mae": float(metrics.mae),
                "mse": float(metrics.mse),
                "rmse": float(metrics.rmse),
                "r2": float(metrics.r2),
                "cultural_consistency": float(metrics.cultural_consistency),
                "overall_score": float(metrics.overall_score)
            },
            "data_quality": {
                "score": float(self.data_enhancer.quality_metrics.overall_score),
                "completeness": float(self.data_enhancer.quality_metrics.completeness),
                "uniqueness": float(self.data_enhancer.quality_metrics.uniqueness),
                "consistency": float(self.data_enhancer.quality_metrics.consistency),
                "validity": float(self.data_enhancer.quality_metrics.validity),
                "balance": float(self.data_enhancer.quality_metrics.balance)
            },
            "timestamp": datetime.now().isoformat()
        }

        # 保存训练报告
        with open(output_dir / "improved_model_training_report.json", "w") as f:
            json.dump(training_report, f, indent=2)

        logger.info(f"📋 最终报告已保存: {output_dir / 'improved_model_training_report.json'}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="使用改进型模型架构训练文化适配模型")
    parser.add_argument("--data-path", type=str, default="data/real_data/user_preferences.json", help="数据路径")
    parser.add_argument("--config-path", type=str, default="configs/improved_model_config.yaml", help="配置文件路径")
    parser.add_argument("--output-dir", type=str, default="outputs/improved_model", help="输出目录")
    args = parser.parse_args()

    print("\n" + "="*80)
    print("🚀 使用改进型模型架构训练文化适配模型")
    print("="*80 + "\n")

    # 加载配置
    try:
        from utils.config_loader import ConfigLoader
        config_loader = ConfigLoader(args.config_path)
        config = config_loader.get_config("improved_model")
        logger.info(f"✅ 配置已加载: {args.config_path}")
    except Exception as e:
        logger.warning(f"⚠️ 配置加载失败: {e}，使用默认配置")
        # 默认配置
        config = {
            "data": {
                "augmentation_factor": 3.0
            },
            "model": {
                "cultural_feature_dim": 256,
                "hidden_dim": 512,
                "lora_rank": 64,
                "dropout": 0.2,
                "use_batch_norm": True,
                "use_residual": True,
                "num_layers": 4,
                "num_heads": 8,
                "activation": "gelu",
                "use_attention": True
            },
            "training": {
                "num_epochs": 200,
                "batch_size": 64,
                "learning_rate": 0.001,
                "patience": 30,
                "min_delta": 1e-6
            },
            "optimization": {
                "optimizer": {
                    "type": "AdamW",
                    "weight_decay": 0.05,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8
                },
                "scheduler": {
                    "type": "CosineAnnealingWarmRestarts",
                    "T_0": 20,
                    "T_mult": 2,
                    "eta_min": 1e-6
                },
                "mixed_precision": {
                    "enabled": True
                },
                "regularization": {
                    "gradient_clipping": {
                        "enabled": True,
                        "max_norm": 1.0
                    }
                },
                "model_ema": {
                    "enabled": True,
                    "decay": 0.999
                },
                "warmup": {
                    "enabled": True,
                    "steps": 10,
                    "base_lr": 1e-6
                }
            }
        }

    # 设置输出目录
    config["output_dir"] = args.output_dir

    # 创建训练器
    trainer = ImprovedModelTrainer(config)

    # 加载数据
    data_points = trainer.load_data(args.data_path)

    # 准备数据
    trainer.prepare_data(data_points)

    # 构建模型
    trainer.build_model()

    # 训练模型
    metrics = trainer.train()

    # 输出最终结果
    logger.info("🎉 训练完成！")
    logger.info("📊 最终结果:")
    logger.info(f"   MAE: {metrics.mae:.6f}")
    logger.info(f"   R²: {metrics.r2:.6f}")
    logger.info(f"   综合评分: {metrics.overall_score:.6f}")

    # 性能评级
    if metrics.overall_score > 0.8:
        grade = "优秀 (A)"
    elif metrics.overall_score > 0.7:
        grade = "良好 (B)"
    elif metrics.overall_score > 0.6:
        grade = "中等 (C)"
    elif metrics.overall_score > 0.4:
        grade = "需改进 (D)"
    else:
        grade = "不足 (F)"

    logger.info(f"   性能评级: 📈 {grade}")

    print("\n" + "="*80)
    print("✅ 训练完成！")
    print(f"📁 输出目录: {trainer.output_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
