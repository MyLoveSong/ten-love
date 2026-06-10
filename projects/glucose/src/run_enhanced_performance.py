#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强性能训练脚本
使用优化后的配置和模型架构进行训练
"""

import os
import sys
import logging
import argparse
import torch
import numpy as np
import copy
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.config_loader import ConfigLoader
from enhanced_stage1_cultural_adaptation import EnhancedCulturalDataSource
from enhanced_model_architecture import EnhancedCulturalModel
from utils.data_quality_enhancer import DataQualityEnhancer
from utils.enhanced_evaluation import EnhancedEvaluator, EarlyStoppingCallback, EvaluationMetrics
from utils.training_optimizer import TrainingOptimizer

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_banner():
    """打印启动横幅"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                  🚀 增强性能文化适配训练                      ║
    ║                                                              ║
    ║  优化原则: "先调数据，再调模型，监控验证，正则跟上，轮数最后看"  ║
    ║                                                              ║
    ║  ✅ 深度模型架构    ✅ 高级特征提取                           ║
    ║  ✅ 多层次融合      ✅ RSLoRA适配                            ║
    ║  ✅ 增强正则化      ✅ 模型EMA                               ║
    ║  ✅ 高级学习率调度  ✅ 优化超参数                             ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_environment():
    """检查训练环境"""
    logger.info("🔍 验证训练环境...")

    # 检查PyTorch版本
    import torch
    logger.info(f"✅ PyTorch版本: {torch.__version__}")

    # 检查CUDA可用性
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"✅ GPU: {device} ({memory:.1f}GB)")
    else:
        logger.warning("⚠️ 未检测到GPU，将使用CPU训练")

    # 检查数据可用性
    data_path = Path("data/cultural_enhanced")
    if data_path.exists() and any(data_path.glob("*.json")):
        logger.info("✅ 找到真实数据")
    else:
        logger.warning("⚠️ 未发现真实数据，将使用模拟数据")

    logger.info("✅ 环境验证通过")


class EnhancedTrainer:
    """增强性能训练器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 数据源
        self.data_source = EnhancedCulturalDataSource(
            use_real_data=config["data"].get("use_real_data", True)
        )

        # 数据质量增强器
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

        # 模型和数据加载器
        self.model = None
        self.train_loader = None
        self.val_loader = None

    def prepare_data(self, sample_size: int):
        """准备训练数据"""
        logger.info("📊 步骤1: 数据质量与规模优化")

        # 获取原始数据
        logger.info("获取原始数据...")
        raw_data = self.data_source.fetch_cultural_data(sample_size)

        # 数据质量增强
        logger.info("执行数据质量增强...")
        enhanced_data, quality_metrics = self.data_enhancer.enhance_cultural_data(raw_data)

        # 输出数据质量报告
        logger.info(f"数据质量评分: {quality_metrics.overall_score:.3f}")
        for rec in quality_metrics.recommendations:
            logger.info(f"建议: {rec}")

        # 创建数据加载器
        train_data, val_data = self.data_enhancer.create_stratified_splits(enhanced_data)

        # 分析数据特征
        regions = set(dp.region for dp in enhanced_data)
        cuisines = set(dp.cuisine_type for dp in enhanced_data)
        logger.info(f"地区数量: {len(regions)}, 菜系数量: {len(cuisines)}")

        # 创建映射
        region_to_id = {region: i for i, region in enumerate(regions)}
        cuisine_to_id = {cuisine: i for i, cuisine in enumerate(cuisines)}

        # 创建训练和验证数据加载器
        self.train_loader, self.val_loader = self._create_data_loaders(
            train_data, val_data, region_to_id, cuisine_to_id
        )

        logger.info(f"训练集: {len(train_data)}, 验证集: {len(val_data)}")

        return len(regions), len(cuisines)

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
                region_id = self.region_map[dp.region]
                cuisine_id = self.cuisine_map[dp.cuisine_type]

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
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

        return train_loader, val_loader

    def build_model(self, num_regions: int, num_cuisines: int):
        """构建增强型模型"""
        logger.info("🧠 步骤2: 模型设计与任务匹配")

        model_config = self.config["model"]

        # 创建增强型文化适配模型
        self.model = EnhancedCulturalModel(
            num_regions=num_regions,
            num_cuisines=num_cuisines,
            feature_dim=model_config.get("cultural_feature_dim", 192),
            hidden_dim=model_config.get("hidden_dim", 384),
            lora_rank=model_config.get("lora_rank", 48),
            dropout=model_config.get("dropout", 0.2),
            use_batch_norm=model_config.get("use_batch_norm", True),
            use_residual=model_config.get("use_residual", True),
            num_layers=model_config.get("num_layers", 3),
            activation=model_config.get("activation", "gelu")
        )

        # 移动模型到设备
        self.model = self.model.to(self.device)

        # 输出模型信息
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")

    def train(self):
        """执行训练过程"""
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from tqdm import tqdm
        import numpy as np

        logger.info("📈 步骤3: 评估体系与早停机制")

        # 训练配置
        train_config = self.config["training"]
        num_epochs = train_config.get("num_epochs", 200)
        patience = train_config.get("patience", 30)
        min_delta = train_config.get("min_delta", 1e-6)

        # 自定义早停实现
        best_score = 0.0
        wait = 0
        best_model_state = None

        logger.info("⚙️ 步骤4: 训练策略与正则化")

        # 创建优化器和调度器
        optimizer = self.optimizer_manager.create_optimizer(self.model)
        scheduler = self.optimizer_manager.create_scheduler(optimizer, num_epochs)
        criterion = self.optimizer_manager.create_loss_function()

        # 梯度裁剪
        gradient_clipper = None
        if self.config["optimization"].get("regularization", {}).get("gradient_clipping", {}).get("enabled", True):
            max_norm = self.config["optimization"].get("regularization", {}).get("gradient_clipping", {}).get("max_norm", 1.0)
            from utils.training_optimizer import GradientClipping
            gradient_clipper = GradientClipping(max_norm=max_norm)

        # 混合精度训练
        scaler = torch.cuda.amp.GradScaler() if self.config["optimization"].get("mixed_precision", {}).get("enabled", True) else None

        # 模型EMA
        model_ema = None
        if self.config["optimization"].get("model_ema", {}).get("enabled", True):
            from utils.training_optimizer import ModelEMA
            decay = self.config["optimization"].get("model_ema", {}).get("decay", 0.999)
            model_ema = ModelEMA(self.model, decay=decay)

        logger.info("🏃 步骤5: 执行优化训练")

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

    def _save_model(self):
        """保存模型"""
        import torch

        # 创建输出目录
        output_dir = Path("outputs/enhanced_performance")
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存完整模型
        torch.save(
            self.model.state_dict(),
            output_dir / "enhanced_cultural_model.pt"
        )

        # 保存LoRA权重
        torch.save(
            self.model.lora_layer.state_dict(),
            output_dir / "enhanced_cultural_lora_weights.pt"
        )

        logger.info(f"✅ 增强模型已保存: {output_dir / 'enhanced_cultural_model.pt'}")
        logger.info(f"✅ LoRA权重已保存: {output_dir / 'enhanced_cultural_lora_weights.pt'}")

    def _save_final_report(self, metrics):
        """保存最终报告"""
        import json

        # 创建输出目录
        output_dir = Path("outputs/enhanced_performance")
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
            "config_name": "enhanced_performance"
        }

        with open(output_dir / "enhanced_cultural_adaptation_evaluation_report.json", "w") as f:
            json.dump(report, f, indent=2)

        # 创建最终训练报告
        training_report = {
            "model_info": {
                "hidden_dim": self.config["model"].get("hidden_dim", 384),
                "feature_dim": self.config["model"].get("cultural_feature_dim", 192),
                "lora_rank": self.config["model"].get("lora_rank", 48),
                "num_layers": self.config["model"].get("num_layers", 3)
            },
            "training_info": {
                "epochs": self.config["training"].get("num_epochs", 200),
                "batch_size": self.config["training"].get("batch_size", 64),
                "learning_rate": self.config["training"].get("learning_rate", 0.0003)
            },
            "performance": {
                "mae": float(metrics.mae),
                "mse": float(metrics.mse),
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
        with open(output_dir / "enhanced_training_report.json", "w") as f:
            json.dump(training_report, f, indent=2)

        logger.info(f"📋 最终报告已保存: {output_dir / 'enhanced_training_report.json'}")


def main():
    """主函数"""

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="增强性能文化适配训练")
    parser.add_argument("--sample-size", type=int, default=5000, help="样本数量")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--quick-test", action="store_true", help="快速测试模式")
    args = parser.parse_args()

    # 打印横幅
    print_banner()

    # 检查环境
    check_environment()

    try:
        # 加载配置
        config_loader = ConfigLoader("configs/enhanced_performance.yaml")
        if "enhanced_performance" not in config_loader.configs:
            logger.error("❌ 未找到增强性能配置，请确保配置文件存在")
            return 1
        config = config_loader.configs["enhanced_performance"]

        # 更新样本大小
        config["data"]["sample_size"] = args.sample_size

        # 快速测试模式
        if args.quick_test:
            config["training"]["num_epochs"] = 5
            config["model"]["hidden_dim"] = 64
            config["model"]["cultural_feature_dim"] = 32
            logger.info("⚡ 快速测试模式已启用")

        # 自定义轮数
        if args.epochs:
            config["training"]["num_epochs"] = args.epochs
            logger.info(f"⚙️ 自定义训练轮数: {args.epochs}")

        # 创建训练器
        trainer = EnhancedTrainer(config)

        # 准备数据
        num_regions, num_cuisines = trainer.prepare_data(args.sample_size)

        # 构建模型
        trainer.build_model(num_regions, num_cuisines)

        # 执行训练
        metrics = trainer.train()

        # 输出最终结果
        logger.info("🎉 训练完成！")
        logger.info("📊 最终结果:")
        logger.info(f"   MAE: {metrics.mae:.6f}")
        logger.info(f"   R²: {metrics.r2:.6f}")
        logger.info(f"   综合评分: {metrics.overall_score:.6f}")
        logger.info(f"   文化一致性: {metrics.cultural_consistency:.6f}")

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

        # 数据质量
        logger.info(f"📈 数据质量: {trainer.data_enhancer.quality_metrics.overall_score:.3f}")

        # 后续建议
        logger.info("\n💡 后续建议:")
        if metrics.r2 < 0:
            logger.info("   📊 尝试调整超参数")
            logger.info("   🔧 考虑使用高性能配置")
            logger.info("   📈 增加训练数据量")
        else:
            logger.info("   🎯 模型已达到良好性能，可以进入下一阶段")

        logger.info("=" * 80)
        logger.info("🎉 训练任务完成！")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"❌ 训练失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
