#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
集成训练系统
整合所有优化组件，实现端到端的训练流程
"""

import os
import sys
import json
import logging
import asyncio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import time
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入所有优化组件
from large_scale_data_collector import LargeScaleDataCollector, DataPoint
from smart_api_manager import SmartAPIManager
from enhanced_data_quality_processor import DataQualityProcessor, DataQualityMetrics
from optimized_model_architecture import ModelFactory, ModelConfig, BaseModel
from optimized_lora_trainer import OptimizedLoRATrainer
from utils.enhanced_evaluation import EnhancedEvaluator
from utils.training_optimizer import TrainingOptimizer


@dataclass
class TrainingPipelineConfig:
    """训练管道配置"""
    # 数据收集配置
    target_samples: int = 1000
    use_real_data: bool = True
    enable_data_quality: bool = True

    # 模型配置
    model_type: str = "optimized"  # efficient, optimized, advanced
    use_cross_validation: bool = True
    cv_folds: int = 5

    # 训练配置
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 2e-4
    patience: int = 15

    # 优化配置
    use_mixed_precision: bool = True
    use_ema: bool = True
    gradient_clipping: float = 1.0

    # 输出配置
    save_model: bool = True
    save_results: bool = True
    output_dir: str = "outputs/integrated_training"


class IntegratedTrainingSystem:
    """集成训练系统"""

    def __init__(self, config: TrainingPipelineConfig):
        """初始化训练系统"""
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化组件
        self.data_collector = None
        self.api_manager = None
        self.quality_processor = None
        self.model = None
        self.trainer = None
        self.evaluator = EnhancedEvaluator()
        self.optimizer_manager = TrainingOptimizer()

        # 训练状态
        self.training_history = []
        self.best_model_state = None
        self.best_score = float('-inf')

        logger.info("✅ 集成训练系统初始化完成")

    async def run_complete_pipeline(self) -> Dict[str, Any]:
        """运行完整的训练管道"""
        logger.info("🚀 启动完整训练管道")

        start_time = time.time()
        results = {}

        try:
            # 1. 数据收集阶段
            logger.info("📊 阶段1: 大规模数据收集")
            data_points = await self._collect_large_scale_data()
            results["data_collection"] = {
                "total_samples": len(data_points),
                "success_rate": len([dp for dp in data_points if dp.quality_score > 0.5]) / len(data_points) if data_points else 0
            }

            # 2. 数据质量处理阶段
            logger.info("🔧 阶段2: 数据质量处理")
            processed_data, quality_metrics = self._process_data_quality(data_points)
            results["data_quality"] = {
                "processed_samples": len(processed_data),
                "quality_metrics": {
                    "completeness": quality_metrics.completeness,
                    "consistency": quality_metrics.consistency,
                    "accuracy": quality_metrics.accuracy,
                    "uniqueness": quality_metrics.uniqueness,
                    "validity": quality_metrics.validity,
                    "overall_score": quality_metrics.overall_score
                }
            }

            # 3. 模型创建阶段
            logger.info("🏗️ 阶段3: 模型创建")
            self.model = self._create_optimized_model()
            model_info = self._analyze_model()
            results["model_info"] = model_info

            # 4. 训练阶段
            logger.info("🎯 阶段4: 模型训练")
            training_results = await self._train_model(processed_data)
            results["training_results"] = training_results

            # 5. 评估阶段
            logger.info("📈 阶段5: 模型评估")
            evaluation_results = self._evaluate_model()
            results["evaluation_results"] = evaluation_results

            # 6. 保存结果
            logger.info("💾 阶段6: 保存结果")
            self._save_training_results(results)

            # 计算总时间
            total_time = time.time() - start_time
            results["pipeline_summary"] = {
                "total_time": total_time,
                "success": True,
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"✅ 完整训练管道完成，总耗时: {total_time:.2f}秒")

        except Exception as e:
            logger.error(f"❌ 训练管道失败: {e}")
            results["pipeline_summary"] = {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

        return results

    async def _collect_large_scale_data(self) -> List[DataPoint]:
        """收集大规模数据"""
        # 创建数据收集器
        self.data_collector = LargeScaleDataCollector()

        # 收集数据
        data_points = await self.data_collector.collect_large_scale_data(
            target_samples=self.config.target_samples
        )

        # 保存原始数据
        if self.config.save_results:
            self.data_collector.save_collected_data(
                data_points,
                str(self.output_dir / "raw_data.json")
            )

        logger.info(f"📊 数据收集完成: {len(data_points)} 个样本")
        return data_points

    def _process_data_quality(self, data_points: List[DataPoint]) -> Tuple[List[DataPoint], DataQualityMetrics]:
        """处理数据质量"""
        # 创建质量处理器
        self.quality_processor = DataQualityProcessor()

        # 处理数据
        processed_data, quality_metrics = self.quality_processor.process_data_points(data_points)

        # 保存处理后的数据
        if self.config.save_results:
            self.quality_processor.save_processed_data(
                processed_data,
                str(self.output_dir / "processed_data.json")
            )

        # 打印质量报告
        self.quality_processor.print_quality_report(quality_metrics)

        return processed_data, quality_metrics

    def _create_optimized_model(self) -> BaseModel:
        """创建优化模型"""
        # 获取模型配置
        model_config = ModelFactory.get_recommended_config(self.config.model_type)

        # 创建模型
        model = ModelFactory.create_model(self.config.model_type, model_config)

        logger.info(f"🏗️ 创建 {self.config.model_type} 模型")
        return model

    def _analyze_model(self) -> Dict[str, Any]:
        """分析模型"""
        from optimized_model_architecture import ModelAnalyzer

        analysis = ModelAnalyzer.analyze_model(self.model)

        logger.info(f"📊 模型分析:")
        logger.info(f"   总参数: {analysis['size_info']['total_params']:,}")
        logger.info(f"   可训练参数: {analysis['size_info']['trainable_params']:,}")
        logger.info(f"   模型大小: {analysis['size_info']['model_size_mb']:.2f} MB")

        return analysis

    async def _train_model(self, data_points: List[DataPoint]) -> Dict[str, Any]:
        """训练模型"""
        # 准备训练数据
        X_preferences, X_nutrition, X_cultural, y = self._prepare_training_data(data_points)

        # 创建数据集
        dataset = TensorDataset(X_preferences, X_nutrition, X_cultural, y)

        # 训练配置
        training_config = {
            "data": {
                "cross_validation": self.config.use_cross_validation,
                "cv_folds": self.config.cv_folds
            },
            "model": {
                "cultural_feature_dim": 64,
                "hidden_dim": 256,
                "lora_rank": 16,
                "lora_alpha": 32,
                "dropout": 0.1
            },
            "training": {
                "num_epochs": self.config.num_epochs,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "patience": self.config.patience
            },
            "optimization": {
                "mixed_precision": self.config.use_mixed_precision,
                "ema": self.config.use_ema,
                "gradient_clipping": self.config.gradient_clipping
            }
        }

        # 创建训练器
        self.trainer = OptimizedLoRATrainer()
        self.trainer.config = training_config
        self.trainer.model = self.model

        # 执行训练
        if self.config.use_cross_validation:
            avg_score, cv_scores = self.trainer.train_with_cross_validation(
                dataset, None, None
            )
            training_results = {
                "average_r2": float(avg_score),
                "cv_scores": [float(score) for score in cv_scores],
                "std_r2": float(np.std(cv_scores))
            }
        else:
            # 简单训练
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )

            score = self.trainer._train_single_fold(train_dataset, val_dataset, 0)
            training_results = {
                "r2": float(score)
            }

        logger.info(f"🎯 训练完成，R²: {training_results.get('average_r2', training_results.get('r2', 0)):.4f}")

        return training_results

    def _prepare_training_data(self, data_points: List[DataPoint]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """准备训练数据"""
        preferences = []
        nutrition = []
        cultural = []
        targets = []

        for dp in data_points:
            # 偏好特征 (10维)
            pref_features = [
                dp.cultural_features.get("cultural_significance", 0.5),
                dp.cultural_features.get("flavor_profile", 0.5),
                dp.cultural_features.get("cooking_method", 0.5),
                dp.cultural_features.get("food_category", 0.5),
                dp.cultural_features.get("cuisine_type", 0.5),
                0.5, 0.5, 0.5, 0.5, 0.5  # 填充到10维
            ]

            # 营养特征 (5维)
            nutrition_features = [
                dp.nutrition.get("nf_calories", 0) / 1000,  # 归一化
                dp.nutrition.get("nf_protein", 0) / 100,
                dp.nutrition.get("nf_total_carbohydrate", 0) / 100,
                dp.nutrition.get("nf_total_fat", 0) / 100,
                dp.nutrition.get("nf_dietary_fiber", 0) / 10
            ]

            # 文化特征 (2维: region_id, cuisine_id)
            cuisine_mapping = {
                "川菜": 0, "粤菜": 1, "鲁菜": 2, "苏菜": 3,
                "浙菜": 4, "闽菜": 5, "湘菜": 6, "徽菜": 7, "常见食物": 8
            }
            region_mapping = {
                "川菜": 0, "粤菜": 1, "鲁菜": 2, "苏菜": 3,
                "浙菜": 4, "闽菜": 5, "湘菜": 6, "徽菜": 7, "常见食物": 8
            }

            cuisine_id = cuisine_mapping.get(dp.cuisine, 8)
            region_id = region_mapping.get(dp.cuisine, 8)

            cultural_features = [region_id, cuisine_id]

            # 目标值 (用户接受度)
            target = dp.quality_score

            preferences.append(pref_features)
            nutrition.append(nutrition_features)
            cultural.append(cultural_features)
            targets.append(target)

        # 转换为张量
        X_preferences = torch.tensor(preferences, dtype=torch.float32)
        X_nutrition = torch.tensor(nutrition, dtype=torch.float32)
        X_cultural = torch.tensor(cultural, dtype=torch.float32)
        y = torch.tensor(targets, dtype=torch.float32)

        return X_preferences, X_nutrition, X_cultural, y

    def _evaluate_model(self) -> Dict[str, Any]:
        """评估模型"""
        # 这里可以实现更详细的模型评估
        evaluation_results = {
            "model_ready": self.model is not None,
            "training_completed": True,
            "evaluation_timestamp": datetime.now().isoformat()
        }

        return evaluation_results

    def _save_training_results(self, results: Dict[str, Any]):
        """保存训练结果"""
        # 保存完整结果
        results_path = self.output_dir / "training_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # 保存模型
        if self.config.save_model and self.model is not None:
            model_path = self.output_dir / "trained_model.pt"
            torch.save(self.model.state_dict(), model_path)
            logger.info(f"✅ 模型已保存: {model_path}")

        logger.info(f"✅ 训练结果已保存: {results_path}")

    def print_pipeline_summary(self, results: Dict[str, Any]):
        """打印管道摘要"""
        logger.info("📊 训练管道摘要:")

        # 数据收集摘要
        if "data_collection" in results:
            dc = results["data_collection"]
            logger.info(f"   数据收集: {dc['total_samples']} 样本，成功率: {dc['success_rate']:.2%}")

        # 数据质量摘要
        if "data_quality" in results:
            dq = results["data_quality"]
            logger.info(f"   数据质量: {dq['processed_samples']} 样本，总体分数: {dq['quality_metrics']['overall_score']:.3f}")

        # 模型信息摘要
        if "model_info" in results:
            mi = results["model_info"]
            logger.info(f"   模型信息: {mi['size_info']['total_params']:,} 参数，{mi['size_info']['model_size_mb']:.2f} MB")

        # 训练结果摘要
        if "training_results" in results:
            tr = results["training_results"]
            if "average_r2" in tr:
                logger.info(f"   训练结果: 平均R² = {tr['average_r2']:.4f} ± {tr['std_r2']:.4f}")
            elif "r2" in tr:
                logger.info(f"   训练结果: R² = {tr['r2']:.4f}")

        # 管道摘要
        if "pipeline_summary" in results:
            ps = results["pipeline_summary"]
            logger.info(f"   管道状态: {'成功' if ps['success'] else '失败'}")
            if "total_time" in ps:
                logger.info(f"   总耗时: {ps['total_time']:.2f}秒")


async def main():
    """主函数"""
    logger.info("🚀 启动集成训练系统")

    # 创建训练配置
    config = TrainingPipelineConfig(
        target_samples=500,  # 减少样本数用于测试
        model_type="optimized",
        use_cross_validation=True,
        cv_folds=3,  # 减少折数用于测试
        num_epochs=20,  # 减少轮数用于测试
        batch_size=16,
        learning_rate=2e-4,
        patience=5
    )

    # 创建训练系统
    training_system = IntegratedTrainingSystem(config)

    # 运行完整管道
    results = await training_system.run_complete_pipeline()

    # 打印摘要
    training_system.print_pipeline_summary(results)

    logger.info("✅ 集成训练系统完成")


if __name__ == "__main__":
    asyncio.run(main())
