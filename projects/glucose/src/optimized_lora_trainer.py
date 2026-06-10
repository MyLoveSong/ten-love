#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
优化的LoRA微调训练器
基于现代工程标准，实现高效的LoRA微调训练
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
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import copy
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
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

# 导入必要的模块
from utils.config_loader import ConfigLoader
from utils.enhanced_evaluation import EnhancedEvaluator
from utils.training_optimizer import TrainingOptimizer
from simplified_lora import LoRALinear, LoRAConfig


@dataclass
class LoRATrainingConfig:
    """LoRA训练配置"""
    # LoRA参数
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.1
    use_rslora: bool = True
    init_lora_weights: bool = True

    # 训练参数
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    num_epochs: int = 100
    patience: int = 15
    min_delta: float = 1e-4

    # 优化器参数
    optimizer_type: str = "AdamW"
    scheduler_type: str = "CosineAnnealingWarmRestarts"
    warmup_steps: int = 100

    # 正则化参数
    gradient_clip_norm: float = 1.0
    use_mixed_precision: bool = True
    use_ema: bool = True
    ema_decay: float = 0.999


class OptimizedLoRAModel(nn.Module):
    """优化的LoRA模型"""

    def __init__(
        self,
        num_regions: int,
        num_cuisines: int,
        feature_dim: int = 128,
        hidden_dim: int = 256,
        lora_config: LoRATrainingConfig = None
    ):
        super().__init__()

        self.num_regions = num_regions
        self.num_cuisines = num_cuisines
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.lora_config = lora_config or LoRATrainingConfig()

        # 嵌入层
        self.region_embedding = nn.Embedding(max(2, num_regions), feature_dim // 2)
        self.cuisine_embedding = nn.Embedding(max(2, num_cuisines), feature_dim // 2)

        # 基础特征提取器
        self.base_encoder = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(self.lora_config.dropout),
            nn.Linear(hidden_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(self.lora_config.dropout)
        )

        # LoRA适配器
        lora_config_obj = LoRAConfig(
            r=self.lora_config.rank,
            alpha=self.lora_config.alpha,
            dropout=self.lora_config.dropout,
            use_rslora=self.lora_config.use_rslora,
            init_lora_weights=self.lora_config.init_lora_weights
        )

        self.lora_adapter = LoRALinear(
            in_features=feature_dim,
            out_features=hidden_dim,
            config=lora_config_obj
        )

        # 预测头
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.lora_config.dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)

    def forward(self, preferences, nutrition, cultural):
        """前向传播"""
        # 嵌入
        region_emb = self.region_embedding(cultural[:, 0].long())
        cuisine_emb = self.cuisine_embedding(cultural[:, 1].long())

        # 特征融合
        cultural_emb = torch.cat([region_emb, cuisine_emb], dim=1)
        all_features = torch.cat([preferences, nutrition, cultural_emb], dim=1)

        # 基础编码
        base_features = self.base_encoder(all_features)

        # LoRA适配
        adapted_features = self.lora_adapter(base_features)

        # 预测
        output = self.prediction_head(adapted_features)

        return output.squeeze(-1)

    def get_lora_parameters(self):
        """获取LoRA参数"""
        lora_params = []
        for name, param in self.named_parameters():
            if 'lora' in name:
                lora_params.append(param)
        return lora_params

    def get_trainable_parameters_count(self):
        """获取可训练参数数量"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        lora_params = sum(p.numel() for p in self.get_lora_parameters())

        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "lora_params": lora_params,
            "lora_ratio": lora_params / total_params if total_params > 0 else 0
        }


class ModelEMA:
    """模型指数移动平均"""

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._register_hooks()

    def _register_hooks(self):
        """注册钩子函数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """更新EMA"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self):
        """应用EMA权重"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """恢复原始权重"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class OptimizedLoRATrainer:
    """优化的LoRA训练器"""

    def __init__(self, config_path: str = "configs/optimized_lora_config.yaml"):
        """初始化训练器"""
        self.config_loader = ConfigLoader(config_path)
        self.config = self._get_optimized_config()

        # 初始化组件
        self.evaluator = EnhancedEvaluator()
        self.optimizer_manager = TrainingOptimizer()
        self.model = None
        self.ema = None

        # 训练状态
        self.training_history = []
        self.best_model_state = None
        self.best_score = float('-inf')

        logger.info("✅ 优化LoRA训练器初始化完成")

    def _get_optimized_config(self) -> Dict[str, Any]:
        """获取优化配置"""
        try:
            return self.config_loader.get_config("optimized_lora")
        except:
            # 返回默认优化配置
            return {
                "data": {
                    "sample_size": 1000,
                    "use_real_data": True,
                    "quality_enhancement": True,
                    "cross_validation": True,
                    "cv_folds": 5
                },
                "model": {
                    "cultural_feature_dim": 64,
                    "hidden_dim": 256,
                    "lora_rank": 16,
                    "lora_alpha": 32,
                    "dropout": 0.1,
                    "use_rslora": True
                },
                "training": {
                    "num_epochs": 100,
                    "batch_size": 32,
                    "learning_rate": 2e-4,
                    "weight_decay": 0.01,
                    "patience": 15,
                    "min_delta": 1e-4
                },
                "optimization": {
                    "optimizer": {
                        "type": "AdamW",
                        "weight_decay": 0.01
                    },
                    "scheduler": {
                        "type": "CosineAnnealingWarmRestarts",
                        "T_0": 10,
                        "T_mult": 2
                    },
                    "mixed_precision": True,
                    "gradient_clipping": 1.0,
                    "ema": True,
                    "ema_decay": 0.999
                }
            }

    def load_large_scale_data(self, data_path: str = "data/large_scale_data.json") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """加载大规模数据"""
        logger.info(f"📊 加载大规模数据: {data_path}")

        try:
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            logger.info(f"   数据样本数: {len(data)}")

            # 提取特征
            preferences = []
            nutrition = []
            cultural = []
            targets = []

            for item in data:
                # 偏好特征 (10维)
                pref_features = [
                    item.get("cultural_features", {}).get("cultural_significance", 0.5),
                    item.get("cultural_features", {}).get("flavor_profile", 0.5),
                    item.get("cultural_features", {}).get("cooking_method", 0.5),
                    item.get("cultural_features", {}).get("food_category", 0.5),
                    item.get("cultural_features", {}).get("cuisine_type", 0.5),
                    0.5, 0.5, 0.5, 0.5, 0.5  # 填充到10维
                ]

                # 营养特征 (5维)
                nutrition_features = [
                    item.get("nutrition", {}).get("nf_calories", 0) / 1000,  # 归一化
                    item.get("nutrition", {}).get("nf_protein", 0) / 100,
                    item.get("nutrition", {}).get("nf_total_carbohydrate", 0) / 100,
                    item.get("nutrition", {}).get("nf_total_fat", 0) / 100,
                    item.get("nutrition", {}).get("nf_dietary_fiber", 0) / 10
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

                cuisine_id = cuisine_mapping.get(item.get("cuisine", "常见食物"), 8)
                region_id = region_mapping.get(item.get("cuisine", "常见食物"), 8)

                cultural_features = [region_id, cuisine_id]

                # 目标值 (用户接受度)
                target = item.get("quality_score", 0.5)

                preferences.append(pref_features)
                nutrition.append(nutrition_features)
                cultural.append(cultural_features)
                targets.append(target)

            # 转换为张量
            X_preferences = torch.tensor(preferences, dtype=torch.float32)
            X_nutrition = torch.tensor(nutrition, dtype=torch.float32)
            X_cultural = torch.tensor(cultural, dtype=torch.float32)
            y = torch.tensor(targets, dtype=torch.float32)

            logger.info(f"   偏好特征形状: {X_preferences.shape}")
            logger.info(f"   营养特征形状: {X_nutrition.shape}")
            logger.info(f"   文化特征形状: {X_cultural.shape}")
            logger.info(f"   目标值形状: {y.shape}")

            return X_preferences, X_nutrition, X_cultural, y

        except Exception as e:
            logger.error(f"❌ 数据加载失败: {e}")
            # 返回模拟数据
            return self._generate_mock_data()

    def _generate_mock_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """生成模拟数据"""
        logger.warning("⚠️ 使用模拟数据进行训练")

        n_samples = 1000
        X_preferences = torch.randn(n_samples, 10)
        X_nutrition = torch.randn(n_samples, 5)
        X_cultural = torch.randint(0, 9, (n_samples, 2))
        y = torch.randn(n_samples)

        return X_preferences, X_nutrition, X_cultural, y

    def create_optimized_model(self) -> OptimizedLoRAModel:
        """创建优化的LoRA模型"""
        logger.info("🏗️ 创建优化的LoRA模型")

        # 获取配置
        model_config = self.config["model"]
        lora_config = LoRATrainingConfig(
            rank=model_config.get("lora_rank", 16),
            alpha=model_config.get("lora_alpha", 32),
            dropout=model_config.get("dropout", 0.1),
            use_rslora=model_config.get("use_rslora", True),
            learning_rate=self.config["training"].get("learning_rate", 2e-4),
            weight_decay=self.config["training"].get("weight_decay", 0.01),
            batch_size=self.config["training"].get("batch_size", 32),
            num_epochs=self.config["training"].get("num_epochs", 100),
            patience=self.config["training"].get("patience", 15)
        )

        # 创建模型
        model = OptimizedLoRAModel(
            num_regions=9,
            num_cuisines=9,
            feature_dim=model_config.get("cultural_feature_dim", 64),
            hidden_dim=model_config.get("hidden_dim", 256),
            lora_config=lora_config
        )

        # 打印模型信息
        param_info = model.get_trainable_parameters_count()
        logger.info(f"   总参数: {param_info['total_params']:,}")
        logger.info(f"   可训练参数: {param_info['trainable_params']:,}")
        logger.info(f"   LoRA参数: {param_info['lora_params']:,}")
        logger.info(f"   LoRA比例: {param_info['lora_ratio']:.2%}")

        return model

    def setup_training(self, X_preferences, X_nutrition, X_cultural, y):
        """设置训练"""
        logger.info("⚙️ 设置训练环境")

        # 创建模型
        self.model = self.create_optimized_model()

        # 数据标准化
        from sklearn.preprocessing import StandardScaler
        scaler_pref = StandardScaler()
        scaler_nutr = StandardScaler()

        X_preferences_scaled = torch.tensor(
            scaler_pref.fit_transform(X_preferences.numpy()), dtype=torch.float32
        )
        X_nutrition_scaled = torch.tensor(
            scaler_nutr.fit_transform(X_nutrition.numpy()), dtype=torch.float32
        )

        # 创建数据集
        dataset = TensorDataset(X_preferences_scaled, X_nutrition_scaled, X_cultural, y)

        # 数据分割
        if self.config["data"].get("cross_validation", True):
            logger.info("📊 使用交叉验证")
            return dataset, scaler_pref, scaler_nutr
        else:
            # 简单分割
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size

            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )

            return (train_dataset, val_dataset), scaler_pref, scaler_nutr

    def train_with_cross_validation(self, dataset, scaler_pref, scaler_nutr):
        """使用交叉验证训练"""
        logger.info("🔄 开始交叉验证训练")

        cv_folds = self.config["data"].get("cv_folds", 5)
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

        # 获取数据
        X_pref = dataset.tensors[0]
        X_nutr = dataset.tensors[1]
        X_cult = dataset.tensors[2]
        y = dataset.tensors[3]

        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_pref)):
            logger.info(f"📊 训练第 {fold + 1}/{cv_folds} 折")

            # 分割数据
            train_dataset = TensorDataset(
                X_pref[train_idx], X_nutr[train_idx], X_cult[train_idx], y[train_idx]
            )
            val_dataset = TensorDataset(
                X_pref[val_idx], X_nutr[val_idx], X_cult[val_idx], y[val_idx]
            )

            # 训练当前折
            fold_score = self._train_single_fold(train_dataset, val_dataset, fold)
            cv_scores.append(fold_score)

            logger.info(f"   第 {fold + 1} 折 R²: {fold_score:.4f}")

        # 计算平均分数
        avg_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)

        logger.info(f"📊 交叉验证结果:")
        logger.info(f"   平均 R²: {avg_score:.4f} ± {std_score:.4f}")
        logger.info(f"   各折分数: {[f'{score:.4f}' for score in cv_scores]}")

        return avg_score, cv_scores

    def _train_single_fold(self, train_dataset, val_dataset, fold):
        """训练单个折"""
        # 创建新的模型实例
        model = self.create_optimized_model()

        # 数据加载器
        batch_size = self.config["training"].get("batch_size", 32)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 优化器
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config["training"].get("learning_rate", 2e-4),
            weight_decay=self.config["training"].get("weight_decay", 0.01)
        )

        # 学习率调度器
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
        )

        # EMA
        ema = ModelEMA(model, decay=0.999) if self.config["optimization"].get("ema", True) else None

        # 训练循环
        best_score = float('-inf')
        patience_counter = 0
        patience = self.config["training"].get("patience", 15)

        for epoch in range(self.config["training"].get("num_epochs", 100)):
            # 训练
            model.train()
            train_loss = 0.0

            for batch_pref, batch_nutr, batch_cult, batch_y in train_loader:
                optimizer.zero_grad()

                outputs = model(batch_pref, batch_nutr, batch_cult)
                loss = nn.MSELoss()(outputs, batch_y)

                loss.backward()

                # 梯度裁剪
                if self.config["optimization"].get("gradient_clipping", 1.0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        self.config["optimization"].get("gradient_clipping", 1.0)
                    )

                optimizer.step()

                if ema:
                    ema.update()

                train_loss += loss.item()

            # 验证
            model.eval()
            val_loss = 0.0
            val_predictions = []
            val_targets = []

            with torch.no_grad():
                for batch_pref, batch_nutr, batch_cult, batch_y in val_loader:
                    outputs = model(batch_pref, batch_nutr, batch_cult)
                    loss = nn.MSELoss()(outputs, batch_y)
                    val_loss += loss.item()

                    val_predictions.extend(outputs.cpu().numpy())
                    val_targets.extend(batch_y.cpu().numpy())

            # 计算R²
            val_predictions = np.array(val_predictions)
            val_targets = np.array(val_targets)
            r2 = r2_score(val_targets, val_predictions)

            # 早停
            if r2 > best_score:
                best_score = r2
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"   早停于第 {epoch + 1} 轮")
                    break

            scheduler.step()

        return best_score

    def train_optimized_model(self):
        """训练优化模型"""
        logger.info("🚀 开始优化LoRA训练")

        # 加载数据
        X_preferences, X_nutrition, X_cultural, y = self.load_large_scale_data()

        # 设置训练
        dataset, scaler_pref, scaler_nutr = self.setup_training(
            X_preferences, X_nutrition, X_cultural, y
        )

        # 交叉验证训练
        avg_score, cv_scores = self.train_with_cross_validation(
            dataset, scaler_pref, scaler_nutr
        )

        # 保存结果
        self._save_training_results(avg_score, cv_scores)

        logger.info(f"✅ 优化LoRA训练完成，平均R²: {avg_score:.4f}")
        return avg_score

    def _save_training_results(self, avg_score, cv_scores):
        """保存训练结果"""
        output_dir = Path("outputs/optimized_lora_training")
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存结果
        results = {
            "timestamp": datetime.now().isoformat(),
            "average_r2": float(avg_score),
            "cv_scores": [float(score) for score in cv_scores],
            "std_r2": float(np.std(cv_scores)),
            "config": self.config
        }

        with open(output_dir / "optimized_training_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ 训练结果已保存: {output_dir}")


def main():
    """主函数"""
    logger.info("🚀 启动优化LoRA训练器")

    # 创建训练器
    trainer = OptimizedLoRATrainer()

    # 训练模型
    score = trainer.train_optimized_model()

    logger.info(f"✅ 训练完成，最终R²: {score:.4f}")


if __name__ == "__main__":
    main()
