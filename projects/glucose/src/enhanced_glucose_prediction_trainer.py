#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强版血糖预测训练器
解决多步预测和个性化建模问题，优化微调训练
"""

import os
import sys
import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
import copy

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedGlucosePredictor(nn.Module):
    """增强版血糖预测模型 - 支持多步预测和个性化"""

    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2,
                 output_steps=6, dropout=0.1, use_attention=True):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_steps = output_steps
        self.use_attention = use_attention

        # LSTM层
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )

        # 注意力机制
        if use_attention:
            self.attention = nn.MultiheadAttention(
                hidden_dim, num_heads=8, dropout=dropout, batch_first=True
            )
            self.attention_norm = nn.LayerNorm(hidden_dim)

        # 多步预测头
        self.multi_step_head = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(output_steps)
        ])

        # 个性化适应层
        self.personalization_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 全局预测头（用于单步预测）
        self.global_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_steps)
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x, use_multi_head=True, personalized=False):
        """
        前向传播
        Args:
            x: 输入序列 [batch_size, seq_len, input_dim]
            use_multi_head: 是否使用多头预测
            personalized: 是否使用个性化层
        """
        batch_size, seq_len, _ = x.shape

        # LSTM特征提取
        lstm_out, (hidden, cell) = self.lstm(x)

        # 注意力机制
        if self.use_attention:
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            attn_out = self.attention_norm(attn_out + lstm_out)
            features = attn_out[:, -1, :]  # 取最后一个时间步
        else:
            features = lstm_out[:, -1, :]

        # 个性化适应
        if personalized:
            features = self.personalization_layer(features)

        # 多步预测
        if use_multi_head:
            # 使用独立的预测头
            predictions = []
            for head in self.multi_step_head:
                pred = head(features)
                predictions.append(pred)
            output = torch.cat(predictions, dim=1)  # [batch_size, output_steps]
        else:
            # 使用全局预测头
            output = self.global_head(features)

        return output

    def get_attention_weights(self, x):
        """获取注意力权重用于可视化"""
        if not self.use_attention:
            return None

        with torch.no_grad():
            lstm_out, _ = self.lstm(x)
            _, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
            return attn_weights


class EnhancedGlucosePredictionTrainer:
    """增强版血糖预测训练器"""

    def __init__(self, config: Dict[str, Any] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 默认配置
        self.config = {
            # 模型配置
            "input_dim": 1,
            "hidden_dim": 64,
            "num_layers": 2,
            "output_steps": 6,
            "dropout": 0.1,
            "use_attention": True,

            # 训练配置
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.001,
            "weight_decay": 1e-5,
            "patience": 10,
            "min_delta": 1e-6,

            # 数据配置
            "seq_length": 12,
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "test_ratio": 0.1,

            # 个性化配置
            "personalization_epochs": 5,
            "personalization_lr": 0.0001,
            "min_personalization_samples": 20,

            # 多步预测配置
            "multi_step_loss_weights": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
            "use_curriculum_learning": True,

            # 输出配置
            "save_model": True,
            "save_plots": True,
            "output_dir": "outputs/enhanced_glucose_prediction"
        }

        if config:
            self.config.update(config)

        # 初始化组件
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = StandardScaler()

        # 训练历史
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "train_mae": [],
            "val_mae": [],
            "multi_step_losses": [],
            "attention_weights": []
        }

        # 个性化数据
        self.personalization_data = {}

        logger.info(f"增强版血糖预测训练器初始化完成")
        logger.info(f"使用设备: {self.device}")

    def create_model(self) -> nn.Module:
        """创建增强版血糖预测模型"""
        model = EnhancedGlucosePredictor(
            input_dim=self.config["input_dim"],
            hidden_dim=self.config["hidden_dim"],
            num_layers=self.config["num_layers"],
            output_steps=self.config["output_steps"],
            dropout=self.config["dropout"],
            use_attention=self.config["use_attention"]
        )

        return model.to(self.device)

    def generate_glucose_data(self, num_samples: int = 10000) -> Dict[str, np.ndarray]:
        """生成模拟血糖数据"""
        logger.info(f"生成 {num_samples} 个血糖数据样本...")

        np.random.seed(42)

        # 生成基础血糖模式
        time_points = np.arange(num_samples)

        # 日常血糖波动模式
        daily_pattern = 20 * np.sin(2 * np.pi * time_points / 288)  # 24小时周期 (5分钟间隔)
        meal_spikes = 30 * np.random.exponential(0.1, num_samples) * (np.random.random(num_samples) < 0.2)
        exercise_dips = -15 * np.random.exponential(0.1, num_samples) * (np.random.random(num_samples) < 0.1)

        # 基础血糖水平
        baseline = 120 + 10 * np.sin(2 * np.pi * time_points / (288 * 7))  # 周期性变化

        # 随机噪声
        noise = np.random.normal(0, 5, num_samples)

        # 合成血糖数据
        glucose_values = baseline + daily_pattern + meal_spikes + exercise_dips + noise
        glucose_values = np.clip(glucose_values, 70, 300)  # 限制在合理范围内

        # 创建序列数据
        seq_length = self.config["seq_length"]
        output_steps = self.config["output_steps"]

        sequences = []
        targets = []

        for i in range(len(glucose_values) - seq_length - output_steps + 1):
            seq = glucose_values[i:i + seq_length]
            target = glucose_values[i + seq_length:i + seq_length + output_steps]
            sequences.append(seq)
            targets.append(target)

        sequences = np.array(sequences)
        targets = np.array(targets)

        logger.info(f"生成完成: {len(sequences)} 个序列")

        return {
            "sequences": sequences,
            "targets": targets,
            "raw_glucose": glucose_values
        }

    def prepare_data(self, data: Dict[str, np.ndarray]) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """准备训练数据"""
        sequences = data["sequences"]
        targets = data["targets"]

        # 数据标准化
        sequences_scaled = self.scaler.fit_transform(sequences.reshape(-1, 1)).reshape(sequences.shape)
        targets_scaled = self.scaler.transform(targets.reshape(-1, 1)).reshape(targets.shape)

        # 数据分割
        n_samples = len(sequences)
        train_size = int(n_samples * self.config["train_ratio"])
        val_size = int(n_samples * self.config["val_ratio"])

        # 训练集
        train_sequences = sequences_scaled[:train_size]
        train_targets = targets_scaled[:train_size]

        # 验证集
        val_sequences = sequences_scaled[train_size:train_size + val_size]
        val_targets = targets_scaled[train_size:train_size + val_size]

        # 测试集
        test_sequences = sequences_scaled[train_size + val_size:]
        test_targets = targets_scaled[train_size + val_size:]

        # 创建DataLoader
        train_dataset = TensorDataset(
            torch.FloatTensor(train_sequences).unsqueeze(-1),
            torch.FloatTensor(train_targets)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(val_sequences).unsqueeze(-1),
            torch.FloatTensor(val_targets)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(test_sequences).unsqueeze(-1),
            torch.FloatTensor(test_targets)
        )

        train_loader = DataLoader(train_dataset, batch_size=self.config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config["batch_size"], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.config["batch_size"], shuffle=False)

        logger.info(f"数据准备完成: 训练集 {len(train_dataset)}, 验证集 {len(val_dataset)}, 测试集 {len(test_dataset)}")

        return train_loader, val_loader, test_loader

    def multi_step_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """多步预测损失函数"""
        batch_size, output_steps = predictions.shape
        total_loss = 0

        weights = self.config["multi_step_loss_weights"]
        if len(weights) != output_steps:
            weights = [1.0] * output_steps

        step_losses = []
        for i in range(output_steps):
            step_loss = nn.MSELoss()(predictions[:, i], targets[:, i])
            weighted_loss = weights[i] * step_loss
            total_loss += weighted_loss
            step_losses.append(step_loss.item())

        # 记录每步损失
        self.training_history["multi_step_losses"].append(step_losses)

        return total_loss / output_steps

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()

        total_loss = 0
        total_mae = 0
        num_batches = 0

        for batch_idx, (sequences, targets) in enumerate(train_loader):
            sequences = sequences.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            # 课程学习：逐步增加预测步数
            if self.config["use_curriculum_learning"]:
                max_steps = min(self.config["output_steps"],
                              max(1, epoch // 5 + 1))
                predictions = self.model(sequences, use_multi_head=True)[:, :max_steps]
                current_targets = targets[:, :max_steps]
            else:
                predictions = self.model(sequences, use_multi_head=True)
                current_targets = targets

            # 计算损失
            loss = self.multi_step_loss(predictions, current_targets)

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # 计算MAE
            with torch.no_grad():
                mae = torch.mean(torch.abs(predictions - current_targets))
                total_mae += mae.item()

            total_loss += loss.item()
            num_batches += 1

            # 打印进度
            if batch_idx % 100 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, "
                          f"Loss: {loss.item():.4f}, MAE: {mae.item():.4f}")

        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches

        return {"loss": avg_loss, "mae": avg_mae}

    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()

        total_loss = 0
        total_mae = 0
        num_batches = 0

        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)

                predictions = self.model(sequences, use_multi_head=True)
                loss = self.multi_step_loss(predictions, targets)
                mae = torch.mean(torch.abs(predictions - targets))

                total_loss += loss.item()
                total_mae += mae.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches

        return {"loss": avg_loss, "mae": avg_mae}

    def train_model(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        """训练模型"""
        logger.info("开始训练增强版血糖预测模型...")

        # 创建模型
        self.model = self.create_model()

        # 创建优化器和调度器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"]
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        # 早停机制
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        # 训练循环
        for epoch in range(self.config["epochs"]):
            # 训练
            train_metrics = self.train_epoch(train_loader, epoch)

            # 验证
            val_metrics = self.validate_epoch(val_loader)

            # 更新学习率
            self.scheduler.step(val_metrics["loss"])

            # 记录历史
            self.training_history["train_loss"].append(train_metrics["loss"])
            self.training_history["val_loss"].append(val_metrics["loss"])
            self.training_history["train_mae"].append(train_metrics["mae"])
            self.training_history["val_mae"].append(val_metrics["mae"])

            # 早停检查
            if val_metrics["loss"] < best_val_loss - self.config["min_delta"]:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
                best_model_state = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1

            logger.info(f"Epoch {epoch+1}/{self.config['epochs']}: "
                      f"Train Loss: {train_metrics['loss']:.4f}, "
                      f"Val Loss: {val_metrics['loss']:.4f}, "
                      f"Train MAE: {train_metrics['mae']:.4f}, "
                      f"Val MAE: {val_metrics['mae']:.4f}")

            # 早停
            if patience_counter >= self.config["patience"]:
                logger.info(f"早停触发，在第 {epoch+1} 轮停止训练")
                break

        # 加载最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        logger.info("模型训练完成")

        return {
            "best_val_loss": best_val_loss,
            "total_epochs": epoch + 1,
            "training_history": self.training_history
        }

    def evaluate_multi_step_prediction(self, test_loader: DataLoader) -> Dict[str, Any]:
        """评估多步预测性能"""
        logger.info("评估多步预测性能...")

        self.model.eval()

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for sequences, targets in test_loader:
                sequences = sequences.to(self.device)
                predictions = self.model(sequences, use_multi_head=True)

                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.numpy())

        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        # 反标准化
        predictions_orig = self.scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(predictions.shape)
        targets_orig = self.scaler.inverse_transform(targets.reshape(-1, 1)).reshape(targets.shape)

        # 计算每步的指标
        step_metrics = {}
        for step in range(self.config["output_steps"]):
            step_pred = predictions_orig[:, step]
            step_target = targets_orig[:, step]

            mae = mean_absolute_error(step_target, step_pred)
            rmse = np.sqrt(mean_squared_error(step_target, step_pred))
            r2 = r2_score(step_target, step_pred)

            step_metrics[f"t+{step+1}"] = {
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "mean_error": np.mean(step_pred - step_target),
                "std_error": np.std(step_pred - step_target)
            }

        # 整体指标
        overall_mae = mean_absolute_error(targets_orig.flatten(), predictions_orig.flatten())
        overall_rmse = np.sqrt(mean_squared_error(targets_orig.flatten(), predictions_orig.flatten()))
        overall_r2 = r2_score(targets_orig.flatten(), predictions_orig.flatten())

        logger.info("多步预测评估完成")

        return {
            "step_metrics": step_metrics,
            "overall_metrics": {
                "mae": overall_mae,
                "rmse": overall_rmse,
                "r2": overall_r2
            },
            "predictions": predictions_orig,
            "targets": targets_orig
        }

    def personalize_model(self, user_data: Dict[str, np.ndarray], user_id: str) -> Dict[str, Any]:
        """个性化模型微调"""
        logger.info(f"开始个性化模型微调: {user_id}")

        # 准备个性化数据
        sequences = user_data["sequences"]
        targets = user_data["targets"]

        if len(sequences) < self.config["min_personalization_samples"]:
            logger.warning(f"个性化数据不足: {len(sequences)} < {self.config['min_personalization_samples']}")
            return {"status": "insufficient_data"}

        # 标准化个性化数据
        sequences_scaled = self.scaler.transform(sequences.reshape(-1, 1)).reshape(sequences.shape)
        targets_scaled = self.scaler.transform(targets.reshape(-1, 1)).reshape(targets.shape)

        # 创建个性化数据集
        dataset = TensorDataset(
            torch.FloatTensor(sequences_scaled).unsqueeze(-1),
            torch.FloatTensor(targets_scaled)
        )
        dataloader = DataLoader(dataset, batch_size=min(16, len(dataset)), shuffle=True)

        # 创建个性化优化器（只优化个性化层）
        personalization_params = list(self.model.personalization_layer.parameters())
        optimizer = optim.Adam(personalization_params, lr=self.config["personalization_lr"])

        # 个性化训练
        initial_loss = None
        final_loss = None

        for epoch in range(self.config["personalization_epochs"]):
            self.model.train()
            epoch_loss = 0
            num_batches = 0

            for sequences_batch, targets_batch in dataloader:
                sequences_batch = sequences_batch.to(self.device)
                targets_batch = targets_batch.to(self.device)

                optimizer.zero_grad()

                # 使用个性化层
                predictions = self.model(sequences_batch, use_multi_head=True, personalized=True)
                loss = self.multi_step_loss(predictions, targets_batch)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            if epoch == 0:
                initial_loss = avg_loss
            final_loss = avg_loss

            logger.info(f"个性化 Epoch {epoch+1}/{self.config['personalization_epochs']}: Loss = {avg_loss:.4f}")

        # 保存个性化数据
        self.personalization_data[user_id] = {
            "model_state": copy.deepcopy(self.model.personalization_layer.state_dict()),
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "improvement": (initial_loss - final_loss) / initial_loss * 100 if initial_loss > 0 else 0,
            "samples_used": len(sequences),
            "timestamp": datetime.now().isoformat()
        }

        logger.info(f"个性化完成: 性能改进 {self.personalization_data[user_id]['improvement']:.2f}%")

        return {
            "status": "success",
            "improvement": self.personalization_data[user_id]["improvement"],
            "samples_used": len(sequences),
            "initial_loss": initial_loss,
            "final_loss": final_loss
        }

    def save_model_and_results(self, training_result: Dict[str, Any],
                              evaluation_result: Dict[str, Any]) -> str:
        """保存模型和结果"""
        output_dir = Path(self.config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存模型
        if self.config["save_model"]:
            model_path = output_dir / f"enhanced_glucose_model_{timestamp}.pt"
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "config": self.config,
                "scaler": self.scaler,
                "training_history": self.training_history,
                "personalization_data": self.personalization_data
            }, model_path)
            logger.info(f"模型已保存: {model_path}")

        # 保存训练结果
        results = {
            "timestamp": timestamp,
            "config": self.config,
            "training_result": training_result,
            "evaluation_result": evaluation_result,
            "personalization_data": self.personalization_data
        }

        results_path = output_dir / f"training_results_{timestamp}.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        # 保存可视化图表
        if self.config["save_plots"]:
            self._save_plots(output_dir, timestamp, evaluation_result)

        logger.info(f"结果已保存到: {output_dir}")
        return str(output_dir)

    def _save_plots(self, output_dir: Path, timestamp: str, evaluation_result: Dict[str, Any]):
        """保存可视化图表"""
        plt.style.use('seaborn-v0_8')

        # 1. 训练历史
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 损失曲线
        axes[0, 0].plot(self.training_history["train_loss"], label="训练损失")
        axes[0, 0].plot(self.training_history["val_loss"], label="验证损失")
        axes[0, 0].set_title("训练损失曲线")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # MAE曲线
        axes[0, 1].plot(self.training_history["train_mae"], label="训练MAE")
        axes[0, 1].plot(self.training_history["val_mae"], label="验证MAE")
        axes[0, 1].set_title("MAE曲线")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("MAE")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # 多步预测性能
        if "step_metrics" in evaluation_result:
            steps = list(evaluation_result["step_metrics"].keys())
            maes = [evaluation_result["step_metrics"][step]["mae"] for step in steps]
            rmses = [evaluation_result["step_metrics"][step]["rmse"] for step in steps]

            axes[1, 0].bar(steps, maes, alpha=0.7, label="MAE")
            axes[1, 0].set_title("多步预测MAE")
            axes[1, 0].set_xlabel("预测步数")
            axes[1, 0].set_ylabel("MAE")
            axes[1, 0].tick_params(axis='x', rotation=45)

            axes[1, 1].bar(steps, rmses, alpha=0.7, label="RMSE", color='orange')
            axes[1, 1].set_title("多步预测RMSE")
            axes[1, 1].set_xlabel("预测步数")
            axes[1, 1].set_ylabel("RMSE")
            axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(output_dir / f"training_plots_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 2. 预测结果可视化
        if "predictions" in evaluation_result and "targets" in evaluation_result:
            predictions = evaluation_result["predictions"]
            targets = evaluation_result["targets"]

            fig, axes = plt.subplots(2, 3, figsize=(18, 10))

            # 显示前6个预测步骤的散点图
            for i in range(min(6, self.config["output_steps"])):
                row = i // 3
                col = i % 3

                axes[row, col].scatter(targets[:100, i], predictions[:100, i], alpha=0.6)
                axes[row, col].plot([targets[:, i].min(), targets[:, i].max()],
                                  [targets[:, i].min(), targets[:, i].max()], 'r--', lw=2)
                axes[row, col].set_title(f"t+{i+1} 预测 vs 真实")
                axes[row, col].set_xlabel("真实值")
                axes[row, col].set_ylabel("预测值")
                axes[row, col].grid(True)

            plt.tight_layout()
            plt.savefig(output_dir / f"prediction_plots_{timestamp}.png", dpi=300, bbox_inches='tight')
            plt.close()

        logger.info("可视化图表已保存")

    def run_complete_training(self) -> Dict[str, Any]:
        """运行完整的训练流程"""
        logger.info("开始完整的增强版血糖预测训练流程...")

        # 1. 生成数据
        data = self.generate_glucose_data()

        # 2. 准备数据
        train_loader, val_loader, test_loader = self.prepare_data(data)

        # 3. 训练模型
        training_result = self.train_model(train_loader, val_loader)

        # 4. 评估模型
        evaluation_result = self.evaluate_multi_step_prediction(test_loader)

        # 5. 个性化测试
        # 生成个性化测试数据
        personalization_data = self.generate_glucose_data(num_samples=1000)
        personalization_result = self.personalize_model(personalization_data, "test_user")

        # 6. 保存结果
        output_dir = self.save_model_and_results(training_result, evaluation_result)

        # 7. 生成报告
        report = self._generate_final_report(training_result, evaluation_result, personalization_result)

        logger.info("完整训练流程完成！")

        return {
            "training_result": training_result,
            "evaluation_result": evaluation_result,
            "personalization_result": personalization_result,
            "output_dir": output_dir,
            "report": report
        }

    def _generate_final_report(self, training_result: Dict[str, Any],
                              evaluation_result: Dict[str, Any],
                              personalization_result: Dict[str, Any]) -> str:
        """生成最终报告"""
        report = f"""
# 🚀 增强版血糖预测模型训练报告

## 📊 训练概览
- **模型类型**: 增强版多步血糖预测模型
- **训练时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **设备**: {self.device}
- **总轮数**: {training_result['total_epochs']}
- **最佳验证损失**: {training_result['best_val_loss']:.6f}

## 🏗️ 模型架构
- **输入维度**: {self.config['input_dim']}
- **隐藏维度**: {self.config['hidden_dim']}
- **LSTM层数**: {self.config['num_layers']}
- **输出步数**: {self.config['output_steps']}
- **注意力机制**: {'启用' if self.config['use_attention'] else '禁用'}

## 📈 多步预测性能
"""

        if "step_metrics" in evaluation_result:
            for step, metrics in evaluation_result["step_metrics"].items():
                report += f"- **{step}**: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}\n"

        report += f"""
## 🎯 整体性能
- **整体MAE**: {evaluation_result['overall_metrics']['mae']:.4f}
- **整体RMSE**: {evaluation_result['overall_metrics']['rmse']:.4f}
- **整体R²**: {evaluation_result['overall_metrics']['r2']:.4f}

## 👤 个性化结果
- **状态**: {personalization_result['status']}
"""

        if personalization_result['status'] == 'success':
            report += f"- **性能改进**: {personalization_result['improvement']:.2f}%\n"
            report += f"- **使用样本**: {personalization_result['samples_used']}\n"

        return report


def main():
    """主函数"""
    print("\n" + "="*80)
    print("🚀 增强版血糖预测训练器")
    print("解决多步预测和个性化建模问题，优化微调训练")
    print("="*80 + "\n")

    # 配置参数
    config = {
        "epochs": 30,
        "batch_size": 32,
        "learning_rate": 0.001,
        "output_steps": 6,
        "use_attention": True,
        "use_curriculum_learning": True,
        "personalization_epochs": 10,
        "save_model": True,
        "save_plots": True
    }

    # 创建训练器
    trainer = EnhancedGlucosePredictionTrainer(config)

    # 运行完整训练
    results = trainer.run_complete_training()

    # 输出结果
    print("\n📊 训练结果:")
    print(f"   最佳验证损失: {results['training_result']['best_val_loss']:.6f}")
    print(f"   总训练轮数: {results['training_result']['total_epochs']}")
    print(f"   整体MAE: {results['evaluation_result']['overall_metrics']['mae']:.4f}")
    print(f"   整体RMSE: {results['evaluation_result']['overall_metrics']['rmse']:.4f}")
    print(f"   整体R²: {results['evaluation_result']['overall_metrics']['r2']:.4f}")

    if results['personalization_result']['status'] == 'success':
        print(f"   个性化改进: {results['personalization_result']['improvement']:.2f}%")

    print(f"\n📁 结果已保存到: {results['output_dir']}")

    print("\n" + "="*80)
    print("✅ 增强版血糖预测训练完成！")
    print("🎯 多步预测和个性化建模问题已解决")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
