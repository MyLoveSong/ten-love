"""
训练器核心模块
"""

import os
import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
from datetime import datetime
import json

from .utils.training_utils import EarlyStopping, MetricsTracker, calculate_metrics

logger = logging.getLogger(__name__)


class TrainingConfig:
    """训练配置类 - 单一职责原则"""

    def __init__(
        self,
        num_epochs: int = 5,
        batch_size: int = 16,
        learning_rate: float = 0.001,
        device: Optional[torch.device] = None,
        output_dir: str = "outputs",
        patience: int = 10,
        min_delta: float = 0.001,
        use_amp: bool = True
    ):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(output_dir)
        self.patience = patience
        self.min_delta = min_delta
        self.use_amp = use_amp and torch.cuda.is_available()

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 记录混合精度状态
        if use_amp and not torch.cuda.is_available():
            logger.warning("CUDA不可用，已禁用混合精度训练")
        elif self.use_amp:
            logger.info("混合精度训练已启用")


class ModelTrainer:
    """模型训练器 - 依赖注入和接口分离原则"""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        config: Optional[TrainingConfig] = None
    ):
        """
        初始化训练器

        Args:
            model: 模型
            criterion: 损失函数
            optimizer: 优化器
            scheduler: 学习率调度器
            config: 训练配置
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config or TrainingConfig()

        # 移动模型到设备
        self.model.to(self.config.device)

        # 初始化组件
        self.early_stopping = EarlyStopping(
            patience=self.config.patience,
            min_delta=self.config.min_delta
        )
        self.metrics_tracker = MetricsTracker()

        # 训练状态
        self.best_val_loss = float('inf')
        self.training_history = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': []
        }

        # 初始化混合精度缩放器
        self.scaler = torch.amp.GradScaler('cuda') if self.config.use_amp else None

        logger.info(f"训练器初始化完成，设备: {self.config.device}")
        if self.config.use_amp:
            logger.info("混合精度训练已启用")

    def train_epoch(self, train_loader) -> Dict[str, float]:
        """
        训练一个轮次 - 支持混合精度

        Args:
            train_loader: 训练数据加载器

        Returns:
            训练指标
        """
        self.model.train()
        total_loss = 0.0
        total_metrics = {'mae': 0.0, 'rmse': 0.0}
        num_batches = len(train_loader)

        for batch in train_loader:
            # 数据移动到设备
            inputs = batch['input_sequence'].to(self.config.device)
            targets = batch['target_sequence'].to(self.config.device)

            self.optimizer.zero_grad()

            # 使用autocast进行前向传播
            if self.config.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            # 反向传播
            if self.config.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            # 累积损失
            total_loss += loss.item()

            # 计算指标
            with torch.no_grad():
                batch_metrics = calculate_metrics(outputs, targets)
                total_metrics['mae'] += batch_metrics['mae']
                total_metrics['rmse'] += batch_metrics['rmse']

        # 计算平均值
        avg_loss = total_loss / num_batches
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}

        return {'loss': avg_loss, **avg_metrics}

    def validate_epoch(self, val_loader) -> Dict[str, float]:
        """
        验证一个轮次 - 支持混合精度

        Args:
            val_loader: 验证数据加载器

        Returns:
            验证指标
        """
        self.model.eval()
        total_loss = 0.0
        total_metrics = {'mae': 0.0, 'rmse': 0.0}
        num_batches = len(val_loader)

        with torch.no_grad():
            for batch in val_loader:
                # 数据移动到设备
                inputs = batch['input_sequence'].to(self.config.device)
                targets = batch['target_sequence'].to(self.config.device)

                # 使用autocast进行前向传播（验证时也使用混合精度以保持一致性）
                if self.config.use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                # 累积损失
                total_loss += loss.item()

                # 计算指标
                batch_metrics = calculate_metrics(outputs, targets)
                total_metrics['mae'] += batch_metrics['mae']
                total_metrics['rmse'] += batch_metrics['rmse']

        # 计算平均值
        avg_loss = total_loss / num_batches
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}

        return {'loss': avg_loss, **avg_metrics}

    def test_model(self, test_loader) -> Dict[str, float]:
        """
        测试模型

        Args:
            test_loader: 测试数据加载器

        Returns:
            测试指标
        """
        return self.validate_epoch(test_loader)

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """
        保存检查点

        Args:
            epoch: 当前轮次
            metrics: 当前指标
            is_best: 是否为最佳模型
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': {
                'num_epochs': self.config.num_epochs,
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate
            },
            'timestamp': datetime.now().isoformat()
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # 保存检查点
        checkpoint_path = self.config.output_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)

        # 如果是最佳模型，额外保存
        if is_best:
            best_path = self.config.output_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"✓ 保存新的最佳模型，验证损失: {metrics['loss']:.6f}")

    def display_progress(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """显示训练进度"""
        self.training_history['epochs'].append(epoch + 1)
        self.training_history['train_loss'].append(train_metrics['loss'])
        self.training_history['val_loss'].append(val_metrics['loss'])
        self.training_history['train_mae'].append(train_metrics['mae'])
        self.training_history['val_mae'].append(val_metrics['mae'])

        current_lr = self.optimizer.param_groups[0]['lr']
        logger.info(
            f"Epoch {epoch+1}/{self.config.num_epochs} - "
            f"Train Loss: {train_metrics['loss']:.6f}, Val Loss: {val_metrics['loss']:.6f} - "
            f"Train MAE: {train_metrics['mae']:.4f}, Val MAE: {val_metrics['mae']:.4f} - "
            f"LR: {current_lr:.6f}"
        )

    def fit(self, train_loader, val_loader, test_loader=None) -> Dict[str, Any]:
        """
        训练模型

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器（可选）

        Returns:
            训练结果
        """
        logger.info(f"开始训练，共 {self.config.num_epochs} 轮次")

        # 训练循环
        for epoch in range(self.config.num_epochs):
            # 训练阶段
            train_metrics = self.train_epoch(train_loader)

            # 验证阶段
            val_metrics = self.validate_epoch(val_loader)

            # 更新学习率
            if self.scheduler:
                self.scheduler.step()

            # 更新指标跟踪器
            self.metrics_tracker.update_loss(train_metrics['loss'], val_metrics['loss'])
            self.metrics_tracker.update_metric('mae', train_metrics['mae'], 'train')
            self.metrics_tracker.update_metric('mae', val_metrics['mae'], 'val')
            self.metrics_tracker.update_metric('rmse', train_metrics['rmse'], 'train')
            self.metrics_tracker.update_metric('rmse', val_metrics['rmse'], 'val')

            if self.scheduler:
                self.metrics_tracker.update_learning_rate(self.optimizer.param_groups[0]['lr'])

            # 显示进度
            self.display_progress(epoch, train_metrics, val_metrics)

            # 保存检查点
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']

            self.save_checkpoint(epoch, val_metrics, is_best)

            # 早停检查
            if self.early_stopping(epoch, val_metrics['loss']):
                logger.info(f"早停触发，最佳轮次: {self.early_stopping.best_epoch}")
                break

        # 测试阶段
        test_results = {}
        if test_loader:
            logger.info("开始测试...")
            test_results = self.test_model(test_loader)
            logger.info(f"测试结果 - Loss: {test_results['loss']:.6f}, MAE: {test_results['mae']:.4f}, RMSE: {test_results['rmse']:.4f}")

        # 保存训练结果
        results = {
            'best_val_loss': self.best_val_loss,
            'test_results': test_results,
            'total_epochs': epoch + 1,
            'best_epoch': self.early_stopping.best_epoch if self.early_stopping.early_stop else epoch + 1,
            'training_history': self.training_history
        }

        results_path = self.config.output_dir / 'training_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        # 绘制训练曲线
        plots_dir = self.config.output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        self.metrics_tracker.plot_metrics(str(plots_dir))

        logger.info(f"训练完成！结果保存在: {self.config.output_dir}")
        return results


class TrainerFactory:
    """训练器工厂类 - 依赖注入原则"""

    @staticmethod
    def create_trainer(
        model: nn.Module,
        config: TrainingConfig,
        criterion_type: str = 'mse',
        optimizer_type: str = 'adamw',
        scheduler_type: Optional[str] = 'cosine'
    ) -> ModelTrainer:
        """
        创建训练器

        Args:
            model: 模型
            config: 训练配置
            criterion_type: 损失函数类型
            optimizer_type: 优化器类型
            scheduler_type: 调度器类型

        Returns:
            训练器实例
        """
        # 创建损失函数
        if criterion_type.lower() == 'mse':
            criterion = nn.MSELoss()
        elif criterion_type.lower() == 'mae':
            criterion = nn.L1Loss()
        else:
            raise ValueError(f"不支持的损失函数类型: {criterion_type}")

        # 创建优化器
        if optimizer_type.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=0.01
            )
        elif optimizer_type.lower() == 'adam':
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.learning_rate
            )
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")

        # 创建调度器
        scheduler = None
        if scheduler_type:
            if scheduler_type.lower() == 'cosine':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=config.num_epochs
                )
            elif scheduler_type.lower() == 'step':
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=config.num_epochs // 3,
                    gamma=0.1
                )

        return ModelTrainer(model, criterion, optimizer, scheduler, config)
