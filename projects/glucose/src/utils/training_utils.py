"""
训练工具模块
提供训练、评估和模型管理功能
"""

import os
import json
import logging
import time
from typing import Dict, Any, Optional, Tuple, Callable, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer as Optimizer

logger = logging.getLogger(__name__)

class EarlyStopping:
    """早停机制"""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
        verbose: bool = True
    ):
        """
        初始化早停

        Args:
            patience: 容忍的轮次数
            min_delta: 最小变化量
            mode: 'min'表示监控指标越小越好，'max'表示越大越好
            verbose: 是否打印信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

        if mode == 'min':
            self.val_score = float('inf')
        else:
            self.val_score = float('-inf')

    def __call__(self, epoch: int, val_score: float) -> bool:
        """
        检查是否应该早停

        Args:
            epoch: 当前轮次
            val_score: 验证分数

        Returns:
            是否应该早停
        """
        if self.mode == 'min':
            score = -val_score
        else:
            score = val_score

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                logger.info(f'早停计数器: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0

        return False

class MetricsTracker:
    """指标跟踪器"""

    def __init__(self):
        """初始化指标跟踪器"""
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'test_loss': [],
            'train_metrics': {},
            'val_metrics': {},
            'test_metrics': {},
            'learning_rates': []
        }

    def update_loss(self, train_loss: float, val_loss: Optional[float] = None, test_loss: Optional[float] = None):
        """
        更新损失值

        Args:
            train_loss: 训练损失
            val_loss: 验证损失
            test_loss: 测试损失
        """
        self.metrics['train_loss'].append(train_loss)

        if val_loss is not None:
            self.metrics['val_loss'].append(val_loss)

        if test_loss is not None:
            self.metrics['test_loss'].append(test_loss)

    def update_metric(self, metric_name: str, value: float, phase: str = 'train'):
        """
        更新指标

        Args:
            metric_name: 指标名称
            value: 指标值
            phase: 阶段（train/val/test）
        """
        phase_key = f'{phase}_metrics'

        if metric_name not in self.metrics[phase_key]:
            self.metrics[phase_key][metric_name] = []

        self.metrics[phase_key][metric_name].append(value)

    def update_learning_rate(self, lr: float):
        """
        更新学习率

        Args:
            lr: 学习率
        """
        self.metrics['learning_rates'].append(lr)

    def get_latest_metrics(self) -> Dict[str, Any]:
        """
        获取最新指标

        Returns:
            最新指标字典
        """
        latest = {}

        # 损失
        if self.metrics['train_loss']:
            latest['train_loss'] = self.metrics['train_loss'][-1]

        if self.metrics['val_loss']:
            latest['val_loss'] = self.metrics['val_loss'][-1]

        if self.metrics['test_loss']:
            latest['test_loss'] = self.metrics['test_loss'][-1]

        # 其他指标
        for phase in ['train', 'val', 'test']:
            phase_key = f'{phase}_metrics'
            for metric_name, values in self.metrics[phase_key].items():
                if values:
                    latest[f'{phase}_{metric_name}'] = values[-1]

        # 学习率
        if self.metrics['learning_rates']:
            latest['learning_rate'] = self.metrics['learning_rates'][-1]

        return latest

    def plot_metrics(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        绘制指标图表

        Args:
            save_path: 保存路径

        Returns:
            图表字典
        """
        figures = {}

        # 绘制损失曲线
        if self.metrics['train_loss']:
            fig, ax = plt.subplots(figsize=(10, 6))
            epochs = range(1, len(self.metrics['train_loss']) + 1)

            ax.plot(epochs, self.metrics['train_loss'], 'b-', label='训练损失')

            if self.metrics['val_loss']:
                ax.plot(epochs, self.metrics['val_loss'], 'r-', label='验证损失')

            ax.set_title('训练和验证损失')
            ax.set_xlabel('轮次')
            ax.set_ylabel('损失')
            ax.legend()
            ax.grid(True)

            figures['loss'] = fig

            if save_path:
                os.makedirs(save_path, exist_ok=True)
                fig.savefig(os.path.join(save_path, 'loss_curve.png'))

        # 绘制其他指标
        for phase in ['train', 'val', 'test']:
            phase_key = f'{phase}_metrics'
            for metric_name, values in self.metrics[phase_key].items():
                if values:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    epochs = range(1, len(values) + 1)

                    ax.plot(epochs, values, '-')
                    ax.set_title(f'{phase.capitalize()} {metric_name}')
                    ax.set_xlabel('轮次')
                    ax.set_ylabel(metric_name)
                    ax.grid(True)

                    figures[f'{phase}_{metric_name}'] = fig

                    if save_path:
                        os.makedirs(save_path, exist_ok=True)
                        fig.savefig(os.path.join(save_path, f'{phase}_{metric_name}_curve.png'))

        return figures

    def save_metrics(self, save_path: str):
        """
        保存指标到JSON文件

        Args:
            save_path: 保存路径
        """
        try:
            # 转换NumPy数组为列表
            metrics_json = {}
            for key, value in self.metrics.items():
                if isinstance(value, list):
                    metrics_json[key] = [float(v) if isinstance(v, (np.float32, np.float64)) else v for v in value]
                elif isinstance(value, dict):
                    metrics_json[key] = {}
                    for k, v in value.items():
                        metrics_json[key][k] = [float(x) if isinstance(x, (np.float32, np.float64)) else x for x in v]
                else:
                    metrics_json[key] = value

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(metrics_json, f, indent=2)

            logger.info(f"指标已保存到: {save_path}")

        except Exception as e:
            logger.error(f"保存指标失败: {e}")

def calculate_metrics(
    outputs: torch.Tensor,
    targets: torch.Tensor
) -> Dict[str, float]:
    """
    计算评估指标

    Args:
        outputs: 模型输出
        targets: 目标值

    Returns:
        指标字典
    """
    # 确保是CPU张量并转换为NumPy数组
    outputs_np = outputs.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()

    # 计算均方误差(MSE)
    mse = np.mean((outputs_np - targets_np) ** 2)

    # 计算均方根误差(RMSE)
    rmse = np.sqrt(mse)

    # 计算平均绝对误差(MAE)
    mae = np.mean(np.abs(outputs_np - targets_np))

    # 计算平均绝对百分比误差(MAPE)
    # 避免除以零
    mask = targets_np != 0
    mape = np.mean(np.abs((targets_np[mask] - outputs_np[mask]) / targets_np[mask])) * 100

    # 计算决定系数(R^2)
    if targets_np.size > 1:
        target_mean = np.mean(targets_np)
        ss_tot = np.sum((targets_np - target_mean) ** 2)
        ss_res = np.sum((targets_np - outputs_np) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    else:
        r2 = 0

    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape),
        'r2': float(r2)
    }

def save_model(
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    loss: float,
    metrics: Dict[str, Any],
    save_path: str,
    config: Optional[Dict[str, Any]] = None
):
    """
    保存模型检查点

    Args:
        model: 模型
        optimizer: 优化器
        epoch: 轮次
        loss: 损失
        metrics: 指标
        save_path: 保存路径
        config: 配置
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }

        if config:
            checkpoint['config'] = config

        torch.save(checkpoint, save_path)
        logger.info(f"模型已保存到: {save_path}")

    except Exception as e:
        logger.error(f"保存模型失败: {e}")

def load_model(
    model: nn.Module,
    optimizer: Optional[Optimizer],
    checkpoint_path: str,
    device: torch.device = None
) -> Dict[str, Any]:
    """
    加载模型检查点

    Args:
        model: 模型
        optimizer: 优化器
        checkpoint_path: 检查点路径
        device: 设备

    Returns:
        检查点信息
    """
    try:
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        logger.info(f"模型已从 {checkpoint_path} 加载")

        # 返回检查点信息
        checkpoint_info = {
            'epoch': checkpoint.get('epoch', 0),
            'loss': checkpoint.get('loss', float('inf')),
            'metrics': checkpoint.get('metrics', {}),
            'config': checkpoint.get('config', {}),
            'timestamp': checkpoint.get('timestamp', '')
        }

        return checkpoint_info

    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        raise

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: Callable,
    optimizer: Optimizer,
    device: torch.device,
    scheduler: Optional[Any] = None,
    clip_grad_norm: Optional[float] = None
) -> Tuple[float, Dict[str, float]]:
    """
    训练一个轮次

    Args:
        model: 模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        scheduler: 学习率调度器
        clip_grad_norm: 梯度裁剪范数

    Returns:
        平均损失和指标
    """
    model.train()
    total_loss = 0.0
    all_outputs = []
    all_targets = []

    for batch in train_loader:
        # 解包批次数据
        if isinstance(batch, dict):
            inputs = batch.get('input_sequence', None)
            targets = batch.get('target_sequence', None)
            personal_features = batch.get('personal_features', None)
            image_features = batch.get('image_features', None)
            text_features = batch.get('text_features', None)
        else:
            inputs, targets = batch[0], batch[1]
            personal_features = batch[2] if len(batch) > 2 else None
            image_features = batch[3] if len(batch) > 3 else None
            text_features = batch[4] if len(batch) > 4 else None

        # 移动数据到设备
        inputs = inputs.to(device)
        targets = targets.to(device)
        if personal_features is not None:
            personal_features = personal_features.to(device)
        if image_features is not None:
            image_features = image_features.to(device)
        if text_features is not None:
            text_features = text_features.to(device)

        # 前向传播
        optimizer.zero_grad()

        # 根据模型接口调用
        if hasattr(model, 'forward') and 'personal_features' in model.forward.__code__.co_varnames:
            outputs = model(inputs, personal_features, image_features, text_features)
        else:
            outputs = model(inputs)

        # 计算损失
        loss = criterion(outputs, targets)

        # 反向传播
        loss.backward()

        # 梯度裁剪
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

        # 更新参数
        optimizer.step()

        # 更新学习率
        if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()

        # 累积损失
        total_loss += loss.item() * inputs.size(0)

        # 收集输出和目标
        all_outputs.append(outputs.detach())
        all_targets.append(targets.detach())

    # 计算平均损失
    avg_loss = total_loss / len(train_loader.dataset)

    # 计算指标
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = calculate_metrics(all_outputs, all_targets)

    return avg_loss, metrics

def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: Callable,
    device: torch.device
) -> Tuple[float, Dict[str, float]]:
    """
    验证模型

    Args:
        model: 模型
        val_loader: 验证数据加载器
        criterion: 损失函数
        device: 设备

    Returns:
        平均损失和指标
    """
    model.eval()
    total_loss = 0.0
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            # 解包批次数据
            if isinstance(batch, dict):
                inputs = batch.get('input_sequence', None)
                targets = batch.get('target_sequence', None)
                personal_features = batch.get('personal_features', None)
                image_features = batch.get('image_features', None)
                text_features = batch.get('text_features', None)
            else:
                inputs, targets = batch[0], batch[1]
                personal_features = batch[2] if len(batch) > 2 else None
                image_features = batch[3] if len(batch) > 3 else None
                text_features = batch[4] if len(batch) > 4 else None

            # 移动数据到设备
            inputs = inputs.to(device)
            targets = targets.to(device)
            if personal_features is not None:
                personal_features = personal_features.to(device)
            if image_features is not None:
                image_features = image_features.to(device)
            if text_features is not None:
                text_features = text_features.to(device)

            # 前向传播
            # 根据模型接口调用
            if hasattr(model, 'forward') and 'personal_features' in model.forward.__code__.co_varnames:
                outputs = model(inputs, personal_features, image_features, text_features)
            else:
                outputs = model(inputs)

            # 计算损失
            loss = criterion(outputs, targets)

            # 累积损失
            total_loss += loss.item() * inputs.size(0)

            # 收集输出和目标
            all_outputs.append(outputs)
            all_targets.append(targets)

    # 计算平均损失
    avg_loss = total_loss / len(val_loader.dataset)

    # 计算指标
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = calculate_metrics(all_outputs, all_targets)

    return avg_loss, metrics

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: Callable,
    optimizer: Optimizer,
    device: torch.device,
    num_epochs: int = 100,
    scheduler: Optional[Any] = None,
    early_stopping: Optional[EarlyStopping] = None,
    checkpoint_dir: Optional[str] = None,
    experiment_name: Optional[str] = None,
    clip_grad_norm: Optional[float] = None,
    verbose: bool = True,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    训练模型

    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        num_epochs: 训练轮次数
        scheduler: 学习率调度器
        early_stopping: 早停机制
        checkpoint_dir: 检查点目录
        experiment_name: 实验名称
        clip_grad_norm: 梯度裁剪范数
        verbose: 是否打印详细信息
        config: 配置

    Returns:
        训练后的模型和训练历史
    """
    # 初始化指标跟踪器
    tracker = MetricsTracker()

    # 设置实验名称
    if experiment_name is None:
        experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # 设置检查点目录
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_model_path = os.path.join(checkpoint_dir, f"{experiment_name}_best.pt")
        last_model_path = os.path.join(checkpoint_dir, f"{experiment_name}_last.pt")
    else:
        best_model_path = None
        last_model_path = None

    # 训练开始时间
    start_time = time.time()

    # 最佳验证损失
    best_val_loss = float('inf')

    # 训练循环
    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()

        # 训练一个轮次
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device,
            scheduler if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR) else None,
            clip_grad_norm
        )

        # 验证
        val_loss, val_metrics = validate(model, val_loader, criterion, device)

        # 更新学习率
        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']

        # 更新指标
        tracker.update_loss(train_loss, val_loss)
        tracker.update_learning_rate(current_lr)

        for metric_name, value in train_metrics.items():
            tracker.update_metric(metric_name, value, 'train')

        for metric_name, value in val_metrics.items():
            tracker.update_metric(metric_name, value, 'val')

        # 计算轮次时间
        epoch_time = time.time() - epoch_start_time

        # 打印进度
        if verbose:
            logger.info(
                f"轮次 {epoch}/{num_epochs} - "
                f"时间: {epoch_time:.2f}s - "
                f"训练损失: {train_loss:.6f} - "
                f"验证损失: {val_loss:.6f} - "
                f"学习率: {current_lr:.6f} - "
                f"MAE: {train_metrics['mae']:.4f}/{val_metrics['mae']:.4f}"
            )

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if best_model_path:
                save_model(
                    model, optimizer, epoch, val_loss,
                    {'train': train_metrics, 'val': val_metrics},
                    best_model_path, config
                )
                if verbose:
                    logger.info(f"保存最佳模型到 {best_model_path}")

        # 检查早停
        if early_stopping is not None:
            if early_stopping(epoch, val_loss):
                if verbose:
                    logger.info(f"早停触发，最佳轮次: {early_stopping.best_epoch}")
                break

    # 保存最后一个轮次的模型
    if last_model_path:
        save_model(
            model, optimizer, epoch, val_loss,
            {'train': train_metrics, 'val': val_metrics},
            last_model_path, config
        )
        if verbose:
            logger.info(f"保存最后模型到 {last_model_path}")

    # 计算总训练时间
    total_time = time.time() - start_time

    # 绘制并保存指标图表
    if checkpoint_dir:
        plots_dir = os.path.join(checkpoint_dir, f"{experiment_name}_plots")
        tracker.plot_metrics(plots_dir)

        # 保存指标
        metrics_path = os.path.join(checkpoint_dir, f"{experiment_name}_metrics.json")
        tracker.save_metrics(metrics_path)

    # 训练摘要
    training_summary = {
        'experiment_name': experiment_name,
        'total_epochs': epoch,
        'best_epoch': early_stopping.best_epoch if early_stopping is not None else epoch,
        'best_val_loss': best_val_loss,
        'final_train_loss': train_loss,
        'final_val_loss': val_loss,
        'final_train_metrics': train_metrics,
        'final_val_metrics': val_metrics,
        'total_time': total_time,
        'metrics_history': tracker.metrics
    }

    if verbose:
        logger.info(f"训练完成，总时间: {total_time:.2f}s")
        logger.info(f"最佳验证损失: {best_val_loss:.6f}")

    return model, training_summary

def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: Callable,
    device: torch.device,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    评估模型

    Args:
        model: 模型
        test_loader: 测试数据加载器
        criterion: 损失函数
        device: 设备
        verbose: 是否打印详细信息

    Returns:
        评估结果
    """
    test_loss, test_metrics = validate(model, test_loader, criterion, device)

    if verbose:
        logger.info(f"测试损失: {test_loss:.6f}")
        for name, value in test_metrics.items():
            logger.info(f"测试 {name}: {value:.6f}")

    return {
        'test_loss': test_loss,
        'test_metrics': test_metrics
    }

def get_experiment_name(prefix: str = "experiment") -> str:
    """
    生成实验名称

    Args:
        prefix: 前缀

    Returns:
        实验名称
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{prefix}_{timestamp}"
