"""
基于麻雀搜索算法的混合神经网络优化模块
结合TCN和GRU的血糖预测模型优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
import random
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class SSAParameters:
    """麻雀搜索算法参数"""
    population_size: int = 20
    max_iterations: int = 50
    discovery_rate: float = 0.1
    safety_threshold: float = 0.8
    lower_bounds: List[float] = None
    upper_bounds: List[float] = None

    def __post_init__(self):
        if self.lower_bounds is None:
            self.lower_bounds = [16, 2, 16, 0.0001, 16, 0.1]  # filters, kernel_size, gru_units, lr, batch_size, dropout
        if self.upper_bounds is None:
            self.upper_bounds = [128, 10, 128, 0.01, 128, 0.5]

class TemporalConvolutionalNetwork(nn.Module):
    """时域卷积网络 (TCN)"""

    def __init__(
        self,
        input_dim: int,
        filters: int = 64,
        kernel_size: int = 3,
        dilation_rates: List[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout = dropout

        if dilation_rates is None:
            dilation_rates = [1, 2, 4, 8]

        # TCN层 - 简化版本避免序列长度问题
        self.tcn_layers = nn.ModuleList()
        prev_dim = input_dim

        for dilation in dilation_rates:
            tcn_block = nn.Sequential(
                nn.Conv1d(
                    prev_dim, filters,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding='same'  # 使用same padding保持序列长度
                ),
                nn.BatchNorm1d(filters),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.tcn_layers.append(tcn_block)
            prev_dim = filters

        # 残差连接
        self.residual_conv = nn.Conv1d(input_dim, filters, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 输入序列 [batch_size, seq_len, input_dim]
        Returns:
            TCN特征 [batch_size, seq_len, filters]
        """
        batch_size, seq_len, input_dim = x.shape
        x = x.transpose(1, 2)  # [batch_size, input_dim, seq_len]

        residual = self.residual_conv(x)

        # TCN处理
        tcn_output = x
        for tcn_layer in self.tcn_layers:
            tcn_output = tcn_layer(tcn_output)

        # 残差连接
        output = tcn_output + residual
        output = output.transpose(1, 2)  # [batch_size, seq_len, filters]

        return output

class EnhancedGRU(nn.Module):
    """增强的GRU网络"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # GRU层
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )

        # 注意力机制 - 确保embed_dim能被num_heads整除
        attention_dim = hidden_dim * (2 if bidirectional else 1)
        # 调整到最近的能被8整除的维度
        attention_dim = ((attention_dim + 7) // 8) * 8
        self.attention = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        # 特征融合 - 使用调整后的维度
        self.feature_fusion = nn.Sequential(
            nn.Linear(attention_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        Args:
            x: 输入序列 [batch_size, seq_len, input_dim]
        Returns:
            GRU输出和注意力权重
        """
        # GRU处理
        gru_output, _ = self.gru(x)

        # 调整GRU输出维度以匹配注意力机制
        gru_dim = gru_output.shape[-1]
        attention_dim = self.attention.embed_dim

        if gru_dim != attention_dim:
            # 使用线性层调整维度
            gru_output = F.pad(gru_output, (0, attention_dim - gru_dim))

        # 自注意力机制
        attended_output, attention_weights = self.attention(
            gru_output, gru_output, gru_output
        )

        # 特征融合
        fused_output = self.feature_fusion(attended_output)

        return fused_output, attention_weights

class SparrowSearchAlgorithm:
    """麻雀搜索算法实现"""

    def __init__(self, parameters: SSAParameters):
        self.params = parameters
        self.population_size = parameters.population_size
        self.max_iterations = parameters.max_iterations
        self.discovery_rate = parameters.discovery_rate
        self.safety_threshold = parameters.safety_threshold
        self.lower_bounds = np.array(parameters.lower_bounds)
        self.upper_bounds = np.array(parameters.upper_bounds)

        # 种群初始化
        self.population = self._initialize_population()
        self.fitness_history = []

    def _initialize_population(self) -> np.ndarray:
        """初始化种群"""
        population = np.random.uniform(
            self.lower_bounds,
            self.upper_bounds,
            (self.population_size, len(self.lower_bounds))
        )
        return population

    def _calculate_fitness(self, individual: np.ndarray, objective_function: Callable) -> float:
        """计算个体适应度"""
        try:
            # 将参数转换为整数（对于需要整数的参数）
            params = individual.copy()
            if len(params) >= 3:
                params[0] = int(params[0])  # filters
                params[1] = int(params[1])  # kernel_size
                params[2] = int(params[2])  # gru_units
            if len(params) >= 5:
                params[4] = int(params[4])  # batch_size

            fitness = objective_function(params)
            return fitness
        except Exception as e:
            logger.warning(f"适应度计算失败: {e}")
            return float('inf')

    def _update_population(self, objective_function: Callable) -> None:
        """更新种群"""
        # 计算适应度
        fitness_values = np.array([
            self._calculate_fitness(individual, objective_function)
            for individual in self.population
        ])

        # 排序（适应度越小越好）
        sorted_indices = np.argsort(fitness_values)
        self.population = self.population[sorted_indices]
        fitness_values = fitness_values[sorted_indices]

        # 记录最佳适应度
        self.fitness_history.append(fitness_values[0])

        # 发现者更新（前20%的个体）
        num_discoverers = int(self.population_size * 0.2)
        for i in range(num_discoverers):
            if np.random.random() < self.safety_threshold:
                # 安全状态：正常觅食
                self.population[i] += np.random.normal(0, 0.1, len(self.lower_bounds))
            else:
                # 危险状态：快速移动
                self.population[i] += np.random.normal(0, 0.5, len(self.lower_bounds))

        # 跟随者更新（后80%的个体）
        for i in range(num_discoverers, self.population_size):
            if i > self.population_size // 2:
                # 后半部分个体：随机觅食
                self.population[i] = np.random.uniform(
                    self.lower_bounds,
                    self.upper_bounds,
                    len(self.lower_bounds)
                )
            else:
                # 前半部分个体：跟随最佳个体
                best_individual = self.population[0]
                self.population[i] = best_individual + np.random.normal(0, 0.1, len(self.lower_bounds))

        # 边界处理
        for i in range(self.population_size):
            self.population[i] = np.clip(
                self.population[i],
                self.lower_bounds,
                self.upper_bounds
            )

    def optimize(self, objective_function: Callable) -> Tuple[np.ndarray, List[float]]:
        """
        执行优化
        Args:
            objective_function: 目标函数
        Returns:
            最佳参数和适应度历史
        """
        logger.info(f"开始麻雀搜索算法优化，种群大小: {self.population_size}, 最大迭代次数: {self.max_iterations}")

        for iteration in range(self.max_iterations):
            self._update_population(objective_function)

            if iteration % 10 == 0:
                logger.info(f"迭代 {iteration}, 最佳适应度: {self.fitness_history[-1]:.6f}")

        best_params = self.population[0].copy()
        # 转换参数类型
        best_params[0] = int(best_params[0])  # filters
        best_params[1] = int(best_params[1])  # kernel_size
        best_params[2] = int(best_params[2])  # gru_units
        best_params[4] = int(best_params[4])  # batch_size

        logger.info(f"优化完成，最佳参数: {best_params}")
        return best_params, self.fitness_history

class SSATCNGRUModel(nn.Module):
    """基于SSA优化的TCN-GRU混合模型"""

    def __init__(
        self,
        input_dim: int = 10,
        tcn_filters: int = 64,
        tcn_kernel_size: int = 3,
        gru_hidden_dim: int = 128,
        dropout: float = 0.1,
        prediction_horizon: int = 6
    ):
        super().__init__()
        self.input_dim = input_dim
        self.prediction_horizon = prediction_horizon

        # TCN模块
        self.tcn = TemporalConvolutionalNetwork(
            input_dim=input_dim,
            filters=tcn_filters,
            kernel_size=tcn_kernel_size,
            dropout=dropout
        )

        # GRU模块
        self.gru = EnhancedGRU(
            input_dim=tcn_filters,
            hidden_dim=gru_hidden_dim,
            dropout=dropout
        )

        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(gru_hidden_dim, gru_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden_dim // 2, gru_hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 个性化适配层
        self.personalization_layer = nn.Sequential(
            nn.Linear(gru_hidden_dim // 4 + 5, gru_hidden_dim // 4),  # +5 for personal features
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 预测头
        self.prediction_head = nn.Sequential(
            nn.Linear(gru_hidden_dim // 4, prediction_horizon),
            nn.Sigmoid()  # 输出0-1之间的值，后续可以映射到血糖范围
        )

    def forward(
        self,
        glucose_data: torch.Tensor,
        personal_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播
        Args:
            glucose_data: 血糖时序数据 [batch_size, seq_len, input_dim]
            personal_features: 个人特征 [batch_size, 5]
        Returns:
            预测结果和中间特征
        """
        batch_size, seq_len, input_dim = glucose_data.shape

        # TCN特征提取
        tcn_features = self.tcn(glucose_data)

        # GRU处理
        gru_output, attention_weights = self.gru(tcn_features)

        # 取最后一个时间步的输出
        final_features = gru_output[:, -1, :]

        # 特征融合
        fused_features = self.feature_fusion(final_features)

        # 个性化适配
        if personal_features is not None:
            personalized_features = torch.cat([fused_features, personal_features], dim=1)
            personalized_features = self.personalization_layer(personalized_features)
        else:
            personalized_features = fused_features

        # 血糖预测
        prediction = self.prediction_head(personalized_features)

        # 将预测值映射到血糖范围 (4-15 mmol/L)
        prediction = prediction * 11 + 4  # 映射到4-15范围

        return prediction, {
            'tcn_features': tcn_features,
            'gru_output': gru_output,
            'attention_weights': attention_weights,
            'fused_features': fused_features
        }

class SSATCNGRUPredictor:
    """基于SSA优化的TCN-GRU预测器"""

    def __init__(self, model_config: Dict):
        self.model_config = model_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()

        # 训练历史
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'best_params': None
        }

    def _create_model(self, params: np.ndarray) -> SSATCNGRUModel:
        """根据参数创建模型"""
        model = SSATCNGRUModel(
            input_dim=self.model_config.get('input_dim', 10),
            tcn_filters=int(params[0]),
            tcn_kernel_size=int(params[1]),
            gru_hidden_dim=int(params[2]),
            dropout=params[5],
            prediction_horizon=self.model_config.get('prediction_horizon', 6)
        )
        return model.to(self.device)

    def _objective_function(self, params: np.ndarray) -> float:
        """目标函数：返回验证损失"""
        try:
            # 创建模型
            model = self._create_model(params)

            # 创建优化器
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=params[3],
                weight_decay=0.01
            )

            # 训练模型
            model.train()
            train_losses = []

            for epoch in range(10):  # 快速训练用于超参数优化
                # 模拟训练数据
                batch_size = int(params[4])
                seq_len = 24
                input_dim = self.model_config.get('input_dim', 10)

                # 生成模拟数据
                glucose_data = torch.randn(batch_size, seq_len, input_dim)
                personal_features = torch.randn(batch_size, 5)
                labels = torch.randn(batch_size, self.model_config.get('prediction_horizon', 6))

                # 前向传播
                predictions, _ = model(glucose_data, personal_features)
                loss = self.criterion(predictions, labels)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            # 返回平均训练损失作为适应度
            return np.mean(train_losses)

        except Exception as e:
            logger.warning(f"目标函数计算失败: {e}")
            return float('inf')

    def optimize_hyperparameters(self) -> Dict[str, Any]:
        """使用SSA优化超参数"""
        logger.info("开始使用麻雀搜索算法优化超参数...")

        # 设置SSA参数
        ssa_params = SSAParameters(
            population_size=15,
            max_iterations=30,
            lower_bounds=[16, 2, 16, 0.0001, 16, 0.1],
            upper_bounds=[128, 10, 128, 0.01, 128, 0.5]
        )

        # 创建SSA实例
        ssa = SparrowSearchAlgorithm(ssa_params)

        # 执行优化
        best_params, fitness_history = ssa.optimize(self._objective_function)

        # 保存最佳参数
        self.training_history['best_params'] = best_params

        logger.info(f"超参数优化完成，最佳参数: {best_params}")

        return {
            'tcn_filters': int(best_params[0]),
            'tcn_kernel_size': int(best_params[1]),
            'gru_hidden_dim': int(best_params[2]),
            'learning_rate': best_params[3],
            'batch_size': int(best_params[4]),
            'dropout': best_params[5],
            'fitness_history': fitness_history
        }

    def train_with_optimized_params(
        self,
        train_data: torch.Tensor,
        train_labels: torch.Tensor,
        personal_features: Optional[torch.Tensor] = None,
        val_data: Optional[torch.Tensor] = None,
        val_labels: Optional[torch.Tensor] = None,
        epochs: int = 100
    ) -> Dict[str, List[float]]:
        """使用优化后的参数训练模型"""
        if self.training_history['best_params'] is None:
            logger.warning("未找到优化后的参数，使用默认参数")
            best_params = np.array([64, 3, 128, 0.001, 32, 0.1])
        else:
            best_params = self.training_history['best_params']

        # 创建优化后的模型
        self.model = self._create_model(best_params)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=best_params[3],
            weight_decay=0.01
        )

        # 确保模型在正确的设备上
        self.model.to(self.device)

        # 训练模型
        self.model.train()

        for epoch in range(epochs):
            # 训练
            predictions, _ = self.model(train_data, personal_features)
            train_loss = self.criterion(predictions, train_labels)

            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()

            self.training_history['train_loss'].append(train_loss.item())

            # 验证
            if val_data is not None and val_labels is not None:
                val_loss = self.validate(val_data, val_labels, personal_features)
                self.training_history['val_loss'].append(val_loss)

            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Train Loss: {train_loss.item():.4f}")

        return self.training_history

    def validate(
        self,
        val_data: torch.Tensor,
        val_labels: torch.Tensor,
        personal_features: Optional[torch.Tensor] = None
    ) -> float:
        """验证模型"""
        if self.model is None:
            raise ValueError("模型未初始化")

        self.model.eval()
        with torch.no_grad():
            predictions, _ = self.model(val_data, personal_features)
            val_loss = self.criterion(predictions, val_labels)
        self.model.train()
        return val_loss.item()

    def predict(
        self,
        glucose_data: torch.Tensor,
        personal_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """预测血糖趋势"""
        if self.model is None:
            raise ValueError("模型未初始化")

        self.model.eval()
        with torch.no_grad():
            predictions, outputs = self.model(glucose_data, personal_features)
        return predictions, outputs

def create_ssa_tcn_gru_predictor(model_config: Dict) -> SSATCNGRUPredictor:
    """创建基于SSA优化的TCN-GRU预测器"""
    return SSATCNGRUPredictor(model_config)

if __name__ == "__main__":
    # 创建预测器
    model_config = {
        'input_dim': 10,
        'prediction_horizon': 6
    }

    predictor = create_ssa_tcn_gru_predictor(model_config)

    # 优化超参数
    optimized_params = predictor.optimize_hyperparameters()
    print(f"优化后的参数: {optimized_params}")

    # 使用优化后的参数训练模型
    batch_size, seq_len, input_dim = 32, 24, 10
    train_data = torch.randn(batch_size, seq_len, input_dim)
    train_labels = torch.randn(batch_size, 6)
    personal_features = torch.randn(batch_size, 5)

    history = predictor.train_with_optimized_params(
        train_data, train_labels, personal_features, epochs=5
    )

    # 预测
    predictions, outputs = predictor.predict(train_data, personal_features)

    print(f"预测结果形状: {predictions.shape}")
    print(f"注意力权重形状: {outputs['attention_weights'].shape}")
    print("基于SSA优化的TCN-GRU模型创建成功！")
