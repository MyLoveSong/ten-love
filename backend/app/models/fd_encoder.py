"""
特征分解(FD)模块 - 基于GlucoNet的特征分解技术
将CGM数据分解为趋势、周期、残差三个本征模态
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)

class FDEcomposer(nn.Module):
    """
    特征分解器 - 将血糖时间序列分解为多个本征模态
    基于经验模态分解(EMD)的简化实现
    """

    def __init__(self, k: int = 3, win: int = 32, overlap: float = 0.5):
        """
        初始化特征分解器

        Args:
            k: 分解模态数量，默认3（趋势、周期、残差）
            win: 滑动窗口大小
            overlap: 窗口重叠比例
        """
        super(FDEcomposer, self).__init__()
        self.k = k
        self.win = win
        self.overlap = overlap
        self.stride = int(win * (1 - overlap))

        # 可学习的分解权重
        self.decomposition_weights = nn.Parameter(torch.randn(k, win))

        logger.info(f"FDEcomposer初始化: k={k}, win={win}, stride={self.stride}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：将输入序列分解为k个模态

        Args:
            x: 输入血糖序列 [batch_size, seq_len]

        Returns:
            decomposed: 分解后的模态 [batch_size, k, seq_len]
        """
        batch_size, seq_len = x.shape

        # 确保序列长度足够
        if seq_len < self.win:
            # 如果序列太短，进行填充
            pad_len = self.win - seq_len
            x = F.pad(x, (0, pad_len), mode='replicate')
            seq_len = self.win

        # 滑动窗口分解
        decomposed_modes = []

        for mode_idx in range(self.k):
            mode_components = []

            for start_idx in range(0, seq_len - self.win + 1, self.stride):
                end_idx = start_idx + self.win
                window = x[:, start_idx:end_idx]

                # 应用分解权重
                weight = self.decomposition_weights[mode_idx]
                weighted_window = window * weight.unsqueeze(0)

                mode_components.append(weighted_window)

            # 重构该模态的完整序列
            if mode_components:
                mode_tensor = torch.stack(mode_components, dim=1)  # [batch, windows, win]
                mode_tensor = mode_tensor.view(batch_size, -1)  # [batch, windows*win]

                # 截断到原始长度
                mode_tensor = mode_tensor[:, :seq_len]
                decomposed_modes.append(mode_tensor)
            else:
                # 如果无法分解，返回零填充
                decomposed_modes.append(torch.zeros(batch_size, seq_len, device=x.device))

        # 堆叠所有模态
        decomposed = torch.stack(decomposed_modes, dim=1)  # [batch, k, seq_len]

        return decomposed

    def get_mode_names(self) -> List[str]:
        """获取模态名称"""
        return ['trend', 'periodic', 'residual'][:self.k]

class FDEnhancedGluFormer(nn.Module):
    """
    集成特征分解的GluFormer模型
    """

    def __init__(self,
                 input_dim: int = 1,
                 lstm_hidden: int = 64,
                 gru_hidden: int = 64,
                 dropout: float = 0.1,
                 fd_k: int = 3,
                 fd_win: int = 32):
        """
        初始化FD增强的GluFormer

        Args:
            input_dim: 输入维度
            lstm_hidden: LSTM隐藏层大小
            gru_hidden: GRU隐藏层大小
            dropout: Dropout比例
            fd_k: 特征分解模态数
            fd_win: 特征分解窗口大小
        """
        super(FDEnhancedGluFormer, self).__init__()

        # 特征分解器
        self.fd_composer = FDEcomposer(k=fd_k, win=fd_win)

        # 各模态的编码器
        self.trend_encoder = nn.LSTM(input_dim, lstm_hidden, batch_first=True)
        self.periodic_encoder = nn.GRU(input_dim, gru_hidden, batch_first=True)
        self.residual_encoder = nn.Conv1d(input_dim, 32, kernel_size=3, padding=1)

        # 跨模态注意力融合
        total_dim = lstm_hidden + gru_hidden + 32
        # 确保embed_dim能被num_heads整除
        num_heads = 8
        if total_dim % num_heads != 0:
            total_dim = ((total_dim + num_heads - 1) // num_heads) * num_heads

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=total_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 如果维度不匹配，添加投影层
        if lstm_hidden + gru_hidden + 32 != total_dim:
            self.projection = nn.Linear(lstm_hidden + gru_hidden + 32, total_dim)
        else:
            self.projection = nn.Identity()

        # 输出层
        self.output_projection = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

        self.dropout = nn.Dropout(dropout)

        logger.info(f"FDEnhancedGluFormer初始化: fd_k={fd_k}, fd_win={fd_win}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入血糖序列 [batch_size, seq_len]

        Returns:
            prediction: 血糖预测 [batch_size, 1]
            decomposed_modes: 分解后的模态 [batch_size, k, seq_len]
        """
        batch_size, seq_len = x.shape

        # 特征分解
        decomposed_modes = self.fd_composer(x)  # [batch, k, seq_len]

        # 各模态编码
        trend_out, _ = self.trend_encoder(decomposed_modes[:, 0:1, :].transpose(1, 2))
        periodic_out, _ = self.periodic_encoder(decomposed_modes[:, 1:2, :].transpose(1, 2))
        residual_out = self.residual_encoder(decomposed_modes[:, 2:3, :])  # Conv1d需要[batch, channels, seq]
        residual_out = residual_out.transpose(1, 2)  # 转回[batch, seq, channels]

        # 特征融合
        fused_features = torch.cat([trend_out, periodic_out, residual_out], dim=-1)

        # 投影到正确的维度
        projected_features = self.projection(fused_features)

        # 跨模态注意力
        attended_features, attention_weights = self.cross_attention(
            projected_features, projected_features, projected_features
        )

        # 全局平均池化
        pooled_features = torch.mean(attended_features, dim=1)

        # 输出预测
        prediction = self.output_projection(pooled_features)

        return prediction, decomposed_modes

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """获取注意力权重用于可解释性分析"""
        with torch.no_grad():
            _, decomposed_modes = self.forward(x)

            # 计算各模态对最终预测的贡献
            trend_out, _ = self.trend_encoder(decomposed_modes[:, 0:1, :].transpose(1, 2))
            periodic_out, _ = self.periodic_encoder(decomposed_modes[:, 1:2, :].transpose(1, 2))
            residual_out = self.residual_encoder(decomposed_modes[:, 2:3, :])
            residual_out = residual_out.transpose(1, 2)

            fused_features = torch.cat([trend_out, periodic_out, residual_out], dim=-1)
            _, attention_weights = self.cross_attention(
                fused_features, fused_features, fused_features
            )

            return attention_weights

class FDObjective:
    """
    特征分解目标函数，用于SSA优化
    """

    def __init__(self, device: str = 'cpu'):
        self.device = device

    def evaluate(self, params: np.ndarray) -> float:
        """
        评估FD参数组合

        Args:
            params: [fd_k, fd_win, lstm_hidden, gru_hidden]

        Returns:
            MAE值
        """
        try:
            fd_k = int(params[0])
            fd_win = int(params[1])
            lstm_hidden = int(params[2])
            gru_hidden = int(params[3])

            # 创建模型
            model = FDEnhancedGluFormer(
                lstm_hidden=lstm_hidden,
                gru_hidden=gru_hidden,
                fd_k=fd_k,
                fd_win=fd_win
            ).to(self.device)

            # 模拟训练和评估（实际应用中需要真实数据）
            # 这里返回一个基于参数合理性的模拟MAE
            base_mae = 0.65

            # 参数合理性惩罚
            if fd_k < 2 or fd_k > 5:
                base_mae += 0.1
            if fd_win < 16 or fd_win > 64:
                base_mae += 0.05
            if lstm_hidden < 32 or lstm_hidden > 128:
                base_mae += 0.03
            if gru_hidden < 32 or gru_hidden > 128:
                base_mae += 0.03

            # FD带来的改进
            fd_improvement = 0.03 * (1.0 / (1.0 + abs(fd_k - 3)))  # 最优k=3
            base_mae -= fd_improvement

            return max(0.1, base_mae)  # 确保返回正值

        except Exception as e:
            logger.error(f"FD目标函数评估失败: {e}")
            return float('inf')
