"""
增强版自适应特征门控机制 (Enhanced Adaptive Feature Gating Mechanism)

改进点:
1. 增加网络深度 (3层 → 5层)
2. 增加隐藏层维度 (16 → 32)
3. 添加 BatchNorm 稳定训练
4. 添加 Dropout 正则化
5. 增加特征交互层
6. 多尺度门控输出

预期改进: +8.7% → +15-20%
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class EnhancedFeatureGating(nn.Module):
    """
    增强版自适应特征门控机制

    改进策略:
    1. 更深的网络 (5层 MLP)
    2. 更大的隐藏维度 (32)
    3. BatchNorm 稳定化
    4. Dropout 正则化
    5. 特征交互层
    6. 动态门控缩放

    Args:
        feature_dim: 临床特征维度 (default: 6)
        hidden_dim: 隐藏层维度 (default: 32)
        output_dim: 输出门控维度 (default: 1)
        dropout: Dropout 比率 (default: 0.1)
        use_residual: 是否使用残差连接 (default: True)
        use_feature_interaction: 是否使用特征交互 (default: True)
    """

    def __init__(
        self,
        feature_dim: int = 6,
        hidden_dim: int = 32,
        output_dim: int = 1,
        dropout: float = 0.1,
        use_residual: bool = True,
        use_feature_interaction: bool = True,
        use_multi_scale: bool = True,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_residual = use_residual
        self.use_feature_interaction = use_feature_interaction
        self.use_multi_scale = use_multi_scale

        # 特征交互层 (捕获特征间关系)
        if use_feature_interaction:
            self.feature_interaction = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.Tanh(),
            )
            interaction_output_dim = hidden_dim
        else:
            self.feature_interaction = None
            interaction_output_dim = feature_dim

        # 主门控网络 (5层深度)
        self.gate_network = nn.Sequential(
            # Layer 1
            nn.Linear(interaction_output_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            # Layer 2
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            # Layer 3
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            # Layer 4
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            # Layer 5 (输出层)
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()  # 输出 [0, 1] 范围内的门控权重
        )

        # 多尺度门控 (可选)
        if use_multi_scale:
            self.scale_gate = nn.Sequential(
                nn.Linear(interaction_output_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 3),  # 3个尺度
                nn.Softmax(dim=-1)
            )

        # 特征重要性投影（用于可解释性分析）
        self.feature_importance = nn.Linear(feature_dim, 1, bias=False)

        # 残差权重（可学习）
        if use_residual:
            self.residual_weight = nn.Parameter(torch.tensor(0.1))

        # 动态门控缩放因子
        self.gate_scale = nn.Parameter(torch.tensor(1.0))

        # 初始化
        self._init_weights()

        logger.info(
            f"EnhancedFeatureGating initialized: "
            f"feature_dim={feature_dim}, hidden_dim={hidden_dim}, "
            f"output_dim={output_dim}, params={self.count_parameters()}"
        )

    def _init_weights(self):
        """Xavier初始化 + BatchNorm 偏置置零"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def count_parameters(self) -> int:
        """计算参数量"""
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        clinical_features: torch.Tensor,
        base_representation: Optional[torch.Tensor] = None,
        return_importance: bool = False,
        return_gate_info: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        """
        前向传播

        Args:
            clinical_features: 临床特征 [batch, feature_dim]
            base_representation: 基础表示 [batch, d_model]
            return_importance: 是否返回特征重要性
            return_gate_info: 是否返回门控信息

        Returns:
            gated_output: 门控后的输出
            feature_importance: 特征重要性 (如果return_importance=True)
            gate_info: 门控信息字典 (如果return_gate_info=True)
        """
        batch_size = clinical_features.size(0)

        # 特征交互
        if self.use_feature_interaction and self.feature_interaction is not None:
            # 需要处理 BatchNorm 的维度问题
            if batch_size > 1:
                clinical_repr = self.feature_interaction(clinical_features)
            else:
                # 单样本时跳过 BatchNorm，直接使用 Tanh
                clinical_repr = torch.tanh(clinical_features)
        else:
            clinical_repr = clinical_features

        # 计算门控权重
        gate_weights = self.gate_network(clinical_repr)  # [batch, output_dim]

        # 多尺度门控 (可选)
        if self.use_multi_scale:
            scale_weights = self.scale_gate(clinical_repr)  # [batch, 3]
        else:
            scale_weights = None

        # 门控信息
        gate_info = {
            'gate_weights': gate_weights,
            'scale_weights': scale_weights if scale_weights is not None else None,
            'gate_scale': self.gate_scale,
        }

        # 特征重要性 (如果需要)
        if return_importance:
            raw_importance = torch.abs(self.feature_importance(clinical_features))
            feature_importance = F.softmax(raw_importance, dim=1)
        else:
            feature_importance = None

        # 应用门控
        if base_representation is not None:
            # 缩放门控权重
            scaled_gate = gate_weights * self.gate_scale

            if scaled_gate.dim() == 1:
                scaled_gate = scaled_gate.unsqueeze(-1)

            if scaled_gate.size(-1) == 1:
                scaled_gate = scaled_gate.expand_as(base_representation)

            if self.use_residual:
                residual = torch.sigmoid(self.residual_weight)
                gated_output = scaled_gate * base_representation + (1 - scaled_gate) * base_representation * residual
            else:
                gated_output = scaled_gate * base_representation
        else:
            gated_output = scaled_gate if scaled_gate.dim() > 1 else scaled_gate.squeeze(-1)

        if return_importance or return_gate_info:
            return gated_output, feature_importance, gate_info
        else:
            return gated_output


class EnhancedCulturalGateGluFormer(nn.Module):
    """
    集成增强版 Cultural Gate 的 GluFormer 模型

    用于对比实验:
    - Baseline: 无 Cultural Gate
    - Original: 原始 Cultural Gate (2参数)
    - Enhanced: 增强 Cultural Gate (预计 +12-15% 改进)
    """

    def __init__(self, config: Dict):
        super().__init__()

        d_model = config.get('d_model', 64)
        seq_len = config.get('seq_len', 48)
        clinical_dim = config.get('clinical_dim', 6)
        output_horizon = config.get('output_horizon', 6)
        dropout = config.get('dropout', 0.1)

        # CGM 编码
        self.cgm_proj = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.get('nhead', 4),
            dim_feedforward=config.get('dim_feedforward', 128),
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.get('num_layers', 2))
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        # 临床特征编码
        self.clinical_embed = nn.Linear(clinical_dim, d_model)

        # 增强版 Cultural Gate
        self.cultural_gate = EnhancedFeatureGating(
            feature_dim=clinical_dim,
            hidden_dim=32,
            output_dim=1,
            dropout=dropout,
            use_residual=True,
            use_feature_interaction=True,
            use_multi_scale=True,
        )

        # 预测头
        self.predictor = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_horizon)
        )

        # 初始化
        self._init_weights()

        logger.info(f"EnhancedCulturalGateGluFormer initialized: params={self.count_parameters()}")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, cgm_seq: torch.Tensor, clinical: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            cgm_seq: CGM 序列 [batch, seq_len]
            clinical: 临床特征 [batch, clinical_dim]

        Returns:
            predictions: 预测值 [batch, output_horizon]
        """
        if cgm_seq.dim() == 2:
            cgm_seq = cgm_seq.unsqueeze(-1)

        # CGM 编码
        cgm_emb = self.cgm_proj(cgm_seq) + self.pos_embed[:, :cgm_seq.size(1), :]
        cgm_encoded = self.transformer(cgm_emb)
        cgm_out = cgm_encoded[:, -1, :]  # [batch, d_model]

        # 临床特征编码
        clinical_out = self.clinical_embed(clinical)

        # 增强版 Cultural Gate
        gated_cgm, _, gate_info = self.cultural_gate(
            clinical,
            cgm_out,
            return_importance=False,
            return_gate_info=True
        )

        # 融合
        fused = torch.cat([gated_cgm, clinical_out], dim=-1)

        # 预测
        output = self.predictor(fused)

        return output


if __name__ == "__main__":
    # 测试
    config = {
        'd_model': 64,
        'seq_len': 48,
        'clinical_dim': 6,
        'output_horizon': 6,
        'nhead': 4,
        'dim_feedforward': 128,
        'num_layers': 2,
        'dropout': 0.1,
    }

    model = EnhancedCulturalGateGluFormer(config)

    # 参数统计
    total_params = model.count_parameters()
    gate_params = model.cultural_gate.count_parameters()

    print(f"总参数量: {total_params:,}")
    print(f"Cultural Gate 参数量: {gate_params:,}")
    print(f"Gate 占比: {gate_params/total_params*100:.2f}%")

    # 测试前向传播
    batch_size = 4
    cgm_seq = torch.randn(batch_size, 48)
    clinical = torch.randn(batch_size, 6)

    with torch.no_grad():
        output = model(cgm_seq, clinical)

    print(f"输入形状: CGM={cgm_seq.shape}, Clinical={clinical.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出范围: [{output.min():.3f}, {output.max():.3f}]")
