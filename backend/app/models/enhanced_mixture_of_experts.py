"""
增强混合专家系统 (Enhanced Mixture of Experts)
基于医学知识增强的可解释混合专家决策系统
实现轻量级MoE模型，支持动态路由门控和低资源微调
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass
import math

logger = logging.getLogger(__name__)

@dataclass
class MoEConfig:
    """混合专家系统配置"""
    input_dim: int = 256
    expert_dim: int = 128
    num_experts: int = 4
    num_experts_per_token: int = 2
    dropout: float = 0.1
    use_auxiliary_loss: bool = True
    auxiliary_loss_weight: float = 0.01
    use_lora: bool = True
    lora_rank: int = 16
    use_dynamic_routing: bool = True
    routing_temperature: float = 1.0

class LoRALayer(nn.Module):
    """低秩适应层 (LoRA)"""

    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: float = 1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha

        # LoRA参数
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # 原始权重（冻结）
        self.original_weight = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始输出
        original_output = F.linear(x, self.original_weight)

        # LoRA输出 - 修复维度问题
        # x: [batch_size, seq_len, input_dim]
        # lora_A: [rank, input_dim] -> 需要转置为 [input_dim, rank]
        # lora_B: [out_features, rank] -> 需要转置为 [rank, out_features]
        lora_output = F.linear(F.linear(x, self.lora_A.T), self.lora_B.T)

        return original_output + (self.alpha / self.rank) * lora_output

class LightweightExpert(nn.Module):
    """轻量级专家网络"""

    def __init__(self, input_dim: int, expert_dim: int, output_dim: int, use_lora: bool = True, lora_rank: int = 16):
        super().__init__()
        self.input_dim = input_dim
        self.expert_dim = expert_dim
        self.output_dim = output_dim
        self.use_lora = use_lora

        if use_lora:
            # 使用LoRA的轻量级专家
            self.expert_layers = nn.Sequential(
                LoRALayer(input_dim, expert_dim, lora_rank),
                nn.ReLU(),
                nn.Dropout(0.1),
                LoRALayer(expert_dim, expert_dim, lora_rank),
                nn.ReLU(),
                nn.Dropout(0.1),
                LoRALayer(expert_dim, output_dim, lora_rank)
            )
        else:
            # 传统专家网络
            self.expert_layers = nn.Sequential(
                nn.Linear(input_dim, expert_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(expert_dim, expert_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(expert_dim, output_dim)
            )

        # 专家特定参数
        self.expert_scale = nn.Parameter(torch.ones(1))
        self.expert_bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expert_output = self.expert_layers(x)
        return self.expert_scale * expert_output + self.expert_bias

class DynamicRoutingGating(nn.Module):
    """动态路由门控网络"""

    def __init__(self, input_dim: int, num_experts: int, num_experts_per_token: int, temperature: float = 1.0):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.temperature = temperature

        # 门控网络
        self.gating_network = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, num_experts)
        )

        # 动态温度调整
        self.temperature_controller = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )

        # 专家负载均衡
        self.load_balancer = nn.Parameter(torch.ones(num_experts))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        动态路由门控
        Args:
            x: 输入特征 [batch_size, seq_len, input_dim]
        Returns:
            门控权重和专家选择掩码
        """
        batch_size, seq_len, input_dim = x.shape

        # 计算门控分数
        gate_scores = self.gating_network(x)  # [batch_size, seq_len, num_experts]

        # 动态温度调整 - 修复维度问题
        # 将 [batch_size, seq_len, input_dim] 展平为 [batch_size, input_dim] 用于温度控制器
        x_flat = x.view(batch_size, -1)  # [batch_size, seq_len * input_dim]
        # 或者使用平均池化来保持维度兼容性
        x_pooled = x.mean(dim=1)  # [batch_size, input_dim]
        temperature_factor = self.temperature_controller(x_pooled)  # [batch_size, 1]
        temperature_factor = temperature_factor.unsqueeze(1)  # [batch_size, 1, 1] 用于广播
        adjusted_temperature = self.temperature * (0.5 + 0.5 * temperature_factor)

        # 应用温度
        gate_scores = gate_scores / adjusted_temperature

        # Top-k专家选择
        top_k_scores, top_k_indices = torch.topk(gate_scores, self.num_experts_per_token, dim=-1)

        # 创建专家选择掩码
        expert_mask = torch.zeros_like(gate_scores)
        expert_mask.scatter_(-1, top_k_indices, 1.0)

        # 软门控权重
        gate_weights = F.softmax(top_k_scores, dim=-1)

        # 负载均衡
        expert_usage = expert_mask.mean(dim=(0, 1))  # [num_experts]
        load_balance_loss = torch.var(expert_usage * self.load_balancer)

        return gate_weights, expert_mask, load_balance_loss

class BiochemicalAnalysisExpert(LightweightExpert):
    """生化数据分析专家"""

    def __init__(self, input_dim: int, expert_dim: int, output_dim: int, use_lora: bool = True):
        super().__init__(input_dim, expert_dim, output_dim, use_lora)

        # 生化指标特定处理
        self.biochemical_processor = nn.Sequential(
            nn.Linear(input_dim, expert_dim),
            nn.ReLU(),
            nn.LayerNorm(expert_dim)
        )

        # 时间序列分析
        self.temporal_analyzer = nn.LSTM(
            input_size=expert_dim,
            hidden_size=expert_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 生化指标处理
        biochemical_features = self.biochemical_processor(x)

        # 时间序列分析
        temporal_features, _ = self.temporal_analyzer(biochemical_features)

        # 专家输出
        expert_output = self.expert_layers(temporal_features)

        return expert_output

class MedicalDiagnosisExpert(LightweightExpert):
    """医学诊断专家"""

    def __init__(self, input_dim: int, expert_dim: int, output_dim: int, use_lora: bool = True):
        super().__init__(input_dim, expert_dim, output_dim, use_lora)

        # 医学知识编码器
        self.medical_encoder = nn.Sequential(
            nn.Linear(input_dim, expert_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(expert_dim, expert_dim),
            nn.ReLU()
        )

        # 诊断决策层
        self.diagnosis_layer = nn.Sequential(
            nn.Linear(expert_dim, expert_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(expert_dim // 2, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 医学知识编码
        medical_features = self.medical_encoder(x)

        # 诊断决策
        diagnosis_output = self.diagnosis_layer(medical_features)

        return diagnosis_output

class LifestyleAssessmentExpert(LightweightExpert):
    """生活习惯评估专家"""

    def __init__(self, input_dim: int, expert_dim: int, output_dim: int, use_lora: bool = True):
        super().__init__(input_dim, expert_dim, output_dim, use_lora)

        # 多模态特征融合
        self.multimodal_fusion = nn.Sequential(
            nn.Linear(input_dim, expert_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 生活习惯分析
        self.lifestyle_analyzer = nn.Sequential(
            nn.Linear(expert_dim, expert_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(expert_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 多模态特征融合
        fused_features = self.multimodal_fusion(x)

        # 生活习惯分析
        lifestyle_output = self.lifestyle_analyzer(fused_features)

        return lifestyle_output

class EnhancedMixtureOfExperts(nn.Module):
    """增强混合专家系统"""

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config

        # 专家网络
        self.experts = nn.ModuleDict({
            'biochemical': BiochemicalAnalysisExpert(
                config.input_dim, config.expert_dim, config.expert_dim, config.use_lora
            ),
            'medical': MedicalDiagnosisExpert(
                config.input_dim, config.expert_dim, config.expert_dim, config.use_lora
            ),
            'lifestyle': LifestyleAssessmentExpert(
                config.input_dim, config.expert_dim, config.expert_dim, config.use_lora
            ),
            'general': LightweightExpert(
                config.input_dim, config.expert_dim, config.expert_dim, config.use_lora
            )
        })

        # 动态路由门控
        if config.use_dynamic_routing:
            self.routing_gate = DynamicRoutingGating(
                config.input_dim, config.num_experts, config.num_experts_per_token, config.routing_temperature
            )
        else:
            self.routing_gate = nn.Linear(config.input_dim, config.num_experts)

        # 输出融合层
        self.output_fusion = nn.Sequential(
            nn.Linear(config.expert_dim * config.num_experts_per_token, config.expert_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.expert_dim, config.expert_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.expert_dim // 2, config.expert_dim)
        )

        # 可解释性分析
        self.explainability_layer = nn.Sequential(
            nn.Linear(config.expert_dim, config.expert_dim // 4),
            nn.ReLU(),
            nn.Linear(config.expert_dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        前向传播
        Args:
            x: 输入特征 [batch_size, seq_len, input_dim]
        Returns:
            输出和解释信息
        """
        batch_size, seq_len, input_dim = x.shape

        # 动态路由
        if self.config.use_dynamic_routing:
            gate_weights, expert_mask, load_balance_loss = self.routing_gate(x)
        else:
            gate_scores = self.routing_gate(x)
            gate_weights = F.softmax(gate_scores, dim=-1)
            expert_mask = torch.ones_like(gate_weights)
            load_balance_loss = torch.tensor(0.0)

        # 专家输出
        expert_outputs = {}
        for name, expert in self.experts.items():
            expert_outputs[name] = expert(x)

        # 加权融合
        weighted_outputs = []
        for i in range(self.config.num_experts_per_token):
            expert_idx = i
            expert_names = list(self.experts.keys())
            if expert_idx < len(expert_names):
                expert_name = expert_names[expert_idx]
                weight = gate_weights[:, :, i:i+1]  # [batch_size, seq_len, 1]
                expert_output = expert_outputs[expert_name]  # [batch_size, seq_len, expert_dim]
                # 确保权重维度与专家输出维度匹配
                weighted_output = weight * expert_output  # [batch_size, seq_len, expert_dim]
                weighted_outputs.append(weighted_output)

        # 输出融合
        if weighted_outputs:
            concatenated_output = torch.cat(weighted_outputs, dim=-1)
            final_output = self.output_fusion(concatenated_output)
        else:
            final_output = torch.zeros(batch_size, seq_len, self.config.expert_dim)

        # 可解释性分析
        explainability_score = self.explainability_layer(final_output)

        # 解释信息
        explanation_info = {
            'gate_weights': gate_weights.detach().cpu().numpy(),
            'expert_mask': expert_mask.detach().cpu().numpy(),
            'expert_outputs': {k: v.detach().cpu().numpy() for k, v in expert_outputs.items()},
            'load_balance_loss': load_balance_loss.item(),
            'explainability_score': explainability_score.detach().cpu().numpy(),
            'selected_experts': [list(self.experts.keys())[i] for i in range(self.config.num_experts_per_token)]
        }

        return final_output, explanation_info

    def get_expert_importance(self, x: torch.Tensor) -> Dict[str, float]:
        """获取专家重要性"""
        with torch.no_grad():
            _, explanation_info = self.forward(x)
            gate_weights = explanation_info['gate_weights']

            # 计算每个专家的平均权重
            expert_importance = {}
            expert_names = list(self.experts.keys())
            for i, name in enumerate(expert_names):
                if i < gate_weights.shape[-1]:
                    importance = float(np.mean(gate_weights[:, :, i]))
                    expert_importance[name] = importance

            return expert_importance

    def compute_auxiliary_loss(self, x: torch.Tensor) -> torch.Tensor:
        """计算辅助损失（负载均衡）"""
        if self.config.use_auxiliary_loss:
            _, explanation_info = self.forward(x)
            return explanation_info['load_balance_loss']
        return torch.tensor(0.0)

class EnhancedMoEService:
    """增强混合专家服务"""

    def __init__(self, config: MoEConfig):
        self.config = config
        self.model = EnhancedMixtureOfExperts(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=0.001,
            weight_decay=0.01
        )

        # 训练历史
        self.training_history = {
            'loss': [],
            'auxiliary_loss': [],
            'expert_usage': []
        }

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """预测"""
        self.model.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)
            x = x.to(self.device)

            output, explanation = self.model(x)

            return output.cpu().numpy(), explanation

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """训练步骤"""
        self.model.train()

        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        if isinstance(y, np.ndarray):
            y = torch.FloatTensor(y)

        x = x.to(self.device)
        y = y.to(self.device)

        # 前向传播
        output, explanation = self.model(x)

        # 主损失
        main_loss = F.mse_loss(output, y)

        # 辅助损失
        auxiliary_loss = self.model.compute_auxiliary_loss(x)

        # 总损失
        total_loss = main_loss + self.config.auxiliary_loss_weight * auxiliary_loss

        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # 记录历史
        self.training_history['loss'].append(main_loss.item())
        self.training_history['auxiliary_loss'].append(auxiliary_loss.item())

        return {
            'main_loss': main_loss.item(),
            'auxiliary_loss': auxiliary_loss.item(),
            'total_loss': total_loss.item()
        }

    def get_expert_analysis(self, x: torch.Tensor) -> Dict[str, Any]:
        """获取专家分析"""
        expert_importance = self.model.get_expert_importance(x)

        return {
            'expert_importance': expert_importance,
            'most_important_expert': max(expert_importance, key=expert_importance.get),
            'expert_diversity': len([imp for imp in expert_importance.values() if imp > 0.1])
        }

# 工厂函数
def create_enhanced_moe_service(config: Optional[MoEConfig] = None) -> EnhancedMoEService:
    """创建增强混合专家服务"""
    if config is None:
        config = MoEConfig()

    service = EnhancedMoEService(config)
    return service

if __name__ == "__main__":
    # 测试增强混合专家系统
    config = MoEConfig(
        input_dim=256,
        expert_dim=128,
        num_experts=4,
        num_experts_per_token=2,
        use_lora=True,
        use_dynamic_routing=True
    )

    service = create_enhanced_moe_service(config)

    # 模拟数据
    batch_size, seq_len, input_dim = 32, 24, 256
    x = torch.randn(batch_size, seq_len, input_dim)
    y = torch.randn(batch_size, seq_len, 128)

    # 预测测试
    output, explanation = service.predict(x)

    print(f"增强混合专家系统创建成功！")
    print(f"输出形状: {output.shape}")
    print(f"选择的专家: {explanation['selected_experts']}")
    print(f"负载均衡损失: {explanation['load_balance_loss']:.4f}")

    # 专家分析
    analysis = service.get_expert_analysis(x)
    print(f"专家重要性: {analysis['expert_importance']}")
    print(f"最重要专家: {analysis['most_important_expert']}")
