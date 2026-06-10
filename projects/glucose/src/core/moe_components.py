"""
Mixture of Experts (MoE) components for glucose prediction.
Provides dynamic routing and expert specialization capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ExpertNetwork(nn.Module):
    """Individual expert network for MoE."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 dropout: float = 0.1, expert_id: str = "expert"):
        super().__init__()
        self.expert_id = expert_id
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        # Expert specialization bias
        self.specialization_bias = nn.Parameter(torch.randn(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through expert network."""
        output = self.network(x)
        # Add specialization bias for expert diversity
        output = output + self.specialization_bias
        return output


class GatingNetwork(nn.Module):
    """Gating network for dynamic expert selection."""

    def __init__(self, input_dim: int, num_experts: int,
                 hidden_dim: int = 32, temperature: float = 1.0):
        super().__init__()
        self.num_experts = num_experts
        self.temperature = temperature

        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_experts)
        )

        # Load balancing auxiliary loss weight
        self.load_balance_weight = 0.01

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through gating network.

        Returns:
            gates: Expert weights (batch_size, num_experts)
            load_balance_loss: Auxiliary loss for load balancing
        """
        # Compute gate logits
        gate_logits = self.gate(x)  # (batch_size, num_experts)

        # Apply temperature scaling and softmax
        gates = F.softmax(gate_logits / self.temperature, dim=-1)

        # Compute load balancing loss
        # Encourage uniform expert usage across the batch
        mean_gates = torch.mean(gates, dim=0)  # (num_experts,)
        load_balance_loss = self.load_balance_weight * torch.sum(mean_gates * torch.log(mean_gates + 1e-8))

        return gates, load_balance_loss


class MoEGlucoseHead(nn.Module):
    """
    Mixture of Experts head for glucose prediction.
    Replaces traditional multi-step prediction heads with dynamic expert routing.
    """

    def __init__(self, input_dim: int, output_steps: int = 6, num_experts: int = 4,
                 expert_hidden_dim: int = 32, gate_hidden_dim: int = 32,
                 dropout: float = 0.1, temperature: float = 1.0,
                 top_k: int = 2, device: Optional[torch.device] = None):
        super().__init__()
        self.input_dim = input_dim
        self.output_steps = output_steps
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create expert networks
        self.experts = nn.ModuleList([
            ExpertNetwork(
                input_dim=input_dim,
                hidden_dim=expert_hidden_dim,
                output_dim=output_steps,
                dropout=dropout,
                expert_id=f"expert_{i}"
            ) for i in range(num_experts)
        ])

        # Gating network
        self.gate = GatingNetwork(
            input_dim=input_dim,
            num_experts=num_experts,
            hidden_dim=gate_hidden_dim,
            temperature=temperature
        )

        # Expert specialization (optional)
        self.expert_specializations = {
            0: "short_term",    # 1-2 steps ahead
            1: "medium_term",   # 3-4 steps ahead
            2: "long_term",     # 5-6 steps ahead
            3: "general"        # all steps
        }

        self.to(self.device)

    def forward(self, x: torch.Tensor, return_expert_info: bool = False) -> torch.Tensor:
        """
        Forward pass through MoE head.

        Args:
            x: Input features (batch_size, input_dim)
            return_expert_info: Whether to return expert routing information

        Returns:
            predictions: Multi-step predictions (batch_size, output_steps)
            expert_info: Optional expert routing information
        """
        batch_size = x.shape[0]

        # Get expert gates and load balance loss
        gates, load_balance_loss = self.gate(x)  # (batch_size, num_experts)

        # Top-k expert selection
        top_k_gates, top_k_indices = torch.topk(gates, self.top_k, dim=-1)

        # Renormalize top-k gates
        top_k_gates = F.softmax(top_k_gates, dim=-1)

        # Compute expert outputs
        expert_outputs = []
        for expert in self.experts:
            output = expert(x)  # (batch_size, output_steps)
            expert_outputs.append(output)

        expert_outputs = torch.stack(expert_outputs, dim=1)  # (batch_size, num_experts, output_steps)

        # Weighted combination of top-k experts
        final_output = torch.zeros(batch_size, self.output_steps, device=self.device)

        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]  # (batch_size,)
            expert_weight = top_k_gates[:, i:i+1]  # (batch_size, 1)

            # Gather expert outputs
            expert_output = expert_outputs[torch.arange(batch_size), expert_idx]  # (batch_size, output_steps)

            # Add weighted contribution
            final_output += expert_weight * expert_output

        # Store load balance loss for training
        self.load_balance_loss = load_balance_loss

        if return_expert_info:
            expert_info = {
                'gates': gates.detach().cpu(),
                'top_k_indices': top_k_indices.detach().cpu(),
                'top_k_gates': top_k_gates.detach().cpu(),
                'load_balance_loss': load_balance_loss.item(),
                'expert_usage': torch.mean(gates, dim=0).detach().cpu()
            }
            return final_output, expert_info

        return final_output

    def get_expert_specialization_loss(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute expert specialization loss to encourage expert diversity.
        """
        if targets.shape[1] != self.output_steps:
            return torch.tensor(0.0, device=self.device)

        specialization_loss = 0.0

        # Short-term expert (expert 0) should be better at steps 1-2
        if 0 < self.num_experts:
            short_pred = self.experts[0](x)
            short_loss = F.mse_loss(short_pred[:, :2], targets[:, :2])
            specialization_loss += short_loss

        # Medium-term expert (expert 1) should be better at steps 3-4
        if 1 < self.num_experts and self.output_steps >= 4:
            medium_pred = self.experts[1](x)
            medium_loss = F.mse_loss(medium_pred[:, 2:4], targets[:, 2:4])
            specialization_loss += medium_loss

        # Long-term expert (expert 2) should be better at steps 5-6
        if 2 < self.num_experts and self.output_steps >= 6:
            long_pred = self.experts[2](x)
            long_loss = F.mse_loss(long_pred[:, 4:6], targets[:, 4:6])
            specialization_loss += long_loss

        return specialization_loss / min(3, self.num_experts)


class MoELoss(nn.Module):
    """Combined loss for MoE training."""

    def __init__(self, prediction_weight: float = 1.0,
                 load_balance_weight: float = 0.01,
                 specialization_weight: float = 0.1):
        super().__init__()
        self.prediction_weight = prediction_weight
        self.load_balance_weight = load_balance_weight
        self.specialization_weight = specialization_weight

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                moe_head: MoEGlucoseHead, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute combined MoE loss.

        Returns:
            Dictionary with individual loss components
        """
        # Main prediction loss
        prediction_loss = F.mse_loss(predictions, targets)

        # Load balance loss (from gating network)
        load_balance_loss = getattr(moe_head, 'load_balance_loss', torch.tensor(0.0))

        # Expert specialization loss
        specialization_loss = moe_head.get_expert_specialization_loss(features, targets)

        # Combined loss
        total_loss = (
            self.prediction_weight * prediction_loss +
            self.load_balance_weight * load_balance_loss +
            self.specialization_weight * specialization_loss
        )

        return {
            'total_loss': total_loss,
            'prediction_loss': prediction_loss,
            'load_balance_loss': load_balance_loss,
            'specialization_loss': specialization_loss
        }
