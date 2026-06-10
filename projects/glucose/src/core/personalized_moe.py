"""
Personalized MoE components combining LoRA adaptation with MoE prediction heads.
Provides patient-specific expert routing and adaptation capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import logging

from .moe_components import MoEGlucoseHead, MoELoss
from .adapters import LoRAAdapter, AdaptedLinear

logger = logging.getLogger(__name__)


class TinyLoRALayer(nn.Module):
    """Minimal LoRA layer holding parameters for personalized adapters."""
    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        # parameters
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # output in out_features
        return F.linear(x, self.lora_B @ self.lora_A) * self.scaling


class PersonalizedMoEHead(nn.Module):
    """
    MoE head with LoRA adaptation for personalized glucose prediction.
    Combines expert routing with patient-specific adaptations.
    """

    def __init__(self, input_dim: int, output_steps: int = 6, num_experts: int = 4,
                 expert_hidden_dim: int = 32, gate_hidden_dim: int = 32,
                 dropout: float = 0.1, temperature: float = 1.0,
                 top_k: int = 2, lora_rank: int = 4, lora_alpha: float = 8.0,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.input_dim = input_dim
        self.output_steps = output_steps
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Base MoE head (frozen during personalization)
        self.base_moe = MoEGlucoseHead(
            input_dim=input_dim,
            output_steps=output_steps,
            num_experts=num_experts,
            expert_hidden_dim=expert_hidden_dim,
            gate_hidden_dim=gate_hidden_dim,
            dropout=dropout,
            temperature=temperature,
            top_k=top_k,
            device=device
        )

        # Patient-specific LoRA adaptations for experts
        self.expert_adaptations: Dict[str, nn.ModuleList] = {}

        # Patient-specific gating adaptations
        self.gate_adaptations: Dict[str, TinyLoRALayer] = {}

        # Current active patient
        self.active_patient: Optional[str] = None

        self.to(self.device)

    def create_patient_adaptation(self, patient_id: str) -> None:
        """Create LoRA adaptations for a specific patient."""
        if patient_id in self.expert_adaptations:
            return

        logger.info(f"Creating personalized MoE adaptation for patient {patient_id}")

        # Create LoRA adaptations for each expert
        expert_loras = nn.ModuleList()
        for i in range(self.num_experts):
            # Adapt the expert networks with LoRA
            expert_adaptation = nn.ModuleDict({
                'input_lora': TinyLoRALayer(
                    self.base_moe.experts[i].network[0].in_features,
                    self.base_moe.experts[i].network[0].out_features,
                    self.lora_rank,
                    self.lora_alpha
                ),
                'hidden_lora': TinyLoRALayer(
                    self.base_moe.experts[i].network[3].in_features,
                    self.base_moe.experts[i].network[3].out_features,
                    self.lora_rank,
                    self.lora_alpha
                ),
                'output_lora': TinyLoRALayer(
                    self.base_moe.experts[i].network[6].in_features,
                    self.base_moe.experts[i].network[6].out_features,
                    self.lora_rank,
                    self.lora_alpha
                )
            })
            expert_loras.append(expert_adaptation)

        self.expert_adaptations[patient_id] = expert_loras

        # Create gating adaptation
        gate_adaptation = TinyLoRALayer(
            self.base_moe.gate.gate[0].in_features,
            self.base_moe.gate.gate[0].out_features,
            self.lora_rank,
            self.lora_alpha
        )
        self.gate_adaptations[patient_id] = gate_adaptation

        logger.info(f"Created {self.num_experts} expert adaptations + 1 gate adaptation for patient {patient_id}")

    def _apply_lora_adaptation(self, base_output: torch.Tensor,
                              lora_layer: nn.Module, input_tensor: torch.Tensor) -> torch.Tensor:
        """Apply LoRA adaptation to base layer output."""
        lora_output = lora_layer(input_tensor)
        return base_output + lora_output

    def set_active_patient(self, patient_id: str) -> None:
        """Set the active patient for personalized inference."""
        if patient_id not in self.expert_adaptations:
            self.create_patient_adaptation(patient_id)

        self.active_patient = patient_id
        logger.info(f"Activated personalized MoE for patient {patient_id}")

    def forward(self, x: torch.Tensor, patient_id: Optional[str] = None,
                return_expert_info: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Forward pass with optional patient-specific adaptations.

        Args:
            x: Input features (batch_size, input_dim)
            patient_id: Patient ID for personalization (uses active_patient if None)
            return_expert_info: Whether to return expert routing information

        Returns:
            predictions: Multi-step predictions (batch_size, output_steps)
            expert_info: Optional expert routing information
        """
        # Use specified patient or active patient
        current_patient = patient_id or self.active_patient

        if current_patient is None or current_patient not in self.expert_adaptations:
            # Fall back to base MoE without personalization
            return self.base_moe(x, return_expert_info)

        batch_size = x.shape[0]

        # Personalized gating
        base_gate_logits = self.base_moe.gate.gate(x)
        gate_adaptation = self.gate_adaptations[current_patient]
        adapted_gate_logits = self._apply_lora_adaptation(base_gate_logits, gate_adaptation, x)

        # Apply temperature and softmax
        gates = F.softmax(adapted_gate_logits / self.base_moe.gate.temperature, dim=-1)

        # Top-k expert selection
        top_k_gates, top_k_indices = torch.topk(gates, self.top_k, dim=-1)
        top_k_gates = F.softmax(top_k_gates, dim=-1)

        # Compute personalized expert outputs
        expert_outputs = []
        expert_adaptations = self.expert_adaptations[current_patient]

        for i, expert in enumerate(self.base_moe.experts):
            # Base expert computation through each layer
            expert_input = x

            # First layer (input -> hidden) with LoRA adaptation
            base_hidden = expert.network[0](expert_input)  # Linear
            adapted_hidden = self._apply_lora_adaptation(
                base_hidden, expert_adaptations[i]['input_lora'], expert_input
            )
            hidden = F.relu(adapted_hidden)  # ReLU
            hidden = expert.network[2](hidden)  # Dropout

            # Second layer (hidden -> hidden/2) with LoRA adaptation
            base_hidden2 = expert.network[3](hidden)  # Linear
            adapted_hidden2 = self._apply_lora_adaptation(
                base_hidden2, expert_adaptations[i]['hidden_lora'], hidden
            )
            hidden2 = F.relu(adapted_hidden2)  # ReLU
            hidden2 = expert.network[5](hidden2)  # Dropout

            # Output layer with LoRA adaptation
            base_output = expert.network[6](hidden2)  # Linear
            adapted_output = self._apply_lora_adaptation(
                base_output, expert_adaptations[i]['output_lora'], hidden2
            )

            # Add expert specialization bias
            final_output = adapted_output + expert.specialization_bias
            expert_outputs.append(final_output)

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

        # Compute load balance loss for training
        mean_gates = torch.mean(gates, dim=0)
        load_balance_loss = self.base_moe.gate.load_balance_weight * torch.sum(
            mean_gates * torch.log(mean_gates + 1e-8)
        )
        self.load_balance_loss = load_balance_loss

        if return_expert_info:
            expert_info = {
                'gates': gates.detach().cpu(),
                'top_k_indices': top_k_indices.detach().cpu(),
                'top_k_gates': top_k_gates.detach().cpu(),
                'load_balance_loss': load_balance_loss.item(),
                'expert_usage': torch.mean(gates, dim=0).detach().cpu(),
                'patient_id': current_patient,
                'personalized': True
            }
            return final_output, expert_info

        return final_output

    def get_patient_parameters(self, patient_id: str) -> List[nn.Parameter]:
        """Get trainable parameters for a specific patient."""
        if patient_id not in self.expert_adaptations:
            return []

        params: List[nn.Parameter] = []

        # Expert adaptation parameters
        for expert_adaptation in self.expert_adaptations[patient_id]:
            for lora_layer in expert_adaptation.values():
                params.extend([lora_layer.lora_A, lora_layer.lora_B])

        # Gate adaptation parameters
        gate_adaptation = self.gate_adaptations[patient_id]
        params.extend([gate_adaptation.lora_A, gate_adaptation.lora_B])

        return params

    def freeze_base_model(self) -> None:
        """Freeze base MoE parameters for personalization training."""
        for param in self.base_moe.parameters():
            param.requires_grad = False

    def unfreeze_base_model(self) -> None:
        """Unfreeze base MoE parameters."""
        for param in self.base_moe.parameters():
            param.requires_grad = True

    def get_personalization_stats(self) -> Dict[str, Union[int, List[str]]]:
        """Get statistics about personalized adaptations."""
        return {
            'num_patients': len(self.expert_adaptations),
            'patient_ids': list(self.expert_adaptations.keys()),
            'active_patient': self.active_patient,
            'lora_rank': self.lora_rank,
            'lora_alpha': self.lora_alpha,
            'num_experts': self.num_experts
        }


class PersonalizedMoELoss(MoELoss):
    """Enhanced MoE loss for personalized training."""

    def __init__(self, prediction_weight: float = 1.0,
                 load_balance_weight: float = 0.01,
                 specialization_weight: float = 0.1,
                 personalization_weight: float = 0.05):
        super().__init__(prediction_weight, load_balance_weight, specialization_weight)
        self.personalization_weight = personalization_weight

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                moe_head: PersonalizedMoEHead, features: torch.Tensor,
                patient_id: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss for personalized MoE training.

        Returns:
            Dictionary with individual loss components
        """
        # Base MoE losses
        base_losses = super().forward(predictions, targets, moe_head.base_moe, features)

        # Personalization regularization loss
        personalization_loss = torch.tensor(0.0, device=predictions.device)

        if patient_id and patient_id in moe_head.expert_adaptations:
            # L2 regularization on LoRA parameters to prevent overfitting
            patient_params = moe_head.get_patient_parameters(patient_id)
            for param in patient_params:
                personalization_loss += torch.norm(param, p=2)

            personalization_loss = personalization_loss / len(patient_params)

        # Combined loss
        total_loss = (
            base_losses['total_loss'] +
            self.personalization_weight * personalization_loss
        )

        return {
            'total_loss': total_loss,
            'prediction_loss': base_losses['prediction_loss'],
            'load_balance_loss': base_losses['load_balance_loss'],
            'specialization_loss': base_losses['specialization_loss'],
            'personalization_loss': personalization_loss
        }
