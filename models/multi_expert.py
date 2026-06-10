#!/usr/bin/env python3
"""
Multi-expert architecture for GluFormer.

Implements specialized expert modules and routing mechanism
inspired by NSFC project's "线性修正的混合专家对话微调".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
import math


class CulturalExpert(nn.Module):
    """
    Cultural adaptation expert.

    Specializes in cultural preference modeling and cross-cultural adaptation.
    """

    def __init__(
        self,
        input_dim: int = 128,
        cultural_dim: int = 64,
        num_cultural_factors: int = 8,
        hidden_dim: int = 128
    ):
        super().__init__()

        self.input_dim = input_dim
        self.cultural_dim = cultural_dim
        self.num_cultural_factors = num_cultural_factors

        # Cultural factor embeddings
        self.cultural_embeddings = nn.Embedding(num_cultural_factors, cultural_dim)

        # Cultural preference encoder
        self.cultural_encoder = nn.Sequential(
            nn.Linear(cultural_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Cross-cultural adaptation
        self.adaptation_layer = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )

        # Cultural preference predictor
        self.preference_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output preference score [0, 1]
        )

    def forward(
        self,
        user_features: torch.Tensor,
        cultural_factors: torch.Tensor,
        item_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply cultural adaptation.

        Args:
            user_features: [batch_size, input_dim] User features
            cultural_factors: [batch_size, num_cultural_factors] Cultural factor indices/weights
            item_features: Optional [batch_size, item_dim] Item features for preference prediction

        Returns:
            adapted_features: [batch_size, input_dim] Culturally adapted features
            preference_scores: [batch_size] or [batch_size, num_items] Cultural preference scores
        """
        batch_size = user_features.size(0)

        # Encode cultural factors
        if cultural_factors.size(-1) == self.cultural_dim and cultural_factors.dtype == torch.float:
            # Input is already properly formatted weights
            cultural_weights = cultural_factors
        elif cultural_factors.size(-1) == self.num_cultural_factors:
            # Input might be indices or small weights - check if values are in index range
            if cultural_factors.dtype == torch.long or (cultural_factors.dtype == torch.float and cultural_factors.max() < self.num_cultural_factors):
                # Treat as indices
                cultural_indices = cultural_factors.long()
                cultural_embed = self.cultural_embeddings(cultural_indices)  # [batch_size, num_cultural_factors, cultural_dim]
                cultural_weights = cultural_embed.mean(dim=1)  # [batch_size, cultural_dim]
            else:
                # Treat as weights but wrong dimension - project
                cultural_weights = torch.nn.functional.adaptive_avg_pool1d(
                    cultural_factors.unsqueeze(1), self.cultural_dim
                ).squeeze(1)
        else:
            # Arbitrary dimension - project to expected size
            if cultural_factors.size(-1) > self.cultural_dim:
                cultural_weights = cultural_factors[:, :self.cultural_dim]
            else:
                # Pad with zeros
                padding = torch.zeros(cultural_factors.size(0), self.cultural_dim - cultural_factors.size(-1),
                                    device=cultural_factors.device, dtype=cultural_factors.dtype)
                cultural_weights = torch.cat([cultural_factors, padding], dim=1)

        cultural_encoded = self.cultural_encoder(cultural_weights)  # [batch_size, hidden_dim]

        # Apply cultural adaptation
        combined = torch.cat([user_features, cultural_encoded], dim=1)
        adapted_features = self.adaptation_layer(combined)

        # Predict cultural preferences
        if item_features is not None:
            # Predict preferences for specific items
            item_batch = item_features.size(0)
            if item_batch == batch_size:
                # Same batch size - predict for each user-item pair
                preference_scores = self.preference_predictor(cultural_encoded)  # [batch_size, 1]
                preference_scores = preference_scores.squeeze(1)  # [batch_size]
            else:
                # Different batch sizes - broadcast
                cultural_expanded = cultural_encoded.unsqueeze(1).expand(-1, item_batch // batch_size, -1)
                cultural_expanded = cultural_expanded.reshape(-1, cultural_encoded.size(-1))
                preference_scores = self.preference_predictor(cultural_expanded).squeeze(1)
        else:
            # General cultural preference score
            preference_scores = self.preference_predictor(cultural_encoded).squeeze(1)

        return adapted_features, preference_scores


class ClinicalExpert(nn.Module):
    """
    Clinical safety expert.

    Specializes in medical guideline compliance and clinical risk assessment.
    """

    def __init__(
        self,
        input_dim: int = 128,
        clinical_dim: int = 64,
        num_clinical_factors: int = 10,
        hidden_dim: int = 128
    ):
        super().__init__()

        self.input_dim = input_dim
        self.clinical_dim = clinical_dim
        self.num_clinical_factors = num_clinical_factors

        # Clinical factor embeddings
        self.clinical_embeddings = nn.Embedding(num_clinical_factors, clinical_dim)

        # Clinical condition encoder
        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Clinical constraint application
        self.constraint_layer = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )

        # Safety score predictor (0-1 scale, higher = safer)
        self.safety_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        user_features: torch.Tensor,
        clinical_factors: torch.Tensor,
        item_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply clinical constraints and safety assessment.

        Args:
            user_features: [batch_size, input_dim] User features
            clinical_factors: [batch_size, num_clinical_factors] Clinical factor indices/weights
            item_features: Optional [batch_size, item_dim] Item features for safety prediction

        Returns:
            constrained_features: [batch_size, input_dim] Clinically constrained features
            safety_scores: [batch_size] or [batch_size, num_items] Clinical safety scores
        """
        batch_size = user_features.size(0)

        # Encode clinical factors
        if clinical_factors.size(-1) == self.clinical_dim and clinical_factors.dtype == torch.float:
            # Input is already properly formatted weights
            clinical_weights = clinical_factors
        elif clinical_factors.size(-1) == self.num_clinical_factors:
            # Input might be indices or small weights - check if values are in index range
            if clinical_factors.dtype == torch.long or (clinical_factors.dtype == torch.float and clinical_factors.max() < self.num_clinical_factors):
                # Treat as indices
                clinical_indices = clinical_factors.long()
                clinical_embed = self.clinical_embeddings(clinical_indices)  # [batch_size, num_clinical_factors, clinical_dim]
                clinical_weights = clinical_embed.mean(dim=1)  # [batch_size, clinical_dim]
            else:
                # Treat as weights but wrong dimension - project
                clinical_weights = torch.nn.functional.adaptive_avg_pool1d(
                    clinical_factors.unsqueeze(1), self.clinical_dim
                ).squeeze(1)
        else:
            # Arbitrary dimension - project to expected size
            if clinical_factors.size(-1) > self.clinical_dim:
                clinical_weights = clinical_factors[:, :self.clinical_dim]
            else:
                # Pad with zeros
                padding = torch.zeros(clinical_factors.size(0), self.clinical_dim - clinical_factors.size(-1),
                                    device=clinical_factors.device, dtype=clinical_factors.dtype)
                clinical_weights = torch.cat([clinical_factors, padding], dim=1)

        clinical_encoded = self.clinical_encoder(clinical_weights)  # [batch_size, hidden_dim]

        # Apply clinical constraints
        combined = torch.cat([user_features, clinical_encoded], dim=1)
        constrained_features = self.constraint_layer(combined)

        # Predict clinical safety
        if item_features is not None:
            # Item-specific safety scores
            safety_scores = self.safety_predictor(clinical_encoded).squeeze(1)
        else:
            # General clinical safety score
            safety_scores = self.safety_predictor(clinical_encoded).squeeze(1)

        return constrained_features, safety_scores


class NutritionalExpert(nn.Module):
    """
    Nutritional analysis expert.

    Specializes in nutritional composition analysis and dietary balance assessment.
    """

    def __init__(
        self,
        input_dim: int = 128,
        nutrition_dim: int = 64,
        num_nutrients: int = 15,  # macronutrients, micronutrients
        hidden_dim: int = 128
    ):
        super().__init__()

        self.input_dim = input_dim
        self.nutrition_dim = nutrition_dim
        self.num_nutrients = num_nutrients

        # Nutrient embeddings
        self.nutrient_embeddings = nn.Embedding(num_nutrients, nutrition_dim)

        # Nutritional profile encoder
        self.nutrition_encoder = nn.Sequential(
            nn.Linear(nutrition_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Nutritional enhancement layer
        self.enhancement_layer = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )

        # Dietary balance scorer
        self.balance_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Balance score [0, 1]
        )

    def forward(
        self,
        user_features: torch.Tensor,
        nutritional_profile: torch.Tensor,
        item_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply nutritional analysis and enhancement.

        Args:
            user_features: [batch_size, input_dim] User features
            nutritional_profile: [batch_size, num_nutrients] Nutritional factor values
            item_features: Optional [batch_size, item_dim] Item features

        Returns:
            enhanced_features: [batch_size, input_dim] Nutritionally enhanced features
            balance_scores: [batch_size] or [batch_size, num_items] Dietary balance scores
        """
        batch_size = user_features.size(0)

        # Encode nutritional profile
        if nutritional_profile.size(-1) == self.num_nutrients:
            # Direct nutritional values
            nutrition_embed = self.nutrient_embeddings(
                torch.arange(self.num_nutrients, device=nutritional_profile.device)
            )  # [num_nutrients, nutrition_dim]

            # Weight by nutritional values
            nutrition_weights = nutritional_profile.unsqueeze(2) * nutrition_embed.unsqueeze(0)
            nutrition_weights = nutrition_weights.mean(dim=1)  # [batch_size, nutrition_dim]
        else:
            # Use as indices
            nutrition_indices = nutritional_profile.long()
            nutrition_weights = self.nutrient_embeddings(nutrition_indices).mean(dim=1)

        nutrition_encoded = self.nutrition_encoder(nutrition_weights)  # [batch_size, hidden_dim]

        # Apply nutritional enhancement
        combined = torch.cat([user_features, nutrition_encoded], dim=1)
        enhanced_features = self.enhancement_layer(combined)

        # Predict dietary balance
        balance_scores = self.balance_predictor(nutrition_encoded).squeeze(1)

        return enhanced_features, balance_scores


class ExpertRouter(nn.Module):
    """
    Router for selecting appropriate experts based on context.

    Uses a gating mechanism to determine which expert(s) to activate.
    """

    def __init__(
        self,
        input_dim: int = 128,
        num_experts: int = 3,
        hidden_dim: int = 64
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_experts = num_experts

        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Expert routing gates
        self.routing_gates = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_experts),
            nn.Softmax(dim=1)  # Normalized routing weights
        )

        # Expert importance weights (learned)
        self.expert_importance = nn.Parameter(torch.ones(num_experts))

    def forward(
        self,
        context_features: torch.Tensor,
        task_type: Optional[str] = None
    ) -> torch.Tensor:
        """
        Route to appropriate experts based on context.

        Args:
            context_features: [batch_size, input_dim] Context features
            task_type: Optional task type hint ("cultural", "clinical", "nutritional")

        Returns:
            routing_weights: [batch_size, num_experts] Routing weights for each expert
        """
        # Encode context
        context_encoded = self.context_encoder(context_features)  # [batch_size, hidden_dim]

        # Compute routing weights
        routing_logits = self.routing_gates(context_encoded)  # [batch_size, num_experts]

        # Apply learned importance weights
        routing_logits = routing_logits * self.expert_importance.unsqueeze(0)

        # Task-specific bias (optional)
        if task_type == "cultural":
            routing_logits = routing_logits + torch.tensor([1.0, 0.0, 0.0], device=routing_logits.device)
        elif task_type == "clinical":
            routing_logits = routing_logits + torch.tensor([0.0, 1.0, 0.0], device=routing_logits.device)
        elif task_type == "nutritional":
            routing_logits = routing_logits + torch.tensor([0.0, 0.0, 1.0], device=routing_logits.device)

        # Normalize to probabilities
        routing_weights = F.softmax(routing_logits, dim=1)

        return routing_weights

    def get_top_experts(self, routing_weights: torch.Tensor, top_k: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get top-k experts for each sample.

        Returns:
            expert_indices: [batch_size, top_k] Expert indices
            expert_weights: [batch_size, top_k] Expert weights
        """
        top_weights, top_indices = torch.topk(routing_weights, top_k, dim=1)
        return top_indices, top_weights


class MultiExpertModule(nn.Module):
    """
    Complete multi-expert system integrating all experts and routing.
    """

    def __init__(
        self,
        input_dim: int = 128,
        expert_dims: Dict[str, int] = None,
        router_hidden_dim: int = 64
    ):
        super().__init__()

        self.input_dim = input_dim
        expert_dims = expert_dims or {
            "cultural": 64,
            "clinical": 64,
            "nutritional": 64
        }

        # Initialize experts
        self.cultural_expert = CulturalExpert(
            input_dim=input_dim,
            cultural_dim=expert_dims["cultural"],
            hidden_dim=expert_dims["cultural"] * 2
        )

        self.clinical_expert = ClinicalExpert(
            input_dim=input_dim,
            clinical_dim=expert_dims["clinical"],
            hidden_dim=expert_dims["clinical"] * 2
        )

        self.nutritional_expert = NutritionalExpert(
            input_dim=input_dim,
            nutrition_dim=expert_dims["nutritional"],
            hidden_dim=expert_dims["nutritional"] * 2
        )

        # Expert router
        self.router = ExpertRouter(
            input_dim=input_dim,
            num_experts=3,  # cultural, clinical, nutritional
            hidden_dim=router_hidden_dim
        )

        # Expert output fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(input_dim * 3, input_dim),  # Concatenate all expert outputs
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim)
        )

        # Final confidence predictor
        self.confidence_predictor = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid()  # Confidence score [0, 1]
        )

    def forward(
        self,
        user_features: torch.Tensor,
        cultural_factors: torch.Tensor,
        clinical_factors: torch.Tensor,
        nutritional_profile: torch.Tensor,
        task_type: Optional[str] = None,
        return_expert_outputs: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through multi-expert system.

        Args:
            user_features: [batch_size, input_dim] User features
            cultural_factors: [batch_size, num_cultural_factors] Cultural factors
            clinical_factors: [batch_size, num_clinical_factors] Clinical factors
            nutritional_profile: [batch_size, num_nutrients] Nutritional profile
            task_type: Optional task type hint
            return_expert_outputs: Whether to return individual expert outputs

        Returns:
            outputs: Dictionary containing fused features and confidence
        """
        # Route to experts
        routing_weights = self.router(user_features, task_type)  # [batch_size, 3]

        # Get expert outputs
        cultural_output, cultural_scores = self.cultural_expert(
            user_features, cultural_factors, user_features
        )

        clinical_output, clinical_scores = self.clinical_expert(
            user_features, clinical_factors, user_features
        )

        nutritional_output, nutritional_scores = self.nutritional_expert(
            user_features, nutritional_profile, user_features
        )

        # Weighted fusion of expert outputs
        weighted_outputs = torch.stack([cultural_output, clinical_output, nutritional_output], dim=1)  # [batch_size, 3, input_dim]
        routing_weights_expanded = routing_weights.unsqueeze(2)  # [batch_size, 3, 1]
        fused_features = (weighted_outputs * routing_weights_expanded).sum(dim=1)  # [batch_size, input_dim]

        # Additional fusion layer
        final_features = self.fusion_layer(
            torch.cat([cultural_output, clinical_output, nutritional_output], dim=1)
        ) + fused_features

        # Predict confidence
        confidence = self.confidence_predictor(final_features).squeeze(1)

        # Combine expert scores
        combined_scores = (
            routing_weights[:, 0] * cultural_scores +
            routing_weights[:, 1] * clinical_scores +
            routing_weights[:, 2] * nutritional_scores
        )

        outputs = {
            "fused_features": final_features,
            "confidence": confidence,
            "routing_weights": routing_weights,
            "combined_scores": combined_scores
        }

        if return_expert_outputs:
            outputs.update({
                "cultural_output": cultural_output,
                "cultural_scores": cultural_scores,
                "clinical_output": clinical_output,
                "clinical_scores": clinical_scores,
                "nutritional_output": nutritional_output,
                "nutritional_scores": nutritional_scores
            })

        return outputs

    def get_expert_stats(self) -> Dict[str, Any]:
        """Get statistics about expert usage and performance."""
        return {
            "num_experts": 3,
            "expert_names": ["cultural", "clinical", "nutritional"],
            "router_importance": self.router.expert_importance.detach().cpu().numpy()
        }


# Example usage and testing
if __name__ == "__main__":
    print("Testing Multi-Expert Architecture...")

    batch_size = 4
    input_dim = 128

    # Create multi-expert system
    multi_expert = MultiExpertModule(input_dim=input_dim)

    # Test inputs
    user_features = torch.randn(batch_size, input_dim)
    cultural_factors = torch.randn(batch_size, 8)  # 8 cultural factors
    clinical_factors = torch.randn(batch_size, 10)  # 10 clinical factors
    nutritional_profile = torch.randn(batch_size, 15)  # 15 nutrients

    # Forward pass
    outputs = multi_expert(
        user_features=user_features,
        cultural_factors=cultural_factors,
        clinical_factors=clinical_factors,
        nutritional_profile=nutritional_profile,
        task_type="cultural",
        return_expert_outputs=True
    )

    print(f"Fused features shape: {outputs['fused_features'].shape}")
    print(f"Confidence shape: {outputs['confidence'].shape}")
    print(f"Routing weights shape: {outputs['routing_weights'].shape}")
    print(f"Combined scores shape: {outputs['combined_scores'].shape}")

    # Check routing weights for cultural task
    print(f"Routing weights (cultural task): {outputs['routing_weights'].mean(dim=0)}")

    # Expert stats
    stats = multi_expert.get_expert_stats()
    print(f"Expert stats: {stats}")

    print("\nMulti-expert architecture ready!")
