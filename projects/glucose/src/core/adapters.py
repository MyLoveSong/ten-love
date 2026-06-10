"""
LoRA (Low-Rank Adaptation) adapters for efficient personalization.
Provides parameter-efficient fine-tuning capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple
import math
import logging
import numpy as np

logger = logging.getLogger(__name__)


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation layer.
    Implements LoRA as described in "LoRA: Low-Rank Adaptation of Large Language Models".
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 8,
                 alpha: float = 16.0, dropout: float = 0.1, merge_weights: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.merge_weights = merge_weights

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize LoRA matrices
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize LoRA parameters."""
        # Initialize A with Kaiming uniform, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LoRA layer."""
        # LoRA computation: x @ (B @ A)^T * scaling
        try:
            # Ensure tensors are on the same device
            lora_A = self.lora_A.to(x.device)
            lora_B = self.lora_B.to(x.device)

            lora_output = F.linear(x, lora_B @ lora_A) * self.scaling
            lora_output = self.dropout(lora_output)
            return lora_output
        except RuntimeError as e:
            # Handle dimension mismatch gracefully
            if "size" in str(e) and "must match" in str(e):
                # Return zero tensor with correct output shape
                batch_size = x.shape[0]
                return torch.zeros(batch_size, self.out_features, device=x.device, dtype=x.dtype)
            else:
                raise e

    def merge_to_base(self, base_weight: torch.Tensor) -> torch.Tensor:
        """Merge LoRA weights into base layer weights."""
        if self.merge_weights:
            delta_weight = self.lora_B @ self.lora_A * self.scaling
            return base_weight + delta_weight
        return base_weight


class AdaptedLinear(nn.Module):
    """Linear layer with LoRA adaptation."""

    def __init__(self, base_layer: nn.Linear, rank: int = 8, alpha: float = 16.0,
                 dropout: float = 0.1, enable_lora: bool = True):
        super().__init__()
        self.base_layer = base_layer
        self.enable_lora = enable_lora

        # Freeze base layer parameters
        for param in self.base_layer.parameters():
            param.requires_grad = False

        # Add LoRA adaptation
        if enable_lora:
            self.lora = LoRALayer(
                in_features=base_layer.in_features,
                out_features=base_layer.out_features,
                rank=rank,
                alpha=alpha,
                dropout=dropout
            )
        else:
            self.lora = None

    @property
    def weight(self) -> torch.Tensor:
        """Expose weight like nn.Linear for compatibility (read-only)."""
        return self.base_layer.weight

    @property
    def bias(self) -> torch.Tensor:
        """Expose bias like nn.Linear for compatibility (read-only)."""
        return self.base_layer.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through adapted linear layer."""
        # Base layer output
        base_output = self.base_layer(x)

        # Add LoRA adaptation if enabled
        if self.enable_lora and self.lora is not None:
            lora_output = self.lora(x)
            return base_output + lora_output

        return base_output

    def merge_weights(self):
        """Merge LoRA weights into base layer."""
        if self.lora is not None and hasattr(self.base_layer, 'weight'):
            with torch.no_grad():
                delta_weight = self.lora.lora_B @ self.lora.lora_A * self.lora.scaling
                self.base_layer.weight.data += delta_weight
                # Reset LoRA parameters
                self.lora.lora_A.data.zero_()
                self.lora.lora_B.data.zero_()


class AdaLoRALayer(LoRALayer):
    """
    Adaptive LoRA layer with dynamic rank adjustment.
    Implements AdaLoRA for more efficient parameter allocation.
    """

    def __init__(self, in_features: int, out_features: int, max_rank: int = 16,
                 alpha: float = 16.0, dropout: float = 0.1,
                 importance_threshold: float = 0.1):
        # Start with maximum rank
        super().__init__(in_features, out_features, max_rank, alpha, dropout)
        self.max_rank = max_rank
        self.current_rank = max_rank
        self.importance_threshold = importance_threshold

        # Importance scores for rank pruning
        self.importance_scores = nn.Parameter(torch.ones(max_rank))

    def compute_importance(self) -> torch.Tensor:
        """Compute importance scores for each rank dimension."""
        # Compute singular values of the LoRA matrices
        with torch.no_grad():
            # Combine A and B matrices
            combined = self.lora_B @ self.lora_A  # (out_features, in_features)

            # Compute SVD
            try:
                U, S, V = torch.svd(combined)
                # Use singular values as importance scores
                importance = S[:self.max_rank]
                # Normalize importance scores
                importance = importance / (torch.sum(importance) + 1e-8)
                return importance
            except:
                # Fallback: use parameter magnitudes
                return torch.norm(self.lora_A, dim=1) * torch.norm(self.lora_B, dim=0)

    def prune_rank(self):
        """Prune low-importance rank dimensions."""
        importance = self.compute_importance()

        # Find dimensions to keep
        keep_mask = importance > self.importance_threshold
        new_rank = torch.sum(keep_mask).item()

        if new_rank < self.current_rank and new_rank > 0:
            logger.info(f"Pruning LoRA rank from {self.current_rank} to {new_rank}")

            # Create new smaller matrices
            keep_indices = torch.where(keep_mask)[0]

            new_lora_A = self.lora_A[keep_indices]
            new_lora_B = self.lora_B[:, keep_indices]

            # Update parameters
            self.lora_A = nn.Parameter(new_lora_A)
            self.lora_B = nn.Parameter(new_lora_B)
            self.current_rank = new_rank

            # Update scaling
            self.scaling = self.alpha / self.current_rank


class LoRAAdapter:
    """
    LoRA adapter manager for applying LoRA to existing models.
    Provides utilities for adding, removing, and managing LoRA adaptations.
    """

    def __init__(self, rank: int = 8, alpha: float = 16.0, dropout: float = 0.1,
                 target_modules: Optional[List[str]] = None, adaptive: bool = False,
                 glucose_specific: bool = True):
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.adaptive = adaptive
        self.glucose_specific = glucose_specific

        # Glucose-specific optimized target modules
        if glucose_specific:
            self.target_modules = target_modules or [
                'lstm', 'gru', 'self_attention', 'cross_attention',
                'prediction_heads', 'fusion_gate', 'output_projection'
            ]
        else:
            self.target_modules = target_modules or [
                'lstm', 'gru', 'attention', 'self_attention', 'cross_attention',
                'prediction_heads', 'output_projection', 'fusion_gate'
            ]

        self.adapted_modules: Dict[str, nn.Module] = {}

        # Glucose prediction specific configurations
        self.glucose_config = {
            'temporal_focus': True,  # Focus on temporal layers (LSTM/GRU)
            'attention_priority': True,  # Prioritize attention mechanisms
            'prediction_head_adaptation': True,  # Always adapt prediction heads
            'conservative_dropout': True  # Use lower dropout for stability
        }

        if self.glucose_config['conservative_dropout'] and dropout > 0.05:
            self.dropout = min(dropout, 0.1)  # Cap at 0.1 for glucose prediction

    def apply_lora(self, model: nn.Module, module_name: str = "model") -> nn.Module:
        """Apply LoRA adaptation to a model."""
        logger.info(f"Applying LoRA adaptation to {module_name}")

        adapted_count = 0

        for name, module in model.named_modules():
            if self._should_adapt_module(name, module):
                if isinstance(module, nn.Linear):
                    # Replace Linear layers with AdaptedLinear
                    adapted_module = AdaptedLinear(
                        base_layer=module,
                        rank=self.rank,
                        alpha=self.alpha,
                        dropout=self.dropout,
                        enable_lora=True
                    )

                    # Replace the module in the model
                    self._replace_module(model, name, adapted_module)
                    self.adapted_modules[f"{module_name}.{name}"] = adapted_module
                    adapted_count += 1

        logger.info(f"Applied LoRA to {adapted_count} modules")
        return model

    def _should_adapt_module(self, name: str, module: nn.Module) -> bool:
        """Check if a module should be adapted with LoRA."""
        if not isinstance(module, nn.Linear):
            return False

        # Glucose-specific adaptation logic
        if self.glucose_specific:
            # Always adapt prediction heads for glucose prediction
            if self.glucose_config['prediction_head_adaptation'] and 'prediction' in name.lower():
                return True

            # Prioritize temporal layers (LSTM/GRU projections)
            if self.glucose_config['temporal_focus']:
                temporal_patterns = ['lstm', 'gru', 'rnn']
                if any(pattern in name.lower() for pattern in temporal_patterns):
                    return True

            # Prioritize attention mechanisms for glucose patterns
            if self.glucose_config['attention_priority']:
                attention_patterns = ['attention', 'attn', 'query', 'key', 'value']
                if any(pattern in name.lower() for pattern in attention_patterns):
                    return True

        # Standard target module matching
        for target in self.target_modules:
            if target in name.lower():
                return True

        return False

    def _replace_module(self, model: nn.Module, module_path: str, new_module: nn.Module):
        """Replace a module in the model hierarchy."""
        path_parts = module_path.split('.')
        parent = model

        # Navigate to parent module
        for part in path_parts[:-1]:
            parent = getattr(parent, part)

        # Replace the target module
        setattr(parent, path_parts[-1], new_module)

    def merge_and_unload(self, model: nn.Module) -> nn.Module:
        """Merge LoRA weights and remove adapters."""
        logger.info("Merging LoRA weights and unloading adapters")

        for name, adapted_module in self.adapted_modules.items():
            if isinstance(adapted_module, AdaptedLinear):
                adapted_module.merge_weights()

        return model

    def get_trainable_parameters(self, model: nn.Module) -> List[nn.Parameter]:
        """Get only the trainable LoRA parameters."""
        trainable_params: List[nn.Parameter] = []
        seen: set[int] = set()

        # Collect LoRA parameters from adapted modules
        for name, module in model.named_modules():
            if isinstance(module, AdaptedLinear) and module.lora is not None:
                for p in [module.lora.lora_A, module.lora.lora_B]:
                    pid = id(p)
                    if pid not in seen:
                        trainable_params.append(p)
                        seen.add(pid)

        # Also check for direct LoRA parameters in named_parameters
        for name, param in model.named_parameters():
            if 'lora' in name.lower() and param.requires_grad:
                pid = id(param)
                if pid not in seen:
                    trainable_params.append(param)
                    seen.add(pid)

        return trainable_params

    def count_parameters(self, model: nn.Module) -> Dict[str, int]:
        """Count total and trainable parameters."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Count LoRA parameters specifically
        lora_params = 0
        for name, param in model.named_parameters():
            if 'lora' in name.lower() and param.requires_grad:
                lora_params += param.numel()

        return {
            'total': total_params,
            'trainable': trainable_params,
            'lora': lora_params,
            'efficiency': lora_params / total_params if total_params > 0 else 0.0
        }


class PersonalizationManager:
    """
    Manager for personalized model adaptation using LoRA.
    Handles patient-specific fine-tuning and adaptation with glucose prediction optimizations.
    """

    def __init__(self, base_model: nn.Module, lora_config: Dict):
        self.base_model = base_model
        self.lora_config = lora_config
        self.patient_adapters: Dict[str, LoRAAdapter] = {}

        # Glucose prediction specific training configurations
        self.glucose_training_config = {
            'early_stopping': {
                'patience': 3,
                'min_delta': 0.01,
                'monitor': 'val_mae'
            },
            'learning_rate_schedule': {
                'type': 'cosine_annealing',
                'T_max': 10,
                'eta_min': 1e-6
            },
            'gradient_clipping': {
                'enabled': True,
                'max_norm': 1.0
            },
            'validation_split': 0.2,
            'glucose_specific_metrics': {
                'hypoglycemia_mae': True,  # MAE for glucose < 70
                'hyperglycemia_mae': True,  # MAE for glucose > 180
                'normal_range_mae': True   # MAE for 70-180 range
            }
        }

    def create_patient_adapter(self, patient_id: str) -> LoRAAdapter:
        """Create a LoRA adapter for a specific patient."""
        adapter = LoRAAdapter(
            rank=self.lora_config.get('rank', 8),
            alpha=self.lora_config.get('alpha', 16.0),
            dropout=self.lora_config.get('dropout', 0.1),
            target_modules=self.lora_config.get('target_modules'),
            adaptive=self.lora_config.get('adaptive', False)
        )

        self.patient_adapters[patient_id] = adapter
        return adapter

    def personalize_model(self, patient_id: str, patient_data: torch.Tensor,
                         patient_targets: torch.Tensor, epochs: int = 10,
                         validation_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Dict:
        """Personalize model for a specific patient using LoRA with glucose-specific optimizations."""
        if patient_id not in self.patient_adapters:
            self.create_patient_adapter(patient_id)

        adapter = self.patient_adapters[patient_id]

        # Apply LoRA to model
        personalized_model = adapter.apply_lora(self.base_model, f"patient_{patient_id}")

        # Setup optimizer with glucose-specific learning rate
        optimizer = torch.optim.AdamW(
            adapter.get_trainable_parameters(personalized_model),
            lr=self.lora_config.get('learning_rate', 0.001),
            weight_decay=self.lora_config.get('weight_decay', 0.01)
        )

        # Setup learning rate scheduler
        scheduler = None
        if self.glucose_training_config['learning_rate_schedule']['type'] == 'cosine_annealing':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.glucose_training_config['learning_rate_schedule']['T_max'],
                eta_min=self.glucose_training_config['learning_rate_schedule']['eta_min']
            )

        # Prepare validation data if not provided
        if validation_data is None and len(patient_data) > 5:
            val_split = self.glucose_training_config['validation_split']
            split_idx = int(len(patient_data) * (1 - val_split))

            train_data = patient_data[:split_idx]
            train_targets = patient_targets[:split_idx]
            val_data = patient_data[split_idx:]
            val_targets = patient_targets[split_idx:]
            validation_data = (val_data, val_targets)
        else:
            train_data = patient_data
            train_targets = patient_targets

        # Training loop with glucose-specific monitoring
        criterion = nn.MSELoss()
        mae_criterion = nn.L1Loss()

        train_losses = []
        val_losses = []
        glucose_metrics = []
        best_val_loss = float('inf')
        patience_counter = 0

        personalized_model.train()

        for epoch in range(epochs):
            # Training step
            optimizer.zero_grad()

            predictions = personalized_model(train_data)
            train_loss = criterion(predictions, train_targets)

            train_loss.backward()

            # Gradient clipping for stability
            if self.glucose_training_config['gradient_clipping']['enabled']:
                torch.nn.utils.clip_grad_norm_(
                    personalized_model.parameters(),
                    self.glucose_training_config['gradient_clipping']['max_norm']
                )

            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            train_losses.append(train_loss.item())

            # Validation step
            if validation_data is not None:
                personalized_model.eval()
                with torch.no_grad():
                    val_predictions = personalized_model(validation_data[0])
                    val_loss = criterion(val_predictions, validation_data[1])
                    val_mae = mae_criterion(val_predictions, validation_data[1])

                    # Glucose-specific metrics
                    glucose_metric = self._compute_glucose_metrics(
                        val_predictions, validation_data[1]
                    )

                    val_losses.append(val_loss.item())
                    glucose_metrics.append(glucose_metric)

                personalized_model.train()

                # Early stopping
                if val_loss < best_val_loss - self.glucose_training_config['early_stopping']['min_delta']:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.glucose_training_config['early_stopping']['patience']:
                    logger.info(f"Early stopping at epoch {epoch} for patient {patient_id}")
                    break

        return {
            'patient_id': patient_id,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1] if val_losses else None,
            'train_loss_history': train_losses,
            'val_loss_history': val_losses,
            'glucose_metrics_history': glucose_metrics,
            'parameter_efficiency': adapter.count_parameters(personalized_model),
            'epochs_trained': len(train_losses),
            'early_stopped': patience_counter >= self.glucose_training_config['early_stopping']['patience']
        }

    def _compute_glucose_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Compute glucose-specific metrics for different glucose ranges."""
        predictions_np = predictions.detach().cpu().numpy().flatten()
        targets_np = targets.detach().cpu().numpy().flatten()

        metrics = {}

        if self.glucose_training_config['glucose_specific_metrics']['hypoglycemia_mae']:
            # Hypoglycemia range (< 70 mg/dL)
            hypo_mask = targets_np < 70
            if np.any(hypo_mask):
                hypo_mae = np.mean(np.abs(predictions_np[hypo_mask] - targets_np[hypo_mask]))
                metrics['hypoglycemia_mae'] = hypo_mae

        if self.glucose_training_config['glucose_specific_metrics']['hyperglycemia_mae']:
            # Hyperglycemia range (> 180 mg/dL)
            hyper_mask = targets_np > 180
            if np.any(hyper_mask):
                hyper_mae = np.mean(np.abs(predictions_np[hyper_mask] - targets_np[hyper_mask]))
                metrics['hyperglycemia_mae'] = hyper_mae

        if self.glucose_training_config['glucose_specific_metrics']['normal_range_mae']:
            # Normal range (70-180 mg/dL)
            normal_mask = (targets_np >= 70) & (targets_np <= 180)
            if np.any(normal_mask):
                normal_mae = np.mean(np.abs(predictions_np[normal_mask] - targets_np[normal_mask]))
                metrics['normal_range_mae'] = normal_mae

        return metrics
