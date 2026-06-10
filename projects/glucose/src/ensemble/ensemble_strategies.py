"""
Ensemble strategies for glucose prediction models.
Implements various ensemble methods following modern ML practices.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.interfaces import EnsembleInterface, ModelInterface
from core.exceptions import EnsembleError

logger = logging.getLogger(__name__)


class WeightedEnsemble(EnsembleInterface):
    """Weighted ensemble strategy with dynamic weight adjustment."""

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models: List[ModelInterface] = []
        self.weights: List[float] = []
        self.model_names: List[str] = []
        self.performance_history: Dict[str, List[float]] = {}

    def add_model(self, model: ModelInterface, weight: float = 1.0, name: str = None) -> None:
        """Add a model to the ensemble with specified weight."""
        try:
            self.models.append(model)
            self.weights.append(weight)
            model_name = name or f"model_{len(self.models)}"
            self.model_names.append(model_name)
            self.performance_history[model_name] = []

            logger.info(f"Added model '{model_name}' with weight {weight}")

        except Exception as e:
            raise EnsembleError(f"Failed to add model: {str(e)}")

    def predict(self, data: torch.Tensor) -> torch.Tensor:
        """Make weighted ensemble predictions."""
        if not self.models:
            raise EnsembleError("No models in ensemble")

        try:
            predictions = []
            data = data.to(self.device)

            for model in self.models:
                model_pred = model.predict(data)
                predictions.append(model_pred)

            # Stack predictions and apply weights
            stacked_predictions = torch.stack(predictions, dim=0)
            weights_tensor = torch.tensor(self.weights, device=self.device).view(-1, 1, 1)

            # Normalize weights
            weights_tensor = weights_tensor / torch.sum(weights_tensor)

            # Weighted average
            ensemble_prediction = torch.sum(stacked_predictions * weights_tensor, dim=0)

            return ensemble_prediction

        except Exception as e:
            raise EnsembleError(f"Ensemble prediction failed: {str(e)}")

    def update_weights(self, validation_data: torch.Tensor,
                      validation_targets: torch.Tensor) -> None:
        """Update model weights based on validation performance."""
        try:
            performances = []

            for i, model in enumerate(self.models):
                metrics = model.evaluate(validation_data, validation_targets)
                mae = metrics['mae']
                performances.append(1.0 / (mae + 1e-8))  # Inverse MAE as weight

                # Update performance history
                model_name = self.model_names[i]
                self.performance_history[model_name].append(mae)

            # Normalize weights
            total_performance = sum(performances)
            self.weights = [p / total_performance for p in performances]

            logger.info(f"Updated weights: {dict(zip(self.model_names, self.weights))}")

        except Exception as e:
            raise EnsembleError(f"Weight update failed: {str(e)}")

    def get_model_weights(self) -> Dict[str, float]:
        """Get current model weights."""
        return dict(zip(self.model_names, self.weights))


class VotingEnsemble(EnsembleInterface):
    """Voting ensemble strategy for classification-like decisions."""

    def __init__(self, voting_strategy: str = "soft", device: Optional[torch.device] = None):
        self.voting_strategy = voting_strategy  # "soft" or "hard"
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models: List[ModelInterface] = []
        self.model_names: List[str] = []

    def add_model(self, model: ModelInterface, weight: float = 1.0, name: str = None) -> None:
        """Add a model to the voting ensemble."""
        try:
            self.models.append(model)
            model_name = name or f"model_{len(self.models)}"
            self.model_names.append(model_name)

            logger.info(f"Added model '{model_name}' to voting ensemble")

        except Exception as e:
            raise EnsembleError(f"Failed to add model: {str(e)}")

    def predict(self, data: torch.Tensor) -> torch.Tensor:
        """Make voting ensemble predictions."""
        if not self.models:
            raise EnsembleError("No models in ensemble")

        try:
            predictions = []
            data = data.to(self.device)

            for model in self.models:
                model_pred = model.predict(data)
                predictions.append(model_pred)

            if self.voting_strategy == "soft":
                # Soft voting: average predictions
                stacked_predictions = torch.stack(predictions, dim=0)
                ensemble_prediction = torch.mean(stacked_predictions, dim=0)
            else:
                # Hard voting: majority vote (for discrete predictions)
                stacked_predictions = torch.stack(predictions, dim=0)
                ensemble_prediction = torch.median(stacked_predictions, dim=0)[0]

            return ensemble_prediction

        except Exception as e:
            raise EnsembleError(f"Voting prediction failed: {str(e)}")

    def get_model_weights(self) -> Dict[str, float]:
        """Get model weights (equal for voting ensemble)."""
        equal_weight = 1.0 / len(self.models) if self.models else 0.0
        return {name: equal_weight for name in self.model_names}


class StackingEnsemble(EnsembleInterface):
    """Stacking ensemble with meta-learner."""

    def __init__(self, meta_learner: Optional[torch.nn.Module] = None,
                 device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models: List[ModelInterface] = []
        self.model_names: List[str] = []
        self.meta_learner = meta_learner or self._create_default_meta_learner()
        self.is_meta_trained = False

    def _create_default_meta_learner(self) -> torch.nn.Module:
        """Create default meta-learner architecture."""
        return torch.nn.Sequential(
            torch.nn.Linear(6, 32),  # Assuming 6 outputs from base models
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 6)  # Final prediction
        ).to(self.device)

    def add_model(self, model: ModelInterface, weight: float = 1.0, name: str = None) -> None:
        """Add a model to the stacking ensemble."""
        try:
            self.models.append(model)
            model_name = name or f"model_{len(self.models)}"
            self.model_names.append(model_name)

            # Reset meta-learner when new model is added
            self.is_meta_trained = False

            logger.info(f"Added model '{model_name}' to stacking ensemble")

        except Exception as e:
            raise EnsembleError(f"Failed to add model: {str(e)}")

    def train_meta_learner(self, train_data: torch.Tensor,
                          train_targets: torch.Tensor,
                          val_data: torch.Tensor,
                          val_targets: torch.Tensor) -> None:
        """Train the meta-learner using base model predictions."""
        try:
            if not self.models:
                raise EnsembleError("No base models available")

            # Generate base model predictions
            train_meta_features = self._generate_meta_features(train_data)
            val_meta_features = self._generate_meta_features(val_data)

            # Train meta-learner
            optimizer = torch.optim.AdamW(self.meta_learner.parameters(), lr=0.001)
            criterion = torch.nn.MSELoss()

            # Create data loaders
            train_dataset = torch.utils.data.TensorDataset(train_meta_features, train_targets)
            val_dataset = torch.utils.data.TensorDataset(val_meta_features, val_targets)

            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

            best_val_loss = float('inf')
            patience = 10
            patience_counter = 0

            for epoch in range(100):
                # Training
                self.meta_learner.train()
                train_loss = 0.0

                for batch_features, batch_targets in train_loader:
                    batch_features = batch_features.to(self.device)
                    batch_targets = batch_targets.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.meta_learner(batch_features)
                    loss = criterion(outputs, batch_targets)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()

                # Validation
                self.meta_learner.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for batch_features, batch_targets in val_loader:
                        batch_features = batch_features.to(self.device)
                        batch_targets = batch_targets.to(self.device)

                        outputs = self.meta_learner(batch_features)
                        loss = criterion(outputs, batch_targets)
                        val_loss += loss.item()

                val_loss /= len(val_loader)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    break

            self.is_meta_trained = True
            logger.info("Meta-learner training completed")

        except Exception as e:
            raise EnsembleError(f"Meta-learner training failed: {str(e)}")

    def _generate_meta_features(self, data: torch.Tensor) -> torch.Tensor:
        """Generate meta-features from base model predictions."""
        meta_features = []
        data = data.to(self.device)

        for model in self.models:
            pred = model.predict(data)
            meta_features.append(pred)

        # Concatenate all predictions as meta-features
        return torch.cat(meta_features, dim=1)

    def predict(self, data: torch.Tensor) -> torch.Tensor:
        """Make stacking ensemble predictions."""
        if not self.models:
            raise EnsembleError("No models in ensemble")

        if not self.is_meta_trained:
            raise EnsembleError("Meta-learner not trained")

        try:
            # Generate meta-features
            meta_features = self._generate_meta_features(data)

            # Meta-learner prediction
            self.meta_learner.eval()
            with torch.no_grad():
                ensemble_prediction = self.meta_learner(meta_features)

            return ensemble_prediction

        except Exception as e:
            raise EnsembleError(f"Stacking prediction failed: {str(e)}")

    def get_model_weights(self) -> Dict[str, float]:
        """Get model weights (not directly applicable for stacking)."""
        return {name: 1.0 for name in self.model_names}
