"""
Interface definitions for glucose prediction system.
Defines contracts for all major components following SOLID principles.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import torch
from torch import nn


class ModelInterface(ABC):
    """Abstract interface for glucose prediction models."""

    @abstractmethod
    def train(self, train_data: torch.Tensor, train_targets: torch.Tensor,
              val_data: torch.Tensor, val_targets: torch.Tensor) -> Dict[str, Any]:
        """Train the model with given data."""
        pass

    @abstractmethod
    def predict(self, data: torch.Tensor) -> torch.Tensor:
        """Make predictions on input data."""
        pass

    @abstractmethod
    def evaluate(self, test_data: torch.Tensor, test_targets: torch.Tensor) -> Dict[str, float]:
        """Evaluate model performance."""
        pass

    @abstractmethod
    def save_model(self, path: str) -> None:
        """Save model to specified path."""
        pass

    @abstractmethod
    def load_model(self, path: str) -> None:
        """Load model from specified path."""
        pass


class DataProcessorInterface(ABC):
    """Abstract interface for data processing components."""

    @abstractmethod
    def preprocess(self, raw_data: np.ndarray) -> np.ndarray:
        """Preprocess raw data."""
        pass

    @abstractmethod
    def postprocess(self, processed_data: np.ndarray) -> np.ndarray:
        """Postprocess data back to original scale."""
        pass

    @abstractmethod
    def validate_data(self, data: np.ndarray) -> bool:
        """Validate data integrity and format."""
        pass


class EnsembleInterface(ABC):
    """Abstract interface for model ensemble strategies."""

    @abstractmethod
    def add_model(self, model: ModelInterface, weight: float = 1.0) -> None:
        """Add a model to the ensemble."""
        pass

    @abstractmethod
    def predict(self, data: torch.Tensor) -> torch.Tensor:
        """Make ensemble predictions."""
        pass

    @abstractmethod
    def get_model_weights(self) -> Dict[str, float]:
        """Get current model weights."""
        pass


class MonitorInterface(ABC):
    """Abstract interface for model monitoring and drift detection."""

    @abstractmethod
    def log_prediction(self, input_data: torch.Tensor, prediction: torch.Tensor,
                      actual: Optional[torch.Tensor] = None) -> None:
        """Log prediction for monitoring."""
        pass

    @abstractmethod
    def detect_drift(self) -> Dict[str, Any]:
        """Detect model drift."""
        pass

    @abstractmethod
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        pass


class ExplainerInterface(ABC):
    """Abstract interface for model explainability."""

    @abstractmethod
    def explain_prediction(self, model: ModelInterface, data: torch.Tensor) -> Dict[str, Any]:
        """Explain model prediction."""
        pass

    @abstractmethod
    def get_feature_importance(self, model: ModelInterface) -> Dict[str, float]:
        """Get feature importance scores."""
        pass
