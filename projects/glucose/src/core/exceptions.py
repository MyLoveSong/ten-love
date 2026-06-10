"""
Custom exceptions for glucose prediction system.
Provides specific error types for better error handling.
"""


class GlucosePredictionError(Exception):
    """Base exception for glucose prediction system."""
    pass


class ModelNotFoundError(GlucosePredictionError):
    """Raised when a model file cannot be found."""
    pass


class DataValidationError(GlucosePredictionError):
    """Raised when data validation fails."""
    pass


class ModelTrainingError(GlucosePredictionError):
    """Raised when model training fails."""
    pass


class EnsembleError(GlucosePredictionError):
    """Raised when ensemble operations fail."""
    pass


class MonitoringError(GlucosePredictionError):
    """Raised when monitoring operations fail."""
    pass


class ConfigurationError(GlucosePredictionError):
    """Raised when configuration is invalid."""
    pass
