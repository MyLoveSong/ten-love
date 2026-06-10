"""
Enhanced Glucose Prediction System - Main Integration Module
Integrates all improved components following modern engineering standards.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime
import json

# Core imports
from .core.base_models import (
    LSTMGlucosePredictor,
    TransformerGlucosePredictor,
    GluFormerPredictor,
    WaveletGluFormerPredictor
)
from .core.moe_components import MoEGlucoseHead, MoELoss
from .core.adapters import LoRAAdapter, PersonalizationManager
from .core.personalized_moe import PersonalizedMoEHead, PersonalizedMoELoss
from .core.exceptions import GlucosePredictionError, ModelTrainingError

# Ensemble imports
from ensemble.ensemble_strategies import WeightedEnsemble, VotingEnsemble, StackingEnsemble

# Monitoring imports
from monitoring.drift_detector import DriftDetector
from monitoring.anomaly_autoencoder import GlucoseAnomalyDetector

# Data processing imports
from data_processing.data_augmentation import GlucoseDataAugmenter
from data_processing.rare_event_augment import RareEventAugmenter

logger = logging.getLogger(__name__)


class EnhancedGlucosePredictionSystem:
    """
    Enhanced glucose prediction system with ensemble learning,
    real-time monitoring, and advanced data processing.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = self._get_default_config()
        if config:
            self.config.update(config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize components
        self.models: Dict[str, Any] = {}
        self.ensemble = None
        self.drift_detector = None
        self.anomaly_detector = None
        self.data_augmenter = GlucoseDataAugmenter(self.device)
        self.rare_event_augmenter = RareEventAugmenter()
        self.lora_adapter = None
        self.personalization_manager = None
        self.personalized_moe_head = None

        # Training state
        self.is_trained = False
        self.training_history = {}

        # Output directory
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Enhanced Glucose Prediction System initialized")
        logger.info(f"Device: {self.device}")
        logger.info(f"Output directory: {self.output_dir}")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default system configuration."""
        return {
            # Model configurations
            'models': {
                'lstm': {
                    'input_dim': 1,
                    'hidden_dim': 64,
                    'output_dim': 6,
                    'dropout': 0.1
                },
                'transformer': {
                    'input_dim': 1,
                    'hidden_dim': 64,
                    'output_dim': 6,
                    'dropout': 0.1
                },
                'gluformer': {
                    'input_dim': 1,
                    'hidden_dim': 64,
                    'output_dim': 6,
                    'dropout': 0.1
                },
                'wavelet_gluformer': {
                    'input_dim': 1,
                    'hidden_dim': 64,
                    'output_dim': 6,
                    'dropout': 0.1,
                    'wavelet': 'db4',
                    'wavelet_levels': 3
                }
            },

            # Training configurations
            'training': {
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.001,
                'patience': 15,
                'val_split': 0.2,
                'gradient_clip': 1.0,
                'lr_scheduler': {
                    'type': 'cosine_warmup',
                    'warmup_epochs': 8,
                    'warmup_start_factor': 0.2,
                    'min_lr': 1e-5
                },
                'early_stopping': {
                    'patience': 8,
                    'min_delta': 5e-4,
                    'burn_in_epochs': 10,
                    'cooldown': 1
                }
            },

            # Ensemble configurations
            'ensemble': {
                'strategy': 'weighted',  # 'weighted', 'voting', 'stacking'
                'auto_weight_update': True
            },

            # Feature engineering and data augmentation configurations
            'feature_engineering': {
                'enabled': True,
                'features': ['delta', 'rolling_mean'],
                'rolling_window': 4,
                'normalize': True
            },

            'augmentation': {
                'enabled': True,
                'factor': 2.0,
                'methods': ['noise', 'scaling', 'time_warping', 'magnitude_warping']
            },

            # Rare event augmentation
            'rare_event_augmentation': {
                'enabled': True,
                'factor': 2.0,
                'methods': ['smote', 'noise_injection', 'temporal_shift', 'magnitude_scaling'],
                'event_types': ['hypoglycemia', 'hyperglycemia', 'rapid_drop', 'rapid_rise']
            },

            # MoE configurations
            'moe': {
                'enabled': False,  # Optional enhancement
                'num_experts': 4,
                'expert_hidden_dim': 32,
                'top_k': 2,
                'temperature': 1.0
            },

            # LoRA configurations
            'lora': {
                'enabled': True,  # Enable for personalization
                'rank': 4,  # Smaller rank for glucose prediction stability
                'alpha': 8.0,  # Conservative alpha for better convergence
                'dropout': 0.05,  # Lower dropout for glucose prediction
                'target_modules': ['lstm', 'gru', 'self_attention', 'cross_attention', 'prediction_heads'],
                'glucose_specific': True,  # Use glucose-optimized configurations
                'learning_rate': 0.001,
                'weight_decay': 0.01,
                'personalization': {
                    'enabled': True,
                    'early_stopping_patience': 3,
                    'validation_split': 0.2,
                    'max_epochs': 10,
                    'glucose_metrics': True
                }
            },

            # Personalized MoE configurations
            'personalized_moe': {
                'enabled': True,  # Enable personalized MoE
                'lora_rank': 4,
                'lora_alpha': 8.0,
                'personalization_weight': 0.05
            },

            # Anomaly detection
            'anomaly_detection': {
                'enabled': True,
                'sequence_length': 12,
                'normal_threshold': 0.1,
                'anomaly_threshold': 0.3,
                'critical_threshold': 0.5
            },

            # Monitoring configurations
            'monitoring': {
                'enabled': True,
                'window_size': 1000,
                'drift_threshold': 0.05
            },

            # Output configurations
            'output_dir': 'outputs/enhanced_glucose_system',
            'save_models': True,
            'save_plots': True
        }

    def prepare_data(self, sequences: np.ndarray, targets: np.ndarray) -> Dict[str, torch.Tensor]:
        """Prepare and augment training data."""
        try:
            logger.info("Preparing training data...")

            # Data validation
            if len(sequences) != len(targets):
                raise GlucosePredictionError("Sequences and targets length mismatch")

            # Feature engineering
            sequences = self._apply_feature_engineering(sequences)

            # Data augmentation
            if self.config['augmentation']['enabled']:
                logger.info("Applying data augmentation...")
                sequences, targets = self.data_augmenter.augment_data(
                    sequences, targets,
                    augmentation_factor=self.config['augmentation']['factor'],
                    methods=self.config['augmentation']['methods']
                )

            # Rare event augmentation
            if self.config['rare_event_augmentation']['enabled']:
                logger.info("Applying rare event augmentation...")
                rare_config = self.config['rare_event_augmentation']
                sequences, targets = self.rare_event_augmenter.augment_rare_events(
                    sequences, targets,
                    event_types=rare_config['event_types']
                )

            # Train/validation split
            val_split = self.config['training']['val_split']
            split_idx = int(len(sequences) * (1 - val_split))

            # Shuffle data
            indices = np.random.permutation(len(sequences))
            sequences = sequences[indices]
            targets = targets[indices]

            # Split data
            train_sequences = sequences[:split_idx]
            train_targets = targets[:split_idx]
            val_sequences = sequences[split_idx:]
            val_targets = targets[split_idx:]

            # Convert to tensors
            data_dict = {
                'train_sequences': torch.FloatTensor(train_sequences),
                'train_targets': torch.FloatTensor(train_targets),
                'val_sequences': torch.FloatTensor(val_sequences),
                'val_targets': torch.FloatTensor(val_targets)
            }

            # Add sequence dimension if needed
            for key in data_dict:
                if 'sequences' in key and data_dict[key].dim() == 2:
                    data_dict[key] = data_dict[key].unsqueeze(-1)

            logger.info(f"Data prepared: Train={len(train_sequences)}, Val={len(val_sequences)}")

            # Update model input dimensions to match engineered features
            self._update_model_input_dims(data_dict['train_sequences'].shape[-1])

            return data_dict

        except Exception as e:
            raise GlucosePredictionError(f"Data preparation failed: {str(e)}")

    def _apply_feature_engineering(self, sequences: np.ndarray) -> np.ndarray:
        """Optionally add derived channels like deltas or rolling stats."""
        if sequences.ndim == 2:
            sequences = sequences[..., np.newaxis]

        feature_cfg = self.config.get('feature_engineering', {})
        if not feature_cfg.get('enabled', False):
            return sequences

        sequences = sequences.astype(np.float32, copy=False)
        features_to_add = set(feature_cfg.get('features', []))
        engineered_channels = [sequences]

        if 'delta' in features_to_add:
            deltas = np.diff(sequences, axis=1, prepend=sequences[:, :1, :])
            engineered_channels.append(self._normalize_feature(deltas, feature_cfg))

        if 'rolling_mean' in features_to_add:
            window = max(1, int(feature_cfg.get('rolling_window', 3)))
            rolling_mean = self._rolling_statistic(sequences, window, np.mean)
            engineered_channels.append(self._normalize_feature(rolling_mean, feature_cfg))

        if 'rolling_std' in features_to_add:
            window = max(2, int(feature_cfg.get('rolling_window', 3)))
            rolling_std = self._rolling_statistic(sequences, window, np.std)
            engineered_channels.append(self._normalize_feature(rolling_std, feature_cfg))

        if len(engineered_channels) == 1:
            return sequences

        combined = np.concatenate(engineered_channels, axis=-1)
        return combined

    def _rolling_statistic(
        self,
        data: np.ndarray,
        window: int,
        reducer
    ) -> np.ndarray:
        """Compute rolling statistic along the time axis."""
        n_samples, seq_len, channels = data.shape
        result = np.empty((n_samples, seq_len, channels), dtype=data.dtype)
        for idx in range(seq_len):
            start = max(0, idx - window + 1)
            segment = data[:, start:idx + 1, :]
            result[:, idx, :] = reducer(segment, axis=1)
        return result

    def _normalize_feature(self, feature: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """Apply optional normalization to engineered features."""
        if not config.get('normalize', True):
            return feature
        mean = feature.mean(axis=(0, 1), keepdims=True)
        std = feature.std(axis=(0, 1), keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        return (feature - mean) / std

    def _update_model_input_dims(self, input_dim: int) -> None:
        """Ensure every model consumes the engineered feature dimension."""
        for model_name, model_config in self.config['models'].items():
            if model_config.get('input_dim') != input_dim:
                logger.debug(
                    f"Updating {model_name} input_dim from "
                    f"{model_config.get('input_dim')} to {input_dim}"
                )
            model_config['input_dim'] = input_dim

    def create_models(self) -> None:
        """Create and initialize all models."""
        try:
            logger.info("Creating models...")

            # LSTM model
            lstm_config = self.config['models']['lstm']
            self.models['lstm'] = LSTMGlucosePredictor(
                input_dim=lstm_config['input_dim'],
                hidden_dim=lstm_config['hidden_dim'],
                output_dim=lstm_config['output_dim'],
                dropout=lstm_config['dropout'],
                device=self.device
            )

            # Transformer model
            transformer_config = self.config['models']['transformer']
            self.models['transformer'] = TransformerGlucosePredictor(
                input_dim=transformer_config['input_dim'],
                hidden_dim=transformer_config['hidden_dim'],
                output_dim=transformer_config['output_dim'],
                dropout=transformer_config['dropout'],
                device=self.device
            )

            # GluFormer model
            gluformer_config = self.config['models']['gluformer']
            self.models['gluformer'] = GluFormerPredictor(
                input_dim=gluformer_config['input_dim'],
                hidden_dim=gluformer_config['hidden_dim'],
                output_dim=gluformer_config['output_dim'],
                dropout=gluformer_config['dropout'],
                device=self.device
            )

            # Wavelet GluFormer model
            wavelet_config = self.config['models']['wavelet_gluformer']
            self.models['wavelet_gluformer'] = WaveletGluFormerPredictor(
                input_dim=wavelet_config['input_dim'],
                hidden_dim=wavelet_config['hidden_dim'],
                output_dim=wavelet_config['output_dim'],
                dropout=wavelet_config['dropout'],
                wavelet=wavelet_config['wavelet'],
                wavelet_levels=wavelet_config['wavelet_levels'],
                device=self.device
            )

            logger.info(f"Created {len(self.models)} models: {list(self.models.keys())}")

        except Exception as e:
            raise ModelTrainingError(f"Model creation failed: {str(e)}")

    def train_models(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Train all models individually."""
        try:
            logger.info("Training individual models...")

            training_config = self.config['training']
            training_results = {}

            scheduler_cfg = training_config.get('lr_scheduler', {})
            early_stop_cfg = training_config.get('early_stopping', {})
            grad_clip = training_config.get('gradient_clip', 1.0)

            for model_name, model in self.models.items():
                logger.info(f"Training {model_name} model...")

                result = model.train_model(
                    train_data=data_dict['train_sequences'],
                    train_targets=data_dict['train_targets'],
                    val_data=data_dict['val_sequences'],
                    val_targets=data_dict['val_targets'],
                    epochs=training_config['epochs'],
                    learning_rate=training_config['learning_rate'],
                    batch_size=training_config['batch_size'],
                    patience=training_config['patience'],
                    scheduler_config=scheduler_cfg,
                    early_stopping_config=early_stop_cfg,
                    grad_clip=grad_clip
                )

                training_results[model_name] = result

                # Save individual model
                if self.config['save_models']:
                    model_path = self.output_dir / f"{model_name}_model.pt"
                    model.save_model(str(model_path))

            self.training_history = training_results
            logger.info("Individual model training completed")

            return training_results

        except Exception as e:
            raise ModelTrainingError(f"Model training failed: {str(e)}")

    def create_ensemble(self, data_dict: Dict[str, torch.Tensor]) -> None:
        """Create and configure ensemble."""
        try:
            logger.info("Creating ensemble...")

            ensemble_config = self.config['ensemble']
            strategy = ensemble_config['strategy']

            if strategy == 'weighted':
                self.ensemble = WeightedEnsemble(self.device)
            elif strategy == 'voting':
                self.ensemble = VotingEnsemble(device=self.device)
            elif strategy == 'stacking':
                self.ensemble = StackingEnsemble(device=self.device)
            else:
                raise GlucosePredictionError(f"Unknown ensemble strategy: {strategy}")

            # Add models to ensemble
            for model_name, model in self.models.items():
                self.ensemble.add_model(model, name=model_name)

            # Train stacking ensemble if needed
            if strategy == 'stacking':
                self.ensemble.train_meta_learner(
                    data_dict['train_sequences'],
                    data_dict['train_targets'],
                    data_dict['val_sequences'],
                    data_dict['val_targets']
                )

            # Update weights for weighted ensemble
            if strategy == 'weighted' and ensemble_config['auto_weight_update']:
                self.ensemble.update_weights(
                    data_dict['val_sequences'],
                    data_dict['val_targets']
                )

            logger.info(f"Ensemble created with strategy: {strategy}")
            logger.info(f"Model weights: {self.ensemble.get_model_weights()}")

        except Exception as e:
            raise GlucosePredictionError(f"Ensemble creation failed: {str(e)}")

    def setup_monitoring(self, reference_data: torch.Tensor) -> None:
        """Setup drift detection and anomaly detection monitoring."""
        try:
            if not self.config['monitoring']['enabled']:
                return

            logger.info("Setting up monitoring...")

            # Setup drift detection
            monitoring_config = self.config['monitoring']
            self.drift_detector = DriftDetector(
                reference_data=reference_data,
                window_size=monitoring_config['window_size'],
                drift_threshold=monitoring_config['drift_threshold']
            )

            # Setup anomaly detection
            if self.config['anomaly_detection']['enabled']:
                logger.info("Setting up anomaly detection...")
                anomaly_config = self.config['anomaly_detection']
                self.anomaly_detector = GlucoseAnomalyDetector(
                    sequence_length=anomaly_config['sequence_length'],
                    normal_threshold=anomaly_config['normal_threshold'],
                    anomaly_threshold=anomaly_config['anomaly_threshold'],
                    critical_threshold=anomaly_config['critical_threshold'],
                    device=self.device
                )

                # Train anomaly detector on normal sequences
                if reference_data.dim() == 3:
                    normal_sequences = reference_data.cpu().numpy()
                    self.anomaly_detector.train_autoencoder(normal_sequences, epochs=50)

            logger.info("Monitoring setup completed")

        except Exception as e:
            logger.warning(f"Monitoring setup failed: {str(e)}")

    def train_complete_system(self, sequences: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
        """Train the complete enhanced system."""
        try:
            logger.info("Starting complete system training...")

            # Prepare data
            data_dict = self.prepare_data(sequences, targets)

            # Create models
            self.create_models()

            # Train individual models
            training_results = self.train_models(data_dict)

            # Create ensemble
            self.create_ensemble(data_dict)

            # Setup monitoring
            self.setup_monitoring(data_dict['train_sequences'])

            # Evaluate ensemble
            ensemble_metrics = self.evaluate_ensemble(
                data_dict['val_sequences'],
                data_dict['val_targets']
            )

            # Save system state
            self.save_system_state()

            self.is_trained = True

            final_results = {
                'individual_models': training_results,
                'ensemble_metrics': ensemble_metrics,
                'ensemble_weights': self.ensemble.get_model_weights(),
                'training_completed': True,
                'timestamp': datetime.now().isoformat()
            }

            logger.info("Complete system training finished successfully")

            return final_results

        except Exception as e:
            raise ModelTrainingError(f"Complete system training failed: {str(e)}")

    def predict(self, sequences: np.ndarray, use_monitoring: bool = True) -> Dict[str, Any]:
        """Make predictions using the ensemble system."""
        if not self.is_trained:
            raise GlucosePredictionError("System not trained")

        try:
            # Convert to tensor
            sequences_tensor = torch.FloatTensor(sequences)
            if sequences_tensor.dim() == 2:
                sequences_tensor = sequences_tensor.unsqueeze(-1)

            # Ensemble prediction
            ensemble_prediction = self.ensemble.predict(sequences_tensor)

            # Individual model predictions for comparison
            individual_predictions = {}
            for model_name, model in self.models.items():
                individual_predictions[model_name] = model.predict(sequences_tensor)

            # Monitoring
            if use_monitoring and self.drift_detector:
                self.drift_detector.log_prediction(sequences_tensor, ensemble_prediction)

            results = {
                'ensemble_prediction': ensemble_prediction.cpu().numpy(),
                'individual_predictions': {
                    name: pred.cpu().numpy()
                    for name, pred in individual_predictions.items()
                },
                'model_weights': self.ensemble.get_model_weights(),
                'timestamp': datetime.now().isoformat()
            }

            return results

        except Exception as e:
            raise GlucosePredictionError(f"Prediction failed: {str(e)}")

    def evaluate_ensemble(self, test_sequences: torch.Tensor,
                         test_targets: torch.Tensor) -> Dict[str, float]:
        """Evaluate ensemble performance."""
        try:
            # Move tensors to device
            test_sequences = test_sequences.to(self.device)
            test_targets = test_targets.to(self.device)

            # Ensemble prediction
            ensemble_pred = self.ensemble.predict(test_sequences)

            # Calculate metrics
            mse = torch.mean((ensemble_pred - test_targets) ** 2).item()
            mae = torch.mean(torch.abs(ensemble_pred - test_targets)).item()
            rmse = np.sqrt(mse)

            # R² score
            ss_res = torch.sum((test_targets - ensemble_pred) ** 2)
            ss_tot = torch.sum((test_targets - torch.mean(test_targets)) ** 2)
            r2 = (1 - ss_res / ss_tot).item()

            # Multi-step analysis
            step_metrics = {}
            for step in range(test_targets.shape[1]):
                step_pred = ensemble_pred[:, step]
                step_target = test_targets[:, step]

                step_mae = torch.mean(torch.abs(step_pred - step_target)).item()
                step_rmse = torch.sqrt(torch.mean((step_pred - step_target) ** 2)).item()

                step_metrics[f't+{step+1}'] = {
                    'mae': step_mae,
                    'rmse': step_rmse
                }

            metrics = {
                'overall_mae': mae,
                'overall_rmse': rmse,
                'overall_mse': mse,
                'overall_r2': r2,
                'step_metrics': step_metrics
            }

            logger.info(f"Ensemble evaluation: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")

            return metrics

        except Exception as e:
            raise GlucosePredictionError(f"Ensemble evaluation failed: {str(e)}")

    def check_drift(self) -> Dict[str, Any]:
        """Check for model drift."""
        if not self.drift_detector:
            return {'drift_monitoring': 'disabled'}

        try:
            drift_results = self.drift_detector.detect_drift()

            if drift_results['drift_detected']:
                logger.warning("Model drift detected!")
                logger.warning(f"Drift types: {drift_results['drift_types']}")

            return drift_results

        except Exception as e:
            logger.error(f"Drift detection failed: {str(e)}")
            return {'error': str(e)}

    def save_system_state(self) -> None:
        """Save complete system state."""
        try:
            # Save configuration
            config_path = self.output_dir / 'system_config.json'
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)

            # Save training history
            history_path = self.output_dir / 'training_history.json'
            with open(history_path, 'w') as f:
                # Convert torch tensors to lists for JSON serialization
                serializable_history = {}
                for model_name, history in self.training_history.items():
                    serializable_history[model_name] = {
                        'history': history['history'],
                        'best_val_loss': history['best_val_loss'],
                        'total_epochs': history['total_epochs']
                    }
                json.dump(serializable_history, f, indent=2)

            # Save ensemble weights
            if self.ensemble:
                weights_path = self.output_dir / 'ensemble_weights.json'
                with open(weights_path, 'w') as f:
                    json.dump(self.ensemble.get_model_weights(), f, indent=2)

            logger.info(f"System state saved to {self.output_dir}")

        except Exception as e:
            logger.error(f"Failed to save system state: {str(e)}")

    def generate_system_report(self) -> str:
        """Generate comprehensive system report."""
        if not self.is_trained:
            return "System not trained - no report available"

        report_lines = [
            "# Enhanced Glucose Prediction System Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## System Configuration",
            f"- Models: {list(self.models.keys())}",
            f"- Ensemble Strategy: {self.config['ensemble']['strategy']}",
            f"- Data Augmentation: {'Enabled' if self.config['augmentation']['enabled'] else 'Disabled'}",
            f"- Monitoring: {'Enabled' if self.config['monitoring']['enabled'] else 'Disabled'}",
            "",
            "## Model Performance",
        ]

        # Add individual model performance
        for model_name, history in self.training_history.items():
            report_lines.extend([
                f"### {model_name.upper()} Model",
                f"- Best Validation Loss: {history['best_val_loss']:.6f}",
                f"- Training Epochs: {history['total_epochs']}",
                f"- Final Train MAE: {history['history']['train_mae'][-1]:.4f}",
                f"- Final Val MAE: {history['history']['val_mae'][-1]:.4f}",
                ""
            ])

        # Add ensemble weights
        if self.ensemble:
            report_lines.extend([
                "## Ensemble Configuration",
                "Model Weights:"
            ])
            for model_name, weight in self.ensemble.get_model_weights().items():
                report_lines.append(f"- {model_name}: {weight:.4f}")

        return "\n".join(report_lines)

    def setup_personalization(self) -> None:
        """Setup LoRA personalization components."""
        if not self.config['lora']['enabled']:
            return

        logger.info("Setting up LoRA personalization...")

        # Initialize LoRA adapter
        self.lora_adapter = LoRAAdapter(
            rank=self.config['lora']['rank'],
            alpha=self.config['lora']['alpha'],
            dropout=self.config['lora']['dropout'],
            target_modules=self.config['lora']['target_modules'],
            glucose_specific=self.config['lora']['glucose_specific']
        )

        # Initialize personalization manager
        if self.models and 'gluformer' in self.models:
            self.personalization_manager = PersonalizationManager(
                base_model=self.models['gluformer'],
                lora_config=self.config['lora']
            )

        # Initialize personalized MoE head if enabled
        if self.config['personalized_moe']['enabled'] and 'gluformer' in self.models:
            self.personalized_moe_head = PersonalizedMoEHead(
                input_dim=self.config['models']['gluformer']['hidden_dim'],
                output_steps=self.config['models']['gluformer']['output_dim'],
                lora_rank=self.config['personalized_moe']['lora_rank'],
                lora_alpha=self.config['personalized_moe']['lora_alpha'],
                device=self.device
            )

        logger.info("LoRA personalization setup completed")

    def personalize_for_patient(self, patient_id: str, patient_data: np.ndarray,
                               patient_targets: np.ndarray, epochs: int = None) -> Dict[str, Any]:
        """
        Personalize the model for a specific patient using LoRA.

        Args:
            patient_id: Unique patient identifier
            patient_data: Patient-specific glucose sequences (n_samples, seq_len, features)
            patient_targets: Patient-specific targets (n_samples, output_dim)
            epochs: Training epochs (uses config default if None)

        Returns:
            Personalization results including metrics and efficiency stats
        """
        if not self.config['lora']['enabled']:
            raise GlucosePredictionError("LoRA personalization not enabled in configuration")

        if self.personalization_manager is None:
            self.setup_personalization()

        logger.info(f"Starting personalization for patient {patient_id}")

        # Convert to tensors
        patient_data_tensor = torch.FloatTensor(patient_data).to(self.device)
        patient_targets_tensor = torch.FloatTensor(patient_targets).to(self.device)

        # Use configured epochs or default
        training_epochs = epochs or self.config['lora']['personalization']['max_epochs']

        # Personalize model
        personalization_results = self.personalization_manager.personalize_model(
            patient_id=patient_id,
            patient_data=patient_data_tensor,
            patient_targets=patient_targets_tensor,
            epochs=training_epochs
        )

        # Setup personalized MoE if available
        if self.personalized_moe_head is not None:
            self.personalized_moe_head.create_patient_adaptation(patient_id)
            logger.info(f"Created personalized MoE adaptation for patient {patient_id}")

        logger.info(f"Personalization completed for patient {patient_id}")

        return {
            'patient_id': patient_id,
            'personalization_results': personalization_results,
            'has_personalized_moe': self.personalized_moe_head is not None,
            'data_samples': len(patient_data),
            'glucose_metrics': personalization_results.get('glucose_metrics_history', [])
        }

    def predict_personalized(self, sequences: np.ndarray, patient_id: str) -> Dict[str, Any]:
        """
        Make personalized predictions for a specific patient.

        Args:
            sequences: Input sequences (n_samples, seq_len, features)
            patient_id: Patient identifier for personalization

        Returns:
            Personalized predictions and routing information
        """
        if not self.config['lora']['enabled']:
            return self.predict(sequences, use_monitoring=False)

        if patient_id not in self.personalization_manager.patient_adapters:
            logger.warning(f"No personalization found for patient {patient_id}, using base model")
            return self.predict(sequences, use_monitoring=False)

        # Convert to tensor
        sequences_tensor = torch.FloatTensor(sequences).to(self.device)
        if sequences_tensor.dim() == 2:
            sequences_tensor = sequences_tensor.unsqueeze(-1)

        # Get personalized model
        adapter = self.personalization_manager.patient_adapters[patient_id]
        personalized_model = adapter.apply_lora(
            self.personalization_manager.base_model,
            f"patient_{patient_id}"
        )

        # Make predictions
        personalized_model.eval()
        with torch.no_grad():
            base_predictions = personalized_model(sequences_tensor)

        results = {
            'personalized_prediction': base_predictions.cpu().numpy(),
            'patient_id': patient_id,
            'model_type': 'personalized_lora'
        }

        # Add personalized MoE predictions if available
        if self.personalized_moe_head is not None:
            self.personalized_moe_head.set_active_patient(patient_id)

            with torch.no_grad():
                # Extract features from base model for MoE
                if hasattr(personalized_model, 'lstm'):
                    lstm_out, _ = personalized_model.lstm(sequences_tensor)
                    features = lstm_out[:, -1, :]  # Last time step
                elif hasattr(personalized_model, 'gru'):
                    gru_out, _ = personalized_model.gru(sequences_tensor)
                    features = gru_out[:, -1, :]
                else:
                    # Fallback: use mean of sequence
                    features = torch.mean(sequences_tensor, dim=1).squeeze(-1)

                moe_predictions, expert_info = self.personalized_moe_head(
                    features, patient_id=patient_id, return_expert_info=True
                )

                results.update({
                    'personalized_moe_prediction': moe_predictions.cpu().numpy(),
                    'expert_routing': {
                        'patient_id': expert_info['patient_id'],
                        'expert_usage': expert_info['expert_usage'].tolist(),
                        'top_k_experts': expert_info['top_k_indices'].tolist()
                    }
                })

        return results

    def get_personalization_stats(self) -> Dict[str, Any]:
        """Get statistics about current personalizations."""
        if not self.config['lora']['enabled'] or self.personalization_manager is None:
            return {'personalization_enabled': False}

        stats = {
            'personalization_enabled': True,
            'num_patients': len(self.personalization_manager.patient_adapters),
            'patient_ids': list(self.personalization_manager.patient_adapters.keys()),
            'lora_config': {
                'rank': self.config['lora']['rank'],
                'alpha': self.config['lora']['alpha'],
                'dropout': self.config['lora']['dropout']
            }
        }

        if self.personalized_moe_head is not None:
            moe_stats = self.personalized_moe_head.get_personalization_stats()
            stats['personalized_moe'] = moe_stats

        return stats


def main():
    """Main function for testing the enhanced system."""
    print("\n" + "="*80)
    print("Enhanced Glucose Prediction System")
    print("Modern Engineering Standards Implementation")
    print("="*80 + "\n")

    # Generate sample data
    np.random.seed(42)
    num_samples = 5000
    seq_length = 12
    output_steps = 6

    # Create realistic glucose sequences
    sequences = []
    targets = []

    for i in range(num_samples):
        # Base glucose pattern
        base_glucose = np.random.normal(0.5, 0.1)

        # Create sequence with trend
        sequence = []
        current = base_glucose

        for j in range(seq_length):
            # Add some trend and noise
            trend = np.random.normal(0, 0.02)
            noise = np.random.normal(0, 0.01)
            current = np.clip(current + trend + noise, 0, 1)
            sequence.append(current)

        # Create targets (future values)
        target_sequence = []
        for j in range(output_steps):
            trend = np.random.normal(0, 0.02)
            noise = np.random.normal(0, 0.01)
            current = np.clip(current + trend + noise, 0, 1)
            target_sequence.append(current)

        sequences.append(sequence)
        targets.append(target_sequence)

    sequences = np.array(sequences)
    targets = np.array(targets)

    # Initialize system
    config = {
        'training': {
            'epochs': 20,  # Reduced for demo
            'batch_size': 32,
            'learning_rate': 0.001,
            'patience': 10,
            'val_split': 0.2
        },
        'augmentation': {
            'enabled': True,
            'factor': 1.5,  # Reduced for demo
            'methods': ['noise', 'scaling']
        }
    }

    system = EnhancedGlucosePredictionSystem(config)

    # Train system
    results = system.train_complete_system(sequences, targets)

    # Make test predictions
    test_sequences = sequences[:100]  # Use first 100 for testing
    predictions = system.predict(test_sequences)

    # Check for drift
    drift_status = system.check_drift()

    # Generate report
    report = system.generate_system_report()

    # Print results
    print("Training Results:")
    print(f"   Ensemble MAE: {results['ensemble_metrics']['overall_mae']:.4f}")
    print(f"   Ensemble RMSE: {results['ensemble_metrics']['overall_rmse']:.4f}")
    print(f"   Ensemble R2: {results['ensemble_metrics']['overall_r2']:.4f}")

    print(f"\nModel Weights:")
    for model, weight in results['ensemble_weights'].items():
        print(f"   {model}: {weight:.4f}")

    print(f"\nTest Predictions Shape: {predictions['ensemble_prediction'].shape}")
    print(f"Drift Status: {'Detected' if drift_status.get('drift_detected', False) else 'Not Detected'}")

    print(f"\nResults saved to: {system.output_dir}")

    print("\n" + "="*80)
    print("Enhanced Glucose Prediction System Demo Complete!")
    print("All modern engineering standards implemented")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
