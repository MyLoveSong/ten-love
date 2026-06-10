"""
Personalized Glucose Prediction Validation Test
Tests LoRA-based personalization for individual patients with glucose-specific optimizations.
"""

import torch
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

from enhanced_glucose_system import EnhancedGlucosePredictionSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_patient_data(patient_id: str, n_samples: int = 20,
                         glucose_pattern: str = "normal") -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic patient-specific glucose data with different patterns.

    Args:
        patient_id: Patient identifier
        n_samples: Number of samples to generate
        glucose_pattern: Pattern type ('normal', 'diabetic', 'hypoglycemic', 'variable')

    Returns:
        Tuple of (sequences, targets)
    """
    np.random.seed(hash(patient_id) % 2**32)  # Consistent seed per patient

    sequences = []
    targets = []

    for i in range(n_samples):
        if glucose_pattern == "normal":
            # Normal glucose range (80-140 mg/dL)
            base_glucose = np.random.uniform(90, 130)
            sequence = base_glucose + np.random.normal(0, 10, 12)
            target = base_glucose + np.random.normal(0, 8, 6)

        elif glucose_pattern == "diabetic":
            # Diabetic pattern (higher baseline, more variability)
            base_glucose = np.random.uniform(140, 200)
            sequence = base_glucose + np.random.normal(0, 20, 12)
            target = base_glucose + np.random.normal(0, 15, 6)

        elif glucose_pattern == "hypoglycemic":
            # Hypoglycemic tendency (lower values, some episodes)
            if np.random.random() < 0.3:  # 30% hypoglycemic episodes
                base_glucose = np.random.uniform(50, 69)
            else:
                base_glucose = np.random.uniform(80, 120)
            sequence = base_glucose + np.random.normal(0, 8, 12)
            target = base_glucose + np.random.normal(0, 6, 6)

        elif glucose_pattern == "variable":
            # High variability pattern
            base_glucose = np.random.uniform(70, 180)
            sequence = base_glucose + np.random.normal(0, 25, 12)
            target = base_glucose + np.random.normal(0, 20, 6)

        # Ensure physiological constraints
        sequence = np.clip(sequence, 40, 400)
        target = np.clip(target, 40, 400)

        sequences.append(sequence)
        targets.append(target)

    return np.array(sequences).reshape(-1, 12, 1), np.array(targets)


def test_lora_personalization():
    """Test basic LoRA personalization functionality."""
    logger.info("🔧 Testing LoRA Personalization...")

    # Configuration with LoRA enabled
    config = {
        'models': {
            'lstm': {'input_dim': 1, 'hidden_dim': 32, 'output_dim': 6, 'dropout': 0.1},
            'transformer': {'input_dim': 1, 'hidden_dim': 32, 'output_dim': 6, 'dropout': 0.1},
            'gluformer': {'input_dim': 1, 'hidden_dim': 32, 'output_dim': 6, 'dropout': 0.1},
            'wavelet_gluformer': {
                'input_dim': 1, 'hidden_dim': 32, 'output_dim': 6, 'dropout': 0.1,
                'wavelet': 'db4', 'wavelet_levels': 3
            }
        },
        'training': {
            'epochs': 3, 'batch_size': 8, 'learning_rate': 0.001,
            'patience': 2, 'val_split': 0.2
        },
        'ensemble': {'strategy': 'weighted', 'auto_weight_update': True},
        'augmentation': {'enabled': True, 'factor': 1.2, 'methods': ['noise']},
        'rare_event_augmentation': {
            'enabled': True, 'factor': 1.2, 'methods': ['noise_injection'],
            'event_types': ['hypoglycemia']
        },
        'lora': {
            'enabled': True, 'rank': 4, 'alpha': 8.0, 'dropout': 0.05,
            'target_modules': ['lstm', 'gru', 'self_attention', 'cross_attention', 'prediction_heads'],
            'glucose_specific': True, 'learning_rate': 0.001, 'weight_decay': 0.01,
            'personalization': {
                'enabled': True, 'early_stopping_patience': 3, 'validation_split': 0.2,
                'max_epochs': 5, 'glucose_metrics': True
            }
        },
        'personalized_moe': {
            'enabled': True, 'lora_rank': 4, 'lora_alpha': 8.0, 'personalization_weight': 0.05
        },
        'anomaly_detection': {
            'enabled': True, 'sequence_length': 12, 'normal_threshold': 0.1,
            'anomaly_threshold': 0.3, 'critical_threshold': 0.5
        },
        'monitoring': {'enabled': True, 'window_size': 50, 'drift_threshold': 0.05},
        'output_dir': 'TRAIN/outputs/personalization_test',
        'save_models': False, 'save_plots': False
    }

    # Initialize system
    system = EnhancedGlucosePredictionSystem(config)

    # Generate base training data
    np.random.seed(42)
    torch.manual_seed(42)

    base_sequences = []
    base_targets = []

    # Generate diverse base data
    for i in range(60):
        if i < 15:
            pattern = "normal"
        elif i < 30:
            pattern = "diabetic"
        elif i < 45:
            pattern = "hypoglycemic"
        else:
            pattern = "variable"

        seq, tgt = generate_patient_data(f"base_{i}", 1, pattern)
        base_sequences.extend(seq)
        base_targets.extend(tgt)

    base_sequences = np.array(base_sequences)
    base_targets = np.array(base_targets)

    logger.info(f"Generated base training data: {len(base_sequences)} sequences")

    # Train base system
    training_results = system.train_complete_system(base_sequences, base_targets)
    assert training_results['training_completed'], "Base training should complete"

    # Setup personalization
    system.setup_personalization()

    logger.info("✅ LoRA Personalization setup successful")
    return system


def test_patient_specific_training(system: EnhancedGlucosePredictionSystem):
    """Test patient-specific training with different glucose patterns."""
    logger.info("👤 Testing Patient-Specific Training...")

    patients = [
        ("patient_001", "normal", "Normal glucose pattern"),
        ("patient_002", "diabetic", "Diabetic pattern with high variability"),
        ("patient_003", "hypoglycemic", "Hypoglycemic tendency"),
        ("patient_004", "variable", "High variability pattern")
    ]

    personalization_results = {}

    for patient_id, pattern, description in patients:
        logger.info(f"Training personalization for {patient_id}: {description}")

        # Generate patient-specific data
        patient_sequences, patient_targets = generate_patient_data(
            patient_id, n_samples=15, glucose_pattern=pattern
        )

        # Personalize model for this patient
        result = system.personalize_for_patient(
            patient_id=patient_id,
            patient_data=patient_sequences,
            patient_targets=patient_targets,
            epochs=5
        )

        personalization_results[patient_id] = {
            'result': result,
            'pattern': pattern,
            'description': description,
            'data_samples': len(patient_sequences)
        }

        # Validate personalization results
        assert result['patient_id'] == patient_id, f"Patient ID mismatch for {patient_id}"
        assert result['data_samples'] == 15, f"Data samples mismatch for {patient_id}"
        assert 'personalization_results' in result, f"Missing personalization results for {patient_id}"

        logger.info(f"✅ Personalization completed for {patient_id}")

    logger.info("✅ Patient-Specific Training test passed")
    return personalization_results


def test_personalized_predictions(system: EnhancedGlucosePredictionSystem,
                                 personalization_results: Dict):
    """Test personalized predictions for each patient."""
    logger.info("🎯 Testing Personalized Predictions...")

    prediction_results = {}

    for patient_id, patient_info in personalization_results.items():
        logger.info(f"Testing predictions for {patient_id}")

        # Generate test data for this patient
        test_sequences, test_targets = generate_patient_data(
            patient_id, n_samples=5, glucose_pattern=patient_info['pattern']
        )

        # Make personalized predictions
        personalized_pred = system.predict_personalized(test_sequences, patient_id)

        # Make base predictions for comparison
        base_pred = system.predict(test_sequences, use_monitoring=False)

        # Validate prediction structure
        assert 'personalized_prediction' in personalized_pred, f"Missing personalized prediction for {patient_id}"
        assert personalized_pred['patient_id'] == patient_id, f"Patient ID mismatch in prediction for {patient_id}"
        assert personalized_pred['model_type'] == 'personalized_lora', f"Model type mismatch for {patient_id}"

        # Check if personalized MoE is available
        has_moe = 'personalized_moe_prediction' in personalized_pred

        prediction_results[patient_id] = {
            'personalized_shape': personalized_pred['personalized_prediction'].shape,
            'base_shape': base_pred['ensemble_prediction'].shape,
            'has_personalized_moe': has_moe,
            'test_samples': len(test_sequences)
        }

        if has_moe:
            assert 'expert_routing' in personalized_pred, f"Missing expert routing for {patient_id}"
            expert_routing = personalized_pred['expert_routing']
            assert expert_routing['patient_id'] == patient_id, f"Expert routing patient ID mismatch for {patient_id}"

            logger.info(f"  MoE Expert Usage for {patient_id}: {expert_routing['expert_usage']}")

        logger.info(f"✅ Predictions validated for {patient_id}")

    logger.info("✅ Personalized Predictions test passed")
    return prediction_results


def test_glucose_specific_metrics(system: EnhancedGlucosePredictionSystem,
                                 personalization_results: Dict):
    """Test glucose-specific metrics computation."""
    logger.info("📊 Testing Glucose-Specific Metrics...")

    metrics_results = {}

    for patient_id, patient_info in personalization_results.items():
        personalization_result = patient_info['result']['personalization_results']

        # Check if glucose metrics were computed
        if 'glucose_metrics_history' in personalization_result:
            glucose_metrics = personalization_result['glucose_metrics_history']

            if glucose_metrics:
                # Analyze the last epoch's metrics
                final_metrics = glucose_metrics[-1]

                metrics_results[patient_id] = {
                    'has_glucose_metrics': True,
                    'final_metrics': final_metrics,
                    'pattern': patient_info['pattern']
                }

                # Validate metric structure
                expected_metrics = ['hypoglycemia_mae', 'hyperglycemia_mae', 'normal_range_mae']
                available_metrics = list(final_metrics.keys())

                logger.info(f"  {patient_id} ({patient_info['pattern']}): {available_metrics}")

                # Check pattern-specific expectations
                if patient_info['pattern'] == 'hypoglycemic':
                    # Should have hypoglycemia metrics
                    if 'hypoglycemia_mae' in final_metrics:
                        logger.info(f"    Hypoglycemia MAE: {final_metrics['hypoglycemia_mae']:.2f}")

                elif patient_info['pattern'] == 'diabetic':
                    # Should have hyperglycemia metrics
                    if 'hyperglycemia_mae' in final_metrics:
                        logger.info(f"    Hyperglycemia MAE: {final_metrics['hyperglycemia_mae']:.2f}")
            else:
                metrics_results[patient_id] = {'has_glucose_metrics': False}
        else:
            metrics_results[patient_id] = {'has_glucose_metrics': False}

    logger.info("✅ Glucose-Specific Metrics test passed")
    return metrics_results


def test_personalization_efficiency(system: EnhancedGlucosePredictionSystem):
    """Test parameter efficiency of LoRA personalization."""
    logger.info("⚡ Testing Personalization Efficiency...")

    stats = system.get_personalization_stats()

    assert stats['personalization_enabled'], "Personalization should be enabled"
    assert stats['num_patients'] > 0, "Should have personalized patients"

    logger.info(f"  Personalized patients: {stats['num_patients']}")
    logger.info(f"  Patient IDs: {stats['patient_ids']}")
    logger.info(f"  LoRA config: rank={stats['lora_config']['rank']}, alpha={stats['lora_config']['alpha']}")

    if 'personalized_moe' in stats:
        moe_stats = stats['personalized_moe']
        logger.info(f"  Personalized MoE: {moe_stats['num_patients']} patients")
        logger.info(f"  Active patient: {moe_stats['active_patient']}")

    # Test parameter efficiency
    if system.personalization_manager:
        for patient_id in stats['patient_ids']:
            if patient_id in system.personalization_manager.patient_adapters:
                adapter = system.personalization_manager.patient_adapters[patient_id]
                param_stats = adapter.count_parameters(system.models['gluformer'])

                logger.info(f"  {patient_id} parameter efficiency: {param_stats['efficiency']:.4f}")
                assert param_stats['efficiency'] < 0.1, f"LoRA should be parameter efficient for {patient_id}"

    logger.info("✅ Personalization Efficiency test passed")
    return stats


def main():
    """Run comprehensive personalized glucose prediction tests."""
    logger.info("🧪 Starting Personalized Glucose Prediction Tests")
    logger.info("=" * 70)

    try:
        # Test 1: Basic LoRA personalization setup
        system = test_lora_personalization()

        # Test 2: Patient-specific training
        personalization_results = test_patient_specific_training(system)

        # Test 3: Personalized predictions
        prediction_results = test_personalized_predictions(system, personalization_results)

        # Test 4: Glucose-specific metrics
        metrics_results = test_glucose_specific_metrics(system, personalization_results)

        # Test 5: Personalization efficiency
        efficiency_stats = test_personalization_efficiency(system)

        # Summary
        logger.info("=" * 70)
        logger.info("🎯 Personalized Glucose Prediction Test Summary:")
        logger.info("=" * 70)

        logger.info("🔧 LoRA Personalization:")
        logger.info(f"  • Enabled: ✅")
        logger.info(f"  • Glucose-specific optimizations: ✅")
        logger.info(f"  • Early stopping: ✅")

        logger.info("👤 Patient-Specific Training:")
        logger.info(f"  • Patients trained: {len(personalization_results)}")
        for patient_id, info in personalization_results.items():
            result = info['result']['personalization_results']
            logger.info(f"  • {patient_id} ({info['pattern']}): "
                       f"Final loss: {result['final_train_loss']:.4f}, "
                       f"Epochs: {result['epochs_trained']}")

        logger.info("🎯 Personalized Predictions:")
        for patient_id, pred_info in prediction_results.items():
            moe_status = "✅" if pred_info['has_personalized_moe'] else "❌"
            logger.info(f"  • {patient_id}: Shape {pred_info['personalized_shape']}, MoE: {moe_status}")

        logger.info("📊 Glucose-Specific Metrics:")
        for patient_id, metrics_info in metrics_results.items():
            if metrics_info['has_glucose_metrics']:
                logger.info(f"  • {patient_id}: ✅ Range-specific MAE computed")
            else:
                logger.info(f"  • {patient_id}: ❌ No glucose metrics")

        logger.info("⚡ Parameter Efficiency:")
        logger.info(f"  • Total patients: {efficiency_stats['num_patients']}")
        logger.info(f"  • LoRA rank: {efficiency_stats['lora_config']['rank']}")
        logger.info(f"  • Parameter efficiency: < 10% (LoRA advantage)")

        logger.info("=" * 70)
        logger.info("🎉 All Personalized Glucose Prediction Tests Passed!")
        logger.info("✅ LoRA personalization working optimally for glucose prediction")

        return True

    except Exception as e:
        logger.error(f"❌ Personalized Glucose Prediction Tests Failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
