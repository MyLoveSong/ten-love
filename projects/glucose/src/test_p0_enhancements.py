"""
P0 Enhancements Validation Test
Tests MoE, LoRA, Rare Event Augmentation, and Anomaly Detection integrations.
"""

import torch
import numpy as np
import logging
from pathlib import Path
from datetime import datetime

from enhanced_glucose_system import EnhancedGlucosePredictionSystem
from core.moe_components import MoEGlucoseHead, MoELoss
from core.adapters import LoRAAdapter, PersonalizationManager
from data_processing.rare_event_augment import RareEventAugmenter, RareEventDetector
from monitoring.anomaly_autoencoder import GlucoseAnomalyDetector, AnomalyAlert

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_moe_components():
    """Test MoE components functionality."""
    logger.info("🧠 Testing MoE Components...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test MoE head
    moe_head = MoEGlucoseHead(
        input_dim=64,
        output_steps=6,
        num_experts=4,
        top_k=2,
        device=device
    )

    # Test forward pass
    batch_size = 8
    test_input = torch.randn(batch_size, 64, device=device)

    # Forward pass with expert info
    predictions, expert_info = moe_head(test_input, return_expert_info=True)

    assert predictions.shape == (batch_size, 6), f"Expected (8, 6), got {predictions.shape}"
    assert 'gates' in expert_info, "Expert info should contain gates"
    assert 'top_k_indices' in expert_info, "Expert info should contain top_k_indices"

    # Test MoE loss
    targets = torch.randn(batch_size, 6, device=device)
    moe_loss = MoELoss()
    loss_dict = moe_loss(predictions, targets, moe_head, test_input)

    assert 'total_loss' in loss_dict, "Loss dict should contain total_loss"
    assert 'prediction_loss' in loss_dict, "Loss dict should contain prediction_loss"
    assert 'load_balance_loss' in loss_dict, "Loss dict should contain load_balance_loss"

    logger.info("✅ MoE Components test passed")

    return {
        'predictions_shape': predictions.shape,
        'expert_usage': expert_info['expert_usage'].tolist(),
        'total_loss': loss_dict['total_loss'].item()
    }


def test_lora_adapters():
    """Test LoRA adapter functionality."""
    logger.info("🔧 Testing LoRA Adapters...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a simple model to adapt
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = torch.nn.LSTM(1, 32, batch_first=True)
            self.attention = torch.nn.MultiheadAttention(32, 4, batch_first=True)
            self.prediction_heads = torch.nn.ModuleList([
                torch.nn.Linear(32, 1) for _ in range(6)
            ])

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            features = attn_out[:, -1, :]
            predictions = [head(features) for head in self.prediction_heads]
            return torch.cat(predictions, dim=1)

    model = SimpleModel().to(device)

    # Test LoRA adapter
    lora_adapter = LoRAAdapter(rank=4, alpha=8.0, dropout=0.1)

    # Count parameters before adaptation
    original_params = lora_adapter.count_parameters(model)

    # Apply LoRA
    adapted_model = lora_adapter.apply_lora(model)

    # Count parameters after adaptation
    adapted_params = lora_adapter.count_parameters(adapted_model)

    # Test forward pass
    test_input = torch.randn(4, 12, 1, device=device)
    output = adapted_model(test_input)

    assert output.shape == (4, 6), f"Expected (4, 6), got {output.shape}"
    assert adapted_params['lora'] > 0, "Should have LoRA parameters"
    assert adapted_params['efficiency'] < 0.1, "LoRA should be parameter efficient"

    # Test personalization manager
    personalization_manager = PersonalizationManager(
        base_model=model,
        lora_config={'rank': 4, 'alpha': 8.0, 'learning_rate': 0.001}
    )

    # Create mock patient data
    patient_data = torch.randn(10, 12, 1, device=device)
    patient_targets = torch.randn(10, 6, device=device)

    # Test personalization
    result = personalization_manager.personalize_model(
        patient_id="test_patient",
        patient_data=patient_data,
        patient_targets=patient_targets,
        epochs=3
    )

    assert 'patient_id' in result, "Result should contain patient_id"
    assert 'final_loss' in result, "Result should contain final_loss"

    logger.info("✅ LoRA Adapters test passed")

    return {
        'original_params': original_params['total'],
        'lora_params': adapted_params['lora'],
        'efficiency': adapted_params['efficiency'],
        'personalization_loss': result['final_loss']
    }


def test_rare_event_augmentation():
    """Test rare event augmentation functionality."""
    logger.info("🎯 Testing Rare Event Augmentation...")

    # Create synthetic glucose data with rare events
    np.random.seed(42)

    # Normal sequences
    normal_sequences = []
    normal_targets = []

    # Rare event sequences (hypoglycemia)
    rare_sequences = []
    rare_targets = []

    for i in range(50):
        # Normal sequence (80-150 mg/dL)
        normal_seq = np.random.uniform(80, 150, 12)
        normal_tgt = np.random.uniform(80, 150, 6)
        normal_sequences.append(normal_seq)
        normal_targets.append(normal_tgt)

        # Rare event sequence (hypoglycemia: <70 mg/dL)
        if i < 10:  # Only 10 rare events
            rare_seq = np.random.uniform(50, 69, 12)
            rare_tgt = np.random.uniform(50, 69, 6)
            rare_sequences.append(rare_seq)
            rare_targets.append(rare_tgt)

    # Combine sequences
    all_sequences = np.array(normal_sequences + rare_sequences).reshape(-1, 12, 1)
    all_targets = np.array(normal_targets + rare_targets)

    # Test rare event detector
    detector = RareEventDetector()
    events = detector.detect_events(rare_sequences[0])

    assert 'hypoglycemia' in events, "Should detect hypoglycemia events"
    assert len(events['hypoglycemia']) > 0, "Should find hypoglycemia events"

    # Test rare event augmenter
    augmenter = RareEventAugmenter(
        augmentation_factor=2.0,
        methods=['smote', 'noise_injection'],
        random_state=42
    )

    # Get augmentation stats before
    original_count = len(all_sequences)

    # Apply augmentation
    augmented_sequences, augmented_targets = augmenter.augment_rare_events(
        all_sequences, all_targets,
        event_types=['hypoglycemia', 'hyperglycemia']
    )

    # Get stats
    stats = augmenter.get_augmentation_stats(all_sequences, augmented_sequences)

    assert len(augmented_sequences) >= original_count, "Should have more sequences after augmentation"
    assert stats['synthetic_sequences'] > 0, "Should generate synthetic sequences"

    logger.info("✅ Rare Event Augmentation test passed")

    return {
        'original_sequences': stats['original_sequences'],
        'synthetic_sequences': stats['synthetic_sequences'],
        'augmentation_factor': stats['augmentation_factor'],
        'rare_event_increase': stats['rare_event_increase']
    }


def test_anomaly_detection():
    """Test anomaly detection functionality."""
    logger.info("🚨 Testing Anomaly Detection...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create anomaly detector
    anomaly_detector = GlucoseAnomalyDetector(
        sequence_length=12,
        device=device
    )

    # Generate normal training data
    np.random.seed(42)
    normal_data = []
    for _ in range(100):
        # Normal glucose pattern (80-150 mg/dL)
        sequence = np.random.uniform(80, 150, 12)
        normal_data.append(sequence)

    normal_data = np.array(normal_data).reshape(-1, 12, 1)

    # Train autoencoder
    training_result = anomaly_detector.train_autoencoder(
        normal_data, epochs=20, batch_size=16
    )

    assert anomaly_detector.is_trained, "Anomaly detector should be trained"
    assert 'training_losses' in training_result, "Should return training losses"

    # Test normal sequence detection
    normal_sequence = np.random.uniform(80, 150, 12).tolist()
    normal_alert = anomaly_detector.detect_anomaly(normal_sequence)

    assert isinstance(normal_alert, AnomalyAlert), "Should return AnomalyAlert"
    assert normal_alert.severity in ['low', 'medium'], "Normal sequence should have low/medium severity"

    # Test anomalous sequence detection
    anomalous_sequence = [50, 45, 40, 35, 30, 35, 40, 45, 50, 55, 60, 65]  # Severe hypoglycemia
    anomaly_alert = anomaly_detector.detect_anomaly(anomalous_sequence)

    assert anomaly_alert.severity in ['high', 'critical'], "Anomalous sequence should have high/critical severity"
    assert anomaly_alert.alert_type in ['severe_hypoglycemia', 'hypoglycemia'], "Should detect hypoglycemia"

    # Test statistics
    stats = anomaly_detector.get_anomaly_statistics()

    assert 'total_alerts_24h' in stats, "Stats should contain total alerts"
    assert stats['is_trained'], "Should show as trained"

    logger.info("✅ Anomaly Detection test passed")

    return {
        'training_loss': training_result['final_loss'],
        'normal_alert_severity': normal_alert.severity,
        'anomaly_alert_severity': anomaly_alert.severity,
        'anomaly_score': anomaly_alert.anomaly_score
    }


def test_integrated_system():
    """Test integrated system with all P0 enhancements."""
    logger.info("🚀 Testing Integrated System...")

    # Configuration with P0 enhancements enabled
    config = {
        'models': {
            'gluformer': {
                'input_dim': 1,
                'hidden_dim': 32,  # Smaller for testing
                'output_dim': 6,
                'dropout': 0.1
            }
        },
        'training': {
            'epochs': 5,  # Quick test
            'batch_size': 16,
            'learning_rate': 0.001,
            'patience': 3,
            'val_split': 0.2
        },
        'ensemble': {
            'strategy': 'weighted',
            'auto_weight_update': True
        },
        'augmentation': {
            'enabled': True,
            'factor': 1.5,
            'methods': ['noise', 'scaling']
        },
        'rare_event_augmentation': {
            'enabled': True,
            'factor': 1.5,
            'methods': ['smote', 'noise_injection'],
            'event_types': ['hypoglycemia', 'hyperglycemia']
        },
        'anomaly_detection': {
            'enabled': True,
            'sequence_length': 12,
            'normal_threshold': 0.1,
            'anomaly_threshold': 0.3,
            'critical_threshold': 0.5
        },
        'monitoring': {
            'enabled': True,
            'window_size': 100,
            'drift_threshold': 0.05
        },
        'output_dir': 'TRAIN/outputs/p0_test',
        'save_models': True,
        'save_plots': False
    }

    # Initialize system
    system = EnhancedGlucosePredictionSystem(config)

    # Generate test data with rare events
    np.random.seed(42)
    torch.manual_seed(42)

    sequences = []
    targets = []

    # Generate mixed data (normal + rare events)
    for i in range(100):
        if i < 10:  # 10% rare events
            # Hypoglycemia sequence
            seq = np.random.uniform(50, 69, 12)
            tgt = np.random.uniform(50, 69, 6)
        elif i < 20:  # 10% hyperglycemia
            seq = np.random.uniform(181, 250, 12)
            tgt = np.random.uniform(181, 250, 6)
        else:  # 80% normal
            seq = np.random.uniform(80, 150, 12)
            tgt = np.random.uniform(80, 150, 6)

        sequences.append(seq)
        targets.append(tgt)

    sequences = np.array(sequences).reshape(-1, 12, 1)
    targets = np.array(targets)

    logger.info(f"Generated test data: {len(sequences)} sequences")

    # Train complete system
    training_results = system.train_complete_system(sequences, targets)

    assert training_results['training_completed'], "Training should complete successfully"
    assert 'individual_models' in training_results, "Should contain individual model results"
    assert 'ensemble_metrics' in training_results, "Should contain ensemble metrics"

    # Test prediction with anomaly detection
    test_sequences = sequences[:5]
    predictions = system.predict(test_sequences, use_monitoring=True)

    assert 'ensemble_prediction' in predictions, "Should contain ensemble predictions"

    # Test anomaly detection if enabled
    if system.anomaly_detector:
        anomaly_alert = system.anomaly_detector.detect_anomaly(test_sequences[0, :, 0].tolist())
        assert isinstance(anomaly_alert, AnomalyAlert), "Should return anomaly alert"

    logger.info("✅ Integrated System test passed")

    return {
        'training_completed': training_results['training_completed'],
        'ensemble_mae': training_results['ensemble_metrics']['mae'],
        'ensemble_r2': training_results['ensemble_metrics']['r2'],
        'prediction_shape': predictions['ensemble_prediction'].shape,
        'has_anomaly_detector': system.anomaly_detector is not None
    }


def main():
    """Run all P0 enhancement tests."""
    logger.info("🧪 Starting P0 Enhancements Validation Tests")
    logger.info("=" * 60)

    results = {}

    try:
        # Test individual components
        results['moe'] = test_moe_components()
        results['lora'] = test_lora_adapters()
        results['rare_events'] = test_rare_event_augmentation()
        results['anomaly_detection'] = test_anomaly_detection()

        # Test integrated system
        results['integrated_system'] = test_integrated_system()

        # Summary
        logger.info("=" * 60)
        logger.info("🎯 P0 Enhancements Test Summary:")
        logger.info("=" * 60)

        logger.info("🧠 MoE Components:")
        logger.info(f"  • Expert usage distribution: {results['moe']['expert_usage']}")
        logger.info(f"  • Total loss: {results['moe']['total_loss']:.4f}")

        logger.info("🔧 LoRA Adapters:")
        logger.info(f"  • Parameter efficiency: {results['lora']['efficiency']:.4f}")
        logger.info(f"  • LoRA parameters: {results['lora']['lora_params']}")

        logger.info("🎯 Rare Event Augmentation:")
        logger.info(f"  • Augmentation factor: {results['rare_events']['augmentation_factor']:.2f}")
        logger.info(f"  • Synthetic sequences: {results['rare_events']['synthetic_sequences']}")

        logger.info("🚨 Anomaly Detection:")
        logger.info(f"  • Training loss: {results['anomaly_detection']['training_loss']:.4f}")
        logger.info(f"  • Anomaly detection working: ✅")

        logger.info("🚀 Integrated System:")
        logger.info(f"  • Ensemble MAE: {results['integrated_system']['ensemble_mae']:.4f}")
        logger.info(f"  • Ensemble R²: {results['integrated_system']['ensemble_r2']:.4f}")
        logger.info(f"  • Has anomaly detector: {results['integrated_system']['has_anomaly_detector']}")

        logger.info("=" * 60)
        logger.info("🎉 All P0 Enhancement Tests Passed Successfully!")

        return True, results

    except Exception as e:
        logger.error(f"❌ P0 Enhancement Tests Failed: {e}")
        return False, results


if __name__ == "__main__":
    success, results = main()
    exit(0 if success else 1)
