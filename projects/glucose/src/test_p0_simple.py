"""
Simplified P0 Enhancements Test
Tests core functionality without complex integrations.
"""

import torch
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_moe_simple():
    """Test MoE components in isolation."""
    logger.info("🧠 Testing MoE Components...")

    from core.moe_components import MoEGlucoseHead, MoELoss

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
    test_input = torch.randn(4, 64, device=device)
    predictions = moe_head(test_input)

    assert predictions.shape == (4, 6), f"Expected (4, 6), got {predictions.shape}"
    logger.info("✅ MoE Components test passed")
    return True


def test_rare_events_simple():
    """Test rare event detection in isolation."""
    logger.info("🎯 Testing Rare Event Detection...")

    from data_processing.rare_event_augment import RareEventDetector

    # Create test sequence with hypoglycemia
    glucose_sequence = [65, 60, 55, 50, 45, 50, 55, 60, 65, 70, 75, 80]

    detector = RareEventDetector()
    events = detector.detect_events(np.array(glucose_sequence))

    assert 'hypoglycemia' in events, "Should detect hypoglycemia"
    assert len(events['hypoglycemia']) > 0, "Should find hypoglycemia events"

    logger.info("✅ Rare Event Detection test passed")
    return True


def test_anomaly_simple():
    """Test anomaly detection autoencoder."""
    logger.info("🚨 Testing Anomaly Detection...")

    from monitoring.anomaly_autoencoder import LSTMAnomalyAutoencoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create autoencoder
    autoencoder = LSTMAnomalyAutoencoder(
        input_dim=1,
        hidden_dim=32,
        sequence_length=12,
        device=device
    )

    # Test forward pass
    test_input = torch.randn(4, 12, 1, device=device)
    reconstructed, latent = autoencoder(test_input)

    assert reconstructed.shape == test_input.shape, "Reconstruction shape should match input"
    assert latent.shape[0] == 4, "Latent batch size should match input"

    # Test reconstruction error
    error = autoencoder.compute_reconstruction_error(test_input, reconstructed)
    assert error.shape == (4,), "Error should be per sequence"

    logger.info("✅ Anomaly Detection test passed")
    return True


def test_gluformer_with_enhancements():
    """Test GluFormer with basic enhancements."""
    logger.info("🚀 Testing Enhanced GluFormer...")

    from enhanced_glucose_system import EnhancedGlucosePredictionSystem

    # Simple configuration
    config = {
        'models': {
            'lstm': {
                'input_dim': 1,
                'hidden_dim': 32,
                'output_dim': 6,
                'dropout': 0.1
            },
            'transformer': {
                'input_dim': 1,
                'hidden_dim': 32,
                'output_dim': 6,
                'dropout': 0.1
            },
            'gluformer': {
                'input_dim': 1,
                'hidden_dim': 32,
                'output_dim': 6,
                'dropout': 0.1
            },
            'wavelet_gluformer': {
                'input_dim': 1,
                'hidden_dim': 32,
                'output_dim': 6,
                'dropout': 0.1,
                'wavelet': 'db4',
                'wavelet_levels': 3
            }
        },
        'training': {
            'epochs': 3,
            'batch_size': 8,
            'learning_rate': 0.001,
            'patience': 2,
            'val_split': 0.2
        },
        'ensemble': {
            'strategy': 'weighted',
            'auto_weight_update': True
        },
        'augmentation': {
            'enabled': True,
            'factor': 1.2,
            'methods': ['noise']
        },
        'rare_event_augmentation': {
            'enabled': True,
            'factor': 1.2,
            'methods': ['noise_injection'],
            'event_types': ['hypoglycemia']
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
            'window_size': 50,
            'drift_threshold': 0.05
        },
        'output_dir': 'TRAIN/outputs/p0_simple_test',
        'save_models': False,
        'save_plots': False
    }

    # Initialize system
    system = EnhancedGlucosePredictionSystem(config)

    # Generate simple test data
    np.random.seed(42)
    torch.manual_seed(42)

    sequences = []
    targets = []

    # Generate 50 sequences
    for i in range(50):
        if i < 5:  # Some rare events
            seq = np.random.uniform(50, 69, 12)  # Hypoglycemia
            tgt = np.random.uniform(50, 69, 6)
        else:  # Normal sequences
            seq = np.random.uniform(80, 150, 12)
            tgt = np.random.uniform(80, 150, 6)

        sequences.append(seq)
        targets.append(tgt)

    sequences = np.array(sequences).reshape(-1, 12, 1)
    targets = np.array(targets)

    logger.info(f"Generated {len(sequences)} test sequences")

    # Train system
    try:
        results = system.train_complete_system(sequences, targets)
        assert results['training_completed'], "Training should complete"
        logger.info("✅ Enhanced GluFormer training completed")

        # Test prediction
        test_seq = sequences[:3]
        predictions = system.predict(test_seq)
        assert 'ensemble_prediction' in predictions, "Should have ensemble predictions"
        logger.info("✅ Enhanced GluFormer prediction working")

        return True

    except Exception as e:
        logger.error(f"Enhanced GluFormer test failed: {e}")
        return False


def main():
    """Run simplified P0 tests."""
    logger.info("🧪 Starting Simplified P0 Enhancement Tests")
    logger.info("=" * 50)

    tests = [
        ("MoE Components", test_moe_simple),
        ("Rare Event Detection", test_rare_events_simple),
        ("Anomaly Detection", test_anomaly_simple),
        ("Enhanced GluFormer", test_gluformer_with_enhancements)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            logger.info(f"Running {test_name}...")
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED: {e}")

    logger.info("=" * 50)
    logger.info(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All P0 Enhancement Tests Passed!")
        return True
    else:
        logger.error("❌ Some tests failed")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
