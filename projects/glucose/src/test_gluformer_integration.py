"""
Test script for GluFormer integration.
Validates the new architecture components work correctly.
"""

import torch
import numpy as np
import logging
from pathlib import Path

from enhanced_glucose_system import EnhancedGlucosePredictionSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_gluformer_models():
    """Test GluFormer model creation and basic functionality."""
    logger.info("Testing GluFormer model integration...")

    # Create test configuration
    config = {
        'models': {
            'gluformer': {
                'input_dim': 1,
                'hidden_dim': 32,  # Smaller for testing
                'output_dim': 6,
                'dropout': 0.1
            },
            'wavelet_gluformer': {
                'input_dim': 1,
                'hidden_dim': 32,  # Smaller for testing
                'output_dim': 6,
                'dropout': 0.1,
                'wavelet': 'db4',
                'wavelet_levels': 2  # Fewer levels for testing
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
        'monitoring': {
            'enabled': True,
            'window_size': 100,
            'drift_threshold': 0.05
        },
        'output_dir': 'TRAIN/outputs/gluformer_test',
        'save_models': True,
        'save_plots': False
    }

    # Initialize system
    system = EnhancedGlucosePredictionSystem(config)

    # Generate synthetic test data
    np.random.seed(42)
    torch.manual_seed(42)

    # Create synthetic glucose time series
    n_samples = 200
    seq_length = 12

    # Generate realistic glucose patterns
    base_glucose = 120 + 20 * np.sin(np.linspace(0, 4*np.pi, n_samples))
    noise = np.random.normal(0, 5, n_samples)
    glucose_data = base_glucose + noise
    glucose_data = np.clip(glucose_data, 70, 200)  # Realistic glucose range

    logger.info(f"Generated synthetic data: {n_samples} samples")

    # Create sequences and targets for multi-step prediction
    sequences = []
    targets = []

    for i in range(len(glucose_data) - seq_length - 6):
        seq = glucose_data[i:i + seq_length]
        target = glucose_data[i + seq_length:i + seq_length + 6]  # 6-step ahead prediction
        sequences.append(seq)
        targets.append(target)

    sequences = np.array(sequences).reshape(-1, seq_length, 1)
    targets = np.array(targets)

    logger.info(f"Created sequences: {sequences.shape}, targets: {targets.shape}")

    # Prepare data
    data_dict = system.prepare_data(sequences, targets)
    logger.info("Data preparation successful")

    # Create models (only GluFormer variants)
    system.models = {}  # Clear default models

    # Test GluFormer
    from core.base_models import GluFormerPredictor, WaveletGluFormerPredictor

    logger.info("Creating GluFormer model...")
    gluformer_config = config['models']['gluformer']
    gluformer = GluFormerPredictor(
        input_dim=gluformer_config['input_dim'],
        hidden_dim=gluformer_config['hidden_dim'],
        output_dim=gluformer_config['output_dim'],
        dropout=gluformer_config['dropout'],
        device=system.device
    )

    # Test forward pass
    test_input = data_dict['train_sequences'][:4].to(system.device)  # Small batch, move to device
    logger.info(f"Testing GluFormer forward pass with input shape: {test_input.shape}")

    with torch.no_grad():
        gluformer_output = gluformer(test_input)
        logger.info(f"GluFormer output shape: {gluformer_output.shape}")
        assert gluformer_output.shape == (4, 6), f"Expected (4, 6), got {gluformer_output.shape}"

    logger.info("✅ GluFormer forward pass successful")

    # Test Wavelet GluFormer
    logger.info("Creating Wavelet GluFormer model...")
    try:
        wavelet_config = config['models']['wavelet_gluformer']
        wavelet_gluformer = WaveletGluFormerPredictor(
            input_dim=wavelet_config['input_dim'],
            hidden_dim=wavelet_config['hidden_dim'],
            output_dim=wavelet_config['output_dim'],
            dropout=wavelet_config['dropout'],
            wavelet=wavelet_config['wavelet'],
            wavelet_levels=wavelet_config['wavelet_levels'],
            device=system.device
        )

        # Test forward pass (test_input is already on device)
        logger.info(f"Testing Wavelet GluFormer forward pass with input shape: {test_input.shape}")

        with torch.no_grad():
            wavelet_output = wavelet_gluformer(test_input)
            logger.info(f"Wavelet GluFormer output shape: {wavelet_output.shape}")
            assert wavelet_output.shape == (4, 6), f"Expected (4, 6), got {wavelet_output.shape}"

        logger.info("✅ Wavelet GluFormer forward pass successful")

    except ImportError as e:
        logger.warning(f"⚠️ Wavelet GluFormer test skipped (missing pywt): {e}")
    except Exception as e:
        logger.error(f"❌ Wavelet GluFormer test failed: {e}")
        raise

    # Test training (quick)
    logger.info("Testing GluFormer training...")
    system.models['gluformer'] = gluformer

    try:
        training_results = system.train_models(data_dict)
        logger.info("✅ GluFormer training successful")
        logger.info(f"Training results keys: {list(training_results.keys())}")

        # Test prediction
        test_predictions = gluformer.predict(test_input)
        logger.info(f"✅ GluFormer prediction successful, shape: {test_predictions.shape}")

    except Exception as e:
        logger.error(f"❌ GluFormer training failed: {e}")
        raise

    logger.info("🎉 All GluFormer integration tests passed!")

    return {
        'gluformer_output_shape': gluformer_output.shape,
        'training_epochs': training_results.get('gluformer', {}).get('total_epochs', 0),
        'final_val_loss': training_results.get('gluformer', {}).get('best_val_loss', 0.0)
    }


def main():
    """Run all tests."""
    try:
        results = test_gluformer_models()

        logger.info("=" * 50)
        logger.info("🎯 GluFormer Integration Test Summary:")
        logger.info(f"  • GluFormer output shape: {results['gluformer_output_shape']}")
        logger.info(f"  • Training epochs: {results['training_epochs']}")
        logger.info(f"  • Final validation loss: {results['final_val_loss']:.4f}")
        logger.info("=" * 50)

        return True

    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
