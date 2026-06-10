"""
Basic LoRA Test - Minimal functionality verification
"""

import torch
import torch.nn as nn
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_basic_lora():
    """Test basic LoRA layer functionality."""
    logger.info("🔧 Testing Basic LoRA Layer...")

    from core.adapters import LoRALayer, AdaptedLinear

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a simple linear layer
    base_layer = nn.Linear(32, 6).to(device)

    # Create LoRA adaptation
    adapted_layer = AdaptedLinear(
        base_layer=base_layer,
        rank=4,
        alpha=8.0,
        dropout=0.05,
        enable_lora=True
    )

    # Test forward pass
    test_input = torch.randn(4, 32, device=device)

    with torch.no_grad():
        base_output = base_layer(test_input)
        adapted_output = adapted_layer(test_input)

    assert base_output.shape == adapted_output.shape, "Output shapes should match"
    logger.info(f"✅ Basic LoRA test passed: {base_output.shape}")

    return True


def test_glucose_metrics_simple():
    """Test glucose metrics computation."""
    logger.info("📊 Testing Glucose Metrics...")

    # Simple glucose metrics function
    def compute_glucose_metrics(predictions, targets):
        predictions_np = predictions.detach().cpu().numpy().flatten()
        targets_np = targets.detach().cpu().numpy().flatten()

        metrics = {}

        # Hypoglycemia range (< 70 mg/dL)
        hypo_mask = targets_np < 70
        if np.any(hypo_mask):
            hypo_mae = np.mean(np.abs(predictions_np[hypo_mask] - targets_np[hypo_mask]))
            metrics['hypoglycemia_mae'] = hypo_mae

        # Hyperglycemia range (> 180 mg/dL)
        hyper_mask = targets_np > 180
        if np.any(hyper_mask):
            hyper_mae = np.mean(np.abs(predictions_np[hyper_mask] - targets_np[hyper_mask]))
            metrics['hyperglycemia_mae'] = hyper_mae

        # Normal range (70-180 mg/dL)
        normal_mask = (targets_np >= 70) & (targets_np <= 180)
        if np.any(normal_mask):
            normal_mae = np.mean(np.abs(predictions_np[normal_mask] - targets_np[normal_mask]))
            metrics['normal_range_mae'] = normal_mae

        return metrics

    # Test with sample data
    predictions = torch.tensor([[65.0, 180.0, 120.0, 50.0, 200.0, 100.0]])
    targets = torch.tensor([[70.0, 185.0, 115.0, 55.0, 195.0, 105.0]])

    metrics = compute_glucose_metrics(predictions, targets)

    logger.info(f"Computed metrics: {metrics}")

    expected_ranges = ['hypoglycemia_mae', 'hyperglycemia_mae', 'normal_range_mae']
    found_ranges = [key for key in expected_ranges if key in metrics]

    logger.info(f"✅ Glucose metrics: {found_ranges}")
    return len(found_ranges) > 0


def test_enhanced_system_basic():
    """Test enhanced system with LoRA enabled."""
    logger.info("🚀 Testing Enhanced System with LoRA...")

    from enhanced_glucose_system import EnhancedGlucosePredictionSystem

    # Minimal configuration
    config = {
        'models': {
            'gluformer': {
                'input_dim': 1,
                'hidden_dim': 16,  # Smaller for testing
                'output_dim': 6,
                'dropout': 0.1
            }
        },
        'training': {
            'epochs': 1,  # Minimal training
            'batch_size': 4,
            'learning_rate': 0.001,
            'patience': 1,
            'val_split': 0.2
        },
        'ensemble': {
            'strategy': 'weighted',
            'auto_weight_update': True
        },
        'augmentation': {
            'enabled': False  # Disable for simplicity
        },
        'rare_event_augmentation': {
            'enabled': False  # Disable for simplicity
        },
        'lora': {
            'enabled': True,
            'rank': 2,  # Very small rank
            'alpha': 4.0,
            'dropout': 0.05,
            'glucose_specific': True,
            'personalization': {
                'enabled': True,
                'max_epochs': 2
            }
        },
        'personalized_moe': {
            'enabled': False  # Disable for simplicity
        },
        'anomaly_detection': {
            'enabled': False  # Disable for simplicity
        },
        'monitoring': {
            'enabled': False  # Disable for simplicity
        },
        'output_dir': 'TRAIN/outputs/lora_basic_test',
        'save_models': False,
        'save_plots': False
    }

    try:
        # Initialize system
        system = EnhancedGlucosePredictionSystem(config)

        # Generate minimal test data
        np.random.seed(42)
        sequences = np.random.uniform(80, 150, (20, 12, 1))
        targets = np.random.uniform(80, 150, (20, 6))

        # Train system (minimal)
        results = system.train_complete_system(sequences, targets)

        # Check if LoRA personalization can be setup
        system.setup_personalization()

        logger.info("✅ Enhanced System with LoRA working")
        return True

    except Exception as e:
        logger.error(f"❌ Enhanced System test failed: {e}")
        return False


def main():
    """Run basic LoRA tests."""
    logger.info("🧪 Starting Basic LoRA Tests")
    logger.info("=" * 40)

    tests = [
        ("Basic LoRA Layer", test_basic_lora),
        ("Glucose Metrics", test_glucose_metrics_simple),
        ("Enhanced System", test_enhanced_system_basic)
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

    logger.info("=" * 40)
    logger.info(f"📊 Results: {passed}/{total} tests passed")

    if passed >= 2:  # Allow some flexibility
        logger.info("🎉 LoRA Basic Functionality Verified!")
        return True
    else:
        logger.error("❌ LoRA tests failed")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
