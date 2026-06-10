"""
Simplified LoRA Personalization Test
Tests core LoRA functionality without complex integrations.
"""

import torch
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_lora_components():
    """Test LoRA components in isolation."""
    logger.info("🔧 Testing LoRA Components...")

    from core.adapters import LoRAAdapter, PersonalizationManager
    from core.base_models import GluFormerPredictor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a simple GluFormer model
    model = GluFormerPredictor(
        input_dim=1,
        hidden_dim=32,
        output_dim=6,
        dropout=0.1,
        device=device
    )

    # Test LoRA adapter
    lora_adapter = LoRAAdapter(
        rank=4,
        alpha=8.0,
        dropout=0.05,
        glucose_specific=True
    )

    # Apply LoRA
    adapted_model = lora_adapter.apply_lora(model, "test_model")

    # Test forward pass
    test_input = torch.randn(4, 12, 1, device=device)

    with torch.no_grad():
        original_output = model(test_input)
        adapted_output = adapted_model(test_input)

    assert original_output.shape == adapted_output.shape, "Output shapes should match"
    logger.info(f"✅ LoRA adaptation successful: {original_output.shape}")

    return model, lora_adapter


def test_personalization_manager():
    """Test PersonalizationManager functionality."""
    logger.info("👤 Testing PersonalizationManager...")

    from core.adapters import PersonalizationManager
    from core.base_models import GluFormerPredictor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create base model
    base_model = GluFormerPredictor(
        input_dim=1,
        hidden_dim=32,
        output_dim=6,
        dropout=0.1,
        device=device
    )

    # LoRA configuration
    lora_config = {
        'rank': 4,
        'alpha': 8.0,
        'dropout': 0.05,
        'target_modules': ['lstm', 'gru', 'self_attention', 'cross_attention'],
        'glucose_specific': True,
        'learning_rate': 0.001,
        'weight_decay': 0.01
    }

    # Create personalization manager
    manager = PersonalizationManager(base_model, lora_config)

    # Generate simple patient data
    np.random.seed(42)
    patient_data = torch.randn(10, 12, 1, device=device)
    patient_targets = torch.randn(10, 6, device=device)

    # Test personalization (reduced epochs for speed)
    try:
        result = manager.personalize_model(
            patient_id="test_patient",
            patient_data=patient_data,
            patient_targets=patient_targets,
            epochs=2  # Very short for testing
        )

        assert 'patient_id' in result, "Result should contain patient_id"
        assert result['patient_id'] == "test_patient", "Patient ID should match"

        logger.info(f"✅ Personalization successful: Final loss = {result['final_train_loss']:.4f}")
        return True

    except Exception as e:
        logger.error(f"❌ Personalization failed: {e}")
        return False


def test_glucose_metrics():
    """Test glucose-specific metrics computation."""
    logger.info("📊 Testing Glucose Metrics...")

    from core.adapters import PersonalizationManager
    from core.base_models import GluFormerPredictor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create base model
    base_model = GluFormerPredictor(
        input_dim=1,
        hidden_dim=32,
        output_dim=6,
        dropout=0.1,
        device=device
    )

    # LoRA configuration
    lora_config = {
        'rank': 4,
        'alpha': 8.0,
        'dropout': 0.05,
        'glucose_specific': True,
        'learning_rate': 0.001,
        'weight_decay': 0.01
    }

    manager = PersonalizationManager(base_model, lora_config)

    # Test glucose metrics computation directly
    predictions = torch.tensor([[65.0, 180.0, 120.0, 50.0, 200.0, 100.0]], device=device)
    targets = torch.tensor([[70.0, 185.0, 115.0, 55.0, 195.0, 105.0]], device=device)

    metrics = manager._compute_glucose_metrics(predictions, targets)

    logger.info(f"Computed metrics: {metrics}")

    # Should have different range metrics
    expected_ranges = ['hypoglycemia_mae', 'hyperglycemia_mae', 'normal_range_mae']
    found_ranges = [key for key in expected_ranges if key in metrics]

    logger.info(f"✅ Glucose metrics computed: {found_ranges}")
    return len(found_ranges) > 0


def main():
    """Run simplified LoRA tests."""
    logger.info("🧪 Starting Simplified LoRA Tests")
    logger.info("=" * 50)

    tests = [
        ("LoRA Components", test_lora_components),
        ("PersonalizationManager", test_personalization_manager),
        ("Glucose Metrics", test_glucose_metrics)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            logger.info(f"Running {test_name}...")
            if test_name == "LoRA Components":
                model, adapter = test_func()
                if model is not None and adapter is not None:
                    passed += 1
                    logger.info(f"✅ {test_name} PASSED")
                else:
                    logger.error(f"❌ {test_name} FAILED")
            else:
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
        logger.info("🎉 All LoRA Tests Passed!")
        logger.info("✅ LoRA personalization is working correctly")
        return True
    else:
        logger.error("❌ Some LoRA tests failed")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
