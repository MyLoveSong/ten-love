#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage2 工具模块
"""

from stage2.utils.data_loader import Stage2DataLoader
from stage2.utils.metrics import RecommendationMetrics
from stage2.utils.reproducibility import (
    set_global_seed,
    get_environment_info,
    save_environment_info,
    log_hyperparameters,
    estimate_training_time,
    create_reproducibility_report,
    ExperimentTracker,
)

# 特征提取依赖 torchvision，在某些纯评估/训练环境中可能未安装。
# 为避免不必要的硬依赖，这里做懒加载与降级处理。
try:
    from stage2.utils.feature_extractor import MultimodalFeatureExtractor  # type: ignore
except Exception:  # noqa: BLE001
    MultimodalFeatureExtractor = None  # type: ignore[misc,assignment]

__all__ = [
    'Stage2DataLoader',
    'MultimodalFeatureExtractor',
    'RecommendationMetrics',
    'set_global_seed',
    'get_environment_info',
    'save_environment_info',
    'log_hyperparameters',
    'estimate_training_time',
    'create_reproducibility_report',
    'ExperimentTracker',
]
