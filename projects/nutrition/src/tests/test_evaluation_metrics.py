#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage1 评估指标单元测试

针对 calculate_metrics 与 compare_with_baseline 的核心数值逻辑，
构造可手算的小规模用例，避免后续修改时悄悄引入错误。
"""

import numpy as np

from .evaluate_integrated_model import (
    calculate_metrics,
    compare_with_baseline,
)


def test_calculate_metrics_basic_regression_case() -> None:
    """
    一个易于手算的回归指标示例，验证 MAE/MSE/RMSE/R2 是否一致。
    """
    # 预测与标签：故意设置成简单分布，便于手算
    preds = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    targets = np.array([0.0, 1.0, 1.0], dtype=np.float32)

    metrics = calculate_metrics(preds, targets)

    # MAE = (0 + 0.5 + 0) / 3 = 1/6
    expected_mae = 1.0 / 6.0
    assert np.isclose(metrics["mae"], expected_mae, atol=1e-6)

    # MSE = (0^2 + 0.5^2 + 0^2) / 3 = 0.25 / 3
    expected_mse = 0.25 / 3.0
    assert np.isclose(metrics["mse"], expected_mse, atol=1e-6)

    # RMSE = sqrt(MSE)
    expected_rmse = np.sqrt(expected_mse)
    assert np.isclose(metrics["rmse"], expected_rmse, atol=1e-6)

    # R2 手动计算：1 - SS_res / (SS_tot + epsilon)
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    expected_r2 = 1.0 - ss_res / (ss_tot + 1e-6)
    assert np.isclose(metrics["r2"], expected_r2, atol=1e-6)


def test_compare_with_baseline_improvement_rate() -> None:
    """
    验证整体 MAE 改进量与改进率的计算是否与手算一致。
    """
    integrated_results = {
        "overall": {
            "taste": {"mae": 0.014},
            "health": {"mae": 0.015},
        },
        "high_health_dishes": {
            "taste": {"mae": 0.014},
            "health": {"mae": 0.015},
        },
    }

    baseline_results = {
        "overall": {
            "taste": {"mae": 0.2214},
            "health": {"mae": 0.1868},
        }
    }

    comparison = compare_with_baseline(integrated_results, baseline_results)
    overall_mae = comparison["overall_mae"]

    # 手算期望改进与改进率
    baseline_taste = baseline_results["overall"]["taste"]["mae"]
    integrated_taste = integrated_results["overall"]["taste"]["mae"]
    expected_improve = baseline_taste - integrated_taste
    expected_rate = expected_improve / baseline_taste * 100.0

    assert np.isclose(
        overall_mae["improvement_taste"],
        expected_improve,
        atol=1e-8,
    )
    assert np.isclose(
        overall_mae["improvement_rate_taste"],
        expected_rate,
        atol=1e-8,
    )
