#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
额外指标工具
- ECE（Expected Calibration Error）
- Brier 分数（针对[0,1]归一化回归视作概率回归）
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Dict


def compute_ece(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 15
) -> Tuple[float, Dict[str, np.ndarray]]:
    """
    计算期望校准误差（ECE）。
    将连续的[0,1]回归预测作为概率进行分箱评估。
    返回 ece 及可用于绘图的分箱统计。
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.clip(np.asarray(y_pred).reshape(-1), 0.0, 1.0)

    # 分箱边界
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_pred, bin_edges[1:-1], right=False)

    ece = 0.0
    bin_empirical = []
    bin_conf = []
    bin_counts = []

    for b in range(n_bins):
        mask = bin_ids == b
        count = np.sum(mask)
        if count == 0:
            bin_empirical.append(0.0)
            bin_conf.append(((bin_edges[b] + bin_edges[b + 1]) / 2.0))
            bin_counts.append(0)
            continue
        pred_confidence = np.mean(y_pred[mask])
        empirical = np.mean(y_true[mask])
        ece += (count / len(y_true)) * np.abs(empirical - pred_confidence)
        bin_empirical.append(empirical)
        bin_conf.append(pred_confidence)
        bin_counts.append(int(count))

    stats = {
        "bin_edges": bin_edges,
        "bin_empirical": np.asarray(bin_empirical),
        "bin_confidence": np.asarray(bin_conf),
        "bin_counts": np.asarray(bin_counts),
    }
    return float(ece), stats


def compute_brier(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """
    计算 Brier 分数。
    在连续目标[0,1]设定下，等价于 MSE，但保留语义命名以用于论文呈现。
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.clip(np.asarray(y_pred).reshape(-1), 0.0, 1.0)
    return float(np.mean((y_pred - y_true) ** 2))


def regression_metrics_bundle(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    便捷打包：返回包含 MAE/MSE/RMSE/ECE/Brier 的基础集合（不含R²，避免sklearn依赖）。
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mae = float(np.mean(np.abs(y_pred - y_true)))
    mse = float(np.mean((y_pred - y_true) ** 2))
    rmse = float(np.sqrt(mse))
    ece, _ = compute_ece(y_true, y_pred)
    brier = compute_brier(y_true, y_pred)
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "ece": ece,
        "brier": brier,
    }
