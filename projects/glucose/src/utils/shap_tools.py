#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SHAP 可解释性工具（通用回归接口）
提供对表格特征模型的 KernelExplainer/FastTreeExplainer 适配。
"""

from __future__ import annotations

from typing import Callable, Dict, Optional
import numpy as np


def compute_shap_values(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    background: np.ndarray,
    samples: np.ndarray,
    max_background: int = 200,
    link: str = "identity"
) -> Dict[str, np.ndarray]:
    """
    计算 SHAP 值（KernelExplainer）。
    - predict_fn: 输入 [N, D] numpy，输出 [N] 或 [N,1] numpy
    - background: 背景样本（用于近似期望值）
    - samples: 需要解释的样本
    """
    import shap  # 延迟导入，避免无依赖环境报错

    bg = background
    if bg.shape[0] > max_background:
        idx = np.random.RandomState(42).choice(bg.shape[0], size=max_background, replace=False)
        bg = bg[idx]

    explainer = shap.KernelExplainer(lambda X: _ensure_1d(predict_fn(X)), bg, link=link)
    shap_values = explainer.shap_values(samples, l1_reg="num_features(10)")
    expected_value = explainer.expected_value

    # shap_values 可能为 list（多输出），统一为 ndarray
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
        if isinstance(expected_value, (list, tuple, np.ndarray)):
            expected_value = expected_value[0]

    return {
        "shap_values": np.asarray(shap_values),  # [N, D]
        "expected_value": np.asarray(expected_value).reshape(()),
    }


def _ensure_1d(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == 1:
        return y[:, 0]
    return y.reshape(-1)
