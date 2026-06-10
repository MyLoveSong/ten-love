#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
不确定性评估工具（MC Dropout）
"""

from __future__ import annotations

from typing import Callable, Dict
import numpy as np
import torch
import torch.nn as nn


def enable_dropout(module: nn.Module) -> None:
    """在评估模式下强制启用 Dropout 层。"""
    for m in module.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            m.train()


@torch.no_grad()
def mc_dropout_predict(
    model: nn.Module,
    forward_fn: Callable[[nn.Module], torch.Tensor],
    num_samples: int = 30,
    clamp01: bool = True
) -> Dict[str, np.ndarray]:
    """
    使用 MC Dropout 进行多次采样预测，返回均值、标准差与置信区间。

    参数：
    - forward_fn: 封装好的前向函数，内部完成数据准备与 model(...)，返回 [N] 或 [N,1] 张量。
    """
    model.eval()
    enable_dropout(model)

    preds = []
    for _ in range(num_samples):
        y = forward_fn(model)
        y = y.view(-1).detach().cpu().numpy()
        if clamp01:
            y = np.clip(y, 0.0, 1.0)
        preds.append(y)

    samples = np.stack(preds, axis=0)  # [T, N]
    mean = samples.mean(axis=0)
    std = samples.std(axis=0)

    # 近似95%置信区间（假设正态）
    ci_low = mean - 1.96 * std
    ci_high = mean + 1.96 * std
    if clamp01:
        ci_low = np.clip(ci_low, 0.0, 1.0)
        ci_high = np.clip(ci_high, 0.0, 1.0)

    return {
        "mean": mean,
        "std": std,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "samples": samples,
    }
