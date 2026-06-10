#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简化版LoRA实现
用于增强型文化适配模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Optional, List

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LoRAConfig:
    """LoRA配置类"""

    def __init__(
        self,
        r: int = 8,
        alpha: float = 16,
        dropout: float = 0.1,
        use_rslora: bool = False,
        init_lora_weights: bool = True
    ):
        """
        初始化LoRA配置

        Args:
            r: LoRA秩
            alpha: LoRA缩放因子
            dropout: Dropout比率
            use_rslora: 是否使用Rank-Stabilized LoRA
            init_lora_weights: 是否初始化LoRA权重
        """
        self.r = r
        self.alpha = alpha
        self.dropout = dropout
        self.use_rslora = use_rslora
        self.init_lora_weights = init_lora_weights

        # 验证配置
        if self.r <= 0:
            raise ValueError(f"LoRA秩必须为正数，当前值: {self.r}")

        if self.alpha <= 0:
            raise ValueError(f"LoRA缩放因子必须为正数，当前值: {self.alpha}")


class LoRALinear(nn.Module):
    """LoRA线性层，替代nn.Linear"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: LoRAConfig,
        bias: bool = True
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.config = config

        # 创建基础线性层
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # 创建LoRA组件
        self.lora_dropout = nn.Dropout(p=config.dropout)
        self.lora_A = nn.Linear(in_features, config.r, bias=False)
        self.lora_B = nn.Linear(config.r, out_features, bias=False)

        # 缩放因子
        self.scaling = config.alpha / config.r

        # 初始化权重
        if config.init_lora_weights:
            self._init_weights()

        # RSLoRA组件
        self.use_rslora = config.use_rslora
        if self.use_rslora:
            self.lora_E = nn.Parameter(torch.ones(1, config.r))

    def _init_weights(self):
        """初始化LoRA权重"""
        # 初始化A为正态分布
        nn.init.normal_(self.lora_A.weight, mean=0.0, std=0.02)
        # 初始化B为零
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 基础线性变换
        base_output = self.linear(x)

        # LoRA路径
        lora_x = self.lora_dropout(x)
        lora_x = self.lora_A(lora_x)

        # 应用RSLoRA（如果启用）
        if self.use_rslora:
            lora_x = lora_x * self.lora_E

        lora_output = self.lora_B(lora_x)

        # 应用缩放并与基础输出相加
        return base_output + lora_output * self.scaling
