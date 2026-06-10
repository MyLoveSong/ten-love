#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
阶段一：文化适配微调（LoRA）入口模块
对接 backend.models.cultural_adaptation 并提供训练封装
"""

from .cultural_finetune import CulturalStage1Trainer, run

__all__ = [
    'CulturalStage1Trainer',
    'run',
]
