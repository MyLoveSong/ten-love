"""
训练策略优化器
实现学习率调度和训练技术
"""

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


class GradientClipping:
    """梯度裁剪"""

    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type

    def __call__(self, model: nn.Module) -> float:
        return torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm, norm_type=self.norm_type)


class TrainingOptimizer:
    """训练策略优化器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config: Dict[str, Any] = config or self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'optimizer': {'type': 'AdamW', 'lr': 0.001, 'weight_decay': 0.01},
            'scheduler': {'type': 'CosineAnnealingLR', 'T_max': 100, 'eta_min': 1e-6},
        }

    def create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """创建优化器"""
        opt_config = self.config['optimizer']
        opt_type = opt_config.get('type', 'AdamW')

        if opt_type == 'AdamW':
            optimizer = optim.AdamW(model.parameters(), lr=opt_config.get('lr', 0.001),
                                   weight_decay=opt_config.get('weight_decay', 0.01))
        elif opt_type == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=opt_config.get('lr', 0.001),
                                  weight_decay=opt_config.get('weight_decay', 0.01))
        elif opt_type == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=opt_config.get('lr', 0.001),
                                 momentum=0.9, weight_decay=opt_config.get('weight_decay', 0.01))
        else:
            optimizer = optim.Adam(model.parameters())

        logger.info(f"创建优化器: {opt_type}, lr={opt_config.get('lr', 0.001)}")
        return optimizer

    def create_scheduler(self, optimizer: optim.Optimizer, num_epochs: int) -> Optional[Any]:
        """创建学习率调度器"""
        sched_config = self.config.get('scheduler', {})
        sched_type = sched_config.get('type', 'CosineAnnealingLR')

        if sched_type == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs,
                                         eta_min=sched_config.get('eta_min', 1e-6))
        elif sched_type == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        elif sched_type == 'StepLR':
            scheduler = StepLR(optimizer, step_size=num_epochs // 3, gamma=0.1)
        else:
            scheduler = None

        if scheduler:
            logger.info(f"创建学习率调度器: {sched_type}")
        return scheduler
