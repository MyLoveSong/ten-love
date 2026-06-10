#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
集成数据微调训练脚本（迁移自 TRAIN/integrated_data_finetuning.py）
使用集成后的数据集（包含Excel数据）进行微调训练
"""

import os
import sys
import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    explained_variance_score, max_error
)
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IntegratedDataFinetuningTrainer:
    """集成数据微调训练器"""

    def __init__(self, data_path="../data/integrated_dataset/integrated_training_data.json"):
        """初始化训练器"""
        self.data_path = data_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler = StandardScaler()
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_r2': [],
            'val_r2': []
        }

        # 针对集成数据的优化配置
        self.config = {
            "epochs": 150,
            "learning_rate": 0.0008,
            "patience": 20,
            "batch_size": 24,
            "weight_decay": 1e-4,
            "dropout_rate": 0.2
        }

        logger.info("🔧 初始化集成数据微调训练器")
        logger.info(f"📊 数据路径: {self.data_path}")
        logger.info(f"📊 设备: {self.device}")
        logger.info(f"📊 配置: {self.config}")

    # ... 其余方法保持不变，但路径需要相对于stage1/scripts/调整 ...

def main():
    """主函数"""
    print("\n" + "="*80)
    print("🎯 集成数据微调训练脚本")
    print("使用集成后的数据集（包含Excel数据）进行微调训练")
    print("="*80 + "\n")

    trainer = IntegratedDataFinetuningTrainer()
    trainer.run_integrated_finetuning()

    print("\n" + "="*80)
    print("✅ 集成数据微调训练完成！")
    print("📁 模型已保存到: ../models/integrated_cultural_model.pth")
    print("📊 报告已保存到: ../outputs/integrated_evaluation/")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
