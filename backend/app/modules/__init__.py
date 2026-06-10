

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块包初始化文件
自动导入和注册所有预测器模块
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# 自动导入所有预测器模块
try:
    # 导入基础预测器
    from .base_predictor import BasePredictor, PredictorRegistry, register_predictor

    # 导入血糖预测器
    from .glucose_prediction.predictor import GlucosePredictor

    # 导入图像识别器
    from .image_recognition.predictor import ImagePredictor

    logger.info("所有预测器模块导入成功")

except ImportError as e:
    logger.warning(f"部分模块导入失败: {e}")

# 导出主要类
__all__ = [
    'BasePredictor',
    'PredictorRegistry',
    'register_predictor',
    'GlucosePredictor',
    'ImagePredictor'
]
