

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预测器抽象基类
定义所有预测器必须实现的接口规范
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List
import logging
from pathlib import Path
import torch
import numpy as np

logger = logging.getLogger(__name__)

class BasePredictor(ABC):
    """预测器抽象基类"""

    def __init__(self, model_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        初始化预测器

        Args:
            model_path: 模型文件路径
            config: 配置参数字典
        """
        self.model_path = model_path
        self.config = config or {}
        self.model = None
        self.is_loaded = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        logger.info(f"初始化预测器: {self.__class__.__name__}")
        logger.info(f"使用设备: {self.device}")

    @abstractmethod
    def load_model(self) -> bool:
        """
        加载模型

        Returns:
            bool: 加载是否成功
        """
        pass

    @abstractmethod
    def predict(self, data: Union[np.ndarray, torch.Tensor, Dict[str, Any]]) -> Dict[str, Any]:
        """
        执行预测

        Args:
            data: 输入数据

        Returns:
            Dict[str, Any]: 预测结果
        """
        pass

    @abstractmethod
    def preprocess(self, data: Any) -> Union[np.ndarray, torch.Tensor]:
        """
        数据预处理

        Args:
            data: 原始数据

        Returns:
            预处理后的数据
        """
        pass

    @abstractmethod
    def postprocess(self, predictions: Union[np.ndarray, torch.Tensor]) -> Dict[str, Any]:
        """
        结果后处理

        Args:
            predictions: 模型原始输出

        Returns:
            Dict[str, Any]: 处理后的结果
        """
        pass

    def validate_input(self, data: Any) -> bool:
        """
        验证输入数据

        Args:
            data: 输入数据

        Returns:
            bool: 数据是否有效
        """
        if data is None:
            logger.error("输入数据为空")
            return False
        return True

    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息

        Returns:
            Dict[str, Any]: 模型信息
        """
        return {
            "predictor_type": self.__class__.__name__,
            "model_path": self.model_path,
            "is_loaded": self.is_loaded,
            "device": str(self.device),
            "config": self.config
        }

    def save_model(self, save_path: str) -> bool:
        """
        保存模型

        Args:
            save_path: 保存路径

        Returns:
            bool: 保存是否成功
        """
        if self.model is None:
            logger.error("模型未加载，无法保存")
            return False

        try:
            torch.save(self.model.state_dict(), save_path)
            logger.info(f"模型已保存到: {save_path}")
            return True
        except Exception as e:
            logger.error(f"保存模型失败: {e}")
            return False

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(loaded={self.is_loaded}, device={self.device})"

class PredictorRegistry:
    """预测器注册表"""

    _predictors = {}

    @classmethod
    def register(cls, name: str, predictor_class: type):
        """
        注册预测器

        Args:
            name: 预测器名称
            predictor_class: 预测器类
        """
        if not issubclass(predictor_class, BasePredictor):
            raise ValueError(f"{predictor_class} 必须继承自 BasePredictor")

        cls._predictors[name] = predictor_class
        logger.info(f"注册预测器: {name} -> {predictor_class}")

    @classmethod
    def get_predictor(cls, name: str, **kwargs) -> Optional[BasePredictor]:
        """
        获取预测器实例

        Args:
            name: 预测器名称
            **kwargs: 初始化参数

        Returns:
            BasePredictor: 预测器实例
        """
        if name not in cls._predictors:
            logger.error(f"未找到预测器: {name}")
            return None

        predictor_class = cls._predictors[name]
        return predictor_class(**kwargs)

    @classmethod
    def list_predictors(cls) -> List[str]:
        """
        列出所有已注册的预测器

        Returns:
            List[str]: 预测器名称列表
        """
        return list(cls._predictors.keys())

    @classmethod
    def clear(cls):
        """清空注册表"""
        cls._predictors.clear()
        logger.info("预测器注册表已清空")

# 装饰器：自动注册预测器
def register_predictor(name: str):
    """
    预测器注册装饰器

    Args:
        name: 预测器名称
    """
    def decorator(cls):
        PredictorRegistry.register(name, cls)
        return cls
    return decorator

__all__ = ["'logger'", "'BasePredictor'", "'PredictorRegistry'", "'register_predictor'"]
