"""
SSA优化器包 - 麻雀搜索算法优化器
用于GluFormer-MoE框架的超参数优化
"""

from .ssa import SSA, SSACallback
from .objective import ObjectiveFunction, GluFormerObjective, MultiTaskObjective, CulturalObjective, ImageNutritionObjective
from .utils import SSALogger, ConvergencePlotter, ParameterValidator

__version__ = "1.0.0"
__author__ = "Academic Research Team"

__all__ = [
    "SSA",
    "SSACallback",
    "ObjectiveFunction",
    "GluFormerObjective",
    "MultiTaskObjective",
    "CulturalObjective",
    "ImageNutritionObjective",
    "SSALogger",
    "ConvergencePlotter",
    "ParameterValidator"
]
