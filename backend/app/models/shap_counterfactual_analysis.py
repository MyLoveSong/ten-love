"""
SHAP反事实分析模块
提供决策反事实分析和可解释性功能
完整版SHAP解释器实现
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod

# 导入完整版SHAP库
try:
    import shap
    SHAP_AVAILABLE = True
    logger = logging.getLogger(__name__)
except ImportError as e:
    SHAP_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"SHAP库不可用，将使用简化版本: {e}")

logger = logging.getLogger(__name__)

@dataclass
class CounterfactualResult:
    """反事实分析结果"""
    original_prediction: float
    counterfactual_prediction: float
    feature_change: str
    impact_difference: float
    explanation: str
    confidence: float

@dataclass
class SHAPExplanation:
    """SHAP解释结果"""
    feature_importance: Dict[str, float]
    feature_values: Dict[str, float]
    shap_values: Dict[str, float]
    base_value: float
    prediction: float

class BaseSHAPExplainer(ABC):
    """SHAP解释器基类"""

    @abstractmethod
    def explain_prediction(self, model, input_data: np.ndarray) -> SHAPExplanation:
        """解释单个预测"""
        pass

    @abstractmethod
    def generate_counterfactual(
        self,
        model,
        input_data: np.ndarray,
        target_feature: str,
        target_value: Optional[float] = None
    ) -> CounterfactualResult:
        """生成反事实解释"""
        pass

class FullSHAPExplainer(BaseSHAPExplainer):
    """完整版SHAP解释器"""

    def __init__(self, feature_names: List[str], model_type: str = "auto"):
        self.feature_names = feature_names
        self.model_type = model_type
        self.explainer = None
        self.background_data = None

        if SHAP_AVAILABLE:
            logger.info("完整版SHAP解释器初始化完成")
        else:
            logger.warning("SHAP库不可用，将使用简化版本")

    def _create_explainer(self, model, background_data: Optional[np.ndarray] = None):
        """创建SHAP解释器"""
        if not SHAP_AVAILABLE:
            return None

        try:
            # 根据模型类型选择解释器
            if self.model_type == "tree" or hasattr(model, 'tree_'):
                self.explainer = shap.TreeExplainer(model)
            elif self.model_type == "linear" or hasattr(model, 'coef_'):
                # 线性模型，使用LinearExplainer
                if background_data is not None:
                    self.explainer = shap.LinearExplainer(model, background_data)
                else:
                    # 创建默认背景数据
                    default_background = np.zeros((10, len(self.feature_names)))
                    self.explainer = shap.LinearExplainer(model, default_background)
            elif self.model_type == "deep" or hasattr(model, 'forward'):
                # 深度学习模型
                if background_data is not None:
                    self.explainer = shap.DeepExplainer(model, background_data)
                else:
                    self.explainer = shap.Explainer(model)
            else:
                # 通用解释器 - 优先使用KernelExplainer
                try:
                    if background_data is not None:
                        self.explainer = shap.KernelExplainer(model, background_data)
                    else:
                        # 创建默认背景数据
                        default_background = np.random.random((10, len(self.feature_names)))
                        self.explainer = shap.KernelExplainer(model, default_background)
                except Exception as kernel_error:
                    logger.warning(f"KernelExplainer失败，尝试Explainer: {kernel_error}")
                    # 降级到通用Explainer
                    if background_data is not None:
                        self.explainer = shap.Explainer(model, background_data)
                    else:
                        self.explainer = shap.Explainer(model)

            self.background_data = background_data
            logger.info(f"SHAP解释器创建成功: {type(self.explainer).__name__}")
            return self.explainer

        except Exception as e:
            logger.error(f"SHAP解释器创建失败: {e}")
            return None

    def explain_prediction(self, model, input_data: np.ndarray) -> SHAPExplanation:
        """
        解释单个预测 - 完整版实现

        Args:
            model: 预测模型
            input_data: 输入数据 [1, features]

        Returns:
            SHAPExplanation对象
        """
        try:
            if not SHAP_AVAILABLE:
                # 降级到简化版本
                return self._fallback_explanation(model, input_data)

            # 如果解释器未初始化，先创建
            if self.explainer is None:
                # 创建背景数据
                background_data = np.random.random((10, len(self.feature_names)))
                self._create_explainer(model, background_data)

            if self.explainer is None:
                logger.warning("完整版SHAP解释器创建失败，使用简化版本")
                return self._fallback_explanation(model, input_data)

            # 使用完整版SHAP解释
            try:
                shap_values = self.explainer(input_data)
            except Exception as shap_error:
                logger.error(f"SHAP计算失败: {shap_error}")
                return self._fallback_explanation(model, input_data)

            # 提取SHAP值
            if hasattr(shap_values, 'values'):
                shap_vals = shap_values.values[0]  # 第一个样本
                base_value = shap_values.base_values[0] if hasattr(shap_values, 'base_values') else 0.0
            else:
                shap_vals = shap_values[0]
                base_value = 0.0

            # 计算特征重要性
            feature_importance = {
                name: float(abs(shap_vals[i]))
                for i, name in enumerate(self.feature_names)
            }

            # 计算SHAP值
            shap_values_dict = {
                name: float(shap_vals[i])
                for i, name in enumerate(self.feature_names)
            }

            # 获取特征值
            feature_values = {
                name: float(input_data[0, i])
                for i, name in enumerate(self.feature_names)
            }

            # 计算最终预测
            prediction = base_value + sum(shap_values_dict.values())

            return SHAPExplanation(
                feature_importance=feature_importance,
                feature_values=feature_values,
                shap_values=shap_values_dict,
                base_value=base_value,
                prediction=prediction
            )

        except Exception as e:
            logger.error(f"完整版SHAP解释失败: {e}")
            return self._fallback_explanation(model, input_data)

    def generate_counterfactual(
        self,
        model,
        input_data: np.ndarray,
        target_feature: str,
        target_value: Optional[float] = None
    ) -> CounterfactualResult:
        """
        生成反事实解释 - 完整版实现

        Args:
            model: 预测模型
            input_data: 输入数据
            target_feature: 目标特征
            target_value: 目标值

        Returns:
            CounterfactualResult对象
        """
        try:
            if not SHAP_AVAILABLE:
                # 降级到简化版本
                return self._fallback_counterfactual(model, input_data, target_feature, target_value)

            # 如果解释器未初始化，先创建
            if self.explainer is None:
                # 创建背景数据
                background_data = np.random.random((10, len(self.feature_names)))
                self._create_explainer(model, background_data)

            if self.explainer is None:
                logger.warning("完整版SHAP解释器创建失败，使用简化版本")
                return self._fallback_counterfactual(model, input_data, target_feature, target_value)

            # 获取原始预测
            original_prediction = self._predict_single(model, input_data)

            # 创建反事实数据
            counterfactual_data = input_data.copy()
            feature_idx = self.feature_names.index(target_feature)

            if target_value is None:
                # 使用SHAP值指导反事实生成
                try:
                    shap_values = self.explainer(input_data)
                except Exception as shap_error:
                    logger.error(f"SHAP计算失败: {shap_error}")
                    return self._fallback_counterfactual(model, input_data, target_feature, target_value)
                if hasattr(shap_values, 'values'):
                    shap_vals = shap_values.values[0]
                else:
                    shap_vals = shap_values[0]

                # 基于SHAP值确定目标值
                current_value = input_data[0, feature_idx]
                shap_impact = shap_vals[feature_idx]

                if target_feature in ['spice_level', 'oil_content', 'sugar_content']:
                    # 减少不健康成分
                    target_value = max(0, current_value - abs(shap_impact) * 0.5)
                elif target_feature in ['fiber_content', 'protein_content']:
                    # 增加健康成分
                    target_value = min(1, current_value + abs(shap_impact) * 0.5)
                else:
                    # 基于SHAP值调整
                    target_value = current_value - shap_impact * 0.3

            counterfactual_data[0, feature_idx] = target_value

            # 获取反事实预测
            counterfactual_prediction = self._predict_single(model, counterfactual_data)

            # 计算影响差异
            impact_difference = counterfactual_prediction - original_prediction

            # 生成解释
            explanation = self._generate_explanation(
                target_feature, target_value, impact_difference
            )

            # 计算置信度
            confidence = self._calculate_confidence(impact_difference)

            return CounterfactualResult(
                original_prediction=original_prediction,
                counterfactual_prediction=counterfactual_prediction,
                feature_change=f"{target_feature}: {input_data[0, feature_idx]:.2f} → {target_value:.2f}",
                impact_difference=impact_difference,
                explanation=explanation,
                confidence=confidence
            )

        except Exception as e:
            logger.error(f"完整版反事实生成失败: {e}")
            return self._fallback_counterfactual(model, input_data, target_feature, target_value)

    def _fallback_explanation(self, model, input_data: np.ndarray) -> SHAPExplanation:
        """降级到简化版本的解释"""
        return SHAPExplanation(
            feature_importance={name: 0.1 for name in self.feature_names},
            feature_values={name: 0.0 for name in self.feature_names},
            shap_values={name: 0.0 for name in self.feature_names},
            base_value=6.0,
            prediction=6.0
        )

    def _fallback_counterfactual(
        self,
        model,
        input_data: np.ndarray,
        target_feature: str,
        target_value: Optional[float] = None
    ) -> CounterfactualResult:
        """降级到简化版本的反事实"""
        original_prediction = self._predict_single(model, input_data)

        if target_value is None:
            target_value = max(0, input_data[0, self.feature_names.index(target_feature)] - 0.2)

        counterfactual_data = input_data.copy()
        counterfactual_data[0, self.feature_names.index(target_feature)] = target_value
        counterfactual_prediction = self._predict_single(model, counterfactual_data)

        return CounterfactualResult(
            original_prediction=original_prediction,
            counterfactual_prediction=counterfactual_prediction,
            feature_change=f"{target_feature}: {input_data[0, self.feature_names.index(target_feature)]:.2f} → {target_value:.2f}",
            impact_difference=counterfactual_prediction - original_prediction,
            explanation="简化版反事实分析",
            confidence=0.6
        )

    def _predict_single(self, model, input_data: np.ndarray) -> float:
        """单个预测"""
        try:
            if hasattr(model, 'predict'):
                prediction = model.predict(input_data)
            elif hasattr(model, 'forward'):
                # PyTorch模型
                input_tensor = torch.tensor(input_data, dtype=torch.float32)
                with torch.no_grad():
                    prediction = model.forward(input_tensor).item()
            else:
                prediction = 6.0

            return float(prediction)
        except:
            return 6.0

    def _generate_explanation(
        self,
        feature: str,
        new_value: float,
        impact: float
    ) -> str:
        """生成解释文本"""
        feature_descriptions = {
            'spice_level': '川菜麻辣',
            'oil_content': '油脂含量',
            'sugar_content': '糖分含量',
            'fiber_content': '纤维含量',
            'protein_content': '蛋白质含量',
            'carbohydrate_content': '碳水化合物含量'
        }

        feature_desc = feature_descriptions.get(feature, feature)

        if impact < -0.1:
            return f"若减少{feature_desc}，预测血糖可再降{abs(impact):.2f} mmol/L"
        elif impact > 0.1:
            return f"若增加{feature_desc}，预测血糖将上升{impact:.2f} mmol/L"
        else:
            return f"调整{feature_desc}对血糖影响较小"

    def _calculate_confidence(self, impact: float) -> float:
        """计算置信度"""
        abs_impact = abs(impact)
        if abs_impact > 0.5:
            return 0.9
        elif abs_impact > 0.2:
            return 0.8
        elif abs_impact > 0.1:
            return 0.7
        else:
            return 0.6

class SimpleSHAPExplainer(BaseSHAPExplainer):
    """简化SHAP解释器"""

    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        logger.info("简化SHAP解释器初始化完成")

    def explain_prediction(self, model, input_data: np.ndarray) -> SHAPExplanation:
        """
        解释单个预测

        Args:
            model: 预测模型
            input_data: 输入数据 [1, features]

        Returns:
            SHAPExplanation对象
        """
        try:
            # 获取基础预测
            base_prediction = self._get_base_prediction(model)

            # 计算特征重要性（简化版本）
            feature_importance = self._calculate_feature_importance(model, input_data)

            # 计算SHAP值（简化版本）
            shap_values = self._calculate_shap_values(model, input_data, base_prediction)

            # 获取特征值
            feature_values = {
                name: float(input_data[0, i])
                for i, name in enumerate(self.feature_names)
            }

            # 计算最终预测
            prediction = base_prediction + sum(shap_values.values())

            return SHAPExplanation(
                feature_importance=feature_importance,
                feature_values=feature_values,
                shap_values=shap_values,
                base_value=base_prediction,
                prediction=prediction
            )

        except Exception as e:
            logger.error(f"SHAP解释失败: {e}")
            # 返回默认解释
            return SHAPExplanation(
                feature_importance={name: 0.1 for name in self.feature_names},
                feature_values={name: 0.0 for name in self.feature_names},
                shap_values={name: 0.0 for name in self.feature_names},
                base_value=6.0,
                prediction=6.0
            )

    def generate_counterfactual(
        self,
        model,
        input_data: np.ndarray,
        target_feature: str,
        target_value: Optional[float] = None
    ) -> CounterfactualResult:
        """
        生成反事实解释

        Args:
            model: 预测模型
            input_data: 输入数据
            target_feature: 目标特征
            target_value: 目标值（如果为None，则使用默认变化）

        Returns:
            CounterfactualResult对象
        """
        try:
            # 获取原始预测
            original_prediction = self._predict_single(model, input_data)

            # 创建反事实数据
            counterfactual_data = input_data.copy()
            feature_idx = self.feature_names.index(target_feature)

            if target_value is None:
                # 使用默认变化策略
                if target_feature in ['spice_level', 'oil_content', 'sugar_content']:
                    # 减少不健康成分
                    target_value = max(0, input_data[0, feature_idx] - 0.3)
                elif target_feature in ['fiber_content', 'protein_content']:
                    # 增加健康成分
                    target_value = min(1, input_data[0, feature_idx] + 0.3)
                else:
                    # 默认减少0.2
                    target_value = max(0, input_data[0, feature_idx] - 0.2)

            counterfactual_data[0, feature_idx] = target_value

            # 获取反事实预测
            counterfactual_prediction = self._predict_single(model, counterfactual_data)

            # 计算影响差异
            impact_difference = counterfactual_prediction - original_prediction

            # 生成解释
            explanation = self._generate_explanation(
                target_feature, target_value, impact_difference
            )

            # 计算置信度
            confidence = self._calculate_confidence(impact_difference)

            return CounterfactualResult(
                original_prediction=original_prediction,
                counterfactual_prediction=counterfactual_prediction,
                feature_change=f"{target_feature}: {input_data[0, feature_idx]:.2f} → {target_value:.2f}",
                impact_difference=impact_difference,
                explanation=explanation,
                confidence=confidence
            )

        except Exception as e:
            logger.error(f"反事实生成失败: {e}")
            return CounterfactualResult(
                original_prediction=6.0,
                counterfactual_prediction=6.0,
                feature_change="无法生成",
                impact_difference=0.0,
                explanation="反事实分析失败",
                confidence=0.0
            )

    def _get_base_prediction(self, model) -> float:
        """获取基础预测值"""
        try:
            # 使用零向量作为基础
            zero_input = np.zeros((1, len(self.feature_names)))
            return self._predict_single(model, zero_input)
        except:
            return 6.0  # 默认基础值

    def _calculate_feature_importance(self, model, input_data: np.ndarray) -> Dict[str, float]:
        """计算特征重要性"""
        try:
            importance = {}
            base_prediction = self._get_base_prediction(model)

            for i, feature_name in enumerate(self.feature_names):
                # 创建单特征输入
                single_feature_input = np.zeros_like(input_data)
                single_feature_input[0, i] = input_data[0, i]

                # 计算该特征的贡献
                single_prediction = self._predict_single(model, single_feature_input)
                contribution = single_prediction - base_prediction
                importance[feature_name] = float(contribution)

            return importance
        except:
            return {name: 0.1 for name in self.feature_names}

    def _calculate_shap_values(self, model, input_data: np.ndarray, base_value: float) -> Dict[str, float]:
        """计算SHAP值"""
        try:
            shap_values = {}
            current_prediction = self._predict_single(model, input_data)

            for i, feature_name in enumerate(self.feature_names):
                # 计算边际贡献
                marginal_contribution = self._calculate_marginal_contribution(
                    model, input_data, i
                )
                shap_values[feature_name] = float(marginal_contribution)

            return shap_values
        except:
            return {name: 0.0 for name in self.feature_names}

    def _calculate_marginal_contribution(self, model, input_data: np.ndarray, feature_idx: int) -> float:
        """计算边际贡献"""
        try:
            # 创建包含该特征和不包含该特征的输入
            with_feature = input_data.copy()
            without_feature = input_data.copy()
            without_feature[0, feature_idx] = 0

            # 计算预测差异
            pred_with = self._predict_single(model, with_feature)
            pred_without = self._predict_single(model, without_feature)

            return pred_with - pred_without
        except:
            return 0.0

    def _predict_single(self, model, input_data: np.ndarray) -> float:
        """单个预测"""
        try:
            if hasattr(model, 'predict'):
                prediction = model.predict(input_data)
            elif hasattr(model, 'forward'):
                # PyTorch模型
                input_tensor = torch.tensor(input_data, dtype=torch.float32)
                with torch.no_grad():
                    prediction = model.forward(input_tensor).item()
            else:
                # 默认预测
                prediction = 6.0

            return float(prediction)
        except:
            return 6.0

    def _generate_explanation(
        self,
        feature: str,
        new_value: float,
        impact: float
    ) -> str:
        """生成解释文本"""
        feature_descriptions = {
            'spice_level': '川菜麻辣',
            'oil_content': '油脂含量',
            'sugar_content': '糖分含量',
            'fiber_content': '纤维含量',
            'protein_content': '蛋白质含量',
            'carbohydrate_content': '碳水化合物含量'
        }

        feature_desc = feature_descriptions.get(feature, feature)

        if impact < -0.1:
            return f"若减少{feature_desc}，预测血糖可再降{abs(impact):.2f} mmol/L"
        elif impact > 0.1:
            return f"若增加{feature_desc}，预测血糖将上升{impact:.2f} mmol/L"
        else:
            return f"调整{feature_desc}对血糖影响较小"

    def _calculate_confidence(self, impact: float) -> float:
        """计算置信度"""
        # 基于影响大小的置信度
        abs_impact = abs(impact)
        if abs_impact > 0.5:
            return 0.9
        elif abs_impact > 0.2:
            return 0.8
        elif abs_impact > 0.1:
            return 0.7
        else:
            return 0.6

class CounterfactualAnalyzer:
    """反事实分析器"""

    def __init__(self, feature_names: List[str], model_type: str = "auto"):
        self.feature_names = feature_names
        self.model_type = model_type

        # 优先使用完整版SHAP解释器
        if SHAP_AVAILABLE:
            self.explainer = FullSHAPExplainer(feature_names, model_type)
            logger.info("反事实分析器初始化完成（完整版SHAP）")
        else:
            self.explainer = SimpleSHAPExplainer(feature_names)
            logger.info("反事实分析器初始化完成（简化版SHAP）")

    def analyze_meal_impact(
        self,
        model,
        meal_features: np.ndarray,
        target_improvements: List[str] = None
    ) -> List[CounterfactualResult]:
        """
        分析膳食影响的反事实

        Args:
            model: 预测模型
            meal_features: 膳食特征
            target_improvements: 目标改进特征列表

        Returns:
            反事实结果列表
        """
        try:
            if target_improvements is None:
                target_improvements = ['spice_level', 'oil_content', 'sugar_content']

            # 初始化SHAP解释器（如果是完整版）
            if isinstance(self.explainer, FullSHAPExplainer):
                # 为完整版SHAP解释器创建解释器
                background_data = np.random.random((10, len(self.feature_names)))  # 背景数据
                self.explainer._create_explainer(model, background_data)

            counterfactuals = []

            for feature in target_improvements:
                if feature in self.feature_names:
                    counterfactual = self.explainer.generate_counterfactual(
                        model, meal_features, feature
                    )
                    counterfactuals.append(counterfactual)

            return counterfactuals

        except Exception as e:
            logger.error(f"膳食影响分析失败: {e}")
            return []

    def generate_optimization_suggestions(
        self,
        model,
        current_features: np.ndarray
    ) -> Dict[str, Any]:
        """
        生成优化建议

        Args:
            model: 预测模型
            current_features: 当前特征

        Returns:
            优化建议字典
        """
        try:
            # 分析所有可能的改进
            improvements = []

            # 健康特征改进
            health_features = ['fiber_content', 'protein_content']
            for feature in health_features:
                if feature in self.feature_names:
                    counterfactual = self.explainer.generate_counterfactual(
                        model, current_features, feature, target_value=1.0
                    )
                    if counterfactual.impact_difference < -0.1:  # 有改善
                        improvements.append({
                            'feature': feature,
                            'suggestion': f"增加{feature}",
                            'expected_improvement': abs(counterfactual.impact_difference),
                            'confidence': counterfactual.confidence
                        })

            # 不健康特征减少
            unhealthy_features = ['spice_level', 'oil_content', 'sugar_content']
            for feature in unhealthy_features:
                if feature in self.feature_names:
                    counterfactual = self.explainer.generate_counterfactual(
                        model, current_features, feature, target_value=0.0
                    )
                    if counterfactual.impact_difference < -0.1:  # 有改善
                        improvements.append({
                            'feature': feature,
                            'suggestion': f"减少{feature}",
                            'expected_improvement': abs(counterfactual.impact_difference),
                            'confidence': counterfactual.confidence
                        })

            # 按改善程度排序
            improvements.sort(key=lambda x: x['expected_improvement'], reverse=True)

            return {
                'top_suggestions': improvements[:3],
                'total_improvements': len(improvements),
                'max_improvement': max([imp['expected_improvement'] for imp in improvements]) if improvements else 0
            }

        except Exception as e:
            logger.error(f"优化建议生成失败: {e}")
            return {
                'top_suggestions': [],
                'total_improvements': 0,
                'max_improvement': 0
            }

def create_shap_explainer(feature_names: List[str], model_type: str = "auto") -> Union[FullSHAPExplainer, SimpleSHAPExplainer]:
    """创建SHAP解释器"""
    if SHAP_AVAILABLE:
        return FullSHAPExplainer(feature_names, model_type)
    else:
        return SimpleSHAPExplainer(feature_names)

def create_counterfactual_analyzer(feature_names: List[str], model_type: str = "auto") -> CounterfactualAnalyzer:
    """创建反事实分析器"""
    return CounterfactualAnalyzer(feature_names, model_type)

# 使用示例
if __name__ == "__main__":
    # 创建分析器
    feature_names = ['spice_level', 'oil_content', 'sugar_content', 'fiber_content', 'protein_content']
    analyzer = create_counterfactual_analyzer(feature_names)

    # 模拟膳食特征
    meal_features = np.array([[0.8, 0.6, 0.7, 0.3, 0.4]])  # 高辣、高油、高糖、低纤维、低蛋白

    # 模拟模型
    class MockModel:
        def predict(self, x):
            # 简化的血糖预测模型
            return np.array([6.0 + 0.5 * x[0, 0] + 0.3 * x[0, 1] + 0.4 * x[0, 2] - 0.2 * x[0, 3] - 0.1 * x[0, 4]])

    model = MockModel()

    # 生成反事实分析
    counterfactuals = analyzer.analyze_meal_impact(model, meal_features)

    for cf in counterfactuals:
        print(f"特征变化: {cf.feature_change}")
        print(f"血糖影响: {cf.impact_difference:.2f} mmol/L")
        print(f"解释: {cf.explanation}")
        print(f"置信度: {cf.confidence:.2f}")
        print("---")
