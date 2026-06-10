"""
推荐系统参数优化器
应用启发式算法优化推荐系统的关键参数
支持多目标优化和自适应参数调整
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """优化配置"""
    algorithm: str = 'adaptive_weighted'  # 优化算法名称
    max_iterations: int = 50  # 最大迭代次数
    population_size: int = 20  # 种群大小（如适用）
    convergence_threshold: float = 1e-4  # 收敛阈值
    multi_objective: bool = False  # 是否多目标优化


class FlavorWeightOptimizer:
    """口味权重优化器"""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.optimization_history = []

    def optimize_flavor_weights(
        self,
        user_preferences: Dict[str, float],
        food_samples: List[Dict[str, Any]],
        target_scores: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """
        优化口味权重

        基于用户偏好和食物样本，优化6个口味因子的权重
        使用自适应加权优化算法

        Args:
            user_preferences: 用户口味偏好
            food_samples: 食物样本列表（包含口味特征）
            target_scores: 目标分数列表（可选，用于监督学习）

        Returns:
            Dict[str, float]: 优化后的口味权重
        """
        # 初始化权重（当前固定权重作为起点）
        initial_weights = {
            'spicy': 0.30,
            'sweet': 0.20,
            'sour': 0.15,
            'salty': 0.15,
            'umami': 0.12,
            'bitter': 0.08
        }

        # 如果使用自适应优化
        if self.config.algorithm == 'adaptive_weighted':
            optimized_weights = self._adaptive_weighted_optimization(
                initial_weights, user_preferences, food_samples, target_scores
            )
        else:
            # 默认返回初始权重
            optimized_weights = initial_weights

        # 归一化权重（确保总和为1）
        total_weight = sum(optimized_weights.values())
        if total_weight > 0:
            optimized_weights = {k: v / total_weight for k, v in optimized_weights.items()}

        return optimized_weights

    def _adaptive_weighted_optimization(
        self,
        initial_weights: Dict[str, float],
        user_preferences: Dict[str, float],
        food_samples: List[Dict[str, Any]],
        target_scores: Optional[List[float]]
    ) -> Dict[str, float]:
        """
        自适应加权优化

        基于用户偏好和食物样本，自适应调整权重
        参考田忌赛马优化算法的思想：通过比较不同权重组合的效果来优化
        参考螳螂虾优化算法的快速收敛特性：快速定位重要因子
        """
        weights = initial_weights.copy()
        flavor_types = list(weights.keys())

        # 计算每个口味因子的重要性得分
        importance_scores = {}
        for flavor_type in flavor_types:
            importance = self._calculate_flavor_importance(
                flavor_type, user_preferences, food_samples
            )
            importance_scores[flavor_type] = importance

        # 基于重要性得分调整权重（参考田忌赛马策略：在关键因子上有优势）
        total_importance = sum(importance_scores.values())
        if total_importance > 0:
            # 归一化重要性得分
            normalized_importance = {
                k: v / total_importance for k, v in importance_scores.items()
            }

            # 动态调整策略：高重要性因子获得更多权重
            for flavor_type in flavor_types:
                importance_weight = normalized_importance[flavor_type]
                initial_weight = initial_weights[flavor_type]

                # 自适应混合：重要性越高，权重调整幅度越大
                # 参考田忌赛马：在关键比赛上投入更多资源
                if importance_weight > 0.2:  # 高重要性因子
                    # 更激进地调整高重要性因子
                    weights[flavor_type] = 0.8 * importance_weight + 0.2 * initial_weight
                else:  # 低重要性因子
                    # 保守地调整低重要性因子
                    weights[flavor_type] = 0.6 * importance_weight + 0.4 * initial_weight

        return weights

    def _calculate_flavor_importance(
        self,
        flavor_type: str,
        user_preferences: Dict[str, float],
        food_samples: List[Dict[str, Any]]
    ) -> float:
        """
        计算口味因子的重要性

        参考螳螂虾优化算法的快速定位特性：快速识别关键因子
        """
        if not food_samples:
            return 0.5

        user_pref = user_preferences.get(flavor_type, 0.5)

        # 计算该口味因子在样本中的分布特征
        flavor_values = []
        for sample in food_samples:
            if 'flavor_profile' in sample:
                flavor_value = sample['flavor_profile'].get(flavor_type, 0.5)
                flavor_values.append(flavor_value)

        if not flavor_values:
            return 0.5

        # 计算与用户偏好的匹配度
        match_scores = [1.0 - abs(flavor_value - user_pref) for flavor_value in flavor_values]

        # 重要性 = 区分度 + 匹配度 + 用户偏好强度
        # 区分度：方差越大，区分度越高
        discrimination = np.std(flavor_values) if len(flavor_values) > 1 else 0.5

        # 匹配度：与用户偏好的平均匹配度
        match_quality = np.mean(match_scores) if match_scores else 0.5

        # 用户偏好强度：偏离中性值越远，重要性越高
        preference_strength = abs(user_pref - 0.5) * 2  # 归一化到[0,1]

        # 综合重要性：区分度40% + 匹配度30% + 偏好强度30%
        importance = 0.4 * discrimination + 0.3 * match_quality + 0.3 * preference_strength

        return min(1.0, max(0.0, importance))


class ConfidenceWeightOptimizer:
    """置信度权重优化器"""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.optimization_history = []

    def optimize_confidence_weights(
        self,
        user_profile_samples: List[Dict[str, Any]],
        actual_confidence_scores: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """
        优化置信度计算的权重

        基于用户档案样本和实际置信度，优化4个因子的权重

        Args:
            user_profile_samples: 用户档案样本列表
            actual_confidence_scores: 实际置信度分数（可选，用于监督学习）

        Returns:
            Dict[str, float]: 优化后的权重
        """
        initial_weights = {
            'cultural': 0.3,
            'nutritional': 0.3,
            'profile': 0.2,
            'data': 0.2
        }

        if self.config.algorithm == 'adaptive_weighted':
            optimized_weights = self._adaptive_confidence_optimization(
                initial_weights, user_profile_samples, actual_confidence_scores
            )
        else:
            optimized_weights = initial_weights

        # 归一化
        total_weight = sum(optimized_weights.values())
        if total_weight > 0:
            optimized_weights = {k: v / total_weight for k, v in optimized_weights.items()}

        return optimized_weights

    def _adaptive_confidence_optimization(
        self,
        initial_weights: Dict[str, float],
        user_profile_samples: List[Dict[str, Any]],
        actual_confidence_scores: Optional[List[float]]
    ) -> Dict[str, float]:
        """自适应置信度权重优化"""
        weights = initial_weights.copy()

        # 计算每个因子的可靠性得分
        reliability_scores = {}
        for factor_name in weights.keys():
            reliability = self._calculate_factor_reliability(
                factor_name, user_profile_samples
            )
            reliability_scores[factor_name] = reliability

        # 基于可靠性调整权重
        total_reliability = sum(reliability_scores.values())
        if total_reliability > 0:
            for factor_name in weights.keys():
                reliability_weight = reliability_scores[factor_name] / total_reliability
                weights[factor_name] = 0.7 * reliability_weight + 0.3 * initial_weights[factor_name]

        return weights

    def _calculate_factor_reliability(
        self,
        factor_name: str,
        user_profile_samples: List[Dict[str, Any]]
    ) -> float:
        """计算因子的可靠性"""
        if not user_profile_samples:
            return 0.5

        # 计算因子值的完整性和稳定性
        factor_values = []
        for sample in user_profile_samples:
            if factor_name in sample:
                factor_values.append(sample[factor_name])

        if not factor_values:
            return 0.0

        # 可靠性 = 完整率 + 稳定性
        completeness = len(factor_values) / len(user_profile_samples)
        stability = 1.0 - np.std(factor_values) if len(factor_values) > 1 else 0.5

        reliability = 0.6 * completeness + 0.4 * stability

        return min(1.0, max(0.0, reliability))


class MultiObjectiveRecommendationOptimizer:
    """多目标推荐优化器"""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.optimization_history = []

    def optimize_recommendation_parameters(
        self,
        objectives: Dict[str, Callable],
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        多目标优化推荐参数

        参考田忌赛马优化算法思想：通过比较不同参数组合的效果来优化
        平衡多个目标：文化适配度、营养平衡、用户满意度

        Args:
            objectives: 目标函数字典 {目标名: 目标函数}
            constraints: 约束条件

        Returns:
            Dict[str, float]: 优化后的参数
        """
        # 使用加权求和法进行多目标优化
        # 参考田忌赛马算法的"比较策略"思想
        if self.config.algorithm == 'adaptive_weighted':
            return self._tianji_style_optimization(objectives, constraints)
        else:
            # 默认返回均衡权重
            return {
                'cultural_weight': 0.4,
                'nutritional_weight': 0.3,
                'satisfaction_weight': 0.3
            }

    def _tianji_style_optimization(
        self,
        objectives: Dict[str, Callable],
        constraints: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        田忌赛马风格优化

        核心思想：通过比较不同参数组合的效果，选择最优组合
        类似田忌赛马的策略：合理分配资源，在关键目标上取得优势
        """
        # 初始化参数
        param_names = list(objectives.keys())
        num_params = len(param_names)

        # 初始化权重（均匀分布）
        initial_weights = {name: 1.0 / num_params for name in param_names}

        # 迭代优化
        current_weights = initial_weights.copy()
        best_fitness = float('-inf')
        best_weights = current_weights.copy()

        for iteration in range(self.config.max_iterations):
            # 评估当前权重
            current_fitness = self._evaluate_multi_objective(current_weights, objectives)

            # 更新最佳权重
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_weights = current_weights.copy()

            # 自适应调整权重（参考田忌赛马策略）
            current_weights = self._adaptive_weight_update(
                current_weights, objectives, iteration
            )

            # 检查收敛
            if iteration > 0 and abs(current_fitness - best_fitness) < self.config.convergence_threshold:
                break

        return best_weights

    def _evaluate_multi_objective(
        self,
        weights: Dict[str, float],
        objectives: Dict[str, Callable]
    ) -> float:
        """评估多目标函数"""
        total_score = 0.0

        for obj_name, obj_func in objectives.items():
            weight = weights.get(obj_name, 0.0)
            try:
                score = obj_func()
                total_score += weight * score
            except Exception as e:
                logger.warning(f"目标函数 {obj_name} 评估失败: {e}")
                continue

        return total_score

    def _adaptive_weight_update(
        self,
        current_weights: Dict[str, float],
        objectives: Dict[str, Callable],
        iteration: int
    ) -> Dict[str, float]:
        """自适应权重更新"""
        # 计算每个目标的历史表现
        performance_scores = {}
        for obj_name in current_weights.keys():
            try:
                score = objectives[obj_name]()
                performance_scores[obj_name] = score
            except:
                performance_scores[obj_name] = 0.5

        # 基于表现调整权重（表现好的目标权重增加）
        total_performance = sum(performance_scores.values())
        if total_performance > 0:
            new_weights = {}
            for obj_name in current_weights.keys():
                performance_weight = performance_scores[obj_name] / total_performance
                # 加权平均：60%性能 + 40%当前权重
                new_weights[obj_name] = 0.6 * performance_weight + 0.4 * current_weights[obj_name]

            # 归一化
            total_weight = sum(new_weights.values())
            if total_weight > 0:
                new_weights = {k: v / total_weight for k, v in new_weights.items()}

            return new_weights

        return current_weights


class RecommendationParameterOptimizer:
    """推荐系统参数优化器（统一接口）"""

    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.flavor_optimizer = FlavorWeightOptimizer(self.config)
        self.confidence_optimizer = ConfidenceWeightOptimizer(self.config)
        self.multi_objective_optimizer = MultiObjectiveRecommendationOptimizer(self.config)

    def optimize_all_parameters(
        self,
        user_data: Dict[str, Any],
        food_samples: List[Dict[str, Any]],
        user_profile_samples: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        优化所有推荐参数

        Args:
            user_data: 用户数据
            food_samples: 食物样本
            user_profile_samples: 用户档案样本

        Returns:
            Dict[str, Any]: 优化后的所有参数
        """
        optimized_params = {}

        # 1. 优化口味权重
        user_preferences = user_data.get('cultural_preferences', {})
        optimized_params['flavor_weights'] = self.flavor_optimizer.optimize_flavor_weights(
            user_preferences, food_samples
        )

        # 2. 优化置信度权重
        optimized_params['confidence_weights'] = self.confidence_optimizer.optimize_confidence_weights(
            user_profile_samples
        )

        return optimized_params
