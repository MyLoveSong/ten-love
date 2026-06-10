"""
麻雀搜索算法（Sparrow Search Algorithm, SSA）核心实现
用于GluFormer-MoE框架的超参数优化
支持单目标和多目标优化
"""

import numpy as np
import random
import torch
import logging
import math
from typing import Callable, Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import time
import json
from pathlib import Path
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class SSAConfig:
    """SSA算法配置"""
    dim: int = 7  # 搜索维度
    pop_size: int = 20  # 种群大小
    max_iter: int = 30  # 最大迭代次数
    lb: List[float] = None  # 下边界
    ub: List[float] = None  # 上边界
    discovery_rate: float = 0.1  # 发现者比例
    safety_threshold: float = 0.8  # 安全阈值
    levy_flight: bool = True  # 是否使用Levy飞行
    adaptive_chaos: bool = True  # 是否使用自适应混沌因子
    early_stopping: bool = True  # 是否早停
    patience: int = 10  # 早停耐心值
    min_delta: float = 1e-6  # 早停最小变化量

    # 多目标优化配置
    multi_objective: bool = False  # 是否多目标优化
    objectives: List[str] = None  # 目标函数列表
    pareto_size: int = 50  # Pareto前沿大小

    def __post_init__(self):
        if self.lb is None:
            self.lb = [0.0] * self.dim
        if self.ub is None:
            self.ub = [1.0] * self.dim
        if self.objectives is None:
            self.objectives = ['prediction_accuracy', 'model_complexity', 'computational_efficiency']

class SSACallback:
    """SSA优化回调函数"""

    def __init__(self):
        self.best_fitness_history = []
        self.iteration_times = []
        self.convergence_data = []

    def on_iteration_start(self, iteration: int, ssa_instance: 'SSA'):
        """迭代开始回调"""
        pass

    def on_iteration_end(self, iteration: int, ssa_instance: 'SSA', best_fitness: float):
        """迭代结束回调"""
        self.best_fitness_history.append(best_fitness)
        self.iteration_times.append(time.time())

        # 记录收敛数据
        self.convergence_data.append({
            'iteration': iteration,
            'best_fitness': best_fitness,
            'best_position': ssa_instance.best_position.copy(),
            'population_diversity': ssa_instance._calculate_diversity(),
            'convergence_rate': ssa_instance._calculate_convergence_rate()
        })

    def on_optimization_complete(self, ssa_instance: 'SSA'):
        """优化完成回调"""
        logger.info(f"SSA优化完成，最佳适应度: {ssa_instance.best_fitness:.6f}")
        logger.info(f"最佳参数: {ssa_instance.best_position}")

class MultiObjectiveObjective(ABC):
    """多目标优化目标函数基类"""

    @abstractmethod
    def evaluate(self, params: np.ndarray) -> List[float]:
        """评估多个目标函数"""
        pass

class MultiObjectiveSSA:
    """多目标麻雀搜索算法"""

    def __init__(self, config: SSAConfig, objective_functions: List[MultiObjectiveObjective]):
        self.config = config
        self.objective_functions = objective_functions

        # 计算目标函数数量
        self.num_objectives = 0
        for obj_func in objective_functions:
            # 测试一个目标函数来获取目标数量
            test_params = np.random.uniform(config.lb, config.ub)
            test_result = obj_func.evaluate(test_params)
            if isinstance(test_result, (list, tuple)):
                self.num_objectives += len(test_result)
            else:
                self.num_objectives += 1

        # Pareto前沿
        self.pareto_front = []
        self.pareto_solutions = []

        # 种群初始化
        self.population = self._initialize_population()
        self.fitness_matrix = np.zeros((self.config.pop_size, self.num_objectives))

        # 计算初始适应度
        for i in range(self.config.pop_size):
            self.fitness_matrix[i] = self._evaluate_objectives(self.population[i])

        # 更新Pareto前沿
        self._update_pareto_front()

        logger.info(f"多目标SSA初始化完成，目标数量: {self.num_objectives}")

    def _initialize_population(self) -> np.ndarray:
        """初始化种群"""
        population = np.zeros((self.config.pop_size, self.config.dim))

        for i in range(self.config.pop_size):
            for j in range(self.config.dim):
                population[i, j] = random.uniform(self.config.lb[j], self.config.ub[j])

        return population

    def _evaluate_objectives(self, params: np.ndarray) -> np.ndarray:
        """评估多个目标函数"""
        objectives = []
        for obj_func in self.objective_functions:
            result = obj_func.evaluate(params)
            if isinstance(result, (list, tuple)):
                objectives.extend(result)
            else:
                objectives.append(result)
        return np.array(objectives)

    def _update_pareto_front(self):
        """更新Pareto前沿"""
        # 找到非支配解
        non_dominated = []
        for i in range(self.config.pop_size):
            is_dominated = False
            for j in range(self.config.pop_size):
                if i != j:
                    if self._dominates(self.fitness_matrix[j], self.fitness_matrix[i]):
                        is_dominated = True
                        break
            if not is_dominated:
                non_dominated.append(i)

        # 更新Pareto前沿
        self.pareto_front = [self.fitness_matrix[i] for i in non_dominated]
        self.pareto_solutions = [self.population[i] for i in non_dominated]

        # 限制Pareto前沿大小
        if len(self.pareto_front) > self.config.pareto_size:
            self._reduce_pareto_front()

    def _dominates(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """判断obj1是否支配obj2"""
        return np.all(obj1 <= obj2) and np.any(obj1 < obj2)

    def _reduce_pareto_front(self):
        """减少Pareto前沿大小"""
        # 使用拥挤距离排序
        distances = self._calculate_crowding_distance()
        sorted_indices = np.argsort(distances)[::-1]

        # 保留前pareto_size个解
        keep_indices = sorted_indices[:self.config.pareto_size]
        self.pareto_front = [self.pareto_front[i] for i in keep_indices]
        self.pareto_solutions = [self.pareto_solutions[i] for i in keep_indices]

    def _calculate_crowding_distance(self) -> np.ndarray:
        """计算拥挤距离"""
        distances = np.zeros(len(self.pareto_front))

        for m in range(self.num_objectives):
            # 按第m个目标排序
            sorted_indices = np.argsort([obj[m] for obj in self.pareto_front])

            # 边界解距离设为无穷大
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')

            # 计算中间解的拥挤距离
            obj_range = self.pareto_front[sorted_indices[-1]][m] - self.pareto_front[sorted_indices[0]][m]
            if obj_range > 0:
                for i in range(1, len(sorted_indices) - 1):
                    distances[sorted_indices[i]] += (
                        self.pareto_front[sorted_indices[i+1]][m] -
                        self.pareto_front[sorted_indices[i-1]][m]
                    ) / obj_range

        return distances

    def optimize(self) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[str, Any]]:
        """执行多目标优化"""
        logger.info("开始多目标SSA优化...")

        start_time = time.time()

        for iteration in range(self.config.max_iter):
            # 更新种群
            self._update_population()

            # 更新Pareto前沿
            self._update_pareto_front()

            logger.info(f"迭代 {iteration+1}/{self.config.max_iter}, Pareto解数量: {len(self.pareto_front)}")

        end_time = time.time()

        optimization_info = {
            'total_time': end_time - start_time,
            'iterations': self.config.max_iter,
            'pareto_size': len(self.pareto_front),
            'num_objectives': self.num_objectives
        }

        logger.info(f"多目标SSA优化完成，耗时: {optimization_info['total_time']:.2f}s")

        return self.pareto_solutions, self.pareto_front, optimization_info

    def _update_population(self):
        """更新种群"""
        new_population = self.population.copy()

        for i in range(self.config.pop_size):
            # 选择更新策略
            if random.random() < self.config.discovery_rate:
                # 发现者更新
                new_population[i] = self._discoverer_update(i)
            else:
                # 跟随者更新
                new_population[i] = self._follower_update(i)

            # 边界处理
            new_population[i] = np.clip(new_population[i], self.config.lb, self.config.ub)

        # 评估新种群
        for i in range(self.config.pop_size):
            new_fitness = self._evaluate_objectives(new_population[i])

            # 如果新解不被任何Pareto解支配，则接受
            if not self._is_dominated_by_pareto(new_fitness):
                self.population[i] = new_population[i]
                self.fitness_matrix[i] = new_fitness

    def _discoverer_update(self, idx: int) -> np.ndarray:
        """发现者更新策略"""
        if random.random() < self.config.safety_threshold:
            # 安全状态：正常觅食
            return self.population[idx] + np.random.normal(0, 0.1, self.config.dim)
        else:
            # 危险状态：快速移动
            return self.population[idx] + np.random.normal(0, 0.5, self.config.dim)

    def _follower_update(self, idx: int) -> np.ndarray:
        """跟随者更新策略"""
        # 选择Pareto前沿中的一个解作为目标
        target_idx = random.randint(0, len(self.pareto_solutions) - 1)
        target_solution = self.pareto_solutions[target_idx]

        # 向目标解移动
        direction = target_solution - self.population[idx]
        return self.population[idx] + 0.5 * direction + np.random.normal(0, 0.1, self.config.dim)

    def _is_dominated_by_pareto(self, fitness: np.ndarray) -> bool:
        """检查解是否被Pareto前沿中的任何解支配"""
        for pareto_fitness in self.pareto_front:
            if self._dominates(pareto_fitness, fitness):
                return True
        return False

class SSA:
    """
    麻雀搜索算法（Sparrow Search Algorithm）

    用于优化GluFormer-MoE框架的超参数，包括：
    1. GluFormer时序模型超参数
    2. 多任务损失权重
    3. 文化适配正则系数
    4. 图像-营养对齐参数
    5. 支持多目标优化
    """

    def __init__(
        self,
        config: SSAConfig,
        objective_function: Callable[[np.ndarray], float],
        callback: Optional[SSACallback] = None
    ):
        self.config = config
        self.objective_function = objective_function
        self.callback = callback or SSACallback()

        # 验证配置
        self._validate_config()

        # 初始化种群
        self.population = self._initialize_population()
        self.fitness = np.zeros(self.config.pop_size)

        # 计算初始适应度
        for i in range(self.config.pop_size):
            self.fitness[i] = self.objective_function(self.population[i])

        # 找到最佳个体
        self.best_idx = np.argmin(self.fitness)
        self.best_position = self.population[self.best_idx].copy()
        self.best_fitness = self.fitness[self.best_idx]

        # 早停相关
        self.no_improvement_count = 0
        self.best_fitness_history = [self.best_fitness]

        # 统计信息
        self.iteration_count = 0
        self.total_evaluations = self.config.pop_size

        logger.info(f"SSA初始化完成，种群大小: {self.config.pop_size}, 维度: {self.config.dim}")
        logger.info(f"初始最佳适应度: {self.best_fitness:.6f}")

    def _validate_config(self):
        """验证配置参数"""
        if len(self.config.lb) != self.config.dim or len(self.config.ub) != self.config.dim:
            raise ValueError("边界数组长度必须等于搜索维度")

        if any(lb >= ub for lb, ub in zip(self.config.lb, self.config.ub)):
            raise ValueError("下边界必须小于上边界")

        if self.config.pop_size < 4:
            raise ValueError("种群大小至少为4")

        if self.config.max_iter < 1:
            raise ValueError("最大迭代次数至少为1")

    def _initialize_population(self) -> np.ndarray:
        """初始化种群"""
        population = np.zeros((self.config.pop_size, self.config.dim))

        for i in range(self.config.pop_size):
            for j in range(self.config.dim):
                population[i, j] = random.uniform(self.config.lb[j], self.config.ub[j])

        return population

    def _calculate_diversity(self) -> float:
        """计算种群多样性"""
        if self.config.pop_size <= 1:
            return 0.0

        # 计算种群中个体间的平均距离
        distances = []
        for i in range(self.config.pop_size):
            for j in range(i + 1, self.config.pop_size):
                dist = np.linalg.norm(self.population[i] - self.population[j])
                distances.append(dist)

        return np.mean(distances) if distances else 0.0

    def _calculate_convergence_rate(self) -> float:
        """计算收敛率"""
        if len(self.best_fitness_history) < 2:
            return 0.0

        # 计算最近几代适应度的变化率
        recent_fitness = self.best_fitness_history[-min(5, len(self.best_fitness_history)):]
        if len(recent_fitness) < 2:
            return 0.0

        improvement = recent_fitness[0] - recent_fitness[-1]
        return improvement / abs(recent_fitness[0]) if recent_fitness[0] != 0 else 0.0

    def _levy_flight(self, beta: float = 1.5) -> np.ndarray:
        """Levy飞行扰动"""
        if not self.config.levy_flight:
            return np.random.randn(self.config.dim)

        # 生成Levy分布随机数
        sigma = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                (math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2)))) ** (1 / beta)

        u = np.random.randn(self.config.dim) * sigma
        v = np.random.randn(self.config.dim)
        step = u / (np.abs(v) ** (1 / beta))

        return step

    def _adaptive_chaos_factor(self, iteration: int) -> float:
        """自适应混沌因子"""
        if not self.config.adaptive_chaos:
            return 1.0

        # 基于迭代次数和收敛情况调整混沌因子
        progress = iteration / self.config.max_iter
        convergence_rate = self._calculate_convergence_rate()

        # 早期使用较大混沌因子，后期逐渐减小
        base_factor = 1.0 - 0.5 * progress

        # 根据收敛情况调整
        if convergence_rate < 0.01:  # 收敛缓慢时增加混沌
            base_factor *= 1.5
        elif convergence_rate > 0.1:  # 收敛过快时减少混沌
            base_factor *= 0.5

        return max(0.1, min(2.0, base_factor))

    def _update_discoverers(self, iteration: int):
        """更新发现者（探索者）"""
        num_discoverers = int(self.config.pop_size * self.config.discovery_rate)

        for i in range(num_discoverers):
            if self.fitness[i] > np.mean(self.fitness):
                # 处于危险状态，需要快速移动
                if random.random() < self.config.safety_threshold:
                    # 安全状态，正常觅食
                    alpha = random.random()
                    self.population[i] = self.population[i] * np.exp(
                        -alpha * (i + 1) / (self.config.max_iter * random.random())
                    )
                else:
                    # 危险状态，快速移动
                    chaos_factor = self._adaptive_chaos_factor(iteration)
                    levy_step = self._levy_flight() * chaos_factor
                    self.population[i] = self.population[i] + levy_step
            else:
                # 安全状态，正常觅食
                alpha = random.random()
                self.population[i] = self.population[i] * np.exp(
                    -alpha * (i + 1) / (self.config.max_iter * random.random())
                )

            # 边界处理
            self.population[i] = np.clip(self.population[i], self.config.lb, self.config.ub)

    def _update_followers(self, iteration: int):
        """更新跟随者"""
        num_discoverers = int(self.config.pop_size * self.config.discovery_rate)

        for i in range(num_discoverers, self.config.pop_size):
            if i > self.config.pop_size / 2:
                # 后半部分跟随者，随机觅食
                chaos_factor = self._adaptive_chaos_factor(iteration)
                levy_step = self._levy_flight() * chaos_factor
                self.population[i] = levy_step * np.exp(
                    (self.best_position - self.population[i]) / (i + 1) ** 2
                )
            else:
                # 前半部分跟随者，跟随最佳个体
                self.population[i] = self.best_position + np.abs(
                    self.population[i] - self.best_position
                ) * np.random.randn(self.config.dim) * 0.5

            # 边界处理
            self.population[i] = np.clip(self.population[i], self.config.lb, self.config.ub)

    def _update_scouts(self, iteration: int):
        """更新侦察者（警戒者）"""
        # 随机选择部分个体作为侦察者
        num_scouts = max(1, int(self.config.pop_size * 0.1))
        scout_indices = random.sample(range(self.config.pop_size), num_scouts)

        for idx in scout_indices:
            if self.fitness[idx] > np.mean(self.fitness):
                # 处于危险状态，需要逃离
                chaos_factor = self._adaptive_chaos_factor(iteration)
                levy_step = self._levy_flight() * chaos_factor
                self.population[idx] = self.best_position + levy_step
            else:
                # 安全状态，向最佳个体靠近
                self.population[idx] = self.population[idx] + np.random.randn(self.config.dim) * 0.1

            # 边界处理
            self.population[idx] = np.clip(self.population[idx], self.config.lb, self.config.ub)

    def _evaluate_population(self):
        """评估种群适应度"""
        for i in range(self.config.pop_size):
            try:
                self.fitness[i] = self.objective_function(self.population[i])
                self.total_evaluations += 1
            except Exception as e:
                logger.warning(f"适应度评估失败，个体 {i}: {e}")
                self.fitness[i] = float('inf')

    def _update_best(self):
        """更新最佳个体"""
        current_best_idx = np.argmin(self.fitness)
        current_best_fitness = self.fitness[current_best_idx]

        if current_best_fitness < self.best_fitness:
            self.best_fitness = current_best_fitness
            self.best_position = self.population[current_best_idx].copy()
            self.best_idx = current_best_idx
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        self.best_fitness_history.append(self.best_fitness)

    def _check_early_stopping(self) -> bool:
        """检查早停条件"""
        if not self.config.early_stopping:
            return False

        return self.no_improvement_count >= self.config.patience

    def optimize(self) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """
        执行SSA优化

        Returns:
            best_position: 最佳参数位置
            best_fitness: 最佳适应度值
            optimization_info: 优化信息字典
        """
        logger.info("开始SSA优化...")
        start_time = time.time()

        for iteration in range(self.config.max_iter):
            iteration_start_time = time.time()

            # 回调：迭代开始
            if self.callback:
                self.callback.on_iteration_start(iteration, self)

            # 更新发现者
            self._update_discoverers(iteration)

            # 更新跟随者
            self._update_followers(iteration)

            # 更新侦察者
            self._update_scouts(iteration)

            # 评估种群
            self._evaluate_population()

            # 更新最佳个体
            self._update_best()

            # 回调：迭代结束
            if self.callback:
                self.callback.on_iteration_end(iteration, self, self.best_fitness)

            # 记录迭代信息
            iteration_time = time.time() - iteration_start_time
            logger.info(
                f"迭代 {iteration + 1}/{self.config.max_iter}, "
                f"最佳适应度: {self.best_fitness:.6f}, "
                f"耗时: {iteration_time:.2f}s"
            )

            # 检查早停
            if self._check_early_stopping():
                logger.info(f"早停触发，连续 {self.config.patience} 代无改进")
                break

            self.iteration_count = iteration + 1

        # 优化完成
        total_time = time.time() - start_time

        if self.callback:
            self.callback.on_optimization_complete(self)

        # 构建优化信息
        optimization_info = {
            'total_iterations': self.iteration_count,
            'total_evaluations': self.total_evaluations,
            'total_time': total_time,
            'convergence_curve': self.best_fitness_history,
            'final_diversity': self._calculate_diversity(),
            'convergence_rate': self._calculate_convergence_rate(),
            'early_stopped': self._check_early_stopping(),
            'callback_data': self.callback.convergence_data if self.callback else []
        }

        logger.info(f"SSA优化完成，总耗时: {total_time:.2f}s")
        logger.info(f"最佳适应度: {self.best_fitness:.6f}")
        logger.info(f"最佳参数: {self.best_position}")

        return self.best_position, self.best_fitness, optimization_info

    def save_results(self, filepath: str):
        """保存优化结果"""
        results = {
            'best_position': self.best_position.tolist(),
            'best_fitness': self.best_fitness,
            'optimization_info': {
                'total_iterations': self.iteration_count,
                'total_evaluations': self.total_evaluations,
                'convergence_curve': self.best_fitness_history,
                'callback_data': self.callback.convergence_data if self.callback else []
            },
            'config': {
                'dim': self.config.dim,
                'pop_size': self.config.pop_size,
                'max_iter': self.config.max_iter,
                'lb': self.config.lb,
                'ub': self.config.ub
            }
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"优化结果已保存到: {filepath}")

    def load_results(self, filepath: str):
        """加载优化结果"""
        with open(filepath, 'r', encoding='utf-8') as f:
            results = json.load(f)

        self.best_position = np.array(results['best_position'])
        self.best_fitness = results['best_fitness']
        self.best_fitness_history = results['optimization_info']['convergence_curve']

        logger.info(f"优化结果已从 {filepath} 加载")
        return results

def create_ssa_optimizer(
    dim: int,
    pop_size: int = 20,
    max_iter: int = 30,
    lb: Optional[List[float]] = None,
    ub: Optional[List[float]] = None,
    objective_function: Optional[Callable[[np.ndarray], float]] = None
) -> SSA:
    """
    创建SSA优化器实例

    Args:
        dim: 搜索维度
        pop_size: 种群大小
        max_iter: 最大迭代次数
        lb: 下边界
        ub: 上边界
        objective_function: 目标函数

    Returns:
        SSA优化器实例
    """
    config = SSAConfig(
        dim=dim,
        pop_size=pop_size,
        max_iter=max_iter,
        lb=lb or [0.0] * dim,
        ub=ub or [1.0] * dim
    )

    if objective_function is None:
        # 默认目标函数（用于测试）
        def default_objective(x):
            return np.sum(x ** 2)
        objective_function = default_objective

    return SSA(config, objective_function)

if __name__ == "__main__":
    # 测试SSA优化器
    def test_objective(x):
        """测试目标函数"""
        return np.sum(x ** 2) + 0.1 * np.sin(10 * x[0])

    # 创建优化器
    ssa = create_ssa_optimizer(
        dim=3,
        pop_size=20,
        max_iter=50,
        lb=[-5.0, -5.0, -5.0],
        ub=[5.0, 5.0, 5.0],
        objective_function=test_objective
    )

    # 执行优化
    best_pos, best_fit, info = ssa.optimize()

    print(f"最佳位置: {best_pos}")
    print(f"最佳适应度: {best_fit}")
    print(f"总迭代次数: {info['total_iterations']}")
    print(f"总评估次数: {info['total_evaluations']}")
    print(f"总耗时: {info['total_time']:.2f}s")
