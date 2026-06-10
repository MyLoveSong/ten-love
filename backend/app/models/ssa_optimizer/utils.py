"""
SSA优化器工具模块
提供日志记录、收敛曲线绘制、参数验证等功能
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import seaborn as sns
import logging
import json
import time
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from pathlib import Path
import pandas as pd
from dataclasses import dataclass, asdict
import warnings

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

logger = logging.getLogger(__name__)

@dataclass
class OptimizationMetrics:
    """优化指标"""
    best_fitness: float
    convergence_rate: float
    diversity: float
    evaluation_count: int
    total_time: float
    improvement_percentage: float
    stability_score: float

class SSALogger:
    """SSA优化器日志记录器"""

    def __init__(self, log_dir: str = "logs/ssa", log_level: str = "INFO"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 设置日志级别
        level = getattr(logging, log_level.upper())

        # 创建日志记录器
        self.logger = logging.getLogger("SSA_Optimizer")
        self.logger.setLevel(level)

        # 清除现有处理器
        self.logger.handlers.clear()

        # 文件处理器
        file_handler = logging.FileHandler(
            self.log_dir / f"ssa_optimization_{time.strftime('%Y%m%d_%H%M%S')}.log",
            encoding='utf-8'
        )
        file_handler.setLevel(level)

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        # 格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # 优化历史记录
        self.optimization_history = []
        self.iteration_logs = []

        self.logger.info("SSA日志记录器初始化完成")

    def log_optimization_start(self, config: Dict[str, Any]):
        """记录优化开始"""
        self.logger.info("=" * 50)
        self.logger.info("SSA优化开始")
        self.logger.info(f"配置参数: {json.dumps(config, indent=2, ensure_ascii=False)}")
        self.logger.info("=" * 50)

        self.optimization_history.append({
            'timestamp': time.time(),
            'event': 'optimization_start',
            'config': config
        })

    def log_iteration(self, iteration: int, best_fitness: float, best_position: np.ndarray,
                     diversity: float, convergence_rate: float, iteration_time: float):
        """记录迭代信息"""
        log_entry = {
            'iteration': iteration,
            'best_fitness': best_fitness,
            'best_position': best_position.tolist(),
            'diversity': diversity,
            'convergence_rate': convergence_rate,
            'iteration_time': iteration_time,
            'timestamp': time.time()
        }

        self.iteration_logs.append(log_entry)

        self.logger.info(
            f"迭代 {iteration}: 最佳适应度={best_fitness:.6f}, "
            f"多样性={diversity:.4f}, 收敛率={convergence_rate:.4f}, "
            f"耗时={iteration_time:.2f}s"
        )

    def log_optimization_end(self, best_position: np.ndarray, best_fitness: float,
                           total_time: float, total_iterations: int):
        """记录优化结束"""
        self.logger.info("=" * 50)
        self.logger.info("SSA优化完成")
        self.logger.info(f"最佳适应度: {best_fitness:.6f}")
        self.logger.info(f"最佳参数: {best_position}")
        self.logger.info(f"总迭代次数: {total_iterations}")
        self.logger.info(f"总耗时: {total_time:.2f}s")
        self.logger.info("=" * 50)

        self.optimization_history.append({
            'timestamp': time.time(),
            'event': 'optimization_end',
            'best_position': best_position.tolist(),
            'best_fitness': best_fitness,
            'total_time': total_time,
            'total_iterations': total_iterations
        })

    def log_warning(self, message: str):
        """记录警告"""
        self.logger.warning(message)

    def log_error(self, message: str, exception: Optional[Exception] = None):
        """记录错误"""
        if exception:
            self.logger.error(f"{message}: {str(exception)}", exc_info=True)
        else:
            self.logger.error(message)

    def save_optimization_log(self, filepath: str):
        """保存优化日志"""
        log_data = {
            'optimization_history': self.optimization_history,
            'iteration_logs': self.iteration_logs,
            'summary': {
                'total_iterations': len(self.iteration_logs),
                'final_fitness': self.iteration_logs[-1]['best_fitness'] if self.iteration_logs else None,
                'total_time': sum(log['iteration_time'] for log in self.iteration_logs)
            }
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"优化日志已保存到: {filepath}")

    def get_optimization_summary(self) -> Dict[str, Any]:
        """获取优化摘要"""
        if not self.iteration_logs:
            return {}

        fitness_values = [log['best_fitness'] for log in self.iteration_logs]
        diversity_values = [log['diversity'] for log in self.iteration_logs]
        convergence_rates = [log['convergence_rate'] for log in self.iteration_logs]

        return {
            'total_iterations': len(self.iteration_logs),
            'initial_fitness': fitness_values[0],
            'final_fitness': fitness_values[-1],
            'best_fitness': min(fitness_values),
            'improvement_percentage': ((fitness_values[0] - fitness_values[-1]) / fitness_values[0]) * 100,
            'average_diversity': np.mean(diversity_values),
            'average_convergence_rate': np.mean(convergence_rates),
            'total_time': sum(log['iteration_time'] for log in self.iteration_logs)
        }

class ConvergencePlotter:
    """收敛曲线绘制器"""

    def __init__(self, output_dir: str = "outputs/ssa_plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 设置绘图样式
        sns.set_style("whitegrid")
        plt.style.use('seaborn-v0_8')

        logger.info(f"收敛曲线绘制器初始化完成，输出目录: {self.output_dir}")

    def plot_convergence_curve(
        self,
        fitness_history: List[float],
        title: str = "SSA收敛曲线",
        save_path: Optional[str] = None
    ) -> str:
        """绘制收敛曲线"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            # 绘制收敛曲线
            ax.plot(fitness_history, 'b-', linewidth=2, label='最佳适应度')

            # 添加移动平均线
            if len(fitness_history) > 10:
                window_size = min(10, len(fitness_history) // 3)
                moving_avg = pd.Series(fitness_history).rolling(window=window_size).mean()
                ax.plot(moving_avg, 'r--', linewidth=1, alpha=0.7, label=f'移动平均({window_size})')

            # 设置标签和标题
            ax.set_xlabel('迭代次数', fontsize=12)
            ax.set_ylabel('适应度值', fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # 设置y轴为对数刻度（如果适应度值变化很大）
            if max(fitness_history) / min(fitness_history) > 100:
                ax.set_yscale('log')

            # 保存图片
            if save_path is None:
                save_path = self.output_dir / f"convergence_curve_{time.strftime('%Y%m%d_%H%M%S')}.png"

            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"收敛曲线已保存到: {save_path}")
            return str(save_path)

        except Exception as e:
            logger.error(f"绘制收敛曲线失败: {e}")
            return ""

    def plot_parameter_evolution(
        self,
        parameter_history: List[np.ndarray],
        parameter_names: List[str],
        title: str = "参数演化过程",
        save_path: Optional[str] = None
    ) -> str:
        """绘制参数演化过程"""
        try:
            if not parameter_history or not parameter_names:
                logger.warning("参数历史或参数名称为空，跳过绘制")
                return ""

            # 转换为DataFrame
            data = []
            for i, params in enumerate(parameter_history):
                for j, (name, value) in enumerate(zip(parameter_names, params)):
                    data.append({
                        'iteration': i,
                        'parameter': name,
                        'value': value
                    })

            df = pd.DataFrame(data)

            # 创建子图
            n_params = len(parameter_names)
            n_cols = min(3, n_params)
            n_rows = (n_params + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            if n_params == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes.reshape(1, -1)

            # 绘制每个参数的演化
            for i, param_name in enumerate(parameter_names):
                row = i // n_cols
                col = i % n_cols
                ax = axes[row, col] if n_rows > 1 else axes[col]

                param_data = df[df['parameter'] == param_name]
                ax.plot(param_data['iteration'], param_data['value'], 'b-', linewidth=2)
                ax.set_title(f'{param_name}演化', fontsize=12)
                ax.set_xlabel('迭代次数')
                ax.set_ylabel('参数值')
                ax.grid(True, alpha=0.3)

            # 隐藏多余的子图
            for i in range(n_params, n_rows * n_cols):
                row = i // n_cols
                col = i % n_cols
                ax = axes[row, col] if n_rows > 1 else axes[col]
                ax.set_visible(False)

            plt.suptitle(title, fontsize=14, fontweight='bold')
            plt.tight_layout()

            # 保存图片
            if save_path is None:
                save_path = self.output_dir / f"parameter_evolution_{time.strftime('%Y%m%d_%H%M%S')}.png"

            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"参数演化图已保存到: {save_path}")
            return str(save_path)

        except Exception as e:
            logger.error(f"绘制参数演化图失败: {e}")
            return ""

    def plot_diversity_convergence(
        self,
        diversity_history: List[float],
        convergence_history: List[float],
        title: str = "多样性与收敛性分析",
        save_path: Optional[str] = None
    ) -> str:
        """绘制多样性与收敛性分析图"""
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

            # 绘制多样性曲线
            ax1.plot(diversity_history, 'g-', linewidth=2, label='种群多样性')
            ax1.set_xlabel('迭代次数')
            ax1.set_ylabel('多样性值')
            ax1.set_title('种群多样性变化', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 绘制收敛率曲线
            ax2.plot(convergence_history, 'r-', linewidth=2, label='收敛率')
            ax2.set_xlabel('迭代次数')
            ax2.set_ylabel('收敛率')
            ax2.set_title('收敛率变化', fontsize=12)
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.suptitle(title, fontsize=14, fontweight='bold')
            plt.tight_layout()

            # 保存图片
            if save_path is None:
                save_path = self.output_dir / f"diversity_convergence_{time.strftime('%Y%m%d_%H%M%S')}.png"

            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"多样性收敛图已保存到: {save_path}")
            return str(save_path)

        except Exception as e:
            logger.error(f"绘制多样性收敛图失败: {e}")
            return ""

    def plot_optimization_summary(
        self,
        optimization_metrics: OptimizationMetrics,
        title: str = "优化结果摘要",
        save_path: Optional[str] = None
    ) -> str:
        """绘制优化结果摘要"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

            # 指标数据
            metrics = [
                ('最佳适应度', optimization_metrics.best_fitness, 'blue'),
                ('收敛率', optimization_metrics.convergence_rate, 'green'),
                ('多样性', optimization_metrics.diversity, 'orange'),
                ('稳定性得分', optimization_metrics.stability_score, 'red')
            ]

            # 绘制指标条形图
            names, values, colors = zip(*metrics)
            bars = ax1.bar(names, values, color=colors, alpha=0.7)
            ax1.set_title('关键指标', fontsize=12)
            ax1.set_ylabel('指标值')

            # 添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.4f}', ha='center', va='bottom')

            # 绘制评估次数饼图
            ax2.pie([optimization_metrics.evaluation_count, 1000],
                   labels=['实际评估', '最大评估'],
                   autopct='%1.1f%%',
                   colors=['lightblue', 'lightgray'])
            ax2.set_title('评估次数分布', fontsize=12)

            # 绘制时间分析
            time_data = [optimization_metrics.total_time, 3600 - optimization_metrics.total_time]
            ax3.pie(time_data,
                   labels=['优化时间', '剩余时间'],
                   autopct='%1.1f%%',
                   colors=['lightcoral', 'lightgray'])
            ax3.set_title('时间分配', fontsize=12)

            # 绘制改进百分比
            improvement_data = [optimization_metrics.improvement_percentage,
                              100 - optimization_metrics.improvement_percentage]
            ax4.pie(improvement_data,
                   labels=['改进', '未改进'],
                   autopct='%1.1f%%',
                   colors=['lightgreen', 'lightgray'])
            ax4.set_title('改进效果', fontsize=12)

            plt.suptitle(title, fontsize=14, fontweight='bold')
            plt.tight_layout()

            # 保存图片
            if save_path is None:
                save_path = self.output_dir / f"optimization_summary_{time.strftime('%Y%m%d_%H%M%S')}.png"

            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"优化摘要图已保存到: {save_path}")
            return str(save_path)

        except Exception as e:
            logger.error(f"绘制优化摘要图失败: {e}")
            return ""

class ParameterValidator:
    """参数验证器"""

    def __init__(self):
        self.validation_rules = {}
        self.validation_history = []

        logger.info("参数验证器初始化完成")

    def add_validation_rule(self, parameter_name: str, rule: Callable[[Any], bool],
                           error_message: str = ""):
        """添加验证规则"""
        self.validation_rules[parameter_name] = {
            'rule': rule,
            'error_message': error_message
        }

    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """验证参数"""
        errors = []

        for param_name, param_value in parameters.items():
            if param_name in self.validation_rules:
                rule_info = self.validation_rules[param_name]
                rule = rule_info['rule']
                error_msg = rule_info['error_message']

                try:
                    if not rule(param_value):
                        errors.append(f"{param_name}: {error_msg}")
                except Exception as e:
                    errors.append(f"{param_name}: 验证规则执行失败 - {str(e)}")

        # 记录验证历史
        self.validation_history.append({
            'timestamp': time.time(),
            'parameters': parameters.copy(),
            'is_valid': len(errors) == 0,
            'errors': errors.copy()
        })

        return len(errors) == 0, errors

    def validate_ssa_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """验证SSA配置"""
        errors = []

        # 检查必需参数
        required_params = ['dim', 'pop_size', 'max_iter', 'lb', 'ub']
        for param in required_params:
            if param not in config:
                errors.append(f"缺少必需参数: {param}")

        # 检查参数类型和范围
        if 'dim' in config:
            if not isinstance(config['dim'], int) or config['dim'] <= 0:
                errors.append("dim必须是正整数")

        if 'pop_size' in config:
            if not isinstance(config['pop_size'], int) or config['pop_size'] < 4:
                errors.append("pop_size必须是大于等于4的整数")

        if 'max_iter' in config:
            if not isinstance(config['max_iter'], int) or config['max_iter'] <= 0:
                errors.append("max_iter必须是正整数")

        # 检查边界参数
        if 'lb' in config and 'ub' in config:
            lb = config['lb']
            ub = config['ub']

            if not isinstance(lb, (list, np.ndarray)) or not isinstance(ub, (list, np.ndarray)):
                errors.append("lb和ub必须是列表或数组")
            elif len(lb) != len(ub):
                errors.append("lb和ub的长度必须相等")
            elif any(l >= u for l, u in zip(lb, ub)):
                errors.append("lb中的每个元素必须小于对应的ub元素")

        return len(errors) == 0, errors

    def get_validation_summary(self) -> Dict[str, Any]:
        """获取验证摘要"""
        if not self.validation_history:
            return {}

        total_validations = len(self.validation_history)
        successful_validations = sum(1 for h in self.validation_history if h['is_valid'])
        failed_validations = total_validations - successful_validations

        # 统计错误类型
        error_types = {}
        for history in self.validation_history:
            for error in history['errors']:
                error_type = error.split(':')[0] if ':' in error else 'unknown'
                error_types[error_type] = error_types.get(error_type, 0) + 1

        return {
            'total_validations': total_validations,
            'successful_validations': successful_validations,
            'failed_validations': failed_validations,
            'success_rate': successful_validations / total_validations if total_validations > 0 else 0,
            'error_types': error_types
        }

def create_ssa_logger(log_dir: str = "logs/ssa", log_level: str = "INFO") -> SSALogger:
    """创建SSA日志记录器"""
    return SSALogger(log_dir, log_level)

def create_convergence_plotter(output_dir: str = "outputs/ssa_plots") -> ConvergencePlotter:
    """创建收敛曲线绘制器"""
    return ConvergencePlotter(output_dir)

def create_parameter_validator() -> ParameterValidator:
    """创建参数验证器"""
    return ParameterValidator()

if __name__ == "__main__":
    # 测试工具模块
    print("测试SSA日志记录器...")
    logger = create_ssa_logger()
    logger.log_optimization_start({'dim': 7, 'pop_size': 20})
    logger.log_iteration(1, 0.5, np.array([1, 2, 3]), 0.8, 0.1, 1.0)
    logger.log_optimization_end(np.array([1, 2, 3]), 0.3, 10.0, 5)

    print("测试收敛曲线绘制器...")
    plotter = create_convergence_plotter()
    fitness_history = [1.0, 0.8, 0.6, 0.4, 0.3, 0.25, 0.2]
    plot_path = plotter.plot_convergence_curve(fitness_history)
    print(f"收敛曲线已保存到: {plot_path}")

    print("测试参数验证器...")
    validator = create_parameter_validator()
    validator.add_validation_rule('dim', lambda x: isinstance(x, int) and x > 0, "维度必须是正整数")

    test_params = {'dim': 7, 'pop_size': 20}
    is_valid, errors = validator.validate_parameters(test_params)
    print(f"参数验证结果: {is_valid}, 错误: {errors}")

    print("所有工具模块测试完成！")
