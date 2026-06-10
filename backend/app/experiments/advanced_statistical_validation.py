

"""
高级统计验证模块
支持Bootstrap、Permutation Testing、置信区间计算等高级统计方法
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)

@dataclass
class StatisticalTestResult:
    """统计检验结果"""
    test_name: str
    statistic: float
    p_value: float
    confidence_interval: Optional[Tuple[float, float]] = None
    effect_size: Optional[float] = None
    interpretation: str = ""

@dataclass
class BootstrapResult:
    """Bootstrap结果"""
    original_statistic: float
    bootstrap_statistics: List[float]
    confidence_interval: Tuple[float, float]
    bias: float
    standard_error: float
    percentile_ci: Tuple[float, float]

class AdvancedStatisticalValidator:
    """高级统计验证器"""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.logger = logging.getLogger(__name__)

    def bootstrap_confidence_interval(
        self,
        data: Union[List[float], np.ndarray],
        statistic_func: callable,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        method: str = "percentile"
    ) -> BootstrapResult:
        """Bootstrap置信区间计算"""

        data = np.array(data)
        original_statistic = statistic_func(data)

        bootstrap_stats = []
        for _ in range(n_bootstrap):
            # 有放回抽样
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_stat = statistic_func(bootstrap_sample)
            bootstrap_stats.append(bootstrap_stat)

        bootstrap_stats = np.array(bootstrap_stats)

        # 计算偏差和标准误差
        bias = np.mean(bootstrap_stats) - original_statistic
        standard_error = np.std(bootstrap_stats, ddof=1)

        # 计算置信区间
        alpha = 1 - confidence_level
        if method == "percentile":
            lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
            upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
        elif method == "bias_corrected":
            # 偏差校正的百分位数方法
            z0 = stats.norm.ppf(np.mean(bootstrap_stats < original_statistic))
            z_alpha_2 = stats.norm.ppf(alpha / 2)
            z_1_alpha_2 = stats.norm.ppf(1 - alpha / 2)

            lower_percentile = stats.norm.cdf(2 * z0 + z_alpha_2) * 100
            upper_percentile = stats.norm.cdf(2 * z0 + z_1_alpha_2) * 100

            lower = np.percentile(bootstrap_stats, lower_percentile)
            upper = np.percentile(bootstrap_stats, upper_percentile)
        else:
            # 正态分布方法
            margin = stats.norm.ppf(1 - alpha / 2) * standard_error
            lower = original_statistic - margin
            upper = original_statistic + margin

        return BootstrapResult(
            original_statistic=original_statistic,
            bootstrap_statistics=bootstrap_stats.tolist(),
            confidence_interval=(lower, upper),
            bias=bias,
            standard_error=standard_error,
            percentile_ci=(np.percentile(bootstrap_stats, 2.5), np.percentile(bootstrap_stats, 97.5))
        )

    def permutation_test(
        self,
        group1: Union[List[float], np.ndarray],
        group2: Union[List[float], np.ndarray],
        statistic_func: callable = np.mean,
        n_permutations: int = 10000,
        alternative: str = "two-sided"
    ) -> StatisticalTestResult:
        """置换检验"""

        group1 = np.array(group1)
        group2 = np.array(group2)

        # 计算原始统计量
        original_diff = statistic_func(group1) - statistic_func(group2)

        # 合并数据
        combined = np.concatenate([group1, group2])
        n1 = len(group1)

        # 执行置换
        permuted_diffs = []
        for _ in range(n_permutations):
            # 随机打乱
            permuted = np.random.permutation(combined)
            perm_group1 = permuted[:n1]
            perm_group2 = permuted[n1:]

            perm_diff = statistic_func(perm_group1) - statistic_func(perm_group2)
            permuted_diffs.append(perm_diff)

        permuted_diffs = np.array(permuted_diffs)

        # 计算p值
        if alternative == "two-sided":
            p_value = np.mean(np.abs(permuted_diffs) >= np.abs(original_diff))
        elif alternative == "greater":
            p_value = np.mean(permuted_diffs >= original_diff)
        elif alternative == "less":
            p_value = np.mean(permuted_diffs <= original_diff)
        else:
            raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

        # 计算效应量（Cohen's d）
        pooled_std = np.sqrt(((n1 - 1) * np.var(group1) + (len(group2) - 1) * np.var(group2)) /
                            (n1 + len(group2) - 2))
        effect_size = original_diff / pooled_std if pooled_std > 0 else 0

        # 解释结果
        if p_value < 0.001:
            interpretation = "极显著差异 (p < 0.001)"
        elif p_value < 0.01:
            interpretation = "高度显著差异 (p < 0.01)"
        elif p_value < 0.05:
            interpretation = "显著差异 (p < 0.05)"
        elif p_value < 0.1:
            interpretation = "边际显著差异 (p < 0.1)"
        else:
            interpretation = "无显著差异 (p >= 0.1)"

        return StatisticalTestResult(
            test_name="Permutation Test",
            statistic=original_diff,
            p_value=p_value,
            effect_size=effect_size,
            interpretation=interpretation
        )

    def paired_t_test(
        self,
        before: Union[List[float], np.ndarray],
        after: Union[List[float], np.ndarray],
        alternative: str = "two-sided"
    ) -> StatisticalTestResult:
        """配对t检验"""

        before = np.array(before)
        after = np.array(after)

        if len(before) != len(after):
            raise ValueError("配对数据长度必须相等")

        # 计算差值
        differences = after - before

        # 执行配对t检验
        t_stat, p_value = stats.ttest_rel(after, before)

        # 计算置信区间
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        n = len(differences)
        se = std_diff / np.sqrt(n)

        # 95%置信区间
        t_critical = stats.t.ppf(0.975, n - 1)
        margin = t_critical * se
        ci = (mean_diff - margin, mean_diff + margin)

        # 计算效应量（Cohen's d）
        effect_size = mean_diff / std_diff if std_diff > 0 else 0

        # 解释结果
        if p_value < 0.001:
            interpretation = "极显著差异 (p < 0.001)"
        elif p_value < 0.01:
            interpretation = "高度显著差异 (p < 0.01)"
        elif p_value < 0.05:
            interpretation = "显著差异 (p < 0.05)"
        elif p_value < 0.1:
            interpretation = "边际显著差异 (p < 0.1)"
        else:
            interpretation = "无显著差异 (p >= 0.1)"

        return StatisticalTestResult(
            test_name="Paired t-test",
            statistic=t_stat,
            p_value=p_value,
            confidence_interval=ci,
            effect_size=effect_size,
            interpretation=interpretation
        )

    def mann_whitney_u_test(
        self,
        group1: Union[List[float], np.ndarray],
        group2: Union[List[float], np.ndarray],
        alternative: str = "two-sided"
    ) -> StatisticalTestResult:
        """Mann-Whitney U检验（非参数检验）"""

        group1 = np.array(group1)
        group2 = np.array(group2)

        # 执行Mann-Whitney U检验
        statistic, p_value = stats.mannwhitneyu(group1, group2, alternative=alternative)

        # 计算效应量（r = Z / sqrt(N)）
        n1, n2 = len(group1), len(group2)
        z_score = stats.norm.ppf(p_value / 2) if alternative == "two-sided" else stats.norm.ppf(p_value)
        effect_size = abs(z_score) / np.sqrt(n1 + n2)

        # 解释结果
        if p_value < 0.001:
            interpretation = "极显著差异 (p < 0.001)"
        elif p_value < 0.01:
            interpretation = "高度显著差异 (p < 0.01)"
        elif p_value < 0.05:
            interpretation = "显著差异 (p < 0.05)"
        elif p_value < 0.1:
            interpretation = "边际显著差异 (p < 0.1)"
        else:
            interpretation = "无显著差异 (p >= 0.1)"

        return StatisticalTestResult(
            test_name="Mann-Whitney U test",
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            interpretation=interpretation
        )

    def anova_test(
        self,
        groups: List[Union[List[float], np.ndarray]],
        group_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """方差分析（ANOVA）"""

        groups = [np.array(group) for group in groups]

        # 执行单因素方差分析
        f_stat, p_value = stats.f_oneway(*groups)

        # 计算效应量（eta-squared）
        all_data = np.concatenate(groups)
        grand_mean = np.mean(all_data)

        # 组间平方和
        ss_between = sum(len(group) * (np.mean(group) - grand_mean) ** 2 for group in groups)
        # 总平方和
        ss_total = sum((all_data - grand_mean) ** 2)

        eta_squared = ss_between / ss_total if ss_total > 0 else 0

        # 事后检验（Tukey HSD）
        from scipy.stats import tukey_hsd
        tukey_result = tukey_hsd(*groups)

        # 准备结果
        result = {
            "test_name": "One-way ANOVA",
            "f_statistic": f_stat,
            "p_value": p_value,
            "eta_squared": eta_squared,
            "group_means": [np.mean(group) for group in groups],
            "group_stds": [np.std(group, ddof=1) for group in groups],
            "group_sizes": [len(group) for group in groups],
            "tukey_hsd": {
                "statistic": tukey_result.statistic,
                "pvalue": tukey_result.pvalue
            }
        }

        if group_names:
            result["group_names"] = group_names

        # 解释结果
        if p_value < 0.001:
            result["interpretation"] = "极显著差异 (p < 0.001)"
        elif p_value < 0.01:
            result["interpretation"] = "高度显著差异 (p < 0.01)"
        elif p_value < 0.05:
            result["interpretation"] = "显著差异 (p < 0.05)"
        elif p_value < 0.1:
            result["interpretation"] = "边际显著差异 (p < 0.1)"
        else:
            result["interpretation"] = "无显著差异 (p >= 0.1)"

        return result

    def model_comparison_statistical_test(
        self,
        model_results: Dict[str, List[float]],
        metric_name: str = "accuracy",
        test_type: str = "paired_t_test"
    ) -> Dict[str, Any]:
        """模型比较的统计检验"""

        if len(model_results) < 2:
            return {"error": "至少需要两个模型进行比较"}

        model_names = list(model_results.keys())
        results = {}

        # 找到最佳模型
        best_model = max(model_names, key=lambda name: np.mean(model_results[name]))

        for model_name in model_names:
            if model_name == best_model:
                continue

            best_scores = model_results[best_model]
            current_scores = model_results[model_name]

            if test_type == "paired_t_test":
                # 配对t检验（假设两个模型在相同数据上测试）
                test_result = self.paired_t_test(best_scores, current_scores)
            elif test_type == "mann_whitney":
                # Mann-Whitney U检验
                test_result = self.mann_whitney_u_test(best_scores, current_scores)
            elif test_type == "permutation":
                # 置换检验
                test_result = self.permutation_test(best_scores, current_scores)
            else:
                # 默认使用独立样本t检验
                t_stat, p_value = stats.ttest_ind(best_scores, current_scores)
                test_result = StatisticalTestResult(
                    test_name="Independent t-test",
                    statistic=t_stat,
                    p_value=p_value,
                    interpretation=f"p = {p_value:.4f}"
                )

            results[f"{model_name}_vs_{best_model}"] = {
                "best_model": best_model,
                "current_model": model_name,
                "best_mean": np.mean(best_scores),
                "current_mean": np.mean(current_scores),
                "difference": np.mean(best_scores) - np.mean(current_scores),
                "test_result": test_result
            }

        return {
            "metric": metric_name,
            "best_model": best_model,
            "comparisons": results,
            "summary": {
                "total_models": len(model_names),
                "best_mean_score": np.mean(model_results[best_model]),
                "best_std_score": np.std(model_results[best_model], ddof=1)
            }
        }

    def calculate_effect_size_cohens_d(
        self,
        group1: Union[List[float], np.ndarray],
        group2: Union[List[float], np.ndarray]
    ) -> float:
        """计算Cohen's d效应量"""

        group1 = np.array(group1)
        group2 = np.array(group2)

        n1, n2 = len(group1), len(group2)
        mean1, mean2 = np.mean(group1), np.mean(group2)

        # 合并标准差
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        return (mean1 - mean2) / pooled_std if pooled_std > 0 else 0

    def power_analysis(
        self,
        effect_size: float,
        alpha: float = 0.05,
        power: float = 0.8,
        test_type: str = "two_sample"
    ) -> Dict[str, Any]:
        """统计功效分析"""

        from statsmodels.stats.power import ttest_power, ttest_solve_power

        if test_type == "two_sample":
            # 双样本t检验的功效分析
            n_required = ttest_solve_power(
                effect_size=effect_size,
                alpha=alpha,
                power=power,
                alternative='two-sided'
            )

            actual_power = ttest_power(
                effect_size=effect_size,
                nobs1=n_required,
                alpha=alpha,
                alternative='two-sided'
            )

            return {
                "test_type": "Two-sample t-test",
                "effect_size": effect_size,
                "alpha": alpha,
                "target_power": power,
                "required_sample_size": int(np.ceil(n_required)),
                "actual_power": actual_power
            }

        return {"error": "不支持的检验类型"}

# 全局统计验证器实例
statistical_validator = AdvancedStatisticalValidator()

__all__ = ["'logger'", "'StatisticalTestResult'", "'BootstrapResult'", "'AdvancedStatisticalValidator'", "'statistical_validator'"]
