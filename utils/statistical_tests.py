#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计显著性检验工具
用于顶会顶刊论文的统计报告

参考: BEST_PRACTICES_AND_DATASETS.md
"""

import numpy as np
from scipy import stats
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


def statistical_significance_test(
    baseline_scores: List[float],
    our_scores: List[float],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    统计显著性检验

    使用配对t-test和Wilcoxon signed-rank test

    Args:
        baseline_scores: baseline方法的指标列表 [seed1, seed2, ...]
        our_scores: 我们方法的指标列表 [seed1, seed2, ...]
        alpha: 显著性水平 (默认0.05)

    Returns:
        result: 包含统计检验结果的字典
            - t_statistic: t统计量
            - p_value: p值 (t-test)
            - p_value_wilcoxon: p值 (Wilcoxon test)
            - is_significant: 是否显著 (p < alpha)
            - improvement: 改进幅度 (均值差)
            - improvement_percent: 改进百分比
            - mean_baseline: baseline均值
            - mean_our: 我们的均值
            - std_baseline: baseline标准差
            - std_our: 我们的标准差
    """
    if len(baseline_scores) != len(our_scores):
        raise ValueError(f"Score lists must have same length: {len(baseline_scores)} vs {len(our_scores)}")

    if len(baseline_scores) < 2:
        logger.warning("Need at least 2 samples for statistical test")
        return {
            't_statistic': float('nan'),
            'p_value': float('nan'),
            'p_value_wilcoxon': float('nan'),
            'is_significant': False,
            'improvement': np.mean(our_scores) - np.mean(baseline_scores),
            'improvement_percent': 0.0,
            'mean_baseline': np.mean(baseline_scores),
            'mean_our': np.mean(our_scores),
            'std_baseline': np.std(baseline_scores, ddof=1),
            'std_our': np.std(our_scores, ddof=1)
        }

    baseline_scores = np.array(baseline_scores)
    our_scores = np.array(our_scores)

    # 配对t-test
    t_statistic, p_value = stats.ttest_rel(our_scores, baseline_scores)

    # Wilcoxon signed-rank test (非参数，更稳健)
    try:
        statistic_wilcoxon, p_value_wilcoxon = stats.wilcoxon(our_scores, baseline_scores)
    except ValueError as e:
        logger.warning(f"Wilcoxon test failed: {e}")
        p_value_wilcoxon = float('nan')

    # 计算改进
    mean_baseline = np.mean(baseline_scores)
    mean_our = np.mean(our_scores)
    improvement = mean_our - mean_baseline
    improvement_percent = (improvement / mean_baseline * 100) if mean_baseline > 0 else 0.0

    is_significant = p_value < alpha

    return {
        't_statistic': float(t_statistic),
        'p_value': float(p_value),
        'p_value_wilcoxon': float(p_value_wilcoxon) if not np.isnan(p_value_wilcoxon) else float('nan'),
        'is_significant': is_significant,
        'improvement': float(improvement),
        'improvement_percent': float(improvement_percent),
        'mean_baseline': float(mean_baseline),
        'mean_our': float(mean_our),
        'std_baseline': float(np.std(baseline_scores, ddof=1)),
        'std_our': float(np.std(our_scores, ddof=1))
    }


def compute_confidence_interval(
    scores: List[float],
    confidence: float = 0.95
) -> Dict[str, float]:
    """
    计算置信区间

    Args:
        scores: 指标列表
        confidence: 置信水平 (默认0.95)

    Returns:
        result: 包含置信区间的字典
            - mean: 均值
            - std: 标准差
            - ci_low: 置信区间下界
            - ci_high: 置信区间上界
            - margin: 置信区间半宽
    """
    if not scores:
        return {
            'mean': 0.0,
            'std': 0.0,
            'ci_low': 0.0,
            'ci_high': 0.0,
            'margin': 0.0
        }

    scores = np.array(scores)
    mean = np.mean(scores)
    std = np.std(scores, ddof=1)  # 样本标准差
    n = len(scores)

    if n == 1:
        return {
            'mean': float(mean),
            'std': float(std),
            'ci_low': float(mean),
            'ci_high': float(mean),
            'margin': 0.0
        }

    # t分布的临界值
    t_critical = stats.t.ppf((1 + confidence) / 2, df=n-1)

    # 标准误差
    se = std / np.sqrt(n)

    # 置信区间
    margin = t_critical * se
    ci_low = mean - margin
    ci_high = mean + margin

    return {
        'mean': float(mean),
        'std': float(std),
        'ci_low': float(ci_low),
        'ci_high': float(ci_high),
        'margin': float(margin)
    }


def format_statistical_report(
    test_result: Dict[str, Any],
    metric_name: str = "Metric"
) -> str:
    """
    格式化统计检验报告

    Args:
        test_result: statistical_significance_test的返回结果
        metric_name: 指标名称

    Returns:
        formatted_str: 格式化后的字符串
    """
    lines = [
        f"### {metric_name} - Statistical Significance Test",
        "",
        f"**Baseline**: {test_result['mean_baseline']:.4f} ± {test_result['std_baseline']:.4f}",
        f"**Our Method**: {test_result['mean_our']:.4f} ± {test_result['std_our']:.4f}",
        f"**Improvement**: {test_result['improvement']:.4f} ({test_result['improvement_percent']:+.2f}%)",
        "",
        f"**Paired t-test**:",
        f"  - t-statistic: {test_result['t_statistic']:.4f}",
        f"  - p-value: {test_result['p_value']:.4f}",
        f"  - Significant: {'Yes' if test_result['is_significant'] else 'No'} (α = 0.05)",
        ""
    ]

    if not np.isnan(test_result['p_value_wilcoxon']):
        lines.append(f"**Wilcoxon signed-rank test**:")
        lines.append(f"  - p-value: {test_result['p_value_wilcoxon']:.4f}")
        lines.append("")

    return "\n".join(lines)


def compare_multiple_methods(
    method_scores: Dict[str, List[float]],
    baseline_name: str = "baseline",
    alpha: float = 0.05
) -> Dict[str, Dict[str, Any]]:
    """
    比较多个方法（相对于baseline）

    Args:
        method_scores: {method_name: [score1, score2, ...]}
        baseline_name: baseline方法名称
        alpha: 显著性水平

    Returns:
        comparisons: {method_name: test_result}
    """
    if baseline_name not in method_scores:
        raise ValueError(f"Baseline method '{baseline_name}' not found in method_scores")

    baseline_scores = method_scores[baseline_name]
    comparisons = {}

    for method_name, scores in method_scores.items():
        if method_name == baseline_name:
            continue

        comparisons[method_name] = statistical_significance_test(
            baseline_scores=baseline_scores,
            our_scores=scores,
            alpha=alpha
        )

    return comparisons
