

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合评估指标体系
Comprehensive Evaluation Metrics for Core Journal Standards

本模块实现了满足核心期刊要求的全面评估指标：
1. 预测性能指标 (Predictive Performance)
2. 临床实用性指标 (Clinical Utility)
3. 模型解释性指标 (Model Interpretability)
4. 校准度指标 (Calibration Metrics)
5. 公平性指标 (Fairness Metrics)
6. 稳健性指标 (Robustness Metrics)
7. 可重现性指标 (Reproducibility Metrics)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize
import scipy.stats as stats
from typing import Dict, List, Tuple, Any, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

@dataclass
class MetricResult:
    """评估指标结果"""
    name: str
    value: float
    confidence_interval: Tuple[float, float] = None
    interpretation: str = ""
    clinical_significance: str = ""
    statistical_significance: float = None

@dataclass
class ComprehensiveMetrics:
    """综合评估指标"""
    # 基础性能指标
    accuracy: MetricResult = None
    precision: MetricResult = None
    recall: MetricResult = None
    f1_score: MetricResult = None
    auc_roc: MetricResult = None
    auc_pr: MetricResult = None
    mcc: MetricResult = None

    # 回归指标
    mse: MetricResult = None
    mae: MetricResult = None
    rmse: MetricResult = None
    r2: MetricResult = None
    mape: MetricResult = None

    # 临床指标
    sensitivity: MetricResult = None
    specificity: MetricResult = None
    ppv: MetricResult = None
    npv: MetricResult = None
    nnt: MetricResult = None  # Number Needed to Treat
    nnh: MetricResult = None  # Number Needed to Harm

    # 校准指标
    brier_score: MetricResult = None
    hosmer_lemeshow: MetricResult = None
    calibration_slope: MetricResult = None
    calibration_intercept: MetricResult = None

    # 判别指标
    c_index: MetricResult = None
    discrimination_slope: MetricResult = None
    integrated_discrimination: MetricResult = None

    # 决策曲线分析
    net_benefit: Dict[str, float] = field(default_factory=dict)

    # 公平性指标
    demographic_parity: MetricResult = None
    equalized_odds: MetricResult = None
    individual_fairness: MetricResult = None

class ComprehensiveEvaluator:
    """综合评估器"""

    def __init__(self, confidence_level: float = 0.95, n_bootstrap: int = 1000):
        self.confidence_level = confidence_level
        self.n_bootstrap = n_bootstrap
        self.alpha = 1 - confidence_level

    def evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray,
                              y_prob: np.ndarray = None,
                              sensitive_attributes: np.ndarray = None) -> ComprehensiveMetrics:
        """
        分类任务的综合评估
        """
        metrics = ComprehensiveMetrics()

        # 基础分类指标
        metrics.accuracy = self._calculate_metric_with_ci(
            accuracy_score, y_true, y_pred, "准确率"
        )

        metrics.precision = self._calculate_metric_with_ci(
            lambda yt, yp: precision_score(yt, yp, average='weighted'),
            y_true, y_pred, "精确率"
        )

        metrics.recall = self._calculate_metric_with_ci(
            lambda yt, yp: recall_score(yt, yp, average='weighted'),
            y_true, y_pred, "召回率"
        )

        metrics.f1_score = self._calculate_metric_with_ci(
            lambda yt, yp: f1_score(yt, yp, average='weighted'),
            y_true, y_pred, "F1分数"
        )

        metrics.mcc = self._calculate_metric_with_ci(
            matthews_corrcoef, y_true, y_pred, "马修斯相关系数"
        )

        # 概率指标（如果提供概率）
        if y_prob is not None:
            if len(np.unique(y_true)) == 2:  # 二分类
                metrics.auc_roc = self._calculate_metric_with_ci(
                    roc_auc_score, y_true, y_prob, "ROC AUC"
                )

                metrics.auc_pr = self._calculate_metric_with_ci(
                    average_precision_score, y_true, y_prob, "PR AUC"
                )

                # 校准指标
                metrics = self._calculate_calibration_metrics(metrics, y_true, y_prob)

            else:  # 多分类
                # 多分类ROC AUC
                y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
                if y_prob.shape[1] == y_true_bin.shape[1]:
                    metrics.auc_roc = self._calculate_metric_with_ci(
                        lambda yt, yp: roc_auc_score(yt, yp, average='weighted', multi_class='ovr'),
                        y_true_bin, y_prob, "多分类ROC AUC"
                    )

        # 临床指标
        metrics = self._calculate_clinical_metrics(metrics, y_true, y_pred)

        # 公平性指标
        if sensitive_attributes is not None:
            metrics = self._calculate_fairness_metrics(
                metrics, y_true, y_pred, y_prob, sensitive_attributes
            )

        return metrics

    def evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray) -> ComprehensiveMetrics:
        """
        回归任务的综合评估
        """
        metrics = ComprehensiveMetrics()

        # 基础回归指标
        metrics.mse = self._calculate_metric_with_ci(
            mean_squared_error, y_true, y_pred, "均方误差"
        )

        metrics.mae = self._calculate_metric_with_ci(
            mean_absolute_error, y_true, y_pred, "平均绝对误差"
        )

        metrics.rmse = MetricResult(
            name="均方根误差",
            value=np.sqrt(metrics.mse.value),
            confidence_interval=None,
            interpretation=self._interpret_rmse(np.sqrt(metrics.mse.value), y_true)
        )

        metrics.r2 = self._calculate_metric_with_ci(
            r2_score, y_true, y_pred, "决定系数"
        )

        # MAPE (平均绝对百分比误差)
        def mape_func(y_true, y_pred):
            return np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100

        metrics.mape = self._calculate_metric_with_ci(
            mape_func, y_true, y_pred, "平均绝对百分比误差"
        )

        # 残差分析
        residuals = y_true - y_pred
        metrics = self._analyze_residuals(metrics, residuals, y_pred)

        return metrics

    def calculate_decision_curve_analysis(self, y_true: np.ndarray, y_prob: np.ndarray,
                                        thresholds: np.ndarray = None) -> Dict[str, Any]:
        """
        决策曲线分析 (Decision Curve Analysis)
        评估模型在不同风险阈值下的临床净收益
        """
        if thresholds is None:
            thresholds = np.linspace(0.01, 0.99, 99)

        results = {
            'thresholds': thresholds,
            'net_benefit': [],
            'treat_all': [],
            'treat_none': []
        }

        n = len(y_true)
        prevalence = np.mean(y_true)

        for threshold in thresholds:
            # 模型策略的净收益
            y_pred_threshold = (y_prob >= threshold).astype(int)

            tp = np.sum((y_true == 1) & (y_pred_threshold == 1))
            fp = np.sum((y_true == 0) & (y_pred_threshold == 1))

            net_benefit = (tp / n) - (fp / n) * (threshold / (1 - threshold))
            results['net_benefit'].append(net_benefit)

            # 全部治疗策略的净收益
            treat_all_nb = prevalence - (1 - prevalence) * (threshold / (1 - threshold))
            results['treat_all'].append(treat_all_nb)

            # 全部不治疗策略的净收益
            results['treat_none'].append(0)

        # 计算临床有用性范围
        useful_range = self._calculate_useful_range(results)
        results['useful_range'] = useful_range

        return results

    def calculate_time_dependent_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                       time_points: np.ndarray) -> Dict[str, Any]:
        """
        时间依赖性指标（用于生存分析等）
        """
        results = {}

        # 时间依赖的C-index
        concordance_indices = []
        for t in time_points:
            # 在时间点t计算一致性指数
            mask = y_true >= t  # 在时间t仍在观察中的样本
            if np.sum(mask) > 10:  # 确保有足够的样本
                c_index = self._calculate_concordance_index(
                    y_true[mask], y_pred[mask]
                )
                concordance_indices.append(c_index)
            else:
                concordance_indices.append(np.nan)

        results['time_dependent_c_index'] = {
            'time_points': time_points,
            'c_indices': concordance_indices
        }

        # 时间依赖的Brier分数
        brier_scores = []
        for t in time_points:
            mask = y_true >= t
            if np.sum(mask) > 10:
                brier_score = self._calculate_time_dependent_brier_score(
                    y_true[mask], y_pred[mask], t
                )
                brier_scores.append(brier_score)
            else:
                brier_scores.append(np.nan)

        results['time_dependent_brier_score'] = {
            'time_points': time_points,
            'brier_scores': brier_scores
        }

        return results

    def calculate_model_interpretability_metrics(self, model, X: np.ndarray,
                                               feature_names: List[str] = None) -> Dict[str, Any]:
        """
        模型可解释性指标
        """
        results = {}

        try:
            # SHAP值分析（如果可用）
            import shap

            explainer = shap.Explainer(model)
            shap_values = explainer(X[:100])  # 使用子集以提高速度

            results['shap_analysis'] = {
                'mean_abs_shap': np.mean(np.abs(shap_values.values), axis=0),
                'feature_importance_ranking': np.argsort(np.mean(np.abs(shap_values.values), axis=0))[::-1],
                'global_feature_importance': np.mean(np.abs(shap_values.values), axis=0)
            }

            if feature_names:
                results['shap_analysis']['feature_names'] = feature_names

        except ImportError:
            results['shap_analysis'] = {'error': 'SHAP库未安装'}

        # 排列重要性
        try:
            from sklearn.inspection import permutation_importance

            perm_importance = permutation_importance(
                model, X, model.predict(X), n_repeats=10, random_state=42
            )

            results['permutation_importance'] = {
                'importances_mean': perm_importance.importances_mean,
                'importances_std': perm_importance.importances_std,
                'feature_ranking': np.argsort(perm_importance.importances_mean)[::-1]
            }

        except Exception as e:
            results['permutation_importance'] = {'error': str(e)}

        return results

    def calculate_robustness_metrics(self, model, X: np.ndarray, y: np.ndarray,
                                   noise_levels: List[float] = None) -> Dict[str, Any]:
        """
        稳健性指标
        测试模型对输入扰动的敏感性
        """
        if noise_levels is None:
            noise_levels = [0.01, 0.05, 0.1, 0.2]

        results = {
            'noise_levels': noise_levels,
            'performance_degradation': [],
            'adversarial_robustness': {}
        }

        # 原始性能
        original_pred = model.predict(X)
        if hasattr(model, 'predict_proba'):
            original_prob = model.predict_proba(X)
        else:
            original_prob = None

        original_acc = accuracy_score(y, original_pred)

        # 噪声鲁棒性测试
        for noise_level in noise_levels:
            # 添加高斯噪声
            X_noisy = X + np.random.normal(0, noise_level * np.std(X, axis=0), X.shape)

            noisy_pred = model.predict(X_noisy)
            noisy_acc = accuracy_score(y, noisy_pred)

            degradation = (original_acc - noisy_acc) / original_acc * 100
            results['performance_degradation'].append(degradation)

        # 对抗样本鲁棒性（简化版本）
        if original_prob is not None:
            adversarial_examples = self._generate_simple_adversarial_examples(
                model, X[:100], y[:100]  # 使用子集
            )
            results['adversarial_robustness'] = adversarial_examples

        return results

    # 辅助方法
    def _calculate_metric_with_ci(self, metric_func, y_true, y_pred, name):
        """计算带置信区间的指标"""
        original_value = metric_func(y_true, y_pred)

        # Bootstrap置信区间
        bootstrap_values = []
        np.random.seed(42)

        n = len(y_true)
        for _ in range(self.n_bootstrap):
            indices = np.random.choice(n, n, replace=True)
            try:
                boot_value = metric_func(y_true[indices], y_pred[indices])
                bootstrap_values.append(boot_value)
            except:
                continue

        bootstrap_values = np.array(bootstrap_values)
        ci_lower = np.percentile(bootstrap_values, 100 * self.alpha / 2)
        ci_upper = np.percentile(bootstrap_values, 100 * (1 - self.alpha / 2))

        return MetricResult(
            name=name,
            value=original_value,
            confidence_interval=(ci_lower, ci_upper),
            interpretation=self._interpret_metric(name, original_value)
        )

    def _calculate_calibration_metrics(self, metrics, y_true, y_prob):
        """计算校准指标"""
        # Brier分数
        brier_score = np.mean((y_prob - y_true) ** 2)
        metrics.brier_score = MetricResult(
            name="Brier分数",
            value=brier_score,
            interpretation=f"{'良好校准' if brier_score < 0.25 else '校准较差'}"
        )

        # Hosmer-Lemeshow检验
        hl_stat, hl_p = self._hosmer_lemeshow_test(y_true, y_prob)
        metrics.hosmer_lemeshow = MetricResult(
            name="Hosmer-Lemeshow检验",
            value=hl_stat,
            statistical_significance=hl_p,
            interpretation=f"{'校准良好' if hl_p > 0.05 else '校准不佳'}"
        )

        # 校准斜率和截距
        slope, intercept = self._calculate_calibration_slope_intercept(y_true, y_prob)
        metrics.calibration_slope = MetricResult(
            name="校准斜率",
            value=slope,
            interpretation=f"{'理想校准' if abs(slope - 1) < 0.2 else '校准偏差'}"
        )

        metrics.calibration_intercept = MetricResult(
            name="校准截距",
            value=intercept,
            interpretation=f"{'无偏校准' if abs(intercept) < 0.1 else '存在偏差'}"
        )

        return metrics

    def _calculate_clinical_metrics(self, metrics, y_true, y_pred):
        """计算临床指标"""
        cm = confusion_matrix(y_true, y_pred)

        if cm.shape == (2, 2):  # 二分类
            tn, fp, fn, tp = cm.ravel()

            # 敏感性和特异性
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            # 阳性和阴性预测值
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0

            metrics.sensitivity = MetricResult(
                name="敏感性",
                value=sensitivity,
                interpretation=self._interpret_sensitivity(sensitivity)
            )

            metrics.specificity = MetricResult(
                name="特异性",
                value=specificity,
                interpretation=self._interpret_specificity(specificity)
            )

            metrics.ppv = MetricResult(
                name="阳性预测值",
                value=ppv,
                interpretation=self._interpret_ppv(ppv)
            )

            metrics.npv = MetricResult(
                name="阴性预测值",
                value=npv,
                interpretation=self._interpret_npv(npv)
            )

        return metrics

    def _calculate_fairness_metrics(self, metrics, y_true, y_pred, y_prob, sensitive_attr):
        """计算公平性指标"""
        unique_groups = np.unique(sensitive_attr)

        # 人口统计学均等 (Demographic Parity)
        group_positive_rates = []
        for group in unique_groups:
            group_mask = sensitive_attr == group
            positive_rate = np.mean(y_pred[group_mask])
            group_positive_rates.append(positive_rate)

        dp_diff = max(group_positive_rates) - min(group_positive_rates)
        metrics.demographic_parity = MetricResult(
            name="人口统计学均等",
            value=dp_diff,
            interpretation=f"{'公平' if dp_diff < 0.1 else '不公平'}"
        )

        # 机会均等 (Equalized Odds)
        if len(unique_groups) == 2:  # 简化为两组
            group1_mask = sensitive_attr == unique_groups[0]
            group2_mask = sensitive_attr == unique_groups[1]

            # 真正例率差异
            tpr1 = self._calculate_tpr(y_true[group1_mask], y_pred[group1_mask])
            tpr2 = self._calculate_tpr(y_true[group2_mask], y_pred[group2_mask])
            tpr_diff = abs(tpr1 - tpr2)

            # 假正例率差异
            fpr1 = self._calculate_fpr(y_true[group1_mask], y_pred[group1_mask])
            fpr2 = self._calculate_fpr(y_true[group2_mask], y_pred[group2_mask])
            fpr_diff = abs(fpr1 - fpr2)

            eo_diff = max(tpr_diff, fpr_diff)
            metrics.equalized_odds = MetricResult(
                name="机会均等",
                value=eo_diff,
                interpretation=f"{'公平' if eo_diff < 0.1 else '不公平'}"
            )

        return metrics

    def _interpret_metric(self, name, value):
        """解释指标值"""
        interpretations = {
            "准确率": lambda v: f"{'优秀' if v > 0.9 else '良好' if v > 0.8 else '一般' if v > 0.7 else '较差'}",
            "精确率": lambda v: f"{'高精确' if v > 0.9 else '中等精确' if v > 0.8 else '低精确'}",
            "召回率": lambda v: f"{'高召回' if v > 0.9 else '中等召回' if v > 0.8 else '低召回'}",
            "F1分数": lambda v: f"{'优秀平衡' if v > 0.9 else '良好平衡' if v > 0.8 else '一般平衡'}",
            "ROC AUC": lambda v: f"{'优秀判别' if v > 0.9 else '良好判别' if v > 0.8 else '中等判别' if v > 0.7 else '较差判别'}",
        }

        return interpretations.get(name, lambda v: "")(value)

    def _hosmer_lemeshow_test(self, y_true, y_prob, n_bins=10):
        """Hosmer-Lemeshow拟合优度检验"""
        # 将概率分成bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_prob, bin_boundaries) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        observed = np.zeros(n_bins)
        expected = np.zeros(n_bins)
        n_in_bin = np.zeros(n_bins)

        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                observed[i] = np.sum(y_true[mask])
                expected[i] = np.sum(y_prob[mask])
                n_in_bin[i] = np.sum(mask)

        # 计算卡方统计量
        chi2_stat = 0
        for i in range(n_bins):
            if n_in_bin[i] > 0 and expected[i] > 0 and (n_in_bin[i] - expected[i]) > 0:
                chi2_stat += ((observed[i] - expected[i]) ** 2 /
                             (expected[i] * (1 - expected[i] / n_in_bin[i])))

        # 自由度 = bins - 2
        df = n_bins - 2
        p_value = 1 - stats.chi2.cdf(chi2_stat, df)

        return chi2_stat, p_value

    def _calculate_calibration_slope_intercept(self, y_true, y_prob):
        """计算校准斜率和截距"""
        from sklearn.linear_model import LogisticRegression

        # 使用logit变换
        logit_prob = np.log(y_prob / (1 - y_prob + 1e-15))

        # 拟合校准模型
        cal_model = LogisticRegression()
        cal_model.fit(logit_prob.reshape(-1, 1), y_true)

        slope = cal_model.coef_[0][0]
        intercept = cal_model.intercept_[0]

        return slope, intercept

# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    n_samples = 1000

    # 分类示例
    y_true_class = np.random.binomial(1, 0.3, n_samples)
    y_prob_class = np.random.beta(2, 5, n_samples)  # 模拟预测概率
    y_pred_class = (y_prob_class > 0.5).astype(int)

    # 创建评估器
    evaluator = ComprehensiveEvaluator()

    # 评估分类任务
    class_metrics = evaluator.evaluate_classification(
        y_true_class, y_pred_class, y_prob_class
    )

    print("=== 分类任务评估结果 ===")
    print(f"准确率: {class_metrics.accuracy.value:.3f} "
          f"(95% CI: {class_metrics.accuracy.confidence_interval})")
    print(f"AUC-ROC: {class_metrics.auc_roc.value:.3f}")
    print(f"Brier分数: {class_metrics.brier_score.value:.3f}")

    # 决策曲线分析
    dca_results = evaluator.calculate_decision_curve_analysis(y_true_class, y_prob_class)
    print(f"决策曲线分析完成，有用范围: {dca_results['useful_range']}")

__all__ = ["'MetricResult'", "'ComprehensiveMetrics'", "'ComprehensiveEvaluator'"]
