

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
严谨实验设计模块
Rigorous Experimental Design for Core Journal Standards

本模块实现了满足核心期刊标准的实验设计：
1. 样本量计算和功效分析
2. 分层随机抽样
3. 交叉验证策略优化
4. 时间序列验证
5. 外部验证集
6. 敏感性分析
7. 偏差分析
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold, TimeSeriesSplit, GroupKFold,
    train_test_split, cross_validate
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import scipy.stats as stats
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class ExperimentalDesignConfig:
    """实验设计配置"""
    study_name: str
    primary_endpoint: str
    secondary_endpoints: List[str] = field(default_factory=list)
    alpha: float = 0.05
    power: float = 0.8
    effect_size: float = 0.5
    allocation_ratio: float = 1.0  # 组间分配比例
    stratification_factors: List[str] = field(default_factory=list)
    blocking_factors: List[str] = field(default_factory=list)
    minimum_sample_size: int = 30
    maximum_sample_size: int = 1000

class RigorousExperimentalDesign:
    """严谨实验设计类"""

    def __init__(self, config: ExperimentalDesignConfig):
        self.config = config
        self.design_results = {}
        self.validation_strategy = None

    def calculate_sample_size(self, test_type: str = 'two_sample_ttest') -> Dict[str, Any]:
        """
        严格的样本量计算
        支持多种统计检验的样本量计算
        """
        results = {}

        if test_type == 'two_sample_ttest':
            # 双样本t检验样本量计算
            from statsmodels.stats.power import tt_solve_power

            n = tt_solve_power(
                effect_size=self.config.effect_size,
                power=self.config.power,
                alpha=self.config.alpha,
                alternative='two-sided'
            )

            # 考虑分配比例
            n1 = n / (1 + 1/self.config.allocation_ratio)
            n2 = n / (1 + self.config.allocation_ratio)

            results['two_sample_ttest'] = {
                'total_sample_size': int(np.ceil(n * 2)),
                'group1_size': int(np.ceil(n1)),
                'group2_size': int(np.ceil(n2)),
                'power_achieved': self.config.power,
                'effect_size': self.config.effect_size,
                'alpha': self.config.alpha
            }

        elif test_type == 'paired_ttest':
            # 配对t检验样本量计算
            from statsmodels.stats.power import ttest_power

            n = stats.power.tt_solve_power(
                effect_size=self.config.effect_size,
                power=self.config.power,
                alpha=self.config.alpha,
                alternative='two-sided'
            )

            results['paired_ttest'] = {
                'sample_size': int(np.ceil(n)),
                'power_achieved': self.config.power,
                'effect_size': self.config.effect_size
            }

        elif test_type == 'anova':
            # 方差分析样本量计算
            from statsmodels.stats.power import ftest_anova_power

            # 假设3组比较
            n_groups = 3
            n_per_group = stats.power.ftest_anova_power(
                effect_size=self.config.effect_size,
                power=self.config.power,
                alpha=self.config.alpha,
                k_groups=n_groups
            )

            results['anova'] = {
                'total_sample_size': int(np.ceil(n_per_group * n_groups)),
                'n_per_group': int(np.ceil(n_per_group)),
                'n_groups': n_groups,
                'power_achieved': self.config.power
            }

        # 样本量调整
        results = self._adjust_sample_size_for_attrition(results)

        return results

    def design_stratified_sampling(self, data: pd.DataFrame,
                                 target_column: str,
                                 stratification_columns: List[str] = None) -> Dict[str, Any]:
        """
        分层随机抽样设计
        确保样本在重要协变量上的平衡
        """
        if stratification_columns is None:
            stratification_columns = self.config.stratification_factors

        results = {}

        # 创建分层变量
        if stratification_columns:
            # 组合分层因子
            data['strata'] = data[stratification_columns].apply(
                lambda x: '_'.join(x.astype(str)), axis=1
            )

            strata_info = data['strata'].value_counts()
            results['strata_distribution'] = strata_info.to_dict()

            # 检查每个分层的最小样本量
            min_stratum_size = strata_info.min()
            results['minimum_stratum_size'] = min_stratum_size

            if min_stratum_size < 10:
                results['warning'] = f"某些分层样本量过小 (最小: {min_stratum_size})"

        # 分层抽样
        train_indices, test_indices = self._stratified_split(
            data, target_column, stratification_columns
        )

        results['train_indices'] = train_indices
        results['test_indices'] = test_indices
        results['train_size'] = len(train_indices)
        results['test_size'] = len(test_indices)

        # 验证分层效果
        results['stratification_balance'] = self._validate_stratification_balance(
            data, train_indices, test_indices, stratification_columns
        )

        return results

    def design_cross_validation_strategy(self, data: pd.DataFrame,
                                       cv_type: str = 'stratified') -> Dict[str, Any]:
        """
        设计交叉验证策略
        根据数据特性选择最适合的交叉验证方法
        """
        n_samples = len(data)
        results = {}

        # 根据样本量选择折数
        if n_samples < 100:
            n_folds = 5
        elif n_samples < 1000:
            n_folds = 10
        else:
            n_folds = 10  # 标准10折

        if cv_type == 'stratified':
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            results['cv_type'] = 'StratifiedKFold'

        elif cv_type == 'time_series':
            cv = TimeSeriesSplit(n_splits=n_folds)
            results['cv_type'] = 'TimeSeriesSplit'

        elif cv_type == 'group':
            # 需要group信息
            if 'group' in data.columns:
                cv = GroupKFold(n_splits=n_folds)
                results['cv_type'] = 'GroupKFold'
            else:
                cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
                results['cv_type'] = 'StratifiedKFold (fallback)'

        else:
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            results['cv_type'] = 'StratifiedKFold (default)'

        results['n_folds'] = n_folds
        results['cv_object'] = cv

        # 验证交叉验证的合理性
        results['validation_assessment'] = self._assess_cv_validity(data, cv)

        return results

    def design_temporal_validation(self, data: pd.DataFrame,
                                 time_column: str,
                                 validation_periods: int = 3) -> Dict[str, Any]:
        """
        时间序列验证设计
        对于包含时间信息的数据，设计时间上的验证策略
        """
        results = {}

        # 确保时间列是datetime类型
        if not pd.api.types.is_datetime64_any_dtype(data[time_column]):
            data[time_column] = pd.to_datetime(data[time_column])

        # 按时间排序
        data_sorted = data.sort_values(time_column)

        # 计算时间范围
        time_span = data_sorted[time_column].max() - data_sorted[time_column].min()
        period_length = time_span / validation_periods

        # 创建时间段
        periods = []
        start_time = data_sorted[time_column].min()

        for i in range(validation_periods):
            end_time = start_time + period_length
            period_data = data_sorted[
                (data_sorted[time_column] >= start_time) &
                (data_sorted[time_column] < end_time)
            ]

            periods.append({
                'period': i + 1,
                'start_time': start_time,
                'end_time': end_time,
                'n_samples': len(period_data),
                'indices': period_data.index.tolist()
            })

            start_time = end_time

        results['temporal_periods'] = periods
        results['validation_strategy'] = 'forward_chaining'

        # 前向链式验证
        cv_splits = []
        for i in range(1, validation_periods):
            train_indices = []
            for j in range(i):
                train_indices.extend(periods[j]['indices'])
            test_indices = periods[i]['indices']

            cv_splits.append({
                'fold': i,
                'train_indices': train_indices,
                'test_indices': test_indices,
                'train_size': len(train_indices),
                'test_size': len(test_indices)
            })

        results['cv_splits'] = cv_splits
        results['temporal_gap_analysis'] = self._analyze_temporal_gaps(data_sorted, time_column)

        return results

    def design_external_validation(self, internal_data: pd.DataFrame,
                                 external_sources: List[str]) -> Dict[str, Any]:
        """
        外部验证设计
        设计多中心或多数据源的外部验证策略
        """
        results = {}

        # 内部验证集特征
        internal_features = {
            'sample_size': len(internal_data),
            'feature_names': list(internal_data.columns),
            'data_types': internal_data.dtypes.to_dict(),
            'missing_rates': (internal_data.isnull().sum() / len(internal_data)).to_dict()
        }

        results['internal_validation'] = internal_features

        # 外部验证要求
        external_requirements = {
            'minimum_sample_size': max(100, len(internal_data) * 0.2),
            'required_features': internal_features['feature_names'],
            'acceptable_missing_rate': 0.2,
            'temporal_requirements': {
                'minimum_follow_up': '6 months',
                'outcome_assessment': 'standardized'
            }
        }

        results['external_validation_requirements'] = external_requirements

        # 验证计划
        validation_plan = []
        for i, source in enumerate(external_sources):
            plan = {
                'source': source,
                'validation_type': 'prospective' if i == 0 else 'retrospective',
                'sample_size_target': external_requirements['minimum_sample_size'],
                'timeline': f"Phase {i+1}",
                'success_criteria': {
                    'auc_threshold': 0.7,
                    'calibration_p_value': 0.05,
                    'discrimination_slope': 0.1
                }
            }
            validation_plan.append(plan)

        results['validation_plan'] = validation_plan

        return results

    def sensitivity_analysis_design(self, analysis_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        敏感性分析设计
        设计针对关键假设和参数的敏感性分析
        """
        results = {}

        # 参数敏感性分析
        sensitivity_scenarios = []

        # 缺失数据处理敏感性
        missing_data_scenarios = [
            {'method': 'complete_case', 'description': '完整病例分析'},
            {'method': 'mean_imputation', 'description': '均值填充'},
            {'method': 'multiple_imputation', 'description': '多重填充'},
            {'method': 'worst_case', 'description': '最坏情况'},
            {'method': 'best_case', 'description': '最好情况'}
        ]
        sensitivity_scenarios.extend(missing_data_scenarios)

        # 阈值敏感性
        if 'classification_threshold' in analysis_parameters:
            threshold_scenarios = []
            base_threshold = analysis_parameters['classification_threshold']
            for delta in [-0.1, -0.05, 0.05, 0.1]:
                threshold_scenarios.append({
                    'method': f'threshold_{base_threshold + delta}',
                    'description': f'分类阈值 {base_threshold + delta}'
                })
            sensitivity_scenarios.extend(threshold_scenarios)

        # 特征选择敏感性
        feature_scenarios = [
            {'method': 'all_features', 'description': '所有特征'},
            {'method': 'univariate_selection', 'description': '单变量特征选择'},
            {'method': 'recursive_elimination', 'description': '递归特征消除'},
            {'method': 'lasso_selection', 'description': 'LASSO特征选择'}
        ]
        sensitivity_scenarios.extend(feature_scenarios)

        results['sensitivity_scenarios'] = sensitivity_scenarios

        # 稳健性检验
        robustness_tests = [
            {
                'test': 'outlier_removal',
                'description': '移除异常值后的结果',
                'method': 'IQR_method'
            },
            {
                'test': 'bootstrap_validation',
                'description': 'Bootstrap重采样验证',
                'n_bootstrap': 1000
            },
            {
                'test': 'subgroup_analysis',
                'description': '亚组分析',
                'subgroups': ['age_group', 'gender', 'comorbidity']
            }
        ]

        results['robustness_tests'] = robustness_tests

        return results

    def bias_analysis_design(self) -> Dict[str, Any]:
        """
        偏差分析设计
        识别和量化潜在的偏差来源
        """
        results = {}

        # 选择偏差分析
        selection_bias = {
            'description': '选择偏差评估',
            'methods': [
                'propensity_score_matching',
                'inverse_probability_weighting',
                'stratification_analysis'
            ],
            'sensitivity_analysis': 'unmeasured_confounding'
        }

        # 信息偏差分析
        information_bias = {
            'description': '信息偏差评估',
            'sources': [
                'measurement_error',
                'recall_bias',
                'observer_bias',
                'misclassification'
            ],
            'quantification_methods': [
                'validation_substudy',
                'probabilistic_bias_analysis'
            ]
        }

        # 混杂偏差分析
        confounding_bias = {
            'description': '混杂偏差评估',
            'methods': [
                'directed_acyclic_graphs',
                'negative_control_outcomes',
                'instrumental_variables'
            ],
            'unmeasured_confounding': {
                'e_value_analysis': True,
                'sensitivity_scenarios': True
            }
        }

        results['selection_bias'] = selection_bias
        results['information_bias'] = information_bias
        results['confounding_bias'] = confounding_bias

        # 偏差量化指标
        bias_metrics = [
            {'metric': 'e_value', 'description': 'E值分析'},
            {'metric': 'bias_factor', 'description': '偏差因子'},
            {'metric': 'confounding_strength', 'description': '混杂强度'},
            {'metric': 'measurement_error_variance', 'description': '测量误差方差'}
        ]

        results['bias_quantification_metrics'] = bias_metrics

        return results

    # 辅助方法
    def _adjust_sample_size_for_attrition(self, sample_size_results: Dict[str, Any],
                                        attrition_rate: float = 0.1) -> Dict[str, Any]:
        """调整样本量以考虑脱落率"""
        adjusted_results = {}

        for test_type, results in sample_size_results.items():
            adjusted_results[test_type] = results.copy()

            if 'total_sample_size' in results:
                original_n = results['total_sample_size']
                adjusted_n = int(np.ceil(original_n / (1 - attrition_rate)))
                adjusted_results[test_type]['adjusted_total_sample_size'] = adjusted_n
                adjusted_results[test_type]['attrition_rate'] = attrition_rate

            if 'sample_size' in results:
                original_n = results['sample_size']
                adjusted_n = int(np.ceil(original_n / (1 - attrition_rate)))
                adjusted_results[test_type]['adjusted_sample_size'] = adjusted_n
                adjusted_results[test_type]['attrition_rate'] = attrition_rate

        return adjusted_results

    def _stratified_split(self, data: pd.DataFrame, target_column: str,
                         stratification_columns: List[str], test_size: float = 0.2):
        """分层分割数据"""
        if stratification_columns:
            # 创建分层变量
            stratify_var = data[stratification_columns].apply(
                lambda x: '_'.join(x.astype(str)), axis=1
            )
        else:
            stratify_var = data[target_column]

        train_idx, test_idx = train_test_split(
            data.index,
            test_size=test_size,
            stratify=stratify_var,
            random_state=42
        )

        return train_idx.tolist(), test_idx.tolist()

    def _validate_stratification_balance(self, data: pd.DataFrame,
                                       train_indices: List[int],
                                       test_indices: List[int],
                                       stratification_columns: List[str]):
        """验证分层平衡性"""
        balance_results = {}

        train_data = data.loc[train_indices]
        test_data = data.loc[test_indices]

        for col in stratification_columns:
            if data[col].dtype in ['object', 'category']:
                # 分类变量 - 卡方检验
                train_counts = train_data[col].value_counts()
                test_counts = test_data[col].value_counts()

                # 确保两个序列有相同的索引
                all_categories = set(train_counts.index) | set(test_counts.index)
                train_counts = train_counts.reindex(all_categories, fill_value=0)
                test_counts = test_counts.reindex(all_categories, fill_value=0)

                chi2, p_value = stats.chisquare(test_counts, train_counts)

                balance_results[col] = {
                    'type': 'categorical',
                    'chi2_statistic': chi2,
                    'p_value': p_value,
                    'balanced': p_value > 0.05
                }
            else:
                # 连续变量 - t检验
                t_stat, p_value = stats.ttest_ind(
                    train_data[col].dropna(),
                    test_data[col].dropna()
                )

                balance_results[col] = {
                    'type': 'continuous',
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'balanced': p_value > 0.05,
                    'train_mean': train_data[col].mean(),
                    'test_mean': test_data[col].mean()
                }

        return balance_results

    def _assess_cv_validity(self, data: pd.DataFrame, cv):
        """评估交叉验证的有效性"""
        assessment = {}

        # 检查每折的样本量
        fold_sizes = []
        for train_idx, test_idx in cv.split(data, data.iloc[:, 0]):
            fold_sizes.append({
                'train_size': len(train_idx),
                'test_size': len(test_idx)
            })

        assessment['fold_sizes'] = fold_sizes
        assessment['min_test_size'] = min([f['test_size'] for f in fold_sizes])
        assessment['max_test_size'] = max([f['test_size'] for f in fold_sizes])
        assessment['cv_validity'] = assessment['min_test_size'] >= 10  # 最小测试集要求

        return assessment

    def _analyze_temporal_gaps(self, data: pd.DataFrame, time_column: str):
        """分析时间间隔"""
        time_diffs = data[time_column].diff().dropna()

        analysis = {
            'mean_gap': time_diffs.mean(),
            'median_gap': time_diffs.median(),
            'min_gap': time_diffs.min(),
            'max_gap': time_diffs.max(),
            'gap_std': time_diffs.std(),
            'irregular_gaps': (time_diffs > time_diffs.quantile(0.95)).sum()
        }

        return analysis

# 使用示例
if __name__ == "__main__":
    # 创建示例配置
    config = ExperimentalDesignConfig(
        study_name="多模态健康监测研究",
        primary_endpoint="健康评分",
        secondary_endpoints=["血糖预测准确性", "图像识别准确性"],
        effect_size=0.5,
        power=0.8,
        stratification_factors=["age_group", "gender", "comorbidity"]
    )

    # 创建实验设计实例
    designer = RigorousExperimentalDesign(config)

    # 样本量计算
    sample_size_results = designer.calculate_sample_size('two_sample_ttest')
    print("样本量计算结果:", sample_size_results)

    # 创建示例数据进行演示
    np.random.seed(42)
    data = pd.DataFrame({
        'age_group': np.random.choice(['young', 'middle', 'old'], 1000),
        'gender': np.random.choice(['M', 'F'], 1000),
        'comorbidity': np.random.choice([0, 1], 1000),
        'outcome': np.random.normal(0, 1, 1000)
    })

    # 分层抽样设计
    stratified_design = designer.design_stratified_sampling(
        data, 'outcome', ['age_group', 'gender']
    )
    print("分层抽样结果:", stratified_design['stratification_balance'])

__all__ = ["'ExperimentalDesignConfig'", "'RigorousExperimentalDesign'"]
