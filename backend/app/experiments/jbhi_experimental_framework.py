
"""
AIIM标准实验框架
符合Artificial Intelligence in Medicine (AIIM)期刊的实验标准
支持J-BHI兼容模式（向后兼容）
"""

import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
)
import logging
from collections import defaultdict

from app.experiments.statistical_methods import (
    StatisticalMethods,
    SampleSizeCalculation,
    NormalityTestResult,
    VarianceTestResult,
    StatisticalComparisonResult
)

logger = logging.getLogger(__name__)


@dataclass
class BaselineComparison:
    """基线对比结果"""
    baseline_name: str
    metrics: Dict[str, float]
    comparison_result: Optional[StatisticalComparisonResult] = None


@dataclass
class AblationStudy:
    """消融实验结果"""
    component_name: str
    removed: bool
    metrics: Dict[str, float]
    comparison_result: Optional[StatisticalComparisonResult] = None


@dataclass
class CrossValidationResult:
    """交叉验证结果"""
    fold: int
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]


@dataclass
class FailureCase:
    """失败案例分析（AIIM要求）"""
    case_id: str
    true_value: float
    predicted_value: float
    error: float
    data_features: Dict[str, Any]
    model_behavior: Optional[Dict[str, Any]] = None
    expert_analysis: Optional[str] = None


@dataclass
class ExperimentalResults:
    """实验结果汇总"""
    sample_size_calculation: Optional[SampleSizeCalculation] = None
    normality_tests: Dict[str, NormalityTestResult] = field(default_factory=dict)
    variance_tests: Dict[str, VarianceTestResult] = field(default_factory=dict)
    baseline_comparisons: List[BaselineComparison] = field(default_factory=list)
    ablation_studies: List[AblationStudy] = field(default_factory=list)
    cross_validation_results: List[CrossValidationResult] = field(default_factory=list)
    overall_metrics: Dict[str, float] = field(default_factory=dict)
    statistical_comparisons: List[StatisticalComparisonResult] = field(default_factory=list)
    failure_cases: List[FailureCase] = field(default_factory=list)
    robustness_results: Optional[Dict[str, Any]] = None
    explainability_results: Optional[Dict[str, Any]] = None
    distribution_shift_results: Optional[Dict[str, Any]] = None
    detailed_failure_analysis: Optional[Dict[str, Any]] = None


def _is_continuous_distribution(values: np.ndarray, threshold: float = 0.1) -> bool:
    """
    判断值是否为连续分布

    Args:
        values: 待判断的值数组
        threshold: 判断阈值，如果相邻唯一值的最小间隔小于阈值，认为是连续分布

    Returns:
        bool: 是否为连续分布
    """
    if len(values) < 2:
        return False

    unique_vals = np.sort(np.unique(values))
    if len(unique_vals) < 2:
        return False

    # 计算相邻唯一值的最小间隔
    min_gap = np.min(np.diff(unique_vals))

    # 如果最小间隔很小，认为是连续分布
    return min_gap < threshold


class JBHIExperimentalFramework:
    """
    AIIM标准实验框架（兼容J-BHI）

    AIIM期刊要求：
    - 统计功效 (power) ≥ 0.90（默认0.90，J-BHI兼容模式0.8）
    - 样本量基于临床意义计算
    - 随机化协议（分层随机、盲法评估）
    - 完整的基线对比（5+基线方法）
    - 深度消融实验（分层消融、失败案例分析）
    - 可解释性验证（SHAP等）
    - 鲁棒性测试（噪声测试、分布偏移）
    """

    def __init__(
        self,
        alpha: float = 0.05,
        power: float = 0.90,
        n_folds: int = 5,
        random_seed: int = 42,
        journal_standard: str = "AIIM"
    ):
        """
        初始化J-BHI实验框架

        Args:
            alpha: 显著性水平，默认0.05
            power: 统计功效（1-β），默认0.8
            n_folds: 交叉验证折数，默认5
            random_seed: 随机种子，默认42
        """
        self.statistical_methods = StatisticalMethods(alpha=alpha, power=power)
        self.alpha = alpha
        self.power = power
        self.n_folds = n_folds
        self.random_seed = random_seed
        self.journal_standard = journal_standard
        self.results = ExperimentalResults()

        # AIIM标准检查
        if journal_standard == "AIIM" and power < 0.90:
            logger.warning(
                f"⚠️ AIIM期刊要求统计功效≥0.90，当前power={power}。"
                f"建议将power设置为0.90以确保符合AIIM标准。"
            )

    def calculate_sample_size(
        self,
        effect_size: float,
        description: str = ""
    ) -> SampleSizeCalculation:
        """
        计算样本量

        Args:
            effect_size: 效应量（Cohen's d）
            description: 效应量确定依据的描述

        Returns:
            SampleSizeCalculation: 样本量计算结果
        """
        result = self.statistical_methods.calculate_sample_size(effect_size)
        self.results.sample_size_calculation = result

        logger.info(f"样本量计算完成: n={result.n}, α={result.alpha}, 1-β={result.power}, d={result.effect_size}")
        if description:
            logger.info(f"效应量确定依据: {description}")

        return result

    def test_data_normality(
        self,
        data: Union[List[float], np.ndarray],
        data_name: str,
        method: str = "shapiro-wilk"
    ) -> NormalityTestResult:
        """
        检验数据正态性

        Args:
            data: 待检验数据
            data_name: 数据名称
            method: 检验方法（"shapiro-wilk" 或 "jarque-bera"）

        Returns:
            NormalityTestResult: 正态性检验结果
        """
        result = self.statistical_methods.test_normality(data, method=method)
        self.results.normality_tests[data_name] = result

        logger.info(f"正态性检验 ({data_name}): {result.method}, p={result.p_value:.4f}, "
                   f"is_normal={result.is_normal}")

        return result

    def test_variance_homogeneity(
        self,
        groups: Dict[str, Union[List[float], np.ndarray]],
        method: str = "levene"
    ) -> VarianceTestResult:
        """
        检验方差齐性

        Args:
            groups: 数据组字典
            method: 检验方法（"levene" 或 "bartlett"）

        Returns:
            VarianceTestResult: 方差齐性检验结果
        """
        group_list = list(groups.values())
        result = self.statistical_methods.test_variance_homogeneity(*group_list, method=method)

        test_name = "_vs_".join(groups.keys())
        self.results.variance_tests[test_name] = result

        logger.info(f"方差齐性检验 ({test_name}): {result.method}, p={result.p_value:.4f}, "
                   f"is_equal_variance={result.is_equal_variance}")

        return result

    def run_baseline_comparison(
        self,
        baseline_method: callable,
        proposed_method: callable,
        test_data: List[Dict[str, Any]],
        baseline_name: str = "Baseline",
        proposed_name: str = "Proposed"
    ) -> BaselineComparison:
        """
        运行基线对比实验

        Args:
            baseline_method: 基线方法（接受test_data，返回预测结果）
            proposed_method: 提出的方法（接受test_data，返回预测结果）
            test_data: 测试数据
            baseline_name: 基线方法名称
            proposed_name: 提出的方法名称

        Returns:
            BaselineComparison: 基线对比结果
        """
        logger.info(f"开始基线对比实验: {baseline_name} vs {proposed_name}")

        # 运行基线方法
        baseline_predictions = baseline_method(test_data)
        baseline_metrics = self._calculate_metrics(test_data, baseline_predictions)

        # 运行提出的方法
        proposed_predictions = proposed_method(test_data)
        proposed_metrics = self._calculate_metrics(test_data, proposed_predictions)

        # 统计比较
        comparison_result = None
        if 'scores' in baseline_metrics and 'scores' in proposed_metrics:
            comparison_result = self.statistical_methods.compare_groups(
                baseline_metrics['scores'],
                proposed_metrics['scores'],
                paired=False
            )
            self.results.statistical_comparisons.append(comparison_result)

        baseline_comparison = BaselineComparison(
            baseline_name=baseline_name,
            metrics=baseline_metrics,
            comparison_result=comparison_result
        )

        self.results.baseline_comparisons.append(baseline_comparison)

        logger.info(f"基线对比完成: {baseline_name} vs {proposed_name}")
        if comparison_result:
            # 格式化 p-value 显示（避免显示为 0.0000）
            p_val = comparison_result.p_value
            if p_val < 0.0001:
                p_value_display = f"{p_val:.2e}"  # 科学计数法
                p_value_note = "（p值极小，可能因样本量过大或差异极大）"
            else:
                p_value_display = f"{p_val:.4f}"
                p_value_note = ""

            logger.info(f"统计显著性: p={p_value_display}{p_value_note}, "
                       f"is_significant={comparison_result.is_significant}")

            # 输出效应量信息（如果可用）
            if comparison_result.effect_size is not None:
                effect_size = comparison_result.effect_size
                if comparison_result.effect_size_ci:
                    logger.info(f"效应量 (Cohen's d): {effect_size:.4f} [{comparison_result.effect_size_ci[0]:.4f}, {comparison_result.effect_size_ci[1]:.4f}]")
                else:
                    logger.info(f"效应量 (Cohen's d): {effect_size:.4f}")
                if abs(effect_size) < 0.2:
                    logger.warning(f"⚠️ 效应量较小 (|d|={abs(effect_size):.4f} < 0.2)，差异的实际意义可能有限")

            # 输出Cliff's delta（如果可用）
            if comparison_result.cliffs_delta is not None:
                cliffs_delta = comparison_result.cliffs_delta
                logger.info(f"效应量 (Cliff's delta): {cliffs_delta:.4f}")

            # 输出临床意义信息（如果可用）
            if comparison_result.clinical_significance:
                clinical = comparison_result.clinical_significance
                logger.info(f"临床意义评估:")
                logger.info(f"  - 临床重要差异比例 (>{clinical.get('mcid', 0.03):.3f}): {clinical.get('proportion_clinically_important', 0):.1%}")
                logger.info(f"  - 决策改变率: {clinical.get('decision_change_rate', 0):.1%}")
                logger.info(f"  - 患者获益指数: {clinical.get('patient_benefit_index', 0):.3f}")

            # 输出NNT信息（如果可用）
            if comparison_result.nnt is not None:
                if comparison_result.nnt_ci:
                    logger.info(f"NNT (Number Needed to Treat): {comparison_result.nnt:.2f} [{comparison_result.nnt_ci[0]:.2f}, {comparison_result.nnt_ci[1]:.2f}]")
                else:
                    logger.info(f"NNT (Number Needed to Treat): {comparison_result.nnt:.2f}")
                if comparison_result.nnt < 5:
                    logger.info(f"✅ NNT较小 ({comparison_result.nnt:.2f} < 5)，临床获益显著")
                elif comparison_result.nnt < 10:
                    logger.info(f"✅ NNT中等 ({comparison_result.nnt:.2f} < 10)，临床获益可接受")
                elif comparison_result.nnt < 20:
                    logger.warning(f"⚠️ NNT较大 ({comparison_result.nnt:.2f} < 20)，临床获益有限")
                else:
                    logger.warning(f"⚠️ NNT很大 ({comparison_result.nnt:.2f} ≥ 20)，临床获益可能不显著")

        return baseline_comparison

    def run_ablation_study(
        self,
        full_method: callable,
        ablated_method: callable,
        test_data: List[Dict[str, Any]],
        component_name: str,
        removed: bool = True
    ) -> AblationStudy:
        """
        运行消融实验

        Args:
            full_method: 完整方法（接受test_data，返回预测结果）
            ablated_method: 消融方法（移除某个组件，接受test_data，返回预测结果）
            test_data: 测试数据
            component_name: 组件名称
            removed: 是否移除该组件

        Returns:
            AblationStudy: 消融实验结果
        """
        logger.info(f"开始消融实验: 移除组件 '{component_name}'")

        # 运行完整方法
        full_predictions = full_method(test_data)
        full_metrics = self._calculate_metrics(test_data, full_predictions)

        # 运行消融方法
        ablated_predictions = ablated_method(test_data)
        ablated_metrics = self._calculate_metrics(test_data, ablated_predictions)

        # 统计比较
        comparison_result = None
        if 'scores' in full_metrics and 'scores' in ablated_metrics:
            comparison_result = self.statistical_methods.compare_groups(
                full_metrics['scores'],
                ablated_metrics['scores'],
                paired=False
            )
            self.results.statistical_comparisons.append(comparison_result)

        ablation_study = AblationStudy(
            component_name=component_name,
            removed=removed,
            metrics=ablated_metrics,
            comparison_result=comparison_result
        )

        self.results.ablation_studies.append(ablation_study)

        logger.info(f"消融实验完成: 组件 '{component_name}'")
        if comparison_result:
            # 格式化 p-value 显示（避免显示为 0.0000）
            p_val = comparison_result.p_value
            if p_val < 0.0001:
                p_value_display = f"{p_val:.2e}"  # 科学计数法
                p_value_note = "（p值极小，可能因样本量过大或差异极大）"
            else:
                p_value_display = f"{p_val:.4f}"
                p_value_note = ""

            logger.info(f"统计显著性: p={p_value_display}{p_value_note}, "
                       f"is_significant={comparison_result.is_significant}")

            # 输出效应量信息（如果可用）
            if comparison_result.effect_size is not None:
                effect_size = comparison_result.effect_size
                if comparison_result.effect_size_ci:
                    logger.info(f"效应量 (Cohen's d): {effect_size:.4f} [{comparison_result.effect_size_ci[0]:.4f}, {comparison_result.effect_size_ci[1]:.4f}]")
                else:
                    logger.info(f"效应量 (Cohen's d): {effect_size:.4f}")
                if abs(effect_size) < 0.2:
                    logger.warning(f"⚠️ 效应量较小 (|d|={abs(effect_size):.4f} < 0.2)，差异的实际意义可能有限")

            # 输出Cliff's delta（如果可用）
            if comparison_result.cliffs_delta is not None:
                cliffs_delta = comparison_result.cliffs_delta
                logger.info(f"效应量 (Cliff's delta): {cliffs_delta:.4f}")

            # 输出临床意义信息（如果可用）
            if comparison_result.clinical_significance:
                clinical = comparison_result.clinical_significance
                logger.info(f"临床意义评估:")
                logger.info(f"  - 临床重要差异比例 (>{clinical.get('mcid', 0.03):.3f}): {clinical.get('proportion_clinically_important', 0):.1%}")
                logger.info(f"  - 决策改变率: {clinical.get('decision_change_rate', 0):.1%}")
                logger.info(f"  - 患者获益指数: {clinical.get('patient_benefit_index', 0):.3f}")

            # 输出NNT信息（如果可用）
            if comparison_result.nnt is not None:
                if comparison_result.nnt_ci:
                    logger.info(f"NNT (Number Needed to Treat): {comparison_result.nnt:.2f} [{comparison_result.nnt_ci[0]:.2f}, {comparison_result.nnt_ci[1]:.2f}]")
                else:
                    logger.info(f"NNT (Number Needed to Treat): {comparison_result.nnt:.2f}")
                if comparison_result.nnt < 5:
                    logger.info(f"✅ NNT较小 ({comparison_result.nnt:.2f} < 5)，临床获益显著")
                elif comparison_result.nnt < 10:
                    logger.info(f"✅ NNT中等 ({comparison_result.nnt:.2f} < 10)，临床获益可接受")
                elif comparison_result.nnt < 20:
                    logger.warning(f"⚠️ NNT较大 ({comparison_result.nnt:.2f} < 20)，临床获益有限")
                else:
                    logger.warning(f"⚠️ NNT很大 ({comparison_result.nnt:.2f} ≥ 20)，临床获益可能不显著")

        return ablation_study

    def run_cross_validation(
        self,
        method: callable,
        data: List[Dict[str, Any]],
        target_key: str = "score",
        stratified: bool = False
    ) -> List[CrossValidationResult]:
        """
        运行交叉验证

        Args:
            method: 方法（接受train_data和test_data，返回预测结果）
            data: 数据列表
            target_key: 目标键名
            stratified: 是否使用分层交叉验证

        Returns:
            List[CrossValidationResult]: 交叉验证结果列表
        """
        logger.info(f"开始{self.n_folds}折交叉验证")

        # 准备数据
        X = np.array([d.get('features', []) for d in data])
        y = np.array([d.get(target_key, 0.0) for d in data])

        # 选择交叉验证方法
        if stratified:
            cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_seed)
            splits = cv.split(X, y)
        else:
            cv = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_seed)
            splits = cv.split(X, y)

        cv_results = []

        for fold, (train_idx, test_idx) in enumerate(splits):
            logger.info(f"处理第 {fold+1}/{self.n_folds} 折")

            train_data = [data[i] for i in train_idx]
            test_data = [data[i] for i in test_idx]

            # 运行方法
            predictions = method(train_data, test_data)

            # 计算指标：method可能只返回test_data的预测，也可能返回train+test的预测
            # 根据predictions的长度判断
            if len(predictions) == len(train_data) + len(test_data):
                # 返回了训练集和测试集的预测
                train_predictions = predictions[:len(train_data)]
                test_predictions = predictions[len(train_data):]
            elif len(predictions) == len(test_data):
                # 只返回了测试集的预测（常见情况）
                train_predictions = None
                test_predictions = predictions
            else:
                # 长度不匹配，只使用测试集预测
                logger.warning(f"预测结果长度不匹配: 期望 {len(train_data) + len(test_data)} 或 {len(test_data)}, 实际 {len(predictions)}")
                train_predictions = None
                test_predictions = predictions[:len(test_data)] if len(predictions) >= len(test_data) else predictions

            # 验证预测结果长度
            if len(test_predictions) != len(test_data):
                logger.warning(f"第 {fold+1} 折预测结果长度不匹配: "
                             f"期望 {len(test_data)}, 实际 {len(test_predictions)}")

            # 计算指标
            if train_predictions is not None:
                train_metrics = self._calculate_metrics(train_data, train_predictions)
            else:
                # 如果没有训练集预测，使用默认值
                train_metrics = {
                    'mae': 0.0, 'mse': 0.0, 'rmse': 0.0, 'r2': 0.0,
                    'correlation': 0.0, 'accuracy': 0.0
                }
            test_metrics = self._calculate_metrics(test_data, test_predictions)

            # 数据泄露检测：检查是否所有指标为0或R2为1（可能是数据泄露）
            if test_metrics.get('mae', 1) == 0.0 and test_metrics.get('r2', 0) == 1.0:
                logger.warning(
                    f"⚠️ 第 {fold+1} 折数据泄露警告：MAE=0.0, R²=1.0！"
                    f"这通常表示预测值与真实值完全相同，可能存在数据泄露。"
                    f"请检查：1) 预测函数是否直接返回标签；2) 训练/测试集是否分离；3) 真实值计算逻辑。"
                )
                # 输出前5个样本的预测值和真实值用于调试
                sample_y_true = np.array([d.get('score', d.get('target', 0.0)) for d in test_data[:5]])
                logger.warning(
                    f"前5个样本 - 真实值: {sample_y_true}, "
                    f"预测值: {test_predictions[:5] if len(test_predictions) >= 5 else test_predictions}"
                )

            cv_result = CrossValidationResult(
                fold=fold + 1,
                train_metrics=train_metrics,
                test_metrics=test_metrics
            )

            cv_results.append(cv_result)
            self.results.cross_validation_results.append(cv_result)

            # 使用更合适的指标进行日志输出
            test_mae = test_metrics.get('mae', 0)
            test_r2 = test_metrics.get('r2', 0)
            logger.info(f"第 {fold+1} 折完成: 测试 MAE={test_mae:.4f}, R²={test_r2:.4f}")

        # 计算平均指标
        avg_test_metrics = self._calculate_average_metrics(cv_results, 'test')
        self.results.overall_metrics = avg_test_metrics

        # 使用更合适的指标进行日志输出
        avg_mae = avg_test_metrics.get('mae', 0)
        avg_r2 = avg_test_metrics.get('r2', 0)
        avg_correlation = avg_test_metrics.get('correlation', 0)
        logger.info(f"交叉验证完成: 平均测试 MAE={avg_mae:.4f}, R²={avg_r2:.4f}, 相关系数={avg_correlation:.4f}")

        return cv_results

    def _calculate_metrics(
        self,
        data: List[Dict[str, Any]],
        predictions: Union[List[float], np.ndarray]
    ) -> Dict[str, float]:
        """
        计算评估指标

        Args:
            data: 数据列表
            predictions: 预测结果

        Returns:
            Dict[str, float]: 评估指标字典
        """
        # 提取真实值
        y_true = np.array([d.get('score', d.get('target', 0.0)) for d in data])
        y_pred = np.array(predictions)

        # 验证数据长度一致
        if len(y_true) != len(y_pred):
            logger.warning(f"真实值和预测值长度不匹配: {len(y_true)} vs {len(y_pred)}")
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]

        # 数据泄露检测：检查预测值和真实值是否完全相同
        if len(y_true) > 0 and len(y_pred) > 0:
            are_identical = np.allclose(y_true, y_pred, atol=1e-6)
            if are_identical:
                logger.error(
                    f"❌ 严重数据泄露：预测值与真实值完全相同！"
                    f"前5个样本 - 真实值: {y_true[:5]}, 预测值: {y_pred[:5]}"
                    f"\n可能原因："
                    f"\n1. 预测函数直接返回了数据中的score字段"
                    f"\n2. 真实值和预测值使用了相同的计算逻辑"
                    f"\n3. 训练/测试集未正确分离"
                )
            else:
                # 记录差异统计（用于验证修复效果）
                diff = np.abs(y_true - y_pred)
                mean_diff = np.mean(diff)
                max_diff = np.max(diff)
                min_diff = np.min(diff)

                # 如果差异过小（<0.001），也发出警告
                if mean_diff < 0.001:
                    logger.warning(
                        f"⚠️ 预测值与真实值差异过小（平均差异={mean_diff:.6f}），"
                        f"可能存在轻微数据泄露或评估逻辑问题"
                    )
                else:
                    logger.info(
                        f"✅ 预测值与真实值存在合理差异 - "
                        f"平均差异: {mean_diff:.6f}, "
                        f"最大差异: {max_diff:.6f}, "
                        f"最小差异: {min_diff:.6f}"
                    )

        # 基础回归指标
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)

        # R2 计算（处理边界情况）
        try:
            r2 = r2_score(y_true, y_pred)
            # 如果 R2 为 NaN 或 inf，可能是所有值相同或方差为0
            if np.isnan(r2) or np.isinf(r2):
                if np.var(y_true) == 0:
                    # 所有真实值相同，如果预测值也相同且等于真实值，R2=1
                    r2 = 1.0 if np.allclose(y_true, y_pred) else 0.0
                else:
                    r2 = 0.0
        except Exception as e:
            logger.warning(f"R2 计算出错: {e}")
            r2 = 0.0

        metrics = {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'r2': float(r2),
            'scores': y_pred.tolist()
        }

        # 计算相关系数（更稳健的相似度指标）
        try:
            if len(y_true) > 1 and np.var(y_true) > 0:
                correlation = np.corrcoef(y_true, y_pred)[0, 1]
                metrics['correlation'] = float(correlation) if not np.isnan(correlation) else 0.0
            else:
                metrics['correlation'] = 1.0 if np.allclose(y_true, y_pred) else 0.0
        except Exception:
            metrics['correlation'] = 0.0

        # 任务类型判断：区分分类任务和回归任务
        # 分类任务特征：
        # 1. 唯一值数量少（<=20）
        # 2. 值为离散整数（或接近整数）
        # 3. 值域在合理范围内（0-1或0-N的整数）
        is_classification = False
        unique_values = np.unique(y_true)

        if len(unique_values) <= 20 and len(y_true) > 0:
            # 检查是否为离散整数（分类任务的特征）
            # 如果所有唯一值都是整数或接近整数（误差<0.01），且值域合理，则认为是分类任务
            is_integer_like = all(
                abs(val - round(val)) < 0.01
                for val in unique_values
            )
            # 检查值域：分类任务通常是0-1（二分类）或0-N（多分类）
            is_reasonable_range = (
                (np.min(y_true) >= 0 and np.max(y_true) <= 1) or  # 二分类概率
                (np.min(y_true) >= 0 and np.max(y_true) < 100 and is_integer_like)  # 多分类标签
            )

            # 进一步检查：如果值都是0或1，且唯一值只有2个，则明确是二分类
            if len(unique_values) == 2 and set(unique_values).issubset({0, 1}):
                is_classification = True
            # 如果值都是整数且值域合理，则可能是多分类
            elif is_integer_like and is_reasonable_range and len(unique_values) >= 2:
                is_classification = True
            # 如果值在0-1之间但唯一值很少（<=5），且不是连续分布，可能是分类概率
            elif (np.min(y_true) >= 0 and np.max(y_true) <= 1 and
                  len(unique_values) <= 5 and
                  not _is_continuous_distribution(y_true)):
                # 这种情况可能是分类任务的概率输出，但我们需要原始标签
                # 如果没有原始标签，则按回归任务处理
                is_classification = False

        # 分类指标（仅当明确判断为分类任务时计算）
        if is_classification:
            try:
                # 对于二分类，使用原始值（0/1）
                if len(unique_values) == 2 and set(unique_values).issubset({0, 1}):
                    y_pred_binary = np.round(np.clip(y_pred, 0, 1)).astype(int)
                    y_true_binary = y_true.astype(int)
                else:
                    # 对于多分类，需要将预测值转换为类别标签
                    # 如果预测值是概率，取最大概率对应的类别
                    if np.max(y_pred) <= 1.0 and np.min(y_pred) >= 0.0:
                        # 可能是概率输出，需要转换为类别
                        # 这里假设预测值已经是类别标签
                        y_pred_binary = np.round(np.clip(y_pred, 0, len(unique_values) - 1)).astype(int)
                        y_true_binary = y_true.astype(int)
                    else:
                        # 预测值不在0-1范围，可能是类别标签
                        y_pred_binary = np.round(y_pred).astype(int)
                        y_true_binary = y_true.astype(int)

                # 确保预测值在有效范围内
                y_pred_binary = np.clip(y_pred_binary, int(np.min(y_true_binary)), int(np.max(y_true_binary)))

                metrics['accuracy'] = float(accuracy_score(y_true_binary, y_pred_binary))

                # 对于二分类，计算precision, recall, f1
                if len(unique_values) == 2:
                    metrics['precision'] = float(precision_score(y_true_binary, y_pred_binary, zero_division=0))
                    metrics['recall'] = float(recall_score(y_true_binary, y_pred_binary, zero_division=0))
                    metrics['f1'] = float(f1_score(y_true_binary, y_pred_binary, zero_division=0))

                    # AUC 计算（需要至少两个不同的类别）
                    if len(np.unique(y_true_binary)) > 1:
                        try:
                            # 对于二分类，使用原始预测值（概率）计算AUC
                            metrics['auc'] = float(roc_auc_score(y_true_binary, y_pred))
                        except Exception:
                            metrics['auc'] = None
                    else:
                        metrics['auc'] = None
                else:
                    # 多分类：使用macro平均
                    metrics['precision'] = float(precision_score(y_true_binary, y_pred_binary, average='macro', zero_division=0))
                    metrics['recall'] = float(recall_score(y_true_binary, y_pred_binary, average='macro', zero_division=0))
                    metrics['f1'] = float(f1_score(y_true_binary, y_pred_binary, average='macro', zero_division=0))
                    metrics['auc'] = None  # 多分类AUC需要特殊处理

            except Exception as e:
                logger.warning(f"分类指标计算出错: {e}")
                metrics['accuracy'] = None
                metrics['precision'] = None
                metrics['recall'] = None
                metrics['f1'] = None
                metrics['auc'] = None
        else:
            # 回归任务：不计算分类指标
            # 保持metrics中只有回归指标，避免混淆
            pass

        return metrics

    def _calculate_average_metrics(
        self,
        cv_results: List[CrossValidationResult],
        metric_type: str = 'test'
    ) -> Dict[str, float]:
        """
        计算平均指标

        Args:
            cv_results: 交叉验证结果列表
            metric_type: 指标类型（'train' 或 'test'）

        Returns:
            Dict[str, float]: 平均指标字典
        """
        all_metrics = defaultdict(list)

        for result in cv_results:
            metrics = result.train_metrics if metric_type == 'train' else result.test_metrics
            for key, value in metrics.items():
                if value is not None and not isinstance(value, list):
                    all_metrics[key].append(value)

        avg_metrics = {
            key: np.mean(values) if values else None
            for key, values in all_metrics.items()
        }

        return avg_metrics

    def generate_report(self) -> Dict[str, Any]:
        """
        生成实验报告

        Returns:
            Dict[str, Any]: 实验报告字典
        """
        report = {
            'sample_size': self.results.sample_size_calculation.__dict__ if self.results.sample_size_calculation else None,
            'normality_tests': {
                name: result.__dict__ for name, result in self.results.normality_tests.items()
            },
            'variance_tests': {
                name: result.__dict__ for name, result in self.results.variance_tests.items()
            },
            'baseline_comparisons': [
                {
                    'baseline_name': comp.baseline_name,
                    'metrics': comp.metrics,
                    'comparison': comp.comparison_result.__dict__ if comp.comparison_result else None
                }
                for comp in self.results.baseline_comparisons
            ],
            'ablation_studies': [
                {
                    'component_name': study.component_name,
                    'removed': study.removed,
                    'metrics': study.metrics,
                    'comparison': study.comparison_result.__dict__ if study.comparison_result else None
                }
                for study in self.results.ablation_studies
            ],
            'cross_validation': {
                'n_folds': self.n_folds,
                'fold_results': [
                    {
                        'fold': result.fold,
                        'train_metrics': result.train_metrics,
                        'test_metrics': result.test_metrics
                    }
                    for result in self.results.cross_validation_results
                ],
                'average_metrics': self.results.overall_metrics
            },
            'statistical_comparisons': [
                comp.__dict__ for comp in self.results.statistical_comparisons
            ],
            'failure_cases': [
                {
                    'case_id': case.case_id,
                    'true_value': case.true_value,
                    'predicted_value': case.predicted_value,
                    'error': case.error,
                    'data_features': case.data_features,
                    'model_behavior': case.model_behavior,
                    'expert_analysis': case.expert_analysis
                }
                for case in self.results.failure_cases
            ],
            'robustness_results': self.results.robustness_results,
            'explainability_results': self.results.explainability_results,
            'distribution_shift_results': self.results.distribution_shift_results,
            'detailed_failure_analysis': self.results.detailed_failure_analysis
        }

        return report

    def analyze_failure_cases(
        self,
        data: List[Dict[str, Any]],
        predictions: List[float],
        error_threshold: float = 0.1,
        max_cases: int = 10
    ) -> List[FailureCase]:
        """
        分析失败案例（AIIM要求）

        Args:
            data: 数据列表
            predictions: 预测结果
            error_threshold: 失败阈值（预测误差>此值视为失败），默认0.1
            max_cases: 最大返回案例数，默认10

        Returns:
            List[FailureCase]: 失败案例列表
        """
        logger.info(f"开始失败案例分析（阈值={error_threshold}）")

        failure_cases = []
        y_true = np.array([d.get('score', d.get('target', 0.0)) for d in data])
        y_pred = np.array(predictions)

        errors = np.abs(y_true - y_pred)
        failure_indices = np.where(errors > error_threshold)[0]

        # 按误差大小排序，选择最严重的失败案例
        sorted_indices = failure_indices[np.argsort(errors[failure_indices])[::-1]]
        selected_indices = sorted_indices[:max_cases]

        for idx in selected_indices:
            case = FailureCase(
                case_id=f"failure_case_{idx}",
                true_value=float(y_true[idx]),
                predicted_value=float(y_pred[idx]),
                error=float(errors[idx]),
                data_features={
                    'food_name': data[idx].get('food_name', ''),
                    'user_id': data[idx].get('user_id', ''),
                    'nutritional_info': data[idx].get('nutritional_info', {}),
                    'user_profile': str(data[idx].get('user_profile', ''))
                },
                model_behavior=None,
                expert_analysis=None
            )
            failure_cases.append(case)

        self.results.failure_cases.extend(failure_cases)
        logger.info(f"失败案例分析完成：发现{len(failure_indices)}个失败案例，详细分析{len(failure_cases)}个")

        return failure_cases

    def test_robustness(
        self,
        method: callable,
        test_data: List[Dict[str, Any]],
        noise_levels: List[float] = [0.0, 0.05, 0.10, 0.15, 0.20],
        noise_type: str = "gaussian"
    ) -> Dict[str, Any]:
        """
        鲁棒性测试（AIIM要求）

        Args:
            method: 预测方法
            test_data: 测试数据
            noise_levels: 噪声水平列表，默认[0.0, 0.05, 0.10, 0.15, 0.20]
            noise_type: 噪声类型，默认"gaussian"（可选"uniform", "missing"）

        Returns:
            Dict[str, Any]: 鲁棒性测试结果
        """
        logger.info(f"开始鲁棒性测试（噪声类型={noise_type}）")

        # 基准性能（无噪声）
        baseline_predictions = method(test_data)
        baseline_metrics = self._calculate_metrics(test_data, baseline_predictions)
        baseline_mae = baseline_metrics.get('mae', 0.0)

        robustness_results = {
            'baseline_mae': baseline_mae,
            'noise_tests': []
        }

        for noise_level in noise_levels:
            if noise_level == 0.0:
                continue

            # 添加噪声到测试数据
            noisy_data = self._add_noise_to_data(test_data, noise_level, noise_type)

            # 测试性能
            noisy_predictions = method(noisy_data)
            noisy_metrics = self._calculate_metrics(test_data, noisy_predictions)
            noisy_mae = noisy_metrics.get('mae', 0.0)

            # 计算性能下降率
            performance_degradation = (noisy_mae - baseline_mae) / baseline_mae if baseline_mae > 0 else 0.0

            robustness_results['noise_tests'].append({
                'noise_level': noise_level,
                'mae': noisy_mae,
                'performance_degradation': performance_degradation,
                'meets_aiim_standard': performance_degradation < 0.15  # AIIM要求<15%
            })

            logger.info(
                f"噪声水平={noise_level:.2f}: MAE={noisy_mae:.4f}, "
                f"性能下降={performance_degradation*100:.1f}%"
            )

        self.results.robustness_results = robustness_results
        logger.info("鲁棒性测试完成")

        return robustness_results

    def _add_noise_to_data(
        self,
        data: List[Dict[str, Any]],
        noise_level: float,
        noise_type: str
    ) -> List[Dict[str, Any]]:
        """
        向数据添加噪声

        Args:
            data: 原始数据
            noise_level: 噪声水平
            noise_type: 噪声类型（"gaussian", "uniform", "missing"）

        Returns:
            List[Dict[str, Any]]: 添加噪声后的数据
        """
        np.random.seed(self.random_seed)
        noisy_data = []

        for d in data:
            noisy_d = d.copy()

            if noise_type == "gaussian":
                # 高斯噪声：添加到营养信息
                if 'nutritional_info' in noisy_d:
                    nutrition = noisy_d['nutritional_info'].copy()
                    for key in ['calories', 'protein', 'carbs', 'fat']:
                        if key in nutrition:
                            nutrition[key] = max(0, nutrition[key] + np.random.normal(0, noise_level * nutrition[key]))
                    noisy_d['nutritional_info'] = nutrition

            elif noise_type == "uniform":
                # 均匀噪声
                if 'nutritional_info' in noisy_d:
                    nutrition = noisy_d['nutritional_info'].copy()
                    for key in ['calories', 'protein', 'carbs', 'fat']:
                        if key in nutrition:
                            noise = np.random.uniform(-noise_level, noise_level) * nutrition[key]
                            nutrition[key] = max(0, nutrition[key] + noise)
                    noisy_d['nutritional_info'] = nutrition

            elif noise_type == "missing":
                # 缺失数据：随机删除部分特征
                if 'nutritional_info' in noisy_d:
                    nutrition = noisy_d['nutritional_info'].copy()
                    for key in nutrition:
                        if np.random.random() < noise_level:
                            nutrition[key] = None
                    noisy_d['nutritional_info'] = nutrition

            noisy_data.append(noisy_d)

        return noisy_data

    def stratified_randomization(
        self,
        data: List[Dict[str, Any]],
        stratify_by: List[str] = ['region', 'diabetes_severity'],
        test_ratio: float = 0.2
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        分层随机化（AIIM要求）

        Args:
            data: 数据列表
            stratify_by: 分层变量列表，默认['region', 'diabetes_severity']
            test_ratio: 测试集比例，默认0.2

        Returns:
            Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: (训练集, 测试集)
        """
        logger.info(f"开始分层随机化（分层变量={stratify_by}）")

        np.random.seed(self.random_seed)

        # 按分层变量分组
        stratified_groups = defaultdict(list)
        for d in data:
            key = tuple(d.get(var, 'unknown') for var in stratify_by)
            stratified_groups[key].append(d)

        train_data = []
        test_data = []

        # 每组内随机划分
        for group_key, group_data in stratified_groups.items():
            np.random.shuffle(group_data)
            n_test = int(len(group_data) * test_ratio)
            test_data.extend(group_data[:n_test])
            train_data.extend(group_data[n_test:])

        logger.info(f"分层随机化完成：训练集={len(train_data)}, 测试集={len(test_data)}")

        return train_data, test_data

    def analyze_explainability(
        self,
        method: callable,
        test_data: List[Dict[str, Any]],
        sample_size: int = 50,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        可解释性分析（AIIM要求）

        Args:
            method: 预测方法
            test_data: 测试数据
            sample_size: 分析样本数，默认50
            feature_names: 特征名称列表，如果为None则自动提取

        Returns:
            Dict[str, Any]: 可解释性分析结果
        """
        logger.info(f"开始可解释性分析（样本数={sample_size}）")

        try:
            import shap
            SHAP_AVAILABLE = True
        except ImportError:
            SHAP_AVAILABLE = False
            logger.warning("SHAP库不可用，将使用简化版本的可解释性分析")

        # 选择代表性样本
        np.random.seed(self.random_seed)
        if len(test_data) > sample_size:
            selected_indices = np.random.choice(len(test_data), size=sample_size, replace=False)
            selected_data = [test_data[i] for i in selected_indices]
        else:
            selected_data = test_data

        # 提取特征（简化版本：基于营养信息和文化特征）
        if feature_names is None:
            feature_names = [
                'calories', 'protein', 'carbs', 'fat',
                'cultural_match', 'spice_tolerance', 'sweet_preference'
            ]

        # 构建特征矩阵
        feature_matrix = []
        for d in selected_data:
            features = []
            nutrition = d.get('nutritional_info', {})
            features.append(nutrition.get('calories', 200))
            features.append(nutrition.get('protein', 10))
            features.append(nutrition.get('carbs', 30))
            features.append(nutrition.get('fat', 10))

            # 文化特征（简化提取）
            user_profile = d.get('user_profile')
            if user_profile:
                cultural_match = 0.5  # 简化：实际应从user_profile提取
                spice_tolerance = 0.5
                sweet_preference = 0.5
            else:
                cultural_match = 0.5
                spice_tolerance = 0.5
                sweet_preference = 0.5

            features.extend([cultural_match, spice_tolerance, sweet_preference])
            feature_matrix.append(features)

        feature_matrix = np.array(feature_matrix)

        # 创建预测函数包装器
        def predict_wrapper(X: np.ndarray) -> np.ndarray:
            """将特征矩阵转换回数据格式并预测"""
            predictions = []
            for i, x in enumerate(X):
                # 创建临时数据字典
                temp_data = selected_data[i].copy()
                temp_data['nutritional_info'] = {
                    'calories': x[0],
                    'protein': x[1],
                    'carbs': x[2],
                    'fat': x[3]
                }
                # 预测
                pred = method([temp_data])
                predictions.append(pred[0] if isinstance(pred, list) else pred)
            return np.array(predictions)

        explainability_results = {
            'shap_available': SHAP_AVAILABLE,
            'feature_names': feature_names,
            'sample_size': len(selected_data),
            'feature_importance': {},
            'shap_values': None
        }

        if SHAP_AVAILABLE:
            try:
                # 使用KernelExplainer（适用于任何模型）
                background_size = min(50, len(selected_data))
                background_data = feature_matrix[:background_size]

                explainer = shap.KernelExplainer(
                    predict_wrapper,
                    background_data
                )

                # 计算SHAP值
                shap_values = explainer.shap_values(feature_matrix, l1_reg="num_features(10)")

                # 计算特征重要性（平均绝对SHAP值）
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]

                shap_values_array = np.array(shap_values)
                feature_importance = {
                    name: float(np.mean(np.abs(shap_values_array[:, i])))
                    for i, name in enumerate(feature_names)
                }

                explainability_results['feature_importance'] = feature_importance
                explainability_results['shap_values'] = shap_values_array.tolist()
                explainability_results['expected_value'] = float(explainer.expected_value)

                logger.info("SHAP分析完成")
                logger.info(f"特征重要性排序: {sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]}")

            except Exception as e:
                logger.warning(f"SHAP分析失败: {e}，使用简化版本")
                # 降级到简化版本
                explainability_results['shap_available'] = False

        if not explainability_results.get('shap_available', False):
            # 简化版本：基于预测差异的特征重要性
            base_predictions = method(selected_data)
            feature_importance = {}

            for i, feature_name in enumerate(feature_names):
                perturbations = []
                for d in selected_data:
                    perturbed_data = d.copy()
                    if i < 4:  # 营养特征
                        nutrition = perturbed_data.get('nutritional_info', {}).copy()
                        if feature_name in nutrition:
                            nutrition[feature_name] *= 1.1  # 增加10%
                            perturbed_data['nutritional_info'] = nutrition

                    perturbed_pred = method([perturbed_data])
                    perturbations.append(perturbed_pred[0] if isinstance(perturbed_pred, list) else perturbed_pred)

                # 计算预测差异
                importance = np.mean(np.abs(np.array(perturbations) - np.array(base_predictions)))
                feature_importance[feature_name] = float(importance)

            explainability_results['feature_importance'] = feature_importance
            logger.info("简化版可解释性分析完成")
            logger.info(f"特征重要性排序: {sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]}")

        self.results.explainability_results = explainability_results
        logger.info("可解释性分析完成")

        return explainability_results

    def test_distribution_shift(
        self,
        method: callable,
        train_data: List[Dict[str, Any]],
        test_data: List[Dict[str, Any]],
        shift_type: str = "cultural"
    ) -> Dict[str, Any]:
        """
        分布偏移测试（AIIM要求）

        Args:
            method: 预测方法
            train_data: 训练数据（源分布）
            test_data: 测试数据（目标分布，可能发生偏移）
            shift_type: 偏移类型，默认"cultural"（可选"disease", "geographic"）

        Returns:
            Dict[str, Any]: 分布偏移测试结果
        """
        logger.info(f"开始分布偏移测试（偏移类型={shift_type}）")

        # 基准性能（训练分布）
        train_predictions = method(train_data)
        train_metrics = self._calculate_metrics(train_data, train_predictions)
        train_mae = train_metrics.get('mae', 0.0)
        train_r2 = train_metrics.get('r2', 0.0)

        # 目标分布性能
        test_predictions = method(test_data)
        test_metrics = self._calculate_metrics(test_data, test_predictions)
        test_mae = test_metrics.get('mae', 0.0)
        test_r2 = test_metrics.get('r2', 0.0)

        # 计算性能下降
        mae_degradation = (test_mae - train_mae) / train_mae if train_mae > 0 else 0.0
        r2_degradation = (train_r2 - test_r2) if train_r2 > 0 else 0.0

        # 分析分布差异
        distribution_analysis = self._analyze_distribution_difference(train_data, test_data, shift_type)

        shift_results = {
            'shift_type': shift_type,
            'train_metrics': {
                'mae': train_mae,
                'r2': train_r2
            },
            'test_metrics': {
                'mae': test_mae,
                'r2': test_r2
            },
            'performance_degradation': {
                'mae_degradation': mae_degradation,
                'r2_degradation': r2_degradation
            },
            'distribution_analysis': distribution_analysis,
            'meets_aiim_standard': mae_degradation < 0.20  # AIIM要求性能下降<20%
        }

        logger.info(f"分布偏移测试完成:")
        logger.info(f"  训练集 MAE={train_mae:.4f}, R²={train_r2:.4f}")
        logger.info(f"  测试集 MAE={test_mae:.4f}, R²={test_r2:.4f}")
        logger.info(f"  MAE下降率={mae_degradation*100:.1f}%")

        if mae_degradation >= 0.20:
            logger.warning(f"⚠️ 分布偏移导致性能下降≥20%，可能影响临床部署")
        else:
            logger.info(f"✅ 分布偏移下性能保持稳定（下降<20%）")

        return shift_results

    def _analyze_distribution_difference(
        self,
        train_data: List[Dict[str, Any]],
        test_data: List[Dict[str, Any]],
        shift_type: str
    ) -> Dict[str, Any]:
        """
        分析分布差异

        Args:
            train_data: 训练数据
            test_data: 测试数据
            shift_type: 偏移类型

        Returns:
            Dict[str, Any]: 分布差异分析结果
        """
        analysis = {
            'shift_type': shift_type,
            'feature_differences': {}
        }

        if shift_type == "cultural":
            # 分析文化背景分布差异
            def extract_region(data_item: Dict[str, Any]) -> str:
                """提取区域信息"""
                user_profile = data_item.get('user_profile')
                if user_profile is None:
                    return 'unknown'
                # 处理UserProfile对象或字典
                if hasattr(user_profile, 'region'):
                    return getattr(user_profile, 'region', 'unknown')
                elif isinstance(user_profile, dict):
                    return user_profile.get('region', 'unknown')
                else:
                    return 'unknown'

            train_regions = [extract_region(d) for d in train_data]
            test_regions = [extract_region(d) for d in test_data]

            from collections import Counter
            train_region_dist = Counter(train_regions)
            test_region_dist = Counter(test_regions)

            analysis['feature_differences']['region_distribution'] = {
                'train': dict(train_region_dist),
                'test': dict(test_region_dist)
            }

        # 分析营养信息分布差异
        train_calories = [d.get('nutritional_info', {}).get('calories', 200) for d in train_data]
        test_calories = [d.get('nutritional_info', {}).get('calories', 200) for d in test_data]

        analysis['feature_differences']['nutritional'] = {
            'calories': {
                'train_mean': float(np.mean(train_calories)),
                'test_mean': float(np.mean(test_calories)),
                'difference': float(np.mean(test_calories) - np.mean(train_calories))
            }
        }

        return analysis

    def analyze_failure_cases_detailed(
        self,
        data: List[Dict[str, Any]],
        predictions: List[float],
        error_threshold: float = 0.1,
        max_cases: int = 10
    ) -> Dict[str, Any]:
        """
        深度失败案例分析（AIIM要求）

        Args:
            data: 数据列表
            predictions: 预测结果
            error_threshold: 失败阈值，默认0.1
            max_cases: 最大分析案例数，默认10

        Returns:
            Dict[str, Any]: 深度失败分析结果
        """
        logger.info(f"开始深度失败案例分析（阈值={error_threshold}）")

        failure_cases = self.analyze_failure_cases(data, predictions, error_threshold, max_cases)

        # 分析失败模式
        failure_patterns = {
            'by_food_type': defaultdict(list),
            'by_cultural_background': defaultdict(list),
            'by_error_magnitude': {
                'severe': [],  # error > 0.15
                'moderate': [],  # 0.1 < error <= 0.15
                'mild': []  # 0.05 < error <= 0.1
            }
        }

        for case in failure_cases:
            food_name = case.data_features.get('food_name', 'unknown')
            failure_patterns['by_food_type'][food_name].append(case.error)

            # 按误差大小分类
            if case.error > 0.15:
                failure_patterns['by_error_magnitude']['severe'].append(case)
            elif case.error > 0.1:
                failure_patterns['by_error_magnitude']['moderate'].append(case)
            else:
                failure_patterns['by_error_magnitude']['mild'].append(case)

        # 计算失败率统计
        y_true = np.array([d.get('score', d.get('target', 0.0)) for d in data])
        y_pred = np.array(predictions)
        errors = np.abs(y_true - y_pred)

        failure_rate = np.mean(errors > error_threshold)
        severe_failure_rate = np.mean(errors > 0.15)

        detailed_analysis = {
            'total_failures': len(failure_cases),
            'failure_rate': float(failure_rate),
            'severe_failure_rate': float(severe_failure_rate),
            'failure_patterns': {
                'by_food_type': {
                    food: {
                        'count': len(errors_list),
                        'mean_error': float(np.mean(errors_list)),
                        'max_error': float(np.max(errors_list))
                    }
                    for food, errors_list in failure_patterns['by_food_type'].items()
                },
                'by_error_magnitude': {
                    level: len(cases)
                    for level, cases in failure_patterns['by_error_magnitude'].items()
                }
            },
            'top_failure_cases': [
                {
                    'case_id': case.case_id,
                    'error': case.error,
                    'food_name': case.data_features.get('food_name', 'unknown'),
                    'true_value': case.true_value,
                    'predicted_value': case.predicted_value
                }
                for case in sorted(failure_cases, key=lambda x: x.error, reverse=True)[:5]
            ],
            'recommendations': self._generate_failure_recommendations(failure_patterns)
        }

        logger.info(f"深度失败分析完成:")
        logger.info(f"  总失败率: {failure_rate*100:.1f}%")
        logger.info(f"  严重失败率: {severe_failure_rate*100:.1f}%")
        logger.info(f"  失败案例数: {len(failure_cases)}")

        return detailed_analysis

    def _generate_failure_recommendations(
        self,
        failure_patterns: Dict[str, Any]
    ) -> List[str]:
        """
        生成失败案例改进建议

        Args:
            failure_patterns: 失败模式分析结果

        Returns:
            List[str]: 改进建议列表
        """
        recommendations = []

        # 分析食物类型失败模式
        food_failures = failure_patterns.get('by_food_type', {})
        if food_failures:
            worst_food = max(food_failures.items(), key=lambda x: np.mean(x[1]) if x[1] else 0)
            if worst_food[1]:
                recommendations.append(
                    f"针对'{worst_food[0]}'类食物，建议增加训练样本或优化特征提取"
                )

        # 分析严重失败
        severe_failures = failure_patterns.get('by_error_magnitude', {}).get('severe', [])
        if len(severe_failures) > 0:
            recommendations.append(
                f"发现{len(severe_failures)}个严重失败案例，建议进行专家审查和模型调整"
            )

        if not recommendations:
            recommendations.append("失败案例分析未发现明显模式，建议进行更深入的特征分析")

        return recommendations
