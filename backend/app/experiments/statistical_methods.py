"""
统计方法模块 - 符合J-BHI标准
提供样本量计算、正态分布检验、方差齐性检验等统计方法
"""

import numpy as np
from scipy import stats
from scipy.stats import norm, shapiro, levene, mannwhitneyu, ttest_ind, ttest_rel
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SampleSizeCalculation:
    """样本量计算结果"""
    n: int
    alpha: float
    beta: float
    power: float
    effect_size: float
    formula: str
    z_alpha: float
    z_beta: float


@dataclass
class NormalityTestResult:
    """正态性检验结果"""
    statistic: float
    p_value: float
    is_normal: bool
    method: str
    significance_level: float


@dataclass
class VarianceTestResult:
    """方差齐性检验结果"""
    statistic: float
    p_value: float
    is_equal_variance: bool
    method: str
    significance_level: float


@dataclass
class StatisticalComparisonResult:
    """统计比较结果"""
    statistic: float
    p_value: float
    is_significant: bool
    method: str
    significance_level: float
    effect_size: Optional[float] = None
    effect_size_ci: Optional[Tuple[float, float]] = None
    cliffs_delta: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    clinical_significance: Optional[Dict[str, Any]] = None
    nnt: Optional[float] = None
    nnt_ci: Optional[Tuple[float, float]] = None


class StatisticalMethods:
    """统计方法类 - 符合J-BHI标准"""

    def __init__(self, alpha: float = 0.05, power: float = 0.8):
        """
        初始化统计方法

        Args:
            alpha: 显著性水平（第一类错误概率），默认0.05
            power: 统计功效（1-β），默认0.8
        """
        self.alpha = alpha
        self.beta = 1 - power
        self.power = power

    def calculate_sample_size(
        self,
        effect_size: float,
        alpha: Optional[float] = None,
        power: Optional[float] = None
    ) -> SampleSizeCalculation:
        """
        计算样本量 - 使用标准公式：n ≈ 2 × (Z₁-α/2 + Z₁-β)² / d²

        Args:
            effect_size: 效应量（d），通常为0.2（小效应）、0.5（中等效应）、0.8（大效应）
            alpha: 显著性水平，默认使用self.alpha
            power: 统计功效，默认使用self.power

        Returns:
            SampleSizeCalculation: 样本量计算结果

        References:
            J-BHI标准样本量计算公式：
            n ≈ 2 × (Z₁-α/2 + Z₁-β)² / d²
            其中：
            - Z₁-α/2: 标准正态分布的双侧检验临界值（α=0.05时，Z=1.96）
            - Z₁-β: 标准正态分布的单侧检验临界值（1-β=0.8时，Z=0.84）
            - d: 效应量（Cohen's d）
        """
        alpha = alpha or self.alpha
        power = power or self.power
        beta = 1 - power

        # 计算Z值
        z_alpha_half = norm.ppf(1 - alpha / 2)  # Z₁-α/2
        z_beta = norm.ppf(power)  # Z₁-β

        # 计算样本量
        n = 2 * (z_alpha_half + z_beta) ** 2 / (effect_size ** 2)
        n = int(np.ceil(n))

        formula = f"n ≈ 2 × (Z₁-α/2 + Z₁-β)² / d²"

        logger.info(f"样本量计算: n={n}, α={alpha}, 1-β={power}, d={effect_size}")
        logger.info(f"Z₁-α/2={z_alpha_half:.4f}, Z₁-β={z_beta:.4f}")

        return SampleSizeCalculation(
            n=n,
            alpha=alpha,
            beta=beta,
            power=power,
            effect_size=effect_size,
            formula=formula,
            z_alpha=z_alpha_half,
            z_beta=z_beta
        )

    def test_normality(
        self,
        data: Union[List[float], np.ndarray],
        method: str = "shapiro-wilk",
        significance_level: Optional[float] = None
    ) -> NormalityTestResult:
        """
        正态性检验

        Args:
            data: 待检验数据
            method: 检验方法（"shapiro-wilk" 或 "jarque-bera"）
            significance_level: 显著性水平，默认使用self.alpha

        Returns:
            NormalityTestResult: 正态性检验结果

        References:
            - Shapiro-Wilk检验：适用于小样本（n<50）
            - Jarque-Bera检验：适用于大样本（n≥50）
        """
        data = np.array(data)
        significance_level = significance_level or self.alpha

        if method.lower() == "shapiro-wilk":
            # Shapiro-Wilk检验（适用于小样本）
            if len(data) < 3:
                raise ValueError("Shapiro-Wilk检验需要至少3个样本")
            if len(data) > 5000:
                logger.warning("样本量过大，Shapiro-Wilk检验可能不准确，建议使用Jarque-Bera检验")

            statistic, p_value = shapiro(data)
            method_name = "Shapiro-Wilk"

        elif method.lower() == "jarque-bera":
            # Jarque-Bera检验（适用于大样本）
            if len(data) < 3:
                raise ValueError("Jarque-Bera检验需要至少3个样本")

            statistic, p_value = stats.jarque_bera(data)
            method_name = "Jarque-Bera"

        else:
            raise ValueError(f"不支持的检验方法: {method}")

        is_normal = p_value > significance_level

        logger.info(f"正态性检验 ({method_name}): statistic={statistic:.4f}, p={p_value:.4f}, "
                   f"is_normal={is_normal} (α={significance_level})")

        return NormalityTestResult(
            statistic=statistic,
            p_value=p_value,
            is_normal=is_normal,
            method=method_name,
            significance_level=significance_level
        )

    def test_variance_homogeneity(
        self,
        *groups: Union[List[float], np.ndarray],
        method: str = "levene",
        significance_level: Optional[float] = None
    ) -> VarianceTestResult:
        """
        方差齐性检验

        Args:
            *groups: 多个数据组
            method: 检验方法（"levene" 或 "bartlett"）
            significance_level: 显著性水平，默认使用self.alpha

        Returns:
            VarianceTestResult: 方差齐性检验结果

        References:
            - Levene检验：适用于非正态分布数据
            - Bartlett检验：适用于正态分布数据
        """
        if len(groups) < 2:
            raise ValueError("方差齐性检验需要至少2组数据")

        groups = [np.array(g) for g in groups]
        significance_level = significance_level or self.alpha

        if method.lower() == "levene":
            # Levene检验（适用于非正态分布）
            statistic, p_value = levene(*groups, center='median')
            method_name = "Levene"

        elif method.lower() == "bartlett":
            # Bartlett检验（适用于正态分布）
            statistic, p_value = stats.bartlett(*groups)
            method_name = "Bartlett"

        else:
            raise ValueError(f"不支持的检验方法: {method}")

        is_equal_variance = p_value > significance_level

        logger.info(f"方差齐性检验 ({method_name}): statistic={statistic:.4f}, p={p_value:.4f}, "
                   f"is_equal_variance={is_equal_variance} (α={significance_level})")

        return VarianceTestResult(
            statistic=statistic,
            p_value=p_value,
            is_equal_variance=is_equal_variance,
            method=method_name,
            significance_level=significance_level
        )

    def _calculate_cliffs_delta(
        self,
        group1: np.ndarray,
        group2: np.ndarray
    ) -> Optional[float]:
        """
        计算Cliff's delta（非参数效应量）

        Cliff's delta = (P(X > Y) - P(X < Y))
        其中X来自group1，Y来自group2

        Returns:
            float: Cliff's delta值，范围[-1, 1]
        """
        group1 = np.array(group1)
        group2 = np.array(group2)

        if len(group1) == 0 or len(group2) == 0:
            return None

        # 计算P(X > Y)和P(X < Y)
        greater_count = 0
        less_count = 0
        equal_count = 0

        for x in group1:
            for y in group2:
                if x > y:
                    greater_count += 1
                elif x < y:
                    less_count += 1
                else:
                    equal_count += 1

        total = len(group1) * len(group2)
        if total == 0:
            return None

        # Cliff's delta = (P(X > Y) - P(X < Y))
        # 处理相等的情况：将相等的一半分配给greater，一半分配给less
        delta = (greater_count + equal_count / 2 - less_count - equal_count / 2) / total

        return float(delta)

    def _calculate_clinical_significance(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        mcid: float = 0.03
    ) -> Dict[str, Any]:
        """
        计算临床意义评估指标

        Args:
            group1: 第一组数据（通常是基线方法）
            group2: 第二组数据（通常是提出的方法）
            mcid: 最小临床重要差异（Minimum Clinically Important Difference），默认0.03

        Returns:
            Dict[str, Any]: 临床意义评估结果
        """
        group1 = np.array(group1)
        group2 = np.array(group2)

        if len(group1) != len(group2) or len(group1) == 0:
            return {}

        # 1. 临床重要差异的比例（差异小于MCID的比例）
        differences = np.abs(group1 - group2)
        clinically_important = differences < mcid
        proportion_clinically_important = np.mean(clinically_important)

        # 2. 决策改变率（预测值四舍五入到0.1后是否改变）
        # 将预测值四舍五入到0.1（相当于10分制）
        group1_rounded = np.round(group1 * 10) / 10
        group2_rounded = np.round(group2 * 10) / 10
        decision_changed = group1_rounded != group2_rounded
        decision_change_rate = np.mean(decision_changed)

        # 3. 患者获益指数
        # 基于误差大小分配获益分数
        benefit_scores = []
        for diff in differences:
            if diff < mcid:
                benefit_scores.append(1.0)  # 优秀
            elif diff < mcid * 1.67:  # 约0.05
                benefit_scores.append(0.7)  # 良好
            elif diff < mcid * 3.33:  # 约0.1
                benefit_scores.append(0.3)  # 可接受
            else:
                benefit_scores.append(0.0)  # 不可接受

        patient_benefit_index = np.mean(benefit_scores) if benefit_scores else 0.0

        # 4. 平均绝对差异
        mean_absolute_difference = np.mean(differences)

        return {
            'proportion_clinically_important': float(proportion_clinically_important),
            'decision_change_rate': float(decision_change_rate),
            'patient_benefit_index': float(patient_benefit_index),
            'mean_absolute_difference': float(mean_absolute_difference),
            'mcid': mcid
        }

    def calculate_clinical_sample_size(
        self,
        min_clinical_difference: float = 0.03,
        expected_std: float = 0.15,
        alpha: Optional[float] = None,
        power: Optional[float] = None
    ) -> SampleSizeCalculation:
        """
        基于最小临床重要差异(MCID)的样本量计算

        Args:
            min_clinical_difference: 最小临床重要差异（MCID），默认0.03
            expected_std: 预期标准差，默认0.15
            alpha: 显著性水平，默认使用self.alpha
            power: 统计功效，默认使用self.power

        Returns:
            SampleSizeCalculation: 样本量计算结果

        References:
            AIIM标准：样本量应基于临床意义而非统计显著性
            AIIM要求统计功效≥0.90（J-BHI兼容模式0.8）
            MCID=0.03: 文化适配度差异0.03被认为是临床重要
        """
        alpha = alpha or self.alpha
        power = power or self.power

        # AIIM标准检查
        if power < 0.90:
            logger.warning(
                f"⚠️ AIIM期刊要求统计功效≥0.90，当前power={power}。"
                f"建议将power设置为0.90以确保符合AIIM标准。"
            )
        beta = 1 - power

        # 计算标准化效应量（d = MCID / 标准差）
        effect_size = min_clinical_difference / expected_std if expected_std > 0 else 0.2

        # 使用标准公式计算样本量
        z_alpha_half = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(power)

        n = 2 * (z_alpha_half + z_beta) ** 2 / (effect_size ** 2)
        n = int(np.ceil(n))

        formula = f"n ≈ 2 × (Z₁-α/2 + Z₁-β)² / d², where d = MCID / σ"

        logger.info(f"基于临床意义的样本量计算: n={n}, MCID={min_clinical_difference}, "
                   f"σ={expected_std}, d={effect_size:.4f}, α={alpha}, 1-β={power}")
        logger.info(f"Z₁-α/2={z_alpha_half:.4f}, Z₁-β={z_beta:.4f}")

        return SampleSizeCalculation(
            n=n,
            alpha=alpha,
            beta=beta,
            power=power,
            effect_size=effect_size,
            formula=formula,
            z_alpha=z_alpha_half,
            z_beta=z_beta
        )

    def _calculate_cohens_d_ci(
        self,
        cohens_d: float,
        n1: int,
        n2: int,
        confidence: float = 0.95
    ) -> Optional[Tuple[float, float]]:
        """
        计算Cohen's d的置信区间

        Args:
            cohens_d: Cohen's d值
            n1: 第一组样本量
            n2: 第二组样本量
            confidence: 置信水平，默认0.95

        Returns:
            Tuple[float, float]: 置信区间下界和上界
        """
        if n1 <= 1 or n2 <= 1:
            return None

        # 计算标准误
        # SE(d) = sqrt((n1 + n2) / (n1 * n2) + d² / (2 * (n1 + n2 - 2)))
        se = np.sqrt((n1 + n2) / (n1 * n2) + cohens_d ** 2 / (2 * (n1 + n2 - 2)))

        # 使用t分布（自由度 = n1 + n2 - 2）
        df = n1 + n2 - 2
        t_critical = stats.t.ppf(1 - (1 - confidence) / 2, df)

        # 计算置信区间
        lower = cohens_d - t_critical * se
        upper = cohens_d + t_critical * se

        return (float(lower), float(upper))

    def _calculate_cohens_d_ci_paired(
        self,
        cohens_d: float,
        n: int,
        confidence: float = 0.95
    ) -> Optional[Tuple[float, float]]:
        """
        计算配对数据Cohen's d的置信区间

        Args:
            cohens_d: Cohen's d值
            n: 样本量
            confidence: 置信水平，默认0.95

        Returns:
            Tuple[float, float]: 置信区间下界和上界
        """
        if n <= 1:
            return None

        # 配对数据的标准误
        # SE(d) = sqrt((1 / n) + d² / (2 * (n - 1)))
        se = np.sqrt(1 / n + cohens_d ** 2 / (2 * (n - 1)))

        # 使用t分布（自由度 = n - 1）
        df = n - 1
        t_critical = stats.t.ppf(1 - (1 - confidence) / 2, df)

        # 计算置信区间
        lower = cohens_d - t_critical * se
        upper = cohens_d + t_critical * se

        return (float(lower), float(upper))

    def _calculate_nnt(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        mcid: float = 0.03
    ) -> Optional[Dict[str, Any]]:
        """
        计算NNT（Number Needed to Treat）

        NNT = 1 / (P(proposed < MCID) - P(baseline < MCID))
        其中P(proposed < MCID)是提出方法误差小于MCID的比例

        Args:
            group1: 第一组数据（通常是基线方法）
            group2: 第二组数据（通常是提出的方法）
            mcid: 最小临床重要差异，默认0.03

        Returns:
            Dict[str, Any]: NNT结果，包含NNT值和置信区间
        """
        group1 = np.array(group1)
        group2 = np.array(group2)

        if len(group1) != len(group2) or len(group1) == 0:
            return None

        # 假设group1是基线，group2是提出的方法
        # 计算误差小于MCID的比例（这里假设是预测误差）
        # 实际应用中，需要根据具体场景调整
        # 这里使用差异小于MCID的比例作为"成功"的定义

        # 对于文化适配度预测，我们定义"成功"为预测误差小于MCID
        # 但这里我们比较的是预测值本身，所以需要调整逻辑
        # 假设我们比较的是预测误差（越小越好）
        # 如果group2的误差小于group1，则视为"治疗成功"

        # 简化版本：比较两组中"好结果"的比例
        # 这里假设值越大越好（文化适配度分数）
        # 如果group2 > group1 + MCID，视为"成功"

        # 计算"成功"比例
        # 对于预测任务，我们比较预测误差
        # 假设group1和group2是预测误差（越小越好）
        # 如果group2 < group1 - MCID，视为"成功"

        # 更合理的定义：比较达到临床重要改善的比例
        # 如果group2的预测误差比group1小MCID以上，视为"成功"
        improvements = group2 < (group1 - mcid)
        success_rate = np.mean(improvements)

        # 如果成功率为0或负，NNT无意义
        if success_rate <= 0:
            return None

        # NNT = 1 / 成功率
        nnt = 1.0 / success_rate

        # 计算NNT的置信区间（使用Wilson score interval）
        n = len(group1)
        z = stats.norm.ppf(0.975)  # 95%置信区间

        # Wilson score interval for proportion
        denominator = 1 + (z ** 2) / n
        center = (success_rate + (z ** 2) / (2 * n)) / denominator
        margin = z * np.sqrt((success_rate * (1 - success_rate) + (z ** 2) / (4 * n)) / n) / denominator

        success_rate_lower = max(0, center - margin)
        success_rate_upper = min(1, center + margin)

        # 转换为NNT置信区间（注意：NNT是倒数的，所以区间需要反转）
        nnt_lower = 1.0 / success_rate_upper if success_rate_upper > 0 else np.inf
        nnt_upper = 1.0 / success_rate_lower if success_rate_lower > 0 else np.inf

        return {
            'nnt': float(nnt),
            'nnt_ci': (float(nnt_lower), float(nnt_upper)),
            'success_rate': float(success_rate),
            'success_rate_ci': (float(success_rate_lower), float(success_rate_upper))
        }

    def compare_groups(
        self,
        group1: Union[List[float], np.ndarray],
        group2: Union[List[float], np.ndarray],
        paired: bool = False,
        significance_level: Optional[float] = None,
        assume_normal: Optional[bool] = None,
        assume_equal_variance: Optional[bool] = None
    ) -> StatisticalComparisonResult:
        """
        两组数据比较

        Args:
            group1: 第一组数据
            group2: 第二组数据
            paired: 是否为配对数据
            significance_level: 显著性水平，默认使用self.alpha
            assume_normal: 是否假设数据正态分布（None时自动检验）
            assume_equal_variance: 是否假设方差齐性（None时自动检验）

        Returns:
            StatisticalComparisonResult: 统计比较结果

        References:
            - 配对t检验：适用于配对数据且满足正态分布
            - 独立样本t检验：适用于独立数据且满足正态分布和方差齐性
            - Mann-Whitney U检验：适用于非正态分布或小样本数据
        """
        group1 = np.array(group1)
        group2 = np.array(group2)
        significance_level = significance_level or self.alpha

        # 自动检验正态性和方差齐性（如果未指定）
        if assume_normal is None:
            try:
                norm_test1 = self.test_normality(group1, method="shapiro-wilk" if len(group1) < 50 else "jarque-bera")
                norm_test2 = self.test_normality(group2, method="shapiro-wilk" if len(group2) < 50 else "jarque-bera")
                assume_normal = norm_test1.is_normal and norm_test2.is_normal
            except Exception as e:
                logger.warning(f"正态性检验失败: {e}，使用非参数检验")
                assume_normal = False

        if assume_equal_variance is None and not paired:
            try:
                var_test = self.test_variance_homogeneity(group1, group2, method="levene")
                assume_equal_variance = var_test.is_equal_variance
            except Exception as e:
                logger.warning(f"方差齐性检验失败: {e}，假设方差不齐")
                assume_equal_variance = False

        # 选择适当的检验方法
        if paired:
            # 配对数据
            if assume_normal:
                # 配对t检验
                statistic, p_value = ttest_rel(group1, group2)
                method_name = "配对t检验 (Paired t-test)"
            else:
                # Wilcoxon符号秩检验
                statistic, p_value = stats.wilcoxon(group1, group2)
                method_name = "Wilcoxon符号秩检验 (Wilcoxon signed-rank test)"
        else:
            # 独立数据
            if assume_normal:
                if assume_equal_variance:
                    # 独立样本t检验（等方差）
                    statistic, p_value = ttest_ind(group1, group2, equal_var=True)
                    method_name = "独立样本t检验 (Independent t-test, equal variance)"
                else:
                    # Welch's t检验（不等方差）
                    statistic, p_value = ttest_ind(group1, group2, equal_var=False)
                    method_name = "Welch's t检验 (Welch's t-test, unequal variance)"
            else:
                # Mann-Whitney U检验
                statistic, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
                method_name = "Mann-Whitney U检验 (Mann-Whitney U test)"

        is_significant = p_value < significance_level

        # 计算效应量（Cohen's d）- 改进：在所有情况下都计算
        effect_size = None
        effect_size_ci = None
        if not paired:
            pooled_std = np.sqrt((np.var(group1, ddof=1) + np.var(group2, ddof=1)) / 2)
            if pooled_std > 0:
                effect_size = (np.mean(group1) - np.mean(group2)) / pooled_std
                # 计算Cohen's d的95%置信区间
                n1, n2 = len(group1), len(group2)
                effect_size_ci = self._calculate_cohens_d_ci(effect_size, n1, n2)
        elif paired:
            # 配对数据：使用配对差异的标准差
            diff = group1 - group2
            if len(diff) > 1 and np.std(diff, ddof=1) > 0:
                effect_size = np.mean(diff) / np.std(diff, ddof=1)
                # 配对数据的Cohen's d置信区间
                n = len(diff)
                effect_size_ci = self._calculate_cohens_d_ci_paired(effect_size, n)

        # 计算Cliff's delta（非参数效应量）- 适用于所有情况
        cliffs_delta = self._calculate_cliffs_delta(group1, group2)

        # 计算置信区间
        confidence_interval = None
        if assume_normal:
            if paired:
                diff = group1 - group2
                mean_diff = np.mean(diff)
                std_diff = np.std(diff, ddof=1) / np.sqrt(len(diff))
                t_critical = stats.t.ppf(1 - significance_level / 2, len(diff) - 1)
                margin = t_critical * std_diff
                confidence_interval = (mean_diff - margin, mean_diff + margin)
            else:
                mean1, mean2 = np.mean(group1), np.mean(group2)
                std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
                n1, n2 = len(group1), len(group2)
                pooled_se = np.sqrt((std1**2 / n1) + (std2**2 / n2))
                t_critical = stats.t.ppf(1 - significance_level / 2, n1 + n2 - 2)
                margin = t_critical * pooled_se
                diff = mean1 - mean2
                confidence_interval = (diff - margin, diff + margin)

        # 格式化 p-value 显示（避免显示为 0.0000）
        if p_value < 0.0001:
            p_value_display = f"{p_value:.2e}"  # 科学计数法
            p_value_warning = "（p值极小，可能因样本量过大或差异极大）"
        else:
            p_value_display = f"{p_value:.4f}"
            p_value_warning = ""

        logger.info(f"统计比较 ({method_name}): statistic={statistic:.4f}, p={p_value_display}{p_value_warning}, "
                   f"is_significant={is_significant} (α={significance_level})")

        # 检查样本量是否合理
        n1, n2 = len(group1), len(group2)
        if n1 + n2 > 1000 and p_value < 0.001:
            logger.warning(
                f"⚠️ 样本量较大 (n1={n1}, n2={n2}) 且 p值极小 ({p_value:.2e})，"
                f"可能检测到统计显著但实际意义不大的差异。建议同时关注效应量。"
            )

        # 计算临床意义评估
        clinical_significance = self._calculate_clinical_significance(group1, group2)

        # 计算NNT（Number Needed to Treat）- 如果适用
        nnt = None
        nnt_ci = None
        if clinical_significance and 'proportion_clinically_important' in clinical_significance:
            nnt_result = self._calculate_nnt(group1, group2, clinical_significance.get('mcid', 0.03))
            if nnt_result:
                nnt = nnt_result.get('nnt')
                nnt_ci = nnt_result.get('nnt_ci')

        # 输出效应量信息
        if effect_size is not None:
            if effect_size_ci:
                logger.info(f"效应量 (Cohen's d): {effect_size:.4f} [{effect_size_ci[0]:.4f}, {effect_size_ci[1]:.4f}]")
            else:
                logger.info(f"效应量 (Cohen's d): {effect_size:.4f}")
            # 评估效应量的实际意义
            if abs(effect_size) < 0.2:
                logger.warning(f"⚠️ 效应量较小 (|d|={abs(effect_size):.4f} < 0.2)，差异的实际意义可能有限")
            elif abs(effect_size) < 0.5:
                logger.info(f"效应量为中等 (|d|={abs(effect_size):.4f})")
            else:
                logger.info(f"效应量较大 (|d|={abs(effect_size):.4f})")

        if cliffs_delta is not None:
            logger.info(f"效应量 (Cliff's delta): {cliffs_delta:.4f}")
            # 评估Cliff's delta的实际意义
            abs_delta = abs(cliffs_delta)
            if abs_delta < 0.147:
                logger.warning(f"⚠️ Cliff's delta较小 (|δ|={abs_delta:.4f} < 0.147)，差异的实际意义可能有限")
            elif abs_delta < 0.33:
                logger.info(f"Cliff's delta为小到中等 (|δ|={abs_delta:.4f})")
            elif abs_delta < 0.474:
                logger.info(f"Cliff's delta为中等 (|δ|={abs_delta:.4f})")
            else:
                logger.info(f"Cliff's delta较大 (|δ|={abs_delta:.4f})")

        # 输出临床意义信息
        if clinical_significance:
            logger.info(f"临床意义评估:")
            logger.info(f"  - 临床重要差异比例 (>{clinical_significance.get('mcid', 0.03):.3f}): {clinical_significance.get('proportion_clinically_important', 0):.1%}")
            logger.info(f"  - 决策改变率: {clinical_significance.get('decision_change_rate', 0):.1%}")
            logger.info(f"  - 患者获益指数: {clinical_significance.get('patient_benefit_index', 0):.3f}")

        # 输出NNT信息
        if nnt is not None:
            if nnt_ci:
                logger.info(f"NNT (Number Needed to Treat): {nnt:.2f} [{nnt_ci[0]:.2f}, {nnt_ci[1]:.2f}]")
            else:
                logger.info(f"NNT (Number Needed to Treat): {nnt:.2f}")
            # 评估NNT的临床意义
            if nnt < 5:
                logger.info(f"✅ NNT较小 ({nnt:.2f} < 5)，临床获益显著")
            elif nnt < 10:
                logger.info(f"✅ NNT中等 ({nnt:.2f} < 10)，临床获益可接受")
            elif nnt < 20:
                logger.warning(f"⚠️ NNT较大 ({nnt:.2f} < 20)，临床获益有限")
            else:
                logger.warning(f"⚠️ NNT很大 ({nnt:.2f} ≥ 20)，临床获益可能不显著")

        return StatisticalComparisonResult(
            statistic=statistic,
            p_value=p_value,
            is_significant=is_significant,
            method=method_name,
            significance_level=significance_level,
            effect_size=effect_size,
            effect_size_ci=effect_size_ci,
            cliffs_delta=cliffs_delta,
            confidence_interval=confidence_interval,
            clinical_significance=clinical_significance,
            nnt=nnt,
            nnt_ci=nnt_ci
        )
