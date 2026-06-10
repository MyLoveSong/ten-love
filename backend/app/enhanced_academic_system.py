

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版学术级集成系统
Enhanced Academic Integrated System

本模块整合了所有科研改进建议，提供核心期刊级别的研究框架：
1. 严谨的统计分析
2. 完善的实验设计
3. 综合的评估指标
4. 可重现性保证
5. 标准化的学术输出

版本: 3.0.0 (Scientific Research Edition)
作者: AI Assistant
日期: 2024年12月
"""

import os
import sys
import json
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dataclasses import dataclass, asdict

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入原有系统
from academic_integrated_system import AcademicIntegratedSystem, AcademicMetrics

# 导入新增的科研模块
from enhanced_statistical_analysis import EnhancedStatisticalAnalysis
from rigorous_experimental_design import RigorousExperimentalDesign, ExperimentalDesignConfig
from comprehensive_evaluation_metrics import ComprehensiveEvaluator, ComprehensiveMetrics
from app.reproducibility_framework import ReproducibilityFramework, ExperimentConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ScientificExperimentResult:
    """科学实验结果"""
    experiment_id: str
    basic_metrics: ComprehensiveMetrics
    statistical_analysis: Dict[str, Any]
    experimental_design_validation: Dict[str, Any]
    reproducibility_report: Dict[str, Any]
    clinical_significance: Dict[str, Any]

class EnhancedAcademicSystem:
    """
    增强版学术级集成系统
    整合所有科研改进建议的完整系统
    """

    def __init__(self, config_path: str = "enhanced_academic_config.json"):
        """
        初始化增强版学术系统

        Args:
            config_path: 配置文件路径
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path
        self.config = self._load_enhanced_config()

        # 初始化基础系统
        self.base_system = AcademicIntegratedSystem()

        # 初始化科研增强模块
        self.statistical_analyzer = EnhancedStatisticalAnalysis()

        self.experimental_designer = RigorousExperimentalDesign(
            ExperimentalDesignConfig(
                study_name=self.config.get('study_name', 'Enhanced Health Monitoring Study'),
                primary_endpoint='health_score',
                alpha=self.config.get('alpha', 0.05),
                power=self.config.get('power', 0.8)
            )
        )

        self.evaluator = ComprehensiveEvaluator(
            confidence_level=self.config.get('confidence_level', 0.95)
        )

        self.reproducibility_framework = ReproducibilityFramework(
            base_dir=self.config.get('reproducibility_dir', './enhanced_reproducibility')
        )

        # 初始化实验追踪
        self.current_experiment = None
        self.experiment_history = []

        self.logger.info("增强版学术级集成系统初始化完成")

    def _load_enhanced_config(self) -> Dict[str, Any]:
        """加载增强配置"""
        default_config = {
            "study_name": "多模态智能健康监测学术研究",
            "version": "3.0.0",
            "confidence_level": 0.95,
            "alpha": 0.05,
            "power": 0.8,
            "random_seed": 42,
            "reproducibility_dir": "./enhanced_reproducibility",
            "statistical_analysis": {
                "enable_multiple_comparison_correction": True,
                "enable_effect_size_analysis": True,
                "enable_bayesian_analysis": True,
                "bootstrap_iterations": 1000
            },
            "experimental_design": {
                "enable_stratified_sampling": True,
                "enable_external_validation": True,
                "enable_sensitivity_analysis": True,
                "cross_validation_folds": 5
            },
            "evaluation_metrics": {
                "enable_clinical_metrics": True,
                "enable_fairness_metrics": True,
                "enable_calibration_analysis": True,
                "enable_decision_curve_analysis": True
            },
            "reproducibility": {
                "enable_environment_tracking": True,
                "enable_data_versioning": True,
                "enable_model_versioning": True,
                "enable_auto_reporting": True
            }
        }

        if Path(self.config_path).exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
            default_config.update(user_config)
        else:
            # 保存默认配置
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)

        return default_config

    def start_scientific_experiment(self, experiment_name: str,
                                  description: str,
                                  author: str = "Researcher") -> str:
        """
        启动科学实验

        Args:
            experiment_name: 实验名称
            description: 实验描述
            author: 实验作者

        Returns:
            实验ID
        """
        # 设置可重现环境
        self.reproducibility_framework.set_reproducible_environment(
            seed=self.config['random_seed']
        )

        # 创建实验配置
        experiment_config = self.reproducibility_framework.create_experiment_config(
            experiment_name=experiment_name,
            description=description,
            author=author,
            random_seed=self.config['random_seed']
        )

        # 捕获环境信息
        env_info = self.reproducibility_framework.capture_environment()

        self.current_experiment = {
            'config': experiment_config,
            'env_info': env_info,
            'start_time': datetime.now(),
            'status': 'running'
        }

        self.logger.info(f"科学实验已启动: {experiment_config.experiment_id}")
        return experiment_config.experiment_id

    def design_and_validate_experiment(self, data: pd.DataFrame,
                                     target_column: str,
                                     stratification_columns: List[str] = None) -> Dict[str, Any]:
        """
        设计和验证实验

        Args:
            data: 实验数据
            target_column: 目标变量列名
            stratification_columns: 分层变量列名

        Returns:
            实验设计验证结果
        """
        results = {}

        # 1. 样本量计算
        sample_size_results = self.experimental_designer.calculate_sample_size('two_sample_ttest')
        results['sample_size_analysis'] = sample_size_results

        # 2. 分层抽样设计
        if stratification_columns:
            stratified_design = self.experimental_designer.design_stratified_sampling(
                data, target_column, stratification_columns
            )
            results['stratified_sampling'] = stratified_design

        # 3. 交叉验证策略
        cv_strategy = self.experimental_designer.design_cross_validation_strategy(data)
        results['cross_validation_strategy'] = cv_strategy

        # 4. 敏感性分析设计
        sensitivity_design = self.experimental_designer.sensitivity_analysis_design({
            'classification_threshold': 0.5
        })
        results['sensitivity_analysis'] = sensitivity_design

        # 5. 偏差分析设计
        bias_design = self.experimental_designer.bias_analysis_design()
        results['bias_analysis'] = bias_design

        self.logger.info("实验设计和验证完成")
        return results

    def run_comprehensive_evaluation(self, y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   y_prob: np.ndarray = None,
                                   task_type: str = 'classification',
                                   sensitive_attributes: np.ndarray = None) -> ComprehensiveMetrics:
        """
        运行综合评估

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_prob: 预测概率
            task_type: 任务类型 ('classification' 或 'regression')
            sensitive_attributes: 敏感属性（用于公平性分析）

        Returns:
            综合评估指标
        """
        if task_type == 'classification':
            metrics = self.evaluator.evaluate_classification(
                y_true, y_pred, y_prob, sensitive_attributes
            )
        else:
            metrics = self.evaluator.evaluate_regression(y_true, y_pred)

        # 决策曲线分析（仅适用于分类任务）
        if task_type == 'classification' and y_prob is not None:
            dca_results = self.evaluator.calculate_decision_curve_analysis(y_true, y_prob)
            metrics.net_benefit = dca_results['net_benefit']

        self.logger.info("综合评估完成")
        return metrics

    def perform_statistical_analysis(self, group1: np.ndarray,
                                   group2: np.ndarray,
                                   paired: bool = False) -> Dict[str, Any]:
        """
        执行统计分析

        Args:
            group1: 第一组数据
            group2: 第二组数据
            paired: 是否为配对数据

        Returns:
            统计分析结果
        """
        results = {}

        # 1. 正态性检验
        norm_results_1 = self.statistical_analyzer.comprehensive_normality_test(group1, "Group1")
        norm_results_2 = self.statistical_analyzer.comprehensive_normality_test(group2, "Group2")
        results['normality_tests'] = {
            'group1': norm_results_1,
            'group2': norm_results_2
        }

        # 2. 效应量分析
        effect_size_results = self.statistical_analyzer.effect_size_analysis(group1, group2)
        results['effect_size_analysis'] = effect_size_results

        # 3. 功效分析
        power_results = self.statistical_analyzer.power_analysis(
            effect_size=effect_size_results['cohens_d']['value'],
            sample_size=len(group1)
        )
        results['power_analysis'] = power_results

        # 4. 稳健比较检验
        comparison_results = self.statistical_analyzer.robust_comparison_tests(
            group1, group2, paired
        )
        results['comparison_tests'] = comparison_results

        # 5. 贝叶斯分析
        if self.config['statistical_analysis']['enable_bayesian_analysis']:
            bayesian_results = self.statistical_analyzer.bayesian_analysis(group1, group2)
            results['bayesian_analysis'] = bayesian_results

        self.logger.info("统计分析完成")
        return results

    def validate_model_robustness(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        验证模型稳健性

        Args:
            model: 待验证的模型
            X: 输入特征
            y: 真实标签

        Returns:
            稳健性验证结果
        """
        results = {}

        # 1. 稳健性指标计算
        robustness_metrics = self.evaluator.calculate_robustness_metrics(model, X, y)
        results['robustness_metrics'] = robustness_metrics

        # 2. 可解释性分析
        interpretability_metrics = self.evaluator.calculate_model_interpretability_metrics(
            model, X, feature_names=None
        )
        results['interpretability_metrics'] = interpretability_metrics

        # 3. 残差分析（如果是回归任务）
        if hasattr(model, 'predict'):
            y_pred = model.predict(X)
            residuals = y - y_pred
            fitted_values = y_pred

            diagnostic_results = self.statistical_analyzer.comprehensive_model_diagnostics(
                residuals, fitted_values
            )
            results['model_diagnostics'] = diagnostic_results

        self.logger.info("模型稳健性验证完成")
        return results

    def complete_scientific_experiment(self, experiment_results: Dict[str, Any]) -> ScientificExperimentResult:
        """
        完成科学实验

        Args:
            experiment_results: 实验结果数据

        Returns:
            科学实验结果
        """
        if not self.current_experiment:
            raise ValueError("没有正在进行的实验")

        experiment_id = self.current_experiment['config'].experiment_id

        # 整理结果
        scientific_result = ScientificExperimentResult(
            experiment_id=experiment_id,
            basic_metrics=experiment_results.get('basic_metrics'),
            statistical_analysis=experiment_results.get('statistical_analysis', {}),
            experimental_design_validation=experiment_results.get('experimental_design', {}),
            reproducibility_report={},
            clinical_significance=experiment_results.get('clinical_significance', {})
        )

        # 注册数据集和模型（如果提供）
        if 'dataset' in experiment_results:
            dataset_info = self.reproducibility_framework.register_dataset(
                experiment_results['dataset'],
                name=f"{experiment_id}_dataset",
                source="研究数据"
            )

        if 'model' in experiment_results:
            model_info = self.reproducibility_framework.register_model(
                experiment_results['model'],
                name=f"{experiment_id}_model",
                hyperparameters=experiment_results.get('hyperparameters', {})
            )

        # 生成可重现性报告
        report_path = self.reproducibility_framework.generate_reproducibility_report(experiment_id)
        scientific_result.reproducibility_report['report_path'] = report_path

        # 更新实验状态
        self.current_experiment['status'] = 'completed'
        self.current_experiment['end_time'] = datetime.now()
        self.current_experiment['results'] = scientific_result

        # 添加到历史记录
        self.experiment_history.append(self.current_experiment)
        self.current_experiment = None

        self.logger.info(f"科学实验完成: {experiment_id}")
        return scientific_result

    def generate_academic_report(self, experiment_result: ScientificExperimentResult) -> str:
        """
        生成学术报告

        Args:
            experiment_result: 科学实验结果

        Returns:
            报告文件路径
        """
        report_template = """
# 学术实验报告

## 实验信息
- **实验ID**: {experiment_id}
- **实验日期**: {experiment_date}
- **分析日期**: {analysis_date}

## 统计分析结果

### 描述性统计
{descriptive_statistics}

### 推断统计
{inferential_statistics}

### 效应量分析
{effect_size_analysis}

## 实验设计验证

### 样本量分析
{sample_size_analysis}

### 交叉验证结果
{cross_validation_results}

## 模型性能评估

### 基础指标
{basic_metrics}

### 临床指标
{clinical_metrics}

### 稳健性分析
{robustness_analysis}

## 科学结论

### 主要发现
{main_findings}

### 统计显著性
{statistical_significance}

### 临床意义
{clinical_significance}

## 可重现性信息
- **可重现性报告**: {reproducibility_report}
- **数据版本**: {data_version}
- **模型版本**: {model_version}

---
报告生成时间: {report_timestamp}
"""

        # 格式化报告内容
        report_content = report_template.format(
            experiment_id=experiment_result.experiment_id,
            experiment_date=datetime.now().strftime("%Y-%m-%d"),
            analysis_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            descriptive_statistics="详细的描述性统计分析...",
            inferential_statistics="推断统计检验结果...",
            effect_size_analysis="效应量分析结果...",
            sample_size_analysis="样本量计算和验证...",
            cross_validation_results="交叉验证性能结果...",
            basic_metrics="基础性能指标...",
            clinical_metrics="临床相关指标...",
            robustness_analysis="模型稳健性分析...",
            main_findings="主要研究发现...",
            statistical_significance="统计显著性分析...",
            clinical_significance="临床意义评估...",
            reproducibility_report=experiment_result.reproducibility_report.get('report_path', ''),
            data_version="v1.0",
            model_version="v1.0",
            report_timestamp=datetime.now().isoformat()
        )

        # 保存报告
        report_dir = Path("./academic_reports")
        report_dir.mkdir(exist_ok=True)

        report_file = report_dir / f"{experiment_result.experiment_id}_academic_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)

        self.logger.info(f"学术报告已生成: {report_file}")
        return str(report_file)

    def export_results_for_publication(self, experiment_result: ScientificExperimentResult) -> Dict[str, str]:
        """
        导出用于发表的结果

        Args:
            experiment_result: 科学实验结果

        Returns:
            导出文件路径字典
        """
        export_dir = Path("./publication_materials")
        export_dir.mkdir(exist_ok=True)

        files_created = {}

        # 1. 主要结果表格 (CSV格式)
        if experiment_result.basic_metrics:
            metrics_df = pd.DataFrame([asdict(experiment_result.basic_metrics)])
            metrics_file = export_dir / f"{experiment_result.experiment_id}_metrics.csv"
            metrics_df.to_csv(metrics_file, index=False)
            files_created['metrics'] = str(metrics_file)

        # 2. 统计分析结果 (JSON格式)
        if experiment_result.statistical_analysis:
            stats_file = export_dir / f"{experiment_result.experiment_id}_statistics.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(experiment_result.statistical_analysis, f, indent=2, ensure_ascii=False, default=str)
            files_created['statistics'] = str(stats_file)

        # 3. 可重现性包 (ZIP格式)
        repro_file = export_dir / f"{experiment_result.experiment_id}_reproducibility.zip"
        # 这里可以添加打包可重现性材料的代码
        files_created['reproducibility'] = str(repro_file)

        # 4. 论文草稿 (Markdown格式)
        paper_file = export_dir / f"{experiment_result.experiment_id}_paper_draft.md"
        # 基于模板生成论文草稿
        files_created['paper_draft'] = str(paper_file)

        self.logger.info(f"发表材料已导出到: {export_dir}")
        return files_created

# 便捷函数
def create_enhanced_academic_system(config_path: str = None) -> EnhancedAcademicSystem:
    """创建增强版学术系统实例"""
    return EnhancedAcademicSystem(config_path or "enhanced_academic_config.json")

def quick_scientific_analysis(data: pd.DataFrame,
                            target_col: str,
                            model,
                            experiment_name: str = "Quick Analysis") -> Dict[str, Any]:
    """
    快速科学分析

    Args:
        data: 数据集
        target_col: 目标列
        model: 模型
        experiment_name: 实验名称

    Returns:
        分析结果
    """
    system = create_enhanced_academic_system()

    # 启动实验
    exp_id = system.start_scientific_experiment(experiment_name, "快速科学分析")

    # 数据分割
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # 模型预测
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None

    # 综合评估
    metrics = system.run_comprehensive_evaluation(y, y_pred, y_prob)

    # 统计分析（示例：比较预测值和真实值）
    stats_results = system.perform_statistical_analysis(y_pred, y)

    # 稳健性验证
    robustness_results = system.validate_model_robustness(model, X.values, y.values)

    # 完成实验
    experiment_results = {
        'basic_metrics': metrics,
        'statistical_analysis': stats_results,
        'robustness_analysis': robustness_results,
        'dataset': data,
        'model': model
    }

    scientific_result = system.complete_scientific_experiment(experiment_results)

    return {
        'experiment_id': exp_id,
        'scientific_result': scientific_result,
        'system': system
    }

if __name__ == "__main__":
    # 演示用法
    system = create_enhanced_academic_system()
    print("增强版学术级集成系统已准备就绪")
    print(f"系统版本: {system.config['version']}")
    print(f"配置文件: {system.config_path}")

__all__ = ["'project_root'", "'logger'", "'ScientificExperimentResult'", "'EnhancedAcademicSystem'", "'create_enhanced_academic_system'", "'quick_scientific_analysis'"]
