

"""
综合评估指标模块
支持多任务学习、多类别分类、不平衡数据集等复杂场景的评估指标
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report, multilabel_confusion_matrix
)
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef
import logging

logger = logging.getLogger(__name__)

@dataclass
class ClassificationMetrics:
    """分类任务指标"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None
    specificity: Optional[float] = None
    sensitivity: Optional[float] = None
    cohen_kappa: Optional[float] = None
    matthews_corrcoef: Optional[float] = None

@dataclass
class RegressionMetrics:
    """回归任务指标"""
    mse: float
    rmse: float
    mae: float
    r2: float
    adjusted_r2: Optional[float] = None
    mape: Optional[float] = None  # Mean Absolute Percentage Error
    smape: Optional[float] = None  # Symmetric Mean Absolute Percentage Error

@dataclass
class MultiTaskMetrics:
    """多任务学习指标"""
    task_metrics: Dict[str, Union[ClassificationMetrics, RegressionMetrics]]
    overall_score: float
    task_weights: Dict[str, float]
    weighted_f1: Optional[float] = None
    weighted_auc: Optional[float] = None

class ComprehensiveEvaluator:
    """综合评估器"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def evaluate_classification(
        self,
        y_true: Union[List, np.ndarray],
        y_pred: Union[List, np.ndarray],
        y_prob: Optional[Union[List, np.ndarray]] = None,
        average: str = "macro",
        labels: Optional[List] = None
    ) -> ClassificationMetrics:
        """评估分类任务"""

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # 基本指标
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average=average, zero_division=0)
        recall = recall_score(y_true, y_pred, average=average, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=average, zero_division=0)

        # ROC-AUC
        roc_auc = None
        if y_prob is not None:
            y_prob = np.array(y_prob)
            try:
                if len(np.unique(y_true)) == 2:
                    # 二分类
                    roc_auc = roc_auc_score(y_true, y_prob)
                else:
                    # 多分类
                    roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average=average)
            except Exception as e:
                self.logger.warning(f"无法计算ROC-AUC: {e}")

        # PR-AUC
        pr_auc = None
        if y_prob is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    pr_auc = average_precision_score(y_true, y_prob)
                else:
                    pr_auc = average_precision_score(y_true, y_prob, average=average)
            except Exception as e:
                self.logger.warning(f"无法计算PR-AUC: {e}")

        # 特异性（仅二分类）
        specificity = None
        if len(np.unique(y_true)) == 2:
            try:
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            except Exception:
                pass

        # Cohen's Kappa
        cohen_kappa = None
        try:
            cohen_kappa = cohen_kappa_score(y_true, y_pred)
        except Exception as e:
            self.logger.warning(f"无法计算Cohen's Kappa: {e}")

        # Matthews相关系数
        mcc = None
        try:
            mcc = matthews_corrcoef(y_true, y_pred)
        except Exception as e:
            self.logger.warning(f"无法计算MCC: {e}")

        return ClassificationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            specificity=specificity,
            sensitivity=recall,  # 敏感性等于召回率
            cohen_kappa=cohen_kappa,
            matthews_corrcoef=mcc
        )

    def evaluate_regression(
        self,
        y_true: Union[List, np.ndarray],
        y_pred: Union[List, np.ndarray],
        n_features: Optional[int] = None
    ) -> RegressionMetrics:
        """评估回归任务"""

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # 基本指标
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # 调整R²
        adjusted_r2 = None
        if n_features is not None and len(y_true) > n_features + 1:
            n = len(y_true)
            adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)

        # MAPE (Mean Absolute Percentage Error)
        mape = None
        try:
            # 避免除零
            mask = y_true != 0
            if np.any(mask):
                mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        except Exception as e:
            self.logger.warning(f"无法计算MAPE: {e}")

        # SMAPE (Symmetric Mean Absolute Percentage Error)
        smape = None
        try:
            denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
            mask = denominator != 0
            if np.any(mask):
                smape = np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100
        except Exception as e:
            self.logger.warning(f"无法计算SMAPE: {e}")

        return RegressionMetrics(
            mse=mse,
            rmse=rmse,
            mae=mae,
            r2=r2,
            adjusted_r2=adjusted_r2,
            mape=mape,
            smape=smape
        )

    def evaluate_multitask(
        self,
        task_results: Dict[str, Dict[str, Any]],
        task_weights: Optional[Dict[str, float]] = None
    ) -> MultiTaskMetrics:
        """评估多任务学习"""

        if not task_results:
            raise ValueError("任务结果不能为空")

        # 默认权重（均等）
        if task_weights is None:
            task_weights = {task: 1.0 / len(task_results) for task in task_results.keys()}

        # 归一化权重
        total_weight = sum(task_weights.values())
        task_weights = {task: weight / total_weight for task, weight in task_weights.items()}

        # 解析每个任务的指标
        task_metrics = {}
        weighted_scores = []

        for task_name, results in task_results.items():
            task_type = results.get('task_type', 'classification')

            if task_type == 'classification':
                metrics = ClassificationMetrics(**results['metrics'])
                task_metrics[task_name] = metrics
                # 使用F1分数作为主要指标
                weighted_scores.append(metrics.f1_score * task_weights[task_name])
            elif task_type == 'regression':
                metrics = RegressionMetrics(**results['metrics'])
                task_metrics[task_name] = metrics
                # 使用R²作为主要指标
                weighted_scores.append(metrics.r2 * task_weights[task_name])
            else:
                raise ValueError(f"不支持的任务类型: {task_type}")

        # 计算加权F1和AUC（仅分类任务）
        weighted_f1 = None
        weighted_auc = None

        classification_tasks = [name for name, metrics in task_metrics.items()
                               if isinstance(metrics, ClassificationMetrics)]

        if classification_tasks:
            f1_scores = [task_metrics[task].f1_score for task in classification_tasks]
            f1_weights = [task_weights[task] for task in classification_tasks]
            weighted_f1 = sum(f1 * w for f1, w in zip(f1_scores, f1_weights))

            # 计算加权AUC
            auc_scores = [task_metrics[task].roc_auc for task in classification_tasks
                         if task_metrics[task].roc_auc is not None]
            if auc_scores:
                auc_weights = [task_weights[task] for task in classification_tasks
                              if task_metrics[task].roc_auc is not None]
                weighted_auc = sum(auc * w for auc, w in zip(auc_scores, auc_weights))

        return MultiTaskMetrics(
            task_metrics=task_metrics,
            overall_score=sum(weighted_scores),
            task_weights=task_weights,
            weighted_f1=weighted_f1,
            weighted_auc=weighted_auc
        )

    def evaluate_imbalanced_classification(
        self,
        y_true: Union[List, np.ndarray],
        y_pred: Union[List, np.ndarray],
        y_prob: Optional[Union[List, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """评估不平衡分类任务"""

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # 基本指标
        metrics = self.evaluate_classification(y_true, y_pred, y_prob)

        # 类别分布
        unique_classes, class_counts = np.unique(y_true, return_counts=True)
        class_distribution = dict(zip(unique_classes, class_counts))

        # 计算每个类别的指标
        per_class_metrics = {}
        for class_label in unique_classes:
            class_mask = y_true == class_label
            class_pred = y_pred[class_mask]
            class_true = y_true[class_mask]

            if len(class_true) > 0:
                precision = precision_score([class_label], [class_label] if class_label in class_pred else [0],
                                          average='binary', zero_division=0)
                recall = recall_score([class_label], [class_label] if class_label in class_pred else [0],
                                    average='binary', zero_division=0)
                f1 = f1_score([class_label], [class_label] if class_label in class_pred else [0],
                            average='binary', zero_division=0)

                per_class_metrics[class_label] = {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'support': len(class_true)
                }

        # 计算不平衡指标
        imbalance_ratio = max(class_counts) / min(class_counts) if len(class_counts) > 1 else 1.0

        return {
            'overall_metrics': metrics,
            'per_class_metrics': per_class_metrics,
            'class_distribution': class_distribution,
            'imbalance_ratio': imbalance_ratio,
            'is_imbalanced': imbalance_ratio > 2.0
        }

    def evaluate_multilabel_classification(
        self,
        y_true: Union[List, np.ndarray],
        y_pred: Union[List, np.ndarray],
        y_prob: Optional[Union[List, np.ndarray]] = None,
        labels: Optional[List] = None
    ) -> Dict[str, Any]:
        """评估多标签分类任务"""

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        if labels is None:
            labels = list(range(y_true.shape[1]))

        # 每个标签的指标
        label_metrics = {}
        for i, label in enumerate(labels):
            label_true = y_true[:, i]
            label_pred = y_pred[:, i]

            metrics = self.evaluate_classification(label_true, label_pred)
            label_metrics[label] = metrics

        # 整体指标
        overall_metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
            'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
            'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
            'precision_samples': precision_score(y_true, y_pred, average='samples', zero_division=0),
            'recall_samples': recall_score(y_true, y_pred, average='samples', zero_division=0),
            'f1_samples': f1_score(y_true, y_pred, average='samples', zero_division=0)
        }

        # 子集准确率（所有标签都预测正确）
        subset_accuracy = accuracy_score(y_true, y_pred)

        return {
            'overall_metrics': overall_metrics,
            'per_label_metrics': label_metrics,
            'subset_accuracy': subset_accuracy,
            'num_labels': len(labels)
        }

    def evaluate_time_series(
        self,
        y_true: Union[List, np.ndarray],
        y_pred: Union[List, np.ndarray],
        horizon: int = 1
    ) -> Dict[str, Any]:
        """评估时间序列预测任务"""

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # 基本回归指标
        regression_metrics = self.evaluate_regression(y_true, y_pred)

        # 时间序列特定指标
        # MAPE for time series
        mape = None
        try:
            mask = y_true != 0
            if np.any(mask):
                mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        except Exception:
            pass

        # SMAPE
        smape = None
        try:
            denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
            mask = denominator != 0
            if np.any(mask):
                smape = np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100
        except Exception:
            pass

        # 方向准确率（预测方向是否正确）
        direction_accuracy = None
        if len(y_true) > 1:
            true_directions = np.diff(y_true) > 0
            pred_directions = np.diff(y_pred) > 0
            direction_accuracy = accuracy_score(true_directions, pred_directions)

        return {
            'regression_metrics': regression_metrics,
            'mape': mape,
            'smape': smape,
            'direction_accuracy': direction_accuracy,
            'horizon': horizon
        }

    def get_confusion_matrix_details(
        self,
        y_true: Union[List, np.ndarray],
        y_pred: Union[List, np.ndarray],
        labels: Optional[List] = None
    ) -> Dict[str, Any]:
        """获取混淆矩阵详细信息"""

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        cm = confusion_matrix(y_true, y_pred, labels=labels)

        if labels is None:
            labels = list(range(len(cm)))

        # 计算每个类别的指标
        class_metrics = {}
        for i, label in enumerate(labels):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            class_metrics[label] = {
                'precision': precision,
                'recall': recall,
                'specificity': specificity,
                'f1_score': f1,
                'support': tp + fn,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn
            }

        return {
            'confusion_matrix': cm.tolist(),
            'labels': labels,
            'class_metrics': class_metrics,
            'total_samples': cm.sum()
        }

    def calculate_feature_importance_metrics(
        self,
        feature_importance: Union[List, np.ndarray],
        feature_names: Optional[List[str]] = None,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """计算特征重要性指标"""

        importance = np.array(feature_importance)

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]

        # 排序
        sorted_indices = np.argsort(importance)[::-1]
        sorted_importance = importance[sorted_indices]
        sorted_names = [feature_names[i] for i in sorted_indices]

        # 归一化
        normalized_importance = sorted_importance / np.sum(sorted_importance)

        # 累积重要性
        cumulative_importance = np.cumsum(normalized_importance)

        # 前k个特征
        top_features = {
            'names': sorted_names[:top_k],
            'importance': sorted_importance[:top_k].tolist(),
            'normalized_importance': normalized_importance[:top_k].tolist(),
            'cumulative_importance': cumulative_importance[:top_k].tolist()
        }

        # 重要性统计
        stats = {
            'mean_importance': float(np.mean(importance)),
            'std_importance': float(np.std(importance)),
            'max_importance': float(np.max(importance)),
            'min_importance': float(np.min(importance)),
            'importance_range': float(np.max(importance) - np.min(importance)),
            'gini_impurity': float(np.sum(normalized_importance * (1 - normalized_importance)))
        }

        return {
            'top_features': top_features,
            'statistics': stats,
            'total_features': len(importance),
            'top_k': top_k
        }

# 全局评估器实例
comprehensive_evaluator = ComprehensiveEvaluator()

__all__ = ["'logger'", "'ClassificationMetrics'", "'RegressionMetrics'", "'MultiTaskMetrics'", "'ComprehensiveEvaluator'", "'comprehensive_evaluator'"]
