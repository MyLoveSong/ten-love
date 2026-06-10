

"""
增强可视化模块
支持ROC曲线、特征重要性、误差分析等高级可视化功能
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
import logging
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from app.matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("Matplotlib not available, visualization features disabled")

try:
    from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
    from sklearn.metrics import roc_auc_score, average_precision_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available, some visualization features disabled")

logger = logging.getLogger(__name__)

@dataclass
class VisualizationConfig:
    """可视化配置"""
    figure_size: Tuple[int, int] = (10, 8)
    dpi: int = 300
    style: str = "whitegrid"
    color_palette: str = "husl"
    font_size: int = 12
    save_format: str = "png"

class AdvancedVisualizer:
    """高级可视化器"""

    def __init__(self, output_dir: str = "visualizations", config: Optional[VisualizationConfig] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.config = config or VisualizationConfig()
        self.logger = logging.getLogger(__name__)

        if MATPLOTLIB_AVAILABLE:
            # 设置matplotlib样式
            plt.style.use('default')
            sns.set_style(self.config.style)
            sns.set_palette(self.config.color_palette)
            plt.rcParams['font.size'] = self.config.font_size

    def plot_roc_curve(
        self,
        y_true: Union[List, np.ndarray],
        y_prob: Union[List, np.ndarray],
        model_name: str = "Model",
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """绘制ROC曲线"""

        if not MATPLOTLIB_AVAILABLE or not SKLEARN_AVAILABLE:
            self.logger.warning("Matplotlib or scikit-learn not available")
            return None

        y_true = np.array(y_true)
        y_prob = np.array(y_prob)

        # 计算ROC曲线
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        auc_score = roc_auc_score(y_true, y_prob)

        # 绘制图形
        fig, ax = plt.subplots(figsize=self.config.figure_size)

        ax.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc_score:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')

        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - {model_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 保存图形
        if save_path is None:
            save_path = self.output_dir / f"roc_curve_{model_name}.png"

        plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()

        self.logger.info(f"ROC曲线已保存: {save_path}")
        return str(save_path)

    def plot_precision_recall_curve(
        self,
        y_true: Union[List, np.ndarray],
        y_prob: Union[List, np.ndarray],
        model_name: str = "Model",
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """绘制精确率-召回率曲线"""

        if not MATPLOTLIB_AVAILABLE or not SKLEARN_AVAILABLE:
            self.logger.warning("Matplotlib or scikit-learn not available")
            return None

        y_true = np.array(y_true)
        y_prob = np.array(y_prob)

        # 计算PR曲线
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        ap_score = average_precision_score(y_true, y_prob)

        # 绘制图形
        fig, ax = plt.subplots(figsize=self.config.figure_size)

        ax.plot(recall, precision, linewidth=2, label=f'{model_name} (AP = {ap_score:.3f})')

        # 添加基线（随机分类器）
        baseline = np.sum(y_true) / len(y_true)
        ax.axhline(y=baseline, color='k', linestyle='--', alpha=0.5, label=f'Random (AP = {baseline:.3f})')

        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Curve - {model_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 保存图形
        if save_path is None:
            save_path = self.output_dir / f"pr_curve_{model_name}.png"

        plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()

        self.logger.info(f"PR曲线已保存: {save_path}")
        return str(save_path)

    def plot_confusion_matrix(
        self,
        y_true: Union[List, np.ndarray],
        y_pred: Union[List, np.ndarray],
        labels: Optional[List] = None,
        model_name: str = "Model",
        normalize: bool = False,
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """绘制混淆矩阵"""

        if not MATPLOTLIB_AVAILABLE or not SKLEARN_AVAILABLE:
            self.logger.warning("Matplotlib or scikit-learn not available")
            return None

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title_suffix = ' (Normalized)'
        else:
            fmt = 'd'
            title_suffix = ''

        # 绘制图形
        fig, ax = plt.subplots(figsize=self.config.figure_size)

        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', ax=ax)

        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(f'Confusion Matrix - {model_name}{title_suffix}')

        if labels:
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)

        # 保存图形
        if save_path is None:
            suffix = "_normalized" if normalize else ""
            save_path = self.output_dir / f"confusion_matrix_{model_name}{suffix}.png"

        plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()

        self.logger.info(f"混淆矩阵已保存: {save_path}")
        return str(save_path)

    def plot_feature_importance(
        self,
        feature_importance: Union[List, np.ndarray],
        feature_names: Optional[List[str]] = None,
        top_k: int = 20,
        model_name: str = "Model",
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """绘制特征重要性"""

        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning("Matplotlib not available")
            return None

        importance = np.array(feature_importance)

        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(importance))]

        # 排序并选择前k个
        sorted_indices = np.argsort(importance)[::-1][:top_k]
        sorted_importance = importance[sorted_indices]
        sorted_names = [feature_names[i] for i in sorted_indices]

        # 绘制图形
        fig, ax = plt.subplots(figsize=(self.config.figure_size[0], max(6, top_k * 0.3)))

        bars = ax.barh(range(len(sorted_names)), sorted_importance)
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names)
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Top {top_k} Feature Importance - {model_name}')

        # 添加数值标签
        for i, (bar, value) in enumerate(zip(bars, sorted_importance)):
            ax.text(value + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{value:.3f}', va='center', ha='left')

        ax.grid(True, alpha=0.3, axis='x')

        # 保存图形
        if save_path is None:
            save_path = self.output_dir / f"feature_importance_{model_name}.png"

        plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()

        self.logger.info(f"特征重要性图已保存: {save_path}")
        return str(save_path)

    def plot_error_analysis(
        self,
        y_true: Union[List, np.ndarray],
        y_pred: Union[List, np.ndarray],
        model_name: str = "Model",
        task_type: str = "regression",
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """绘制误差分析图"""

        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning("Matplotlib not available")
            return None

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        if task_type == "regression":
            return self._plot_regression_error_analysis(y_true, y_pred, model_name, save_path)
        elif task_type == "classification":
            return self._plot_classification_error_analysis(y_true, y_pred, model_name, save_path)
        else:
            self.logger.warning(f"不支持的任务类型: {task_type}")
            return None

    def _plot_regression_error_analysis(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """回归任务误差分析"""

        errors = y_pred - y_true

        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Error Analysis - {model_name}', fontsize=16)

        # 1. 预测值 vs 真实值
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6)
        axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('True Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title('Predicted vs True Values')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 残差图
        axes[0, 1].scatter(y_pred, errors, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 误差分布
        axes[1, 0].hist(errors, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Residual Distribution')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Q-Q图
        from scipy import stats
        stats.probplot(errors, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图形
        if save_path is None:
            save_path = self.output_dir / f"error_analysis_{model_name}.png"

        plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()

        self.logger.info(f"误差分析图已保存: {save_path}")
        return str(save_path)

    def _plot_classification_error_analysis(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """分类任务误差分析"""

        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Classification Error Analysis - {model_name}', fontsize=16)

        # 1. 混淆矩阵
        if SKLEARN_AVAILABLE:
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
            axes[0, 0].set_title('Confusion Matrix')
            axes[0, 0].set_xlabel('Predicted')
            axes[0, 0].set_ylabel('True')

        # 2. 准确率按类别
        unique_classes = np.unique(y_true)
        class_accuracies = []
        for cls in unique_classes:
            mask = y_true == cls
            if np.sum(mask) > 0:
                accuracy = np.sum((y_pred[mask] == cls)) / np.sum(mask)
                class_accuracies.append(accuracy)
            else:
                class_accuracies.append(0)

        axes[0, 1].bar(range(len(unique_classes)), class_accuracies)
        axes[0, 1].set_xlabel('Class')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Accuracy by Class')
        axes[0, 1].set_xticks(range(len(unique_classes)))
        axes[0, 1].set_xticklabels(unique_classes)
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 预测分布
        axes[1, 0].hist(y_pred, bins=len(unique_classes), alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Predicted Class')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Prediction Distribution')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 真实分布
        axes[1, 1].hist(y_true, bins=len(unique_classes), alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('True Class')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('True Distribution')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图形
        if save_path is None:
            save_path = self.output_dir / f"classification_error_analysis_{model_name}.png"

        plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()

        self.logger.info(f"分类误差分析图已保存: {save_path}")
        return str(save_path)

    def plot_learning_curves(
        self,
        train_scores: List[float],
        val_scores: List[float],
        train_sizes: Optional[List[int]] = None,
        model_name: str = "Model",
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """绘制学习曲线"""

        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning("Matplotlib not available")
            return None

        if train_sizes is None:
            train_sizes = list(range(1, len(train_scores) + 1))

        # 绘制图形
        fig, ax = plt.subplots(figsize=self.config.figure_size)

        ax.plot(train_sizes, train_scores, 'o-', label='Training Score', linewidth=2)
        ax.plot(train_sizes, val_scores, 'o-', label='Validation Score', linewidth=2)

        ax.set_xlabel('Training Set Size')
        ax.set_ylabel('Score')
        ax.set_title(f'Learning Curves - {model_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 保存图形
        if save_path is None:
            save_path = self.output_dir / f"learning_curves_{model_name}.png"

        plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()

        self.logger.info(f"学习曲线已保存: {save_path}")
        return str(save_path)

    def plot_model_comparison(
        self,
        model_results: Dict[str, Dict[str, Any]],
        metric_name: str = "score",
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """绘制模型比较图"""

        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning("Matplotlib not available")
            return None

        model_names = list(model_results.keys())
        scores = [model_results[name].get(metric_name, 0) for name in model_names]
        errors = [model_results[name].get('std', 0) for name in model_names]

        # 绘制图形
        fig, ax = plt.subplots(figsize=(12, 6))

        bars = ax.bar(model_names, scores, yerr=errors, capsize=5, alpha=0.7)

        # 高亮最佳模型
        best_idx = scores.index(max(scores))
        bars[best_idx].set_color('gold')

        ax.set_xlabel('Models')
        ax.set_ylabel(metric_name.title())
        ax.set_title(f'Model Comparison - {metric_name.title()}')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)

        # 添加数值标签
        for bar, score, error in zip(bars, scores, errors):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + error + 0.01,
                   f'{score:.3f}', ha='center', va='bottom')

        # 保存图形
        if save_path is None:
            save_path = self.output_dir / f"model_comparison_{metric_name}.png"

        plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()

        self.logger.info(f"模型比较图已保存: {save_path}")
        return str(save_path)

    def create_visualization_dashboard(
        self,
        experiment_data: Dict[str, Any],
        model_name: str = "Model",
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """创建可视化仪表板"""

        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning("Matplotlib not available")
            return None

        # 创建大图
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(f'Visualization Dashboard - {model_name}', fontsize=16)

        # 根据可用数据创建子图
        subplot_count = 0

        # 1. 如果有预测概率，绘制ROC曲线
        if 'y_prob' in experiment_data and 'y_true' in experiment_data:
            subplot_count += 1
            ax1 = plt.subplot(2, 3, subplot_count)
            self._plot_roc_subplot(ax1, experiment_data['y_true'], experiment_data['y_prob'])

        # 2. 如果有预测值，绘制混淆矩阵
        if 'y_pred' in experiment_data and 'y_true' in experiment_data:
            subplot_count += 1
            ax2 = plt.subplot(2, 3, subplot_count)
            self._plot_confusion_matrix_subplot(ax2, experiment_data['y_true'], experiment_data['y_pred'])

        # 3. 如果有特征重要性，绘制特征重要性图
        if 'feature_importance' in experiment_data:
            subplot_count += 1
            ax3 = plt.subplot(2, 3, subplot_count)
            self._plot_feature_importance_subplot(ax3, experiment_data['feature_importance'])

        # 4. 如果有分数，绘制分数分布
        if 'scores' in experiment_data:
            subplot_count += 1
            ax4 = plt.subplot(2, 3, subplot_count)
            self._plot_scores_distribution_subplot(ax4, experiment_data['scores'])

        # 5. 如果有学习曲线数据
        if 'train_scores' in experiment_data and 'val_scores' in experiment_data:
            subplot_count += 1
            ax5 = plt.subplot(2, 3, subplot_count)
            self._plot_learning_curves_subplot(ax5, experiment_data['train_scores'], experiment_data['val_scores'])

        plt.tight_layout()

        # 保存图形
        if save_path is None:
            save_path = self.output_dir / f"dashboard_{model_name}.png"

        plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()

        self.logger.info(f"可视化仪表板已保存: {save_path}")
        return str(save_path)

    def _plot_roc_subplot(self, ax, y_true, y_prob):
        """ROC曲线子图"""
        if SKLEARN_AVAILABLE:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc_score = roc_auc_score(y_true, y_prob)
            ax.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc_score:.3f}')
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')
            ax.legend()
            ax.grid(True, alpha=0.3)

    def _plot_confusion_matrix_subplot(self, ax, y_true, y_pred):
        """混淆矩阵子图"""
        if SKLEARN_AVAILABLE:
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title('Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')

    def _plot_feature_importance_subplot(self, ax, feature_importance):
        """特征重要性子图"""
        importance = np.array(feature_importance)
        sorted_indices = np.argsort(importance)[::-1][:10]  # 前10个
        sorted_importance = importance[sorted_indices]

        ax.barh(range(len(sorted_importance)), sorted_importance)
        ax.set_xlabel('Importance')
        ax.set_title('Top 10 Feature Importance')
        ax.set_yticks(range(len(sorted_importance)))
        ax.set_yticklabels([f'Feature_{i}' for i in sorted_indices])

    def _plot_scores_distribution_subplot(self, ax, scores):
        """分数分布子图"""
        ax.hist(scores, bins=10, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Score Distribution')
        ax.grid(True, alpha=0.3)

    def _plot_learning_curves_subplot(self, ax, train_scores, val_scores):
        """学习曲线子图"""
        train_sizes = list(range(1, len(train_scores) + 1))
        ax.plot(train_sizes, train_scores, 'o-', label='Training', linewidth=2)
        ax.plot(train_sizes, val_scores, 'o-', label='Validation', linewidth=2)
        ax.set_xlabel('Training Set Size')
        ax.set_ylabel('Score')
        ax.set_title('Learning Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)

# 全局可视化器实例
advanced_visualizer = AdvancedVisualizer()

__all__ = ["'logger'", "'VisualizationConfig'", "'AdvancedVisualizer'", "'advanced_visualizer'"]
