#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage2 评估指标模块
实现顶会+顶刊双重评估体系
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from sklearn.metrics import ndcg_score, average_precision_score
import logging
import math

logger = logging.getLogger(__name__)


class RecommendationMetrics:
    """
    推荐系统评估指标

    顶会指标 (NeurIPS/ICML):
    - Recall@K
    - Precision@K
    - NDCG@K
    - MAP (Mean Average Precision)
    - MRR (Mean Reciprocal Rank)

    顶刊指标 (AIIM):
    - Clinical Guideline Compliance Rate
    - NNT (Number Needed to Treat)
    - Expert Satisfaction Score
    - Adherence Prediction Accuracy
    """

    def __init__(self, k_values: List[int] = [5, 10, 20]):
        """
        Args:
            k_values: 评估的K值列表
        """
        self.k_values = k_values
        logger.info(f"RecommendationMetrics initialized with k_values={k_values}")

    def compute_recall_at_k(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        k: int,
        num_items: Optional[torch.Tensor] = None
    ) -> float:
        """
        计算Recall@K

        Args:
            predictions: [batch_size, num_items] 预测分数
            targets: [batch_size, num_items] 真实标签 (0/1)
            k: Top-K
            num_items: [batch_size] 每个用户的实际候选项目数 (可选)

        Returns:
            recall_at_k: Recall@K分数
        """
        batch_size = predictions.size(0)

        # 计算每个样本的Recall
        recalls = []
        for i in range(batch_size):
            pred_scores = predictions[i]  # [num_items]
            true_labels = targets[i]      # [num_items]

            # 确定这个用户的有效项目数
            if num_items is not None:
                valid_items = min(int(num_items[i].item()), len(pred_scores))
            else:
                valid_items = len(pred_scores)

            # 只考虑有效项目
            valid_pred_scores = pred_scores[:valid_items]
            valid_true_labels = true_labels[:valid_items]

            # 获取真实正样本
            true_positives = valid_true_labels.nonzero(as_tuple=True)[0]
            if len(true_positives) == 0:
                continue

            # 获取Top-K预测 (不超过有效项目数)
            effective_k = min(k, valid_items)
            _, top_k_indices = torch.topk(valid_pred_scores, effective_k)

            # 计算Top-K中命中的正样本数
            hits = sum(idx.item() in true_positives for idx in top_k_indices)
            recall = hits / len(true_positives)
            recalls.append(recall)

        return float(np.mean(recalls)) if recalls else 0.0

    def compute_precision_at_k(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        k: int
    ) -> float:
        """
        计算Precision@K

        Args:
            predictions: [batch_size, num_items] 预测分数
            targets: [batch_size, num_items] 真实标签 (0/1)
            k: Top-K

        Returns:
            precision_at_k: Precision@K分数
        """
        batch_size = predictions.size(0)

        # 获取Top-K预测
        _, top_k_indices = torch.topk(predictions, k, dim=1)

        # 计算每个样本的Precision
        precisions = []
        for i in range(batch_size):
            # 获取真实正样本
            true_positives = targets[i].nonzero(as_tuple=True)[0]

            # 计算Top-K中命中的正样本数
            hits = sum(idx.item() in true_positives for idx in top_k_indices[i])
            precision = hits / k
            precisions.append(precision)

        return np.mean(precisions)

    def compute_ndcg_at_k(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        k: int
    ) -> float:
        """
        计算NDCG@K (Normalized Discounted Cumulative Gain)

        Args:
            predictions: [batch_size, num_items] 预测分数
            targets: [batch_size, num_items] 真实标签 (0/1)
            k: Top-K

        Returns:
            ndcg_at_k: NDCG@K分数
        """
        batch_size = predictions.size(0)
        ndcg_scores = []

        for i in range(batch_size):
            pred_scores = predictions[i]  # [num_items]
            true_labels = targets[i]      # [num_items]

            # 获取Top-K预测的索引
            _, top_k_indices = torch.topk(pred_scores, k)

            # 计算DCG: 相关性 / log2(rank + 1)
            dcg = 0.0
            for rank, item_idx in enumerate(top_k_indices):
                relevance = true_labels[item_idx].item()
                dcg += relevance / math.log2(rank + 2)  # rank从1开始，所以+2

            # 计算IDCG (理想情况下的DCG)
            # 按相关性降序排列，取前K个
            sorted_relevances = torch.sort(true_labels, descending=True)[0][:k]
            idcg = 0.0
            for rank, relevance in enumerate(sorted_relevances):
                idcg += relevance.item() / math.log2(rank + 2)

            # 计算NDCG
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_scores.append(ndcg)

        return float(np.mean(ndcg_scores))

    def compute_map(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> float:
        """
        计算MAP (Mean Average Precision)

        Args:
            predictions: [batch_size, num_items] 预测分数
            targets: [batch_size, num_items] 真实标签 (0/1)

        Returns:
            map_score: MAP分数
        """
        predictions_np = predictions.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()

        aps = []
        for i in range(len(predictions_np)):
            if targets_np[i].sum() > 0:
                ap = average_precision_score(targets_np[i], predictions_np[i])
                aps.append(ap)

        return np.mean(aps) if aps else 0.0

    def compute_mrr(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> float:
        """
        计算MRR (Mean Reciprocal Rank)

        Args:
            predictions: [batch_size, num_items] 预测分数
            targets: [batch_size, num_items] 真实标签 (0/1)

        Returns:
            mrr_score: MRR分数
        """
        batch_size = predictions.size(0)

        # 按预测分数排序
        _, sorted_indices = torch.sort(predictions, descending=True, dim=1)

        reciprocal_ranks = []
        for i in range(batch_size):
            # 找到第一个正样本的排名
            true_positives = targets[i].nonzero(as_tuple=True)[0]
            if len(true_positives) == 0:
                continue

            for rank, idx in enumerate(sorted_indices[i], 1):
                if idx.item() in true_positives:
                    reciprocal_ranks.append(1.0 / rank)
                    break

        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    def compute_confidence_interval(self, values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        if not values:
            return 0.0, 0.0
        mean = float(np.mean(values))
        if len(values) == 1:
            return mean, mean
        std = float(np.std(values))
        stderr = std / math.sqrt(len(values))
        margin = 1.96 * stderr if confidence == 0.95 else stderr
        return mean - margin, mean + margin

    def compute_clinical_guideline_compliance(
        self,
        nutrition_profiles: torch.Tensor,
        clinical_constraint_layer: 'ClinicalConstraintLayer'
    ) -> float:
        """
        计算临床指南遵循率 (顶刊核心指标)

        Args:
            nutrition_profiles: [batch_size, 10] 营养档案
            clinical_constraint_layer: 临床约束层

        Returns:
            compliance_rate: 遵循率 [0, 1]
        """
        return clinical_constraint_layer.compute_guideline_compliance_rate(nutrition_profiles)

    def compute_nnt(
        self,
        predicted_improvements: torch.Tensor,
        threshold: float = 0.5
    ) -> float:
        """
        计算NNT (Number Needed to Treat)

        NNT = 1 / (受益率)
        受益率 = 预测改善比例

        Args:
            predicted_improvements: [batch_size] 预测的健康改善分数
            threshold: 改善阈值

        Returns:
            nnt: NNT值 (越小越好,理想值接近1.0)
        """
        improvements = (predicted_improvements > threshold).float()
        benefit_rate = improvements.mean().item()

        if benefit_rate > 0:
            nnt = 1.0 / benefit_rate
        else:
            nnt = float('inf')

        return nnt

    def compute_all_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        nutrition_profiles: Optional[torch.Tensor] = None,
        health_impacts: Optional[torch.Tensor] = None,
        clinical_constraint_layer: Optional['ClinicalConstraintLayer'] = None,
        k_for_compliance: int = 10,
        user_cultures: Optional[List[Any]] = None,
        recommended_cultures: Optional[List[List[Any]]] = None,
        stage1_feature_importance: Optional[Dict[str, float]] = None,
        stage2_feature_importance: Optional[Dict[str, float]] = None,
        shap_top_k: int = 5,
        num_items: Optional[torch.Tensor] = None,
        culture_match_rate: Optional[float] = None
    ) -> Dict[str, float]:
        """
        计算所有评估指标

        Args:
            predictions: [batch_size, num_items] 预测分数
            targets: [batch_size, num_items] 真实标签
            nutrition_profiles: [batch_size, num_items, 10] 或 [batch_size * num_items, 10] 营养档案 (可选)
            health_impacts: [batch_size] 健康影响预测 (可选)
            clinical_constraint_layer: 临床约束层 (可选)
            k_for_compliance: 用于计算guideline compliance的Top-K值 (默认10)

        Returns:
            metrics: 所有评估指标
        """
        metrics = {}

        # 顶会指标
        for k in self.k_values:
            metrics[f'recall@{k}'] = self.compute_recall_at_k(predictions, targets, k, num_items)
            metrics[f'precision@{k}'] = self.compute_precision_at_k(predictions, targets, k)
            metrics[f'ndcg@{k}'] = self.compute_ndcg_at_k(predictions, targets, k)

        metrics['map'] = self.compute_map(predictions, targets)
        metrics['mrr'] = self.compute_mrr(predictions, targets)

        # 顶刊指标 (如果提供)
        if nutrition_profiles is not None and clinical_constraint_layer is not None:
            # 修复: 只对Top-K推荐结果计算guideline compliance
            # 获取Top-K推荐的索引
            _, top_k_indices = torch.topk(predictions, k=k_for_compliance, dim=1)  # [batch_size, k]

            # 处理nutrition_profiles的形状
            batch_size = predictions.size(0)
            num_items = predictions.size(1)

            if nutrition_profiles.dim() == 2:
                # 形状为 [batch_size * num_items, 10]，需要reshape
                nutrition_profiles = nutrition_profiles.view(batch_size, num_items, -1)
            elif nutrition_profiles.dim() == 3:
                # 形状为 [batch_size, num_items, 10]，直接使用
                pass
            else:
                raise ValueError(f"Unexpected nutrition_profiles shape: {nutrition_profiles.shape}")

            # 提取Top-K推荐的nutrition profiles
            top_k_nutrition = []
            for i in range(batch_size):
                top_k_nutrition.append(nutrition_profiles[i, top_k_indices[i]])  # [k, 10]
            top_k_nutrition = torch.cat(top_k_nutrition, dim=0)  # [batch_size * k, 10]

            # 关键修复: 过滤掉零填充的items (所有营养值为0的items)
            # 检查是否有有效的营养数据 (至少有一个非零值)
            # 使用更宽松的阈值，因为营养值可能很小
            valid_mask = top_k_nutrition.abs().sum(dim=1) > 1e-3  # [batch_size * k]

            if valid_mask.sum() > 0:
                # 只对有效items计算compliance
                valid_nutrition = top_k_nutrition[valid_mask]  # [num_valid, 10]

                # 确保营养档案在正确的设备上
                if valid_nutrition.device != next(clinical_constraint_layer.parameters()).device:
                    valid_nutrition = valid_nutrition.to(next(clinical_constraint_layer.parameters()).device)

                # 计算compliance
                compliance = self.compute_clinical_guideline_compliance(
                    valid_nutrition,
                    clinical_constraint_layer
                )

                # 调试信息 (仅在需要时启用)
                if compliance == 0.0 and valid_mask.sum() > 5:  # 如果有足够样本但compliance为0，记录警告
                    import logging
                    logging.warning(
                        f"Guideline compliance is 0.0 for {valid_mask.sum()} valid items. "
                        f"Nutrition profile stats: min={valid_nutrition.min().item():.4f}, "
                        f"max={valid_nutrition.max().item():.4f}, "
                        f"mean={valid_nutrition.mean().item():.4f}"
                    )

                metrics['guideline_compliance'] = compliance
            else:
                # 如果没有有效items，返回0 (不应该发生，但防止崩溃)
                import logging
                logging.warning(f"No valid nutrition profiles found in Top-{k_for_compliance} recommendations!")
                metrics['guideline_compliance'] = 0.0

        if health_impacts is not None:
            metrics['nnt'] = self.compute_nnt(health_impacts)

        if user_cultures is not None and recommended_cultures is not None:
            metrics['culture_match_rate'] = self.compute_culture_match_rate(user_cultures, recommended_cultures)

        if 'guideline_compliance' in metrics:
            metrics['clinical_constraint_satisfaction'] = metrics['guideline_compliance']

        if stage1_feature_importance and stage2_feature_importance:
            metrics['shap_alignment'] = self.compute_shap_alignment(stage1_feature_importance, stage2_feature_importance, shap_top_k)

        # 添加文化匹配率指标
        if culture_match_rate is not None:
            metrics['culture_match_rate'] = culture_match_rate

        return metrics

    def format_metrics_for_paper(
        self,
        metrics: Dict[str, float],
        target: str = 'conference'
    ) -> str:
        """
        格式化指标用于论文报告

        Args:
            metrics: 评估指标字典
            target: 目标类型 ['conference', 'journal']

        Returns:
            formatted_str: 格式化后的字符串
        """
        if target == 'conference':
            # 顶会格式 (强调方法性能)
            formatted = "### Conference Metrics (NeurIPS/ICML)\n\n"
            formatted += f"- **Recall@10**: {metrics.get('recall@10', 0.0):.4f}\n"
            formatted += f"- **Precision@10**: {metrics.get('precision@10', 0.0):.4f}\n"
            formatted += f"- **NDCG@10**: {metrics.get('ndcg@10', 0.0):.4f}\n"
            formatted += f"- **MAP**: {metrics.get('map', 0.0):.4f}\n"
            formatted += f"- **MRR**: {metrics.get('mrr', 0.0):.4f}\n"

        elif target == 'journal':
            # 顶刊格式 (强调临床价值)
            formatted = "### Journal Metrics (AIIM)\n\n"
            formatted += f"- **Clinical Guideline Compliance**: {metrics.get('guideline_compliance', 0.0):.2%}\n"
            formatted += f"- **NNT**: {metrics.get('nnt', float('inf')):.2f}\n"
            formatted += f"- **Recall@10**: {metrics.get('recall@10', 0.0):.4f}\n"

        else:
            formatted = "Invalid target type"

        return formatted

    def compare_with_baselines(
        self,
        our_metrics: Dict[str, float],
        baselines: Dict[str, Dict[str, float]]
    ) -> str:
        """
        与baseline方法对比

        Args:
            our_metrics: 我们的方法指标
            baselines: baseline方法指标字典 {method_name: metrics}

        Returns:
            comparison_table: 对比表格(Markdown格式)
        """
        # 构建对比表格
        table = "| Method | Recall@10 | NDCG@10 | MAP | Guideline Compliance | NNT |\n"
        table += "|--------|-----------|---------|-----|----------------------|-----|\n"

        # 添加baseline
        for method_name, metrics in baselines.items():
            table += f"| {method_name} | "
            table += f"{metrics.get('recall@10', 0.0):.4f} | "
            table += f"{metrics.get('ndcg@10', 0.0):.4f} | "
            table += f"{metrics.get('map', 0.0):.4f} | "
            table += f"{metrics.get('guideline_compliance', 0.0):.2%} | "
            table += f"{metrics.get('nnt', float('inf')):.2f} |\n"

        # 添加我们的方法 (加粗)
        table += f"| **Ours** | "
        table += f"**{our_metrics.get('recall@10', 0.0):.4f}** | "
        table += f"**{our_metrics.get('ndcg@10', 0.0):.4f}** | "
        table += f"**{our_metrics.get('map', 0.0):.4f}** | "
        table += f"**{our_metrics.get('guideline_compliance', 0.0):.2%}** | "
        table += f"**{our_metrics.get('nnt', float('inf')):.2f}** |\n"

        # 计算相对改进
        table += "\n### Relative Improvements\n\n"

        for baseline_name, baseline_metrics in baselines.items():
            table += f"**vs {baseline_name}:**\n"
            for metric_name in ['recall@10', 'ndcg@10', 'map']:
                if metric_name in baseline_metrics and metric_name in our_metrics:
                    improvement = (our_metrics[metric_name] - baseline_metrics[metric_name]) / baseline_metrics[metric_name] * 100
                    table += f"- {metric_name}: {improvement:+.2f}%\n"
            table += "\n"

        return table

    # Backward compatibility aliases for legacy code
    def recall_at_k(self, predictions, targets, k: int) -> float:
        """Legacy API: use compute_recall_at_k instead

        Args:
            predictions: List of Top-K predictions [batch_size, k] or tensor [batch_size, num_items]
            targets: List of binary labels [batch_size, num_items] or tensor
            k: Top-K value
        """
        # Handle predictions as list of Top-K indices
        if isinstance(predictions, list) and len(predictions) > 0:
            if isinstance(predictions[0], list):
                # predictions: [[idx1, idx2, ...], ...] - convert to hit matrix
                batch_size = len(predictions)
                num_items = len(targets[0]) if isinstance(targets[0], list) else targets[0].size(-1)
                pred_tensor = torch.zeros(batch_size, num_items)
                for i, pred_list in enumerate(predictions):
                    for idx in pred_list[:k]:
                        if idx < num_items:
                            pred_tensor[i, idx] = 1.0
                predictions = pred_tensor
            else:
                predictions = torch.tensor(predictions)

        if isinstance(targets, list):
            targets = torch.tensor(targets, dtype=torch.float32)

        return self.compute_recall_at_k(predictions, targets, k)

    def precision_at_k(self, predictions, targets, k: int) -> float:
        """Legacy API: use compute_precision_at_k instead"""
        if isinstance(predictions, list) and len(predictions) > 0:
            if isinstance(predictions[0], list):
                batch_size = len(predictions)
                num_items = len(targets[0]) if isinstance(targets[0], list) else targets[0].size(-1)
                pred_tensor = torch.zeros(batch_size, num_items)
                for i, pred_list in enumerate(predictions):
                    for idx in pred_list[:k]:
                        if idx < num_items:
                            pred_tensor[i, idx] = 1.0
                predictions = pred_tensor
            else:
                predictions = torch.tensor(predictions)

        if isinstance(targets, list):
            targets = torch.tensor(targets, dtype=torch.float32)

        return self.compute_precision_at_k(predictions, targets, k)

    def ndcg_at_k(self, predictions, targets, k: int) -> float:
        """Legacy API: use compute_ndcg_at_k instead"""
        if isinstance(predictions, list) and len(predictions) > 0:
            if isinstance(predictions[0], list):
                batch_size = len(predictions)
                num_items = len(targets[0]) if isinstance(targets[0], list) else targets[0].size(-1)
                pred_tensor = torch.zeros(batch_size, num_items)
                for i, pred_list in enumerate(predictions):
                    for rank, idx in enumerate(pred_list[:k]):
                        if idx < num_items:
                            pred_tensor[i, idx] = k - rank
                predictions = pred_tensor
            else:
                predictions = torch.tensor(predictions)

        if isinstance(targets, list):
            targets = torch.tensor(targets, dtype=torch.float32)

        return self.compute_ndcg_at_k(predictions, targets, k)

    def mean_reciprocal_rank(self, predictions, targets) -> float:
        """Legacy API: use compute_mrr instead"""
        if isinstance(predictions, list) and len(predictions) > 0:
            if isinstance(predictions[0], list):
                batch_size = len(predictions)
                num_items = len(targets[0]) if isinstance(targets[0], list) else targets[0].size(-1)
                pred_tensor = torch.zeros(batch_size, num_items)
                for i, pred_list in enumerate(predictions):
                    for rank, idx in enumerate(pred_list):
                        if idx < num_items:
                            pred_tensor[i, idx] = 1.0 / (rank + 1)
                predictions = pred_tensor
            else:
                predictions = torch.tensor(predictions)

        if isinstance(targets, list):
            targets = torch.tensor(targets, dtype=torch.float32)

        return self.compute_mrr(predictions, targets)

    def compute_culture_match_rate(
        self,
        user_preferences: List[Any],
        recommended_cultures: List[List[Any]],
        top_k: int = 10
    ) -> float:
        total = 0
        hits = 0
        for pref, recs in zip(user_preferences, recommended_cultures):
            pref_set = set(pref) if isinstance(pref, (list, tuple, set)) else {pref}
            if not pref_set:
                continue
            if not isinstance(recs, list):
                recs = [recs]
            trimmed = recs[:top_k]
            total += 1
            if any(tag in pref_set for tag in trimmed if tag):
                hits += 1
        return hits / total if total else 0.0

    def compute_shap_alignment(
        self,
        stage1_importance: Dict[str, float],
        stage2_importance: Dict[str, float],
        top_k: int = 5
    ) -> Dict[str, float]:
        shared = set(stage1_importance.keys()) & set(stage2_importance.keys())
        if len(shared) < 2:
            return {
                'spearman': float('nan'),
                'topk_overlap': 0.0,
                'stage2_top_features': sorted(stage2_importance.items(), key=lambda kv: -kv[1])[:top_k]
            }

        rank1 = self._rank_dict({k: stage1_importance[k] for k in shared})
        rank2 = self._rank_dict({k: stage2_importance[k] for k in shared})
        n = len(shared)
        diff_sum = sum((rank1[k] - rank2[k]) ** 2 for k in shared)
        spearman = 1 - (6 * diff_sum) / (n * (n * n - 1))
        overlap = self._topk_overlap(stage1_importance, stage2_importance, top_k)
        return {
            'spearman': float(spearman),
            'topk_overlap': float(overlap),
            'stage2_top_features': sorted(stage2_importance.items(), key=lambda kv: -kv[1])[:top_k]
        }

    @staticmethod
    def _rank_dict(values: Dict[str, float]) -> Dict[str, float]:
        sorted_items = sorted(values.items(), key=lambda kv: (-kv[1], kv[0]))
        ranks: Dict[str, float] = {}
        current_rank = 1
        prev_value = None
        for idx, (key, value) in enumerate(sorted_items, start=1):
            if prev_value is None or value != prev_value:
                current_rank = idx
            ranks[key] = current_rank
            prev_value = value
        return ranks

    @staticmethod
    def _topk_overlap(stage1_importance: Dict[str, float], stage2_importance: Dict[str, float], top_k: int) -> float:
        top_stage1 = {k for k, _ in sorted(stage1_importance.items(), key=lambda kv: -kv[1])[:top_k]}
        top_stage2 = {k for k, _ in sorted(stage2_importance.items(), key=lambda kv: -kv[1])[:top_k]}
        if not top_stage1:
            return 0.0
        return len(top_stage1 & top_stage2) / min(len(top_stage1), top_k)
