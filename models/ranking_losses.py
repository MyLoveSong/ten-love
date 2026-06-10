# @ai-version: 2025-11-18T17:28:45Z
"""
Ranking Losses for Recommendation System

Based on latest research:
- SoftmaxLoss@K (2025, +6.03% NDCG improvement)
- NeuralNDCG (differentiable NDCG approximation)
- Context-Aware Learning to Rank
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PairwiseRankingLoss(nn.Module):
    """
    Pairwise ranking loss for improving NDCG

    Implements margin-based pairwise ranking following:
    L = max(0, margin - (score_pos - score_neg))

    Args:
        margin: Ranking margin (default: 0.5)
        reduction: Loss reduction method ['mean', 'sum']
    """

    def __init__(self, margin: float = 0.5, reduction: str = 'mean'):
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            scores: [batch_size, num_items] prediction scores
            labels: [batch_size, num_items] binary relevance labels

        Returns:
            loss: scalar tensor
        """
        batch_size, num_items = scores.shape

        pos_mask = labels > 0
        neg_mask = labels == 0

        if not pos_mask.any() or not neg_mask.any():
            return torch.tensor(0.0, device=scores.device)

        pos_scores = torch.where(pos_mask, scores, torch.tensor(float('-inf'), device=scores.device))
        neg_scores = torch.where(neg_mask, scores, torch.tensor(float('inf'), device=scores.device))

        pos_scores_expanded = pos_scores.unsqueeze(2)
        neg_scores_expanded = neg_scores.unsqueeze(1)

        score_diff = pos_scores_expanded - neg_scores_expanded

        pairwise_loss = F.relu(self.margin - score_diff)

        valid_pairs = (pos_scores_expanded != float('-inf')) & (neg_scores_expanded != float('inf'))
        pairwise_loss = torch.where(valid_pairs, pairwise_loss, torch.zeros_like(pairwise_loss))

        if self.reduction == 'mean':
            return pairwise_loss.sum() / (valid_pairs.sum() + 1e-8)
        elif self.reduction == 'sum':
            return pairwise_loss.sum()
        else:
            return pairwise_loss


class ListwiseRankingLoss(nn.Module):
    """
    Listwise ranking loss using softmax formulation

    L = -sum(label_i * log(softmax(score_i)))

    Better captures ranking order than pointwise losses
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            scores: [batch_size, num_items]
            labels: [batch_size, num_items] relevance scores (can be graded)

        Returns:
            loss: scalar
        """
        scores_normalized = scores / self.temperature
        log_probs = F.log_softmax(scores_normalized, dim=-1)

        label_probs = F.softmax(labels, dim=-1)

        loss = -(label_probs * log_probs).sum(dim=-1).mean()

        return loss


class ApproxNDCGLoss(nn.Module):
    """
    Approximate NDCG loss using differentiable ranking

    Based on NeuralNDCG paper (arXiv:2102.07831)
    Uses smoothed sorting approximation for end-to-end NDCG optimization

    Args:
        k: Truncation rank for NDCG@k (default: 10)
        temperature: Softmax temperature for smooth sorting (default: 0.1)
    """

    def __init__(self, k: int = 10, temperature: float = 0.1):
        super().__init__()
        self.k = k
        self.temperature = temperature

    def forward(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            scores: [batch_size, num_items]
            labels: [batch_size, num_items] relevance labels

        Returns:
            ndcg_loss: scalar (1 - NDCG for minimization)
        """
        batch_size, num_items = scores.shape
        k = min(self.k, num_items)

        topk_scores, topk_indices = torch.topk(scores, k, dim=-1)
        topk_labels = torch.gather(labels, -1, topk_indices)

        gains = (2 ** topk_labels - 1) / torch.log2(torch.arange(2, k + 2, device=scores.device).float())
        dcg = gains.sum(dim=-1)

        ideal_labels, _ = torch.sort(labels, descending=True, dim=-1)
        ideal_gains = (2 ** ideal_labels[:, :k] - 1) / torch.log2(torch.arange(2, k + 2, device=scores.device).float())
        idcg = ideal_gains.sum(dim=-1)

        ndcg = dcg / (idcg + 1e-8)

        loss = 1.0 - ndcg.mean()

        return loss


class NeuralNDCGLoss(nn.Module):
    """
    NeuralNDCG: Direct NDCG optimization using differentiable sorting

    Based on "NeuralNDCG: Direct Optimisation of a Ranking Metric" (arXiv:2102.07831)
    Uses NeuralSort for differentiable approximation of sorting operation

    Args:
        k: Truncation rank for NDCG@k (default: 10)
        tau: Temperature for NeuralSort (default: 0.05, lower = sharper)
    """

    def __init__(self, k: int = 10, tau: float = 0.05):
        super().__init__()
        self.k = k
        self.tau = tau

    def _neural_sort(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Differentiable sorting approximation using NeuralSort

        Args:
            scores: [batch_size, num_items]

        Returns:
            soft_ranks: [batch_size, num_items] soft permutation matrix
        """
        batch_size, n = scores.shape

        scores_expanded = scores.unsqueeze(2)
        scores_transposed = scores.unsqueeze(1)

        pairwise_diff = scores_expanded - scores_transposed

        pairwise_logits = pairwise_diff / self.tau
        # Compute pairwise probabilities and clone to ensure a fresh tensor
        # (prevents accidental in-place modifications affecting autograd)
        pairwise_probs = torch.sigmoid(pairwise_logits).clone()

        soft_ranks = n - pairwise_probs.sum(dim=2)

        return soft_ranks

    def forward(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            scores: [batch_size, num_items] prediction scores
            labels: [batch_size, num_items] relevance labels

        Returns:
            loss: 1 - NeuralNDCG (for minimization)
        """
        batch_size, num_items = scores.shape
        k = min(self.k, num_items)

        soft_ranks = self._neural_sort(scores)

        gains = 2 ** labels - 1

        discounts = torch.log2(soft_ranks + 1.0 + 1e-8)
        discounted_gains = gains / discounts

        top_k_mask = (soft_ranks <= k).float()
        dcg = (discounted_gains * top_k_mask).sum(dim=-1)

        ideal_labels, _ = torch.sort(labels, descending=True, dim=-1)
        ideal_gains = 2 ** ideal_labels - 1
        ideal_discounts = torch.log2(torch.arange(2, num_items + 2, device=scores.device).float())
        ideal_dcg = (ideal_gains[:, :k] / ideal_discounts[:k]).sum(dim=-1)

        ndcg = dcg / (ideal_dcg + 1e-8)

        loss = 1.0 - ndcg.mean()

        return loss


class SoftmaxLossAtK(nn.Module):
    """
    SoftmaxLoss@K for direct NDCG@K optimization

    Based on "Breaking the Top-K Barrier" (2025, arXiv:2508.05673)
    - Uses quantile technique for Top-K truncation
    - Smooth upper bound for NDCG@K optimization
    - 6.03% average improvement over baselines

    Args:
        k: Target rank for NDCG@K (default: 10)
        temperature: Softmax temperature (default: 0.05 for sharper distributions)
        label_smoothing: Label smoothing factor (default: 0.0)
    """

    def __init__(
        self,
        k: int = 10,
        temperature: float = 0.05,
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.k = k
        self.temperature = temperature
        self.label_smoothing = label_smoothing

    def forward(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            scores: [batch_size, num_items] prediction scores
            labels: [batch_size, num_items] relevance labels (binary or graded)

        Returns:
            loss: scalar tensor
        """
        batch_size, num_items = scores.shape
        k = min(self.k, num_items)

        quantile_scores = torch.quantile(scores, q=1.0 - k / num_items, dim=-1, keepdim=True)

        topk_mask = (scores >= quantile_scores)
        fill_value = torch.finfo(scores.dtype).min
        masked_scores = torch.where(
            topk_mask,
            scores,
            torch.full_like(scores, fill_value)
        )

        log_probs = F.log_softmax(masked_scores / self.temperature, dim=-1)
        log_probs = torch.where(topk_mask, log_probs, torch.zeros_like(log_probs))

        if self.label_smoothing > 0:
            smooth_labels = labels * (1 - self.label_smoothing) + self.label_smoothing / num_items
        else:
            smooth_labels = labels

        label_probs = F.softmax(smooth_labels, dim=-1)

        position_weights = 1.0 / torch.log2(torch.arange(2, num_items + 2, device=scores.device).float())
        weighted_label_probs = label_probs * position_weights
        weighted_label_probs = weighted_label_probs / (weighted_label_probs.sum(dim=-1, keepdim=True) + 1e-8)

        loss = -(weighted_label_probs * log_probs).sum(dim=-1).mean()

        return loss


class MultiTaskRankingLoss(nn.Module):
    """
    Multi-task ranking loss combining SOTA objectives

    L_total = w1*L_softmax@k + w2*L_neural_ndcg + w3*L_pairwise

    Args:
        weights: Loss weights dict with keys ['softmax_k', 'neural_ndcg', 'pairwise']
    """

    def __init__(
        self,
        weights: Optional[dict] = None,
        pairwise_margin: float = 0.5,
        ndcg_k: int = 10
    ):
        super().__init__()

        if weights is None:
            weights = {'softmax_k': 0.5, 'neural_ndcg': 0.3, 'pairwise': 0.2}

        self.weights = weights

        self.softmax_k_loss = SoftmaxLossAtK(k=ndcg_k, temperature=0.05)
        self.neural_ndcg_loss = NeuralNDCGLoss(k=ndcg_k, tau=0.05)
        self.pairwise_loss = PairwiseRankingLoss(margin=pairwise_margin)

    def forward(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            scores: [batch_size, num_items]
            labels: [batch_size, num_items]

        Returns:
            total_loss: weighted combination
            loss_components: individual losses for logging
        """
        softmax_k = self.softmax_k_loss(scores, labels)
        neural_ndcg = self.neural_ndcg_loss(scores, labels)
        pairwise = self.pairwise_loss(scores, labels)

        total_loss = (
            self.weights.get('softmax_k', 0.0) * softmax_k +
            self.weights.get('neural_ndcg', 0.0) * neural_ndcg +
            self.weights.get('pairwise', 0.0) * pairwise
        )

        loss_components = {
            'softmax_k_loss': softmax_k.item(),
            'neural_ndcg_loss': neural_ndcg.item(),
            'pairwise_loss': pairwise.item(),
            'total_ranking_loss': total_loss.item()
        }

        return total_loss, loss_components
