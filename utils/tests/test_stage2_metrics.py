#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage2 推荐指标单元测试

验证 RecommendationMetrics 的核心 Top-K 指标计算是否与手算结果一致，
为后续顶会/顶刊实验提供数值安全网。
"""

import numpy as np
import torch

from utils.metrics import RecommendationMetrics


def test_compute_all_metrics_basic_topk() -> None:
    """
    构造一个极简二分类 Top-K 场景，校验 Recall@K / Precision@K / NDCG@K / MAP / MRR。

    设有 2 个用户、3 个候选 item：
    - user 0: 真实正样本在 index=0，模型预测正确排在第一。
    - user 1: 真实正样本在 index=0，但模型把 index=2 排在第一。
    """
    # 预测分数 [batch_size=2, num_items=3]
    preds = torch.tensor(
        [
            [0.9, 0.2, 0.1],  # user 0: 命中在 rank=1
            [0.1, 0.2, 0.9],  # user 1: 正样本排在最后
        ],
        dtype=torch.float32,
    )
    # 二值标签：每个用户只有一个正样本 (index=0)
    targets = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    metrics = RecommendationMetrics(k_values=[1, 3])
    out = metrics.compute_all_metrics(predictions=preds, targets=targets)

    # 手算:
    # user0: Recall@1=1, Recall@3=1, Precision@1=1
    # user1: Recall@1=0, Recall@3=1, Precision@1=0
    # => Recall@1 = 0.5, Recall@3 = 1.0, Precision@1 = 0.5
    assert np.isclose(out["recall@1"], 0.5, atol=1e-6)
    assert np.isclose(out["recall@3"], 1.0, atol=1e-6)
    assert np.isclose(out["precision@1"], 0.5, atol=1e-6)

    # NDCG:
    # user0: 正样本在 rank=1 -> NDCG@1 = 1
    # user1: 正样本在 rank=3 -> 对 NDCG@1 贡献为 0
    # => NDCG@1 ≈ 0.5
    assert np.isclose(out["ndcg@1"], 0.5, atol=1e-6)

    # MAP 与 MRR:
    # user0: AP=1, RR=1
    # user1: AP=1/3, RR=1/3
    # => MAP ≈ 2/3, MRR ≈ 2/3
    assert np.isclose(out["map"], 2.0 / 3.0, atol=1e-6)
    assert np.isclose(out["mrr"], 2.0 / 3.0, atol=1e-6)


def test_legacy_recall_at_k_with_index_list() -> None:
    """
    验证兼容接口 RecommendationMetrics.recall_at_k 能正确处理
    “每个用户给出 Top-K 索引列表”的旧调用方式。
    """
    metrics = RecommendationMetrics()

    # predictions: 每个子列表表示该用户的 Top-3 排序索引
    predictions = [
        [2, 0, 1],  # user0: 正样本 index=0 出现在 rank=2
        [0, 2, 1],  # user1: 正样本 index=0 出现在 rank=1
    ]
    # targets: one-hot 形式 (2 用户 × 3 item)
    targets = [
        [0.0, 0.0, 1.0],  # user0: 正样本在 index=2
        [1.0, 0.0, 0.0],  # user1: 正样本在 index=0
    ]

    # 对 Top-1 而言:
    # user0: Top-1=2 命中 -> 1
    # user1: Top-1=0 命中 -> 1
    # => Recall@1 = 1.0
    recall_1 = metrics.recall_at_k(predictions, targets, k=1)
    assert np.isclose(recall_1, 1.0, atol=1e-6)
