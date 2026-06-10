# Recommendation: 文化感知推荐系统

## 📋 项目概述

本项目是文化感知的多模态个性化推荐系统，面向顶会（NeurIPS/ICML）和顶刊（AIIM）发表。

核心创新：
- **文化适配门控机制**: 动态适应不同文化背景的用户偏好
- **多模态融合跨注意力**: 分层融合图像和文本特征
- **临床约束优化层**: 集成ADA糖尿病指南

## 📁 目录结构

```
recommendation/
├── configs/           # 推荐系统配置
│   └── stage1_bridge_data.json
├── models/           # 预训练模型
│   └── pretrained/vit-base-food101/  # ViT-Food101预训练权重
├── results/          # 实验结果
│   └── constraint_gate_diagnosis.json
└── README.md        # 本文件
```

## 🎯 评估指标

### 顶会指标 (NeurIPS/ICML)
- Recall@10 > 0.75
- Precision@10 > 0.65
- NDCG@10 > 0.80

### 顶刊指标 (AIIM)
- 临床指南遵循率 > 95%
- NNT < 1.25
- 专家满意度 > 4.2/5.0

## 🔗 依赖关系

- 复用 `projects/nutrition/` 的特征提取器
- 复用 `projects/glucose/` 的血糖预测能力
- 通过 `projects/glucose/src/utils/stage2_evaluation_sync.py` 同步训练指标

## 📖 详细文档

请参考 `system/README_stage2.md`（位于 system 根目录）。
