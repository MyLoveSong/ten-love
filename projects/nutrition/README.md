# Nutrition: 营养健康多任务模型

## 📋 项目概述

本项目专注于营养健康多任务模型的训练、评估与优化，包括：
- 扩展高分菜品训练集
- 分层校准机制
- 温度缩放与校准
- 多尺度特征融合
- 高级数据增强
- 细粒度特征提取
- 系统化A/B消融实验
- 对抗训练增强

## 📁 目录结构

```
nutrition/
├── src/                    # 训练脚本、评估工具、数据处理
│   ├── cultural_finetune*.py    # 文化适配微调
│   ├── enhance*.py              # 模型增强
│   ├── train_integrated_model.py # 集成训练
│   ├── evaluate*.py             # 模型评估
│   ├── optimize*.py             # 优化脚本
│   ├── scripts/                # 辅助脚本
│   └── tests/                  # 测试文件
├── data/                    # 营养数据集
│   ├── external_datasets/    # 外部数据集（67个JSON）
│   ├── preprocessed/         # 预处理后数据
│   └── high_health_dishes_expanded.json
├── configs/                  # 训练配置文件
├── outputs/                  # 训练结果和模型检查点
└── README.md                # 本文件
```

## 🚀 快速开始

### 训练集成模型
```bash
cd system/projects/nutrition
python src/train_integrated_model.py
```

### 运行评估
```bash
python src/evaluate_integrated_model.py
```

### 文化微调
```bash
python src/cultural_finetune.py
```

## 📊 评估指标

- **MAE**: 平均绝对误差
- **MSE**: 均方误差
- **R²**: 决定系数
- **临床指南遵循率**: ADA指南符合度

## 📖 详细文档

- `IMPROVEMENTS_FINAL_SUMMARY.md` - 模型改进最终总结
- `COMPREHENSIVE_EVALUATION_REPORT.md` - 综合评估报告
- `VERIFICATION_RESULTS.md` - 模型改进效果验证报告
