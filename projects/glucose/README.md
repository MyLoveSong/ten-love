# Glucose: 血糖预测训练系统

## 📋 项目概述

本项目专注于血糖预测模型的训练与评估，包括：
- GluFormer（LSTM-GRU时序感知模型）
- 多模态数据融合
- LoRA个性化微调
- 混合专家（MoE）机制
- 异常检测与漂移监控

## 📁 目录结构

```
glucose/
├── src/                    # 核心源码
│   ├── core/              # GluFormer、增强版模型、MoE组件、LoRA适配器
│   ├── utils/             # 训练工具、评估、数据处理
│   ├── data_processing/   # PhysioNet数据预处理
│   ├── data_sources/      # 数据下载与清洗
│   ├── monitoring/        # 漂移检测、异常检测
│   ├── models/           # 微调模型
│   ├── ensemble/          # 集成策略
│   ├── analysis/          # 分析工具
│   ├── tools/             # 辅助工具
│   └── main.py            # 主入口
├── data/                  # 血糖数据集
│   ├── physionet_big_ideas/  # Big Ideas血糖数据
│   ├── physionet_cgmacros/   # CGM宏数据
│   ├── cleaned_dataset/       # 清洗后数据
│   └── augmented_dataset/     # 增强数据集
├── configs/               # YAML配置文件
├── outputs/               # 训练结果、模型检查点
├── suya_model/           # 历史训练模型
└── README.md            # 本文件
```

## 🚀 快速开始

### 检查血糖数据
```bash
cd system/projects/glucose
python src/main.py --check-data
```

### 运行基础训练
```bash
python src/run_glucose_training.py \
  --inputs data/cleaned_dataset/unified_cleaned_glucose.json \
  --in_len 12 \
  --out_len 6
```

### 运行LoRA个性化微调
```bash
python src/run_glucose_training.py \
  --inputs data/cleaned_dataset/unified_cleaned_glucose.json \
  --in_len 12 \
  --out_len 6 \
  --run_lora \
  --use_patient_ids
```

### 同步训练指标到推荐系统
```bash
python src/utils/stage2_evaluation_sync.py \
  --training "suya_model/final_model/1/training_results.json" \
  --evaluation ../recommendation/results/models/evaluation_results.json
```

## 📊 性能目标

| 指标 | 目标值 | 当前值 |
|------|--------|--------|
| MAE | < 5.0 mg/dL | 4.84 mg/dL |
| RMSE | < 7.0 mg/dL | - |
| R² | > 0.85 | 0.856 |

## 📖 详细文档

- `ENHANCED_GLUCOSE_PREDICTION_FINAL_REPORT.md` - 血糖预测最终报告
- `FINAL_COMPARISON_REPORT.md` - 模型对比报告
