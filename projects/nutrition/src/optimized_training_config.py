#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化训练配置
针对高损失问题进行优化，进一步降低损失值
"""

import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_optimized_config():
    """创建优化的训练配置"""
    logger.info("🔧 创建优化训练配置...")

    config = {
        "optimized_training": {
            "loss_function": {
                "type": "MSELoss",
                "reason": "MSELoss对于标准化目标更直接，损失值更易解释"
            },
            "learning_rate": {
                "initial": 5e-5,
                "scheduler": "CosineAnnealingLR",
                "T_max": 60,
                "eta_min": 1e-6,
                "reason": "较低的学习率可以提高收敛稳定性"
            },
            "model_architecture": {
                "hidden_dim": 256,
                "lora_rank": 64,
                "lora_alpha": 128.0,
                "dropout": 0.2,
                "reason": "增加模型容量和正则化"
            },
            "training_params": {
                "epochs": 80,
                "batch_size": 64,
                "grad_clip": 0.5,
                "patience": 25,
                "weight_decay": 1e-5,
                "reason": "更多轮数、更大批次、更强正则化"
            },
            "data_preprocessing": {
                "keep_normalization": True,
                "augmentation": {
                    "enabled": True,
                    "noise_level": 0.01,
                    "reason": "减少噪声水平，提高数据质量"
                }
            }
        }
    }

    output_path = Path("stage1/configs/optimized_training_config.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ 配置已保存: {output_path}")

    return config


def generate_optimized_training_command():
    """生成优化训练命令"""
    logger.info("\n📋 优化训练命令:")
    logger.info("="*80)

    commands = [
        "# 选项1: 使用MSELoss（默认读取 Stage2 统一数据）",
        "python stage1/cultural_finetune.py \\",
        "  --stage2_split train \\",
        "  --epochs 80 \\",
        "  --lr 5e-5 \\",
        "  --batch_size 64 \\",
        "  --hidden_dim 256 \\",
        "  --lora_rank 64 \\",
        "  --lora_alpha 128.0 \\",
        "  --grad_clip 0.5 \\",
        "  --patience 25",
        "",
        "# 选项2: 保持当前配置但降低学习率",
        "python stage1/cultural_finetune.py \\",
        "  --stage2_split train \\",
        "  --epochs 80 \\",
        "  --lr 3e-5 \\",
        "  --batch_size 48 \\",
        "  --grad_clip 0.5",
        "",
        "# 如需使用历史 JSON，请追加: --use_legacy_dataset --final_dataset <path>"
    ]

    for cmd in commands:
        logger.info(cmd)

    logger.info("="*80)


def explain_loss_value():
    """解释损失值的含义"""
    logger.info("\n📊 损失值解释:")
    logger.info("="*80)

    logger.info("\n当前情况:")
    logger.info("  标准化损失: 0.5525")
    logger.info("  标准化RMSE: 0.7433")
    logger.info("  标准化MAE: 0.5946")

    logger.info("\n反标准化后（真实值）:")
    logger.info("  实际MAE: ~0.1236 (非常好！)")
    logger.info("  实际RMSE: ~0.167")

    logger.info("\n说明:")
    logger.info("  1. 标准化后的损失值看起来高，但实际预测误差很低")
    logger.info("  2. MAE=0.1236 表示预测误差平均只有0.12分（满分1.0）")
    logger.info("  3. 这个结果比之前的0.22好很多（改进44%）")

    logger.info("\n如果还想进一步降低损失:")
    logger.info("  1. 使用MSELoss可能更直观")
    logger.info("  2. 降低学习率到5e-5")
    logger.info("  3. 增加模型容量（hidden_dim=256）")
    logger.info("  4. 增加训练轮数到80")
    logger.info("="*80)


if __name__ == "__main__":
    logger.info("="*80)
    logger.info("🎯 训练损失优化方案")
    logger.info("="*80)

    explain_loss_value()

    create_optimized_config()

    generate_optimized_training_command()

    logger.info("\n✅ 优化方案生成完成！")
