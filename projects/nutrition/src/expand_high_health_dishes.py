#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扩展高分健康菜品训练集
从营养数据库中筛选高健康价值菜品（健康分>=0.8），并生成扩展的训练数据
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def load_nutrition_database() -> List[Dict]:
    """加载营养数据库"""
    nutrition_path = project_root / "data_nutrition" / "nutrition_db_public_candidate.json"
    if not nutrition_path.exists():
        print(f"警告: 营养数据库文件不存在: {nutrition_path}")
        return []

    with open(nutrition_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def filter_high_health_dishes(data: List[Dict], threshold: float = 0.8) -> List[Dict]:
    """筛选高健康价值菜品"""
    high_health = []
    for item in data:
        health_score = item.get('health_score', 0)
        # 过滤掉明显不合理的高分（如所有都是1.0的情况）
        # 同时检查营养数据是否合理
        calories = item.get('calories', 0)
        sodium = item.get('sodium', 0)
        fat = item.get('fat', 0)

        # 基本合理性检查：卡路里应该在合理范围内，钠不应该过高
        if health_score >= threshold and 50 <= calories <= 800 and sodium <= 2000:
            high_health.append(item)
    return high_health

def create_training_samples(high_health_dishes: List[Dict]) -> List[Dict]:
    """创建训练样本格式"""
    training_samples = []

    # 已有的高分菜品（带期望值）
    known_dishes = {
        "清蒸鲈鱼": {"expected_health": 0.85, "expected_taste": 0.85, "priority": 1},
        "蒸蛋羹": {"expected_health": 0.75, "expected_taste": 0.75, "priority": 1},
        "清炒时蔬": {"expected_health": 0.9, "expected_taste": 0.7, "priority": 1},
        "白切鸡": {"expected_health": 0.78, "expected_taste": 0.72, "priority": 2},
        "白灼虾": {"expected_health": 0.82, "expected_taste": 0.78, "priority": 2},
        "菌菇汤": {"expected_health": 0.8, "expected_taste": 0.7, "priority": 2},
    }

    # 处理已知菜品
    for dish_name, info in known_dishes.items():
        # 尝试从营养数据库中找到匹配的菜品
        matched = None
        for item in high_health_dishes:
            if dish_name in item.get('food_name', '') or item.get('food_name', '') in dish_name:
                matched = item
                break

        if matched:
            training_samples.append({
                "name": dish_name,
                "nutrition": {
                    "calories": matched.get('calories', 0),
                    "protein": matched.get('protein', 0),
                    "carbs": matched.get('carbs', 0),
                    "fat": matched.get('fat', 0),
                    "fiber": matched.get('fiber', 0),
                    "sodium": matched.get('sodium', 0),
                    "sugar": matched.get('sugar', 0),
                },
                "target_health": info["expected_health"],
                "target_taste": info["expected_taste"],
                "weight": 3.0 if info["priority"] == 1 else 2.0,
                "source": "known_high_health"
            })

    # 处理其他高健康价值菜品（使用营养数据库中的健康分作为目标）
    for item in high_health_dishes:
        food_name = item.get('food_name', '')
        # 跳过已处理的已知菜品
        if any(dish_name in food_name or food_name in dish_name for dish_name in known_dishes.keys()):
            continue

        health_score = item.get('health_score', 0)
        # 对于高健康价值菜品，假设口味分略低于健康分
        taste_score = max(0.5, health_score - 0.1)

        training_samples.append({
            "name": food_name,
            "nutrition": {
                "calories": item.get('calories', 0),
                "protein": item.get('protein', 0),
                "carbs": item.get('carbs', 0),
                "fat": item.get('fat', 0),
                "fiber": item.get('fiber', 0),
                "sodium": item.get('sodium', 0),
                "sugar": item.get('sugar', 0),
            },
            "target_health": health_score,
            "target_taste": taste_score,
            "weight": 1.5,  # 中等权重
            "source": "nutrition_database"
        })

    return training_samples

def augment_samples(samples: List[Dict], noise_level: float = 0.05, num_augmentations: int = 3) -> List[Dict]:
    """数据增强：添加噪声生成新样本"""
    augmented = []

    for sample in samples:
        augmented.append(sample)  # 保留原始样本

        for _ in range(num_augmentations):
            aug_sample = sample.copy()
            aug_nutrition = aug_sample["nutrition"].copy()

            # 对营养值添加小幅度噪声
            for key in aug_nutrition:
                if isinstance(aug_nutrition[key], (int, float)):
                    noise = np.random.normal(0, noise_level * abs(aug_nutrition[key]))
                    aug_nutrition[key] = max(0, aug_nutrition[key] + noise)

            aug_sample["nutrition"] = aug_nutrition
            aug_sample["source"] = f"{sample['source']}_augmented"
            augmented.append(aug_sample)

    return augmented

def main():
    """主函数"""
    print("=" * 60)
    print("扩展高分健康菜品训练集")
    print("=" * 60)

    # 1. 加载营养数据库
    print("\n[1/4] 加载营养数据库...")
    nutrition_data = load_nutrition_database()
    print(f"  加载了 {len(nutrition_data)} 条营养数据")

    # 2. 筛选高健康价值菜品
    print("\n[2/4] 筛选高健康价值菜品（健康分>=0.8）...")
    high_health_dishes = filter_high_health_dishes(nutrition_data, threshold=0.8)
    print(f"  找到 {len(high_health_dishes)} 个高健康价值菜品")

    # 显示前10个示例
    print("\n  示例菜品（前10个）:")
    for i, item in enumerate(high_health_dishes[:10], 1):
        name = item.get('food_name', 'Unknown')
        health = item.get('health_score', 0)
        print(f"    {i}. {name}: 健康分={health:.3f}")

    # 3. 创建训练样本
    print("\n[3/4] 创建训练样本...")
    training_samples = create_training_samples(high_health_dishes)
    print(f"  生成了 {len(training_samples)} 个训练样本")

    # 统计来源
    source_counts = {}
    for sample in training_samples:
        source = sample.get('source', 'unknown')
        source_counts[source] = source_counts.get(source, 0) + 1
    print(f"  样本来源分布: {source_counts}")

    # 4. 数据增强
    print("\n[4/4] 进行数据增强...")
    augmented_samples = augment_samples(training_samples, noise_level=0.05, num_augmentations=3)
    print(f"  增强后共有 {len(augmented_samples)} 个训练样本")

    # 5. 保存结果
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "high_health_dishes_expanded.json"

    result = {
        "version": "1.0",
        "generated_at": datetime.now().isoformat(),
        "total_samples": len(augmented_samples),
        "original_samples": len(training_samples),
        "augmented_samples": len(augmented_samples) - len(training_samples),
        "high_health_dishes_count": len(high_health_dishes),
        "source_distribution": source_counts,
        "samples": augmented_samples
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n[完成] 结果已保存到: {output_path}")
    print(f"  - 总样本数: {len(augmented_samples)}")
    print(f"  - 原始样本: {len(training_samples)}")
    print(f"  - 增强样本: {len(augmented_samples) - len(training_samples)}")

    return output_path

if __name__ == "__main__":
    main()
