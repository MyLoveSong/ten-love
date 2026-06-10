#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
从文化数据中提取用户偏好样本
用于训练模型的数据准备工作
"""

import os
import sys
import json
import logging
import argparse
import random
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_samples_from_cultural_data(data_path: str, output_path: str, num_samples: int = 5000):
    """从文化数据中提取用户偏好样本"""
    logger.info(f"从 {data_path} 提取用户偏好样本...")

    # 加载文化数据
    with open(data_path, 'r', encoding='utf-8') as f:
        cultural_data = json.load(f)

    # 提取地区信息
    regions = []
    if "regional_data" in cultural_data:
        regions = list(cultural_data["regional_data"].keys())

    # 提取菜系信息
    cuisines = []
    if "cuisine_data" in cultural_data:
        cuisines = list(cultural_data["cuisine_data"].keys())

    # 如果没有足够的地区或菜系信息，使用默认值
    if not regions:
        regions = ["华北", "华南", "华东", "华西", "华中", "东北", "西北", "西南"]

    if not cuisines:
        cuisines = ["川菜", "粤菜", "鲁菜", "苏菜", "浙菜", "闽菜", "湘菜", "徽菜"]

    # 提取营养数据库
    nutrition_db = {}
    if "nutrition_database" in cultural_data:
        nutrition_db = cultural_data["nutrition_database"]

    # 生成用户偏好样本
    samples = []

    for i in range(num_samples):
        # 选择地区和菜系
        region = random.choice(regions)
        cuisine = random.choice(cuisines)

        # 生成偏好数据
        sample = {
            "user_id": f"user_{i:04d}",
            "region": region,
            "cuisine_type": cuisine,
            "spice_tolerance": round(random.uniform(0.0, 1.0), 2),
            "sweet_preference": round(random.uniform(0.0, 1.0), 2),
            "salt_preference": round(random.uniform(0.0, 1.0), 2),
            "sour_preference": round(random.uniform(0.0, 1.0), 2),
            "bitter_preference": round(random.uniform(0.0, 1.0), 2),
            "umami_preference": round(random.uniform(0.0, 1.0), 2),
            "preferred_cooking": random.choice(["炒", "蒸", "煮", "烤", "炖", "煎", "炸"]),
            "meal_time": random.choice(["早餐", "午餐", "晚餐", "夜宵"]),
            "dietary_restrictions": random.sample(["素食", "清真", "无辣", "低盐", "低糖", "无海鲜", "无坚果"],
                                                k=random.randint(0, 3)),
            "acceptance_score": round(random.uniform(0.3, 0.9), 2)
        }

        samples.append(sample)

    # 保存样本
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ 已提取 {len(samples)} 个用户偏好样本到 {output_path}")

    return samples


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="从文化数据中提取用户偏好样本")
    parser.add_argument("--data-path", type=str, default="data/real_data/cultural_data.json", help="文化数据路径")
    parser.add_argument("--output-path", type=str, default="data/real_data/user_preferences.json", help="输出路径")
    parser.add_argument("--num-samples", type=int, default=5000, help="样本数量")
    args = parser.parse_args()

    extract_samples_from_cultural_data(args.data_path, args.output_path, args.num_samples)


if __name__ == "__main__":
    main()
