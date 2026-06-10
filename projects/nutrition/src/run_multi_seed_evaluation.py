#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多随机种子评估脚本
基于已训练好的 Stage1 集成模型，重复进行多次随机划分测试集的评估，
统计关键指标（MAE / RMSE / R² 等）的均值和标准差，贴近顶会/顶刊的实验报告规范。
"""

import argparse
import json
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

# 添加项目根目录，保持与其他 Stage1 脚本一致的导入方式
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from .evaluate_integrated_model import evaluate_model
from .integrate_improvements import create_integrated_model
from .train_integrated_model import NutritionDataset


@dataclass
class MetricStats:
    mean: float
    std: float


def _set_global_seed(seed: int) -> None:
    """设置 Python / NumPy / PyTorch 的全局随机种子，保证可复现的划分与评估。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _aggregate_scalar(values: List[float]) -> MetricStats:
    arr = np.array(values, dtype=np.float64)
    return MetricStats(mean=float(arr.mean()), std=float(arr.std(ddof=1)) if len(arr) > 1 else 0.0)


def _extract_key_metrics(results: Dict) -> Dict[str, Dict[str, float]]:
    """从单次 evaluate_model 的结果中抽取关键标量指标，便于做多次统计。"""
    overall = results["overall"]
    robustness = results.get("robustness", {})
    return {
        "taste": {
            "mae": overall["taste"]["mae"],
            "rmse": overall["taste"]["rmse"],
            "r2": overall["taste"]["r2"],
        },
        "health": {
            "mae": overall["health"]["mae"],
            "rmse": overall["health"]["rmse"],
            "r2": overall["health"]["r2"],
        },
        "robustness": {
            "mae_normal": robustness.get("mae_normal"),
            "mae_adversarial": robustness.get("mae_adversarial"),
            "robustness_gap": robustness.get("robustness_gap"),
        },
    }


def run_multi_seed_evaluation(
    num_runs: int = 5,
    test_ratio: float = 0.2,
    batch_size: int = 32,
) -> Dict:
    """对同一已训练模型进行多次随机划分测试集的评估，并统计指标分布。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = Path("stage1/outputs/integrated_model.pt")
    if not model_path.exists():
        raise FileNotFoundError(f"未找到已训练的集成模型权重: {model_path}")

    data_path = Path("stage1/data/high_health_dishes_expanded.json")
    if not data_path.exists():
        raise FileNotFoundError(f"未找到评估数据文件: {data_path}")

    # 预先加载数据集（每次 run 仅改变 random_split 的随机种子）
    dataset = NutritionDataset(str(data_path), use_augmentation=False)
    total_size = len(dataset)
    test_size = int(test_ratio * total_size)

    # 为了专注于评估方差，这里对每次 run 复用同一训练好的模型权重
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    per_run_metrics: List[Dict] = []

    for run_idx in range(num_runs):
        seed = 42 + run_idx  # 简单地从一个固定偏移开始
        _set_global_seed(seed)

        # 创建并加载模型
        model = create_integrated_model(use_all_improvements=True, device=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        # 按当前种子进行一次新的随机划分（只关心测试集）
        # 注意：这里不重新训练模型，而是考察在不同随机子集上的评估稳定性
        generator = torch.Generator().manual_seed(seed)
        _, test_dataset = torch.utils.data.random_split(
            dataset,
            [total_size - test_size, test_size],
            generator=generator,
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        results = evaluate_model(model, test_loader, device=device)
        key_metrics = _extract_key_metrics(results)

        per_run_metrics.append(
            {
                "seed": seed,
                "test_samples": len(test_dataset),
                "key_metrics": key_metrics,
            }
        )

    # 统计各指标的均值和标准差
    def _collect(metric_path: List[str]) -> MetricStats:
        vals: List[float] = []
        for run in per_run_metrics:
            cur = run["key_metrics"]
            for key in metric_path:
                if cur is None:
                    break
                cur = cur.get(key)
            if cur is not None:
                vals.append(cur)
        return _aggregate_scalar(vals) if vals else MetricStats(mean=0.0, std=0.0)

    summary = {
        "taste": {
            "mae": asdict(_collect(["taste", "mae"])),
            "rmse": asdict(_collect(["taste", "rmse"])),
            "r2": asdict(_collect(["taste", "r2"])),
        },
        "health": {
            "mae": asdict(_collect(["health", "mae"])),
            "rmse": asdict(_collect(["health", "rmse"])),
            "r2": asdict(_collect(["health", "r2"])),
        },
        "robustness": {
            "mae_normal": asdict(_collect(["robustness", "mae_normal"])),
            "mae_adversarial": asdict(_collect(["robustness", "mae_adversarial"])),
            "robustness_gap": asdict(_collect(["robustness", "robustness_gap"])),
        },
    }

    return {
        "num_runs": num_runs,
        "test_ratio": test_ratio,
        "batch_size": batch_size,
        "device": device,
        "total_samples": total_size,
        "per_run": per_run_metrics,
        "summary": summary,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage1 集成模型多随机种子评估")
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="评估运行次数（随机种子数，默认: 5）",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
        help="测试集占比（默认: 0.2）",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="评估 batch size（默认: 32）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="stage1/outputs/multi_seed_evaluation_results.json",
        help="结果保存路径（默认: stage1/outputs/multi_seed_evaluation_results.json）",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print("=" * 60)
    print("Stage1 集成模型多随机种子评估")
    print("=" * 60)

    print(
        f"\n配置: runs={args.runs}, test_ratio={args.test_ratio}, "
        f"batch_size={args.batch_size}"
    )

    results = run_multi_seed_evaluation(
        num_runs=args.runs,
        test_ratio=args.test_ratio,
        batch_size=args.batch_size,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 多随机种子评估结果已保存到: {output_path}")
    print("  你可以在论文中直接使用 summary 中的 mean/std 作为主表的统计量。")


if __name__ == "__main__":
    main()
