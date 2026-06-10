#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集成模型评估脚本
全面评估模型性能，对比优化前后的性能指标
"""

import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List
from datetime import datetime

# 添加项目根目录
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from .integrate_improvements import create_integrated_model
from .train_integrated_model import NutritionDataset
from torch.utils.data import DataLoader
from .adversarial_training_enhancer import HighHealthDishAdversarialTrainer


def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """计算评估指标"""
    mae = np.mean(np.abs(predictions - targets))
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)

    # 计算R²
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-6))

    # 计算预测多样性（标准差）
    diversity = np.std(predictions)

    return {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'r2': float(r2),
        'diversity': float(diversity)
    }


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = 'cpu',
    high_health_threshold: float = 0.7,
) -> Dict:
    """评估模型性能"""
    torch_device = torch.device(device) if not isinstance(device, torch.device) else device
    model = model.to(torch_device)
    model.eval()

    all_taste_preds = []
    all_taste_targets = []
    all_health_preds = []
    all_health_targets = []
    all_health_scores = []

    high_health_taste_preds = []
    high_health_taste_targets = []
    high_health_health_preds = []
    high_health_health_targets = []

    with torch.no_grad():
        for batch in data_loader:
            region_ids = batch['region_id'].to(torch_device)
            cuisine_ids = batch['cuisine_id'].to(torch_device)
            preferences = batch['preferences'].to(torch_device)
            nutrition_features = batch['nutrition_features'].to(torch_device)
            dish_names = batch['dish_name']
            taste_targets = batch['taste_target'].to(torch_device)
            health_targets = batch['health_target'].to(torch_device)
            health_scores = batch['health_score'].to(torch_device)

            outputs = model(
                region_ids=region_ids,
                cuisine_ids=cuisine_ids,
                preferences=preferences,
                nutrition_features=nutrition_features,
                dish_names=dish_names
            )

            taste_pred = outputs['taste_pred'].cpu().numpy()
            health_pred = outputs['health_pred'].cpu().numpy()
            taste_target = taste_targets.cpu().numpy()
            health_target = health_targets.cpu().numpy()
            health_score = health_scores.cpu().numpy()

            all_taste_preds.extend(taste_pred)
            all_taste_targets.extend(taste_target)
            all_health_preds.extend(health_pred)
            all_health_targets.extend(health_target)
            all_health_scores.extend(health_score)

            # 高分菜品
            high_health_mask = health_score >= high_health_threshold
            if high_health_mask.any():
                high_health_taste_preds.extend(taste_pred[high_health_mask])
                high_health_taste_targets.extend(taste_target[high_health_mask])
                high_health_health_preds.extend(health_pred[high_health_mask])
                high_health_health_targets.extend(health_target[high_health_mask])

    # 转换为numpy数组
    all_taste_preds = np.array(all_taste_preds)
    all_taste_targets = np.array(all_taste_targets)
    all_health_preds = np.array(all_health_preds)
    all_health_targets = np.array(all_health_targets)

    # 计算整体指标
    taste_metrics = calculate_metrics(all_taste_preds, all_taste_targets)
    health_metrics = calculate_metrics(all_health_preds, all_health_targets)

    # 计算高分菜品指标
    high_health_taste_metrics = {}
    high_health_health_metrics = {}

    if len(high_health_taste_preds) > 0:
        high_health_taste_preds = np.array(high_health_taste_preds)
        high_health_taste_targets = np.array(high_health_taste_targets)
        high_health_health_preds = np.array(high_health_health_preds)
        high_health_health_targets = np.array(high_health_health_targets)

        high_health_taste_metrics = calculate_metrics(
            high_health_taste_preds, high_health_taste_targets
        )
        high_health_health_metrics = calculate_metrics(
            high_health_health_preds, high_health_health_targets
        )

    # 评估鲁棒性（基于健康预测头的对抗评估）
    robustness_metrics: Dict = {}
    try:
        # 仅使用一小批样本进行鲁棒性评估，避免评估时间过长
        sample_batch = next(iter(data_loader))
        nutrition_features = sample_batch['nutrition_features'].to(torch_device)
        health_targets = sample_batch['health_target'].to(torch_device)
        health_scores = sample_batch['health_score'].to(torch_device)
        region_ids = sample_batch['region_id'].to(torch_device)
        cuisine_ids = sample_batch['cuisine_id'].to(torch_device)
        preferences = sample_batch['preferences'].to(torch_device)
        dish_names = sample_batch['dish_name']

        class _HealthPredictionWrapper(nn.Module):
            """
            轻量级包装器：将集成模型封装为仅基于营养特征输出健康预测的回归模型，
            以复用 HighHealthDishAdversarialTrainer 的通用对抗评估逻辑。
            """

            def __init__(
                self,
                base_model: nn.Module,
                region_ids: torch.Tensor,
                cuisine_ids: torch.Tensor,
                preferences: torch.Tensor,
                dish_names: List[str],
            ):
                super().__init__()
                self.base_model = base_model
                self.region_ids = region_ids
                self.cuisine_ids = cuisine_ids
                self.preferences = preferences
                self.dish_names = dish_names

            def forward(self, nutrition_features: torch.Tensor) -> torch.Tensor:
                outputs = self.base_model(
                    region_ids=self.region_ids,
                    cuisine_ids=self.cuisine_ids,
                    preferences=self.preferences,
                    nutrition_features=nutrition_features,
                    dish_names=self.dish_names,
                )
                # 仅关注健康预测，用于鲁棒性评估
                return outputs['health_pred']

        health_model = _HealthPredictionWrapper(
            base_model=model,
            region_ids=region_ids,
            cuisine_ids=cuisine_ids,
            preferences=preferences,
            dish_names=dish_names,
        ).to(torch_device)

        adversarial_trainer = HighHealthDishAdversarialTrainer(
            model=health_model,
            attack_method='fgm',
            epsilon=0.1,
            device=str(torch_device),
        )

        robustness_metrics = adversarial_trainer.evaluate_robustness(
            features=nutrition_features,
            targets=health_targets,
            health_scores=health_scores,
        )
    except Exception as e:
        robustness_metrics = {'error': str(e)}

    return {
        'overall': {
            'taste': taste_metrics,
            'health': health_metrics
        },
        'high_health_dishes': {
            'count': len(high_health_taste_preds),
            'taste': high_health_taste_metrics,
            'health': high_health_health_metrics
        },
        'robustness': robustness_metrics,
        'prediction_distribution': {
            'taste_min': float(np.min(all_taste_preds)),
            'taste_max': float(np.max(all_taste_preds)),
            'taste_range': float(np.max(all_taste_preds) - np.min(all_taste_preds)),
            'health_min': float(np.min(all_health_preds)),
            'health_max': float(np.max(all_health_preds)),
            'health_range': float(np.max(all_health_preds) - np.min(all_health_preds))
        }
    }


def compare_with_baseline(integrated_results: Dict, baseline_results: Dict = None) -> Dict:
    """对比集成模型和基线模型"""
    comparison = {
        'overall_mae': {
            'integrated_taste': integrated_results['overall']['taste']['mae'],
            'integrated_health': integrated_results['overall']['health']['mae']
        },
        'high_health_mae': {
            'integrated_taste': integrated_results['high_health_dishes']['taste'].get('mae', 0),
            'integrated_health': integrated_results['high_health_dishes']['health'].get('mae', 0)
        },
        'prediction_diversity': {
            'integrated_taste': integrated_results['overall']['taste']['diversity'],
            'integrated_health': integrated_results['overall']['health']['diversity']
        }
    }

    if baseline_results:
        comparison['overall_mae']['baseline_taste'] = baseline_results['overall']['taste']['mae']
        comparison['overall_mae']['baseline_health'] = baseline_results['overall']['health']['mae']
        comparison['overall_mae']['improvement_taste'] = (
            baseline_results['overall']['taste']['mae'] -
            integrated_results['overall']['taste']['mae']
        )
        comparison['overall_mae']['improvement_health'] = (
            baseline_results['overall']['health']['mae'] -
            integrated_results['overall']['health']['mae']
        )
        comparison['overall_mae']['improvement_rate_taste'] = (
            comparison['overall_mae']['improvement_taste'] /
            baseline_results['overall']['taste']['mae'] * 100
        )
        comparison['overall_mae']['improvement_rate_health'] = (
            comparison['overall_mae']['improvement_health'] /
            baseline_results['overall']['health']['mae'] * 100
        )

    return comparison


def main():
    """主函数"""
    print("=" * 60)
    print("集成模型性能评估")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 加载模型
    print("\n[1/3] 加载模型...")
    model_path = Path("stage1/outputs/integrated_model.pt")
    if not model_path.exists():
        print(f"✗ 模型文件不存在: {model_path}")
        print("  请先运行 train_integrated_model.py 训练模型")
        return

    # 集成模型 checkpoint 由本项目生成，内部包含 numpy 标量等安全对象，
    # 因此在 PyTorch 2.6+ 下需要显式设置 weights_only=False
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = create_integrated_model(use_all_improvements=True, device=device)
    # 训练阶段保存的是“裸模型”权重（即使使用多卡训练），因此这里可以直接加载
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ 模型加载成功")

    # 加载测试数据
    print("\n[2/3] 加载测试数据...")
    data_path = Path("stage1/data/high_health_dishes_expanded.json")
    if not data_path.exists():
        print(f"✗ 数据文件不存在: {data_path}")
        return

    dataset = NutritionDataset(str(data_path), use_augmentation=False)
    # 使用20%作为测试集
    test_size = int(0.2 * len(dataset))
    _, test_dataset = torch.utils.data.random_split(
        dataset, [len(dataset) - test_size, test_size]
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(f"✓ 测试集: {len(test_dataset):,} 个样本")

    # 评估模型
    print("\n[3/3] 评估模型性能...")
    results = evaluate_model(model, test_loader, device=device)

    # 打印结果
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)

    print("\n整体性能:")
    print(f"  口味预测 - MAE: {results['overall']['taste']['mae']:.4f}, "
          f"R²: {results['overall']['taste']['r2']:.4f}, "
          f"多样性: {results['overall']['taste']['diversity']:.4f}")
    print(f"  健康预测 - MAE: {results['overall']['health']['mae']:.4f}, "
          f"R²: {results['overall']['health']['r2']:.4f}, "
          f"多样性: {results['overall']['health']['diversity']:.4f}")

    if results['high_health_dishes']['count'] > 0:
        print(f"\n高分菜品性能 (阈值>=0.7, 共{results['high_health_dishes']['count']}个):")
        print(f"  口味预测 - MAE: {results['high_health_dishes']['taste'].get('mae', 0):.4f}")
        print(f"  健康预测 - MAE: {results['high_health_dishes']['health'].get('mae', 0):.4f}")

    print(f"\n预测分布:")
    print(f"  口味: [{results['prediction_distribution']['taste_min']:.4f}, "
          f"{results['prediction_distribution']['taste_max']:.4f}], "
          f"范围: {results['prediction_distribution']['taste_range']:.4f}")
    print(f"  健康: [{results['prediction_distribution']['health_min']:.4f}, "
          f"{results['prediction_distribution']['health_max']:.4f}], "
          f"范围: {results['prediction_distribution']['health_range']:.4f}")

    if 'mae_adversarial' in results['robustness']:
        print(f"\n模型鲁棒性:")
        print(f"  正常样本MAE: {results['robustness']['mae_normal']:.4f}")
        print(f"  对抗样本MAE: {results['robustness']['mae_adversarial']:.4f}")
        print(f"  鲁棒性差距: {results['robustness']['robustness_gap']:.4f}")

    # 对比基线（如果有）
    baseline_mae_taste = 0.2214  # 从之前的报告
    baseline_mae_health = 0.1868
    baseline_high_health_mae = 0.3914  # 清蒸鲈鱼误差

    print(f"\n性能对比:")
    print(f"  整体MAE - 口味:")
    print(f"    基线: {baseline_mae_taste:.4f}")
    print(f"    集成: {results['overall']['taste']['mae']:.4f}")
    print(f"    改进: {baseline_mae_taste - results['overall']['taste']['mae']:.4f} "
          f"({(baseline_mae_taste - results['overall']['taste']['mae']) / baseline_mae_taste * 100:.1f}%)")

    print(f"  整体MAE - 健康:")
    print(f"    基线: {baseline_mae_health:.4f}")
    print(f"    集成: {results['overall']['health']['mae']:.4f}")
    print(f"    改进: {baseline_mae_health - results['overall']['health']['mae']:.4f} "
          f"({(baseline_mae_health - results['overall']['health']['mae']) / baseline_mae_health * 100:.1f}%)")

    if results['high_health_dishes']['count'] > 0:
        high_health_mae = results['high_health_dishes']['health'].get('mae', 0)
        print(f"  高分菜品MAE:")
        print(f"    基线: {baseline_high_health_mae:.4f}")
        print(f"    集成: {high_health_mae:.4f}")
        print(f"    改进: {baseline_high_health_mae - high_health_mae:.4f} "
              f"({(baseline_high_health_mae - high_health_mae) / baseline_high_health_mae * 100:.1f}%)")

    # 保存结果
    output_path = Path("stage1/outputs/evaluation_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    comparison = compare_with_baseline(results, {
        'overall': {
            'taste': {'mae': baseline_mae_taste},
            'health': {'mae': baseline_mae_health}
        }
    })

    # 可选：加载 Stage2 约束门控诊断结果，用于在同一评估文件中对齐约束相关统计
    constraint_diag_path = Path("stage2/results/constraint_gate_diagnosis.json")
    constraint_diagnosis = None
    if constraint_diag_path.exists():
        try:
            with open(constraint_diag_path, 'r', encoding='utf-8') as f:
                constraint_diagnosis = json.load(f)
        except Exception as e:
            # 如果诊断文件存在但解析失败，不影响主流程，只记录错误信息
            constraint_diagnosis = {'error': str(e)}

    final_results = {
        'evaluation_date': datetime.now().isoformat(),
        'model_path': str(model_path),
        'test_samples': len(test_dataset),
        'results': results,
        'comparison': comparison,
        'baseline': {
            'mae_taste': baseline_mae_taste,
            'mae_health': baseline_mae_health,
            'high_health_mae': baseline_high_health_mae
        },
        # 将 Stage2 的约束门控诊断结果直接写入评估输出，便于后续联动分析
        'constraint_diagnosis': constraint_diagnosis
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 评估结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
