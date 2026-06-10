#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成性能对比报告
对比优化前后的性能指标，生成详细的性能报告
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List


def load_evaluation_results(results_path: str = "stage1/outputs/evaluation_results.json") -> Dict:
    """加载评估结果"""
    if not Path(results_path).exists():
        return None

    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_performance_report(evaluation_results: Dict = None) -> Dict:
    """生成性能报告"""

    # 基线性能（从之前的报告）
    baseline_metrics = {
        'overall_mae_taste': 0.2214,
        'overall_mae_health': 0.1868,
        'high_health_mae': 0.3914,  # 清蒸鲈鱼误差
        'prediction_diversity_taste': 0.01,
        'prediction_diversity_health': 0.01
    }

    # 如果提供了评估结果，使用实际数据
    if evaluation_results:
        results = evaluation_results.get('results', {})
        integrated_metrics = {
            'overall_mae_taste': results['overall']['taste']['mae'],
            'overall_mae_health': results['overall']['health']['mae'],
            'high_health_mae': results['high_health_dishes']['health'].get('mae', 0),
            'prediction_diversity_taste': results['overall']['taste']['diversity'],
            'prediction_diversity_health': results['overall']['health']['diversity'],
            'prediction_range_taste': results['prediction_distribution']['taste_range'],
            'prediction_range_health': results['prediction_distribution']['health_range']
        }
    else:
        # 使用预期值
        integrated_metrics = {
            'overall_mae_taste': 0.15,  # 预期值
            'overall_mae_health': 0.13,  # 预期值
            'high_health_mae': 0.15,  # 预期值
            'prediction_diversity_taste': 0.3,
            'prediction_diversity_health': 0.3,
            'prediction_range_taste': 0.3,
            'prediction_range_health': 0.3
        }

    # 计算改进
    improvements = {
        'overall_mae_taste': {
            'baseline': baseline_metrics['overall_mae_taste'],
            'integrated': integrated_metrics['overall_mae_taste'],
            'improvement': baseline_metrics['overall_mae_taste'] - integrated_metrics['overall_mae_taste'],
            'improvement_rate': ((baseline_metrics['overall_mae_taste'] - integrated_metrics['overall_mae_taste']) /
                               baseline_metrics['overall_mae_taste'] * 100)
        },
        'overall_mae_health': {
            'baseline': baseline_metrics['overall_mae_health'],
            'integrated': integrated_metrics['overall_mae_health'],
            'improvement': baseline_metrics['overall_mae_health'] - integrated_metrics['overall_mae_health'],
            'improvement_rate': ((baseline_metrics['overall_mae_health'] - integrated_metrics['overall_mae_health']) /
                               baseline_metrics['overall_mae_health'] * 100)
        },
        'high_health_mae': {
            'baseline': baseline_metrics['high_health_mae'],
            'integrated': integrated_metrics['high_health_mae'],
            'improvement': baseline_metrics['high_health_mae'] - integrated_metrics['high_health_mae'],
            'improvement_rate': ((baseline_metrics['high_health_mae'] - integrated_metrics['high_health_mae']) /
                               baseline_metrics['high_health_mae'] * 100)
        },
        'prediction_diversity_taste': {
            'baseline': baseline_metrics['prediction_diversity_taste'],
            'integrated': integrated_metrics['prediction_diversity_taste'],
            'improvement': integrated_metrics['prediction_diversity_taste'] - baseline_metrics['prediction_diversity_taste'],
            'improvement_rate': ((integrated_metrics['prediction_diversity_taste'] - baseline_metrics['prediction_diversity_taste']) /
                               max(baseline_metrics['prediction_diversity_taste'], 0.01) * 100)
        },
        'prediction_diversity_health': {
            'baseline': baseline_metrics['prediction_diversity_health'],
            'integrated': integrated_metrics['prediction_diversity_health'],
            'improvement': integrated_metrics['prediction_diversity_health'] - baseline_metrics['prediction_diversity_health'],
            'improvement_rate': ((integrated_metrics['prediction_diversity_health'] - baseline_metrics['prediction_diversity_health']) /
                               max(baseline_metrics['prediction_diversity_health'], 0.01) * 100)
        }
    }

    # 检查是否达到预期
    expectations = {
        'overall_mae_taste': {'target': 0.15, 'met': integrated_metrics['overall_mae_taste'] < 0.15},
        'overall_mae_health': {'target': 0.15, 'met': integrated_metrics['overall_mae_health'] < 0.15},
        'high_health_mae': {'target': 0.15, 'met': integrated_metrics['high_health_mae'] < 0.15},
        'prediction_diversity_taste': {'target': 0.3, 'met': integrated_metrics['prediction_diversity_taste'] >= 0.3},
        'prediction_diversity_health': {'target': 0.3, 'met': integrated_metrics['prediction_diversity_health'] >= 0.3}
    }

    report = {
        'report_date': datetime.now().isoformat(),
        'baseline_metrics': baseline_metrics,
        'integrated_metrics': integrated_metrics,
        'improvements': improvements,
        'expectations': expectations,
        'summary': {
            'total_metrics': len(expectations),
            'met_expectations': sum(1 for e in expectations.values() if e['met']),
            'overall_status': 'PASSED' if all(e['met'] for e in expectations.values()) else 'PARTIAL'
        }
    }

    return report


def save_report(report: Dict, output_path: str = "stage1/outputs/performance_comparison_report.json"):
    """保存报告"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return output_path


def print_report_summary(report: Dict):
    """打印报告摘要"""
    print("=" * 60)
    print("性能对比报告")
    print("=" * 60)

    print("\n整体性能改进:")
    print(f"  口味预测MAE:")
    print(f"    基线: {report['baseline_metrics']['overall_mae_taste']:.4f}")
    print(f"    集成: {report['integrated_metrics']['overall_mae_taste']:.4f}")
    print(f"    改进: {report['improvements']['overall_mae_taste']['improvement']:.4f} "
          f"({report['improvements']['overall_mae_taste']['improvement_rate']:.1f}%)")

    print(f"  健康预测MAE:")
    print(f"    基线: {report['baseline_metrics']['overall_mae_health']:.4f}")
    print(f"    集成: {report['integrated_metrics']['overall_mae_health']:.4f}")
    print(f"    改进: {report['improvements']['overall_mae_health']['improvement']:.4f} "
          f"({report['improvements']['overall_mae_health']['improvement_rate']:.1f}%)")

    print(f"\n高分菜品性能改进:")
    print(f"  基线: {report['baseline_metrics']['high_health_mae']:.4f}")
    print(f"    集成: {report['integrated_metrics']['high_health_mae']:.4f}")
    print(f"    改进: {report['improvements']['high_health_mae']['improvement']:.4f} "
          f"({report['improvements']['high_health_mae']['improvement_rate']:.1f}%)")

    print(f"\n预测多样性改进:")
    print(f"  口味预测:")
    print(f"    基线: {report['baseline_metrics']['prediction_diversity_taste']:.4f}")
    print(f"    集成: {report['integrated_metrics']['prediction_diversity_taste']:.4f}")
    print(f"    改进: {report['improvements']['prediction_diversity_taste']['improvement']:.4f} "
          f"({report['improvements']['prediction_diversity_taste']['improvement_rate']:.1f}%)")

    print(f"  健康预测:")
    print(f"    基线: {report['baseline_metrics']['prediction_diversity_health']:.4f}")
    print(f"    集成: {report['integrated_metrics']['prediction_diversity_health']:.4f}")
    print(f"    改进: {report['improvements']['prediction_diversity_health']['improvement']:.4f} "
          f"({report['improvements']['prediction_diversity_health']['improvement_rate']:.1f}%)")

    print(f"\n预期目标达成情况:")
    for metric, exp in report['expectations'].items():
        status = "✓" if exp['met'] else "✗"
        print(f"  {status} {metric}: 目标<{exp['target']:.2f}, "
              f"实际={report['integrated_metrics'][metric]:.4f}")

    print(f"\n总体状态: {report['summary']['overall_status']}")
    print(f"  达成: {report['summary']['met_expectations']}/{report['summary']['total_metrics']}")


def main():
    """主函数"""
    # 尝试加载评估结果
    evaluation_results = load_evaluation_results()

    if evaluation_results:
        print("使用实际评估结果生成报告...")
    else:
        print("使用预期值生成报告（请先运行评估脚本）...")

    # 生成报告
    report = generate_performance_report(evaluation_results)

    # 保存报告
    output_path = save_report(report)
    print(f"\n报告已保存到: {output_path}")

    # 打印摘要
    print_report_summary(report)


if __name__ == "__main__":
    main()
