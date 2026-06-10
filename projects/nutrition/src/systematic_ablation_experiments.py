#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统化A/B消融实验
"""

import json
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import itertools


@dataclass
class AblationConfig:
    """消融实验配置"""
    name: str
    components: Dict[str, bool]
    hyperparameters: Dict[str, float]


@dataclass
class AblationResult:
    """消融实验结果"""
    config_name: str
    metrics: Dict[str, float]
    component_contributions: Dict[str, float]


class AblationExperimentDesigner:
    """消融实验设计器"""

    def __init__(self):
        self.components = {
            'hierarchical_calibration': '分层校准',
            'temperature_scaling': '温度缩放',
            'multi_scale_fusion': '多尺度融合',
            'fine_grained_features': '细粒度特征',
            'advanced_augmentation': '数据增强',
            'high_health_expansion': '高分扩展',
        }

    def design_experiments(self, components: List[str] = None) -> List[AblationConfig]:
        """设计消融实验组合"""
        if components is None:
            components = list(self.components.keys())

        configs = []
        for i in range(2 ** len(components)):
            binary = format(i, f'0{len(components)}b')
            states = {comp: (binary[j] == '1') for j, comp in enumerate(components)}
            configs.append(AblationConfig(
                name=f"config_{i:03d}",
                components=states,
                hyperparameters={}
            ))
        return configs

    def design_grid_search(self, param_ranges: Dict[str, List]) -> List[Dict]:
        """设计网格搜索实验"""
        keys = list(param_ranges.keys())
        values = list(param_ranges.values())
        configs = []

        for combo in itertools.product(*values):
            configs.append({k: v for k, v in zip(keys, combo)})

        return configs


def run_ablation_experiment(configs: List[AblationConfig],
                            run_fn) -> List[AblationResult]:
    """运行消融实验"""
    results = []

    for config in configs:
        metrics = run_fn(config)
        contributions = calculate_contributions(results)

        results.append(AblationResult(
            config_name=config.name,
            metrics=metrics,
            component_contributions=contributions
        ))

    return results


def calculate_contributions(results: List[AblationResult]) -> Dict[str, float]:
    """计算组件贡献度"""
    if not results:
        return {}

    contributions = {}
    component_names = list(results[0].metrics.keys())

    for comp in component_names:
        values = [r.metrics.get(comp, 0) for r in results]
        contributions[comp] = float(np.mean(values))

    return contributions


def generate_report(results: List[AblationResult], output_path: str):
    """生成消融实验报告"""
    report = {
        'total_experiments': len(results),
        'best_config': max(results, key=lambda r: r.metrics.get('score', 0)).config_name,
        'results': [
            {
                'config': r.config_name,
                'metrics': r.metrics,
                'contributions': r.component_contributions
            }
            for r in results
        ]
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)


def compare_ablations(baseline_result, experiment_result) -> Dict:
    """比较消融结果"""
    improvements = {}

    for key in baseline_result.metrics:
        baseline_val = baseline_result.metrics[key]
        exp_val = experiment_result.metrics[key]

        if baseline_val != 0:
            improvement = (exp_val - baseline_val) / baseline_val * 100
            improvements[key] = improvement

    return {
        'improvements': improvements,
        'overall_improvement': np.mean(list(improvements.values())) if improvements else 0.0
    }