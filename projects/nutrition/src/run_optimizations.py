#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行所有优化脚本的主程序

按优先级顺序执行各个优化方面
"""

import os
import sys
import subprocess
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 优化脚本列表（按优先级排序）
OPTIMIZATION_SCRIPTS = [
    {
        "name": "高分健康菜品预测优化",
        "script": "optimize_high_health_dishes.py",
        "priority": "P0",
        "description": "优化清蒸鲈鱼、蒸蛋羹等高健康价值菜品的预测准确性"
    },
    {
        "name": "提升临床评估指标",
        "script": "optimize_clinical_metrics.py",
        "priority": "P0",
        "description": "提升健康合规率、临床实用性和临床评分"
    },
    {
        "name": "增强模型可解释性",
        "script": "enhance_model_interpretability.py",
        "priority": "P1",
        "description": "实现SHAP分析和特征重要性报告"
    },
    # 可以继续添加其他优化脚本
]


def run_optimization(script_name: str) -> bool:
    """运行单个优化脚本"""
    script_path = Path(__file__).parent / script_name

    if not script_path.exists():
        logger.error(f"脚本不存在: {script_path}")
        return False

    logger.info(f"运行优化脚本: {script_name}")
    logger.info("-" * 60)

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=True
        )

        logger.info(result.stdout)
        if result.stderr:
            logger.warning(f"警告: {result.stderr}")

        logger.info(f"✅ {script_name} 执行成功")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {script_name} 执行失败:")
        logger.error(e.stdout)
        logger.error(e.stderr)
        return False
    except Exception as e:
        logger.error(f"❌ {script_name} 执行出错: {e}")
        return False


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("Stage1 模型优化 - 批量执行")
    logger.info("=" * 60)

    # 创建输出目录
    output_dir = Path(__file__).parent / "outputs" / "optimization_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "total_optimizations": len(OPTIMIZATION_SCRIPTS),
        "successful": 0,
        "failed": 0,
        "details": []
    }

    # 按优先级执行
    for opt in OPTIMIZATION_SCRIPTS:
        logger.info(f"\n[{opt['priority']}] {opt['name']}")
        logger.info(f"描述: {opt['description']}")

        success = run_optimization(opt['script'])

        results["details"].append({
            "name": opt['name'],
            "script": opt['script'],
            "priority": opt['priority'],
            "success": success
        })

        if success:
            results["successful"] += 1
        else:
            results["failed"] += 1

    # 打印总结
    logger.info("\n" + "=" * 60)
    logger.info("优化执行总结")
    logger.info("=" * 60)
    logger.info(f"总计: {results['total_optimizations']} 个优化")
    logger.info(f"成功: {results['successful']} 个")
    logger.info(f"失败: {results['failed']} 个")

    # 保存结果
    import json
    from datetime import datetime

    summary_path = output_dir / "optimization_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            **results
        }, f, ensure_ascii=False, indent=2)

    logger.info(f"\n优化总结已保存到: {summary_path}")

    if results["failed"] > 0:
        logger.warning("\n⚠️  部分优化失败，请检查日志")
        return 1
    else:
        logger.info("\n✅ 所有优化执行完成！")
        return 0


if __name__ == "__main__":
    exit(main())
