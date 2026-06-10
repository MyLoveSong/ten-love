#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据充足性评估脚本

评估当前数据量是否足够，是否需要采集更多数据
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any
import logging
from datetime import datetime

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataAdequacyEvaluator:
    """数据充足性评估器"""

    def __init__(self):
        self.project_root = project_root
        self.current_samples = 2994  # 从model_performance_log.json获取
        self.optimization_samples = {
            "high_health_dishes": 6,
            "clinical_metrics": 5
        }

    def evaluate_data_adequacy(self) -> Dict[str, Any]:
        """评估数据充足性"""

        evaluation = {
            "timestamp": datetime.now().isoformat(),
            "current_data_status": {},
            "data_adequacy_assessment": {},
            "recommendations": []
        }

        # 1. 当前数据量评估
        evaluation["current_data_status"] = {
            "total_training_samples": self.current_samples,
            "optimization_samples": sum(self.optimization_samples.values()),
            "data_sources": [
                "Stage2统一数据集",
                "优化生成的训练样本"
            ]
        }

        # 2. 数据充足性标准（基于机器学习最佳实践）
        adequacy_standards = {
            "minimum_samples": {
                "simple_model": 100,
                "medium_model": 1000,
                "complex_model": 5000,
                "deep_learning": 10000
            },
            "recommended_samples": {
                "simple_model": 500,
                "medium_model": 5000,
                "complex_model": 20000,
                "deep_learning": 50000
            },
            "current_model_complexity": "medium_model"  # LoRA微调，中等复杂度
        }

        # 3. 数据充足性评估
        current_samples = self.current_samples
        model_complexity = adequacy_standards["current_model_complexity"]
        min_samples = adequacy_standards["minimum_samples"][model_complexity]
        rec_samples = adequacy_standards["recommended_samples"][model_complexity]

        adequacy_score = min(1.0, current_samples / rec_samples)

        evaluation["data_adequacy_assessment"] = {
            "current_samples": current_samples,
            "minimum_required": min_samples,
            "recommended": rec_samples,
            "adequacy_score": round(adequacy_score, 3),
            "adequacy_level": self._get_adequacy_level(adequacy_score),
            "model_complexity": model_complexity,
            "meets_minimum": current_samples >= min_samples,
            "meets_recommended": current_samples >= rec_samples
        }

        # 4. 数据质量评估
        quality_assessment = {
            "coverage": {
                "high_health_dishes": "limited",  # 高健康价值菜品样本较少
                "clinical_compliance": "moderate",  # 临床合规样本中等
                "diversity": "good"  # 数据多样性良好
            },
            "issues": [
                "高健康价值菜品样本不足（仅6个优化样本）",
                "临床标准样本较少（仅5个）",
                "需要更多营养师专家标注数据"
            ]
        }

        evaluation["data_quality"] = quality_assessment

        # 5. 生成建议
        recommendations = []

        if current_samples < min_samples:
            recommendations.append({
                "priority": "HIGH",
                "action": "立即采集数据",
                "target": f"至少采集 {min_samples - current_samples} 个样本",
                "reason": "当前数据量低于最低要求"
            })
        elif current_samples < rec_samples:
            recommendations.append({
                "priority": "MEDIUM",
                "action": "建议补充数据",
                "target": f"建议补充 {rec_samples - current_samples} 个样本",
                "reason": "达到推荐数据量可进一步提升模型性能"
            })
        else:
            recommendations.append({
                "priority": "LOW",
                "action": "数据量充足",
                "target": "无需额外采集",
                "reason": "当前数据量已达到推荐标准"
            })

        # 针对特定问题的建议
        if quality_assessment["coverage"]["high_health_dishes"] == "limited":
            recommendations.append({
                "priority": "HIGH",
                "action": "增加高健康价值菜品样本",
                "target": "至少500个高健康价值菜品样本",
                "reason": "当前仅6个优化样本，不足以训练高分菜品预测"
            })

        if quality_assessment["coverage"]["clinical_compliance"] == "moderate":
            recommendations.append({
                "priority": "MEDIUM",
                "action": "增加临床标准样本",
                "target": "至少200个符合临床标准的样本",
                "reason": "提升临床评估指标的准确性"
            })

        recommendations.append({
            "priority": "MEDIUM",
            "action": "引入营养师专家标注",
            "target": "至少200个专家标注样本",
            "reason": "提升标注质量和一致性"
        })

        evaluation["recommendations"] = recommendations

        # 6. 数据采集优先级
        evaluation["data_collection_priority"] = {
            "immediate": [
                "高健康价值菜品样本（500+）",
                "营养师专家标注数据（200+）"
            ],
            "short_term": [
                "临床标准样本（200+）",
                "不同菜系和地区的代表性样本"
            ],
            "long_term": [
                "扩大数据集到推荐规模（5000+）",
                "建立持续数据采集机制"
            ]
        }

        return evaluation

    def _get_adequacy_level(self, score: float) -> str:
        """获取充足性等级"""
        if score >= 1.0:
            return "充足 (Adequate)"
        elif score >= 0.8:
            return "良好 (Good)"
        elif score >= 0.5:
            return "中等 (Moderate)"
        elif score >= 0.3:
            return "不足 (Inadequate)"
        else:
            return "严重不足 (Severely Inadequate)"

    def generate_report(self, evaluation: Dict[str, Any]) -> str:
        """生成评估报告"""
        report_path = self.project_root / "stage1" / "outputs" / "data_adequacy_evaluation.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation, f, indent=2, ensure_ascii=False)

        logger.info(f"数据充足性评估报告已保存: {report_path}")
        return str(report_path)


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("数据充足性评估")
    logger.info("=" * 60)

    evaluator = DataAdequacyEvaluator()
    evaluation = evaluator.evaluate_data_adequacy()

    # 显示评估结果
    print("\n" + "=" * 60)
    print("数据充足性评估结果")
    print("=" * 60)

    assessment = evaluation["data_adequacy_assessment"]
    print(f"\n当前数据量: {assessment['current_samples']} 个样本")
    print(f"最低要求: {assessment['minimum_required']} 个样本")
    print(f"推荐数量: {assessment['recommended']} 个样本")
    print(f"充足性评分: {assessment['adequacy_score']:.1%}")
    print(f"充足性等级: {assessment['adequacy_level']}")
    print(f"达到最低要求: {'是' if assessment['meets_minimum'] else '否'}")
    print(f"达到推荐标准: {'是' if assessment['meets_recommended'] else '否'}")

    print("\n" + "=" * 60)
    print("数据质量评估")
    print("=" * 60)
    quality = evaluation["data_quality"]
    print(f"高健康价值菜品覆盖: {quality['coverage']['high_health_dishes']}")
    print(f"临床合规样本覆盖: {quality['coverage']['clinical_compliance']}")
    print(f"数据多样性: {quality['coverage']['diversity']}")

    if quality["issues"]:
        print("\n主要问题:")
        for i, issue in enumerate(quality["issues"], 1):
            print(f"  {i}. {issue}")

    print("\n" + "=" * 60)
    print("数据采集建议")
    print("=" * 60)

    high_priority = [r for r in evaluation["recommendations"] if r["priority"] == "HIGH"]
    medium_priority = [r for r in evaluation["recommendations"] if r["priority"] == "MEDIUM"]
    low_priority = [r for r in evaluation["recommendations"] if r["priority"] == "LOW"]

    if high_priority:
        print("\n[高优先级]:")
        for r in high_priority:
            print(f"  * {r['action']}: {r['target']}")
            print(f"    原因: {r['reason']}")

    if medium_priority:
        print("\n[中优先级]:")
        for r in medium_priority:
            print(f"  * {r['action']}: {r['target']}")
            print(f"    原因: {r['reason']}")

    if low_priority:
        print("\n[低优先级]:")
        for r in low_priority:
            print(f"  * {r['action']}: {r['target']}")

    # 保存报告
    report_path = evaluator.generate_report(evaluation)

    print(f"\n详细评估报告已保存: {report_path}")

    # 最终结论
    print("\n" + "=" * 60)
    print("最终结论")
    print("=" * 60)

    if assessment['meets_recommended']:
        print("[通过] 数据量充足，可以开始实施优化方案")
    elif assessment['meets_minimum']:
        print("[警告] 数据量达到最低要求，建议补充数据后再实施优化")
    else:
        print("[失败] 数据量不足，需要先采集更多数据")

    return evaluation


if __name__ == "__main__":
    main()
