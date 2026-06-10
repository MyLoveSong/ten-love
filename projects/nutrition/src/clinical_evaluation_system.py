#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
临床对齐的评估体系
"""

import json
import logging
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class HealthRiskLevel(Enum):
    """健康风险等级"""
    LOW = "低风险"
    MODERATE = "中等风险"
    HIGH = "高风险"


class ClinicalRecommendation(Enum):
    """临床推荐等级"""
    STRONGLY_RECOMMEND = "强烈推荐"
    RECOMMEND = "推荐"
    NEUTRAL = "中性"
    CAUTION = "谨慎"
    NOT_RECOMMEND = "不推荐"


@dataclass
class NutritionProfile:
    """营养成分档案"""
    sodium_mg: float
    fat_g: float
    sugar_g: float
    fiber_g: float
    protein_g: float
    calories: float


class NutritionistExpertSystem:
    """营养师专家系统"""

    def __init__(self):
        self.sodium_thresholds = {"low": 400, "moderate": 800, "high": 1200}
        self.fat_thresholds = {"low": 10, "moderate": 20, "high": 30}
        self.sugar_thresholds = {"low": 10, "moderate": 20, "high": 30}

    def assess_nutrition(self, nutrition: NutritionProfile) -> Dict:
        """评估营养状况"""
        sodium_level = "low" if nutrition.sodium_mg < self.sodium_thresholds["low"] else \
            "high" if nutrition.sodium_mg > self.sodium_thresholds["high"] else "moderate"

        fat_level = "low" if nutrition.fat_g < self.fat_thresholds["low"] else \
            "high" if nutrition.fat_g > self.fat_thresholds["high"] else "moderate"

        sugar_level = "low" if nutrition.sugar_g < self.sugar_thresholds["low"] else \
            "high" if nutrition.sugar_g > self.sugar_thresholds["high"] else "moderate"

        risk_factors, benefits = [], []

        if sodium_level == "high":
            risk_factors.append("高钠摄入增加高血压风险")
        if fat_level == "high":
            risk_factors.append("高脂肪摄入增加心血管疾病风险")
        if sugar_level == "high":
            risk_factors.append("高糖摄入增加糖尿病风险")

        if nutrition.fiber_g > 10:
            benefits.append("富含膳食纤维，有助于消化")
        if nutrition.protein_g > 15:
            benefits.append("优质蛋白质来源")

        return {
            'sodium_level': sodium_level,
            'fat_level': fat_level,
            'sugar_level': sugar_level,
            'risk_factors': risk_factors,
            'health_benefits': benefits,
        }

    def get_recommendation(self, nutrition: NutritionProfile) -> ClinicalRecommendation:
        """获取推荐等级"""
        assessment = self.assess_nutrition(nutrition)

        risk_count = sum([1 for level in ['sodium_level', 'fat_level', 'sugar_level']
                         if assessment[level] == "high"])

        if risk_count >= 2:
            return ClinicalRecommendation.NOT_RECOMMEND
        elif risk_count == 1:
            return ClinicalRecommendation.CAUTION
        elif assessment['health_benefits']:
            return ClinicalRecommendation.RECOMMEND
        return ClinicalRecommendation.NEUTRAL


class ClinicalEvaluationSystem:
    """临床评估系统"""

    def __init__(self, model=None):
        self.model = model
        self.nutritionist = NutritionistExpertSystem()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate_single_dish(self, dish_name: str, nutrition: NutritionProfile,
                           region_id: int, cuisine_id: int, preferences: List[float]) -> Dict:
        """评估单个菜品"""
        assessment = self.nutritionist.assess_nutrition(nutrition)

        predicted_taste = 0.5
        predicted_health = nutrition.protein_g / 50.0 if nutrition.protein_g > 0 else 0.3

        risk_level = HealthRiskLevel.MODERATE
        if assessment['risk_factors']:
            if len(assessment['risk_factors']) > 1:
                risk_level = HealthRiskLevel.HIGH
            else:
                risk_level = HealthRiskLevel.LOW

        recommendation = self.nutritionist.get_recommendation(nutrition)

        return {
            'dish_name': dish_name,
            'predicted_taste': predicted_taste,
            'predicted_health': predicted_health,
            'nutrition': {
                'sodium_mg': nutrition.sodium_mg,
                'fat_g': nutrition.fat_g,
                'sugar_g': nutrition.sugar_g,
                'fiber_g': nutrition.fiber_g,
                'protein_g': nutrition.protein_g,
                'calories': nutrition.calories
            },
            'health_risk_level': risk_level.value,
            'recommendation': recommendation.value,
            'risk_factors': assessment['risk_factors'],
            'health_benefits': assessment['health_benefits'],
        }

    def evaluate_batch(self, dishes: List[Dict]) -> Dict:
        """批量评估"""
        results = []
        for dish in dishes:
            nutrition = NutritionProfile(
                sodium_mg=dish.get('sodium_mg', 500),
                fat_g=dish.get('fat_g', 15),
                sugar_g=dish.get('sugar_g', 10),
                fiber_g=dish.get('fiber_g', 3),
                protein_g=dish.get('protein_g', 10),
                calories=dish.get('calories', 300)
            )
            result = self.evaluate_single_dish(
                dish['name'],
                nutrition,
                dish.get('region_id', 0),
                dish.get('cuisine_id', 0),
                dish.get('preferences', [0.5] * 6)
            )
            results.append(result)

        return {
            'evaluations': results,
            'summary': self._generate_summary(results)
        }

    def _generate_summary(self, results: List[Dict]) -> Dict:
        """生成评估摘要"""
        recommendations = [r['recommendation'] for r in results]

        return {
            'total_dishes': len(results),
            'strongly_recommend': recommendations.count("强烈推荐"),
            'recommend': recommendations.count("推荐"),
            'neutral': recommendations.count("中性"),
            'caution': recommendations.count("谨慎"),
            'not_recommend': recommendations.count("不推荐"),
            'avg_predicted_health': np.mean([r['predicted_health'] for r in results]),
        }


def run_clinical_evaluation(model=None) -> Dict:
    """运行临床评估"""
    system = ClinicalEvaluationSystem(model)

    test_dishes = [
        {"name": "清蒸鲈鱼", "sodium_mg": 120, "fat_g": 8, "sugar_g": 0, "fiber_g": 0, "protein_g": 20, "calories": 150},
        {"name": "红烧肉", "sodium_mg": 900, "fat_g": 35, "sugar_g": 8, "fiber_g": 0, "protein_g": 20, "calories": 450},
        {"name": "清炒时蔬", "sodium_mg": 200, "fat_g": 5, "sugar_g": 3, "fiber_g": 5, "protein_g": 3, "calories": 80},
    ]

    return system.evaluate_batch(test_dishes)


if __name__ == '__main__':
    results = run_clinical_evaluation()
    print(json.dumps(results, indent=2, ensure_ascii=False))
