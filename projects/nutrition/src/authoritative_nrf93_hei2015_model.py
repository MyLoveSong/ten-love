#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
权威营养评分简化实现
提供 NRF9.3 与 HEI-2015 的轻量级近似，保证演示与验证流程可运行。
"""

from dataclasses import dataclass
from typing import Dict, List
import math


@dataclass
class AuthoritativeScores:
    nrf93_raw: float
    nrf93_normalized: float
    nrf93_grade: str
    hei_raw: float
    hei_normalized: float
    hei_grade: str
    composite_health_score: float
    health_classification: str
    clinical_recommendations: List[str]


class AuthoritativeNutritionDatabase:
    """
    简化版权威营养数据库：
    - 使用预定义营养档案近似 NRF9.3 与 HEI-2015
    - 若无命中则返回平均值
    """

    def __init__(self):
        self._profiles = {
            "清蒸鲈鱼": {"protein": 20, "fiber": 0, "sodium": 120, "fat": 8, "sugar": 0, "calorie": 150},
            "蒸蛋羹": {"protein": 12, "fiber": 0, "sodium": 200, "fat": 6, "sugar": 1, "calorie": 120},
            "清炒时蔬": {"protein": 4, "fiber": 8, "sodium": 300, "fat": 5, "sugar": 3, "calorie": 80},
            "宫保鸡丁": {"protein": 18, "fiber": 2, "sodium": 800, "fat": 15, "sugar": 8, "calorie": 250},
            "麻婆豆腐": {"protein": 15, "fiber": 3, "sodium": 900, "fat": 12, "sugar": 5, "calorie": 200},
            "红烧肉": {"protein": 20, "fiber": 0, "sodium": 1200, "fat": 25, "sugar": 12, "calorie": 400},
        }

    def _grade(self, score: float) -> str:
        if score >= 0.8:
            return "优秀"
        if score >= 0.6:
            return "良好"
        if score >= 0.4:
            return "一般"
        return "需改进"

    def _clinical_recs(self, profile: Dict[str, float], composite: float) -> List[str]:
        recs = []
        if profile["sodium"] > 800:
            recs.append("减少钠盐，建议使用低钠酱油或蒸煮方式。")
        if profile["fat"] > 20:
            recs.append("控制饱和脂肪，增加蒸煮替代油炸。")
        if profile["fiber"] < 3:
            recs.append("增加蔬菜与全谷物以提升膳食纤维。")
        if composite < 0.6:
            recs.append("整体健康评分偏低，建议重构食谱比例。")
        if not recs:
            recs.append("营养均衡，可保持现有制作方式。")
        return recs

    def get_authoritative_scores(self, dish_name: str) -> Dict:
        profile = self._profiles.get(dish_name, {"protein": 12, "fiber": 2, "sodium": 700, "fat": 12, "sugar": 6, "calorie": 220})

        # 简化 NRF9.3：蛋白质/纤维奖励 - 钠/脂/糖惩罚
        positive = profile["protein"] * 0.6 + profile["fiber"] * 1.2
        negative = (profile["sodium"] / 1000.0) * 0.8 + profile["fat"] * 0.05 + profile["sugar"] * 0.04
        nrf93_raw = max(0.0, positive - negative)
        nrf93_normalized = max(0.0, min(1.0, nrf93_raw / 15.0))

        # 简化 HEI-2015：热量密度与营养平衡
        calorie_norm = max(0.1, profile["calorie"] / 500.0)
        balance = (profile["protein"] * 0.4 + profile["fiber"] * 0.6) / calorie_norm
        penalty = (profile["sodium"] / 1200.0) * 0.3 + (profile["fat"] / 30.0) * 0.3
        hei_raw = max(0.0, balance - penalty)
        hei_normalized = max(0.0, min(1.0, hei_raw / 8.0))

        composite = 0.55 * nrf93_normalized + 0.45 * hei_normalized
        classification = self._grade(composite)

        return {
            "nrf93_results": {
                "nrf93_raw": float(nrf93_raw),
                "nrf93_normalized": float(nrf93_normalized),
                "health_grade": self._grade(nrf93_normalized),
            },
            "hei2015_results": {
                "hei_raw": float(hei_raw),
                "hei_normalized": float(hei_normalized),
                "health_grade": self._grade(hei_normalized),
            },
            "composite_health_score": float(composite),
            "health_classification": classification,
            "clinical_recommendations": self._clinical_recs(profile, composite),
        }


def demonstrate_authoritative_standards(dish_name: str) -> AuthoritativeScores:
    """供演示脚本使用的简化接口。"""
    db = AuthoritativeNutritionDatabase()
    scores = db.get_authoritative_scores(dish_name)
    nrf = scores["nrf93_results"]
    hei = scores["hei2015_results"]
    return AuthoritativeScores(
        nrf93_raw=nrf["nrf93_raw"],
        nrf93_normalized=nrf["nrf93_normalized"],
        nrf93_grade=nrf["health_grade"],
        hei_raw=hei["hei_raw"],
        hei_normalized=hei["hei_normalized"],
        hei_grade=hei["health_grade"],
        composite_health_score=scores["composite_health_score"],
        health_classification=scores["health_classification"],
        clinical_recommendations=scores["clinical_recommendations"],
    )
