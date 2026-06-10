"""
健康评估服务模块
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import asyncio
import json

"""
健康评估服务
提供个性化健康评估和推荐功能
"""

import logging
logger = logging.getLogger(__name__)
from datetime import datetime, timedelta
import math

logger = logging.getLogger(__name__)

class HealthAssessmentService:
    """健康评估服务类"""

    def __init__(self):
        self.logger = logger

    async def assess_health(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行健康评估"""
        try:
            # 计算BMI
            bmi = self._calculate_bmi(request_data['height'], request_data['weight'])

            # 评估各个健康分类
            cardiovascular_score = self._assess_cardiovascular_health(
                request_data['blood_pressure_systolic'],
                request_data['blood_pressure_diastolic'],
                request_data['cholesterol'],
                bmi,
                request_data['age']
            )

            metabolic_score = self._assess_metabolic_health(
                request_data['glucose'],
                request_data.get('hba1c'),
                bmi,
                request_data['age']
            )

            lifestyle_score = self._assess_lifestyle_health(
                request_data['exercise_frequency'],
                request_data['exercise_duration'],
                request_data['sleep_hours'],
                request_data['stress_level'],
                request_data['alcohol_consumption']
            )

            nutrition_score = self._assess_nutrition_health(
                request_data['meal_frequency'],
                request_data['water_intake'],
                request_data['dietary_preferences'] or [],
                request_data['food_allergies'] or []
            )

            # 计算总体评分
            overall_score = (cardiovascular_score['score'] +
                           metabolic_score['score'] +
                           lifestyle_score['score'] +
                           nutrition_score['score']) / 4

            # 确定风险等级
            risk_level = self._determine_risk_level(overall_score)

            # 生成个性化建议
            personalized_recommendations = self._generate_personalized_recommendations(
                request_data, overall_score, risk_level
            )

            # 识别风险和保护因素
            risk_factors = self._identify_risk_factors(request_data, overall_score)
            protective_factors = self._identify_protective_factors(request_data, overall_score)

            # 计算下次评估日期
            next_assessment_date = (datetime.now() + timedelta(days=30)).isoformat()

            return {
                "overall_score": round(overall_score, 1),
                "risk_level": risk_level,
                "health_categories": {
                    "cardiovascular": cardiovascular_score,
                    "metabolic": metabolic_score,
                    "lifestyle": lifestyle_score,
                    "nutrition": nutrition_score
                },
                "personalized_recommendations": personalized_recommendations,
                "risk_factors": risk_factors,
                "protective_factors": protective_factors,
                "next_assessment_date": next_assessment_date
            }

        except Exception as e:
            self.logger.error(f"健康评估失败: {e}")
            raise

    def _calculate_bmi(self, height: float, weight: float) -> float:
        """计算BMI"""
        height_m = height / 100
        return weight / (height_m ** 2)

    def _assess_cardiovascular_health(self, systolic: int, diastolic: int,
                                    cholesterol: float, bmi: float, age: int) -> Dict[str, Any]:
        """评估心血管健康"""
        score = 100

        # 血压评估
        if systolic >= 140 or diastolic >= 90:
            score -= 30
        elif systolic >= 130 or diastolic >= 80:
            score -= 20
        elif systolic >= 120 or diastolic >= 80:
            score -= 10

        # 胆固醇评估
        if cholesterol >= 240:
            score -= 25
        elif cholesterol >= 200:
            score -= 15
        elif cholesterol >= 180:
            score -= 5

        # BMI评估
        if bmi >= 30:
            score -= 20
        elif bmi >= 25:
            score -= 10
        elif bmi < 18.5:
            score -= 5

        # 年龄因素
        if age >= 65:
            score -= 10
        elif age >= 50:
            score -= 5

        score = max(0, min(100, score))

        if score >= 80:
            status = "优秀"
        elif score >= 60:
            status = "良好"
        elif score >= 40:
            status = "一般"
        else:
            status = "需要改善"

        recommendations = []
        if systolic >= 130 or diastolic >= 80:
            recommendations.append("建议控制血压，减少钠盐摄入")
        if cholesterol >= 200:
            recommendations.append("建议控制胆固醇，增加膳食纤维")
        if bmi >= 25:
            recommendations.append("建议控制体重，增加运动")

        return {
            "score": round(score, 1),
            "status": status,
            "recommendations": recommendations
        }

    def _assess_metabolic_health(self, glucose: float, hba1c: Optional[float],
                               bmi: float, age: int) -> Dict[str, Any]:
        """评估代谢健康"""
        score = 100

        # 血糖评估
        if glucose >= 126:
            score -= 35
        elif glucose >= 100:
            score -= 20
        elif glucose >= 90:
            score -= 10

        # 糖化血红蛋白评估
        if hba1c:
            if hba1c >= 6.5:
                score -= 30
            elif hba1c >= 5.7:
                score -= 15
            elif hba1c >= 5.0:
                score -= 5

        # BMI对代谢的影响
        if bmi >= 30:
            score -= 20
        elif bmi >= 25:
            score -= 10

        # 年龄因素
        if age >= 65:
            score -= 10
        elif age >= 45:
            score -= 5

        score = max(0, min(100, score))

        if score >= 80:
            status = "优秀"
        elif score >= 60:
            status = "良好"
        elif score >= 40:
            status = "一般"
        else:
            status = "需要改善"

        recommendations = []
        if glucose >= 100:
            recommendations.append("建议控制血糖，减少精制碳水化合物摄入")
        if hba1c and hba1c >= 5.7:
            recommendations.append("建议定期监测血糖，控制饮食")
        if bmi >= 25:
            recommendations.append("建议减重，改善胰岛素敏感性")

        return {
            "score": round(score, 1),
            "status": status,
            "recommendations": recommendations
        }

    def _assess_lifestyle_health(self, exercise_freq: int, exercise_duration: int,
                               sleep_hours: float, stress_level: int,
                               alcohol_consumption: int) -> Dict[str, Any]:
        """评估生活方式健康"""
        score = 100

        # 运动评估
        weekly_exercise_minutes = exercise_freq * exercise_duration
        if weekly_exercise_minutes >= 150:
            score += 10  # 超过推荐量
        elif weekly_exercise_minutes >= 75:
            score += 0   # 达到推荐量
        else:
            score -= (75 - weekly_exercise_minutes) // 15 * 5

        # 睡眠评估
        if 7 <= sleep_hours <= 9:
            score += 0   # 理想睡眠时长
        elif 6 <= sleep_hours < 7 or 9 < sleep_hours <= 10:
            score -= 10
        else:
            score -= 20

        # 压力评估
        if stress_level <= 3:
            score += 5
        elif stress_level <= 5:
            score += 0
        elif stress_level <= 7:
            score -= 10
        else:
            score -= 20

        # 饮酒评估
        if alcohol_consumption == 0:
            score += 5
        elif alcohol_consumption <= 7:
            score += 0
        else:
            score -= alcohol_consumption * 2

        score = max(0, min(100, score))

        if score >= 80:
            status = "优秀"
        elif score >= 60:
            status = "良好"
        elif score >= 40:
            status = "一般"
        else:
            status = "需要改善"

        recommendations = []
        if weekly_exercise_minutes < 75:
            recommendations.append("建议增加运动量，每周至少150分钟中等强度运动")
        if sleep_hours < 7 or sleep_hours > 9:
            recommendations.append("建议调整睡眠时间，保持7-9小时规律睡眠")
        if stress_level > 5:
            recommendations.append("建议学习压力管理技巧，如冥想、深呼吸")
        if alcohol_consumption > 7:
            recommendations.append("建议减少饮酒量，每周不超过7次")

        return {
            "score": round(score, 1),
            "status": status,
            "recommendations": recommendations
        }

    def _assess_nutrition_health(self, meal_freq: int, water_intake: int,
                               dietary_preferences: List[str],
                               food_allergies: List[str]) -> Dict[str, Any]:
        """评估营养健康"""
        score = 100

        # 餐次评估
        if meal_freq == 3:
            score += 0   # 理想餐次
        elif meal_freq == 4 or meal_freq == 5:
            score += 5   # 少食多餐
        elif meal_freq < 3:
            score -= 15
        else:
            score -= 5

        # 饮水量评估
        if water_intake >= 2000:
            score += 5
        elif water_intake >= 1500:
            score += 0
        elif water_intake >= 1000:
            score -= 10
        else:
            score -= 20

        # 饮食偏好评估
        healthy_preferences = ['mediterranean', 'low_carb', 'high_protein', 'low_sodium']
        unhealthy_preferences = ['high_fat', 'high_sugar']

        for preference in dietary_preferences:
            if preference in healthy_preferences:
                score += 5
            elif preference in unhealthy_preferences:
                score -= 10

        # 食物过敏处理
        if food_allergies:
            score += 5  # 有过敏意识是好事

        score = max(0, min(100, score))

        if score >= 80:
            status = "优秀"
        elif score >= 60:
            status = "良好"
        elif score >= 40:
            status = "一般"
        else:
            status = "需要改善"

        recommendations = []
        if meal_freq < 3:
            recommendations.append("建议保持规律三餐，避免过度节食")
        if water_intake < 1500:
            recommendations.append("建议增加饮水量，每日至少1500ml")
        if not any(pref in dietary_preferences for pref in healthy_preferences):
            recommendations.append("建议采用更健康的饮食模式，如地中海饮食")

        return {
            "score": round(score, 1),
            "status": status,
            "recommendations": recommendations
        }

    def _determine_risk_level(self, overall_score: float) -> str:
        """确定风险等级"""
        if overall_score >= 80:
            return "低风险"
        elif overall_score >= 60:
            return "中等风险"
        else:
            return "高风险"

    def _generate_personalized_recommendations(self, request_data: Dict[str, Any],
                                            overall_score: float, risk_level: str) -> Dict[str, Any]:
        """生成个性化建议"""
        recommendations = {
            "immediate_actions": [],
            "long_term_goals": [],
            "dietary_suggestions": [],
            "exercise_plan": []
        }

        # 立即行动
        if request_data['glucose'] >= 100:
            recommendations["immediate_actions"].append("立即开始血糖监测")
        if request_data['blood_pressure_systolic'] >= 130:
            recommendations["immediate_actions"].append("减少钠盐摄入")
        if request_data['stress_level'] > 7:
            recommendations["immediate_actions"].append("学习压力管理技巧")

        # 长期目标
        if overall_score < 60:
            recommendations["long_term_goals"].append("制定3个月健康改善计划")
            recommendations["long_term_goals"].append("建立健康生活习惯")
        else:
            recommendations["long_term_goals"].append("维持当前健康状态")
            recommendations["long_term_goals"].append("持续优化生活方式")

        # 饮食建议
        cultural_background = request_data.get('cultural_background', 'chinese')
        if cultural_background == 'chinese':
            recommendations["dietary_suggestions"].append("增加蔬菜摄入，减少精制米面")
            recommendations["dietary_suggestions"].append("选择蒸煮烹饪方式")
        else:
            recommendations["dietary_suggestions"].append("增加全谷物和蔬菜摄入")
            recommendations["dietary_suggestions"].append("减少加工食品")

        # 运动计划
        if request_data['exercise_frequency'] < 3:
            recommendations["exercise_plan"].append("从每周3次30分钟中等强度运动开始")
        else:
            recommendations["exercise_plan"].append("增加运动强度或时长")

        return recommendations

    def _identify_risk_factors(self, request_data: Dict[str, Any], overall_score: float) -> List[str]:
        """识别风险因素"""
        risk_factors = []

        if request_data['glucose'] >= 100:
            risk_factors.append("血糖偏高")
        if request_data['blood_pressure_systolic'] >= 130:
            risk_factors.append("血压偏高")
        if request_data['cholesterol'] >= 200:
            risk_factors.append("胆固醇偏高")

        bmi = self._calculate_bmi(request_data['height'], request_data['weight'])
        if bmi >= 25:
            risk_factors.append("体重超标")

        if request_data['exercise_frequency'] < 3:
            risk_factors.append("运动不足")
        if request_data['sleep_hours'] < 7:
            risk_factors.append("睡眠不足")
        if request_data['stress_level'] > 7:
            risk_factors.append("压力过大")

        return risk_factors

    def _identify_protective_factors(self, request_data: Dict[str, Any], overall_score: float) -> List[str]:
        """识别保护因素"""
        protective_factors = []

        if request_data['glucose'] < 100:
            protective_factors.append("血糖正常")
        if request_data['blood_pressure_systolic'] < 120:
            protective_factors.append("血压正常")
        if request_data['cholesterol'] < 200:
            protective_factors.append("胆固醇正常")

        bmi = self._calculate_bmi(request_data['height'], request_data['weight'])
        if 18.5 <= bmi < 25:
            protective_factors.append("体重正常")

        if request_data['exercise_frequency'] >= 3:
            protective_factors.append("规律运动")
        if 7 <= request_data['sleep_hours'] <= 9:
            protective_factors.append("充足睡眠")
        if request_data['stress_level'] <= 5:
            protective_factors.append("压力管理良好")
        if request_data['water_intake'] >= 1500:
            protective_factors.append("充足饮水")

        return protective_factors

__all__ = ["'logger'", "'HealthAssessmentService'"]
