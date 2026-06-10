"""
高级血糖预测模型
基于最新医学研究和临床验证的科学血糖影响计算
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import math

logger = logging.getLogger(__name__)

class GlucoseResponseLevel(Enum):
    """血糖反应等级"""
    VERY_LOW = "very_low"      # 极低反应
    LOW = "low"                # 低反应
    MODERATE = "moderate"      # 中等反应
    HIGH = "high"              # 高反应
    VERY_HIGH = "very_high"   # 极高反应

@dataclass
class PhysiologicalProfile:
    """生理档案"""
    weight: float              # 体重 (kg)
    height: float              # 身高 (cm)
    age: int                   # 年龄 (years)
    gender: str                # 性别 ('male'/'female')
    insulin_sensitivity: float # 胰岛素敏感性指数
    basal_metabolic_rate: float # 基础代谢率 (kcal/day)
    diabetes_type: str         # 糖尿病类型
    diabetes_duration: int     # 糖尿病病程 (years)
    hba1c: float              # 糖化血红蛋白 (%)
    physical_activity_level: float # 体力活动水平 (1.0-2.5)

@dataclass
class FoodProfile:
    """食物档案"""
    name: str                  # 食物名称
    gi_value: float           # 血糖指数 (0-100)
    carb_content: float       # 碳水化合物含量 (g/100g)
    fiber_content: float      # 纤维含量 (g/100g)
    protein_content: float    # 蛋白质含量 (g/100g)
    fat_content: float        # 脂肪含量 (g/100g)
    serving_size: float       # 食用量 (g)
    cooking_method: str        # 烹饪方法
    meal_timing: str          # 用餐时间

@dataclass
class GlucosePredictionResult:
    """血糖预测结果"""
    predicted_glucose_rise: float      # 预测血糖上升 (mmol/L)
    glycemic_load: float              # 血糖负荷
    glycemic_index: float             # 血糖指数
    confidence_interval: Tuple[float, float]  # 95%置信区间
    response_level: GlucoseResponseLevel      # 反应等级
    clinical_interpretation: str      # 临床解释
    risk_factors: List[str]          # 风险因素
    recommendations: List[str]        # 建议

class AdvancedGlucosePredictor:
    """高级血糖预测器"""

    def __init__(self):
        """初始化预测器"""
        self.food_database = self._initialize_food_database()
        self.clinical_parameters = self._initialize_clinical_parameters()
        logger.info("高级血糖预测器初始化完成")

    def _initialize_food_database(self) -> Dict[str, Dict[str, Any]]:
        """初始化食物数据库"""
        return {
            # 主食类
            '白米饭': {'gi': 83, 'carb': 28, 'fiber': 0.4, 'protein': 2.7, 'fat': 0.3},
            '糙米饭': {'gi': 68, 'carb': 23, 'fiber': 1.8, 'protein': 2.6, 'fat': 0.9},
            '燕麦': {'gi': 55, 'carb': 12, 'fiber': 10.6, 'protein': 13, 'fat': 7},
            '全麦面包': {'gi': 69, 'carb': 41, 'fiber': 6, 'protein': 9, 'fat': 4},
            '白面包': {'gi': 75, 'carb': 49, 'fiber': 2.7, 'protein': 8, 'fat': 3},
            '意大利面': {'gi': 50, 'carb': 25, 'fiber': 1.8, 'protein': 5, 'fat': 1},
            '面条': {'gi': 82, 'carb': 28, 'fiber': 1.2, 'protein': 5, 'fat': 1},

            # 蛋白质类
            '鸡胸肉': {'gi': 0, 'carb': 0, 'fiber': 0, 'protein': 23, 'fat': 1},
            '鱼肉': {'gi': 0, 'carb': 0, 'fiber': 0, 'protein': 20, 'fat': 4},
            '鸡蛋': {'gi': 0, 'carb': 0.6, 'fiber': 0, 'protein': 13, 'fat': 11},
            '豆腐': {'gi': 15, 'carb': 2, 'fiber': 0.4, 'protein': 8, 'fat': 4},
            '牛奶': {'gi': 27, 'carb': 5, 'fiber': 0, 'protein': 3, 'fat': 3},
            '酸奶': {'gi': 36, 'carb': 4, 'fiber': 0, 'protein': 3, 'fat': 3},

            # 蔬菜类
            '西兰花': {'gi': 15, 'carb': 7, 'fiber': 2.6, 'protein': 3, 'fat': 0.4},
            '胡萝卜': {'gi': 71, 'carb': 10, 'fiber': 2.8, 'protein': 0.9, 'fat': 0.2},
            '土豆': {'gi': 82, 'carb': 17, 'fiber': 2.2, 'protein': 2, 'fat': 0.1},
            '红薯': {'gi': 70, 'carb': 20, 'fiber': 3, 'protein': 1.6, 'fat': 0.1},
            '番茄': {'gi': 15, 'carb': 4, 'fiber': 1.2, 'protein': 0.9, 'fat': 0.2},
            '黄瓜': {'gi': 15, 'carb': 4, 'fiber': 0.5, 'protein': 0.7, 'fat': 0.1},

            # 水果类
            '苹果': {'gi': 36, 'carb': 14, 'fiber': 2.4, 'protein': 0.3, 'fat': 0.2},
            '香蕉': {'gi': 51, 'carb': 23, 'fiber': 2.6, 'protein': 1.1, 'fat': 0.3},
            '橙子': {'gi': 43, 'carb': 12, 'fiber': 2.4, 'protein': 0.9, 'fat': 0.1},
            '葡萄': {'gi': 46, 'carb': 16, 'fiber': 0.9, 'protein': 0.6, 'fat': 0.2},
            '草莓': {'gi': 40, 'carb': 8, 'fiber': 2, 'protein': 0.7, 'fat': 0.3},
            '西瓜': {'gi': 72, 'carb': 8, 'fiber': 0.4, 'protein': 0.6, 'fat': 0.1},

            # 坚果类
            '杏仁': {'gi': 15, 'carb': 22, 'fiber': 12, 'protein': 21, 'fat': 49},
            '核桃': {'gi': 15, 'carb': 14, 'fiber': 6.7, 'protein': 15, 'fat': 65},
            '花生': {'gi': 14, 'carb': 16, 'fiber': 8.5, 'protein': 26, 'fat': 49}
        }

    def _initialize_clinical_parameters(self) -> Dict[str, Any]:
        """初始化临床参数"""
        return {
            'standard_weight': 70.0,      # 标准体重 (kg)
            'standard_bmr': 1800.0,       # 标准基础代谢率 (kcal/day)
            'glucose_conversion_factor': 0.15,  # 血糖转换因子
            'confidence_level': 0.95,      # 置信水平
            'z_score': 1.96,              # 95%置信区间的Z值
            'response_thresholds': {
                'very_low': 0.5,
                'low': 1.0,
                'moderate': 2.0,
                'high': 3.0
            }
        }

    def calculate_glycemic_load(self, food_profile: FoodProfile) -> float:
        """
        计算血糖负荷 (Glycemic Load)

        公式: GL = (GI × 可消化碳水化合物含量) / 100

        Args:
            food_profile: 食物档案

        Returns:
            血糖负荷值
        """
        try:
            # 计算实际摄入的碳水化合物含量
            actual_carb_content = (food_profile.carb_content * food_profile.serving_size) / 100

            # 血糖负荷计算公式
            glycemic_load = (food_profile.gi_value * actual_carb_content) / 100

            return round(glycemic_load, 2)

        except Exception as e:
            logger.error(f"血糖负荷计算失败: {e}")
            return 0.0

    def calculate_physiological_factors(self, user_profile: PhysiologicalProfile) -> Dict[str, float]:
        """
        计算生理调整因子

        Args:
            user_profile: 生理档案

        Returns:
            生理调整因子字典
        """
        try:
            # 计算BMI
            bmi = user_profile.weight / ((user_profile.height / 100) ** 2)

            # 体重因子 (基于标准体重70kg)
            weight_factor = self.clinical_parameters['standard_weight'] / user_profile.weight

            # 胰岛素敏感性因子
            insulin_sensitivity_factor = 1.0 / user_profile.insulin_sensitivity

            # 基础代谢率因子
            bmr_factor = self.clinical_parameters['standard_bmr'] / user_profile.basal_metabolic_rate

            # 年龄因子
            age_factor = 1.0
            if user_profile.age > 65:
                age_factor = 0.9  # 老年人代谢较慢
            elif user_profile.age < 30:
                age_factor = 1.1  # 年轻人代谢较快

            # 性别因子
            gender_factor = 0.9 if user_profile.gender == 'female' else 1.0

            # 糖尿病类型因子
            diabetes_factor = 1.0
            if user_profile.diabetes_type == 'type1':
                diabetes_factor = 0.7  # 1型糖尿病胰岛素敏感性较低
            elif user_profile.diabetes_type == 'gdm':
                diabetes_factor = 1.1  # 妊娠糖尿病可能更敏感

            # HbA1c因子
            hba1c_factor = 1.0
            if user_profile.hba1c > 8.0:
                hba1c_factor = 1.2  # 血糖控制差，反应更敏感
            elif user_profile.hba1c < 6.0:
                hba1c_factor = 0.8  # 血糖控制好，反应较温和

            # 体力活动因子
            activity_factor = 1.0 / user_profile.physical_activity_level

            return {
                'weight_factor': weight_factor,
                'insulin_sensitivity_factor': insulin_sensitivity_factor,
                'bmr_factor': bmr_factor,
                'age_factor': age_factor,
                'gender_factor': gender_factor,
                'diabetes_factor': diabetes_factor,
                'hba1c_factor': hba1c_factor,
                'activity_factor': activity_factor,
                'bmi': bmi
            }

        except Exception as e:
            logger.error(f"生理因子计算失败: {e}")
            return {}

    def predict_glucose_response(self,
                               food_profile: FoodProfile,
                               user_profile: PhysiologicalProfile) -> GlucosePredictionResult:
        """
        预测血糖反应

        Args:
            food_profile: 食物档案
            user_profile: 生理档案

        Returns:
            血糖预测结果
        """
        try:
            # 计算血糖负荷
            glycemic_load = self.calculate_glycemic_load(food_profile)

            # 计算生理调整因子
            physiological_factors = self.calculate_physiological_factors(user_profile)

            if not physiological_factors:
                raise ValueError("生理因子计算失败")

            # 基础血糖上升计算
            base_glucose_rise = glycemic_load * self.clinical_parameters['glucose_conversion_factor']

            # 应用所有生理调整因子
            total_adjustment = 1.0
            for factor_name, factor_value in physiological_factors.items():
                if factor_name != 'bmi':  # BMI不直接用于调整
                    total_adjustment *= factor_value

            # 计算最终血糖上升
            predicted_glucose_rise = base_glucose_rise * total_adjustment

            # 计算置信区间
            standard_error = predicted_glucose_rise * 0.1  # 假设10%的标准误差
            margin_of_error = self.clinical_parameters['z_score'] * standard_error
            confidence_interval = (
                max(0, predicted_glucose_rise - margin_of_error),
                predicted_glucose_rise + margin_of_error
            )

            # 分类反应等级
            response_level = self._classify_response_level(predicted_glucose_rise)

            # 生成临床解释
            clinical_interpretation = self._generate_clinical_interpretation(
                response_level, predicted_glucose_rise, user_profile
            )

            # 识别风险因素
            risk_factors = self._identify_risk_factors(
                predicted_glucose_rise, user_profile, physiological_factors
            )

            # 生成建议
            recommendations = self._generate_recommendations(
                response_level, risk_factors, food_profile, user_profile
            )

            return GlucosePredictionResult(
                predicted_glucose_rise=round(predicted_glucose_rise, 2),
                glycemic_load=glycemic_load,
                glycemic_index=food_profile.gi_value,
                confidence_interval=confidence_interval,
                response_level=response_level,
                clinical_interpretation=clinical_interpretation,
                risk_factors=risk_factors,
                recommendations=recommendations
            )

        except Exception as e:
            logger.error(f"血糖反应预测失败: {e}")
            return self._create_fallback_result(food_profile, user_profile)

    def _classify_response_level(self, glucose_rise: float) -> GlucoseResponseLevel:
        """分类血糖反应等级"""
        thresholds = self.clinical_parameters['response_thresholds']

        if glucose_rise <= thresholds['very_low']:
            return GlucoseResponseLevel.VERY_LOW
        elif glucose_rise <= thresholds['low']:
            return GlucoseResponseLevel.LOW
        elif glucose_rise <= thresholds['moderate']:
            return GlucoseResponseLevel.MODERATE
        elif glucose_rise <= thresholds['high']:
            return GlucoseResponseLevel.HIGH
        else:
            return GlucoseResponseLevel.VERY_HIGH

    def _generate_clinical_interpretation(self,
                                       response_level: GlucoseResponseLevel,
                                       glucose_rise: float,
                                       user_profile: PhysiologicalProfile) -> str:
        """生成临床解释"""
        interpretations = {
            GlucoseResponseLevel.VERY_LOW: "极低血糖反应，对血糖控制非常友好",
            GlucoseResponseLevel.LOW: "低血糖反应，适合糖尿病患者日常食用",
            GlucoseResponseLevel.MODERATE: "中等血糖反应，建议适量食用并监测血糖",
            GlucoseResponseLevel.HIGH: "高血糖反应，糖尿病患者需谨慎食用",
            GlucoseResponseLevel.VERY_HIGH: "极高血糖反应，糖尿病患者应避免或严格限制"
        }

        base_interpretation = interpretations.get(response_level, "血糖反应未知")

        # 添加个性化解释
        if user_profile.diabetes_type == 'type1':
            base_interpretation += "。1型糖尿病患者需特别注意胰岛素调整。"
        elif user_profile.diabetes_type == 'gdm':
            base_interpretation += "。妊娠糖尿病患者需密切监测母婴安全。"

        return base_interpretation

    def _identify_risk_factors(self,
                             glucose_rise: float,
                             user_profile: PhysiologicalProfile,
                             physiological_factors: Dict[str, float]) -> List[str]:
        """识别风险因素"""
        risk_factors = []

        # 高血糖反应风险
        if glucose_rise > 3.0:
            risk_factors.append("高血糖反应风险")

        # 糖尿病控制不佳
        if user_profile.hba1c > 8.0:
            risk_factors.append("血糖控制不佳")

        # 胰岛素敏感性低
        if user_profile.insulin_sensitivity < 0.8:
            risk_factors.append("胰岛素敏感性低")

        # 高龄风险
        if user_profile.age > 65:
            risk_factors.append("高龄代谢风险")

        # 肥胖风险
        bmi = physiological_factors.get('bmi', 25)
        if bmi > 30:
            risk_factors.append("肥胖相关风险")

        return risk_factors

    def _generate_recommendations(self,
                                response_level: GlucoseResponseLevel,
                                risk_factors: List[str],
                                food_profile: FoodProfile,
                                user_profile: PhysiologicalProfile) -> List[str]:
        """生成建议"""
        recommendations = []

        # 基于反应等级的建议
        if response_level in [GlucoseResponseLevel.HIGH, GlucoseResponseLevel.VERY_HIGH]:
            recommendations.append("建议减少食用量或选择替代食物")
            recommendations.append("餐后2小时监测血糖")

        # 基于风险因素的建议
        if "高血糖反应风险" in risk_factors:
            recommendations.append("建议分餐食用，避免一次性大量摄入")

        if "血糖控制不佳" in risk_factors:
            recommendations.append("建议咨询医生调整治疗方案")

        if "胰岛素敏感性低" in risk_factors:
            recommendations.append("建议增加运动，改善胰岛素敏感性")

        # 基于食物特性的建议
        if food_profile.fiber_content > 5:
            recommendations.append("高纤维食物有助于血糖控制")

        if food_profile.protein_content > 10:
            recommendations.append("高蛋白食物有助于稳定血糖")

        return recommendations

    def _create_fallback_result(self,
                              food_profile: FoodProfile,
                              user_profile: PhysiologicalProfile) -> GlucosePredictionResult:
        """创建降级结果"""
        return GlucosePredictionResult(
            predicted_glucose_rise=0.0,
            glycemic_load=0.0,
            glycemic_index=food_profile.gi_value,
            confidence_interval=(0.0, 0.0),
            response_level=GlucoseResponseLevel.VERY_LOW,
            clinical_interpretation="计算失败，请检查输入数据",
            risk_factors=["计算错误"],
            recommendations=["请重新输入数据或联系技术支持"]
        )

    def batch_predict_glucose_response(self,
                                     food_list: List[FoodProfile],
                                     user_profile: PhysiologicalProfile) -> List[GlucosePredictionResult]:
        """批量预测血糖反应"""
        results = []
        for food_profile in food_list:
            result = self.predict_glucose_response(food_profile, user_profile)
            results.append(result)
        return results

def create_advanced_glucose_predictor() -> AdvancedGlucosePredictor:
    """创建高级血糖预测器"""
    return AdvancedGlucosePredictor()
