

"""
多模态融合模块 - 健康决策引擎
整合血糖预测和图像识别结果，提供综合健康评估
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.system_config import HEALTH_CONFIG, FUSION_CONFIG
from backend.app.modules.fusionmultimodal_fusion import MultimodalFusionModel

class HealthDecisionEngine:
    """健康决策引擎类"""

    def __init__(self, glucose_predictor=None, image_predictor=None):
        """
        初始化健康决策引擎

        Args:
            glucose_predictor: 血糖预测器实例
            image_predictor: 图像识别预测器实例
        """
        self.logger = logging.getLogger(__name__)

        # 初始化预测器
        self.glucose_predictor = glucose_predictor
        self.image_predictor = image_predictor

        # 初始化融合模型
        self.fusion_model = None
        self._initialize_fusion_model()

        # 健康评估权重
        self.assessment_weights = HEALTH_CONFIG["assessment_weights"]

        # 风险阈值
        self.risk_thresholds = HEALTH_CONFIG["risk_thresholds"]
        self.glucose_ranges = HEALTH_CONFIG["glucose_ranges"]

        self.logger.info("健康决策引擎初始化完成")

    def _initialize_fusion_model(self):
        """初始化多模态融合模型"""
        try:
            self.fusion_model = MultimodalFusionModel(
                glucose_feature_dim=512,
                image_feature_dim=2048,
                hidden_dim=FUSION_CONFIG["hidden_dim"],
                num_heads=FUSION_CONFIG["attention_heads"],
                dropout=FUSION_CONFIG["dropout"]
            )

            self.logger.info("多模态融合模型初始化完成")

        except Exception as e:
            self.logger.error(f"融合模型初始化失败: {e}")
            self.fusion_model = None

    def assess_health(self, glucose_data: Dict[str, Any],
                     image_data: Optional[Dict[str, Any]] = None,
                     user_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        综合健康评估

        Args:
            glucose_data: 血糖相关数据
            image_data: 图像数据（可选）
            user_profile: 用户档案（可选）

        Returns:
            综合健康评估结果
        """
        try:
            # 初始化评估结果
            assessment = {
                "health_score": 0.0,
                "risk_assessment": {},
                "recommendations": [],
                "next_checkup": "",
                "timestamp": datetime.now().isoformat()
            }

            # 1. 血糖预测评估
            glucose_assessment = self._assess_glucose(glucose_data)

            # 2. 图像分析评估
            image_assessment = self._assess_image(image_data) if image_data else None

            # 3. 用户档案评估
            profile_assessment = self._assess_user_profile(user_profile) if user_profile else None

            # 4. 历史趋势评估
            trend_assessment = self._assess_historical_trend(glucose_data)

            # 5. 多模态融合评估
            fusion_assessment = self._perform_fusion_assessment(
                glucose_assessment, image_assessment, profile_assessment, trend_assessment
            )

            # 6. 综合评分和风险评估
            final_assessment = self._calculate_final_assessment(
                glucose_assessment, image_assessment, profile_assessment,
                trend_assessment, fusion_assessment
            )

            assessment.update(final_assessment)

            return assessment

        except Exception as e:
            self.logger.error(f"健康评估失败: {e}")
            raise

    def _assess_glucose(self, glucose_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        血糖评估

        Args:
            glucose_data: 血糖数据

        Returns:
            血糖评估结果
        """
        try:
            if not self.glucose_predictor:
                return self._assess_glucose_manual(glucose_data)

            # 使用血糖预测器
            prediction_result = self.glucose_predictor.predict(glucose_data)

            glucose_level = prediction_result["predicted_glucose"]
            confidence = prediction_result["confidence"]

            # 评估血糖风险
            risk_level = self._assess_glucose_risk(glucose_level)

            # 计算血糖评分
            glucose_score = self._calculate_glucose_score(glucose_level, confidence)

            return {
                "glucose_level": glucose_level,
                "confidence": confidence,
                "risk_level": risk_level,
                "score": glucose_score,
                "recommendations": prediction_result.get("recommendations", [])
            }

        except Exception as e:
            self.logger.error(f"血糖评估失败: {e}")
            return {"score": 50.0, "risk_level": "unknown"}

    def _assess_glucose_manual(self, glucose_data: Dict[str, Any]) -> Dict[str, Any]:
        """手动血糖评估（当预测器不可用时）"""
        try:
            fasting_glucose = glucose_data.get("fasting_glucose", 100)
            age = glucose_data.get("age", 45)
            bmi = glucose_data.get("bmi", 25)

            # 简单的风险评估
            if fasting_glucose < 70:
                risk_level = "low"
                score = 60.0
            elif fasting_glucose < 100:
                risk_level = "normal"
                score = 85.0
            elif fasting_glucose < 126:
                risk_level = "medium"
                score = 65.0
            else:
                risk_level = "high"
                score = 40.0

            # 考虑年龄和BMI
            if age > 50:
                score *= 0.95
            if bmi > 25:
                score *= 0.9

            return {
                "glucose_level": fasting_glucose,
                "confidence": 0.7,
                "risk_level": risk_level,
                "score": score,
                "recommendations": self._generate_glucose_recommendations(fasting_glucose)
            }

        except Exception as e:
            self.logger.error(f"手动血糖评估失败: {e}")
            return {"score": 50.0, "risk_level": "unknown"}

    def _assess_image(self, image_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        图像分析评估

        Args:
            image_data: 图像数据

        Returns:
            图像评估结果
        """
        try:
            if not self.image_predictor or "path" not in image_data:
                return None

            # 使用图像识别预测器
            prediction_result = self.image_predictor.predict(image_data["path"])

            classification = prediction_result["classification"]
            confidence = prediction_result["confidence"]

            # 评估图像风险
            risk_level = self._assess_image_risk(classification)

            # 计算图像评分
            image_score = self._calculate_image_score(classification, confidence)

            return {
                "classification": classification,
                "confidence": confidence,
                "risk_level": risk_level,
                "score": image_score,
                "recommendations": prediction_result.get("recommendations", [])
            }

        except Exception as e:
            self.logger.error(f"图像评估失败: {e}")
            return {"score": 70.0, "risk_level": "unknown"}

    def _assess_user_profile(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        用户档案评估

        Args:
            user_profile: 用户档案

        Returns:
            用户档案评估结果
        """
        try:
            score = 100.0
            risk_factors = []

            # 吸烟评估
            if user_profile.get("smoking", 0):
                score -= 20
                risk_factors.append("吸烟")

            # 饮酒评估
            if user_profile.get("alcohol", 0):
                score -= 15
                risk_factors.append("饮酒")

            # 压力水平评估
            stress_level = user_profile.get("stress_level", 3)
            if stress_level > 3:
                score -= (stress_level - 3) * 5
                risk_factors.append("高压力")

            # 睡眠评估
            sleep_hours = user_profile.get("sleep_hours", 7)
            if sleep_hours < 6 or sleep_hours > 9:
                score -= 10
                risk_factors.append("睡眠不足或过多")

            # 确定风险等级
            if score >= 80:
                risk_level = "low"
            elif score >= 60:
                risk_level = "medium"
            else:
                risk_level = "high"

            return {
                "score": max(0, score),
                "risk_level": risk_level,
                "risk_factors": risk_factors,
                "recommendations": self._generate_profile_recommendations(user_profile)
            }

        except Exception as e:
            self.logger.error(f"用户档案评估失败: {e}")
            return {"score": 70.0, "risk_level": "medium"}

    def _assess_historical_trend(self, glucose_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        历史趋势评估

        Args:
            glucose_data: 血糖数据

        Returns:
            历史趋势评估结果
        """
        try:
            # 这里可以添加历史数据分析逻辑
            # 目前使用简单的静态评估

            fasting_glucose = glucose_data.get("fasting_glucose", 100)
            age = glucose_data.get("age", 45)

            # 基于年龄和血糖水平的趋势评估
            if age > 50 and fasting_glucose > 100:
                trend_score = 60.0
                trend_direction = "increasing"
            elif fasting_glucose < 100:
                trend_score = 85.0
                trend_direction = "stable"
            else:
                trend_score = 70.0
                trend_direction = "stable"

            return {
                "score": trend_score,
                "trend_direction": trend_direction,
                "recommendations": ["建议定期监测血糖变化趋势"]
            }

        except Exception as e:
            self.logger.error(f"历史趋势评估失败: {e}")
            return {"score": 70.0, "trend_direction": "unknown"}

    def _perform_fusion_assessment(self, glucose_assessment: Dict[str, Any],
                                 image_assessment: Optional[Dict[str, Any]],
                                 profile_assessment: Optional[Dict[str, Any]],
                                 trend_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行多模态融合评估

        Args:
            glucose_assessment: 血糖评估结果
            image_assessment: 图像评估结果
            profile_assessment: 用户档案评估结果
            trend_assessment: 历史趋势评估结果

        Returns:
            融合评估结果
        """
        try:
            if not self.fusion_model:
                return self._simple_fusion(glucose_assessment, image_assessment,
                                         profile_assessment, trend_assessment)

            # 准备融合特征
            features = []

            # 血糖特征
            glucose_features = [
                glucose_assessment.get("glucose_level", 100),
                glucose_assessment.get("confidence", 0.7),
                glucose_assessment.get("score", 70.0)
            ]
            features.extend(glucose_features)

            # 图像特征（如果有）
            if image_assessment:
                image_features = [
                    image_assessment.get("confidence", 0.7),
                    image_assessment.get("score", 70.0)
                ]
                features.extend(image_features)
            else:
                features.extend([0.5, 70.0])  # 默认值

            # 用户档案特征（如果有）
            if profile_assessment:
                profile_features = [
                    profile_assessment.get("score", 70.0)
                ]
                features.extend(profile_features)
            else:
                features.extend([70.0])

            # 趋势特征
            trend_features = [
                trend_assessment.get("score", 70.0)
            ]
            features.extend(trend_features)

            # 转换为张量
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

            # 融合预测
            with torch.no_grad():
                fusion_score = self.fusion_model(features_tensor).item()

            return {
                "fusion_score": fusion_score,
                "method": "neural_fusion"
            }

        except Exception as e:
            self.logger.error(f"融合评估失败: {e}")
            return self._simple_fusion(glucose_assessment, image_assessment,
                                     profile_assessment, trend_assessment)

    def _simple_fusion(self, glucose_assessment: Dict[str, Any],
                      image_assessment: Optional[Dict[str, Any]],
                      profile_assessment: Optional[Dict[str, Any]],
                      trend_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """简单融合方法"""
        try:
            scores = []
            weights = []

            # 血糖评分
            scores.append(glucose_assessment.get("score", 70.0))
            weights.append(self.assessment_weights["glucose_prediction"])

            # 图像评分
            if image_assessment:
                scores.append(image_assessment.get("score", 70.0))
                weights.append(self.assessment_weights["image_analysis"])

            # 用户档案评分
            if profile_assessment:
                scores.append(profile_assessment.get("score", 70.0))
                weights.append(self.assessment_weights["user_profile"])

            # 趋势评分
            scores.append(trend_assessment.get("score", 70.0))
            weights.append(self.assessment_weights["historical_trend"])

            # 加权平均
            total_weight = sum(weights)
            fusion_score = sum(s * w for s, w in zip(scores, weights)) / total_weight

            return {
                "fusion_score": fusion_score,
                "method": "weighted_average"
            }

        except Exception as e:
            self.logger.error(f"简单融合失败: {e}")
            return {"fusion_score": 70.0, "method": "fallback"}

    def _calculate_final_assessment(self, glucose_assessment: Dict[str, Any],
                                  image_assessment: Optional[Dict[str, Any]],
                                  profile_assessment: Optional[Dict[str, Any]],
                                  trend_assessment: Dict[str, Any],
                                  fusion_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """
        计算最终评估结果

        Args:
            glucose_assessment: 血糖评估结果
            image_assessment: 图像评估结果
            profile_assessment: 用户档案评估结果
            trend_assessment: 历史趋势评估结果
            fusion_assessment: 融合评估结果

        Returns:
            最终评估结果
        """
        try:
            # 综合健康评分
            health_score = fusion_assessment.get("fusion_score", 70.0)

            # 风险评估
            risk_assessment = self._calculate_risk_assessment(
                glucose_assessment, image_assessment, profile_assessment
            )

            # 综合建议
            recommendations = self._generate_comprehensive_recommendations(
                glucose_assessment, image_assessment, profile_assessment,
                trend_assessment, health_score
            )

            # 下次检查时间
            next_checkup = self._calculate_next_checkup(health_score, risk_assessment)

            return {
                "health_score": health_score,
                "risk_assessment": risk_assessment,
                "recommendations": recommendations,
                "next_checkup": next_checkup
            }

        except Exception as e:
            self.logger.error(f"最终评估计算失败: {e}")
            return {
                "health_score": 70.0,
                "risk_assessment": {"level": "medium", "factors": ["评估失败"]},
                "recommendations": ["建议咨询专业医生"],
                "next_checkup": (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
            }

    def _calculate_risk_assessment(self, glucose_assessment: Dict[str, Any],
                                 image_assessment: Optional[Dict[str, Any]],
                                 profile_assessment: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """计算风险评估"""
        try:
            risk_factors = []
            risk_level = "low"

            # 血糖风险
            glucose_risk = glucose_assessment.get("risk_level", "unknown")
            if glucose_risk in ["medium", "high", "critical"]:
                risk_factors.append("血糖异常")
                if glucose_risk in ["high", "critical"]:
                    risk_level = "high"

            # 图像风险
            if image_assessment:
                image_risk = image_assessment.get("risk_level", "unknown")
                if image_risk in ["medium", "high"]:
                    risk_factors.append("图像异常")
                    if image_risk == "high":
                        risk_level = "high"

            # 用户档案风险
            if profile_assessment:
                profile_risk = profile_assessment.get("risk_level", "unknown")
                if profile_risk in ["medium", "high"]:
                    risk_factors.append("生活方式风险")
                    if profile_risk == "high":
                        risk_level = "high"

            # 如果没有风险因素，设为低风险
            if not risk_factors:
                risk_factors.append("无明显风险因素")

            return {
                "level": risk_level,
                "factors": risk_factors
            }

        except Exception as e:
            self.logger.error(f"风险评估计算失败: {e}")
            return {"level": "unknown", "factors": ["评估失败"]}

    def _generate_comprehensive_recommendations(self, glucose_assessment: Dict[str, Any],
                                              image_assessment: Optional[Dict[str, Any]],
                                              profile_assessment: Optional[Dict[str, Any]],
                                              trend_assessment: Dict[str, Any],
                                              health_score: float) -> List[str]:
        """生成综合建议"""
        try:
            recommendations = []

            # 添加血糖建议
            if glucose_assessment.get("recommendations"):
                recommendations.extend(glucose_assessment["recommendations"][:2])

            # 添加图像建议
            if image_assessment and image_assessment.get("recommendations"):
                recommendations.extend(image_assessment["recommendations"][:2])

            # 添加用户档案建议
            if profile_assessment and profile_assessment.get("recommendations"):
                recommendations.extend(profile_assessment["recommendations"][:2])

            # 基于健康评分的建议
            if health_score < 60:
                recommendations.append("健康评分较低，建议立即咨询医生进行全面检查")
            elif health_score < 80:
                recommendations.append("健康评分中等，建议改善生活方式并定期体检")
            else:
                recommendations.append("健康评分良好，继续保持健康的生活方式")

            # 去重并限制数量
            unique_recommendations = list(dict.fromkeys(recommendations))
            return unique_recommendations[:5]

        except Exception as e:
            self.logger.error(f"综合建议生成失败: {e}")
            return ["建议咨询专业医生获取个性化建议"]

    def _calculate_next_checkup(self, health_score: float, risk_assessment: Dict[str, Any]) -> str:
        """计算下次检查时间"""
        try:
            risk_level = risk_assessment.get("level", "medium")

            if risk_level == "high" or health_score < 60:
                days = 7  # 一周内
            elif risk_level == "medium" or health_score < 80:
                days = 30  # 一个月内
            else:
                days = 90  # 三个月内

            next_date = datetime.now() + timedelta(days=days)
            return next_date.strftime("%Y-%m-%d")

        except Exception as e:
            self.logger.error(f"下次检查时间计算失败: {e}")
            return (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")

    def _assess_glucose_risk(self, glucose_level: float) -> str:
        """评估血糖风险等级"""
        if glucose_level < 70:
            return "low"
        elif glucose_level < 100:
            return "normal"
        elif glucose_level < 126:
            return "medium"
        elif glucose_level < 200:
            return "high"
        else:
            return "critical"

    def _assess_image_risk(self, classification: str) -> str:
        """评估图像风险等级"""
        high_risk_classes = ["diabetic_retinopathy", "glaucoma", "cardiovascular"]
        medium_risk_classes = ["cataract", "hypertension", "kidney_disease", "liver_disease"]

        if classification in high_risk_classes:
            return "high"
        elif classification in medium_risk_classes:
            return "medium"
        else:
            return "low"

    def _calculate_glucose_score(self, glucose_level: float, confidence: float) -> float:
        """计算血糖评分"""
        base_score = 100.0

        if glucose_level < 70:
            base_score = 60.0
        elif glucose_level < 100:
            base_score = 85.0
        elif glucose_level < 126:
            base_score = 65.0
        else:
            base_score = 40.0

        return base_score * confidence

    def _calculate_image_score(self, classification: str, confidence: float) -> float:
        """计算图像评分"""
        base_scores = {
            "normal": 90.0,
            "diabetic_retinopathy": 30.0,
            "glaucoma": 40.0,
            "cataract": 60.0,
            "hypertension": 50.0,
            "cardiovascular": 35.0,
            "kidney_disease": 45.0,
            "liver_disease": 50.0,
            "thyroid": 70.0,
            "other": 60.0
        }

        base_score = base_scores.get(classification, 70.0)
        return base_score * confidence

    def _generate_glucose_recommendations(self, glucose_level: float) -> List[str]:
        """生成血糖建议"""
        if glucose_level < 70:
            return ["血糖偏低，建议适当增加碳水化合物摄入"]
        elif glucose_level < 100:
            return ["血糖水平正常，继续保持"]
        elif glucose_level < 126:
            return ["血糖偏高，建议控制饮食并增加运动"]
        else:
            return ["血糖明显偏高，建议咨询医生"]

    def _generate_profile_recommendations(self, user_profile: Dict[str, Any]) -> List[str]:
        """生成用户档案建议"""
        recommendations = []

        if user_profile.get("smoking", 0):
            recommendations.append("建议戒烟以改善健康状况")

        if user_profile.get("alcohol", 0):
            recommendations.append("建议适量饮酒或戒酒")

        stress_level = user_profile.get("stress_level", 3)
        if stress_level > 3:
            recommendations.append("建议学习压力管理技巧")

        sleep_hours = user_profile.get("sleep_hours", 7)
        if sleep_hours < 6 or sleep_hours > 9:
            recommendations.append("建议保持7-8小时的规律睡眠")

        return recommendations[:3]
__all__ = ["'project_root'", "'HealthDecisionEngine'"]
