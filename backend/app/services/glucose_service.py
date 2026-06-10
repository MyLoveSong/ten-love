"""
血糖预测服务模块
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
血糖预测服务
企业级血糖预测业务逻辑
"""

import logging
logger = logging.getLogger(__name__)
import numpy as np
import torch
from datetime import datetime

from backend.app.core.exceptions import ModelError, ValidationError
from backend.app.models.glucose_models import (
    AcademicGlucosePredictor,
    EnhancedGluFormer,
    EnhancedMultiModalFusion
)
from backend.app.services.cultural_service import CulturalAdaptationService
from backend.app.services.merl_service import MERLService

logger = logging.getLogger(__name__)

class GlucosePredictionService:
    """血糖预测服务"""

    def __init__(self):
        self.glucose_predictor = None
        self.enhanced_gluformer = None
        self.multimodal_fusion = None
        self.cultural_service = None
        self.merl_service = None

        # 初始化模型
        self._initialize_models()

    def _initialize_models(self):
        """初始化AI模型"""
        try:
            # 初始化基础血糖预测器
            self.glucose_predictor = AcademicGlucosePredictor()
            logger.info("✅ 基础血糖预测器初始化完成")

            # 初始化增强GluFormer
            self.enhanced_gluformer = EnhancedGluFormer(
                input_dim=10,
                hidden_dim=128,
                num_layers=3,
                cultural_dim=64,
                temporal_dim=32
            )
            logger.info("✅ 增强GluFormer初始化完成")

            # 初始化多模态融合
            self.multimodal_fusion = EnhancedMultiModalFusion(
                glucose_dim=10,
                image_dim=512,
                text_dim=256,
                behavioral_dim=8,
                medical_dim=16,
                cultural_dim=64,
                hidden_dim=256
            )
            logger.info("✅ 多模态融合初始化完成")

            # 初始化文化适配服务
            self.cultural_service = CulturalAdaptationService()
            logger.info("✅ 文化适配服务初始化完成")

            # 初始化MERL服务
            self.merl_service = MERLService()
            logger.info("✅ MERL服务初始化完成")

        except Exception as e:
            logger.error(f"❌ 模型初始化失败: {e}")
            raise ModelError(f"模型初始化失败: {e}")

    async def predict_glucose(self, request) -> Dict[str, Any]:
        """基础血糖预测"""
        try:
            # 准备输入数据
            input_data = self._prepare_basic_input(request)

            # 调用基础预测器
            prediction = self.glucose_predictor.predict(input_data)

            # 计算置信度和风险等级
            confidence = self._calculate_confidence(prediction, input_data)
            risk_level = self._assess_risk_level(prediction)
            recommendations = self._generate_recommendations(prediction, risk_level)

            return {
                "prediction": float(prediction),
                "confidence": confidence,
                "risk_level": risk_level,
                "recommendations": recommendations,
                "model_info": {
                    "model_name": "基础血糖预测模型",
                    "version": "1.0.0",
                    "timestamp": datetime.now().isoformat()
                }
            }

        except Exception as e:
            logger.error(f"基础血糖预测失败: {e}")
            raise ModelError(f"基础血糖预测失败: {e}")

    async def predict_glucose_enhanced(self, request) -> Dict[str, Any]:
        """增强血糖预测"""
        try:
            # 准备增强输入数据
            input_data = self._prepare_enhanced_input(request)

            # 多模态融合
            fused_features = self.multimodal_fusion(input_data)

            # 增强预测
            prediction = self.enhanced_gluformer.predict(fused_features)

            # 文化适配
            cultural_adaptations = await self.cultural_service.adapt_recommendations(
                request.cultural_id, prediction
            )

            # 计算置信度和风险等级
            confidence = self._calculate_enhanced_confidence(prediction, fused_features)
            risk_level = self._assess_risk_level(prediction)
            recommendations = self._generate_enhanced_recommendations(
                prediction, risk_level, cultural_adaptations
            )

            return {
                "prediction": float(prediction),
                "confidence": confidence,
                "risk_level": risk_level,
                "recommendations": recommendations,
                "cultural_adaptations": cultural_adaptations,
                "temporal_insights": self._analyze_temporal_patterns(input_data),
                "multimodal_analysis": self._analyze_multimodal_data(input_data),
                "model_info": {
                    "model_name": "增强血糖预测模型",
                    "version": "2.0.0",
                    "timestamp": datetime.now().isoformat()
                }
            }

        except Exception as e:
            logger.error(f"增强血糖预测失败: {e}")
            raise ModelError(f"增强血糖预测失败: {e}")

    async def predict_glucose_gluformer(self, request) -> Dict[str, Any]:
        """GluFormer血糖预测"""
        try:
            # 准备GluFormer输入数据
            input_data = self._prepare_gluformer_input(request)

            # GluFormer预测
            prediction = self.enhanced_gluformer.predict(input_data)

            # MERL优化
            merl_insights = await self.merl_service.optimize_prediction(
                input_data, prediction
            )

            # 文化适配
            cultural_adaptations = await self.cultural_service.adapt_recommendations(
                request.cultural_id, prediction
            )

            # 计算置信度和风险等级
            confidence = self._calculate_gluformer_confidence(prediction, input_data)
            risk_level = self._assess_risk_level(prediction)
            recommendations = self._generate_gluformer_recommendations(
                prediction, risk_level, cultural_adaptations, merl_insights
            )

            return {
                "prediction": float(prediction),
                "confidence": confidence,
                "risk_level": risk_level,
                "recommendations": recommendations,
                "cultural_adaptations": cultural_adaptations,
                "temporal_insights": self._analyze_temporal_patterns(input_data),
                "multimodal_analysis": self._analyze_multimodal_data(input_data),
                "merl_insights": merl_insights,
                "model_info": {
                    "model_name": "GluFormer模型",
                    "version": "3.0.0",
                    "timestamp": datetime.now().isoformat()
                }
            }

        except Exception as e:
            logger.error(f"GluFormer血糖预测失败: {e}")
            raise ModelError(f"GluFormer血糖预测失败: {e}")

    def _prepare_basic_input(self, request) -> np.ndarray:
        """准备基础输入数据"""
        return np.array([
            request.glucose,
            request.hour,
            request.carbohydrates,
            request.insulin or 0,
            request.exercise or 0,
            request.stress or 5
        ]).reshape(1, -1)

    def _prepare_enhanced_input(self, request) -> Dict[str, Any]:
        """准备增强输入数据"""
        return {
            "glucose": np.array([request.glucose]),
            "temporal": np.array([request.hour]),
            "nutritional": np.array([request.carbohydrates]),
            "medical": np.array([request.insulin or 0]),
            "behavioral": np.array([request.exercise or 0, request.stress or 5]),
            "cultural": np.array([hash(request.cultural_id or "default") % 100]),
            "multimodal": request.multimodal_data or {}
        }

    def _prepare_gluformer_input(self, request) -> Dict[str, Any]:
        """准备GluFormer输入数据"""
        return {
            "glucose": np.array([request.glucose]),
            "temporal": np.array([request.hour]),
            "nutritional": np.array([request.carbohydrates]),
            "medical": np.array([request.insulin or 0]),
            "behavioral": np.array([request.exercise or 0, request.stress or 5]),
            "cultural": np.array([hash(request.cultural_id or "default") % 100]),
            "temporal_features": request.temporal_features or {},
            "multimodal": request.multimodal_data or {}
        }

    def _calculate_confidence(self, prediction: float, input_data: np.ndarray) -> float:
        """计算基础预测置信度"""
        # 基于输入数据的完整性和预测值的合理性
        completeness = np.count_nonzero(input_data) / len(input_data.flatten())
        reasonableness = 1.0 - abs(prediction - 100) / 100  # 假设100为正常值
        return min(0.95, max(0.5, (completeness + reasonableness) / 2))

    def _calculate_enhanced_confidence(self, prediction: float, fused_features: torch.Tensor) -> float:
        """计算增强预测置信度"""
        # 基于融合特征的质量和预测值的合理性
        feature_quality = torch.mean(torch.abs(fused_features)).item()
        reasonableness = 1.0 - abs(prediction - 100) / 100
        return min(0.98, max(0.7, (feature_quality + reasonableness) / 2))

    def _calculate_gluformer_confidence(self, prediction: float, input_data: Dict[str, Any]) -> float:
        """计算GluFormer预测置信度"""
        # 基于时序特征的质量和预测值的合理性
        temporal_quality = len(input_data.get("temporal_features", {})) / 10
        reasonableness = 1.0 - abs(prediction - 100) / 100
        return min(0.99, max(0.8, (temporal_quality + reasonableness) / 2))

    def _assess_risk_level(self, prediction: float) -> str:
        """评估风险等级"""
        if prediction < 70:
            return "低血糖风险"
        elif prediction < 100:
            return "正常"
        elif prediction < 140:
            return "轻度升高"
        elif prediction < 200:
            return "中度升高"
        else:
            return "高风险"

    def _generate_recommendations(self, prediction: float, risk_level: str) -> list:
        """生成基础建议"""
        recommendations = []

        if risk_level == "低血糖风险":
            recommendations.extend([
                "立即摄入15-20g快速碳水化合物",
                "15分钟后重新检测血糖",
                "如症状持续，寻求医疗帮助"
            ])
        elif risk_level == "高风险":
            recommendations.extend([
                "立即联系医疗专业人员",
                "检查酮体水平",
                "调整胰岛素剂量"
            ])
        else:
            recommendations.extend([
                "继续监测血糖",
                "保持健康饮食",
                "适量运动"
            ])

        return recommendations

    def _generate_enhanced_recommendations(self, prediction: float, risk_level: str,
                                         cultural_adaptations: Dict[str, Any]) -> list:
        """生成增强建议"""
        recommendations = self._generate_recommendations(prediction, risk_level)

        # 添加文化适配建议
        if cultural_adaptations.get("dietary_recommendations"):
            recommendations.extend(cultural_adaptations["dietary_recommendations"])

        return recommendations

    def _generate_gluformer_recommendations(self, prediction: float, risk_level: str,
                                           cultural_adaptations: Dict[str, Any],
                                           merl_insights: Dict[str, Any]) -> list:
        """生成GluFormer建议"""
        recommendations = self._generate_enhanced_recommendations(
            prediction, risk_level, cultural_adaptations
        )

        # 添加MERL洞察建议
        if merl_insights.get("optimization_suggestions"):
            recommendations.extend(merl_insights["optimization_suggestions"])

        return recommendations

    def _analyze_temporal_patterns(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析时序模式"""
        return {
            "hour_pattern": f"当前时间: {input_data.get('temporal', [0])[0]}点",
            "trend_analysis": "基于历史数据的趋势分析",
            "seasonal_effects": "季节性影响评估"
        }

    def _analyze_multimodal_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析多模态数据"""
        return {
            "data_sources": list(input_data.get("multimodal", {}).keys()),
            "fusion_quality": "多模态融合质量评估",
            "correlation_analysis": "跨模态相关性分析"
        }

__all__ = ["'logger'", "'GlucosePredictionService'"]
