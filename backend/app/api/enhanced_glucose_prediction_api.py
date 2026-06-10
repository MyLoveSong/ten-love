"""
增强的血糖预测API
集成麻雀搜索算法优化的TCN-GRU模型
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

from backend.app.services.integrated_diabetes_service import (
    create_integrated_diabetes_service,
    IntegratedDiabetesService
)

logger = logging.getLogger(__name__)

# 创建API路由器
router = APIRouter(prefix="/api/v2/glucose", tags=["Enhanced Glucose Prediction"])

# 全局服务实例
diabetes_service: Optional[IntegratedDiabetesService] = None

def get_diabetes_service() -> IntegratedDiabetesService:
    """获取糖尿病服务实例"""
    global diabetes_service
    if diabetes_service is None:
        diabetes_service = create_integrated_diabetes_service(use_ssa_optimization=True)
    return diabetes_service

# Pydantic模型
class UserRegistrationRequest(BaseModel):
    user_id: str
    age: int
    gender: str
    height: float
    weight: float
    diabetes_type: str = "type2"
    diabetes_duration: int = 0
    region: str = "广东"
    cultural_preferences: Dict[str, Any] = {}
    dietary_restrictions: List[str] = []
    medication_info: Dict[str, Any] = {}
    target_glucose_range: List[float] = [4.0, 7.0]

class GlucoseReadingRequest(BaseModel):
    timestamp: str
    glucose_value: float
    measurement_type: str = "random"
    notes: Optional[str] = None

class MealRecordRequest(BaseModel):
    meal_type: str
    food_items: List[Dict[str, Any]]
    total_calories: float
    total_carbs: float
    total_protein: float
    total_fat: float
    estimated_gi: float
    image_path: Optional[str] = None
    description: Optional[str] = None

class PredictionRequest(BaseModel):
    prediction_hours: int = 6
    current_glucose: Optional[float] = None

class OptimizationRequest(BaseModel):
    population_size: int = 15
    max_iterations: int = 30
    use_advanced_optimization: bool = True

class OptimizationResponse(BaseModel):
    status: str
    optimized_parameters: Optional[Dict[str, Any]] = None
    optimization_timestamp: Optional[str] = None
    error_message: Optional[str] = None

class PerformanceMetricsResponse(BaseModel):
    status: str
    model_type: str
    final_train_loss: Optional[float] = None
    final_val_loss: Optional[float] = None
    improvement_trend: Optional[float] = None
    best_params: Optional[Dict[str, Any]] = None
    training_epochs: Optional[int] = None
    error_message: Optional[str] = None

# API端点
@router.post("/register")
async def register_user(request: UserRegistrationRequest):
    """注册用户"""
    try:
        service = get_diabetes_service()
        user_data = request.dict()
        user_id = service.register_user(user_data)

        return {
            "status": "success",
            "user_id": user_id,
            "message": "用户注册成功"
        }
    except Exception as e:
        logger.error(f"用户注册失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/glucose/{user_id}")
async def record_glucose(user_id: str, request: GlucoseReadingRequest):
    """记录血糖读数"""
    try:
        service = get_diabetes_service()
        glucose_data = request.dict()
        glucose_id = service.record_glucose(user_id, glucose_data)

        return {
            "status": "success",
            "glucose_id": glucose_id,
            "message": "血糖记录成功"
        }
    except Exception as e:
        logger.error(f"血糖记录失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/meal/{user_id}")
async def record_meal(user_id: str, request: MealRecordRequest):
    """记录膳食"""
    try:
        service = get_diabetes_service()
        meal_data = request.dict()
        meal_id = service.record_meal(
            user_id,
            meal_data,
            request.image_path,
            request.description
        )

        return {
            "status": "success",
            "meal_id": meal_id,
            "message": "膳食记录成功"
        }
    except Exception as e:
        logger.error(f"膳食记录失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/predict/{user_id}")
async def predict_glucose_trend(user_id: str, request: PredictionRequest):
    """预测血糖趋势"""
    try:
        service = get_diabetes_service()
        prediction_result = service.predict_glucose_trend(
            user_id,
            request.prediction_hours
        )

        return {
            "status": "success",
            "prediction_result": prediction_result,
            "model_type": "SSA-TCN-GRU" if service.use_ssa_model else "GluFormer"
        }
    except Exception as e:
        logger.error(f"血糖预测失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/recommend/{user_id}")
async def get_personalized_recommendation(
    user_id: str,
    meal_type: str = "lunch",
    current_glucose: Optional[float] = None
):
    """获取个性化膳食推荐"""
    try:
        service = get_diabetes_service()
        recommendation = service.get_personalized_recommendation(
            user_id,
            meal_type,
            current_glucose
        )

        return {
            "status": "success",
            "recommendation": {
                "recommended_foods": recommendation.recommended_foods,
                "predicted_glucose_impact": recommendation.predicted_glucose_impact,
                "cultural_compatibility": recommendation.cultural_compatibility,
                "nutritional_balance_score": recommendation.nutritional_balance_score,
                "explanation": recommendation.explanation,
                "confidence_score": recommendation.confidence_score,
                "alternative_options": recommendation.alternative_options,
                "expert_reasoning": recommendation.expert_reasoning
            }
        }
    except Exception as e:
        logger.error(f"个性化推荐失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/optimize")
async def optimize_model_hyperparameters(request: OptimizationRequest):
    """优化模型超参数"""
    try:
        service = get_diabetes_service()

        if not service.use_ssa_model:
            return OptimizationResponse(
                status="not_applicable",
                error_message="当前使用的是传统GluFormer模型，无法进行超参数优化"
            )

        # 执行优化
        optimization_result = service.optimize_model_hyperparameters()

        return OptimizationResponse(**optimization_result)

    except Exception as e:
        logger.error(f"超参数优化失败: {e}")
        return OptimizationResponse(
            status="error",
            error_message=str(e)
        )

@router.get("/performance")
async def get_model_performance_metrics():
    """获取模型性能指标"""
    try:
        service = get_diabetes_service()
        performance_metrics = service.get_model_performance_metrics()

        return PerformanceMetricsResponse(**performance_metrics)

    except Exception as e:
        logger.error(f"获取性能指标失败: {e}")
        return PerformanceMetricsResponse(
            status="error",
            model_type="unknown",
            error_message=str(e)
        )

@router.get("/analyze/{user_id}/{meal_id}")
async def analyze_meal_impact(user_id: str, meal_id: str):
    """分析膳食影响"""
    try:
        service = get_diabetes_service()
        analysis_result = service.analyze_meal_impact(user_id, meal_id)

        return {
            "status": "success",
            "analysis_result": analysis_result
        }
    except Exception as e:
        logger.error(f"膳食影响分析失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/report/{user_id}")
async def get_comprehensive_report(user_id: str, days: int = 7):
    """获取综合报告"""
    try:
        service = get_diabetes_service()
        report = service.get_comprehensive_report(user_id, days)

        return {
            "status": "success",
            "report": report
        }
    except Exception as e:
        logger.error(f"综合报告生成失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/health")
async def health_check():
    """健康检查"""
    try:
        service = get_diabetes_service()
        return {
            "status": "healthy",
            "model_type": "SSA-TCN-GRU" if service.use_ssa_model else "GluFormer",
            "timestamp": datetime.now().isoformat(),
            "features": [
                "麻雀搜索算法超参数优化",
                "TCN-GRU混合神经网络",
                "个性化血糖预测",
                "文化适配膳食推荐",
                "可解释性分析"
            ]
        }
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(router, host="0.0.0.0", port=8001)
