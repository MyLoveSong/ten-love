"""
糖尿病膳食管理API端点
提供完整的糖尿病膳食记录和管理功能
"""

from fastapi import APIRouter, HTTPException, Depends, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime
import json
import os
import tempfile

from backend.app.services.integrated_diabetes_service import create_integrated_diabetes_service, IntegratedDiabetesService

logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(prefix="/diabetes", tags=["diabetes_management"])

# 全局服务实例
diabetes_service = create_integrated_diabetes_service()

# Pydantic模型定义
class UserRegistrationRequest(BaseModel):
    """用户注册请求"""
    user_id: str = Field(..., description="用户ID")
    age: int = Field(..., ge=1, le=120, description="年龄")
    gender: str = Field(..., regex="^(male|female)$", description="性别")
    height: float = Field(..., gt=0, description="身高(cm)")
    weight: float = Field(..., gt=0, description="体重(kg)")
    diabetes_type: str = Field("type2", regex="^(type1|type2)$", description="糖尿病类型")
    diabetes_duration: int = Field(0, ge=0, description="糖尿病病程(年)")
    region: str = Field("广东", description="地区")
    cultural_preferences: Dict[str, Any] = Field(default_factory=dict, description="文化偏好")
    dietary_restrictions: List[str] = Field(default_factory=list, description="饮食限制")
    medication_info: Dict[str, Any] = Field(default_factory=dict, description="用药信息")
    target_glucose_range: List[float] = Field([4.0, 7.0], description="目标血糖范围")

class MealRecordRequest(BaseModel):
    """膳食记录请求"""
    user_id: str = Field(..., description="用户ID")
    meal_type: str = Field(..., description="餐次类型")
    food_items: List[Dict[str, Any]] = Field(..., description="食物项目")
    total_calories: float = Field(0, ge=0, description="总热量")
    total_carbs: float = Field(0, ge=0, description="总碳水化合物")
    total_protein: float = Field(0, ge=0, description="总蛋白质")
    total_fat: float = Field(0, ge=0, description="总脂肪")
    estimated_gi: float = Field(50, ge=0, le=100, description="估算GI值")
    description: Optional[str] = Field(None, description="描述")

class GlucoseRecordRequest(BaseModel):
    """血糖记录请求"""
    user_id: str = Field(..., description="用户ID")
    glucose_value: float = Field(..., gt=0, description="血糖值(mmol/L)")
    measurement_type: str = Field("random", description="测量类型")
    notes: Optional[str] = Field(None, description="备注")

class PredictionRequest(BaseModel):
    """血糖预测请求"""
    user_id: str = Field(..., description="用户ID")
    prediction_hours: int = Field(6, ge=1, le=24, description="预测时长(小时)")

class RecommendationRequest(BaseModel):
    """推荐请求"""
    user_id: str = Field(..., description="用户ID")
    meal_type: str = Field("lunch", description="餐次类型")
    current_glucose: Optional[float] = Field(None, description="当前血糖值")

# API端点实现
@router.post("/users/register")
async def register_user(request: UserRegistrationRequest) -> JSONResponse:
    """注册用户"""
    try:
        user_data = request.dict()
        # 转换target_glucose_range为元组
        user_data['target_glucose_range'] = tuple(user_data['target_glucose_range'])

        user_id = diabetes_service.register_user(user_data)

        return JSONResponse(
            status_code=201,
            content={
                "success": True,
                "message": "用户注册成功",
                "data": {
                    "user_id": user_id
                }
            }
        )
    except Exception as e:
        logger.error(f"用户注册失败: {e}")
        raise HTTPException(status_code=400, detail=f"用户注册失败: {str(e)}")

@router.post("/meals/record")
async def record_meal(request: MealRecordRequest) -> JSONResponse:
    """记录膳食"""
    try:
        meal_data = request.dict()
        user_id = meal_data.pop('user_id')
        description = meal_data.pop('description', None)

        meal_id = diabetes_service.record_meal(user_id, meal_data, description=description)

        return JSONResponse(
            status_code=201,
            content={
                "success": True,
                "message": "膳食记录成功",
                "data": {
                    "meal_id": meal_id
                }
            }
        )
    except Exception as e:
        logger.error(f"膳食记录失败: {e}")
        raise HTTPException(status_code=400, detail=f"膳食记录失败: {str(e)}")

@router.post("/meals/record-with-image")
async def record_meal_with_image(
    user_id: str = Form(...),
    meal_type: str = Form(...),
    description: str = Form(""),
    image: UploadFile = File(...)
) -> JSONResponse:
    """带图像的膳食记录"""
    try:
        # 保存上传的图像
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            content = await image.read()
            temp_file.write(content)
            temp_image_path = temp_file.name

        try:
            # 构建膳食数据
            meal_data = {
                'meal_type': meal_type,
                'food_items': [],  # 将通过图像识别填充
                'total_calories': 0,
                'total_carbs': 0,
                'total_protein': 0,
                'total_fat': 0,
                'estimated_gi': 50
            }

            meal_id = diabetes_service.record_meal(
                user_id, meal_data, image_path=temp_image_path, description=description
            )

            return JSONResponse(
                status_code=201,
                content={
                    "success": True,
                    "message": "带图像的膳食记录成功",
                    "data": {
                        "meal_id": meal_id
                    }
                }
            )
        finally:
            # 清理临时文件
            if os.path.exists(temp_image_path):
                os.unlink(temp_image_path)

    except Exception as e:
        logger.error(f"带图像的膳食记录失败: {e}")
        raise HTTPException(status_code=400, detail=f"膳食记录失败: {str(e)}")

@router.post("/glucose/record")
async def record_glucose(request: GlucoseRecordRequest) -> JSONResponse:
    """记录血糖"""
    try:
        glucose_data = request.dict()
        glucose_data['timestamp'] = datetime.now().isoformat()

        user_id = glucose_data.pop('user_id')
        glucose_id = diabetes_service.record_glucose(user_id, glucose_data)

        return JSONResponse(
            status_code=201,
            content={
                "success": True,
                "message": "血糖记录成功",
                "data": {
                    "glucose_id": glucose_id
                }
            }
        )
    except Exception as e:
        logger.error(f"血糖记录失败: {e}")
        raise HTTPException(status_code=400, detail=f"血糖记录失败: {str(e)}")

@router.post("/glucose/predict")
async def predict_glucose(request: PredictionRequest) -> JSONResponse:
    """预测血糖趋势"""
    try:
        prediction_result = diabetes_service.predict_glucose_trend(
            request.user_id, request.prediction_hours
        )

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "血糖预测成功",
                "data": prediction_result
            }
        )
    except Exception as e:
        logger.error(f"血糖预测失败: {e}")
        raise HTTPException(status_code=400, detail=f"血糖预测失败: {str(e)}")

@router.post("/recommendations/get")
async def get_recommendation(request: RecommendationRequest) -> JSONResponse:
    """获取个性化膳食推荐"""
    try:
        recommendation = diabetes_service.get_personalized_recommendation(
            request.user_id, request.meal_type, request.current_glucose
        )

        # 转换为可序列化的字典
        recommendation_dict = {
            'recommended_foods': recommendation.recommended_foods,
            'predicted_glucose_impact': recommendation.predicted_glucose_impact,
            'cultural_compatibility': recommendation.cultural_compatibility,
            'nutritional_balance_score': recommendation.nutritional_balance_score,
            'explanation': recommendation.explanation,
            'confidence_score': recommendation.confidence_score,
            'alternative_options': recommendation.alternative_options,
            'expert_reasoning': recommendation.expert_reasoning
        }

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "个性化推荐成功",
                "data": recommendation_dict
            }
        )
    except Exception as e:
        logger.error(f"个性化推荐失败: {e}")
        raise HTTPException(status_code=400, detail=f"个性化推荐失败: {str(e)}")

@router.get("/meals/{meal_id}/analysis")
async def analyze_meal_impact(meal_id: str, user_id: str) -> JSONResponse:
    """分析膳食影响"""
    try:
        analysis_result = diabetes_service.analyze_meal_impact(user_id, meal_id)

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "膳食影响分析成功",
                "data": analysis_result
            }
        )
    except Exception as e:
        logger.error(f"膳食影响分析失败: {e}")
        raise HTTPException(status_code=400, detail=f"膳食影响分析失败: {str(e)}")

@router.get("/users/{user_id}/report")
async def get_comprehensive_report(user_id: str, days: int = 7) -> JSONResponse:
    """获取综合报告"""
    try:
        if days < 1 or days > 30:
            raise ValueError("报告天数必须在1-30天之间")

        report = diabetes_service.get_comprehensive_report(user_id, days)

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "综合报告生成成功",
                "data": report
            }
        )
    except Exception as e:
        logger.error(f"综合报告生成失败: {e}")
        raise HTTPException(status_code=400, detail=f"综合报告生成失败: {str(e)}")

@router.get("/users/{user_id}/profile")
async def get_user_profile(user_id: str) -> JSONResponse:
    """获取用户档案"""
    try:
        if user_id not in diabetes_service.user_profiles:
            raise ValueError(f"用户 {user_id} 不存在")

        user_profile = diabetes_service.user_profiles[user_id]

        # 转换为可序列化的字典
        profile_dict = {
            'user_id': user_profile.user_id,
            'age': user_profile.age,
            'gender': user_profile.gender,
            'height': user_profile.height,
            'weight': user_profile.weight,
            'bmi': user_profile.weight / ((user_profile.height / 100) ** 2),
            'diabetes_type': user_profile.diabetes_type,
            'diabetes_duration': user_profile.diabetes_duration,
            'region': user_profile.region,
            'cultural_preferences': user_profile.cultural_preferences,
            'dietary_restrictions': user_profile.dietary_restrictions,
            'target_glucose_range': user_profile.target_glucose_range
        }

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "用户档案获取成功",
                "data": profile_dict
            }
        )
    except Exception as e:
        logger.error(f"用户档案获取失败: {e}")
        raise HTTPException(status_code=404, detail=f"用户档案获取失败: {str(e)}")

@router.get("/users/{user_id}/meals")
async def get_user_meals(user_id: str, limit: int = 10) -> JSONResponse:
    """获取用户膳食记录"""
    try:
        if user_id not in diabetes_service.meal_records:
            raise ValueError(f"用户 {user_id} 不存在")

        meals = diabetes_service.meal_records[user_id][-limit:]

        # 转换为可序列化的字典
        meals_data = []
        for meal in meals:
            meal_dict = {
                'meal_id': meal.meal_id,
                'timestamp': meal.timestamp.isoformat(),
                'meal_type': meal.meal_type,
                'food_items': meal.food_items,
                'total_calories': meal.total_calories,
                'total_carbs': meal.total_carbs,
                'total_protein': meal.total_protein,
                'total_fat': meal.total_fat,
                'estimated_gi': meal.estimated_gi,
                'description': meal.description
            }
            meals_data.append(meal_dict)

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "膳食记录获取成功",
                "data": {
                    "meals": meals_data,
                    "total_count": len(diabetes_service.meal_records[user_id])
                }
            }
        )
    except Exception as e:
        logger.error(f"膳食记录获取失败: {e}")
        raise HTTPException(status_code=404, detail=f"膳食记录获取失败: {str(e)}")

@router.get("/users/{user_id}/glucose")
async def get_user_glucose(user_id: str, limit: int = 20) -> JSONResponse:
    """获取用户血糖记录"""
    try:
        if user_id not in diabetes_service.glucose_records:
            raise ValueError(f"用户 {user_id} 不存在")

        glucose_readings = diabetes_service.glucose_records[user_id][-limit:]

        # 转换为可序列化的字典
        glucose_data = []
        for reading in glucose_readings:
            reading_dict = {
                'timestamp': reading.timestamp.isoformat(),
                'glucose_value': reading.glucose_value,
                'measurement_type': reading.measurement_type,
                'notes': reading.notes
            }
            glucose_data.append(reading_dict)

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "血糖记录获取成功",
                "data": {
                    "glucose_readings": glucose_data,
                    "total_count": len(diabetes_service.glucose_records[user_id])
                }
            }
        )
    except Exception as e:
        logger.error(f"血糖记录获取失败: {e}")
        raise HTTPException(status_code=404, detail=f"血糖记录获取失败: {str(e)}")

@router.get("/health")
async def health_check() -> JSONResponse:
    """健康检查"""
    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "message": "糖尿病膳食管理服务运行正常",
            "service": "diabetes_management",
            "version": "1.0.0",
            "features": [
                "用户管理",
                "膳食记录",
                "血糖监测",
                "智能预测",
                "个性化推荐",
                "文化适配",
                "专家系统",
                "可解释AI",
                "多源数据融合"
            ]
        }
    )

# 导出路由器
__all__ = ["router"]
