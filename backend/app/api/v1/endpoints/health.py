"""
健康检查端点
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

"""
健康检查与评估API端点
"""

logger = logging.getLogger(__name__)
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
import logging
from datetime import datetime, timedelta
import math

from backend.app.core.database import check_db_health
from backend.app.core.exceptions import ModelError, ValidationError
from backend.app.services.health_service import HealthAssessmentService

logger = logging.getLogger(__name__)
router = APIRouter()

# 请求模型
class HealthAssessmentRequest(BaseModel):
    """健康评估请求"""
    # 基本信息
    age: int = Field(..., ge=18, le=100, description="年龄")
    gender: str = Field(..., description="性别")
    height: float = Field(..., ge=100, le=250, description="身高 (cm)")
    weight: float = Field(..., ge=30, le=200, description="体重 (kg)")

    # 健康指标
    glucose: float = Field(..., ge=50, le=500, description="血糖 (mg/dL)")
    blood_pressure_systolic: int = Field(..., ge=80, le=200, description="收缩压 (mmHg)")
    blood_pressure_diastolic: int = Field(..., ge=50, le=120, description="舒张压 (mmHg)")
    cholesterol: float = Field(..., ge=100, le=400, description="胆固醇 (mg/dL)")
    hba1c: Optional[float] = Field(None, ge=3, le=15, description="糖化血红蛋白 (%)")

    # 生活方式
    exercise_frequency: int = Field(..., ge=0, le=7, description="每周运动次数")
    exercise_duration: int = Field(..., ge=0, le=300, description="每次运动时长 (分钟)")
    sleep_hours: float = Field(..., ge=4, le=12, description="每日睡眠时长 (小时)")
    stress_level: int = Field(..., ge=1, le=10, description="压力水平 (1-10)")

    # 饮食习惯
    meal_frequency: int = Field(..., ge=1, le=6, description="每日餐数")
    water_intake: int = Field(..., ge=500, le=5000, description="每日饮水量 (ml)")
    alcohol_consumption: int = Field(..., ge=0, le=14, description="每周饮酒次数")

    # 文化背景
    cultural_background: str = Field(..., description="文化背景")
    dietary_preferences: Optional[List[str]] = Field(None, description="饮食偏好")
    food_allergies: Optional[List[str]] = Field(None, description="食物过敏")

# 响应模型
class HealthCategoryScore(BaseModel):
    """健康分类评分"""
    score: float = Field(..., ge=0, le=100, description="评分")
    status: str = Field(..., description="状态")
    recommendations: List[str] = Field(..., description="建议")

class PersonalizedRecommendations(BaseModel):
    """个性化建议"""
    immediate_actions: List[str] = Field(..., description="立即行动")
    long_term_goals: List[str] = Field(..., description="长期目标")
    dietary_suggestions: List[str] = Field(..., description="饮食建议")
    exercise_plan: List[str] = Field(..., description="运动计划")

class HealthAssessmentResponse(BaseModel):
    """健康评估响应"""
    overall_score: float = Field(..., ge=0, le=100, description="总体评分")
    risk_level: str = Field(..., description="风险等级")
    health_categories: Dict[str, HealthCategoryScore] = Field(..., description="健康分类")
    personalized_recommendations: PersonalizedRecommendations = Field(..., description="个性化建议")
    risk_factors: List[str] = Field(..., description="风险因素")
    protective_factors: List[str] = Field(..., description="保护因素")
    next_assessment_date: str = Field(..., description="下次评估日期")

# 依赖注入
def get_health_service() -> HealthAssessmentService:
    """获取健康评估服务"""
    return HealthAssessmentService()

@router.get("/")
async def health_check():
    """系统健康检查"""
    db_healthy = await check_db_health()

    return {
        "status": "healthy" if db_healthy else "degraded",
        "version": "3.0.0",
        "service": "academic-health-monitoring-system",
        "database": "healthy" if db_healthy else "unhealthy",
        "timestamp": datetime.now().isoformat()
    }

@router.post("/assess", response_model=HealthAssessmentResponse)
async def assess_health(
    request: HealthAssessmentRequest,
    service: HealthAssessmentService = Depends(get_health_service)
):
    """健康评估"""
    try:
        logger.info(f"收到健康评估请求: {request}")

        # 调用评估服务
        result = await service.assess_health(request.dict())

        logger.info(f"健康评估完成: {result}")
        return result

    except ValidationError as e:
        logger.warning(f"验证错误: {e.detail}")
        raise HTTPException(status_code=422, detail=e.detail)
    except ModelError as e:
        logger.error(f"模型错误: {e.detail}")
        raise HTTPException(status_code=500, detail=e.detail)
    except Exception as e:
        logger.error(f"健康评估失败: {e}")
        raise HTTPException(status_code=500, detail="健康评估服务暂时不可用")

@router.get("/trend")
async def get_health_trend():
    """获取健康趋势数据"""
    try:
        # 模拟血糖趋势数据
        trend_data = []
        base_time = datetime.now() - timedelta(hours=24)

        for i in range(24):
            time_point = base_time + timedelta(hours=i)
            glucose = 100 + 20 * math.sin(i * math.pi / 12) + (i % 3) * 5
            prediction = glucose + 10 * math.cos(i * math.pi / 8)

            trend_data.append({
                "time": time_point.strftime("%H:%M"),
                "glucose": round(glucose, 1),
                "prediction": round(prediction, 1)
            })

        return {
            "success": True,
            "data": trend_data
        }

    except Exception as e:
        logger.error(f"获取健康趋势失败: {e}")
        raise HTTPException(status_code=500, detail="获取健康趋势失败")

@router.get("/distribution")
async def get_health_distribution():
    """获取健康状态分布"""
    try:
        distribution_data = [
            {"category": "健康", "count": 45, "color": "#52c41a"},
            {"category": "亚健康", "count": 30, "color": "#faad14"},
            {"category": "风险", "count": 20, "color": "#ff4d4f"},
            {"category": "高风险", "count": 5, "color": "#722ed1"}
        ]

        return {
            "success": True,
            "data": distribution_data
        }

    except Exception as e:
        logger.error(f"获取健康分布失败: {e}")
        raise HTTPException(status_code=500, detail="获取健康分布失败")

@router.get("/cultural/options")
async def get_cultural_options():
    """获取文化背景选项"""
    try:
        options = [
            {"value": "chinese", "label": "中华文化"},
            {"value": "western", "label": "西方文化"},
            {"value": "indian", "label": "印度文化"},
            {"value": "mediterranean", "label": "地中海文化"},
            {"value": "japanese", "label": "日本文化"},
            {"value": "korean", "label": "韩国文化"},
            {"value": "thai", "label": "泰国文化"},
            {"value": "mexican", "label": "墨西哥文化"},
        ]

        return {
            "success": True,
            "data": options
        }

    except Exception as e:
        logger.error(f"获取文化选项失败: {e}")
        raise HTTPException(status_code=500, detail="获取文化选项失败")

@router.get("/recipes/dietary-preferences")
async def get_dietary_preferences():
    """获取饮食偏好选项"""
    try:
        preferences = [
            {"value": "vegetarian", "label": "素食主义"},
            {"value": "vegan", "label": "纯素食"},
            {"value": "keto", "label": "生酮饮食"},
            {"value": "low_carb", "label": "低碳水化合物"},
            {"value": "low_fat", "label": "低脂肪"},
            {"value": "low_sodium", "label": "低钠"},
            {"value": "high_protein", "label": "高蛋白"},
            {"value": "mediterranean", "label": "地中海饮食"},
            {"value": "paleo", "label": "原始饮食"},
            {"value": "gluten_free", "label": "无麸质"},
        ]

        return {
            "success": True,
            "data": preferences
        }

    except Exception as e:
        logger.error(f"获取饮食偏好失败: {e}")
        raise HTTPException(status_code=500, detail="获取饮食偏好失败")

__all__ = ["'logger'", "'router'", "'HealthAssessmentRequest'", "'HealthCategoryScore'", "'PersonalizedRecommendations'", "'HealthAssessmentResponse'", "'get_health_service'"]
