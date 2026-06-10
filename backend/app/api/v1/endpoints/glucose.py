"""
血糖预测端点
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

"""
血糖预测API端点
企业级血糖预测服务
"""

logger = logging.getLogger(__name__)
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
import logging

from backend.app.core.exceptions import ModelError, ValidationError
from backend.app.services.glucose_service import GlucosePredictionService

logger = logging.getLogger(__name__)
router = APIRouter()

# 请求模型
class GlucosePredictionRequest(BaseModel):
    """血糖预测请求"""
    glucose: float = Field(..., ge=0, le=600, description="当前血糖值 (mg/dL)")
    hour: int = Field(..., ge=0, le=23, description="当前小时 (0-23)")
    carbohydrates: float = Field(..., ge=0, le=200, description="碳水化合物摄入量 (g)")
    insulin: Optional[float] = Field(None, ge=0, le=50, description="胰岛素剂量 (U)")
    exercise: Optional[float] = Field(None, ge=0, le=300, description="运动强度 (分钟)")
    stress: Optional[float] = Field(None, ge=0, le=10, description="压力水平 (1-10)")
    cultural_id: Optional[str] = Field(None, description="文化ID")
    temporal_features: Optional[Dict[str, Any]] = Field(None, description="时序特征")

class EnhancedGlucosePredictionRequest(BaseModel):
    """增强血糖预测请求"""
    glucose: float = Field(..., ge=0, le=600, description="当前血糖值 (mg/dL)")
    hour: int = Field(..., ge=0, le=23, description="当前小时 (0-23)")
    carbohydrates: float = Field(..., ge=0, le=200, description="碳水化合物摄入量 (g)")
    insulin: Optional[float] = Field(None, ge=0, le=50, description="胰岛素剂量 (U)")
    exercise: Optional[float] = Field(None, ge=0, le=300, description="运动强度 (分钟)")
    stress: Optional[float] = Field(None, ge=0, le=10, description="压力水平 (1-10)")
    cultural_id: Optional[str] = Field(None, description="文化ID")
    temporal_features: Optional[Dict[str, Any]] = Field(None, description="时序特征")
    multimodal_data: Optional[Dict[str, Any]] = Field(None, description="多模态数据")

# 响应模型
class GlucosePredictionResponse(BaseModel):
    """血糖预测响应"""
    prediction: float = Field(..., description="预测血糖值 (mg/dL)")
    confidence: float = Field(..., ge=0, le=1, description="置信度")
    risk_level: str = Field(..., description="风险等级")
    recommendations: list = Field(..., description="建议")
    model_info: Dict[str, Any] = Field(..., description="模型信息")

class EnhancedGlucosePredictionResponse(BaseModel):
    """增强血糖预测响应"""
    prediction: float = Field(..., description="预测血糖值 (mg/dL)")
    confidence: float = Field(..., ge=0, le=1, description="置信度")
    risk_level: str = Field(..., description="风险等级")
    recommendations: list = Field(..., description="建议")
    cultural_adaptations: Dict[str, Any] = Field(..., description="文化适配")
    temporal_insights: Dict[str, Any] = Field(..., description="时序洞察")
    multimodal_analysis: Dict[str, Any] = Field(..., description="多模态分析")
    model_info: Dict[str, Any] = Field(..., description="模型信息")

# 依赖注入
def get_glucose_service() -> GlucosePredictionService:
    """获取血糖预测服务"""
    return GlucosePredictionService()

@router.post("/predict", response_model=GlucosePredictionResponse)
async def predict_glucose(
    request: GlucosePredictionRequest,
    service: GlucosePredictionService = Depends(get_glucose_service)
):
    """基础血糖预测"""
    try:
        logger.info(f"收到血糖预测请求: {request}")

        # 调用预测服务
        result = await service.predict_glucose(request)

        logger.info(f"血糖预测完成: {result}")
        return result

    except ValidationError as e:
        logger.warning(f"验证错误: {e.detail}")
        raise HTTPException(status_code=422, detail=e.detail)
    except ModelError as e:
        logger.error(f"模型错误: {e.detail}")
        raise HTTPException(status_code=500, detail=e.detail)
    except Exception as e:
        logger.error(f"血糖预测失败: {e}")
        raise HTTPException(status_code=500, detail="血糖预测服务暂时不可用")

@router.post("/predict/enhanced", response_model=EnhancedGlucosePredictionResponse)
async def predict_glucose_enhanced(
    request: EnhancedGlucosePredictionRequest,
    service: GlucosePredictionService = Depends(get_glucose_service)
):
    """增强血糖预测"""
    try:
        logger.info(f"收到增强血糖预测请求: {request}")

        # 调用增强预测服务
        result = await service.predict_glucose_enhanced(request)

        logger.info(f"增强血糖预测完成: {result}")
        return result

    except ValidationError as e:
        logger.warning(f"验证错误: {e.detail}")
        raise HTTPException(status_code=422, detail=e.detail)
    except ModelError as e:
        logger.error(f"模型错误: {e.detail}")
        raise HTTPException(status_code=500, detail=e.detail)
    except Exception as e:
        logger.error(f"增强血糖预测失败: {e}")
        raise HTTPException(status_code=500, detail="增强血糖预测服务暂时不可用")

@router.post("/predict/gluformer", response_model=EnhancedGlucosePredictionResponse)
async def predict_glucose_gluformer(
    request: EnhancedGlucosePredictionRequest,
    service: GlucosePredictionService = Depends(get_glucose_service)
):
    """GluFormer血糖预测"""
    try:
        logger.info(f"收到GluFormer血糖预测请求: {request}")

        # 调用GluFormer预测服务
        result = await service.predict_glucose_gluformer(request)

        logger.info(f"GluFormer血糖预测完成: {result}")
        return result

    except ValidationError as e:
        logger.warning(f"验证错误: {e.detail}")
        raise HTTPException(status_code=422, detail=e.detail)
    except ModelError as e:
        logger.error(f"模型错误: {e.detail}")
        raise HTTPException(status_code=500, detail=e.detail)
    except Exception as e:
        logger.error(f"GluFormer血糖预测失败: {e}")
        raise HTTPException(status_code=500, detail="GluFormer血糖预测服务暂时不可用")

@router.get("/models/info")
async def get_models_info():
    """获取模型信息"""
    return {
        "models": [
            {
                "name": "基础血糖预测模型",
                "version": "1.0.0",
                "description": "基于传统机器学习的血糖预测模型",
                "endpoint": "/api/v1/glucose/predict"
            },
            {
                "name": "增强血糖预测模型",
                "version": "2.0.0",
                "description": "集成多模态数据的增强预测模型",
                "endpoint": "/api/v1/glucose/predict/enhanced"
            },
            {
                "name": "GluFormer模型",
                "version": "3.0.0",
                "description": "基于LSTM-GRU融合的时序感知血糖预测模型",
                "endpoint": "/api/v1/glucose/predict/gluformer"
            }
        ]
    }

__all__ = ["'logger'", "'router'", "'GlucosePredictionRequest'", "'EnhancedGlucosePredictionRequest'", "'GlucosePredictionResponse'", "'EnhancedGlucosePredictionResponse'", "'get_glucose_service'"]
