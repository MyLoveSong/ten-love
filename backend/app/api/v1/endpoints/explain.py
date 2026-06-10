"""
模型解释端点
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

"""
可解释性API端点
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

class ExplanationRequest(BaseModel):
    prediction_id: str
    explanation_type: str = "full"

@router.post("/prediction")
async def explain_prediction(request: ExplanationRequest):
    """解释预测结果"""
    try:
        # 模拟可解释性分析
        explanation = {
            "prediction_id": request.prediction_id,
            "explanation_type": request.explanation_type,
            "feature_importance": {
                "glucose": 0.4,
                "carbohydrates": 0.3,
                "hour": 0.2,
                "exercise": 0.1
            },
            "reasoning_path": [
                "血糖值影响最大",
                "碳水化合物摄入次之",
                "时间因素也有影响"
            ],
            "confidence": 0.88
        }

        logger.info(f"可解释性分析完成: {request.prediction_id}")
        return explanation

    except Exception as e:
        logger.error(f"可解释性分析失败: {e}")
        raise HTTPException(status_code=500, detail="可解释性服务暂时不可用")

__all__ = ["'logger'", "'router'", "'ExplanationRequest'"]
