

"""
Web服务接口
提供RESTful API服务，支持血糖预测、图像识别和综合健康评估
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from config.system_config import SYSTEM_CONFIG, API_CONFIG
from modules.glucose_prediction.predictor import GlucosePredictor
from modules.image_recognition.predictor import ImagePredictor
from modules.fusion.decision_engine import HealthDecisionEngine
from utils.logger import setup_logger

# 设置日志
logger = setup_logger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title=API_CONFIG["title"],
    description=API_CONFIG["description"],
    version=API_CONFIG["version"],
    docs_url=API_CONFIG["docs_url"],
    redoc_url=API_CONFIG["redoc_url"],
    openapi_url=API_CONFIG["openapi_url"]
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
glucose_predictor = None
image_predictor = None
health_engine = None

# 数据模型
class GlucosePredictionRequest(BaseModel):
    """血糖预测请求模型"""
    age: float = Field(..., ge=0, le=120, description="年龄")
    bmi: float = Field(..., ge=10, le=60, description="BMI")
    blood_pressure: float = Field(..., ge=60, le=200, description="血压")
    fasting_glucose: float = Field(..., ge=50, le=300, description="空腹血糖")
    gender: int = Field(..., ge=0, le=1, description="性别 (0=女性, 1=男性)")
    family_history: int = Field(..., ge=0, le=1, description="糖尿病家族史 (0=无, 1=有)")
    exercise_frequency: int = Field(..., ge=0, le=7, description="每周运动次数")

class HealthAssessmentRequest(BaseModel):
    """健康评估请求模型"""
    glucose_data: GlucosePredictionRequest
    user_profile: Optional[Dict[str, Any]] = Field(default=None, description="用户档案")

class SystemStatusResponse(BaseModel):
    """系统状态响应模型"""
    status: str
    modules: Dict[str, bool]
    system_info: Dict[str, Any]
    timestamp: str

# 启动事件
@app.on_event("startup")
async def startup_event():
    """应用启动时初始化模块"""
    global glucose_predictor, image_predictor, health_engine

    try:
        logger.info("正在初始化系统模块...")

        # 初始化血糖预测器
        glucose_predictor = GlucosePredictor()
        logger.info("血糖预测器初始化完成")

        # 初始化图像识别预测器
        image_predictor = ImagePredictor()
        logger.info("图像识别预测器初始化完成")

        # 初始化健康决策引擎
        health_engine = HealthDecisionEngine(
            glucose_predictor=glucose_predictor,
            image_predictor=image_predictor
        )
        logger.info("健康决策引擎初始化完成")

        logger.info("所有模块初始化完成")

    except Exception as e:
        logger.error(f"模块初始化失败: {e}")
        raise

# 根路径
@app.get("/")
async def root():
    """根路径，返回系统信息"""
    return {
        "message": "智能健康监测集成系统",
        "version": API_CONFIG["version"],
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "docs": "/docs"
    }

# 健康检查
@app.get("/health")
async def health_check():
    """健康检查接口"""
    try:
        status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "modules": {
                "glucose_predictor": glucose_predictor is not None,
                "image_predictor": image_predictor is not None,
                "health_engine": health_engine is not None
            }
        }
        return status
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        raise HTTPException(status_code=500, detail="系统健康检查失败")

# 系统状态
@app.get("/status", response_model=SystemStatusResponse)
async def get_system_status():
    """获取系统状态"""
    try:
        # 检查模块状态
        modules = {
            "glucose_predictor": glucose_predictor is not None,
            "image_predictor": image_predictor is not None,
            "health_engine": health_engine is not None
        }

        # 获取系统信息
        system_info = {
            "python_version": sys.version,
            "platform": sys.platform,
            "api_version": API_CONFIG["version"]
        }

        # 获取模型信息
        if glucose_predictor:
            system_info["glucose_model"] = glucose_predictor.get_model_info()
        if image_predictor:
            system_info["image_model"] = image_predictor.get_model_info()

        return SystemStatusResponse(
            status="running" if all(modules.values()) else "partial",
            modules=modules,
            system_info=system_info,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"获取系统状态失败: {e}")
        raise HTTPException(status_code=500, detail="获取系统状态失败")

# 血糖预测API
@app.post("/predict/glucose")
async def predict_glucose(request: GlucosePredictionRequest):
    """血糖预测接口"""
    try:
        if not glucose_predictor:
            raise HTTPException(status_code=503, detail="血糖预测模块未初始化")

        # 转换请求数据
        input_data = request.dict()

        # 进行预测
        result = glucose_predictor.predict(input_data)

        return {
            "success": True,
            "data": result,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"血糖预测失败: {e}")
        raise HTTPException(status_code=500, detail=f"血糖预测失败: {str(e)}")

# 图像识别API
@app.post("/predict/image")
async def predict_image(image: UploadFile = File(...)):
    """图像识别接口"""
    try:
        if not image_predictor:
            raise HTTPException(status_code=503, detail="图像识别模块未初始化")

        # 检查文件类型
        if not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="上传的文件不是图像")

        # 保存临时文件
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)

        temp_file_path = temp_dir / f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{image.filename}"

        with open(temp_file_path, "wb") as buffer:
            content = await image.read()
            buffer.write(content)

        try:
            # 进行预测
            result = image_predictor.predict(str(temp_file_path))

            return {
                "success": True,
                "data": result,
                "timestamp": datetime.now().isoformat()
            }

        finally:
            # 清理临时文件
            if temp_file_path.exists():
                temp_file_path.unlink()

    except Exception as e:
        logger.error(f"图像识别失败: {e}")
        raise HTTPException(status_code=500, detail=f"图像识别失败: {str(e)}")

# 综合健康评估API
@app.post("/assess/health")
async def assess_health(request: HealthAssessmentRequest):
    """综合健康评估接口"""
    try:
        if not health_engine:
            raise HTTPException(status_code=503, detail="健康决策引擎未初始化")

        # 转换请求数据
        glucose_data = request.glucose_data.dict()
        user_profile = request.user_profile or {}

        # 进行综合评估
        result = health_engine.assess_health(
            glucose_data=glucose_data,
            user_profile=user_profile
        )

        return {
            "success": True,
            "data": result,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"健康评估失败: {e}")
        raise HTTPException(status_code=500, detail=f"健康评估失败: {str(e)}")

# 批量血糖预测API
@app.post("/predict/glucose/batch")
async def batch_predict_glucose(requests: List[GlucosePredictionRequest]):
    """批量血糖预测接口"""
    try:
        if not glucose_predictor:
            raise HTTPException(status_code=503, detail="血糖预测模块未初始化")

        # 转换请求数据
        input_data_list = [request.dict() for request in requests]

        # 进行批量预测
        results = glucose_predictor.batch_predict(input_data_list)

        return {
            "success": True,
            "data": results,
            "count": len(results),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"批量血糖预测失败: {e}")
        raise HTTPException(status_code=500, detail=f"批量血糖预测失败: {str(e)}")

# 批量图像识别API
@app.post("/predict/image/batch")
async def batch_predict_image(images: List[UploadFile] = File(...)):
    """批量图像识别接口"""
    try:
        if not image_predictor:
            raise HTTPException(status_code=503, detail="图像识别模块未初始化")

        # 检查文件类型
        for image in images:
            if not image.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="上传的文件不是图像")

        # 保存临时文件
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)

        temp_files = []
        try:
            for image in images:
                temp_file_path = temp_dir / f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{image.filename}"

                with open(temp_file_path, "wb") as buffer:
                    content = await image.read()
                    buffer.write(content)

                temp_files.append(temp_file_path)

            # 进行批量预测
            image_paths = [str(f) for f in temp_files]
            results = image_predictor.batch_predict(image_paths)

            return {
                "success": True,
                "data": results,
                "count": len(results),
                "timestamp": datetime.now().isoformat()
            }

        finally:
            # 清理临时文件
            for temp_file in temp_files:
                if temp_file.exists():
                    temp_file.unlink()

    except Exception as e:
        logger.error(f"批量图像识别失败: {e}")
        raise HTTPException(status_code=500, detail=f"批量图像识别失败: {str(e)}")

# 模型信息API
@app.get("/models/info")
async def get_models_info():
    """获取模型信息"""
    try:
        models_info = {}

        if glucose_predictor:
            models_info["glucose"] = glucose_predictor.get_model_info()

        if image_predictor:
            models_info["image"] = image_predictor.get_model_info()

        return {
            "success": True,
            "data": models_info,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"获取模型信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取模型信息失败: {str(e)}")

# 错误处理
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理器"""
    logger.error(f"未处理的异常: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "内部服务器错误",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

def start_web_server():
    """启动Web服务器"""
    try:
        host = SYSTEM_CONFIG["host"]
        port = SYSTEM_CONFIG["port"]
        reload = SYSTEM_CONFIG["reload"]

        logger.info(f"启动Web服务器: http://{host}:{port}")
        logger.info(f"API文档: http://{host}:{port}/docs")

        uvicorn.run(
            "web_service:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )

    except Exception as e:
        logger.error(f"Web服务器启动失败: {e}")
        raise

if __name__ == "__main__":
    start_web_server()
__all__ = ["'project_root'", "'logger'", "'app'", "'glucose_predictor'", "'image_predictor'", "'health_engine'", "'GlucosePredictionRequest'", "'HealthAssessmentRequest'", "'SystemStatusResponse'", "'start_web_server'"]
