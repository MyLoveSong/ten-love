"""
训练模型API端点
提供训练好的模型预测服务
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from app.models.trained_model_service import (
    get_glucose_prediction_service,
    TrainedModelServiceFactory,
    PredictionInput,
    PredictionOutput,
    GlucosePredictionService
)

# 设置日志
logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter()


@router.get("/models/available", response_model=List[Dict[str, Any]])
async def get_available_models():
    """
    获取可用的训练模型列表

    Returns:
        可用模型信息列表
    """
    try:
        models = TrainedModelServiceFactory.get_available_models()
        logger.info(f"返回{len(models)}个可用模型")
        return models
    except Exception as e:
        logger.error(f"获取可用模型失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取模型列表失败: {str(e)}")


@router.get("/models/current/info")
async def get_current_model_info():
    """
    获取当前加载模型的信息

    Returns:
        当前模型信息
    """
    try:
        service = get_glucose_prediction_service()
        if not service.metadata:
            raise HTTPException(status_code=404, detail="未找到加载的模型")

        return {
            "model_name": service.metadata.model_name,
            "model_version": service.metadata.model_version,
            "training_date": service.metadata.training_date,
            "performance": {
                "best_val_loss": service.metadata.best_val_loss,
                "test_mae": service.metadata.test_mae,
                "test_rmse": service.metadata.test_rmse
            },
            "config": service.metadata.config,
            "model_path": service.metadata.model_path
        }
    except Exception as e:
        logger.error(f"获取当前模型信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取模型信息失败: {str(e)}")


@router.post("/predict/glucose", response_model=Dict[str, Any])
async def predict_glucose(
    glucose_sequence: List[float],
    timestamps: List[str],
    user_features: Optional[Dict[str, Any]] = None,
    meal_info: Optional[Dict[str, Any]] = None
):
    """
    血糖预测接口

    Args:
        glucose_sequence: 血糖序列数据
        timestamps: 对应的时间戳
        user_features: 用户特征（可选）
        meal_info: 餐食信息（可选）

    Returns:
        预测结果
    """
    try:
        # 验证输入
        if not glucose_sequence or not timestamps:
            raise HTTPException(status_code=400, detail="血糖序列和时间戳不能为空")

        if len(glucose_sequence) != len(timestamps):
            raise HTTPException(status_code=400, detail="血糖序列和时间戳长度不匹配")

        # 创建预测输入
        prediction_input = PredictionInput(
            glucose_sequence=glucose_sequence,
            timestamps=timestamps,
            user_features=user_features,
            meal_info=meal_info
        )

        # 获取预测服务
        service = get_glucose_prediction_service()

        # 执行预测
        result = service.predict(prediction_input)

        # 返回结果
        return {
            "success": True,
            "data": {
                "predictions": result.predictions,
                "prediction_times": result.prediction_times,
                "confidence_scores": result.confidence_scores,
                "risk_level": result.risk_level,
                "recommendations": result.recommendations,
                "model_version": result.model_version,
                "prediction_timestamp": result.prediction_timestamp
            },
            "message": "预测成功"
        }

    except ValueError as e:
        logger.error(f"输入验证失败: {e}")
        raise HTTPException(status_code=400, detail=f"输入数据无效: {str(e)}")
    except Exception as e:
        logger.error(f"血糖预测失败: {e}")
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


@router.post("/predict/glucose/batch", response_model=Dict[str, Any])
async def predict_glucose_batch(
    predictions: List[Dict[str, Any]]
):
    """
    批量血糖预测接口

    Args:
        predictions: 预测请求列表，每个包含glucose_sequence和timestamps

    Returns:
        批量预测结果
    """
    try:
        if not predictions:
            raise HTTPException(status_code=400, detail="预测请求列表不能为空")

        service = get_glucose_prediction_service()
        results = []

        for i, pred_request in enumerate(predictions):
            try:
                # 验证单个请求
                if 'glucose_sequence' not in pred_request or 'timestamps' not in pred_request:
                    results.append({
                        "success": False,
                        "error": "缺少必需字段glucose_sequence或timestamps",
                        "index": i
                    })
                    continue

                # 创建预测输入
                prediction_input = PredictionInput(
                    glucose_sequence=pred_request['glucose_sequence'],
                    timestamps=pred_request['timestamps'],
                    user_features=pred_request.get('user_features'),
                    meal_info=pred_request.get('meal_info')
                )

                # 执行预测
                result = service.predict(prediction_input)

                results.append({
                    "success": True,
                    "data": {
                        "predictions": result.predictions,
                        "prediction_times": result.prediction_times,
                        "confidence_scores": result.confidence_scores,
                        "risk_level": result.risk_level,
                        "recommendations": result.recommendations,
                        "model_version": result.model_version,
                        "prediction_timestamp": result.prediction_timestamp
                    },
                    "index": i
                })

            except Exception as e:
                logger.error(f"批量预测第{i}个请求失败: {e}")
                results.append({
                    "success": False,
                    "error": str(e),
                    "index": i
                })

        # 统计成功率
        success_count = sum(1 for r in results if r.get("success", False))
        success_rate = success_count / len(results) if results else 0

        return {
            "success": True,
            "data": {
                "results": results,
                "summary": {
                    "total_requests": len(predictions),
                    "successful_predictions": success_count,
                    "failed_predictions": len(results) - success_count,
                    "success_rate": success_rate
                }
            },
            "message": f"批量预测完成，成功率: {success_rate:.2%}"
        }

    except Exception as e:
        logger.error(f"批量血糖预测失败: {e}")
        raise HTTPException(status_code=500, detail=f"批量预测失败: {str(e)}")


@router.post("/models/reload")
async def reload_model(
    background_tasks: BackgroundTasks,
    model_path: Optional[str] = None
):
    """
    重新加载模型

    Args:
        model_path: 新模型路径（可选，默认使用最新模型）

    Returns:
        重新加载结果
    """
    try:
        def reload_model_task():
            """后台重新加载模型任务"""
            global _glucose_prediction_service
            from app.models.trained_model_service import _glucose_prediction_service

            try:
                # 创建新的服务实例
                new_service = TrainedModelServiceFactory.create_glucose_prediction_service(model_path)

                # 替换全局实例
                _glucose_prediction_service = new_service

                logger.info(f"模型重新加载成功: {new_service.metadata.model_name if new_service.metadata else 'Unknown'}")

            except Exception as e:
                logger.error(f"后台重新加载模型失败: {e}")

        # 添加后台任务
        background_tasks.add_task(reload_model_task)

        return {
            "success": True,
            "message": "模型重新加载任务已启动",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"启动模型重新加载任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"重新加载模型失败: {str(e)}")


@router.get("/health")
async def health_check():
    """
    健康检查接口

    Returns:
        服务健康状态
    """
    try:
        service = get_glucose_prediction_service()

        # 检查模型是否加载
        model_loaded = service.model is not None
        metadata_available = service.metadata is not None

        return {
            "status": "healthy" if model_loaded else "unhealthy",
            "model_loaded": model_loaded,
            "metadata_available": metadata_available,
            "model_info": {
                "name": service.metadata.model_name if metadata_available else None,
                "version": service.metadata.model_version if metadata_available else None,
                "training_date": service.metadata.training_date if metadata_available else None
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
