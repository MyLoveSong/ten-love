"""
模型生命周期管理API端点
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, List, Optional
import logging

from app.services.model_lifecycle_service import get_model_lifecycle_service
from app.services.enhanced_diabetes_service import get_enhanced_diabetes_service

# 设置日志
logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/status", response_model=Dict[str, Any])
async def get_lifecycle_status():
    """获取模型生命周期服务状态"""
    try:
        service = get_model_lifecycle_service()
        return service.get_service_status()
    except Exception as e:
        logger.error(f"获取生命周期状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取状态失败: {str(e)}")

@router.get("/models", response_model=List[Dict[str, Any]])
async def list_model_versions():
    """获取所有模型版本"""
    try:
        service = get_model_lifecycle_service()
        return service.get_model_versions()
    except Exception as e:
        logger.error(f"获取模型版本列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取模型版本失败: {str(e)}")

@router.get("/models/active", response_model=Dict[str, Any])
async def get_active_model():
    """获取当前活跃模型信息"""
    try:
        service = get_model_lifecycle_service()
        active_model = service.get_active_model_info()

        if not active_model:
            raise HTTPException(status_code=404, detail="没有活跃的模型")

        return active_model
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取活跃模型失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取活跃模型失败: {str(e)}")

@router.post("/training/trigger")
async def trigger_training(
    background_tasks: BackgroundTasks,
    job_type: str = "incremental",
    config: Optional[Dict[str, Any]] = None
):
    """触发模型训练"""
    try:
        service = get_model_lifecycle_service()

        if config is None:
            config = {
                'epochs': 10,
                'batch_size': 32,
                'learning_rate': 0.001,
                'use_amp': True
            }

        job_id = await service.trigger_training(job_type, config)

        return {
            'job_id': job_id,
            'job_type': job_type,
            'config': config,
            'message': f'训练任务已提交: {job_id}'
        }

    except Exception as e:
        logger.error(f"触发训练失败: {e}")
        raise HTTPException(status_code=500, detail=f"触发训练失败: {str(e)}")

@router.post("/models/{version_id}/deploy")
async def deploy_model_version(version_id: str):
    """部署指定版本的模型"""
    try:
        service = get_model_lifecycle_service()
        await service.deploy_model_version(version_id)

        return {
            'version_id': version_id,
            'message': f'模型版本 {version_id} 部署成功'
        }

    except Exception as e:
        logger.error(f"部署模型版本失败: {e}")
        raise HTTPException(status_code=500, detail=f"部署模型失败: {str(e)}")

@router.get("/training/status/{job_id}")
async def get_training_status(job_id: str):
    """获取训练任务状态"""
    try:
        service = get_model_lifecycle_service()
        training_service = service.training_service

        status = training_service.get_job_status(job_id)

        if not status:
            raise HTTPException(status_code=404, detail=f"训练任务不存在: {job_id}")

        return status

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取训练状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取训练状态失败: {str(e)}")

@router.get("/service/model-status")
async def get_service_model_status():
    """获取服务中的模型状态"""
    try:
        diabetes_service = get_enhanced_diabetes_service()
        return diabetes_service.get_model_status()

    except Exception as e:
        logger.error(f"获取服务模型状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取服务模型状态失败: {str(e)}")

@router.post("/service/trigger-training")
async def trigger_service_training(job_type: str = "incremental"):
    """通过服务触发训练"""
    try:
        diabetes_service = get_enhanced_diabetes_service()
        job_id = await diabetes_service.trigger_model_training(job_type)

        if not job_id:
            raise HTTPException(status_code=400, detail="无法触发训练，模型生命周期服务不可用")

        return {
            'job_id': job_id,
            'job_type': job_type,
            'message': f'通过服务触发训练任务: {job_id}'
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"通过服务触发训练失败: {e}")
        raise HTTPException(status_code=500, detail=f"触发训练失败: {str(e)}")

@router.get("/health")
async def health_check():
    """健康检查"""
    try:
        lifecycle_service = get_model_lifecycle_service()
        diabetes_service = get_enhanced_diabetes_service()

        return {
            'status': 'healthy',
            'timestamp': '2025-10-14T21:00:00Z',
            'services': {
                'model_lifecycle': {
                    'running': lifecycle_service.running,
                    'active_model': lifecycle_service.active_model_version is not None
                },
                'diabetes_service': {
                    'trained_model_available': diabetes_service.use_trained_model,
                    'lifecycle_integrated': diabetes_service.model_lifecycle_service is not None
                }
            }
        }

    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        raise HTTPException(status_code=500, detail=f"健康检查失败: {str(e)}")
