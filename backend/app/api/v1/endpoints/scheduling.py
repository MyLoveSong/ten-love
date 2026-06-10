"""scheduling模块\n\n模块描述\n"""
from typing import Dict, Any, Optional
from fastapi import APIRouter
from pydantic import BaseModel

from backend.app.core.exceptions import CustomException
from backend.app.core.scheduling import ResourceSpec, select_engine_by_resources, to_engine_params
from backend.app.data_processing.automated_pipeline import (
    PipelineEngine, academic_pipeline_manager,
    execute_pipeline as execute_auto_pipeline
)

router = APIRouter()

class ScheduleJobRequest(BaseModel):
    pipeline_id: str
    preferred_engine: Optional[str] = None  # dask/spark/tensorflow/custom
    cpu: int = 2
    memory_gb: int = 4
    gpu: int = 0
    gpu_type: Optional[str] = None
    priority: str = "normal"

@router.post("/schedule/execute")
async def schedule_execute(req: ScheduleJobRequest):
    try:
        spec = ResourceSpec(cpu=req.cpu, memory_gb=req.memory_gb, gpu=req.gpu, gpu_type=req.gpu_type, priority=req.priority)
        engine_name = req.preferred_engine or select_engine_by_resources(spec)
        engine_map = {
            "dask": PipelineEngine.DASK,
            "spark": PipelineEngine.SPARK,
            "tensorflow": PipelineEngine.TENSORFLOW,
            "airflow": PipelineEngine.AIRFLOW,
            "gpu": PipelineEngine.DASK,  # 简单映射占位
            "local": PipelineEngine.DASK,
        }
        engine = engine_map.get(engine_name, PipelineEngine.DASK)
        params = to_engine_params(spec)

        execution = execute_auto_pipeline(req.pipeline_id, engine)
        return APIResponse.success(
            data={
                "execution_id": execution.execution_id,
                "engine": engine.value,
                "engine_params": params,
            },
            message="任务已根据资源规格调度执行"
        )
    except Exception as e:
        return APIResponse.error(message="调度执行失败", details={"error": str(e)})

__all__ = ["'router'", "'ScheduleJobRequest'"]
