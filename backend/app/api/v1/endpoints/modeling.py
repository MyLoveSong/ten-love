"""modeling模块

新增：/modeling/distill/align 端点，用于教师-学生跨领域蒸馏KL对齐（DistillationCoordinator）。
"""
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.exceptions import CustomException
from app.models.enhanced_knowledge_distillation import DistillationCoordinator
from app.modeling.model_registry import train as train_model, predict as predict_model, explain as explain_model

router = APIRouter()

class TrainRequest(BaseModel):
    model_name: str
    data: Dict[str, List[Any]]
    target: List[Any]
    params: Optional[Dict[str, Any]] = None

@router.post("/train")
async def train_endpoint(req: TrainRequest):
    try:
        tm = train_model(req.model_name, req.data, req.target, req.params)
        return APIResponse.success(data={
            "model_id": tm.model_id,
            "model_name": tm.model_name,
            "task_type": tm.task_type,
            "metrics": tm.metrics,
        }, message="模型训练完成")
    except Exception as e:
        return APIResponse.error(message="模型训练失败", details={"error": str(e)})

class PredictRequest(BaseModel):
    model_id: str
    data: Dict[str, List[Any]]

@router.post("/predict")
async def predict_endpoint(req: PredictRequest):
    try:
        result = predict_model(req.model_id, req.data)
        return APIResponse.success(data=result, message="预测完成")
    except Exception as e:
        return APIResponse.error(message="预测失败", details={"error": str(e)})

class ExplainRequest(BaseModel):
    model_id: str

@router.post("/explain")
async def explain_endpoint(req: ExplainRequest):
    try:
        result = explain_model(req.model_id)
        return APIResponse.success(data=result, message="解释完成")
    except Exception as e:
        return APIResponse.error(message="解释失败", details={"error": str(e)})

class DistillAlignRequest(BaseModel):
    """蒸馏对齐请求
    输入为各领域教师与学生的浮点矩阵（BxD），字段可包含 medical/nutritional/cultural/behavioral。
    """
    teacher: Dict[str, List[List[float]]] = Field(..., description="教师各领域输出")
    student: Dict[str, List[List[float]]] = Field(..., description="学生各领域输出")
    temperature: float = Field(2.0, ge=0.1, le=10.0)
    domain_weights: Optional[Dict[str, float]] = None
    teacher_uncertainty: Optional[Dict[str, float]] = None

@router.post("/distill/align")
async def distill_align(req: DistillAlignRequest):
    """返回聚合蒸馏损失与分领域损失"""
    try:
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        coord = DistillationCoordinator(temperature=req.temperature, domain_weights=req.domain_weights).to(device)

        def to_tensor_map(d: Dict[str, List[List[float]]]) -> Dict[str, torch.Tensor]:
            out: Dict[str, torch.Tensor] = {}
            for k, v in d.items():
                out[k] = torch.tensor(v, dtype=torch.float32, device=device)
            return out

        t_map = to_tensor_map(req.teacher)
        s_map = to_tensor_map(req.student)

        # 传入教师不确定性（可选）
        u_map = None
        if req.teacher_uncertainty:
            u_map = {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in req.teacher_uncertainty.items()}
        result = coord(t_map, s_map, teacher_uncertainty=u_map)
        total = float(result['distillation_loss'].detach().cpu().item())
        per_domain = {k: float(v.detach().cpu().item()) for k, v in result['per_domain_loss'].items()}
        return {"distillation_loss": total, "per_domain_loss": per_domain}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"蒸馏对齐失败: {e}")

__all__ = ["'router'", "'TrainRequest'", "'PredictRequest'", "'ExplainRequest'", "'DistillAlignRequest'"]
