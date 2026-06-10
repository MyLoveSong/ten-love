"""experiments模块\n\n模块描述\n"""
from typing import Dict, Any, List, Optional
from fastapi import APIRouter
from pydantic import BaseModel

from backend.app.core.exceptions import CustomException
from backend.app.experiments.experiment_runner import (
    kfold_train, multi_run, compare_models,
    enhanced_kfold_train, enhanced_compare_models
)
from app.experiments.automated_report_generation import report_generator
from app.experiments.advanced_visualization import advanced_visualizer
from app.experiments.batch_experiment_executor import batch_executor, ResourceRequirement, TaskPriority

router = APIRouter()

class KFoldRequest(BaseModel):
    model_name: str
    data: Dict[str, List[Any]]
    target: List[Any]
    k: int = 5
    params: Optional[Dict[str, Any]] = None
    enhanced: bool = False
    experiment_config: Optional[Dict[str, Any]] = None

@router.post("/kfold")
async def kfold_endpoint(req: KFoldRequest):
    try:
        if req.enhanced:
            res = enhanced_kfold_train(
                req.model_name, req.data, req.target, req.k, req.params, req.experiment_config
            )
        else:
            res = kfold_train(req.model_name, req.data, req.target, req.k, req.params)
        return APIResponse.success(data=res, message="KFold 完成")
    except Exception as e:
        return APIResponse.error(message="KFold 失败", details={"error": str(e)})

class MultiRunRequest(BaseModel):
    model_name: str
    data: Dict[str, List[Any]]
    target: List[Any]
    runs: int = 5
    params: Optional[Dict[str, Any]] = None

@router.post("/multi-run")
async def multi_run_endpoint(req: MultiRunRequest):
    try:
        res = multi_run(req.model_name, req.data, req.target, req.runs, req.params)
        return APIResponse.success(data=res, message="多次实验完成")
    except Exception as e:
        return APIResponse.error(message="多次实验失败", details={"error": str(e)})

class CompareRequest(BaseModel):
    model_names: List[str]
    data: Dict[str, List[Any]]
    target: List[Any]
    enhanced: bool = False
    runs: int = 5
    experiment_config: Optional[Dict[str, Any]] = None

@router.post("/compare")
async def compare_endpoint(req: CompareRequest):
    try:
        if req.enhanced:
            res = enhanced_compare_models(
                req.model_names, req.data, req.target, req.runs, req.experiment_config
            )
        else:
            res = compare_models(req.model_names, req.data, req.target)
        return APIResponse.success(data=res, message="模型比较完成")
    except Exception as e:
        return APIResponse.error(message="模型比较失败", details={"error": str(e)})

class ReportRequest(BaseModel):
    experiment_data: Dict[str, Any]
    template_name: str = "default"
    format: str = "html"
    include_charts: bool = True

@router.post("/generate-report")
async def generate_report_endpoint(req: ReportRequest):
    """生成实验报告"""
    try:
        report_path = report_generator.generate_experiment_report(
            experiment_data=req.experiment_data,
            template_name=req.template_name,
            format=req.format,
            include_charts=req.include_charts
        )
        return APIResponse.success(
            data={"report_path": report_path},
            message="报告生成完成"
        )
    except Exception as e:
        return APIResponse.error(message="报告生成失败", details={"error": str(e)})

class BatchReportRequest(BaseModel):
    batch_data: List[Dict[str, Any]]
    template_name: str = "batch"
    format: str = "html"

@router.post("/generate-batch-report")
async def generate_batch_report_endpoint(req: BatchReportRequest):
    """生成批量实验报告"""
    try:
        report_path = report_generator.generate_batch_report(
            batch_data=req.batch_data,
            template_name=req.template_name,
            format=req.format
        )
        return APIResponse.success(
            data={"report_path": report_path},
            message="批量报告生成完成"
        )
    except Exception as e:
        return APIResponse.error(message="批量报告生成失败", details={"error": str(e)})

class VisualizationRequest(BaseModel):
    visualization_type: str  # roc, pr, confusion_matrix, feature_importance, error_analysis, learning_curves, dashboard
    data: Dict[str, Any]
    model_name: str = "Model"
    save_path: Optional[str] = None

@router.post("/generate-visualization")
async def generate_visualization_endpoint(req: VisualizationRequest):
    """生成可视化图表"""
    try:
        viz_type = req.visualization_type
        data = req.data

        if viz_type == "roc":
            result = advanced_visualizer.plot_roc_curve(
                data['y_true'], data['y_prob'], req.model_name, req.save_path
            )
        elif viz_type == "pr":
            result = advanced_visualizer.plot_precision_recall_curve(
                data['y_true'], data['y_prob'], req.model_name, req.save_path
            )
        elif viz_type == "confusion_matrix":
            result = advanced_visualizer.plot_confusion_matrix(
                data['y_true'], data['y_pred'], data.get('labels'), req.model_name,
                data.get('normalize', False), req.save_path
            )
        elif viz_type == "feature_importance":
            result = advanced_visualizer.plot_feature_importance(
                data['feature_importance'], data.get('feature_names'),
                data.get('top_k', 20), req.model_name, req.save_path
            )
        elif viz_type == "error_analysis":
            result = advanced_visualizer.plot_error_analysis(
                data['y_true'], data['y_pred'], req.model_name,
                data.get('task_type', 'regression'), req.save_path
            )
        elif viz_type == "learning_curves":
            result = advanced_visualizer.plot_learning_curves(
                data['train_scores'], data['val_scores'],
                data.get('train_sizes'), req.model_name, req.save_path
            )
        elif viz_type == "dashboard":
            result = advanced_visualizer.create_visualization_dashboard(
                data, req.model_name, req.save_path
            )
        else:
            return APIResponse.error(message=f"不支持的可视化类型: {viz_type}")

        if result:
            return APIResponse.success(
                data={"visualization_path": result},
                message=f"{viz_type}可视化生成完成"
            )
        else:
            return APIResponse.error(message="可视化生成失败")

    except Exception as e:
        return APIResponse.error(message="可视化生成失败", details={"error": str(e)})

class BatchExperimentRequest(BaseModel):
    experiment_configs: List[Dict[str, Any]]
    resource_requirements: Optional[List[Dict[str, Any]]] = None
    priorities: Optional[List[str]] = None

@router.post("/submit-batch-experiments")
async def submit_batch_experiments_endpoint(req: BatchExperimentRequest):
    """提交批量实验"""
    try:
        # 转换资源需求
        resource_reqs = []
        if req.resource_requirements:
            for req_dict in req.resource_requirements:
                resource_reqs.append(ResourceRequirement(**req_dict))

        # 转换优先级
        priorities = []
        if req.priorities:
            for priority_str in req.priorities:
                priorities.append(TaskPriority[priority_str.upper()])

        # 启动调度器（如果未启动）
        if not batch_executor.running:
            batch_executor.start_scheduler()

        # 提交批量实验
        task_ids = batch_executor.submit_batch_experiments(
            experiment_configs=req.experiment_configs,
            resource_requirements=resource_reqs,
            priorities=priorities
        )

        return APIResponse.success(
            data={"task_ids": task_ids, "total_tasks": len(task_ids)},
            message="批量实验已提交"
        )
    except Exception as e:
        return APIResponse.error(message="批量实验提交失败", details={"error": str(e)})

@router.get("/batch-status")
async def get_batch_status_endpoint():
    """获取批量执行状态"""
    try:
        status = batch_executor.get_batch_status()
        return APIResponse.success(data=status, message="批量状态获取成功")
    except Exception as e:
        return APIResponse.error(message="批量状态获取失败", details={"error": str(e)})

@router.get("/task-status/{task_id}")
async def get_task_status_endpoint(task_id: str):
    """获取单个任务状态"""
    try:
        status = batch_executor.get_task_status(task_id)
        if status:
            return APIResponse.success(data=status, message="任务状态获取成功")
        else:
            return APIResponse.error(message="任务不存在")
    except Exception as e:
        return APIResponse.error(message="任务状态获取失败", details={"error": str(e)})

@router.post("/cancel-task/{task_id}")
async def cancel_task_endpoint(task_id: str):
    """取消任务"""
    try:
        success = batch_executor.cancel_task(task_id)
        if success:
            return APIResponse.success(message="任务已取消")
        else:
            return APIResponse.error(message="任务无法取消或不存在")
    except Exception as e:
        return APIResponse.error(message="任务取消失败", details={"error": str(e)})

@router.get("/batch-results")
async def get_batch_results_endpoint():
    """获取批量实验结果"""
    try:
        results = batch_executor.get_completed_results()
        return APIResponse.success(data=results, message="批量结果获取成功")
    except Exception as e:
        return APIResponse.error(message="批量结果获取失败", details={"error": str(e)})

@router.get("/batch-summary")
async def get_batch_summary_endpoint():
    """获取批量执行摘要"""
    try:
        summary = batch_executor.generate_batch_summary_report()
        return APIResponse.success(data=summary, message="批量摘要获取成功")
    except Exception as e:
        return APIResponse.error(message="批量摘要获取失败", details={"error": str(e)})

__all__ = ["'router'", "'KFoldRequest'", "'MultiRunRequest'", "'CompareRequest'", "'ReportRequest'", "'BatchReportRequest'", "'VisualizationRequest'", "'BatchExperimentRequest'"]
