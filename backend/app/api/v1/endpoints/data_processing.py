

"""
学术级数据处理API端点
整合数据清洗、特征工程、数据增强、自监督学习、多模态融合、质量监控、管道管理
"""

from typing import Dict, List, Optional, Any, Union
from fastapi import APIRouter, HTTPException, Depends, Request, BackgroundTasks, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import logging
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

from backend.app.core.exceptions import CustomException

# 导入数据处理模块
from backend.app.data_processing.data_cleaner import (
    DataCleaner, DataPreprocessor, PreprocessingConfig, MissingValueStrategy,
    OutlierDetectionMethod, ScalingMethod, EncodingMethod, preprocess_data,
    analyze_data_quality, get_preprocessing_history, get_quality_reports
)
from backend.app.data_processing.feature_engineer import (
    FeatureEngineer, FeatureSelectionMethod, FeatureConstructionMethod,
    FeatureTransformationMethod, DimensionalityReductionMethod, engineer_features,
    get_engineering_history
)
from backend.app.data_processing.data_augmenter import (
    DataAugmenter, AugmentationConfig, AugmentationType, ImageAugmentationMethod,
    TextAugmentationMethod, TimeSeriesAugmentationMethod, TabularAugmentationMethod,
    augment_data, get_augmentation_history
)
from backend.app.data_processing.self_supervised_learning import (
    SelfSupervisedLearner, SelfSupervisedConfig, SelfSupervisedMethod,
    ContrastiveMethod, MaskingStrategy, train_self_supervised, get_self_supervised_history
)
from backend.app.data_processing.multimodal_fusion import (
    MultimodalFusionManager, ModalityConfig, FusionConfig, ModalityType,
    FusionStrategy, AttentionMechanism, fuse_modalities, get_fusion_history
)
from backend.app.data_processing.data_quality_monitor import (
    DataQualityMonitor, QualityMetric, AnomalyType, AlertLevel, check_data_quality,
    detect_anomalies, get_active_alerts, get_quality_summary, resolve_alert
)
from backend.app.data_processing.data_pipeline_manager import (
    DataPipelineManager, PipelineConfig, TaskConfig, TaskType, TriggerType,
    create_pipeline, execute_pipeline, schedule_pipeline, get_pipeline_status,
    get_all_pipelines
)
from backend.app.data_processing.data_annotation import (
    AcademicDataLabeler, LabelingConfig, LabelingStrategy, ContrastiveMethod,
    LabelBalanceMethod, NoiseHandlingMethod, label_data, balance_labels, handle_label_noise
)
from backend.app.data_processing.automated_pipeline import (
    AcademicDataPipelineManager, PipelineConfig as AutoPipelineConfig,
    DataPipelineTask, PipelineEngine, DataSource, ProcessingMode,
    create_pipeline as create_auto_pipeline, execute_pipeline as execute_auto_pipeline,
    create_dask_preprocessing_pipeline, create_dask_contrastive_pipeline
)
from backend.app.data_processing.pipeline_registry import register_task, get_task

logger = logging.getLogger(__name__)
router = APIRouter()

# 请求模型
class PreprocessingRequest(BaseModel):
    data: Dict[str, List[Any]]
    missing_value_strategy: str = "mean"
    outlier_detection_method: str = "z_score"
    scaling_method: str = "standard"
    encoding_method: str = "one_hot"
    remove_duplicates: bool = True
    handle_outliers: bool = True
    scale_features: bool = True
    encode_categorical: bool = True

class FeatureEngineeringRequest(BaseModel):
    data: Dict[str, List[Any]]
    target: Optional[List[Any]] = None
    selection_method: Optional[str] = None
    construction_methods: List[str] = []
    transformation_methods: List[str] = []
    reduction_method: Optional[str] = None

class DataAugmentationRequest(BaseModel):
    data: Dict[str, List[Any]]
    augmentation_type: str
    methods: List[str]
    augmentation_factor: float = 2.0
    preserve_original: bool = True

class SelfSupervisedRequest(BaseModel):
    data: List[List[float]]
    method: str
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 100
    hidden_dim: int = 128

class MultimodalFusionRequest(BaseModel):
    data_dict: Dict[str, List[List[float]]]
    labels: List[int]
    modality_configs: List[Dict[str, Any]]
    fusion_config: Dict[str, Any]

class QualityCheckRequest(BaseModel):
    data: Dict[str, List[Any]]
    data_source: str = "unknown"

class AnomalyDetectionRequest(BaseModel):
    data: Dict[str, List[Any]]
    anomaly_type: str = "statistical"

class PipelineCreateRequest(BaseModel):
    pipeline_id: str
    name: str
    description: str
    version: str
    tasks: List[Dict[str, Any]]
    triggers: List[str]
    schedule: Optional[str] = None
    max_parallel_tasks: int = 5

class DataAnnotationRequest(BaseModel):
    data: Dict[str, List[Any]]
    strategy: str = "self_supervised"
    contrastive_method: Optional[str] = None
    balance_method: Optional[str] = None
    noise_handling: Optional[str] = None
    confidence_threshold: float = 0.8
    max_iterations: int = 100

class AutomatedPipelineRequest(BaseModel):
    pipeline_id: str
    name: str
    description: str
    engine: str
    data_source: str
    processing_mode: str
    schedule: Optional[str] = None
    tasks: List[Dict[str, Any]]
    max_parallel_tasks: int = 5
    memory_limit: str = "2GB"
    cpu_limit: int = 2

class DaskPresetCreateRequest(BaseModel):
    pipeline_id: str
    description: Optional[str] = "Dask预处理流水线"
    max_parallel_tasks: int = 4
    loader_name: str = "loader.csv"
    preprocess_name: str = "preprocess.default"
    save_name: str = "save.parquet"

class DaskContrastivePresetCreateRequest(BaseModel):
    pipeline_id: str
    description: Optional[str] = "Dask对比学习预设"
    max_parallel_tasks: int = 4
    loader_name: str = "loader.csv"
    preprocess_name: str = "preprocess.default"
    contrastive_name: str = "contrastive.simclr"
    save_name: str = "save.parquet"

# 数据预处理API
@router.post("/preprocessing/clean")
async def preprocess_data_endpoint(request: PreprocessingRequest):
    """数据预处理"""
    try:
        # 转换数据
        df = pd.DataFrame(request.data)

        # 创建配置
        config = PreprocessingConfig(
            missing_value_strategy=MissingValueStrategy(request.missing_value_strategy),
            outlier_detection_method=OutlierDetectionMethod(request.outlier_detection_method),
            scaling_method=ScalingMethod(request.scaling_method),
            encoding_method=EncodingMethod(request.encoding_method),
            remove_duplicates=request.remove_duplicates,
            handle_outliers=request.handle_outliers,
            scale_features=request.scale_features,
            encode_categorical=request.encode_categorical
        )

        # 执行预处理
        processed_df, quality_report = preprocess_data(df, config)

        return APIResponse.success(
            data={
                "processed_data": processed_df.to_dict(),
                "quality_report": {
                    "quality_score": quality_report.quality_score,
                    "total_rows": quality_report.total_rows,
                    "total_columns": quality_report.total_columns,
                    "missing_values": quality_report.missing_values,
                    "recommendations": quality_report.recommendations
                }
            },
            message="数据预处理完成"
        )

    except Exception as e:
        logger.error(f"数据预处理失败: {e}")
        return APIResponse.error(message="数据预处理失败", details={"error": str(e)})

@router.post("/preprocessing/analyze")
async def analyze_data_quality_endpoint(request: QualityCheckRequest):
    """分析数据质量"""
    try:
        df = pd.DataFrame(request.data)
        quality_report = analyze_data_quality(df)

        return APIResponse.success(
            data={
                "quality_score": quality_report.quality_score,
                "total_rows": quality_report.total_rows,
                "total_columns": quality_report.total_columns,
                "missing_values": quality_report.missing_values,
                "missing_percentage": quality_report.missing_percentage,
                "duplicate_rows": quality_report.duplicate_rows,
                "outlier_counts": quality_report.outlier_counts,
                "recommendations": quality_report.recommendations
            },
            message="数据质量分析完成"
        )

    except Exception as e:
        logger.error(f"数据质量分析失败: {e}")
        return APIResponse.error(message="数据质量分析失败", details={"error": str(e)})

# 特征工程API
@router.post("/feature-engineering/engineer")
async def engineer_features_endpoint(request: FeatureEngineeringRequest):
    """特征工程"""
    try:
        # 转换数据
        df = pd.DataFrame(request.data)
        y = pd.Series(request.target) if request.target else None

        # 解析配置
        selection_method = FeatureSelectionMethod(request.selection_method) if request.selection_method else None
        construction_methods = [FeatureConstructionMethod(m) for m in request.construction_methods]
        transformation_methods = [FeatureTransformationMethod(m) for m in request.transformation_methods]
        reduction_method = DimensionalityReductionMethod(request.reduction_method) if request.reduction_method else None

        # 执行特征工程
        engineered_df, report = engineer_features(
            df, y, selection_method, construction_methods, transformation_methods, reduction_method
        )

        return APIResponse.success(
            data={
                "engineered_data": engineered_df.to_dict(),
                "report": {
                    "original_features": report.original_features,
                    "constructed_features": report.constructed_features,
                    "final_features": report.final_features,
                    "selection_method": report.selection_method,
                    "construction_methods": report.construction_methods,
                    "transformation_methods": report.transformation_methods
                }
            },
            message="特征工程完成"
        )

    except Exception as e:
        logger.error(f"特征工程失败: {e}")
        return APIResponse.error(message="特征工程失败", details={"error": str(e)})

# 数据增强API
@router.post("/augmentation/augment")
async def augment_data_endpoint(request: DataAugmentationRequest):
    """数据增强"""
    try:
        # 转换数据
        df = pd.DataFrame(request.data)

        # 解析配置
        augmentation_type = AugmentationType(request.augmentation_type)

        if augmentation_type == AugmentationType.IMAGE:
            methods = [ImageAugmentationMethod(m) for m in request.methods]
        elif augmentation_type == AugmentationType.TEXT:
            methods = [TextAugmentationMethod(m) for m in request.methods]
        elif augmentation_type == AugmentationType.TIME_SERIES:
            methods = [TimeSeriesAugmentationMethod(m) for m in request.methods]
        elif augmentation_type == AugmentationType.TABULAR:
            methods = [TabularAugmentationMethod(m) for m in request.methods]
        else:
            methods = []

        config = AugmentationConfig(
            augmentation_type=augmentation_type,
            methods=methods,
            augmentation_factor=request.augmentation_factor,
            preserve_original=request.preserve_original
        )

        # 执行数据增强
        augmented_data, result = augment_data(df, config)

        return APIResponse.success(
            data={
                "augmented_data": augmented_data.to_dict() if hasattr(augmented_data, 'to_dict') else augmented_data,
                "result": {
                    "original_count": result.original_count,
                    "augmented_count": result.augmented_count,
                    "total_count": result.total_count,
                    "augmentation_methods": result.augmentation_methods,
                    "processing_time": result.processing_time
                }
            },
            message="数据增强完成"
        )

    except Exception as e:
        logger.error(f"数据增强失败: {e}")
        return APIResponse.error(message="数据增强失败", details={"error": str(e)})

# 自监督学习API
@router.post("/self-supervised/train")
async def train_self_supervised_endpoint(request: SelfSupervisedRequest):
    """自监督学习训练"""
    try:
        # 转换数据
        data_array = np.array(request.data)

        # 创建配置
        config = SelfSupervisedConfig(
            method=SelfSupervisedMethod(request.method),
            learning_rate=request.learning_rate,
            batch_size=request.batch_size,
            epochs=request.epochs,
            hidden_dim=request.hidden_dim
        )

        # 执行自监督学习
        result = train_self_supervised(data_array, config)

        return APIResponse.success(
            data={
                "method": result.method,
                "training_loss": result.training_loss,
                "representation_quality": result.representation_quality,
                "downstream_performance": result.downstream_performance,
                "training_time": result.training_time,
                "model_size": result.model_size
            },
            message="自监督学习训练完成"
        )

    except Exception as e:
        logger.error(f"自监督学习训练失败: {e}")
        return APIResponse.error(message="自监督学习训练失败", details={"error": str(e)})

# 多模态融合API
@router.post("/multimodal/fuse")
async def fuse_modalities_endpoint(request: MultimodalFusionRequest):
    """多模态数据融合"""
    try:
        # 转换数据
        data_np = {key: np.array(value) for key, value in request.data_dict.items()}
        labels_np = np.array(request.labels)

        # 转换配置
        modality_configs = [ModalityConfig(**config) for config in request.modality_configs]
        fusion_config = FusionConfig(**request.fusion_config)

        # 执行多模态融合
        result = fuse_modalities(data_np, labels_np, modality_configs, fusion_config)

        return APIResponse.success(
            data={
                "fusion_strategy": result.fusion_strategy,
                "modality_types": result.modality_types,
                "training_loss": result.training_loss,
                "test_accuracy": result.test_accuracy,
                "fusion_quality": result.fusion_quality,
                "training_time": result.training_time,
                "model_size": result.model_size
            },
            message="多模态融合完成"
        )

    except Exception as e:
        logger.error(f"多模态融合失败: {e}")
        return APIResponse.error(message="多模态融合失败", details={"error": str(e)})

# 数据质量监控API
@router.post("/quality/check")
async def check_data_quality_endpoint(request: QualityCheckRequest):
    """检查数据质量"""
    try:
        df = pd.DataFrame(request.data)
        report = check_data_quality(df, request.data_source)

        return APIResponse.success(
            data={
                "report_id": report.report_id,
                "data_source": report.data_source,
                "total_records": report.total_records,
                "quality_score": report.quality_score,
                "metric_scores": report.metric_scores,
                "alerts": [
                    {
                        "alert_id": alert.alert_id,
                        "metric": alert.metric.value,
                        "current_value": alert.current_value,
                        "threshold_value": alert.threshold_value,
                        "alert_level": alert.alert_level.value,
                        "message": alert.message
                    }
                    for alert in report.alerts
                ],
                "recommendations": report.recommendations
            },
            message="数据质量检查完成"
        )

    except Exception as e:
        logger.error(f"数据质量检查失败: {e}")
        return APIResponse.error(message="数据质量检查失败", details={"error": str(e)})

@router.post("/quality/detect-anomalies")
async def detect_anomalies_endpoint(request: AnomalyDetectionRequest):
    """检测数据异常"""
    try:
        df = pd.DataFrame(request.data)
        anomaly_type = AnomalyType(request.anomaly_type)
        anomalies = detect_anomalies(df, anomaly_type)

        return APIResponse.success(
            data={
                "anomaly_type": request.anomaly_type,
                "anomaly_count": len(anomalies),
                "anomalies": anomalies
            },
            message="异常检测完成"
        )

    except Exception as e:
        logger.error(f"异常检测失败: {e}")
        return APIResponse.error(message="异常检测失败", details={"error": str(e)})

@router.get("/quality/alerts")
async def get_active_alerts_endpoint():
    """获取活跃告警"""
    try:
        alerts = get_active_alerts()

        return APIResponse.success(
            data=[
                {
                    "alert_id": alert.alert_id,
                    "metric": alert.metric.value,
                    "current_value": alert.current_value,
                    "threshold_value": alert.threshold_value,
                    "alert_level": alert.alert_level.value,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                    "data_source": alert.data_source,
                    "resolution_status": alert.resolution_status
                }
                for alert in alerts
            ],
            message="活跃告警获取成功"
        )

    except Exception as e:
        logger.error(f"获取活跃告警失败: {e}")
        return APIResponse.error(message="获取活跃告警失败", details={"error": str(e)})

@router.post("/quality/resolve-alert/{alert_id}")
async def resolve_alert_endpoint(alert_id: str, resolution_notes: str = ""):
    """解决告警"""
    try:
        resolve_alert(alert_id, resolution_notes)

        return APIResponse.success(message="告警已解决")

    except Exception as e:
        logger.error(f"解决告警失败: {e}")
        return APIResponse.error(message="解决告警失败", details={"error": str(e)})

@router.get("/quality/summary")
async def get_quality_summary_endpoint():
    """获取质量摘要"""
    try:
        summary = get_quality_summary()

        return APIResponse.success(data=summary, message="质量摘要获取成功")

    except Exception as e:
        logger.error(f"获取质量摘要失败: {e}")
        return APIResponse.error(message="获取质量摘要失败", details={"error": str(e)})

# 数据管道API
@router.post("/pipeline/create")
async def create_pipeline_endpoint(request: PipelineCreateRequest):
    """创建数据管道"""
    try:
        # 转换任务配置
        tasks = []
        for task_dict in request.tasks:
            task_config = TaskConfig(**task_dict)
            tasks.append(task_config)

        # 创建管道配置
        config = PipelineConfig(
            pipeline_id=request.pipeline_id,
            name=request.name,
            description=request.description,
            version=request.version,
            tasks=tasks,
            triggers=[TriggerType(t) for t in request.triggers],
            schedule=request.schedule,
            max_parallel_tasks=request.max_parallel_tasks
        )

        # 创建管道
        pipeline = create_pipeline(config)

        return APIResponse.success(
            data={
                "pipeline_id": pipeline.config.pipeline_id,
                "name": pipeline.config.name,
                "version": pipeline.config.version,
                "task_count": len(pipeline.config.tasks)
            },
            message="数据管道创建成功"
        )

    except Exception as e:
        logger.error(f"创建数据管道失败: {e}")
        return APIResponse.error(message="创建数据管道失败", details={"error": str(e)})

@router.post("/pipeline/{pipeline_id}/execute")
async def execute_pipeline_endpoint(pipeline_id: str, trigger_type: str = "manual"):
    """执行数据管道"""
    try:
        trigger_type_enum = TriggerType(trigger_type)
        result = execute_pipeline(pipeline_id, trigger_type_enum)

        return APIResponse.success(
            data={
                "execution_id": result.execution_id,
                "status": result.status.value,
                "start_time": result.start_time.isoformat(),
                "end_time": result.end_time.isoformat() if result.end_time else None,
                "duration": result.duration,
                "total_tasks": result.total_tasks,
                "successful_tasks": result.successful_tasks,
                "failed_tasks": result.failed_tasks,
                "skipped_tasks": result.skipped_tasks
            },
            message="数据管道执行完成"
        )

    except Exception as e:
        logger.error(f"执行数据管道失败: {e}")
        return APIResponse.error(message="执行数据管道失败", details={"error": str(e)})

@router.post("/pipeline/{pipeline_id}/schedule")
async def schedule_pipeline_endpoint(pipeline_id: str, schedule: str):
    """调度数据管道"""
    try:
        schedule_pipeline(pipeline_id, schedule)

        return APIResponse.success(message="数据管道调度成功")

    except Exception as e:
        logger.error(f"调度数据管道失败: {e}")
        return APIResponse.error(message="调度数据管道失败", details={"error": str(e)})

@router.get("/pipeline/{pipeline_id}/status")
async def get_pipeline_status_endpoint(pipeline_id: str):
    """获取管道状态"""
    try:
        status = get_pipeline_status(pipeline_id)

        return APIResponse.success(data=status, message="管道状态获取成功")

    except Exception as e:
        logger.error(f"获取管道状态失败: {e}")
        return APIResponse.error(message="获取管道状态失败", details={"error": str(e)})

@router.get("/pipeline/list")
async def get_all_pipelines_endpoint():
    """获取所有管道"""
    try:
        pipelines = get_all_pipelines()

        return APIResponse.success(data=pipelines, message="管道列表获取成功")

    except Exception as e:
        logger.error(f"获取管道列表失败: {e}")
        return APIResponse.error(message="获取管道列表失败", details={"error": str(e)})

# 综合数据处理API
@router.post("/comprehensive/process")
async def comprehensive_data_processing_endpoint(
    request: Dict[str, Any],
    background_tasks: BackgroundTasks
):
    """综合数据处理"""
    try:
        # 解析请求
        data = request.get("data", {})
        processing_steps = request.get("processing_steps", [])

        df = pd.DataFrame(data)
        results = {}

        # 执行处理步骤
        for step in processing_steps:
            step_type = step.get("type")
            step_config = step.get("config", {})

            if step_type == "preprocessing":
                # 数据预处理
                config = PreprocessingConfig(**step_config)
                processed_df, quality_report = preprocess_data(df, config)
                results["preprocessing"] = {
                    "processed_data": processed_df.to_dict(),
                    "quality_report": {
                        "quality_score": quality_report.quality_score,
                        "recommendations": quality_report.recommendations
                    }
                }
                df = processed_df

            elif step_type == "feature_engineering":
                # 特征工程
                target = step_config.get("target")
                y = pd.Series(target) if target else None

                selection_method = FeatureSelectionMethod(step_config.get("selection_method")) if step_config.get("selection_method") else None
                construction_methods = [FeatureConstructionMethod(m) for m in step_config.get("construction_methods", [])]
                transformation_methods = [FeatureTransformationMethod(m) for m in step_config.get("transformation_methods", [])]
                reduction_method = DimensionalityReductionMethod(step_config.get("reduction_method")) if step_config.get("reduction_method") else None

                engineered_df, report = engineer_features(
                    df, y, selection_method, construction_methods, transformation_methods, reduction_method
                )
                results["feature_engineering"] = {
                    "engineered_data": engineered_df.to_dict(),
                    "report": {
                        "original_features": report.original_features,
                        "final_features": report.final_features
                    }
                }
                df = engineered_df

            elif step_type == "quality_check":
                # 质量检查
                data_source = step_config.get("data_source", "unknown")
                quality_report = check_data_quality(df, data_source)
                results["quality_check"] = {
                    "quality_score": quality_report.quality_score,
                    "alerts": len(quality_report.alerts),
                    "recommendations": quality_report.recommendations
                }

        return APIResponse.success(
            data={
                "final_data": df.to_dict(),
                "processing_results": results,
                "total_steps": len(processing_steps)
            },
            message="综合数据处理完成"
        )

    except Exception as e:
        logger.error(f"综合数据处理失败: {e}")
        return APIResponse.error(message="综合数据处理失败", details={"error": str(e)})

# 数据上传API
@router.post("/upload/data")
async def upload_data_endpoint(file: UploadFile = File(...)):
    """上传数据文件"""
    try:
        # 读取文件内容
        content = await file.read()

        # 根据文件类型解析数据
        if file.filename.endswith('.csv'):
            import io
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        elif file.filename.endswith('.json'):
            data = json.loads(content.decode('utf-8'))
            df = pd.DataFrame(data)
        else:
            return APIResponse.error(message="不支持的文件格式", code=400)

        # 返回数据信息
        return APIResponse.success(
            data={
                "filename": file.filename,
                "file_size": len(content),
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "data_types": df.dtypes.astype(str).to_dict(),
                "sample_data": df.head().to_dict()
            },
            message="数据文件上传成功"
        )

    except Exception as e:
        logger.error(f"数据文件上传失败: {e}")
        return APIResponse.error(message="数据文件上传失败", details={"error": str(e)})

# 数据标注API
@router.post("/annotation/label")
async def label_data_endpoint(request: DataAnnotationRequest):
    """数据标注"""
    try:
        # 转换数据
        df = pd.DataFrame(request.data)

        # 创建配置
        config = LabelingConfig(
            strategy=LabelingStrategy(request.strategy),
            contrastive_method=ContrastiveMethod(request.contrastive_method) if request.contrastive_method else None,
            balance_method=LabelBalanceMethod(request.balance_method) if request.balance_method else None,
            noise_handling=NoiseHandlingMethod(request.noise_handling) if request.noise_handling else None,
            confidence_threshold=request.confidence_threshold,
            max_iterations=request.max_iterations
        )

        # 执行标注
        result = label_data(df, config)

        return APIResponse.success(
            data={
                "labeled_data": result.labeled_data.to_dict(),
                "pseudo_labels": result.pseudo_labels.to_dict() if result.pseudo_labels is not None else None,
                "confidence_scores": result.confidence_scores.tolist() if result.confidence_scores is not None else None,
                "quality_metrics": result.quality_metrics,
                "processing_time": result.processing_time
            },
            message="数据标注完成"
        )

    except Exception as e:
        logger.error(f"数据标注失败: {e}")
        return APIResponse.error(message="数据标注失败", details={"error": str(e)})

@router.post("/annotation/balance-labels")
async def balance_labels_endpoint(request: Dict[str, Any]):
    """标签平衡"""
    try:
        X = np.array(request["X"])
        y = np.array(request["y"])
        method = LabelBalanceMethod(request["method"])

        X_balanced, y_balanced = balance_labels(X, y, method)

        return APIResponse.success(
            data={
                "X_balanced": X_balanced.tolist(),
                "y_balanced": y_balanced.tolist(),
                "original_size": len(X),
                "balanced_size": len(X_balanced)
            },
            message="标签平衡完成"
        )

    except Exception as e:
        logger.error(f"标签平衡失败: {e}")
        return APIResponse.error(message="标签平衡失败", details={"error": str(e)})

@router.post("/annotation/handle-noise")
async def handle_label_noise_endpoint(request: Dict[str, Any]):
    """处理标签噪声"""
    try:
        X = np.array(request["X"])
        y = np.array(request["y"])
        method = NoiseHandlingMethod(request["method"])

        X_clean, y_clean = handle_label_noise(X, y, method)

        return APIResponse.success(
            data={
                "X_clean": X_clean.tolist(),
                "y_clean": y_clean.tolist(),
                "original_size": len(X),
                "cleaned_size": len(X_clean)
            },
            message="标签噪声处理完成"
        )

    except Exception as e:
        logger.error(f"标签噪声处理失败: {e}")
        return APIResponse.error(message="标签噪声处理失败", details={"error": str(e)})

# 自动化管道API
@router.post("/automated-pipeline/create")
async def create_automated_pipeline_endpoint(request: AutomatedPipelineRequest):
    """创建自动化数据管道"""
    try:
        # 转换任务配置
        tasks = []
        for task_dict in request.tasks:
            task = DataPipelineTask(**task_dict)
            tasks.append(task)

        # 创建管道配置
        config = AutoPipelineConfig(
            pipeline_id=request.pipeline_id,
            name=request.name,
            description=request.description,
            engine=PipelineEngine(request.engine),
            data_source=DataSource(request.data_source),
            processing_mode=ProcessingMode(request.processing_mode),
            schedule=request.schedule,
            max_parallel_tasks=request.max_parallel_tasks,
            memory_limit=request.memory_limit,
            cpu_limit=request.cpu_limit
        )

        # 创建管道
        pipeline_id = create_auto_pipeline(config, tasks)

        return APIResponse.success(
            data={
                "pipeline_id": pipeline_id,
                "engine": request.engine,
                "data_source": request.data_source,
                "processing_mode": request.processing_mode,
                "task_count": len(tasks)
            },
            message="自动化数据管道创建成功"
        )

    except Exception as e:
        logger.error(f"创建自动化数据管道失败: {e}")
        return APIResponse.error(message="创建自动化数据管道失败", details={"error": str(e)})

@router.post("/automated-pipeline/preset/dask-preprocess/create")
async def create_dask_preset_pipeline_endpoint(request: DaskPresetCreateRequest):
    """创建Dask预处理预设流水线（加载→预处理→保存）"""
    try:
        pipeline_path = create_dask_preprocessing_pipeline(
            pipeline_id=request.pipeline_id,
            description=request.description or "Dask预处理流水线",
            max_parallel_tasks=request.max_parallel_tasks,
            loader_name=request.loader_name,
            preprocess_name=request.preprocess_name,
            save_name=request.save_name
        )
        return APIResponse.success(data={"pipeline_path": pipeline_path}, message="Dask预设流水线创建成功")
    except Exception as e:
        logger.error(f"创建Dask预设流水线失败: {e}")
        return APIResponse.error(message="创建Dask预设流水线失败", details={"error": str(e)})

@router.post("/automated-pipeline/preset/{pipeline_id}/execute")
async def execute_dask_preset_pipeline_endpoint(pipeline_id: str):
    """执行Dask预设流水线"""
    try:
        execution = execute_auto_pipeline(pipeline_id, PipelineEngine.DASK)
        return APIResponse.success(
            data={
                "execution_id": execution.execution_id,
                "pipeline_id": execution.pipeline_id,
                "status": execution.status,
                "start_time": execution.start_time.isoformat(),
                "total_tasks": execution.total_tasks
            },
            message="Dask预设流水线执行开始"
        )
    except Exception as e:
        logger.error(f"执行Dask预设流水线失败: {e}")
        return APIResponse.error(message="执行Dask预设流水线失败", details={"error": str(e)})

@router.post("/automated-pipeline/preset/dask-contrastive/create")
async def create_dask_contrastive_preset_endpoint(request: DaskContrastivePresetCreateRequest):
    """创建带对比学习节点的Dask预设流水线"""
    try:
        pipeline_path = create_dask_contrastive_pipeline(
            pipeline_id=request.pipeline_id,
            description=request.description or "Dask对比学习预设",
            max_parallel_tasks=request.max_parallel_tasks,
            loader_name=request.loader_name,
            preprocess_name=request.preprocess_name,
            contrastive_name=request.contrastive_name,
            save_name=request.save_name
        )
        return APIResponse.success(data={"pipeline_path": pipeline_path}, message="Dask对比学习预设创建成功")
    except Exception as e:
        logger.error(f"创建Dask对比学习预设失败: {e}")
        return APIResponse.error(message="创建Dask对比学习预设失败", details={"error": str(e)})

@router.post("/automated-pipeline/{pipeline_id}/execute")
async def execute_automated_pipeline_endpoint(pipeline_id: str, engine: str):
    """执行自动化数据管道"""
    try:
        engine_enum = PipelineEngine(engine)
        execution = execute_auto_pipeline(pipeline_id, engine_enum)

        return APIResponse.success(
            data={
                "execution_id": execution.execution_id,
                "pipeline_id": execution.pipeline_id,
                "status": execution.status,
                "start_time": execution.start_time.isoformat(),
                "total_tasks": execution.total_tasks
            },
            message="自动化数据管道执行开始"
        )

    except Exception as e:
        logger.error(f"执行自动化数据管道失败: {e}")
        return APIResponse.error(message="执行自动化数据管道失败", details={"error": str(e)})

@router.get("/automated-pipeline/engines")
async def get_pipeline_engines_endpoint():
    """获取可用的管道引擎"""
    try:
        from backend.app.data_processing.automated_pipeline import academic_pipeline_manager
        engine_status = academic_pipeline_manager.get_engine_status()

        return APIResponse.success(data=engine_status, message="管道引擎状态获取成功")

    except Exception as e:
        logger.error(f"获取管道引擎状态失败: {e}")
        return APIResponse.error(message="获取管道引擎状态失败", details={"error": str(e)})

# 系统状态API
@router.get("/status")
async def get_data_processing_status():
    """获取数据处理系统状态"""
    try:
        status = {
            "timestamp": datetime.now().isoformat(),
            "modules": {
                "data_cleaner": "active",
                "feature_engineer": "active",
                "data_augmenter": "active",
                "self_supervised_learning": "active",
                "multimodal_fusion": "active",
                "data_quality_monitor": "active",
                "data_pipeline_manager": "active",
                "data_annotation": "active",
                "automated_pipeline": "active"
            },
            "statistics": {
                "total_pipelines": len(get_all_pipelines()),
                "active_alerts": len(get_active_alerts()),
                "quality_summary": get_quality_summary()
            },
            "capabilities": {
                "data_cleaning": ["缺失值处理", "异常值检测", "数据标准化", "类型转换"],
                "feature_engineering": ["特征选择", "特征构造", "特征转换", "降维"],
                "data_augmentation": ["图像增强", "文本增强", "时间序列增强", "表格数据增强"],
                "self_supervised_learning": ["对比学习", "掩码预测", "重构任务"],
                "multimodal_fusion": ["早期融合", "晚期融合", "注意力融合"],
                "data_annotation": ["自监督学习", "半监督学习", "标签平衡", "噪声处理"],
                "automated_pipeline": ["Airflow", "TensorFlow", "Dask", "Spark"]
            }
        }

        return APIResponse.success(data=status, message="系统状态获取成功")

    except Exception as e:
        logger.error(f"获取系统状态失败: {e}")
        return APIResponse.error(message="获取系统状态失败", details={"error": str(e)})

__all__ = ["'logger'", "'router'", "'PreprocessingRequest'", "'FeatureEngineeringRequest'", "'DataAugmentationRequest'", "'SelfSupervisedRequest'", "'MultimodalFusionRequest'", "'QualityCheckRequest'", "'AnomalyDetectionRequest'", "'PipelineCreateRequest'", "'DataAnnotationRequest'", "'AutomatedPipelineRequest'", "'DaskPresetCreateRequest'", "'DaskContrastivePresetCreateRequest'"]
