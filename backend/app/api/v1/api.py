"""
API路由模块
企业级API管理
"""

from fastapi import APIRouter
from .endpoints import (
    health,
    glucose,
    image,
    cultural,
    merl,
    explain,
    feedback,
    user,
    workflow,
    data,
    stats,
    monitor,
    database_management,
    data_processing,
    security_performance,
    modeling,
    scheduling,
    experiments,
    trained_model,
    model_lifecycle
)

api_router = APIRouter()

# 健康检查
api_router.include_router(health.router, prefix="/health", tags=["健康检查"])

# 血糖预测
api_router.include_router(glucose.router, prefix="/glucose", tags=["血糖预测"])

# 图像识别
api_router.include_router(image.router, prefix="/image", tags=["图像识别"])

# 文化适配
api_router.include_router(cultural.router, prefix="/cultural", tags=["文化适配"])

# 混合专家强化学习
api_router.include_router(merl.router, prefix="/merl", tags=["MERL系统"])

# 可解释性
api_router.include_router(explain.router, prefix="/explain", tags=["可解释性"])

# 反馈系统
api_router.include_router(feedback.router, prefix="/feedback", tags=["反馈系统"])

# 用户管理
api_router.include_router(user.router, prefix="/user", tags=["用户管理"])

# 工作流
api_router.include_router(workflow.router, prefix="/workflow", tags=["工作流"])

# 数据处理
api_router.include_router(data.router, prefix="/data", tags=["数据处理"])

# 统计信息
api_router.include_router(stats.router, prefix="/stats", tags=["统计信息"])

# 系统监控
api_router.include_router(monitor.router, prefix="/monitor", tags=["系统监控"])

# 数据库管理
api_router.include_router(database_management.router, prefix="/database", tags=["数据库管理"])

# 数据处理
api_router.include_router(data_processing.router, prefix="/data-processing", tags=["数据处理"])

# 数据安全与性能优化
api_router.include_router(security_performance.router, prefix="/security-performance", tags=["数据安全与性能优化"])

# 模型训练/预测/解释
api_router.include_router(modeling.router, prefix="/modeling", tags=["模型"])

# 资源调度
api_router.include_router(scheduling.router, prefix="/scheduling", tags=["资源调度"])

# 实验设计与验证
api_router.include_router(experiments.router, prefix="/experiments", tags=["实验"])

# 训练模型服务
api_router.include_router(trained_model.router, prefix="/trained-model", tags=["训练模型"])

# 模型生命周期管理
api_router.include_router(model_lifecycle.router, prefix="/model-lifecycle", tags=["模型生命周期"])

__all__ = ["'api_router'"]
