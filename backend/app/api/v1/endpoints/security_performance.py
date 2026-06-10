

"""
数据安全与性能优化API端点
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel
import pandas as pd
import numpy as np

from backend.app.core.exceptions import CustomException
from backend.app.core.data_security import (
    academic_data_security_manager, DataSecurityConfig, DataClassification,
    DataRetentionPolicy, AccessLevel, AnonymizationMethod, secure_data,
    check_data_access, record_data_access, cleanup_expired_data
)
from backend.app.core.performance_optimization import (
    academic_performance_optimizer, OptimizationConfig, CacheStrategy,
    OptimizationLevel, optimize_system, get_performance_report, get_cache_stats
)

logger = logging.getLogger(__name__)
router = APIRouter()

# 请求模型
class DataSecurityRequest(BaseModel):
    data: Dict[str, List[Any]]
    classification: str = "confidential"
    retention_policy: str = "medium_term"
    encryption_enabled: bool = True
    anonymization_enabled: bool = True
    access_control_enabled: bool = True
    audit_logging_enabled: bool = True

class DataAccessRequest(BaseModel):
    user_id: str
    data_id: str
    access_type: str = "read"
    ip_address: str
    user_agent: str

class AnonymizationRequest(BaseModel):
    data: Dict[str, List[Any]]
    method: str = "pseudonymization"
    columns: List[str]
    config: Optional[Dict[str, Any]] = None

class PerformanceOptimizationRequest(BaseModel):
    cache_strategy: str = "intelligent"
    optimization_level: str = "advanced"
    max_cache_size: int = 1000
    max_memory_usage: float = 0.8
    auto_optimization: bool = True

class CacheRequest(BaseModel):
    key: str
    value: Any
    ttl: Optional[int] = None

# 数据安全API
@router.post("/security/secure-data")
async def secure_data_endpoint(request: DataSecurityRequest):
    """保护数据"""
    try:
        # 转换数据
        df = pd.DataFrame(request.data)

        # 创建配置
        config = DataSecurityConfig(
            classification=DataClassification(request.classification),
            retention_policy=DataRetentionPolicy(request.retention_policy),
            encryption_enabled=request.encryption_enabled,
            anonymization_enabled=request.anonymization_enabled,
            access_control_enabled=request.access_control_enabled,
            audit_logging_enabled=request.audit_logging_enabled
        )

        # 保护数据
        result = secure_data(df, config)

        return APIResponse.success(
            data={
                "original_shape": df.shape,
                "encrypted_data": result["encrypted_data"],
                "anonymized_shape": result["anonymized_data"].shape if result["anonymized_data"] is not None else None,
                "access_controls": result["access_controls"],
                "retention_info": result["retention_info"]
            },
            message="数据保护完成"
        )

    except Exception as e:
        logger.error(f"数据保护失败: {e}")
        return APIResponse.error(message="数据保护失败", details={"error": str(e)})

@router.post("/security/check-access")
async def check_access_endpoint(request: DataAccessRequest):
    """检查数据访问权限"""
    try:
        access_level = AccessLevel(request.access_type)
        has_access = check_data_access(request.user_id, request.data_id, access_level)

        # 记录访问尝试
        record_data_access(
            request.user_id, request.data_id, access_level,
            request.ip_address, request.user_agent, has_access
        )

        return APIResponse.success(
            data={
                "has_access": has_access,
                "user_id": request.user_id,
                "data_id": request.data_id,
                "access_type": request.access_type
            },
            message="访问权限检查完成"
        )

    except Exception as e:
        logger.error(f"访问权限检查失败: {e}")
        return APIResponse.error(message="访问权限检查失败", details={"error": str(e)})

@router.post("/security/anonymize")
async def anonymize_data_endpoint(request: AnonymizationRequest):
    """匿名化数据"""
    try:
        # 转换数据
        df = pd.DataFrame(request.data)

        # 获取匿名化器
        anonymizer = academic_data_security_manager.anonymizer

        # 执行匿名化
        anonymized_data = await anonymizer.anonymize(
            df,
            AnonymizationMethod(request.method),
            request.columns,
            request.config or {}
        )

        return APIResponse.success(
            data={
                "original_shape": df.shape,
                "anonymized_shape": anonymized_data.shape,
                "method": request.method,
                "columns": request.columns
            },
            message="数据匿名化完成"
        )

    except Exception as e:
        logger.error(f"数据匿名化失败: {e}")
        return APIResponse.error(message="数据匿名化失败", details={"error": str(e)})

@router.get("/security/statistics")
async def get_security_statistics():
    """获取安全统计"""
    try:
        stats = academic_data_security_manager.get_security_statistics()

        return APIResponse.success(data=stats, message="安全统计获取成功")

    except Exception as e:
        logger.error(f"获取安全统计失败: {e}")
        return APIResponse.error(message="获取安全统计失败", details={"error": str(e)})

@router.post("/security/cleanup-expired")
async def cleanup_expired_data_endpoint():
    """清理过期数据"""
    try:
        expired_data_ids = cleanup_expired_data()

        return APIResponse.success(
            data={
                "expired_count": len(expired_data_ids),
                "expired_ids": expired_data_ids
            },
            message="过期数据清理完成"
        )

    except Exception as e:
        logger.error(f"过期数据清理失败: {e}")
        return APIResponse.error(message="过期数据清理失败", details={"error": str(e)})

# 性能优化API
@router.post("/performance/optimize")
async def optimize_system_endpoint(request: PerformanceOptimizationRequest):
    """优化系统性能"""
    try:
        # 更新优化配置
        config = OptimizationConfig(
            cache_strategy=CacheStrategy(request.cache_strategy),
            optimization_level=OptimizationLevel(request.optimization_level),
            max_cache_size=request.max_cache_size,
            max_memory_usage=request.max_memory_usage,
            auto_optimization=request.auto_optimization
        )

        # 执行优化
        result = optimize_system()

        return APIResponse.success(
            data=result,
            message="系统性能优化完成"
        )

    except Exception as e:
        logger.error(f"系统性能优化失败: {e}")
        return APIResponse.error(message="系统性能优化失败", details={"error": str(e)})

@router.get("/performance/report")
async def get_performance_report_endpoint():
    """获取性能报告"""
    try:
        report = get_performance_report()

        return APIResponse.success(data=report, message="性能报告获取成功")

    except Exception as e:
        logger.error(f"获取性能报告失败: {e}")
        return APIResponse.error(message="获取性能报告失败", details={"error": str(e)})

@router.get("/performance/cache/stats")
async def get_cache_stats_endpoint():
    """获取缓存统计"""
    try:
        stats = get_cache_stats()

        return APIResponse.success(data=stats, message="缓存统计获取成功")

    except Exception as e:
        logger.error(f"获取缓存统计失败: {e}")
        return APIResponse.error(message="获取缓存统计失败", details={"error": str(e)})

@router.post("/performance/cache/set")
async def set_cache_endpoint(request: CacheRequest):
    """设置缓存"""
    try:
        cache = academic_performance_optimizer.cache
        success = await cache.set(request.key, request.value, request.ttl)

        return APIResponse.success(
            data={
                "key": request.key,
                "success": success,
                "ttl": request.ttl
            },
            message="缓存设置完成"
        )

    except Exception as e:
        logger.error(f"缓存设置失败: {e}")
        return APIResponse.error(message="缓存设置失败", details={"error": str(e)})

@router.get("/performance/cache/get/{key}")
async def get_cache_endpoint(key: str):
    """获取缓存"""
    try:
        cache = academic_performance_optimizer.cache
        value = await cache.get(key)

        return APIResponse.success(
            data={
                "key": key,
                "value": value,
                "found": value is not None
            },
            message="缓存获取完成"
        )

    except Exception as e:
        logger.error(f"缓存获取失败: {e}")
        return APIResponse.error(message="缓存获取失败", details={"error": str(e)})

@router.delete("/performance/cache/delete/{key}")
async def delete_cache_endpoint(key: str):
    """删除缓存"""
    try:
        cache = academic_performance_optimizer.cache
        success = await cache.delete(key)

        return APIResponse.success(
            data={
                "key": key,
                "success": success
            },
            message="缓存删除完成"
        )

    except Exception as e:
        logger.error(f"缓存删除失败: {e}")
        return APIResponse.error(message="缓存删除失败", details={"error": str(e)})

@router.delete("/performance/cache/clear")
async def clear_cache_endpoint():
    """清空缓存"""
    try:
        cache = academic_performance_optimizer.cache
        success = await cache.clear()

        return APIResponse.success(
            data={"success": success},
            message="缓存清空完成"
        )

    except Exception as e:
        logger.error(f"缓存清空失败: {e}")
        return APIResponse.error(message="缓存清空失败", details={"error": str(e)})

# 系统状态API
@router.get("/status")
async def get_system_status():
    """获取系统状态"""
    try:
        # 获取安全统计
        security_stats = academic_data_security_manager.get_security_statistics()

        # 获取性能报告
        performance_report = get_performance_report()

        # 获取缓存统计
        cache_stats = get_cache_stats()

        status = {
            "timestamp": datetime.now().isoformat(),
            "security": {
                "total_data_records": security_stats["total_data_records"],
                "total_access_records": security_stats["total_access_records"],
                "expired_data_count": security_stats["expired_data_count"],
                "access_attempts_today": security_stats["access_attempts_today"]
            },
            "performance": {
                "current_cpu": performance_report["current_metrics"]["cpu_usage"],
                "current_memory": performance_report["current_metrics"]["memory_usage"],
                "current_disk": performance_report["current_metrics"]["disk_usage"],
                "anomalies_count": len(performance_report["anomalies"])
            },
            "cache": {
                "cache_size": cache_stats["cache_size"],
                "hit_rate": cache_stats["hit_rate"],
                "total_size_bytes": cache_stats["total_size_bytes"]
            },
            "modules": {
                "data_security": "active",
                "performance_optimization": "active",
                "intelligent_cache": "active",
                "system_monitor": "active"
            }
        }

        return APIResponse.success(data=status, message="系统状态获取成功")

    except Exception as e:
        logger.error(f"获取系统状态失败: {e}")
        return APIResponse.error(message="获取系统状态失败", details={"error": str(e)})

# 健康检查API
@router.get("/health")
async def health_check():
    """健康检查"""
    try:
        # 检查各个组件状态
        cache_stats = get_cache_stats()
        security_stats = academic_data_security_manager.get_security_statistics()

        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "data_security": "healthy",
                "performance_optimization": "healthy",
                "intelligent_cache": "healthy",
                "system_monitor": "healthy"
            },
            "metrics": {
                "cache_hit_rate": cache_stats["hit_rate"],
                "data_records": security_stats["total_data_records"],
                "access_records": security_stats["total_access_records"]
            }
        }

        return APIResponse.success(data=health_status, message="健康检查完成")

    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return APIResponse.error(message="健康检查失败", details={"error": str(e)})

__all__ = ["'logger'", "'router'", "'DataSecurityRequest'", "'DataAccessRequest'", "'AnonymizationRequest'", "'PerformanceOptimizationRequest'", "'CacheRequest'"]
