"""
监控端点
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

"""
系统监控和健康检查API
企业级监控端点
"""

from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import logging
import time
import psutil
from datetime import datetime, timedelta
import asyncio

from backend.app.core.database import check_db_health, get_connection_pool_status, db_performance_monitor
from backend.app.core.observability import export_runtime_metrics, evaluate_alerts
from backend.app.core.cache import cache_manager
from backend.app.core.exceptions import CustomException
from backend.app.database.migration_manager import migration_manager

logger = logging.getLogger(__name__)
router = APIRouter()

class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str
    timestamp: str
    version: str
    uptime: float
    services: Dict[str, Any]

class MetricsResponse(BaseModel):
    """指标响应模型"""
    timestamp: str
    system: Dict[str, Any]
    database: Dict[str, Any]
    cache: Dict[str, Any]
    application: Dict[str, Any]

class SystemInfo(BaseModel):
    """系统信息模型"""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, int]
    processes: int

# 全局变量用于跟踪系统启动时间
_start_time = time.time()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查端点"""
    try:
        current_time = datetime.now()
        uptime = time.time() - _start_time

        # 检查各个服务状态
        services = {}

        # 数据库健康检查
        db_healthy, db_status = await check_db_health()
        services["database"] = {
            "status": "healthy" if db_healthy else "unhealthy",
            "details": db_status
        }

        # 缓存健康检查
        cache_status = await cache_manager.health_check()
        services["cache"] = {
            "status": "healthy" if cache_status["redis_connected"] or cache_status["memory_cache_size"] > 0 else "unhealthy",
            "details": cache_status
        }

        # 系统资源检查
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        services["system"] = {
            "status": "healthy" if cpu_usage < 90 and memory.percent < 90 and disk.percent < 90 else "warning",
            "details": {
                "cpu_usage": cpu_usage,
                "memory_usage": memory.percent,
                "disk_usage": disk.percent
            }
        }

        # 确定整体状态
        overall_status = "healthy"
        for service_name, service_info in services.items():
            if service_info["status"] == "unhealthy":
                overall_status = "unhealthy"
                break
            elif service_info["status"] == "warning":
                overall_status = "warning"

        return HealthResponse(
            status=overall_status,
            timestamp=current_time.isoformat(),
            version="3.0.0",
            uptime=uptime,
            services=services
        )

    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        raise HTTPException(status_code=500, detail="健康检查失败")

@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """获取系统指标"""
    try:
        current_time = datetime.now()

        # 系统指标
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network_io = psutil.net_io_counters()

        system_metrics = {
            "cpu": {
                "usage_percent": cpu_usage,
                "count": psutil.cpu_count(),
                "load_avg": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            },
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "usage_percent": memory.percent,
                "cached": getattr(memory, 'cached', 0),
                "buffers": getattr(memory, 'buffers', 0)
            },
            "disk": {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "usage_percent": (disk.used / disk.total) * 100
            },
            "network": {
                "bytes_sent": network_io.bytes_sent,
                "bytes_recv": network_io.bytes_recv,
                "packets_sent": network_io.packets_sent,
                "packets_recv": network_io.packets_recv
            },
            "processes": {
                "count": len(psutil.pids()),
                "running": len([p for p in psutil.process_iter(['status']) if p.info['status'] == 'running'])
            }
        }

        # 数据库指标
        db_pool_status = get_connection_pool_status()
        db_performance_stats = db_performance_monitor.get_performance_stats()

        database_metrics = {
            "connection_pool": db_pool_status,
            "performance": db_performance_stats,
            "migration_status": migration_manager.get_database_status()
        }

        # 缓存指标
        cache_stats = cache_manager.get_stats()
        cache_metrics = {
            "stats": cache_stats,
            "health": await cache_manager.health_check()
        }

        # 应用指标
        application_metrics = {
            "uptime": time.time() - _start_time,
            "version": "3.0.0",
            "python_version": f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}.{psutil.sys.version_info.micro}",
            "platform": psutil.sys.platform
        }

        return MetricsResponse(
            timestamp=current_time.isoformat(),
            system=system_metrics,
            database=database_metrics,
            cache=cache_metrics,
            application=application_metrics
        )

    except Exception as e:
        logger.error(f"获取指标失败: {e}")
        raise HTTPException(status_code=500, detail="获取指标失败")

@router.get("/status")
async def get_status():
    """获取系统状态摘要"""
    try:
        # 基本状态信息
        status = {
            "timestamp": datetime.now().isoformat(),
            "uptime": time.time() - _start_time,
            "version": "3.0.0"
        }

        # 快速健康检查
        db_healthy, _ = await check_db_health()
        cache_status = await cache_manager.health_check()

        status["services"] = {
            "database": "healthy" if db_healthy else "unhealthy",
            "cache": "healthy" if cache_status["redis_connected"] or cache_status["memory_cache_size"] > 0 else "unhealthy"
        }

        # 系统资源快速检查
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory_usage = psutil.virtual_memory().percent

        status["resources"] = {
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "status": "healthy" if cpu_usage < 80 and memory_usage < 80 else "warning"
        }

        return status

    except Exception as e:
        logger.error(f"获取状态失败: {e}")
        raise HTTPException(status_code=500, detail="获取状态失败")

@router.get("/database/status")
async def get_database_status():
    """获取数据库状态"""
    try:
        # 连接池状态
        pool_status = get_connection_pool_status()

        # 性能统计
        performance_stats = db_performance_monitor.get_performance_stats()

        # 迁移状态
        migration_status = migration_manager.get_database_status()

        # 健康检查
        healthy, health_details = await check_db_health()

        return {
            "timestamp": datetime.now().isoformat(),
            "healthy": healthy,
            "health_details": health_details,
            "connection_pool": pool_status,
            "performance": performance_stats,
            "migration": migration_status
        }

    except Exception as e:
        logger.error(f"获取数据库状态失败: {e}")
        raise HTTPException(status_code=500, detail="获取数据库状态失败")

@router.get("/cache/status")
async def get_cache_status():
    """获取缓存状态"""
    try:
        stats = cache_manager.get_stats()
        health = await cache_manager.health_check()

        return {
            "timestamp": datetime.now().isoformat(),
            "stats": stats,
            "health": health
        }

    except Exception as e:
        logger.error(f"获取缓存状态失败: {e}")
        raise HTTPException(status_code=500, detail="获取缓存状态失败")

@router.post("/cache/clear")
async def clear_cache():
    """清理缓存"""
    try:
        # 清理所有缓存
        await cache_manager.clear_pattern("*")

        return {
            "success": True,
            "message": "缓存清理完成",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"清理缓存失败: {e}")
        raise HTTPException(status_code=500, detail="清理缓存失败")

@router.get("/performance")
async def get_performance_metrics():
    """获取性能指标"""
    try:
        # 数据库性能
        db_performance = db_performance_monitor.get_performance_stats()

        # 系统性能
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()

        # 缓存性能
        cache_stats = cache_manager.get_stats()

        return {
            "timestamp": datetime.now().isoformat(),
            "database": db_performance,
            "system": {
                "cpu_usage": cpu_usage,
                "memory_usage": memory.percent,
                "memory_available": memory.available
            },
            "cache": cache_stats
        }

    except Exception as e:
        logger.error(f"获取性能指标失败: {e}")
        raise HTTPException(status_code=500, detail="获取性能指标失败")

@router.get("/runtime")
async def get_runtime_metrics():
    """导出运行时聚合指标（HTTP摘要、DB连接池）"""
    try:
        return export_runtime_metrics()
    except Exception as e:
        logger.error(f"获取运行时指标失败: {e}")
        raise HTTPException(status_code=500, detail="获取运行时指标失败")

@router.get("/alerts")
async def get_alerts():
    """基于简单阈值的告警（可对接 Alertmanager）"""
    try:
        return evaluate_alerts()
    except Exception as e:
        logger.error(f"获取告警失败: {e}")
        raise HTTPException(status_code=500, detail="获取告警失败")

@router.get("/stream")
async def stream_runtime(interval: float = 2.0):
    """Server-Sent Events: 周期推送运行时指标与告警摘要"""
    async def event_gen():
        while True:
            try:
                data = evaluate_alerts()
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                await asyncio.sleep(max(0.5, interval))
            except Exception as e:
                yield f"event: error\n" f"data: {str(e)}\n\n"
                await asyncio.sleep(2.0)
    return StreamingResponse(event_gen(), media_type="text/event-stream")

@router.get("/logs")
async def get_recent_logs(limit: int = 100):
    """获取最近的日志"""
    try:
        # 这里可以实现日志查询功能
        # 目前返回模拟数据
        logs = [
            {
                "timestamp": datetime.now().isoformat(),
                "level": "INFO",
                "message": "系统运行正常",
                "module": "monitor"
            }
        ]

        return {
            "timestamp": datetime.now().isoformat(),
            "logs": logs[:limit],
            "total": len(logs)
        }

    except Exception as e:
        logger.error(f"获取日志失败: {e}")
        raise HTTPException(status_code=500, detail="获取日志失败")

# 系统信息端点
@router.get("/system/info")
async def get_system_info():
    """获取系统信息"""
    try:
        return {
            "timestamp": datetime.now().isoformat(),
            "system": {
                "platform": psutil.sys.platform,
                "python_version": f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}.{psutil.sys.version_info.micro}",
                "cpu_count": psutil.cpu_count(),
                "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat()
            },
            "application": {
                "version": "3.0.0",
                "uptime": time.time() - _start_time,
                "start_time": datetime.fromtimestamp(_start_time).isoformat()
            }
        }

    except Exception as e:
        logger.error(f"获取系统信息失败: {e}")
        raise HTTPException(status_code=500, detail="获取系统信息失败")

__all__ = ["'logger'", "'router'", "'HealthResponse'", "'MetricsResponse'", "'SystemInfo'"]
