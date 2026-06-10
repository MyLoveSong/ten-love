"""
数据库管理端点
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

"""
高级数据库管理API
整合分片、备份、性能优化、分区、ACID事务、异步同步等功能
"""

from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging
import asyncio
from datetime import datetime, timedelta

from backend.app.core.database import check_db_health, get_connection_pool_status, db_performance_monitor
from backend.app.core.cache import cache_manager
from backend.app.core.exceptions import CustomException

# 导入所有数据库管理模块
from backend.app.database.sharding_manager import (
    sharding_manager, get_sharding_status, add_sharding_rule, rebalance_shards
)
from backend.app.database.backup_manager import (
    backup_manager, create_backup, restore_backup, get_backup_list,
    get_backup_info, delete_backup, get_backup_stats, cleanup_old_backups
)
from backend.app.database.performance_optimizer import (
    performance_optimizer, analyze_database_performance, auto_optimize_database,
    get_performance_summary, get_optimization_history
)
from backend.app.database.partition_manager import (
    partition_manager, get_partition_info, create_new_partition,
    drop_old_partitions, get_partition_stats
)
from backend.app.database.acid_manager import (
    acid_manager, get_transaction_stats, get_lock_status, register_saga, execute_saga
)
from backend.app.database.async_sync_manager import (
    delayed_write_manager, sync_manager, get_delayed_write_status,
    get_sync_stats, enable_delayed_write, disable_delayed_write
)
from app.core.task_queue import enqueue_task, get_task_status

logger = logging.getLogger(__name__)
router = APIRouter()

# 请求模型
class ShardingRuleRequest(BaseModel):
    table_name: str
    shard_key: str
    strategy: str
    shard_count: int
    shard_configs: List[Dict[str, Any]]

class BackupRequest(BaseModel):
    backup_type: str = "full"
    metadata: Optional[Dict[str, Any]] = None

class PartitionRequest(BaseModel):
    table_name: str
    partition_type: str
    partition_strategy: str
    partition_column: str
    partition_count: Optional[int] = None
    retention_days: Optional[int] = None

class SagaRequest(BaseModel):
    saga_id: str
    steps: List[Dict[str, Any]]
    context: Dict[str, Any]

class WriteOperationRequest(BaseModel):
    data_type: str
    table_name: str
    operation_type: str
    data: Dict[str, Any]
    priority: int = 5
    immediate: bool = False

# 数据库概览
@router.get("/overview")
async def get_database_overview():
    """获取数据库概览"""
    try:
        # 健康检查
        db_healthy, db_status = await check_db_health()

        # 连接池状态
        pool_status = get_connection_pool_status()

        # 性能统计
        performance_stats = db_performance_monitor.get_performance_stats()

        # 缓存状态
        cache_status = await cache_manager.health_check()

        # 分片状态
        sharding_status = get_sharding_status()

        # 备份统计
        backup_stats = get_backup_stats()

        # 分区统计
        partition_stats = get_partition_stats()

        # 事务统计
        transaction_stats = get_transaction_stats()

        # 延迟写状态
        delayed_write_status = get_delayed_write_status()

        # 同步统计
        sync_stats = get_sync_stats()

        overview = {
            "timestamp": datetime.now().isoformat(),
            "database": {
                "healthy": db_healthy,
                "status": db_status,
                "connection_pool": pool_status,
                "performance": performance_stats
            },
            "cache": cache_status,
            "sharding": sharding_status,
            "backup": backup_stats,
            "partitioning": partition_stats,
            "transactions": transaction_stats,
            "delayed_write": delayed_write_status,
            "sync": sync_stats
        }

        return APIResponse.success(data=overview, message="数据库概览获取成功")

    except Exception as e:
        logger.error(f"获取数据库概览失败: {e}")
        return APIResponse.error(message="获取数据库概览失败", details={"error": str(e)})

# 分片管理
@router.post("/sharding/rules")
async def add_sharding_rule_endpoint(request: ShardingRuleRequest):
    """添加分片规则"""
    try:
        from backend.app.database.sharding_manager import ShardingRule, ShardingStrategy, ShardConfig

        # 创建分片配置
        shard_configs = [
            ShardConfig(
                shard_id=config["shard_id"],
                database_url=config["database_url"],
                weight=config.get("weight", 1),
                is_active=config.get("is_active", True)
            )
            for config in request.shard_configs
        ]

        # 创建分片规则
        rule = ShardingRule(
            table_name=request.table_name,
            shard_key=request.shard_key,
            strategy=ShardingStrategy(request.strategy),
            shard_count=request.shard_count,
            shard_configs=shard_configs
        )

        add_sharding_rule(request.table_name, rule)

        return APIResponse.success(message="分片规则添加成功")

    except Exception as e:
        logger.error(f"添加分片规则失败: {e}")
        return APIResponse.error(message="添加分片规则失败", details={"error": str(e)})

@router.get("/sharding/status")
async def get_sharding_status_endpoint():
    """获取分片状态"""
    try:
        status = get_sharding_status()
        return APIResponse.success(data=status, message="分片状态获取成功")
    except Exception as e:
        logger.error(f"获取分片状态失败: {e}")
        return APIResponse.error(message="获取分片状态失败", details={"error": str(e)})

@router.post("/sharding/rebalance/{table_name}")
async def rebalance_shards_endpoint(table_name: str, new_configs: List[Dict[str, Any]]):
    """重新平衡分片"""
    try:
        from backend.app.database.sharding_manager import ShardConfig

        configs = [
            ShardConfig(
                shard_id=config["shard_id"],
                database_url=config["database_url"],
                weight=config.get("weight", 1),
                is_active=config.get("is_active", True)
            )
            for config in new_configs
        ]

        rebalance_shards(table_name, configs)

        return APIResponse.success(message="分片重新平衡成功")

    except Exception as e:
        logger.error(f"分片重新平衡失败: {e}")
        return APIResponse.error(message="分片重新平衡失败", details={"error": str(e)})

# 备份管理
@router.post("/backup/create")
async def create_backup_endpoint(request: BackupRequest, background_tasks: BackgroundTasks):
    """创建备份"""
    try:
        from backend.app.database.backup_manager import BackupType

        backup_type = BackupType(request.backup_type)
        backup_id = create_backup(backup_type, request.metadata)

        return APIResponse.success(
            data={"backup_id": backup_id},
            message="备份任务已创建"
        )

    except Exception as e:
        logger.error(f"创建备份失败: {e}")
        return APIResponse.error(message="创建备份失败", details={"error": str(e)})

@router.get("/backup/list")
async def get_backup_list_endpoint():
    """获取备份列表"""
    try:
        backups = get_backup_list()
        return APIResponse.success(data=backups, message="备份列表获取成功")
    except Exception as e:
        logger.error(f"获取备份列表失败: {e}")
        return APIResponse.error(message="获取备份列表失败", details={"error": str(e)})

@router.get("/backup/{backup_id}")
async def get_backup_info_endpoint(backup_id: str):
    """获取备份信息"""
    try:
        backup_info = get_backup_info(backup_id)
        if not backup_info:
            return APIResponse.error(message="备份不存在", code=404)

        return APIResponse.success(data=backup_info, message="备份信息获取成功")

    except Exception as e:
        logger.error(f"获取备份信息失败: {e}")
        return APIResponse.error(message="获取备份信息失败", details={"error": str(e)})

@router.post("/backup/{backup_id}/restore")
async def restore_backup_endpoint(backup_id: str, target_database_url: Optional[str] = None):
    """恢复备份"""
    try:
        success = await restore_backup(backup_id, target_database_url)
        if success:
            return APIResponse.success(message="备份恢复成功")
        else:
            return APIResponse.error(message="备份恢复失败")

    except Exception as e:
        logger.error(f"恢复备份失败: {e}")
        return APIResponse.error(message="恢复备份失败", details={"error": str(e)})

@router.delete("/backup/{backup_id}")
async def delete_backup_endpoint(backup_id: str):
    """删除备份"""
    try:
        success = delete_backup(backup_id)
        if success:
            return APIResponse.success(message="备份删除成功")
        else:
            return APIResponse.error(message="备份不存在", code=404)

    except Exception as e:
        logger.error(f"删除备份失败: {e}")
        return APIResponse.error(message="删除备份失败", details={"error": str(e)})

@router.get("/backup/stats")
async def get_backup_stats_endpoint():
    """获取备份统计"""
    try:
        stats = get_backup_stats()
        return APIResponse.success(data=stats, message="备份统计获取成功")
    except Exception as e:
        logger.error(f"获取备份统计失败: {e}")
        return APIResponse.error(message="获取备份统计失败", details={"error": str(e)})

# 性能优化
@router.get("/performance/analysis")
async def analyze_performance_endpoint():
    """分析数据库性能"""
    try:
        analysis = await analyze_database_performance()
        return APIResponse.success(data=analysis, message="性能分析完成")
    except Exception as e:
        logger.error(f"性能分析失败: {e}")
        return APIResponse.error(message="性能分析失败", details={"error": str(e)})

@router.post("/performance/optimize")
async def auto_optimize_endpoint():
    """自动优化数据库"""
    try:
        optimization_result = await auto_optimize_database()
        return APIResponse.success(data=optimization_result, message="自动优化完成")
    except Exception as e:
        logger.error(f"自动优化失败: {e}")
        return APIResponse.error(message="自动优化失败", details={"error": str(e)})

@router.get("/performance/summary")
async def get_performance_summary_endpoint():
    """获取性能摘要"""
    try:
        summary = get_performance_summary()
        return APIResponse.success(data=summary, message="性能摘要获取成功")
    except Exception as e:
        logger.error(f"获取性能摘要失败: {e}")
        return APIResponse.error(message="获取性能摘要失败", details={"error": str(e)})

@router.get("/performance/history")
async def get_optimization_history_endpoint():
    """获取优化历史"""
    try:
        history = get_optimization_history()
        return APIResponse.success(data=history, message="优化历史获取成功")
    except Exception as e:
        logger.error(f"获取优化历史失败: {e}")
        return APIResponse.error(message="获取优化历史失败", details={"error": str(e)})

# 分区管理
@router.post("/partition/create")
async def create_partition_endpoint(request: PartitionRequest):
    """创建分区"""
    try:
        from backend.app.database.partition_manager import PartitionConfig, PartitionType, PartitionStrategy

        config = PartitionConfig(
            table_name=request.table_name,
            partition_type=PartitionType(request.partition_type),
            partition_strategy=PartitionStrategy(request.partition_strategy),
            partition_column=request.partition_column,
            partition_count=request.partition_count,
            retention_days=request.retention_days
        )

        success = await partition_manager.create_partitioned_table(request.table_name, config)

        if success:
            return APIResponse.success(message="分区创建成功")
        else:
            return APIResponse.error(message="分区创建失败")

    except Exception as e:
        logger.error(f"创建分区失败: {e}")
        return APIResponse.error(message="创建分区失败", details={"error": str(e)})

@router.get("/partition/{table_name}")
async def get_partition_info_endpoint(table_name: str):
    """获取分区信息"""
    try:
        partition_info = await get_partition_info(table_name)
        return APIResponse.success(data=partition_info, message="分区信息获取成功")
    except Exception as e:
        logger.error(f"获取分区信息失败: {e}")
        return APIResponse.error(message="获取分区信息失败", details={"error": str(e)})

@router.post("/partition/{table_name}/create")
async def create_new_partition_endpoint(table_name: str, partition_value: str):
    """创建新分区"""
    try:
        success = await create_new_partition(table_name, partition_value)
        if success:
            return APIResponse.success(message="新分区创建成功")
        else:
            return APIResponse.error(message="新分区创建失败")

    except Exception as e:
        logger.error(f"创建新分区失败: {e}")
        return APIResponse.error(message="创建新分区失败", details={"error": str(e)})

@router.post("/partition/{table_name}/cleanup")
async def cleanup_old_partitions_endpoint(table_name: str):
    """清理旧分区"""
    try:
        dropped_count = await drop_old_partitions(table_name)
        return APIResponse.success(
            data={"dropped_count": dropped_count},
            message=f"清理了 {dropped_count} 个旧分区"
        )
    except Exception as e:
        logger.error(f"清理旧分区失败: {e}")
        return APIResponse.error(message="清理旧分区失败", details={"error": str(e)})

@router.get("/partition/stats")
async def get_partition_stats_endpoint():
    """获取分区统计"""
    try:
        stats = get_partition_stats()
        return APIResponse.success(data=stats, message="分区统计获取成功")
    except Exception as e:
        logger.error(f"获取分区统计失败: {e}")
        return APIResponse.error(message="获取分区统计失败", details={"error": str(e)})

# ACID事务管理
@router.get("/transactions/stats")
async def get_transaction_stats_endpoint():
    """获取事务统计"""
    try:
        stats = get_transaction_stats()
        return APIResponse.success(data=stats, message="事务统计获取成功")
    except Exception as e:
        logger.error(f"获取事务统计失败: {e}")
        return APIResponse.error(message="获取事务统计失败", details={"error": str(e)})

@router.get("/transactions/locks/{resource_id}")
async def get_lock_status_endpoint(resource_id: str):
    """获取锁状态"""
    try:
        lock_status = get_lock_status(resource_id)
        return APIResponse.success(data=lock_status, message="锁状态获取成功")
    except Exception as e:
        logger.error(f"获取锁状态失败: {e}")
        return APIResponse.error(message="获取锁状态失败", details={"error": str(e)})

@router.post("/transactions/saga")
async def execute_saga_endpoint(request: SagaRequest):
    """执行SAGA事务"""
    try:
        success = await execute_saga(request.saga_id, request.context)
        if success:
            return APIResponse.success(message="SAGA事务执行成功")
        else:
            return APIResponse.error(message="SAGA事务执行失败")

    except Exception as e:
        logger.error(f"SAGA事务执行失败: {e}")
        return APIResponse.error(message="SAGA事务执行失败", details={"error": str(e)})

# 异步同步管理
@router.post("/sync/write")
async def write_operation_endpoint(request: WriteOperationRequest):
    """写入操作"""
    try:
        from backend.app.database.async_sync_manager import DataType, write_operation

        data_type = DataType(request.data_type)
        operation_id = write_operation(
            data_type=data_type,
            table_name=request.table_name,
            operation_type=request.operation_type,
            data=request.data,
            priority=request.priority,
            immediate=request.immediate
        )

        return APIResponse.success(
            data={"operation_id": operation_id},
            message="写入操作已提交"
        )

    except Exception as e:
        logger.error(f"写入操作失败: {e}")
        return APIResponse.error(message="写入操作失败", details={"error": str(e)})

@router.post("/sync/flush")
async def flush_operations_endpoint():
    """刷新所有操作"""
    try:
        from backend.app.database.async_sync_manager import flush_all_operations

        flushed_operations = flush_all_operations()
        total_operations = sum(len(ops) for ops in flushed_operations.values())

        return APIResponse.success(
            data={"flushed_operations": total_operations},
            message=f"刷新了 {total_operations} 个操作"
        )

    except Exception as e:
        logger.error(f"刷新操作失败: {e}")
        return APIResponse.error(message="刷新操作失败", details={"error": str(e)})

@router.get("/sync/status")
async def get_delayed_write_status_endpoint():
    """获取延迟写状态"""
    try:
        status = get_delayed_write_status()
        return APIResponse.success(data=status, message="延迟写状态获取成功")
    except Exception as e:
        logger.error(f"获取延迟写状态失败: {e}")
        return APIResponse.error(message="获取延迟写状态失败", details={"error": str(e)})

@router.post("/sync/enable")
async def enable_delayed_write_endpoint():
    """启用延迟写"""
    try:
        enable_delayed_write()
        return APIResponse.success(message="延迟写已启用")
    except Exception as e:
        logger.error(f"启用延迟写失败: {e}")
        return APIResponse.error(message="启用延迟写失败", details={"error": str(e)})

@router.post("/sync/disable")
async def disable_delayed_write_endpoint():
    """禁用延迟写"""
    try:
        disable_delayed_write()
        return APIResponse.success(message="延迟写已禁用")
    except Exception as e:
        logger.error(f"禁用延迟写失败: {e}")
        return APIResponse.error(message="禁用延迟写失败", details={"error": str(e)})

@router.get("/sync/stats")
async def get_sync_stats_endpoint():
    """获取同步统计"""
    try:
        stats = get_sync_stats()
        return APIResponse.success(data=stats, message="同步统计获取成功")
    except Exception as e:
        logger.error(f"获取同步统计失败: {e}")
        return APIResponse.error(message="获取同步统计失败", details={"error": str(e)})

# 任务管理
@router.get("/tasks/{task_id}")
async def get_task_status_endpoint(task_id: str):
    """获取任务状态"""
    try:
        status = await get_task_status(task_id)
        if status:
            return APIResponse.success(data=status, message="任务状态获取成功")
        else:
            return APIResponse.error(message="任务不存在", code=404)

    except Exception as e:
        logger.error(f"获取任务状态失败: {e}")
        return APIResponse.error(message="获取任务状态失败", details={"error": str(e)})

# 系统维护
@router.post("/maintenance/cleanup")
async def system_cleanup_endpoint():
    """系统清理"""
    try:
        # 清理旧备份
        backup_cleanup_count = await cleanup_old_backups()

        # 清理旧分区
        partition_cleanup_count = 0
        for table_name in ["glucose_predictions", "user_activities"]:
            try:
                count = await drop_old_partitions(table_name)
                partition_cleanup_count += count
            except Exception as e:
                logger.warning(f"清理分区失败 {table_name}: {e}")

        # 清理缓存
        await cache_manager.clear_pattern("temp:*")

        cleanup_result = {
            "backup_cleanup": backup_cleanup_count,
            "partition_cleanup": partition_cleanup_count,
            "cache_cleanup": "completed"
        }

        return APIResponse.success(
            data=cleanup_result,
            message="系统清理完成"
        )

    except Exception as e:
        logger.error(f"系统清理失败: {e}")
        return APIResponse.error(message="系统清理失败", details={"error": str(e)})

@router.post("/maintenance/optimize")
async def system_optimize_endpoint():
    """系统优化"""
    try:
        # 自动优化数据库
        optimization_result = await auto_optimize_database()

        # 刷新所有延迟写操作
        from backend.app.database.async_sync_manager import flush_all_operations
        flushed_operations = flush_all_operations()

        # 清理过期锁
        acid_manager.transaction_manager.pessimistic_lock_manager.release_expired_locks()

        optimize_result = {
            "database_optimization": optimization_result,
            "flushed_operations": sum(len(ops) for ops in flushed_operations.values()),
            "lock_cleanup": "completed"
        }

        return APIResponse.success(
            data=optimize_result,
            message="系统优化完成"
        )

    except Exception as e:
        logger.error(f"系统优化失败: {e}")
        return APIResponse.error(message="系统优化失败", details={"error": str(e)})

# 健康检查
@router.get("/health")
async def database_health_check():
    """数据库健康检查"""
    try:
        # 基本健康检查
        db_healthy, db_status = await check_db_health()

        # 检查各个组件状态
        components_status = {
            "database": db_healthy,
            "cache": await cache_manager.health_check(),
            "sharding": len(get_sharding_status().get("rules", {})) > 0,
            "backup": get_backup_stats().get("total_backups", 0) > 0,
            "partitioning": get_partition_stats().get("total_tables", 0) > 0,
            "transactions": get_transaction_stats().get("active_transactions", 0) >= 0,
            "delayed_write": get_delayed_write_status().get("enabled", False),
            "sync": get_sync_stats().get("total_syncs", 0) >= 0
        }

        overall_health = all(components_status.values())

        health_status = {
            "overall_health": overall_health,
            "timestamp": datetime.now().isoformat(),
            "components": components_status,
            "database_status": db_status
        }

        if overall_health:
            return APIResponse.success(data=health_status, message="数据库健康检查通过")
        else:
            return APIResponse.error(
                message="数据库健康检查失败",
                code=503,
                details=health_status
            )

    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return APIResponse.error(message="健康检查失败", details={"error": str(e)})

__all__ = ["'logger'", "'router'", "'ShardingRuleRequest'", "'BackupRequest'", "'PartitionRequest'", "'SagaRequest'", "'WriteOperationRequest'"]
