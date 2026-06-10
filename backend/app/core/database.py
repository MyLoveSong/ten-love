"""
数据库核心功能
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
import json

"""
数据库配置模块
企业级数据库管理
"""

from sqlalchemy import create_engine, MetaData, event, text
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool, QueuePool
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DisconnectionError
import logging
import time
from contextlib import contextmanager
from typing import Generator

from .config import settings

logger = logging.getLogger(__name__)

# 创建数据库引擎
if settings.DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        settings.DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=settings.DATABASE_ECHO,
    )
else:
    engine = create_engine(
        settings.DATABASE_URL,
        echo=settings.DATABASE_ECHO,
        # PostgreSQL/MySQL连接池配置
        pool_size=20,
        max_overflow=30,
        pool_pre_ping=True,
        pool_recycle=3600,
        pool_timeout=30,
        # 事务隔离级别
        isolation_level="READ_COMMITTED",
        # 连接参数
        connect_args={
            "application_name": "academic_health_system"
        }
    )

# 创建会话工厂
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False  # 防止对象过期
)

# 创建基础模型类
Base = declarative_base()

# 元数据
metadata = MetaData()

# 添加连接事件监听器
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """设置SQLite连接参数"""
    if "sqlite" in settings.DATABASE_URL:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA cache_size=10000")
        cursor.execute("PRAGMA temp_store=MEMORY")
        cursor.close()

@event.listens_for(engine, "checkout")
def receive_checkout(dbapi_connection, connection_record, connection_proxy):
    """连接检出事件"""
    logger.debug("数据库连接被检出")

@event.listens_for(engine, "checkin")
def receive_checkin(dbapi_connection, connection_record):
    """连接检入事件"""
    logger.debug("数据库连接被检入")

async def init_db():
    """初始化数据库"""
    try:
        # 导入所有模型以确保表被创建
        from app.database.models import Base as ModelsBase

        # 创建所有表
        ModelsBase.metadata.create_all(bind=engine)
        logger.info("✅ 数据库表创建完成")

        # 可以在这里添加初始数据
        # await seed_initial_data()

    except Exception as e:
        logger.error(f"❌ 数据库初始化失败: {e}")
        raise

def get_db():
    """获取数据库会话"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@contextmanager
def get_db_transaction() -> Generator:
    """获取数据库事务上下文管理器"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"数据库事务回滚: {e}")
        raise
    finally:
        db.close()

# 数据库健康检查
async def check_db_health():
    """检查数据库连接健康状态"""
    try:
        with engine.connect() as connection:
            # 执行简单查询测试连接
            result = connection.execute(text("SELECT 1"))
            result.fetchone()

            # 检查连接池状态
            pool = engine.pool
            if hasattr(pool, 'size') and hasattr(pool, 'checkedin'):
                pool_status = {
                    "size": pool.size(),
                    "checked_in": pool.checkedin(),
                    "checked_out": pool.checkedout(),
                    "overflow": pool.overflow(),
                    "invalid": pool.invalid() if hasattr(pool, 'invalid') else 0
                }
            else:
                # SQLite StaticPool没有这些方法
                pool_status = {
                    "type": "StaticPool",
                    "status": "active"
                }

            logger.info(f"数据库连接池状态: {pool_status}")
            return True, pool_status
    except Exception as e:
        logger.error(f"数据库健康检查失败: {e}")
        return False, {"error": str(e)}

def get_connection_pool_status():
    """获取连接池状态"""
    pool = engine.pool
    if hasattr(pool, 'size'):
        return {
            "pool_size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "invalid": pool.invalid(),
            "total_connections": pool.size() + pool.overflow()
        }
    else:
        # SQLite StaticPool
        return {
            "pool_type": "StaticPool",
            "status": "active"
        }

# 数据库性能监控
class DatabasePerformanceMonitor:
    """数据库性能监控器"""

    def __init__(self):
        self.query_times = []
        self.slow_queries = []
        self.connection_errors = 0

    def log_query_time(self, query_time: float, query: str = ""):
        """记录查询时间"""
        self.query_times.append(query_time)

        # 记录慢查询
        if query_time > 1.0:  # 超过1秒的查询
            self.slow_queries.append({
                "time": query_time,
                "query": query[:100] if query else "unknown",
                "timestamp": time.time()
            })

    def get_performance_stats(self):
        """获取性能统计"""
        if not self.query_times:
            return {"avg_time": 0, "max_time": 0, "slow_queries": 0}

        return {
            "avg_time": sum(self.query_times) / len(self.query_times),
            "max_time": max(self.query_times),
            "slow_queries": len(self.slow_queries),
            "total_queries": len(self.query_times)
        }

# 全局性能监控实例
db_performance_monitor = DatabasePerformanceMonitor()

# 添加查询时间监控
@event.listens_for(engine, "before_cursor_execute")
def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    """查询执行前事件"""
    context._query_start_time = time.time()

@event.listens_for(engine, "after_cursor_execute")
def receive_after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    """查询执行后事件"""
    if hasattr(context, '_query_start_time'):
        query_time = time.time() - context._query_start_time
        db_performance_monitor.log_query_time(query_time, statement)

        # 记录到性能优化器
        try:
            from app.database.performance_optimizer import record_query_execution
            record_query_execution(statement, query_time)
        except ImportError:
            pass

__all__ = ["'logger'", "'SessionLocal'", "'Base'", "'metadata'", "'set_sqlite_pragma'", "'receive_checkout'", "'receive_checkin'", "'get_db'", "'get_db_transaction'", "'get_connection_pool_status'", "'DatabasePerformanceMonitor'", "'db_performance_monitor'", "'receive_before_cursor_execute'", "'receive_after_cursor_execute'"]
