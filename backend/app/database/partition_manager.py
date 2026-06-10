

"""
数据库分区管理器
支持表分区和分区管理
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json

from sqlalchemy import create_engine, text, MetaData, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

from backend.app.core.config import settings
from backend.app.core.exceptions import DatabaseError, CustomException
from backend.app.core.database import engine, SessionLocal
from backend.app.core.task_queue import async_task, TaskPriority

logger = logging.getLogger(__name__)

class PartitionType(Enum):
    """分区类型枚举"""
    RANGE = "range"  # 范围分区
    HASH = "hash"    # 哈希分区
    LIST = "list"    # 列表分区
    TIME = "time"    # 时间分区

class PartitionStrategy(Enum):
    """分区策略枚举"""
    YEARLY = "yearly"    # 按年分区
    MONTHLY = "monthly"  # 按月分区
    WEEKLY = "weekly"    # 按周分区
    DAILY = "daily"      # 按日分区
    CUSTOM = "custom"    # 自定义分区

@dataclass
class PartitionConfig:
    """分区配置"""
    table_name: str
    partition_type: PartitionType
    partition_strategy: PartitionStrategy
    partition_column: str
    partition_count: Optional[int] = None
    partition_ranges: Optional[List[Tuple[Any, Any]]] = None
    partition_values: Optional[List[List[Any]]] = None
    auto_create: bool = True
    retention_days: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class PartitionInfo:
    """分区信息"""
    partition_name: str
    table_name: str
    partition_type: PartitionType
    partition_value: Optional[Any] = None
    partition_range: Optional[Tuple[Any, Any]] = None
    row_count: Optional[int] = None
    size_bytes: Optional[int] = None
    created_at: Optional[datetime] = None
    last_accessed: Optional[datetime] = None

class DatabasePartitionManager:
    """数据库分区管理器"""

    def __init__(self):
        self.partition_configs: Dict[str, PartitionConfig] = {}
        self.partition_info: Dict[str, List[PartitionInfo]] = {}

        # 初始化默认分区配置
        self._init_default_partitions()

    def _init_default_partitions(self):
        """初始化默认分区配置"""
        # 血糖预测表按时间分区
        glucose_config = PartitionConfig(
            table_name="glucose_predictions",
            partition_type=PartitionType.TIME,
            partition_strategy=PartitionStrategy.MONTHLY,
            partition_column="prediction_time",
            auto_create=True,
            retention_days=365
        )

        self.add_partition_config("glucose_predictions", glucose_config)

        # 用户活动日志按时间分区
        activity_config = PartitionConfig(
            table_name="user_activities",
            partition_type=PartitionType.TIME,
            partition_strategy=PartitionStrategy.DAILY,
            partition_column="created_at",
            auto_create=True,
            retention_days=90
        )

        self.add_partition_config("user_activities", activity_config)

    def add_partition_config(self, table_name: str, config: PartitionConfig):
        """添加分区配置"""
        self.partition_configs[table_name] = config
        logger.info(f"添加分区配置: {table_name} - {config.partition_strategy.value}")

    async def create_partitioned_table(self, table_name: str, config: PartitionConfig) -> bool:
        """创建分区表"""
        try:
            if settings.DATABASE_URL.startswith("sqlite"):
                # SQLite不支持原生分区，使用应用层分区
                return await self._create_sqlite_partitioned_table(table_name, config)
            elif settings.DATABASE_URL.startswith("postgresql"):
                return await self._create_postgresql_partitioned_table(table_name, config)
            elif settings.DATABASE_URL.startswith("mysql"):
                return await self._create_mysql_partitioned_table(table_name, config)
            else:
                raise CustomException(f"不支持的数据库类型: {settings.DATABASE_URL}")

        except Exception as e:
            logger.error(f"创建分区表失败 {table_name}: {e}")
            raise DatabaseError(f"创建分区表失败: {e}")

    async def _create_sqlite_partitioned_table(self, table_name: str, config: PartitionConfig) -> bool:
        """创建SQLite分区表（应用层实现）"""
        # SQLite不支持原生分区，这里实现应用层分区
        logger.info(f"SQLite不支持原生分区，使用应用层分区: {table_name}")

        # 创建分区表
        partition_tables = self._generate_partition_tables(table_name, config)

        with engine.connect() as connection:
            for partition_table in partition_tables:
                try:
                    connection.execute(text(partition_table["sql"]))
                    connection.commit()
                    logger.info(f"创建分区表: {partition_table['name']}")
                except Exception as e:
                    logger.error(f"创建分区表失败 {partition_table['name']}: {e}")

        return True

    async def _create_postgresql_partitioned_table(self, table_name: str, config: PartitionConfig) -> bool:
        """创建PostgreSQL分区表"""
        with engine.connect() as connection:
            try:
                # 创建主表
                if config.partition_type == PartitionType.TIME:
                    partition_sql = f"""
                        CREATE TABLE IF NOT EXISTS {table_name} (
                            id SERIAL PRIMARY KEY,
                            {config.partition_column} TIMESTAMP NOT NULL,
                            -- 其他列定义
                        ) PARTITION BY RANGE ({config.partition_column});
                    """
                elif config.partition_type == PartitionType.HASH:
                    partition_sql = f"""
                        CREATE TABLE IF NOT EXISTS {table_name} (
                            id SERIAL PRIMARY KEY,
                            {config.partition_column} INTEGER NOT NULL,
                            -- 其他列定义
                        ) PARTITION BY HASH ({config.partition_column});
                    """
                else:
                    raise CustomException(f"不支持的分区类型: {config.partition_type}")

                connection.execute(text(partition_sql))

                # 创建分区
                await self._create_postgresql_partitions(connection, table_name, config)

                connection.commit()
                logger.info(f"PostgreSQL分区表创建成功: {table_name}")
                return True

            except Exception as e:
                connection.rollback()
                raise e

    async def _create_mysql_partitioned_table(self, table_name: str, config: PartitionConfig) -> bool:
        """创建MySQL分区表"""
        with engine.connect() as connection:
            try:
                # 创建主表
                if config.partition_type == PartitionType.TIME:
                    partition_sql = f"""
                        CREATE TABLE IF NOT EXISTS {table_name} (
                            id INT AUTO_INCREMENT PRIMARY KEY,
                            {config.partition_column} DATETIME NOT NULL,
                            -- 其他列定义
                        ) PARTITION BY RANGE (YEAR({config.partition_column}) * 100 + MONTH({config.partition_column}));
                    """
                elif config.partition_type == PartitionType.HASH:
                    partition_sql = f"""
                        CREATE TABLE IF NOT EXISTS {table_name} (
                            id INT AUTO_INCREMENT PRIMARY KEY,
                            {config.partition_column} INT NOT NULL,
                            -- 其他列定义
                        ) PARTITION BY HASH ({config.partition_column}) PARTITIONS {config.partition_count or 4};
                    """
                else:
                    raise CustomException(f"不支持的分区类型: {config.partition_type}")

                connection.execute(text(partition_sql))

                # 创建分区
                await self._create_mysql_partitions(connection, table_name, config)

                connection.commit()
                logger.info(f"MySQL分区表创建成功: {table_name}")
                return True

            except Exception as e:
                connection.rollback()
                raise e

    async def _create_postgresql_partitions(self, connection, table_name: str, config: PartitionConfig):
        """创建PostgreSQL分区"""
        if config.partition_type == PartitionType.TIME:
            # 创建时间分区
            current_date = datetime.now()

            if config.partition_strategy == PartitionStrategy.MONTHLY:
                # 创建过去12个月和未来3个月的分区
                for i in range(-12, 4):
                    partition_date = current_date + timedelta(days=30 * i)
                    partition_name = f"{table_name}_{partition_date.strftime('%Y_%m')}"

                    start_date = partition_date.replace(day=1)
                    if start_date.month == 12:
                        end_date = start_date.replace(year=start_date.year + 1, month=1)
                    else:
                        end_date = start_date.replace(month=start_date.month + 1)

                    partition_sql = f"""
                        CREATE TABLE IF NOT EXISTS {partition_name}
                        PARTITION OF {table_name}
                        FOR VALUES FROM ('{start_date.isoformat()}') TO ('{end_date.isoformat()}');
                    """

                    try:
                        connection.execute(text(partition_sql))
                        logger.info(f"创建PostgreSQL分区: {partition_name}")
                    except Exception as e:
                        logger.warning(f"创建分区失败 {partition_name}: {e}")

    async def _create_mysql_partitions(self, connection, table_name: str, config: PartitionConfig):
        """创建MySQL分区"""
        if config.partition_type == PartitionType.TIME:
            # MySQL时间分区需要手动定义
            current_date = datetime.now()

            if config.partition_strategy == PartitionStrategy.MONTHLY:
                # 创建过去12个月和未来3个月的分区
                for i in range(-12, 4):
                    partition_date = current_date + timedelta(days=30 * i)
                    partition_name = f"p_{partition_date.strftime('%Y%m')}"

                    start_date = partition_date.replace(day=1)
                    if start_date.month == 12:
                        end_date = start_date.replace(year=start_date.year + 1, month=1)
                    else:
                        end_date = start_date.replace(month=start_date.month + 1)

                    partition_sql = f"""
                        ALTER TABLE {table_name}
                        ADD PARTITION (
                            PARTITION {partition_name}
                            VALUES LESS THAN ({end_date.year * 100 + end_date.month})
                        );
                    """

                    try:
                        connection.execute(text(partition_sql))
                        logger.info(f"创建MySQL分区: {partition_name}")
                    except Exception as e:
                        logger.warning(f"创建分区失败 {partition_name}: {e}")

    def _generate_partition_tables(self, table_name: str, config: PartitionConfig) -> List[Dict[str, str]]:
        """生成分区表定义（SQLite应用层分区）"""
        partition_tables = []
        current_date = datetime.now()

        if config.partition_strategy == PartitionStrategy.MONTHLY:
            # 创建过去12个月和未来3个月的分区表
            for i in range(-12, 4):
                partition_date = current_date + timedelta(days=30 * i)
                partition_name = f"{table_name}_{partition_date.strftime('%Y_%m')}"

                # 这里需要根据实际表结构生成SQL
                partition_sql = f"""
                    CREATE TABLE IF NOT EXISTS {partition_name} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        {config.partition_column} TIMESTAMP NOT NULL,
                        -- 其他列定义
                        CHECK ({config.partition_column} >= '{partition_date.replace(day=1).isoformat()}'
                               AND {config.partition_column} < '{(partition_date.replace(day=1) + timedelta(days=32)).replace(day=1).isoformat()}')
                    );
                """

                partition_tables.append({
                    "name": partition_name,
                    "sql": partition_sql,
                    "date_range": (partition_date.replace(day=1), (partition_date.replace(day=1) + timedelta(days=32)).replace(day=1))
                })

        return partition_tables

    async def get_partition_info(self, table_name: str) -> List[PartitionInfo]:
        """获取分区信息"""
        if table_name not in self.partition_configs:
            return []

        config = self.partition_configs[table_name]
        partition_info = []

        try:
            if settings.DATABASE_URL.startswith("sqlite"):
                partition_info = await self._get_sqlite_partition_info(table_name, config)
            elif settings.DATABASE_URL.startswith("postgresql"):
                partition_info = await self._get_postgresql_partition_info(table_name, config)
            elif settings.DATABASE_URL.startswith("mysql"):
                partition_info = await self._get_mysql_partition_info(table_name, config)

            self.partition_info[table_name] = partition_info
            return partition_info

        except Exception as e:
            logger.error(f"获取分区信息失败 {table_name}: {e}")
            return []

    async def _get_sqlite_partition_info(self, table_name: str, config: PartitionConfig) -> List[PartitionInfo]:
        """获取SQLite分区信息"""
        partition_info = []

        with engine.connect() as connection:
            # 获取所有分区表
            tables_query = text("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name LIKE :pattern
            """)

            result = connection.execute(tables_query, {"pattern": f"{table_name}_%"})

            for row in result:
                partition_name = row[0]

                # 获取行数
                count_query = text(f"SELECT COUNT(*) FROM {partition_name}")
                count_result = connection.execute(count_query)
                row_count = count_result.scalar()

                # 获取表大小（SQLite不直接支持，这里用行数估算）
                size_bytes = row_count * 100  # 估算每行100字节

                partition_info.append(PartitionInfo(
                    partition_name=partition_name,
                    table_name=table_name,
                    partition_type=config.partition_type,
                    row_count=row_count,
                    size_bytes=size_bytes,
                    created_at=datetime.now()  # SQLite不记录创建时间
                ))

        return partition_info

    async def _get_postgresql_partition_info(self, table_name: str, config: PartitionConfig) -> List[PartitionInfo]:
        """获取PostgreSQL分区信息"""
        partition_info = []

        with engine.connect() as connection:
            query = text("""
                SELECT
                    schemaname,
                    tablename,
                    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
                    pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
                FROM pg_tables
                WHERE tablename LIKE :pattern
            """)

            result = connection.execute(query, {"pattern": f"{table_name}_%"})

            for row in result:
                partition_name = row[1]

                # 获取行数
                count_query = text(f"SELECT COUNT(*) FROM {partition_name}")
                count_result = connection.execute(count_query)
                row_count = count_result.scalar()

                partition_info.append(PartitionInfo(
                    partition_name=partition_name,
                    table_name=table_name,
                    partition_type=config.partition_type,
                    row_count=row_count,
                    size_bytes=row[3],
                    created_at=datetime.now()  # PostgreSQL需要额外查询获取创建时间
                ))

        return partition_info

    async def _get_mysql_partition_info(self, table_name: str, config: PartitionConfig) -> List[PartitionInfo]:
        """获取MySQL分区信息"""
        partition_info = []

        with engine.connect() as connection:
            query = text("""
                SELECT
                    PARTITION_NAME,
                    TABLE_ROWS,
                    DATA_LENGTH + INDEX_LENGTH as size_bytes
                FROM INFORMATION_SCHEMA.PARTITIONS
                WHERE TABLE_NAME = :table_name AND PARTITION_NAME IS NOT NULL
            """)

            result = connection.execute(query, {"table_name": table_name})

            for row in result:
                partition_info.append(PartitionInfo(
                    partition_name=row[0],
                    table_name=table_name,
                    partition_type=config.partition_type,
                    row_count=row[1],
                    size_bytes=row[2],
                    created_at=datetime.now()  # MySQL需要额外查询获取创建时间
                ))

        return partition_info

    async def create_new_partition(self, table_name: str, partition_value: Any) -> bool:
        """创建新分区"""
        if table_name not in self.partition_configs:
            raise CustomException(f"表 {table_name} 没有分区配置")

        config = self.partition_configs[table_name]

        try:
            if settings.DATABASE_URL.startswith("sqlite"):
                return await self._create_sqlite_partition(table_name, config, partition_value)
            elif settings.DATABASE_URL.startswith("postgresql"):
                return await self._create_postgresql_partition(table_name, config, partition_value)
            elif settings.DATABASE_URL.startswith("mysql"):
                return await self._create_mysql_partition(table_name, config, partition_value)
            else:
                raise CustomException(f"不支持的数据库类型: {settings.DATABASE_URL}")

        except Exception as e:
            logger.error(f"创建新分区失败 {table_name}: {e}")
            raise DatabaseError(f"创建新分区失败: {e}")

    async def _create_sqlite_partition(self, table_name: str, config: PartitionConfig, partition_value: Any) -> bool:
        """创建SQLite分区"""
        partition_name = f"{table_name}_{partition_value}"

        with engine.connect() as connection:
            partition_sql = f"""
                CREATE TABLE IF NOT EXISTS {partition_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    {config.partition_column} TIMESTAMP NOT NULL,
                    -- 其他列定义
                );
            """

            connection.execute(text(partition_sql))
            connection.commit()

            logger.info(f"创建SQLite分区: {partition_name}")
            return True

    async def _create_postgresql_partition(self, table_name: str, config: PartitionConfig, partition_value: Any) -> bool:
        """创建PostgreSQL分区"""
        partition_name = f"{table_name}_{partition_value}"

        with engine.connect() as connection:
            # 根据分区类型创建分区
            if config.partition_type == PartitionType.TIME:
                start_date = partition_value
                end_date = start_date + timedelta(days=32)  # 下个月

                partition_sql = f"""
                    CREATE TABLE {partition_name}
                    PARTITION OF {table_name}
                    FOR VALUES FROM ('{start_date.isoformat()}') TO ('{end_date.isoformat()}');
                """

            connection.execute(text(partition_sql))
            connection.commit()

            logger.info(f"创建PostgreSQL分区: {partition_name}")
            return True

    async def _create_mysql_partition(self, table_name: str, config: PartitionConfig, partition_value: Any) -> bool:
        """创建MySQL分区"""
        partition_name = f"p_{partition_value}"

        with engine.connect() as connection:
            if config.partition_type == PartitionType.TIME:
                end_value = partition_value + 1  # 下个月

                partition_sql = f"""
                    ALTER TABLE {table_name}
                    ADD PARTITION (
                        PARTITION {partition_name}
                        VALUES LESS THAN ({end_value})
                    );
                """

            connection.execute(text(partition_sql))
            connection.commit()

            logger.info(f"创建MySQL分区: {partition_name}")
            return True

    async def drop_old_partitions(self, table_name: str) -> int:
        """删除旧分区"""
        if table_name not in self.partition_configs:
            return 0

        config = self.partition_configs[table_name]
        if not config.retention_days:
            return 0

        dropped_count = 0
        cutoff_date = datetime.now() - timedelta(days=config.retention_days)

        try:
            partition_info = await self.get_partition_info(table_name)

            for partition in partition_info:
                # 根据分区名称解析日期
                partition_date = self._parse_partition_date(partition.partition_name, config)

                if partition_date and partition_date < cutoff_date:
                    await self._drop_partition(table_name, partition.partition_name)
                    dropped_count += 1
                    logger.info(f"删除过期分区: {partition.partition_name}")

        except Exception as e:
            logger.error(f"删除旧分区失败 {table_name}: {e}")

        return dropped_count

    def _parse_partition_date(self, partition_name: str, config: PartitionConfig) -> Optional[datetime]:
        """解析分区日期"""
        try:
            if config.partition_strategy == PartitionStrategy.MONTHLY:
                # 解析格式: table_name_2024_01
                date_part = partition_name.split('_')[-2:]
                if len(date_part) == 2:
                    year, month = int(date_part[0]), int(date_part[1])
                    return datetime(year, month, 1)
            elif config.partition_strategy == PartitionStrategy.DAILY:
                # 解析格式: table_name_2024_01_01
                date_part = partition_name.split('_')[-3:]
                if len(date_part) == 3:
                    year, month, day = int(date_part[0]), int(date_part[1]), int(date_part[2])
                    return datetime(year, month, day)
        except (ValueError, IndexError):
            pass

        return None

    async def _drop_partition(self, table_name: str, partition_name: str):
        """删除分区"""
        with engine.connect() as connection:
            if settings.DATABASE_URL.startswith("sqlite"):
                drop_sql = f"DROP TABLE IF EXISTS {partition_name}"
            elif settings.DATABASE_URL.startswith("postgresql"):
                drop_sql = f"DROP TABLE IF EXISTS {partition_name}"
            elif settings.DATABASE_URL.startswith("mysql"):
                drop_sql = f"ALTER TABLE {table_name} DROP PARTITION {partition_name}"
            else:
                raise CustomException(f"不支持的数据库类型: {settings.DATABASE_URL}")

            connection.execute(text(drop_sql))
            connection.commit()

    def get_partition_stats(self) -> Dict[str, Any]:
        """获取分区统计信息"""
        stats = {
            "total_tables": len(self.partition_configs),
            "total_partitions": sum(len(partitions) for partitions in self.partition_info.values()),
            "tables": {}
        }

        for table_name, config in self.partition_configs.items():
            partitions = self.partition_info.get(table_name, [])

            stats["tables"][table_name] = {
                "partition_type": config.partition_type.value,
                "partition_strategy": config.partition_strategy.value,
                "partition_column": config.partition_column,
                "partition_count": len(partitions),
                "total_rows": sum(p.row_count or 0 for p in partitions),
                "total_size_bytes": sum(p.size_bytes or 0 for p in partitions),
                "retention_days": config.retention_days
            }

        return stats

# 全局分区管理器实例
partition_manager = DatabasePartitionManager()

# 异步任务
@async_task("create_partitioned_table", TaskPriority.HIGH)
def create_partitioned_table_task(table_name: str, config_dict: Dict[str, Any]):
    """创建分区表任务"""
    config = PartitionConfig(**config_dict)
    return asyncio.run(partition_manager.create_partitioned_table(table_name, config))

@async_task("create_new_partition", TaskPriority.NORMAL)
def create_new_partition_task(table_name: str, partition_value: str):
    """创建新分区任务"""
    return asyncio.run(partition_manager.create_new_partition(table_name, partition_value))

@async_task("drop_old_partitions", TaskPriority.LOW)
def drop_old_partitions_task(table_name: str):
    """删除旧分区任务"""
    return asyncio.run(partition_manager.drop_old_partitions(table_name))

# 分区管理API
def add_partition_config(table_name: str, config: PartitionConfig):
    """添加分区配置"""
    partition_manager.add_partition_config(table_name, config)

async def create_partitioned_table(table_name: str, config: PartitionConfig) -> bool:
    """创建分区表"""
    return await partition_manager.create_partitioned_table(table_name, config)

async def get_partition_info(table_name: str) -> List[PartitionInfo]:
    """获取分区信息"""
    return await partition_manager.get_partition_info(table_name)

async def create_new_partition(table_name: str, partition_value: Any) -> bool:
    """创建新分区"""
    return await partition_manager.create_new_partition(table_name, partition_value)

async def drop_old_partitions(table_name: str) -> int:
    """删除旧分区"""
    return await partition_manager.drop_old_partitions(table_name)

def get_partition_stats() -> Dict[str, Any]:
    """获取分区统计"""
    return partition_manager.get_partition_stats()

if __name__ == "__main__":
    # 测试分区管理器
    print("分区管理器统计:")
    print(json.dumps(get_partition_stats(), indent=2, ensure_ascii=False))

    # 获取分区信息
    partition_info = asyncio.run(get_partition_info("glucose_predictions"))
    print(f"血糖预测表分区信息: {len(partition_info)} 个分区")

__all__ = ["'logger'", "'PartitionType'", "'PartitionStrategy'", "'PartitionConfig'", "'PartitionInfo'", "'DatabasePartitionManager'", "'partition_manager'", "'create_partitioned_table_task'", "'create_new_partition_task'", "'drop_old_partitions_task'", "'add_partition_config'", "'get_partition_stats'"]
