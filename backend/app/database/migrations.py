"""
数据库迁移管理
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
import json

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库迁移管理器
支持数据库版本升级和迁移
"""

import logging
logger = logging.getLogger(__name__)
from sqlalchemy import text, inspect
from sqlalchemy.orm import Session
from datetime import datetime
import json
import os

from .models import Base

logger = logging.getLogger(__name__)

class MigrationManager:
    """数据库迁移管理器"""

    def __init__(self, session: Session):
        self.session = session
        self.migrations_table = "schema_migrations"

    def create_migrations_table(self):
        """创建迁移记录表"""
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.migrations_table} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version VARCHAR(50) NOT NULL UNIQUE,
            description TEXT,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            checksum VARCHAR(64)
        )
        """
        self.session.execute(text(create_table_sql))
        self.session.commit()
        logger.info("迁移记录表创建完成")

    def get_applied_migrations(self) -> List[str]:
        """获取已应用的迁移版本"""
        try:
            result = self.session.execute(text(f"SELECT version FROM {self.migrations_table} ORDER BY version"))
            return [row[0] for row in result.fetchall()]
        except Exception as e:
            logger.warning(f"获取已应用迁移失败: {e}")
            return []

    def record_migration(self, version: str, description: str, checksum: str = ""):
        """记录迁移应用"""
        insert_sql = f"""
        INSERT INTO {self.migrations_table} (version, description, checksum, applied_at)
        VALUES (:version, :description, :checksum, :applied_at)
        """
        self.session.execute(text(insert_sql), {
            "version": version,
            "description": description,
            "checksum": checksum,
            "applied_at": datetime.utcnow()
        })
        self.session.commit()
        logger.info(f"记录迁移: {version} - {description}")

    def apply_migration(self, version: str, description: str, migration_sql: str, checksum: str = ""):
        """应用迁移"""
        try:
            # 执行迁移SQL
            self.session.execute(text(migration_sql))
            self.session.commit()

            # 记录迁移
            self.record_migration(version, description, checksum)

            logger.info(f"迁移应用成功: {version}")
            return True
        except Exception as e:
            self.session.rollback()
            logger.error(f"迁移应用失败: {version} - {e}")
            return False

    def rollback_migration(self, version: str, rollback_sql: str):
        """回滚迁移"""
        try:
            # 执行回滚SQL
            self.session.execute(text(rollback_sql))

            # 删除迁移记录
            delete_sql = f"DELETE FROM {self.migrations_table} WHERE version = :version"
            self.session.execute(text(delete_sql), {"version": version})

            self.session.commit()
            logger.info(f"迁移回滚成功: {version}")
            return True
        except Exception as e:
            self.session.rollback()
            logger.error(f"迁移回滚失败: {version} - {e}")
            return False

    def get_database_schema_version(self) -> str:
        """获取数据库架构版本"""
        try:
            result = self.session.execute(text(f"SELECT MAX(version) FROM {self.migrations_table}"))
            version = result.fetchone()[0]
            return version if version else "0.0.0"
        except Exception as e:
            logger.warning(f"获取数据库版本失败: {e}")
            return "0.0.0"

    def check_schema_consistency(self) -> Dict[str, Any]:
        """检查架构一致性"""
        inspector = inspect(engine)
        existing_tables = set(inspector.get_table_names())

        # 获取所有模型表名
        expected_tables = set(Base.metadata.tables.keys())

        missing_tables = expected_tables - existing_tables
        extra_tables = existing_tables - expected_tables

        return {
            "consistent": len(missing_tables) == 0 and len(extra_tables) == 0,
            "missing_tables": list(missing_tables),
            "extra_tables": list(extra_tables),
            "existing_tables": list(existing_tables),
            "expected_tables": list(expected_tables)
        }

    def initialize_database(self):
        """初始化数据库"""
        try:
            # 创建迁移记录表
            self.create_migrations_table()

            # 创建所有表
            Base.metadata.create_all(bind=engine)

            # 记录初始迁移
            initial_version = "1.0.0"
            applied_migrations = self.get_applied_migrations()

            if initial_version not in applied_migrations:
                self.record_migration(
                    initial_version,
                    "初始数据库架构创建",
                    "initial_schema"
                )

            logger.info("数据库初始化完成")
            return True
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            return False

    def upgrade_database(self, target_version: str = None):
        """升级数据库到指定版本"""
        try:
            # 获取当前版本
            current_version = self.get_database_schema_version()
            logger.info(f"当前数据库版本: {current_version}")

            # 定义迁移脚本
            migrations = self._get_migration_scripts()

            # 应用未应用的迁移
            applied_migrations = self.get_applied_migrations()

            for version, migration_info in migrations.items():
                if version not in applied_migrations:
                    if target_version and version > target_version:
                        break

                    logger.info(f"应用迁移: {version}")
                    success = self.apply_migration(
                        version,
                        migration_info["description"],
                        migration_info["sql"],
                        migration_info.get("checksum", "")
                    )

                    if not success:
                        logger.error(f"迁移失败，停止升级: {version}")
                        return False

            logger.info("数据库升级完成")
            return True
        except Exception as e:
            logger.error(f"数据库升级失败: {e}")
            return False

    def _get_migration_scripts(self) -> Dict[str, Dict[str, Any]]:
        """获取迁移脚本定义"""
        return {
            "1.0.1": {
                "description": "添加用户配置数据字段",
                "sql": """
                ALTER TABLE users ADD COLUMN profile_data TEXT DEFAULT '{}';
                """,
                "checksum": "add_profile_data"
            },
            "1.0.2": {
                "description": "添加食物项目英文名称字段",
                "sql": """
                ALTER TABLE food_items ADD COLUMN name_en VARCHAR(200);
                CREATE INDEX idx_food_name_en ON food_items(name_en);
                """,
                "checksum": "add_food_name_en"
            },
            "1.0.3": {
                "description": "添加血糖预测上下文数据字段",
                "sql": """
                ALTER TABLE glucose_predictions ADD COLUMN context_data TEXT DEFAULT '{}';
                ALTER TABLE glucose_predictions ADD COLUMN model_version VARCHAR(50) DEFAULT '1.0.0';
                """,
                "checksum": "add_context_data"
            },
            "1.0.4": {
                "description": "添加工作流复杂度评分字段",
                "sql": """
                ALTER TABLE workflow_definitions ADD COLUMN complexity_score FLOAT;
                ALTER TABLE workflow_definitions ADD COLUMN category VARCHAR(100);
                CREATE INDEX idx_workflow_category ON workflow_definitions(category);
                """,
                "checksum": "add_workflow_complexity"
            },
            "1.0.5": {
                "description": "添加系统指标标签字段",
                "sql": """
                ALTER TABLE system_metrics ADD COLUMN tags TEXT DEFAULT '[]';
                ALTER TABLE system_metrics ADD COLUMN metric_unit VARCHAR(20);
                """,
                "checksum": "add_metrics_tags"
            },
            "1.0.6": {
                "description": "添加告警通知状态字段",
                "sql": """
                ALTER TABLE alerts ADD COLUMN notification_sent BOOLEAN DEFAULT FALSE;
                """,
                "checksum": "add_notification_status"
            },
            "1.0.7": {
                "description": "添加反馈类型和评分字段",
                "sql": """
                ALTER TABLE feedback_records ADD COLUMN feedback_type VARCHAR(50) DEFAULT 'prediction';
                ALTER TABLE feedback_records ADD COLUMN accuracy_rating INTEGER;
                ALTER TABLE feedback_records ADD COLUMN recommendation_helpfulness INTEGER;
                CREATE INDEX idx_feedback_type ON feedback_records(feedback_type);
                """,
                "checksum": "add_feedback_types"
            },
            "1.0.8": {
                "description": "添加实验数据增强记录字段",
                "sql": """
                ALTER TABLE experiment_results ADD COLUMN mae FLOAT;
                ALTER TABLE experiment_results ADD COLUMN rmse FLOAT;
                ALTER TABLE experiment_results ADD COLUMN training_time FLOAT;
                ALTER TABLE experiment_results ADD COLUMN inference_time FLOAT;
                ALTER TABLE experiment_results ADD COLUMN experiment_notes TEXT;
                """,
                "checksum": "add_experiment_metrics"
            },
            "1.0.9": {
                "description": "添加数据增强记录处理时间字段",
                "sql": """
                ALTER TABLE data_augmentation_records ADD COLUMN processing_time FLOAT;
                ALTER TABLE data_augmentation_records ADD COLUMN dataset_name VARCHAR(200);
                CREATE INDEX idx_augmentation_dataset ON data_augmentation_records(dataset_name);
                """,
                "checksum": "add_augmentation_processing"
            },
            "1.1.0": {
                "description": "添加文化推荐营养分析字段",
                "sql": """
                ALTER TABLE cultural_recommendations ADD COLUMN nutritional_analysis TEXT DEFAULT '{}';
                ALTER TABLE cultural_recommendations ADD COLUMN served_at TIMESTAMP;
                """,
                "checksum": "add_nutritional_analysis"
            }
        }

    def backup_database(self, backup_path: str):
        """备份数据库"""
        try:
            if "sqlite" in str(engine.url):
                # SQLite备份
                import shutil
                db_path = str(engine.url).replace("sqlite:///", "")
                shutil.copy2(db_path, backup_path)
                logger.info(f"SQLite数据库备份完成: {backup_path}")
            else:
                # PostgreSQL备份需要pg_dump
                logger.warning("PostgreSQL备份需要手动执行pg_dump命令")

            return True
        except Exception as e:
            logger.error(f"数据库备份失败: {e}")
            return False

    def restore_database(self, backup_path: str):
        """恢复数据库"""
        try:
            if "sqlite" in str(engine.url):
                # SQLite恢复
                import shutil
                db_path = str(engine.url).replace("sqlite:///", "")
                shutil.copy2(backup_path, db_path)
                logger.info(f"SQLite数据库恢复完成: {backup_path}")
            else:
                # PostgreSQL恢复需要psql
                logger.warning("PostgreSQL恢复需要手动执行psql命令")

            return True
        except Exception as e:
            logger.error(f"数据库恢复失败: {e}")
            return False

    def get_migration_status(self) -> Dict[str, Any]:
        """获取迁移状态"""
        try:
            applied_migrations = self.get_applied_migrations()
            all_migrations = list(self._get_migration_scripts().keys())

            pending_migrations = [m for m in all_migrations if m not in applied_migrations]

            return {
                "current_version": self.get_database_schema_version(),
                "applied_migrations": applied_migrations,
                "pending_migrations": pending_migrations,
                "total_migrations": len(all_migrations),
                "schema_consistency": self.check_schema_consistency()
            }
        except Exception as e:
            logger.error(f"获取迁移状态失败: {e}")
            return {"error": str(e)}

def run_migrations():
    """运行数据库迁移"""
    session = SessionLocal()
    try:
        migration_manager = MigrationManager(session)

        # 初始化数据库
        if not migration_manager.initialize_database():
            logger.error("数据库初始化失败")
            return False

        # 升级到最新版本
        if not migration_manager.upgrade_database():
            logger.error("数据库升级失败")
            return False

        # 显示迁移状态
        status = migration_manager.get_migration_status()
        logger.info(f"迁移状态: {json.dumps(status, indent=2, ensure_ascii=False)}")

        return True
    finally:
        session.close()

if __name__ == "__main__":
    # 运行迁移
    run_migrations()

__all__ = ["'logger'", "'MigrationManager'", "'run_migrations'"]
