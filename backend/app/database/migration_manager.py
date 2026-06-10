"""
数据库迁移管理器
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import asyncio
import json

"""
数据库迁移管理
使用Alembic进行数据库版本控制
"""

import os
import logging
logger = logging.getLogger(__name__)
from typing import Optional, List, Dict, Any

try:
    from alembic import command
    from alembic.config import Config
    from alembic.script import ScriptDirectory
    from alembic.runtime.environment import EnvironmentContext
    from alembic.runtime.migration import MigrationContext
    ALEMBIC_AVAILABLE = True
except ImportError:
    ALEMBIC_AVAILABLE = False
    command = None
    Config = None

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from backend.app.core.config import settings
from backend.app.core.database import engine, SessionLocal

logger = logging.getLogger(__name__)

class DatabaseMigrationManager:
    """数据库迁移管理器"""

    def __init__(self):
        self.alembic_cfg = None
        self.script_dir = None

        if ALEMBIC_AVAILABLE:
            self._init_alembic()
        else:
            logger.warning("Alembic未安装，使用简单迁移管理")

    def _init_alembic(self):
        """初始化Alembic配置"""
        try:
            # 设置Alembic配置路径
            alembic_dir = Path(__file__).parent.parent / "alembic"
            alembic_cfg_path = alembic_dir / "alembic.ini"

            if alembic_cfg_path.exists():
                self.alembic_cfg = Config(str(alembic_cfg_path))
                self.alembic_cfg.set_main_option("sqlalchemy.url", settings.DATABASE_URL)
                self.script_dir = ScriptDirectory.from_config(self.alembic_cfg)
                logger.info("Alembic配置初始化成功")
            else:
                logger.warning("Alembic配置文件不存在，将创建")
                self._create_alembic_config()
        except Exception as e:
            logger.error(f"Alembic初始化失败: {e}")
            self.alembic_cfg = None

    def _create_alembic_config(self):
        """创建Alembic配置"""
        try:
            # 创建alembic目录
            alembic_dir = Path(__file__).parent.parent / "alembic"
            alembic_dir.mkdir(exist_ok=True)

            # 创建alembic.ini文件
            alembic_cfg_path = alembic_dir / "alembic.ini"
            alembic_cfg_content = f"""[alembic]
script_location = alembic
prepend_sys_path = .
version_path_separator = os
sqlalchemy.url = {settings.DATABASE_URL}

[post_write_hooks]
hooks = black
black.type = console_scripts
black.entrypoint = black
black.options = -l 79 REVISION_SCRIPT_FILENAME

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
"""

            with open(alembic_cfg_path, 'w', encoding='utf-8') as f:
                f.write(alembic_cfg_content)

            # 创建versions目录
            versions_dir = alembic_dir / "versions"
            versions_dir.mkdir(exist_ok=True)

            # 创建env.py文件
            env_py_content = '''"""Alembic环境配置"""
from app.logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from app.alembic import context
import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.database.models import Base
from backend.app.core.config import settings

# Alembic配置对象
config = context.config

# 设置SQLAlchemy URL
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

# 配置日志
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 目标元数据
target_metadata = Base.metadata

def run_migrations_offline():
    """离线模式运行迁移"""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    """在线模式运行迁移"""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
'''

            env_py_path = alembic_dir / "env.py"
            with open(env_py_path, 'w', encoding='utf-8') as f:
                f.write(env_py_content)

            # 重新初始化
            self.alembic_cfg = Config(str(alembic_cfg_path))
            self.alembic_cfg.set_main_option("sqlalchemy.url", settings.DATABASE_URL)
            self.script_dir = ScriptDirectory.from_config(self.alembic_cfg)

            logger.info("Alembic配置创建成功")

        except Exception as e:
            logger.error(f"创建Alembic配置失败: {e}")

    def get_current_revision(self) -> Optional[str]:
        """获取当前数据库版本"""
        if not self.alembic_cfg:
            return self._get_simple_version()

        try:
            with engine.connect() as connection:
                context = MigrationContext.configure(connection)
                return context.get_current_revision()
        except Exception as e:
            logger.error(f"获取当前版本失败: {e}")
            return None

    def get_head_revision(self) -> Optional[str]:
        """获取最新版本"""
        if not self.script_dir:
            return None

        try:
            return self.script_dir.get_current_head()
        except Exception as e:
            logger.error(f"获取最新版本失败: {e}")
            return None

    def get_migration_history(self) -> List[Dict[str, Any]]:
        """获取迁移历史"""
        if not self.script_dir:
            return []

        try:
            revisions = []
            for revision in self.script_dir.walk_revisions():
                revisions.append({
                    "revision": revision.revision,
                    "down_revision": revision.down_revision,
                    "branch_labels": revision.branch_labels,
                    "depends_on": revision.depends_on,
                    "comment": revision.comment,
                    "doc": revision.doc
                })
            return revisions
        except Exception as e:
            logger.error(f"获取迁移历史失败: {e}")
            return []

    def create_migration(self, message: str) -> bool:
        """创建新的迁移"""
        if not self.alembic_cfg:
            logger.warning("Alembic未配置，无法创建迁移")
            return False

        try:
            command.revision(self.alembic_cfg, message=message, autogenerate=True)
            logger.info(f"迁移创建成功: {message}")
            return True
        except Exception as e:
            logger.error(f"创建迁移失败: {e}")
            return False

    def upgrade_database(self, revision: str = "head") -> bool:
        """升级数据库"""
        if not self.alembic_cfg:
            logger.warning("Alembic未配置，无法升级数据库")
            return False

        try:
            command.upgrade(self.alembic_cfg, revision)
            logger.info(f"数据库升级成功到版本: {revision}")
            return True
        except Exception as e:
            logger.error(f"数据库升级失败: {e}")
            return False

    def downgrade_database(self, revision: str) -> bool:
        """降级数据库"""
        if not self.alembic_cfg:
            logger.warning("Alembic未配置，无法降级数据库")
            return False

        try:
            command.downgrade(self.alembic_cfg, revision)
            logger.info(f"数据库降级成功到版本: {revision}")
            return True
        except Exception as e:
            logger.error(f"数据库降级失败: {e}")
            return False

    def _get_simple_version(self) -> Optional[str]:
        """获取简单版本信息"""
        try:
            with engine.connect() as connection:
                result = connection.execute(text("SELECT version FROM schema_migrations ORDER BY id DESC LIMIT 1"))
                row = result.fetchone()
                return row[0] if row else None
        except Exception:
            return None

    def init_database(self) -> bool:
        """初始化数据库"""
        try:
            # 创建所有表
            from app.database.models import Base
            Base.metadata.create_all(bind=engine)

            # 创建版本表
            with engine.connect() as connection:
                connection.execute(text("""
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        version VARCHAR(50) NOT NULL UNIQUE,
                        description TEXT,
                        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                connection.commit()

            logger.info("数据库初始化成功")
            return True
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            return False

    def get_database_status(self) -> Dict[str, Any]:
        """获取数据库状态"""
        status = {
            "alembic_available": ALEMBIC_AVAILABLE,
            "alembic_configured": self.alembic_cfg is not None,
            "current_revision": self.get_current_revision(),
            "head_revision": self.get_head_revision(),
            "migration_count": len(self.get_migration_history())
        }

        # 检查是否需要迁移
        if status["current_revision"] and status["head_revision"]:
            status["needs_migration"] = status["current_revision"] != status["head_revision"]
        else:
            status["needs_migration"] = False

        return status

    def backup_database(self, backup_path: str) -> bool:
        """备份数据库"""
        try:
            if settings.DATABASE_URL.startswith("sqlite"):
                import shutil
                db_path = settings.DATABASE_URL.replace("sqlite:///", "")
                shutil.copy2(db_path, backup_path)
                logger.info(f"数据库备份成功: {backup_path}")
                return True
            else:
                logger.warning("非SQLite数据库，备份功能待实现")
                return False
        except Exception as e:
            logger.error(f"数据库备份失败: {e}")
            return False

    def restore_database(self, backup_path: str) -> bool:
        """恢复数据库"""
        try:
            if settings.DATABASE_URL.startswith("sqlite"):
                import shutil
                db_path = settings.DATABASE_URL.replace("sqlite:///", "")
                shutil.copy2(backup_path, db_path)
                logger.info(f"数据库恢复成功: {backup_path}")
                return True
            else:
                logger.warning("非SQLite数据库，恢复功能待实现")
                return False
        except Exception as e:
            logger.error(f"数据库恢复失败: {e}")
            return False

# 全局迁移管理器实例
migration_manager = DatabaseMigrationManager()

# 迁移命令接口
def create_migration(message: str) -> bool:
    """创建迁移"""
    return migration_manager.create_migration(message)

def upgrade_database(revision: str = "head") -> bool:
    """升级数据库"""
    return migration_manager.upgrade_database(revision)

def downgrade_database(revision: str) -> bool:
    """降级数据库"""
    return migration_manager.downgrade_database(revision)

def get_database_status() -> Dict[str, Any]:
    """获取数据库状态"""
    return migration_manager.get_database_status()

def init_database() -> bool:
    """初始化数据库"""
    return migration_manager.init_database()

if __name__ == "__main__":
    # 命令行工具
    import sys

    if len(sys.argv) < 2:
        print("用法: python migration.py <command> [args]")
        print("命令:")
        print("  init - 初始化数据库")
        print("  status - 查看数据库状态")
        print("  create <message> - 创建迁移")
        print("  upgrade [revision] - 升级数据库")
        print("  downgrade <revision> - 降级数据库")
        sys.exit(1)

    command = sys.argv[1]

    if command == "init":
        success = init_database()
        print("数据库初始化成功" if success else "数据库初始化失败")

    elif command == "status":
        status = get_database_status()
        print("数据库状态:")
        for key, value in status.items():
            print(f"  {key}: {value}")

    elif command == "create":
        if len(sys.argv) < 3:
            print("请提供迁移消息")
            sys.exit(1)
        message = sys.argv[2]
        success = create_migration(message)
        print(f"迁移创建{'成功' if success else '失败'}")

    elif command == "upgrade":
        revision = sys.argv[2] if len(sys.argv) > 2 else "head"
        success = upgrade_database(revision)
        print(f"数据库升级{'成功' if success else '失败'}")

    elif command == "downgrade":
        if len(sys.argv) < 3:
            print("请提供目标版本")
            sys.exit(1)
        revision = sys.argv[2]
        success = downgrade_database(revision)
        print(f"数据库降级{'成功' if success else '失败'}")

    else:
        print(f"未知命令: {command}")
        sys.exit(1)

__all__ = ["'logger'", "'DatabaseMigrationManager'", "'migration_manager'", "'create_migration'", "'upgrade_database'", "'downgrade_database'", "'get_database_status'", "'init_database'"]
