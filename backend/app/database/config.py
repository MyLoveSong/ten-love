

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库配置管理
支持开发环境(SQLite)和生产环境(PostgreSQL)的配置切换
"""

import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """数据库配置类"""
    # 数据库类型
    db_type: str = "sqlite"  # sqlite, postgresql, mysql

    # 连接参数
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None
    database: str = "academic_system"

    # SQLite特定配置
    sqlite_path: str = "academic_system.db"

    # 连接池配置
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 300

    # 其他配置
    echo: bool = False
    echo_pool: bool = False
    pool_pre_ping: bool = True

    def get_database_url(self) -> str:
        """获取数据库连接URL"""
        if self.db_type == "sqlite":
            return f"sqlite:///{self.sqlite_path}"
        elif self.db_type == "postgresql":
            return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        elif self.db_type == "mysql":
            return f"mysql+pymysql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        else:
            raise ValueError(f"不支持的数据库类型: {self.db_type}")

    def get_engine_kwargs(self) -> Dict[str, Any]:
        """获取SQLAlchemy引擎参数"""
        kwargs = {
            "echo": self.echo,
            "echo_pool": self.echo_pool,
            "pool_pre_ping": self.pool_pre_ping,
            "pool_recycle": self.pool_recycle,
        }

        if self.db_type != "sqlite":
            kwargs.update({
                "pool_size": self.pool_size,
                "max_overflow": self.max_overflow,
                "pool_timeout": self.pool_timeout,
            })
        else:
            # SQLite特定配置
            kwargs["connect_args"] = {"check_same_thread": False}

        return kwargs

class DatabaseConfigManager:
    """数据库配置管理器"""

    def __init__(self):
        self.configs: Dict[str, DatabaseConfig] = {}
        self._load_configs()

    def _load_configs(self):
        """加载配置"""
        # 开发环境配置 (SQLite)
        self.configs["development"] = DatabaseConfig(
            db_type="sqlite",
            sqlite_path="academic_system_dev.db",
            echo=True  # 开发环境显示SQL
        )

        # 测试环境配置 (SQLite)
        self.configs["testing"] = DatabaseConfig(
            db_type="sqlite",
            sqlite_path="academic_system_test.db",
            echo=False
        )

        # 生产环境配置 (PostgreSQL)
        self.configs["production"] = DatabaseConfig(
            db_type="postgresql",
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            username=os.getenv("DB_USERNAME", "academic_user"),
            password=os.getenv("DB_PASSWORD", "academic_password"),
            database=os.getenv("DB_NAME", "academic_system"),
            pool_size=10,
            max_overflow=20,
            echo=False
        )

        # 从环境变量加载当前环境配置
        self._load_from_env()

    def _load_from_env(self):
        """从环境变量加载配置"""
        env = os.getenv("ENVIRONMENT", "development")

        if env in self.configs:
            config = self.configs[env]

            # 覆盖环境变量中的配置
            if os.getenv("DB_TYPE"):
                config.db_type = os.getenv("DB_TYPE")

            if os.getenv("DB_HOST"):
                config.host = os.getenv("DB_HOST")

            if os.getenv("DB_PORT"):
                config.port = int(os.getenv("DB_PORT"))

            if os.getenv("DB_USERNAME"):
                config.username = os.getenv("DB_USERNAME")

            if os.getenv("DB_PASSWORD"):
                config.password = os.getenv("DB_PASSWORD")

            if os.getenv("DB_NAME"):
                config.database = os.getenv("DB_NAME")

            if os.getenv("DB_SQLITE_PATH"):
                config.sqlite_path = os.getenv("DB_SQLITE_PATH")

            if os.getenv("DB_ECHO"):
                config.echo = os.getenv("DB_ECHO").lower() == "true"

            logger.info(f"加载环境配置: {env}")

    def get_config(self, environment: str = None) -> DatabaseConfig:
        """获取指定环境的配置"""
        if environment is None:
            environment = os.getenv("ENVIRONMENT", "development")

        if environment not in self.configs:
            logger.warning(f"环境 {environment} 不存在，使用开发环境配置")
            environment = "development"

        return self.configs[environment]

    def get_current_config(self) -> DatabaseConfig:
        """获取当前环境配置"""
        return self.get_config()

    def add_custom_config(self, name: str, config: DatabaseConfig):
        """添加自定义配置"""
        self.configs[name] = config
        logger.info(f"添加自定义配置: {name}")

    def list_configs(self) -> Dict[str, str]:
        """列出所有可用配置"""
        return {name: config.db_type for name, config in self.configs.items()}

    def validate_config(self, config: DatabaseConfig) -> bool:
        """验证配置有效性"""
        try:
            # 检查数据库类型
            if config.db_type not in ["sqlite", "postgresql", "mysql"]:
                logger.error(f"不支持的数据库类型: {config.db_type}")
                return False

            # 检查必要参数
            if config.db_type == "sqlite":
                if not config.sqlite_path:
                    logger.error("SQLite路径不能为空")
                    return False
            else:
                if not all([config.host, config.port, config.username, config.password, config.database]):
                    logger.error(f"{config.db_type} 数据库缺少必要参数")
                    return False

            # 检查端口范围
            if config.port and (config.port < 1 or config.port > 65535):
                logger.error(f"端口号无效: {config.port}")
                return False

            logger.info(f"配置验证通过: {config.db_type}")
            return True

        except Exception as e:
            logger.error(f"配置验证失败: {e}")
            return False

    def get_connection_info(self, config: DatabaseConfig) -> Dict[str, Any]:
        """获取连接信息（隐藏敏感信息）"""
        info = {
            "db_type": config.db_type,
            "database": config.database,
            "echo": config.echo,
            "pool_size": config.pool_size,
        }

        if config.db_type == "sqlite":
            info["sqlite_path"] = config.sqlite_path
        else:
            info.update({
                "host": config.host,
                "port": config.port,
                "username": config.username,
                "password": "***" if config.password else None,
            })

        return info

# 全局配置管理器实例
config_manager = DatabaseConfigManager()

def get_database_config(environment: str = None) -> DatabaseConfig:
    """获取数据库配置的便捷函数"""
    return config_manager.get_config(environment)

def get_current_database_config() -> DatabaseConfig:
    """获取当前数据库配置的便捷函数"""
    return config_manager.get_current_config()

def validate_database_config(config: DatabaseConfig) -> bool:
    """验证数据库配置的便捷函数"""
    return config_manager.validate_config(config)

if __name__ == "__main__":
    # 测试配置管理器
    print("可用配置:")
    for name, db_type in config_manager.list_configs().items():
        print(f"  {name}: {db_type}")

    print("\n当前配置:")
    current_config = config_manager.get_current_config()
    print(f"  数据库类型: {current_config.db_type}")
    print(f"  连接URL: {current_config.get_database_url()}")

    print("\n连接信息:")
    conn_info = config_manager.get_connection_info(current_config)
    for key, value in conn_info.items():
        print(f"  {key}: {value}")

    print(f"\n配置验证: {'通过' if config_manager.validate_config(current_config) else '失败'}")

__all__ = ["'logger'", "'DatabaseConfig'", "'DatabaseConfigManager'", "'config_manager'", "'get_database_config'", "'get_current_database_config'", "'validate_database_config'"]
