"""
配置管理模块
"""

import os
import sys
from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
import logging

"""
核心配置模块
企业级配置管理
"""

import os
logger = logging.getLogger(__name__)
from pydantic_settings import BaseSettings
from pydantic import validator, field_validator
from pathlib import Path

class Settings(BaseSettings):
    """应用配置"""
    # 允许额外环境变量（如 NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD）不报错
    model_config = ConfigDict(extra='allow')

    # 基础配置
    APP_NAME: str = "增强学术级智能健康监测集成系统"
    APP_VERSION: str = "3.0.0"
    DEBUG: bool = False

    # 服务器配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # 数据库配置
    DATABASE_URL: str = "sqlite:///./academic_system_dev.db"
    DATABASE_ECHO: bool = False

    # Redis配置
    REDIS_URL: str = "redis://localhost:6379"

    # 安全配置
    SECRET_KEY: str = "your-secret-key-here-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ALGORITHM: str = "HS256"

    # CORS配置
    ALLOWED_HOSTS: List[str] = ["*"]
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000"
    ]

    # AI模型配置
    MODEL_CACHE_DIR: str = "./models/cache"
    MODEL_DEVICE: str = "cpu"  # cpu, cuda, mps

    # 文件上传配置
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    UPLOAD_DIR: str = "./uploads"

    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/app.log"

    # API配置
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Academic Health Monitoring System"

    # 监控配置
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090

    # 缓存配置
    CACHE_ENABLED: bool = True
    CACHE_TTL_DEFAULT: int = 3600
    CACHE_MAX_SIZE: int = 1000

    # 限流配置
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60

    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/app.log"
    LOG_FORMAT: str = "json"  # json, text

    # 安全配置
    SECURITY_HEADERS_ENABLED: bool = True
    CORS_ENABLED: bool = True

    # 性能配置
    PERFORMANCE_MONITORING: bool = True
    SLOW_QUERY_THRESHOLD: float = 1.0  # 秒

    @field_validator("ALLOWED_HOSTS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v):
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    # 将Config类配置迁移到model_config
    model_config = ConfigDict(
        extra='allow',
        env_file='.env',
        case_sensitive=True
    )

# 创建全局设置实例
settings = Settings()

# 确保必要的目录存在
def ensure_directories():
    """确保必要的目录存在"""
    directories = [
        settings.MODEL_CACHE_DIR,
        settings.UPLOAD_DIR,
        Path(settings.LOG_FILE).parent,
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

# 初始化时创建目录
ensure_directories()

__all__ = ["'Settings'", "'settings'", "'ensure_directories'"]
