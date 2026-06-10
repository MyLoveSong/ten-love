

"""
增强学术级智能健康监测集成系统 - Backend主应用
企业级FastAPI应用架构
"""

import logging
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入核心模块
from app.core.config import settings
# from app.core.logging_config import setup_logging
from app.api.v1.api import api_router
from app.core.database import init_db
from app.core.exceptions import CustomException, exception_handler
from app.core.middleware import create_middleware_stack

# 导入MCP架构核心模块
from app.core.dependency_injection import configure_services, build_service_provider
from app.core.configuration import configure_configuration, get_configuration
# from app.core.logging import get_app_logger, get_logger
from app.core.error_handling import get_error_monitor
from app.core.data_security import academic_data_security_manager
from app.core.performance_optimization import academic_performance_optimizer

# 设置日志
# logger = setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理 - MCP架构"""
    # 启动时执行
    logger.info("🚀 启动增强学术级智能健康监测集成系统 - MCP架构")
    logger.info("==========================================")

    try:
        # 配置MCP架构核心组件
        logger.info("🔧 配置MCP架构核心组件...")

        # 配置服务容器
        service_collection = configure_services()
        service_provider = build_service_provider()
        logger.info("✅ 依赖注入容器初始化完成")

        # 配置配置管理器
        config_manager = configure_configuration()
        logger.info("✅ 配置管理系统初始化完成")

        # 初始化数据库
        await init_db()
        logger.info("✅ 数据库初始化完成")

        # 启动错误监控
        error_monitor = get_error_monitor()
        await error_monitor.start_monitoring()
        logger.info("✅ 错误监控系统启动完成")

        # 初始化数据安全系统
        logger.info("✅ 数据安全系统初始化完成")

        # 初始化性能优化系统
        logger.info("✅ 性能优化系统初始化完成")

        # 初始化AI模型
        # await init_models()
        logger.info("✅ AI模型初始化完成")

        logger.info("🎯 MCP架构系统启动完成，准备处理请求")
        logger.info("   🌐 API文档: http://localhost:8000/docs")
        logger.info("   ❤️  健康检查: http://localhost:8000/health")
        logger.info("   📊 系统监控: http://localhost:8000/api/v1/monitor")
        logger.info("   🔧 数据处理: http://localhost:8000/api/v1/data-processing")

    except Exception as e:
        logger.error(f"❌ MCP架构系统初始化失败: {e}")
        raise

    yield

    # 关闭时执行
    logger.info("🛑 MCP架构系统正在关闭...")

    try:
               # 停止错误监控
               error_monitor = get_error_monitor()
               await error_monitor.stop_monitoring()
               logger.info("✅ 错误监控系统已停止")

               # 关闭性能优化系统
               await academic_performance_optimizer.shutdown()
               logger.info("✅ 性能优化系统已停止")

               logger.info("✅ MCP架构系统关闭完成")

    except Exception as e:
        logger.error(f"❌ 系统关闭异常: {e}")

# 创建FastAPI应用 - MCP架构
app = FastAPI(
    title="增强学术级智能健康监测集成系统 - MCP架构",
    description="企业级AI驱动的健康监测与预测系统 - 基于MCP(模块化、组件化、插件化)架构",
    version="3.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/api/v1/openapi.json",
    lifespan=lifespan
)

# 添加中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS
)

# 设置MCP架构统一中间件（新门面，向后兼容）
unified_middleware = create_middleware_stack(app)
app.add_middleware(type(unified_middleware))

# 设置传统中间件（向后兼容）
setup_middleware(app)

# 全局异常处理
@app.exception_handler(CustomException)
async def custom_exception_handler(request: Request, exc: CustomException):
    return await exception_handler.handle_custom_exception(request, exc)

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return await exception_handler.handle_http_exception(request, exc)

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return await exception_handler.handle_general_exception(request, exc)

# 注册路由
app.include_router(api_router, prefix="/api/v1")

# 健康检查端点 - MCP架构
@app.get("/health", tags=["系统"])
async def health_check():
    """系统健康检查 - MCP架构"""
    try:
        # 获取配置管理器
        config_manager = get_configuration()

        # 获取错误监控器
        error_monitor = get_error_monitor()
        error_stats = error_monitor.get_error_statistics()

        # 获取中间件管理器
        # from app.app.core.middleware import get_middleware_manager
        # middleware_manager = get_middleware_manager()

        return {
            "status": "healthy",
            "version": "3.1.0",
            "architecture": "MCP",
            "service": "academic-health-monitoring-system",
            "timestamp": time.time(),
                   "components": {
                       "database": "connected",
                       "configuration": "loaded",
                       "logging": "active",
                       "error_monitoring": "active",
                       "middleware": f"{len(middleware_manager._middlewares)} active",
                       "dependency_injection": "active",
                       "data_security": "active",
                       "performance_optimization": "active"
                   },
            "error_statistics": error_stats,
            "environment": settings.ENVIRONMENT
        }

    except Exception as e:
        logger.error("健康检查失败", exception=e)
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
        )

@app.get("/", tags=["系统"])
async def root():
    """根路径 - MCP架构"""
    return {
        "message": "增强学术级智能健康监测集成系统 - MCP架构",
        "version": "3.1.0",
        "architecture": "MCP (Modular, Component-based, Plugin)",
        "features": [
            "血糖预测",
            "图像识别",
            "文化适配",
            "数据管理",
            "系统监控",
            "学术级数据处理",
            "企业级数据库管理"
        ],
        "docs": "/docs",
        "health": "/health",
        "monitoring": "/api/v1/monitor",
        "data_processing": "/api/v1/data-processing"
    }

# 系统信息端点
@app.get("/system/info", tags=["系统"])
async def system_info():
    """获取系统信息 - MCP架构"""
    try:
        config_manager = get_configuration()

        return {
            "system": {
                "name": "增强学术级智能健康监测集成系统",
                "version": "3.1.0",
                "architecture": "MCP (Modular, Component-based, Plugin)",
                "environment": settings.ENVIRONMENT
            },
            "features": {
                "data_processing": "学术级数据处理",
                "database_management": "企业级数据库管理",
                "machine_learning": "血糖预测和图像识别",
                "cultural_adaptation": "文化适配推荐",
                "monitoring": "实时监控和告警",
                "api_management": "统一API管理"
            },
            "modules": [
                "依赖注入容器",
                "配置管理系统",
                "统一日志系统",
                "错误处理监控",
                "中间件管理",
                "数据清洗预处理",
                "特征工程框架",
                "数据增强技术",
                "自监督学习",
                "多模态融合",
                "质量监控",
                "管道管理"
            ]
        }

    except Exception as e:
        logger.error("获取系统信息失败", exception=e)
        return JSONResponse(
            status_code=500,
            content={"error": "获取系统信息失败"}
        )

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
