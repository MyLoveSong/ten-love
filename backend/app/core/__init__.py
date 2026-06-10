"""
核心模块统一导出
遵循ESM标准，消除重复导入
"""

# MCP注册中心
from .mcp_registry import (
    mcp_registry,
    register_module,
    register_service,
    inject,
    MCPModule,
    ModuleMetadata
)

# ESM转换器
from .esm_converter import (
    ESMConverter,
    ImportOptimizer
)

# 依赖分析器
from .dependency_analyzer import (
    DependencyAnalyzer,
    ModuleOptimizer
)

# 现有核心模块
try:
    from .configuration import get_configuration
except ImportError:
    get_configuration = None

try:
    from .exceptions import CustomException
except ImportError:
    CustomException = None

try:
    from .structured_logging import get_logger
except ImportError:
    get_logger = None

try:
    from .middleware import APIResponse
except ImportError:
    APIResponse = None

try:
    from .database import check_db_health, get_connection_pool_status
except ImportError:
    check_db_health = None
    get_connection_pool_status = None

try:
    from .cache import cache_manager
except ImportError:
    cache_manager = None

try:
    from .observability import export_runtime_metrics, evaluate_alerts
except ImportError:
    export_runtime_metrics = None
    evaluate_alerts = None

# 统一导出
__all__ = [
    # MCP相关
    'mcp_registry',
    'register_module',
    'register_service',
    'inject',
    'MCPModule',
    'ModuleMetadata',

    # ESM相关
    'ESMConverter',
    'ImportOptimizer',

    # 依赖分析
    'DependencyAnalyzer',
    'ModuleOptimizer',

    # 现有核心功能
    'get_configuration',
    'CustomException',
    'get_logger',
    'APIResponse',
    'check_db_health',
    'get_connection_pool_status',
    'cache_manager',
    'export_runtime_metrics',
    'evaluate_alerts'
]
