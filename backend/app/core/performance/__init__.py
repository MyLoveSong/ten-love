"""性能子包门面
逐步从 legacy `app.core.performance_optimization` 迁移到此处的细分模块
当前提供占位与轻量适配，避免破坏对外契约
"""

from .metrics import (
    PerformanceMetricsCollector,
    get_default_metrics_collector,
)
from .profilers import (
    CpuProfiler,
    IoProfiler,
    SqlProfiler,
    MemoryProfiler,
)
from .optimizers import (
    BatchOptimizer,
    CacheOptimizer,
)
from .advice import (
    OptimizationAdvisor,
)

__all__ = [
    "PerformanceMetricsCollector",
    "get_default_metrics_collector",
    "CpuProfiler",
    "IoProfiler",
    "SqlProfiler",
    "MemoryProfiler",
    "BatchOptimizer",
    "CacheOptimizer",
    "OptimizationAdvisor",
]
