"""observability模块\n\n模块描述\n"""
"""
可观测性模块
"""

from typing import Dict, Any, List

from backend.app.core.performance import get_default_metrics_collector
from backend.app.core.database import get_connection_pool_status

def export_runtime_metrics() -> Dict[str, Any]:
    collector = get_default_metrics_collector()
    return {
        "http": collector.summary(),
        "db_pool": get_connection_pool_status(),
    }

def evaluate_alerts() -> Dict[str, Any]:
    """基于简单阈值的告警评估（可后续外置到配置中心/MCP）"""
    metrics = export_runtime_metrics()
    alerts: List[Dict[str, Any]] = []

    # HTTP 平均响应时间阈值
    http_summary = metrics.get("http", {})
    avg_over_1s = []
    for name, stat in http_summary.items():
        if stat.get("avg", 0) > 1.0:
            avg_over_1s.append({"metric": name, "avg": stat.get("avg", 0)})
    if avg_over_1s:
        alerts.append({
            "type": "http_latency",
            "level": "warning",
            "details": avg_over_1s
        })

    # 数据库连接池使用率阈值
    pool = metrics.get("db_pool", {})
    size = pool.get("pool_size", 0) or 0
    checked_out = pool.get("checked_out", 0) or 0
    if size and (checked_out / size) > 0.8:
        alerts.append({
            "type": "db_pool_pressure",
            "level": "warning",
            "details": {"checked_out": checked_out, "pool_size": size}
        })

    return {"alerts": alerts, "metrics": metrics}

__all__ = ["'export_runtime_metrics'", "'evaluate_alerts'"]
