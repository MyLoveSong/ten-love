"""
性能建议模块
"""

from typing import Dict, Any

class OptimizationAdvisor:
    """基于指标的简单建议占位"""

    def advise(self, metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        tips = []
        for name, stat in metrics.items():
            if stat.get("avg", 0) > 1.0:
                tips.append({
                    "metric": name,
                    "advice": "平均耗时较高，考虑批处理/缓存/并行化"
                })
        return {"tips": tips}

__all__ = ["'OptimizationAdvisor'"]
