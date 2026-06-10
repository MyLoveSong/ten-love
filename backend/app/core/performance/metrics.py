"""metrics模块\n\n模块描述\n"""
"""
性能指标模块
"""

from typing import Dict, List
import time

class PerformanceMetricsCollector:
    """轻量性能指标采集占位（请求/任务级别）"""

    def __init__(self):
        self._counters: Dict[str, int] = {}
        self._timings: Dict[str, List[float]] = {}

    def incr(self, name: str, value: int = 1) -> None:
        self._counters[name] = self._counters.get(name, 0) + value

    def timeit(self, name: str, duration: float) -> None:
        arr = self._timings.setdefault(name, [])
        arr.append(duration)
        if len(arr) > 1000:
            self._timings[name] = arr[-1000:]

    def summary(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for k, arr in self._timings.items():
            if arr:
                out[k] = {
                    "count": float(len(arr)),
                    "avg": sum(arr) / len(arr),
                    "min": min(arr),
                    "max": max(arr),
                }
        return out

_default_collector = PerformanceMetricsCollector()

def get_default_metrics_collector() -> PerformanceMetricsCollector:
    return _default_collector

__all__ = ["'PerformanceMetricsCollector'", "'get_default_metrics_collector'"]
