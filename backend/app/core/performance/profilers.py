class CpuProfiler:
    """CPU 采样占位"""

    def start(self):
        return self

    def stop(self):
        return {}

class IoProfiler:
    """IO 监控占位"""

    def start(self):
        return self

    def stop(self):
        return {}

class SqlProfiler:
    """SQL 监控占位（可结合 SQLAlchemy 事件）"""

    def start(self):
        return self

    def stop(self):
        return {}

class MemoryProfiler:
    """内存剖析占位"""

    def start(self):
        return self

    def stop(self):
        return {}

__all__ = ["'CpuProfiler'", "'IoProfiler'", "'SqlProfiler'", "'MemoryProfiler'"]
