"""scheduling模块\n\n模块描述\n"""
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class ResourceSpec:
    cpu: int = 2
    memory_gb: int = 4
    gpu: int = 0
    gpu_type: Optional[str] = None  # e.g., "nvidia-tesla-t4"
    priority: str = "normal"        # low/normal/high

def select_engine_by_resources(spec: ResourceSpec, policy: Optional[Dict[str, Any]] = None) -> str:
    """根据资源规格选择执行引擎（简单映射占位）"""
    # 策略优先
    if policy:
        # 例如: {"gpu": "spark", "cpu>=16": "dask"}
        try:
            if spec.gpu and policy.get("gpu"):
                return str(policy["gpu"])
            if spec.cpu >= int(policy.get("cpu_ge", 1)) and policy.get("cpu_engine"):
                return str(policy["cpu_engine"])
        except Exception:
            pass
    if spec.gpu and spec.gpu > 0:
        return "gpu"
    if spec.cpu >= 8 or spec.memory_gb >= 16:
        return "dask"
    return "local"

def to_engine_params(spec: ResourceSpec) -> Dict[str, Any]:
    """将资源规格映射为引擎参数"""
    return {
        "n_workers": max(1, spec.cpu // 2),
        "memory_limit": f"{spec.memory_gb}GB",
        "gpu": spec.gpu,
        "priority": spec.priority,
        "gpu_type": spec.gpu_type,
    }

__all__ = ["'ResourceSpec'", "'select_engine_by_resources'", "'to_engine_params'"]
