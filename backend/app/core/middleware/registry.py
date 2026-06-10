from typing import Any

def create_middleware_stack(app: Any):
    """门面：创建统一中间件栈
    当前委托到 legacy 的 create_unified_middleware 以保持兼容，后续可逐步拆分到本子包
    """
    # 延迟导入避免循环依赖
    from app.app.core.middleware import create_unified_middleware

    unified = create_unified_middleware(app)
    return unified

__all__ = ["'create_middleware_stack'"]
