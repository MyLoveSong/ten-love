"""tracing模块\n\n模块描述\n"""
from typing import Callable
import time

def request_timer_middleware(get_response: Callable):
    """占位：请求计时中间件（未来迁移到此）"""
    async def middleware(request):
        start = time.time()
        response = await get_response(request)
        duration = time.time() - start
        # 这里未来接入统一日志
        return response

    return middleware

__all__ = ["'request_timer_middleware'"]
