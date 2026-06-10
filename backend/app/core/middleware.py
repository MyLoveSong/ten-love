

"""
统一中间件系统
基于MCP架构的中间件管理和请求处理
"""

import logging
import time
import uuid
import asyncio
from typing import Dict, List, Optional, Any, Union, Callable, Type
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
import json
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager

from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.app.core.dependency_injection import injectable, singleton, get_service
from app.app.core.configuration import get_configuration
from app.core.logging import get_logger, log_function_call, log_performance
try:
    from backend.app.core.performance import get_default_metrics_collector
    _METRICS_AVAILABLE = True
except Exception:
    _METRICS_AVAILABLE = False
from app.app.core.error_handling import get_error_monitor, ErrorContext, handle_error

logger = get_logger("middleware")

class MiddlewarePriority(Enum):
    """中间件优先级"""
    HIGHEST = 1000
    HIGH = 800
    NORMAL = 500
    LOW = 200
    LOWEST = 100

class RequestPhase(Enum):
    """请求阶段"""
    BEFORE_REQUEST = "before_request"
    AFTER_REQUEST = "after_request"
    ON_EXCEPTION = "on_exception"

@dataclass
class RequestContext:
    """请求上下文"""
    request_id: str
    start_time: datetime
    method: str
    url: str
    headers: Dict[str, str]
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    extra_data: Dict[str, Any] = None

@dataclass
class ResponseContext:
    """响应上下文"""
    status_code: int
    response_time: float
    response_size: int
    headers: Dict[str, str]
    extra_data: Dict[str, Any] = None

class IMiddleware(ABC):
    """中间件接口"""

    @abstractmethod
    async def process_request(self, request: Request, context: RequestContext) -> Optional[Response]:
        """处理请求"""
        pass

    @abstractmethod
    async def process_response(self, request: Request, response: Response,
                              request_context: RequestContext, response_context: ResponseContext) -> Response:
        """处理响应"""
        pass

    @abstractmethod
    async def process_exception(self, request: Request, exception: Exception,
                               request_context: RequestContext) -> Optional[Response]:
        """处理异常"""
        pass

@singleton(IMiddleware)
class RequestIdMiddleware(IMiddleware):
    """请求ID中间件"""

    def __init__(self):
        self.priority = MiddlewarePriority.HIGHEST

    async def process_request(self, request: Request, context: RequestContext) -> Optional[Response]:
        """处理请求"""
        # 生成或获取请求ID
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            request_id = str(uuid.uuid4())

        context.request_id = request_id

        # 设置请求ID到请求对象
        request.state.request_id = request_id

        logger.info(f"请求开始: {request_id}", {
            "request_id": request_id,
            "method": context.method,
            "url": context.url
        })

        return None

    async def process_response(self, request: Request, response: Response,
                              request_context: RequestContext, response_context: ResponseContext) -> Response:
        """处理响应"""
        # 添加请求ID到响应头
        response.headers["X-Request-ID"] = request_context.request_id

        logger.info(f"请求完成: {request_context.request_id}", {
            "request_id": request_context.request_id,
            "status_code": response_context.status_code,
            "response_time": response_context.response_time
        })

        return response

    async def process_exception(self, request: Request, exception: Exception,
                              request_context: RequestContext) -> Optional[Response]:
        """处理异常"""
        logger.error(f"请求异常: {request_context.request_id}", {
            "request_id": request_context.request_id,
            "exception": str(exception)
        }, exception=exception)

        return None

@singleton(IMiddleware)
class PerformanceMiddleware(IMiddleware):
    """性能监控中间件"""

    def __init__(self):
        self.priority = MiddlewarePriority.HIGH
        self._performance_data: Dict[str, List[float]] = {}

    async def process_request(self, request: Request, context: RequestContext) -> Optional[Response]:
        """处理请求"""
        # 记录开始时间
        request.state.start_time = time.time()

        return None

    async def process_response(self, request: Request, response: Response,
                              request_context: RequestContext, response_context: ResponseContext) -> Response:
        """处理响应"""
        # 计算响应时间
        if hasattr(request.state, 'start_time'):
            response_time = time.time() - request.state.start_time
            response_context.response_time = response_time

            # 记录性能数据
            endpoint = f"{request_context.method} {request_context.url}"
            if endpoint not in self._performance_data:
                self._performance_data[endpoint] = []

            self._performance_data[endpoint].append(response_time)

            # 保持最近1000个记录
            if len(self._performance_data[endpoint]) > 1000:
                self._performance_data[endpoint] = self._performance_data[endpoint][-1000:]

            # 添加性能头
            response.headers["X-Response-Time"] = str(response_time)

            # 推送到全局指标采集器
            if _METRICS_AVAILABLE:
                try:
                    collector = get_default_metrics_collector()
                    collector.timeit("http_response_time", response_time)
                except Exception:
                    pass

            # 记录慢请求
            if response_time > 1.0:  # 超过1秒
                logger.warning(f"慢请求: {request_context.request_id}", {
                    "request_id": request_context.request_id,
                    "response_time": response_time,
                    "endpoint": endpoint
                })

        return response

    async def process_exception(self, request: Request, exception: Exception,
                              request_context: RequestContext) -> Optional[Response]:
        """处理异常"""
        # 计算异常时的响应时间
        if hasattr(request.state, 'start_time'):
            response_time = time.time() - request.state.start_time

            logger.error(f"请求异常性能: {request_context.request_id}", {
                "request_id": request_context.request_id,
                "response_time": response_time,
                "exception": str(exception)
            })

        return None

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        stats = {}

        for endpoint, times in self._performance_data.items():
            if times:
                stats[endpoint] = {
                    "count": len(times),
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "p95_time": sorted(times)[int(len(times) * 0.95)] if len(times) > 20 else max(times)
                }

        return stats

@singleton(IMiddleware)
class SecurityMiddleware(IMiddleware):
    """安全中间件"""

    def __init__(self):
        self.priority = MiddlewarePriority.HIGH
        self.config = get_configuration()

    async def process_request(self, request: Request, context: RequestContext) -> Optional[Response]:
        """处理请求"""
        # 添加安全头
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin"
        }

        # 检查请求来源
        origin = request.headers.get("Origin")
        if origin:
            # 这里可以添加CORS检查逻辑
            pass

        # 检查请求大小
        content_length = request.headers.get("Content-Length")
        if content_length:
            max_size = await self.config.get_int("max_request_size", 10 * 1024 * 1024)  # 10MB
            if int(content_length) > max_size:
                logger.warning(f"请求过大: {request_context.request_id}", {
                    "request_id": request_context.request_id,
                    "content_length": content_length,
                    "max_size": max_size
                })
                return JSONResponse(
                    status_code=413,
                    content={"error": "Request too large"}
                )

        return None

    async def process_response(self, request: Request, response: Response,
                              request_context: RequestContext, response_context: ResponseContext) -> Response:
        """处理响应"""
        # 添加安全头
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin"
        }

        for header, value in security_headers.items():
            response.headers[header] = value

        return response

    async def process_exception(self, request: Request, exception: Exception,
                              request_context: RequestContext) -> Optional[Response]:
        """处理异常"""
        # 记录安全相关异常
        if isinstance(exception, HTTPException):
            if exception.status_code in [401, 403, 429]:
                logger.warning(f"安全异常: {request_context.request_id}", {
                    "request_id": request_context.request_id,
                    "status_code": exception.status_code,
                    "exception": str(exception)
                })

        return None

@singleton(IMiddleware)
class RateLimitMiddleware(IMiddleware):
    """限流中间件"""

    def __init__(self):
        self.priority = MiddlewarePriority.NORMAL
        self.config = get_configuration()
        self._request_counts: Dict[str, List[float]] = {}
        self._lock = asyncio.Lock()

    async def process_request(self, request: Request, context: RequestContext) -> Optional[Response]:
        """处理请求"""
        # 获取客户端IP
        client_ip = request.client.host if request.client else "unknown"

        # 获取限流配置
        rate_limit_enabled = await self.config.get_bool("rate_limit_enabled", True)
        if not rate_limit_enabled:
            return None

        requests_per_minute = await self.config.get_int("rate_limit_requests_per_minute", 100)

        async with self._lock:
            current_time = time.time()

            # 清理过期记录
            if client_ip in self._request_counts:
                self._request_counts[client_ip] = [
                    req_time for req_time in self._request_counts[client_ip]
                    if current_time - req_time < 60  # 保留最近1分钟的记录
                ]
            else:
                self._request_counts[client_ip] = []

            # 检查限流
            if len(self._request_counts[client_ip]) >= requests_per_minute:
                logger.warning(f"限流触发: {client_ip}", {
                    "client_ip": client_ip,
                    "request_count": len(self._request_counts[client_ip]),
                    "limit": requests_per_minute
                })

                return JSONResponse(
                    status_code=429,
                    content={"error": "Rate limit exceeded"},
                    headers={"Retry-After": "60"}
                )

            # 记录请求
            self._request_counts[client_ip].append(current_time)

        return None

    async def process_response(self, request: Request, response: Response,
                              request_context: RequestContext, response_context: ResponseContext) -> Response:
        """处理响应"""
        return response

    async def process_exception(self, request: Request, exception: Exception,
                              request_context: RequestContext) -> Optional[Response]:
        """处理异常"""
        return None

@singleton(IMiddleware)
class ErrorHandlingMiddleware(IMiddleware):
    """错误处理中间件"""

    def __init__(self):
        self.priority = MiddlewarePriority.LOW
        self.error_monitor = get_error_monitor()

    async def process_request(self, request: Request, context: RequestContext) -> Optional[Response]:
        """处理请求"""
        return None

    async def process_response(self, request: Request, response: Response,
                              request_context: RequestContext, response_context: ResponseContext) -> Response:
        """处理响应"""
        return response

    async def process_exception(self, request: Request, exception: Exception,
                              request_context: RequestContext) -> Optional[Response]:
        """处理异常"""
        # 创建错误上下文
        error_context = ErrorContext(
            request_id=request_context.request_id,
            module=request.url.path,
            function=request.method,
            extra_data={
                "url": str(request.url),
                "headers": dict(request.headers),
                "client_ip": request.client.host if request.client else None
            }
        )

        # 处理错误
        error_record = await self.error_monitor.handle_error(exception, error_context)

        # 返回错误响应
        if isinstance(exception, HTTPException):
            return JSONResponse(
                status_code=exception.status_code,
                content={
                    "error": exception.detail,
                    "error_id": error_record.error_id,
                    "request_id": request_context.request_id
                }
            )
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "error_id": error_record.error_id,
                    "request_id": request_context.request_id
                }
            )

class MiddlewareManager:
    """中间件管理器"""

    def __init__(self):
        self._middlewares: List[IMiddleware] = []
        self._setup_default_middlewares()

    def _setup_default_middlewares(self):
        """设置默认中间件"""
        # 按优先级顺序添加中间件
        middlewares = [
            RequestIdMiddleware(),
            PerformanceMiddleware(),
            SecurityMiddleware(),
            RateLimitMiddleware(),
            ErrorHandlingMiddleware()
        ]

        # 按优先级排序
        middlewares.sort(key=lambda m: m.priority.value, reverse=True)

        self._middlewares.extend(middlewares)

        logger.info(f"已注册 {len(middlewares)} 个中间件")

    def add_middleware(self, middleware: IMiddleware):
        """添加中间件"""
        self._middlewares.append(middleware)
        # 重新排序
        self._middlewares.sort(key=lambda m: m.priority.value, reverse=True)

        logger.info(f"添加中间件: {middleware.__class__.__name__}")

    def remove_middleware(self, middleware_class: Type[IMiddleware]):
        """移除中间件"""
        self._middlewares = [m for m in self._middlewares if not isinstance(m, middleware_class)]

        logger.info(f"移除中间件: {middleware_class.__name__}")

    async def process_request(self, request: Request) -> Optional[Response]:
        """处理请求"""
        # 创建请求上下文
        context = RequestContext(
            request_id="",
            start_time=datetime.now(),
            method=request.method,
            url=str(request.url),
            headers=dict(request.headers)
        )

        # 执行请求处理中间件
        for middleware in self._middlewares:
            try:
                response = await middleware.process_request(request, context)
                if response:
                    return response
            except Exception as e:
                logger.error(f"中间件处理请求失败: {middleware.__class__.__name__}", exception=e)

        return None

    async def process_response(self, request: Request, response: Response) -> Response:
        """处理响应"""
        # 创建响应上下文
        response_context = ResponseContext(
            status_code=response.status_code,
            response_time=0.0,
            response_size=len(response.body) if hasattr(response, 'body') else 0,
            headers=dict(response.headers)
        )

        # 获取请求上下文
        request_context = getattr(request.state, 'request_context', None)
        if not request_context:
            request_context = RequestContext(
                request_id=getattr(request.state, 'request_id', ''),
                start_time=datetime.now(),
                method=request.method,
                url=str(request.url),
                headers=dict(request.headers)
            )

        # 执行响应处理中间件
        for middleware in self._middlewares:
            try:
                response = await middleware.process_response(request, response, request_context, response_context)
            except Exception as e:
                logger.error(f"中间件处理响应失败: {middleware.__class__.__name__}", exception=e)

        return response

    async def process_exception(self, request: Request, exception: Exception) -> Optional[Response]:
        """处理异常"""
        # 获取请求上下文
        request_context = getattr(request.state, 'request_context', None)
        if not request_context:
            request_context = RequestContext(
                request_id=getattr(request.state, 'request_id', ''),
                start_time=datetime.now(),
                method=request.method,
                url=str(request.url),
                headers=dict(request.headers)
            )

        # 执行异常处理中间件
        for middleware in self._middlewares:
            try:
                response = await middleware.process_exception(request, exception, request_context)
                if response:
                    return response
            except Exception as e:
                logger.error(f"中间件处理异常失败: {middleware.__class__.__name__}", exception=e)

        return None

class UnifiedMiddleware(BaseHTTPMiddleware):
    """统一中间件"""

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.middleware_manager = MiddlewareManager()

    async def dispatch(self, request: Request, call_next):
        """分发请求"""
        try:
            # 处理请求
            response = await self.middleware_manager.process_request(request)
            if response:
                return response

            # 调用下一个中间件或应用
            response = await call_next(request)

            # 处理响应
            response = await self.middleware_manager.process_response(request, response)

            return response

        except Exception as e:
            # 处理异常
            response = await self.middleware_manager.process_exception(request, e)
            if response:
                return response

            # 如果没有中间件处理异常，重新抛出
            raise

# 全局中间件管理器
_middleware_manager = MiddlewareManager()

def get_middleware_manager() -> MiddlewareManager:
    """获取中间件管理器"""
    return _middleware_manager

def create_unified_middleware(app: ASGIApp) -> UnifiedMiddleware:
    """创建统一中间件"""
    return UnifiedMiddleware(app)

if __name__ == "__main__":
    # 测试中间件系统
    async def test_middleware():
        # 创建测试请求
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()

        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}

        # 添加中间件
        app.add_middleware(UnifiedMiddleware)

        # 测试客户端
        client = TestClient(app)

        # 发送请求
        response = client.get("/test")
        print(f"状态码: {response.status_code}")
        print(f"响应头: {dict(response.headers)}")
        print(f"响应内容: {response.json()}")

    # 运行测试
    asyncio.run(test_middleware())
__all__ = ["'logger'", "'MiddlewarePriority'", "'RequestPhase'", "'RequestContext'", "'ResponseContext'", "'IMiddleware'", "'RequestIdMiddleware'", "'PerformanceMiddleware'", "'SecurityMiddleware'", "'RateLimitMiddleware'", "'ErrorHandlingMiddleware'", "'MiddlewareManager'", "'UnifiedMiddleware'", "'get_middleware_manager'", "'create_unified_middleware'"]
