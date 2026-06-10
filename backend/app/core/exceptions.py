"""
异常处理模块
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime

"""
异常处理模块
企业级异常管理
"""

logger = logging.getLogger(__name__)
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
import logging
import traceback
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class ErrorCode(Enum):
    """错误代码枚举"""
    # 通用错误
    INTERNAL_ERROR = "INTERNAL_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    CONFLICT = "CONFLICT"

    # 业务错误
    MODEL_ERROR = "MODEL_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    CACHE_ERROR = "CACHE_ERROR"
    EXTERNAL_API_ERROR = "EXTERNAL_API_ERROR"

    # 数据错误
    DATA_NOT_FOUND = "DATA_NOT_FOUND"
    DATA_VALIDATION_FAILED = "DATA_VALIDATION_FAILED"
    DATA_PROCESSING_ERROR = "DATA_PROCESSING_ERROR"

    # 用户相关错误
    USER_NOT_FOUND = "USER_NOT_FOUND"
    USER_ALREADY_EXISTS = "USER_ALREADY_EXISTS"
    INVALID_CREDENTIALS = "INVALID_CREDENTIALS"
    ACCOUNT_DISABLED = "ACCOUNT_DISABLED"

    # 预测相关错误
    PREDICTION_FAILED = "PREDICTION_FAILED"
    MODEL_NOT_LOADED = "MODEL_NOT_LOADED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"

    # 文件相关错误
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    INVALID_FILE_FORMAT = "INVALID_FILE_FORMAT"
    UPLOAD_FAILED = "UPLOAD_FAILED"

class CustomException(Exception):
    """自定义异常基类"""

    def __init__(
        self,
        detail: str,
        status_code: int = 500,
        error_code: Optional[ErrorCode] = None,
        headers: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        self.detail = detail
        self.status_code = status_code
        self.error_code = error_code or ErrorCode.INTERNAL_ERROR
        self.headers = headers
        self.context = context or {}
        self.timestamp = datetime.now()
        super().__init__(detail)

class ValidationError(CustomException):
    """验证错误"""

    def __init__(self, detail: str, field: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(
            detail=detail,
            status_code=422,
            error_code=ErrorCode.VALIDATION_ERROR,
            context={"field": field, **(context or {})}
        )

class NotFoundError(CustomException):
    """资源未找到错误"""

    def __init__(self, resource: str, resource_id: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(
            detail=f"{resource} with id '{resource_id}' not found",
            status_code=404,
            error_code=ErrorCode.NOT_FOUND,
            context={"resource": resource, "resource_id": resource_id, **(context or {})}
        )

class UnauthorizedError(CustomException):
    """未授权错误"""

    def __init__(self, detail: str = "Unauthorized", context: Optional[Dict[str, Any]] = None):
        super().__init__(
            detail=detail,
            status_code=401,
            error_code=ErrorCode.UNAUTHORIZED,
            context=context or {}
        )

class ForbiddenError(CustomException):
    """禁止访问错误"""

    def __init__(self, detail: str = "Forbidden", context: Optional[Dict[str, Any]] = None):
        super().__init__(
            detail=detail,
            status_code=403,
            error_code=ErrorCode.FORBIDDEN,
            context=context or {}
        )

class ConflictError(CustomException):
    """冲突错误"""

    def __init__(self, detail: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(
            detail=detail,
            status_code=409,
            error_code=ErrorCode.CONFLICT,
            context=context or {}
        )

class ModelError(CustomException):
    """AI模型错误"""

    def __init__(self, detail: str, model_name: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(
            detail=detail,
            status_code=500,
            error_code=ErrorCode.MODEL_ERROR,
            context={"model_name": model_name, **(context or {})}
        )

class DatabaseError(CustomException):
    """数据库错误"""

    def __init__(self, detail: str, operation: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(
            detail=detail,
            status_code=500,
            error_code=ErrorCode.DATABASE_ERROR,
            context={"operation": operation, **(context or {})}
        )

class CacheError(CustomException):
    """缓存错误"""

    def __init__(self, detail: str, cache_key: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(
            detail=detail,
            status_code=500,
            error_code=ErrorCode.CACHE_ERROR,
            context={"cache_key": cache_key, **(context or {})}
        )

class ExternalAPIError(CustomException):
    """外部API错误"""

    def __init__(self, detail: str, api_name: Optional[str] = None, status_code: Optional[int] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(
            detail=detail,
            status_code=500,
            error_code=ErrorCode.EXTERNAL_API_ERROR,
            context={"api_name": api_name, "external_status_code": status_code, **(context or {})}
        )

class PredictionError(CustomException):
    """预测错误"""

    def __init__(self, detail: str, model_type: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(
            detail=detail,
            status_code=500,
            error_code=ErrorCode.PREDICTION_FAILED,
            context={"model_type": model_type, **(context or {})}
        )

class FileError(CustomException):
    """文件错误"""

    def __init__(self, detail: str, filename: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(
            detail=detail,
            status_code=400,
            error_code=ErrorCode.FILE_NOT_FOUND,
            context={"filename": filename, **(context or {})}
        )

# 异常处理器
class ExceptionHandler:
    """异常处理器"""

    @staticmethod
    def create_error_response(
        request: Request,
        exc: CustomException,
        include_traceback: bool = False
    ) -> JSONResponse:
        """创建错误响应"""

        # 构建错误响应
        error_response = {
            "success": False,
            "error": {
                "code": exc.error_code.value,
                "message": exc.detail,
                "timestamp": exc.timestamp.isoformat(),
                "path": str(request.url),
                "method": request.method
            }
        }

        # 添加上下文信息
        if exc.context:
            error_response["error"]["context"] = exc.context

        # 添加请求ID（如果存在）
        if hasattr(request.state, "request_id"):
            error_response["error"]["request_id"] = request.state.request_id

        # 在开发环境中添加堆栈跟踪
        if include_traceback:
            error_response["error"]["traceback"] = traceback.format_exc()

        # 记录错误日志
        logger.error(
            f"异常处理: {exc.error_code.value} - {exc.detail}",
            extra={
                "error_code": exc.error_code.value,
                "status_code": exc.status_code,
                "context": exc.context,
                "path": str(request.url),
                "method": request.method
            }
        )

        return JSONResponse(
            status_code=exc.status_code,
            content=error_response,
            headers=exc.headers
        )

    @staticmethod
    async def handle_custom_exception(request: Request, exc: CustomException) -> JSONResponse:
        """处理自定义异常"""
        return ExceptionHandler.create_error_response(request, exc)

    @staticmethod
    async def handle_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
        """处理HTTP异常"""
        error_response = {
            "success": False,
            "error": {
                "code": "HTTP_ERROR",
                "message": exc.detail,
                "timestamp": datetime.now().isoformat(),
                "path": str(request.url),
                "method": request.method,
                "status_code": exc.status_code
            }
        }

        logger.warning(
            f"HTTP异常: {exc.status_code} - {exc.detail}",
            extra={
                "status_code": exc.status_code,
                "path": str(request.url),
                "method": request.method
            }
        )

        return JSONResponse(
            status_code=exc.status_code,
            content=error_response,
            headers=exc.headers
        )

    @staticmethod
    async def handle_general_exception(request: Request, exc: Exception) -> JSONResponse:
        """处理通用异常"""
        error_response = {
            "success": False,
            "error": {
                "code": ErrorCode.INTERNAL_ERROR.value,
                "message": "内部服务器错误",
                "timestamp": datetime.now().isoformat(),
                "path": str(request.url),
                "method": request.method
            }
        }

        # 记录详细错误信息
        logger.error(
            f"未处理的异常: {type(exc).__name__} - {str(exc)}",
            extra={
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "path": str(request.url),
                "method": request.method,
                "traceback": traceback.format_exc()
            }
        )

        return JSONResponse(
            status_code=500,
            content=error_response
        )

# 全局异常处理器实例
exception_handler = ExceptionHandler()

# 异常处理装饰器
def handle_exceptions(func):
    """异常处理装饰器"""
    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except CustomException:
            raise  # 重新抛出自定义异常
        except Exception as e:
            logger.error(f"函数 {func.__name__} 执行异常: {e}")
            raise CustomException(
                detail=f"执行 {func.__name__} 时发生错误: {str(e)}",
                context={"function": func.__name__, "original_error": str(e)}
            )

    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except CustomException:
            raise  # 重新抛出自定义异常
        except Exception as e:
            logger.error(f"函数 {func.__name__} 执行异常: {e}")
            raise CustomException(
                detail=f"执行 {func.__name__} 时发生错误: {str(e)}",
                context={"function": func.__name__, "original_error": str(e)}
            )

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper

__all__ = ["'logger'", "'ErrorCode'", "'CustomException'", "'ValidationError'", "'NotFoundError'", "'UnauthorizedError'", "'ForbiddenError'", "'ConflictError'", "'ModelError'", "'DatabaseError'", "'CacheError'", "'ExternalAPIError'", "'PredictionError'", "'FileError'", "'ExceptionHandler'", "'exception_handler'", "'handle_exceptions'"]
