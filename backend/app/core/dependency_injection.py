

"""
依赖注入容器
基于MCP架构的服务管理和依赖注入
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Type, TypeVar, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import inspect
import threading
from contextlib import asynccontextmanager
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

T = TypeVar('T')

class ServiceScope(Enum):
    """服务作用域"""
    SINGLETON = "singleton"      # 单例
    TRANSIENT = "transient"      # 瞬时
    SCOPED = "scoped"           # 作用域

class ServiceLifetime(Enum):
    """服务生命周期"""
    APPLICATION = "application"  # 应用级别
    REQUEST = "request"         # 请求级别
    SESSION = "session"         # 会话级别

@dataclass
class ServiceDescriptor:
    """服务描述符"""
    service_type: Type
    implementation_type: Optional[Type] = None
    factory: Optional[Callable] = None
    instance: Optional[Any] = None
    scope: ServiceScope = ServiceScope.SINGLETON
    lifetime: ServiceLifetime = ServiceLifetime.APPLICATION
    dependencies: List[Type] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class ServiceContainer:
    """服务容器"""

    def __init__(self):
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._instances: Dict[Type, Any] = {}
        self._scoped_instances: Dict[str, Dict[Type, Any]] = {}
        self._lock = threading.RLock()
        self._current_scope: Optional[str] = None

    def register_singleton(self, service_type: Type[T], implementation_type: Optional[Type[T]] = None,
                          factory: Optional[Callable[[], T]] = None) -> 'ServiceContainer':
        """注册单例服务"""
        descriptor = ServiceDescriptor(
            service_type=service_type,
            implementation_type=implementation_type,
            factory=factory,
            scope=ServiceScope.SINGLETON,
            lifetime=ServiceLifetime.APPLICATION
        )

        self._register_service(descriptor)
        return self

    def register_transient(self, service_type: Type[T], implementation_type: Optional[Type[T]] = None,
                          factory: Optional[Callable[[], T]] = None) -> 'ServiceContainer':
        """注册瞬时服务"""
        descriptor = ServiceDescriptor(
            service_type=service_type,
            implementation_type=implementation_type,
            factory=factory,
            scope=ServiceScope.TRANSIENT,
            lifetime=ServiceLifetime.REQUEST
        )

        self._register_service(descriptor)
        return self

    def register_scoped(self, service_type: Type[T], implementation_type: Optional[Type[T]] = None,
                       factory: Optional[Callable[[], T]] = None) -> 'ServiceContainer':
        """注册作用域服务"""
        descriptor = ServiceDescriptor(
            service_type=service_type,
            implementation_type=implementation_type,
            factory=factory,
            scope=ServiceScope.SCOPED,
            lifetime=ServiceLifetime.REQUEST
        )

        self._register_service(descriptor)
        return self

    def _register_service(self, descriptor: ServiceDescriptor):
        """注册服务"""
        with self._lock:
            self._services[descriptor.service_type] = descriptor

            # 分析依赖
            if descriptor.implementation_type:
                dependencies = self._analyze_dependencies(descriptor.implementation_type)
                descriptor.dependencies = dependencies

            logger.debug(f"注册服务: {descriptor.service_type.__name__}")

    def _analyze_dependencies(self, implementation_type: Type) -> List[Type]:
        """分析依赖"""
        dependencies = []

        try:
            # 获取构造函数签名
            init_signature = inspect.signature(implementation_type.__init__)

            for param_name, param in init_signature.parameters.items():
                if param_name == 'self':
                    continue

                if param.annotation != inspect.Parameter.empty:
                    dependencies.append(param.annotation)

        except Exception as e:
            logger.warning(f"分析依赖失败 {implementation_type.__name__}: {e}")

        return dependencies

    def get_service(self, service_type: Type[T]) -> T:
        """获取服务实例"""
        with self._lock:
            if service_type not in self._services:
                raise ValueError(f"服务未注册: {service_type.__name__}")

            descriptor = self._services[service_type]

            # 根据作用域返回实例
            if descriptor.scope == ServiceScope.SINGLETON:
                return self._get_singleton_instance(descriptor)
            elif descriptor.scope == ServiceScope.TRANSIENT:
                return self._create_transient_instance(descriptor)
            elif descriptor.scope == ServiceScope.SCOPED:
                return self._get_scoped_instance(descriptor)
            else:
                raise ValueError(f"不支持的作用域: {descriptor.scope}")

    def _get_singleton_instance(self, descriptor: ServiceDescriptor) -> Any:
        """获取单例实例"""
        if descriptor.instance is not None:
            return descriptor.instance

        if descriptor.service_type in self._instances:
            descriptor.instance = self._instances[descriptor.service_type]
            return descriptor.instance

        # 创建新实例
        instance = self._create_instance(descriptor)
        self._instances[descriptor.service_type] = instance
        descriptor.instance = instance

        return instance

    def _create_transient_instance(self, descriptor: ServiceDescriptor) -> Any:
        """创建瞬时实例"""
        return self._create_instance(descriptor)

    def _get_scoped_instance(self, descriptor: ServiceDescriptor) -> Any:
        """获取作用域实例"""
        if not self._current_scope:
            raise ValueError("当前没有活动的作用域")

        if self._current_scope not in self._scoped_instances:
            self._scoped_instances[self._current_scope] = {}

        scope_instances = self._scoped_instances[self._current_scope]

        if descriptor.service_type in scope_instances:
            return scope_instances[descriptor.service_type]

        # 创建新实例
        instance = self._create_instance(descriptor)
        scope_instances[descriptor.service_type] = instance

        return instance

    def _create_instance(self, descriptor: ServiceDescriptor) -> Any:
        """创建服务实例"""
        try:
            if descriptor.factory:
                # 使用工厂方法
                return descriptor.factory()

            if descriptor.implementation_type:
                # 使用实现类型
                implementation_type = descriptor.implementation_type

                # 解析依赖
                dependencies = []
                for dep_type in descriptor.dependencies:
                    dep_instance = self.get_service(dep_type)
                    dependencies.append(dep_instance)

                # 创建实例
                return implementation_type(*dependencies)

            # 使用服务类型本身
            service_type = descriptor.service_type

            # 解析依赖
            dependencies = []
            for dep_type in descriptor.dependencies:
                dep_instance = self.get_service(dep_type)
                dependencies.append(dep_instance)

            return service_type(*dependencies)

        except Exception as e:
            logger.error(f"创建服务实例失败 {descriptor.service_type.__name__}: {e}")
            raise

    @asynccontextmanager
    async def scope(self, scope_name: str):
        """创建作用域"""
        old_scope = self._current_scope
        self._current_scope = scope_name

        try:
            yield self
        finally:
            self._current_scope = old_scope
            # 清理作用域实例
            if scope_name in self._scoped_instances:
                del self._scoped_instances[scope_name]

    def is_registered(self, service_type: Type) -> bool:
        """检查服务是否已注册"""
        return service_type in self._services

    def get_registered_services(self) -> List[Type]:
        """获取所有已注册的服务类型"""
        return list(self._services.keys())

    def clear(self):
        """清空容器"""
        with self._lock:
            self._services.clear()
            self._instances.clear()
            self._scoped_instances.clear()
            self._current_scope = None

class ServiceProvider:
    """服务提供者"""

    def __init__(self, container: ServiceContainer):
        self._container = container

    def get_service(self, service_type: Type[T]) -> T:
        """获取服务"""
        return self._container.get_service(service_type)

    def get_required_service(self, service_type: Type[T]) -> T:
        """获取必需的服务"""
        if not self._container.is_registered(service_type):
            raise ValueError(f"必需的服务未注册: {service_type.__name__}")

        return self.get_service(service_type)

    def get_optional_service(self, service_type: Type[T]) -> Optional[T]:
        """获取可选的服务"""
        if not self._container.is_registered(service_type):
            return None

        try:
            return self.get_service(service_type)
        except Exception:
            return None

    def create_scope(self, scope_name: str):
        """创建作用域"""
        return self._container.scope(scope_name)

class ServiceCollection:
    """服务集合"""

    def __init__(self):
        self._container = ServiceContainer()

    def add_singleton(self, service_type: Type[T], implementation_type: Optional[Type[T]] = None,
                     factory: Optional[Callable[[], T]] = None) -> 'ServiceCollection':
        """添加单例服务"""
        self._container.register_singleton(service_type, implementation_type, factory)
        return self

    def add_transient(self, service_type: Type[T], implementation_type: Optional[Type[T]] = None,
                     factory: Optional[Callable[[], T]] = None) -> 'ServiceCollection':
        """添加瞬时服务"""
        self._container.register_transient(service_type, implementation_type, factory)
        return self

    def add_scoped(self, service_type: Type[T], implementation_type: Optional[Type[T]] = None,
                  factory: Optional[Callable[[], T]] = None) -> 'ServiceCollection':
        """添加作用域服务"""
        self._container.register_scoped(service_type, implementation_type, factory)
        return self

    def build(self) -> ServiceProvider:
        """构建服务提供者"""
        return ServiceProvider(self._container)

# 全局服务容器
_service_collection = ServiceCollection()

def configure_services() -> ServiceCollection:
    """配置服务"""
    return _service_collection

def build_service_provider(service_collection: Optional[ServiceCollection] = None) -> ServiceProvider:
    """构建服务提供者。
    兼容旧调用签名：允许传入 `service_collection` 参数但不依赖它，
    实际以模块级 `_service_collection` 为准，避免错误调用导致崩溃。
    """
    # 为向后兼容而忽略传入参数，统一使用模块级集合
    return _service_collection.build()

# 装饰器
def injectable(service_type: Optional[Type] = None, scope: ServiceScope = ServiceScope.SINGLETON):
    """可注入服务装饰器"""
    def decorator(cls):
        nonlocal service_type
        if service_type is None:
            service_type = cls

        _service_collection.add_singleton(service_type, cls)
        return cls

    return decorator

def singleton(service_type: Optional[Type] = None):
    """单例服务装饰器"""
    return injectable(service_type, ServiceScope.SINGLETON)

def transient(service_type: Optional[Type] = None):
    """瞬时服务装饰器"""
    return injectable(service_type, ServiceScope.TRANSIENT)

def scoped(service_type: Optional[Type] = None):
    """作用域服务装饰器"""
    return injectable(service_type, ServiceScope.SCOPED)

# 依赖注入函数
def get_service(service_type: Type[T]) -> T:
    """获取服务"""
    provider = build_service_provider()
    return provider.get_service(service_type)

def get_required_service(service_type: Type[T]) -> T:
    """获取必需的服务"""
    provider = build_service_provider()
    return provider.get_required_service(service_type)

def get_optional_service(service_type: Type[T]) -> Optional[T]:
    """获取可选的服务"""
    provider = build_service_provider()
    return provider.get_optional_service(service_type)

if __name__ == "__main__":
    # 测试依赖注入容器
    class ILogger(ABC):
        @abstractmethod
        def log(self, message: str):
            pass

    @singleton(ILogger)
    class Logger(ILogger):
        def log(self, message: str):
            print(f"Log: {message}")

    class IDataProcessor(ABC):
        @abstractmethod
        def process(self, data: str) -> str:
            pass

    @singleton(IDataProcessor)
    class DataProcessor(IDataProcessor):
        def __init__(self, logger: ILogger):
            self.logger = logger

        def process(self, data: str) -> str:
            self.logger.log(f"Processing: {data}")
            return f"Processed: {data}"

    # 配置服务
    configure_services()

    # 构建服务提供者
    provider = build_service_provider()

    # 使用服务
    processor = provider.get_service(IDataProcessor)
    result = processor.process("test data")
    print(result)

__all__ = ["'logger'", "'T'", "'ServiceScope'", "'ServiceLifetime'", "'ServiceDescriptor'", "'ServiceContainer'", "'ServiceProvider'", "'ServiceCollection'", "'configure_services'", "'build_service_provider'", "'injectable'", "'singleton'", "'transient'", "'scoped'", "'get_service'", "'get_required_service'", "'get_optional_service'"]
