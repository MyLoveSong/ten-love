"""
MCP模块注册中心
实现模块化控制平台的核心功能
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime

"""
MCP (Model Context Protocol) 注册中心
统一管理所有模块的注册、发现和依赖注入
遵循SOLID原则，消除重复导入
"""

import logging
logger = logging.getLogger(__name__)
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import inspect
import threading
from pathlib import Path
import json

logger = logging.getLogger(__name__)

@dataclass
class ModuleMetadata:
    """模块元数据"""
    name: str
    version: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    path: Optional[str] = None

class MCPRegistry:
    """MCP注册中心 - 单例模式"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._modules: Dict[str, ModuleMetadata] = {}
            self._services: Dict[str, object] = {}
            self._factories: Dict[str, Callable] = {}
            self._initialized = True
            logger.info("MCP注册中心已初始化")

    def register_module(
        self,
        name: str,
        version: str = "1.0.0",
        description: str = "",
        dependencies: Optional[List[str]] = None,
        exports: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        path: Optional[str] = None
    ) -> None:
        """注册模块"""
        metadata = ModuleMetadata(
            name=name,
            version=version,
            description=description,
            dependencies=dependencies or [],
            exports=exports or [],
            tags=tags or [],
            path=path
        )

        self._modules[name] = metadata
        logger.info(f"模块已注册: {name} v{version}")

    def register_service(
        self,
        name: str,
        service: object,
        singleton: bool = True
    ) -> None:
        """注册服务"""
        if singleton:
            self._services[name] = service
        else:
            # 非单例服务存储工厂函数
            self._factories[name] = lambda: service

        logger.info(f"服务已注册: {name} (单例: {singleton})")

    def register_factory(
        self,
        name: str,
        factory: Callable
    ) -> None:
        """注册工厂函数"""
        self._factories[name] = factory
        logger.info(f"工厂函数已注册: {name}")

    def get_service(self, name: str) -> object:
        """获取服务"""
        if name in self._services:
            return self._services[name]

        if name in self._factories:
            return self._factories[name]()

        raise ValueError(f"服务未找到: {name}")

    def get_module(self, name: str) -> Optional[ModuleMetadata]:
        """获取模块元数据"""
        return self._modules.get(name)

    def list_modules(self, tag: Optional[str] = None) -> List[ModuleMetadata]:
        """列出模块"""
        if tag:
            return [m for m in self._modules.values() if tag in m.tags]
        return list(self._modules.values())

    def resolve_dependencies(self, module_name: str) -> List[str]:
        """解析模块依赖"""
        resolved = []
        to_resolve = [module_name]

        while to_resolve:
            current = to_resolve.pop(0)
            if current in resolved:
                continue

            module = self._modules.get(current)
            if module:
                resolved.append(current)
                to_resolve.extend(module.dependencies)

        return resolved

    def export_registry(self, file_path: str) -> None:
        """导出注册表"""
        data = {
            "modules": {
                name: {
                    "version": meta.version,
                    "description": meta.description,
                    "dependencies": meta.dependencies,
                    "exports": meta.exports,
                    "tags": meta.tags,
                    "path": meta.path
                }
                for name, meta in self._modules.items()
            },
            "services": list(self._services.keys()),
            "factories": list(self._factories.keys())
        }

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"注册表已导出到: {file_path}")

# 全局注册中心实例
mcp_registry = MCPRegistry()

def register_module(
    name: str,
    version: str = "1.0.0",
    description: str = "",
    dependencies: Optional[List[str]] = None,
    exports: Optional[List[str]] = None,
    tags: Optional[List[str]] = None
):
    """模块注册装饰器"""
    def decorator(cls):
        mcp_registry.register_module(
            name=name,
            version=version,
            description=description,
            dependencies=dependencies,
            exports=exports,
            tags=tags,
            path=inspect.getfile(cls)
        )
        return cls
    return decorator

def register_service(name: str, singleton: bool = True):
    """服务注册装饰器"""
    def decorator(cls):
        mcp_registry.register_service(name, cls, singleton)
        return cls
    return decorator

def inject(service_name: str):
    """依赖注入装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            service = mcp_registry.get_service(service_name)
            return func(service, *args, **kwargs)
        return wrapper
    return decorator

class MCPModule(ABC):
    """MCP模块基类"""

    def __init__(self):
        self.registry = mcp_registry
        self._setup()

    @abstractmethod
    def _setup(self) -> None:
        """模块初始化"""
        pass

    def get_dependency(self, name: str) -> object:
        """获取依赖"""
        return self.registry.get_service(name)

    def register_export(self, name: str, obj: object) -> None:
        """注册导出"""
        self.registry.register_service(name, obj)

# 自动注册核心模块
@register_module(
    name="core",
    version="1.0.0",
    description="核心功能模块",
    tags=["core", "base"]
)
class CoreModule(MCPModule):
    """核心模块"""

    def _setup(self) -> None:
        logger.info("核心模块已初始化")

__all__ = ["'logger'", "'ModuleMetadata'", "'MCPRegistry'", "'mcp_registry'", "'register_module'", "'register_service'", "'inject'", "'MCPModule'", "'CoreModule'"]
