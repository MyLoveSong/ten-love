

"""
统一配置管理系统
基于MCP架构的配置管理和环境隔离
"""

import logging
import os
import json
import yaml
from typing import Dict, List, Optional, Any, Union, Type, TypeVar
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import asyncio
from datetime import datetime, timedelta
import threading
from abc import ABC, abstractmethod

from backend.app.core.dependency_injection import injectable, singleton

logger = logging.getLogger(__name__)

T = TypeVar('T')

class Environment(Enum):
    """环境类型"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

class ConfigSource(Enum):
    """配置源"""
    ENVIRONMENT = "environment"
    FILE = "file"
    DATABASE = "database"
    REMOTE = "remote"
    DEFAULT = "default"

@dataclass
class ConfigValue:
    """配置值"""
    key: str
    value: Any
    source: ConfigSource
    environment: Environment
    last_updated: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConfigSection:
    """配置节"""
    name: str
    values: Dict[str, ConfigValue] = field(default_factory=dict)
    description: str = ""
    required: bool = True
    validation_rules: Dict[str, Any] = field(default_factory=dict)

class IConfigurationProvider(ABC):
    """配置提供者接口"""

    @abstractmethod
    async def get_value(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        pass

    @abstractmethod
    async def set_value(self, key: str, value: Any) -> None:
        """设置配置值"""
        pass

    @abstractmethod
    async def get_section(self, section_name: str) -> ConfigSection:
        """获取配置节"""
        pass

    @abstractmethod
    async def reload(self) -> None:
        """重新加载配置"""
        pass

@singleton(IConfigurationProvider)
class EnvironmentConfigurationProvider(IConfigurationProvider):
    """环境变量配置提供者"""

    def __init__(self):
        self._values: Dict[str, ConfigValue] = {}
        self._load_environment_variables()

    def _load_environment_variables(self):
        """加载环境变量"""
        for key, value in os.environ.items():
            if key.startswith('APP_'):
                config_key = key[4:].lower().replace('_', '.')
                self._values[config_key] = ConfigValue(
                    key=config_key,
                    value=self._parse_value(value),
                    source=ConfigSource.ENVIRONMENT,
                    environment=self._get_current_environment(),
                    last_updated=datetime.now()
                )

    def _parse_value(self, value: str) -> Any:
        """解析值"""
        # 尝试解析为JSON
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            pass

        # 尝试解析为布尔值
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'

        # 尝试解析为数字
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass

        # 返回字符串
        return value

    def _get_current_environment(self) -> Environment:
        """获取当前环境"""
        env_name = os.getenv('APP_ENVIRONMENT', 'development').lower()
        try:
            return Environment(env_name)
        except ValueError:
            return Environment.DEVELOPMENT

    async def get_value(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        if key in self._values:
            return self._values[key].value
        return default

    async def set_value(self, key: str, value: Any) -> None:
        """设置配置值"""
        self._values[key] = ConfigValue(
            key=key,
            value=value,
            source=ConfigSource.ENVIRONMENT,
            environment=self._get_current_environment(),
            last_updated=datetime.now()
        )

    async def get_section(self, section_name: str) -> ConfigSection:
        """获取配置节"""
        section_values = {}
        prefix = f"{section_name}."

        for key, config_value in self._values.items():
            if key.startswith(prefix):
                section_key = key[len(prefix):]
                section_values[section_key] = config_value

        return ConfigSection(
            name=section_name,
            values=section_values,
            description=f"Environment configuration section: {section_name}"
        )

    async def reload(self) -> None:
        """重新加载配置"""
        self._values.clear()
        self._load_environment_variables()

@singleton(IConfigurationProvider)
class FileConfigurationProvider(IConfigurationProvider):
    """文件配置提供者"""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self._values: Dict[str, ConfigValue] = {}
        self._file_watchers: Dict[str, float] = {}
        self._load_config_files()

    def _load_config_files(self):
        """加载配置文件"""
        if not self.config_dir.exists():
            self.config_dir.mkdir(parents=True, exist_ok=True)
            return

        # 加载JSON文件
        for json_file in self.config_dir.glob("*.json"):
            self._load_json_file(json_file)

        # 加载YAML文件
        for yaml_file in self.config_dir.glob("*.yaml"):
            self._load_yaml_file(yaml_file)

        for yaml_file in self.config_dir.glob("*.yml"):
            self._load_yaml_file(yaml_file)

    def _load_json_file(self, file_path: Path):
        """加载JSON文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self._load_data(data, file_path.stem, file_path)
            self._file_watchers[str(file_path)] = file_path.stat().st_mtime

        except Exception as e:
            logger.error(f"加载JSON配置文件失败 {file_path}: {e}")

    def _load_yaml_file(self, file_path: Path):
        """加载YAML文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            self._load_data(data, file_path.stem, file_path)
            self._file_watchers[str(file_path)] = file_path.stat().st_mtime

        except Exception as e:
            logger.error(f"加载YAML配置文件失败 {file_path}: {e}")

    def _load_data(self, data: Dict[str, Any], section_name: str, file_path: Path):
        """加载数据"""
        for key, value in data.items():
            config_key = f"{section_name}.{key}" if section_name != "app" else key

            self._values[config_key] = ConfigValue(
                key=config_key,
                value=value,
                source=ConfigSource.FILE,
                environment=self._get_current_environment(),
                last_updated=datetime.now(),
                metadata={"file_path": str(file_path)}
            )

    def _get_current_environment(self) -> Environment:
        """获取当前环境"""
        env_name = os.getenv('APP_ENVIRONMENT', 'development').lower()
        try:
            return Environment(env_name)
        except ValueError:
            return Environment.DEVELOPMENT

    async def get_value(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        if key in self._values:
            return self._values[key].value
        return default

    async def set_value(self, key: str, value: Any) -> None:
        """设置配置值"""
        self._values[key] = ConfigValue(
            key=key,
            value=value,
            source=ConfigSource.FILE,
            environment=self._get_current_environment(),
            last_updated=datetime.now()
        )

    async def get_section(self, section_name: str) -> ConfigSection:
        """获取配置节"""
        section_values = {}
        prefix = f"{section_name}."

        for key, config_value in self._values.items():
            if key.startswith(prefix):
                section_key = key[len(prefix):]
                section_values[section_key] = config_value

        return ConfigSection(
            name=section_name,
            values=section_values,
            description=f"File configuration section: {section_name}"
        )

    async def reload(self) -> None:
        """重新加载配置"""
        self._values.clear()
        self._file_watchers.clear()
        self._load_config_files()

class ConfigurationManager:
    """配置管理器"""

    def __init__(self):
        self._providers: List[IConfigurationProvider] = []
        self._cache: Dict[str, ConfigValue] = {}
        self._cache_lock = threading.RLock()
        self._cache_ttl = timedelta(minutes=5)
        self._cache_timestamps: Dict[str, datetime] = {}

    def add_provider(self, provider: IConfigurationProvider, priority: int = 0):
        """添加配置提供者"""
        self._providers.insert(priority, provider)

    async def get_value(self, key: str, default: Any = None,
                       environment: Optional[Environment] = None) -> Any:
        """获取配置值"""
        # 检查缓存
        if self._is_cached(key):
            return self._cache[key].value

        # 从提供者获取值
        for provider in self._providers:
            try:
                value = await provider.get_value(key, default)
                if value is not None:
                    # 缓存值
                    self._cache_value(key, value, provider)
                    return value
            except Exception as e:
                logger.warning(f"从提供者获取配置值失败 {key}: {e}")

        return default

    async def get_string(self, key: str, default: str = "") -> str:
        """获取字符串配置"""
        value = await self.get_value(key, default)
        return str(value) if value is not None else default

    async def get_int(self, key: str, default: int = 0) -> int:
        """获取整数配置"""
        value = await self.get_value(key, default)
        try:
            return int(value) if value is not None else default
        except (ValueError, TypeError):
            return default

    async def get_float(self, key: str, default: float = 0.0) -> float:
        """获取浮点数配置"""
        value = await self.get_value(key, default)
        try:
            return float(value) if value is not None else default
        except (ValueError, TypeError):
            return default

    async def get_bool(self, key: str, default: bool = False) -> bool:
        """获取布尔配置"""
        value = await self.get_value(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'on')
        return bool(value) if value is not None else default

    async def get_list(self, key: str, default: List[Any] = None) -> List[Any]:
        """获取列表配置"""
        if default is None:
            default = []

        value = await self.get_value(key, default)
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value.split(',')
        return default

    async def get_dict(self, key: str, default: Dict[str, Any] = None) -> Dict[str, Any]:
        """获取字典配置"""
        if default is None:
            default = {}

        value = await self.get_value(key, default)
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return default
        return default

    async def get_section(self, section_name: str) -> ConfigSection:
        """获取配置节"""
        for provider in self._providers:
            try:
                section = await provider.get_section(section_name)
                if section.values:
                    return section
            except Exception as e:
                logger.warning(f"从提供者获取配置节失败 {section_name}: {e}")

        return ConfigSection(name=section_name)

    async def set_value(self, key: str, value: Any) -> None:
        """设置配置值"""
        # 更新缓存
        self._cache_value(key, value, None)

        # 更新所有提供者
        for provider in self._providers:
            try:
                await provider.set_value(key, value)
            except Exception as e:
                logger.warning(f"设置配置值失败 {key}: {e}")

    async def reload(self) -> None:
        """重新加载配置"""
        self._cache.clear()
        self._cache_timestamps.clear()

        for provider in self._providers:
            try:
                await provider.reload()
            except Exception as e:
                logger.warning(f"重新加载配置提供者失败: {e}")

    def _is_cached(self, key: str) -> bool:
        """检查是否缓存"""
        with self._cache_lock:
            if key not in self._cache:
                return False

            if key not in self._cache_timestamps:
                return False

            cache_time = self._cache_timestamps[key]
            return datetime.now() - cache_time < self._cache_ttl

    def _cache_value(self, key: str, value: Any, provider: Optional[IConfigurationProvider]):
        """缓存值"""
        with self._cache_lock:
            self._cache[key] = ConfigValue(
                key=key,
                value=value,
                source=ConfigSource.FILE if provider else ConfigSource.DEFAULT,
                environment=self._get_current_environment(),
                last_updated=datetime.now()
            )
            self._cache_timestamps[key] = datetime.now()

    def _get_current_environment(self) -> Environment:
        """获取当前环境"""
        env_name = os.getenv('APP_ENVIRONMENT', 'development').lower()
        try:
            return Environment(env_name)
        except ValueError:
            return Environment.DEVELOPMENT

# 全局配置管理器
_config_manager = ConfigurationManager()

def configure_configuration():
    """配置配置管理器"""
    # 添加环境变量提供者
    env_provider = EnvironmentConfigurationProvider()
    _config_manager.add_provider(env_provider, priority=0)

    # 添加文件提供者
    file_provider = FileConfigurationProvider()
    _config_manager.add_provider(file_provider, priority=1)

    return _config_manager

def get_configuration() -> ConfigurationManager:
    """获取配置管理器"""
    return _config_manager

# 便捷函数
async def get_config(key: str, default: Any = None) -> Any:
    """获取配置值"""
    return await _config_manager.get_value(key, default)

async def get_config_string(key: str, default: str = "") -> str:
    """获取字符串配置"""
    return await _config_manager.get_string(key, default)

async def get_config_int(key: str, default: int = 0) -> int:
    """获取整数配置"""
    return await _config_manager.get_int(key, default)

async def get_config_float(key: str, default: float = 0.0) -> float:
    """获取浮点数配置"""
    return await _config_manager.get_float(key, default)

async def get_config_bool(key: str, default: bool = False) -> bool:
    """获取布尔配置"""
    return await _config_manager.get_bool(key, default)

async def get_config_list(key: str, default: List[Any] = None) -> List[Any]:
    """获取列表配置"""
    return await _config_manager.get_list(key, default)

async def get_config_dict(key: str, default: Dict[str, Any] = None) -> Dict[str, Any]:
    """获取字典配置"""
    return await _config_manager.get_dict(key, default)

async def set_config(key: str, value: Any) -> None:
    """设置配置值"""
    await _config_manager.set_value(key, value)

async def reload_config() -> None:
    """重新加载配置"""
    await _config_manager.reload()

# 配置装饰器
def config_value(key: str, default: Any = None):
    """配置值装饰器"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            value = await get_config(key, default)
            return await func(value, *args, **kwargs)
        return wrapper
    return decorator

if __name__ == "__main__":
    # 测试配置管理
    async def test_configuration():
        # 配置配置管理器
        configure_configuration()

        # 设置测试配置
        await set_config("test.string", "hello world")
        await set_config("test.number", 42)
        await set_config("test.boolean", True)
        await set_config("test.list", [1, 2, 3])
        await set_config("test.dict", {"key": "value"})

        # 获取配置
        string_value = await get_config_string("test.string", "default")
        number_value = await get_config_int("test.number", 0)
        boolean_value = await get_config_bool("test.boolean", False)
        list_value = await get_config_list("test.list", [])
        dict_value = await get_config_dict("test.dict", {})

        print(f"String: {string_value}")
        print(f"Number: {number_value}")
        print(f"Boolean: {boolean_value}")
        print(f"List: {list_value}")
        print(f"Dict: {dict_value}")

        # 获取配置节
        section = await _config_manager.get_section("test")
        print(f"Section: {section.name}, Values: {len(section.values)}")

    # 运行测试
    asyncio.run(test_configuration())

__all__ = ["'logger'", "'T'", "'Environment'", "'ConfigSource'", "'ConfigValue'", "'ConfigSection'", "'IConfigurationProvider'", "'EnvironmentConfigurationProvider'", "'FileConfigurationProvider'", "'ConfigurationManager'", "'configure_configuration'", "'get_configuration'", "'config_value'"]
