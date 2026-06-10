"""
增强集成模块
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime
import asyncio
import json

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版API和集成能力
支持跨系统集成、多协议支持、自定义API
"""

import asyncio
import json
import logging
logger = logging.getLogger(__name__)
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import aiohttp
import xml.etree.ElementTree as ET
import csv
import io
from pathlib import Path
import yaml
import base64
import hashlib
import hmac
from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)

class ProtocolType(Enum):
    """协议类型"""
    REST = "rest"
    GRAPHQL = "graphql"
    WEBSOCKET = "websocket"
    GRPC = "grpc"
    MQTT = "mqtt"
    AMQP = "amqp"

class DataFormat(Enum):
    """数据格式"""
    JSON = "json"
    XML = "xml"
    CSV = "csv"
    YAML = "yaml"
    PROTOBUF = "protobuf"
    MSGPACK = "msgpack"

class IntegrationType(Enum):
    """集成类型"""
    EHR = "ehr"  # 电子健康记录
    HIS = "his"  # 医院信息系统
    WEARABLE = "wearable"  # 可穿戴设备
    IOT = "iot"  # 物联网设备
    EXTERNAL_API = "external_api"  # 外部API
    DATABASE = "database"  # 数据库

@dataclass
class IntegrationConfig:
    """集成配置"""
    integration_id: str
    name: str
    integration_type: IntegrationType
    protocol: ProtocolType
    endpoint: str
    authentication: Dict[str, Any] = field(default_factory=dict)
    data_format: DataFormat = DataFormat.JSON
    timeout: int = 30
    retry_count: int = 3
    rate_limit: Optional[int] = None
    health_check_interval: int = 300  # 5分钟
    enabled: bool = True

@dataclass
class CustomAPIDefinition:
    """自定义API定义"""
    api_id: str
    name: str
    description: str
    endpoint: str
    method: str  # GET, POST, PUT, DELETE
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    response_schema: Dict[str, Any] = field(default_factory=dict)
    authentication_required: bool = True
    rate_limit: Optional[int] = None
    timeout: int = 30
    enabled: bool = True

@dataclass
class DataTransformationRule:
    """数据转换规则"""
    rule_id: str
    name: str
    source_format: DataFormat
    target_format: DataFormat
    transformation_mapping: Dict[str, str] = field(default_factory=dict)
    validation_rules: List[Dict[str, Any]] = field(default_factory=list)
    enabled: bool = True

class MultiProtocolAdapter:
    """多协议适配器"""

    def __init__(self):
        self.protocol_handlers: Dict[ProtocolType, Callable] = {}
        self.data_formatters: Dict[DataFormat, Callable] = {}

        # 初始化协议处理器
        self._initialize_protocol_handlers()

        # 初始化数据格式化器
        self._initialize_data_formatters()

    def _initialize_protocol_handlers(self):
        """初始化协议处理器"""
        self.protocol_handlers[ProtocolType.REST] = self._handle_rest_request
        self.protocol_handlers[ProtocolType.GRAPHQL] = self._handle_graphql_request
        self.protocol_handlers[ProtocolType.WEBSOCKET] = self._handle_websocket_request
        self.protocol_handlers[ProtocolType.MQTT] = self._handle_mqtt_request

    def _initialize_data_formatters(self):
        """初始化数据格式化器"""
        self.data_formatters[DataFormat.JSON] = self._format_json
        self.data_formatters[DataFormat.XML] = self._format_xml
        self.data_formatters[DataFormat.CSV] = self._format_csv
        self.data_formatters[DataFormat.YAML] = self._format_yaml

    async def _handle_rest_request(self, config: IntegrationConfig,
                                 data: Dict[str, Any]) -> Dict[str, Any]:
        """处理REST请求"""
        async with aiohttp.ClientSession() as session:
            headers = self._prepare_headers(config)

            # 格式化数据
            formatted_data = self._format_data(data, config.data_format)

            async with session.request(
                method='POST',
                url=config.endpoint,
                headers=headers,
                data=formatted_data,
                timeout=aiohttp.ClientTimeout(total=config.timeout)
            ) as response:
                response_data = await response.text()
                return await self._parse_response(response_data, config.data_format)

    async def _handle_graphql_request(self, config: IntegrationConfig,
                                    query: str, variables: Dict[str, Any] = None) -> Dict[str, Any]:
        """处理GraphQL请求"""
        async with aiohttp.ClientSession() as session:
            headers = self._prepare_headers(config)
            headers['Content-Type'] = 'application/json'

            payload = {
                'query': query,
                'variables': variables or {}
            }

            async with session.post(
                config.endpoint,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=config.timeout)
            ) as response:
                return await response.json()

    async def _handle_websocket_request(self, config: IntegrationConfig,
                                      message: Dict[str, Any]) -> Dict[str, Any]:
        """处理WebSocket请求"""
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(config.endpoint) as ws:
                await ws.send_str(json.dumps(message))
                response = await ws.receive()
                return json.loads(response.data)

    async def _handle_mqtt_request(self, config: IntegrationConfig,
                                 topic: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """处理MQTT请求"""
        # 这里需要集成MQTT客户端库，如aiomqtt
        # 为了简化，这里返回模拟响应
        logger.info(f"MQTT消息发送到主题 {topic}: {message}")
        return {"status": "sent", "topic": topic}

    def _prepare_headers(self, config: IntegrationConfig) -> Dict[str, str]:
        """准备请求头"""
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'Health-Monitoring-System/1.0'
        }

        # 添加认证头
        if config.authentication:
            auth_type = config.authentication.get('type', 'bearer')
            if auth_type == 'bearer':
                token = config.authentication.get('token')
                if token:
                    headers['Authorization'] = f'Bearer {token}'
            elif auth_type == 'api_key':
                api_key = config.authentication.get('api_key')
                key_name = config.authentication.get('key_name', 'X-API-Key')
                if api_key:
                    headers[key_name] = api_key

        return headers

    def _format_data(self, data: Dict[str, Any], data_format: DataFormat) -> Union[str, bytes]:
        """格式化数据"""
        formatter = self.data_formatters.get(data_format)
        if formatter:
            return formatter(data)
        else:
            return json.dumps(data)

    def _format_json(self, data: Dict[str, Any]) -> str:
        """JSON格式化"""
        return json.dumps(data, ensure_ascii=False, indent=2)

    def _format_xml(self, data: Dict[str, Any]) -> str:
        """XML格式化"""
        root = ET.Element("data")
        self._dict_to_xml(data, root)
        return ET.tostring(root, encoding='unicode')

    def _dict_to_xml(self, data: Dict[str, Any], parent: ET.Element):
        """字典转XML"""
        for key, value in data.items():
            element = ET.SubElement(parent, key)
            if isinstance(value, dict):
                self._dict_to_xml(value, element)
            else:
                element.text = str(value)

    def _format_csv(self, data: Dict[str, Any]) -> str:
        """CSV格式化"""
        output = io.StringIO()
        if isinstance(data, list):
            if data and isinstance(data[0], dict):
                writer = csv.DictWriter(output, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
        else:
            writer = csv.writer(output)
            writer.writerow(data.keys())
            writer.writerow(data.values())

        return output.getvalue()

    def _format_yaml(self, data: Dict[str, Any]) -> str:
        """YAML格式化"""
        return yaml.dump(data, default_flow_style=False, allow_unicode=True)

    async def _parse_response(self, response_data: str, data_format: DataFormat) -> Dict[str, Any]:
        """解析响应"""
        try:
            if data_format == DataFormat.JSON:
                return json.loads(response_data)
            elif data_format == DataFormat.XML:
                root = ET.fromstring(response_data)
                return self._xml_to_dict(root)
            elif data_format == DataFormat.YAML:
                return yaml.safe_load(response_data)
            else:
                return {"raw_data": response_data}
        except Exception as e:
            logger.error(f"响应解析失败: {e}")
            return {"error": str(e), "raw_data": response_data}

    def _xml_to_dict(self, element: ET.Element) -> Dict[str, Any]:
        """XML转字典"""
        result = {}
        for child in element:
            if len(child) == 0:
                result[child.tag] = child.text
            else:
                result[child.tag] = self._xml_to_dict(child)
        return result

class CrossSystemIntegrator:
    """跨系统集成器"""

    def __init__(self):
        self.integrations: Dict[str, IntegrationConfig] = {}
        self.protocol_adapter = MultiProtocolAdapter()
        self.data_transformations: Dict[str, DataTransformationRule] = {}
        self.health_monitors: Dict[str, Dict[str, Any]] = {}

        # 初始化默认集成
        self._initialize_default_integrations()

        # 启动健康检查任务
        self._start_health_check_tasks()

    def _initialize_default_integrations(self):
        """初始化默认集成"""
        default_integrations = [
            IntegrationConfig(
                integration_id="ehr_system",
                name="电子健康记录系统",
                integration_type=IntegrationType.EHR,
                protocol=ProtocolType.REST,
                endpoint="https://ehr.example.com/api/v1",
                authentication={"type": "bearer", "token": "ehr_token"},
                data_format=DataFormat.JSON
            ),
            IntegrationConfig(
                integration_id="wearable_device",
                name="可穿戴设备",
                integration_type=IntegrationType.WEARABLE,
                protocol=ProtocolType.MQTT,
                endpoint="mqtt://wearable.example.com:1883",
                data_format=DataFormat.JSON
            ),
            IntegrationConfig(
                integration_id="external_api",
                name="外部健康API",
                integration_type=IntegrationType.EXTERNAL_API,
                protocol=ProtocolType.GRAPHQL,
                endpoint="https://api.health.example.com/graphql",
                authentication={"type": "api_key", "api_key": "external_key", "key_name": "X-API-Key"},
                data_format=DataFormat.JSON
            )
        ]

        for integration in default_integrations:
            self.integrations[integration.integration_id] = integration

    def _start_health_check_tasks(self):
        """启动健康检查任务"""
        # 延迟启动，避免在初始化时创建异步任务
        self._health_check_task_started = False

    async def _ensure_health_check_task(self):
        """确保健康检查任务正在运行"""
        if not self._health_check_task_started:
            self._health_check_task_started = True
            asyncio.create_task(self._health_check_loop())

    async def _health_check_loop(self):
        """健康检查循环"""
        while True:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(60)  # 每分钟检查一次
            except Exception as e:
                logger.error(f"健康检查失败: {e}")
                await asyncio.sleep(30)

    async def register_integration(self, config: IntegrationConfig):
        """注册集成"""
        self.integrations[config.integration_id] = config

        # 初始化健康监控
        self.health_monitors[config.integration_id] = {
            'last_check': None,
            'status': 'unknown',
            'response_time': None,
            'error_count': 0,
            'success_count': 0
        }

        logger.info(f"注册集成: {config.name}")

    async def send_data_to_integration(self, integration_id: str,
                                     data: Dict[str, Any]) -> Dict[str, Any]:
        """发送数据到集成系统"""
        if integration_id not in self.integrations:
            raise ValueError(f"集成 {integration_id} 不存在")

        config = self.integrations[integration_id]

        if not config.enabled:
            raise ValueError(f"集成 {integration_id} 已禁用")

        try:
            # 数据转换
            transformed_data = await self._transform_data(data, config)

            # 发送数据
            handler = self.protocol_adapter.protocol_handlers.get(config.protocol)
            if handler:
                result = await handler(config, transformed_data)

                # 更新健康监控
                await self._update_health_monitor(integration_id, True, None)

                return result
            else:
                raise ValueError(f"不支持的协议: {config.protocol}")

        except Exception as e:
            logger.error(f"发送数据到集成失败 ({integration_id}): {e}")
            await self._update_health_monitor(integration_id, False, str(e))
            raise

    async def _transform_data(self, data: Dict[str, Any],
                           config: IntegrationConfig) -> Dict[str, Any]:
        """转换数据格式"""
        # 查找适用的转换规则
        transformation_rule = None
        for rule in self.data_transformations.values():
            if (rule.source_format == DataFormat.JSON and
                rule.target_format == config.data_format and
                rule.enabled):
                transformation_rule = rule
                break

        if transformation_rule:
            return self._apply_transformation(data, transformation_rule)
        else:
            return data

    def _apply_transformation(self, data: Dict[str, Any],
                            rule: DataTransformationRule) -> Dict[str, Any]:
        """应用数据转换"""
        transformed_data = {}

        for source_key, target_key in rule.transformation_mapping.items():
            if source_key in data:
                transformed_data[target_key] = data[source_key]

        # 应用验证规则
        for validation_rule in rule.validation_rules:
            self._apply_validation(transformed_data, validation_rule)

        return transformed_data

    def _apply_validation(self, data: Dict[str, Any], validation_rule: Dict[str, Any]):
        """应用验证规则"""
        field = validation_rule.get('field')
        rule_type = validation_rule.get('type')
        value = validation_rule.get('value')

        if field in data:
            if rule_type == 'required' and not data[field]:
                raise ValueError(f"字段 {field} 是必需的")
            elif rule_type == 'min_length' and len(str(data[field])) < value:
                raise ValueError(f"字段 {field} 长度不能少于 {value}")
            elif rule_type == 'max_length' and len(str(data[field])) > value:
                raise ValueError(f"字段 {field} 长度不能超过 {value}")

    async def _update_health_monitor(self, integration_id: str, success: bool, error: str = None):
        """更新健康监控"""
        if integration_id in self.health_monitors:
            monitor = self.health_monitors[integration_id]
            monitor['last_check'] = datetime.now()

            if success:
                monitor['status'] = 'healthy'
                monitor['success_count'] += 1
            else:
                monitor['status'] = 'unhealthy'
                monitor['error_count'] += 1
                if error:
                    logger.error(f"集成 {integration_id} 错误: {error}")

    async def _perform_health_checks(self):
        """执行健康检查"""
        for integration_id, config in self.integrations.items():
            if not config.enabled:
                continue

            try:
                # 发送健康检查请求
                health_data = {"health_check": True, "timestamp": datetime.now().isoformat()}
                await self.send_data_to_integration(integration_id, health_data)

            except Exception as e:
                logger.warning(f"集成 {integration_id} 健康检查失败: {e}")

    def get_integration_status(self) -> Dict[str, Any]:
        """获取集成状态"""
        status = {}
        for integration_id, config in self.integrations.items():
            monitor = self.health_monitors.get(integration_id, {})
            status[integration_id] = {
                'name': config.name,
                'type': config.integration_type.value,
                'protocol': config.protocol.value,
                'enabled': config.enabled,
                'status': monitor.get('status', 'unknown'),
                'last_check': monitor.get('last_check'),
                'success_count': monitor.get('success_count', 0),
                'error_count': monitor.get('error_count', 0)
            }
        return status

class CustomAPIManager:
    """自定义API管理器"""

    def __init__(self):
        self.custom_apis: Dict[str, CustomAPIDefinition] = {}
        self.api_handlers: Dict[str, Callable] = {}
        self.rate_limiters: Dict[str, Dict[str, Any]] = {}

        # 初始化默认API
        self._initialize_default_apis()

    def _initialize_default_apis(self):
        """初始化默认API"""
        default_apis = [
            CustomAPIDefinition(
                api_id="workflow_custom",
                name="自定义工作流API",
                description="允许用户自定义工作流配置",
                endpoint="/api/v1/custom/workflow",
                method="POST",
                parameters=[
                    {"name": "workflow_config", "type": "object", "required": True},
                    {"name": "user_id", "type": "string", "required": True}
                ],
                response_schema={
                    "type": "object",
                    "properties": {
                        "workflow_id": {"type": "string"},
                        "status": {"type": "string"},
                        "message": {"type": "string"}
                    }
                }
            ),
            CustomAPIDefinition(
                api_id="data_export",
                name="数据导出API",
                description="导出用户健康数据",
                endpoint="/api/v1/custom/export",
                method="GET",
                parameters=[
                    {"name": "user_id", "type": "string", "required": True},
                    {"name": "format", "type": "string", "required": False, "default": "json"},
                    {"name": "date_range", "type": "object", "required": False}
                ],
                response_schema={
                    "type": "object",
                    "properties": {
                        "export_url": {"type": "string"},
                        "file_size": {"type": "number"},
                        "expires_at": {"type": "string"}
                    }
                }
            )
        ]

        for api in default_apis:
            self.custom_apis[api.api_id] = api

    async def register_custom_api(self, api_definition: CustomAPIDefinition,
                                handler: Callable):
        """注册自定义API"""
        self.custom_apis[api_definition.api_id] = api_definition
        self.api_handlers[api_definition.api_id] = handler

        # 初始化速率限制器
        if api_definition.rate_limit:
            self.rate_limiters[api_definition.api_id] = {
                'limit': api_definition.rate_limit,
                'window': 3600,  # 1小时窗口
                'requests': []
            }

        logger.info(f"注册自定义API: {api_definition.name}")

    async def execute_custom_api(self, api_id: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行自定义API"""
        if api_id not in self.custom_apis:
            raise ValueError(f"API {api_id} 不存在")

        api_definition = self.custom_apis[api_id]

        if not api_definition.enabled:
            raise ValueError(f"API {api_id} 已禁用")

        # 检查速率限制
        if not await self._check_rate_limit(api_id):
            raise ValueError(f"API {api_id} 速率限制超出")

        # 验证参数
        await self._validate_parameters(request_data, api_definition.parameters)

        # 执行API处理器
        handler = self.api_handlers.get(api_id)
        if handler:
            result = await handler(request_data)

            # 更新速率限制器
            await self._update_rate_limiter(api_id)

            return result
        else:
            raise ValueError(f"API {api_id} 处理器未找到")

    async def _check_rate_limit(self, api_id: str) -> bool:
        """检查速率限制"""
        if api_id not in self.rate_limiters:
            return True

        limiter = self.rate_limiters[api_id]
        current_time = datetime.now()

        # 清理过期请求
        limiter['requests'] = [
            req_time for req_time in limiter['requests']
            if (current_time - req_time).total_seconds() < limiter['window']
        ]

        # 检查是否超出限制
        return len(limiter['requests']) < limiter['limit']

    async def _update_rate_limiter(self, api_id: str):
        """更新速率限制器"""
        if api_id in self.rate_limiters:
            self.rate_limiters[api_id]['requests'].append(datetime.now())

    async def _validate_parameters(self, request_data: Dict[str, Any],
                                 parameters: List[Dict[str, Any]]):
        """验证参数"""
        for param in parameters:
            param_name = param['name']
            param_type = param['type']
            required = param.get('required', False)

            if required and param_name not in request_data:
                raise ValueError(f"必需参数 {param_name} 缺失")

            if param_name in request_data:
                value = request_data[param_name]
                if not self._validate_parameter_type(value, param_type):
                    raise ValueError(f"参数 {param_name} 类型错误，期望 {param_type}")

    def _validate_parameter_type(self, value: Any, expected_type: str) -> bool:
        """验证参数类型"""
        type_map = {
            'string': str,
            'integer': int,
            'number': (int, float),
            'boolean': bool,
            'object': dict,
            'array': list
        }

        expected_python_type = type_map.get(expected_type)
        if expected_python_type:
            return isinstance(value, expected_python_type)

        return True

    def get_custom_api_list(self) -> List[Dict[str, Any]]:
        """获取自定义API列表"""
        api_list = []
        for api_id, api_definition in self.custom_apis.items():
            api_list.append({
                'api_id': api_id,
                'name': api_definition.name,
                'description': api_definition.description,
                'endpoint': api_definition.endpoint,
                'method': api_definition.method,
                'enabled': api_definition.enabled,
                'rate_limit': api_definition.rate_limit
            })
        return api_list

# 创建全局集成管理器实例
cross_system_integrator = CrossSystemIntegrator()
custom_api_manager = CustomAPIManager()

logger.info("增强版API和集成管理器已创建")

__all__ = ["'logger'", "'ProtocolType'", "'DataFormat'", "'IntegrationType'", "'IntegrationConfig'", "'CustomAPIDefinition'", "'DataTransformationRule'", "'MultiProtocolAdapter'", "'CrossSystemIntegrator'", "'CustomAPIManager'", "'cross_system_integrator'", "'custom_api_manager'"]
