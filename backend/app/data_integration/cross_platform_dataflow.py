

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨平台数据流模块
支持多平台数据源集成，模块化架构，动态路由和负载均衡
"""

import asyncio
import aiohttp
import logging
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
import hashlib
import time
from collections import defaultdict, deque
import threading

logger = logging.getLogger(__name__)

class DataSourceType(Enum):
    """数据源类型"""
    CGM = "cgm"
    NUTRITION = "nutrition"
    ACTIVITY = "activity"
    IMAGE = "image"
    MEDICAL = "medical"
    ENVIRONMENT = "environment"
    SOCIAL = "social"

class DataQuality(Enum):
    """数据质量等级"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INVALID = "invalid"

@dataclass
class DataPacket:
    """数据包"""
    id: str
    source_type: DataSourceType
    source_platform: str
    timestamp: datetime
    user_id: str
    data: Dict[str, Any]
    quality: DataQuality = DataQuality.MEDIUM
    metadata: Dict[str, Any] = field(default_factory=dict)
    routing_info: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProcessingResult:
    """处理结果"""
    packet_id: str
    success: bool
    processed_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0
    quality_score: float = 0.0

class DataSourceAdapter(ABC):
    """数据源适配器基类"""

    def __init__(self, platform_name: str, config: Dict[str, Any]):
        self.platform_name = platform_name
        self.config = config
        self.is_connected = False
        self.last_health_check = None

    @abstractmethod
    async def connect(self) -> bool:
        """连接到数据源"""
        pass

    @abstractmethod
    async def fetch_data(self, params: Dict[str, Any]) -> List[DataPacket]:
        """获取数据"""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """健康检查"""
        pass

    @abstractmethod
    def normalize_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """数据标准化"""
        pass

class DifyAdapter(DataSourceAdapter):
    """Dify工作流平台适配器"""

    async def connect(self) -> bool:
        """连接到Dify"""
        try:
            headers = {
                'Authorization': f"Bearer {self.config.get('api_key')}",
                'Content-Type': 'application/json'
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config['base_url']}/v1/apps",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    self.is_connected = response.status == 200
                    return self.is_connected
        except Exception as e:
            logger.error(f"Dify连接失败: {e}")
            return False

    async def fetch_data(self, params: Dict[str, Any]) -> List[DataPacket]:
        """从Dify工作流获取数据"""
        packets = []
        try:
            headers = {
                'Authorization': f"Bearer {self.config.get('api_key')}",
                'Content-Type': 'application/json'
            }

            async with aiohttp.ClientSession() as session:
                # 触发工作流
                workflow_data = {
                    'inputs': params,
                    'response_mode': 'blocking',
                    'user': params.get('user_id', 'system')
                }

                async with session.post(
                    f"{self.config['base_url']}/v1/workflows/{self.config['workflow_id']}/run",
                    headers=headers,
                    json=workflow_data,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:

                    if response.status == 200:
                        result = await response.json()

                        # 解析工作流返回的数据
                        outputs = result.get('data', {}).get('outputs', {})

                        for data_type, data_content in outputs.items():
                            packet = DataPacket(
                                id=f"dify_{int(time.time() * 1000)}_{len(packets)}",
                                source_type=self._map_data_type(data_type),
                                source_platform="dify",
                                timestamp=datetime.now(),
                                user_id=params.get('user_id', 'unknown'),
                                data=self.normalize_data(data_content),
                                metadata={
                                    'workflow_id': self.config['workflow_id'],
                                    'execution_id': result.get('data', {}).get('id'),
                                    'raw_data_type': data_type
                                }
                            )
                            packets.append(packet)

        except Exception as e:
            logger.error(f"Dify数据获取失败: {e}")

        return packets

    async def health_check(self) -> bool:
        """Dify健康检查"""
        try:
            headers = {
                'Authorization': f"Bearer {self.config.get('api_key')}",
                'Content-Type': 'application/json'
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config['base_url']}/v1/apps",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    is_healthy = response.status == 200
                    self.last_health_check = datetime.now()
                    return is_healthy
        except:
            return False

    def normalize_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """标准化Dify数据"""
        if isinstance(raw_data, str):
            try:
                raw_data = json.loads(raw_data)
            except:
                raw_data = {'text': raw_data}

        return raw_data

    def _map_data_type(self, dify_type: str) -> DataSourceType:
        """映射Dify数据类型到标准类型"""
        mapping = {
            'cgm_data': DataSourceType.CGM,
            'nutrition_data': DataSourceType.NUTRITION,
            'activity_data': DataSourceType.ACTIVITY,
            'image_data': DataSourceType.IMAGE,
            'medical_data': DataSourceType.MEDICAL
        }
        return mapping.get(dify_type, DataSourceType.CGM)

class APIAdapter(DataSourceAdapter):
    """通用API适配器"""

    async def connect(self) -> bool:
        """连接测试"""
        try:
            headers = self.config.get('headers', {})
            if self.config.get('api_key'):
                headers['Authorization'] = f"Bearer {self.config['api_key']}"

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config['base_url']}/health",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    self.is_connected = response.status < 400
                    return self.is_connected
        except Exception as e:
            logger.warning(f"API连接检查失败: {e}")
            # 对于某些API，健康检查端点可能不存在，标记为已连接
            self.is_connected = True
            return True

    async def fetch_data(self, params: Dict[str, Any]) -> List[DataPacket]:
        """从API获取数据"""
        packets = []
        try:
            headers = self.config.get('headers', {})
            if self.config.get('api_key'):
                headers['Authorization'] = f"Bearer {self.config['api_key']}"

            endpoint = self.config.get('endpoint', '/data')

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config['base_url']}{endpoint}",
                    headers=headers,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:

                    if response.status == 200:
                        data = await response.json()

                        # 处理不同的数据格式
                        if isinstance(data, list):
                            for i, item in enumerate(data):
                                packet = self._create_packet(item, params, i)
                                packets.append(packet)
                        elif isinstance(data, dict):
                            packet = self._create_packet(data, params, 0)
                            packets.append(packet)

        except Exception as e:
            logger.error(f"API数据获取失败 {self.platform_name}: {e}")

        return packets

    async def health_check(self) -> bool:
        """API健康检查"""
        return await self.connect()

    def normalize_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """标准化API数据"""
        # 应用数据映射规则
        mapping_rules = self.config.get('field_mapping', {})
        normalized = {}

        for standard_field, source_field in mapping_rules.items():
            if source_field in raw_data:
                normalized[standard_field] = raw_data[source_field]

        # 保留原始数据
        normalized['_raw'] = raw_data

        return normalized

    def _create_packet(self, data: Dict[str, Any], params: Dict[str, Any], index: int) -> DataPacket:
        """创建数据包"""
        return DataPacket(
            id=f"{self.platform_name}_{int(time.time() * 1000)}_{index}",
            source_type=DataSourceType(self.config.get('data_type', 'cgm')),
            source_platform=self.platform_name,
            timestamp=datetime.now(),
            user_id=params.get('user_id', 'unknown'),
            data=self.normalize_data(data),
            metadata={'api_config': self.config}
        )

class DataRouter:
    """数据路由器"""

    def __init__(self):
        self.routes: Dict[str, List[Callable]] = defaultdict(list)
        self.load_balancers: Dict[str, int] = defaultdict(int)
        self.processor_health: Dict[str, bool] = {}

    def register_processor(self, data_type: str, processor: Callable,
                          processor_id: str = None):
        """注册数据处理器"""
        if processor_id is None:
            processor_id = f"processor_{len(self.routes[data_type])}"

        self.routes[data_type].append((processor_id, processor))
        self.processor_health[processor_id] = True
        logger.info(f"注册处理器 {processor_id} for {data_type}")

    async def route_data(self, packet: DataPacket) -> List[ProcessingResult]:
        """路由数据到处理器"""
        data_type = packet.source_type.value

        if data_type not in self.routes:
            logger.warning(f"未找到数据类型 {data_type} 的处理器")
            return []

        # 负载均衡选择处理器
        available_processors = [
            (pid, proc) for pid, proc in self.routes[data_type]
            if self.processor_health.get(pid, True)
        ]

        if not available_processors:
            logger.error(f"没有可用的处理器处理 {data_type} 数据")
            return []

        results = []

        # 轮询方式分发（可以改为其他策略）
        lb_index = self.load_balancers[data_type] % len(available_processors)
        processor_id, processor = available_processors[lb_index]
        self.load_balancers[data_type] += 1

        try:
            start_time = time.time()

            if asyncio.iscoroutinefunction(processor):
                result_data = await processor(packet)
            else:
                result_data = processor(packet)

            processing_time = time.time() - start_time

            result = ProcessingResult(
                packet_id=packet.id,
                success=True,
                processed_data=result_data,
                processing_time=processing_time,
                quality_score=self._calculate_quality_score(packet, result_data)
            )

            self.processor_health[processor_id] = True

        except Exception as e:
            logger.error(f"处理器 {processor_id} 处理失败: {e}")

            result = ProcessingResult(
                packet_id=packet.id,
                success=False,
                error_message=str(e),
                processing_time=time.time() - start_time
            )

            self.processor_health[processor_id] = False

        results.append(result)
        return results

    def _calculate_quality_score(self, packet: DataPacket,
                               processed_data: Dict[str, Any]) -> float:
        """计算数据质量分数"""
        score = 0.5  # 基础分数

        # 数据完整性检查
        if processed_data and 'error' not in processed_data:
            score += 0.2

        # 时效性检查
        if packet.timestamp and (datetime.now() - packet.timestamp).seconds < 300:  # 5分钟内
            score += 0.1

        # 来源可靠性
        reliable_sources = ['dify', 'official_api']
        if packet.source_platform in reliable_sources:
            score += 0.1

        # 数据量检查
        if processed_data and len(str(processed_data)) > 100:
            score += 0.1

        return min(1.0, score)

class DataQualityManager:
    """数据质量管理器"""

    def __init__(self):
        self.quality_metrics = defaultdict(list)
        self.quality_thresholds = {
            DataQuality.HIGH: 0.8,
            DataQuality.MEDIUM: 0.5,
            DataQuality.LOW: 0.2
        }

    def assess_quality(self, packet: DataPacket) -> DataQuality:
        """评估数据质量"""
        score = 0.0

        # 数据完整性
        if packet.data:
            completeness = len([v for v in packet.data.values() if v is not None]) / len(packet.data)
            score += completeness * 0.4

        # 时效性
        if packet.timestamp:
            age_minutes = (datetime.now() - packet.timestamp).total_seconds() / 60
            timeliness = max(0, 1 - age_minutes / 60)  # 1小时内为满分
            score += timeliness * 0.3

        # 来源可靠性
        reliability_scores = {
            'dify': 0.9,
            'official_api': 0.8,
            'third_party': 0.6,
            'unknown': 0.3
        }
        reliability = reliability_scores.get(packet.source_platform, 0.3)
        score += reliability * 0.3

        # 确定质量等级
        if score >= self.quality_thresholds[DataQuality.HIGH]:
            quality = DataQuality.HIGH
        elif score >= self.quality_thresholds[DataQuality.MEDIUM]:
            quality = DataQuality.MEDIUM
        elif score >= self.quality_thresholds[DataQuality.LOW]:
            quality = DataQuality.LOW
        else:
            quality = DataQuality.INVALID

        # 记录质量指标
        self.quality_metrics[packet.source_platform].append({
            'timestamp': packet.timestamp,
            'score': score,
            'quality': quality
        })

        return quality

    def get_quality_report(self, platform: str = None) -> Dict[str, Any]:
        """获取质量报告"""
        if platform:
            metrics = self.quality_metrics.get(platform, [])
            platforms = [platform]
        else:
            metrics = []
            for platform_metrics in self.quality_metrics.values():
                metrics.extend(platform_metrics)
            platforms = list(self.quality_metrics.keys())

        if not metrics:
            return {'platforms': platforms, 'average_score': 0, 'quality_distribution': {}}

        scores = [m['score'] for m in metrics]
        qualities = [m['quality'] for m in metrics]

        quality_counts = defaultdict(int)
        for q in qualities:
            quality_counts[q.value] += 1

        return {
            'platforms': platforms,
            'total_samples': len(metrics),
            'average_score': np.mean(scores),
            'quality_distribution': dict(quality_counts),
            'score_trend': scores[-10:] if len(scores) > 10 else scores
        }

class CrossPlatformDataFlowManager:
    """跨平台数据流管理器"""

    def __init__(self):
        self.adapters: Dict[str, DataSourceAdapter] = {}
        self.router = DataRouter()
        self.quality_manager = DataQualityManager()
        self.data_buffer: deque = deque(maxlen=1000)
        self.processing_stats = defaultdict(int)
        self.is_running = False
        self._background_tasks = []

    def register_adapter(self, platform_name: str, adapter: DataSourceAdapter):
        """注册数据源适配器"""
        self.adapters[platform_name] = adapter
        logger.info(f"注册数据源适配器: {platform_name}")

    def register_processor(self, data_type: str, processor: Callable):
        """注册数据处理器"""
        self.router.register_processor(data_type, processor)

    async def start(self):
        """启动数据流管理器"""
        self.is_running = True

        # 连接所有数据源
        for platform_name, adapter in self.adapters.items():
            try:
                connected = await adapter.connect()
                if connected:
                    logger.info(f"数据源 {platform_name} 连接成功")
                else:
                    logger.warning(f"数据源 {platform_name} 连接失败")
            except Exception as e:
                logger.error(f"数据源 {platform_name} 连接异常: {e}")

        # 启动后台任务
        self._background_tasks = [
            asyncio.create_task(self._health_check_loop()),
            asyncio.create_task(self._data_processing_loop())
        ]

        logger.info("跨平台数据流管理器启动完成")

    async def stop(self):
        """停止数据流管理器"""
        self.is_running = False

        # 取消后台任务
        for task in self._background_tasks:
            task.cancel()

        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        logger.info("跨平台数据流管理器已停止")

    async def fetch_data_from_platform(self, platform_name: str,
                                     params: Dict[str, Any]) -> List[DataPacket]:
        """从指定平台获取数据"""
        if platform_name not in self.adapters:
            logger.error(f"未找到平台适配器: {platform_name}")
            return []

        adapter = self.adapters[platform_name]

        if not adapter.is_connected:
            logger.warning(f"平台 {platform_name} 未连接，尝试重新连接...")
            connected = await adapter.connect()
            if not connected:
                logger.error(f"平台 {platform_name} 连接失败")
                return []

        try:
            packets = await adapter.fetch_data(params)

            # 评估数据质量
            for packet in packets:
                packet.quality = self.quality_manager.assess_quality(packet)

            # 添加到缓冲区
            self.data_buffer.extend(packets)
            self.processing_stats['fetched'] += len(packets)

            logger.info(f"从 {platform_name} 获取 {len(packets)} 个数据包")
            return packets

        except Exception as e:
            logger.error(f"从 {platform_name} 获取数据失败: {e}")
            return []

    async def process_data_packet(self, packet: DataPacket) -> List[ProcessingResult]:
        """处理单个数据包"""
        try:
            # 质量检查
            if packet.quality == DataQuality.INVALID:
                logger.warning(f"跳过无效数据包: {packet.id}")
                return []

            # 路由到处理器
            results = await self.router.route_data(packet)
            self.processing_stats['processed'] += len(results)

            return results

        except Exception as e:
            logger.error(f"处理数据包失败 {packet.id}: {e}")
            return []

    async def process_batch_data(self, user_id: str, platforms: List[str] = None) -> Dict[str, Any]:
        """批量处理数据"""
        if platforms is None:
            platforms = list(self.adapters.keys())

        all_results = []
        fetch_params = {'user_id': user_id, 'limit': 100}

        # 并发获取数据
        fetch_tasks = []
        for platform in platforms:
            if platform in self.adapters:
                task = self.fetch_data_from_platform(platform, fetch_params)
                fetch_tasks.append((platform, task))

        # 等待所有获取任务完成
        platform_packets = {}
        for platform, task in fetch_tasks:
            try:
                packets = await task
                platform_packets[platform] = packets
            except Exception as e:
                logger.error(f"平台 {platform} 数据获取失败: {e}")
                platform_packets[platform] = []

        # 处理所有数据包
        processing_tasks = []
        for platform, packets in platform_packets.items():
            for packet in packets:
                task = self.process_data_packet(packet)
                processing_tasks.append(task)

        # 等待所有处理任务完成
        if processing_tasks:
            batch_results = await asyncio.gather(*processing_tasks, return_exceptions=True)
            for results in batch_results:
                if isinstance(results, list):
                    all_results.extend(results)

        # 统计结果
        successful_results = [r for r in all_results if r.success]
        failed_results = [r for r in all_results if not r.success]

        return {
            'user_id': user_id,
            'platforms_processed': list(platform_packets.keys()),
            'total_packets': sum(len(packets) for packets in platform_packets.values()),
            'successful_processing': len(successful_results),
            'failed_processing': len(failed_results),
            'results': successful_results,
            'errors': [r.error_message for r in failed_results if r.error_message],
            'average_processing_time': np.mean([r.processing_time for r in successful_results]) if successful_results else 0,
            'quality_report': self.quality_manager.get_quality_report()
        }

    async def _health_check_loop(self):
        """健康检查循环"""
        while self.is_running:
            try:
                for platform_name, adapter in self.adapters.items():
                    is_healthy = await adapter.health_check()
                    if not is_healthy:
                        logger.warning(f"平台 {platform_name} 健康检查失败")
                        adapter.is_connected = False

                await asyncio.sleep(60)  # 每分钟检查一次

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"健康检查异常: {e}")
                await asyncio.sleep(30)

    async def _data_processing_loop(self):
        """数据处理循环"""
        while self.is_running:
            try:
                if self.data_buffer:
                    # 批量处理缓冲区中的数据
                    packets_to_process = []
                    for _ in range(min(10, len(self.data_buffer))):  # 每次处理最多10个
                        if self.data_buffer:
                            packets_to_process.append(self.data_buffer.popleft())

                    if packets_to_process:
                        processing_tasks = [
                            self.process_data_packet(packet)
                            for packet in packets_to_process
                        ]
                        await asyncio.gather(*processing_tasks, return_exceptions=True)

                await asyncio.sleep(1)  # 避免过度占用CPU

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"数据处理循环异常: {e}")
                await asyncio.sleep(5)

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        adapter_status = {}
        for platform_name, adapter in self.adapters.items():
            adapter_status[platform_name] = {
                'connected': adapter.is_connected,
                'last_health_check': adapter.last_health_check.isoformat() if adapter.last_health_check else None
            }

        return {
            'is_running': self.is_running,
            'registered_adapters': list(self.adapters.keys()),
            'adapter_status': adapter_status,
            'processing_stats': dict(self.processing_stats),
            'buffer_size': len(self.data_buffer),
            'quality_report': self.quality_manager.get_quality_report()
        }

# 使用示例
async def main():
    """使用示例"""
    # 创建数据流管理器
    manager = CrossPlatformDataFlowManager()

    # 注册Dify适配器
    dify_config = {
        'base_url': 'https://api.dify.ai',
        'api_key': 'your_dify_api_key',
        'workflow_id': 'your_workflow_id'
    }
    dify_adapter = DifyAdapter('dify', dify_config)
    manager.register_adapter('dify', dify_adapter)

    # 注册API适配器
    api_config = {
        'base_url': 'https://api.example.com',
        'api_key': 'your_api_key',
        'endpoint': '/cgm/data',
        'data_type': 'cgm',
        'field_mapping': {
            'glucose': 'glucose_value',
            'timestamp': 'recorded_at'
        }
    }
    api_adapter = APIAdapter('example_api', api_config)
    manager.register_adapter('example_api', api_adapter)

    # 注册数据处理器
    async def cgm_processor(packet: DataPacket) -> Dict[str, Any]:
        """CGM数据处理器示例"""
        logger.info(f"处理CGM数据包: {packet.id}")
        return {
            'processed_glucose': packet.data.get('glucose', 0),
            'quality': packet.quality.value,
            'source': packet.source_platform
        }

    manager.register_processor('cgm', cgm_processor)

    # 启动系统
    await manager.start()

    try:
        # 批量处理数据
        result = await manager.process_batch_data('user_123', ['dify', 'example_api'])
        print(f"批量处理结果: {result}")

        # 获取系统状态
        status = manager.get_system_status()
        print(f"系统状态: {status}")

        # 运行一段时间
        await asyncio.sleep(60)

    finally:
        # 停止系统
        await manager.stop()

if __name__ == "__main__":
    asyncio.run(main())

__all__ = ["'logger'", "'DataSourceType'", "'DataQuality'", "'DataPacket'", "'ProcessingResult'", "'DataSourceAdapter'", "'DifyAdapter'", "'APIAdapter'", "'DataRouter'", "'DataQualityManager'", "'CrossPlatformDataFlowManager'"]
