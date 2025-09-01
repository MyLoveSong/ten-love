#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能监控配置
Performance Monitoring Configuration
"""

import time
import logging
import psutil
import threading
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import os

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/system.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """系统指标数据类"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used: int
    memory_total: int
    disk_usage: float
    network_io: Dict[str, int]
    active_connections: int
    api_response_time: float
    model_inference_time: float
    error_count: int
    request_count: int

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, log_interval: int = 60):
        self.log_interval = log_interval
        self.metrics_history = []
        self.is_monitoring = False
        self.monitor_thread = None
        self.start_time = time.time()
        
        # 创建日志目录
        os.makedirs('logs', exist_ok=True)
        
    def start_monitoring(self):
        """开始监控"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            logger.info("性能监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("性能监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # 保持历史记录在合理范围内
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-500:]
                
                # 检查性能阈值
                self._check_performance_thresholds(metrics)
                
                # 记录到文件
                self._log_metrics(metrics)
                
                time.sleep(self.log_interval)
                
            except Exception as e:
                logger.error(f"监控循环出错: {str(e)}")
                time.sleep(10)
    
    def _collect_metrics(self) -> SystemMetrics:
        """收集系统指标"""
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # 内存使用情况
        memory = psutil.virtual_memory()
        
        # 磁盘使用情况
        disk = psutil.disk_usage('/')
        
        # 网络IO
        network = psutil.net_io_counters()
        
        # 当前时间
        timestamp = time.time()
        
        return SystemMetrics(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used=memory.used,
            memory_total=memory.total,
            disk_usage=disk.percent,
            network_io={
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv
            },
            active_connections=0,  # 需要从应用层获取
            api_response_time=0,   # 需要从应用层获取
            model_inference_time=0, # 需要从应用层获取
            error_count=0,         # 需要从应用层获取
            request_count=0        # 需要从应用层获取
        )
    
    def _check_performance_thresholds(self, metrics: SystemMetrics):
        """检查性能阈值"""
        warnings = []
        
        # CPU使用率检查
        if metrics.cpu_percent > 80:
            warnings.append(f"CPU使用率过高: {metrics.cpu_percent}%")
        
        # 内存使用率检查
        if metrics.memory_percent > 85:
            warnings.append(f"内存使用率过高: {metrics.memory_percent}%")
        
        # 磁盘使用率检查
        if metrics.disk_usage > 90:
            warnings.append(f"磁盘使用率过高: {metrics.disk_usage}%")
        
        # API响应时间检查
        if metrics.api_response_time > 5.0:
            warnings.append(f"API响应时间过长: {metrics.api_response_time}s")
        
        # 模型推理时间检查
        if metrics.model_inference_time > 2.0:
            warnings.append(f"模型推理时间过长: {metrics.model_inference_time}s")
        
        # 错误率检查
        if metrics.request_count > 0 and metrics.error_count / metrics.request_count > 0.05:
            warnings.append(f"错误率过高: {metrics.error_count/metrics.request_count*100:.2f}%")
        
        # 记录警告
        for warning in warnings:
            logger.warning(warning)
    
    def _log_metrics(self, metrics: SystemMetrics):
        """记录指标到文件"""
        log_entry = {
            'timestamp': datetime.fromtimestamp(metrics.timestamp).isoformat(),
            'cpu_percent': metrics.cpu_percent,
            'memory_percent': metrics.memory_percent,
            'memory_used_mb': metrics.memory_used // (1024 * 1024),
            'memory_total_mb': metrics.memory_total // (1024 * 1024),
            'disk_usage': metrics.disk_usage,
            'network_io': metrics.network_io,
            'api_response_time': metrics.api_response_time,
            'model_inference_time': metrics.model_inference_time,
            'error_count': metrics.error_count,
            'request_count': metrics.request_count
        }
        
        with open('logs/metrics.json', 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def get_system_summary(self) -> Dict[str, Any]:
        """获取系统摘要"""
        if not self.metrics_history:
            return {}
        
        recent_metrics = self.metrics_history[-10:]  # 最近10个指标
        
        return {
            'uptime_seconds': time.time() - self.start_time,
            'avg_cpu_percent': sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics),
            'avg_memory_percent': sum(m.memory_percent for m in recent_metrics) / len(recent_metrics),
            'avg_api_response_time': sum(m.api_response_time for m in recent_metrics) / len(recent_metrics),
            'avg_model_inference_time': sum(m.model_inference_time for m in recent_metrics) / len(recent_metrics),
            'total_requests': sum(m.request_count for m in recent_metrics),
            'total_errors': sum(m.error_count for m in recent_metrics),
            'error_rate': sum(m.error_count for m in recent_metrics) / max(sum(m.request_count for m in recent_metrics), 1) * 100
        }

class APIMetrics:
    """API指标收集器"""
    
    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.response_times = []
        self.inference_times = []
        self.lock = threading.Lock()
    
    def record_request(self, response_time: float, inference_time: float = 0, is_error: bool = False):
        """记录请求指标"""
        with self.lock:
            self.request_count += 1
            if is_error:
                self.error_count += 1
            
            self.response_times.append(response_time)
            if inference_time > 0:
                self.inference_times.append(inference_time)
            
            # 保持历史记录在合理范围内
            if len(self.response_times) > 1000:
                self.response_times = self.response_times[-500:]
            if len(self.inference_times) > 1000:
                self.inference_times = self.inference_times[-500:]
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取当前指标"""
        with self.lock:
            return {
                'request_count': self.request_count,
                'error_count': self.error_count,
                'avg_response_time': sum(self.response_times) / len(self.response_times) if self.response_times else 0,
                'avg_inference_time': sum(self.inference_times) / len(self.inference_times) if self.inference_times else 0,
                'error_rate': self.error_count / max(self.request_count, 1) * 100
            }

# 全局监控实例
performance_monitor = PerformanceMonitor()
api_metrics = APIMetrics()

def start_monitoring():
    """启动监控"""
    performance_monitor.start_monitoring()

def stop_monitoring():
    """停止监控"""
    performance_monitor.stop_monitoring()

def get_system_status() -> Dict[str, Any]:
    """获取系统状态"""
    return {
        'system_summary': performance_monitor.get_system_summary(),
        'api_metrics': api_metrics.get_metrics(),
        'is_monitoring': performance_monitor.is_monitoring
    }

if __name__ == "__main__":
    # 测试监控功能
    start_monitoring()
    
    try:
        # 模拟一些API调用
        for i in range(10):
            api_metrics.record_request(
                response_time=0.1 + i * 0.05,
                inference_time=0.05 + i * 0.02,
                is_error=(i % 10 == 0)
            )
            time.sleep(1)
        
        # 打印系统状态
        print(json.dumps(get_system_status(), indent=2))
        
    finally:
        stop_monitoring() 