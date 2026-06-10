"""
实时异常检测系统 - 检测血糖异常波动
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from collections import deque
import time
from enum import Enum

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """警报级别"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class AnomalyConfig:
    """异常检测配置"""
    window_size: int = 12  # 时间窗口大小
    threshold_multiplier: float = 2.0  # 阈值倍数
    min_samples: int = 10  # 最小样本数
    update_frequency: int = 1  # 更新频率（分钟）
    alert_cooldown: int = 30  # 警报冷却时间（分钟）

class AlertSystem:
    """警报系统"""

    def __init__(self, config: AnomalyConfig):
        self.config = config
        self.alert_history = deque(maxlen=100)
        self.last_alert_time = {}

        logger.info("警报系统初始化完成")

    def send_alert(self, alert_type: str, level: AlertLevel,
                   message: str, glucose_value: float = None) -> bool:
        """发送警报"""
        current_time = time.time()

        # 检查冷却时间
        if alert_type in self.last_alert_time:
            time_since_last = current_time - self.last_alert_time[alert_type]
            if time_since_last < self.config.alert_cooldown * 60:
                return False

        # 创建警报记录
        alert = {
            'timestamp': current_time,
            'type': alert_type,
            'level': level.value,
            'message': message,
            'glucose_value': glucose_value
        }

        self.alert_history.append(alert)
        self.last_alert_time[alert_type] = current_time

        # 记录日志
        logger.warning(f"警报 [{level.value.upper()}] {alert_type}: {message}")

        return True

    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """获取最近的警报"""
        current_time = time.time()
        cutoff_time = current_time - (hours * 3600)

        recent_alerts = [
            alert for alert in self.alert_history
            if alert['timestamp'] > cutoff_time
        ]

        return recent_alerts

class ThresholdModel:
    """阈值模型"""

    def __init__(self, config: AnomalyConfig):
        self.config = config
        self.baseline_mean = None
        self.baseline_std = None
        self.is_trained = False

    def fit(self, glucose_data: List[float]):
        """训练阈值模型"""
        if len(glucose_data) < self.config.min_samples:
            logger.warning(f"样本数量不足，需要至少{self.config.min_samples}个样本")
            return False

        glucose_array = np.array(glucose_data)
        self.baseline_mean = np.mean(glucose_array)
        self.baseline_std = np.std(glucose_array)
        self.is_trained = True

        logger.info(f"阈值模型训练完成，均值: {self.baseline_mean:.2f}, 标准差: {self.baseline_std:.2f}")
        return True

    def predict_anomaly(self, glucose_value: float) -> Tuple[bool, AlertLevel]:
        """预测异常"""
        if not self.is_trained:
            return False, AlertLevel.LOW

        # 计算Z分数
        z_score = abs(glucose_value - self.baseline_mean) / self.baseline_std

        # 确定警报级别
        if z_score > 3.0:
            return True, AlertLevel.CRITICAL
        elif z_score > 2.5:
            return True, AlertLevel.HIGH
        elif z_score > 2.0:
            return True, AlertLevel.MEDIUM
        elif z_score > 1.5:
            return True, AlertLevel.LOW
        else:
            return False, AlertLevel.LOW

class LSTMAutoencoder(nn.Module):
    """LSTM自编码器用于异常检测"""

    def __init__(self, input_dim: int = 1, hidden_dim: int = 64,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()

        # 编码器
        self.encoder = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout
        )

        # 解码器
        self.decoder = nn.LSTM(
            hidden_dim, input_dim, num_layers,
            batch_first=True, dropout=dropout
        )

        # 输出层
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # 编码
        encoded, (hidden, cell) = self.encoder(x)

        # 解码
        decoded, _ = self.decoder(encoded, (hidden, cell))

        # 输出
        output = self.output_layer(encoded)

        return output, decoded

class AnomalyDetection:
    """异常检测系统"""

    def __init__(self, config: AnomalyConfig):
        self.config = config
        self.alert_system = AlertSystem(config)
        self.threshold_models = {}

        # 初始化LSTM自编码器
        self.lstm_autoencoder = LSTMAutoencoder()
        self.is_lstm_trained = False

        # 数据缓冲区
        self.data_buffer = deque(maxlen=config.window_size * 2)

        logger.info("异常检测系统初始化完成")

    def add_glucose_reading(self, glucose_value: float, timestamp: float = None) -> Dict[str, Any]:
        """添加血糖读数"""
        if timestamp is None:
            timestamp = time.time()

        # 添加到缓冲区
        self.data_buffer.append({
            'value': glucose_value,
            'timestamp': timestamp
        })

        # 检测异常
        anomaly_result = self._detect_anomaly(glucose_value)

        return anomaly_result

    def _detect_anomaly(self, glucose_value: float) -> Dict[str, Any]:
        """检测异常"""
        result = {
            'is_anomaly': False,
            'alert_level': AlertLevel.LOW,
            'anomaly_type': None,
            'confidence': 0.0,
            'message': ''
        }

        # 阈值检测
        threshold_anomaly, threshold_level = self._threshold_detection(glucose_value)

        # LSTM自编码器检测
        lstm_anomaly, lstm_confidence = self._lstm_detection(glucose_value)

        # 趋势检测
        trend_anomaly, trend_level = self._trend_detection(glucose_value)

        # 综合判断
        if threshold_anomaly or lstm_anomaly or trend_anomaly:
            result['is_anomaly'] = True

            # 确定最高警报级别
            levels = [threshold_level, trend_level]
            if threshold_anomaly:
                levels.append(threshold_level)
            if trend_anomaly:
                levels.append(trend_level)

            result['alert_level'] = max(levels, key=lambda x: self._get_level_priority(x))
            result['confidence'] = max(0.5, lstm_confidence)

            # 生成消息
            result['message'] = self._generate_anomaly_message(
                glucose_value, threshold_anomaly, lstm_anomaly, trend_anomaly
            )

            # 发送警报
            self._send_anomaly_alert(result, glucose_value)

        return result

    def _threshold_detection(self, glucose_value: float) -> Tuple[bool, AlertLevel]:
        """阈值检测"""
        if len(self.data_buffer) < self.config.min_samples:
            return False, AlertLevel.LOW

        # 获取历史数据
        historical_values = [item['value'] for item in self.data_buffer]

        # 训练阈值模型
        threshold_model = ThresholdModel(self.config)
        threshold_model.fit(historical_values)

        return threshold_model.predict_anomaly(glucose_value)

    def _lstm_detection(self, glucose_value: float) -> Tuple[bool, float]:
        """LSTM自编码器检测"""
        if len(self.data_buffer) < self.config.window_size:
            return False, 0.0

        # 准备输入数据
        recent_values = [item['value'] for item in list(self.data_buffer)[-self.config.window_size:]]
        input_tensor = torch.tensor(recent_values).unsqueeze(0).unsqueeze(-1).float()

        # 预测
        self.lstm_autoencoder.eval()
        with torch.no_grad():
            output, decoded = self.lstm_autoencoder(input_tensor)

            # 计算重构误差
            reconstruction_error = torch.mean((input_tensor - decoded) ** 2).item()

            # 判断异常
            threshold = 0.1  # 可调整的阈值
            is_anomaly = reconstruction_error > threshold

            # 计算置信度
            confidence = min(1.0, reconstruction_error / threshold)

            return is_anomaly, confidence

    def _trend_detection(self, glucose_value: float) -> Tuple[bool, AlertLevel]:
        """趋势检测"""
        if len(self.data_buffer) < 3:
            return False, AlertLevel.LOW

        # 获取最近3个值
        recent_values = [item['value'] for item in list(self.data_buffer)[-3:]]

        # 计算趋势
        trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]

        # 判断趋势异常
        if abs(trend) > 2.0:  # 快速变化
            if trend > 0:
                return True, AlertLevel.HIGH  # 快速上升
            else:
                return True, AlertLevel.MEDIUM  # 快速下降
        elif abs(trend) > 1.0:  # 中等变化
            return True, AlertLevel.LOW

        return False, AlertLevel.LOW

    def _get_level_priority(self, level: AlertLevel) -> int:
        """获取警报级别优先级"""
        priority_map = {
            AlertLevel.LOW: 1,
            AlertLevel.MEDIUM: 2,
            AlertLevel.HIGH: 3,
            AlertLevel.CRITICAL: 4
        }
        return priority_map.get(level, 0)

    def _generate_anomaly_message(self, glucose_value: float,
                                threshold_anomaly: bool,
                                lstm_anomaly: bool,
                                trend_anomaly: bool) -> str:
        """生成异常消息"""
        message_parts = []

        if threshold_anomaly:
            if glucose_value > 10.0:
                message_parts.append("血糖值过高")
            elif glucose_value < 3.0:
                message_parts.append("血糖值过低")
            else:
                message_parts.append("血糖值异常")

        if trend_anomaly:
            message_parts.append("血糖变化趋势异常")

        if lstm_anomaly:
            message_parts.append("血糖模式异常")

        if message_parts:
            return f"检测到异常：{', '.join(message_parts)}"
        else:
            return "血糖正常"

    def _send_anomaly_alert(self, result: Dict[str, Any], glucose_value: float):
        """发送异常警报"""
        alert_type = f"glucose_anomaly_{result['alert_level'].value}"

        self.alert_system.send_alert(
            alert_type=alert_type,
            level=result['alert_level'],
            message=result['message'],
            glucose_value=glucose_value
        )

    def real_time_monitoring(self, glucose_data_stream: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """实时监控血糖数据流"""
        results = []

        for data_point in glucose_data_stream:
            glucose_value = data_point.get('glucose_value', 0)
            timestamp = data_point.get('timestamp', time.time())

            # 添加读数并检测异常
            result = self.add_glucose_reading(glucose_value, timestamp)
            result['timestamp'] = timestamp
            result['glucose_value'] = glucose_value

            results.append(result)

        return results

    def get_monitoring_summary(self, hours: int = 24) -> Dict[str, Any]:
        """获取监控摘要"""
        current_time = time.time()
        cutoff_time = current_time - (hours * 3600)

        # 获取时间窗口内的数据
        recent_data = [
            item for item in self.data_buffer
            if item['timestamp'] > cutoff_time
        ]

        if not recent_data:
            return {'status': 'no_data'}

        # 统计信息
        glucose_values = [item['value'] for item in recent_data]

        summary = {
            'total_readings': len(recent_data),
            'avg_glucose': np.mean(glucose_values),
            'min_glucose': np.min(glucose_values),
            'max_glucose': np.max(glucose_values),
            'std_glucose': np.std(glucose_values),
            'anomaly_count': sum(1 for item in recent_data if item.get('is_anomaly', False)),
            'recent_alerts': self.alert_system.get_recent_alerts(hours)
        }

        return summary

    def train_lstm_model(self, training_data: List[float]):
        """训练LSTM自编码器"""
        if len(training_data) < self.config.window_size * 10:
            logger.warning("训练数据不足")
            return False

        # 准备训练数据
        sequences = []
        for i in range(len(training_data) - self.config.window_size + 1):
            sequence = training_data[i:i + self.config.window_size]
            sequences.append(sequence)

        # 转换为张量
        X = torch.tensor(sequences).unsqueeze(-1).float()

        # 训练模型
        optimizer = torch.optim.Adam(self.lstm_autoencoder.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        self.lstm_autoencoder.train()
        for epoch in range(50):  # 简化训练
            optimizer.zero_grad()
            output, decoded = self.lstm_autoencoder(X)
            loss = criterion(decoded, X)
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                logger.info(f"LSTM训练轮次 {epoch}, 损失: {loss.item():.4f}")

        self.is_lstm_trained = True
        logger.info("LSTM自编码器训练完成")
        return True
