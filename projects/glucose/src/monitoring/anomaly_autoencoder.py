"""
异常检测模块
使用LSTM自编码器进行葡萄糖监测的实时异常检测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from collections import deque
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AnomalyAlert:
    """异常警报数据结构"""
    timestamp: datetime
    severity: str
    anomaly_score: float
    glucose_value: float
    alert_type: str
    message: str


class GlucoseAnomalyDetector:
    """葡萄糖异常检测系统"""

    def __init__(self, sequence_length: int = 12,
                 normal_threshold: float = 0.1,
                 anomaly_threshold: float = 0.3,
                 critical_threshold: float = 0.5,
                 window_size: int = 100,
                 device: Optional[torch.device] = None):

        self.sequence_length = sequence_length
        self.normal_threshold = normal_threshold
        self.anomaly_threshold = anomaly_threshold
        self.critical_threshold = critical_threshold
        self.window_size = window_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化自编码器
        self.autoencoder = LSTMAnomalyAutoencoder(
            sequence_length=sequence_length
        )

        self.is_trained = False
        self.error_history = deque(maxlen=window_size)
        self.alert_history: List[AnomalyAlert] = []

        # 临床阈值
        self.clinical_thresholds = {
            'severe_low': 54.0,
            'low': 70.0,
            'high': 180.0,
            'severe_high': 250.0,
            'rapid_change': 20.0
        }

    def train(self, normal_sequences: np.ndarray, epochs: int = 100, batch_size: int = 32) -> Dict:
        """训练自编码器"""
        logger.info("训练葡萄糖异常检测自编码器...")

        if isinstance(normal_sequences, np.ndarray):
            normal_sequences = torch.FloatTensor(normal_sequences).to(self.device)

        dataset = torch.utils.data.TensorDataset(normal_sequences)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.autoencoder.parameters(), lr=0.001, weight_decay=1e-5)

        self.autoencoder.train()

        for epoch in range(epochs):
            epoch_loss = 0.0

            for batch in dataloader:
                sequences = batch[0]
                optimizer.zero_grad()

                reconstructed, latent = self.autoencoder(sequences)
                loss = F.mse_loss(reconstructed, sequences)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()

            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}: Loss = {epoch_loss / len(dataloader):.6f}")

        self.is_trained = True
        logger.info("自编码器训练完成")

        return {'training_completed': True}

    def detect(self, glucose_sequence: Union[np.ndarray, List[float]],
               timestamp: Optional[datetime] = None) -> AnomalyAlert:
        """检测葡萄糖序列中的异常"""
        if not self.is_trained:
            raise RuntimeError("自编码器必须在异常检测前训练")

        if timestamp is None:
            timestamp = datetime.now()

        if len(glucose_sequence) < self.sequence_length:
            last_value = glucose_sequence[-1] if glucose_sequence else 100.0
            glucose_sequence = list(glucose_sequence) + [last_value] * (self.sequence_length - len(glucose_sequence))
        elif len(glucose_sequence) > self.sequence_length:
            glucose_sequence = glucose_sequence[-self.sequence_length:]

        sequence_tensor = torch.FloatTensor(glucose_sequence).reshape(1, -1, 1).to(self.device)

        self.autoencoder.eval()
        with torch.no_grad():
            reconstructed, _ = self.autoencoder(sequence_tensor)
            mse = F.mse_loss(reconstructed, sequence_tensor, reduction='none')
            reconstruction_error = torch.mean(mse).item()

        current_glucose = glucose_sequence[-1]
        anomaly_score = reconstruction_error

        # 严重程度判断
        if current_glucose < self.clinical_thresholds['severe_low'] or \
           current_glucose > self.clinical_thresholds['severe_high']:
            severity = 'critical'
        elif current_glucose < self.clinical_thresholds['low'] or \
             current_glucose > self.clinical_thresholds['high']:
            severity = 'high'
        elif anomaly_score > self.normal_threshold:
            severity = 'medium'
        else:
            severity = 'low'

        # 警报类型判断
        if current_glucose < self.clinical_thresholds['severe_low']:
            alert_type = 'severe_hypoglycemia'
        elif current_glucose < self.clinical_thresholds['low']:
            alert_type = 'hypoglycemia'
        elif current_glucose > self.clinical_thresholds['severe_high']:
            alert_type = 'severe_hyperglycemia'
        elif current_glucose > self.clinical_thresholds['high']:
            alert_type = 'hyperglycemia'
        elif len(glucose_sequence) >= 2 and abs(glucose_sequence[-1] - glucose_sequence[-2]) > self.clinical_thresholds['rapid_change']:
            alert_type = 'rapid_change'
        else:
            alert_type = 'anomaly_pattern'

        message = f"检测到{alert_type}警报: {current_glucose:.1f} mg/dL (分数: {anomaly_score:.3f})"

        alert = AnomalyAlert(
            timestamp=timestamp,
            severity=severity,
            anomaly_score=anomaly_score,
            glucose_value=current_glucose,
            alert_type=alert_type,
            message=message
        )

        self.error_history.append(reconstruction_error)
        self.alert_history.append(alert)

        # 保留最近24小时的警报
        cutoff_time = timestamp - timedelta(hours=24)
        self.alert_history = [a for a in self.alert_history if a.timestamp > cutoff_time]

        return alert

    def get_statistics(self) -> Dict:
        """获取异常检测统计信息"""
        recent_alerts = self.get_recent_alerts(24)

        severity_counts = {}
        for alert in recent_alerts:
            severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1

        return {
            'total_alerts_24h': len(recent_alerts),
            'severity_distribution': severity_counts,
            'is_trained': self.is_trained
        }

    def get_recent_alerts(self, hours: int = 24) -> List[AnomalyAlert]:
        """获取最近指定时间范围内的警报"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert.timestamp > cutoff_time]


class LSTMAnomalyAutoencoder(nn.Module):
    """基于LSTM的自编码器用于葡萄糖异常检测"""

    def __init__(self, input_dim: int = 1, hidden_dim: int = 64,
                 num_layers: int = 2, dropout: float = 0.1,
                 sequence_length: int = 12):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.sequence_length = sequence_length

        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0)

        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0)

        self.output_projection = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        encoded, _ = self.encoder_lstm(x)
        decoded, _ = self.decoder_lstm(encoded)
        reconstructed = self.output_projection(decoded)
        latent = encoded[:, -1, :]
        return reconstructed, latent
