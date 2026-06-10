"""
多源数据融合框架和语义对齐模块
实现跨模态语义对齐和多源异构数据整合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json

logger = logging.getLogger(__name__)

@dataclass
class DataSource:
    """数据源定义"""
    source_id: str
    source_type: str  # 'clinical', 'nutrition', 'behavioral', 'environmental'
    data_format: str  # 'structured', 'text', 'image', 'time_series'
    sampling_rate: Optional[float] = None
    quality_score: float = 1.0

@dataclass
class SemanticAlignment:
    """语义对齐结果"""
    source_a: str
    source_b: str
    alignment_score: float
    semantic_mapping: Dict[str, str]
    confidence: float

class DataPreprocessor(ABC):
    """数据预处理器抽象基类"""

    @abstractmethod
    def preprocess(self, data: Any) -> torch.Tensor:
        """预处理数据"""
        pass

    @abstractmethod
    def get_feature_dim(self) -> int:
        """获取特征维度"""
        pass

class ClinicalDataPreprocessor(DataPreprocessor):
    """临床数据预处理器"""

    def __init__(self):
        self.feature_names = [
            'glucose_level', 'hba1c', 'bmi', 'blood_pressure_systolic',
            'blood_pressure_diastolic', 'age', 'gender', 'diabetes_duration'
        ]
        self.normalizers = {
            'glucose_level': (5.5, 2.0),  # (mean, std)
            'hba1c': (7.0, 1.5),
            'bmi': (25.0, 5.0),
            'blood_pressure_systolic': (120, 20),
            'blood_pressure_diastolic': (80, 10),
            'age': (50, 15),
            'diabetes_duration': (5, 3)
        }

    def preprocess(self, data: Dict[str, float]) -> torch.Tensor:
        """预处理临床数据"""
        features = []

        for feature_name in self.feature_names:
            if feature_name in data:
                value = data[feature_name]

                # 特殊处理性别
                if feature_name == 'gender':
                    value = 1.0 if value == 'male' else 0.0
                else:
                    # 标准化数值特征
                    if feature_name in self.normalizers:
                        mean, std = self.normalizers[feature_name]
                        value = (value - mean) / std

                features.append(value)
            else:
                features.append(0.0)  # 缺失值填充

        return torch.tensor(features, dtype=torch.float32)

    def get_feature_dim(self) -> int:
        return len(self.feature_names)

class NutritionDataPreprocessor(DataPreprocessor):
    """营养数据预处理器"""

    def __init__(self):
        self.nutrient_names = [
            'calories', 'carbohydrates', 'protein', 'fat', 'fiber',
            'sugar', 'sodium', 'cholesterol', 'gi_value', 'gl_value'
        ]

    def preprocess(self, data: Dict[str, float]) -> torch.Tensor:
        """预处理营养数据"""
        features = []

        for nutrient in self.nutrient_names:
            value = data.get(nutrient, 0.0)

            # 营养素特定的归一化
            if nutrient == 'calories':
                value = value / 2000.0  # 基于2000卡路里标准
            elif nutrient in ['carbohydrates', 'protein', 'fat']:
                value = value / 100.0  # 基于100g标准
            elif nutrient == 'gi_value':
                value = value / 100.0  # GI值0-100
            elif nutrient == 'sodium':
                value = value / 2300.0  # 基于每日推荐摄入量

            features.append(value)

        return torch.tensor(features, dtype=torch.float32)

    def get_feature_dim(self) -> int:
        return len(self.nutrient_names)

class BehavioralDataPreprocessor(DataPreprocessor):
    """行为数据预处理器"""

    def __init__(self):
        self.behavior_features = [
            'exercise_duration', 'exercise_intensity', 'sleep_hours',
            'stress_level', 'meal_timing', 'medication_adherence'
        ]

    def preprocess(self, data: Dict[str, float]) -> torch.Tensor:
        """预处理行为数据"""
        features = []

        for feature in self.behavior_features:
            value = data.get(feature, 0.0)

            # 行为特征归一化
            if feature == 'exercise_duration':
                value = min(value / 120.0, 1.0)  # 最大2小时
            elif feature == 'exercise_intensity':
                value = value / 10.0  # 强度1-10
            elif feature == 'sleep_hours':
                value = value / 12.0  # 最大12小时
            elif feature == 'stress_level':
                value = value / 10.0  # 压力级别1-10
            elif feature == 'medication_adherence':
                value = value / 100.0  # 依从性百分比

            features.append(value)

        return torch.tensor(features, dtype=torch.float32)

    def get_feature_dim(self) -> int:
        return len(self.behavior_features)

class SemanticAlignmentModule(nn.Module):
    """语义对齐模块"""

    def __init__(self, source_dims: Dict[str, int], alignment_dim: int = 128):
        super().__init__()
        self.source_dims = source_dims
        self.alignment_dim = alignment_dim

        # 为每个数据源创建投影层
        self.projectors = nn.ModuleDict()
        for source_name, dim in source_dims.items():
            self.projectors[source_name] = nn.Sequential(
                nn.Linear(dim, alignment_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(alignment_dim, alignment_dim)
            )

        # 语义对齐网络
        self.alignment_network = nn.Sequential(
            nn.Linear(alignment_dim * 2, alignment_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(alignment_dim, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        source_data: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        语义对齐前向传播
        Args:
            source_data: 各数据源的特征字典
        Returns:
            对齐后的特征和对齐分数
        """
        # 投影到统一语义空间
        aligned_features = {}
        for source_name, data in source_data.items():
            if source_name in self.projectors:
                aligned_features[source_name] = self.projectors[source_name](data)

        # 计算两两对齐分数
        alignment_scores = {}
        source_names = list(aligned_features.keys())

        for i, source_a in enumerate(source_names):
            for j, source_b in enumerate(source_names[i+1:], i+1):
                # 拼接特征
                combined = torch.cat([
                    aligned_features[source_a],
                    aligned_features[source_b]
                ], dim=-1)

                # 计算对齐分数
                score = self.alignment_network(combined)
                alignment_scores[f"{source_a}_{source_b}"] = score

        return {
            'aligned_features': aligned_features,
            'alignment_scores': alignment_scores
        }

class TemporalAlignmentModule(nn.Module):
    """时序对齐模块"""

    def __init__(self, feature_dim: int, max_seq_len: int = 100):
        super().__init__()
        self.feature_dim = feature_dim
        self.max_seq_len = max_seq_len

        # 时序编码器
        self.temporal_encoder = nn.LSTM(
            input_size=feature_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=True
        )

        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=256,  # bidirectional LSTM output
            num_heads=8,
            batch_first=True
        )

        # 时序对齐网络
        self.alignment_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )

    def forward(
        self,
        temporal_data: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        时序对齐
        Args:
            temporal_data: 时序数据 [batch_size, seq_len, feature_dim]
            timestamps: 时间戳 [batch_size, seq_len]
        Returns:
            对齐后的特征和注意力权重
        """
        batch_size, seq_len, feature_dim = temporal_data.shape

        # LSTM编码
        lstm_output, _ = self.temporal_encoder(temporal_data)

        # 自注意力
        attended_output, attention_weights = self.attention(
            lstm_output, lstm_output, lstm_output
        )

        # 对齐投影
        aligned_output = self.alignment_layer(attended_output)

        return aligned_output, attention_weights

class MultiSourceDataFusion(nn.Module):
    """多源数据融合网络"""

    def __init__(
        self,
        source_dims: Dict[str, int],
        fusion_dim: int = 256,
        output_dim: int = 128
    ):
        super().__init__()
        self.source_dims = source_dims
        self.fusion_dim = fusion_dim

        # 语义对齐模块
        self.semantic_alignment = SemanticAlignmentModule(source_dims, fusion_dim // 2)

        # 时序对齐模块
        self.temporal_alignment = TemporalAlignmentModule(fusion_dim // 2)

        # 融合网络
        total_aligned_dim = len(source_dims) * (fusion_dim // 2)
        self.fusion_network = nn.Sequential(
            nn.Linear(total_aligned_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, output_dim)
        )

        # 质量评估网络
        self.quality_assessor = nn.Sequential(
            nn.Linear(output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        multi_source_data: Dict[str, torch.Tensor],
        data_quality_scores: Optional[Dict[str, float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        多源数据融合
        Args:
            multi_source_data: 多源数据字典
            data_quality_scores: 数据质量分数
        Returns:
            融合结果
        """
        # 语义对齐
        alignment_results = self.semantic_alignment(multi_source_data)
        aligned_features = alignment_results['aligned_features']

        # 拼接对齐后的特征
        feature_list = []
        for source_name in self.source_dims.keys():
            if source_name in aligned_features:
                feature_list.append(aligned_features[source_name])

        if not feature_list:
            raise ValueError("No valid source data provided")

        concatenated_features = torch.cat(feature_list, dim=-1)

        # 数据融合
        fused_features = self.fusion_network(concatenated_features)

        # 质量评估
        quality_score = self.quality_assessor(fused_features)

        return {
            'fused_features': fused_features,
            'quality_score': quality_score,
            'aligned_features': aligned_features,
            'alignment_scores': alignment_results['alignment_scores']
        }

class DataIntegrationService:
    """数据集成服务"""

    def __init__(self):
        # 数据预处理器
        self.preprocessors = {
            'clinical': ClinicalDataPreprocessor(),
            'nutrition': NutritionDataPreprocessor(),
            'behavioral': BehavioralDataPreprocessor()
        }

        # 获取各数据源维度
        source_dims = {
            name: processor.get_feature_dim()
            for name, processor in self.preprocessors.items()
        }

        # 多源数据融合网络
        self.fusion_network = MultiSourceDataFusion(source_dims)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fusion_network.to(self.device)

        # 数据源注册表
        self.data_sources = {}

    def register_data_source(self, data_source: DataSource):
        """注册数据源"""
        self.data_sources[data_source.source_id] = data_source
        logger.info(f"注册数据源: {data_source.source_id} ({data_source.source_type})")

    def preprocess_multi_source_data(
        self,
        raw_data: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """预处理多源数据"""
        processed_data = {}

        for source_type, data in raw_data.items():
            if source_type in self.preprocessors:
                processed_tensor = self.preprocessors[source_type].preprocess(data)
                processed_data[source_type] = processed_tensor.unsqueeze(0).to(self.device)
            else:
                logger.warning(f"未找到数据源 {source_type} 的预处理器")

        return processed_data

    def integrate_data(
        self,
        raw_data: Dict[str, Any],
        quality_scores: Optional[Dict[str, float]] = None
    ) -> Dict[str, Union[torch.Tensor, float, Dict]]:
        """集成多源数据"""
        # 预处理数据
        processed_data = self.preprocess_multi_source_data(raw_data)

        if not processed_data:
            raise ValueError("没有有效的数据源")

        # 数据融合
        self.fusion_network.eval()
        with torch.no_grad():
            fusion_results = self.fusion_network(processed_data, quality_scores)

        # 计算整体数据质量
        overall_quality = self._calculate_overall_quality(
            fusion_results['quality_score'], quality_scores
        )

        # 生成集成报告
        integration_report = self._generate_integration_report(
            processed_data, fusion_results, overall_quality
        )

        return {
            'integrated_features': fusion_results['fused_features'],
            'quality_score': overall_quality,
            'alignment_scores': fusion_results['alignment_scores'],
            'integration_report': integration_report,
            'source_contributions': self._calculate_source_contributions(
                fusion_results['aligned_features']
            )
        }

    def _calculate_overall_quality(
        self,
        model_quality: torch.Tensor,
        input_quality: Optional[Dict[str, float]]
    ) -> float:
        """计算整体数据质量"""
        model_score = model_quality.item()

        if input_quality:
            input_avg = np.mean(list(input_quality.values()))
            overall_quality = 0.6 * model_score + 0.4 * input_avg
        else:
            overall_quality = model_score

        return float(overall_quality)

    def _generate_integration_report(
        self,
        processed_data: Dict[str, torch.Tensor],
        fusion_results: Dict[str, torch.Tensor],
        overall_quality: float
    ) -> Dict[str, Any]:
        """生成集成报告"""
        report = {
            'data_sources': list(processed_data.keys()),
            'integration_quality': overall_quality,
            'feature_dimensions': {
                source: tensor.shape[-1]
                for source, tensor in processed_data.items()
            },
            'fusion_dimension': fusion_results['fused_features'].shape[-1],
            'alignment_summary': {}
        }

        # 对齐分数摘要
        for alignment_pair, score in fusion_results['alignment_scores'].items():
            report['alignment_summary'][alignment_pair] = {
                'score': score.item(),
                'quality': 'good' if score.item() > 0.7 else 'moderate' if score.item() > 0.5 else 'poor'
            }

        return report

    def _calculate_source_contributions(
        self,
        aligned_features: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """计算各数据源的贡献度"""
        contributions = {}

        # 基于特征范数计算贡献度
        total_norm = 0
        source_norms = {}

        for source, features in aligned_features.items():
            norm = torch.norm(features).item()
            source_norms[source] = norm
            total_norm += norm

        # 归一化贡献度
        for source, norm in source_norms.items():
            contributions[source] = norm / total_norm if total_norm > 0 else 0.0

        return contributions

    def validate_data_quality(
        self,
        integrated_data: Dict[str, torch.Tensor]
    ) -> Dict[str, Union[bool, float, List[str]]]:
        """验证数据质量"""
        quality_checks = {
            'completeness': True,
            'consistency': True,
            'quality_score': 0.0,
            'issues': []
        }

        # 检查数据完整性
        for source, data in integrated_data.items():
            if torch.isnan(data).any():
                quality_checks['completeness'] = False
                quality_checks['issues'].append(f"{source}数据包含NaN值")

            if torch.isinf(data).any():
                quality_checks['consistency'] = False
                quality_checks['issues'].append(f"{source}数据包含无穷值")

        # 计算质量分数
        if quality_checks['completeness'] and quality_checks['consistency']:
            quality_checks['quality_score'] = 1.0
        elif quality_checks['completeness'] or quality_checks['consistency']:
            quality_checks['quality_score'] = 0.5
        else:
            quality_checks['quality_score'] = 0.0

        return quality_checks

def create_data_integration_service() -> DataIntegrationService:
    """创建数据集成服务"""
    service = DataIntegrationService()

    # 注册默认数据源
    service.register_data_source(DataSource(
        source_id="clinical_db",
        source_type="clinical",
        data_format="structured",
        quality_score=0.9
    ))

    service.register_data_source(DataSource(
        source_id="nutrition_db",
        source_type="nutrition",
        data_format="structured",
        quality_score=0.8
    ))

    service.register_data_source(DataSource(
        source_id="behavioral_tracker",
        source_type="behavioral",
        data_format="time_series",
        sampling_rate=0.1,  # 每10秒一个数据点
        quality_score=0.7
    ))

    return service

if __name__ == "__main__":
    # 创建服务
    service = create_data_integration_service()

    # 模拟多源数据
    raw_data = {
        'clinical': {
            'glucose_level': 7.2,
            'hba1c': 7.5,
            'bmi': 26.5,
            'blood_pressure_systolic': 135,
            'blood_pressure_diastolic': 85,
            'age': 45,
            'gender': 'male',
            'diabetes_duration': 3
        },
        'nutrition': {
            'calories': 350,
            'carbohydrates': 45,
            'protein': 20,
            'fat': 15,
            'fiber': 5,
            'gi_value': 65
        },
        'behavioral': {
            'exercise_duration': 30,
            'exercise_intensity': 6,
            'sleep_hours': 7,
            'stress_level': 4,
            'medication_adherence': 85
        }
    }

    # 数据集成
    integration_result = service.integrate_data(raw_data)

    print("数据集成结果:")
    print(f"集成特征维度: {integration_result['integrated_features'].shape}")
    print(f"整体质量分数: {integration_result['quality_score']:.3f}")
    print(f"数据源贡献度: {integration_result['source_contributions']}")
    print(f"对齐分数: {integration_result['integration_report']['alignment_summary']}")
    print("多源数据融合系统创建成功！")
