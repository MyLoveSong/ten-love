"""
训练模型服务 - 集成TRAIN目录下训练好的模型到项目中
遵循SOLID原则的模块化设计
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import logging
from datetime import datetime
import json
from dataclasses import dataclass
from abc import ABC, abstractmethod
import threading

# 设置日志
logger = logging.getLogger(__name__)

train_path = Path(__file__).resolve().parent.parent.parent.parent / "TRAIN"
if str(train_path) not in sys.path:
    sys.path.insert(0, str(train_path))

try:
    from auto_train import GlucosePredictor
    from utils.training_utils import calculate_metrics
    TRAINED_MODEL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"无法导入训练模型: {e}")
    TRAINED_MODEL_AVAILABLE = False


@dataclass
class ModelMetadata:
    """模型元数据"""
    model_name: str
    model_version: str
    training_date: str
    best_val_loss: float
    test_mae: float
    test_rmse: float
    model_path: str
    config: Dict[str, Any]


@dataclass
class PredictionInput:
    """预测输入数据结构"""
    glucose_sequence: List[float]  # 血糖序列
    timestamps: List[str]          # 时间戳
    user_features: Optional[Dict[str, Any]] = None  # 用户特征
    meal_info: Optional[Dict[str, Any]] = None      # 餐食信息


@dataclass
class PredictionOutput:
    """预测输出数据结构"""
    predictions: List[float]       # 预测值
    prediction_times: List[str]    # 预测时间
    confidence_scores: List[float] # 置信度
    risk_level: str               # 风险等级
    recommendations: List[str]     # 建议
    model_version: str            # 模型版本
    prediction_timestamp: str     # 预测时间戳


class IModelLoader(ABC):
    """模型加载器接口 - 接口分离原则"""

    @abstractmethod
    def load_model(self, model_path: str) -> nn.Module:
        """加载模型"""
        pass

    @abstractmethod
    def get_model_metadata(self, model_path: str) -> ModelMetadata:
        """获取模型元数据"""
        pass


class IPredictor(ABC):
    """预测器接口 - 接口分离原则"""

    @abstractmethod
    def predict(self, input_data: PredictionInput) -> PredictionOutput:
        """执行预测"""
        pass

    @abstractmethod
    def validate_input(self, input_data: PredictionInput) -> bool:
        """验证输入数据"""
        pass


class TrainedModelLoader(IModelLoader):
    """训练模型加载器 - 单一职责原则"""

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"模型加载器初始化，设备: {self.device}")

    def load_model(self, model_path: str) -> nn.Module:
        """
        加载训练好的模型 - 支持不同架构

        Args:
            model_path: 模型文件路径

        Returns:
            加载的模型
        """
        if not TRAINED_MODEL_AVAILABLE:
            raise ImportError("训练模型模块不可用")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        try:
            # 加载检查点
            state_dict = torch.load(model_path, map_location=self.device)

            # 判断是否为完整检查点还是仅权重
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                model_weights = state_dict['model_state_dict']
            else:
                model_weights = state_dict

            # 根据模型路径判断架构类型
            if "stage2" in model_path or "glucose_head" in model_path:
                # 三阶段训练的增强模型
                logger.info("加载三阶段训练的增强模型架构")
                model = GlucosePredictor(
                    input_size=1,
                    hidden_size=128,  # 增强版使用更大的隐藏层
                    num_layers=3,     # 更多层数
                    output_size=6,
                    dropout=0.1
                )
            else:
                # 标准模型架构
                logger.info("加载标准模型架构")
                model = GlucosePredictor(
                    input_size=1,
                    hidden_size=64,
                    num_layers=2,
                    output_size=6,
                    dropout=0.1
                )

            # 尝试加载模型权重
            try:
                model.load_state_dict(model_weights)
                logger.info("✅ 使用匹配的模型架构加载成功")
            except RuntimeError as e:
                logger.warning(f"架构不匹配，尝试标准架构: {e}")
                # 如果加载失败，回退到标准架构
                model = GlucosePredictor(
                    input_size=1,
                    hidden_size=64,
                    num_layers=2,
                    output_size=6,
                    dropout=0.1
                )
                model.load_state_dict(model_weights)
                logger.info("✅ 使用标准架构加载成功")

            model.to(self.device)
            model.eval()

            logger.info(f"成功加载模型: {model_path}")
            return model

        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise

    def get_model_metadata(self, model_path: str) -> ModelMetadata:
        """
        获取模型元数据 - 支持三阶段训练结果

        Args:
            model_path: 模型文件路径

        Returns:
            模型元数据
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        try:
            # 检查是否为三阶段训练模型
            if "stage2" in model_path:
                return self._get_three_stage_metadata(model_path)
            else:
                return self._get_standard_metadata(model_path)

        except Exception as e:
            logger.error(f"获取模型元数据失败: {e}")
            raise

    def _get_three_stage_metadata(self, model_path: str) -> ModelMetadata:
        """获取三阶段训练模型的元数据"""
        model_dir = Path(model_path).parent
        results_file = model_dir / 'training_results.json'

        # 查找三阶段训练报告
        pipeline_report_path = Path(model_path).parent.parent.parent.parent / "TRAIN" / "TRAIN" / "outputs" / "demo_three_stage" / "pipeline_report_20251014_203704.json"

        if pipeline_report_path.exists():
            with open(pipeline_report_path, 'r', encoding='utf-8') as f:
                pipeline_results = json.load(f)

            stage2_results = pipeline_results.get("stage_results", {}).get("stage2", {})
            summary = pipeline_results.get("summary", {})

            return ModelMetadata(
                model_name="Enhanced GlucosePredictor (Three-Stage)",
                model_version="2.0.0",
                training_date=pipeline_results.get("pipeline_info", {}).get("execution_time", ""),
                best_val_loss=stage2_results.get("best_val_loss", summary.get("stage2_val_loss", 0.0)),
                test_mae=summary.get("stage2_test_mae", 0.0),
                test_rmse=0.0,
                model_path=model_path,
                config={
                    "pipeline_type": "three_stage",
                    "stage1_acceptance": summary.get("stage1_acceptance", 0.0),
                    "stage2_config": pipeline_results.get("pipeline_info", {}).get("config", {}).get("stage2", {}),
                    "training_type": "cultural_adaptation + glucose_head_tuning + online_learning"
                }
            )
        elif results_file.exists():
            # 使用阶段2的训练结果
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)

            test_results = results.get('test_results', {})
            return ModelMetadata(
                model_name="Enhanced GlucosePredictor (Stage2)",
                model_version="1.5.0",
                training_date=datetime.fromtimestamp(Path(model_path).stat().st_mtime).isoformat(),
                best_val_loss=results.get('best_val_loss', 0.0),
                test_mae=test_results.get('mae', 0.0),
                test_rmse=test_results.get('rmse', 0.0),
                model_path=model_path,
                config=results.get('config', {})
            )
        else:
            # 默认元数据
            return ModelMetadata(
                model_name="Enhanced GlucosePredictor",
                model_version="1.5.0",
                training_date=datetime.fromtimestamp(Path(model_path).stat().st_mtime).isoformat(),
                best_val_loss=0.0,
                test_mae=0.0,
                test_rmse=0.0,
                model_path=model_path,
                config={"pipeline_type": "stage2_only"}
            )

    def _get_standard_metadata(self, model_path: str) -> ModelMetadata:
        """获取标准训练模型的元数据"""
        checkpoint = torch.load(model_path, map_location='cpu')

        # 查找训练结果文件
        model_dir = Path(model_path).parent
        results_file = model_dir / 'training_results.json'

        if results_file.exists():
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)

            test_results = results.get('test_results', {})
            return ModelMetadata(
                model_name="GlucosePredictor",
                model_version="1.0.0",
                training_date=results.get('training_date', datetime.now().isoformat()),
                best_val_loss=results.get('best_val_loss', 0.0),
                test_mae=test_results.get('mae', 0.0),
                test_rmse=test_results.get('rmse', 0.0),
                model_path=model_path,
                config=results.get('config', {})
            )
        else:
            # 从检查点获取基本信息
            return ModelMetadata(
                model_name="GlucosePredictor",
                model_version="1.0.0",
                training_date=datetime.fromtimestamp(Path(model_path).stat().st_mtime).isoformat(),
                best_val_loss=0.0,
                test_mae=0.0,
                test_rmse=0.0,
                model_path=model_path,
                config={}
            )


class GlucosePredictionService(IPredictor):
    """血糖预测服务 - 单一职责原则"""

    def __init__(
        self,
        model_loader: IModelLoader,
        model_path: Optional[str] = None,
        sequence_length: int = 24,
        prediction_horizon: int = 6
    ):
        self.model_loader = model_loader
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.model: Optional[nn.Module] = None
        self.metadata: Optional[ModelMetadata] = None

        # 自动查找最新模型
        if model_path is None:
            model_path = self._find_latest_model()

        if model_path:
            self.load_model(model_path)

    def _find_latest_model(self) -> Optional[str]:
        """查找最新的训练模型 - 优先三阶段训练结果"""
        project_root = Path(__file__).parent.parent.parent.parent

        # 按优先级搜索模型文件
        search_paths = [
            # 三阶段训练结果（最高优先级）
            project_root / "TRAIN" / "outputs" / "stage2_glucose_head" / "glucose_head_best.pt",
            project_root / "TRAIN" / "outputs" / "stage2_glucose_head" / "best_model.pt",
            project_root / "TRAIN" / "TRAIN" / "outputs" / "stage2_glucose_head" / "glucose_head_best.pt",

            # 单独训练结果
            project_root / "TRAIN" / "outputs" / "glucose_prediction" / "best_model.pt",
        ]

        # 添加通用搜索结果
        train_outputs_dir = project_root / "TRAIN" / "outputs"
        if train_outputs_dir.exists():
            search_paths.extend(list(train_outputs_dir.rglob("*best*.pt")))
            search_paths.extend(list(train_outputs_dir.rglob("best_model.pt")))

        # 查找存在的文件并按修改时间排序
        existing_files = []
        for path in search_paths:
            if path.exists():
                existing_files.append((path, path.stat().st_mtime))

        if not existing_files:
            logger.warning("未找到任何训练好的模型文件")
            return None

        # 按修改时间排序，返回最新的
        latest_model_path, _ = max(existing_files, key=lambda x: x[1])

        # 判断模型类型
        if "stage2" in str(latest_model_path):
            logger.info(f"找到三阶段训练模型: {latest_model_path}")
        else:
            logger.info(f"找到标准训练模型: {latest_model_path}")

        return str(latest_model_path)

    def load_model(self, model_path: str):
        """加载模型"""
        try:
            self.model = self.model_loader.load_model(model_path)
            self.metadata = self.model_loader.get_model_metadata(model_path)
            logger.info(f"模型加载成功: {self.metadata.model_name} v{self.metadata.model_version}")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise

    def validate_input(self, input_data: PredictionInput) -> bool:
        """
        验证输入数据

        Args:
            input_data: 输入数据

        Returns:
            是否有效
        """
        if not input_data.glucose_sequence:
            logger.error("血糖序列为空")
            return False

        if len(input_data.glucose_sequence) < self.sequence_length:
            logger.error(f"血糖序列长度不足，需要至少{self.sequence_length}个数据点")
            return False

        if len(input_data.glucose_sequence) != len(input_data.timestamps):
            logger.error("血糖序列和时间戳长度不匹配")
            return False

        # 检查数值范围
        for glucose in input_data.glucose_sequence:
            if not (2.0 <= glucose <= 30.0):  # 合理的血糖范围 mmol/L
                logger.error(f"血糖值超出合理范围: {glucose}")
                return False

        return True

    def predict(self, input_data: PredictionInput) -> PredictionOutput:
        """
        执行血糖预测

        Args:
            input_data: 输入数据

        Returns:
            预测结果
        """
        if not self.model:
            raise RuntimeError("模型未加载")

        if not self.validate_input(input_data):
            raise ValueError("输入数据验证失败")

        try:
            # 准备输入数据
            glucose_sequence = np.array(input_data.glucose_sequence[-self.sequence_length:])

            # 归一化（简单的min-max归一化）
            glucose_min, glucose_max = 2.0, 20.0
            normalized_sequence = (glucose_sequence - glucose_min) / (glucose_max - glucose_min)

            # 转换为张量
            input_tensor = torch.FloatTensor(normalized_sequence).unsqueeze(0).unsqueeze(-1)
            input_tensor = input_tensor.to(self.model_loader.device)

            # 执行预测
            with torch.no_grad():
                predictions = self.model(input_tensor)
                predictions = predictions.squeeze().cpu().numpy()

            # 反归一化
            denormalized_predictions = predictions * (glucose_max - glucose_min) + glucose_min

            # 生成预测时间
            from datetime import datetime, timedelta
            last_time = datetime.fromisoformat(input_data.timestamps[-1].replace('Z', '+00:00'))
            prediction_times = [
                (last_time + timedelta(hours=i+1)).isoformat()
                for i in range(len(denormalized_predictions))
            ]

            # 计算置信度（简化版本）
            confidence_scores = [0.85] * len(denormalized_predictions)  # 固定置信度

            # 评估风险等级
            risk_level = self._assess_risk_level(denormalized_predictions)

            # 生成建议
            recommendations = self._generate_recommendations(denormalized_predictions, risk_level)

            return PredictionOutput(
                predictions=denormalized_predictions.tolist(),
                prediction_times=prediction_times,
                confidence_scores=confidence_scores,
                risk_level=risk_level,
                recommendations=recommendations,
                model_version=self.metadata.model_version if self.metadata else "unknown",
                prediction_timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            logger.error(f"预测失败: {e}")
            raise

    def _assess_risk_level(self, predictions: np.ndarray) -> str:
        """评估风险等级"""
        max_glucose = np.max(predictions)
        min_glucose = np.min(predictions)

        if max_glucose > 15.0 or min_glucose < 3.0:
            return "高风险"
        elif max_glucose > 12.0 or min_glucose < 4.0:
            return "中风险"
        else:
            return "低风险"

    def _generate_recommendations(self, predictions: np.ndarray, risk_level: str) -> List[str]:
        """生成建议"""
        recommendations = []

        if risk_level == "高风险":
            recommendations.extend([
                "建议立即咨询医生",
                "密切监测血糖变化",
                "避免高糖食物摄入"
            ])
        elif risk_level == "中风险":
            recommendations.extend([
                "注意饮食控制",
                "增加血糖监测频率",
                "适量运动"
            ])
        else:
            recommendations.extend([
                "保持当前生活方式",
                "定期监测血糖",
                "均衡饮食"
            ])

        return recommendations


class TrainedModelServiceFactory:
    """训练模型服务工厂 - 依赖注入原则"""

    @staticmethod
    def create_glucose_prediction_service(
        model_path: Optional[str] = None,
        device: Optional[torch.device] = None
    ) -> GlucosePredictionService:
        """
        创建血糖预测服务

        Args:
            model_path: 模型路径
            device: 计算设备

        Returns:
            血糖预测服务实例
        """
        model_loader = TrainedModelLoader(device)
        return GlucosePredictionService(model_loader, model_path)

    @staticmethod
    def get_available_models() -> List[Dict[str, Any]]:
        """
        获取可用的模型列表

        Returns:
            模型信息列表
        """
        train_output_dir = Path(__file__).parent.parent.parent.parent / "TRAIN" / "outputs"

        if not train_output_dir.exists():
            return []

        models = []
        for model_file in train_output_dir.rglob("best_model.pt"):
            try:
                loader = TrainedModelLoader()
                metadata = loader.get_model_metadata(str(model_file))
                models.append({
                    'name': metadata.model_name,
                    'version': metadata.model_version,
                    'path': metadata.model_path,
                    'training_date': metadata.training_date,
                    'performance': {
                        'val_loss': metadata.best_val_loss,
                        'test_mae': metadata.test_mae,
                        'test_rmse': metadata.test_rmse
                    }
                })
            except Exception as e:
                logger.warning(f"无法读取模型元数据 {model_file}: {e}")

        return models


# 全局服务实例（单例模式）
_glucose_prediction_service: Optional[GlucosePredictionService] = None


def get_glucose_prediction_service() -> GlucosePredictionService:
    """
    获取血糖预测服务实例（单例）

    Returns:
        血糖预测服务实例
    """
    global _glucose_prediction_service

    if _glucose_prediction_service is None:
        try:
            _glucose_prediction_service = TrainedModelServiceFactory.create_glucose_prediction_service()
            logger.info("血糖预测服务初始化成功")
        except Exception as e:
            logger.error(f"血糖预测服务初始化失败: {e}")
            raise

    return _glucose_prediction_service


def create_trained_model_service(model_path: Optional[str] = None) -> GlucosePredictionService:
    """
    创建训练模型服务（工厂函数）

    Args:
        model_path: 模型路径

    Returns:
        血糖预测服务实例
    """
    return TrainedModelServiceFactory.create_glucose_prediction_service(model_path)


# 导出所有必要的类和函数
__all__ = [
    'ModelMetadata',
    'PredictionInput',
    'PredictionOutput',
    'IModelLoader',
    'TrainedModelLoader',
    'IPredictor',
    'GlucosePredictionService',
    'TrainedModelServiceFactory',
    'get_glucose_prediction_service',
    'create_trained_model_service',
    'TRAINED_MODEL_AVAILABLE'
]
