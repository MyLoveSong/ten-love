

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像识别器模块
基于深度学习的图像分类和识别功能
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import cv2
from app.PIL import Image
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import logging
from datetime import datetime

# 导入基类
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from modules.base_predictor import BasePredictor, register_predictor

logger = logging.getLogger(__name__)

class ImageRecognitionModel(nn.Module):
    """图像识别模型"""

    def __init__(self, num_classes: int = 10, model_type: str = "resnet18", pretrained: bool = True):
        super().__init__()

        self.model_type = model_type
        self.num_classes = num_classes

        # 选择预训练模型
        if model_type == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif model_type == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        elif model_type == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            feature_dim = 1280
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

        # 修改分类头
        if hasattr(self.backbone, 'classifier'):
            # EfficientNet
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(feature_dim, num_classes)
            )
        else:
            # ResNet
            self.backbone.fc = nn.Linear(feature_dim, num_classes)

        # 添加注意力机制
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, feature_dim),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入图像张量 [batch_size, 3, height, width]

        Returns:
            torch.Tensor: 分类结果 [batch_size, num_classes]
        """
        # 提取特征
        if hasattr(self.backbone, 'classifier'):
            # EfficientNet
            x = self.backbone.features(x)
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
            features = x
            x = self.backbone.classifier(x)
        else:
            # ResNet
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)

            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)

            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
            features = x
            x = self.backbone.fc(x)

        # 应用注意力机制
        attention_weights = self.attention(features)
        attended_features = features * attention_weights

        return x, attended_features

@register_predictor("image_predictor")
class ImagePredictor(BasePredictor):
    """图像识别器"""

    def __init__(self, model_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        初始化图像识别器

        Args:
            model_path: 模型文件路径
            config: 配置参数
        """
        super().__init__(model_path, config)

        # 默认配置
        self.default_config = {
            "model_type": "resnet18",
            "num_classes": 10,
            "pretrained": True,
            "image_size": 224,
            "class_names": [
                "正常", "异常", "病变", "炎症", "肿瘤",
                "出血", "感染", "损伤", "退化", "其他"
            ],
            "confidence_threshold": 0.5,
            "supported_formats": [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
        }

        # 合并配置
        self.config = {**self.default_config, **self.config}

        # 初始化模型
        self.model = ImageRecognitionModel(
            num_classes=self.config["num_classes"],
            model_type=self.config["model_type"],
            pretrained=self.config["pretrained"]
        )

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((self.config["image_size"], self.config["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        logger.info(f"图像识别器初始化完成: {self}")

    def load_model(self) -> bool:
        """
        加载模型

        Returns:
            bool: 加载是否成功
        """
        if self.model_path is None:
            logger.warning("模型路径未指定，使用默认模型")
            self.is_loaded = True
            return True

        try:
            if Path(self.model_path).exists():
                # 加载模型权重
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
                self.is_loaded = True
                logger.info(f"模型加载成功: {self.model_path}")
                return True
            else:
                logger.warning(f"模型文件不存在: {self.model_path}")
                self.is_loaded = True  # 使用默认模型
                return True
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False

    def preprocess(self, data: Union[str, bytes, np.ndarray, Image.Image]) -> torch.Tensor:
        """
        图像预处理

        Args:
            data: 输入图像数据

        Returns:
            torch.Tensor: 预处理后的图像张量
        """
        try:
            # 处理不同类型的输入
            if isinstance(data, str):
                # 文件路径
                if not Path(data).exists():
                    raise FileNotFoundError(f"图像文件不存在: {data}")
                image = Image.open(data).convert('RGB')
            elif isinstance(data, bytes):
                # 字节数据
                import io
                image = Image.open(io.BytesIO(data)).convert('RGB')
            elif isinstance(data, np.ndarray):
                # OpenCV格式
                if len(data.shape) == 3 and data.shape[2] == 3:
                    # BGR转RGB
                    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(data)
            elif isinstance(data, Image.Image):
                # PIL图像
                image = data.convert('RGB')
            else:
                raise ValueError(f"不支持的图像数据类型: {type(data)}")

            # 应用变换
            image_tensor = self.transform(image)

            # 添加批次维度
            image_tensor = image_tensor.unsqueeze(0)

            return image_tensor

        except Exception as e:
            logger.error(f"图像预处理失败: {e}")
            raise

    def predict(self, data: Union[str, bytes, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """
        执行图像识别

        Args:
            data: 输入图像数据

        Returns:
            Dict[str, Any]: 识别结果
        """
        if not self.validate_input(data):
            return {"error": "输入数据无效"}

        try:
            # 确保模型已加载
            if not self.is_loaded:
                self.load_model()

            # 图像预处理
            input_tensor = self.preprocess(data)
            input_tensor = input_tensor.to(self.device)

            # 模型推理
            with torch.no_grad():
                predictions, features = self.model(input_tensor)
                probabilities = torch.softmax(predictions, dim=1)
                probabilities = probabilities.cpu().numpy().flatten()

                # 获取预测类别
                predicted_class = np.argmax(probabilities)
                confidence = probabilities[predicted_class]

            # 结果后处理
            results = self.postprocess(predictions.cpu().numpy())

            # 添加元信息
            results.update({
                "predictor_type": "image_predictor",
                "model_info": self.get_model_info(),
                "prediction_timestamp": datetime.now().isoformat(),
                "confidence": float(confidence),
                "predicted_class": int(predicted_class),
                "class_name": self.config["class_names"][predicted_class] if predicted_class < len(self.config["class_names"]) else "未知"
            })

            logger.info(f"图像识别完成: {results}")
            return results

        except Exception as e:
            logger.error(f"图像识别失败: {e}")
            return {"error": str(e)}

    def postprocess(self, predictions: np.ndarray) -> Dict[str, Any]:
        """
        结果后处理

        Args:
            predictions: 模型原始输出

        Returns:
            Dict[str, Any]: 处理后的结果
        """
        try:
            # 计算概率分布
            probabilities = torch.softmax(torch.from_numpy(predictions), dim=1).numpy().flatten()

            # 获取top-k预测结果
            top_k = 3
            top_indices = np.argsort(probabilities)[::-1][:top_k]

            # 构建结果
            top_predictions = []
            for idx in top_indices:
                class_name = self.config["class_names"][idx] if idx < len(self.config["class_names"]) else f"类别{idx}"
                top_predictions.append({
                    "class_id": int(idx),
                    "class_name": class_name,
                    "probability": float(probabilities[idx])
                })

            # 生成分析建议
            recommendations = self._generate_recommendations(top_predictions)

            return {
                "top_predictions": top_predictions,
                "recommendations": recommendations,
                "analysis_summary": self._generate_analysis_summary(top_predictions)
            }

        except Exception as e:
            logger.error(f"结果后处理失败: {e}")
            return {"error": str(e)}

    def _generate_recommendations(self, top_predictions: List[Dict[str, Any]]) -> List[str]:
        """
        生成分析建议

        Args:
            top_predictions: 前K个预测结果

        Returns:
            List[str]: 建议列表
        """
        recommendations = []

        if not top_predictions:
            return ["无法识别图像内容，建议重新上传清晰的图像"]

        # 获取最高置信度的预测
        best_prediction = top_predictions[0]
        class_name = best_prediction["class_name"]
        confidence = best_prediction["probability"]

        # 基于置信度的建议
        if confidence < 0.3:
            recommendations.append("识别置信度较低，建议上传更清晰的图像")
        elif confidence < 0.7:
            recommendations.append("识别置信度中等，建议结合其他检查方法")
        else:
            recommendations.append("识别置信度较高，结果可信")

        # 基于类别的建议
        if class_name == "正常":
            recommendations.extend([
                "图像显示正常，建议定期检查",
                "保持良好的生活习惯"
            ])
        elif class_name in ["异常", "病变", "炎症"]:
            recommendations.extend([
                "发现异常情况，建议进一步检查",
                "及时咨询专业医生",
                "避免延误治疗时机"
            ])
        elif class_name == "肿瘤":
            recommendations.extend([
                "发现疑似肿瘤，立即就医",
                "进行专业医学检查",
                "制定治疗方案"
            ])
        elif class_name == "出血":
            recommendations.extend([
                "发现出血情况，立即就医",
                "避免剧烈运动",
                "保持休息"
            ])
        elif class_name == "感染":
            recommendations.extend([
                "发现感染迹象，及时治疗",
                "注意个人卫生",
                "遵医嘱用药"
            ])
        else:
            recommendations.append("建议咨询专业医生进行详细分析")

        return recommendations[:5]  # 限制建议数量

    def _generate_analysis_summary(self, top_predictions: List[Dict[str, Any]]) -> str:
        """
        生成分析摘要

        Args:
            top_predictions: 前K个预测结果

        Returns:
            str: 分析摘要
        """
        if not top_predictions:
            return "无法识别图像内容"

        best_prediction = top_predictions[0]
        class_name = best_prediction["class_name"]
        confidence = best_prediction["probability"]

        if confidence > 0.8:
            confidence_level = "高度可信"
        elif confidence > 0.6:
            confidence_level = "中等可信"
        else:
            confidence_level = "低可信度"

        summary = f"图像识别结果：{class_name}（{confidence_level}，置信度：{confidence:.2%}）"

        if len(top_predictions) > 1:
            second_prediction = top_predictions[1]
            summary += f"，次要可能：{second_prediction['class_name']}（{second_prediction['probability']:.2%}）"

        return summary

    def batch_predict(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        批量图像识别

        Args:
            image_paths: 图像文件路径列表

        Returns:
            List[Dict[str, Any]]: 识别结果列表
        """
        results = []

        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                results.append(result)
            except Exception as e:
                logger.error(f"处理图像失败 {image_path}: {e}")
                results.append({"error": str(e), "image_path": image_path})

        return results

    def extract_features(self, data: Union[str, bytes, np.ndarray, Image.Image]) -> np.ndarray:
        """
        提取图像特征

        Args:
            data: 输入图像数据

        Returns:
            np.ndarray: 特征向量
        """
        try:
            if not self.is_loaded:
                self.load_model()

            input_tensor = self.preprocess(data)
            input_tensor = input_tensor.to(self.device)

            with torch.no_grad():
                _, features = self.model(input_tensor)
                features = features.cpu().numpy().flatten()

            return features

        except Exception as e:
            logger.error(f"特征提取失败: {e}")
            raise
__all__ = ["'logger'", "'ImageRecognitionModel'", "'ImagePredictor'"]
