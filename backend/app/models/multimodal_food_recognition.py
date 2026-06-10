"""
多模态食物识别和营养成分分析系统
实现视觉-文本语义对齐和混合菜品识别
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import BertModel, BertTokenizer
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from PIL import Image
import cv2

logger = logging.getLogger(__name__)

class VisualEncoder(nn.Module):
    """视觉编码器 - 基于ResNet的食物图像特征提取"""

    def __init__(self, pretrained: bool = True, feature_dim: int = 512):
        super().__init__()
        # 使用预训练的ResNet-50
        self.backbone = models.resnet50(pretrained=pretrained)

        # 移除最后的分类层
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # 添加自定义特征提取层
        self.feature_projection = nn.Sequential(
            nn.Linear(2048, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )

        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            batch_first=True
        )

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        视觉特征提取
        Args:
            images: 输入图像 [batch_size, 3, height, width]
        Returns:
            特征向量和注意力权重
        """
        batch_size = images.shape[0]

        # 特征提取
        features = self.backbone(images)  # [batch_size, 2048]
        features = features.view(batch_size, -1)

        # 特征投影
        projected_features = self.feature_projection(features)

        # 添加序列维度用于注意力计算
        projected_features = projected_features.unsqueeze(1)  # [batch_size, 1, feature_dim]

        # 自注意力
        attended_features, attention_weights = self.attention(
            projected_features, projected_features, projected_features
        )

        return attended_features.squeeze(1), attention_weights.squeeze(1)

class TextEncoder(nn.Module):
    """文本编码器 - 基于BERT的食物描述语义提取"""

    def __init__(self, model_name: str = 'bert-base-chinese', feature_dim: int = 512):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

        # 特征投影层
        self.feature_projection = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )

        # 营养术语增强
        self.nutrition_enhancer = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, text_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        文本特征提取
        Args:
            text_inputs: BERT输入字典
        Returns:
            文本特征向量
        """
        # BERT编码
        outputs = self.bert(**text_inputs)
        pooled_output = outputs.pooler_output  # [batch_size, hidden_size]

        # 特征投影
        projected_features = self.feature_projection(pooled_output)

        # 营养术语增强
        enhanced_features = self.nutrition_enhancer(projected_features)

        return enhanced_features

class CrossModalAttention(nn.Module):
    """跨模态注意力机制"""

    def __init__(self, feature_dim: int = 512):
        super().__init__()
        self.feature_dim = feature_dim

        # 视觉到文本的注意力
        self.v2t_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            batch_first=True
        )

        # 文本到视觉的注意力
        self.t2v_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            batch_first=True
        )

        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        跨模态注意力融合
        Args:
            visual_features: 视觉特征 [batch_size, feature_dim]
            text_features: 文本特征 [batch_size, feature_dim]
        Returns:
            融合特征和注意力权重
        """
        # 添加序列维度
        v_features = visual_features.unsqueeze(1)
        t_features = text_features.unsqueeze(1)

        # 视觉到文本注意力
        v2t_output, v2t_weights = self.v2t_attention(v_features, t_features, t_features)

        # 文本到视觉注意力
        t2v_output, t2v_weights = self.t2v_attention(t_features, v_features, v_features)

        # 特征融合
        fused_features = torch.cat([v2t_output, t2v_output], dim=-1)
        fused_features = self.fusion_layer(fused_features)

        return fused_features.squeeze(1), {
            'v2t_attention': v2t_weights,
            't2v_attention': t2v_weights
        }

class NutritionAnalyzer(nn.Module):
    """营养成分分析器"""

    def __init__(self, feature_dim: int = 512, num_nutrients: int = 9):
        super().__init__()
        self.num_nutrients = num_nutrients

        # 营养成分预测头
        self.nutrition_heads = nn.ModuleDict({
            'calories': nn.Linear(feature_dim, 1),
            'carbohydrates': nn.Linear(feature_dim, 1),
            'protein': nn.Linear(feature_dim, 1),
            'fat': nn.Linear(feature_dim, 1),
            'fiber': nn.Linear(feature_dim, 1),
            'sugar': nn.Linear(feature_dim, 1),
            'sodium': nn.Linear(feature_dim, 1),
            'cholesterol': nn.Linear(feature_dim, 1),
            'vitamins': nn.Linear(feature_dim, 1)
        })

        # GI/GL预测
        self.gi_predictor = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()  # GI值范围0-1
        )

        self.gl_predictor = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        营养成分预测
        Args:
            features: 融合特征 [batch_size, feature_dim]
        Returns:
            营养成分预测结果
        """
        nutrition_predictions = {}

        # 预测各种营养成分
        for nutrient, head in self.nutrition_heads.items():
            nutrition_predictions[nutrient] = head(features)

        # 预测GI/GL
        nutrition_predictions['gi'] = self.gi_predictor(features) * 100  # 转换为0-100范围
        nutrition_predictions['gl'] = self.gl_predictor(features)

        return nutrition_predictions

class MixedDishRecognizer(nn.Module):
    """混合菜品识别器"""

    def __init__(self, feature_dim: int = 512, max_ingredients: int = 10):
        super().__init__()
        self.max_ingredients = max_ingredients

        # 成分检测网络
        self.ingredient_detector = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, max_ingredients)
        )

        # 成分权重预测
        self.ingredient_weights = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, max_ingredients),
            nn.Softmax(dim=-1)
        )

        # 菜品分类
        self.dish_classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 50)  # 50种常见菜品类型
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        混合菜品识别
        Args:
            features: 融合特征 [batch_size, feature_dim]
        Returns:
            成分检测和菜品分类结果
        """
        # 成分检测
        ingredient_scores = self.ingredient_detector(features)
        ingredient_weights = self.ingredient_weights(features)

        # 菜品分类
        dish_classification = self.dish_classifier(features)

        return {
            'ingredient_scores': ingredient_scores,
            'ingredient_weights': ingredient_weights,
            'dish_classification': dish_classification
        }

class MultiModalFoodRecognition(nn.Module):
    """多模态食物识别系统"""

    def __init__(
        self,
        visual_feature_dim: int = 512,
        text_feature_dim: int = 512,
        num_nutrients: int = 9,
        max_ingredients: int = 10
    ):
        super().__init__()

        # 编码器
        self.visual_encoder = VisualEncoder(feature_dim=visual_feature_dim)
        self.text_encoder = TextEncoder(feature_dim=text_feature_dim)

        # 跨模态注意力
        self.cross_modal_attention = CrossModalAttention(visual_feature_dim)

        # 分析器
        self.nutrition_analyzer = NutritionAnalyzer(visual_feature_dim, num_nutrients)
        self.mixed_dish_recognizer = MixedDishRecognizer(visual_feature_dim, max_ingredients)

        # 最终融合层
        self.final_fusion = nn.Sequential(
            nn.Linear(visual_feature_dim * 2, visual_feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(visual_feature_dim, visual_feature_dim)
        )

    def forward(
        self,
        images: torch.Tensor,
        text_inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        多模态食物识别
        Args:
            images: 食物图像 [batch_size, 3, height, width]
            text_inputs: 文本输入字典
        Returns:
            识别和分析结果
        """
        # 视觉特征提取
        visual_features, visual_attention = self.visual_encoder(images)

        # 文本特征提取
        text_features = self.text_encoder(text_inputs)

        # 跨模态注意力融合
        fused_features, cross_attention = self.cross_modal_attention(
            visual_features, text_features
        )

        # 最终特征融合
        final_features = torch.cat([visual_features, text_features], dim=-1)
        final_features = self.final_fusion(final_features)

        # 营养成分分析
        nutrition_results = self.nutrition_analyzer(final_features)

        # 混合菜品识别
        mixed_dish_results = self.mixed_dish_recognizer(final_features)

        return {
            'visual_features': visual_features,
            'text_features': text_features,
            'fused_features': fused_features,
            'final_features': final_features,
            'nutrition_analysis': nutrition_results,
            'mixed_dish_recognition': mixed_dish_results,
            'attention_weights': {
                'visual_attention': visual_attention,
                'cross_attention': cross_attention
            }
        }

class FoodRecognitionService:
    """食物识别服务"""

    def __init__(self):
        self.model = MultiModalFoodRecognition()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # 营养成分数据库
        self.nutrition_database = self._load_nutrition_database()

        # 菜品分类标签
        self.dish_labels = self._load_dish_labels()

    def _load_nutrition_database(self) -> Dict[str, Dict[str, float]]:
        """加载营养成分数据库"""
        return {
            '米饭': {'calories': 130, 'carbohydrates': 28, 'protein': 2.7, 'fat': 0.3, 'gi': 83},
            '面条': {'calories': 131, 'carbohydrates': 25, 'protein': 5, 'fat': 1.1, 'gi': 82},
            '鸡蛋': {'calories': 155, 'carbohydrates': 1.1, 'protein': 13, 'fat': 11, 'gi': 0},
            '鸡肉': {'calories': 165, 'carbohydrates': 0, 'protein': 31, 'fat': 3.6, 'gi': 0},
            '蔬菜': {'calories': 25, 'carbohydrates': 5, 'protein': 2, 'fat': 0.2, 'gi': 15},
            # 更多食物数据...
        }

    def _load_dish_labels(self) -> List[str]:
        """加载菜品分类标签"""
        return [
            '川菜', '粤菜', '鲁菜', '苏菜', '浙菜', '闽菜', '湘菜', '徽菜',
            '汤类', '炒菜', '蒸菜', '烤菜', '炸菜', '凉菜', '主食', '甜品',
            '西餐', '日料', '韩料', '东南亚菜', '其他'
        ]

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """图像预处理"""
        # 加载图像
        image = Image.open(image_path).convert('RGB')

        # 调整大小
        image = image.resize((224, 224))

        # 转换为tensor
        image_tensor = torch.from_numpy(np.array(image)).float()
        image_tensor = image_tensor.permute(2, 0, 1) / 255.0

        # 标准化
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        image_tensor = (image_tensor - mean.view(3, 1, 1)) / std.view(3, 1, 1)

        return image_tensor.unsqueeze(0)  # 添加batch维度

    def preprocess_text(self, text: str) -> Dict[str, torch.Tensor]:
        """文本预处理"""
        tokenizer = self.text_encoder.tokenizer

        # 分词和编码
        inputs = tokenizer(
            text,
            max_length=128,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        return inputs

    def recognize_food(
        self,
        image_path: str,
        description: str = ""
    ) -> Dict[str, Union[float, List[str], Dict]]:
        """
        食物识别和分析
        Args:
            image_path: 图像路径
            description: 食物描述文本
        Returns:
            识别和分析结果
        """
        self.model.eval()

        # 预处理
        image_tensor = self.preprocess_image(image_path)
        text_inputs = self.preprocess_text(description)

        # 移动到设备
        image_tensor = image_tensor.to(self.device)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

        # 推理
        with torch.no_grad():
            results = self.model(image_tensor, text_inputs)

        # 后处理
        nutrition_analysis = results['nutrition_analysis']
        mixed_dish_results = results['mixed_dish_recognition']

        # 提取营养成分
        nutrition_values = {}
        for nutrient, values in nutrition_analysis.items():
            nutrition_values[nutrient] = values.item()

        # 提取成分检测结果
        ingredient_scores = mixed_dish_results['ingredient_scores'].cpu().numpy()[0]
        ingredient_weights = mixed_dish_results['ingredient_weights'].cpu().numpy()[0]

        # 获取主要成分
        top_ingredients = np.argsort(ingredient_scores)[-5:][::-1]
        ingredient_names = [f"成分_{i}" for i in top_ingredients]

        # 菜品分类
        dish_scores = mixed_dish_results['dish_classification'].cpu().numpy()[0]
        top_dish = np.argmax(dish_scores)
        dish_name = self.dish_labels[top_dish]

        return {
            'nutrition_analysis': nutrition_values,
            'ingredients': ingredient_names,
            'ingredient_weights': ingredient_weights[top_ingredients].tolist(),
            'dish_classification': dish_name,
            'dish_confidence': float(dish_scores[top_dish]),
            'attention_weights': results['attention_weights']
        }

    def batch_recognize(self, image_paths: List[str], descriptions: List[str]) -> List[Dict]:
        """批量识别"""
        results = []
        for img_path, desc in zip(image_paths, descriptions):
            result = self.recognize_food(img_path, desc)
            results.append(result)
        return results

# 使用示例
def create_food_recognition_service() -> FoodRecognitionService:
    """创建食物识别服务"""
    service = FoodRecognitionService()
    return service

if __name__ == "__main__":
    # 创建服务
    service = create_food_recognition_service()

    # 模拟识别
    result = service.recognize_food("sample_food.jpg", "这是一道宫保鸡丁")

    print("食物识别结果:")
    print(f"营养成分: {result['nutrition_analysis']}")
    print(f"主要成分: {result['ingredients']}")
    print(f"菜品分类: {result['dish_classification']}")
    print("多模态食物识别系统创建成功！")
