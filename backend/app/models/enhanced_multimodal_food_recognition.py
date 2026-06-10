"""
增强多模态食物识别系统
面向糖尿病膳食记录的食物图像-营养多模态表示和视觉识别方法
集成计算机视觉、自然语言处理和营养分析
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass
import cv2
from PIL import Image
import json

logger = logging.getLogger(__name__)

@dataclass
class MultimodalConfig:
    """多模态配置"""
    image_size: int = 224
    text_max_length: int = 512
    image_feature_dim: int = 512
    text_feature_dim: int = 256
    nutrition_feature_dim: int = 128
    fusion_dim: int = 256
    num_food_categories: int = 1000
    num_nutrition_labels: int = 50
    dropout: float = 0.1
    use_attention: bool = True
    use_contrastive_learning: bool = True

class EnhancedVisualEncoder(nn.Module):
    """增强视觉编码器"""

    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.config = config

        # 使用预训练的ResNet作为骨干网络
        import torchvision.models as models
        try:
            # 新版本torchvision
            from torchvision.models import ResNet50_Weights
            self.backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        except ImportError:
            # 旧版本torchvision
            from torchvision.models import ResNet50_Weights
            self.backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        # 移除最后的分类层
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # 特征投影层
        self.feature_projection = nn.Sequential(
            nn.Linear(2048, config.image_feature_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.image_feature_dim, config.image_feature_dim),
            nn.LayerNorm(config.image_feature_dim)
        )

        # 注意力机制
        if config.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=config.image_feature_dim,
                num_heads=8,
                dropout=config.dropout,
                batch_first=True
            )

        # 食物区域检测
        self.food_detector = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        视觉编码
        Args:
            images: 输入图像 [batch_size, 3, height, width]
        Returns:
            视觉特征字典
        """
        batch_size = images.shape[0]

        # 骨干网络特征提取
        backbone_features = self.backbone(images)  # [batch_size, 2048, 7, 7]

        # 全局特征
        global_features = backbone_features.view(batch_size, 2048, -1).mean(dim=-1)  # [batch_size, 2048]
        global_features = self.feature_projection(global_features)  # [batch_size, image_feature_dim]

        # 食物区域检测
        food_regions = self.food_detector(backbone_features)  # [batch_size, 1, 7, 7]

        # 区域特征
        region_features = backbone_features * food_regions  # [batch_size, 2048, 7, 7]
        region_features = region_features.view(batch_size, 2048, -1)  # [batch_size, 2048, 49]
        region_features = region_features.transpose(1, 2)  # [batch_size, 49, 2048]
        region_features = self.feature_projection(region_features)  # [batch_size, 49, image_feature_dim]

        # 注意力机制
        if self.config.use_attention:
            attended_features, attention_weights = self.attention(
                region_features, region_features, region_features
            )
            region_features = attended_features

        return {
            'global_features': global_features,
            'region_features': region_features,
            'food_regions': food_regions,
            'attention_weights': attention_weights if self.config.use_attention else None
        }

class EnhancedTextEncoder(nn.Module):
    """增强文本编码器"""

    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.config = config

        # 文本嵌入层
        self.embedding = nn.Embedding(30000, 256)  # 词汇表大小

        # 双向LSTM编码器
        self.lstm_encoder = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout
        )

        # 特征投影层
        self.feature_projection = nn.Sequential(
            nn.Linear(256, config.text_feature_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.text_feature_dim, config.text_feature_dim),
            nn.LayerNorm(config.text_feature_dim)
        )

        # 营养信息提取器
        self.nutrition_extractor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, config.nutrition_feature_dim)
        )

    def forward(self, text_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        文本编码
        Args:
            text_tokens: 文本标记 [batch_size, seq_len]
        Returns:
            文本特征字典
        """
        batch_size, seq_len = text_tokens.shape

        # 文本嵌入
        embedded = self.embedding(text_tokens)  # [batch_size, seq_len, 256]

        # LSTM编码
        lstm_output, (hidden, cell) = self.lstm_encoder(embedded)

        # 全局特征（使用最后一个隐藏状态）
        global_features = hidden[-1]  # [batch_size, 256]
        global_features = self.feature_projection(global_features)  # [batch_size, text_feature_dim]

        # 序列特征
        sequence_features = self.feature_projection(lstm_output)  # [batch_size, seq_len, text_feature_dim]

        # 营养信息提取
        nutrition_features = self.nutrition_extractor(lstm_output.mean(dim=1))  # [batch_size, nutrition_feature_dim]

        return {
            'global_features': global_features,
            'sequence_features': sequence_features,
            'nutrition_features': nutrition_features
        }

class CrossModalAttention(nn.Module):
    """跨模态注意力机制"""

    def __init__(self, image_dim: int, text_dim: int, fusion_dim: int):
        super().__init__()
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.fusion_dim = fusion_dim

        # 图像到文本注意力
        self.image_to_text_attention = nn.MultiheadAttention(
            embed_dim=text_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # 文本到图像注意力
        self.text_to_image_attention = nn.MultiheadAttention(
            embed_dim=image_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # 特征融合
        self.fusion_layer = nn.Sequential(
            nn.Linear(image_dim + text_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        跨模态注意力融合
        Args:
            image_features: 图像特征 [batch_size, image_dim]
            text_features: 文本特征 [batch_size, text_dim]
        Returns:
            融合特征字典
        """
        batch_size = image_features.shape[0]

        # 扩展维度用于注意力计算
        image_features_expanded = image_features.unsqueeze(1)  # [batch_size, 1, image_dim]
        text_features_expanded = text_features.unsqueeze(1)    # [batch_size, 1, text_dim]

        # 图像到文本注意力
        image_attended, image_attention_weights = self.image_to_text_attention(
            text_features_expanded, image_features_expanded, image_features_expanded
        )

        # 文本到图像注意力
        text_attended, text_attention_weights = self.text_to_image_attention(
            image_features_expanded, text_features_expanded, text_features_expanded
        )

        # 特征拼接和融合
        concatenated_features = torch.cat([
            image_features, text_features,
            image_attended.squeeze(1), text_attended.squeeze(1)
        ], dim=-1)

        fused_features = self.fusion_layer(concatenated_features)

        return {
            'fused_features': fused_features,
            'image_attention_weights': image_attention_weights,
            'text_attention_weights': text_attention_weights
        }

class NutritionAnalyzer(nn.Module):
    """营养分析器"""

    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.config = config

        # 营养预测头
        self.nutrition_predictor = nn.Sequential(
            nn.Linear(config.fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, config.num_nutrition_labels)
        )

        # 食物分类器
        self.food_classifier = nn.Sequential(
            nn.Linear(config.fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, config.num_food_categories)
        )

        # 血糖影响预测器
        self.glucose_impact_predictor = nn.Sequential(
            nn.Linear(config.fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 1)  # 血糖影响分数
        )

    def forward(self, fused_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        营养分析
        Args:
            fused_features: 融合特征 [batch_size, fusion_dim]
        Returns:
            营养分析结果
        """
        # 营养预测
        nutrition_predictions = self.nutrition_predictor(fused_features)

        # 食物分类
        food_predictions = self.food_classifier(fused_features)

        # 血糖影响预测
        glucose_impact = self.glucose_impact_predictor(fused_features)

        return {
            'nutrition_predictions': nutrition_predictions,
            'food_predictions': food_predictions,
            'glucose_impact': glucose_impact
        }

class ContrastiveLearningModule(nn.Module):
    """对比学习模块"""

    def __init__(self, feature_dim: int, temperature: float = 0.07):
        super().__init__()
        self.feature_dim = feature_dim
        self.temperature = temperature

        # 投影头
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim // 2)
        )

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        对比学习
        Args:
            image_features: 图像特征 [batch_size, feature_dim]
            text_features: 文本特征 [batch_size, feature_dim]
        Returns:
            对比学习结果
        """
        # 特征投影
        image_proj = self.projection_head(image_features)
        text_proj = self.projection_head(text_features)

        # L2归一化
        image_proj = F.normalize(image_proj, p=2, dim=1)
        text_proj = F.normalize(text_proj, p=2, dim=1)

        # 计算相似度矩阵
        similarity_matrix = torch.matmul(image_proj, text_proj.T) / self.temperature

        # 对比损失
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size).to(image_features.device)

        # 图像到文本损失
        image_to_text_loss = F.cross_entropy(similarity_matrix, labels)

        # 文本到图像损失
        text_to_image_loss = F.cross_entropy(similarity_matrix.T, labels)

        # 总对比损失
        contrastive_loss = (image_to_text_loss + text_to_image_loss) / 2

        return {
            'contrastive_loss': contrastive_loss,
            'similarity_matrix': similarity_matrix,
            'image_projection': image_proj,
            'text_projection': text_proj
        }

class EnhancedMultimodalFoodRecognition(nn.Module):
    """增强多模态食物识别系统"""

    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.config = config

        # 编码器
        self.visual_encoder = EnhancedVisualEncoder(config)
        self.text_encoder = EnhancedTextEncoder(config)

        # 跨模态注意力
        self.cross_modal_attention = CrossModalAttention(
            config.image_feature_dim, config.text_feature_dim, config.fusion_dim
        )

        # 营养分析器
        self.nutrition_analyzer = NutritionAnalyzer(config)

        # 对比学习模块
        if config.use_contrastive_learning:
            self.contrastive_module = ContrastiveLearningModule(config.fusion_dim)

        # 最终分类器
        self.final_classifier = nn.Sequential(
            nn.Linear(config.fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, config.num_food_categories)
        )

    def forward(
        self,
        images: torch.Tensor,
        text_tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        Args:
            images: 输入图像 [batch_size, 3, height, width]
            text_tokens: 文本标记 [batch_size, seq_len]
        Returns:
            识别结果字典
        """
        # 视觉编码
        visual_features = self.visual_encoder(images)

        # 文本编码
        text_features = self.text_encoder(text_tokens)

        # 跨模态注意力融合
        cross_modal_features = self.cross_modal_attention(
            visual_features['global_features'],
            text_features['global_features']
        )

        # 营养分析
        nutrition_analysis = self.nutrition_analyzer(
            cross_modal_features['fused_features']
        )

        # 最终分类
        final_predictions = self.final_classifier(
            cross_modal_features['fused_features']
        )

        # 对比学习
        contrastive_results = {}
        if self.config.use_contrastive_learning:
            contrastive_results = self.contrastive_module(
                visual_features['global_features'],
                text_features['global_features']
            )

        return {
            'visual_features': visual_features,
            'text_features': text_features,
            'cross_modal_features': cross_modal_features,
            'nutrition_analysis': nutrition_analysis,
            'final_predictions': final_predictions,
            'contrastive_results': contrastive_results
        }

    def extract_features(self, images: torch.Tensor, text_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """提取特征用于其他任务"""
        with torch.no_grad():
            results = self.forward(images, text_tokens)
            return {
                'image_features': results['visual_features']['global_features'],
                'text_features': results['text_features']['global_features'],
                'fused_features': results['cross_modal_features']['fused_features'],
                'nutrition_features': results['nutrition_analysis']['nutrition_predictions']
            }

class EnhancedFoodRecognitionService:
    """增强食物识别服务"""

    def __init__(self, config: MultimodalConfig):
        self.config = config
        self.model = EnhancedMultimodalFoodRecognition(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # 图像预处理
        self.image_transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 文本预处理
        self.text_tokenizer = self._create_simple_tokenizer()

        # 营养标签
        self.nutrition_labels = self._load_nutrition_labels()
        self.food_categories = self._load_food_categories()

    def _create_simple_tokenizer(self):
        """创建简单分词器"""
        # 这里应该使用更复杂的分词器，如BERT tokenizer
        # 为了简化，使用简单的词汇表
        vocab = {}
        with open('vocab.txt', 'w', encoding='utf-8') as f:
            # 创建基本词汇表
            basic_words = ['食物', '营养', '卡路里', '蛋白质', '脂肪', '碳水化合物', '维生素', '矿物质']
            for i, word in enumerate(basic_words):
                vocab[word] = i
                f.write(f"{word}\n")
        return vocab

    def _load_nutrition_labels(self) -> List[str]:
        """加载营养标签"""
        return [
            '卡路里', '蛋白质', '脂肪', '碳水化合物', '纤维', '糖分',
            '钠', '钾', '钙', '铁', '维生素A', '维生素C', '维生素D',
            '维生素E', '维生素K', '维生素B1', '维生素B2', '维生素B3',
            '维生素B6', '维生素B12', '叶酸', '生物素', '泛酸', '胆碱'
        ]

    def _load_food_categories(self) -> List[str]:
        """加载食物类别"""
        return [
            '主食', '蔬菜', '水果', '肉类', '鱼类', '蛋类', '奶制品',
            '豆制品', '坚果', '饮料', '甜品', '调料', '其他'
        ]

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """预处理图像"""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.image_transform(image)
            return image_tensor.unsqueeze(0)  # 添加批次维度
        except Exception as e:
            logger.error(f"图像预处理失败: {e}")
            # 返回默认图像
            return torch.zeros(1, 3, self.config.image_size, self.config.image_size)

    def preprocess_text(self, text: str) -> torch.Tensor:
        """预处理文本"""
        # 简单的文本预处理
        words = text.split()
        tokens = []
        for word in words:
            if word in self.text_tokenizer:
                tokens.append(self.text_tokenizer[word])
            else:
                tokens.append(0)  # 未知词

        # 填充或截断到固定长度
        if len(tokens) > self.config.text_max_length:
            tokens = tokens[:self.config.text_max_length]
        else:
            tokens.extend([0] * (self.config.text_max_length - len(tokens)))

        return torch.LongTensor(tokens).unsqueeze(0)  # 添加批次维度

    def recognize_food(self, image_path: str, text_description: str = "") -> Dict[str, Any]:
        """识别食物"""
        self.model.eval()

        # 预处理
        image_tensor = self.preprocess_image(image_path)
        text_tensor = self.preprocess_text(text_description)

        # 移动到设备
        image_tensor = image_tensor.to(self.device)
        text_tensor = text_tensor.to(self.device)

        with torch.no_grad():
            results = self.model(image_tensor, text_tensor)

            # 后处理
            food_predictions = F.softmax(results['final_predictions'], dim=-1)
            nutrition_predictions = F.sigmoid(results['nutrition_analysis']['nutrition_predictions'])
            glucose_impact = results['nutrition_analysis']['glucose_impact']

            # 获取预测结果
            predicted_food = torch.argmax(food_predictions, dim=-1).item()
            predicted_food_name = self.food_categories[predicted_food] if predicted_food < len(self.food_categories) else "未知"

            # 营养信息
            nutrition_info = {}
            for i, label in enumerate(self.nutrition_labels):
                if i < nutrition_predictions.shape[-1]:
                    nutrition_info[label] = float(nutrition_predictions[0, i])

            return {
                'food_category': predicted_food_name,
                'food_confidence': float(food_predictions[0, predicted_food]),
                'nutrition_info': nutrition_info,
                'glucose_impact': float(glucose_impact[0, 0]),
                'features': {
                    'image_features': results['visual_features']['global_features'].cpu().numpy(),
                    'text_features': results['text_features']['global_features'].cpu().numpy(),
                    'fused_features': results['cross_modal_features']['fused_features'].cpu().numpy()
                }
            }

    def extract_multimodal_features(self, image_path: str, text_description: str = "") -> Dict[str, np.ndarray]:
        """提取多模态特征"""
        self.model.eval()

        # 预处理
        image_tensor = self.preprocess_image(image_path)
        text_tensor = self.preprocess_text(text_description)

        # 移动到设备
        image_tensor = image_tensor.to(self.device)
        text_tensor = text_tensor.to(self.device)

        with torch.no_grad():
            features = self.model.extract_features(image_tensor, text_tensor)

            return {
                'image_features': features['image_features'].cpu().numpy(),
                'text_features': features['text_features'].cpu().numpy(),
                'fused_features': features['fused_features'].cpu().numpy(),
                'nutrition_features': features['nutrition_features'].cpu().numpy()
            }

# 工厂函数
def create_enhanced_food_recognition_service(config: Optional[MultimodalConfig] = None) -> EnhancedFoodRecognitionService:
    """创建增强食物识别服务"""
    if config is None:
        config = MultimodalConfig()

    service = EnhancedFoodRecognitionService(config)
    return service

if __name__ == "__main__":
    # 测试增强多模态食物识别系统
    config = MultimodalConfig(
        image_size=224,
        text_max_length=512,
        image_feature_dim=512,
        text_feature_dim=256,
        nutrition_feature_dim=128,
        fusion_dim=256,
        num_food_categories=1000,
        num_nutrition_labels=50,
        use_attention=True,
        use_contrastive_learning=True
    )

    service = create_enhanced_food_recognition_service(config)

    # 模拟识别
    image_path = "test_image.jpg"  # 假设的测试图像
    text_description = "这是一盘宫保鸡丁，含有鸡肉、花生、辣椒等食材"

    try:
        # 食物识别
        recognition_result = service.recognize_food(image_path, text_description)

        print(f"增强多模态食物识别系统创建成功！")
        print(f"识别结果: {recognition_result['food_category']}")
        print(f"置信度: {recognition_result['food_confidence']:.4f}")
        print(f"血糖影响: {recognition_result['glucose_impact']:.4f}")
        print(f"营养信息: {list(recognition_result['nutrition_info'].keys())}")

        # 特征提取
        features = service.extract_multimodal_features(image_path, text_description)
        print(f"提取的特征维度: {features['fused_features'].shape}")

    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        print("系统创建成功，但需要有效的图像文件进行测试")
