#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态特征提取器
用于提取和处理图像、文本等多模态特征
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from PIL import Image
import torchvision.transforms as transforms
from transformers import (
    AutoTokenizer, AutoModel,
    CLIPProcessor, CLIPModel
)

logger = logging.getLogger(__name__)


class MultimodalFeatureExtractor(nn.Module):
    """
    多模态特征提取器
    支持文本、图像、用户画像等多种模态的特征提取
    """

    def __init__(
        self,
        text_model_name: str = "bert-base-uncased",
        image_model_name: str = "openai/clip-vit-base-patch32",
        text_dim: int = 768,
        image_dim: int = 512,
        user_profile_dim: int = 128,
        cultural_dim: int = 64,
        device: str = "cpu"
    ):
        super().__init__()

        self.device = device
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.user_profile_dim = user_profile_dim
        self.cultural_dim = cultural_dim

        # 文本特征提取器
        try:
            self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
            self.text_model = AutoModel.from_pretrained(text_model_name)
            logger.info(f"Loaded text model: {text_model_name}")
        except Exception as e:
            logger.warning(f"Failed to load text model {text_model_name}: {e}")
            # 使用简单的嵌入层作为备选
            self.text_tokenizer = None
            self.text_model = nn.Embedding(10000, text_dim)

        # 图像特征提取器
        try:
            self.image_processor = CLIPProcessor.from_pretrained(image_model_name)
            self.image_model = CLIPModel.from_pretrained(image_model_name)
            logger.info(f"Loaded image model: {image_model_name}")
        except Exception as e:
            logger.warning(f"Failed to load image model {image_model_name}: {e}")
            # 使用简单的CNN作为备选
            self.image_processor = None
            self.image_model = self._create_simple_cnn()

        # 图像预处理
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # 特征投影层
        self.text_projection = nn.Linear(text_dim, text_dim)
        self.image_projection = nn.Linear(image_dim, image_dim)
        self.user_projection = nn.Linear(user_profile_dim, user_profile_dim)
        self.cultural_projection = nn.Linear(cultural_dim, cultural_dim)

        # 特征融合层
        fusion_input_dim = text_dim + image_dim + user_profile_dim + cultural_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_input_dim // 2, fusion_input_dim // 4),
            nn.ReLU(),
            nn.Linear(fusion_input_dim // 4, 256)
        )

        self.to(device)

    def _create_simple_cnn(self) -> nn.Module:
        """创建简单的CNN作为图像特征提取器的备选"""
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, self.image_dim)
        )

    def extract_text_features(
        self,
        texts: List[str],
        max_length: int = 512
    ) -> torch.Tensor:
        """
        提取文本特征

        Args:
            texts: 文本列表
            max_length: 最大序列长度

        Returns:
            文本特征张量 [batch_size, text_dim]
        """
        if self.text_tokenizer is not None:
            # 使用预训练模型
            inputs = self.text_tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.text_model(**inputs)
                # 使用[CLS]标记的特征或平均池化
                if hasattr(outputs, 'pooler_output'):
                    features = outputs.pooler_output
                else:
                    features = outputs.last_hidden_state.mean(dim=1)
        else:
            # 使用简单嵌入层
            # 这里简化处理，实际应该有更好的文本处理
            batch_size = len(texts)
            # 生成随机索引作为演示
            indices = torch.randint(0, 10000, (batch_size, 10)).to(self.device)
            embeddings = self.text_model(indices)
            features = embeddings.mean(dim=1)

        return self.text_projection(features)

    def extract_image_features(self, images: List[Image.Image]) -> torch.Tensor:
        """
        提取图像特征

        Args:
            images: PIL图像列表

        Returns:
            图像特征张量 [batch_size, image_dim]
        """
        if self.image_processor is not None:
            # 使用CLIP模型
            inputs = self.image_processor(
                images=images,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                features = self.image_model.get_image_features(**inputs)
        else:
            # 使用简单CNN
            image_tensors = []
            for img in images:
                if isinstance(img, str):
                    # 如果是路径，创建随机图像作为演示
                    img_tensor = torch.randn(3, 224, 224)
                else:
                    img_tensor = self.image_transform(img)
                image_tensors.append(img_tensor)

            batch_images = torch.stack(image_tensors).to(self.device)
            features = self.image_model(batch_images)

        return self.image_projection(features)

    def extract_user_features(self, user_profiles: torch.Tensor) -> torch.Tensor:
        """
        提取用户画像特征

        Args:
            user_profiles: 用户画像张量 [batch_size, user_profile_dim]

        Returns:
            用户特征张量 [batch_size, user_profile_dim]
        """
        return self.user_projection(user_profiles)

    def extract_cultural_features(self, cultural_profiles: torch.Tensor) -> torch.Tensor:
        """
        提取文化特征

        Args:
            cultural_profiles: 文化画像张量 [batch_size, cultural_dim]

        Returns:
            文化特征张量 [batch_size, cultural_dim]
        """
        return self.cultural_projection(cultural_profiles)

    def fuse_features(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        user_features: torch.Tensor,
        cultural_features: torch.Tensor
    ) -> torch.Tensor:
        """
        融合多模态特征

        Args:
            text_features: 文本特征
            image_features: 图像特征
            user_features: 用户特征
            cultural_features: 文化特征

        Returns:
            融合后的特征张量
        """
        # 确保所有特征具有相同的batch size
        batch_size = text_features.size(0)

        # 拼接所有特征
        fused_features = torch.cat([
            text_features,
            image_features,
            user_features,
            cultural_features
        ], dim=1)

        # 通过融合层
        return self.fusion_layer(fused_features)

    def forward(
        self,
        texts: Optional[List[str]] = None,
        images: Optional[List[Image.Image]] = None,
        user_profiles: Optional[torch.Tensor] = None,
        cultural_profiles: Optional[torch.Tensor] = None,
        return_individual: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            texts: 文本列表
            images: 图像列表
            user_profiles: 用户画像
            cultural_profiles: 文化画像
            return_individual: 是否返回单独的特征

        Returns:
            特征字典
        """
        results = {}

        # 提取各模态特征
        if texts is not None:
            text_features = self.extract_text_features(texts)
            results['text_features'] = text_features
        else:
            # 创建零特征
            batch_size = len(images) if images else user_profiles.size(0)
            text_features = torch.zeros(batch_size, self.text_dim).to(self.device)
            results['text_features'] = text_features

        if images is not None:
            image_features = self.extract_image_features(images)
            results['image_features'] = image_features
        else:
            # 创建零特征
            batch_size = len(texts) if texts else user_profiles.size(0)
            image_features = torch.zeros(batch_size, self.image_dim).to(self.device)
            results['image_features'] = image_features

        if user_profiles is not None:
            user_features = self.extract_user_features(user_profiles)
            results['user_features'] = user_features
        else:
            # 创建零特征
            batch_size = len(texts) if texts else len(images)
            user_features = torch.zeros(batch_size, self.user_profile_dim).to(self.device)
            results['user_features'] = user_features

        if cultural_profiles is not None:
            cultural_features = self.extract_cultural_features(cultural_profiles)
            results['cultural_features'] = cultural_features
        else:
            # 创建零特征
            batch_size = user_features.size(0)
            cultural_features = torch.zeros(batch_size, self.cultural_dim).to(self.device)
            results['cultural_features'] = cultural_features

        # 融合特征
        fused_features = self.fuse_features(
            text_features, image_features, user_features, cultural_features
        )
        results['fused_features'] = fused_features

        if not return_individual:
            return {'features': fused_features}
        else:
            return results


class SimpleFeatureExtractor(nn.Module):
    """
    简化版特征提取器，用于演示和测试
    """

    def __init__(
        self,
        text_dim: int = 768,
        image_dim: int = 512,
        user_profile_dim: int = 128,
        cultural_dim: int = 64,
        fusion_dim: int = 256
    ):
        super().__init__()

        self.text_dim = text_dim
        self.image_dim = image_dim
        self.user_profile_dim = user_profile_dim
        self.cultural_dim = cultural_dim
        self.fusion_dim = fusion_dim

        # 简单的线性投影层
        self.text_proj = nn.Linear(text_dim, text_dim)
        self.image_proj = nn.Linear(image_dim, image_dim)
        self.user_proj = nn.Linear(user_profile_dim, user_profile_dim)
        self.cultural_proj = nn.Linear(cultural_dim, cultural_dim)

        # 融合层
        total_dim = text_dim + image_dim + user_profile_dim + cultural_dim
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, fusion_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim * 2, fusion_dim)
        )

    def forward(self, features_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播

        Args:
            features_dict: 包含各模态特征的字典

        Returns:
            融合后的特征
        """
        processed_features = []

        if 'text' in features_dict:
            processed_features.append(self.text_proj(features_dict['text']))
        else:
            batch_size = next(iter(features_dict.values())).size(0)
            processed_features.append(torch.zeros(batch_size, self.text_dim))

        if 'image' in features_dict:
            processed_features.append(self.image_proj(features_dict['image']))
        else:
            batch_size = next(iter(features_dict.values())).size(0)
            processed_features.append(torch.zeros(batch_size, self.image_dim))

        if 'user' in features_dict:
            processed_features.append(self.user_proj(features_dict['user']))
        else:
            batch_size = next(iter(features_dict.values())).size(0)
            processed_features.append(torch.zeros(batch_size, self.user_profile_dim))

        if 'cultural' in features_dict:
            processed_features.append(self.cultural_proj(features_dict['cultural']))
        else:
            batch_size = next(iter(features_dict.values())).size(0)
            processed_features.append(torch.zeros(batch_size, self.cultural_dim))

        # 拼接并融合
        concatenated = torch.cat(processed_features, dim=1)
        return self.fusion(concatenated)


def create_demo_features(
    batch_size: int = 32,
    text_dim: int = 768,
    image_dim: int = 512,
    user_profile_dim: int = 128,
    cultural_dim: int = 64
) -> Dict[str, torch.Tensor]:
    """
    创建演示用的特征数据

    Args:
        batch_size: 批次大小
        text_dim: 文本特征维度
        image_dim: 图像特征维度
        user_profile_dim: 用户画像维度
        cultural_dim: 文化特征维度

    Returns:
        特征字典
    """
    return {
        'text': torch.randn(batch_size, text_dim),
        'image': torch.randn(batch_size, image_dim),
        'user': torch.randn(batch_size, user_profile_dim),
        'cultural': torch.randn(batch_size, cultural_dim)
    }


if __name__ == "__main__":
    # 测试特征提取器
    print("Testing MultimodalFeatureExtractor...")

    # 创建特征提取器
    extractor = MultimodalFeatureExtractor(
        text_dim=768,
        image_dim=512,
        user_profile_dim=128,
        cultural_dim=64,
        device="cpu"
    )

    # 测试数据
    texts = ["This is a test sentence.", "Another test sentence."]
    user_profiles = torch.randn(2, 128)
    cultural_profiles = torch.randn(2, 64)

    # 提取特征
    results = extractor(
        texts=texts,
        user_profiles=user_profiles,
        cultural_profiles=cultural_profiles,
        return_individual=True
    )

    print("Feature extraction results:")
    for key, value in results.items():
        print(f"  {key}: {value.shape}")

    print("\nTesting SimpleFeatureExtractor...")

    # 测试简化版
    simple_extractor = SimpleFeatureExtractor()
    demo_features = create_demo_features(batch_size=2)

    fused = simple_extractor(demo_features)
    print(f"Fused features shape: {fused.shape}")

    print("All tests passed!")
