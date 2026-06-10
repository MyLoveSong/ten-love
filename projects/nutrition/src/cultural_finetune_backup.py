#!/usr/bin/env python3
"""
Stage1: 文化特征提取器 - 简化的可运行版本
用于替代缺失的backend模块依赖
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CulturalFeatureExtractor(nn.Module):
    """
    文化特征提取器
    从原始文化输入（8维）提取文化特征（64维）
    """

    def __init__(self, model_name: str = "bert-base-chinese", cultural_dim: int = 64):
        super().__init__()
        self.model_name = model_name
        self.cultural_dim = cultural_dim

        # 文化特征编码器
        # 输入: 8维原始文化特征
        # 输出: 64维文化嵌入
        self.cultural_encoder = nn.Sequential(
            nn.Linear(8, cultural_dim // 2),
            nn.LayerNorm(cultural_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(cultural_dim // 2, cultural_dim),
            nn.LayerNorm(cultural_dim)
        )

        # 用户-文化交互编码器
        self.user_cultural_interaction = nn.Sequential(
            nn.Linear(128 + cultural_dim, cultural_dim),  # 用户特征(128) + 文化特征(cultural_dim)
            nn.LayerNorm(cultural_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(cultural_dim, cultural_dim)
        )

        logger.info(f"CulturalFeatureExtractor initialized: model={model_name}, cultural_dim={cultural_dim}")

    def extract_cultural_features(self, cultural_raw: torch.Tensor,
                                user_profile: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        提取文化特征

        Args:
            cultural_raw: 原始文化特征 [batch_size, 8]
            user_profile: 用户基本信息 [batch_size, 128], 可选

        Returns:
            文化特征嵌入 [batch_size, cultural_dim]
        """
        # 基础文化编码
        cultural_embed = self.cultural_encoder(cultural_raw)

        # 如果提供了用户profile，进行交互编码
        if user_profile is not None:
            combined = torch.cat([user_profile, cultural_embed], dim=1)
            cultural_embed = self.user_cultural_interaction(combined)

        return cultural_embed

    def forward(self, cultural_raw: torch.Tensor,
               user_profile: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播"""
        return self.extract_cultural_features(cultural_raw, user_profile)

    def load_pretrained_model(self, model_path: str):
        """加载预训练模型"""
        try:
            if Path(model_path).exists():
                state_dict = torch.load(model_path, map_location='cpu')
                self.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded pretrained model from {model_path}")
            else:
                logger.warning(f"Model path does not exist: {model_path}")
        except Exception as e:
            logger.warning(f"Could not load pretrained model: {e}")

    def save_model(self, save_path: str):
        """保存模型"""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")


# 兼容性别名
CulturalFeatureExtractorClass = CulturalFeatureExtractor


def create_cultural_extractor(model_name: str = "bert-base-chinese",
                            cultural_dim: int = 64,
                            pretrained_path: Optional[str] = None) -> CulturalFeatureExtractor:
    """
    创建文化特征提取器

    Args:
        model_name: 模型名称
        cultural_dim: 文化特征维度
        pretrained_path: 预训练模型路径

    Returns:
        CulturalFeatureExtractor实例
    """
    extractor = CulturalFeatureExtractor(model_name, cultural_dim)

    if pretrained_path:
        extractor.load_pretrained_model(pretrained_path)

    return extractor


# 测试函数
def test_cultural_extractor():
    """测试文化特征提取器"""
    extractor = create_cultural_extractor()

    # 测试数据
    batch_size = 4
    cultural_raw = torch.randn(batch_size, 8)
    user_profile = torch.randn(batch_size, 128)

    # 测试基本提取
    features_basic = extractor.extract_cultural_features(cultural_raw)
    print(f"Basic extraction: {cultural_raw.shape} -> {features_basic.shape}")

    # 测试带用户信息的提取
    features_with_user = extractor.extract_cultural_features(cultural_raw, user_profile)
    print(f"With user profile: {features_with_user.shape}")

    print("✓ CulturalFeatureExtractor test passed!")


if __name__ == "__main__":
    test_cultural_extractor()
