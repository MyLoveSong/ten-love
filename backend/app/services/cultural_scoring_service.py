"""
文化评分服务
使用Stage1训练模型进行文化适配评分预测
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Optional, Any
import torch

project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from system.projects.nutrition.src.cultural_finetune import CulturalStage1Trainer, _build_feature_vector

logger = logging.getLogger(__name__)


class CulturalScoringService:
    """文化评分服务：封装Stage1模型预测功能"""

    _instance: Optional['CulturalScoringService'] = None
    _initialized: bool = False

    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化服务（仅执行一次）"""
        if self._initialized:
            return

        self.trainer: Optional[CulturalStage1Trainer] = None
        self.model_path: Optional[Path] = None
        self._initialized = True

    def initialize(self, model_path: Optional[Path] = None) -> bool:
        """
        初始化模型

        Args:
            model_path: 模型权重路径，默认使用最佳模型

        Returns:
            bool: 初始化是否成功
        """
        try:
            if model_path is None:
                model_path = project_root / 'TRAIN' / 'outputs' / 'stage1_cultural' / 'cultural_lora_weights_best.pt'

            if not model_path.exists():
                logger.warning(f"模型文件不存在: {model_path}，使用规则计算")
                return False

            self.model_path = model_path

            # 初始化训练器（与训练时参数一致）
            self.trainer = CulturalStage1Trainer(
                input_dim=256,
                hidden_dim=128,
                num_experts=4,
                expert_dim=64,
                lora_rank=16,
                lora_alpha=32.0
            )

            # 加载权重
            self.trainer.load_lora(model_path)

            logger.info(f"文化评分模型初始化成功: {model_path}")
            return True

        except Exception as e:
            logger.error(f"文化评分模型初始化失败: {e}")
            self.trainer = None
            return False

    def is_available(self) -> bool:
        """检查模型是否可用"""
        return self.trainer is not None

    def calculate_cultural_score(
        self,
        recipe: Any,
        nutritional_info: Dict[str, Any],
        cultural_profile: Dict[str, Any]
    ) -> float:
        """
        计算文化适配分数

        Args:
            recipe: Recipe对象或包含菜品信息的字典
            nutritional_info: 营养信息字典
            cultural_profile: 用户文化偏好配置

        Returns:
            float: 文化适配分数 [0, 1]
        """
        if not self.is_available():
            logger.warning("模型未初始化，使用规则计算")
            return self._calculate_cultural_score_rule_based(recipe, cultural_profile)

        try:
            # 构建特征向量
            feature_dict = self._extract_features_from_recipe(recipe, nutritional_info)
            feature_vector = _build_feature_vector(feature_dict)

            if feature_vector is None:
                logger.warning("无法构建特征向量，使用规则计算")
                return self._calculate_cultural_score_rule_based(recipe, cultural_profile)

            # 转换为tensor并预测
            features_tensor = torch.tensor([feature_vector], dtype=torch.float32)
            prediction = self.trainer.predict(features_tensor)

            # 提取预测值（取平均值，因为输出是256维）
            cultural_score = float(prediction['distilled_output'].mean().item())

            # 确保在[0, 1]范围内
            cultural_score = max(0.0, min(1.0, cultural_score))

            return cultural_score

        except Exception as e:
            logger.error(f"模型预测失败: {e}，回退到规则计算")
            return self._calculate_cultural_score_rule_based(recipe, cultural_profile)

    def _extract_features_from_recipe(
        self,
        recipe: Any,
        nutritional_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """从Recipe对象或字典中提取特征"""
        if isinstance(recipe, dict):
            food_name = recipe.get('name') or recipe.get('food_name', '')
            return {
                'food_name': food_name,
                'calories': nutritional_info.get('calories', 0),
                'protein': nutritional_info.get('protein', 0),
                'carbs': nutritional_info.get('carbs', 0),
                'fat': nutritional_info.get('fat', 0),
                'fiber': nutritional_info.get('fiber', 0),
                'sugar': nutritional_info.get('sugar', 0),
                'sodium': nutritional_info.get('sodium', 0)
            }
        else:
            # Recipe对象
            return {
                'food_name': getattr(recipe, 'name', ''),
                'calories': nutritional_info.get('calories', 0),
                'protein': nutritional_info.get('protein', 0),
                'carbs': nutritional_info.get('carbs', 0),
                'fat': nutritional_info.get('fat', 0),
                'fiber': nutritional_info.get('fiber', 0),
                'sugar': nutritional_info.get('sugar', 0),
                'sodium': nutritional_info.get('sodium', 0)
            }

    def _calculate_cultural_score_rule_based(
        self,
        recipe: Any,
        cultural_profile: Dict[str, Any]
    ) -> float:
        """规则基础的文化分数计算（回退方案）"""
        score = 0.5  # 基础分数

        if hasattr(recipe, 'cultural_tags'):
            recipe_cultural_tags = recipe.cultural_tags or []
        elif isinstance(recipe, dict):
            recipe_cultural_tags = recipe.get('cultural_tags', [])
        else:
            recipe_cultural_tags = []

        user_cultural_tags = cultural_profile.get('cultural_tags', [])

        if user_cultural_tags:
            match_count = sum(1 for tag in user_cultural_tags if tag in recipe_cultural_tags)
            score += (match_count / len(user_cultural_tags)) * 0.5

        return min(score, 1.0)


def get_cultural_scoring_service() -> CulturalScoringService:
    """获取文化评分服务单例"""
    service = CulturalScoringService()
    if not service.is_available():
        service.initialize()
    return service
