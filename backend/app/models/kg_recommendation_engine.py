"""
基于知识图谱的个性化健康营养推荐系统
结合用户画像和知识图谱推理的智能推荐
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import networkx as nx

from app.models.knowledge_graph import KnowledgeGraphBuilder, Entity, Relation, KnowledgeGraphConfig

logger = logging.getLogger(__name__)

@dataclass
class UserProfile:
    """用户画像"""
    user_id: str
    age: int
    gender: str
    height: float
    weight: float
    region: str
    health_conditions: List[str]
    dietary_restrictions: List[str]
    allergies: List[str]
    activity_level: str
    target_goals: List[str]
    bmi: float = 0.0
    food_preferences: List[str] = None
    cultural_background: str = ""
    created_at: datetime = None
    updated_at: datetime = None

    def __post_init__(self):
        if self.bmi == 0.0:
            self.bmi = self.weight / ((self.height / 100) ** 2)
        if self.food_preferences is None:
            self.food_preferences = []
        if self.cultural_background == "":
            self.cultural_background = self.region
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()

@dataclass
class RecommendationItem:
    """推荐项目"""
    item_id: str
    item_type: str  # food, recipe, exercise
    name: str
    score: float
    confidence: float
    reasoning: str
    nutritional_info: Dict[str, Any]
    health_benefits: List[str]
    cultural_fit: float
    personal_fit: float

@dataclass
class RecommendationResult:
    """推荐结果"""
    user_id: str
    recommendations: List[RecommendationItem]
    total_score: float
    reasoning_summary: str
    generated_at: datetime
    valid_until: datetime

class UserProfileBuilder:
    """用户画像构建器"""

    def __init__(self, kg_builder: KnowledgeGraphBuilder):
        self.kg_builder = kg_builder

    def build_user_profile(self, user_data: Dict[str, Any]) -> UserProfile:
        """构建用户画像"""
        try:
            # 计算BMI
            height_m = user_data['height'] / 100
            bmi = user_data['weight'] / (height_m ** 2)

            # 分析健康状态
            health_conditions = self._analyze_health_conditions(user_data, bmi)

            # 分析饮食偏好
            food_preferences = self._analyze_food_preferences(user_data)

            # 分析文化背景
            cultural_background = self._analyze_cultural_background(user_data)

            profile = UserProfile(
                user_id=user_data['user_id'],
                age=user_data['age'],
                gender=user_data['gender'],
                height=user_data['height'],
                weight=user_data['weight'],
                bmi=bmi,
                health_conditions=health_conditions,
                dietary_restrictions=user_data.get('dietary_restrictions', []),
                food_preferences=food_preferences,
                allergies=user_data.get('allergies', []),
                activity_level=user_data.get('activity_level', 'moderate'),
                target_goals=user_data.get('target_goals', []),
                cultural_background=cultural_background,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )

            # 将用户画像存储到知识图谱
            self._store_user_profile(profile)

            return profile

        except Exception as e:
            logger.error(f"用户画像构建失败: {e}")
            raise

    def _analyze_health_conditions(self, user_data: Dict[str, Any], bmi: float) -> List[str]:
        """分析健康状态"""
        conditions = []

        # BMI分析
        if bmi < 18.5:
            conditions.append('underweight')
        elif bmi > 25:
            conditions.append('overweight')
        if bmi > 30:
            conditions.append('obesity')

        # 年龄相关
        age = user_data['age']
        if age > 65:
            conditions.append('elderly')
        elif age < 18:
            conditions.append('adolescent')

        # 性别相关
        gender = user_data['gender']
        if gender == 'female' and age > 50:
            conditions.append('postmenopausal')

        # 已有疾病
        existing_conditions = user_data.get('health_conditions', [])
        conditions.extend(existing_conditions)

        return list(set(conditions))

    def _analyze_food_preferences(self, user_data: Dict[str, Any]) -> List[str]:
        """分析饮食偏好"""
        preferences = []

        # 基于文化背景
        region = user_data.get('region', '')
        if region in ['四川', '重庆']:
            preferences.extend(['spicy', 'sichuan_cuisine'])
        elif region in ['广东', '香港']:
            preferences.extend(['light', 'cantonese_cuisine'])
        elif region in ['北京', '天津']:
            preferences.extend(['northern_cuisine'])

        # 基于饮食习惯
        dietary_habits = user_data.get('dietary_habits', [])
        preferences.extend(dietary_habits)

        return list(set(preferences))

    def _analyze_cultural_background(self, user_data: Dict[str, Any]) -> str:
        """分析文化背景"""
        region = user_data.get('region', '全国')
        return region

    def _store_user_profile(self, profile: UserProfile):
        """将用户画像存储到知识图谱"""
        try:
            user_entity = Entity(
                id=f"user_{profile.user_id}",
                name=f"User_{profile.user_id}",
                entity_type='User',
                properties={
                    'age': profile.age,
                    'gender': profile.gender,
                    'bmi': profile.bmi,
                    'health_conditions': profile.health_conditions,
                    'dietary_restrictions': profile.dietary_restrictions,
                    'food_preferences': profile.food_preferences,
                    'allergies': profile.allergies,
                    'activity_level': profile.activity_level,
                    'target_goals': profile.target_goals,
                    'cultural_background': profile.cultural_background
                }
            )

            self.kg_builder.add_entity(user_entity)

        except Exception as e:
            logger.error(f"用户画像存储失败: {e}")

class KnowledgeGraphRecommendationEngine:
    """基于知识图谱的推荐引擎"""

    def __init__(self, kg_builder: KnowledgeGraphBuilder):
        self.kg_builder = kg_builder
        self.user_profile_builder = UserProfileBuilder(kg_builder)

    def generate_recommendations(
        self,
        user_profile: UserProfile,
        recommendation_type: str = 'food',
        num_recommendations: int = 10
    ) -> RecommendationResult:
        """生成个性化推荐"""
        try:
            logger.info(f"为用户 {user_profile.user_id} 生成 {recommendation_type} 推荐")

            if recommendation_type == 'food':
                recommendations = self._recommend_foods(user_profile, num_recommendations)
            elif recommendation_type == 'recipe':
                recommendations = self._recommend_recipes(user_profile, num_recommendations)
            elif recommendation_type == 'exercise':
                recommendations = self._recommend_exercises(user_profile, num_recommendations)
            else:
                raise ValueError(f"不支持的推荐类型: {recommendation_type}")

            # 计算总评分
            total_score = sum(rec.score for rec in recommendations) / len(recommendations) if recommendations else 0

            # 生成推理总结
            reasoning_summary = self._generate_reasoning_summary(user_profile, recommendations)

            result = RecommendationResult(
                user_id=user_profile.user_id,
                recommendations=recommendations,
                total_score=total_score,
                reasoning_summary=reasoning_summary,
                generated_at=datetime.now(),
                valid_until=datetime.now() + timedelta(days=7)
            )

            logger.info(f"推荐生成完成，共 {len(recommendations)} 个推荐")
            return result

        except Exception as e:
            logger.error(f"推荐生成失败: {e}")
            raise

    def _recommend_foods(self, user_profile: UserProfile, num_recommendations: int) -> List[RecommendationItem]:
        """推荐食物"""
        try:
            # 获取所有食物实体
            food_entities = self.kg_builder.query_entities('Food')

            recommendations = []

            for food_data in food_entities:
                food_id = food_data['id']
                food_name = food_data['name']
                food_properties = json.loads(food_data['properties']) if isinstance(food_data['properties'], str) else food_data['properties']

                # 计算推荐分数
                score, confidence, reasoning = self._calculate_food_score(
                    user_profile, food_id, food_name, food_properties
                )

                if score > 0.3:  # 只推荐分数较高的食物
                    # 获取健康益处
                    health_benefits = self._get_food_health_benefits(food_id)

                    # 计算文化适配度
                    cultural_fit = self._calculate_cultural_fit(user_profile, food_name)

                    # 计算个人适配度
                    personal_fit = self._calculate_personal_fit(user_profile, food_properties)

                    recommendation = RecommendationItem(
                        item_id=food_id,
                        item_type='food',
                        name=food_name,
                        score=score,
                        confidence=confidence,
                        reasoning=reasoning,
                        nutritional_info=food_properties,
                        health_benefits=health_benefits,
                        cultural_fit=cultural_fit,
                        personal_fit=personal_fit
                    )

                    recommendations.append(recommendation)

            # 按分数排序并返回前N个
            recommendations.sort(key=lambda x: x.score, reverse=True)
            return recommendations[:num_recommendations]

        except Exception as e:
            logger.error(f"食物推荐失败: {e}")
            return []

    def _calculate_food_score(
        self,
        user_profile: UserProfile,
        food_id: str,
        food_name: str,
        food_properties: Dict[str, Any]
    ) -> Tuple[float, float, str]:
        """计算食物推荐分数"""
        try:
            score = 0.0
            confidence = 0.0
            reasoning_parts = []

            # 1. 健康状态匹配
            health_score = self._calculate_health_match_score(user_profile, food_id)
            score += health_score * 0.4
            confidence += 0.4
            if health_score > 0.7:
                reasoning_parts.append(f"适合您的健康状况")

            # 2. 营养需求匹配
            nutrition_score = self._calculate_nutrition_match_score(user_profile, food_properties)
            score += nutrition_score * 0.3
            confidence += 0.3
            if nutrition_score > 0.7:
                reasoning_parts.append(f"满足您的营养需求")

            # 3. 饮食限制检查
            restriction_penalty = self._check_dietary_restrictions(user_profile, food_name, food_properties)
            score -= restriction_penalty * 0.2
            if restriction_penalty > 0:
                reasoning_parts.append(f"符合您的饮食限制")

            # 4. 过敏检查
            allergy_penalty = self._check_allergies(user_profile, food_name)
            score -= allergy_penalty * 0.3
            if allergy_penalty > 0:
                reasoning_parts.append(f"无过敏风险")

            # 5. 文化适配
            cultural_score = self._calculate_cultural_fit(user_profile, food_name)
            score += cultural_score * 0.1
            confidence += 0.1

            # 确保分数在0-1之间
            score = max(0, min(1, score))
            confidence = max(0, min(1, confidence))

            reasoning = "; ".join(reasoning_parts) if reasoning_parts else "基于综合评估"

            return score, confidence, reasoning

        except Exception as e:
            logger.error(f"食物分数计算失败: {e}")
            return 0.0, 0.0, "计算失败"

    def _calculate_health_match_score(self, user_profile: UserProfile, food_id: str) -> float:
        """计算健康状态匹配分数"""
        try:
            score = 0.5  # 基础分数

            # 查找食物与疾病的关系
            beneficial_relations = self.kg_builder.find_related_entities(food_id, 'BENEFICIAL_FOR')
            harmful_relations = self.kg_builder.find_related_entities(food_id, 'HARMFUL_FOR')

            # 检查是否有益于用户的健康状态
            for condition in user_profile.health_conditions:
                condition_id = f"disease_{condition}"

                # 检查有益关系
                for relation in beneficial_relations:
                    if relation['id'] == condition_id:
                        score += relation['weight'] * 0.3

                # 检查有害关系
                for relation in harmful_relations:
                    if relation['id'] == condition_id:
                        score -= relation['weight'] * 0.5

            return max(0, min(1, score))

        except Exception as e:
            logger.error(f"健康匹配分数计算失败: {e}")
            return 0.5

    def _calculate_nutrition_match_score(self, user_profile: UserProfile, food_properties: Dict[str, Any]) -> float:
        """计算营养需求匹配分数"""
        try:
            score = 0.5  # 基础分数

            # 根据BMI调整营养需求
            if user_profile.bmi < 18.5:  # 偏瘦
                # 需要更多卡路里和蛋白质
                calories = food_properties.get('calories', 0)
                protein = food_properties.get('protein', 0)
                if calories > 200:
                    score += 0.2
                if protein > 10:
                    score += 0.2

            elif user_profile.bmi > 25:  # 超重
                # 需要低卡路里、高纤维
                calories = food_properties.get('calories', 0)
                fiber = food_properties.get('fiber', 0)
                if calories < 150:
                    score += 0.2
                if fiber > 5:
                    score += 0.2

            # 根据活动水平调整
            if user_profile.activity_level == 'high':
                # 需要更多蛋白质
                protein = food_properties.get('protein', 0)
                if protein > 15:
                    score += 0.1

            return max(0, min(1, score))

        except Exception as e:
            logger.error(f"营养匹配分数计算失败: {e}")
            return 0.5

    def _check_dietary_restrictions(self, user_profile: UserProfile, food_name: str, food_properties: Dict[str, Any]) -> float:
        """检查饮食限制"""
        penalty = 0.0

        for restriction in user_profile.dietary_restrictions:
            if restriction.lower() in food_name.lower():
                penalty += 0.5

            # 检查营养成分
            if restriction == 'low_sodium':
                sodium = food_properties.get('sodium', 0)
                if sodium > 200:
                    penalty += 0.3

            elif restriction == 'low_sugar':
                sugar = food_properties.get('sugar', 0)
                if sugar > 10:
                    penalty += 0.3

        return penalty

    def _check_allergies(self, user_profile: UserProfile, food_name: str) -> float:
        """检查过敏风险"""
        penalty = 0.0

        for allergy in user_profile.allergies:
            if allergy.lower() in food_name.lower():
                penalty += 1.0  # 完全排除过敏食物

        return penalty

    def _calculate_cultural_fit(self, user_profile: UserProfile, food_name: str) -> float:
        """计算文化适配度"""
        try:
            cultural_background = user_profile.cultural_background

            # 简化的文化适配逻辑
            cultural_foods = {
                '四川': ['川菜', '麻辣', '火锅', '宫保鸡丁', '麻婆豆腐'],
                '广东': ['粤菜', '清淡', '白切鸡', '清蒸鱼', '煲汤'],
                '北京': ['京菜', '烤鸭', '炸酱面', '豆汁'],
                '上海': ['本帮菜', '小笼包', '红烧肉', '糖醋排骨']
            }

            if cultural_background in cultural_foods:
                for cultural_food in cultural_foods[cultural_background]:
                    if cultural_food in food_name:
                        return 0.8

            return 0.5  # 默认适配度

        except Exception as e:
            logger.error(f"文化适配度计算失败: {e}")
            return 0.5

    def _calculate_personal_fit(self, user_profile: UserProfile, food_properties: Dict[str, Any]) -> float:
        """计算个人适配度"""
        try:
            fit_score = 0.5

            # 基于食物偏好
            for preference in user_profile.food_preferences:
                if preference in food_properties.get('category', ''):
                    fit_score += 0.2

            # 基于目标
            for goal in user_profile.target_goals:
                if goal == 'weight_loss':
                    calories = food_properties.get('calories', 0)
                    if calories < 100:
                        fit_score += 0.2
                elif goal == 'muscle_gain':
                    protein = food_properties.get('protein', 0)
                    if protein > 15:
                        fit_score += 0.2

            return max(0, min(1, fit_score))

        except Exception as e:
            logger.error(f"个人适配度计算失败: {e}")
            return 0.5

    def _get_food_health_benefits(self, food_id: str) -> List[str]:
        """获取食物健康益处"""
        try:
            benefits = []

            # 查找有益关系
            beneficial_relations = self.kg_builder.find_related_entities(food_id, 'BENEFICIAL_FOR')

            for relation in beneficial_relations:
                disease_name = relation['name']
                benefits.append(f"有助于{disease_name}的预防和管理")

            return benefits

        except Exception as e:
            logger.error(f"健康益处获取失败: {e}")
            return []

    def _recommend_recipes(self, user_profile: UserProfile, num_recommendations: int) -> List[RecommendationItem]:
        """推荐食谱（预留接口）"""
        # 这里可以实现基于知识图谱的食谱推荐
        return []

    def _recommend_exercises(self, user_profile: UserProfile, num_recommendations: int) -> List[RecommendationItem]:
        """推荐运动（预留接口）"""
        # 这里可以实现基于知识图谱的运动推荐
        return []

    def _generate_reasoning_summary(self, user_profile: UserProfile, recommendations: List[RecommendationItem]) -> str:
        """生成推理总结"""
        try:
            summary_parts = []

            # 基于健康状态
            if user_profile.health_conditions:
                conditions_str = "、".join(user_profile.health_conditions)
                summary_parts.append(f"考虑到您的健康状况（{conditions_str}）")

            # 基于目标
            if user_profile.target_goals:
                goals_str = "、".join(user_profile.target_goals)
                summary_parts.append(f"结合您的目标（{goals_str}）")

            # 基于文化背景
            if user_profile.cultural_background:
                summary_parts.append(f"融入您的文化背景（{user_profile.cultural_background}）")

            # 基于推荐质量
            if recommendations:
                avg_score = sum(rec.score for rec in recommendations) / len(recommendations)
                if avg_score > 0.8:
                    summary_parts.append("为您精选了高匹配度的推荐")
                elif avg_score > 0.6:
                    summary_parts.append("为您提供了合适的推荐")

            return "，".join(summary_parts) + "。"

        except Exception as e:
            logger.error(f"推理总结生成失败: {e}")
            return "基于您的个人资料和健康需求生成推荐。"

def create_recommendation_engine(kg_builder: KnowledgeGraphBuilder) -> KnowledgeGraphRecommendationEngine:
    """创建推荐引擎"""
    return KnowledgeGraphRecommendationEngine(kg_builder)

if __name__ == "__main__":
    # 测试推荐系统
    from app.models.knowledge_graph import create_knowledge_graph_builder

    # 创建知识图谱构建器
    kg_builder = create_knowledge_graph_builder()

    # 创建推荐引擎
    recommendation_engine = create_recommendation_engine(kg_builder)

    # 示例用户数据
    user_data = {
        'user_id': 'test_user_001',
        'age': 35,
        'gender': 'male',
        'height': 175,
        'weight': 75,
        'region': '四川',
        'health_conditions': ['糖尿病'],
        'dietary_restrictions': ['low_sugar'],
        'allergies': ['花生'],
        'activity_level': 'moderate',
        'target_goals': ['weight_loss']
    }

    # 构建用户画像
    user_profile = recommendation_engine.user_profile_builder.build_user_profile(user_data)

    # 生成推荐
    recommendations = recommendation_engine.generate_recommendations(
        user_profile,
        recommendation_type='food',
        num_recommendations=5
    )

    print(f"为用户 {user_profile.user_id} 生成了 {len(recommendations.recommendations)} 个推荐")
    for rec in recommendations.recommendations:
        print(f"- {rec.name}: {rec.score:.2f} ({rec.reasoning})")

    print(f"推理总结: {recommendations.reasoning_summary}")

    kg_builder.close()
