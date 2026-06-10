"""
食谱服务模块
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import asyncio
import json

"""
CookLikeHOC菜谱服务
提供菜谱管理、营养分析、个性化推荐等功能
"""

import logging
logger = logging.getLogger(__name__)
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc
from datetime import datetime, timedelta
import json
import math

from app.models.recipe_models import (
    Recipe, IngredientNutrition, RecipeRecommendation,
    UserRecipePreference, RecipeDataValidator
)
from app.services.cultural_scoring_service import get_cultural_scoring_service

logger = logging.getLogger(__name__)

class RecipeService:
    """菜谱服务类"""

    def __init__(self, db_session: Session):
        self.db = db_session
        self.validator = RecipeDataValidator()
        self.cultural_scoring_service = get_cultural_scoring_service()

    # ==================== 菜谱管理 ====================

    def create_recipe(self, recipe_data: Dict[str, Any]) -> Recipe:
        """创建新菜谱"""
        try:
            # 验证数据
            validated_data = self.validator.validate_recipe_data(recipe_data)

            # 计算营养信息
            nutritional_info = self._calculate_recipe_nutrition(validated_data['ingredients'])
            validated_data['nutritional_info'] = nutritional_info

            # 分析健康标签
            health_tags = self._analyze_health_tags(validated_data)
            validated_data['health_tags'] = health_tags

            # 创建菜谱对象
            recipe = Recipe(**validated_data)
            self.db.add(recipe)
            self.db.commit()
            self.db.refresh(recipe)

            logger.info(f"创建菜谱成功: {recipe.name}")
            return recipe

        except Exception as e:
            self.db.rollback()
            logger.error(f"创建菜谱失败: {e}")
            raise

    def get_recipe_by_id(self, recipe_id: int) -> Optional[Recipe]:
        """根据ID获取菜谱"""
        return self.db.query(Recipe).filter(
            Recipe.id == recipe_id,
            Recipe.is_active == True
        ).first()

    def get_recipes(self, filters: Dict[str, Any] = None,
                   limit: int = 20, offset: int = 0,
                   order_by: str = "created_at", order_direction: str = "desc") -> List[Recipe]:
        """获取菜谱列表"""
        query = self.db.query(Recipe).filter(Recipe.is_active == True)

        if filters:
            # 按类别筛选
            if filters.get('category'):
                query = query.filter(Recipe.category == filters['category'])

            # 按健康标签筛选
            if filters.get('health_tags'):
                health_tags = filters['health_tags']
                for tag in health_tags:
                    query = query.filter(Recipe.health_tags.contains([tag]))

            # 按文化标签筛选
            if filters.get('cultural_tags'):
                cultural_tags = filters['cultural_tags']
                for tag in cultural_tags:
                    query = query.filter(Recipe.cultural_tags.contains([tag]))

            # 按烹饪时间筛选
            if filters.get('max_cooking_time'):
                query = query.filter(Recipe.cooking_time <= filters['max_cooking_time'])

            # 按难度等级筛选
            if filters.get('max_difficulty'):
                query = query.filter(Recipe.difficulty_level <= filters['max_difficulty'])

            # 按关键词搜索
            if filters.get('keyword'):
                keyword = filters['keyword']
                query = query.filter(
                    or_(
                        Recipe.name.contains(keyword),
                        Recipe.description.contains(keyword),
                        Recipe.ingredients.contains([{"name": keyword}])
                    )
                )

        # 排序
        if order_by == "name":
            order_column = Recipe.name
        elif order_by == "cooking_time":
            order_column = Recipe.cooking_time
        elif order_by == "difficulty_level":
            order_column = Recipe.difficulty_level
        else:
            order_column = Recipe.created_at

        if order_direction == "asc":
            query = query.order_by(asc(order_column))
        else:
            query = query.order_by(desc(order_column))

        return query.offset(offset).limit(limit).all()

    def update_recipe(self, recipe_id: int, update_data: Dict[str, Any]) -> Optional[Recipe]:
        """更新菜谱"""
        try:
            recipe = self.get_recipe_by_id(recipe_id)
            if not recipe:
                return None

            # 更新字段
            for key, value in update_data.items():
                if hasattr(recipe, key):
                    setattr(recipe, key, value)

            # 如果更新了食材，重新计算营养信息
            if 'ingredients' in update_data:
                recipe.nutritional_info = self._calculate_recipe_nutrition(update_data['ingredients'])
                recipe.health_tags = self._analyze_health_tags(update_data)

            recipe.updated_at = datetime.utcnow()
            self.db.commit()
            self.db.refresh(recipe)

            logger.info(f"更新菜谱成功: {recipe.name}")
            return recipe

        except Exception as e:
            self.db.rollback()
            logger.error(f"更新菜谱失败: {e}")
            raise

    def delete_recipe(self, recipe_id: int) -> bool:
        """删除菜谱（软删除）"""
        try:
            recipe = self.get_recipe_by_id(recipe_id)
            if not recipe:
                return False

            recipe.is_active = False
            recipe.updated_at = datetime.utcnow()
            self.db.commit()

            logger.info(f"删除菜谱成功: {recipe.name}")
            return True

        except Exception as e:
            self.db.rollback()
            logger.error(f"删除菜谱失败: {e}")
            raise

    # ==================== 营养分析 ====================

    def _calculate_recipe_nutrition(self, ingredients: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算菜谱营养信息"""
        total_nutrition = {
            'calories': 0.0,
            'protein': 0.0,
            'carbs': 0.0,
            'fat': 0.0,
            'fiber': 0.0,
            'sugar': 0.0,
            'sodium': 0.0,
            'glycemic_index': 0.0
        }

        total_weight = 0.0

        for ingredient in ingredients:
            ingredient_name = ingredient['name']
            amount_str = ingredient.get('amount', '100g')

            # 解析数量
            amount = self._parse_amount(amount_str)
            if amount <= 0:
                continue

            # 获取食材营养信息
            nutrition = self.get_ingredient_nutrition(ingredient_name)
            if nutrition:
                weight_ratio = amount / 100.0  # 转换为100g基准
                total_weight += amount

                for nutrient in total_nutrition:
                    if nutrient in nutrition:
                        total_nutrition[nutrient] += nutrition[nutrient] * weight_ratio

        # 计算平均血糖指数
        if total_weight > 0:
            total_nutrition['glycemic_index'] = total_nutrition['glycemic_index'] / len(ingredients)

        return total_nutrition

    def _parse_amount(self, amount_str: str) -> float:
        """解析食材数量"""
        try:
            # 移除单位，提取数字
            import re
            numbers = re.findall(r'\d+\.?\d*', amount_str)
            if numbers:
                return float(numbers[0])
            return 0.0
        except:
            return 0.0

    def _analyze_health_tags(self, recipe_data: Dict[str, Any]) -> List[str]:
        """分析菜谱健康标签"""
        health_tags = []
        ingredients = recipe_data.get('ingredients', [])

        # 基于食材分析健康属性
        ingredient_names = [ing['name'] for ing in ingredients]

        # 高蛋白检测
        protein_ingredients = ['鸡肉', '猪肉', '牛肉', '鱼肉', '虾', '蛋', '豆腐', '豆类']
        if any(ing in ' '.join(ingredient_names) for ing in protein_ingredients):
            health_tags.append('高蛋白')

        # 高纤维检测
        fiber_ingredients = ['蔬菜', '水果', '全麦', '燕麦', '豆类', '坚果']
        if any(ing in ' '.join(ingredient_names) for ing in fiber_ingredients):
            health_tags.append('高纤维')

        # 低脂肪检测
        low_fat_ingredients = ['蔬菜', '水果', '瘦肉', '鱼类']
        if all(ing in ' '.join(ingredient_names) for ing in low_fat_ingredients):
            health_tags.append('低脂肪')

        # 低糖检测
        low_sugar_ingredients = ['蔬菜', '瘦肉', '豆类']
        high_sugar_ingredients = ['糖', '蜂蜜', '果酱', '甜点']
        if any(ing in ' '.join(ingredient_names) for ing in low_sugar_ingredients) and \
           not any(ing in ' '.join(ingredient_names) for ing in high_sugar_ingredients):
            health_tags.append('低糖')

        # 低钠检测
        if not any(ing in ' '.join(ingredient_names) for ing in ['盐', '酱油', '味精']):
            health_tags.append('低钠')

        return health_tags

    # ==================== 食材营养管理 ====================

    def create_ingredient_nutrition(self, nutrition_data: Dict[str, Any]) -> IngredientNutrition:
        """创建食材营养信息"""
        try:
            validated_data = self.validator.validate_ingredient_nutrition(nutrition_data)

            nutrition = IngredientNutrition(**validated_data)
            self.db.add(nutrition)
            self.db.commit()
            self.db.refresh(nutrition)

            logger.info(f"创建食材营养信息成功: {nutrition.ingredient_name}")
            return nutrition

        except Exception as e:
            self.db.rollback()
            logger.error(f"创建食材营养信息失败: {e}")
            raise

    def get_ingredient_nutrition(self, ingredient_name: str) -> Optional[Dict[str, Any]]:
        """获取食材营养信息"""
        nutrition = self.db.query(IngredientNutrition).filter(
            IngredientNutrition.ingredient_name == ingredient_name
        ).first()

        if nutrition:
            return nutrition.to_dict()
        return None

    def search_ingredients(self, keyword: str) -> List[IngredientNutrition]:
        """搜索食材"""
        return self.db.query(IngredientNutrition).filter(
            or_(
                IngredientNutrition.ingredient_name.contains(keyword),
                IngredientNutrition.common_names.contains([keyword])
            )
        ).all()

    # ==================== 个性化推荐 ====================

    def generate_personalized_recommendations(
        self,
        user_id: str,
        health_profile: Dict[str, Any],
        cultural_profile: Dict[str, Any],
        meal_type: str = "lunch",
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """生成个性化推荐"""
        try:
            # 获取用户偏好
            user_preferences = self.get_user_preferences(user_id)

            # 提取健康约束
            health_constraints = self._extract_health_constraints(health_profile)

            # 提取文化偏好
            cultural_preferences = self._extract_cultural_preferences(cultural_profile)

            # 筛选候选菜谱
            candidate_recipes = self._filter_recipes_by_constraints(
                health_constraints, cultural_preferences, meal_type
            )

            # 计算推荐分数
            scored_recipes = self._calculate_recommendation_scores(
                candidate_recipes, health_profile, cultural_profile, user_preferences
            )

            # 排序并返回推荐结果
            scored_recipes.sort(key=lambda x: x['recommendation_score'], reverse=True)

            recommendations = []
            for recipe_score in scored_recipes[:limit]:
                recipe = recipe_score['recipe']
                recommendation = {
                    'recipe': recipe.to_dict(),
                    'recommendation_score': recipe_score['recommendation_score'],
                    'health_score': recipe_score['health_score'],
                    'cultural_score': recipe_score['cultural_score'],
                    'nutritional_score': recipe_score['nutritional_score'],
                    'recommendation_reason': recipe_score['recommendation_reason']
                }
                recommendations.append(recommendation)

                # 记录推荐
                self._record_recommendation(user_id, recipe.id, meal_type, recipe_score)

            logger.info(f"为用户{user_id}生成{len(recommendations)}个推荐")
            return recommendations

        except Exception as e:
            logger.error(f"生成个性化推荐失败: {e}")
            raise

    def _extract_health_constraints(self, health_profile: Dict[str, Any]) -> Dict[str, Any]:
        """提取健康约束条件"""
        constraints = {
            'max_calories': 600,
            'max_sugar': 30,
            'max_sodium': 600,
            'min_protein': 20,
            'max_glycemic_index': 70,
            'dietary_restrictions': [],
            'health_conditions': []
        }

        # 根据用户健康档案调整约束
        if health_profile.get('diabetes'):
            constraints['max_glycemic_index'] = 55
            constraints['max_sugar'] = 15
            constraints['health_conditions'].append('diabetes')

        if health_profile.get('hypertension'):
            constraints['max_sodium'] = 400
            constraints['health_conditions'].append('hypertension')

        if health_profile.get('heart_disease'):
            constraints['max_sodium'] = 300
            constraints['max_fat'] = 20
            constraints['health_conditions'].append('heart_disease')

        return constraints

    def _extract_cultural_preferences(self, cultural_profile: Dict[str, Any]) -> Dict[str, Any]:
        """提取文化偏好"""
        return {
            'preferred_cuisines': cultural_profile.get('cuisines', ['中式']),
            'meal_times': cultural_profile.get('meal_times', {}),
            'cooking_preferences': cultural_profile.get('cooking_preferences', []),
            'cultural_tags': cultural_profile.get('cultural_tags', [])
        }

    def _filter_recipes_by_constraints(
        self,
        health_constraints: Dict[str, Any],
        cultural_preferences: Dict[str, Any],
        meal_type: str
    ) -> List[Recipe]:
        """根据约束条件筛选菜谱"""
        query = self.db.query(Recipe).filter(Recipe.is_active == True)

        # 健康约束筛选
        if health_constraints.get('max_calories'):
            # 这里需要根据营养信息筛选，简化处理
            pass

        # 文化偏好筛选
        if cultural_preferences.get('cultural_tags'):
            for tag in cultural_preferences['cultural_tags']:
                query = query.filter(Recipe.cultural_tags.contains([tag]))

        # 餐型筛选（简化处理）
        meal_categories = {
            'breakfast': ['早餐', '粥', '包子', '面条'],
            'lunch': ['炒菜', '汤', '主食'],
            'dinner': ['汤', '炖菜', '蒸菜'],
            'snack': ['凉拌', '小食']
        }

        if meal_type in meal_categories:
            categories = meal_categories[meal_type]
            query = query.filter(Recipe.category.in_(categories))

        return query.limit(50).all()  # 限制候选数量

    def _calculate_recommendation_scores(
        self,
        recipes: List[Recipe],
        health_profile: Dict[str, Any],
        cultural_profile: Dict[str, Any],
        user_preferences: Optional[UserRecipePreference]
    ) -> List[Dict[str, Any]]:
        """计算推荐分数"""
        scored_recipes = []

        for recipe in recipes:
            # 健康分数
            health_score = self._calculate_health_score(recipe, health_profile)

            # 文化适配分数
            cultural_score = self._calculate_cultural_score(recipe, cultural_profile)

            # 营养分数
            nutritional_score = self._calculate_nutritional_score(recipe, health_profile)

            # 综合推荐分数
            recommendation_score = (
                health_score * 0.4 +
                cultural_score * 0.3 +
                nutritional_score * 0.3
            )

            # 推荐理由
            recommendation_reason = self._generate_recommendation_reason(
                recipe, health_score, cultural_score, nutritional_score
            )

            scored_recipes.append({
                'recipe': recipe,
                'recommendation_score': recommendation_score,
                'health_score': health_score,
                'cultural_score': cultural_score,
                'nutritional_score': nutritional_score,
                'recommendation_reason': recommendation_reason
            })

        return scored_recipes

    def _calculate_health_score(self, recipe: Recipe, health_profile: Dict[str, Any]) -> float:
        """计算健康分数"""
        score = 0.5  # 基础分数

        # 基于健康标签加分
        health_tags = recipe.health_tags or []
        if '高蛋白' in health_tags:
            score += 0.1
        if '高纤维' in health_tags:
            score += 0.1
        if '低脂肪' in health_tags:
            score += 0.1
        if '低糖' in health_tags:
            score += 0.1
        if '低钠' in health_tags:
            score += 0.1

        # 根据健康条件调整
        if health_profile.get('diabetes') and '低糖' in health_tags:
            score += 0.2

        if health_profile.get('hypertension') and '低钠' in health_tags:
            score += 0.2

        return min(score, 1.0)

    def _calculate_cultural_score(self, recipe: Recipe, cultural_profile: Dict[str, Any]) -> float:
        """计算文化适配分数（使用Stage1模型预测）"""
        nutritional_info = recipe.nutritional_info or {}
        return self.cultural_scoring_service.calculate_cultural_score(
            recipe=recipe,
            nutritional_info=nutritional_info,
            cultural_profile=cultural_profile
        )

    def _calculate_nutritional_score(self, recipe: Recipe, health_profile: Dict[str, Any]) -> float:
        """计算营养分数"""
        score = 0.5  # 基础分数

        nutritional_info = recipe.nutritional_info or {}

        # 基于营养信息评分
        if nutritional_info.get('protein', 0) > 20:
            score += 0.1
        if nutritional_info.get('fiber', 0) > 5:
            score += 0.1
        if nutritional_info.get('sugar', 0) < 10:
            score += 0.1

        return min(score, 1.0)

    def _generate_recommendation_reason(
        self,
        recipe: Recipe,
        health_score: float,
        cultural_score: float,
        nutritional_score: float
    ) -> Dict[str, Any]:
        """生成推荐理由"""
        reasons = []

        if health_score > 0.7:
            reasons.append("符合您的健康需求")

        if cultural_score > 0.7:
            reasons.append("符合您的饮食文化偏好")

        if nutritional_score > 0.7:
            reasons.append("营养搭配均衡")

        health_tags = recipe.health_tags or []
        if health_tags:
            reasons.append(f"具有{', '.join(health_tags)}特点")

        return {
            'reasons': reasons,
            'scores': {
                'health': health_score,
                'cultural': cultural_score,
                'nutritional': nutritional_score
            }
        }

    def _record_recommendation(
        self,
        user_id: str,
        recipe_id: int,
        meal_type: str,
        recipe_score: Dict[str, Any]
    ):
        """记录推荐"""
        try:
            recommendation = RecipeRecommendation(
                user_id=user_id,
                recipe_id=recipe_id,
                meal_type=meal_type,
                recommendation_score=recipe_score['recommendation_score'],
                health_score=recipe_score['health_score'],
                cultural_score=recipe_score['cultural_score'],
                nutritional_score=recipe_score['nutritional_score'],
                recommendation_reason=recipe_score['recommendation_reason']
            )
            self.db.add(recommendation)
            self.db.commit()
        except Exception as e:
            logger.error(f"记录推荐失败: {e}")

    # ==================== 用户偏好管理 ====================

    def get_user_preferences(self, user_id: str) -> Optional[UserRecipePreference]:
        """获取用户偏好"""
        return self.db.query(UserRecipePreference).filter(
            UserRecipePreference.user_id == user_id
        ).first()

    def update_user_preferences(self, user_id: str, preferences_data: Dict[str, Any]) -> UserRecipePreference:
        """更新用户偏好"""
        try:
            preferences = self.get_user_preferences(user_id)

            if not preferences:
                preferences = UserRecipePreference(user_id=user_id)
                self.db.add(preferences)

            # 更新字段
            for key, value in preferences_data.items():
                if hasattr(preferences, key):
                    setattr(preferences, key, value)

            preferences.updated_at = datetime.utcnow()
            self.db.commit()
            self.db.refresh(preferences)

            logger.info(f"更新用户偏好成功: {user_id}")
            return preferences

        except Exception as e:
            self.db.rollback()
            logger.error(f"更新用户偏好失败: {e}")
            raise

    def get_user_recommendation_history(self, user_id: str, limit: int = 20) -> List[RecipeRecommendation]:
        """获取用户推荐历史"""
        return self.db.query(RecipeRecommendation).filter(
            RecipeRecommendation.user_id == user_id
        ).order_by(desc(RecipeRecommendation.created_at)).limit(limit).all()

    def update_recommendation_feedback(
        self,
        recommendation_id: int,
        user_satisfaction: int,
        feedback_notes: Optional[str] = None
    ) -> Optional[RecipeRecommendation]:
        """更新推荐反馈"""
        try:
            recommendation = self.db.query(RecipeRecommendation).filter(
                RecipeRecommendation.id == recommendation_id
            ).first()

            if not recommendation:
                return None

            recommendation.user_satisfaction = user_satisfaction
            recommendation.feedback_notes = feedback_notes
            recommendation.updated_at = datetime.utcnow()

            self.db.commit()
            self.db.refresh(recommendation)

            logger.info(f"更新推荐反馈成功: {recommendation_id}")
            return recommendation

        except Exception as e:
            self.db.rollback()
            logger.error(f"更新推荐反馈失败: {e}")
            raise

__all__ = ["'logger'", "'RecipeService'"]
