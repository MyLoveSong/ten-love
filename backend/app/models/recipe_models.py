"""
食谱模型定义
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import Optional, Dict, Any

"""
CookLikeHOC菜谱数据模型
集成到健康监测系统的菜谱管理模块
"""

from sqlalchemy import Column, Integer, String, Text, JSON, DateTime, Float, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
Base = declarative_base()
from typing import Dict, List, Optional, Any
import json

Base = declarative_base()

class Recipe(Base):
    """菜谱数据模型"""
    __tablename__ = "recipes"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, index=True)
    category = Column(String(50), nullable=False, index=True)  # 主食、凉拌、卤菜等
    description = Column(Text)
    ingredients = Column(JSON)  # 食材列表
    cooking_method = Column(String(100))  # 烹饪方式
    cultural_tags = Column(JSON)  # 文化标签
    nutritional_info = Column(JSON)  # 营养信息
    health_tags = Column(JSON)  # 健康标签
    difficulty_level = Column(Integer, default=1)  # 难度等级 1-5
    cooking_time = Column(Integer)  # 烹饪时间(分钟)
    servings = Column(Integer, default=1)  # 份数
    source = Column(String(100), default="CookLikeHOC")  # 来源
    image_url = Column(String(500))  # 图片URL
    instructions = Column(Text)  # 制作步骤
    tips = Column(Text)  # 制作小贴士
    is_active = Column(Boolean, default=True)  # 是否启用
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'id': self.id,
            'name': self.name,
            'category': self.category,
            'description': self.description,
            'ingredients': self.ingredients,
            'cooking_method': self.cooking_method,
            'cultural_tags': self.cultural_tags,
            'nutritional_info': self.nutritional_info,
            'health_tags': self.health_tags,
            'difficulty_level': self.difficulty_level,
            'cooking_time': self.cooking_time,
            'servings': self.servings,
            'source': self.source,
            'image_url': self.image_url,
            'instructions': self.instructions,
            'tips': self.tips,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class IngredientNutrition(Base):
    """食材营养数据库"""
    __tablename__ = "ingredient_nutrition"

    id = Column(Integer, primary_key=True, index=True)
    ingredient_name = Column(String(100), nullable=False, index=True)
    calories_per_100g = Column(Float, default=0.0)
    protein = Column(Float, default=0.0)  # 蛋白质(g/100g)
    carbs = Column(Float, default=0.0)  # 碳水化合物(g/100g)
    fat = Column(Float, default=0.0)  # 脂肪(g/100g)
    fiber = Column(Float, default=0.0)  # 纤维(g/100g)
    sugar = Column(Float, default=0.0)  # 糖(g/100g)
    sodium = Column(Float, default=0.0)  # 钠(mg/100g)
    glycemic_index = Column(Float, default=0.0)  # 血糖指数
    health_benefits = Column(JSON)  # 健康益处
    cultural_significance = Column(Text)  # 文化意义
    common_names = Column(JSON)  # 常见名称
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'id': self.id,
            'ingredient_name': self.ingredient_name,
            'calories_per_100g': self.calories_per_100g,
            'protein': self.protein,
            'carbs': self.carbs,
            'fat': self.fat,
            'fiber': self.fiber,
            'sugar': self.sugar,
            'sodium': self.sodium,
            'glycemic_index': self.glycemic_index,
            'health_benefits': self.health_benefits,
            'cultural_significance': self.cultural_significance,
            'common_names': self.common_names,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class RecipeRecommendation(Base):
    """菜谱推荐记录"""
    __tablename__ = "recipe_recommendations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(50), nullable=False, index=True)
    recipe_id = Column(Integer, nullable=False, index=True)
    meal_type = Column(String(20), nullable=False)  # breakfast, lunch, dinner, snack
    recommendation_score = Column(Float, nullable=False)  # 推荐分数
    health_score = Column(Float)  # 健康分数
    cultural_score = Column(Float)  # 文化适配分数
    nutritional_score = Column(Float)  # 营养分数
    user_satisfaction = Column(Integer)  # 用户满意度 1-5
    feedback_notes = Column(Text)  # 反馈备注
    recommendation_reason = Column(JSON)  # 推荐理由
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'recipe_id': self.recipe_id,
            'meal_type': self.meal_type,
            'recommendation_score': self.recommendation_score,
            'health_score': self.health_score,
            'cultural_score': self.cultural_score,
            'nutritional_score': self.nutritional_score,
            'user_satisfaction': self.user_satisfaction,
            'feedback_notes': self.feedback_notes,
            'recommendation_reason': self.recommendation_reason,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class UserRecipePreference(Base):
    """用户菜谱偏好"""
    __tablename__ = "user_recipe_preferences"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(50), nullable=False, index=True)
    preferred_categories = Column(JSON)  # 偏好菜谱类别
    preferred_ingredients = Column(JSON)  # 偏好食材
    dietary_restrictions = Column(JSON)  # 饮食限制
    health_goals = Column(JSON)  # 健康目标
    cultural_background = Column(JSON)  # 文化背景
    cooking_skill_level = Column(Integer, default=1)  # 烹饪技能等级
    available_cooking_time = Column(Integer)  # 可用烹饪时间(分钟)
    kitchen_equipment = Column(JSON)  # 厨房设备
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'preferred_categories': self.preferred_categories,
            'preferred_ingredients': self.preferred_ingredients,
            'dietary_restrictions': self.dietary_restrictions,
            'health_goals': self.health_goals,
            'cultural_background': self.cultural_background,
            'cooking_skill_level': self.cooking_skill_level,
            'available_cooking_time': self.available_cooking_time,
            'kitchen_equipment': self.kitchen_equipment,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

# 数据模型验证和工具函数
class RecipeDataValidator:
    """菜谱数据验证器"""

    @staticmethod
    def validate_recipe_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """验证菜谱数据"""
        required_fields = ['name', 'category', 'ingredients']
        for field in required_fields:
            if field not in data or not data[field]:
                raise ValueError(f"缺少必需字段: {field}")

        # 验证食材格式
        if not isinstance(data['ingredients'], list):
            raise ValueError("ingredients必须是列表格式")

        for ingredient in data['ingredients']:
            if not isinstance(ingredient, dict) or 'name' not in ingredient:
                raise ValueError("每个食材必须包含name字段")

        return data

    @staticmethod
    def validate_ingredient_nutrition(data: Dict[str, Any]) -> Dict[str, Any]:
        """验证食材营养数据"""
        required_fields = ['ingredient_name']
        for field in required_fields:
            if field not in data or not data[field]:
                raise ValueError(f"缺少必需字段: {field}")

        # 验证数值字段
        numeric_fields = ['calories_per_100g', 'protein', 'carbs', 'fat', 'fiber', 'sugar', 'sodium', 'glycemic_index']
        for field in numeric_fields:
            if field in data and data[field] is not None:
                if not isinstance(data[field], (int, float)) or data[field] < 0:
                    raise ValueError(f"{field}必须是大于等于0的数值")

        return data

# 示例数据
SAMPLE_RECIPES = [
    {
        "name": "宫保鸡丁",
        "category": "炒菜",
        "description": "经典川菜，麻辣鲜香",
        "ingredients": [
            {"name": "鸡胸肉", "amount": "300g"},
            {"name": "花生米", "amount": "50g"},
            {"name": "干辣椒", "amount": "10个"},
            {"name": "花椒", "amount": "1茶匙"},
            {"name": "大葱", "amount": "2根"},
            {"name": "生姜", "amount": "1块"},
            {"name": "大蒜", "amount": "3瓣"},
            {"name": "生抽", "amount": "2汤匙"},
            {"name": "老抽", "amount": "1汤匙"},
            {"name": "料酒", "amount": "1汤匙"},
            {"name": "白糖", "amount": "1茶匙"},
            {"name": "醋", "amount": "1茶匙"},
            {"name": "盐", "amount": "适量"},
            {"name": "食用油", "amount": "3汤匙"}
        ],
        "cooking_method": "炒",
        "cultural_tags": ["川菜", "经典", "家常"],
        "health_tags": ["高蛋白", "适量脂肪"],
        "difficulty_level": 3,
        "cooking_time": 20,
        "servings": 2,
        "instructions": "1. 鸡胸肉切丁，用料酒、生抽腌制15分钟\n2. 热锅下油，爆炒花生米至金黄盛起\n3. 下鸡丁炒至变色，加入干辣椒、花椒炒香\n4. 加入葱姜蒜爆炒，调入生抽、老抽、白糖、醋\n5. 最后加入花生米炒匀即可",
        "tips": "鸡丁要切得均匀，炒制时火候要掌握好"
    },
    {
        "name": "小笼包",
        "category": "早餐",
        "description": "上海传统点心，皮薄馅嫩",
        "ingredients": [
            {"name": "面粉", "amount": "300g"},
            {"name": "猪肉馅", "amount": "200g"},
            {"name": "虾仁", "amount": "100g"},
            {"name": "生姜", "amount": "1块"},
            {"name": "大葱", "amount": "2根"},
            {"name": "生抽", "amount": "2汤匙"},
            {"name": "料酒", "amount": "1汤匙"},
            {"name": "香油", "amount": "1茶匙"},
            {"name": "盐", "amount": "适量"},
            {"name": "糖", "amount": "1茶匙"}
        ],
        "cooking_method": "蒸",
        "cultural_tags": ["上海菜", "传统", "点心"],
        "health_tags": ["高蛋白", "低脂肪"],
        "difficulty_level": 4,
        "cooking_time": 60,
        "servings": 4,
        "instructions": "1. 面粉加水和成面团，醒发30分钟\n2. 猪肉馅加虾仁、调料拌匀\n3. 面团擀成薄皮，包入馅料\n4. 上锅蒸15分钟即可",
        "tips": "面皮要擀得薄而均匀，蒸制时间要掌握好"
    }
]

SAMPLE_INGREDIENT_NUTRITION = [
    {
        "ingredient_name": "鸡胸肉",
        "calories_per_100g": 165.0,
        "protein": 31.0,
        "carbs": 0.0,
        "fat": 3.6,
        "fiber": 0.0,
        "sugar": 0.0,
        "sodium": 74.0,
        "glycemic_index": 0.0,
        "health_benefits": ["高蛋白", "低脂肪", "富含维生素B"],
        "cultural_significance": "中式烹饪常用食材，营养丰富",
        "common_names": ["鸡肉", "鸡胸", "鸡脯肉"]
    },
    {
        "ingredient_name": "花生米",
        "calories_per_100g": 567.0,
        "protein": 25.8,
        "carbs": 16.1,
        "fat": 49.2,
        "fiber": 8.5,
        "sugar": 4.7,
        "sodium": 18.0,
        "glycemic_index": 14.0,
        "health_benefits": ["富含不饱和脂肪酸", "高蛋白", "富含维生素E"],
        "cultural_significance": "中式烹饪常用坚果，增香增味",
        "common_names": ["花生", "落花生", "长生果"]
    }
]

__all__ = ["'Base'", "'Recipe'", "'IngredientNutrition'", "'RecipeRecommendation'", "'UserRecipePreference'", "'RecipeDataValidator'", "'SAMPLE_RECIPES'", "'SAMPLE_INGREDIENT_NUTRITION'"]
