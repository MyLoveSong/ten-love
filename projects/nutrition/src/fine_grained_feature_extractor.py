#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
细粒度特征提取器
提取烹饪方式、食材组合、营养比例、季节性、地域性等细粒度特征
"""

import re
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import Counter


@dataclass
class CookingMethod:
    """烹饪方式编码"""
    STEAM = "steam"  # 蒸
    BOIL = "boil"  # 煮
    FRY = "fry"  # 炒
    DEEP_FRY = "deep_fry"  # 炸
    ROAST = "roast"  # 烤
    BRAISE = "braise"  # 炖
    STIR_FRY = "stir_fry"  # 爆炒
    GRILL = "grill"  # 烤
    RAW = "raw"  # 生食


class CookingMethodExtractor:
    """烹饪方式提取器"""

    def __init__(self):
        # 烹饪方式关键词映射
        self.method_keywords = {
            CookingMethod.STEAM: ['蒸', 'steam', 'steamed'],
            CookingMethod.BOIL: ['煮', 'boil', 'boiled', '白灼', '白切'],
            CookingMethod.FRY: ['炒', 'fry', 'fried', 'stir'],
            CookingMethod.DEEP_FRY: ['炸', 'deep_fry', 'deep-fried', 'fried'],
            CookingMethod.ROAST: ['烤', 'roast', 'roasted', 'baked'],
            CookingMethod.BRAISE: ['炖', 'braise', 'braised', '红烧', '焖'],
            CookingMethod.STIR_FRY: ['爆', 'stir_fry', 'quick-fry'],
            CookingMethod.GRILL: ['烤', 'grill', 'grilled', 'barbecue'],
            CookingMethod.RAW: ['生', 'raw', 'sashimi', '刺身']
        }

        # 烹饪方式健康分数（相对健康程度）
        self.method_health_scores = {
            CookingMethod.STEAM: 0.9,
            CookingMethod.BOIL: 0.85,
            CookingMethod.RAW: 0.8,
            CookingMethod.GRILL: 0.7,
            CookingMethod.ROAST: 0.65,
            CookingMethod.FRY: 0.5,
            CookingMethod.STIR_FRY: 0.45,
            CookingMethod.BRAISE: 0.4,
            CookingMethod.DEEP_FRY: 0.3
        }

    def extract(self, dish_name: str) -> Dict[str, float]:
        """
        从菜品名称提取烹饪方式

        Args:
            dish_name: 菜品名称

        Returns:
            烹饪方式特征向量（one-hot编码）
        """
        dish_name_lower = dish_name.lower()
        features = {method: 0.0 for method in self.method_keywords.keys()}

        # 匹配烹饪方式关键词
        for method, keywords in self.method_keywords.items():
            for keyword in keywords:
                if keyword.lower() in dish_name_lower:
                    features[method] = 1.0
                    break

        # 如果没有匹配到，使用默认值（基于常见烹饪方式）
        if sum(features.values()) == 0:
            # 默认假设是炒
            features[CookingMethod.FRY] = 1.0

        return features

    def get_health_score(self, dish_name: str) -> float:
        """根据烹饪方式获取健康分数"""
        features = self.extract(dish_name)
        for method, value in features.items():
            if value > 0:
                return self.method_health_scores[method]
        return 0.5  # 默认值


class IngredientCompositionExtractor:
    """食材组合特征提取器"""

    def __init__(self):
        # 主要食材类别
        self.ingredient_categories = {
            'meat': ['肉', '鸡', '鸭', '鱼', '虾', '蟹', '牛', '羊', '猪',
                    'meat', 'chicken', 'fish', 'shrimp', 'beef', 'pork'],
            'vegetable': ['菜', '蔬', '豆', '菇', '菌', '笋', '瓜', '茄',
                         'vegetable', 'cabbage', 'tomato', 'mushroom'],
            'grain': ['米', '面', '粉', '饭', '饼', 'bread', 'rice', 'noodle'],
            'dairy': ['奶', '蛋', 'cheese', 'milk', 'egg', 'dairy'],
            'nut': ['果', '仁', 'nut', 'seed'],
            'spice': ['椒', '蒜', '姜', '葱', 'pepper', 'garlic', 'ginger']
        }

        # 食材组合健康分数
        self.combination_scores = {
            ('vegetable', 'meat'): 0.7,
            ('vegetable', 'grain'): 0.85,
            ('vegetable',): 0.9,
            ('meat',): 0.6,
            ('grain',): 0.75,
            ('dairy', 'grain'): 0.8,
            ('vegetable', 'dairy'): 0.85
        }

    def extract(self, dish_name: str) -> Dict[str, float]:
        """
        提取食材组合特征

        Args:
            dish_name: 菜品名称

        Returns:
            食材类别特征向量
        """
        dish_name_lower = dish_name.lower()
        features = {category: 0.0 for category in self.ingredient_categories.keys()}

        # 匹配食材类别
        for category, keywords in self.ingredient_categories.items():
            for keyword in keywords:
                if keyword.lower() in dish_name_lower:
                    features[category] = 1.0
                    break

        return features

    def get_combination_score(self, dish_name: str) -> float:
        """根据食材组合获取健康分数"""
        features = self.extract(dish_name)
        present_categories = tuple(sorted([cat for cat, val in features.items() if val > 0]))

        # 查找匹配的组合
        for combination, score in self.combination_scores.items():
            if set(combination).issubset(set(present_categories)):
                return score

        # 默认分数
        if len(present_categories) > 0:
            return 0.65
        return 0.5


class NutritionRatioExtractor:
    """营养比例特征提取器"""

    def extract(self, nutrition: Dict) -> Dict[str, float]:
        """
        提取营养比例特征

        Args:
            nutrition: 营养数据字典

        Returns:
            营养比例特征
        """
        protein = float(nutrition.get('protein', 0))
        carbs = float(nutrition.get('carbs', 0))
        fat = float(nutrition.get('fat', 0))
        fiber = float(nutrition.get('fiber', 0))
        calories = float(nutrition.get('calories', 1))

        features = {}

        # 宏量营养素比例
        total_macro = protein + carbs + fat
        if total_macro > 0:
            features['protein_ratio'] = protein / total_macro
            features['carbs_ratio'] = carbs / total_macro
            features['fat_ratio'] = fat / total_macro
        else:
            features['protein_ratio'] = 0.0
            features['carbs_ratio'] = 0.0
            features['fat_ratio'] = 0.0

        # 蛋白质/脂肪比
        if fat > 0:
            features['protein_fat_ratio'] = protein / fat
        else:
            features['protein_fat_ratio'] = protein if protein > 0 else 0.0

        # 纤维/碳水比
        if carbs > 0:
            features['fiber_carbs_ratio'] = fiber / carbs
        else:
            features['fiber_carbs_ratio'] = fiber if fiber > 0 else 0.0

        # 能量密度（卡路里/100g）
        features['energy_density'] = calories / 100.0 if calories > 0 else 0.0

        # 蛋白质质量分数（蛋白质卡路里/总卡路里）
        if calories > 0:
            protein_calories = protein * 4
            features['protein_quality'] = protein_calories / calories
        else:
            features['protein_quality'] = 0.0

        # 健康脂肪比例（假设总脂肪的30%是健康脂肪）
        if fat > 0:
            features['healthy_fat_ratio'] = 0.3  # 简化假设
        else:
            features['healthy_fat_ratio'] = 0.0

        return features


class RegionalFeatureExtractor:
    """地域性特征提取器"""

    def __init__(self):
        # 中国菜系地域映射
        self.regional_keywords = {
            'sichuan': ['川', 'sichuan', 'spicy', '麻辣', '宫保', '麻婆'],
            'cantonese': ['粤', 'cantonese', '清蒸', '白切', '白灼'],
            'shandong': ['鲁', 'shandong', 'shandong'],
            'jiangsu': ['苏', 'jiangsu', 'sweet', 'sour', '红烧'],
            'zhejiang': ['浙', 'zhejiang', 'zhejiang'],
            'fujian': ['闽', 'fujian', 'fujian'],
            'hunan': ['湘', 'hunan', 'hunan'],
            'anhui': ['徽', 'anhui', 'anhui']
        }

        # 地域健康分数（基于地域饮食习惯）
        self.regional_health_scores = {
            'cantonese': 0.85,  # 粤菜偏清淡
            'zhejiang': 0.8,
            'jiangsu': 0.75,
            'fujian': 0.75,
            'sichuan': 0.6,  # 川菜偏重油重盐
            'hunan': 0.6,
            'shandong': 0.65,
            'anhui': 0.7
        }

    def extract(self, dish_name: str) -> Dict[str, float]:
        """
        提取地域特征

        Args:
            dish_name: 菜品名称

        Returns:
            地域特征向量
        """
        dish_name_lower = dish_name.lower()
        features = {region: 0.0 for region in self.regional_keywords.keys()}

        # 匹配地域关键词
        for region, keywords in self.regional_keywords.items():
            for keyword in keywords:
                if keyword.lower() in dish_name_lower:
                    features[region] = 1.0
                    break

        return features

    def get_regional_score(self, dish_name: str) -> float:
        """根据地域获取健康分数"""
        features = self.extract(dish_name)
        for region, value in features.items():
            if value > 0:
                return self.regional_health_scores.get(region, 0.65)
        return 0.65  # 默认值


class SeasonalFeatureExtractor:
    """季节性特征提取器"""

    def __init__(self):
        # 季节性食材关键词
        self.seasonal_keywords = {
            'spring': ['春', '笋', 'spring', 'bamboo'],
            'summer': ['夏', '瓜', 'summer', 'melon', 'cucumber'],
            'autumn': ['秋', 'autumn', 'fall'],
            'winter': ['冬', 'winter', 'radish']
        }

    def extract(self, dish_name: str) -> Dict[str, float]:
        """
        提取季节性特征

        Args:
            dish_name: 菜品名称

        Returns:
            季节性特征向量
        """
        dish_name_lower = dish_name.lower()
        features = {season: 0.0 for season in self.seasonal_keywords.keys()}

        # 匹配季节性关键词
        for season, keywords in self.seasonal_keywords.items():
            for keyword in keywords:
                if keyword.lower() in dish_name_lower:
                    features[season] = 1.0
                    break

        return features


class FineGrainedFeatureExtractor:
    """细粒度特征提取器主类"""

    def __init__(self):
        self.cooking_extractor = CookingMethodExtractor()
        self.ingredient_extractor = IngredientCompositionExtractor()
        self.nutrition_extractor = NutritionRatioExtractor()
        self.regional_extractor = RegionalFeatureExtractor()
        self.seasonal_extractor = SeasonalFeatureExtractor()

    def extract_all(self,
                   dish_name: str,
                   nutrition: Dict) -> Dict[str, float]:
        """
        提取所有细粒度特征

        Args:
            dish_name: 菜品名称
            nutrition: 营养数据字典

        Returns:
            所有特征的合并字典
        """
        features = {}

        # 烹饪方式特征
        cooking_features = self.cooking_extractor.extract(dish_name)
        features.update({f'cooking_{k}': v for k, v in cooking_features.items()})

        # 食材组合特征
        ingredient_features = self.ingredient_extractor.extract(dish_name)
        features.update({f'ingredient_{k}': v for k, v in ingredient_features.items()})

        # 营养比例特征
        nutrition_features = self.nutrition_extractor.extract(nutrition)
        features.update({f'ratio_{k}': v for k, v in nutrition_features.items()})

        # 地域特征
        regional_features = self.regional_extractor.extract(dish_name)
        features.update({f'regional_{k}': v for k, v in regional_features.items()})

        # 季节性特征
        seasonal_features = self.seasonal_extractor.extract(dish_name)
        features.update({f'seasonal_{k}': v for k, v in seasonal_features.items()})

        # 综合健康分数（基于多个特征）
        cooking_score = self.cooking_extractor.get_health_score(dish_name)
        ingredient_score = self.ingredient_extractor.get_combination_score(dish_name)
        regional_score = self.regional_extractor.get_regional_score(dish_name)

        # 加权平均
        features['composite_health_score'] = (
            0.4 * cooking_score +
            0.3 * ingredient_score +
            0.3 * regional_score
        )

        return features

    def get_feature_dimension(self) -> int:
        """获取特征维度"""
        # 烹饪方式: 9
        # 食材类别: 6
        # 营养比例: 8
        # 地域: 8
        # 季节性: 4
        # 综合分数: 1
        return 9 + 6 + 8 + 8 + 4 + 1
