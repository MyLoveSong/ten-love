#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强版数据处理器
结合多个API数据源，提取文化特征，优化数据质量
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedDataProcessor:
    """增强版数据处理器"""

    def __init__(self):
        """初始化数据处理器"""
        self.data_sources = {
            "cultural_nutrition": "data/real_data/enhanced_cultural_nutrition.json",
            "common_nutrition": "data/real_data/enhanced_common_nutrition.json",
            "cultural_recipes": "data/real_data/cultural_recipes_data.json",
            "common_ingredients": "data/real_data/common_ingredients_data.json",
            "cultural_nutrition_old": "data/real_data/cultural_nutrition_data.json",
            "common_nutrition_old": "data/real_data/common_nutrition_data.json"
        }
        self.processed_data = {}
        self.cultural_features = {}

    def load_all_data_sources(self) -> Dict[str, Any]:
        """加载所有数据源"""
        logger.info("📊 加载所有数据源...")

        loaded_data = {}

        for source_name, file_path in self.data_sources.items():
            try:
                if Path(file_path).exists():
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    loaded_data[source_name] = data
                    logger.info(f"   ✅ {source_name}: {len(data)} 条记录")
                else:
                    logger.warning(f"   ⚠️ {source_name}: 文件不存在 - {file_path}")
            except Exception as e:
                logger.error(f"   ❌ {source_name}: 加载失败 - {e}")

        self.processed_data = loaded_data
        return loaded_data

    def extract_cultural_features(self) -> Dict[str, Any]:
        """提取文化特征"""
        logger.info("🎭 提取文化特征...")

        cultural_features = {
            "cuisine_nutrition_profiles": {},
            "cuisine_ingredient_patterns": {},
            "cuisine_cooking_methods": {},
            "cuisine_flavor_profiles": {},
            "regional_nutrition_differences": {}
        }

        # 1. 分析菜系营养特征
        if "cultural_nutrition" in self.processed_data:
            cuisine_nutrition = self.processed_data["cultural_nutrition"]
            for cuisine, foods in cuisine_nutrition.items():
                nutrition_profile = self._analyze_cuisine_nutrition(foods)
                cultural_features["cuisine_nutrition_profiles"][cuisine] = nutrition_profile

        # 2. 分析菜系食材模式
        if "cultural_recipes" in self.processed_data:
            cuisine_recipes = self.processed_data["cultural_recipes"]
            for cuisine, recipes in cuisine_recipes.items():
                ingredient_pattern = self._analyze_cuisine_ingredients(recipes)
                cultural_features["cuisine_ingredient_patterns"][cuisine] = ingredient_pattern

        # 3. 分析烹饪方法
        cultural_features["cuisine_cooking_methods"] = self._analyze_cooking_methods()

        # 4. 分析口味特征
        cultural_features["cuisine_flavor_profiles"] = self._analyze_flavor_profiles()

        # 5. 分析地区营养差异
        cultural_features["regional_nutrition_differences"] = self._analyze_regional_differences()

        self.cultural_features = cultural_features
        logger.info(f"✅ 文化特征提取完成: {len(cultural_features)} 个特征类别")

        return cultural_features

    def _analyze_cuisine_nutrition(self, foods: Dict[str, Any]) -> Dict[str, Any]:
        """分析菜系营养特征"""
        nutrition_data = []

        for food_name, food_info in foods.items():
            nutrition = food_info.get("nutrition", {})
            if nutrition and not nutrition.get("is_mock_data", False):
                nutrition_data.append({
                    "calories": nutrition.get("nf_calories", 0),
                    "protein": nutrition.get("nf_protein", 0),
                    "carbs": nutrition.get("nf_total_carbohydrate", 0),
                    "fat": nutrition.get("nf_total_fat", 0),
                    "fiber": nutrition.get("nf_dietary_fiber", 0),
                    "sodium": nutrition.get("nf_sodium", 0)
                })

        if not nutrition_data:
            return {"error": "无有效营养数据"}

        # 计算统计特征
        df = pd.DataFrame(nutrition_data)
        stats = {
            "avg_calories": df["calories"].mean(),
            "avg_protein": df["protein"].mean(),
            "avg_carbs": df["carbs"].mean(),
            "avg_fat": df["fat"].mean(),
            "avg_fiber": df["fiber"].mean(),
            "avg_sodium": df["sodium"].mean(),
            "calorie_range": [df["calories"].min(), df["calories"].max()],
            "protein_ratio": df["protein"].mean() / df["calories"].mean() if df["calories"].mean() > 0 else 0,
            "fat_ratio": df["fat"].mean() / df["calories"].mean() if df["calories"].mean() > 0 else 0,
            "fiber_ratio": df["fiber"].mean() / df["calories"].mean() if df["calories"].mean() > 0 else 0
        }

        return stats

    def _analyze_cuisine_ingredients(self, recipes: Dict[str, Any]) -> Dict[str, Any]:
        """分析菜系食材模式"""
        ingredient_counts = {}
        cooking_methods = {}

        for recipe_name, recipe_info in recipes.items():
            # 分析食材（基于菜名推断）
            ingredients = self._extract_ingredients_from_name(recipe_name)
            for ingredient in ingredients:
                ingredient_counts[ingredient] = ingredient_counts.get(ingredient, 0) + 1

            # 分析烹饪方法
            cooking_method = self._extract_cooking_method(recipe_name)
            if cooking_method:
                cooking_methods[cooking_method] = cooking_methods.get(cooking_method, 0) + 1

        return {
            "top_ingredients": sorted(ingredient_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            "cooking_methods": cooking_methods,
            "ingredient_diversity": len(ingredient_counts),
            "method_diversity": len(cooking_methods)
        }

    def _extract_ingredients_from_name(self, recipe_name: str) -> List[str]:
        """从菜名中提取食材"""
        # 简单的食材提取逻辑
        ingredients = []

        # 肉类
        if any(meat in recipe_name for meat in ["鸡", "鸭", "鹅", "猪", "牛", "羊", "鱼", "虾", "蟹"]):
            if "鸡" in recipe_name:
                ingredients.append("鸡肉")
            if "鸭" in recipe_name:
                ingredients.append("鸭肉")
            if "鹅" in recipe_name:
                ingredients.append("鹅肉")
            if "猪" in recipe_name:
                ingredients.append("猪肉")
            if "牛" in recipe_name:
                ingredients.append("牛肉")
            if "羊" in recipe_name:
                ingredients.append("羊肉")
            if "鱼" in recipe_name:
                ingredients.append("鱼肉")
            if "虾" in recipe_name:
                ingredients.append("虾")
            if "蟹" in recipe_name:
                ingredients.append("蟹")

        # 蔬菜
        if any(veg in recipe_name for veg in ["豆", "菜", "笋", "菇", "瓜", "椒"]):
            if "豆" in recipe_name:
                ingredients.append("豆腐")
            if "菜" in recipe_name:
                ingredients.append("蔬菜")
            if "笋" in recipe_name:
                ingredients.append("竹笋")
            if "菇" in recipe_name:
                ingredients.append("蘑菇")
            if "瓜" in recipe_name:
                ingredients.append("瓜类")
            if "椒" in recipe_name:
                ingredients.append("辣椒")

        return ingredients

    def _extract_cooking_method(self, recipe_name: str) -> Optional[str]:
        """从菜名中提取烹饪方法"""
        cooking_methods = {
            "炒": ["炒", "爆", "煸"],
            "煮": ["煮", "炖", "煲", "汤"],
            "蒸": ["蒸", "蒸蛋"],
            "烤": ["烤", "烧", "叉烧"],
            "炸": ["炸", "煎"],
            "腌": ["腌", "糟", "醉"],
            "凉拌": ["凉", "拌", "白切"]
        }

        for method, keywords in cooking_methods.items():
            if any(keyword in recipe_name for keyword in keywords):
                return method

        return None

    def _analyze_cooking_methods(self) -> Dict[str, Any]:
        """分析烹饪方法特征"""
        return {
            "川菜": {"主要方法": ["炒", "煮", "炸"], "特点": "重油重辣"},
            "粤菜": {"主要方法": ["蒸", "煮", "白切"], "特点": "清淡鲜美"},
            "鲁菜": {"主要方法": ["炸", "烧", "炖"], "特点": "浓油赤酱"},
            "苏菜": {"主要方法": ["蒸", "煮", "炖"], "特点": "清淡甜鲜"},
            "浙菜": {"主要方法": ["蒸", "煮", "炸"], "特点": "清淡鲜嫩"},
            "闽菜": {"主要方法": ["蒸", "煮", "炸"], "特点": "清淡鲜香"},
            "湘菜": {"主要方法": ["炒", "煮", "炸"], "特点": "重油重辣"},
            "徽菜": {"主要方法": ["烧", "炖", "腌"], "特点": "重油重色"}
        }

    def _analyze_flavor_profiles(self) -> Dict[str, Any]:
        """分析口味特征"""
        return {
            "川菜": {"口味": "麻辣", "调料": ["花椒", "辣椒", "豆瓣酱"], "特点": "重口味"},
            "粤菜": {"口味": "清淡", "调料": ["生抽", "蚝油", "糖"], "特点": "原味"},
            "鲁菜": {"口味": "咸鲜", "调料": ["酱油", "盐", "糖"], "特点": "浓重"},
            "苏菜": {"口味": "甜鲜", "调料": ["糖", "醋", "生抽"], "特点": "甜味"},
            "浙菜": {"口味": "清淡", "调料": ["生抽", "糖", "醋"], "特点": "清淡"},
            "闽菜": {"口味": "清淡", "调料": ["生抽", "糖", "醋"], "特点": "鲜香"},
            "湘菜": {"口味": "辣", "调料": ["辣椒", "花椒", "豆瓣酱"], "特点": "重辣"},
            "徽菜": {"口味": "咸鲜", "调料": ["酱油", "盐", "糖"], "特点": "重色"}
        }

    def _analyze_regional_differences(self) -> Dict[str, Any]:
        """分析地区营养差异"""
        return {
            "北方": {"特点": "高热量", "主食": "面食", "口味": "偏咸"},
            "南方": {"特点": "清淡", "主食": "米饭", "口味": "偏甜"},
            "西南": {"特点": "重辣", "调料": "花椒辣椒", "口味": "麻辣"},
            "华东": {"特点": "甜鲜", "调料": "糖醋", "口味": "偏甜"},
            "华南": {"特点": "清淡", "调料": "生抽蚝油", "口味": "原味"}
        }

    def create_enhanced_training_data(self) -> Dict[str, Any]:
        """创建增强版训练数据"""
        logger.info("🚀 创建增强版训练数据...")

        training_data = {
            "cultural_preferences": [],
            "nutrition_features": [],
            "cultural_features": [],
            "metadata": {
                "total_samples": 0,
                "cuisine_distribution": {},
                "feature_dimensions": {},
                "data_quality_score": 0
            }
        }

        # 1. 处理文化偏好数据
        if "cultural_nutrition" in self.processed_data:
            cultural_data = self._process_cultural_preferences()
            training_data["cultural_preferences"] = cultural_data

        # 2. 处理营养特征
        nutrition_features = self._process_nutrition_features()
        training_data["nutrition_features"] = nutrition_features

        # 3. 处理文化特征
        cultural_features = self._process_cultural_features()
        training_data["cultural_features"] = cultural_features

        # 4. 计算元数据
        training_data["metadata"] = self._calculate_metadata(training_data)

        logger.info(f"✅ 增强版训练数据创建完成: {training_data['metadata']['total_samples']} 个样本")

        return training_data

    def _process_cultural_preferences(self) -> List[Dict[str, Any]]:
        """处理文化偏好数据"""
        preferences = []

        if "cultural_nutrition" in self.processed_data:
            cuisine_data = self.processed_data["cultural_nutrition"]

            for cuisine, foods in cuisine_data.items():
                for food_name, food_info in foods.items():
                    nutrition = food_info.get("nutrition", {})

                    if nutrition and not nutrition.get("is_mock_data", False):
                        preference = {
                            "region": cuisine,
                            "cuisine_type": cuisine,
                            "food_name": food_name,
                            "calories": nutrition.get("nf_calories", 0),
                            "protein": nutrition.get("nf_protein", 0),
                            "carbs": nutrition.get("nf_total_carbohydrate", 0),
                            "fat": nutrition.get("nf_total_fat", 0),
                            "fiber": nutrition.get("nf_dietary_fiber", 0),
                            "sodium": nutrition.get("nf_sodium", 0),
                            "acceptance_score": self._calculate_acceptance_score(nutrition, cuisine),
                            "cultural_relevance": self._calculate_cultural_relevance(food_name, cuisine)
                        }
                        preferences.append(preference)

        return preferences

    def _calculate_acceptance_score(self, nutrition: Dict[str, Any], cuisine: str) -> float:
        """计算接受度分数"""
        # 基于营养数据和菜系特征计算接受度
        calories = nutrition.get("nf_calories", 0)
        protein = nutrition.get("nf_protein", 0)
        fat = nutrition.get("nf_total_fat", 0)

        # 基础分数
        base_score = 0.5

        # 营养平衡加分
        if calories > 0:
            protein_ratio = protein / calories * 4  # 蛋白质比例
            fat_ratio = fat / calories * 9  # 脂肪比例

            if 0.1 <= protein_ratio <= 0.3:  # 合理蛋白质比例
                base_score += 0.2
            if 0.2 <= fat_ratio <= 0.4:  # 合理脂肪比例
                base_score += 0.1

        # 菜系特征加分
        cuisine_bonus = {
            "川菜": 0.1, "粤菜": 0.15, "鲁菜": 0.1, "苏菜": 0.15,
            "浙菜": 0.15, "闽菜": 0.15, "湘菜": 0.1, "徽菜": 0.1
        }
        base_score += cuisine_bonus.get(cuisine, 0)

        return min(1.0, max(0.0, base_score))

    def _calculate_cultural_relevance(self, food_name: str, cuisine: str) -> float:
        """计算文化相关性"""
        # 基于菜名和菜系的匹配度
        relevance = 0.5

        # 菜系关键词匹配
        cuisine_keywords = {
            "川菜": ["麻", "辣", "椒", "豆", "鱼"],
            "粤菜": ["白", "切", "烧", "蒸", "虾"],
            "鲁菜": ["糖", "醋", "扒", "烧", "海"],
            "苏菜": ["桂", "鱼", "蟹", "粉", "狮"],
            "浙菜": ["醋", "鱼", "东", "坡", "龙"],
            "闽菜": ["佛", "跳", "荔", "枝", "醉"],
            "湘菜": ["剁", "椒", "鱼", "头", "毛"],
            "徽菜": ["臭", "鳜", "毛", "豆", "腐"]
        }

        keywords = cuisine_keywords.get(cuisine, [])
        matches = sum(1 for keyword in keywords if keyword in food_name)
        relevance += matches * 0.1

        return min(1.0, relevance)

    def _process_nutrition_features(self) -> List[Dict[str, Any]]:
        """处理营养特征"""
        features = []

        # 从所有营养数据中提取特征
        for source_name, data in self.processed_data.items():
            if "nutrition" in source_name:
                if isinstance(data, dict):
                    for cuisine, foods in data.items():
                        if isinstance(foods, dict):
                            for food_name, food_info in foods.items():
                                if isinstance(food_info, dict):
                                    nutrition = food_info.get("nutrition", {})
                                    if nutrition and not nutrition.get("is_mock_data", False):
                                        feature = {
                                            "food_name": food_name,
                                            "cuisine": cuisine,
                                            "calories": nutrition.get("nf_calories", 0),
                                            "protein": nutrition.get("nf_protein", 0),
                                            "carbs": nutrition.get("nf_total_carbohydrate", 0),
                                            "fat": nutrition.get("nf_total_fat", 0),
                                            "fiber": nutrition.get("nf_dietary_fiber", 0),
                                            "sodium": nutrition.get("nf_sodium", 0)
                                        }
                                        features.append(feature)

        return features

    def _process_cultural_features(self) -> List[Dict[str, Any]]:
        """处理文化特征"""
        features = []

        # 使用提取的文化特征
        for cuisine, nutrition_profile in self.cultural_features.get("cuisine_nutrition_profiles", {}).items():
            if "error" not in nutrition_profile:
                feature = {
                    "cuisine": cuisine,
                    "avg_calories": nutrition_profile.get("avg_calories", 0),
                    "avg_protein": nutrition_profile.get("avg_protein", 0),
                    "avg_carbs": nutrition_profile.get("avg_carbs", 0),
                    "avg_fat": nutrition_profile.get("avg_fat", 0),
                    "protein_ratio": nutrition_profile.get("protein_ratio", 0),
                    "fat_ratio": nutrition_profile.get("fat_ratio", 0),
                    "fiber_ratio": nutrition_profile.get("fiber_ratio", 0)
                }
                features.append(feature)

        return features

    def _calculate_metadata(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """计算元数据"""
        total_samples = len(training_data["cultural_preferences"])

        # 计算菜系分布
        cuisine_distribution = {}
        for preference in training_data["cultural_preferences"]:
            cuisine = preference["cuisine_type"]
            cuisine_distribution[cuisine] = cuisine_distribution.get(cuisine, 0) + 1

        # 计算特征维度
        feature_dimensions = {
            "nutrition_features": len(training_data["nutrition_features"]),
            "cultural_features": len(training_data["cultural_features"]),
            "cultural_preferences": len(training_data["cultural_preferences"])
        }

        # 计算数据质量分数
        quality_score = self._calculate_data_quality_score(training_data)

        return {
            "total_samples": total_samples,
            "cuisine_distribution": cuisine_distribution,
            "feature_dimensions": feature_dimensions,
            "data_quality_score": quality_score,
            "timestamp": datetime.now().isoformat()
        }

    def _calculate_data_quality_score(self, training_data: Dict[str, Any]) -> float:
        """计算数据质量分数"""
        score = 0.0

        # 样本数量分数 (0-0.3)
        total_samples = training_data["metadata"]["total_samples"]
        if total_samples > 100:
            score += 0.3
        elif total_samples > 50:
            score += 0.2
        elif total_samples > 20:
            score += 0.1

        # 菜系多样性分数 (0-0.3)
        cuisine_count = len(training_data["metadata"]["cuisine_distribution"])
        if cuisine_count >= 8:
            score += 0.3
        elif cuisine_count >= 6:
            score += 0.2
        elif cuisine_count >= 4:
            score += 0.1

        # 特征完整性分数 (0-0.4)
        nutrition_features = training_data["nutrition_features"]
        cultural_features = training_data["cultural_features"]

        if len(nutrition_features) > 50 and len(cultural_features) > 5:
            score += 0.4
        elif len(nutrition_features) > 20 and len(cultural_features) > 3:
            score += 0.3
        elif len(nutrition_features) > 10:
            score += 0.2

        return min(1.0, score)

    def save_enhanced_data(self, training_data: Dict[str, Any]) -> None:
        """保存增强版数据"""
        output_dir = Path("data/enhanced_training")
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存训练数据
        with open(output_dir / "enhanced_training_data.json", "w", encoding="utf-8") as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False, default=str)

        # 保存文化特征
        with open(output_dir / "cultural_features.json", "w", encoding="utf-8") as f:
            json.dump(self.cultural_features, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"💾 增强版数据已保存: {output_dir}")

    def run_enhanced_processing(self) -> Dict[str, Any]:
        """运行增强版数据处理"""
        logger.info("🚀 开始增强版数据处理...")

        # 1. 加载所有数据源
        self.load_all_data_sources()

        # 2. 提取文化特征
        self.extract_cultural_features()

        # 3. 创建增强版训练数据
        training_data = self.create_enhanced_training_data()

        # 4. 保存数据
        self.save_enhanced_data(training_data)

        # 5. 生成报告
        self._generate_processing_report(training_data)

        return training_data

    def _generate_processing_report(self, training_data: Dict[str, Any]) -> None:
        """生成处理报告"""
        logger.info("📊 生成数据处理报告...")

        report = {
            "processing_summary": {
                "total_samples": training_data["metadata"]["total_samples"],
                "cuisine_distribution": training_data["metadata"]["cuisine_distribution"],
                "data_quality_score": training_data["metadata"]["data_quality_score"],
                "feature_dimensions": training_data["metadata"]["feature_dimensions"]
            },
            "cultural_features_summary": {
                "cuisine_nutrition_profiles": len(self.cultural_features.get("cuisine_nutrition_profiles", {})),
                "cuisine_ingredient_patterns": len(self.cultural_features.get("cuisine_ingredient_patterns", {})),
                "cuisine_cooking_methods": len(self.cultural_features.get("cuisine_cooking_methods", {})),
                "cuisine_flavor_profiles": len(self.cultural_features.get("cuisine_flavor_profiles", {}))
            },
            "data_sources_used": list(self.processed_data.keys()),
            "processing_timestamp": datetime.now().isoformat()
        }

        # 保存报告
        output_dir = Path("outputs/enhanced_processing")
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "processing_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"📋 处理报告已保存: {output_dir / 'processing_report.json'}")


def main():
    """主函数"""
    print("\n" + "="*80)
    print("🚀 增强版数据处理器")
    print("结合多个API数据源，提取文化特征，优化数据质量")
    print("="*80 + "\n")

    processor = EnhancedDataProcessor()
    training_data = processor.run_enhanced_processing()

    print("\n" + "="*80)
    print("✅ 增强版数据处理完成！")
    print(f"📊 总样本数: {training_data['metadata']['total_samples']}")
    print(f"🎯 数据质量分数: {training_data['metadata']['data_quality_score']:.2f}")
    print("📁 数据已保存到: data/enhanced_training/")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
