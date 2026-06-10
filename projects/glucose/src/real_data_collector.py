#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
真实数据收集器
专门收集真实的API数据样本，确保数据质量
"""

import os
import sys
import json
import logging
import requests
import time
import asyncio
import aiohttp
from pathlib import Path
from typing import Dict, Any, List, Optional
import random

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RealDataCollector:
    """真实数据收集器"""

    def __init__(self):
        """初始化收集器"""
        self.api_keys = self._load_api_keys()
        self.usda_base_url = "https://api.nal.usda.gov/fdc/v1"
        self.openfoodfacts_base_url = "https://world.openfoodfacts.org/api/v0"
        self.rate_limits = {
            "usda": {"requests_per_hour": 1000, "current_count": 0, "reset_time": 0},
            "openfoodfacts": {"requests_per_minute": 100, "current_count": 0, "reset_time": 0}
        }
        self.real_data = []
        self.failed_queries = []

        # 真实食物查询列表（英文，更容易匹配API）
        self.real_food_queries = {
            "fruits": [
                "apple", "banana", "orange", "grape", "strawberry", "blueberry", "cherry", "peach", "pear", "plum",
                "watermelon", "cantaloupe", "pineapple", "mango", "kiwi", "lemon", "lime", "pomegranate", "lychee", "longan"
            ],
            "vegetables": [
                "tomato", "cucumber", "carrot", "potato", "onion", "cabbage", "spinach", "celery", "lettuce", "eggplant",
                "pepper", "mushroom", "green beans", "broccoli", "cauliflower", "radish", "lotus root", "yam", "pumpkin", "squash"
            ],
            "meats": [
                "chicken breast", "beef steak", "pork chop", "lamb", "salmon", "shrimp", "crab", "lobster", "duck", "turkey",
                "rabbit", "venison", "quail", "pigeon", "tuna", "cod", "bass", "carp", "tilapia", "mackerel"
            ],
            "grains": [
                "white rice", "brown rice", "pasta", "bread", "oats", "corn", "wheat", "barley", "quinoa", "buckwheat",
                "millet", "rye", "spelt", "bulgur", "couscous", "noodles", "vermicelli", "rice noodles", "udon", "soba"
            ],
            "dairy": [
                "milk", "yogurt", "cheese", "butter", "cream", "ice cream", "cottage cheese", "mozzarella", "cheddar", "parmesan"
            ],
            "beverages": [
                "water", "tea", "coffee", "orange juice", "apple juice", "coca cola", "beer", "wine", "whiskey", "vodka"
            ],
            "nuts_seeds": [
                "peanuts", "walnuts", "almonds", "cashews", "pistachios", "hazelnuts", "pine nuts", "sunflower seeds", "sesame seeds", "flax seeds"
            ]
        }

        logger.info("🔧 初始化真实数据收集器")
        logger.info(f"📊 目标: 收集真实API数据样本")
        logger.info(f"📊 查询食物: {sum(len(foods) for foods in self.real_food_queries.values())} 个")

    def _load_api_keys(self) -> Dict[str, str]:
        """加载API密钥"""
        try:
            from utils.config_loader import ConfigLoader
            config_loader = ConfigLoader("configs/api_keys.yaml")
            return config_loader.get_config("api_keys")
        except Exception as e:
            logger.warning(f"无法加载API密钥配置: {e}")
            return {
                "usda": os.getenv("USDA_API_KEY", ""),
                "openfoodfacts": "no_key_needed"
            }

    def _check_rate_limit(self, api_name: str) -> bool:
        """检查API速率限制"""
        current_time = time.time()
        limit_info = self.rate_limits[api_name]

        # 重置计数器
        if api_name == "usda" and current_time > limit_info["reset_time"]:
            limit_info["current_count"] = 0
            limit_info["reset_time"] = current_time + 3600
        elif api_name == "openfoodfacts" and current_time > limit_info["reset_time"]:
            limit_info["current_count"] = 0
            limit_info["reset_time"] = current_time + 60

        # 检查是否超出限制
        if api_name == "usda" and limit_info["current_count"] >= limit_info["requests_per_hour"]:
            return False
        elif api_name == "openfoodfacts" and limit_info["current_count"] >= limit_info["requests_per_minute"]:
            return False

        return True

    def _increment_rate_limit(self, api_name: str):
        """增加API调用计数"""
        self.rate_limits[api_name]["current_count"] += 1

    async def query_usda_api_detailed(self, query: str) -> Optional[Dict[str, Any]]:
        """详细查询USDA API"""
        if not self._check_rate_limit("usda"):
            logger.warning(f"USDA API速率限制，跳过查询: {query}")
            return None

        try:
            url = f"{self.usda_base_url}/foods/search"
            params = {
                "query": query,
                "api_key": self.api_keys.get("usda", ""),
                "pageSize": 1,
                "dataType": ["Foundation", "SR Legacy"]
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=15) as response:
                    if response.status == 200:
                        data = await response.json()
                        self._increment_rate_limit("usda")

                        if "foods" in data and data["foods"]:
                            food = data["foods"][0]
                            nutrients = food.get("foodNutrients", [])

                            if len(nutrients) >= 5:
                                logger.info(f"✅ USDA API查询成功: {query} ({len(nutrients)} 个营养数据)")
                                return data

                            logger.warning(f"⚠️ USDA API数据不足: {query} ({len(nutrients)} 个营养数据)")
                            return None

                        logger.warning(f"⚠️ USDA API无结果: {query}")
                        return None

                    else:
                        logger.warning(f"❌ USDA API查询失败: {query} - {response.status}")
                        return None

        except Exception as e:
            logger.error(f"❌ USDA API查询异常: {query} - {e}")
            return None

    async def query_openfoodfacts_api_detailed(self, query: str) -> Optional[Dict[str, Any]]:
        """详细查询OpenFoodFacts API"""
        if not self._check_rate_limit("openfoodfacts"):
            logger.warning(f"OpenFoodFacts API速率限制，跳过查询: {query}")
            return None

        try:
            url = f"{self.openfoodfacts_base_url}/cgi/search.pl"
            params = {
                "search_terms": query,
                "search_simple": 1,
                "action": "process",
                "json": 1,
                "page_size": 1
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=15) as response:
                    if response.status == 200:
                        data = await response.json()
                        self._increment_rate_limit("openfoodfacts")

                        # 详细检查数据质量
                        if "products" in data and data["products"]:
                            product = data["products"][0]
                            nutriments = product.get("nutriments", {})

                            # 检查关键营养数据
                            key_nutrients = ["energy-kcal_100g", "proteins_100g", "carbohydrates_100g", "fat_100g"]
                            available_nutrients = sum(1 for key in key_nutrients if key in nutriments and nutriments[key] is not None)

                            if available_nutrients >= 3:
                                logger.info(f"✅ OpenFoodFacts API查询成功: {query} ({available_nutrients} 个关键营养数据)")
                                return data
                            else:
                                logger.warning(f"⚠️ OpenFoodFacts API数据不足: {query} ({available_nutrients} 个关键营养数据)")
                                return None
                        else:
                            logger.warning(f"⚠️ OpenFoodFacts API无结果: {query}")
                            return None
                    else:
                        logger.warning(f"❌ OpenFoodFacts API查询失败: {query} - {response.status}")
            return None

        except Exception as e:
            logger.error(f"❌ OpenFoodFacts API查询异常: {query} - {e}")
            return None

    def extract_real_nutrition_data(self, usda_data: Dict[str, Any], off_data: Dict[str, Any], food_name: str, category: str) -> Optional[Dict[str, Any]]:
        """提取真实营养数据"""
        nutrition_data = {
            "food_name": food_name,
            "category": category,
            "source": "real_api",
            "calories": 0,
            "protein": 0,
            "carbs": 0,
            "fat": 0,
            "fiber": 0,
            "sugar": 0,
            "sodium": 0,
            "serving_size": 100,
            "data_quality": "unknown",
            "api_sources": []
        }

        data_found = False

        # 从USDA数据提取
        if usda_data and "foods" in usda_data and usda_data["foods"]:
            usda_food = usda_data["foods"][0]
            food_nutrients = usda_food.get("foodNutrients", [])

            for nutrient in food_nutrients:
                nutrient_name = nutrient.get("nutrient", {}).get("name", "").lower()
                amount = nutrient.get("amount", 0)

                if "energy" in nutrient_name or "calories" in nutrient_name:
                    nutrition_data["calories"] = amount
                    data_found = True
                elif "protein" in nutrient_name:
                    nutrition_data["protein"] = amount
                    data_found = True
                elif "carbohydrate" in nutrient_name or "carb" in nutrient_name:
                    nutrition_data["carbs"] = amount
                    data_found = True
                elif "fat" in nutrient_name or "lipid" in nutrient_name:
                    nutrition_data["fat"] = amount
                    data_found = True
                elif "fiber" in nutrient_name:
                    nutrition_data["fiber"] = amount
                    data_found = True
                elif "sugar" in nutrient_name:
                    nutrition_data["sugar"] = amount
                    data_found = True
                elif "sodium" in nutrient_name:
                    nutrition_data["sodium"] = amount
                    data_found = True

            if data_found:
                nutrition_data["data_quality"] = "usda_high"
                nutrition_data["api_sources"].append("usda")

        # 从OpenFoodFacts数据提取
        if off_data and "products" in off_data and off_data["products"]:
            off_product = off_data["products"][0]
            nutriments = off_product.get("nutriments", {})

            off_data_found = False
            if "energy-kcal_100g" in nutriments and nutriments["energy-kcal_100g"] is not None:
                nutrition_data["calories"] = nutriments["energy-kcal_100g"]
                off_data_found = True
            if "proteins_100g" in nutriments and nutriments["proteins_100g"] is not None:
                nutrition_data["protein"] = nutriments["proteins_100g"]
                off_data_found = True
            if "carbohydrates_100g" in nutriments and nutriments["carbohydrates_100g"] is not None:
                nutrition_data["carbs"] = nutriments["carbohydrates_100g"]
                off_data_found = True
            if "fat_100g" in nutriments and nutriments["fat_100g"] is not None:
                nutrition_data["fat"] = nutriments["fat_100g"]
                off_data_found = True
            if "fiber_100g" in nutriments and nutriments["fiber_100g"] is not None:
                nutrition_data["fiber"] = nutriments["fiber_100g"]
                off_data_found = True
            if "sugars_100g" in nutriments and nutriments["sugars_100g"] is not None:
                nutrition_data["sugar"] = nutriments["sugars_100g"]
                off_data_found = True
            if "sodium_100g" in nutriments and nutriments["sodium_100g"] is not None:
                nutrition_data["sodium"] = nutriments["sodium_100g"]
                off_data_found = True

            if off_data_found:
                if nutrition_data["data_quality"] == "usda_high":
                    nutrition_data["data_quality"] = "combined_high"
                else:
                    nutrition_data["data_quality"] = "off_medium"
                nutrition_data["api_sources"].append("openfoodfacts")

        # 检查数据质量
        total_nutrients = sum([
            nutrition_data["calories"], nutrition_data["protein"],
            nutrition_data["carbs"], nutrition_data["fat"]
        ])

        if total_nutrients > 0 and data_found:
            return nutrition_data
        else:
            logger.warning(f"⚠️ 数据质量不足，跳过: {food_name}")
            return None

    async def collect_real_food_data(self, food_name: str, category: str) -> Optional[Dict[str, Any]]:
        """收集真实食物数据"""
        logger.info(f"🍽️ 收集真实食物数据: {food_name}")

        # 并行查询两个API
        usda_task = self.query_usda_api_detailed(food_name)
        off_task = self.query_openfoodfacts_api_detailed(food_name)

        usda_data, off_data = await asyncio.gather(usda_task, off_task)

        # 提取真实营养数据
        nutrition_data = self.extract_real_nutrition_data(usda_data, off_data, food_name, category)

        if nutrition_data:
            logger.info(f"   ✅ 真实数据收集成功: {food_name}")
            logger.info(f"   📊 卡路里: {nutrition_data['calories']:.1f}, 蛋白质: {nutrition_data['protein']:.1f}g")
            logger.info(f"   🔍 数据源: {', '.join(nutrition_data['api_sources'])}")
            logger.info(f"   📈 质量: {nutrition_data['data_quality']}")
        else:
            logger.warning(f"   ❌ 真实数据收集失败: {food_name}")

        return nutrition_data

    async def collect_category_real_data(self, category: str, foods: List[str]) -> List[Dict[str, Any]]:
        """收集特定类别的真实数据"""
        logger.info(f"🍽️ 收集 {category} 真实数据 ({len(foods)} 个食物)...")

        category_data = []
        success_count = 0

        for i, food in enumerate(foods):
            logger.info(f"   📊 进度: {i+1}/{len(foods)} - {food}")

            nutrition_data = await self.collect_real_food_data(food, category)

            if nutrition_data:
                category_data.append(nutrition_data)
                success_count += 1
                logger.info(f"   ✅ 成功: {success_count}/{i+1}")
            else:
                logger.warning(f"   ❌ 失败: {food}")

            # 避免API限制
            await asyncio.sleep(2)

        logger.info(f"✅ {category} 真实数据收集完成: {success_count}/{len(foods)} 成功")
        return category_data

    async def run_real_data_collection(self) -> None:
        """运行真实数据收集"""
        logger.info("🚀 开始真实数据收集...")
        logger.info(f"📊 目标: 收集真实API数据样本")

        all_real_data = []
        total_success = 0
        total_attempts = 0

        for category, foods in self.real_food_queries.items():
            category_data = await self.collect_category_real_data(category, foods)
            all_real_data.extend(category_data)
            total_success += len(category_data)
            total_attempts += len(foods)

            logger.info(f"📊 累计真实数据: {total_success} 个样本")

            # 保存中间结果
            if total_success % 10 == 0:
                self._save_intermediate_results(all_real_data, total_success)

        # 保存最终结果
        self._save_final_results(all_real_data)

        # 生成统计报告
        self._generate_real_data_report(all_real_data, total_success, total_attempts)

        logger.info(f"✅ 真实数据收集完成! 总计: {total_success}/{total_attempts} 个真实样本")

    def _save_intermediate_results(self, data: List[Dict[str, Any]], count: int) -> None:
        """保存中间结果"""
        output_dir = Path("data/real_data_collection")
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"real_data_intermediate_{count}.json"
        file_path = output_dir / filename

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"💾 中间真实数据已保存: {file_path}")

    def _save_final_results(self, data: List[Dict[str, Any]]) -> None:
        """保存最终结果"""
        output_dir = Path("data/real_data_collection")
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存完整数据
        with open(output_dir / "real_data_final.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        # 保存训练格式数据
        training_data = self._convert_to_training_format(data)
        with open(output_dir / "real_training_data.json", "w", encoding="utf-8") as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"💾 最终真实数据已保存: {output_dir}")

    def _convert_to_training_format(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """转换为训练格式"""
        training_samples = []

        for item in data:
            # 构建特征向量
            features = [
                item.get("calories", 0),
                item.get("protein", 0),
                item.get("carbs", 0),
                item.get("fat", 0),
                item.get("fiber", 0),
                item.get("sugar", 0),
                item.get("sodium", 0),
                # 添加类别编码
                1 if item.get("category") == "fruits" else 0,
                1 if item.get("category") == "vegetables" else 0,
                1 if item.get("category") == "meats" else 0,
                1 if item.get("category") == "grains" else 0,
                1 if item.get("category") == "dairy" else 0,
                1 if item.get("category") == "beverages" else 0,
                1 if item.get("category") == "nuts_seeds" else 0,
                1 if item.get("category") == "chinese_cuisine" else 0
            ]

            # 计算营养评分
            nutrition_score = self._calculate_nutrition_score(item)

            training_samples.append({
                "food_name": item.get("food_name", ""),
                "category": item.get("category", ""),
                "features": features,
                "nutrition_score": nutrition_score,
                "data_quality": item.get("data_quality", "unknown"),
                "api_sources": item.get("api_sources", []),
                "source": "real_api"
            })

        return {
            "training_samples": training_samples,
            "total_samples": len(training_samples),
            "feature_dimensions": len(features),
            "categories": list(set(item.get("category", "") for item in data)),
            "data_quality": "real_api"
        }

    def _calculate_nutrition_score(self, item: Dict[str, Any]) -> float:
        """计算营养评分"""
        calories = item.get("calories", 0)
        protein = item.get("protein", 0)
        carbs = item.get("carbs", 0)
        fat = item.get("fat", 0)
        fiber = item.get("fiber", 0)
        sugar = item.get("sugar", 0)
        sodium = item.get("sodium", 0)

        # 简化的营养评分算法
        score = 0.0

        # 蛋白质加分
        score += min(protein / 20, 1.0) * 0.3

        # 纤维加分
        score += min(fiber / 10, 1.0) * 0.2

        # 糖分减分
        score -= min(sugar / 50, 1.0) * 0.2

        # 钠分减分
        score -= min(sodium / 1000, 1.0) * 0.1

        # 卡路里平衡
        if 100 <= calories <= 400:
            score += 0.2
        elif calories < 100 or calories > 400:
            score -= 0.1

        return max(0.0, min(1.0, score))

    def _generate_real_data_report(self, data: List[Dict[str, Any]], success_count: int, total_attempts: int) -> None:
        """生成真实数据报告"""
        logger.info("📊 生成真实数据收集报告...")

        # 统计数据源
        api_sources = {}
        data_quality = {}
        categories = {}

        for item in data:
            sources = item.get("api_sources", [])
            quality = item.get("data_quality", "unknown")
            category = item.get("category", "unknown")

            for source in sources:
                api_sources[source] = api_sources.get(source, 0) + 1

            data_quality[quality] = data_quality.get(quality, 0) + 1
            categories[category] = categories.get(category, 0) + 1

        # 生成报告
        report = {
            "collection_summary": {
                "total_samples": len(data),
                "success_rate": f"{success_count}/{total_attempts} ({success_count/total_attempts*100:.1f}%)",
                "api_sources": api_sources,
                "data_quality": data_quality,
                "categories": categories
            },
            "api_usage": self.rate_limits,
            "data_authenticity": {
                "real_api_data": len(data),
                "percentage": 100.0,
                "description": "所有数据都来自真实API"
            },
            "recommendations": {
                "sufficient_for_training": len(data) >= 100,
                "next_steps": [
                    "使用真实数据进行模型训练",
                    "与模拟数据对比性能",
                    "评估真实数据的训练效果",
                    "考虑混合使用真实和模拟数据"
                ]
            }
        }

        # 保存报告
        output_dir = Path("outputs/real_data_collection")
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "real_data_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"📋 真实数据报告已保存: {output_dir / 'real_data_report.json'}")


async def main():
    """主函数"""
    print("\n" + "="*80)
    print("🍽️ 真实数据收集器")
    print("专门收集真实的API数据样本，确保数据质量")
    print("="*80 + "\n")

    collector = RealDataCollector()
    await collector.run_real_data_collection()

    print("\n" + "="*80)
    print("✅ 真实数据收集完成！")
    print("📁 真实数据已保存到: data/real_data_collection/")
    print("📊 报告已保存到: outputs/real_data_collection/")
    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
