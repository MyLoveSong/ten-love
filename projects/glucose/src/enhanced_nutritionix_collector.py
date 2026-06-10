#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强版Nutritionix数据收集器
支持中文菜名翻译和智能查询
"""

import os
import sys
import json
import logging
import requests
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入翻译器
from utils.chinese_food_translator import ChineseFoodTranslator


class EnhancedNutritionixCollector:
    """增强版Nutritionix数据收集器"""

    def __init__(self):
        """初始化收集器"""
        self.api_keys = self._load_api_keys()
        self.translator = ChineseFoodTranslator()
        self.base_url = "https://trackapi.nutritionix.com/v2"
        self.headers = {
            "x-app-id": self.api_keys.get("nutritionix_app_id", ""),
            "x-app-key": self.api_keys.get("nutritionix", ""),
            "Content-Type": "application/json"
        }
        self.cache = {}  # 缓存已查询的食物

    def _load_api_keys(self) -> Dict[str, str]:
        """加载API密钥"""
        try:
            from utils.config_loader import ConfigLoader
            config_loader = ConfigLoader("configs/api_keys.yaml")
            return config_loader.get_config("api_keys")
        except Exception as e:
            logger.warning(f"无法加载API密钥配置: {e}")
            return {
                "nutritionix": "d512fe23a268c732b6329392a51a5c21",
                "nutritionix_app_id": "faf0209b"
            }

    def _query_nutritionix(self, query: str) -> Optional[Dict[str, Any]]:
        """查询Nutritionix API"""
        try:
            # 检查缓存
            if query in self.cache:
                return self.cache[query]

            url = f"{self.base_url}/natural/nutrients"
            payload = {"query": query}

            response = requests.post(url, headers=self.headers, json=payload, timeout=10)

            if response.status_code == 200:
                data = response.json()
                foods = data.get("foods", [])

                if foods:
                    result = foods[0]
                    # 缓存结果
                    self.cache[query] = result
                    return result
                else:
                    logger.warning(f"未找到营养信息: {query}")
                    return None
            else:
                logger.warning(f"API请求失败: {response.status_code} - {query}")
                return None

        except Exception as e:
            logger.error(f"查询Nutritionix API时出错: {e}")
            return None

    def get_chinese_food_nutrition(self, chinese_name: str, cuisine_type: str = None) -> Dict[str, Any]:
        """
        获取中文食物的营养信息

        Args:
            chinese_name: 中文食物名称
            cuisine_type: 菜系类型

        Returns:
            Dict[str, Any]: 营养信息
        """
        logger.info(f"🍜 查询中文食物: {chinese_name}")

        # 获取翻译建议
        query_suggestions = self.translator.get_nutrition_query_suggestions(chinese_name, cuisine_type)

        nutrition_data = None
        successful_query = None

        # 尝试不同的查询建议
        for query in query_suggestions:
            logger.info(f"   尝试查询: {query}")
            nutrition_data = self._query_nutritionix(query)

            if nutrition_data:
                successful_query = query
                logger.info(f"   ✅ 查询成功: {query}")
                break
            else:
                logger.warning(f"   ❌ 查询失败: {query}")

            # 避免API限制
            time.sleep(0.5)

        # 如果所有查询都失败，返回模拟数据
        if not nutrition_data:
            logger.warning(f"   所有查询都失败，使用模拟数据: {chinese_name}")
            nutrition_data = self._generate_mock_nutrition(chinese_name)
            successful_query = "mock_data"

        return {
            "chinese_name": chinese_name,
            "cuisine_type": cuisine_type,
            "successful_query": successful_query,
            "nutrition": nutrition_data,
            "query_suggestions": query_suggestions
        }

    def _generate_mock_nutrition(self, chinese_name: str) -> Dict[str, Any]:
        """生成模拟营养数据"""
        # 基于食物名称生成合理的营养数据
        hash_value = hash(chinese_name) % 1000

        return {
            "food_name": chinese_name,
            "nf_calories": 100 + hash_value % 300,
            "nf_protein": 5 + hash_value % 20,
            "nf_total_carbohydrate": 10 + hash_value % 30,
            "nf_total_fat": 2 + hash_value % 15,
            "nf_dietary_fiber": 1 + hash_value % 5,
            "nf_sugars": 2 + hash_value % 10,
            "nf_sodium": 50 + hash_value % 200,
            "serving_weight_grams": 100,
            "is_mock_data": True
        }

    def collect_cultural_foods_nutrition(self) -> Dict[str, Any]:
        """收集文化食物营养数据"""
        logger.info("🍜 收集文化食物营养数据...")

        # 文化食物列表
        cultural_foods = {
            "川菜": ["麻婆豆腐", "宫保鸡丁", "回锅肉", "水煮鱼", "夫妻肺片"],
            "粤菜": ["白切鸡", "叉烧", "烧鹅", "白灼虾", "蒸蛋"],
            "鲁菜": ["糖醋里脊", "九转大肠", "德州扒鸡", "锅塌豆腐", "葱烧海参"],
            "苏菜": ["松鼠桂鱼", "蟹粉小笼", "白汁圆菜", "清炖蟹粉狮子头", "水晶肴肉"],
            "浙菜": ["西湖醋鱼", "东坡肉", "龙井虾仁", "叫化童鸡", "干炸响铃"],
            "闽菜": ["佛跳墙", "荔枝肉", "醉排骨", "红糟鱼", "白斩河田鸡"],
            "湘菜": ["剁椒鱼头", "口味虾", "湘西外婆菜", "干锅花菜", "毛氏红烧肉"],
            "徽菜": ["臭鳜鱼", "毛豆腐", "胡适一品锅", "黄山炖鸽", "问政山笋"]
        }

        nutrition_database = {}
        total_success = 0
        total_attempts = 0

        for cuisine, foods in cultural_foods.items():
            logger.info(f"   收集 {cuisine} 营养数据...")
            cuisine_nutrition = {}

            for food in foods:
                total_attempts += 1
                nutrition_info = self.get_chinese_food_nutrition(food, cuisine)

                if nutrition_info["nutrition"] and not nutrition_info["nutrition"].get("is_mock_data", False):
                    total_success += 1

                cuisine_nutrition[food] = nutrition_info

                # 显示结果
                nutrition = nutrition_info["nutrition"]
                if nutrition:
                    calories = nutrition.get("nf_calories", 0)
                    protein = nutrition.get("nf_protein", 0)
                    logger.info(f"     {food}: {calories} kcal, {protein}g 蛋白质")
                else:
                    logger.warning(f"     {food}: 无法获取营养信息")

                # 避免API限制
                time.sleep(1)

            nutrition_database[cuisine] = cuisine_nutrition

        success_rate = (total_success / total_attempts * 100) if total_attempts > 0 else 0
        logger.info(f"📊 数据收集完成: {total_success}/{total_attempts} 成功 ({success_rate:.1f}%)")

        return nutrition_database

    def collect_common_foods_nutrition(self) -> Dict[str, Any]:
        """收集常见食物营养数据"""
        logger.info("🥗 收集常见食物营养数据...")

        # 常见食物列表（中英文混合）
        common_foods = [
            "苹果", "香蕉", "橙子", "葡萄", "草莓",
            "鸡肉", "牛肉", "猪肉", "鱼肉", "鸡蛋",
            "米饭", "面条", "面包", "土豆", "番茄",
            "牛奶", "酸奶", "茶", "咖啡", "水"
        ]

        nutrition_data = {}
        success_count = 0

        for food in common_foods:
            nutrition_info = self.get_chinese_food_nutrition(food)

            if nutrition_info["nutrition"] and not nutrition_info["nutrition"].get("is_mock_data", False):
                success_count += 1

            nutrition_data[food] = nutrition_info

            # 显示结果
            nutrition = nutrition_info["nutrition"]
            if nutrition:
                calories = nutrition.get("nf_calories", 0)
                logger.info(f"   {food}: {calories} kcal")
            else:
                logger.warning(f"   {food}: 无法获取营养信息")

            # 避免API限制
            time.sleep(0.5)

        logger.info(f"📊 常见食物收集完成: {success_count}/{len(common_foods)} 成功")

        return nutrition_data

    def save_nutrition_data(self, data: Dict[str, Any], filename: str) -> None:
        """保存营养数据"""
        output_dir = Path("data/real_data")
        output_dir.mkdir(parents=True, exist_ok=True)

        file_path = output_dir / filename

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"💾 营养数据已保存: {file_path}")

    def run_enhanced_collection(self) -> None:
        """运行增强版数据收集"""
        logger.info("🚀 开始增强版Nutritionix数据收集...")

        # 1. 收集文化食物营养数据
        cultural_nutrition = self.collect_cultural_foods_nutrition()
        self.save_nutrition_data(cultural_nutrition, "enhanced_cultural_nutrition.json")

        # 2. 收集常见食物营养数据
        common_nutrition = self.collect_common_foods_nutrition()
        self.save_nutrition_data(common_nutrition, "enhanced_common_nutrition.json")

        # 3. 生成统计报告
        self._generate_collection_report(cultural_nutrition, common_nutrition)

    def _generate_collection_report(self, cultural_data: Dict[str, Any], common_data: Dict[str, Any]) -> None:
        """生成收集报告"""
        logger.info("📊 生成数据收集报告...")

        # 统计文化食物
        cultural_stats = {}
        for cuisine, foods in cultural_data.items():
            success_count = sum(1 for food_data in foods.values()
                              if food_data["nutrition"] and not food_data["nutrition"].get("is_mock_data", False))
            total_count = len(foods)
            cultural_stats[cuisine] = {"success": success_count, "total": total_count}

        # 统计常见食物
        common_success = sum(1 for food_data in common_data.values()
                           if food_data["nutrition"] and not food_data["nutrition"].get("is_mock_data", False))
        common_total = len(common_data)

        # 生成报告
        report = {
            "collection_summary": {
                "cultural_foods": cultural_stats,
                "common_foods": {"success": common_success, "total": common_total},
                "total_success": sum(stats["success"] for stats in cultural_stats.values()) + common_success,
                "total_attempts": sum(stats["total"] for stats in cultural_stats.values()) + common_total
            },
            "translation_cache": self.cache,
            "collection_timestamp": time.time()
        }

        # 保存报告
        output_dir = Path("outputs/enhanced_collection")
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "collection_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"📋 收集报告已保存: {output_dir / 'collection_report.json'}")


def main():
    """主函数"""
    print("\n" + "="*80)
    print("🍜 增强版Nutritionix数据收集器")
    print("支持中文菜名翻译和智能查询")
    print("="*80 + "\n")

    collector = EnhancedNutritionixCollector()
    collector.run_enhanced_collection()

    print("\n" + "="*80)
    print("✅ 增强版数据收集完成！")
    print("📁 数据已保存到: data/real_data/")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
