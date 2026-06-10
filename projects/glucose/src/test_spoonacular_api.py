#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试Spoonacular API连接和功能
验证API密钥是否有效，并演示数据获取功能
"""

import os
import sys
import json
import logging
import requests
from pathlib import Path
from typing import Dict, Any, List

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SpoonacularAPITester:
    """Spoonacular API测试器"""

    def __init__(self):
        """初始化API测试器"""
        self.api_keys = self._load_api_keys()
        self.base_url = "https://api.spoonacular.com"
        self.api_key = self.api_keys.get("spoonacular", "")

    def _load_api_keys(self) -> Dict[str, str]:
        """加载API密钥"""
        try:
            from utils.config_loader import ConfigLoader
            config_loader = ConfigLoader("configs/api_keys.yaml")
            return config_loader.get_config("api_keys")
        except Exception as e:
            logger.warning(f"无法加载API密钥配置: {e}")
            return {
                "spoonacular": "373f3604a64a46dc95c567e895dd07e9"
            }

    def test_api_connection(self) -> Dict[str, Any]:
        """测试API连接"""
        logger.info("🔍 测试Spoonacular API连接...")

        try:
            # 测试搜索食谱端点
            url = f"{self.base_url}/recipes/search"
            params = {
                "apiKey": self.api_key,
                "query": "pasta",
                "number": 5
            }

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                logger.info(f"✅ Spoonacular API连接成功，找到 {len(results)} 个食谱")
                return {"status": "success", "message": "API连接正常", "data": data}
            else:
                return {
                    "status": "error",
                    "message": f"API请求失败: {response.status_code} - {response.text}"
                }

        except Exception as e:
            return {"status": "error", "message": f"API连接测试失败: {str(e)}"}

    def test_recipe_search(self) -> Dict[str, Any]:
        """测试食谱搜索功能"""
        logger.info("🍝 测试食谱搜索功能...")

        try:
            # 搜索食谱
            url = f"{self.base_url}/recipes/search"
            params = {
                "apiKey": self.api_key,
                "query": "chicken",
                "number": 5
            }

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])

                logger.info(f"✅ 搜索成功，找到 {len(results)} 个食谱")

                # 显示前几个结果
                for i, recipe in enumerate(results[:3]):
                    logger.info(f"  {i+1}. {recipe.get('title', 'Unknown')}")

                return {
                    "status": "success",
                    "message": f"成功搜索到 {len(results)} 个食谱",
                    "recipes": results
                }
            else:
                return {
                    "status": "error",
                    "message": f"搜索失败: {response.status_code} - {response.text}"
                }

        except Exception as e:
            return {"status": "error", "message": f"食谱搜索测试失败: {str(e)}"}

    def test_recipe_nutrition(self) -> Dict[str, Any]:
        """测试食谱营养分析功能"""
        logger.info("🥗 测试食谱营养分析功能...")

        try:
            # 获取食谱营养信息
            url = f"{self.base_url}/recipes/guessNutrition"
            params = {
                "apiKey": self.api_key,
                "title": "Chicken Pasta"
            }

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ 营养分析成功")
                logger.info(f"   卡路里: {data.get('calories', 0)}")
                logger.info(f"   蛋白质: {data.get('protein', 0)}g")
                logger.info(f"   碳水化合物: {data.get('carbs', 0)}g")
                logger.info(f"   脂肪: {data.get('fat', 0)}g")

                return {
                    "status": "success",
                    "message": "营养分析成功",
                    "nutrition": data
                }
            else:
                return {
                    "status": "error",
                    "message": f"营养分析失败: {response.status_code} - {response.text}"
                }

        except Exception as e:
            return {"status": "error", "message": f"营养分析测试失败: {str(e)}"}

    def test_ingredient_search(self) -> Dict[str, Any]:
        """测试食材搜索功能"""
        logger.info("🥕 测试食材搜索功能...")

        try:
            # 搜索食材
            url = f"{self.base_url}/food/ingredients/search"
            params = {
                "apiKey": self.api_key,
                "query": "tomato",
                "number": 5
            }

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])

                logger.info(f"✅ 食材搜索成功，找到 {len(results)} 个食材")

                # 显示前几个结果
                for i, ingredient in enumerate(results[:3]):
                    logger.info(f"  {i+1}. {ingredient.get('name', 'Unknown')}")

                return {
                    "status": "success",
                    "message": f"成功搜索到 {len(results)} 个食材",
                    "ingredients": results
                }
            else:
                return {
                    "status": "error",
                    "message": f"食材搜索失败: {response.status_code} - {response.text}"
                }

        except Exception as e:
            return {"status": "error", "message": f"食材搜索测试失败: {str(e)}"}

    def test_food_nutrition(self) -> Dict[str, Any]:
        """测试食物营养信息功能"""
        logger.info("🍎 测试食物营养信息功能...")

        try:
            # 获取食物营养信息
            url = f"{self.base_url}/food/ingredients/9266/information"
            params = {
                "apiKey": self.api_key,
                "amount": 100,
                "unit": "grams"
            }

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                nutrition = data.get("nutrition", {})
                nutrients = nutrition.get("nutrients", [])

                logger.info(f"✅ 营养信息获取成功")

                # 显示主要营养素
                for nutrient in nutrients[:5]:
                    name = nutrient.get("name", "Unknown")
                    amount = nutrient.get("amount", 0)
                    unit = nutrient.get("unit", "")
                    logger.info(f"   {name}: {amount} {unit}")

                return {
                    "status": "success",
                    "message": "营养信息获取成功",
                    "nutrition": data
                }
            else:
                return {
                    "status": "error",
                    "message": f"营养信息获取失败: {response.status_code} - {response.text}"
                }

        except Exception as e:
            return {"status": "error", "message": f"营养信息测试失败: {str(e)}"}

    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        logger.info("🚀 开始Spoonacular API测试...")

        results = {
            "api_connection": self.test_api_connection(),
            "recipe_search": self.test_recipe_search(),
            "recipe_nutrition": self.test_recipe_nutrition(),
            "ingredient_search": self.test_ingredient_search(),
            "food_nutrition": self.test_food_nutrition()
        }

        # 统计结果
        success_count = sum(1 for result in results.values() if result["status"] == "success")
        total_count = len(results)

        logger.info(f"📊 测试完成: {success_count}/{total_count} 项测试通过")

        # 保存测试结果
        self._save_test_results(results)

        return results

    def _save_test_results(self, results: Dict[str, Any]) -> None:
        """保存测试结果"""
        output_dir = Path("outputs/spoonacular_test")
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "spoonacular_test_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"📋 测试结果已保存: {output_dir / 'spoonacular_test_results.json'}")


def main():
    """主函数"""
    print("\n" + "="*80)
    print("🔍 Spoonacular API 连接和功能测试")
    print("="*80 + "\n")

    tester = SpoonacularAPITester()
    results = tester.run_all_tests()

    # 输出测试摘要
    print("\n" + "="*80)
    print("📊 测试结果摘要")
    print("="*80)

    for test_name, result in results.items():
        status_icon = "✅" if result["status"] == "success" else "❌"
        print(f"{status_icon} {test_name}: {result['message']}")

    print("\n" + "="*80)
    print("✅ Spoonacular API测试完成！")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
