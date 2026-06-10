#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试Nutritionix API连接和功能
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


class NutritionixAPITester:
    """Nutritionix API测试器"""

    def __init__(self):
        """初始化API测试器"""
        self.api_keys = self._load_api_keys()
        self.base_url = "https://trackapi.nutritionix.com/v2"
        self.headers = {
            "x-app-id": self.api_keys.get("nutritionix_app_id", ""),
            "x-app-key": self.api_keys.get("nutritionix", ""),
            "Content-Type": "application/json"
        }

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

    def test_api_connection(self) -> Dict[str, Any]:
        """测试API连接"""
        logger.info("🔍 测试Nutritionix API连接...")

        try:
            # 测试搜索端点
            url = f"{self.base_url}/search/instant"
            params = {
                "query": "apple"
            }

            response = requests.get(url, headers=self.headers, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ Nutritionix API连接成功")
                return {"status": "success", "message": "API连接正常", "data": data}
            else:
                return {
                    "status": "error",
                    "message": f"API请求失败: {response.status_code} - {response.text}"
                }

        except Exception as e:
            return {"status": "error", "message": f"API连接测试失败: {str(e)}"}

    def test_food_search(self) -> Dict[str, Any]:
        """测试食物搜索功能"""
        logger.info("🍎 测试食物搜索功能...")

        try:
            # 搜索食物
            url = f"{self.base_url}/search/instant"
            params = {
                "query": "apple"
            }

            response = requests.get(url, headers=self.headers, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                common_foods = data.get("common", [])
                branded_foods = data.get("branded", [])

                logger.info(f"✅ 搜索成功，找到 {len(common_foods)} 个常见食物，{len(branded_foods)} 个品牌食物")

                # 显示前几个结果
                for i, food in enumerate(common_foods[:3]):
                    logger.info(f"  {i+1}. {food.get('food_name', 'Unknown')}")

                return {
                    "status": "success",
                    "message": f"成功搜索到 {len(common_foods)} 个常见食物",
                    "common_foods": common_foods[:5],
                    "branded_foods": branded_foods[:5]
                }
            else:
                return {
                    "status": "error",
                    "message": f"搜索失败: {response.status_code} - {response.text}"
                }

        except Exception as e:
            return {"status": "error", "message": f"食物搜索测试失败: {str(e)}"}

    def test_nutrition_analysis(self) -> Dict[str, Any]:
        """测试营养分析功能"""
        logger.info("🥗 测试营养分析功能...")

        try:
            # 营养分析请求
            url = f"{self.base_url}/natural/nutrients"

            # 测试不同的食物
            test_foods = [
                "1 large apple",
                "1 banana",
                "1 cup rice",
                "2 slices bread",
                "100g chicken breast"
            ]

            nutrition_data = {}

            for food in test_foods:
                logger.info(f"   分析: {food}")

                payload = {
                    "query": food
                }

                response = requests.post(url, headers=self.headers, json=payload, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    foods = data.get("foods", [])

                    if foods:
                        food_info = foods[0]
                        nutrition_info = {
                            "food_name": food_info.get("food_name", food),
                            "calories": food_info.get("nf_calories", 0),
                            "protein": food_info.get("nf_protein", 0),
                            "carbs": food_info.get("nf_total_carbohydrate", 0),
                            "fat": food_info.get("nf_total_fat", 0),
                            "fiber": food_info.get("nf_dietary_fiber", 0),
                            "sugar": food_info.get("nf_sugars", 0),
                            "sodium": food_info.get("nf_sodium", 0)
                        }

                        nutrition_data[food] = nutrition_info
                        logger.info(f"     ✅ 卡路里: {nutrition_info['calories']} kcal")
                    else:
                        logger.warning(f"     ⚠️ 未找到营养信息")
                else:
                    logger.warning(f"     ❌ 分析失败: {response.status_code}")

            logger.info(f"✅ 营养分析完成，成功分析了 {len(nutrition_data)} 种食物")

            return {
                "status": "success",
                "message": f"成功分析了 {len(nutrition_data)} 种食物的营养信息",
                "nutrition_data": nutrition_data
            }

        except Exception as e:
            return {"status": "error", "message": f"营养分析测试失败: {str(e)}"}

    def test_food_database_search(self) -> Dict[str, Any]:
        """测试食物数据库搜索"""
        logger.info("🔍 测试食物数据库搜索...")

        try:
            # 搜索食物数据库
            url = f"{self.base_url}/search/item"
            params = {
                "nix_item_id": "513fc9e73fe3ffd40300109e"  # 示例食物ID
            }

            response = requests.get(url, headers=self.headers, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                food = data.get("foods", [])

                if food:
                    food_info = food[0]
                    logger.info(f"✅ 数据库搜索成功")
                    logger.info(f"   食物名称: {food_info.get('food_name', 'Unknown')}")
                    logger.info(f"   品牌: {food_info.get('brand_name', 'Unknown')}")

                    return {
                        "status": "success",
                        "message": "数据库搜索成功",
                        "food_info": food_info
                    }
                else:
                    return {"status": "error", "message": "未找到食物信息"}
            else:
                return {
                    "status": "error",
                    "message": f"数据库搜索失败: {response.status_code} - {response.text}"
                }

        except Exception as e:
            return {"status": "error", "message": f"数据库搜索测试失败: {str(e)}"}

    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        logger.info("🚀 开始Nutritionix API测试...")

        results = {
            "api_connection": self.test_api_connection(),
            "food_search": self.test_food_search(),
            "nutrition_analysis": self.test_nutrition_analysis(),
            "database_search": self.test_food_database_search()
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
        output_dir = Path("outputs/nutritionix_test")
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "nutritionix_test_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"📋 测试结果已保存: {output_dir / 'nutritionix_test_results.json'}")


def main():
    """主函数"""
    print("\n" + "="*80)
    print("🔍 Nutritionix API 连接和功能测试")
    print("="*80 + "\n")

    tester = NutritionixAPITester()
    results = tester.run_all_tests()

    # 输出测试摘要
    print("\n" + "="*80)
    print("📊 测试结果摘要")
    print("="*80)

    for test_name, result in results.items():
        status_icon = "✅" if result["status"] == "success" else "❌"
        print(f"{status_icon} {test_name}: {result['message']}")

    print("\n" + "="*80)
    print("✅ Nutritionix API测试完成！")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
