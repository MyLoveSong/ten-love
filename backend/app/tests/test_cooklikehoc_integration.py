

"""
CookLikeHOC集成测试
测试菜谱管理、推荐、营养分析等功能
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.app.database.database_manager import get_db
from backend.app.services.recipe_service import RecipeService
from backend.app.utils.cooklikehoc_importer import CookLikeHOCDataImporter

logger = logging.getLogger(__name__)

class CookLikeHOCTestSuite:
    """CookLikeHOC测试套件"""

    def __init__(self):
        self.db = next(get_db())
        self.recipe_service = RecipeService(self.db)
        self.importer = CookLikeHOCDataImporter(self.db)
        self.test_results = {}

    async def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        print("🧪 开始CookLikeHOC集成测试")
        print("=" * 50)

        tests = [
            ("数据导入测试", self.test_data_import),
            ("菜谱管理测试", self.test_recipe_management),
            ("营养分析测试", self.test_nutrition_analysis),
            ("个性化推荐测试", self.test_personalized_recommendations),
            ("用户偏好测试", self.test_user_preferences),
            ("搜索功能测试", self.test_search_functionality)
        ]

        for test_name, test_func in tests:
            print(f"\n🔍 运行测试: {test_name}")
            try:
                result = await test_func()
                self.test_results[test_name] = {"status": "PASS", "result": result}
                print(f"✅ {test_name}: 通过")
            except Exception as e:
                self.test_results[test_name] = {"status": "FAIL", "error": str(e)}
                print(f"❌ {test_name}: 失败 - {e}")

        return self.test_results

    async def test_data_import(self) -> Dict[str, Any]:
        """测试数据导入功能"""
        # 导入示例数据
        import_result = await self.importer.import_all_data()

        # 验证导入结果
        assert import_result["ingredients_imported"] > 0, "食材数据导入失败"
        assert import_result["recipes_imported"] > 0, "菜谱数据导入失败"

        return {
            "ingredients_imported": import_result["ingredients_imported"],
            "recipes_imported": import_result["recipes_imported"],
            "errors": import_result["errors"]
        }

    async def test_recipe_management(self) -> Dict[str, Any]:
        """测试菜谱管理功能"""
        # 测试获取菜谱列表
        recipes = self.recipe_service.get_recipes(limit=5)
        assert len(recipes) > 0, "无法获取菜谱列表"

        # 测试获取菜谱详情
        recipe = recipes[0]
        recipe_detail = self.recipe_service.get_recipe_by_id(recipe.id)
        assert recipe_detail is not None, "无法获取菜谱详情"

        # 测试创建新菜谱
        new_recipe_data = {
            "name": "测试菜谱",
            "category": "测试",
            "description": "这是一个测试菜谱",
            "ingredients": [
                {"name": "测试食材", "amount": "100g"}
            ],
            "cooking_method": "测试",
            "cultural_tags": ["测试"],
            "difficulty_level": 1,
            "cooking_time": 10,
            "servings": 1,
            "instructions": "测试制作步骤",
            "tips": "测试小贴士"
        }

        new_recipe = self.recipe_service.create_recipe(new_recipe_data)
        assert new_recipe.id is not None, "创建菜谱失败"

        # 测试更新菜谱
        update_data = {"description": "更新后的描述"}
        updated_recipe = self.recipe_service.update_recipe(new_recipe.id, update_data)
        assert updated_recipe.description == "更新后的描述", "更新菜谱失败"

        # 测试删除菜谱
        success = self.recipe_service.delete_recipe(new_recipe.id)
        assert success, "删除菜谱失败"

        return {
            "recipes_count": len(recipes),
            "recipe_detail": recipe_detail.name,
            "create_success": True,
            "update_success": True,
            "delete_success": True
        }

    async def test_nutrition_analysis(self) -> Dict[str, Any]:
        """测试营养分析功能"""
        # 获取一个菜谱进行营养分析
        recipes = self.recipe_service.get_recipes(limit=1)
        assert len(recipes) > 0, "没有可用的菜谱进行测试"

        recipe = recipes[0]
        nutritional_info = recipe.nutritional_info

        # 验证营养信息
        assert nutritional_info is not None, "菜谱缺少营养信息"
        assert "calories" in nutritional_info, "缺少卡路里信息"
        assert "protein" in nutritional_info, "缺少蛋白质信息"

        # 测试食材营养查询
        ingredient_nutrition = self.recipe_service.get_ingredient_nutrition("鸡胸肉")
        assert ingredient_nutrition is not None, "无法获取食材营养信息"

        return {
            "recipe_name": recipe.name,
            "nutritional_info": nutritional_info,
            "ingredient_nutrition": ingredient_nutrition
        }

    async def test_personalized_recommendations(self) -> Dict[str, Any]:
        """测试个性化推荐功能"""
        # 模拟用户档案
        health_profile = {
            "diabetes": True,
            "hypertension": False,
            "weight_loss": True
        }

        cultural_profile = {
            "cuisines": ["中式"],
            "cultural_tags": ["经典", "家常"],
            "meal_times": {"lunch": 12}
        }

        # 生成推荐
        recommendations = self.recipe_service.generate_personalized_recommendations(
            user_id="test_user_001",
            health_profile=health_profile,
            cultural_profile=cultural_profile,
            meal_type="lunch",
            limit=3
        )

        assert len(recommendations) > 0, "无法生成推荐"

        # 验证推荐结果
        for rec in recommendations:
            assert "recipe" in rec, "推荐缺少菜谱信息"
            assert "recommendation_score" in rec, "推荐缺少分数"
            assert "recommendation_reason" in rec, "推荐缺少理由"

        return {
            "recommendations_count": len(recommendations),
            "sample_recommendation": {
                "recipe_name": recommendations[0]["recipe"]["name"],
                "score": recommendations[0]["recommendation_score"],
                "reasons": recommendations[0]["recommendation_reason"]["reasons"]
            }
        }

    async def test_user_preferences(self) -> Dict[str, Any]:
        """测试用户偏好功能"""
        user_id = "test_user_002"

        # 测试创建用户偏好
        preferences_data = {
            "preferred_categories": ["炒菜", "汤"],
            "preferred_ingredients": ["鸡肉", "蔬菜"],
            "dietary_restrictions": ["低盐"],
            "health_goals": ["减重"],
            "cultural_background": {"region": "北方", "cuisine": "中式"},
            "cooking_skill_level": 2,
            "available_cooking_time": 30
        }

        preferences = self.recipe_service.update_user_preferences(user_id, preferences_data)
        assert preferences.user_id == user_id, "创建用户偏好失败"

        # 测试获取用户偏好
        retrieved_preferences = self.recipe_service.get_user_preferences(user_id)
        assert retrieved_preferences is not None, "无法获取用户偏好"
        assert retrieved_preferences.preferred_categories == ["炒菜", "汤"], "用户偏好数据不正确"

        # 测试推荐历史
        history = self.recipe_service.get_user_recommendation_history(user_id)
        assert isinstance(history, list), "无法获取推荐历史"

        return {
            "preferences_created": True,
            "preferences_retrieved": True,
            "history_count": len(history)
        }

    async def test_search_functionality(self) -> Dict[str, Any]:
        """测试搜索功能"""
        # 测试按类别搜索
        recipes_by_category = self.recipe_service.get_recipes({"category": "炒菜"})
        assert len(recipes_by_category) > 0, "按类别搜索失败"

        # 测试按健康标签搜索
        recipes_by_health = self.recipe_service.get_recipes({"health_tags": ["高蛋白"]})
        assert isinstance(recipes_by_health, list), "按健康标签搜索失败"

        # 测试关键词搜索
        recipes_by_keyword = self.recipe_service.get_recipes({"keyword": "鸡"})
        assert isinstance(recipes_by_keyword, list), "关键词搜索失败"

        # 测试食材搜索
        ingredients = self.recipe_service.search_ingredients("鸡")
        assert isinstance(ingredients, list), "食材搜索失败"

        return {
            "category_search": len(recipes_by_category),
            "health_search": len(recipes_by_health),
            "keyword_search": len(recipes_by_keyword),
            "ingredient_search": len(ingredients)
        }

    def print_test_summary(self):
        """打印测试总结"""
        print("\n" + "=" * 50)
        print("📊 测试结果总结")
        print("=" * 50)

        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result["status"] == "PASS")
        failed_tests = total_tests - passed_tests

        print(f"总测试数: {total_tests}")
        print(f"通过: {passed_tests}")
        print(f"失败: {failed_tests}")
        print(f"成功率: {(passed_tests/total_tests)*100:.1f}%")

        if failed_tests > 0:
            print("\n❌ 失败的测试:")
            for test_name, result in self.test_results.items():
                if result["status"] == "FAIL":
                    print(f"  - {test_name}: {result['error']}")

        print("\n✅ 所有测试完成!")

async def main():
    """主函数"""
    # 设置日志
    logging.basicConfig(level=logging.INFO)

    # 运行测试
    test_suite = CookLikeHOCTestSuite()
    results = await test_suite.run_all_tests()

    # 打印总结
    test_suite.print_test_summary()

    # 检查是否有失败的测试
    failed_tests = sum(1 for result in results.values() if result["status"] == "FAIL")
    if failed_tests > 0:
        print(f"\n⚠️  有 {failed_tests} 个测试失败")
        sys.exit(1)
    else:
        print("\n🎉 所有测试都通过了!")

if __name__ == "__main__":
    asyncio.run(main())

__all__ = ["'project_root'", "'logger'", "'CookLikeHOCTestSuite'"]
