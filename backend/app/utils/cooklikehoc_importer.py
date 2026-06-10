

"""
CookLikeHOC数据导入工具
从CookLikeHOC项目导入菜谱数据到健康监测系统
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
import requests
from sqlalchemy.orm import Session

from ..database.database_manager import DatabaseManager, get_db
from ..models.recipe_models import Recipe, IngredientNutrition, RecipeDataValidator
from ..services.recipe_service import RecipeService

logger = logging.getLogger(__name__)

class CookLikeHOCDataImporter:
    """CookLikeHOC数据导入器"""

    def __init__(self, db_session: Session):
        self.db = db_session
        self.recipe_service = RecipeService(db_session)
        self.validator = RecipeDataValidator()

        # CookLikeHOC项目信息
        self.cooklikehoc_repo = "Gar-b-age/CookLikeHOC"
        self.github_api_base = "https://api.github.com"

        # 菜谱类别映射
        self.category_mapping = {
            "主食": "主食",
            "凉拌": "凉拌",
            "卤菜": "卤菜",
            "早餐": "早餐",
            "汤": "汤",
            "炒菜": "炒菜",
            "炖菜": "炖菜",
            "炸品": "炸品",
            "烤类": "烤类",
            "烫菜": "烫菜",
            "煮锅": "煮锅",
            "砂锅菜": "砂锅菜",
            "蒸菜": "蒸菜",
            "配料": "配料",
            "饮品": "饮品"
        }

        # 食材营养数据库（部分示例数据）
        self.ingredient_nutrition_db = {
            "鸡胸肉": {
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
            "花生米": {
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
            },
            "大米": {
                "calories_per_100g": 130.0,
                "protein": 2.7,
                "carbs": 28.0,
                "fat": 0.3,
                "fiber": 0.4,
                "sugar": 0.1,
                "sodium": 1.0,
                "glycemic_index": 73.0,
                "health_benefits": ["提供能量", "易消化"],
                "cultural_significance": "亚洲主食，文化意义重大",
                "common_names": ["米", "白米", "大米"]
            },
            "猪肉": {
                "calories_per_100g": 250.0,
                "protein": 26.0,
                "carbs": 0.0,
                "fat": 15.0,
                "fiber": 0.0,
                "sugar": 0.0,
                "sodium": 65.0,
                "glycemic_index": 0.0,
                "health_benefits": ["高蛋白", "富含铁质"],
                "cultural_significance": "中式烹饪主要肉类",
                "common_names": ["猪肉", "猪肉丝", "猪肉片"]
            },
            "蔬菜": {
                "calories_per_100g": 25.0,
                "protein": 2.0,
                "carbs": 5.0,
                "fat": 0.2,
                "fiber": 2.5,
                "sugar": 3.0,
                "sodium": 10.0,
                "glycemic_index": 15.0,
                "health_benefits": ["高纤维", "富含维生素", "低热量"],
                "cultural_significance": "健康饮食的重要组成部分",
                "common_names": ["青菜", "蔬菜", "时蔬"]
            }
        }

    async def import_all_data(self) -> Dict[str, Any]:
        """导入所有数据"""
        results = {
            "ingredients_imported": 0,
            "recipes_imported": 0,
            "errors": []
        }

        try:
            # 1. 导入食材营养数据
            logger.info("开始导入食材营养数据...")
            ingredients_result = await self.import_ingredient_nutrition()
            results["ingredients_imported"] = ingredients_result["imported"]
            results["errors"].extend(ingredients_result["errors"])

            # 2. 导入示例菜谱数据
            logger.info("开始导入示例菜谱数据...")
            recipes_result = await self.import_sample_recipes()
            results["recipes_imported"] = recipes_result["imported"]
            results["errors"].extend(recipes_result["errors"])

            logger.info(f"数据导入完成: {results}")
            return results

        except Exception as e:
            logger.error(f"数据导入失败: {e}")
            results["errors"].append(str(e))
            return results

    async def import_ingredient_nutrition(self) -> Dict[str, Any]:
        """导入食材营养数据"""
        result = {"imported": 0, "errors": []}

        try:
            for ingredient_name, nutrition_data in self.ingredient_nutrition_db.items():
                try:
                    # 检查是否已存在
                    existing = self.db.query(IngredientNutrition).filter(
                        IngredientNutrition.ingredient_name == ingredient_name
                    ).first()

                    if existing:
                        logger.info(f"食材 {ingredient_name} 已存在，跳过")
                        continue

                    # 创建食材营养记录
                    nutrition = IngredientNutrition(
                        ingredient_name=ingredient_name,
                        **nutrition_data
                    )

                    self.db.add(nutrition)
                    result["imported"] += 1
                    logger.info(f"导入食材营养数据: {ingredient_name}")

                except Exception as e:
                    error_msg = f"导入食材 {ingredient_name} 失败: {e}"
                    logger.error(error_msg)
                    result["errors"].append(error_msg)

            self.db.commit()
            logger.info(f"食材营养数据导入完成，共导入 {result['imported']} 条")

        except Exception as e:
            self.db.rollback()
            error_msg = f"食材营养数据导入失败: {e}"
            logger.error(error_msg)
            result["errors"].append(error_msg)

        return result

    async def import_sample_recipes(self) -> Dict[str, Any]:
        """导入示例菜谱数据"""
        result = {"imported": 0, "errors": []}

        # 示例菜谱数据
        sample_recipes = [
            {
                "name": "宫保鸡丁",
                "category": "炒菜",
                "description": "经典川菜，麻辣鲜香，营养丰富",
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
                "difficulty_level": 3,
                "cooking_time": 20,
                "servings": 2,
                "instructions": """1. 鸡胸肉切丁，用料酒、生抽腌制15分钟
2. 热锅下油，爆炒花生米至金黄盛起
3. 下鸡丁炒至变色，加入干辣椒、花椒炒香
4. 加入葱姜蒜爆炒，调入生抽、老抽、白糖、醋
5. 最后加入花生米炒匀即可""",
                "tips": "鸡丁要切得均匀，炒制时火候要掌握好，避免过老"
            },
            {
                "name": "小笼包",
                "category": "早餐",
                "description": "上海传统点心，皮薄馅嫩，汤汁丰富",
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
                "difficulty_level": 4,
                "cooking_time": 60,
                "servings": 4,
                "instructions": """1. 面粉加水和成面团，醒发30分钟
2. 猪肉馅加虾仁、调料拌匀，腌制20分钟
3. 面团擀成薄皮，包入馅料，收口
4. 上锅蒸15分钟即可""",
                "tips": "面皮要擀得薄而均匀，蒸制时间要掌握好，避免破皮"
            },
            {
                "name": "西红柿鸡蛋汤",
                "category": "汤",
                "description": "家常汤品，营养丰富，制作简单",
                "ingredients": [
                    {"name": "西红柿", "amount": "2个"},
                    {"name": "鸡蛋", "amount": "2个"},
                    {"name": "大葱", "amount": "1根"},
                    {"name": "生姜", "amount": "1块"},
                    {"name": "盐", "amount": "适量"},
                    {"name": "糖", "amount": "1茶匙"},
                    {"name": "香油", "amount": "1茶匙"},
                    {"name": "水", "amount": "500ml"}
                ],
                "cooking_method": "煮",
                "cultural_tags": ["家常", "简单", "营养"],
                "difficulty_level": 1,
                "cooking_time": 15,
                "servings": 2,
                "instructions": """1. 西红柿切块，鸡蛋打散
2. 热锅下油，爆香葱姜
3. 下西红柿炒出汁水
4. 加水煮开，调入盐糖
5. 淋入蛋液，搅散即可""",
                "tips": "西红柿要炒出汁水，蛋液要慢慢淋入"
            },
            {
                "name": "凉拌黄瓜",
                "category": "凉拌",
                "description": "清爽开胃的凉拌菜，适合夏季",
                "ingredients": [
                    {"name": "黄瓜", "amount": "2根"},
                    {"name": "大蒜", "amount": "3瓣"},
                    {"name": "生姜", "amount": "1块"},
                    {"name": "生抽", "amount": "2汤匙"},
                    {"name": "醋", "amount": "1汤匙"},
                    {"name": "糖", "amount": "1茶匙"},
                    {"name": "香油", "amount": "1茶匙"},
                    {"name": "盐", "amount": "适量"}
                ],
                "cooking_method": "凉拌",
                "cultural_tags": ["清爽", "开胃", "夏季"],
                "difficulty_level": 1,
                "cooking_time": 10,
                "servings": 2,
                "instructions": """1. 黄瓜拍松切段
2. 大蒜生姜切末
3. 调好料汁：生抽、醋、糖、香油、盐
4. 黄瓜段加料汁拌匀
5. 撒上蒜姜末即可""",
                "tips": "黄瓜要拍松，这样更入味"
            },
            {
                "name": "红烧肉",
                "category": "炖菜",
                "description": "经典家常菜，肥而不腻，香甜可口",
                "ingredients": [
                    {"name": "五花肉", "amount": "500g"},
                    {"name": "冰糖", "amount": "50g"},
                    {"name": "生抽", "amount": "3汤匙"},
                    {"name": "老抽", "amount": "1汤匙"},
                    {"name": "料酒", "amount": "2汤匙"},
                    {"name": "大葱", "amount": "2根"},
                    {"name": "生姜", "amount": "1块"},
                    {"name": "八角", "amount": "2个"},
                    {"name": "桂皮", "amount": "1块"},
                    {"name": "盐", "amount": "适量"}
                ],
                "cooking_method": "炖",
                "cultural_tags": ["经典", "家常", "下饭"],
                "difficulty_level": 3,
                "cooking_time": 90,
                "servings": 4,
                "instructions": """1. 五花肉切块，焯水去腥
2. 热锅下冰糖炒糖色
3. 下肉块炒至上色
4. 加入调料和香料
5. 加水炖煮1小时即可""",
                "tips": "炒糖色要掌握火候，避免炒苦"
            }
        ]

        try:
            for recipe_data in sample_recipes:
                try:
                    # 检查是否已存在
                    existing = self.db.query(Recipe).filter(
                        Recipe.name == recipe_data["name"]
                    ).first()

                    if existing:
                        logger.info(f"菜谱 {recipe_data['name']} 已存在，跳过")
                        continue

                    # 创建菜谱
                    recipe = self.recipe_service.create_recipe(recipe_data)
                    result["imported"] += 1
                    logger.info(f"导入菜谱: {recipe.name}")

                except Exception as e:
                    error_msg = f"导入菜谱 {recipe_data['name']} 失败: {e}"
                    logger.error(error_msg)
                    result["errors"].append(error_msg)

            logger.info(f"示例菜谱导入完成，共导入 {result['imported']} 条")

        except Exception as e:
            error_msg = f"示例菜谱导入失败: {e}"
            logger.error(error_msg)
            result["errors"].append(error_msg)

        return result

    async def sync_from_github(self) -> Dict[str, Any]:
        """从GitHub同步CookLikeHOC数据（模拟）"""
        result = {"synced": 0, "errors": []}

        try:
            # 这里可以添加从GitHub API获取数据的逻辑
            # 由于GitHub API的限制，这里提供模拟实现

            logger.info("开始从GitHub同步CookLikeHOC数据...")

            # 模拟同步过程
            await asyncio.sleep(1)

            logger.info("GitHub同步完成（模拟）")
            result["synced"] = 0  # 实际实现中会返回同步的菜谱数量

        except Exception as e:
            error_msg = f"GitHub同步失败: {e}"
            logger.error(error_msg)
            result["errors"].append(error_msg)

        return result

    def export_sample_data(self, output_file: str) -> bool:
        """导出示例数据到JSON文件"""
        try:
            sample_data = {
                "ingredients": self.ingredient_nutrition_db,
                "recipes": [
                    {
                        "name": "宫保鸡丁",
                        "category": "炒菜",
                        "description": "经典川菜，麻辣鲜香",
                        "ingredients": [
                            {"name": "鸡胸肉", "amount": "300g"},
                            {"name": "花生米", "amount": "50g"},
                            {"name": "干辣椒", "amount": "10个"}
                        ],
                        "cooking_method": "炒",
                        "cultural_tags": ["川菜", "经典"],
                        "difficulty_level": 3,
                        "cooking_time": 20,
                        "servings": 2
                    }
                ]
            }

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(sample_data, f, ensure_ascii=False, indent=2)

            logger.info(f"示例数据已导出到: {output_file}")
            return True

        except Exception as e:
            logger.error(f"导出示例数据失败: {e}")
            return False

# 命令行工具
async def main():
    """主函数 - 用于命令行导入数据"""
    import argparse

    parser = argparse.ArgumentParser(description='CookLikeHOC数据导入工具')
    parser.add_argument('--action', choices=['import', 'sync', 'export'],
                       default='import', help='执行的操作')
    parser.add_argument('--output', help='导出文件路径')

    args = parser.parse_args()

    # 初始化数据库连接
    db = next(get_db())
    importer = CookLikeHOCDataImporter(db)

    try:
        if args.action == 'import':
            result = await importer.import_all_data()
            print(f"导入完成: {result}")

        elif args.action == 'sync':
            result = await importer.sync_from_github()
            print(f"同步完成: {result}")

        elif args.action == 'export':
            if not args.output:
                args.output = 'cooklikehoc_sample_data.json'
            success = importer.export_sample_data(args.output)
            print(f"导出{'成功' if success else '失败'}")

    except Exception as e:
        logger.error(f"执行失败: {e}")
        print(f"执行失败: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    asyncio.run(main())

__all__ = ["'logger'", "'CookLikeHOCDataImporter'"]
