#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
中文食物名称翻译工具
将中文菜名翻译为英文，以便在Nutritionix API中查询
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChineseFoodTranslator:
    """中文食物名称翻译器"""

    def __init__(self):
        """初始化翻译器"""
        self.translation_dict = self._load_translation_dict()
        self.cultural_food_mapping = self._load_cultural_food_mapping()

    def _load_translation_dict(self) -> Dict[str, str]:
        """加载基础翻译词典"""
        return {
            # 基础食材
            "苹果": "apple",
            "香蕉": "banana",
            "橙子": "orange",
            "葡萄": "grape",
            "草莓": "strawberry",
            "蓝莓": "blueberry",
            "西瓜": "watermelon",
            "桃子": "peach",
            "梨": "pear",
            "柠檬": "lemon",

            # 蔬菜
            "番茄": "tomato",
            "土豆": "potato",
            "胡萝卜": "carrot",
            "洋葱": "onion",
            "大蒜": "garlic",
            "白菜": "cabbage",
            "菠菜": "spinach",
            "芹菜": "celery",
            "黄瓜": "cucumber",
            "茄子": "eggplant",
            "青椒": "green pepper",
            "红椒": "red pepper",
            "蘑菇": "mushroom",
            "西兰花": "broccoli",
            "花菜": "cauliflower",

            # 肉类
            "鸡肉": "chicken",
            "牛肉": "beef",
            "猪肉": "pork",
            "羊肉": "lamb",
            "鱼肉": "fish",
            "虾": "shrimp",
            "蟹": "crab",
            "鸡蛋": "egg",
            "鸭肉": "duck",
            "鹅肉": "goose",

            # 主食
            "米饭": "rice",
            "面条": "noodles",
            "面包": "bread",
            "馒头": "steamed bun",
            "包子": "baozi",
            "饺子": "dumpling",
            "面条": "pasta",
            "粥": "porridge",

            # 调料
            "盐": "salt",
            "糖": "sugar",
            "酱油": "soy sauce",
            "醋": "vinegar",
            "油": "oil",
            "胡椒": "pepper",
            "辣椒": "chili",
            "姜": "ginger",
            "葱": "scallion",
            "蒜": "garlic",

            # 饮品
            "牛奶": "milk",
            "酸奶": "yogurt",
            "茶": "tea",
            "咖啡": "coffee",
            "果汁": "juice",
            "水": "water",
            "啤酒": "beer",
            "红酒": "red wine",
            "白酒": "white wine"
        }

    def _load_cultural_food_mapping(self) -> Dict[str, Dict[str, str]]:
        """加载文化食物映射"""
        return {
            "川菜": {
                "麻婆豆腐": "mapo tofu",
                "宫保鸡丁": "kung pao chicken",
                "回锅肉": "twice-cooked pork",
                "水煮鱼": "boiled fish",
                "夫妻肺片": "sliced beef and ox tongue in chili sauce",
                "鱼香肉丝": "fish-flavored shredded pork",
                "辣子鸡": "spicy chicken",
                "口水鸡": "mouth-watering chicken"
            },
            "粤菜": {
                "白切鸡": "white cut chicken",
                "叉烧": "char siu",
                "烧鹅": "roast goose",
                "白灼虾": "blanched shrimp",
                "蒸蛋": "steamed egg",
                "白切鸡": "white cut chicken",
                "烧鸭": "roast duck",
                "白切鸡": "white cut chicken"
            },
            "鲁菜": {
                "糖醋里脊": "sweet and sour pork",
                "九转大肠": "nine-turn large intestine",
                "德州扒鸡": "dezhou braised chicken",
                "锅塌豆腐": "pan-fried tofu",
                "葱烧海参": "scallion-braised sea cucumber",
                "糖醋鲤鱼": "sweet and sour carp"
            },
            "苏菜": {
                "松鼠桂鱼": "squirrel mandarin fish",
                "蟹粉小笼": "crab roe xiaolongbao",
                "白汁圆菜": "white sauce round vegetables",
                "清炖蟹粉狮子头": "clear stewed crab roe lion's head",
                "水晶肴肉": "crystal meat",
                "蟹粉豆腐": "crab roe tofu"
            },
            "浙菜": {
                "西湖醋鱼": "west lake vinegar fish",
                "东坡肉": "dongpo pork",
                "龙井虾仁": "longjing shrimp",
                "叫化童鸡": "beggar's chicken",
                "干炸响铃": "dry-fried bell",
                "宋嫂鱼羹": "song sister fish soup"
            },
            "闽菜": {
                "佛跳墙": "buddha jumps over the wall",
                "荔枝肉": "lychee meat",
                "醉排骨": "drunken ribs",
                "红糟鱼": "red wine fish",
                "白斩河田鸡": "white cut hetian chicken",
                "福州鱼丸": "fuzhou fish balls"
            },
            "湘菜": {
                "剁椒鱼头": "chopped chili fish head",
                "口味虾": "flavor shrimp",
                "湘西外婆菜": "western hunan grandmother's vegetables",
                "干锅花菜": "dry pot cauliflower",
                "毛氏红烧肉": "mao's red-braised pork",
                "湘西腊肉": "western hunan cured meat"
            },
            "徽菜": {
                "臭鳜鱼": "stinky mandarin fish",
                "毛豆腐": "hairy tofu",
                "胡适一品锅": "hu shi first-class pot",
                "黄山炖鸽": "huangshan stewed pigeon",
                "问政山笋": "wenzheng mountain bamboo shoots",
                "徽州毛豆腐": "huizhou hairy tofu"
            }
        }

    def translate_food_name(self, chinese_name: str, cuisine_type: str = None) -> List[Tuple[str, float]]:
        """
        翻译中文食物名称为英文

        Args:
            chinese_name: 中文食物名称
            cuisine_type: 菜系类型（可选）

        Returns:
            List[Tuple[str, float]]: 翻译结果列表，包含(英文名称, 置信度)
        """
        # 1. 首先尝试文化食物映射
        if cuisine_type and cuisine_type in self.cultural_food_mapping:
            if chinese_name in self.cultural_food_mapping[cuisine_type]:
                english_name = self.cultural_food_mapping[cuisine_type][chinese_name]
                return [(english_name, 0.95)]  # 高置信度

        # 2. 尝试基础翻译词典
        if chinese_name in self.translation_dict:
            english_name = self.translation_dict[chinese_name]
            return [(english_name, 0.9)]  # 高置信度

        # 3. 尝试部分匹配
        partial_matches = []
        for cn_name, en_name in self.translation_dict.items():
            if cn_name in chinese_name or chinese_name in cn_name:
                confidence = len(cn_name) / len(chinese_name) if len(chinese_name) > 0 else 0
                partial_matches.append((en_name, confidence))

        if partial_matches:
            # 按置信度排序
            partial_matches.sort(key=lambda x: x[1], reverse=True)
            return partial_matches[:3]  # 返回前3个匹配

        # 4. 尝试拼音转换（简单实现）
        pinyin_translation = self._pinyin_to_english(chinese_name)
        if pinyin_translation:
            return [(pinyin_translation, 0.3)]  # 低置信度

        # 5. 返回原始名称（作为最后选择）
        return [(chinese_name, 0.1)]

    def _pinyin_to_english(self, chinese_name: str) -> Optional[str]:
        """简单的拼音到英文转换"""
        # 这里可以实现更复杂的拼音转换逻辑
        # 目前返回None，表示无法转换
        return None

    def get_nutrition_query_suggestions(self, chinese_name: str, cuisine_type: str = None) -> List[str]:
        """
        获取营养查询建议

        Args:
            chinese_name: 中文食物名称
            cuisine_type: 菜系类型

        Returns:
            List[str]: 查询建议列表
        """
        translations = self.translate_food_name(chinese_name, cuisine_type)
        suggestions = []

        for english_name, confidence in translations:
            if confidence > 0.5:  # 只使用高置信度的翻译
                suggestions.append(english_name)

        # 添加一些通用的查询建议
        if not suggestions:
            suggestions = [
                f"1 serving {chinese_name}",
                f"100g {chinese_name}",
                f"chinese {chinese_name}",
                f"traditional {chinese_name}"
            ]

        return suggestions[:5]  # 返回最多5个建议

    def batch_translate_foods(self, food_list: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        批量翻译食物列表

        Args:
            food_list: 食物列表，每个元素包含name和cuisine_type

        Returns:
            List[Dict[str, Any]]: 翻译结果列表
        """
        results = []

        for food in food_list:
            chinese_name = food.get("name", "")
            cuisine_type = food.get("cuisine_type", "")

            translations = self.translate_food_name(chinese_name, cuisine_type)
            suggestions = self.get_nutrition_query_suggestions(chinese_name, cuisine_type)

            result = {
                "chinese_name": chinese_name,
                "cuisine_type": cuisine_type,
                "translations": translations,
                "query_suggestions": suggestions,
                "best_translation": translations[0] if translations else None
            }

            results.append(result)

        return results

    def save_translation_results(self, results: List[Dict[str, Any]], filename: str = "translation_results.json") -> None:
        """保存翻译结果"""
        output_dir = Path("outputs/translation")
        output_dir.mkdir(parents=True, exist_ok=True)

        file_path = output_dir / filename

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"💾 翻译结果已保存: {file_path}")


def main():
    """主函数 - 演示翻译功能"""
    print("\n" + "="*80)
    print("🈳 中文食物名称翻译工具演示")
    print("="*80 + "\n")

    translator = ChineseFoodTranslator()

    # 测试食物列表
    test_foods = [
        {"name": "麻婆豆腐", "cuisine_type": "川菜"},
        {"name": "宫保鸡丁", "cuisine_type": "川菜"},
        {"name": "白切鸡", "cuisine_type": "粤菜"},
        {"name": "苹果", "cuisine_type": ""},
        {"name": "鸡肉", "cuisine_type": ""},
        {"name": "未知食物", "cuisine_type": ""}
    ]

    # 批量翻译
    results = translator.batch_translate_foods(test_foods)

    # 显示结果
    for result in results:
        print(f"中文名称: {result['chinese_name']}")
        print(f"菜系: {result['cuisine_type']}")
        print(f"最佳翻译: {result['best_translation']}")
        print(f"查询建议: {', '.join(result['query_suggestions'][:3])}")
        print("-" * 50)

    # 保存结果
    translator.save_translation_results(results)

    print("\n" + "="*80)
    print("✅ 翻译演示完成！")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
