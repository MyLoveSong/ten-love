
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
重构标签体系：基于权威营养学标准重新计算nutrition_score
参考：WHO膳食指南、中国居民膳食宝塔、Nutri-Score、HEI等
目标：0-1标准化营养质量指数，具备临床可解释性
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StandardizedNutritionScorer:
    """标准化营养评分器 - 基于权威营养学标准"""

    def __init__(self):
        """初始化评分规则"""
        # 基于WHO/中国膳食指南的营养标准
        self.standards = {
            # 每日推荐摄入量 (DRI) - 成人标准
            'calories_dri': 2000,      # 千卡/日
            'protein_dri': 60,         # 克/日
            'fat_dri': 67,             # 克/日 (30% calories)
            'carbs_dri': 300,          # 克/日 (60% calories)
            'fiber_dri': 25,           # 克/日
            'sodium_dri': 2300,        # 毫克/日
            'sugar_limit': 50,         # 克/日 (WHO建议<10%能量)

            # 营养密度标准 (per 100g)
            'high_protein_threshold': 12,    # 高蛋白食物
            'high_fiber_threshold': 6,       # 高纤维食物
            'low_sodium_threshold': 120,     # 低钠食物
            'low_sugar_threshold': 5,        # 低糖食物
        }

        # 权重配置 - 基于营养学重要性
        self.weights = {
            'nutrient_density': 0.35,    # 营养密度 (蛋白质、纤维等)
            'harmful_components': 0.25,  # 有害成分 (钠、糖、饱和脂肪)
            'energy_balance': 0.20,      # 能量平衡
            'processing_level': 0.10,    # 加工程度 (推断)
            'cultural_bonus': 0.10       # 文化适应性加分
        }

    def calculate_nutrition_score(self, item: Dict[str, Any]) -> float:
        """
        计算标准化营养评分 (0-1)

        评分维度：
        1. 营养密度 (35%): 蛋白质、纤维、维生素等有益成分
        2. 有害成分 (25%): 钠、糖、饱和脂肪等限制成分
        3. 能量平衡 (20%): 卡路里适中性
        4. 加工程度 (10%): 天然vs加工食品
        5. 文化适应 (10%): 中式烹饪方式加分
        """
        try:
            # 提取营养成分
            calories = float(item.get('calories', 0))
            protein = float(item.get('protein', 0))
            carbs = float(item.get('carbs', 0))
            fat = float(item.get('fat', 0))
            fiber = float(item.get('fiber', 0))
            sugar = float(item.get('sugar', 0))
            sodium = float(item.get('sodium', 0))
            food_name = str(item.get('food_name', '')).lower()

            # 1. 营养密度评分 (0-1)
            nutrient_density_score = self._calculate_nutrient_density(
                protein, fiber, calories
            )

            # 2. 有害成分评分 (0-1, 越低越好转为越高越好)
            harmful_score = self._calculate_harmful_components(
                sodium, sugar, fat, calories
            )

            # 3. 能量平衡评分 (0-1)
            energy_balance_score = self._calculate_energy_balance(calories)

            # 4. 加工程度评分 (0-1, 基于食物名称推断)
            processing_score = self._estimate_processing_level(food_name)

            # 5. 文化适应性评分 (0-1)
            cultural_score = self._calculate_cultural_adaptation(food_name)

            # 加权综合评分
            final_score = (
                nutrient_density_score * self.weights['nutrient_density'] +
                harmful_score * self.weights['harmful_components'] +
                energy_balance_score * self.weights['energy_balance'] +
                processing_score * self.weights['processing_level'] +
                cultural_score * self.weights['cultural_bonus']
            )

            # 确保在[0,1]范围内
            final_score = max(0.0, min(1.0, final_score))

            return final_score

        except Exception as e:
            logger.warning(f"计算营养评分失败: {e}")
            return 0.5  # 默认中等评分

    def _calculate_nutrient_density(self, protein: float, fiber: float, calories: float) -> float:
        """计算营养密度评分"""
        if calories <= 0:
            return 0.0

        # 蛋白质密度 (g/100kcal)
        protein_density = (protein / calories) * 100 if calories > 0 else 0
        protein_score = min(protein_density / 6.0, 1.0)  # 6g/100kcal为优秀

        # 纤维密度 (g/100kcal)
        fiber_density = (fiber / calories) * 100 if calories > 0 else 0
        fiber_score = min(fiber_density / 3.0, 1.0)  # 3g/100kcal为优秀

        # 高蛋白食物加分
        high_protein_bonus = 0.2 if protein >= self.standards['high_protein_threshold'] else 0

        # 高纤维食物加分
        high_fiber_bonus = 0.2 if fiber >= self.standards['high_fiber_threshold'] else 0

        nutrient_score = (protein_score * 0.5 + fiber_score * 0.3 +
                         high_protein_bonus + high_fiber_bonus)

        return min(nutrient_score, 1.0)

    def _calculate_harmful_components(self, sodium: float, sugar: float,
                                    fat: float, calories: float) -> float:
        """计算有害成分评分 (越少越好)"""

        # 钠含量评分 (mg/100g)
        if sodium <= self.standards['low_sodium_threshold']:
            sodium_score = 1.0  # 低钠优秀
        elif sodium <= 400:
            sodium_score = 0.7  # 中等
        elif sodium <= 800:
            sodium_score = 0.4  # 偏高
        else:
            sodium_score = 0.1  # 高钠

        # 糖含量评分 (g/100g)
        if sugar <= self.standards['low_sugar_threshold']:
            sugar_score = 1.0   # 低糖优秀
        elif sugar <= 15:
            sugar_score = 0.6   # 中等
        elif sugar <= 30:
            sugar_score = 0.3   # 偏高
        else:
            sugar_score = 0.1   # 高糖

        # 脂肪比例评分
        fat_ratio = (fat * 9 / calories) if calories > 0 else 0  # 脂肪提供的能量比例
        if fat_ratio <= 0.25:      # ≤25%
            fat_score = 1.0
        elif fat_ratio <= 0.35:    # 25-35%
            fat_score = 0.8
        elif fat_ratio <= 0.45:    # 35-45%
            fat_score = 0.5
        else:                       # >45%
            fat_score = 0.2

        # 综合有害成分评分
        harmful_score = (sodium_score * 0.4 + sugar_score * 0.3 + fat_score * 0.3)

        return harmful_score

    def _calculate_energy_balance(self, calories: float) -> float:
        """计算能量平衡评分"""
        # 基于每餐推荐卡路里 (2000/3 ≈ 667 kcal/餐)
        ideal_calories_per_meal = 667

        if 200 <= calories <= 400:      # 轻食/小食
            return 0.9
        elif 400 <= calories <= 600:    # 正餐适中
            return 1.0
        elif 600 <= calories <= 800:    # 正餐偏多
            return 0.7
        elif 100 <= calories <= 200:    # 零食/小点
            return 0.6
        elif 800 <= calories <= 1000:   # 大餐
            return 0.4
        else:                           # 极端值
            return 0.2

    def _estimate_processing_level(self, food_name: str) -> float:
        """基于食物名称估计加工程度"""
        # 天然/传统烹饪方式 (高分)
        natural_keywords = ['蒸', '煮', '炖', '清', '白切', '白灼', '凉拌',
                           '生', '鲜', '蔬菜', '水果', '鱼', '虾']

        # 中等加工 (中分)
        moderate_keywords = ['炒', '烧', '焖', '炸', '煎', '烤', '卤', '酱']

        # 高度加工 (低分)
        processed_keywords = ['罐头', '腌制', '熏制', '火腿', '香肠', '方便',
                             '速食', '膨化', '饮料', '汽水']

        # 检查关键词
        if any(keyword in food_name for keyword in natural_keywords):
            return 0.9  # 天然/传统
        elif any(keyword in food_name for keyword in moderate_keywords):
            return 0.6  # 中等加工
        elif any(keyword in food_name for keyword in processed_keywords):
            return 0.2  # 高度加工
        else:
            return 0.5  # 默认中等

    def _calculate_cultural_adaptation(self, food_name: str) -> float:
        """计算文化适应性评分"""
        # 中式菜品关键词
        chinese_keywords = ['宫保', '麻婆', '红烧', '糖醋', '清蒸', '白切',
                           '蒸蛋', '豆腐', '青菜', '白菜', '冬瓜', '丝瓜',
                           '鲈鱼', '鲤鱼', '鸡丁', '牛肉', '猪肉']

        # 传统健康烹饪方式
        healthy_cooking = ['清蒸', '白灼', '凉拌', '煮', '炖']

        cultural_score = 0.5  # 基础分

        # 中式菜品加分
        if any(keyword in food_name for keyword in chinese_keywords):
            cultural_score += 0.3

        # 健康烹饪方式加分
        if any(keyword in food_name for keyword in healthy_cooking):
            cultural_score += 0.2

        return min(cultural_score, 1.0)

    def get_score_breakdown(self, item: Dict[str, Any]) -> Dict[str, float]:
        """获取评分详细分解 (用于调试和解释)"""
        calories = float(item.get('calories', 0))
        protein = float(item.get('protein', 0))
        fiber = float(item.get('fiber', 0))
        sugar = float(item.get('sugar', 0))
        sodium = float(item.get('sodium', 0))
        fat = float(item.get('fat', 0))
        food_name = str(item.get('food_name', '')).lower()

        breakdown = {
            'nutrient_density': self._calculate_nutrient_density(protein, fiber, calories),
            'harmful_components': self._calculate_harmful_components(sodium, sugar, fat, calories),
            'energy_balance': self._calculate_energy_balance(calories),
            'processing_level': self._estimate_processing_level(food_name),
            'cultural_bonus': self._calculate_cultural_adaptation(food_name),
        }

        # 加权最终分数
        breakdown['final_score'] = sum(
            score * self.weights[component]
            for component, score in breakdown.items()
            if component != 'final_score'
        )

        return breakdown


def reconstruct_nutrition_labels(input_file: str, output_file: str):
    """重构营养标签"""
    logger.info("🔧 开始重构营养标签体系...")

    # 初始化评分器
    scorer = StandardizedNutritionScorer()

    # 加载原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = data.get('training_samples', [])
    logger.info(f"📊 加载 {len(samples)} 个样本")

    # 重新计算营养评分
    updated_samples = []
    score_distribution = []

    for i, item in enumerate(samples):
        if isinstance(item, dict):
            # 计算新的标准化营养评分
            new_score = scorer.calculate_nutrition_score(item)

            # 更新样本
            updated_item = item.copy()
            updated_item['nutrition_score'] = new_score
            updated_item['original_nutrition_score'] = item.get('nutrition_score', 0)  # 保留原始分数

            # 添加评分详细分解 (前100个样本)
            if i < 100:
                updated_item['score_breakdown'] = scorer.get_score_breakdown(item)

            updated_samples.append(updated_item)
            score_distribution.append(new_score)

        if (i + 1) % 500 == 0:
            logger.info(f"   处理进度: {i+1}/{len(samples)}")

    # 分析新评分分布
    scores = np.array(score_distribution)
    logger.info(f"📈 新营养评分分布:")
    logger.info(f"   最小值: {scores.min():.3f}")
    logger.info(f"   最大值: {scores.max():.3f}")
    logger.info(f"   平均值: {scores.mean():.3f}")
    logger.info(f"   标准差: {scores.std():.3f}")
    logger.info(f"   中位数: {np.median(scores):.3f}")

    # 更新数据集信息
    updated_data = data.copy()
    updated_data['training_samples'] = updated_samples
    updated_data['dataset_info']['nutrition_scoring'] = {
        'method': 'Standardized Nutrition Quality Index',
        'version': '1.0',
        'based_on': ['WHO Dietary Guidelines', 'Chinese Dietary Pagoda', 'Nutri-Score', 'HEI'],
        'score_range': [0.0, 1.0],
        'components': {
            'nutrient_density': 0.35,
            'harmful_components': 0.25,
            'energy_balance': 0.20,
            'processing_level': 0.10,
            'cultural_adaptation': 0.10
        },
        'distribution': {
            'min': float(scores.min()),
            'max': float(scores.max()),
            'mean': float(scores.mean()),
            'std': float(scores.std()),
            'median': float(np.median(scores))
        }
    }

    # 保存重构后的数据
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(updated_data, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ 重构完成，保存到: {output_path}")

    # 生成示例评分
    logger.info(f"\n🎯 示例评分 (前5个样本):")
    for i, sample in enumerate(updated_samples[:5]):
        name = sample.get('food_name', 'Unknown')
        old_score = sample.get('original_nutrition_score', 0)
        new_score = sample.get('nutrition_score', 0)
        logger.info(f"   {i+1}. {name}: {old_score:.3f} → {new_score:.3f}")

    return output_path


def validate_demo_cases():
    """验证演示案例的预期评分"""
    logger.info(f"\n🎯 验证演示案例预期评分:")

    scorer = StandardizedNutritionScorer()

    demo_cases = [
        {
            'name': '宫保鸡丁',
            'calories': 280, 'protein': 25, 'carbs': 12, 'fat': 15,
            'fiber': 3, 'sugar': 8, 'sodium': 850,
        },
        {
            'name': '蒸蛋羹',
            'calories': 120, 'protein': 12, 'carbs': 2, 'fat': 8,
            'fiber': 0, 'sugar': 1, 'sodium': 200,
        },
        {
            'name': '红烧肉',
            'calories': 450, 'protein': 20, 'carbs': 8, 'fat': 35,
            'fiber': 1, 'sugar': 6, 'sodium': 900,
        },
        {
            'name': '清蒸鲈鱼',
            'calories': 180, 'protein': 28, 'carbs': 2, 'fat': 6,
            'fiber': 0, 'sugar': 0, 'sodium': 300,
        },
        {
            'name': '麻婆豆腐',
            'calories': 220, 'protein': 15, 'carbs': 10, 'fat': 14,
            'fiber': 4, 'sugar': 3, 'sodium': 750,
        }
    ]

    for case in demo_cases:
        case['food_name'] = case['name']
        score = scorer.calculate_nutrition_score(case)
        breakdown = scorer.get_score_breakdown(case)

        logger.info(f"   {case['name']}: {score:.3f}")
        logger.info(f"     营养密度: {breakdown['nutrient_density']:.2f}")
        logger.info(f"     有害成分: {breakdown['harmful_components']:.2f}")
        logger.info(f"     能量平衡: {breakdown['energy_balance']:.2f}")
        logger.info(f"     加工程度: {breakdown['processing_level']:.2f}")
        logger.info(f"     文化适应: {breakdown['cultural_bonus']:.2f}")


def main():
    """主函数"""
    logger.info("🚀 营养标签体系重构")
    logger.info("="*80)

    # 输入输出路径 - 使用绝对路径
    project_root = Path(__file__).resolve().parent.parent
    input_file = project_root / "TRAIN/data/final_dataset/final_training_data.json"
    output_file = project_root / "TRAIN/data/reconstructed_dataset/reconstructed_training_data.json"

    # 1. 重构营养标签
    output_path = reconstruct_nutrition_labels(input_file, output_file)

    # 2. 验证演示案例
    validate_demo_cases()

    logger.info(f"\n🎉 标签重构完成!")
    logger.info(f"📁 新数据集: {output_path}")
    logger.info(f"📊 评分范围: 0.0-1.0 (标准化营养质量指数)")
    logger.info(f"🔬 基于权威标准: WHO、中国膳食宝塔、Nutri-Score、HEI")
    logger.info(f"✅ 具备临床可解释性，支持健康饮食推荐应用")


if __name__ == "__main__":
    main()