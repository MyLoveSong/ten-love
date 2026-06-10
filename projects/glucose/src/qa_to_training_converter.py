#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问答JSON转训练样本转换器
将AI智能制造的问答题目JSON格式转换为训练样本格式，用于微调训练
"""

import json
import re
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QAToTrainingConverter:
    """问答JSON转训练样本转换器"""

    def __init__(self):
        """初始化转换器"""
        # 营养成分提取模式
        self.nutrition_patterns = {
            'calories': [
                r'(\d+(?:\.\d+)?)\s*(?:卡路里|kcal|大卡|千卡|cal)',
                r'热量[：:]?\s*(\d+(?:\.\d+)?)',
                r'能量[：:]?\s*(\d+(?:\.\d+)?)'
            ],
            'protein': [
                r'(\d+(?:\.\d+)?)\s*(?:克|g)\s*(?:蛋白质|蛋白)',
                r'蛋白质[：:]?\s*(\d+(?:\.\d+)?)',
                r'蛋白[：:]?\s*(\d+(?:\.\d+)?)\s*[克g]'
            ],
            'carbs': [
                r'(\d+(?:\.\d+)?)\s*(?:克|g)\s*(?:碳水化合物|碳水|糖类)',
                r'碳水化合物[：:]?\s*(\d+(?:\.\d+)?)',
                r'碳水[：:]?\s*(\d+(?:\.\d+)?)\s*[克g]'
            ],
            'fat': [
                r'(\d+(?:\.\d+)?)\s*(?:克|g)\s*(?:脂肪|脂质)',
                r'脂肪[：:]?\s*(\d+(?:\.\d+)?)',
                r'脂质[：:]?\s*(\d+(?:\.\d+)?)\s*[克g]'
            ],
            'fiber': [
                r'(\d+(?:\.\d+)?)\s*(?:克|g)\s*(?:纤维|膳食纤维|纤维素)',
                r'纤维[：:]?\s*(\d+(?:\.\d+)?)',
                r'膳食纤维[：:]?\s*(\d+(?:\.\d+)?)\s*[克g]'
            ],
            'sugar': [
                r'(\d+(?:\.\d+)?)\s*(?:克|g)\s*(?:糖|蔗糖|添加糖)',
                r'糖[：:]?\s*(\d+(?:\.\d+)?)',
                r'蔗糖[：:]?\s*(\d+(?:\.\d+)?)\s*[克g]'
            ],
            'sodium': [
                r'(\d+(?:\.\d+)?)\s*(?:毫克|mg)\s*(?:钠|盐)',
                r'钠[：:]?\s*(\d+(?:\.\d+)?)\s*[毫克mg]',
                r'盐[：:]?\s*(\d+(?:\.\d+)?)\s*[毫克mg]'
            ]
        }

        # 食物名称提取模式
        self.food_name_patterns = [
            r'([^，。；：\n]{2,15}?)的营养',
            r'([^，。；：\n]{2,15}?)含有',
            r'([^，。；：\n]{2,15}?)的(?:卡路里|热量|蛋白质)',
            r'(?:关于|对于|针对)([^，。；：\n]{2,15}?)的(?:问题|营养)'
        ]

    def extract_nutrition_info(self, text: str) -> Dict[str, float]:
        """从文本中提取营养信息"""
        nutrition = {}

        for nutrient, patterns in self.nutrition_patterns.items():
            value = None
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        value = float(match.group(1))
                        break
                    except (ValueError, IndexError):
                        continue

            # 如果未找到，尝试在JSON结构化数据中查找
            if value is None:
                value = self._extract_from_json_like(text, nutrient)

            nutrition[nutrient] = value if value is not None else 0.0

        return nutrition

    def _extract_from_json_like(self, text: str, nutrient: str) -> Optional[float]:
        """从JSON格式的文本中提取营养信息"""
        # 尝试查找JSON格式的营养数据
        json_pattern = rf'["\']?{nutrient}["\']?\s*[:：]\s*(\d+(?:\.\d+)?)'
        match = re.search(json_pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, IndexError):
                pass
        return None

    def extract_food_name(self, question: str, answer: str) -> str:
        """从问答中提取食物名称"""
        # 从问题中提取
        for pattern in self.food_name_patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                food_name = match.group(1).strip()
                if len(food_name) >= 2 and len(food_name) <= 20:
                    return food_name

        # 从答案中提取（尝试查找食物名称）
        food_keywords = ['宫保鸡丁', '红烧肉', '清蒸鲈鱼', '麻婆豆腐', '蒸蛋羹',
                        '糖醋排骨', '白切鸡', '酸菜鱼', '水煮鱼', '回锅肉']
        for keyword in food_keywords:
            if keyword in answer or keyword in question:
                return keyword

        # 默认返回"未知食物"
        return "未知食物"

    def calculate_nutrition_score(self, nutrition: Dict[str, float]) -> float:
        """计算营养评分 (0-1)"""
        calories = nutrition.get('calories', 0)
        protein = nutrition.get('protein', 0)
        fiber = nutrition.get('fiber', 0)
        sugar = nutrition.get('sugar', 0)
        sodium = nutrition.get('sodium', 0)
        fat = nutrition.get('fat', 0)

        if calories <= 0:
            return 0.5  # 默认中等评分

        # 营养密度评分
        protein_density = (protein / calories) * 100 if calories > 0 else 0
        protein_score = min(protein_density / 6.0, 1.0)

        fiber_density = (fiber / calories) * 100 if calories > 0 else 0
        fiber_score = min(fiber_density / 3.0, 1.0)

        # 有害成分评分（越少越好）
        sodium_score = 1.0 if sodium <= 120 else (0.7 if sodium <= 400 else (0.4 if sodium <= 800 else 0.1))
        sugar_score = 1.0 if sugar <= 5 else (0.6 if sugar <= 15 else (0.3 if sugar <= 30 else 0.1))

        # 能量平衡评分
        if 400 <= calories <= 600:
            energy_score = 1.0
        elif 200 <= calories <= 400 or 600 <= calories <= 800:
            energy_score = 0.8
        else:
            energy_score = 0.5

        # 加权综合评分
        final_score = (
            protein_score * 0.25 +
            fiber_score * 0.15 +
            sodium_score * 0.25 +
            sugar_score * 0.15 +
            energy_score * 0.20
        )

        return max(0.0, min(1.0, final_score))

    def convert_qa_to_sample(self, qa_item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """将单个问答项转换为训练样本"""
        try:
            # 提取问题和答案
            question = qa_item.get('question', '') or qa_item.get('Q', '') or ''
            answer = qa_item.get('answer', '') or qa_item.get('A', '') or qa_item.get('content', '')

            if not question or not answer:
                return None

            # 合并文本用于提取
            full_text = f"{question} {answer}"

            # 提取食物名称
            food_name = qa_item.get('food_name') or self.extract_food_name(question, answer)

            # 提取营养信息
            nutrition = self.extract_nutrition_info(full_text)

            # 如果没有提取到足够的营养信息，尝试从结构化字段获取
            if nutrition.get('calories', 0) == 0:
                nutrition.update({
                    'calories': qa_item.get('calories', qa_item.get('calories_per_100g', 0)) or 0,
                    'protein': qa_item.get('protein', qa_item.get('protein_per_100g', 0)) or 0,
                    'carbs': qa_item.get('carbs', qa_item.get('carbohydrates', 0)) or 0,
                    'fat': qa_item.get('fat', qa_item.get('total_fat', 0)) or 0,
                    'fiber': qa_item.get('fiber', qa_item.get('dietary_fiber', 0)) or 0,
                    'sugar': qa_item.get('sugar', qa_item.get('total_sugar', 0)) or 0,
                    'sodium': qa_item.get('sodium', qa_item.get('salt', 0)) or 0,
                })

            # 检查是否有足够的营养信息
            if nutrition.get('calories', 0) == 0 and nutrition.get('protein', 0) == 0:
                return None  # 数据不足，跳过

            # 计算营养评分
            nutrition_score = self.calculate_nutrition_score(nutrition)

            # 构建训练样本
            sample = {
                'food_name': food_name,
                'calories': float(nutrition['calories']) or 0.0,
                'protein': float(nutrition['protein']) or 0.0,
                'carbs': float(nutrition['carbs']) or 0.0,
                'fat': float(nutrition['fat']) or 0.0,
                'fiber': float(nutrition['fiber']) or 0.0,
                'sugar': float(nutrition['sugar']) or 0.0,
                'sodium': float(nutrition['sodium']) or 0.0,
                'nutrition_score': float(nutrition_score),
                'source': 'qa_json',
                'data_quality': 'qa_extracted',
                'quality_scores': {
                    'completeness': self._calculate_completeness(nutrition),
                    'consistency': 0.8,  # 问答数据假设为中等一致性
                    'overall': self._calculate_overall_quality(nutrition)
                },
                'metadata': {
                    'original_question': question[:200],  # 保留前200字符
                    'original_answer': answer[:500],  # 保留前500字符
                    'extraction_method': 'qa_converter'
                }
            }

            return sample

        except Exception as e:
            logger.warning(f"转换问答项失败: {e}")
            return None

    def _calculate_completeness(self, nutrition: Dict[str, float]) -> float:
        """计算数据完整性评分"""
        required_fields = ['calories', 'protein', 'carbs', 'fat']
        optional_fields = ['fiber', 'sugar', 'sodium']

        required_count = sum(1 for field in required_fields if nutrition.get(field, 0) > 0)
        optional_count = sum(1 for field in optional_fields if nutrition.get(field, 0) > 0)

        completeness = (required_count / len(required_fields)) * 0.7 + (optional_count / len(optional_fields)) * 0.3
        return completeness

    def _calculate_overall_quality(self, nutrition: Dict[str, float]) -> float:
        """计算总体质量评分"""
        completeness = self._calculate_completeness(nutrition)

        # 合理性检查
        calories = nutrition.get('calories', 0)
        protein = nutrition.get('protein', 0)
        carbs = nutrition.get('carbs', 0)
        fat = nutrition.get('fat', 0)

        # 基本合理性：总热量应该约为蛋白质*4 + 碳水*4 + 脂肪*9
        estimated_calories = protein * 4 + carbs * 4 + fat * 9
        if calories > 0:
            calorie_consistency = 1.0 - min(abs(estimated_calories - calories) / max(calories, 1), 0.5)
        else:
            calorie_consistency = 0.5

        overall = (completeness * 0.7 + calorie_consistency * 0.3)
        return max(0.5, min(1.0, overall))  # 问答数据质量假设在0.5-1.0之间

    def convert_qa_file(self, qa_file_path: Path, output_path: Path) -> Dict[str, Any]:
        """转换问答JSON文件为训练样本"""
        logger.info(f"📖 读取问答文件: {qa_file_path}")

        with open(qa_file_path, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)

        # 支持多种JSON格式
        qa_items = []
        if isinstance(qa_data, list):
            qa_items = qa_data
        elif isinstance(qa_data, dict):
            if 'questions' in qa_data:
                qa_items = qa_data['questions']
            elif 'qa_pairs' in qa_data:
                qa_items = qa_data['qa_pairs']
            elif 'data' in qa_data:
                qa_items = qa_data['data']
            else:
                # 尝试直接使用
                qa_items = [qa_data]

        logger.info(f"   找到 {len(qa_items)} 个问答项")

        # 转换每个问答项
        training_samples = []
        success_count = 0
        fail_count = 0

        for i, qa_item in enumerate(qa_items):
            sample = self.convert_qa_to_sample(qa_item)
            if sample:
                training_samples.append(sample)
                success_count += 1
            else:
                fail_count += 1

            if (i + 1) % 100 == 0:
                logger.info(f"   处理进度: {i+1}/{len(qa_items)}, 成功: {success_count}, 失败: {fail_count}")

        logger.info(f"✅ 转换完成: 成功 {success_count}, 失败 {fail_count}")

        # 构建输出数据
        output_data = {
            'training_samples': training_samples,
            'total_samples': len(training_samples),
            'source': 'qa_json_converter',
            'conversion_stats': {
                'total_qa_items': len(qa_items),
                'successful_conversions': success_count,
                'failed_conversions': fail_count,
                'success_rate': success_count / len(qa_items) if qa_items else 0
            },
            'dataset_info': {
                'version': '1.0',
                'created_by': 'QAToTrainingConverter',
                'source_type': 'qa_json'
            }
        }

        # 保存转换后的数据
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        logger.info(f"💾 保存转换结果到: {output_path}")

        return output_data

    def merge_with_existing_data(self, qa_samples: List[Dict], existing_data_path: Path,
                                 output_path: Path) -> Dict[str, Any]:
        """将问答转换的样本合并到现有训练数据中"""
        logger.info(f"🔄 合并问答样本到现有数据...")

        # 加载现有数据
        if existing_data_path.exists():
            with open(existing_data_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)

            existing_samples = existing_data.get('training_samples', [])
            logger.info(f"   现有样本数: {len(existing_samples)}")
        else:
            existing_samples = []
            existing_data = {'dataset_info': {}}

        # 合并样本
        merged_samples = existing_samples + qa_samples

        # 去重（基于food_name和营养成分）
        unique_samples = []
        seen_keys = set()

        for sample in merged_samples:
            # 创建唯一键
            key = (
                sample.get('food_name', ''),
                round(sample.get('calories', 0), 1),
                round(sample.get('protein', 0), 1)
            )

            if key not in seen_keys:
                seen_keys.add(key)
                unique_samples.append(sample)

        logger.info(f"   合并后样本数: {len(merged_samples)}")
        logger.info(f"   去重后样本数: {len(unique_samples)}")
        logger.info(f"   新增样本数: {len(unique_samples) - len(existing_samples)}")

        # 构建合并后的数据
        merged_data = existing_data.copy()
        merged_data['training_samples'] = unique_samples
        merged_data['total_samples'] = len(unique_samples)
        merged_data['dataset_info']['qa_merged'] = True
        merged_data['dataset_info']['qa_added_samples'] = len(unique_samples) - len(existing_samples)

        # 保存合并后的数据
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)

        logger.info(f"💾 保存合并结果到: {output_path}")

        return merged_data


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='问答JSON转训练样本转换器')
    parser.add_argument('--qa_file', type=str, required=True, help='问答JSON文件路径')
    parser.add_argument('--output', type=str, help='输出文件路径（可选）')
    parser.add_argument('--merge', type=str, help='合并到现有训练数据文件路径（可选）')
    parser.add_argument('--merged_output', type=str, help='合并后的输出文件路径（使用--merge时必需）')

    args = parser.parse_args()

    qa_file_path = Path(args.qa_file)
    if not qa_file_path.exists():
        logger.error(f"问答文件不存在: {qa_file_path}")
        return

    converter = QAToTrainingConverter()

    # 转换问答数据
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = qa_file_path.parent / f"{qa_file_path.stem}_training_samples.json"

    converted_data = converter.convert_qa_file(qa_file_path, output_path)

    logger.info(f"\n📊 转换统计:")
    logger.info(f"   总问答项: {converted_data['conversion_stats']['total_qa_items']}")
    logger.info(f"   成功转换: {converted_data['conversion_stats']['successful_conversions']}")
    logger.info(f"   转换成功率: {converted_data['conversion_stats']['success_rate']:.2%}")

    # 如果需要合并到现有数据
    if args.merge:
        if not args.merged_output:
            logger.error("使用--merge时，必须指定--merged_output")
            return

        merge_path = Path(args.merge)
        merged_output_path = Path(args.merged_output)

        merged_data = converter.merge_with_existing_data(
            converted_data['training_samples'],
            merge_path,
            merged_output_path
        )

        logger.info(f"\n✅ 合并完成!")
        logger.info(f"   原始样本数: {len(converted_data['training_samples'])}")
        logger.info(f"   合并后总样本数: {merged_data['total_samples']}")
        logger.info(f"   新增样本数: {merged_data['dataset_info']['qa_added_samples']}")

    logger.info(f"\n🎉 转换完成!")


if __name__ == "__main__":
    main()
