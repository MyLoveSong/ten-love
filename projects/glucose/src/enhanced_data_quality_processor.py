#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强数据质量处理器
实现多源验证、智能清洗、特征工程和质量评估
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
from collections import Counter
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import warnings
warnings.filterwarnings("ignore")

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DataQualityMetrics:
    """数据质量指标"""
    completeness: float = 0.0
    consistency: float = 0.0
    accuracy: float = 0.0
    uniqueness: float = 0.0
    validity: float = 0.0
    overall_score: float = 0.0

    def calculate_overall_score(self) -> float:
        """计算总体质量分数"""
        weights = {
            'completeness': 0.25,
            'consistency': 0.20,
            'accuracy': 0.20,
            'uniqueness': 0.15,
            'validity': 0.20
        }

        self.overall_score = sum(
            getattr(self, metric) * weight
            for metric, weight in weights.items()
        )
        return self.overall_score


@dataclass
class DataPoint:
    """数据点"""
    food_name: str
    cuisine: str
    nutrition: Dict[str, Any]
    cultural_features: Dict[str, Any]
    source: str
    quality_score: float
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataValidator:
    """数据验证器"""

    def __init__(self):
        self.nutrition_ranges = {
            'nf_calories': (0, 2000),
            'nf_protein': (0, 100),
            'nf_total_carbohydrate': (0, 200),
            'nf_total_fat': (0, 100),
            'nf_dietary_fiber': (0, 50),
            'nf_sodium': (0, 5000)
        }

        self.cultural_categories = {
            'cuisine_type': ['川菜', '粤菜', '鲁菜', '苏菜', '浙菜', '闽菜', '湘菜', '徽菜', '常见食物'],
            'food_category': ['主食', '蔬菜', '肉类', '水果', '饮料', '其他'],
            'cooking_method': ['炒', '煮', '烤', '蒸', '炸', '未知'],
            'flavor_profile': ['甜', '辣', '酸', '咸', '苦', '清淡']
        }

    def validate_nutrition_data(self, nutrition: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """验证营养数据"""
        errors = []

        for field, (min_val, max_val) in self.nutrition_ranges.items():
            if field in nutrition:
                value = nutrition[field]
                if not isinstance(value, (int, float)):
                    errors.append(f"{field} 不是数值类型")
                elif value < min_val or value > max_val:
                    errors.append(f"{field} 超出合理范围 [{min_val}, {max_val}]")

        return len(errors) == 0, errors

    def validate_cultural_data(self, cultural_features: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """验证文化特征数据"""
        errors = []

        for field, valid_values in self.cultural_categories.items():
            if field in cultural_features:
                value = cultural_features[field]
                if value not in valid_values:
                    errors.append(f"{field} 值 '{value}' 不在有效范围内")

        return len(errors) == 0, errors

    def validate_data_point(self, data_point: DataPoint) -> Tuple[bool, List[str]]:
        """验证数据点"""
        all_errors = []

        # 验证营养数据
        nutrition_valid, nutrition_errors = self.validate_nutrition_data(data_point.nutrition)
        all_errors.extend(nutrition_errors)

        # 验证文化数据
        cultural_valid, cultural_errors = self.validate_cultural_data(data_point.cultural_features)
        all_errors.extend(cultural_errors)

        # 验证基本字段
        if not data_point.food_name or len(data_point.food_name.strip()) == 0:
            all_errors.append("食物名称不能为空")

        if not data_point.cuisine or len(data_point.cuisine.strip()) == 0:
            all_errors.append("菜系不能为空")

        return len(all_errors) == 0, all_errors


class DataCleaner:
    """数据清洗器"""

    def __init__(self):
        self.nutrition_imputer = SimpleImputer(strategy='median')
        self.cultural_imputer = SimpleImputer(strategy='most_frequent')
        self.outlier_detector = IsolationForest(contamination=0.1, random_state=42)

    def clean_nutrition_data(self, nutrition: Dict[str, Any]) -> Dict[str, Any]:
        """清洗营养数据"""
        cleaned = nutrition.copy()

        # 处理缺失值
        for field in ['nf_calories', 'nf_protein', 'nf_total_carbohydrate', 'nf_total_fat', 'nf_dietary_fiber', 'nf_sodium']:
            if field not in cleaned or cleaned[field] is None:
                # 使用合理的默认值
                defaults = {
                    'nf_calories': 100,
                    'nf_protein': 5,
                    'nf_total_carbohydrate': 20,
                    'nf_total_fat': 3,
                    'nf_dietary_fiber': 2,
                    'nf_sodium': 200
                }
                cleaned[field] = defaults.get(field, 0)
            else:
                # 确保数值类型
                try:
                    cleaned[field] = float(cleaned[field])
                except (ValueError, TypeError):
                    cleaned[field] = 0

        # 处理异常值
        for field, value in cleaned.items():
            if isinstance(value, (int, float)):
                # 使用3σ规则处理异常值
                if field == 'nf_calories' and (value < 0 or value > 2000):
                    cleaned[field] = max(0, min(2000, value))
                elif field in ['nf_protein', 'nf_total_carbohydrate', 'nf_total_fat'] and (value < 0 or value > 100):
                    cleaned[field] = max(0, min(100, value))
                elif field == 'nf_dietary_fiber' and (value < 0 or value > 50):
                    cleaned[field] = max(0, min(50, value))
                elif field == 'nf_sodium' and (value < 0 or value > 5000):
                    cleaned[field] = max(0, min(5000, value))

        return cleaned

    def clean_cultural_data(self, cultural_features: Dict[str, Any]) -> Dict[str, Any]:
        """清洗文化特征数据"""
        cleaned = cultural_features.copy()

        # 处理缺失值
        defaults = {
            'cuisine_type': '常见食物',
            'food_category': '其他',
            'cooking_method': '未知',
            'flavor_profile': '清淡',
            'cultural_significance': 0.5
        }

        for field, default_value in defaults.items():
            if field not in cleaned or cleaned[field] is None or cleaned[field] == '':
                cleaned[field] = default_value

        # 标准化字符串值
        if 'cuisine_type' in cleaned:
            cleaned['cuisine_type'] = str(cleaned['cuisine_type']).strip()
        if 'food_category' in cleaned:
            cleaned['food_category'] = str(cleaned['food_category']).strip()
        if 'cooking_method' in cleaned:
            cleaned['cooking_method'] = str(cleaned['cooking_method']).strip()
        if 'flavor_profile' in cleaned:
            cleaned['flavor_profile'] = str(cleaned['flavor_profile']).strip()

        # 确保cultural_significance是数值
        if 'cultural_significance' in cleaned:
            try:
                cleaned['cultural_significance'] = float(cleaned['cultural_significance'])
                cleaned['cultural_significance'] = max(0, min(1, cleaned['cultural_significance']))
            except (ValueError, TypeError):
                cleaned['cultural_significance'] = 0.5

        return cleaned

    def detect_outliers(self, data_points: List[DataPoint]) -> List[int]:
        """检测异常值"""
        if len(data_points) < 10:
            return []

        # 提取数值特征
        features = []
        for dp in data_points:
            nutrition = dp.nutrition
            feature_vector = [
                nutrition.get('nf_calories', 0),
                nutrition.get('nf_protein', 0),
                nutrition.get('nf_total_carbohydrate', 0),
                nutrition.get('nf_total_fat', 0),
                nutrition.get('nf_dietary_fiber', 0),
                nutrition.get('nf_sodium', 0)
            ]
            features.append(feature_vector)

        features_array = np.array(features)

        # 检测异常值
        outlier_indices = self.outlier_detector.fit_predict(features_array)
        outlier_indices = np.where(outlier_indices == -1)[0].tolist()

        return outlier_indices


class FeatureEngineer:
    """特征工程师"""

    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_selector = None

    def extract_nutrition_features(self, nutrition: Dict[str, Any]) -> List[float]:
        """提取营养特征"""
        features = [
            nutrition.get('nf_calories', 0),
            nutrition.get('nf_protein', 0),
            nutrition.get('nf_total_carbohydrate', 0),
            nutrition.get('nf_total_fat', 0),
            nutrition.get('nf_dietary_fiber', 0),
            nutrition.get('nf_sodium', 0)
        ]

        # 计算衍生特征
        calories = features[0]
        if calories > 0:
            protein_ratio = features[1] / calories * 100
            carb_ratio = features[2] / calories * 100
            fat_ratio = features[3] / calories * 100
        else:
            protein_ratio = carb_ratio = fat_ratio = 0

        # 营养密度
        nutrition_density = (features[1] + features[2] + features[3]) / max(calories, 1)

        # 健康指数（基于营养比例）
        health_score = self._calculate_health_score(features)

        features.extend([
            protein_ratio, carb_ratio, fat_ratio, nutrition_density, health_score
        ])

        return features

    def _calculate_health_score(self, nutrition_features: List[float]) -> float:
        """计算健康分数"""
        calories, protein, carbs, fat, fiber, sodium = nutrition_features

        if calories == 0:
            return 0.5

        # 理想营养比例
        ideal_protein_ratio = 0.15
        ideal_carb_ratio = 0.55
        ideal_fat_ratio = 0.30

        # 计算实际比例
        total_macros = protein + carbs + fat
        if total_macros == 0:
            return 0.5

        actual_protein_ratio = protein / total_macros
        actual_carb_ratio = carbs / total_macros
        actual_fat_ratio = fat / total_macros

        # 计算偏差
        protein_deviation = abs(actual_protein_ratio - ideal_protein_ratio)
        carb_deviation = abs(actual_carb_ratio - ideal_carb_ratio)
        fat_deviation = abs(actual_fat_ratio - ideal_fat_ratio)

        # 纤维加分
        fiber_bonus = min(fiber / 10, 0.1)

        # 钠减分
        sodium_penalty = min(sodium / 1000, 0.1)

        # 计算健康分数
        health_score = 1.0 - (protein_deviation + carb_deviation + fat_deviation) / 3
        health_score += fiber_bonus - sodium_penalty

        return max(0, min(1, health_score))

    def extract_cultural_features(self, cultural_features: Dict[str, Any]) -> List[float]:
        """提取文化特征"""
        # 编码分类特征
        cuisine_encoding = self._encode_cuisine(cultural_features.get('cuisine_type', '常见食物'))
        category_encoding = self._encode_category(cultural_features.get('food_category', '其他'))
        cooking_encoding = self._encode_cooking_method(cultural_features.get('cooking_method', '未知'))
        flavor_encoding = self._encode_flavor(cultural_features.get('flavor_profile', '清淡'))

        # 文化重要性
        cultural_significance = cultural_features.get('cultural_significance', 0.5)

        features = [
            cuisine_encoding,
            category_encoding,
            cooking_encoding,
            flavor_encoding,
            cultural_significance
        ]

        return features

    def _encode_cuisine(self, cuisine: str) -> float:
        """编码菜系"""
        cuisine_scores = {
            '川菜': 0.9, '粤菜': 0.9, '鲁菜': 0.8, '苏菜': 0.8,
            '浙菜': 0.7, '闽菜': 0.7, '湘菜': 0.8, '徽菜': 0.6,
            '常见食物': 0.5
        }
        return cuisine_scores.get(cuisine, 0.5)

    def _encode_category(self, category: str) -> float:
        """编码食物类别"""
        category_scores = {
            '主食': 0.8, '蔬菜': 0.9, '肉类': 0.7, '水果': 0.9,
            '饮料': 0.6, '其他': 0.5
        }
        return category_scores.get(category, 0.5)

    def _encode_cooking_method(self, method: str) -> float:
        """编码烹饪方法"""
        method_scores = {
            '炒': 0.8, '煮': 0.9, '烤': 0.7, '蒸': 0.9,
            '炸': 0.6, '未知': 0.5
        }
        return method_scores.get(method, 0.5)

    def _encode_flavor(self, flavor: str) -> float:
        """编码口味"""
        flavor_scores = {
            '甜': 0.7, '辣': 0.8, '酸': 0.6, '咸': 0.5,
            '苦': 0.4, '清淡': 0.8
        }
        return flavor_scores.get(flavor, 0.5)

    def create_derived_features(self, data_points: List[DataPoint]) -> List[List[float]]:
        """创建衍生特征"""
        derived_features = []

        for dp in data_points:
            # 基础特征
            nutrition_features = self.extract_nutrition_features(dp.nutrition)
            cultural_features = self.extract_cultural_features(dp.cultural_features)

            # 组合特征
            combined_features = nutrition_features + cultural_features

            # 交互特征
            interaction_features = self._create_interaction_features(nutrition_features, cultural_features)

            # 最终特征向量
            final_features = combined_features + interaction_features
            derived_features.append(final_features)

        return derived_features

    def _create_interaction_features(self, nutrition: List[float], cultural: List[float]) -> List[float]:
        """创建交互特征"""
        interactions = []

        # 营养-文化交互
        for n_feat in nutrition[:3]:  # 前3个营养特征
            for c_feat in cultural[:2]:  # 前2个文化特征
                interactions.append(n_feat * c_feat)

        # 营养特征之间的交互
        if len(nutrition) >= 3:
            interactions.append(nutrition[0] * nutrition[1])  # 卡路里 * 蛋白质
            interactions.append(nutrition[1] * nutrition[2])  # 蛋白质 * 碳水化合物
            interactions.append(nutrition[2] * nutrition[3])  # 碳水化合物 * 脂肪

        return interactions


class DataQualityProcessor:
    """数据质量处理器"""

    def __init__(self):
        self.validator = DataValidator()
        self.cleaner = DataCleaner()
        self.feature_engineer = FeatureEngineer()
        self.quality_metrics = DataQualityMetrics()

    def process_data_points(self, data_points: List[DataPoint]) -> Tuple[List[DataPoint], DataQualityMetrics]:
        """处理数据点"""
        logger.info(f"🔧 开始处理 {len(data_points)} 个数据点")

        # 1. 数据验证
        valid_data_points = []
        validation_errors = []

        for i, dp in enumerate(data_points):
            is_valid, errors = self.validator.validate_data_point(dp)
            if is_valid:
                valid_data_points.append(dp)
            else:
                validation_errors.extend([(i, error) for error in errors])
                logger.warning(f"数据点 {i} 验证失败: {errors}")

        logger.info(f"   验证通过: {len(valid_data_points)}/{len(data_points)}")

        # 2. 数据清洗
        cleaned_data_points = []
        for dp in valid_data_points:
            cleaned_dp = DataPoint(
                food_name=dp.food_name.strip(),
                cuisine=dp.cuisine.strip(),
                nutrition=self.cleaner.clean_nutrition_data(dp.nutrition),
                cultural_features=self.cleaner.clean_cultural_data(dp.cultural_features),
                source=dp.source,
                quality_score=dp.quality_score,
                timestamp=dp.timestamp,
                metadata=dp.metadata
            )
            cleaned_data_points.append(cleaned_dp)

        logger.info(f"   清洗完成: {len(cleaned_data_points)} 个数据点")

        # 3. 异常值检测
        outlier_indices = self.cleaner.detect_outliers(cleaned_data_points)
        if outlier_indices:
            logger.info(f"   检测到 {len(outlier_indices)} 个异常值")
            # 可以选择移除或标记异常值
            for idx in sorted(outlier_indices, reverse=True):
                cleaned_data_points[idx].metadata['is_outlier'] = True

        # 4. 特征工程
        derived_features = self.feature_engineer.create_derived_features(cleaned_data_points)
        logger.info(f"   特征工程完成: {len(derived_features)} 个特征向量")

        # 5. 质量评估
        quality_metrics = self._assess_data_quality(cleaned_data_points)

        logger.info(f"✅ 数据处理完成，质量分数: {quality_metrics.overall_score:.3f}")

        return cleaned_data_points, quality_metrics

    def _assess_data_quality(self, data_points: List[DataPoint]) -> DataQualityMetrics:
        """评估数据质量"""
        if not data_points:
            return DataQualityMetrics()

        # 完整性评估
        completeness = self._assess_completeness(data_points)

        # 一致性评估
        consistency = self._assess_consistency(data_points)

        # 准确性评估
        accuracy = self._assess_accuracy(data_points)

        # 唯一性评估
        uniqueness = self._assess_uniqueness(data_points)

        # 有效性评估
        validity = self._assess_validity(data_points)

        metrics = DataQualityMetrics(
            completeness=completeness,
            consistency=consistency,
            accuracy=accuracy,
            uniqueness=uniqueness,
            validity=validity
        )

        metrics.calculate_overall_score()
        return metrics

    def _assess_completeness(self, data_points: List[DataPoint]) -> float:
        """评估完整性"""
        if not data_points:
            return 0.0

        total_fields = 0
        filled_fields = 0

        for dp in data_points:
            # 检查营养数据完整性
            nutrition_fields = ['nf_calories', 'nf_protein', 'nf_total_carbohydrate', 'nf_total_fat', 'nf_dietary_fiber', 'nf_sodium']
            for field in nutrition_fields:
                total_fields += 1
                if field in dp.nutrition and dp.nutrition[field] is not None:
                    filled_fields += 1

            # 检查文化特征完整性
            cultural_fields = ['cuisine_type', 'food_category', 'cooking_method', 'flavor_profile', 'cultural_significance']
            for field in cultural_fields:
                total_fields += 1
                if field in dp.cultural_features and dp.cultural_features[field] is not None:
                    filled_fields += 1

        return filled_fields / total_fields if total_fields > 0 else 0.0

    def _assess_consistency(self, data_points: List[DataPoint]) -> float:
        """评估一致性"""
        if len(data_points) < 2:
            return 1.0

        # 检查相同食物的数据一致性
        food_groups = {}
        for dp in data_points:
            if dp.food_name not in food_groups:
                food_groups[dp.food_name] = []
            food_groups[dp.food_name].append(dp)

        consistency_scores = []
        for food_name, group in food_groups.items():
            if len(group) > 1:
                # 计算营养数据的一致性
                nutrition_consistency = self._calculate_nutrition_consistency(group)
                consistency_scores.append(nutrition_consistency)

        return np.mean(consistency_scores) if consistency_scores else 1.0

    def _calculate_nutrition_consistency(self, group: List[DataPoint]) -> float:
        """计算营养数据一致性"""
        if len(group) < 2:
            return 1.0

        nutrition_fields = ['nf_calories', 'nf_protein', 'nf_total_carbohydrate', 'nf_total_fat']
        consistency_scores = []

        for field in nutrition_fields:
            values = [dp.nutrition.get(field, 0) for dp in group if field in dp.nutrition]
            if len(values) > 1:
                cv = np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
                consistency_scores.append(1 - min(cv, 1))  # 变异系数越小，一致性越高

        return np.mean(consistency_scores) if consistency_scores else 1.0

    def _assess_accuracy(self, data_points: List[DataPoint]) -> float:
        """评估准确性"""
        if not data_points:
            return 0.0

        # 基于源可信度评估准确性
        source_credibility = {
            'nutritionix': 0.9,
            'usda': 0.95,
            'spoonacular': 0.85,
            'mock': 0.3
        }

        accuracy_scores = []
        for dp in data_points:
            source_score = source_credibility.get(dp.source, 0.5)
            quality_score = dp.quality_score
            accuracy_scores.append((source_score + quality_score) / 2)

        return np.mean(accuracy_scores)

    def _assess_uniqueness(self, data_points: List[DataPoint]) -> float:
        """评估唯一性"""
        if not data_points:
            return 0.0

        # 检查重复数据
        food_names = [dp.food_name for dp in data_points]
        unique_names = set(food_names)

        # 计算重复率
        total_count = len(food_names)
        unique_count = len(unique_names)
        uniqueness = unique_count / total_count if total_count > 0 else 0.0

        return uniqueness

    def _assess_validity(self, data_points: List[DataPoint]) -> float:
        """评估有效性"""
        if not data_points:
            return 0.0

        valid_count = 0
        for dp in data_points:
            is_valid, _ = self.validator.validate_data_point(dp)
            if is_valid:
                valid_count += 1

        return valid_count / len(data_points)

    def save_processed_data(self, data_points: List[DataPoint], output_path: str = "data/processed_data.json"):
        """保存处理后的数据"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 转换为可序列化格式
        serializable_data = []
        for dp in data_points:
            serializable_data.append({
                "food_name": dp.food_name,
                "cuisine": dp.cuisine,
                "nutrition": dp.nutrition,
                "cultural_features": dp.cultural_features,
                "source": dp.source,
                "quality_score": dp.quality_score,
                "timestamp": dp.timestamp,
                "metadata": dp.metadata
            })

        # 保存数据
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable_data, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ 处理后数据已保存: {output_path}")

    def print_quality_report(self, metrics: DataQualityMetrics):
        """打印质量报告"""
        logger.info("📊 数据质量报告:")
        logger.info(f"   完整性: {metrics.completeness:.3f}")
        logger.info(f"   一致性: {metrics.consistency:.3f}")
        logger.info(f"   准确性: {metrics.accuracy:.3f}")
        logger.info(f"   唯一性: {metrics.uniqueness:.3f}")
        logger.info(f"   有效性: {metrics.validity:.3f}")
        logger.info(f"   总体分数: {metrics.overall_score:.3f}")


def main():
    """主函数"""
    logger.info("🚀 启动增强数据质量处理器")

    # 创建处理器
    processor = DataQualityProcessor()

    # 模拟数据点
    sample_data = [
        DataPoint(
            food_name="宫保鸡丁",
            cuisine="川菜",
            nutrition={
                "nf_calories": 300,
                "nf_protein": 25,
                "nf_total_carbohydrate": 15,
                "nf_total_fat": 18,
                "nf_dietary_fiber": 2,
                "nf_sodium": 800
            },
            cultural_features={
                "cuisine_type": "川菜",
                "food_category": "肉类",
                "cooking_method": "炒",
                "flavor_profile": "辣",
                "cultural_significance": 0.9
            },
            source="nutritionix",
            quality_score=0.9,
            timestamp=datetime.now().isoformat()
        )
    ]

    # 处理数据
    processed_data, quality_metrics = processor.process_data_points(sample_data)

    # 打印质量报告
    processor.print_quality_report(quality_metrics)

    # 保存数据
    processor.save_processed_data(processed_data)

    logger.info("✅ 数据质量处理完成")


if __name__ == "__main__":
    main()
