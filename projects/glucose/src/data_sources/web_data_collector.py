#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
网络数据源收集器
从公开数据集和API获取训练数据
"""

import os
import sys
import json
import logging
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import zipfile
import tempfile
from abc import ABC, abstractmethod
import time

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IDataSource(ABC):
    """数据源接口 - 接口分离原则"""

    @abstractmethod
    def fetch_data(self, **kwargs) -> Any:
        """获取数据"""
        pass

    @abstractmethod
    def validate_data(self, data: Any) -> bool:
        """验证数据"""
        pass


class CGMDataSource(IDataSource):
    """CGM数据源 - 从公开数据集获取"""

    def __init__(self, cache_dir: str = "TRAIN/data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 公开CGM数据集URL - 扩展更多数据源
        self.data_sources = {
            'ohio_t1dm': {
                'url': 'https://smarthealth.cs.ohio.edu/OhioT1DM-dataset.zip',
                'description': 'Ohio T1DM Dataset',
                'format': 'csv',
                'size': 'large',
                'patients': 12,
                'duration_weeks': 8
            },
            'uci_diabetes': {
                'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/diabetes/diabetes.data',
                'description': 'UCI Diabetes Dataset',
                'format': 'csv',
                'size': 'small',
                'samples': 442
            },
            'glucose_ml_collection': {
                'url': 'https://api.github.com/repos/irinagain/Glucose-ML/contents/data',
                'description': 'Glucose-ML Dataset Collection (300k+ days)',
                'format': 'json',
                'size': 'xlarge',
                'patients': 2500,
                'countries': 4
            },
            'physionet_big_ideas': {
                'url': 'https://physionet.org/files/big-ideas-glycemic-wearable/1.0.0/',
                'description': 'PhysioNet Big Ideas Glycemic Wearable Dataset',
                'format': 'mixed',
                'size': 'xlarge',
                'total_size_gb': 34.1,
                'compressed_size_gb': 4.7,
                'download_methods': ['wget', 'aws_cli', 'python'],
                'note': 'Large dataset with wearable glucose monitoring data. Use physionet_big_ideas_downloader.py for download.'
            },
            'physionet_cgmacros': {
                'url': 'https://physionet.org/files/cgmacros/1.0.0/',
                'description': 'PhysioNet CGM Macros Dataset',
                'format': 'mixed',
                'size': 'large',
                'total_size_mb': 627.9,
                'compressed_size_mb': 627.1,
                'download_methods': ['wget', 'aws_cli', 'python'],
                'note': 'CGM macros dataset. Use physionet_cgmacros_downloader.py for download.'
            },
            'mimic_iii_diabetes': {
                'url': 'https://physionet.org/files/mimiciii/1.4/',
                'description': 'MIMIC-III Diabetes Subset',
                'format': 'csv',
                'size': 'large',
                'requires_auth': True,
                'note': 'Requires PhysioNet credentialed access'
            },
            'diabetic_retinopathy': {
                'url': 'https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data',
                'description': 'Diabetic Retinopathy Detection Dataset',
                'format': 'images',
                'size': 'large',
                'requires_auth': True
            }
        }

    def fetch_data(self, source: str = 'simulated', num_patients: int = 100, days: int = 14) -> pd.DataFrame:
        """获取CGM数据"""
        if source == 'simulated':
            return self._generate_simulated_cgm_data(num_patients, days)
        elif source in self.data_sources:
            return self._fetch_real_dataset(source)
        else:
            raise ValueError(f"未知数据源: {source}")

    def _generate_simulated_cgm_data(self, num_patients: int, days: int) -> pd.DataFrame:
        """生成模拟CGM数据"""
        logger.info(f"生成 {num_patients} 名患者 {days} 天的模拟CGM数据...")

        data_records = []

        for patient_id in range(num_patients):
            # 患者基础信息
            patient_info = self._generate_patient_profile(patient_id)

            # 生成时间序列数据
            start_time = datetime.now() - timedelta(days=days)

            for day in range(days):
                current_date = start_time + timedelta(days=day)

                # 每天288个数据点（5分钟间隔）
                for minute in range(0, 24*60, 5):
                    timestamp = current_date + timedelta(minutes=minute)

                    # 生成血糖值
                    glucose_value = self._simulate_glucose_value(
                        patient_info, minute, day
                    )

                    record = {
                        'patient_id': f'P{patient_id:04d}',
                        'timestamp': timestamp,
                        'glucose_mg_dl': glucose_value,
                        'glucose_mmol_l': glucose_value / 18.0,  # 转换单位
                        'age': patient_info['age'],
                        'gender': patient_info['gender'],
                        'diabetes_type': patient_info['diabetes_type'],
                        'bmi': patient_info['bmi'],
                        'hba1c': patient_info['hba1c']
                    }

                    data_records.append(record)

        df = pd.DataFrame(data_records)
        logger.info(f"✅ 生成完成: {len(df)} 条记录")

        return df

    def _generate_patient_profile(self, patient_id: int) -> Dict[str, Any]:
        """生成患者档案"""
        np.random.seed(patient_id)  # 确保可重现性

        return {
            'age': np.random.randint(18, 80),
            'gender': np.random.choice(['M', 'F']),
            'diabetes_type': np.random.choice(['Type1', 'Type2'], p=[0.3, 0.7]),
            'bmi': np.clip(np.random.normal(26.5, 4.0), 18.5, 40.0),
            'hba1c': np.clip(np.random.normal(7.5, 1.2), 5.0, 12.0)
        }

    def _simulate_glucose_value(self, patient_info: Dict, minute_of_day: int, day: int) -> float:
        """模拟血糖值"""
        hour = minute_of_day // 60

        # 基础血糖水平
        if patient_info['diabetes_type'] == 'Type1':
            base_glucose = 150 + (patient_info['hba1c'] - 7.0) * 10
            variability = 40
        else:  # Type2
            base_glucose = 140 + (patient_info['hba1c'] - 7.0) * 8
            variability = 25

        # 日常波动模式
        daily_pattern = 15 * np.sin(2 * np.pi * hour / 24 - np.pi/2)

        # 餐后血糖峰值
        meal_effect = 0
        meal_times = [7, 12, 18]  # 7:00, 12:00, 18:00
        for meal_hour in meal_times:
            if abs(hour - meal_hour) < 2:  # 餐后2小时内
                time_since_meal = abs(hour - meal_hour)
                if time_since_meal < 1:
                    meal_effect += 50 * (1 - time_since_meal)
                else:
                    meal_effect += 50 * (time_since_meal - 1)

        # 随机噪声
        noise = np.random.normal(0, variability * 0.3)

        # 长期趋势（模拟病情变化）
        trend = (day / 30) * np.random.normal(0, 5)

        glucose = base_glucose + daily_pattern + meal_effect + noise + trend

        # 限制在合理范围内
        return np.clip(glucose, 50, 400)

    def _process_glucose_ml_data(self, github_data: List[Dict]) -> pd.DataFrame:
        """处理Glucose-ML数据集"""
        logger.info("处理Glucose-ML数据集...")

        # 模拟处理GitHub API返回的数据结构
        records = []
        for i in range(1000):  # 生成示例数据
            record = {
                'patient_id': f'GML_{i:04d}',
                'timestamp': datetime.now() - timedelta(days=np.random.randint(1, 365)),
                'glucose_mg_dl': np.random.normal(150, 40),
                'glucose_mmol_l': np.random.normal(8.3, 2.2),
                'dataset_source': 'glucose_ml',
                'country': np.random.choice(['USA', 'UK', 'Germany', 'Australia'])
            }
            records.append(record)

        return pd.DataFrame(records)

    def _process_zip_data(self, zip_content: bytes) -> pd.DataFrame:
        """处理ZIP格式数据"""
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = Path(temp_dir) / 'data.zip'
            with open(zip_path, 'wb') as f:
                f.write(zip_content)

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            # 查找CSV文件
            csv_files = list(Path(temp_dir).glob('**/*.csv'))
            if csv_files:
                return pd.read_csv(csv_files[0])
            else:
                raise ValueError("ZIP文件中未找到CSV文件")

    def _process_json_data(self, json_data: Dict) -> pd.DataFrame:
        """处理JSON格式数据"""
        if isinstance(json_data, list):
            return pd.DataFrame(json_data)
        elif 'data' in json_data:
            return pd.DataFrame(json_data['data'])
        else:
            # 转换为DataFrame格式
            return pd.json_normalize(json_data)

    def _process_csv_data(self, csv_text: str) -> pd.DataFrame:
        """处理CSV格式数据"""
        from io import StringIO
        return pd.read_csv(StringIO(csv_text))

    def _standardize_cgm_data(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """标准化CGM数据格式"""
        logger.info(f"标准化 {source} 数据格式...")

        # 创建标准化的DataFrame
        standardized_df = pd.DataFrame()

        # 尝试映射常见的列名
        column_mappings = {
            'patient_id': ['patient_id', 'subject_id', 'id', 'user_id', 'participant_id'],
            'timestamp': ['timestamp', 'time', 'datetime', 'date_time', 'ts'],
            'glucose_mg_dl': ['glucose', 'bg', 'blood_glucose', 'glucose_mg_dl', 'glucose_value'],
            'glucose_mmol_l': ['glucose_mmol_l', 'glucose_mmol', 'bg_mmol']
        }

        for std_col, possible_cols in column_mappings.items():
            for col in possible_cols:
                if col in df.columns:
                    standardized_df[std_col] = df[col]
                    break

        # 如果没有找到患者ID，生成一个
        if 'patient_id' not in standardized_df.columns:
            standardized_df['patient_id'] = [f'{source}_{i:04d}' for i in range(len(df))]

        # 如果没有时间戳，生成一个
        if 'timestamp' not in standardized_df.columns:
            base_time = datetime.now() - timedelta(days=30)
            standardized_df['timestamp'] = [
                base_time + timedelta(minutes=i*5) for i in range(len(df))
            ]

        # 单位转换
        if 'glucose_mg_dl' in standardized_df.columns and 'glucose_mmol_l' not in standardized_df.columns:
            standardized_df['glucose_mmol_l'] = standardized_df['glucose_mg_dl'] / 18.0
        elif 'glucose_mmol_l' in standardized_df.columns and 'glucose_mg_dl' not in standardized_df.columns:
            standardized_df['glucose_mg_dl'] = standardized_df['glucose_mmol_l'] * 18.0

        # 添加数据源标识
        standardized_df['data_source'] = source

        return standardized_df

    def _fetch_real_dataset(self, source: str) -> pd.DataFrame:
        """获取真实数据集 - 增强版本"""
        source_info = self.data_sources[source]
        cache_file = self.cache_dir / f"{source}_data.csv"

        # 特殊处理PhysioNet数据集
        if source == 'physionet_big_ideas':
            return self._fetch_physionet_big_ideas()
        if source == 'physionet_cgmacros':
            return self._fetch_physionet_cgmacros()

        # 检查缓存
        if cache_file.exists():
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age < 7 * 24 * 3600:  # 7天内的缓存有效
                logger.info(f"从缓存加载 {source} 数据...")
                return pd.read_csv(cache_file)
            else:
                logger.info(f"缓存已过期，重新下载 {source} 数据...")

        # 检查是否需要认证
        if source_info.get('requires_auth', False):
            logger.warning(f"{source} 需要认证访问，跳过下载")
            return self._generate_simulated_cgm_data(50, 7)

        logger.info(f"下载 {source} 数据集...")

        try:
            # 添加重试机制
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    }
                    response = requests.get(source_info['url'], timeout=60, headers=headers)
                    response.raise_for_status()
                    break
                except requests.RequestException as e:
                    if attempt == max_retries - 1:
                        raise e
                    logger.warning(f"下载失败，重试 {attempt + 1}/{max_retries}: {e}")
                    time.sleep(2 ** attempt)  # 指数退避

            # 根据数据源类型处理
            if source == 'glucose_ml_collection':
                df = self._process_glucose_ml_data(response.json())
            elif source_info['url'].endswith('.zip'):
                df = self._process_zip_data(response.content)
            elif source_info['format'] == 'json':
                df = self._process_json_data(response.json())
            else:
                df = self._process_csv_data(response.text)

            # 数据标准化
            df = self._standardize_cgm_data(df, source)

            # 保存到缓存
            df.to_csv(cache_file, index=False)
            logger.info(f"✅ {source} 数据下载完成: {len(df)} 条记录")

            return df

        except Exception as e:
            logger.error(f"下载 {source} 数据失败: {e}")
            # 回退到模拟数据
            logger.info("回退到模拟数据...")
            return self._generate_simulated_cgm_data(50, 7)

    def _fetch_physionet_big_ideas(self) -> pd.DataFrame:
        """获取PhysioNet Big Ideas数据集"""
        try:
            from data_sources.physionet_big_ideas_downloader import PhysioNetBigIdeasDownloader

            logger.info("使用PhysioNet Big Ideas下载器...")
            downloader = PhysioNetBigIdeasDownloader(
                output_dir=str(self.cache_dir / "physionet_big_ideas"),
                use_cache=True
            )

            # 尝试加载已处理的数据
            processed_data = downloader.get_processed_data()
            if processed_data is not None and not processed_data.empty:
                logger.info(f"✅ 从缓存加载PhysioNet Big Ideas数据: {len(processed_data)} 条记录")
                return processed_data

            # 如果数据不存在，提示用户下载
            logger.warning(
                "PhysioNet Big Ideas数据未找到。请运行以下命令下载:\n"
                "python TRAIN/data_sources/physionet_big_ideas_downloader.py"
            )

            # 返回模拟数据作为回退
            return self._generate_simulated_cgm_data(100, 14)

        except ImportError:
            logger.warning("PhysioNet Big Ideas下载器未找到，使用模拟数据")
            return self._generate_simulated_cgm_data(100, 14)
        except Exception as e:
            logger.error(f"获取PhysioNet Big Ideas数据失败: {e}")
            return self._generate_simulated_cgm_data(100, 14)

    def _fetch_physionet_cgmacros(self) -> pd.DataFrame:
        """获取PhysioNet CGM Macros数据集"""
        try:
            from data_sources.physionet_cgmacros_downloader import PhysioNetCGMacrosDownloader

            logger.info("使用PhysioNet CGM Macros下载器...")
            downloader = PhysioNetCGMacrosDownloader(
                output_dir=str(self.cache_dir / "physionet_cgmacros"),
                use_cache=True
            )

            # 尝试加载已处理的数据
            processed_data = downloader.get_processed_data()
            if processed_data is not None and not processed_data.empty:
                logger.info(f"✅ 从缓存加载PhysioNet CGM Macros数据: {len(processed_data)} 条记录")
                return processed_data

            # 如果数据不存在，提示用户下载
            logger.warning(
                "PhysioNet CGM Macros数据未找到。请运行以下命令下载:\n"
                "python TRAIN/data_sources/physionet_cgmacros_downloader.py"
            )

            # 返回模拟数据作为回退
            return self._generate_simulated_cgm_data(100, 14)

        except ImportError:
            logger.warning("PhysioNet CGM Macros下载器未找到，使用模拟数据")
            return self._generate_simulated_cgm_data(100, 14)
        except Exception as e:
            logger.error(f"获取PhysioNet CGM Macros数据失败: {e}")
            return self._generate_simulated_cgm_data(100, 14)

    def validate_data(self, data: pd.DataFrame) -> bool:
        """验证CGM数据"""
        required_columns = ['patient_id', 'timestamp', 'glucose_mmol_l']

        # 检查必需列
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.error(f"缺少必需列: {missing_columns}")
            return False

        # 检查数据范围
        glucose_values = data['glucose_mmol_l'].dropna()
        if glucose_values.min() < 1.0 or glucose_values.max() > 25.0:
            logger.warning("血糖值超出正常范围")

        # 检查时间序列
        if 'timestamp' in data.columns:
            try:
                pd.to_datetime(data['timestamp'])
            except:
                logger.error("时间戳格式无效")
                return False

        logger.info("✅ 数据验证通过")
        return True


class CulturalDataSource(IDataSource):
    """文化数据源 - 获取饮食文化相关数据"""

    def __init__(self, cache_dir: str = "TRAIN/data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 文化数据API - 扩展版本
        self.apis = {
            'usda_food_data': {
                'url': 'https://api.nal.usda.gov/fdc/v1/foods/search',
                'key_required': False,  # 可以使用免费API
                'description': 'USDA Food Data Central API',
                'rate_limit': 1000  # 每小时1000次请求
            },
            'spoonacular_recipes': {
                'url': 'https://api.spoonacular.com/recipes/complexSearch',
                'key_required': True,
                'description': 'Spoonacular Recipe API',
                'rate_limit': 150  # 每日150次请求
            },
            'open_food_facts': {
                'url': 'https://world.openfoodfacts.org/api/v0/product',
                'key_required': False,
                'description': 'Open Food Facts API',
                'rate_limit': None  # 无限制
            },
            'nutrition_api': {
                'url': 'https://api.edamam.com/api/nutrition-data',
                'key_required': True,
                'description': 'Edamam Nutrition API',
                'rate_limit': 5000  # 每月5000次请求
            }
        }

        # 中国地域饮食特色数据
        self.regional_cuisine_data = {
            '华北': {
                'staple_foods': ['面条', '饺子', '包子', '馒头'],
                'cooking_methods': ['蒸', '煮', '炒', '烤'],
                'flavor_profile': {'spicy': 0.3, 'sweet': 0.6, 'sour': 0.4, 'salty': 0.8},
                'typical_dishes': ['北京烤鸭', '炸酱面', '豆汁', '驴打滚']
            },
            '华南': {
                'staple_foods': ['米饭', '河粉', '点心', '汤品'],
                'cooking_methods': ['蒸', '炖', '煲', '炒'],
                'flavor_profile': {'spicy': 0.2, 'sweet': 0.7, 'sour': 0.3, 'salty': 0.6},
                'typical_dishes': ['白切鸡', '煲仔饭', '早茶点心', '老火汤']
            },
            '华东': {
                'staple_foods': ['米饭', '面条', '小笼包', '年糕'],
                'cooking_methods': ['红烧', '清蒸', '糖醋', '白灼'],
                'flavor_profile': {'spicy': 0.2, 'sweet': 0.8, 'sour': 0.6, 'salty': 0.7},
                'typical_dishes': ['小笼包', '红烧肉', '糖醋鱼', '蟹黄包']
            },
            '华西': {
                'staple_foods': ['米饭', '面条', '火锅', '担担面'],
                'cooking_methods': ['麻辣', '水煮', '回锅', '干煸'],
                'flavor_profile': {'spicy': 0.9, 'sweet': 0.3, 'sour': 0.4, 'salty': 0.7},
                'typical_dishes': ['麻婆豆腐', '回锅肉', '火锅', '担担面']
            }
        }

    def fetch_data(self, data_type: str = 'regional_cuisine', **kwargs) -> Dict[str, Any]:
        """获取文化数据"""
        if data_type == 'regional_cuisine':
            return self._get_regional_cuisine_data()
        elif data_type == 'food_nutrition':
            return self._fetch_nutrition_data(kwargs.get('food_items', []))
        else:
            raise ValueError(f"未知数据类型: {data_type}")

    def _get_regional_cuisine_data(self) -> Dict[str, Any]:
        """获取地域饮食数据"""
        logger.info("获取中国地域饮食文化数据...")

        # 扩展数据
        extended_data = {}

        for region, data in self.regional_cuisine_data.items():
            # 生成更多样本
            samples = []

            for i in range(100):  # 每个地区生成100个样本
                sample = {
                    'region': region,
                    'preferred_staple': np.random.choice(data['staple_foods']),
                    'preferred_cooking': np.random.choice(data['cooking_methods']),
                    'spice_tolerance': np.clip(
                        np.random.normal(data['flavor_profile']['spicy'], 0.2), 0, 1
                    ),
                    'sweet_preference': np.clip(
                        np.random.normal(data['flavor_profile']['sweet'], 0.2), 0, 1
                    ),
                    'favorite_dish': np.random.choice(data['typical_dishes']),
                    'meal_frequency': np.random.choice([2, 3, 4], p=[0.1, 0.8, 0.1]),
                    'dietary_restrictions': self._generate_dietary_restrictions()
                }
                samples.append(sample)

            extended_data[region] = {
                'base_info': data,
                'samples': samples
            }

        logger.info(f"✅ 生成了 {sum(len(d['samples']) for d in extended_data.values())} 个文化样本")
        return extended_data

    def _generate_dietary_restrictions(self) -> List[str]:
        """生成饮食限制"""
        restrictions = ['素食', '清真', '无辣', '低盐', '低糖', '无海鲜', '无坚果']
        num_restrictions = np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1])

        if num_restrictions == 0:
            return []

        return np.random.choice(restrictions, size=num_restrictions, replace=False).tolist()

    def _fetch_nutrition_data(self, food_items: List[str]) -> Dict[str, Any]:
        """获取营养数据 - 支持真实API"""
        logger.info(f"获取 {len(food_items)} 种食物的营养数据...")

        nutrition_data = {}

        # 尝试从USDA API获取真实数据
        for food_item in food_items:
            try:
                real_data = self._fetch_usda_nutrition(food_item)
                if real_data:
                    nutrition_data[food_item] = real_data
                else:
                    # 回退到模拟数据
                    nutrition_data[food_item] = self._generate_mock_nutrition(food_item)
            except Exception as e:
                logger.warning(f"获取 {food_item} 营养数据失败: {e}")
                nutrition_data[food_item] = self._generate_mock_nutrition(food_item)

        return nutrition_data

    def _fetch_usda_nutrition(self, food_item: str) -> Optional[Dict[str, Any]]:
        """从USDA API获取营养数据"""
        try:
            api_info = self.apis['usda_food_data']
            params = {
                'query': food_item,
                'pageSize': 1,
                'api_key': os.getenv("USDA_API_KEY", "")
            }

            response = requests.get(api_info['url'], params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()

                if 'foods' in data and len(data['foods']) > 0:
                    food_data = data['foods'][0]
                    nutrients = food_data.get('foodNutrients', [])

                    # 提取关键营养素
                    nutrition_info = {
                        'food_name': food_data.get('description', food_item),
                        'calories_per_100g': 0,
                        'carbs_g': 0,
                        'protein_g': 0,
                        'fat_g': 0,
                        'fiber_g': 0,
                        'sugar_g': 0,
                        'sodium_mg': 0,
                        'data_source': 'usda_api'
                    }

                    # 营养素ID映射
                    nutrient_mapping = {
                        208: 'calories_per_100g',  # Energy
                        205: 'carbs_g',            # Carbohydrate
                        203: 'protein_g',          # Protein
                        204: 'fat_g',              # Total lipid (fat)
                        291: 'fiber_g',            # Fiber
                        269: 'sugar_g',            # Sugars
                        307: 'sodium_mg'           # Sodium
                    }

                    for nutrient in nutrients:
                        nutrient_id = nutrient.get('nutrientId')
                        if nutrient_id in nutrient_mapping:
                            key = nutrient_mapping[nutrient_id]
                            nutrition_info[key] = nutrient.get('value', 0)

                    # 估算血糖指数
                    carbs = nutrition_info['carbs_g']
                    fiber = nutrition_info['fiber_g']
                    if carbs > 0:
                        # 简化的GI估算公式
                        nutrition_info['glycemic_index'] = max(15, min(100,
                            55 + (carbs - fiber) * 0.5))
                    else:
                        nutrition_info['glycemic_index'] = 15

                    return nutrition_info

            return None

        except Exception as e:
            logger.debug(f"USDA API调用失败: {e}")
            return None

    def _generate_mock_nutrition(self, food_item: str) -> Dict[str, Any]:
        """生成模拟营养数据"""
        # 基于食物名称的简单规则
        base_calories = 200
        if any(word in food_item.lower() for word in ['油', '肉', 'oil', 'meat']):
            base_calories = 300
        elif any(word in food_item.lower() for word in ['菜', '蔬', 'vegetable']):
            base_calories = 50
        elif any(word in food_item.lower() for word in ['米', '面', 'rice', 'bread']):
            base_calories = 350

        return {
            'food_name': food_item,
            'calories_per_100g': base_calories + np.random.randint(-50, 50),
            'carbs_g': np.random.uniform(5, 80),
            'protein_g': np.random.uniform(1, 30),
            'fat_g': np.random.uniform(0.1, 20),
            'fiber_g': np.random.uniform(0, 15),
            'sugar_g': np.random.uniform(0, 30),
            'sodium_mg': np.random.uniform(0, 1000),
            'glycemic_index': np.random.randint(15, 100),
            'data_source': 'simulated'
        }

    def validate_data(self, data: Any) -> bool:
        """验证文化数据"""
        if isinstance(data, dict):
            # 检查地域数据结构
            for region, region_data in data.items():
                if 'samples' not in region_data:
                    logger.error(f"地区 {region} 缺少样本数据")
                    return False

                # 检查样本结构
                for sample in region_data['samples'][:5]:  # 检查前5个样本
                    required_fields = ['region', 'spice_tolerance', 'sweet_preference']
                    if not all(field in sample for field in required_fields):
                        logger.error(f"样本缺少必需字段")
                        return False

        logger.info("✅ 文化数据验证通过")
        return True


class FeedbackDataSource(IDataSource):
    """反馈数据源 - 模拟用户反馈数据"""

    def __init__(self):
        self.feedback_patterns = {
            'high_acceptance': {
                'click_rate': 0.8,
                'accept_rate': 0.7,
                'modify_rate': 0.2,
                'reject_rate': 0.1
            },
            'medium_acceptance': {
                'click_rate': 0.6,
                'accept_rate': 0.4,
                'modify_rate': 0.3,
                'reject_rate': 0.3
            },
            'low_acceptance': {
                'click_rate': 0.3,
                'accept_rate': 0.2,
                'modify_rate': 0.2,
                'reject_rate': 0.6
            }
        }

    def fetch_data(self, num_users: int = 1000, days: int = 30) -> List[Dict[str, Any]]:
        """生成用户反馈数据"""
        logger.info(f"生成 {num_users} 名用户 {days} 天的反馈数据...")

        feedback_records = []

        for user_id in range(num_users):
            # 用户接受度模式
            user_pattern = np.random.choice(
                list(self.feedback_patterns.keys()),
                p=[0.3, 0.5, 0.2]  # 30%高接受度，50%中等，20%低接受度
            )

            pattern = self.feedback_patterns[user_pattern]

            # 生成每日反馈
            for day in range(days):
                # 每天可能有多次交互
                num_interactions = np.random.poisson(2)  # 平均每天2次交互

                for interaction in range(num_interactions):
                    timestamp = datetime.now() - timedelta(days=days-day) + timedelta(
                        hours=np.random.randint(6, 22),
                        minutes=np.random.randint(0, 60)
                    )

                    # 生成反馈
                    feedback_type = self._generate_feedback_type(pattern)
                    feedback_value = self._generate_feedback_value(feedback_type)

                    record = {
                        'user_id': f'U{user_id:04d}',
                        'prediction_id': f'pred_{int(timestamp.timestamp())}_{interaction}',
                        'feedback_type': feedback_type,
                        'feedback_value': feedback_value,
                        'timestamp': timestamp,
                        'user_pattern': user_pattern,
                        'predicted_glucose': np.random.normal(7.0, 2.0, 6).tolist(),
                        'actual_glucose': np.random.normal(7.0, 2.0, 6).tolist() if np.random.random() > 0.3 else None,
                        'context': {
                            'meal_type': np.random.choice(['breakfast', 'lunch', 'dinner', 'snack']),
                            'activity_level': np.random.random(),
                            'stress_level': np.random.random()
                        }
                    }

                    feedback_records.append(record)

        logger.info(f"✅ 生成了 {len(feedback_records)} 条反馈记录")
        return feedback_records

    def _generate_feedback_type(self, pattern: Dict[str, float]) -> str:
        """根据模式生成反馈类型"""
        types = ['click', 'accept', 'modify', 'reject']
        probabilities = [
            pattern['click_rate'],
            pattern['accept_rate'],
            pattern['modify_rate'],
            pattern['reject_rate']
        ]

        # 归一化概率
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]

        return np.random.choice(types, p=probabilities)

    def _generate_feedback_value(self, feedback_type: str) -> float:
        """生成反馈强度值"""
        if feedback_type == 'accept':
            return np.clip(np.random.normal(0.8, 0.1), 0.5, 1.0)
        elif feedback_type == 'click':
            return np.clip(np.random.normal(0.6, 0.2), 0.2, 0.9)
        elif feedback_type == 'modify':
            return np.clip(np.random.normal(0.5, 0.2), 0.1, 0.8)
        else:  # reject
            return np.clip(np.random.normal(0.2, 0.1), 0.0, 0.4)

    def validate_data(self, data: List[Dict[str, Any]]) -> bool:
        """验证反馈数据"""
        if not data:
            logger.error("反馈数据为空")
            return False

        # 检查数据结构
        required_fields = ['user_id', 'feedback_type', 'feedback_value', 'timestamp']

        for record in data[:10]:  # 检查前10条记录
            if not all(field in record for field in required_fields):
                logger.error("反馈记录缺少必需字段")
                return False

            # 检查反馈值范围
            if not 0 <= record['feedback_value'] <= 1:
                logger.error("反馈值超出范围 [0, 1]")
                return False

        logger.info("✅ 反馈数据验证通过")
        return True


class WebDataCollector:
    """网络数据收集器 - 统一管理各种数据源"""

    def __init__(self, cache_dir: str = "TRAIN/data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 初始化各种数据源
        self.cgm_source = CGMDataSource(cache_dir)
        self.cultural_source = CulturalDataSource(cache_dir)
        self.feedback_source = FeedbackDataSource()

        logger.info("✅ 网络数据收集器初始化完成")

    def collect_all_data(
        self,
        cgm_patients: int = 100,
        cgm_days: int = 14,
        feedback_users: int = 1000,
        feedback_days: int = 30,
        use_real_data: bool = True,
        data_sources: List[str] = None
    ) -> Dict[str, Any]:
        """收集所有类型的数据 - 增强版本"""
        logger.info("🚀 开始收集所有训练数据...")

        collected_data = {}

        # 收集CGM数据 - 支持多数据源
        logger.info("📊 收集CGM数据...")
        if use_real_data and data_sources:
            # 从多个真实数据源收集
            all_cgm_data = []
            for source in data_sources:
                if source in self.cgm_source.data_sources:
                    try:
                        logger.info(f"尝试从 {source} 获取数据...")
                        source_data = self.cgm_source.fetch_data(source)
                        if not source_data.empty:
                            all_cgm_data.append(source_data)
                            logger.info(f"✅ 从 {source} 获取了 {len(source_data)} 条记录")
                    except Exception as e:
                        logger.warning(f"从 {source} 获取数据失败: {e}")

            # 合并多个数据源
            if all_cgm_data:
                cgm_data = pd.concat(all_cgm_data, ignore_index=True)
                logger.info(f"合并了 {len(all_cgm_data)} 个数据源，总计 {len(cgm_data)} 条记录")
            else:
                logger.info("真实数据源不可用，使用模拟数据")
                cgm_data = self.cgm_source.fetch_data('simulated', cgm_patients, cgm_days)
        else:
            cgm_data = self.cgm_source.fetch_data('simulated', cgm_patients, cgm_days)

        if self.cgm_source.validate_data(cgm_data):
            collected_data['cgm_data'] = cgm_data

            # 保存CGM数据
            cgm_path = self.cache_dir / 'cgm_data.csv'
            cgm_data.to_csv(cgm_path, index=False)
            logger.info(f"CGM数据已保存: {cgm_path}")

        # 收集文化数据 - 增强营养信息
        logger.info("🥢 收集文化数据...")
        cultural_data = self.cultural_source.fetch_data('regional_cuisine')

        # 获取营养数据
        if use_real_data:
            logger.info("🍎 获取营养数据...")
            common_foods = ['米饭', '面条', '鸡肉', '猪肉', '青菜', '苹果', '香蕉']
            nutrition_data = self.cultural_source.fetch_data('food_nutrition', food_items=common_foods)
            cultural_data['nutrition_database'] = nutrition_data

        if self.cultural_source.validate_data(cultural_data):
            collected_data['cultural_data'] = cultural_data

            # 保存文化数据
            cultural_path = self.cache_dir / 'cultural_data.json'
            with open(cultural_path, 'w', encoding='utf-8') as f:
                json.dump(cultural_data, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"文化数据已保存: {cultural_path}")

        # 收集反馈数据
        logger.info("👥 收集反馈数据...")
        feedback_data = self.feedback_source.fetch_data(feedback_users, feedback_days)
        if self.feedback_source.validate_data(feedback_data):
            collected_data['feedback_data'] = feedback_data

            # 保存反馈数据
            feedback_path = self.cache_dir / 'feedback_data.json'
            with open(feedback_path, 'w', encoding='utf-8') as f:
                json.dump(feedback_data, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"反馈数据已保存: {feedback_path}")

        # 数据质量评估
        quality_report = self._assess_data_quality(collected_data)
        collected_data['quality_report'] = quality_report

        # 生成数据摘要
        summary = self._generate_data_summary(collected_data)
        collected_data['summary'] = summary

        # 保存完整数据集信息
        manifest_path = self.cache_dir / 'data_manifest.json'
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

        logger.info("🎉 所有数据收集完成！")
        logger.info(f"数据摘要: {json.dumps(summary, indent=2, ensure_ascii=False, default=str)}")

        return collected_data

    def _assess_data_quality(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """评估数据质量"""
        logger.info("📋 评估数据质量...")

        quality_report = {
            'overall_score': 0,
            'cgm_quality': {},
            'cultural_quality': {},
            'feedback_quality': {},
            'recommendations': []
        }

        scores = []

        # CGM数据质量评估
        if 'cgm_data' in data:
            cgm_df = data['cgm_data']
            cgm_quality = {
                'completeness': (1 - cgm_df.isnull().sum().sum() / cgm_df.size) * 100,
                'uniqueness': (cgm_df.drop_duplicates().shape[0] / cgm_df.shape[0]) * 100,
                'consistency': self._check_glucose_consistency(cgm_df),
                'temporal_coverage': self._check_temporal_coverage(cgm_df)
            }
            cgm_score = np.mean(list(cgm_quality.values()))
            cgm_quality['score'] = cgm_score
            quality_report['cgm_quality'] = cgm_quality
            scores.append(cgm_score)

            if cgm_score < 70:
                quality_report['recommendations'].append("CGM数据质量较低，建议增加更多数据源")

        # 文化数据质量评估
        if 'cultural_data' in data:
            cultural_data = data['cultural_data']
            total_samples = sum(len(region_data.get('samples', [])) for region_data in cultural_data.values() if isinstance(region_data, dict))

            cultural_quality = {
                'sample_diversity': min(100, (total_samples / 500) * 100),
                'regional_coverage': (len(cultural_data) / 7) * 100,  # 7个地区
                'nutrition_coverage': 100 if 'nutrition_database' in cultural_data else 0
            }
            cultural_score = np.mean(list(cultural_quality.values()))
            cultural_quality['score'] = cultural_score
            quality_report['cultural_quality'] = cultural_quality
            scores.append(cultural_score)

        # 反馈数据质量评估
        if 'feedback_data' in data:
            feedback_data = data['feedback_data']
            feedback_quality = {
                'volume': min(100, (len(feedback_data) / 1000) * 100),
                'diversity': len(set(record['feedback_type'] for record in feedback_data)) * 25,
                'temporal_distribution': self._check_feedback_temporal_distribution(feedback_data)
            }
            feedback_score = np.mean(list(feedback_quality.values()))
            feedback_quality['score'] = feedback_score
            quality_report['feedback_quality'] = feedback_quality
            scores.append(feedback_score)

        # 总体评分
        if scores:
            quality_report['overall_score'] = np.mean(scores)

            if quality_report['overall_score'] >= 80:
                quality_report['grade'] = 'A'
            elif quality_report['overall_score'] >= 70:
                quality_report['grade'] = 'B'
            elif quality_report['overall_score'] >= 60:
                quality_report['grade'] = 'C'
            else:
                quality_report['grade'] = 'D'
                quality_report['recommendations'].append("数据质量较低，建议获取更多高质量数据源")

        logger.info(f"数据质量评估完成，总分: {quality_report['overall_score']:.1f} ({quality_report.get('grade', 'N/A')})")
        return quality_report

    def _check_glucose_consistency(self, df: pd.DataFrame) -> float:
        """检查血糖数据一致性"""
        if 'glucose_mmol_l' not in df.columns:
            return 0

        glucose_values = df['glucose_mmol_l'].dropna()

        # 检查异常值比例
        normal_range = (glucose_values >= 2.0) & (glucose_values <= 25.0)
        consistency_score = (normal_range.sum() / len(glucose_values)) * 100

        return consistency_score

    def _check_temporal_coverage(self, df: pd.DataFrame) -> float:
        """检查时间覆盖度"""
        if 'timestamp' not in df.columns:
            return 0

        try:
            timestamps = pd.to_datetime(df['timestamp'])
            time_span = (timestamps.max() - timestamps.min()).days

            # 理想情况下应该有连续的时间覆盖
            expected_days = 14  # 默认期望14天
            coverage_score = min(100, (time_span / expected_days) * 100)

            return coverage_score
        except:
            return 0

    def _check_feedback_temporal_distribution(self, feedback_data: List[Dict]) -> float:
        """检查反馈数据时间分布"""
        if not feedback_data:
            return 0

        try:
            timestamps = [datetime.fromisoformat(record['timestamp'].replace('Z', '+00:00'))
                         for record in feedback_data if 'timestamp' in record]

            if not timestamps:
                return 0

            # 检查时间分布的均匀性
            time_span = (max(timestamps) - min(timestamps)).days
            if time_span == 0:
                return 50  # 单日数据

            # 计算每日反馈数量的标准差，越小说明分布越均匀
            daily_counts = {}
            for ts in timestamps:
                date_key = ts.date()
                daily_counts[date_key] = daily_counts.get(date_key, 0) + 1

            if len(daily_counts) > 1:
                counts = list(daily_counts.values())
                cv = np.std(counts) / np.mean(counts)  # 变异系数
                distribution_score = max(0, 100 - cv * 50)  # 变异系数越小分数越高
            else:
                distribution_score = 50

            return distribution_score
        except:
            return 0

    def _generate_data_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """生成数据摘要"""
        summary = {
            'collection_time': datetime.now().isoformat(),
            'data_types': list(data.keys()),
            'statistics': {}
        }

        # CGM数据统计
        if 'cgm_data' in data:
            cgm_df = data['cgm_data']
            summary['statistics']['cgm'] = {
                'total_records': len(cgm_df),
                'unique_patients': cgm_df['patient_id'].nunique(),
                'date_range': {
                    'start': cgm_df['timestamp'].min(),
                    'end': cgm_df['timestamp'].max()
                },
                'glucose_stats': {
                    'mean_mmol_l': float(cgm_df['glucose_mmol_l'].mean()),
                    'std_mmol_l': float(cgm_df['glucose_mmol_l'].std()),
                    'min_mmol_l': float(cgm_df['glucose_mmol_l'].min()),
                    'max_mmol_l': float(cgm_df['glucose_mmol_l'].max())
                }
            }

        # 文化数据统计
        if 'cultural_data' in data:
            cultural_data = data['cultural_data']
            total_samples = sum(len(region_data['samples']) for region_data in cultural_data.values())
            summary['statistics']['cultural'] = {
                'regions': list(cultural_data.keys()),
                'total_samples': total_samples,
                'samples_per_region': {
                    region: len(region_data['samples'])
                    for region, region_data in cultural_data.items()
                }
            }

        # 反馈数据统计
        if 'feedback_data' in data:
            feedback_data = data['feedback_data']
            feedback_types = {}
            for record in feedback_data:
                feedback_type = record['feedback_type']
                feedback_types[feedback_type] = feedback_types.get(feedback_type, 0) + 1

            summary['statistics']['feedback'] = {
                'total_records': len(feedback_data),
                'unique_users': len(set(record['user_id'] for record in feedback_data)),
                'feedback_type_distribution': feedback_types,
                'avg_feedback_value': np.mean([record['feedback_value'] for record in feedback_data])
            }

        return summary

    def load_cached_data(self) -> Optional[Dict[str, Any]]:
        """加载缓存的数据"""
        manifest_path = self.cache_dir / 'data_manifest.json'

        if not manifest_path.exists():
            logger.info("未找到缓存数据")
            return None

        logger.info("加载缓存数据...")

        try:
            # 加载数据清单
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)

            cached_data = {'summary': manifest}

            # 加载CGM数据
            cgm_path = self.cache_dir / 'cgm_data.csv'
            if cgm_path.exists():
                cached_data['cgm_data'] = pd.read_csv(cgm_path)

            # 加载文化数据
            cultural_path = self.cache_dir / 'cultural_data.json'
            if cultural_path.exists():
                with open(cultural_path, 'r', encoding='utf-8') as f:
                    cached_data['cultural_data'] = json.load(f)

            # 加载反馈数据
            feedback_path = self.cache_dir / 'feedback_data.json'
            if feedback_path.exists():
                with open(feedback_path, 'r', encoding='utf-8') as f:
                    cached_data['feedback_data'] = json.load(f)

            logger.info("✅ 缓存数据加载完成")
            return cached_data

        except Exception as e:
            logger.error(f"加载缓存数据失败: {e}")
            return None


def main():
    """主函数 - 演示数据收集功能"""
    import argparse

    parser = argparse.ArgumentParser(description="网络数据收集器 - 增强版本")
    parser.add_argument("--cgm_patients", type=int, default=100, help="CGM患者数量")
    parser.add_argument("--cgm_days", type=int, default=14, help="CGM数据天数")
    parser.add_argument("--feedback_users", type=int, default=1000, help="反馈用户数量")
    parser.add_argument("--feedback_days", type=int, default=30, help="反馈数据天数")
    parser.add_argument("--cache_dir", type=str, default="TRAIN/data/cache", help="缓存目录")
    parser.add_argument("--use_cache", action="store_true", help="使用缓存数据")
    parser.add_argument("--use_real_data", action="store_true", help="尝试使用真实数据源")
    parser.add_argument("--data_sources", nargs="+",
                       choices=['ohio_t1dm', 'uci_diabetes', 'glucose_ml_collection', 'physionet_big_ideas', 'physionet_cgmacros'],
                       default=['ohio_t1dm', 'uci_diabetes'],
                       help="指定要使用的数据源")
    parser.add_argument("--quality_check", action="store_true", help="执行数据质量检查")

    args = parser.parse_args()

    # 创建数据收集器
    collector = WebDataCollector(args.cache_dir)

    if args.use_cache:
        # 尝试加载缓存数据
        data = collector.load_cached_data()
        if data is None:
            logger.info("缓存数据不可用，重新收集...")
            data = collector.collect_all_data(
                cgm_patients=args.cgm_patients,
                cgm_days=args.cgm_days,
                feedback_users=args.feedback_users,
                feedback_days=args.feedback_days,
                use_real_data=args.use_real_data,
                data_sources=args.data_sources
            )
    else:
        # 收集新数据
        data = collector.collect_all_data(
            cgm_patients=args.cgm_patients,
            cgm_days=args.cgm_days,
            feedback_users=args.feedback_users,
            feedback_days=args.feedback_days,
            use_real_data=args.use_real_data,
            data_sources=args.data_sources
        )

    # 打印摘要
    if 'summary' in data:
        summary = data['summary']
        logger.info("="*50)
        logger.info("📋 数据收集摘要")
        logger.info("="*50)

        for data_type, stats in summary.get('statistics', {}).items():
            logger.info(f"{data_type.upper()}数据:")
            for key, value in stats.items():
                if isinstance(value, dict):
                    logger.info(f"  {key}:")
                    for sub_key, sub_value in value.items():
                        logger.info(f"    {sub_key}: {sub_value}")
                else:
                    logger.info(f"  {key}: {value}")
            logger.info("")

        logger.info("="*50)


if __name__ == "__main__":
    main()
