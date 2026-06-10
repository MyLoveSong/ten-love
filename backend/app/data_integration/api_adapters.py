

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API数据适配器模块
集成多种数据源，支持实时数据流和API接口
"""

import requests
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
import json
import asyncio
import aiohttp
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DataSourceConfig:
    """数据源配置"""
    name: str
    api_url: str
    api_key: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    rate_limit: int = 100  # 每分钟请求数
    timeout: int = 30
    retry_count: int = 3

class BaseDataAdapter(ABC):
    """数据适配器基类"""

    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.session = requests.Session()
        if self.config.headers:
            self.session.headers.update(self.config.headers)
        if self.config.api_key:
            self.session.headers.update({"Authorization": f"Bearer {self.config.api_key}"})

    @abstractmethod
    def fetch_data(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """获取数据"""
        pass

    @abstractmethod
    def transform_data(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
        """数据转换"""
        pass

    def validate_data(self, data: pd.DataFrame) -> bool:
        """数据验证"""
        if data.empty:
            return False

        # 检查必需列
        required_columns = ['timestamp', 'glucose']
        for col in required_columns:
            if col not in data.columns:
                logger.warning(f"缺少必需列: {col}")
                return False

        return True

class CGMApiAdapter(BaseDataAdapter):
    """CGM数据API适配器"""

    def fetch_data(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """获取CGM数据"""
        try:
            # 构建请求参数
            query_params = {
                'start_date': params.get('start_date', (datetime.now() - timedelta(days=7)).isoformat()),
                'end_date': params.get('end_date', datetime.now().isoformat()),
                'granularity': params.get('granularity', '5min'),
                'limit': params.get('limit', 1000)
            }

            response = self.session.get(
                f"{self.config.api_url}/cgm/readings",
                params=query_params,
                timeout=self.config.timeout
            )
            response.raise_for_status()

            return response.json()

        except Exception as e:
            logger.error(f"CGM数据获取失败: {e}")
            return {}

    def transform_data(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
        """转换CGM数据格式"""
        try:
            if not raw_data or 'readings' not in raw_data:
                return pd.DataFrame()

            readings = raw_data['readings']
            df = pd.DataFrame(readings)

            # 标准化列名
            column_mapping = {
                'timestamp': 'timestamp',
                'glucose_value': 'glucose',
                'trend': 'trend',
                'quality': 'quality'
            }

            df = df.rename(columns=column_mapping)

            # 数据类型转换
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['glucose'] = pd.to_numeric(df['glucose'], errors='coerce')

            # 添加衍生特征
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_night'] = (df['hour'] >= 22) | (df['hour'] <= 6)

            return df

        except Exception as e:
            logger.error(f"CGM数据转换失败: {e}")
            return pd.DataFrame()

class NutritionApiAdapter(BaseDataAdapter):
    """营养数据API适配器"""

    def fetch_data(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """获取营养数据"""
        try:
            query_params = {
                'date': params.get('date', datetime.now().date().isoformat()),
                'user_id': params.get('user_id'),
                'include_nutrients': True
            }

            response = self.session.get(
                f"{self.config.api_url}/nutrition/logs",
                params=query_params,
                timeout=self.config.timeout
            )
            response.raise_for_status()

            return response.json()

        except Exception as e:
            logger.error(f"营养数据获取失败: {e}")
            return {}

    def transform_data(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
        """转换营养数据格式"""
        try:
            if not raw_data or 'meals' not in raw_data:
                return pd.DataFrame()

            meals = raw_data['meals']
            df = pd.DataFrame(meals)

            # 标准化列名
            column_mapping = {
                'timestamp': 'timestamp',
                'carbs': 'carbohydrates',
                'protein': 'protein',
                'fat': 'fat',
                'calories': 'calories',
                'meal_type': 'meal_type'
            }

            df = df.rename(columns=column_mapping)

            # 数据类型转换
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            numeric_columns = ['carbohydrates', 'protein', 'fat', 'calories']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

            # 计算餐后血糖影响因子
            df['carb_impact'] = df['carbohydrates'] * 0.8  # 碳水化合物对血糖的影响系数
            df['meal_complexity'] = df['carbohydrates'] + df['protein'] * 0.5 + df['fat'] * 0.3

            return df

        except Exception as e:
            logger.error(f"营养数据转换失败: {e}")
            return pd.DataFrame()

class ActivityApiAdapter(BaseDataAdapter):
    """运动数据API适配器"""

    def fetch_data(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """获取运动数据"""
        try:
            query_params = {
                'start_date': params.get('start_date', (datetime.now() - timedelta(days=7)).isoformat()),
                'end_date': params.get('end_date', datetime.now().isoformat()),
                'activity_types': params.get('activity_types', ['walking', 'running', 'cycling'])
            }

            response = self.session.get(
                f"{self.config.api_url}/activities",
                params=query_params,
                timeout=self.config.timeout
            )
            response.raise_for_status()

            return response.json()

        except Exception as e:
            logger.error(f"运动数据获取失败: {e}")
            return {}

    def transform_data(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
        """转换运动数据格式"""
        try:
            if not raw_data or 'activities' not in raw_data:
                return pd.DataFrame()

            activities = raw_data['activities']
            df = pd.DataFrame(activities)

            # 标准化列名
            column_mapping = {
                'start_time': 'timestamp',
                'activity_type': 'activity_type',
                'duration': 'duration_minutes',
                'calories_burned': 'calories_burned',
                'intensity': 'intensity'
            }

            df = df.rename(columns=column_mapping)

            # 数据类型转换
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['duration_minutes'] = pd.to_numeric(df['duration_minutes'], errors='coerce')
            df['calories_burned'] = pd.to_numeric(df['calories_burned'], errors='coerce')

            # 计算运动对血糖的影响
            df['exercise_impact'] = df['calories_burned'] * 0.1  # 运动对血糖的降低影响
            df['intensity_factor'] = df['intensity'].map({'low': 0.5, 'moderate': 1.0, 'high': 1.5}).fillna(1.0)

            return df

        except Exception as e:
            logger.error(f"运动数据转换失败: {e}")
            return pd.DataFrame()

class DataIntegrationManager:
    """数据集成管理器 - 支持工作流和API集成"""

    def __init__(self):
        self.adapters: Dict[str, BaseDataAdapter] = {}
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self.cache_ttl = 300  # 缓存5分钟
        self.workflow_callbacks: Dict[str, callable] = {}  # 工作流回调函数
        self.real_time_handlers: Dict[str, callable] = {}  # 实时数据处理器

    def register_adapter(self, name: str, adapter: BaseDataAdapter):
        """注册数据适配器"""
        self.adapters[name] = adapter
        logger.info(f"注册数据适配器: {name}")

    def fetch_all_data(self, params: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """获取所有数据源的数据"""
        results = {}

        for name, adapter in self.adapters.items():
            try:
                # 检查缓存
                cache_key = f"{name}_{hash(str(params))}"
                if cache_key in self.data_cache:
                    cache_time, data = self.data_cache[cache_key]
                    if datetime.now().timestamp() - cache_time < self.cache_ttl:
                        results[name] = data
                        continue

                # 获取新数据
                raw_data = adapter.fetch_data(params)
                if raw_data:
                    df = adapter.transform_data(raw_data)
                    if adapter.validate_data(df):
                        results[name] = df
                        # 更新缓存
                        self.data_cache[cache_key] = (datetime.now().timestamp(), df)
                    else:
                        logger.warning(f"数据验证失败: {name}")

            except Exception as e:
                logger.error(f"获取数据失败 {name}: {e}")

        return results

    def merge_data(self, data_sources: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """合并多源数据"""
        try:
            if not data_sources:
                return pd.DataFrame()

            # 以CGM数据为基准
            if 'cgm' in data_sources:
                base_df = data_sources['cgm'].copy()
            else:
                # 如果没有CGM数据，使用第一个数据源
                base_df = list(data_sources.values())[0].copy()

            # 合并其他数据源
            for name, df in data_sources.items():
                if name == 'cgm' or df.empty:
                    continue

                # 基于时间戳合并
                base_df = pd.merge_asof(
                    base_df.sort_values('timestamp'),
                    df.sort_values('timestamp'),
                    on='timestamp',
                    direction='backward',
                    suffixes=('', f'_{name}')
                )

            # 填充缺失值
            numeric_columns = base_df.select_dtypes(include=[np.number]).columns
            base_df[numeric_columns] = base_df[numeric_columns].fillna(method='ffill').fillna(0)

            return base_df

        except Exception as e:
            logger.error(f"数据合并失败: {e}")
            return pd.DataFrame()

    def register_workflow_callback(self, workflow_name: str, callback: callable):
        """注册工作流回调函数"""
        self.workflow_callbacks[workflow_name] = callback
        logger.info(f"注册工作流回调: {workflow_name}")

    def register_real_time_handler(self, data_type: str, handler: callable):
        """注册实时数据处理器"""
        self.real_time_handlers[data_type] = handler
        logger.info(f"注册实时数据处理器: {data_type}")

    async def process_workflow_data(self, workflow_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理工作流数据"""
        try:
            if workflow_name in self.workflow_callbacks:
                return await self.workflow_callbacks[workflow_name](data)
            else:
                logger.warning(f"未找到工作流回调: {workflow_name}")
                return {}
        except Exception as e:
            logger.error(f"工作流数据处理失败: {e}")
            return {}

    def process_real_time_data(self, data_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理实时数据"""
        try:
            if data_type in self.real_time_handlers:
                return self.real_time_handlers[data_type](data)
            else:
                logger.warning(f"未找到实时数据处理器: {data_type}")
                return {}
        except Exception as e:
            logger.error(f"实时数据处理失败: {e}")
            return {}

    def create_feature_matrix(self, merged_data: pd.DataFrame) -> np.ndarray:
        """创建特征矩阵"""
        try:
            if merged_data.empty:
                return np.array([])

            # 选择特征列
            feature_columns = [
                'glucose', 'hour', 'day_of_week', 'is_night',
                'carbohydrates', 'protein', 'fat', 'calories',
                'duration_minutes', 'calories_burned', 'exercise_impact'
            ]

            # 过滤存在的列
            available_columns = [col for col in feature_columns if col in merged_data.columns]
            feature_matrix = merged_data[available_columns].values

            # 处理NaN值
            feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)

            return feature_matrix

        except Exception as e:
            logger.error(f"特征矩阵创建失败: {e}")
            return np.array([])

# 配置示例
def create_default_configs() -> Dict[str, DataSourceConfig]:
    """创建默认配置"""
    configs = {
        'cgm': DataSourceConfig(
            name='CGM',
            api_url='https://api.cgm-provider.com/v1',
            api_key='your_cgm_api_key',
            headers={'Content-Type': 'application/json'}
        ),
        'nutrition': DataSourceConfig(
            name='Nutrition',
            api_url='https://api.nutrition-provider.com/v1',
            api_key='your_nutrition_api_key',
            headers={'Content-Type': 'application/json'}
        ),
        'activity': DataSourceConfig(
            name='Activity',
            api_url='https://api.activity-provider.com/v1',
            api_key='your_activity_api_key',
            headers={'Content-Type': 'application/json'}
        )
    }
    return configs

# 使用示例
def main():
    """使用示例"""
    # 创建数据集成管理器
    manager = DataIntegrationManager()

    # 创建适配器
    configs = create_default_configs()

    cgm_adapter = CGMApiAdapter(configs['cgm'])
    nutrition_adapter = NutritionApiAdapter(configs['nutrition'])
    activity_adapter = ActivityApiAdapter(configs['activity'])

    # 注册适配器
    manager.register_adapter('cgm', cgm_adapter)
    manager.register_adapter('nutrition', nutrition_adapter)
    manager.register_adapter('activity', activity_adapter)

    # 获取数据
    params = {
        'start_date': (datetime.now() - timedelta(days=7)).isoformat(),
        'end_date': datetime.now().isoformat(),
        'user_id': 'user_123'
    }

    data_sources = manager.fetch_all_data(params)
    merged_data = manager.merge_data(data_sources)
    feature_matrix = manager.create_feature_matrix(merged_data)

    print(f"合并数据形状: {merged_data.shape}")
    print(f"特征矩阵形状: {feature_matrix.shape}")

if __name__ == "__main__":
    main()

__all__ = ["'logger'", "'DataSourceConfig'", "'BaseDataAdapter'", "'CGMApiAdapter'", "'NutritionApiAdapter'", "'ActivityApiAdapter'", "'DataIntegrationManager'", "'create_default_configs'", "'main'"]
