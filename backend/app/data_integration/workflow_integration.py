

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工作流集成模块
支持Dify等工作流平台，实现数据流自动化和API无缝集成
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from pathlib import Path
import yaml

from .api_adapters import DataIntegrationManager
from .data_augmentation import DataEnhancementPipeline, AugmentationConfig
from .feedback_optimization import IterativeOptimizationSystem, FeedbackConfig

logger = logging.getLogger(__name__)

@dataclass
class WorkflowConfig:
    """工作流配置"""
    name: str
    endpoint: str
    api_key: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: int = 30
    retry_count: int = 3
    data_format: str = "json"  # json, csv, xml
    trigger_type: str = "api"  # api, webhook, scheduler
    schedule_interval: int = 300  # 秒

@dataclass
class DataFlowConfig:
    """数据流配置"""
    source_type: str  # cgm, nutrition, activity, image
    target_model: str  # glucose_predictor, image_classifier
    preprocessing_steps: List[str] = field(default_factory=list)
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    transformation_config: Dict[str, Any] = field(default_factory=dict)

class WorkflowDataProcessor:
    """工作流数据处理器"""

    def __init__(self, data_manager: DataIntegrationManager):
        self.data_manager = data_manager
        self.processing_pipelines: Dict[str, Callable] = {}

    def register_pipeline(self, data_type: str, pipeline: Callable):
        """注册数据处理流水线"""
        self.processing_pipelines[data_type] = pipeline
        logger.info(f"注册数据处理流水线: {data_type}")

    async def process_cgm_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理CGM数据"""
        try:
            # 数据格式标准化
            readings = data.get('readings', [])
            if not readings:
                return {'error': 'No CGM readings found'}

            # 转换为DataFrame
            df = pd.DataFrame(readings)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['glucose'] = pd.to_numeric(df['glucose'], errors='coerce')

            # 数据质量检查
            valid_readings = df.dropna(subset=['glucose'])
            if len(valid_readings) < len(df) * 0.8:
                logger.warning("CGM数据质量较低，有效读数不足80%")

            # 计算衍生特征
            df['glucose_trend'] = df['glucose'].diff()
            df['glucose_variability'] = df['glucose'].rolling(window=5).std()

            # 异常值检测
            q1 = df['glucose'].quantile(0.25)
            q3 = df['glucose'].quantile(0.75)
            iqr = q3 - q1
            outliers = df[(df['glucose'] < q1 - 1.5 * iqr) | (df['glucose'] > q3 + 1.5 * iqr)]

            if len(outliers) > 0:
                logger.info(f"检测到 {len(outliers)} 个CGM异常值")

            return {
                'processed_data': df.to_dict('records'),
                'statistics': {
                    'total_readings': len(df),
                    'valid_readings': len(valid_readings),
                    'mean_glucose': df['glucose'].mean(),
                    'glucose_std': df['glucose'].std(),
                    'outliers_count': len(outliers)
                },
                'quality_score': len(valid_readings) / len(df)
            }

        except Exception as e:
            logger.error(f"CGM数据处理失败: {e}")
            return {'error': str(e)}

    async def process_nutrition_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理营养数据"""
        try:
            meals = data.get('meals', [])
            if not meals:
                return {'error': 'No meal data found'}

            # 营养成分分析
            total_carbs = sum(meal.get('carbohydrates', 0) for meal in meals)
            total_protein = sum(meal.get('protein', 0) for meal in meals)
            total_fat = sum(meal.get('fat', 0) for meal in meals)
            total_calories = sum(meal.get('calories', 0) for meal in meals)

            # 餐后血糖影响预测
            carb_impact_score = total_carbs * 0.8  # 碳水化合物对血糖的影响系数
            meal_complexity = total_carbs + total_protein * 0.5 + total_fat * 0.3

            # 膳食质量评分
            balanced_ratio = min(total_protein / max(total_carbs, 1), 1.0)
            quality_score = balanced_ratio * 0.6 + min(total_fat / max(total_calories, 1), 0.3) * 0.4

            return {
                'processed_data': meals,
                'nutrition_summary': {
                    'total_carbohydrates': total_carbs,
                    'total_protein': total_protein,
                    'total_fat': total_fat,
                    'total_calories': total_calories,
                    'carb_impact_score': carb_impact_score,
                    'meal_complexity': meal_complexity,
                    'quality_score': quality_score
                },
                'recommendations': self._generate_nutrition_recommendations(
                    total_carbs, total_protein, total_fat, quality_score
                )
            }

        except Exception as e:
            logger.error(f"营养数据处理失败: {e}")
            return {'error': str(e)}

    async def process_activity_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理运动数据"""
        try:
            activities = data.get('activities', [])
            if not activities:
                return {'error': 'No activity data found'}

            # 运动强度分析
            total_duration = sum(activity.get('duration', 0) for activity in activities)
            total_calories = sum(activity.get('calories_burned', 0) for activity in activities)

            # 运动对血糖的影响评估
            exercise_impact = total_calories * 0.1  # 运动对血糖的降低影响
            intensity_factor = np.mean([
                {'low': 0.5, 'moderate': 1.0, 'high': 1.5}.get(
                    activity.get('intensity', 'moderate'), 1.0
                ) for activity in activities
            ])

            # 运动时机分析
            recent_activities = [
                a for a in activities
                if datetime.fromisoformat(a.get('start_time', '2024-01-01')) >
                datetime.now() - timedelta(hours=2)
            ]

            return {
                'processed_data': activities,
                'activity_summary': {
                    'total_duration': total_duration,
                    'total_calories_burned': total_calories,
                    'exercise_impact': exercise_impact,
                    'intensity_factor': intensity_factor,
                    'recent_activities_count': len(recent_activities)
                },
                'recommendations': self._generate_activity_recommendations(
                    total_duration, intensity_factor, len(recent_activities)
                )
            }

        except Exception as e:
            logger.error(f"运动数据处理失败: {e}")
            return {'error': str(e)}

    def _generate_nutrition_recommendations(self, carbs: float, protein: float,
                                          fat: float, quality_score: float) -> List[str]:
        """生成营养建议"""
        recommendations = []

        if carbs > 100:
            recommendations.append("碳水化合物摄入偏高，建议减少精制糖和淀粉类食物")
        if protein < 30:
            recommendations.append("蛋白质摄入不足，建议增加瘦肉、鱼类或豆类")
        if quality_score < 0.5:
            recommendations.append("膳食营养不够均衡，建议调整各营养素比例")
        if carbs / max(protein, 1) > 4:
            recommendations.append("碳水化合物与蛋白质比例失衡，建议增加蛋白质摄入")

        return recommendations

    def _generate_activity_recommendations(self, duration: float, intensity: float,
                                         recent_count: int) -> List[str]:
        """生成运动建议"""
        recommendations = []

        if duration < 30:
            recommendations.append("运动时间不足，建议每天至少30分钟中等强度运动")
        if intensity < 0.8:
            recommendations.append("运动强度偏低，建议适当增加运动强度")
        if recent_count == 0:
            recommendations.append("建议餐后1-2小时进行适量运动以控制血糖")

        return recommendations

class WorkflowOrchestrator:
    """工作流编排器"""

    def __init__(self):
        self.workflows: Dict[str, WorkflowConfig] = {}
        self.data_flows: Dict[str, DataFlowConfig] = {}
        self.active_schedules: Dict[str, asyncio.Task] = {}
        self.data_processor = None
        self.enhancement_pipeline = None
        self.optimization_system = None

    def register_workflow(self, config: WorkflowConfig):
        """注册工作流"""
        self.workflows[config.name] = config
        logger.info(f"注册工作流: {config.name}")

        # 如果是定时触发，启动调度器
        if config.trigger_type == "scheduler":
            self._schedule_workflow(config)

    def register_data_flow(self, flow_name: str, config: DataFlowConfig):
        """注册数据流"""
        self.data_flows[flow_name] = config
        logger.info(f"注册数据流: {flow_name}")

    def setup_processors(self, data_manager: DataIntegrationManager,
                        glucose_predictor, image_predictor):
        """设置处理器"""
        self.data_processor = WorkflowDataProcessor(data_manager)

        # 注册处理流水线
        self.data_processor.register_pipeline('cgm', self.data_processor.process_cgm_data)
        self.data_processor.register_pipeline('nutrition', self.data_processor.process_nutrition_data)
        self.data_processor.register_pipeline('activity', self.data_processor.process_activity_data)

        # 设置数据增强流水线
        aug_config = AugmentationConfig()
        self.enhancement_pipeline = DataEnhancementPipeline(aug_config)

        # 设置优化系统
        feedback_config = FeedbackConfig()
        self.optimization_system = IterativeOptimizationSystem(glucose_predictor, feedback_config)

    async def trigger_workflow(self, workflow_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """触发工作流"""
        if workflow_name not in self.workflows:
            return {'error': f'Workflow {workflow_name} not found'}

        config = self.workflows[workflow_name]

        try:
            async with aiohttp.ClientSession() as session:
                headers = config.headers.copy()
                if config.api_key:
                    headers['Authorization'] = f'Bearer {config.api_key}'

                async with session.post(
                    config.endpoint,
                    json=input_data,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=config.timeout)
                ) as response:

                    if response.status == 200:
                        result_data = await response.json()

                        # 处理工作流返回的数据
                        processed_result = await self._process_workflow_result(
                            workflow_name, result_data
                        )

                        return {
                            'success': True,
                            'workflow': workflow_name,
                            'original_result': result_data,
                            'processed_result': processed_result
                        }
                    else:
                        error_text = await response.text()
                        return {
                            'error': f'Workflow failed with status {response.status}: {error_text}'
                        }

        except Exception as e:
            logger.error(f"工作流触发失败 {workflow_name}: {e}")
            return {'error': str(e)}

    async def _process_workflow_result(self, workflow_name: str, result_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理工作流结果"""
        processed_data = {}

        # 根据工作流类型处理数据
        if 'cgm' in workflow_name.lower():
            processed_data = await self.data_processor.process_cgm_data(result_data)
        elif 'nutrition' in workflow_name.lower():
            processed_data = await self.data_processor.process_nutrition_data(result_data)
        elif 'activity' in workflow_name.lower():
            processed_data = await self.data_processor.process_activity_data(result_data)

        # 数据增强（如果数据量不足）
        if self.enhancement_pipeline and 'processed_data' in processed_data:
            df = pd.DataFrame(processed_data['processed_data'])
            if len(df) < 100:  # 数据量不足时进行增强
                enhanced_datasets = self.enhancement_pipeline.enhance_dataset(df)
                processed_data['enhanced_data'] = enhanced_datasets['final'].to_dict('records')
                logger.info(f"数据增强完成，从 {len(df)} 条增加到 {len(enhanced_datasets['final'])} 条")

        return processed_data

    def _schedule_workflow(self, config: WorkflowConfig):
        """调度工作流"""
        async def scheduled_task():
            while True:
                try:
                    await asyncio.sleep(config.schedule_interval)

                    # 构造默认输入数据
                    default_input = {
                        'timestamp': datetime.now().isoformat(),
                        'source': 'scheduler',
                        'workflow': config.name
                    }

                    result = await self.trigger_workflow(config.name, default_input)
                    logger.info(f"定时工作流 {config.name} 执行完成: {result.get('success', False)}")

                except Exception as e:
                    logger.error(f"定时工作流 {config.name} 执行失败: {e}")

        # 启动定时任务
        task = asyncio.create_task(scheduled_task())
        self.active_schedules[config.name] = task
        logger.info(f"启动定时工作流调度: {config.name}")

    async def process_realtime_data(self, data_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理实时数据"""
        try:
            if data_type in self.data_processor.processing_pipelines:
                processed = await self.data_processor.processing_pipelines[data_type](data)

                # 如果是血糖数据，进行实时预测
                if data_type == 'cgm' and self.optimization_system:
                    prediction_input = self._extract_prediction_features(processed)
                    prediction = self.optimization_system.process_prediction(
                        data.get('user_id', 'unknown'), prediction_input
                    )
                    processed['prediction'] = prediction

                return processed
            else:
                logger.warning(f"未找到数据类型 {data_type} 的处理器")
                return {'error': f'No processor for data type: {data_type}'}

        except Exception as e:
            logger.error(f"实时数据处理失败: {e}")
            return {'error': str(e)}

    def _extract_prediction_features(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """从处理后的数据中提取预测特征"""
        features = {}

        if 'statistics' in processed_data:
            stats = processed_data['statistics']
            features.update({
                'glucose_mean': stats.get('mean_glucose', 0),
                'glucose_std': stats.get('glucose_std', 0),
                'quality_score': processed_data.get('quality_score', 0)
            })

        if 'nutrition_summary' in processed_data:
            nutrition = processed_data['nutrition_summary']
            features.update({
                'carbohydrates': nutrition.get('total_carbohydrates', 0),
                'protein': nutrition.get('total_protein', 0),
                'fat': nutrition.get('total_fat', 0),
                'carb_impact_score': nutrition.get('carb_impact_score', 0)
            })

        if 'activity_summary' in processed_data:
            activity = processed_data['activity_summary']
            features.update({
                'exercise_duration': activity.get('total_duration', 0),
                'exercise_impact': activity.get('exercise_impact', 0),
                'intensity_factor': activity.get('intensity_factor', 0)
            })

        # 添加时间特征
        now = datetime.now()
        features.update({
            'hour': now.hour,
            'day_of_week': now.weekday(),
            'is_weekend': now.weekday() >= 5
        })

        return features

    def get_workflow_status(self) -> Dict[str, Any]:
        """获取工作流状态"""
        return {
            'registered_workflows': list(self.workflows.keys()),
            'active_schedules': list(self.active_schedules.keys()),
            'data_flows': list(self.data_flows.keys()),
            'processor_status': self.data_processor is not None,
            'enhancement_status': self.enhancement_pipeline is not None,
            'optimization_status': self.optimization_system is not None
        }

    def stop_all_schedules(self):
        """停止所有定时任务"""
        for name, task in self.active_schedules.items():
            task.cancel()
            logger.info(f"停止定时任务: {name}")
        self.active_schedules.clear()

# 配置加载器
class WorkflowConfigLoader:
    """工作流配置加载器"""

    @staticmethod
    def load_from_yaml(config_path: str) -> Dict[str, WorkflowConfig]:
        """从YAML文件加载配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)

            workflows = {}
            for name, config in config_data.get('workflows', {}).items():
                workflows[name] = WorkflowConfig(
                    name=name,
                    endpoint=config['endpoint'],
                    api_key=config.get('api_key'),
                    headers=config.get('headers', {}),
                    timeout=config.get('timeout', 30),
                    retry_count=config.get('retry_count', 3),
                    data_format=config.get('data_format', 'json'),
                    trigger_type=config.get('trigger_type', 'api'),
                    schedule_interval=config.get('schedule_interval', 300)
                )

            return workflows

        except Exception as e:
            logger.error(f"配置加载失败: {e}")
            return {}

    @staticmethod
    def create_default_config() -> str:
        """创建默认配置文件"""
        default_config = {
            'workflows': {
                'dify_cgm_collector': {
                    'endpoint': 'https://api.dify.ai/v1/workflows/cgm-data/run',
                    'api_key': 'your_dify_api_key',
                    'headers': {'Content-Type': 'application/json'},
                    'trigger_type': 'scheduler',
                    'schedule_interval': 300
                },
                'nutrition_analyzer': {
                    'endpoint': 'https://api.nutrition-service.com/v1/analyze',
                    'api_key': 'your_nutrition_api_key',
                    'trigger_type': 'api'
                },
                'activity_tracker': {
                    'endpoint': 'https://api.fitness-tracker.com/v1/activities',
                    'api_key': 'your_fitness_api_key',
                    'trigger_type': 'webhook'
                }
            }
        }

        config_path = 'workflow_config.yaml'
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"默认配置文件已创建: {config_path}")
        return config_path

# 使用示例
async def main():
    """使用示例"""
    # 创建工作流编排器
    orchestrator = WorkflowOrchestrator()

    # 注册工作流
    cgm_workflow = WorkflowConfig(
        name='cgm_collector',
        endpoint='https://api.dify.ai/v1/workflows/cgm/run',
        api_key='your_api_key',
        trigger_type='scheduler',
        schedule_interval=300
    )
    orchestrator.register_workflow(cgm_workflow)

    # 模拟触发工作流
    input_data = {
        'user_id': 'user_123',
        'date_range': '7d',
        'data_types': ['cgm', 'nutrition', 'activity']
    }

    result = await orchestrator.trigger_workflow('cgm_collector', input_data)
    print(f"工作流结果: {result}")

    # 获取状态
    status = orchestrator.get_workflow_status()
    print(f"工作流状态: {status}")

if __name__ == "__main__":
    asyncio.run(main())
