

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强型血糖预测系统 - 主集成模块
整合API数据流、数据增强、反馈机制和迭代优化
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import asyncio
import json
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入现有模块
from modules.glucose_prediction.predictor import GlucosePredictor
from modules.image_recognition.predictor import ImagePredictor
from modules.fusion.decision_engine import HealthDecisionEngine

# 导入新增的数据集成模块
from data_integration.api_adapters import (
    DataIntegrationManager, CGMApiAdapter, NutritionApiAdapter,
    ActivityApiAdapter, DataSourceConfig, create_default_configs
)
from app.data_integration.data_augmentation import (
    DataEnhancementPipeline, AugmentationConfig, SyntheticDataGenerator
)
from data_integration.feedback_optimization import (
    IterativeOptimizationSystem, FeedbackConfig
)

from utils.logger import setup_logger
from utils.monitor import SystemMonitor

class EnhancedGlucosePredictionSystem:
    """增强型血糖预测系统"""

    def __init__(self, config_path: Optional[str] = None):
        """初始化系统"""
        self.logger = setup_logger(__name__)
        self.monitor = SystemMonitor()

        # 加载配置
        self.config = self._load_config(config_path)

        # 初始化核心组件
        self.glucose_predictor = None
        self.image_predictor = None
        self.health_engine = None

        # 初始化数据集成组件
        self.data_manager = DataIntegrationManager()
        self.enhancement_pipeline = None
        self.optimization_system = None

        # 系统状态
        self.is_initialized = False
        self.data_sources_connected = False

        self.logger.info("增强型血糖预测系统初始化完成")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """加载配置"""
        default_config = {
            "system": {
                "name": "Enhanced Glucose Prediction System",
                "version": "2.0.0",
                "debug": False
            },
            "data_integration": {
                "enable_api_adapters": True,
                "enable_data_augmentation": True,
                "enable_feedback_optimization": True,
                "cache_ttl": 300
            },
            "augmentation": {
                "noise_level": 0.05,
                "augmentation_ratio": 2.0,
                "synthetic_samples": 1000
            },
            "feedback": {
                "feedback_window": 24,
                "min_feedback_samples": 10,
                "learning_rate": 0.001,
                "update_frequency": 100
            },
            "api_sources": {
                "cgm": {
                    "enabled": True,
                    "api_url": "https://api.cgm-provider.com/v1",
                    "api_key": "your_cgm_api_key"
                },
                "nutrition": {
                    "enabled": True,
                    "api_url": "https://api.nutrition-provider.com/v1",
                    "api_key": "your_nutrition_api_key"
                },
                "activity": {
                    "enabled": True,
                    "api_url": "https://api.activity-provider.com/v1",
                    "api_key": "your_activity_api_key"
                }
            }
        }

        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                self.logger.warning(f"配置文件加载失败: {e}")

        return default_config

    async def initialize_system(self) -> bool:
        """异步初始化系统"""
        try:
            self.logger.info("开始系统初始化...")

            # 1. 初始化核心预测模块
            await self._initialize_core_modules()

            # 2. 初始化数据集成
            await self._initialize_data_integration()

            # 3. 初始化数据增强
            await self._initialize_data_enhancement()

            # 4. 初始化反馈优化
            await self._initialize_feedback_optimization()

            # 5. 连接数据源
            await self._connect_data_sources()

            self.is_initialized = True
            self.logger.info("系统初始化完成")
            return True

        except Exception as e:
            self.logger.error(f"系统初始化失败: {e}")
            return False

    async def _initialize_core_modules(self):
        """初始化核心模块"""
        self.logger.info("初始化核心预测模块...")

        # 初始化血糖预测器
        self.glucose_predictor = GlucosePredictor()

        # 初始化图像识别器
        self.image_predictor = ImagePredictor()

        # 初始化健康决策引擎
        self.health_engine = HealthDecisionEngine(
            glucose_predictor=self.glucose_predictor,
            image_predictor=self.image_predictor
        )

        self.logger.info("核心模块初始化完成")

    async def _initialize_data_integration(self):
        """初始化数据集成"""
        if not self.config["data_integration"]["enable_api_adapters"]:
            return

        self.logger.info("初始化数据集成模块...")

        # 创建API适配器配置
        api_configs = self.config["api_sources"]

        if api_configs["cgm"]["enabled"]:
            cgm_config = DataSourceConfig(
                name="CGM",
                api_url=api_configs["cgm"]["api_url"],
                api_key=api_configs["cgm"]["api_key"]
            )
            cgm_adapter = CGMApiAdapter(cgm_config)
            self.data_manager.register_adapter("cgm", cgm_adapter)

        if api_configs["nutrition"]["enabled"]:
            nutrition_config = DataSourceConfig(
                name="Nutrition",
                api_url=api_configs["nutrition"]["api_url"],
                api_key=api_configs["nutrition"]["api_key"]
            )
            nutrition_adapter = NutritionApiAdapter(nutrition_config)
            self.data_manager.register_adapter("nutrition", nutrition_adapter)

        if api_configs["activity"]["enabled"]:
            activity_config = DataSourceConfig(
                name="Activity",
                api_url=api_configs["activity"]["api_url"],
                api_key=api_configs["activity"]["api_key"]
            )
            activity_adapter = ActivityApiAdapter(activity_config)
            self.data_manager.register_adapter("activity", activity_adapter)

        self.logger.info("数据集成模块初始化完成")

    async def _initialize_data_enhancement(self):
        """初始化数据增强"""
        if not self.config["data_integration"]["enable_data_augmentation"]:
            return

        self.logger.info("初始化数据增强模块...")

        # 创建增强配置
        aug_config = AugmentationConfig(
            noise_level=self.config["augmentation"]["noise_level"],
            augmentation_ratio=self.config["augmentation"]["augmentation_ratio"],
            synthetic_samples=self.config["augmentation"]["synthetic_samples"]
        )

        self.enhancement_pipeline = DataEnhancementPipeline(aug_config)

        self.logger.info("数据增强模块初始化完成")

    async def _initialize_feedback_optimization(self):
        """初始化反馈优化"""
        if not self.config["data_integration"]["enable_feedback_optimization"]:
            return

        self.logger.info("初始化反馈优化模块...")

        # 创建反馈配置
        feedback_config = FeedbackConfig(
            feedback_window=self.config["feedback"]["feedback_window"],
            min_feedback_samples=self.config["feedback"]["min_feedback_samples"],
            learning_rate=self.config["feedback"]["learning_rate"],
            update_frequency=self.config["feedback"]["update_frequency"]
        )

        self.optimization_system = IterativeOptimizationSystem(
            self.glucose_predictor, feedback_config
        )

        self.logger.info("反馈优化模块初始化完成")

    async def _connect_data_sources(self):
        """连接数据源"""
        if not self.data_manager.adapters:
            self.logger.warning("没有可用的数据源适配器")
            return

        self.logger.info("连接数据源...")

        # 测试数据源连接
        test_params = {
            'start_date': (datetime.now() - timedelta(days=1)).isoformat(),
            'end_date': datetime.now().isoformat(),
            'user_id': 'test_user'
        }

        try:
            test_data = self.data_manager.fetch_all_data(test_params)
            if test_data:
                self.data_sources_connected = True
                self.logger.info(f"成功连接 {len(test_data)} 个数据源")
            else:
                self.logger.warning("数据源连接测试失败")
        except Exception as e:
            self.logger.error(f"数据源连接失败: {e}")

    async def get_enhanced_prediction(self, user_id: str,
                                    input_data: Dict[str, Any],
                                    use_real_time_data: bool = True) -> Dict[str, Any]:
        """获取增强预测"""
        if not self.is_initialized:
            raise RuntimeError("系统未初始化")

        try:
            # 1. 获取实时数据（如果启用）
            enhanced_input = input_data.copy()
            if use_real_time_data and self.data_sources_connected:
                real_time_data = await self._get_real_time_data(user_id)
                enhanced_input.update(real_time_data)

            # 2. 数据增强（如果启用）
            if self.enhancement_pipeline:
                enhanced_input = await self._enhance_input_data(enhanced_input)

            # 3. 获取预测
            if self.optimization_system:
                prediction = self.optimization_system.process_prediction(
                    user_id, enhanced_input
                )
            else:
                prediction = self.glucose_predictor.predict(enhanced_input)

            # 4. 添加增强信息
            prediction.update({
                "enhancement_applied": self.enhancement_pipeline is not None,
                "real_time_data_used": use_real_time_data and self.data_sources_connected,
                "optimization_active": self.optimization_system is not None,
                "prediction_timestamp": datetime.now().isoformat()
            })

            return prediction

        except Exception as e:
            self.logger.error(f"增强预测失败: {e}")
            return {"error": str(e)}

    async def _get_real_time_data(self, user_id: str) -> Dict[str, Any]:
        """获取实时数据"""
        try:
            params = {
                'start_date': (datetime.now() - timedelta(hours=24)).isoformat(),
                'end_date': datetime.now().isoformat(),
                'user_id': user_id
            }

            data_sources = self.data_manager.fetch_all_data(params)
            merged_data = self.data_manager.merge_data(data_sources)

            if merged_data.empty:
                return {}

            # 提取最新数据
            latest_data = merged_data.iloc[-1].to_dict()

            # 转换为模型输入格式
            real_time_features = {
                'glucose': latest_data.get('glucose', 0),
                'carbohydrates': latest_data.get('carbohydrates', 0),
                'exercise': latest_data.get('exercise', 0),
                'hour': latest_data.get('hour', datetime.now().hour),
                'day_of_week': latest_data.get('day_of_week', datetime.now().weekday())
            }

            return real_time_features

        except Exception as e:
            self.logger.error(f"获取实时数据失败: {e}")
            return {}

    async def _enhance_input_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """增强输入数据"""
        try:
            # 将输入数据转换为DataFrame
            import pandas as pd
            df = pd.DataFrame([input_data])

            # 应用数据增强
            enhanced_datasets = self.enhancement_pipeline.enhance_dataset(df)

            # 使用增强后的数据
            enhanced_data = enhanced_datasets['final'].iloc[0].to_dict()

            return enhanced_data

        except Exception as e:
            self.logger.error(f"数据增强失败: {e}")
            return input_data

    async def process_user_feedback(self, user_id: str, prediction_id: str,
                                  actual_glucose: float, satisfaction_score: int,
                                  additional_info: Optional[Dict[str, Any]] = None):
        """处理用户反馈"""
        if not self.optimization_system:
            self.logger.warning("反馈优化系统未启用")
            return

        try:
            self.optimization_system.process_feedback(
                user_id, prediction_id, actual_glucose,
                satisfaction_score, additional_info
            )

            self.logger.info(f"用户反馈处理完成: {user_id}")

        except Exception as e:
            self.logger.error(f"反馈处理失败: {e}")

    async def get_personalized_recommendations(self, user_id: str,
                                             user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """获取个性化推荐"""
        if not self.optimization_system:
            # 使用基础推荐
            return self.glucose_predictor._generate_recommendations([6.0])

        try:
            recommendations = self.optimization_system.get_optimized_recommendations(
                user_id, user_profile
            )

            return recommendations

        except Exception as e:
            self.logger.error(f"获取个性化推荐失败: {e}")
            return {"error": str(e)}

    async def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        status = {
            "system_initialized": self.is_initialized,
            "data_sources_connected": self.data_sources_connected,
            "modules_enabled": {
                "data_integration": self.data_manager.adapters is not None,
                "data_augmentation": self.enhancement_pipeline is not None,
                "feedback_optimization": self.optimization_system is not None
            },
            "timestamp": datetime.now().isoformat()
        }

        # 添加优化系统统计
        if self.optimization_system:
            status["optimization_stats"] = self.optimization_system.get_system_stats()

        # 添加系统监控信息
        try:
            system_monitor = self.monitor.get_system_status()
            status["system_resources"] = system_monitor
        except Exception as e:
            self.logger.warning(f"获取系统监控信息失败: {e}")

        return status

    async def run_batch_training(self, training_data_path: str) -> Dict[str, Any]:
        """运行批量训练"""
        try:
            self.logger.info("开始批量训练...")

            # 加载训练数据
            import pandas as pd
            training_data = pd.read_csv(training_data_path)

            # 数据增强
            if self.enhancement_pipeline:
                enhanced_datasets = self.enhancement_pipeline.enhance_dataset(training_data)
                training_data = enhanced_datasets['final']

            # 训练模型
            train_result = self.glucose_predictor.train(training_data)

            # 训练自监督学习特征
            if self.enhancement_pipeline:
                self.enhancement_pipeline.train_self_supervised_features(training_data)

            self.logger.info("批量训练完成")

            return {
                "success": True,
                "training_result": train_result,
                "data_enhanced": self.enhancement_pipeline is not None,
                "samples_used": len(training_data)
            }

        except Exception as e:
            self.logger.error(f"批量训练失败: {e}")
            return {"success": False, "error": str(e)}

# 异步主函数
async def main():
    """主函数"""
    try:
        # 创建增强系统
        system = EnhancedGlucosePredictionSystem()

        # 初始化系统
        if not await system.initialize_system():
            print("❌ 系统初始化失败")
            return

        print("✅ 增强型血糖预测系统启动成功")

        # 显示系统状态
        status = await system.get_system_status()
        print(f"📊 系统状态: {json.dumps(status, indent=2, ensure_ascii=False)}")

        # 示例：获取增强预测
        user_id = "test_user_123"
        input_data = {
            'age': 45,
            'bmi': 25.5,
            'glucose': 6.0,
            'carbohydrates': 50,
            'exercise': 30
        }

        prediction = await system.get_enhanced_prediction(user_id, input_data)
        print(f"🔮 增强预测结果: {json.dumps(prediction, indent=2, ensure_ascii=False)}")

        # 示例：处理用户反馈
        await system.process_user_feedback(
            user_id,
            datetime.now().isoformat(),
            6.2,  # 实际血糖值
            4,    # 满意度评分
            {'age': 45, 'bmi': 25.5}
        )

        # 示例：获取个性化推荐
        user_profile = {
            'age': 45,
            'bmi': 25.5,
            'exercise_level': 3,
            'stress_level': 2,
            'sleep_hours': 7
        }

        recommendations = await system.get_personalized_recommendations(
            user_id, user_profile
        )
        print(f"💡 个性化推荐: {json.dumps(recommendations, indent=2, ensure_ascii=False)}")

    except Exception as e:
        print(f"❌ 系统运行失败: {e}")
        logging.error(f"系统运行失败: {e}")

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 运行异步主函数
    asyncio.run(main())

__all__ = ["'project_root'", "'EnhancedGlucosePredictionSystem'"]
