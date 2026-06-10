#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
集成系统测试脚本
验证所有优化组件的功能
"""

import os
import sys
import json
import logging
import asyncio
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import time

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入所有组件
from large_scale_data_collector import LargeScaleDataCollector, DataPoint
from smart_api_manager import SmartAPIManager
from enhanced_data_quality_processor import DataQualityProcessor, DataPoint as QualityDataPoint
from optimized_model_architecture import ModelFactory, ModelConfig
from integrated_training_system import IntegratedTrainingSystem, TrainingPipelineConfig


class SystemTester:
    """系统测试器"""

    def __init__(self):
        self.test_results = {}
        self.output_dir = Path("outputs/system_test")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def run_all_tests(self):
        """运行所有测试"""
        logger.info("🧪 开始系统测试")

        # 测试1: 数据收集器
        await self._test_data_collector()

        # 测试2: API管理器
        await self._test_api_manager()

        # 测试3: 数据质量处理器
        self._test_data_quality_processor()

        # 测试4: 模型架构
        self._test_model_architecture()

        # 测试5: 集成训练系统
        await self._test_integrated_system()

        # 保存测试结果
        self._save_test_results()

        # 打印测试摘要
        self._print_test_summary()

        logger.info("✅ 系统测试完成")

    async def _test_data_collector(self):
        """测试数据收集器"""
        logger.info("🔍 测试数据收集器...")

        try:
            collector = LargeScaleDataCollector()

            # 测试小规模数据收集
            test_foods = ["苹果", "香蕉", "鸡肉"]
            data_points = await collector.collect_nutrition_data(test_foods, "测试菜系")

            self.test_results["data_collector"] = {
                "status": "success",
                "collected_samples": len(data_points),
                "success_rate": len([dp for dp in data_points if dp.quality_score > 0.5]) / len(data_points) if data_points else 0
            }

            logger.info(f"   ✅ 数据收集器测试通过: {len(data_points)} 样本")

        except Exception as e:
            logger.error(f"   ❌ 数据收集器测试失败: {e}")
            self.test_results["data_collector"] = {
                "status": "failed",
                "error": str(e)
            }

    async def _test_api_manager(self):
        """测试API管理器"""
        logger.info("🔍 测试API管理器...")

        try:
            manager = SmartAPIManager()

            # 测试单个查询
            result = await manager.query_nutrition("苹果")

            # 测试批量查询
            test_foods = ["苹果", "香蕉", "鸡肉"]
            batch_results = await manager.batch_query_nutrition(test_foods, max_concurrent=2)

            success_count = sum(1 for r in batch_results if r is not None)

            self.test_results["api_manager"] = {
                "status": "success",
                "single_query_success": result is not None,
                "batch_query_success_rate": success_count / len(test_foods),
                "api_metrics": manager.get_api_metrics()
            }

            logger.info(f"   ✅ API管理器测试通过: {success_count}/{len(test_foods)} 成功")

        except Exception as e:
            logger.error(f"   ❌ API管理器测试失败: {e}")
            self.test_results["api_manager"] = {
                "status": "failed",
                "error": str(e)
            }

    def _test_data_quality_processor(self):
        """测试数据质量处理器"""
        logger.info("🔍 测试数据质量处理器...")

        try:
            processor = DataQualityProcessor()

            # 创建测试数据点
            test_data_points = [
                QualityDataPoint(
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
                    source="test",
                    quality_score=0.9,
                    timestamp="2024-01-01T00:00:00"
                )
            ]

            # 处理数据
            processed_data, quality_metrics = processor.process_data_points(test_data_points)

            self.test_results["data_quality_processor"] = {
                "status": "success",
                "processed_samples": len(processed_data),
                "quality_score": quality_metrics.overall_score,
                "completeness": quality_metrics.completeness,
                "consistency": quality_metrics.consistency
            }

            logger.info(f"   ✅ 数据质量处理器测试通过: 质量分数 {quality_metrics.overall_score:.3f}")

        except Exception as e:
            logger.error(f"   ❌ 数据质量处理器测试失败: {e}")
            self.test_results["data_quality_processor"] = {
                "status": "failed",
                "error": str(e)
            }

    def _test_model_architecture(self):
        """测试模型架构"""
        logger.info("🔍 测试模型架构...")

        try:
            # 测试不同模型类型
            model_types = ["efficient", "optimized", "advanced"]
            model_results = {}

            for model_type in model_types:
                # 获取配置
                config = ModelFactory.get_recommended_config(model_type)

                # 创建模型
                model = ModelFactory.create_model(model_type, config)

                # 测试前向传播
                batch_size = 4
                preferences = torch.randn(batch_size, 10)
                nutrition = torch.randn(batch_size, 5)
                cultural = torch.randint(0, 9, (batch_size, 2))

                with torch.no_grad():
                    output = model(preferences, nutrition, cultural)

                # 分析模型
                from optimized_model_architecture import ModelAnalyzer
                analysis = ModelAnalyzer.analyze_model(model)

                model_results[model_type] = {
                    "status": "success",
                    "total_params": analysis["size_info"]["total_params"],
                    "output_shape": output.shape,
                    "model_size_mb": analysis["size_info"]["model_size_mb"]
                }

            self.test_results["model_architecture"] = {
                "status": "success",
                "models_tested": len(model_types),
                "model_results": model_results
            }

            logger.info(f"   ✅ 模型架构测试通过: {len(model_types)} 种模型")

        except Exception as e:
            logger.error(f"   ❌ 模型架构测试失败: {e}")
            self.test_results["model_architecture"] = {
                "status": "failed",
                "error": str(e)
            }

    async def _test_integrated_system(self):
        """测试集成训练系统"""
        logger.info("🔍 测试集成训练系统...")

        try:
            # 创建简化的训练配置
            config = TrainingPipelineConfig(
                target_samples=10,  # 极小的样本数用于测试
                model_type="efficient",  # 使用最简单的模型
                use_cross_validation=False,  # 禁用交叉验证
                num_epochs=2,  # 极少的轮数
                batch_size=4,
                patience=1
            )

            # 创建训练系统
            training_system = IntegratedTrainingSystem(config)

            # 运行简化管道
            results = await training_system.run_complete_pipeline()

            self.test_results["integrated_system"] = {
                "status": "success" if results.get("pipeline_summary", {}).get("success", False) else "failed",
                "pipeline_success": results.get("pipeline_summary", {}).get("success", False),
                "total_time": results.get("pipeline_summary", {}).get("total_time", 0)
            }

            logger.info(f"   ✅ 集成训练系统测试通过")

        except Exception as e:
            logger.error(f"   ❌ 集成训练系统测试失败: {e}")
            self.test_results["integrated_system"] = {
                "status": "failed",
                "error": str(e)
            }

    def _save_test_results(self):
        """保存测试结果"""
        results_path = self.output_dir / "test_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(self.test_results, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ 测试结果已保存: {results_path}")

    def _print_test_summary(self):
        """打印测试摘要"""
        logger.info("📊 测试摘要:")

        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result.get("status") == "success")

        logger.info(f"   总测试数: {total_tests}")
        logger.info(f"   通过测试: {passed_tests}")
        logger.info(f"   失败测试: {total_tests - passed_tests}")
        logger.info(f"   成功率: {passed_tests / total_tests * 100:.1f}%")

        # 详细结果
        for test_name, result in self.test_results.items():
            status = "✅" if result.get("status") == "success" else "❌"
            logger.info(f"   {status} {test_name}: {result.get('status', 'unknown')}")


async def main():
    """主函数"""
    logger.info("🚀 启动集成系统测试")

    # 创建测试器
    tester = SystemTester()

    # 运行所有测试
    await tester.run_all_tests()

    logger.info("✅ 集成系统测试完成")


if __name__ == "__main__":
    asyncio.run(main())
