

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强的集成系统主模块
整合工作流、数据增强、跨平台数据流、模型适应、反馈优化等功能
实现完整的自动化、数据驱动的智能健康监测系统
"""

import asyncio
import logging
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import traceback

# FastAPI相关
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# 导入自定义模块
sys.path.append(str(Path(__file__).parent))

from academic_integrated_system import AcademicIntegratedSystem, AcademicGlucosePredictor, AcademicImagePredictor
from app.data_integration.workflow_integration import WorkflowOrchestrator, WorkflowConfig, WorkflowConfigLoader
from data_integration.cross_platform_dataflow import CrossPlatformDataFlowManager, DifyAdapter, APIAdapter
from app.data_integration.data_augmentation import DataEnhancementPipeline, AugmentationConfig
from data_integration.feedback_optimization import IterativeOptimizationSystem, FeedbackConfig
from models.adaptive_model_tuning import AdaptiveModelManager, AdaptationConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_integrated_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedIntegratedSystem:
    """增强的智能健康监测集成系统"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "system_config.json"
        self.config = self._load_config()

        # 核心组件
        self.academic_system = None
        self.workflow_orchestrator = None
        self.dataflow_manager = None
        self.enhancement_pipeline = None
        self.optimization_system = None
        self.adaptive_manager = None

        # FastAPI应用
        self.app = None

        # 系统状态
        self.is_initialized = False
        self.is_running = False

    def _load_config(self) -> Dict[str, Any]:
        """加载系统配置"""
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.warning(f"配置文件不存在: {self.config_path}，使用默认配置")
                return self._create_default_config()
        except Exception as e:
            logger.error(f"加载配置失败: {e}")
            return self._create_default_config()

    def _create_default_config(self) -> Dict[str, Any]:
        """创建默认配置"""
        default_config = {
            "system": {
                "name": "Enhanced Integrated Health Monitoring System",
                "version": "2.0.0",
                "debug": False
            },
            "server": {
                "host": "0.0.0.0",
                "port": 8000,
                "reload": False
            },
            "workflows": {
                "dify_cgm_collector": {
                    "endpoint": "https://api.dify.ai/v1/workflows/cgm-data/run",
                    "api_key": "your_dify_api_key",
                    "trigger_type": "scheduler",
                    "schedule_interval": 300
                },
                "nutrition_analyzer": {
                    "endpoint": "https://api.nutrition-service.com/v1/analyze",
                    "api_key": "your_nutrition_api_key",
                    "trigger_type": "api"
                }
            },
            "data_sources": {
                "cgm_api": {
                    "base_url": "https://api.cgm-provider.com",
                    "api_key": "your_cgm_api_key",
                    "data_type": "cgm"
                },
                "nutrition_api": {
                    "base_url": "https://api.nutrition-provider.com",
                    "api_key": "your_nutrition_api_key",
                    "data_type": "nutrition"
                }
            },
            "adaptation": {
                "learning_rate": 0.001,
                "fine_tune_epochs": 10,
                "min_adaptation_samples": 20
            },
            "feedback": {
                "min_feedback_samples": 10,
                "update_frequency": 100,
                "confidence_threshold": 0.7
            },
            "data_enhancement": {
                "noise_level": 0.05,
                "augmentation_ratio": 2.0,
                "synthetic_samples": 1000
            }
        }

        # 保存默认配置
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)

        logger.info(f"默认配置已保存到: {self.config_path}")
        return default_config

    async def initialize(self):
        """初始化系统"""
        try:
            logger.info("开始初始化增强集成系统...")

            # 1. 初始化学术级基础系统
            logger.info("初始化学术级基础系统...")
            self.academic_system = AcademicIntegratedSystem()

            # 2. 初始化工作流编排器
            logger.info("初始化工作流编排器...")
            self.workflow_orchestrator = WorkflowOrchestrator()

            # 注册工作流
            for workflow_name, workflow_config in self.config.get("workflows", {}).items():
                config = WorkflowConfig(
                    name=workflow_name,
                    endpoint=workflow_config["endpoint"],
                    api_key=workflow_config.get("api_key"),
                    trigger_type=workflow_config.get("trigger_type", "api"),
                    schedule_interval=workflow_config.get("schedule_interval", 300)
                )
                self.workflow_orchestrator.register_workflow(config)

            # 3. 初始化跨平台数据流管理器
            logger.info("初始化跨平台数据流管理器...")
            self.dataflow_manager = CrossPlatformDataFlowManager()

            # 注册数据源适配器
            for source_name, source_config in self.config.get("data_sources", {}).items():
                if source_name.startswith("dify"):
                    adapter = DifyAdapter(source_name, source_config)
                else:
                    adapter = APIAdapter(source_name, source_config)

                self.dataflow_manager.register_adapter(source_name, adapter)

            # 4. 初始化数据增强流水线
            logger.info("初始化数据增强流水线...")
            enhancement_config = AugmentationConfig(**self.config.get("data_enhancement", {}))
            self.enhancement_pipeline = DataEnhancementPipeline(enhancement_config)

            # 5. 初始化自适应模型管理器
            logger.info("初始化自适应模型管理器...")
            adaptation_config = AdaptationConfig(**self.config.get("adaptation", {}))
            # 使用学术系统的血糖预测器模型
            base_model = self.academic_system.glucose_predictor.model if self.academic_system.glucose_predictor.model else None
            if base_model:
                self.adaptive_manager = AdaptiveModelManager(base_model, adaptation_config)
            else:
                logger.warning("基础模型不可用，跳过自适应管理器初始化")

            # 6. 初始化迭代优化系统
            logger.info("初始化迭代优化系统...")
            feedback_config = FeedbackConfig(**self.config.get("feedback", {}))
            self.optimization_system = IterativeOptimizationSystem(
                self.academic_system.glucose_predictor, feedback_config
            )

            # 7. 设置组件间的集成
            logger.info("设置组件集成...")
            await self._setup_integration()

            # 8. 初始化FastAPI应用
            logger.info("初始化FastAPI应用...")
            self._setup_fastapi()

            self.is_initialized = True
            logger.info("增强集成系统初始化完成")

        except Exception as e:
            logger.error(f"系统初始化失败: {e}")
            logger.error(traceback.format_exc())
            raise

    async def _setup_integration(self):
        """设置组件间的集成"""
        try:
            # 设置工作流处理器
            if self.workflow_orchestrator and self.dataflow_manager:
                self.workflow_orchestrator.setup_processors(
                    self.dataflow_manager,
                    self.academic_system.glucose_predictor,
                    self.academic_system.image_predictor
                )

            # 设置优化系统的集成组件
            if self.optimization_system:
                self.optimization_system.set_integration_components(
                    adaptive_manager=self.adaptive_manager,
                    workflow_orchestrator=self.workflow_orchestrator,
                    data_flow_manager=self.dataflow_manager
                )

            # 启动数据流管理器
            if self.dataflow_manager:
                await self.dataflow_manager.start()

            logger.info("组件集成设置完成")

        except Exception as e:
            logger.error(f"组件集成设置失败: {e}")
            raise

    def _setup_fastapi(self):
        """设置FastAPI应用"""
        self.app = FastAPI(
            title=self.config["system"]["name"],
            description="增强的智能健康监测集成系统，支持工作流集成、数据增强、自适应学习和反馈优化",
            version=self.config["system"]["version"]
        )

        # CORS中间件
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # 注册路由
        self._register_routes()

    def _register_routes(self):
        """注册API路由"""

        @self.app.get("/")
        async def root():
            return {
                "message": self.config["system"]["name"],
                "version": self.config["system"]["version"],
                "status": "running" if self.is_running else "stopped",
                "initialized": self.is_initialized
            }

        @self.app.get("/health")
        async def health_check():
            """系统健康检查"""
            health_status = {
                "timestamp": datetime.now().isoformat(),
                "status": "healthy" if self.is_initialized else "unhealthy",
                "components": {
                    "academic_system": self.academic_system is not None,
                    "workflow_orchestrator": self.workflow_orchestrator is not None,
                    "dataflow_manager": self.dataflow_manager is not None,
                    "enhancement_pipeline": self.enhancement_pipeline is not None,
                    "optimization_system": self.optimization_system is not None,
                    "adaptive_manager": self.adaptive_manager is not None
                }
            }

            if self.dataflow_manager:
                health_status["dataflow_status"] = self.dataflow_manager.get_system_status()

            if self.workflow_orchestrator:
                health_status["workflow_status"] = self.workflow_orchestrator.get_workflow_status()

            return health_status

        @self.app.post("/predict/glucose/enhanced")
        async def enhanced_glucose_prediction(data: Dict[str, Any]):
            """增强的血糖预测（集成所有功能）"""
            try:
                if not self.is_initialized:
                    raise HTTPException(status_code=503, detail="System not initialized")

                user_id = data.get('user_id', 'anonymous')

                # 使用优化系统进行预测（已集成自适应、工作流等功能）
                prediction = self.optimization_system.process_prediction(user_id, data)

                return JSONResponse(content=prediction)

            except Exception as e:
                logger.error(f"增强血糖预测失败: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/feedback")
        async def submit_feedback(feedback_data: Dict[str, Any]):
            """提交用户反馈"""
            try:
                if not self.optimization_system:
                    raise HTTPException(status_code=503, detail="Optimization system not available")

                self.optimization_system.process_feedback(
                    user_id=feedback_data['user_id'],
                    prediction_id=feedback_data['prediction_id'],
                    actual_glucose=feedback_data['actual_glucose'],
                    satisfaction_score=feedback_data['satisfaction_score'],
                    additional_info=feedback_data.get('additional_info')
                )

                return {"status": "success", "message": "Feedback submitted successfully"}

            except Exception as e:
                logger.error(f"反馈提交失败: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/user/register")
        async def register_user(user_data: Dict[str, Any]):
            """注册新用户"""
            try:
                if not self.adaptive_manager:
                    raise HTTPException(status_code=503, detail="Adaptive manager not available")

                profile = self.adaptive_manager.register_user(user_data)

                return {
                    "status": "success",
                    "user_id": profile.user_id,
                    "message": "User registered successfully"
                }

            except Exception as e:
                logger.error(f"用户注册失败: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/user/{user_id}/adapt")
        async def adapt_user_model(user_id: str, adaptation_data: List[Dict[str, Any]]):
            """为用户适应模型"""
            try:
                if not self.adaptive_manager:
                    raise HTTPException(status_code=503, detail="Adaptive manager not available")

                result = self.adaptive_manager.adapt_for_user(user_id, adaptation_data)

                return result

            except Exception as e:
                logger.error(f"用户模型适应失败: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/workflow/{workflow_name}/trigger")
        async def trigger_workflow(workflow_name: str, workflow_data: Dict[str, Any]):
            """触发工作流"""
            try:
                if not self.workflow_orchestrator:
                    raise HTTPException(status_code=503, detail="Workflow orchestrator not available")

                result = await self.workflow_orchestrator.trigger_workflow(workflow_name, workflow_data)

                return result

            except Exception as e:
                logger.error(f"工作流触发失败: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/data/process_batch")
        async def process_batch_data(request_data: Dict[str, Any]):
            """批量处理数据"""
            try:
                if not self.dataflow_manager:
                    raise HTTPException(status_code=503, detail="Dataflow manager not available")

                user_id = request_data.get('user_id', 'anonymous')
                platforms = request_data.get('platforms', None)

                result = await self.dataflow_manager.process_batch_data(user_id, platforms)

                return result

            except Exception as e:
                logger.error(f"批量数据处理失败: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/data/enhance")
        async def enhance_data(background_tasks: BackgroundTasks):
            """触发数据增强"""
            try:
                if not self.enhancement_pipeline:
                    raise HTTPException(status_code=503, detail="Enhancement pipeline not available")

                # 在后台执行数据增强
                background_tasks.add_task(self._run_data_enhancement)

                return {"status": "success", "message": "Data enhancement started in background"}

            except Exception as e:
                logger.error(f"数据增强失败: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/system/status")
        async def get_system_status():
            """获取系统状态"""
            try:
                status = {
                    "system_info": {
                        "name": self.config["system"]["name"],
                        "version": self.config["system"]["version"],
                        "initialized": self.is_initialized,
                        "running": self.is_running
                    }
                }

                # 各组件状态
                if self.dataflow_manager:
                    status["dataflow"] = self.dataflow_manager.get_system_status()

                if self.workflow_orchestrator:
                    status["workflow"] = self.workflow_orchestrator.get_workflow_status()

                if self.optimization_system:
                    status["optimization"] = self.optimization_system.get_system_stats()

                if self.adaptive_manager:
                    status["adaptation"] = self.adaptive_manager.get_adaptation_report()

                return status

            except Exception as e:
                logger.error(f"获取系统状态失败: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/reports/adaptation/{user_id}")
        async def get_adaptation_report(user_id: str):
            """获取用户适应报告"""
            try:
                if not self.adaptive_manager:
                    raise HTTPException(status_code=503, detail="Adaptive manager not available")

                report = self.adaptive_manager.get_adaptation_report(user_id)

                return report

            except Exception as e:
                logger.error(f"获取适应报告失败: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    async def _run_data_enhancement(self):
        """运行数据增强（后台任务）"""
        try:
            logger.info("开始后台数据增强...")

            # 这里应该有实际的数据源
            # 暂时使用模拟数据
            import pandas as pd
            import numpy as np

            mock_data = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=100, freq='15min'),
                'glucose': np.random.normal(6.0, 1.0, 100),
                'carbohydrates': np.random.uniform(20, 80, 100),
                'exercise': np.random.uniform(0, 60, 100)
            })

            enhanced_datasets = self.enhancement_pipeline.enhance_dataset(mock_data)

            logger.info(f"数据增强完成，原始数据: {len(mock_data)}, 增强后: {len(enhanced_datasets['final'])}")

        except Exception as e:
            logger.error(f"后台数据增强失败: {e}")

    async def run(self, host: str = None, port: int = None):
        """运行系统"""
        try:
            if not self.is_initialized:
                await self.initialize()

            self.is_running = True

            # 使用配置中的主机和端口，或使用传入的参数
            host = host or self.config["server"]["host"]
            port = port or self.config["server"]["port"]
            reload = self.config["server"]["reload"]

            logger.info(f"启动增强集成系统服务器: {host}:{port}")

            # 运行FastAPI应用
            config = uvicorn.Config(
                app=self.app,
                host=host,
                port=port,
                reload=reload,
                log_level="info"
            )
            server = uvicorn.Server(config)
            await server.serve()

        except Exception as e:
            logger.error(f"系统运行失败: {e}")
            raise
        finally:
            await self.shutdown()

    async def shutdown(self):
        """关闭系统"""
        try:
            logger.info("开始关闭增强集成系统...")

            self.is_running = False

            # 停止各个组件
            if self.dataflow_manager:
                await self.dataflow_manager.stop()

            if self.workflow_orchestrator:
                self.workflow_orchestrator.stop_all_schedules()

            # 保存适应状态
            if self.adaptive_manager:
                try:
                    save_path = "adaptation_state.pkl"
                    self.adaptive_manager.save_adaptation_state(save_path)
                    logger.info(f"适应状态已保存到: {save_path}")
                except Exception as e:
                    logger.error(f"保存适应状态失败: {e}")

            logger.info("增强集成系统已关闭")

        except Exception as e:
            logger.error(f"系统关闭失败: {e}")

# 主函数
async def main():
    """主函数"""
    try:
        # 创建增强集成系统
        system = EnhancedIntegratedSystem()

        # 运行系统
        await system.run()

    except KeyboardInterrupt:
        logger.info("接收到中断信号，正在关闭系统...")
    except Exception as e:
        logger.error(f"系统运行异常: {e}")
        logger.error(traceback.format_exc())
    finally:
        logger.info("系统已退出")

if __name__ == "__main__":
    asyncio.run(main())

__all__ = ["'logger'", "'EnhancedIntegratedSystem'"]
