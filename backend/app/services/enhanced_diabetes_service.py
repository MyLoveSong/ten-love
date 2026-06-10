"""
增强糖尿病服务 - 集成训练好的模型
将TRAIN目录下训练的模型与现有系统无缝集成
"""

import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import numpy as np

from app.services.integrated_diabetes_service import IntegratedDiabetesService
from app.models.trained_model_service import (
    GlucosePredictionService,
    TrainedModelLoader,
    PredictionInput,
    PredictionOutput
)
# 延迟导入以避免循环导入
# from app.services.model_lifecycle_service import get_model_lifecycle_service

# 设置日志
logger = logging.getLogger(__name__)


class EnhancedDiabetesService(IntegratedDiabetesService):
    """
    增强糖尿病服务 - 继承现有服务并集成训练模型
    遵循开闭原则：对扩展开放，对修改封闭
    """

    def __init__(self, *args, **kwargs):
        """初始化增强服务"""
        super().__init__(*args, **kwargs)

        # 集成模型生命周期服务（延迟导入）
        try:
            from app.services.model_lifecycle_service import get_model_lifecycle_service
            self.model_lifecycle_service = get_model_lifecycle_service()

            # 添加模型更新回调
            self.model_lifecycle_service.add_model_update_callback(self._handle_model_update)

            logger.info("✅ 模型生命周期服务集成成功")
        except Exception as e:
            logger.warning(f"模型生命周期服务集成失败: {e}")
            self.model_lifecycle_service = None

        # 集成训练好的模型服务
        try:
            model_loader = TrainedModelLoader()
            self.trained_model_service = GlucosePredictionService(
                model_loader=model_loader,
                model_path=None,  # 自动查找最新模型
                sequence_length=24,
                prediction_horizon=6
            )
            self.use_trained_model = True
            logger.info("✅ 训练模型服务集成成功")

            # 显示模型信息
            if self.trained_model_service.metadata:
                metadata = self.trained_model_service.metadata
                logger.info(f"   模型: {metadata.model_name} v{metadata.model_version}")
                logger.info(f"   测试MAE: {metadata.test_mae:.4f}")
                if "three_stage" in metadata.config.get("pipeline_type", ""):
                    logger.info(f"   文化适配接受度: {metadata.config.get('stage1_acceptance', 0.0):.3f}")
                    logger.info("   🎯 使用三阶段训练管道模型")

        except Exception as e:
            logger.warning(f"训练模型服务集成失败，将使用原有模型: {e}")
            self.trained_model_service = None
            self.use_trained_model = False

    def predict_glucose_trend(
        self,
        user_id: str,
        prediction_hours: int = 6,
        use_trained_model: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        增强血糖趋势预测 - 可选择使用训练模型或原有模型

        Args:
            user_id: 用户ID
            prediction_hours: 预测小时数
            use_trained_model: 是否使用训练模型（None时自动选择）

        Returns:
            预测结果
        """
        # 决定使用哪个模型
        should_use_trained = (
            use_trained_model if use_trained_model is not None
            else self.use_trained_model
        )

        if should_use_trained and self.trained_model_service:
            try:
                return self._predict_with_trained_model(user_id, prediction_hours)
            except Exception as e:
                logger.warning(f"训练模型预测失败，回退到原有模型: {e}")
                return super().predict_glucose_trend(user_id, prediction_hours)
        else:
            return super().predict_glucose_trend(user_id, prediction_hours)

    def _predict_with_trained_model(
        self,
        user_id: str,
        prediction_hours: int = 6
    ) -> Dict[str, Any]:
        """
        使用训练模型进行预测

        Args:
            user_id: 用户ID
            prediction_hours: 预测小时数

        Returns:
            预测结果
        """
        try:
            if user_id not in self.user_profiles:
                raise ValueError(f"用户 {user_id} 不存在")

            user_profile = self.user_profiles[user_id]
            recent_glucose = self.glucose_records[user_id][-24:]  # 最近24个读数
            recent_meals = self.meal_records[user_id][-10:]  # 最近10餐

            if not recent_glucose:
                raise ValueError("没有足够的血糖数据进行预测")

            # 准备训练模型的输入数据
            glucose_sequence = [record['glucose_value'] for record in recent_glucose]
            timestamps = [record['timestamp'] for record in recent_glucose]

            # 用户特征
            user_features = {
                'age': user_profile.get('age', 30),
                'weight': user_profile.get('weight', 70),
                'height': user_profile.get('height', 170),
                'diabetes_type': user_profile.get('diabetes_type', 'type2'),
                'medication': user_profile.get('medication', [])
            }

            # 餐食信息
            meal_info = None
            if recent_meals:
                latest_meal = recent_meals[-1]
                meal_info = {
                    'meal_type': latest_meal.get('meal_type', 'unknown'),
                    'carbs': latest_meal.get('carbs', 0),
                    'calories': latest_meal.get('calories', 0),
                    'time_since_meal': self._calculate_time_since_meal(latest_meal)
                }

            # 创建预测输入
            prediction_input = PredictionInput(
                glucose_sequence=glucose_sequence,
                timestamps=timestamps,
                user_features=user_features,
                meal_info=meal_info
            )

            # 执行预测
            result = self.trained_model_service.predict(prediction_input)

            # 转换为系统标准格式
            prediction_times = [
                datetime.fromisoformat(t.replace('Z', '+00:00'))
                for t in result.prediction_times
            ]

            # 生成详细分析
            detailed_analysis = self._generate_detailed_analysis(
                result.predictions,
                result.risk_level,
                user_profile
            )

            return {
                'user_id': user_id,
                'prediction_timestamp': result.prediction_timestamp,
                'model_version': result.model_version,
                'predictions': [
                    {
                        'time': time.isoformat(),
                        'glucose_value': value,
                        'confidence': confidence,
                        'risk_level': self._assess_individual_risk(value)
                    }
                    for time, value, confidence in zip(
                        prediction_times,
                        result.predictions,
                        result.confidence_scores
                    )
                ],
                'overall_risk_level': result.risk_level,
                'recommendations': result.recommendations,
                'detailed_analysis': detailed_analysis,
                'model_source': 'trained_model'
            }

        except Exception as e:
            logger.error(f"训练模型预测失败，回退到原有模型: {e}")
            # 回退到原有模型
            return super().predict_glucose_trend(user_id, prediction_hours)

    def _calculate_time_since_meal(self, meal_record: Dict[str, Any]) -> float:
        """计算距离上次用餐的时间（小时）"""
        try:
            meal_time = datetime.fromisoformat(meal_record['timestamp'].replace('Z', '+00:00'))
            current_time = datetime.now(meal_time.tzinfo)
            return (current_time - meal_time).total_seconds() / 3600
        except Exception:
            return 2.0  # 默认2小时

    def _assess_individual_risk(self, glucose_value: float) -> str:
        """评估单个血糖值的风险等级"""
        if glucose_value > 15.0 or glucose_value < 3.0:
            return "高风险"
        elif glucose_value > 12.0 or glucose_value < 4.0:
            return "中风险"
        else:
            return "低风险"

    def _generate_detailed_analysis(
        self,
        predictions: List[float],
        risk_level: str,
        user_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成详细分析"""
        predictions_array = np.array(predictions)

        return {
            'trend_analysis': {
                'direction': self._analyze_trend_direction(predictions_array),
                'volatility': float(np.std(predictions_array)),
                'max_value': float(np.max(predictions_array)),
                'min_value': float(np.min(predictions_array)),
                'mean_value': float(np.mean(predictions_array))
            },
            'risk_factors': self._identify_risk_factors(predictions_array, user_profile),
            'time_in_range': self._calculate_time_in_range(predictions_array),
            'glycemic_variability': {
                'coefficient_of_variation': float(np.std(predictions_array) / np.mean(predictions_array) * 100),
                'mean_amplitude': float(np.max(predictions_array) - np.min(predictions_array))
            }
        }

    def _analyze_trend_direction(self, predictions: np.ndarray) -> str:
        """分析趋势方向"""
        if len(predictions) < 2:
            return "stable"

        # 计算线性趋势
        x = np.arange(len(predictions))
        slope = np.polyfit(x, predictions, 1)[0]

        if slope > 0.5:
            return "rising"
        elif slope < -0.5:
            return "falling"
        else:
            return "stable"

    def _identify_risk_factors(
        self,
        predictions: np.ndarray,
        user_profile: Dict[str, Any]
    ) -> List[str]:
        """识别风险因素"""
        risk_factors = []

        # 血糖值风险
        if np.any(predictions > 15.0):
            risk_factors.append("预测存在严重高血糖风险")
        elif np.any(predictions > 12.0):
            risk_factors.append("预测存在高血糖风险")

        if np.any(predictions < 3.0):
            risk_factors.append("预测存在严重低血糖风险")
        elif np.any(predictions < 4.0):
            risk_factors.append("预测存在低血糖风险")

        # 血糖变异性风险
        cv = np.std(predictions) / np.mean(predictions) * 100
        if cv > 36:
            risk_factors.append("血糖变异性过高")

        # 用户特征风险
        age = user_profile.get('age', 30)
        if age > 65:
            risk_factors.append("年龄相关风险")

        diabetes_type = user_profile.get('diabetes_type', 'type2')
        if diabetes_type == 'type1':
            risk_factors.append("1型糖尿病相关风险")

        return risk_factors

    def _calculate_time_in_range(self, predictions: np.ndarray) -> Dict[str, float]:
        """计算目标范围内时间百分比"""
        # 标准目标范围 3.9-10.0 mmol/L
        in_range = np.sum((predictions >= 3.9) & (predictions <= 10.0))
        above_range = np.sum(predictions > 10.0)
        below_range = np.sum(predictions < 3.9)

        total = len(predictions)

        return {
            'time_in_range': float(in_range / total * 100),
            'time_above_range': float(above_range / total * 100),
            'time_below_range': float(below_range / total * 100)
        }

    def get_model_comparison(self, user_id: str) -> Dict[str, Any]:
        """
        比较训练模型和原有模型的预测结果

        Args:
            user_id: 用户ID

        Returns:
            模型比较结果
        """
        try:
            # 使用训练模型预测
            trained_result = self.predict_glucose_trend(user_id, use_trained_model=True)

            # 使用原有模型预测
            original_result = self.predict_glucose_trend(user_id, use_trained_model=False)

            # 提取预测值进行比较
            trained_predictions = [p['glucose_value'] for p in trained_result['predictions']]
            original_predictions = [p['glucose_value'] for p in original_result['predictions']]

            # 计算差异
            differences = np.array(trained_predictions) - np.array(original_predictions)

            return {
                'user_id': user_id,
                'comparison_timestamp': datetime.now().isoformat(),
                'trained_model': {
                    'predictions': trained_predictions,
                    'risk_level': trained_result['overall_risk_level'],
                    'model_version': trained_result.get('model_version', 'unknown')
                },
                'original_model': {
                    'predictions': original_predictions,
                    'risk_level': original_result['overall_risk_level'],
                    'model_version': 'integrated_system'
                },
                'comparison_metrics': {
                    'mean_difference': float(np.mean(differences)),
                    'max_difference': float(np.max(np.abs(differences))),
                    'correlation': float(np.corrcoef(trained_predictions, original_predictions)[0, 1]),
                    'agreement_rate': float(np.sum(np.abs(differences) < 1.0) / len(differences))
                }
            }

        except Exception as e:
            logger.error(f"模型比较失败: {e}")
            raise

    def get_service_status(self) -> Dict[str, Any]:
        """
        获取服务状态

        Returns:
            服务状态信息
        """
        status = {
            'service_name': 'EnhancedDiabetesService',
            'timestamp': datetime.now().isoformat(),
            'original_service_available': True,
            'trained_model_available': self.use_trained_model,
            'active_users': len(self.user_profiles),
            'total_glucose_records': sum(len(records) for records in self.glucose_records.values()),
            'total_meal_records': sum(len(records) for records in self.meal_records.values())
        }

        if self.trained_model_service and self.trained_model_service.metadata:
            status['trained_model_info'] = {
                'name': self.trained_model_service.metadata.model_name,
                'version': self.trained_model_service.metadata.model_version,
                'training_date': self.trained_model_service.metadata.training_date,
                'performance': {
                    'val_loss': self.trained_model_service.metadata.best_val_loss,
                    'test_mae': self.trained_model_service.metadata.test_mae,
                    'test_rmse': self.trained_model_service.metadata.test_rmse
                }
            }

        return status

    def _handle_model_update(self, model_path: str):
        """处理模型更新"""
        logger.info(f"收到模型更新通知: {model_path}")

        try:
            # 重新加载训练模型服务
            if self.trained_model_service:
                model_loader = TrainedModelLoader()
                new_service = GlucosePredictionService(
                    model_loader=model_loader,
                    model_path=model_path,
                    sequence_length=24,
                    prediction_horizon=6
                )

                # 原子性更新
                old_service = self.trained_model_service
                self.trained_model_service = new_service

                logger.info(f"✅ 模型更新成功: {model_path}")

                # 显示新模型信息
                if new_service.metadata:
                    metadata = new_service.metadata
                    logger.info(f"   新模型: {metadata.model_name} v{metadata.model_version}")
                    logger.info(f"   测试MAE: {metadata.test_mae:.4f}")

        except Exception as e:
            logger.error(f"模型更新失败: {e}")

    async def trigger_model_training(self, job_type: str = 'incremental') -> Optional[str]:
        """触发模型训练"""
        if not self.model_lifecycle_service:
            logger.warning("模型生命周期服务不可用，无法触发训练")
            return None

        try:
            job_id = await self.model_lifecycle_service.trigger_training(job_type)
            logger.info(f"触发模型训练: {job_id}")
            return job_id
        except Exception as e:
            logger.error(f"触发模型训练失败: {e}")
            return None

    def get_model_status(self) -> Dict[str, Any]:
        """获取模型状态"""
        status = {
            'trained_model_available': self.use_trained_model,
            'lifecycle_service_available': self.model_lifecycle_service is not None
        }

        if self.trained_model_service and self.trained_model_service.metadata:
            metadata = self.trained_model_service.metadata
            status.update({
                'current_model': {
                    'name': metadata.model_name,
                    'version': metadata.model_version,
                    'path': metadata.model_path,
                    'test_mae': metadata.test_mae,
                    'training_date': metadata.training_date
                }
            })

        if self.model_lifecycle_service:
            status.update({
                'lifecycle_status': self.model_lifecycle_service.get_service_status(),
                'active_model': self.model_lifecycle_service.get_active_model_info(),
                'model_versions': self.model_lifecycle_service.get_model_versions()
            })

        return status


# 全局服务实例
_enhanced_diabetes_service: Optional[EnhancedDiabetesService] = None


def get_enhanced_diabetes_service() -> EnhancedDiabetesService:
    """
    获取增强糖尿病服务实例（单例）

    Returns:
        增强糖尿病服务实例
    """
    global _enhanced_diabetes_service

    if _enhanced_diabetes_service is None:
        try:
            _enhanced_diabetes_service = EnhancedDiabetesService(
                use_enhanced_gluformer=True,
                use_enhanced_moe=True,
                use_enhanced_multimodal=True
            )
            logger.info("增强糖尿病服务初始化成功")
        except Exception as e:
            logger.error(f"增强糖尿病服务初始化失败: {e}")
            raise

    return _enhanced_diabetes_service
