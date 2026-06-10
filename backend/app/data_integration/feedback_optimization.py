

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
反馈机制和迭代优化模块
实现实时反馈、模型微调和强化学习优化
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import pickle
from collections import deque
import random

logger = logging.getLogger(__name__)

@dataclass
class FeedbackConfig:
    """反馈配置"""
    feedback_window: int = 24  # 反馈窗口（小时）
    min_feedback_samples: int = 10  # 最小反馈样本数
    learning_rate: float = 0.001
    batch_size: int = 32
    update_frequency: int = 100  # 更新频率（样本数）
    confidence_threshold: float = 0.7  # 置信度阈值

class FeedbackCollector:
    """反馈收集器"""

    def __init__(self, config: FeedbackConfig):
        self.config = config
        self.feedback_buffer = deque(maxlen=1000)  # 反馈缓冲区
        self.prediction_buffer = deque(maxlen=1000)  # 预测缓冲区

    def add_prediction(self, prediction: Dict[str, Any], user_id: str):
        """添加预测记录"""
        prediction_record = {
            'timestamp': datetime.now(),
            'user_id': user_id,
            'prediction': prediction,
            'feedback_received': False
        }
        self.prediction_buffer.append(prediction_record)
        logger.debug(f"添加预测记录: {user_id}")

    def add_feedback(self, user_id: str, prediction_id: str,
                    actual_glucose: float, satisfaction_score: int,
                    additional_info: Optional[Dict[str, Any]] = None):
        """添加用户反馈"""
        feedback_record = {
            'timestamp': datetime.now(),
            'user_id': user_id,
            'prediction_id': prediction_id,
            'actual_glucose': actual_glucose,
            'satisfaction_score': satisfaction_score,  # 1-5分
            'additional_info': additional_info or {}
        }

        self.feedback_buffer.append(feedback_record)

        # 更新对应的预测记录
        for pred_record in self.prediction_buffer:
            if (pred_record['user_id'] == user_id and
                pred_record['timestamp'].isoformat() == prediction_id):
                pred_record['feedback_received'] = True
                pred_record['feedback'] = feedback_record
                break

        logger.info(f"收到反馈: {user_id}, 满意度: {satisfaction_score}")

    def get_feedback_data(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取反馈数据"""
        if user_id:
            return [fb for fb in self.feedback_buffer if fb['user_id'] == user_id]
        return list(self.feedback_buffer)

    def get_prediction_accuracy(self, user_id: Optional[str] = None) -> Dict[str, float]:
        """计算预测准确性"""
        feedback_data = self.get_feedback_data(user_id)

        if not feedback_data:
            return {'mae': 0.0, 'rmse': 0.0, 'accuracy_rate': 0.0}

        errors = []
        correct_predictions = 0

        for feedback in feedback_data:
            # 找到对应的预测
            for pred_record in self.prediction_buffer:
                if (pred_record['user_id'] == feedback['user_id'] and
                    pred_record['timestamp'].isoformat() == feedback['prediction_id']):

                    predicted_glucose = feedback['prediction']['predictions'][0]
                    actual_glucose = feedback['actual_glucose']

                    error = abs(predicted_glucose - actual_glucose)
                    errors.append(error)

                    # 血糖预测准确性（误差小于1.0 mmol/L认为准确）
                    if error < 1.0:
                        correct_predictions += 1
                    break

        if not errors:
            return {'mae': 0.0, 'rmse': 0.0, 'accuracy_rate': 0.0}

        mae = np.mean(errors)
        rmse = np.sqrt(np.mean([e**2 for e in errors]))
        accuracy_rate = correct_predictions / len(errors)

        return {
            'mae': mae,
            'rmse': rmse,
            'accuracy_rate': accuracy_rate
        }

class ModelFineTuner:
    """模型微调器"""

    def __init__(self, model: nn.Module, config: FeedbackConfig):
        self.model = model
        self.config = config
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )
        self.loss_fn = nn.MSELoss()

    def fine_tune(self, feedback_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """基于反馈微调模型"""
        if len(feedback_data) < self.config.min_feedback_samples:
            logger.warning(f"反馈样本不足: {len(feedback_data)} < {self.config.min_feedback_samples}")
            return {'success': False, 'reason': 'insufficient_samples'}

        try:
            # 准备训练数据
            train_data, train_labels = self._prepare_training_data(feedback_data)

            if len(train_data) == 0:
                return {'success': False, 'reason': 'no_valid_data'}

            # 微调训练
            self.model.train()
            losses = []

            for epoch in range(10):  # 少量epoch避免过拟合
                epoch_loss = 0

                for i in range(0, len(train_data), self.config.batch_size):
                    batch_data = train_data[i:i+self.config.batch_size]
                    batch_labels = train_labels[i:i+self.config.batch_size]

                    # 转换为张量
                    batch_data = torch.FloatTensor(batch_data)
                    batch_labels = torch.FloatTensor(batch_labels).unsqueeze(1)

                    # 前向传播
                    predictions = self.model(batch_data)
                    loss = self.loss_fn(predictions, batch_labels)

                    # 反向传播
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()

                losses.append(epoch_loss / (len(train_data) // self.config.batch_size))
                self.scheduler.step()

            logger.info(f"模型微调完成，最终损失: {losses[-1]:.4f}")

            return {
                'success': True,
                'final_loss': losses[-1],
                'loss_history': losses,
                'samples_used': len(train_data)
            }

        except Exception as e:
            logger.error(f"模型微调失败: {e}")
            return {'success': False, 'reason': str(e)}

    def _prepare_training_data(self, feedback_data: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """准备训练数据"""
        features = []
        labels = []

        for feedback in feedback_data:
            try:
                # 提取特征（这里需要根据实际模型输入调整）
                feature_vector = self._extract_features_from_feedback(feedback)
                if feature_vector is not None:
                    features.append(feature_vector)
                    labels.append(feedback['actual_glucose'])
            except Exception as e:
                logger.warning(f"处理反馈数据失败: {e}")
                continue

        return np.array(features), np.array(labels)

    def _extract_features_from_feedback(self, feedback: Dict[str, Any]) -> Optional[np.ndarray]:
        """从反馈中提取特征"""
        try:
            # 这里需要根据实际的特征工程逻辑调整
            # 示例：提取时间、用户信息等特征
            features = [
                feedback['timestamp'].hour,
                feedback['timestamp'].dayofweek,
                feedback['additional_info'].get('age', 45),
                feedback['additional_info'].get('bmi', 25),
                feedback['additional_info'].get('exercise_level', 3),
                feedback['additional_info'].get('stress_level', 3),
                feedback['satisfaction_score']
            ]

            return np.array(features)
        except Exception as e:
            logger.error(f"特征提取失败: {e}")
            return None

class ReinforcementLearningOptimizer:
    """强化学习优化器"""

    def __init__(self, state_dim: int, action_dim: int, config: FeedbackConfig):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config

        # Q网络
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

        self.target_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)
        self.epsilon = 0.1  # 探索率

    def select_action(self, state: np.ndarray) -> int:
        """选择动作"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action = q_values.argmax().item()

        return action

    def store_experience(self, state: np.ndarray, action: int,
                        reward: float, next_state: np.ndarray, done: bool):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))

    def calculate_reward(self, prediction: Dict[str, Any],
                        actual_glucose: float, satisfaction_score: int) -> float:
        """计算奖励"""
        predicted_glucose = prediction['predictions'][0]

        # 预测准确性奖励
        accuracy_reward = max(0, 1 - abs(predicted_glucose - actual_glucose) / 5.0)

        # 用户满意度奖励
        satisfaction_reward = satisfaction_score / 5.0

        # 综合奖励
        total_reward = 0.7 * accuracy_reward + 0.3 * satisfaction_reward

        return total_reward

    def train(self, batch_size: int = 32):
        """训练Q网络"""
        if len(self.memory) < batch_size:
            return

        # 随机采样经验
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.BoolTensor(dones)

        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            max_next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + 0.99 * max_next_q_values * (~dones)

        # 计算当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # 计算损失
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新目标网络
        if len(self.memory) % 100 == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def optimize_recommendations(self, user_state: np.ndarray) -> Dict[str, Any]:
        """优化推荐策略"""
        action = self.select_action(user_state)

        # 根据动作生成推荐
        recommendations = {
            0: "增加运动量，特别是餐后散步",
            1: "减少精制碳水化合物摄入",
            2: "增加蛋白质和健康脂肪摄入",
            3: "改善睡眠质量，保持规律作息",
            4: "管理压力，尝试冥想或深呼吸"
        }

        return {
            'action': action,
            'recommendation': recommendations.get(action, "保持当前生活方式"),
            'confidence': 0.8
        }

class IterativeOptimizationSystem:
    """增强的迭代优化系统"""

    def __init__(self, glucose_predictor, config: FeedbackConfig):
        self.glucose_predictor = glucose_predictor
        self.config = config

        # 初始化组件
        self.feedback_collector = FeedbackCollector(config)
        self.model_fine_tuner = ModelFineTuner(glucose_predictor.model, config)
        self.rl_optimizer = ReinforcementLearningOptimizer(
            state_dim=10,  # 根据实际状态维度调整
            action_dim=5,  # 推荐动作数量
            config=config
        )

        # 新增组件
        self.adaptive_manager = None  # 将在集成时设置
        self.workflow_orchestrator = None  # 将在集成时设置
        self.data_flow_manager = None  # 将在集成时设置

        self.update_counter = 0
        self.optimization_history = []
        self.performance_metrics = defaultdict(list)

    def set_integration_components(self, adaptive_manager=None, workflow_orchestrator=None, data_flow_manager=None):
        """设置集成组件"""
        self.adaptive_manager = adaptive_manager
        self.workflow_orchestrator = workflow_orchestrator
        self.data_flow_manager = data_flow_manager
        logger.info("反馈优化系统集成组件已设置")

    def process_prediction(self, user_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理预测请求 - 增强版本"""
        start_time = datetime.now()

        try:
            # 1. 优先使用自适应模型进行预测
            if self.adaptive_manager:
                prediction = self.adaptive_manager.predict_with_adaptation(user_id, input_data)
            else:
                prediction = self.glucose_predictor.predict(input_data)

            # 2. 记录预测
            self.feedback_collector.add_prediction(prediction, user_id)

            # 3. 如果有工作流集成，触发相关工作流
            if self.workflow_orchestrator:
                try:
                    workflow_data = {
                        'user_id': user_id,
                        'prediction': prediction,
                        'input_data': input_data,
                        'timestamp': start_time.isoformat()
                    }
                    # 异步触发工作流，不阻塞主流程
                    asyncio.create_task(
                        self.workflow_orchestrator.trigger_workflow('prediction_post_process', workflow_data)
                    )
                except Exception as e:
                    logger.warning(f"工作流触发失败: {e}")

            # 4. 检查是否需要更新模型
            if self.update_counter % self.config.update_frequency == 0:
                self._update_model_with_adaptation(user_id)

            # 5. 记录性能指标
            processing_time = (datetime.now() - start_time).total_seconds()
            self.performance_metrics['prediction_time'].append(processing_time)
            self.performance_metrics['prediction_count'] += 1

            self.update_counter += 1

            # 6. 增强预测结果
            enhanced_prediction = self._enhance_prediction_result(prediction, user_id, processing_time)

            return enhanced_prediction

        except Exception as e:
            logger.error(f"预测处理失败: {e}")
            return {'error': str(e), 'user_id': user_id}

    def process_feedback(self, user_id: str, prediction_id: str,
                        actual_glucose: float, satisfaction_score: int,
                        additional_info: Optional[Dict[str, Any]] = None):
        """处理用户反馈"""
        # 添加反馈
        self.feedback_collector.add_feedback(
            user_id, prediction_id, actual_glucose,
            satisfaction_score, additional_info
        )

        # 计算奖励并更新强化学习
        feedback_data = self.feedback_collector.get_feedback_data(user_id)
        if feedback_data:
            latest_feedback = feedback_data[-1]

            # 找到对应的预测
            for pred_record in self.feedback_collector.prediction_buffer:
                if (pred_record['user_id'] == user_id and
                    pred_record['timestamp'].isoformat() == prediction_id):

                    reward = self.rl_optimizer.calculate_reward(
                        pred_record['prediction'],
                        actual_glucose,
                        satisfaction_score
                    )

                    # 存储经验（这里需要构造状态）
                    state = self._construct_state(additional_info or {})
                    next_state = state  # 简化处理

                    self.rl_optimizer.store_experience(
                        state, 0, reward, next_state, False
                    )
                    break

        # 训练强化学习模型
        self.rl_optimizer.train()

    def get_optimized_recommendations(self, user_id: str,
                                    user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """获取优化后的推荐"""
        # 构造用户状态
        user_state = self._construct_state(user_profile)

        # 获取强化学习优化推荐
        rl_recommendation = self.rl_optimizer.optimize_recommendations(user_state)

        # 结合历史反馈
        feedback_data = self.feedback_collector.get_feedback_data(user_id)
        accuracy_stats = self.feedback_collector.get_prediction_accuracy(user_id)

        return {
            'recommendation': rl_recommendation['recommendation'],
            'confidence': rl_recommendation['confidence'],
            'personalization_level': min(len(feedback_data) / 50, 1.0),
            'accuracy_stats': accuracy_stats,
            'optimization_status': 'active' if len(feedback_data) > 10 else 'learning'
        }

    def _enhance_prediction_result(self, prediction: Dict[str, Any], user_id: str, processing_time: float) -> Dict[str, Any]:
        """增强预测结果"""
        enhanced = prediction.copy()

        # 添加处理时间
        enhanced['processing_time'] = processing_time

        # 添加个性化信息
        if self.adaptive_manager:
            adaptation_report = self.adaptive_manager.get_adaptation_report(user_id)
            enhanced['personalization_level'] = min(adaptation_report.get('total_adaptations', 0) / 10, 1.0)
            enhanced['has_personalized_model'] = adaptation_report.get('has_personalized_model', False)

        # 添加系统状态
        enhanced['system_status'] = {
            'total_predictions': self.update_counter,
            'recent_performance': self.performance_metrics.get('prediction_time', [])[-5:],
            'optimization_active': len(self.optimization_history) > 0
        }

        return enhanced

    def _update_model_with_adaptation(self, user_id: str):
        """增强的模型更新，结合自适应学习"""
        feedback_data = self.feedback_collector.get_feedback_data()

        if len(feedback_data) >= self.config.min_feedback_samples:
            logger.info("开始增强模型更新...")

            # 1. 传统微调
            fine_tune_result = self.model_fine_tuner.fine_tune(feedback_data)

            # 2. 如果有自适应管理器，触发用户特定的适应
            if self.adaptive_manager and user_id:
                try:
                    user_feedback = self.feedback_collector.get_feedback_data(user_id)
                    if len(user_feedback) >= 5:  # 用户特定数据足够
                        # 转换反馈数据为适应数据格式
                        adaptation_data = self._convert_feedback_to_adaptation_data(user_feedback)
                        adaptation_result = self.adaptive_manager.adapt_for_user(user_id, adaptation_data)
                        logger.info(f"用户 {user_id} 自适应更新: {adaptation_result}")
                except Exception as e:
                    logger.error(f"用户 {user_id} 自适应更新失败: {e}")

            # 3. 记录优化历史
            optimization_record = {
                'timestamp': datetime.now(),
                'feedback_samples': len(feedback_data),
                'fine_tune_result': fine_tune_result,
                'user_id': user_id,
                'update_counter': self.update_counter
            }
            self.optimization_history.append(optimization_record)

            if fine_tune_result['success']:
                logger.info("增强模型更新成功")
            else:
                logger.warning(f"模型更新失败: {fine_tune_result['reason']}")

    def _convert_feedback_to_adaptation_data(self, feedback_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """将反馈数据转换为适应数据格式"""
        adaptation_data = []

        for feedback in feedback_data:
            try:
                adaptation_record = {
                    'user_id': feedback['user_id'],
                    'glucose': feedback['actual_glucose'],
                    'target_glucose': feedback['actual_glucose'],
                    'satisfaction_score': feedback['satisfaction_score'],
                    'timestamp': feedback['timestamp']
                }

                # 从additional_info中提取更多特征
                if 'additional_info' in feedback and feedback['additional_info']:
                    additional = feedback['additional_info']
                    adaptation_record.update({
                        'carbohydrates': additional.get('carbohydrates', 0),
                        'protein': additional.get('protein', 0),
                        'fat': additional.get('fat', 0),
                        'exercise_duration': additional.get('exercise_duration', 0),
                        'stress_level': additional.get('stress_level', 3),
                        'sleep_hours': additional.get('sleep_hours', 7),
                        'hour': feedback['timestamp'].hour,
                        'day_of_week': feedback['timestamp'].weekday(),
                        'is_weekend': feedback['timestamp'].weekday() >= 5
                    })

                adaptation_data.append(adaptation_record)

            except Exception as e:
                logger.warning(f"转换反馈数据失败: {e}")
                continue

        return adaptation_data

    def _update_model(self):
        """更新模型（保留原有方法以兼容）"""
        self._update_model_with_adaptation(None)

    def _construct_state(self, user_profile: Dict[str, Any]) -> np.ndarray:
        """构造用户状态"""
        state_features = [
            user_profile.get('age', 45),
            user_profile.get('bmi', 25),
            user_profile.get('exercise_level', 3),
            user_profile.get('stress_level', 3),
            user_profile.get('sleep_hours', 7),
            user_profile.get('diet_quality', 3),
            user_profile.get('medication_adherence', 4),
            user_profile.get('glucose_variability', 2),
            user_profile.get('recent_trend', 0),
            user_profile.get('risk_factors', 2)
        ]

        return np.array(state_features, dtype=np.float32)

    def get_system_stats(self) -> Dict[str, Any]:
        """获取增强的系统统计信息"""
        total_feedback = len(self.feedback_collector.feedback_buffer)
        total_predictions = len(self.feedback_collector.prediction_buffer)

        accuracy_stats = self.feedback_collector.get_prediction_accuracy()

        # 基础统计
        base_stats = {
            'total_feedback_count': total_feedback,
            'total_predictions_count': total_predictions,
            'feedback_rate': total_feedback / max(total_predictions, 1),
            'prediction_accuracy': accuracy_stats,
            'model_update_counter': self.update_counter,
            'rl_epsilon': self.rl_optimizer.epsilon,
            'system_status': 'active'
        }

        # 增强统计
        enhanced_stats = {
            'optimization_history_count': len(self.optimization_history),
            'average_processing_time': np.mean(self.performance_metrics.get('prediction_time', [0])),
            'recent_optimization_count': len([h for h in self.optimization_history
                                           if (datetime.now() - h['timestamp']).days <= 7])
        }

        # 自适应管理器统计
        if self.adaptive_manager:
            adaptation_stats = self.adaptive_manager.get_adaptation_report()
            enhanced_stats.update({
                'total_personalized_users': adaptation_stats.get('total_users', 0),
                'adaptation_success_rate': adaptation_stats.get('success_rate', 0),
                'average_adaptation_time': adaptation_stats.get('average_adaptation_time', 0)
            })

        # 工作流统计
        if self.workflow_orchestrator:
            workflow_stats = self.workflow_orchestrator.get_workflow_status()
            enhanced_stats.update({
                'registered_workflows': len(workflow_stats.get('registered_workflows', [])),
                'active_schedules': len(workflow_stats.get('active_schedules', [])),
                'processor_status': workflow_stats.get('processor_status', False)
            })

        # 数据流统计
        if self.data_flow_manager:
            dataflow_stats = self.data_flow_manager.get_system_status()
            enhanced_stats.update({
                'data_adapters': len(dataflow_stats.get('registered_adapters', [])),
                'buffer_size': dataflow_stats.get('buffer_size', 0),
                'processing_stats': dataflow_stats.get('processing_stats', {})
            })

        base_stats.update(enhanced_stats)
        return base_stats

# 使用示例
def main():
    """使用示例"""
    # 创建配置
    config = FeedbackConfig(
        feedback_window=24,
        min_feedback_samples=10,
        learning_rate=0.001
    )

    # 模拟血糖预测器
    class MockGlucosePredictor:
        def __init__(self):
            self.model = nn.Linear(10, 1)

        def predict(self, data):
            return {
                'predictions': [6.5],
                'confidence': 0.8,
                'recommendations': ['保持健康饮食']
            }

    # 创建迭代优化系统
    glucose_predictor = MockGlucosePredictor()
    optimization_system = IterativeOptimizationSystem(glucose_predictor, config)

    # 模拟使用流程
    user_id = "user_123"
    input_data = {'age': 45, 'bmi': 25, 'glucose': 6.0}

    # 1. 获取预测
    prediction = optimization_system.process_prediction(user_id, input_data)
    print(f"预测结果: {prediction}")

    # 2. 模拟用户反馈
    optimization_system.process_feedback(
        user_id,
        datetime.now().isoformat(),
        6.2,  # 实际血糖值
        4,    # 满意度评分
        {'age': 45, 'bmi': 25}
    )

    # 3. 获取优化推荐
    recommendations = optimization_system.get_optimized_recommendations(
        user_id, {'age': 45, 'bmi': 25}
    )
    print(f"优化推荐: {recommendations}")

    # 4. 获取系统统计
    stats = optimization_system.get_system_stats()
    print(f"系统统计: {stats}")

if __name__ == "__main__":
    main()

__all__ = ["'logger'", "'FeedbackConfig'", "'FeedbackCollector'", "'ModelFineTuner'", "'ReinforcementLearningOptimizer'", "'IterativeOptimizationSystem'", "'main'"]
