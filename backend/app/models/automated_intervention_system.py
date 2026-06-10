

"""
自动化健康干预策略与建议模块
基于用户建议的改进方向设计
实现预测性干预和自适应干预机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
from collections import deque
from datetime import datetime, timedelta
import random
import json

logger = logging.getLogger(__name__)

class PredictiveInterventionEngine(nn.Module):
    """
    预测性干预引擎
    基于用户建议的改进方向设计
    基于当前健康状态和历史数据预测未来健康趋势并提供干预措施
    """

    def __init__(self,
                 input_dim: int = 256,
                 hidden_dim: int = 128,
                 prediction_horizon: int = 24,  # 24小时预测
                 num_intervention_types: int = 5,
                 dropout: float = 0.1):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.prediction_horizon = prediction_horizon
        self.num_intervention_types = num_intervention_types

        # 健康状态编码器
        self.health_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 时序预测网络
        self.temporal_predictor = nn.LSTM(
            hidden_dim, hidden_dim,
            batch_first=True, dropout=dropout
        )

        # 趋势预测头
        self.trend_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, prediction_horizon),
            nn.Tanh()
        )

        # 风险预测头
        self.risk_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # 干预策略生成器
        self.intervention_generator = nn.Sequential(
            nn.Linear(hidden_dim + prediction_horizon, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_intervention_types),
            nn.Softmax(dim=-1)
        )

        # 干预效果预测器
        self.effect_predictor = nn.Sequential(
            nn.Linear(hidden_dim + num_intervention_types, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # 历史数据存储
        self.historical_data = deque(maxlen=1000)

        # 干预历史
        self.intervention_history = deque(maxlen=500)

    def forward(self,
                current_health: torch.Tensor,
                historical_data: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 健康状态编码
        encoded_health = self.health_encoder(current_health)

        # 时序预测
        if historical_data is not None:
            historical_encoded = self.health_encoder(historical_data)
            temporal_output, _ = self.temporal_predictor(historical_encoded)
            temporal_feature = temporal_output[:, -1, :]  # 取最后一个时间步
        else:
            temporal_feature = encoded_health

        # 趋势预测
        trend_prediction = self.trend_predictor(temporal_feature)

        # 风险预测
        risk_prediction = self.risk_predictor(temporal_feature)

        # 干预策略生成
        intervention_input = torch.cat([temporal_feature, trend_prediction], dim=-1)
        intervention_strategy = self.intervention_generator(intervention_input)

        # 干预效果预测
        effect_input = torch.cat([temporal_feature, intervention_strategy], dim=-1)
        intervention_effect = self.effect_predictor(effect_input)

        return {
            'trend_prediction': trend_prediction,
            'risk_prediction': risk_prediction,
            'intervention_strategy': intervention_strategy,
            'intervention_effect': intervention_effect,
            'encoded_health': encoded_health,
            'temporal_feature': temporal_feature
        }

    def predict_health_trend(self,
                            current_health: Dict[str, Any],
                            hours_ahead: int = 24) -> Dict[str, Any]:
        """预测健康趋势"""
        # 转换为张量
        health_tensor = self._health_dict_to_tensor(current_health)

        # 前向传播
        with torch.no_grad():
            result = self.forward(health_tensor)

        # 解析预测结果
        trend_prediction = result['trend_prediction'].squeeze().numpy()
        risk_prediction = result['risk_prediction'].item()

        # 生成趋势分析
        trend_analysis = self._analyze_trend(trend_prediction, hours_ahead)

        # 生成干预建议
        intervention_suggestions = self._generate_intervention_suggestions(
            trend_analysis, risk_prediction
        )

        return {
            'trend_prediction': trend_prediction.tolist(),
            'risk_level': self._risk_level_to_string(risk_prediction),
            'trend_analysis': trend_analysis,
            'intervention_suggestions': intervention_suggestions,
            'prediction_horizon': hours_ahead,
            'timestamp': datetime.now()
        }

    def _health_dict_to_tensor(self, health_data: Dict[str, Any]) -> torch.Tensor:
        """将健康数据字典转换为张量"""
        # 提取关键健康指标
        health_features = [
            health_data.get('glucose', 0.0),
            health_data.get('blood_pressure_systolic', 0.0),
            health_data.get('blood_pressure_diastolic', 0.0),
            health_data.get('heart_rate', 0.0),
            health_data.get('weight', 0.0),
            health_data.get('bmi', 0.0),
            health_data.get('exercise_minutes', 0.0),
            health_data.get('stress_level', 0.0),
            health_data.get('sleep_hours', 0.0),
            health_data.get('medication_dose', 0.0),
            health_data.get('carbohydrates', 0.0),
            health_data.get('protein', 0.0),
            health_data.get('fat', 0.0),
            health_data.get('fiber', 0.0),
            health_data.get('water_intake', 0.0),
            health_data.get('age', 0.0),
            health_data.get('gender', 0.0),
            health_data.get('diabetes_type', 0.0),
            health_data.get('medication_type', 0.0),
            health_data.get('exercise_type', 0.0)
        ]

        # 填充到input_dim
        while len(health_features) < self.input_dim:
            health_features.append(0.0)

        return torch.tensor(health_features[:self.input_dim], dtype=torch.float32).unsqueeze(0)

    def _analyze_trend(self,
                       trend_prediction: np.ndarray,
                       hours_ahead: int) -> Dict[str, Any]:
        """分析趋势"""
        # 计算趋势方向
        if len(trend_prediction) > 1:
            trend_direction = 'increasing' if trend_prediction[-1] > trend_prediction[0] else 'decreasing'
            trend_magnitude = abs(trend_prediction[-1] - trend_prediction[0])
        else:
            trend_direction = 'stable'
            trend_magnitude = 0.0

        # 识别关键时间点
        critical_points = []
        for i in range(1, len(trend_prediction) - 1):
            if (trend_prediction[i] > trend_prediction[i-1] and
                trend_prediction[i] > trend_prediction[i+1]) or \
               (trend_prediction[i] < trend_prediction[i-1] and
                trend_prediction[i] < trend_prediction[i+1]):
                critical_points.append({
                    'hour': i,
                    'value': trend_prediction[i],
                    'type': 'peak' if trend_prediction[i] > trend_prediction[i-1] else 'valley'
                })

        return {
            'direction': trend_direction,
            'magnitude': trend_magnitude,
            'critical_points': critical_points,
            'max_value': float(np.max(trend_prediction)),
            'min_value': float(np.min(trend_prediction)),
            'average_value': float(np.mean(trend_prediction))
        }

    def _generate_intervention_suggestions(self,
                                          trend_analysis: Dict[str, Any],
                                          risk_prediction: float) -> List[Dict[str, Any]]:
        """生成干预建议"""
        suggestions = []

        # 基于风险水平的建议
        if risk_prediction > 0.8:
            suggestions.append({
                'type': 'urgent',
                'priority': 'high',
                'message': '高风险状态检测，建议立即就医',
                'intervention': 'seek_medical_attention',
                'timeline': 'immediate'
            })
        elif risk_prediction > 0.6:
            suggestions.append({
                'type': 'warning',
                'priority': 'medium',
                'message': '中等风险状态，建议调整生活方式',
                'intervention': 'lifestyle_adjustment',
                'timeline': 'within_2_hours'
            })

        # 基于趋势的建议
        if trend_analysis['direction'] == 'increasing':
            if trend_analysis['magnitude'] > 0.5:
                suggestions.append({
                    'type': 'dietary',
                    'priority': 'high',
                    'message': '血糖上升趋势明显，建议减少碳水化合物摄入',
                    'intervention': 'reduce_carbohydrates',
                    'timeline': 'within_1_hour'
                })
            else:
                suggestions.append({
                    'type': 'exercise',
                    'priority': 'medium',
                    'message': '血糖轻微上升，建议进行轻度运动',
                    'intervention': 'light_exercise',
                    'timeline': 'within_2_hours'
                })
        elif trend_analysis['direction'] == 'decreasing':
            if trend_analysis['magnitude'] > 0.5:
                suggestions.append({
                    'type': 'dietary',
                    'priority': 'high',
                    'message': '血糖下降趋势明显，建议补充碳水化合物',
                    'intervention': 'increase_carbohydrates',
                    'timeline': 'within_30_minutes'
                })

        # 基于关键时间点的建议
        for point in trend_analysis['critical_points']:
            if point['type'] == 'peak':
                suggestions.append({
                    'type': 'monitoring',
                    'priority': 'medium',
                    'message': f"预计在{point['hour']}小时后血糖达到峰值",
                    'intervention': 'increase_monitoring',
                    'timeline': f'within_{point["hour"]}_hours'
                })

        return suggestions

    def _risk_level_to_string(self, risk_score: float) -> str:
        """将风险分数转换为字符串"""
        if risk_score > 0.8:
            return 'high'
        elif risk_score > 0.6:
            return 'medium'
        elif risk_score > 0.4:
            return 'low'
        else:
            return 'very_low'

class AdaptiveInterventionMechanism(nn.Module):
    """
    自适应干预机制
    基于用户建议的改进方向设计
    当用户健康状况发生变化时自动调整干预策略
    """

    def __init__(self,
                 state_dim: int = 256,
                 action_dim: int = 10,
                 hidden_dim: int = 128,
                 adaptation_rate: float = 0.01,
                 memory_size: int = 1000):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.adaptation_rate = adaptation_rate

        # 状态变化检测器
        self.change_detector = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # 干预策略网络
        self.intervention_policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

        # 策略调整网络
        self.policy_adjuster = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

        # 效果评估网络
        self.effect_evaluator = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # 用户偏好学习
        self.preference_learner = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

        # 记忆存储
        self.memory = deque(maxlen=memory_size)

        # 用户状态历史
        self.user_state_history = deque(maxlen=100)

        # 干预效果历史
        self.intervention_effect_history = deque(maxlen=500)

    def forward(self,
                current_state: torch.Tensor,
                previous_state: Optional[torch.Tensor] = None,
                previous_action: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 检测状态变化
        if previous_state is not None:
            state_change_input = torch.cat([current_state, previous_state], dim=-1)
            change_detected = self.change_detector(state_change_input)
        else:
            change_detected = torch.tensor(0.0)

        # 生成干预策略
        intervention_policy = self.intervention_policy(current_state)

        # 策略调整
        if previous_action is not None:
            adjustment_input = torch.cat([current_state, previous_action], dim=-1)
            policy_adjustment = self.policy_adjuster(adjustment_input)
            adjusted_policy = intervention_policy + policy_adjustment * self.adaptation_rate
            adjusted_policy = F.softmax(adjusted_policy, dim=-1)
        else:
            adjusted_policy = intervention_policy

        # 效果评估
        if previous_action is not None:
            effect_input = torch.cat([current_state, previous_action], dim=-1)
            intervention_effect = self.effect_evaluator(effect_input)
        else:
            intervention_effect = torch.tensor(0.5)

        # 用户偏好学习
        if previous_action is not None:
            preference_input = torch.cat([current_state, previous_action], dim=-1)
            user_preferences = self.preference_learner(preference_input)
        else:
            user_preferences = torch.ones(self.action_dim) / self.action_dim

        return {
            'change_detected': change_detected,
            'intervention_policy': intervention_policy,
            'adjusted_policy': adjusted_policy,
            'intervention_effect': intervention_effect,
            'user_preferences': user_preferences
        }

    def detect_health_change(self,
                             current_health: Dict[str, Any],
                             previous_health: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """检测健康状态变化"""
        if previous_health is None:
            return {
                'change_detected': False,
                'change_magnitude': 0.0,
                'change_type': 'none'
            }

        # 转换为张量
        current_tensor = self._health_dict_to_tensor(current_health)
        previous_tensor = self._health_dict_to_tensor(previous_health)

        # 检测变化
        with torch.no_grad():
            result = self.forward(current_tensor, previous_tensor)

        change_detected = result['change_detected'].item() > 0.5

        # 计算变化幅度
        change_magnitude = torch.norm(current_tensor - previous_tensor).item()

        # 确定变化类型
        change_type = self._classify_change_type(current_health, previous_health)

        return {
            'change_detected': change_detected,
            'change_magnitude': change_magnitude,
            'change_type': change_type,
            'change_score': result['change_detected'].item()
        }

    def _health_dict_to_tensor(self, health_data: Dict[str, Any]) -> torch.Tensor:
        """将健康数据字典转换为张量"""
        health_features = [
            health_data.get('glucose', 0.0),
            health_data.get('blood_pressure_systolic', 0.0),
            health_data.get('heart_rate', 0.0),
            health_data.get('weight', 0.0),
            health_data.get('exercise_minutes', 0.0),
            health_data.get('stress_level', 0.0),
            health_data.get('sleep_hours', 0.0),
            health_data.get('medication_dose', 0.0)
        ]

        # 填充到state_dim
        while len(health_features) < self.state_dim:
            health_features.append(0.0)

        return torch.tensor(health_features[:self.state_dim], dtype=torch.float32).unsqueeze(0)

    def _classify_change_type(self,
                              current_health: Dict[str, Any],
                              previous_health: Dict[str, Any]) -> str:
        """分类变化类型"""
        # 计算关键指标的变化
        glucose_change = current_health.get('glucose', 0) - previous_health.get('glucose', 0)
        bp_change = current_health.get('blood_pressure_systolic', 0) - previous_health.get('blood_pressure_systolic', 0)
        hr_change = current_health.get('heart_rate', 0) - previous_health.get('heart_rate', 0)

        # 确定主要变化类型
        if abs(glucose_change) > 2.0:
            return 'glucose_spike' if glucose_change > 0 else 'glucose_drop'
        elif abs(bp_change) > 20:
            return 'blood_pressure_change'
        elif abs(hr_change) > 20:
            return 'heart_rate_change'
        else:
            return 'minor_change'

    def adapt_intervention_strategy(self,
                                   current_health: Dict[str, Any],
                                   previous_health: Optional[Dict[str, Any]] = None,
                                   previous_intervention: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """自适应调整干预策略"""
        # 转换为张量
        current_tensor = self._health_dict_to_tensor(current_health)
        previous_tensor = self._health_dict_to_tensor(previous_health) if previous_health else None
        previous_action = self._intervention_to_tensor(previous_intervention) if previous_intervention else None

        # 前向传播
        with torch.no_grad():
            result = self.forward(current_tensor, previous_tensor, previous_action)

        # 生成新的干预策略
        intervention_strategy = result['adjusted_policy'].squeeze().numpy()

        # 选择最佳干预
        best_intervention_idx = np.argmax(intervention_strategy)
        best_intervention = self._get_intervention_by_index(best_intervention_idx)

        # 考虑用户偏好
        user_preferences = result['user_preferences'].squeeze().numpy()
        preference_adjusted_strategy = intervention_strategy * user_preferences
        preference_adjusted_strategy = preference_adjusted_strategy / np.sum(preference_adjusted_strategy)

        # 选择偏好调整后的最佳干预
        preference_best_idx = np.argmax(preference_adjusted_strategy)
        preference_best_intervention = self._get_intervention_by_index(preference_best_idx)

        return {
            'primary_intervention': best_intervention,
            'preference_adjusted_intervention': preference_best_intervention,
            'intervention_strategy': intervention_strategy.tolist(),
            'user_preferences': user_preferences.tolist(),
            'intervention_effect': result['intervention_effect'].item(),
            'change_detected': result['change_detected'].item() > 0.5
        }

    def _intervention_to_tensor(self, intervention: Dict[str, Any]) -> torch.Tensor:
        """将干预转换为张量"""
        intervention_vector = [0.0] * self.action_dim

        intervention_type = intervention.get('type', 'none')
        intervention_mapping = {
            'dietary': 0,
            'exercise': 1,
            'medication': 2,
            'lifestyle': 3,
            'monitoring': 4,
            'medical_attention': 5,
            'stress_management': 6,
            'sleep_improvement': 7,
            'hydration': 8,
            'none': 9
        }

        if intervention_type in intervention_mapping:
            intervention_vector[intervention_mapping[intervention_type]] = 1.0

        return torch.tensor(intervention_vector, dtype=torch.float32).unsqueeze(0)

    def _get_intervention_by_index(self, index: int) -> Dict[str, Any]:
        """根据索引获取干预"""
        interventions = [
            {
                'type': 'dietary',
                'name': '饮食调整',
                'description': '调整饮食结构和营养成分',
                'priority': 'high'
            },
            {
                'type': 'exercise',
                'name': '运动干预',
                'description': '增加或调整运动计划',
                'priority': 'medium'
            },
            {
                'type': 'medication',
                'name': '药物调整',
                'description': '调整药物剂量或类型',
                'priority': 'high'
            },
            {
                'type': 'lifestyle',
                'name': '生活方式调整',
                'description': '调整日常生活习惯',
                'priority': 'medium'
            },
            {
                'type': 'monitoring',
                'name': '监测加强',
                'description': '增加健康指标监测频率',
                'priority': 'low'
            },
            {
                'type': 'medical_attention',
                'name': '医疗关注',
                'description': '寻求专业医疗建议',
                'priority': 'high'
            },
            {
                'type': 'stress_management',
                'name': '压力管理',
                'description': '进行压力缓解活动',
                'priority': 'medium'
            },
            {
                'type': 'sleep_improvement',
                'name': '睡眠改善',
                'description': '改善睡眠质量和时长',
                'priority': 'medium'
            },
            {
                'type': 'hydration',
                'name': '水分补充',
                'description': '增加水分摄入',
                'priority': 'low'
            },
            {
                'type': 'none',
                'name': '无干预',
                'description': '当前状态良好，无需干预',
                'priority': 'none'
            }
        ]

        return interventions[index] if index < len(interventions) else interventions[-1]

    def learn_from_intervention_feedback(self,
                                         intervention: Dict[str, Any],
                                         feedback: Dict[str, Any]) -> Dict[str, float]:
        """从干预反馈中学习"""
        # 存储反馈
        feedback_record = {
            'intervention': intervention,
            'feedback': feedback,
            'timestamp': datetime.now()
        }
        self.intervention_effect_history.append(feedback_record)

        # 计算学习信号
        effectiveness = feedback.get('effectiveness', 0.5)
        satisfaction = feedback.get('satisfaction', 0.5)
        adherence = feedback.get('adherence', 0.5)

        learning_signal = (effectiveness + satisfaction + adherence) / 3.0

        return {
            'learning_signal': learning_signal,
            'effectiveness': effectiveness,
            'satisfaction': satisfaction,
            'adherence': adherence
        }

class AutomatedHealthInterventionSystem(nn.Module):
    """
    自动化健康干预系统
    整合所有干预模块
    """

    def __init__(self,
                 input_dim: int = 256,
                 hidden_dim: int = 128,
                 prediction_horizon: int = 24,
                 adaptation_rate: float = 0.01):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.prediction_horizon = prediction_horizon

        # 预测性干预引擎
        self.predictive_engine = PredictiveInterventionEngine(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            prediction_horizon=prediction_horizon
        )

        # 自适应干预机制
        self.adaptive_mechanism = AdaptiveInterventionMechanism(
            state_dim=input_dim,
            action_dim=10,
            hidden_dim=hidden_dim,
            adaptation_rate=adaptation_rate
        )

        # 干预决策融合器
        self.decision_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # 用户状态跟踪
        self.user_states = {}

        # 干预历史
        self.intervention_history = deque(maxlen=1000)

    def forward(self,
                user_id: str,
                current_health: Dict[str, Any],
                context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """前向传播"""
        # 获取用户历史状态
        previous_health = self.user_states.get(user_id, {}).get('previous_health')
        previous_intervention = self.user_states.get(user_id, {}).get('previous_intervention')

        # 预测性干预
        predictive_result = self.predictive_engine.predict_health_trend(
            current_health, self.prediction_horizon
        )

        # 自适应干预
        adaptive_result = self.adaptive_mechanism.adapt_intervention_strategy(
            current_health, previous_health, previous_intervention
        )

        # 决策融合
        predictive_features = self.predictive_engine.forward(
            self.predictive_engine._health_dict_to_tensor(current_health)
        )['encoded_health']

        adaptive_features = self.adaptive_mechanism.forward(
            self.adaptive_mechanism._health_dict_to_tensor(current_health),
            self.adaptive_mechanism._health_dict_to_tensor(previous_health) if previous_health else None,
            self.adaptive_mechanism._intervention_to_tensor(previous_intervention) if previous_intervention else None
        )['intervention_effect']

        # 融合特征
        fused_features = torch.cat([predictive_features, adaptive_features.unsqueeze(-1)], dim=-1)
        decision_confidence = self.decision_fusion(fused_features)

        # 生成最终干预建议
        final_intervention = self._generate_final_intervention(
            predictive_result, adaptive_result, decision_confidence.item()
        )

        # 更新用户状态
        self._update_user_state(user_id, current_health, final_intervention)

        # 记录干预历史
        self._record_intervention(user_id, final_intervention, current_health)

        return {
            'predictive_result': predictive_result,
            'adaptive_result': adaptive_result,
            'final_intervention': final_intervention,
            'decision_confidence': decision_confidence.item(),
            'user_state': self.user_states.get(user_id, {}),
            'intervention_history': list(self.intervention_history)[-5:]  # 最近5次干预
        }

    def _generate_final_intervention(self,
                                    predictive_result: Dict[str, Any],
                                    adaptive_result: Dict[str, Any],
                                    decision_confidence: float) -> Dict[str, Any]:
        """生成最终干预建议"""
        # 基于决策置信度选择干预
        if decision_confidence > 0.8:
            # 高置信度：使用预测性干预
            primary_intervention = predictive_result['intervention_suggestions'][0] if predictive_result['intervention_suggestions'] else None
            secondary_intervention = adaptive_result['primary_intervention']
        elif decision_confidence > 0.5:
            # 中等置信度：使用自适应干预
            primary_intervention = adaptive_result['primary_intervention']
            secondary_intervention = predictive_result['intervention_suggestions'][0] if predictive_result['intervention_suggestions'] else None
        else:
            # 低置信度：保守干预
            primary_intervention = {
                'type': 'monitoring',
                'priority': 'low',
                'message': '建议加强监测，观察健康指标变化',
                'intervention': 'increase_monitoring',
                'timeline': 'within_2_hours'
            }
            secondary_intervention = None

        return {
            'primary_intervention': primary_intervention,
            'secondary_intervention': secondary_intervention,
            'decision_confidence': decision_confidence,
            'intervention_type': 'automated',
            'timestamp': datetime.now(),
            'rationale': f"基于预测性分析(置信度: {decision_confidence:.2f})和自适应机制的综合决策"
        }

    def _update_user_state(self,
                           user_id: str,
                           current_health: Dict[str, Any],
                           intervention: Dict[str, Any]):
        """更新用户状态"""
        if user_id not in self.user_states:
            self.user_states[user_id] = {}

        # 更新历史状态
        self.user_states[user_id]['previous_health'] = self.user_states[user_id].get('current_health')
        self.user_states[user_id]['current_health'] = current_health
        self.user_states[user_id]['previous_intervention'] = self.user_states[user_id].get('current_intervention')
        self.user_states[user_id]['current_intervention'] = intervention
        self.user_states[user_id]['last_update'] = datetime.now()

    def _record_intervention(self,
                            user_id: str,
                            intervention: Dict[str, Any],
                            health_state: Dict[str, Any]):
        """记录干预历史"""
        intervention_record = {
            'user_id': user_id,
            'intervention': intervention,
            'health_state': health_state,
            'timestamp': datetime.now()
        }
        self.intervention_history.append(intervention_record)

    def get_intervention_report(self, user_id: str) -> Dict[str, Any]:
        """获取干预报告"""
        user_state = self.user_states.get(user_id, {})
        user_interventions = [record for record in self.intervention_history if record['user_id'] == user_id]

        return {
            'user_id': user_id,
            'total_interventions': len(user_interventions),
            'last_intervention': user_interventions[-1] if user_interventions else None,
            'intervention_frequency': len(user_interventions) / 30.0 if user_interventions else 0,  # 假设30天
            'current_health': user_state.get('current_health', {}),
            'last_update': user_state.get('last_update'),
            'intervention_types': self._analyze_intervention_types(user_interventions)
        }

    def _analyze_intervention_types(self,
                                   interventions: List[Dict[str, Any]]) -> Dict[str, int]:
        """分析干预类型"""
        type_counts = {}
        for intervention in interventions:
            intervention_type = intervention['intervention']['primary_intervention'].get('type', 'unknown')
            type_counts[intervention_type] = type_counts.get(intervention_type, 0) + 1

        return type_counts

    def learn_from_feedback(self,
                            user_id: str,
                            feedback: Dict[str, Any]) -> Dict[str, float]:
        """从用户反馈中学习"""
        # 获取最近的干预
        user_interventions = [record for record in self.intervention_history if record['user_id'] == user_id]
        if not user_interventions:
            return {'error': 'No interventions found for user'}

        recent_intervention = user_interventions[-1]['intervention']

        # 学习反馈
        learning_result = self.adaptive_mechanism.learn_from_intervention_feedback(
            recent_intervention, feedback
        )

        return learning_result

# 使用示例
def main():
    """使用示例"""
    # 创建自动化健康干预系统
    system = AutomatedHealthInterventionSystem()

    # 模拟用户健康数据
    current_health = {
        'glucose': 8.5,
        'blood_pressure_systolic': 140,
        'heart_rate': 85,
        'weight': 75,
        'exercise_minutes': 30,
        'stress_level': 6,
        'sleep_hours': 7,
        'medication_dose': 10
    }

    # 生成干预建议
    result = system.forward("user_001", current_health)

    print("干预建议:", result['final_intervention'])
    print("预测结果:", result['predictive_result'])
    print("自适应结果:", result['adaptive_result'])

if __name__ == "__main__":
    main()

__all__ = ["'logger'", "'PredictiveInterventionEngine'", "'AdaptiveInterventionMechanism'", "'AutomatedHealthInterventionSystem'", "'main'"]
