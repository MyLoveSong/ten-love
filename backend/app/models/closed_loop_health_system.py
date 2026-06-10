

"""
健康监测闭环和实时反馈系统
基于项目申请表中的创新点七设计
实现多目标强化学习和实时健康管理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging
from collections import deque
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class MultiObjectiveReinforcementLearning(nn.Module):
    """
    多目标强化学习系统
    基于项目申请表中的创新点七设计
    实现血糖控制、营养均衡、患者满意度等多目标优化
    """

    def __init__(self,
                 state_dim: int = 256,
                 action_dim: int = 10,
                 hidden_dim: int = 128,
                 num_objectives: int = 4,  # 血糖控制、营养均衡、患者满意度、安全性
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_objectives = num_objectives

        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # 多目标价值网络
        self.value_networks = nn.ModuleDict({
            'glucose_control': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            ),
            'nutrition_balance': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            ),
            'patient_satisfaction': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            ),
            'safety': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
        })

        # 策略网络
        self.policy_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

        # 目标权重网络
        self.objective_weight_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_objectives),
            nn.Softmax(dim=-1)
        )

        # 经验回放缓冲区
        self.replay_buffer = deque(maxlen=10000)

        # 优化器
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        # 超参数
        self.gamma = 0.99
        self.epsilon = 0.1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 状态编码
        encoded_state = self.state_encoder(state)

        # 多目标价值估计
        values = {}
        for objective, network in self.value_networks.items():
            values[objective] = network(encoded_state)

        # 策略输出
        policy = self.policy_network(encoded_state)

        # 目标权重
        objective_weights = self.objective_weight_network(encoded_state)

        # 加权总价值
        weighted_value = sum(
            values[obj] * objective_weights[:, i:i+1]
            for i, obj in enumerate(self.value_networks.keys())
        )

        return {
            'policy': policy,
            'values': values,
            'weighted_value': weighted_value,
            'objective_weights': objective_weights,
            'encoded_state': encoded_state
        }

    def select_action(self, state: torch.Tensor, training: bool = True) -> Tuple[int, Dict[str, torch.Tensor]]:
        """选择动作"""
        with torch.no_grad():
            output = self.forward(state)
            policy = output['policy']

            if training and np.random.random() < self.epsilon:
                # 探索
                action = np.random.randint(0, self.action_dim)
            else:
                # 利用
                action = torch.argmax(policy, dim=-1).item()

            return action, output

    def store_experience(self, state: torch.Tensor, action: int,
                        rewards: Dict[str, float], next_state: torch.Tensor,
                        done: bool):
        """存储经验"""
        experience = {
            'state': state,
            'action': action,
            'rewards': rewards,
            'next_state': next_state,
            'done': done,
            'timestamp': datetime.now()
        }
        self.replay_buffer.append(experience)

    def update_policy(self, batch_size: int = 32) -> Dict[str, float]:
        """更新策略"""
        if len(self.replay_buffer) < batch_size:
            return {}

        # 采样经验
        batch = random.sample(self.replay_buffer, batch_size)

        states = torch.stack([exp['state'] for exp in batch])
        actions = torch.tensor([exp['action'] for exp in batch])
        rewards = [exp['rewards'] for exp in batch]
        next_states = torch.stack([exp['next_state'] for exp in batch])
        dones = torch.tensor([exp['done'] for exp in batch], dtype=torch.float32)

        # 计算目标价值
        with torch.no_grad():
            next_output = self.forward(next_states)
            next_values = next_output['values']

            target_values = {}
            for objective in self.value_networks.keys():
                obj_rewards = torch.tensor([r[objective] for r in rewards], dtype=torch.float32)
                target_values[objective] = obj_rewards + self.gamma * next_values[objective].squeeze() * (1 - dones)

        # 前向传播
        current_output = self.forward(states)
        current_values = current_output['values']

        # 计算损失
        value_losses = {}
        total_value_loss = 0

        for objective in self.value_networks.keys():
            value_loss = F.mse_loss(current_values[objective].squeeze(), target_values[objective])
            value_losses[objective] = value_loss.item()
            total_value_loss += value_loss

        # 策略损失
        policy_loss = -torch.mean(torch.log(current_output['policy'][range(batch_size), actions] + 1e-8))

        # 总损失
        total_loss = total_value_loss + policy_loss

        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # 更新epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_losses': value_losses,
            'epsilon': self.epsilon
        }

class RealTimeHealthMonitor(nn.Module):
    """
    实时健康监测器
    基于项目申请表中的创新点七设计
    实现血糖、营养、运动等多维度实时监测
    """

    def __init__(self,
                 input_dim: int = 20,
                 hidden_dim: int = 128,
                 num_health_indicators: int = 6,
                 alert_threshold: float = 0.8):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_health_indicators = num_health_indicators
        self.alert_threshold = alert_threshold

        # 健康指标编码器
        self.health_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # 健康指标预测器
        self.health_predictors = nn.ModuleDict({
            'glucose': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            ),
            'nutrition': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            ),
            'exercise': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            ),
            'stress': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            ),
            'sleep': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            ),
            'medication': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
        })

        # 异常检测器
        self.anomaly_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # 趋势分析器
        self.trend_analyzer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),  # 上升、下降、稳定
            nn.Softmax(dim=-1)
        )

        # 监测历史
        self.monitoring_history = deque(maxlen=1000)

    def forward(self, health_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 健康数据编码
        encoded_health = self.health_encoder(health_data)

        # 健康指标预测
        health_scores = {}
        for indicator, predictor in self.health_predictors.items():
            health_scores[indicator] = predictor(encoded_health)

        # 异常检测
        anomaly_score = self.anomaly_detector(encoded_health)

        # 趋势分析
        trend_probs = self.trend_analyzer(encoded_health)

        # 综合健康评分
        overall_health = torch.mean(torch.stack(list(health_scores.values())), dim=0)

        return {
            'health_scores': health_scores,
            'anomaly_score': anomaly_score,
            'trend_probs': trend_probs,
            'overall_health': overall_health,
            'encoded_health': encoded_health
        }

    def monitor_health(self, health_data: Dict[str, Any]) -> Dict[str, Any]:
        """监测健康状态"""
        # 转换为张量
        health_tensor = torch.tensor([
            health_data.get('glucose', 0.0),
            health_data.get('carbohydrates', 0.0),
            health_data.get('protein', 0.0),
            health_data.get('fat', 0.0),
            health_data.get('exercise', 0.0),
            health_data.get('stress', 0.0),
            health_data.get('sleep_hours', 0.0),
            health_data.get('medication_dose', 0.0),
            health_data.get('blood_pressure', 0.0),
            health_data.get('heart_rate', 0.0),
            health_data.get('weight', 0.0),
            health_data.get('bmi', 0.0),
            health_data.get('age', 0.0),
            health_data.get('gender', 0.0),
            health_data.get('diabetes_type', 0.0),
            health_data.get('medication_type', 0.0),
            health_data.get('exercise_type', 0.0),
            health_data.get('meal_time', 0.0),
            health_data.get('weather', 0.0),
            health_data.get('mood', 0.0)
        ], dtype=torch.float32).unsqueeze(0)

        # 前向传播
        output = self.forward(health_tensor)

        # 检查异常
        alerts = []
        if output['anomaly_score'].item() > self.alert_threshold:
            alerts.append({
                'type': 'anomaly',
                'severity': 'high',
                'message': '检测到健康异常，建议立即就医',
                'timestamp': datetime.now()
            })

        # 检查各指标
        for indicator, score in output['health_scores'].items():
            if score.item() < 0.3:  # 低分阈值
                alerts.append({
                    'type': indicator,
                    'severity': 'medium',
                    'message': f'{indicator}指标偏低，建议关注',
                    'timestamp': datetime.now()
                })

        # 记录监测历史
        monitoring_record = {
            'timestamp': datetime.now(),
            'health_data': health_data,
            'health_scores': {k: v.item() for k, v in output['health_scores'].items()},
            'anomaly_score': output['anomaly_score'].item(),
            'overall_health': output['overall_health'].item(),
            'alerts': alerts
        }
        self.monitoring_history.append(monitoring_record)

        return {
            'health_scores': {k: v.item() for k, v in output['health_scores'].items()},
            'anomaly_score': output['anomaly_score'].item(),
            'trend_probs': output['trend_probs'].detach().numpy().tolist(),
            'overall_health': output['overall_health'].item(),
            'alerts': alerts,
            'monitoring_record': monitoring_record
        }

    def get_health_trends(self, days: int = 7) -> Dict[str, Any]:
        """获取健康趋势"""
        if not self.monitoring_history:
            return {}

        # 获取最近几天的数据
        recent_records = list(self.monitoring_history)[-days:]

        # 计算趋势
        trends = {}
        for indicator in self.health_predictors.keys():
            scores = [record['health_scores'][indicator] for record in recent_records]
            if len(scores) > 1:
                trend = 'improving' if scores[-1] > scores[0] else 'declining' if scores[-1] < scores[0] else 'stable'
                trends[indicator] = {
                    'trend': trend,
                    'current_score': scores[-1],
                    'average_score': np.mean(scores),
                    'change': scores[-1] - scores[0]
                }

        return {
            'trends': trends,
            'overall_trend': 'improving' if recent_records[-1]['overall_health'] > recent_records[0]['overall_health'] else 'declining',
            'period': f'{days} days',
            'total_records': len(recent_records)
        }

class ClosedLoopHealthManagementSystem(nn.Module):
    """
    闭环健康管理系统
    基于项目申请表中的创新点七设计
    整合多目标强化学习和实时监测
    """

    def __init__(self,
                 state_dim: int = 256,
                 action_dim: int = 10,
                 health_input_dim: int = 20,
                 hidden_dim: int = 128):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.health_input_dim = health_input_dim

        # 多目标强化学习系统
        self.morl_system = MultiObjectiveReinforcementLearning(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        )

        # 实时健康监测器
        self.health_monitor = RealTimeHealthMonitor(
            input_dim=health_input_dim,
            hidden_dim=hidden_dim
        )

        # 反馈融合器
        self.feedback_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, state_dim)
        )

        # 推荐生成器
        self.recommendation_generator = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 5)  # 5种推荐类型
        )

        # 系统状态
        self.system_state = {
            'current_health': {},
            'recommendations': [],
            'feedback_history': deque(maxlen=1000),
            'performance_metrics': {}
        }

    def forward(self,
                health_data: Dict[str, Any],
                user_feedback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """前向传播"""
        # 健康监测
        health_monitoring_result = self.health_monitor.monitor_health(health_data)

        # 构建状态
        state = self._build_state(health_monitoring_result, user_feedback)

        # 多目标强化学习决策
        action, rl_output = self.morl_system.select_action(state)

        # 生成推荐
        recommendation = self._generate_recommendation(state, action, health_monitoring_result)

        # 更新系统状态
        self._update_system_state(health_monitoring_result, recommendation, user_feedback)

        return {
            'health_monitoring': health_monitoring_result,
            'action': action,
            'recommendation': recommendation,
            'rl_output': rl_output,
            'system_state': self.system_state.copy()
        }

    def _build_state(self,
                     health_monitoring_result: Dict[str, Any],
                     user_feedback: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """构建状态向量"""
        # 健康指标
        health_scores = health_monitoring_result['health_scores']
        health_vector = [
            health_scores.get('glucose', 0.0),
            health_scores.get('nutrition', 0.0),
            health_scores.get('exercise', 0.0),
            health_scores.get('stress', 0.0),
            health_scores.get('sleep', 0.0),
            health_scores.get('medication', 0.0)
        ]

        # 异常分数
        anomaly_score = health_monitoring_result['anomaly_score']

        # 整体健康
        overall_health = health_monitoring_result['overall_health']

        # 趋势
        trend_probs = health_monitoring_result['trend_probs']

        # 用户反馈
        feedback_vector = [0.0] * 10  # 10维反馈向量
        if user_feedback:
            feedback_vector[0] = user_feedback.get('satisfaction', 0.0)
            feedback_vector[1] = user_feedback.get('compliance', 0.0)
            feedback_vector[2] = user_feedback.get('effectiveness', 0.0)

        # 组合状态向量
        state_vector = health_vector + [anomaly_score, overall_health] + trend_probs + feedback_vector

        # 填充到state_dim
        while len(state_vector) < self.state_dim:
            state_vector.append(0.0)

        return torch.tensor(state_vector[:self.state_dim], dtype=torch.float32).unsqueeze(0)

    def _generate_recommendation(self,
                                state: torch.Tensor,
                                action: int,
                                health_monitoring_result: Dict[str, Any]) -> Dict[str, Any]:
        """生成推荐"""
        # 动作描述
        action_descriptions = {
            0: "调整饮食计划",
            1: "增加运动量",
            2: "调整药物剂量",
            3: "改善睡眠质量",
            4: "减少压力",
            5: "增加血糖监测频率",
            6: "调整餐次时间",
            7: "增加水分摄入",
            8: "调整运动时间",
            9: "寻求医疗建议"
        }

        # 基于健康监测结果生成具体推荐
        recommendations = []

        # 血糖相关推荐
        if health_monitoring_result['health_scores']['glucose'] < 0.4:
            recommendations.append({
                'type': 'diet',
                'priority': 'high',
                'message': '血糖偏低，建议增加碳水化合物摄入',
                'action': 'increase_carbs'
            })
        elif health_monitoring_result['health_scores']['glucose'] > 0.8:
            recommendations.append({
                'type': 'diet',
                'priority': 'high',
                'message': '血糖偏高，建议减少碳水化合物摄入',
                'action': 'decrease_carbs'
            })

        # 运动相关推荐
        if health_monitoring_result['health_scores']['exercise'] < 0.3:
            recommendations.append({
                'type': 'exercise',
                'priority': 'medium',
                'message': '运动量不足，建议增加日常活动',
                'action': 'increase_exercise'
            })

        # 压力相关推荐
        if health_monitoring_result['health_scores']['stress'] > 0.7:
            recommendations.append({
                'type': 'lifestyle',
                'priority': 'medium',
                'message': '压力水平较高，建议进行放松活动',
                'action': 'reduce_stress'
            })

        return {
            'action_description': action_descriptions.get(action, "未知动作"),
            'recommendations': recommendations,
            'priority': 'high' if any(r['priority'] == 'high' for r in recommendations) else 'medium',
            'timestamp': datetime.now(),
            'confidence': 0.85  # 模拟置信度
        }

    def _update_system_state(self,
                            health_monitoring_result: Dict[str, Any],
                            recommendation: Dict[str, Any],
                            user_feedback: Optional[Dict[str, Any]] = None):
        """更新系统状态"""
        # 更新当前健康状态
        self.system_state['current_health'] = health_monitoring_result['health_scores']

        # 添加推荐
        self.system_state['recommendations'].append(recommendation)

        # 添加反馈历史
        if user_feedback:
            feedback_record = {
                'timestamp': datetime.now(),
                'feedback': user_feedback,
                'health_state': health_monitoring_result['health_scores'],
                'recommendation': recommendation
            }
            self.system_state['feedback_history'].append(feedback_record)

        # 更新性能指标
        self._update_performance_metrics(health_monitoring_result, user_feedback)

    def _update_performance_metrics(self,
                                   health_monitoring_result: Dict[str, Any],
                                   user_feedback: Optional[Dict[str, Any]] = None):
        """更新性能指标"""
        if not hasattr(self, 'performance_metrics'):
            self.performance_metrics = {
                'total_recommendations': 0,
                'user_satisfaction': [],
                'health_improvement': [],
                'system_accuracy': []
            }

        # 更新推荐数量
        self.performance_metrics['total_recommendations'] += 1

        # 更新用户满意度
        if user_feedback and 'satisfaction' in user_feedback:
            self.performance_metrics['user_satisfaction'].append(user_feedback['satisfaction'])

        # 更新健康改善情况
        overall_health = health_monitoring_result['overall_health']
        self.performance_metrics['health_improvement'].append(overall_health)

        # 更新系统准确性
        if user_feedback and 'effectiveness' in user_feedback:
            self.performance_metrics['system_accuracy'].append(user_feedback['effectiveness'])

    def get_system_performance(self) -> Dict[str, Any]:
        """获取系统性能"""
        if not hasattr(self, 'performance_metrics'):
            return {}

        metrics = self.performance_metrics

        # 计算平均满意度
        avg_satisfaction = np.mean(metrics['user_satisfaction']) if metrics['user_satisfaction'] else 0.0

        # 计算健康改善趋势
        health_trend = 'improving' if len(metrics['health_improvement']) > 1 and metrics['health_improvement'][-1] > metrics['health_improvement'][0] else 'stable'

        # 计算平均准确性
        avg_accuracy = np.mean(metrics['system_accuracy']) if metrics['system_accuracy'] else 0.0

        return {
            'total_recommendations': metrics['total_recommendations'],
            'average_satisfaction': avg_satisfaction,
            'health_trend': health_trend,
            'average_accuracy': avg_accuracy,
            'total_feedback': len(self.system_state['feedback_history']),
            'current_health_score': self.system_state['current_health'].get('overall', 0.0)
        }

    def train_system(self, training_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """训练系统"""
        total_losses = []

        for data in training_data:
            # 前向传播
            result = self.forward(data['health_data'], data.get('user_feedback'))

            # 计算奖励
            rewards = self._calculate_rewards(data, result)

            # 存储经验
            state = self._build_state(result['health_monitoring'], data.get('user_feedback'))
            next_state = state  # 简化实现
            self.morl_system.store_experience(state, result['action'], rewards, next_state, False)

            # 更新策略
            losses = self.morl_system.update_policy()
            if losses:
                total_losses.append(losses)

        # 计算平均损失
        if total_losses:
            avg_losses = {}
            for key in total_losses[0].keys():
                avg_losses[key] = np.mean([loss[key] for loss in total_losses])
            return avg_losses

        return {}

    def _calculate_rewards(self,
                          data: Dict[str, Any],
                          result: Dict[str, Any]) -> Dict[str, float]:
        """计算奖励"""
        rewards = {}

        # 血糖控制奖励
        glucose_score = result['health_monitoring']['health_scores']['glucose']
        rewards['glucose_control'] = 1.0 if 0.4 <= glucose_score <= 0.8 else -0.5

        # 营养均衡奖励
        nutrition_score = result['health_monitoring']['health_scores']['nutrition']
        rewards['nutrition_balance'] = nutrition_score

        # 患者满意度奖励
        if 'user_feedback' in data and 'satisfaction' in data['user_feedback']:
            rewards['patient_satisfaction'] = data['user_feedback']['satisfaction'] / 5.0
        else:
            rewards['patient_satisfaction'] = 0.5

        # 安全性奖励
        anomaly_score = result['health_monitoring']['anomaly_score']
        rewards['safety'] = 1.0 - anomaly_score

        return rewards

# 使用示例
def main():
    """使用示例"""
    # 创建闭环健康管理系统
    system = ClosedLoopHealthManagementSystem()

    # 模拟健康数据
    health_data = {
        'glucose': 7.5,
        'carbohydrates': 60,
        'protein': 25,
        'fat': 15,
        'exercise': 30,
        'stress': 5,
        'sleep_hours': 7,
        'medication_dose': 10,
        'blood_pressure': 120,
        'heart_rate': 70,
        'weight': 70,
        'bmi': 24,
        'age': 45,
        'gender': 1,
        'diabetes_type': 2,
        'medication_type': 1,
        'exercise_type': 2,
        'meal_time': 12,
        'weather': 1,
        'mood': 3
    }

    # 运行系统
    result = system.forward(health_data)

    print("健康监测结果:", result['health_monitoring'])
    print("推荐:", result['recommendation'])
    print("系统性能:", system.get_system_performance())

if __name__ == "__main__":
    main()

__all__ = ["'logger'", "'MultiObjectiveReinforcementLearning'", "'RealTimeHealthMonitor'", "'ClosedLoopHealthManagementSystem'", "'main'"]
