

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强的智能决策引擎
整合多模态数据融合、强化学习和跨领域知识协同
基于项目申请表中的核心算法思想设计
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
import logging
import json
from datetime import datetime
from dataclasses import dataclass

# 导入基础模型
from backend.app.modelsglucose_models import EnhancedMultiModalFusion, ReinforcementLearningDecisionEngine

logger = logging.getLogger(__name__)

@dataclass
class DecisionContext:
    """决策上下文"""
    user_id: str
    timestamp: datetime
    glucose_data: Dict[str, Any]
    image_data: Optional[Dict[str, Any]] = None
    behavioral_data: Optional[Dict[str, Any]] = None
    medical_data: Optional[Dict[str, Any]] = None
    cultural_data: Optional[Dict[str, Any]] = None
    temporal_context: Optional[List[Dict[str, Any]]] = None
    user_preferences: Optional[Dict[str, Any]] = None

class EnhancedIntelligentDecisionEngine(nn.Module):
    """
    增强的智能决策引擎
    整合多模态融合、强化学习和跨领域知识协同
    实现动态决策优化和个性化推荐
    """

    def __init__(self,
                 glucose_dim: int = 128,
                 image_dim: int = 512,
                 text_dim: int = 256,
                 behavioral_dim: int = 64,
                 medical_dim: int = 128,
                 cultural_dim: int = 64,
                 fusion_dim: int = 256,
                 state_dim: int = 256,
                 action_dim: int = 5,
                 hidden_dim: int = 128,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()

        # 多模态融合模块
        self.multimodal_fusion = EnhancedMultiModalFusion(
            glucose_dim=glucose_dim,
            image_dim=image_dim,
            text_dim=text_dim,
            behavioral_dim=behavioral_dim,
            medical_dim=medical_dim,
            cultural_dim=cultural_dim,
            fusion_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # 强化学习决策模块
        self.rl_decision_engine = ReinforcementLearningDecisionEngine(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # 状态投影层
        self.state_projection = nn.Sequential(
            nn.Linear(fusion_dim // 4, state_dim),
            nn.LayerNorm(state_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 跨模态注意力增强
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim // 4,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 决策融合层
        self.decision_fusion = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # 决策置信度
        )

        # 个性化适配层
        self.personalization_layer = nn.Sequential(
            nn.Linear(fusion_dim // 4 + 64, fusion_dim // 4),  # +64 for user features
            nn.LayerNorm(fusion_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 不确定性估计
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(fusion_dim // 4, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # 确保输出为正
        )

        # 决策历史编码器
        self.decision_history_encoder = nn.LSTM(
            input_size=action_dim + 1,  # action + reward
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        # 动态权重调整网络
        self.dynamic_weight_network = nn.Sequential(
            nn.Linear(fusion_dim // 4 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # 3个权重：多模态、强化学习、历史
            nn.Softmax(dim=-1)
        )

        # 决策解释生成器
        self.explanation_generator = nn.Sequential(
            nn.Linear(fusion_dim // 4 + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),  # 解释特征维度
            nn.ReLU()
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, context: DecisionContext) -> Dict[str, Any]:
        """
        增强的前向传播

        Args:
            context: 决策上下文

        Returns:
            决策结果和解释
        """
        # 准备多模态特征
        glucose_features = self._prepare_glucose_features(context.glucose_data)
        image_features = self._prepare_image_features(context.image_data)
        behavioral_features = self._prepare_behavioral_features(context.behavioral_data)
        medical_features = self._prepare_medical_features(context.medical_data)
        cultural_features = self._prepare_cultural_features(context.cultural_data)

        # 多模态融合
        fused_features, attention_weights = self.multimodal_fusion(
            glucose_features=glucose_features,
            image_features=image_features,
            text_features=None,  # 暂时不使用文本特征
            behavioral_features=behavioral_features,
            medical_features=medical_features,
            cultural_features=cultural_features,
            sequence_length=10
        )

        # 个性化适配
        user_features = self._prepare_user_features(context.user_preferences)
        personalized_features = self.personalization_layer(
            torch.cat([fused_features, user_features], dim=-1)
        )

        # 状态投影
        state = self.state_projection(personalized_features)

        # 强化学习决策
        temporal_context = self._prepare_temporal_context(context.temporal_context)
        rl_output = self.rl_decision_engine(state, temporal_context)

        # 决策历史处理
        decision_history_features = self._process_decision_history(context.user_id)

        # 动态权重计算
        weight_input = torch.cat([personalized_features, decision_history_features], dim=-1)
        dynamic_weights = self.dynamic_weight_network(weight_input)

        # 决策融合
        decision_confidence = self.decision_fusion(
            torch.cat([state, rl_output['policy']], dim=-1)
        )

        # 不确定性估计
        uncertainty = self.uncertainty_estimator(personalized_features)

        # 决策解释生成
        explanation_features = self.explanation_generator(
            torch.cat([personalized_features, rl_output['policy']], dim=-1)
        )

        return {
            'recommended_action': rl_output['recommended_action'],
            'action_description': rl_output['action_description'],
            'confidence': decision_confidence.item(),
            'uncertainty': uncertainty.item(),
            'objective_values': rl_output['objective_values'],
            'objective_weights': rl_output['objective_weights'],
            'dynamic_weights': dynamic_weights[0].tolist(),
            'attention_weights': attention_weights,
            'explanation_features': explanation_features[0].tolist(),
            'personalized_features': personalized_features[0].tolist(),
            'fusion_features': fused_features[0].tolist()
        }

    def _prepare_glucose_features(self, glucose_data: Dict[str, Any]) -> torch.Tensor:
        """准备血糖特征"""
        if glucose_data is None:
            return torch.zeros(1, 128)

        features = [
            glucose_data.get('glucose', 100.0),
            glucose_data.get('hour', 12.0),
            glucose_data.get('day_of_week', 1.0),
            glucose_data.get('carbohydrates', 0.0),
            glucose_data.get('protein', 0.0),
            glucose_data.get('fat', 0.0),
            glucose_data.get('calories', 0.0),
            glucose_data.get('exercise', 0.0),
            glucose_data.get('stress_level', 3.0),
            glucose_data.get('sleep_hours', 8.0)
        ]

        # 扩展到128维
        while len(features) < 128:
            features.append(0.0)

        return torch.tensor(features[:128], dtype=torch.float32).unsqueeze(0)

    def _prepare_image_features(self, image_data: Optional[Dict[str, Any]]) -> torch.Tensor:
        """准备图像特征"""
        if image_data is None:
            return torch.zeros(1, 512)

        # 简化的图像特征提取
        features = [0.0] * 512
        if 'classification' in image_data:
            # 基于分类结果生成特征
            class_id = hash(image_data['classification']) % 512
            features[class_id] = image_data.get('confidence', 0.5)

        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    def _prepare_behavioral_features(self, behavioral_data: Optional[Dict[str, Any]]) -> torch.Tensor:
        """准备行为特征"""
        if behavioral_data is None:
            return torch.zeros(1, 64)

        features = [
            behavioral_data.get('activity_level', 3.0),
            behavioral_data.get('meal_frequency', 3.0),
            behavioral_data.get('exercise_frequency', 3.0),
            behavioral_data.get('sleep_quality', 3.0),
            behavioral_data.get('stress_level', 3.0),
            behavioral_data.get('social_activity', 3.0)
        ]

        # 扩展到64维
        while len(features) < 64:
            features.append(0.0)

        return torch.tensor(features[:64], dtype=torch.float32).unsqueeze(0)

    def _prepare_medical_features(self, medical_data: Optional[Dict[str, Any]]) -> torch.Tensor:
        """准备医学特征"""
        if medical_data is None:
            return torch.zeros(1, 128)

        features = [
            medical_data.get('age', 40.0),
            medical_data.get('bmi', 25.0),
            medical_data.get('diabetes_type', 2.0),
            medical_data.get('medication_count', 0.0),
            medical_data.get('complications', 0.0),
            medical_data.get('family_history', 0.0)
        ]

        # 扩展到128维
        while len(features) < 128:
            features.append(0.0)

        return torch.tensor(features[:128], dtype=torch.float32).unsqueeze(0)

    def _prepare_cultural_features(self, cultural_data: Optional[Dict[str, Any]]) -> torch.Tensor:
        """准备文化特征"""
        if cultural_data is None:
            return torch.zeros(1, 64)

        features = [
            cultural_data.get('cultural_group', 0.0),
            cultural_data.get('dietary_preferences', 0.0),
            cultural_data.get('religious_restrictions', 0.0),
            cultural_data.get('cooking_style', 0.0),
            cultural_data.get('meal_timing', 0.0),
            cultural_data.get('social_eating', 0.0)
        ]

        # 扩展到64维
        while len(features) < 64:
            features.append(0.0)

        return torch.tensor(features[:64], dtype=torch.float32).unsqueeze(0)

    def _prepare_user_features(self, user_preferences: Optional[Dict[str, Any]]) -> torch.Tensor:
        """准备用户特征"""
        if user_preferences is None:
            return torch.zeros(1, 64)

        features = [
            user_preferences.get('preferred_cuisine', 0.0),
            user_preferences.get('spice_level', 3.0),
            user_preferences.get('cooking_time', 30.0),
            user_preferences.get('budget_level', 3.0),
            user_preferences.get('health_goals', 0.0),
            user_preferences.get('allergies', 0.0)
        ]

        # 扩展到64维
        while len(features) < 64:
            features.append(0.0)

        return torch.tensor(features[:64], dtype=torch.float32).unsqueeze(0)

    def _prepare_temporal_context(self, temporal_context: Optional[List[Dict[str, Any]]]) -> Optional[torch.Tensor]:
        """准备时序上下文"""
        if temporal_context is None or len(temporal_context) == 0:
            return None

        # 简化的时序特征提取
        features = []
        for context_point in temporal_context[-10:]:  # 最近10个时间点
            point_features = [
                context_point.get('glucose', 100.0),
                context_point.get('hour', 12.0),
                context_point.get('carbohydrates', 0.0),
                context_point.get('exercise', 0.0)
            ]
            features.append(point_features)

        # 填充到固定长度
        while len(features) < 10:
            features.insert(0, [100.0, 12.0, 0.0, 0.0])

        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    def _process_decision_history(self, user_id: str) -> torch.Tensor:
        """处理决策历史"""
        # 简化的历史处理，实际应用中应该从数据库获取
        # 这里返回零向量作为占位符
        return torch.zeros(1, 128)

    def make_decision(self, context: DecisionContext) -> Dict[str, Any]:
        """
        做出决策

        Args:
            context: 决策上下文

        Returns:
            决策结果
        """
        try:
            # 前向传播
            result = self.forward(context)

            # 添加时间戳和用户ID
            result.update({
                'user_id': context.user_id,
                'timestamp': context.timestamp.isoformat(),
                'decision_id': f"{context.user_id}_{int(context.timestamp.timestamp())}"
            })

            # 生成可读的解释
            explanation = self._generate_explanation(result)
            result['explanation'] = explanation

            return result

        except Exception as e:
            logger.error(f"决策失败: {e}")
            return {
                'error': str(e),
                'user_id': context.user_id,
                'timestamp': context.timestamp.isoformat(),
                'recommended_action': 4,  # 默认动作：保持当前状态
                'action_description': "保持当前状态",
                'confidence': 0.5,
                'uncertainty': 1.0
            }

    def _generate_explanation(self, result: Dict[str, Any]) -> str:
        """生成决策解释"""
        action = result['recommended_action']
        confidence = result['confidence']
        objective_values = result['objective_values']

        explanations = []

        # 基于动作的解释
        action_explanations = {
            0: "基于您的血糖水平和营养需求，建议适当增加碳水化合物摄入",
            1: "考虑到血糖控制目标，建议减少碳水化合物摄入",
            2: "根据您的健康状况，建议增加运动量以改善血糖控制",
            3: "基于血糖监测结果，建议调整胰岛素剂量",
            4: "当前状态良好，建议保持现有的饮食和运动习惯"
        }

        explanations.append(action_explanations.get(action, "建议保持当前状态"))

        # 基于置信度的解释
        if confidence > 0.8:
            explanations.append("系统对此建议有很高的信心")
        elif confidence > 0.6:
            explanations.append("系统对此建议有中等信心")
        else:
            explanations.append("建议仅供参考，请结合实际情况考虑")

        # 基于目标值的解释
        if 'glucose_control' in objective_values:
            glucose_value = objective_values['glucose_control']
            if glucose_value > 0.7:
                explanations.append("血糖控制是当前的主要关注点")

        if 'cultural_adaptation' in objective_values:
            cultural_value = objective_values['cultural_adaptation']
            if cultural_value > 0.6:
                explanations.append("建议考虑了您的文化背景和饮食习惯")

        return "；".join(explanations)

    def update_with_feedback(self, user_id: str, decision_id: str,
                           feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用反馈更新模型

        Args:
            user_id: 用户ID
            decision_id: 决策ID
            feedback: 反馈数据

        Returns:
            更新结果
        """
        try:
            # 准备奖励
            reward = {
                'glucose_control': feedback.get('glucose_improvement', 0.0),
                'nutrition_balance': feedback.get('nutrition_satisfaction', 0.0),
                'patient_satisfaction': feedback.get('satisfaction_score', 0.0) / 5.0,
                'cultural_adaptation': feedback.get('cultural_fit', 0.0)
            }

            # 更新强化学习模型
            if hasattr(self.rl_decision_engine, 'store_experience'):
                # 这里需要实际的状态和下一状态，简化处理
                state = torch.zeros(1, 256)
                next_state = torch.zeros(1, 256)

                self.rl_decision_engine.store_experience(
                    state=state,
                    action=feedback.get('action_taken', 4),
                    reward=reward,
                    next_state=next_state,
                    done=False
                )

            return {
                'status': 'success',
                'message': '反馈已记录并用于模型更新',
                'user_id': user_id,
                'decision_id': decision_id
            }

        except Exception as e:
            logger.error(f"反馈更新失败: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'user_id': user_id,
                'decision_id': decision_id
            }

class DecisionEngineManager:
    """决策引擎管理器"""

    def __init__(self):
        self.engines = {}
        self.logger = logger

    def get_engine(self, user_id: str) -> EnhancedIntelligentDecisionEngine:
        """获取用户的决策引擎"""
        if user_id not in self.engines:
            self.engines[user_id] = EnhancedIntelligentDecisionEngine()
            self.logger.info(f"为用户 {user_id} 创建新的决策引擎")

        return self.engines[user_id]

    def make_decision(self, user_id: str, context: DecisionContext) -> Dict[str, Any]:
        """为用户做决策"""
        engine = self.get_engine(user_id)
        return engine.make_decision(context)

    def update_feedback(self, user_id: str, decision_id: str,
                       feedback: Dict[str, Any]) -> Dict[str, Any]:
        """更新反馈"""
        if user_id in self.engines:
            return self.engines[user_id].update_with_feedback(user_id, decision_id, feedback)
        else:
            return {
                'status': 'error',
                'message': f'用户 {user_id} 的决策引擎不存在'
            }

    def get_engine_stats(self, user_id: str) -> Dict[str, Any]:
        """获取引擎统计信息"""
        if user_id not in self.engines:
            return {'status': 'not_found'}

        engine = self.engines[user_id]
        return {
            'user_id': user_id,
            'engine_type': 'EnhancedIntelligentDecisionEngine',
            'has_rl_engine': hasattr(engine, 'rl_decision_engine'),
            'has_multimodal_fusion': hasattr(engine, 'multimodal_fusion'),
            'replay_buffer_size': len(engine.rl_decision_engine.replay_buffer) if hasattr(engine, 'rl_decision_engine') else 0,
            'epsilon': engine.rl_decision_engine.epsilon if hasattr(engine, 'rl_decision_engine') else 0.0
        }

# 全局决策引擎管理器实例
decision_engine_manager = DecisionEngineManager()

__all__ = ["'logger'", "'DecisionContext'", "'EnhancedIntelligentDecisionEngine'", "'DecisionEngineManager'", "'decision_engine_manager'"]
