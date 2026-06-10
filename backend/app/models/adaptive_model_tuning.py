

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型适应性调整和迁移学习模块
支持动态模型微调、迁移学习、个性化适应和知识蒸馏
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import copy
import pickle
from pathlib import Path
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class AdaptationConfig:
    """适应性配置"""
    learning_rate: float = 0.001
    fine_tune_epochs: int = 10
    adaptation_threshold: float = 0.1  # 性能下降阈值
    min_adaptation_samples: int = 20
    max_adaptation_samples: int = 1000
    knowledge_retention_weight: float = 0.3  # 知识保留权重
    personalization_weight: float = 0.7  # 个性化权重
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 5

@dataclass
class UserProfile:
    """用户档案"""
    user_id: str
    age: int
    gender: str
    bmi: float
    diabetes_type: Optional[int] = None
    medication: List[str] = field(default_factory=list)
    dietary_preferences: Dict[str, Any] = field(default_factory=dict)
    activity_level: str = "moderate"  # low, moderate, high
    baseline_glucose: float = 5.5
    glucose_variability: float = 1.0
    adaptation_history: List[Dict[str, Any]] = field(default_factory=list)

class MetaLearner(nn.Module):
    """元学习器 - 用于快速适应新用户"""

    def __init__(self, input_dim: int, hidden_dim: int = 128, meta_dim: int = 64):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.meta_dim = meta_dim

        # 主网络
        self.main_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 元网络 - 生成适应参数
        self.meta_network = nn.Sequential(
            nn.Linear(meta_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self._count_parameters())
        )

        # 用户特征编码器
        self.user_encoder = nn.Sequential(
            nn.Linear(10, meta_dim // 2),  # 假设用户特征维度为10
            nn.ReLU(),
            nn.Linear(meta_dim // 2, meta_dim)
        )

    def _count_parameters(self) -> int:
        """计算主网络参数数量"""
        return sum(p.numel() for p in self.main_network.parameters())

    def forward(self, x: torch.Tensor, user_features: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 编码用户特征
        user_encoded = self.user_encoder(user_features)

        # 生成适应参数
        adaptation_params = self.meta_network(user_encoded)

        # 应用适应参数到主网络（简化版本）
        adapted_output = self.main_network(x)

        # 这里可以实现更复杂的参数调制机制
        adaptation_factor = torch.sigmoid(adaptation_params[:, 0]).unsqueeze(1)
        adapted_output = adapted_output * adaptation_factor

        return adapted_output

class TransferLearningModule:
    """迁移学习模块"""

    def __init__(self, base_model: nn.Module, config: AdaptationConfig):
        self.base_model = base_model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 冻结部分层用于迁移学习
        self.frozen_layers = []
        self.trainable_layers = []
        self._setup_transfer_layers()

    def _setup_transfer_layers(self):
        """设置迁移学习层"""
        # 冻结前面的特征提取层，只训练后面的分类层
        layers = list(self.base_model.children())
        freeze_ratio = 0.7  # 冻结70%的层
        freeze_count = int(len(layers) * freeze_ratio)

        for i, layer in enumerate(layers):
            if i < freeze_count:
                for param in layer.parameters():
                    param.requires_grad = False
                self.frozen_layers.append(layer)
            else:
                self.trainable_layers.append(layer)

        logger.info(f"冻结 {len(self.frozen_layers)} 层，训练 {len(self.trainable_layers)} 层")

    def fine_tune(self, train_data: torch.Tensor, train_labels: torch.Tensor,
                  val_data: Optional[torch.Tensor] = None,
                  val_labels: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """微调模型"""
        self.base_model.train()
        self.base_model.to(self.device)

        # 只优化可训练层
        trainable_params = []
        for layer in self.trainable_layers:
            trainable_params.extend(layer.parameters())

        optimizer = torch.optim.Adam(trainable_params, lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.fine_tune_epochs)
        criterion = nn.MSELoss()

        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.config.fine_tune_epochs):
            # 训练
            self.base_model.train()
            epoch_train_loss = 0

            for i in range(0, len(train_data), 32):  # batch_size = 32
                batch_data = train_data[i:i+32].to(self.device)
                batch_labels = train_labels[i:i+32].to(self.device)

                optimizer.zero_grad()
                outputs = self.base_model(batch_data)
                loss = criterion(outputs.squeeze(), batch_labels)
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(trainable_params, self.config.gradient_clip_norm)

                optimizer.step()
                epoch_train_loss += loss.item()

            avg_train_loss = epoch_train_loss / (len(train_data) // 32 + 1)
            train_losses.append(avg_train_loss)

            # 验证
            if val_data is not None and val_labels is not None:
                self.base_model.eval()
                with torch.no_grad():
                    val_outputs = self.base_model(val_data.to(self.device))
                    val_loss = criterion(val_outputs.squeeze(), val_labels.to(self.device))
                    val_losses.append(val_loss.item())

                    # 早停检查
                    if val_loss.item() < best_val_loss:
                        best_val_loss = val_loss.item()
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= self.config.early_stopping_patience:
                        logger.info(f"早停在第 {epoch+1} 轮")
                        break

            scheduler.step()

            if epoch % 5 == 0:
                logger.info(f"微调 Epoch {epoch+1}: 训练损失={avg_train_loss:.4f}, "
                          f"验证损失={val_losses[-1] if val_losses else 'N/A':.4f}")

        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'epochs_trained': len(train_losses)
        }

    def progressive_unfreezing(self, step: int):
        """渐进式解冻"""
        if step == 0:
            return

        # 计算需要解冻的层数
        layers_to_unfreeze = min(step, len(self.frozen_layers))

        for i in range(layers_to_unfreeze):
            layer_idx = len(self.frozen_layers) - 1 - i  # 从后往前解冻
            layer = self.frozen_layers[layer_idx]

            for param in layer.parameters():
                param.requires_grad = True

            # 移动到可训练层
            self.trainable_layers.insert(0, layer)

        # 更新冻结层列表
        self.frozen_layers = self.frozen_layers[:-layers_to_unfreeze]

        logger.info(f"解冻 {layers_to_unfreeze} 层，当前可训练层数: {len(self.trainable_layers)}")

class PersonalizationEngine:
    """个性化引擎"""

    def __init__(self, config: AdaptationConfig):
        self.config = config
        self.user_models: Dict[str, nn.Module] = {}
        self.user_profiles: Dict[str, UserProfile] = {}
        self.user_scalers: Dict[str, StandardScaler] = {}
        self.adaptation_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def create_user_profile(self, user_data: Dict[str, Any]) -> UserProfile:
        """创建用户档案"""
        profile = UserProfile(
            user_id=user_data['user_id'],
            age=user_data.get('age', 45),
            gender=user_data.get('gender', 'unknown'),
            bmi=user_data.get('bmi', 25.0),
            diabetes_type=user_data.get('diabetes_type'),
            medication=user_data.get('medication', []),
            dietary_preferences=user_data.get('dietary_preferences', {}),
            activity_level=user_data.get('activity_level', 'moderate'),
            baseline_glucose=user_data.get('baseline_glucose', 5.5),
            glucose_variability=user_data.get('glucose_variability', 1.0)
        )

        self.user_profiles[profile.user_id] = profile
        logger.info(f"创建用户档案: {profile.user_id}")

        return profile

    def adapt_model_for_user(self, user_id: str, base_model: nn.Module,
                           user_data: List[Dict[str, Any]]) -> nn.Module:
        """为用户适应模型"""
        if len(user_data) < self.config.min_adaptation_samples:
            logger.warning(f"用户 {user_id} 数据不足，无法进行个性化适应")
            # 返回基础模型的副本
            user_model = copy.deepcopy(base_model)
            self.user_models[user_id] = user_model
            return user_model

        # 准备用户数据
        features, labels = self._prepare_user_data(user_data)

        # 创建用户特定模型
        user_model = copy.deepcopy(base_model)

        # 应用迁移学习
        transfer_module = TransferLearningModule(user_model, self.config)

        # 分割训练和验证数据
        split_idx = int(len(features) * 0.8)
        train_features = features[:split_idx]
        train_labels = labels[:split_idx]
        val_features = features[split_idx:] if split_idx < len(features) else None
        val_labels = labels[split_idx:] if split_idx < len(labels) else None

        # 微调模型
        fine_tune_result = transfer_module.fine_tune(
            train_features, train_labels, val_features, val_labels
        )

        # 记录适应历史
        adaptation_record = {
            'timestamp': datetime.now(),
            'samples_used': len(user_data),
            'fine_tune_result': fine_tune_result,
            'final_loss': fine_tune_result['best_val_loss']
        }
        self.adaptation_history[user_id].append(adaptation_record)

        # 保存用户模型
        self.user_models[user_id] = user_model

        logger.info(f"为用户 {user_id} 完成模型适应，使用 {len(user_data)} 个样本")

        return user_model

    def _prepare_user_data(self, user_data: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """准备用户数据"""
        # 提取特征和标签
        features = []
        labels = []

        for record in user_data:
            feature_vector = [
                record.get('glucose_history', [0])[-1] if record.get('glucose_history') else 0,
                record.get('carbohydrates', 0),
                record.get('protein', 0),
                record.get('fat', 0),
                record.get('exercise_duration', 0),
                record.get('stress_level', 3),
                record.get('sleep_hours', 7),
                record.get('hour', 12),
                record.get('day_of_week', 0),
                record.get('is_weekend', 0)
            ]

            features.append(feature_vector)
            labels.append(record.get('target_glucose', record.get('glucose', 5.5)))

        # 标准化特征
        user_id = user_data[0].get('user_id', 'unknown')
        if user_id not in self.user_scalers:
            self.user_scalers[user_id] = StandardScaler()

        scaler = self.user_scalers[user_id]
        features_scaled = scaler.fit_transform(features)

        return torch.FloatTensor(features_scaled), torch.FloatTensor(labels)

    def get_user_model(self, user_id: str) -> Optional[nn.Module]:
        """获取用户模型"""
        return self.user_models.get(user_id)

    def predict_for_user(self, user_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """为用户进行预测"""
        user_model = self.get_user_model(user_id)
        if user_model is None:
            return {'error': f'No model found for user {user_id}'}

        try:
            # 准备输入特征
            feature_vector = [
                input_data.get('glucose_history', [0])[-1] if input_data.get('glucose_history') else 0,
                input_data.get('carbohydrates', 0),
                input_data.get('protein', 0),
                input_data.get('fat', 0),
                input_data.get('exercise_duration', 0),
                input_data.get('stress_level', 3),
                input_data.get('sleep_hours', 7),
                input_data.get('hour', 12),
                input_data.get('day_of_week', 0),
                input_data.get('is_weekend', 0)
            ]

            # 标准化
            if user_id in self.user_scalers:
                feature_vector = self.user_scalers[user_id].transform([feature_vector])[0]

            # 预测
            user_model.eval()
            with torch.no_grad():
                input_tensor = torch.FloatTensor(feature_vector).unsqueeze(0)
                prediction = user_model(input_tensor).item()

            # 计算个性化置信度
            confidence = self._calculate_personalized_confidence(user_id, input_data)

            return {
                'prediction': prediction,
                'confidence': confidence,
                'personalized': True,
                'user_id': user_id
            }

        except Exception as e:
            logger.error(f"用户 {user_id} 预测失败: {e}")
            return {'error': str(e)}

    def _calculate_personalized_confidence(self, user_id: str, input_data: Dict[str, Any]) -> float:
        """计算个性化置信度"""
        base_confidence = 0.8

        # 基于适应历史调整置信度
        if user_id in self.adaptation_history:
            adaptation_count = len(self.adaptation_history[user_id])
            history_bonus = min(adaptation_count * 0.02, 0.15)  # 最多增加15%
            base_confidence += history_bonus

        # 基于用户档案完整性调整
        if user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            completeness = self._calculate_profile_completeness(profile)
            base_confidence += completeness * 0.1

        return min(1.0, base_confidence)

    def _calculate_profile_completeness(self, profile: UserProfile) -> float:
        """计算档案完整性"""
        completeness_score = 0.0

        # 检查各个字段的完整性
        if profile.age > 0:
            completeness_score += 0.15
        if profile.gender != 'unknown':
            completeness_score += 0.1
        if profile.bmi > 0:
            completeness_score += 0.15
        if profile.diabetes_type is not None:
            completeness_score += 0.2
        if profile.medication:
            completeness_score += 0.1
        if profile.dietary_preferences:
            completeness_score += 0.1
        if profile.baseline_glucose > 0:
            completeness_score += 0.2

        return completeness_score

class KnowledgeDistillationModule:
    """知识蒸馏模块"""

    def __init__(self, teacher_model: nn.Module, config: AdaptationConfig):
        self.teacher_model = teacher_model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_student_model(self, input_dim: int, compression_ratio: float = 0.5) -> nn.Module:
        """创建学生模型（更小的模型）"""
        # 获取教师模型的结构信息
        teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
        target_params = int(teacher_params * compression_ratio)

        # 创建更小的学生模型
        hidden_dim = 64  # 比教师模型更小的隐藏层

        student_model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )

        student_params = sum(p.numel() for p in student_model.parameters())
        logger.info(f"创建学生模型: 教师参数={teacher_params}, 学生参数={student_params}, "
                   f"压缩比={student_params/teacher_params:.2f}")

        return student_model

    def distill_knowledge(self, student_model: nn.Module, train_data: torch.Tensor,
                         temperature: float = 3.0, alpha: float = 0.7) -> Dict[str, Any]:
        """知识蒸馏训练"""
        self.teacher_model.eval()
        student_model.train()

        self.teacher_model.to(self.device)
        student_model.to(self.device)
        train_data = train_data.to(self.device)

        optimizer = torch.optim.Adam(student_model.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.fine_tune_epochs)

        distillation_losses = []

        for epoch in range(self.config.fine_tune_epochs):
            epoch_loss = 0

            for i in range(0, len(train_data), 32):
                batch_data = train_data[i:i+32]

                # 教师模型预测（软标签）
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(batch_data)
                    soft_targets = F.softmax(teacher_outputs / temperature, dim=-1)

                # 学生模型预测
                student_outputs = student_model(batch_data)
                soft_predictions = F.log_softmax(student_outputs / temperature, dim=-1)

                # 知识蒸馏损失
                distillation_loss = F.kl_div(
                    soft_predictions, soft_targets, reduction='batchmean'
                ) * (temperature ** 2)

                # 如果有真实标签，可以加上硬标签损失
                # hard_loss = F.mse_loss(student_outputs, true_labels)
                # total_loss = alpha * distillation_loss + (1 - alpha) * hard_loss

                total_loss = distillation_loss

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), self.config.gradient_clip_norm)
                optimizer.step()

                epoch_loss += total_loss.item()

            avg_loss = epoch_loss / (len(train_data) // 32 + 1)
            distillation_losses.append(avg_loss)
            scheduler.step()

            if epoch % 5 == 0:
                logger.info(f"知识蒸馏 Epoch {epoch+1}: 损失={avg_loss:.4f}")

        return {
            'distillation_losses': distillation_losses,
            'final_loss': distillation_losses[-1] if distillation_losses else 0,
            'compression_achieved': True
        }

class AdaptiveModelManager:
    """自适应模型管理器"""

    def __init__(self, base_model: nn.Module, config: AdaptationConfig):
        self.base_model = base_model
        self.config = config

        # 各个模块
        self.personalization_engine = PersonalizationEngine(config)
        self.knowledge_distiller = KnowledgeDistillationModule(base_model, config)
        self.meta_learner = MetaLearner(input_dim=10)  # 假设输入维度为10

        # 性能监控
        self.performance_tracker = defaultdict(list)
        self.adaptation_log = []

    def register_user(self, user_data: Dict[str, Any]) -> UserProfile:
        """注册新用户"""
        profile = self.personalization_engine.create_user_profile(user_data)
        logger.info(f"注册新用户: {profile.user_id}")
        return profile

    def adapt_for_user(self, user_id: str, user_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """为用户自适应模型"""
        start_time = datetime.now()

        try:
            # 检查是否需要适应
            if not self._should_adapt(user_id, user_data):
                return {
                    'adapted': False,
                    'reason': 'No adaptation needed',
                    'user_id': user_id
                }

            # 执行个性化适应
            user_model = self.personalization_engine.adapt_model_for_user(
                user_id, self.base_model, user_data
            )

            # 记录适应日志
            adaptation_record = {
                'timestamp': start_time,
                'user_id': user_id,
                'samples_used': len(user_data),
                'adaptation_time': (datetime.now() - start_time).total_seconds(),
                'success': True
            }
            self.adaptation_log.append(adaptation_record)

            return {
                'adapted': True,
                'user_id': user_id,
                'model_updated': True,
                'samples_used': len(user_data),
                'adaptation_time': adaptation_record['adaptation_time']
            }

        except Exception as e:
            logger.error(f"用户 {user_id} 适应失败: {e}")

            adaptation_record = {
                'timestamp': start_time,
                'user_id': user_id,
                'samples_used': len(user_data),
                'adaptation_time': (datetime.now() - start_time).total_seconds(),
                'success': False,
                'error': str(e)
            }
            self.adaptation_log.append(adaptation_record)

            return {
                'adapted': False,
                'error': str(e),
                'user_id': user_id
            }

    def predict_with_adaptation(self, user_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """使用自适应模型进行预测"""
        # 尝试使用个性化模型
        user_prediction = self.personalization_engine.predict_for_user(user_id, input_data)

        if 'error' not in user_prediction:
            # 记录性能
            self.performance_tracker[user_id].append({
                'timestamp': datetime.now(),
                'prediction': user_prediction['prediction'],
                'confidence': user_prediction['confidence']
            })

            return user_prediction

        # 如果个性化模型不可用，使用基础模型
        logger.warning(f"用户 {user_id} 个性化模型不可用，使用基础模型")

        try:
            # 使用基础模型预测
            feature_vector = self._extract_features(input_data)

            self.base_model.eval()
            with torch.no_grad():
                input_tensor = torch.FloatTensor(feature_vector).unsqueeze(0)
                prediction = self.base_model(input_tensor).item()

            return {
                'prediction': prediction,
                'confidence': 0.7,  # 基础模型的默认置信度
                'personalized': False,
                'user_id': user_id
            }

        except Exception as e:
            logger.error(f"基础模型预测失败: {e}")
            return {'error': str(e)}

    def _should_adapt(self, user_id: str, user_data: List[Dict[str, Any]]) -> bool:
        """判断是否需要适应"""
        # 检查数据量
        if len(user_data) < self.config.min_adaptation_samples:
            return False

        # 检查是否已有用户模型
        if user_id not in self.personalization_engine.user_models:
            return True

        # 检查性能是否下降
        if user_id in self.performance_tracker:
            recent_performance = self.performance_tracker[user_id][-10:]  # 最近10次预测
            if len(recent_performance) >= 5:
                avg_confidence = np.mean([p['confidence'] for p in recent_performance])
                if avg_confidence < (1.0 - self.config.adaptation_threshold):
                    logger.info(f"用户 {user_id} 性能下降，触发重新适应")
                    return True

        return False

    def _extract_features(self, input_data: Dict[str, Any]) -> List[float]:
        """提取特征向量"""
        return [
            input_data.get('glucose_history', [0])[-1] if input_data.get('glucose_history') else 0,
            input_data.get('carbohydrates', 0),
            input_data.get('protein', 0),
            input_data.get('fat', 0),
            input_data.get('exercise_duration', 0),
            input_data.get('stress_level', 3),
            input_data.get('sleep_hours', 7),
            input_data.get('hour', 12),
            input_data.get('day_of_week', 0),
            input_data.get('is_weekend', 0)
        ]

    def get_adaptation_report(self, user_id: str = None) -> Dict[str, Any]:
        """获取适应报告"""
        if user_id:
            # 单用户报告
            user_adaptations = [log for log in self.adaptation_log if log['user_id'] == user_id]
            user_performance = self.performance_tracker.get(user_id, [])

            return {
                'user_id': user_id,
                'total_adaptations': len(user_adaptations),
                'successful_adaptations': sum(1 for a in user_adaptations if a['success']),
                'average_adaptation_time': np.mean([a['adaptation_time'] for a in user_adaptations]) if user_adaptations else 0,
                'recent_performance': user_performance[-10:],
                'has_personalized_model': user_id in self.personalization_engine.user_models
            }
        else:
            # 全局报告
            total_adaptations = len(self.adaptation_log)
            successful_adaptations = sum(1 for a in self.adaptation_log if a['success'])

            return {
                'total_users': len(self.personalization_engine.user_models),
                'total_adaptations': total_adaptations,
                'successful_adaptations': successful_adaptations,
                'success_rate': successful_adaptations / max(total_adaptations, 1),
                'average_adaptation_time': np.mean([a['adaptation_time'] for a in self.adaptation_log]) if self.adaptation_log else 0,
                'active_users_with_models': len(self.personalization_engine.user_models)
            }

    def save_adaptation_state(self, save_path: str):
        """保存适应状态"""
        state = {
            'user_models': {uid: model.state_dict() for uid, model in self.personalization_engine.user_models.items()},
            'user_profiles': self.personalization_engine.user_profiles,
            'user_scalers': self.personalization_engine.user_scalers,
            'adaptation_history': self.personalization_engine.adaptation_history,
            'performance_tracker': dict(self.performance_tracker),
            'adaptation_log': self.adaptation_log
        }

        with open(save_path, 'wb') as f:
            pickle.dump(state, f)

        logger.info(f"适应状态已保存到: {save_path}")

    def load_adaptation_state(self, load_path: str):
        """加载适应状态"""
        try:
            with open(load_path, 'rb') as f:
                state = pickle.load(f)

            # 恢复用户模型
            for uid, model_state in state['user_models'].items():
                user_model = copy.deepcopy(self.base_model)
                user_model.load_state_dict(model_state)
                self.personalization_engine.user_models[uid] = user_model

            # 恢复其他状态
            self.personalization_engine.user_profiles = state['user_profiles']
            self.personalization_engine.user_scalers = state['user_scalers']
            self.personalization_engine.adaptation_history = state['adaptation_history']
            self.performance_tracker = defaultdict(list, state['performance_tracker'])
            self.adaptation_log = state['adaptation_log']

            logger.info(f"适应状态已从 {load_path} 加载")

        except Exception as e:
            logger.error(f"加载适应状态失败: {e}")

# 使用示例
def main():
    """使用示例"""
    # 创建基础模型
    base_model = nn.Sequential(
        nn.Linear(10, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )

    # 创建配置
    config = AdaptationConfig(
        learning_rate=0.001,
        fine_tune_epochs=10,
        min_adaptation_samples=20
    )

    # 创建适应性管理器
    manager = AdaptiveModelManager(base_model, config)

    # 注册用户
    user_data = {
        'user_id': 'user_123',
        'age': 45,
        'gender': 'male',
        'bmi': 25.5,
        'diabetes_type': 2,
        'baseline_glucose': 6.0
    }
    profile = manager.register_user(user_data)
    print(f"用户档案: {profile}")

    # 模拟用户数据进行适应
    user_training_data = []
    for i in range(50):
        sample = {
            'user_id': 'user_123',
            'glucose': 5.5 + np.random.normal(0, 0.5),
            'carbohydrates': 50 + np.random.normal(0, 10),
            'protein': 20 + np.random.normal(0, 5),
            'fat': 15 + np.random.normal(0, 3),
            'exercise_duration': 30 + np.random.normal(0, 10),
            'stress_level': 3,
            'sleep_hours': 7,
            'hour': 12,
            'day_of_week': i % 7,
            'is_weekend': (i % 7) >= 5,
            'target_glucose': 5.8 + np.random.normal(0, 0.3)
        }
        user_training_data.append(sample)

    # 执行适应
    adaptation_result = manager.adapt_for_user('user_123', user_training_data)
    print(f"适应结果: {adaptation_result}")

    # 进行预测
    test_input = {
        'glucose_history': [5.5, 5.7, 5.9],
        'carbohydrates': 60,
        'protein': 25,
        'fat': 18,
        'exercise_duration': 45,
        'stress_level': 2,
        'sleep_hours': 8,
        'hour': 14,
        'day_of_week': 1,
        'is_weekend': 0
    }

    prediction = manager.predict_with_adaptation('user_123', test_input)
    print(f"预测结果: {prediction}")

    # 获取适应报告
    report = manager.get_adaptation_report('user_123')
    print(f"适应报告: {report}")

if __name__ == "__main__":
    main()

__all__ = ["'logger'", "'AdaptationConfig'", "'UserProfile'", "'MetaLearner'", "'TransferLearningModule'", "'PersonalizationEngine'", "'KnowledgeDistillationModule'", "'AdaptiveModelManager'", "'main'"]
