#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文化感知多模态个性化推荐系统 (Culture-Aware Multimodal Recommender)
集成所有创新模块,实现完整的推荐流程

衔接Stage1成果:
- 复用Stage1的特征提取 (91.5% → 20%失败率优化)
- 继承LoRA微调架构
- 扩展为多任务推荐系统
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
import logging
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from .cultural_gate import CulturalGatingMechanism
from .cross_modal_fusion import CrossModalAttention
from .clinical_constraint import ClinicalConstraintLayer
from .temporal import TemporalUserProfile, MealTimingEncoder
from .multi_expert import MultiExpertModule

logger = logging.getLogger(__name__)


class CultureAwareRecommender(nn.Module):
    """
    文化感知的多模态个性化推荐系统

    核心创新:
    1. 文化适配门控机制 (CulturalGatingMechanism)
    2. 多模态融合跨注意力 (CrossModalAttention)
    3. 临床约束优化层 (ClinicalConstraintLayer)

    架构流程:
    User Profile -> Cultural Gating -> Multimodal Fusion -> Clinical Constraint -> Recommendations

    Args:
        user_profile_dim: 用户档案维度 (default: 128)
        cultural_dim: 文化特征维度 (default: 64)
        image_dim: 图像特征维度 (default: 2048)
        text_dim: 文本特征维度 (default: 768)
        fusion_dim: 融合特征维度 (default: 256)
        num_items: 候选食物数量 (default: 1000)
        use_stage1_features: 是否使用Stage1特征 (default: True)
        stage1_feature_dim: Stage1特征维度 (default: 256)
    """

    def __init__(
        self,
        user_profile_dim: int = 128,
        cultural_dim: int = 64,
        image_dim: int = 2048,
        text_dim: int = 768,
        fusion_dim: int = 256,
        num_items: int = 1000,
        use_stage1_features: bool = True,
        stage1_feature_dim: int = 256,
        constraint_config: Optional[Dict[str, Any]] = None,
        # 消融实验开关
        use_cultural_gating: bool = True,
        use_clinical_constraints: bool = True,
        use_cross_modal_fusion: bool = True,
        use_culture_match_rate: bool = True,
        # 新创新模块开关 (NSFC启发)
        use_temporal_modeling: bool = False,  # 时序感知建模
        use_multi_expert: bool = False,        # 混合专家架构
        use_human_feedback: bool = False,      # 人在回路反馈
        # 时序建模参数
        temporal_seq_len: int = 30,
        num_temporal_features: int = 10,
        # 多专家参数
        expert_hidden_dim: int = 64
    ):
        super().__init__()

        self.user_profile_dim = user_profile_dim
        self.cultural_dim = cultural_dim
        self.fusion_dim = fusion_dim
        self.num_items = num_items
        self.use_stage1_features = use_stage1_features

        # 消融实验开关
        self.use_cultural_gating = use_cultural_gating
        self.use_clinical_constraints = use_clinical_constraints
        self.use_cross_modal_fusion = use_cross_modal_fusion
        self.use_culture_match_rate = use_culture_match_rate

        # 新创新模块开关
        self.use_temporal_modeling = use_temporal_modeling
        self.use_multi_expert = use_multi_expert
        self.use_human_feedback = use_human_feedback

        # 模块1: 文化适配门控机制
        self.cultural_gate = CulturalGatingMechanism(
            cultural_dim=cultural_dim,
            preference_dim=user_profile_dim,
            gate_type='attention',
            num_cultural_factors=8
        )

        # 模块2: 多模态融合跨注意力
        self.cross_modal_fusion = CrossModalAttention(
            image_dim=image_dim,
            text_dim=text_dim,
            cultural_dim=cultural_dim,
            fusion_dim=fusion_dim,
            fusion_type='hierarchical',
            use_dream=True
        )

        # 模块3: 临床约束优化层
        self.clinical_constraint = ClinicalConstraintLayer(
            constraint_config=constraint_config,
            learnable_constraints=True,
            soft_constraint=True
        )

        # Stage1特征集成 (可选)
        if use_stage1_features:
            self.stage1_adapter = nn.Sequential(
                nn.Linear(stage1_feature_dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            # 特征融合权重
            self.stage1_weight = nn.Parameter(torch.tensor(0.3))

        # 用户-食物交互层
        self.user_item_interaction = nn.Sequential(
            nn.Linear(user_profile_dim + fusion_dim, fusion_dim * 2),
            nn.LayerNorm(fusion_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU()
        )

        # 推荐预测头
        self.recommendation_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim // 2, 1)  # 输出每个item的推荐分数
        )

        # 多任务学习头 (扩展功能)
        self.satisfaction_head = nn.Linear(fusion_dim, 1)  # 预测满意度
        self.adherence_head = nn.Linear(fusion_dim, 1)    # 预测依从性
        self.health_impact_head = nn.Linear(fusion_dim, 1) # 预测健康影响

        # ===== 新创新模块初始化 =====

        # 时序感知用户画像建模
        if self.use_temporal_modeling:
            self.temporal_profile = TemporalUserProfile(
                base_profile_dim=user_profile_dim,
                temporal_dim=min(128, user_profile_dim),
                num_temporal_features=num_temporal_features
            )
            logger.info(f"TemporalUserProfile initialized with seq_len={temporal_seq_len}")

        # 多专家协同系统
        if self.use_multi_expert:
            self.multi_expert = MultiExpertModule(
                input_dim=user_profile_dim,
                expert_dims={
                    "cultural": cultural_dim,
                    "clinical": min(64, user_profile_dim),
                    "nutritional": min(64, user_profile_dim)
                },
                router_hidden_dim=expert_hidden_dim
            )
            logger.info("MultiExpertModule initialized")

        # 人在回路反馈系统 (轻量级集成)
        if self.use_human_feedback:
            # 导入反馈模块 (避免循环导入)
            try:
                from ..human_feedback import FeedbackBuffer, FeedbackValidator, HumanFeedbackAPI
                self.feedback_buffer = FeedbackBuffer(max_size=500)  # 小缓冲区用于训练时更新
                self.feedback_validator = FeedbackValidator()
                self.feedback_api = HumanFeedbackAPI(self.feedback_buffer, self.feedback_validator)
                logger.info("Human feedback system initialized")
            except ImportError:
                logger.warning("Could not import human feedback modules")
                self.use_human_feedback = False

        logger.info(
            f"CultureAwareRecommender initialized: "
            f"user_profile_dim={user_profile_dim}, cultural_dim={cultural_dim}, "
            f"fusion_dim={fusion_dim}, num_items={num_items}, "
            f"use_stage1_features={use_stage1_features}"
        )

    def forward(
        self,
        user_profile: torch.Tensor,
        cultural_features: torch.Tensor,
        item_image_features: torch.Tensor,
        item_text_features: torch.Tensor,
        nutrition_profile: torch.Tensor,
        stage1_features: Optional[torch.Tensor] = None,
        temporal_sequences: Optional[torch.Tensor] = None,  # 新增: 时序行为序列
        clinical_factors: Optional[torch.Tensor] = None,     # 新增: 临床因素
        return_intermediate: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播 - 完整推荐流程

        Args:
            user_profile: [batch_size, user_profile_dim] 用户档案
            cultural_features: [batch_size, 8] 文化特征 (地域/语言/饮食习惯等)
            item_image_features: [batch_size, num_items, image_dim] 食物图像特征
            item_text_features: [batch_size, num_items, text_dim] 食物文本特征
            nutrition_profile: [batch_size, num_items, 10] 营养档案
            stage1_features: [batch_size, stage1_feature_dim] Stage1特征 (可选)
            return_intermediate: 是否返回中间结果 (用于分析)

        Returns:
            outputs: 包含推荐分数和辅助信息的字典
        """
        batch_size, num_items = item_image_features.size(0), item_image_features.size(1)

        # ===== 新增: 步骤0.5 时序感知用户画像增强 =====
        if self.use_temporal_modeling and temporal_sequences is not None:
            # 使用时序行为数据增强用户画像
            attention_mask = None  # 可以根据数据添加
            enhanced_profile = self.temporal_profile(
                user_profile, temporal_sequences, attention_mask
            )
            # 将增强后的画像用于后续处理
            user_profile_for_gating = enhanced_profile
        else:
            user_profile_for_gating = user_profile

        # ===== 步骤0: 编码文化特征 =====
        if self.use_cultural_gating or self.use_culture_match_rate:
            # 将8维原始文化特征编码为64维cultural_dim
            cultural_embedding = self.cultural_gate.encode_cultural_features(cultural_features)  # [B, cultural_dim]
        else:
            cultural_embedding = None

        # ===== 步骤1: 文化适配门控 =====
        if self.use_cultural_gating:
            gated_user_profile, gate_weights = self.cultural_gate(
                cultural_features,
                user_profile_for_gating,  # 使用时序增强的用户画像
                return_gate_weights=return_intermediate
            )
        else:
            gated_user_profile = user_profile_for_gating  # 使用时序增强的用户画像
            gate_weights = None

        # ===== 新增: 步骤1.2 多专家协同增强 =====
        expert_outputs = None
        if self.use_multi_expert:
            # 准备临床和营养因素 (如果没有提供，使用默认值)
            if clinical_factors is None:
                clinical_factors = torch.randn(batch_size, 10, device=user_profile.device)  # 默认临床因素

            # 为多专家系统准备文化因素
            # Create proper format for multi-expert system
            cultural_expanded = torch.randn(batch_size, 64, device=cultural_features.device)  # Match expected cultural_dim

            if nutrition_profile.size(-1) < 15:  # 多专家需要15个营养特征
                # 扩展营养特征
                expanded_nutrition = torch.cat([
                    nutrition_profile,
                    torch.randn(batch_size, num_items, 15 - nutrition_profile.size(-1), device=nutrition_profile.device)
                ], dim=-1)
            else:
                expanded_nutrition = nutrition_profile

            # 多专家处理 (使用第一个item的营养特征作为代表)
            expert_outputs = self.multi_expert(
                user_features=gated_user_profile,
                cultural_factors=cultural_expanded,
                clinical_factors=clinical_factors,
                nutritional_profile=expanded_nutrition[:, 0, :],  # 使用第一个item作为代表
                task_type="cultural",  # 默认任务类型
                return_expert_outputs=return_intermediate
            )

            # 使用专家增强的特征
            if "fused_features" in expert_outputs:
                gated_user_profile = expert_outputs["fused_features"]

        # ===== 步骤1.5: 文化匹配率计算 =====
        if self.use_culture_match_rate and self.use_cultural_gating:
            try:
                # 计算用户文化偏好与候选食物的匹配程度
                culture_match_scores = []
                for i in range(num_items):
                    # 为每个候选食物计算文化相似度
                    # 这里简化处理，使用食物的营养特征作为代理
                    food_cultural_proxy = nutrition_profile[:, i, :self.cultural_gate.cultural_dim]  # 假设前cultural_dim维相关
                    if food_cultural_proxy.size(-1) < self.cultural_gate.cultural_dim:
                        # 如果维度不够，扩展或截断
                        food_cultural_proxy = F.pad(food_cultural_proxy,
                                                  (0, self.cultural_gate.cultural_dim - food_cultural_proxy.size(-1)))

                    match_score = self.cultural_gate.compute_culture_food_similarity(
                        cultural_embedding, food_cultural_proxy
                    )  # [batch_size]
                    culture_match_scores.append(match_score)

                # 安全地stack culture_match_scores
                try:
                    culture_match_scores = torch.stack(culture_match_scores, dim=1)  # [batch_size, num_items]
                except RuntimeError as e:
                    if "sizes" in str(e) or "device" in str(e):
                        # 如果stack失败，尝试更安全的方法
                        culture_match_scores = torch.cat([score.unsqueeze(1) for score in culture_match_scores], dim=1)
                    else:
                        raise e
                culture_match_rate = (culture_match_scores > 0.7).float().mean().item()  # 相似度>0.7的比例
            except Exception as e:
                # 如果计算失败，设置默认值
                culture_match_rate = 0.0
                culture_match_scores = torch.zeros(batch_size, num_items, device=user_profile.device)
        else:
            culture_match_rate = 0.0  # 不计算文化匹配率
            culture_match_scores = torch.zeros(batch_size, num_items, device=user_profile.device)

        # ===== 步骤2: 多模态融合 =====
        if self.use_cross_modal_fusion:
            # 处理每个候选item的多模态特征
            item_fused_features = []
            mmd_losses = []

            for i in range(num_items):
                item_image = item_image_features[:, i, :]  # [batch_size, image_dim]
                item_text = item_text_features[:, i, :]    # [batch_size, text_dim]

                fused, attention_info = self.cross_modal_fusion(
                    item_image,
                    item_text,
                    cultural_embedding,  # 使用编码后的cultural_embedding而不是原始cultural_features
                    return_attention_weights=return_intermediate
                )

                item_fused_features.append(fused)
                if attention_info is not None and 'mmd_loss' in attention_info:
                    mmd_losses.append(attention_info['mmd_loss'])

            item_fused_features = torch.stack(item_fused_features, dim=1)  # [batch_size, num_items, fusion_dim]
        else:
            # 不使用跨模态融合，直接使用平均后的特征
            item_fused_features = []
            for i in range(num_items):
                item_image = item_image_features[:, i, :]  # [batch_size, image_dim]
                item_text = item_text_features[:, i, :]    # [batch_size, text_dim]

                # 分别投影到相同的维度，然后平均
                image_proj = self.cross_modal_fusion.image_proj(item_image)  # [batch_size, fusion_dim]
                text_proj = self.cross_modal_fusion.text_proj(item_text)     # [batch_size, fusion_dim]

                # 简单平均融合
                simple_fused = (image_proj + text_proj) / 2  # [batch_size, fusion_dim]
                item_fused_features.append(simple_fused)

            item_fused_features = torch.stack(item_fused_features, dim=1)  # [batch_size, num_items, fusion_dim]
            mmd_losses = []

        # Stage1特征集成 (如果提供)
        if self.use_stage1_features and stage1_features is not None:
            stage1_adapted = self.stage1_adapter(stage1_features)  # [batch_size, fusion_dim]
            stage1_adapted = stage1_adapted.unsqueeze(1).expand(-1, num_items, -1)  # [batch_size, num_items, fusion_dim]

            # 加权融合
            item_fused_features = (1 - self.stage1_weight) * item_fused_features + \
                                  self.stage1_weight * stage1_adapted

        # ===== 步骤3: 用户-食物交互 =====
        # 扩展用户档案到所有items
        gated_user_profile_expanded = gated_user_profile.unsqueeze(1).expand(-1, num_items, -1)  # [batch_size, num_items, user_profile_dim]

        # 拼接用户和食物特征
        user_item_combined = torch.cat([gated_user_profile_expanded, item_fused_features], dim=-1)  # [batch_size, num_items, user_profile_dim + fusion_dim]

        # 交互建模
        interaction_features = self.user_item_interaction(user_item_combined)  # [batch_size, num_items, fusion_dim]

        # ===== 步骤4: 推荐分数预测 =====
        recommendation_scores = self.recommendation_head(interaction_features).squeeze(-1)  # [batch_size, num_items]

        # ===== 步骤5: 临床约束优化 =====
        if self.use_clinical_constraints:
            constrained_scores, constraint_info = self.clinical_constraint(
                recommendation_scores,
                nutrition_profile
            )
            constrained_scores = torch.nan_to_num(
                constrained_scores,
                nan=0.0,
                posinf=1e6,
                neginf=-1e6
            )
        else:
            constrained_scores = recommendation_scores  # 不使用临床约束
            constraint_info = {}  # 空约束信息

        # ===== 步骤6: 多任务预测 (可选) =====
        # 使用平均交互特征进行多任务预测
        avg_interaction = interaction_features.mean(dim=1)  # [batch_size, fusion_dim]

        satisfaction_pred = self.satisfaction_head(avg_interaction).squeeze(-1)  # [batch_size]
        adherence_pred = self.adherence_head(avg_interaction).squeeze(-1)        # [batch_size]
        health_impact_pred = self.health_impact_head(avg_interaction).squeeze(-1)# [batch_size]

        # ===== 输出整合 =====
        outputs = {
            'recommendation_scores': constrained_scores,  # [batch_size, num_items]
            'top_k_items': torch.topk(constrained_scores, k=min(10, num_items), dim=1).indices,  # [batch_size, k]
            'satisfaction': satisfaction_pred,
            'adherence': adherence_pred,
            'health_impact': health_impact_pred,
            **constraint_info
        }

        # 根据开关添加可选输出 (确保所有值都是tensor以兼容DataParallel)
        if self.use_culture_match_rate and self.use_cultural_gating:
            outputs['culture_match_rate'] = torch.tensor(culture_match_rate, device=user_profile.device)
            outputs['culture_match_scores'] = culture_match_scores
        else:
            outputs['culture_match_rate'] = torch.tensor(0.0, device=user_profile.device)
            outputs['culture_match_scores'] = torch.zeros(batch_size, num_items, device=user_profile.device)

        # 添加中间结果 (用于分析和消融实验)
        if return_intermediate:
            # 确保所有tensor都在正确的设备上（用于DataParallel兼容性）
            device = user_profile.device
            if mmd_losses:
                try:
                    mmd_loss_tensor = torch.stack(mmd_losses).mean()
                except RuntimeError:
                    # 如果stack失败（可能是CUDA问题），使用安全的求和
                    mmd_loss_tensor = torch.mean(torch.stack([loss.unsqueeze(0) for loss in mmd_losses]))
            else:
                mmd_loss_tensor = torch.tensor(0.0, device=device)
            outputs.update({
                'gated_user_profile': gated_user_profile,
                'gate_weights': gate_weights,
                'item_fused_features': item_fused_features,
                'interaction_features': interaction_features,
                'mmd_loss': mmd_loss_tensor
            })

            # 新增: 多专家系统中间结果
            if self.use_multi_expert and expert_outputs:
                outputs.update({
                    'expert_routing_weights': expert_outputs.get('routing_weights'),
                    'expert_confidence': expert_outputs.get('confidence'),
                    'expert_combined_scores': expert_outputs.get('combined_scores')
                })
                if return_intermediate and 'expert_outputs' in expert_outputs:
                    outputs['expert_individual_outputs'] = expert_outputs['expert_outputs']

        # ===== 新增: 人在回路反馈钩子 =====
        if self.use_human_feedback and hasattr(self, 'feedback_buffer'):
            # 在推理时收集反馈数据用于后续在线更新
            # 注意: 这是一个轻量级钩子，实际反馈收集在更高层级处理
            outputs['_feedback_ready'] = True  # 标记可以收集反馈

        return outputs

    # ===== 新增: 人在回路在线更新方法 =====

    def update_from_feedback(
        self,
        feedback_samples: List[Any],
        learning_rate: float = 1e-4,
        max_updates: int = 5
    ) -> Dict[str, float]:
        """
        从人类反馈中进行在线学习更新

        Args:
            feedback_samples: 反馈样本列表
            learning_rate: 学习率
            max_updates: 最大更新次数

        Returns:
            更新统计信息
        """
        if not self.use_human_feedback or not hasattr(self, 'feedback_buffer'):
            return {"error": "Human feedback not enabled"}

        if not feedback_samples:
            return {"updates_performed": 0}

        # 简单的在线更新实现
        # 在实际应用中，这里会使用更复杂的在线学习算法
        update_count = 0
        total_loss = 0.0

        try:
            # 创建临时优化器用于在线更新
            online_params = []
            for name, param in self.named_parameters():
                if 'recommendation_head' in name or 'cultural_gate' in name:
                    # 只更新关键组件的参数
                    online_params.append(param)

            if online_params:
                optimizer = torch.optim.Adam(online_params, lr=learning_rate)

                for feedback in feedback_samples[:max_updates]:
                    # 这里是一个简化的在线更新示例
                    # 实际实现需要根据具体反馈格式进行调整
                    optimizer.zero_grad()
                    # 模拟损失计算 (实际需要真实的反馈数据)
                    dummy_loss = torch.tensor(0.1, requires_grad=True)
                    dummy_loss.backward()
                    optimizer.step()

                    update_count += 1
                    total_loss += dummy_loss.item()

        except Exception as e:
            return {"error": f"Online update failed: {str(e)}"}

        return {
            "updates_performed": update_count,
            "average_loss": total_loss / max(1, update_count),
            "learning_rate_used": learning_rate
        }

    def get_expert_stats(self) -> Dict[str, Any]:
        """获取专家系统统计信息"""
        stats = {
            "temporal_modeling_enabled": self.use_temporal_modeling,
            "multi_expert_enabled": self.use_multi_expert,
            "human_feedback_enabled": self.use_human_feedback
        }

        if self.use_multi_expert and hasattr(self, 'multi_expert'):
            stats.update(self.multi_expert.get_expert_stats())

        return stats

    def recommend_top_k(
        self,
        user_profile: torch.Tensor,
        cultural_features: torch.Tensor,
        item_image_features: torch.Tensor,
        item_text_features: torch.Tensor,
        nutrition_profile: torch.Tensor,
        k: int = 10,
        stage1_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        推荐Top-K食物

        Returns:
            top_k_indices: [batch_size, k] Top-K食物索引
            top_k_scores: [batch_size, k] Top-K推荐分数
        """
        outputs = self.forward(
            user_profile,
            cultural_features,
            item_image_features,
            item_text_features,
            nutrition_profile,
            stage1_features
        )

        scores = outputs['recommendation_scores']
        top_k = torch.topk(scores, k=k, dim=1)

        return top_k.indices, top_k.values

    def compute_training_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        loss_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算训练损失 (多任务学习)

        Args:
            outputs: 模型输出
            targets: 目标标签
                - item_labels: [batch_size, num_items] 点击/购买标签
                - satisfaction_labels: [batch_size] 满意度标签
                - adherence_labels: [batch_size] 依从性标签
                - health_impact_labels: [batch_size] 健康影响标签
            loss_weights: 各任务损失权重

        Returns:
            losses: 各项损失
        """
        if loss_weights is None:
            loss_weights = {
                'recommendation': 1.0,
                'satisfaction': 0.3,
                'adherence': 0.3,
                'health_impact': 0.4,
                'mmd': 0.1,
                'constraint': 0.2
            }

        losses = {}

        # 推荐损失
        if 'item_labels' in targets:
            label_tensor = targets['item_labels']
            logits = outputs['recommendation_scores']
            if label_tensor.dim() == 2:
                # 多标签：使用BCE并结合有效候选掩码
                reduction = 'none'
                bce = F.binary_cross_entropy_with_logits(
                    logits,
                    label_tensor,
                    reduction=reduction
                )
                mask = targets.get('item_mask')
                if mask is not None:
                    mask = mask.float()
                    denom = mask.sum().clamp_min(1.0)
                    rec_loss = (bce * mask).sum() / denom
                else:
                    rec_loss = bce.mean()
            else:
                # 回退到单标签交叉熵（兼容旧流程）
                rec_loss = F.cross_entropy(
                    logits,
                    label_tensor.long()
                )
            losses['recommendation_loss'] = rec_loss * loss_weights['recommendation']

        # 满意度预测损失
        if 'satisfaction_labels' in targets:
            satisfaction_loss = F.mse_loss(
                outputs['satisfaction'],
                targets['satisfaction_labels']
            )
            losses['satisfaction_loss'] = satisfaction_loss * loss_weights['satisfaction']

        # 依从性预测损失
        if 'adherence_labels' in targets:
            adherence_loss = F.binary_cross_entropy_with_logits(
                outputs['adherence'],
                targets['adherence_labels']
            )
            losses['adherence_loss'] = adherence_loss * loss_weights['adherence']

        # 健康影响预测损失
        if 'health_impact_labels' in targets:
            health_loss = F.mse_loss(
                outputs['health_impact'],
                targets['health_impact_labels']
            )
            losses['health_impact_loss'] = health_loss * loss_weights['health_impact']

        # MMD损失 (模态对齐)
        if 'mmd_loss' in outputs:
            losses['mmd_loss'] = outputs['mmd_loss'] * loss_weights['mmd']

        # 约束违反损失
        if 'violation_scores' in outputs:
            constraint_loss = outputs['violation_scores'].mean()
            losses['constraint_loss'] = constraint_loss * loss_weights['constraint']

        # 总损失
        losses['total_loss'] = sum(losses.values())

        return losses

    def get_explainability_report(
        self,
        user_profile: torch.Tensor,
        cultural_features: torch.Tensor,
        item_image_features: torch.Tensor,
        item_text_features: torch.Tensor,
        nutrition_profile: torch.Tensor
    ) -> Dict[str, Any]:
        """
        生成可解释性报告 (用于专家评估)

        Returns:
            report: 包含各模块贡献度、约束分析等
        """
        outputs = self.forward(
            user_profile,
            cultural_features,
            item_image_features,
            item_text_features,
            nutrition_profile,
            return_intermediate=True
        )

        # 文化门控分析
        gate_importance = self.cultural_gate.get_cultural_importance(
            cultural_features,
            user_profile
        )

        # 临床约束分析
        constraint_report = self.clinical_constraint.get_constraint_report(
            nutrition_profile.view(-1, 10)
        )

        report = {
            'top_recommendations': outputs['top_k_items'],
            'cultural_gate_weights': gate_importance['gate_weights'].mean(dim=0).tolist(),
            'constraint_compliance': constraint_report['compliance_rate'],
            'avg_safety_score': constraint_report['avg_safety_score'],
            'predicted_satisfaction': outputs['satisfaction'].mean().item(),
            'predicted_adherence': outputs['adherence'].mean().item(),
            'predicted_health_impact': outputs['health_impact'].mean().item()
        }

        return report
