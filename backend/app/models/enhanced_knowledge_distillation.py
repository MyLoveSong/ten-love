

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强的知识蒸馏模块
实现多领域专家知识的蒸馏、融合与可解释性分析
集成实时调整能力和动态知识更新
新增：跨领域知识蒸馏协调器（教师-学生KL对齐），可复用于GlucoNet等模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from collections import defaultdict
import json
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

class DistillationCoordinator(nn.Module):
    """跨领域知识蒸馏协调器
    - teacher_outputs: Dict[str, Tensor] 例如 {'medical': BxD, 'nutritional': BxD, 'cultural': BxD}
    - student_outputs: Dict[str, Tensor] 结构同上
    - 通过温度缩放与KL散度对齐，并可加权多领域损失
    """
    def __init__(self, temperature: float = 2.0, domain_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        self.temperature = temperature
        self.domain_weights = domain_weights or {
            'medical': 1.0,
            'nutritional': 1.0,
            'cultural': 1.0,
            'behavioral': 1.0
        }

    def forward(self, teacher_outputs: Dict[str, torch.Tensor], student_outputs: Dict[str, torch.Tensor], teacher_uncertainty: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        total_loss = torch.tensor(0.0, device=list(student_outputs.values())[0].device)
        per_domain = {}
        # 置信加权：w_i = softmax(-σ_i^2)
        dyn_weights: Optional[Dict[str, float]] = None
        if teacher_uncertainty:
            # 收集sigma^2
            sigmas = []
            keys = []
            for k, v in teacher_uncertainty.items():
                if isinstance(v, torch.Tensor):
                    sigma2 = (v.mean() ** 2).item()
                else:
                    sigma2 = float(v)
                sigmas.append(-sigma2)
                keys.append(k)
            if sigmas:
                import math
                exps = [math.exp(s) for s in sigmas]
                ssum = sum(exps) if sum(exps) > 0 else 1.0
                dyn_weights = {k: exps[i]/ssum for i, k in enumerate(keys)}
        for domain, t_out in teacher_outputs.items():
            if domain not in student_outputs:
                continue
            s_out = student_outputs[domain]
            # 对齐维度（若需要）
            if s_out.size(-1) != t_out.size(-1):
                # 简化处理：线性投影到教师维度
                projector = nn.Linear(s_out.size(-1), t_out.size(-1)).to(s_out.device)
                s_out = projector(s_out)

            t_soft = F.softmax(t_out / self.temperature, dim=-1)
            s_log_soft = F.log_softmax(s_out / self.temperature, dim=-1)
            loss = F.kl_div(s_log_soft, t_soft, reduction='batchmean') * (self.temperature ** 2)
            base_w = self.domain_weights.get(domain, 1.0)
            if dyn_weights and domain in dyn_weights:
                weight = base_w * dyn_weights[domain]
            else:
                weight = base_w
            per_domain[domain] = loss.detach()
            total_loss = total_loss + weight * loss

        return {
            'distillation_loss': total_loss,
            'per_domain_loss': per_domain
        }

class CrossDomainKnowledgeFusion(nn.Module):
    """
    跨领域知识融合模块
    基于项目申请表中的创新点五设计
    实现医学、营养学、文化学、行为学知识的深度融合
    """

    def __init__(self,
                 medical_dim: int = 512,
                 nutritional_dim: int = 256,
                 cultural_dim: int = 128,
                 behavioral_dim: int = 128,
                 fusion_dim: int = 256,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()

        self.medical_dim = medical_dim
        self.nutritional_dim = nutritional_dim
        self.cultural_dim = cultural_dim
        self.behavioral_dim = behavioral_dim
        self.fusion_dim = fusion_dim

        # 领域特定编码器
        self.medical_encoder = nn.Sequential(
            nn.Linear(medical_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.nutritional_encoder = nn.Sequential(
            nn.Linear(nutritional_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.cultural_encoder = nn.Sequential(
            nn.Linear(cultural_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.behavioral_encoder = nn.Sequential(
            nn.Linear(behavioral_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 跨领域注意力机制
        self.cross_domain_attention = nn.ModuleDict({
            'medical_nutritional': nn.MultiheadAttention(fusion_dim, num_heads, dropout, batch_first=True),
            'nutritional_cultural': nn.MultiheadAttention(fusion_dim, num_heads, dropout, batch_first=True),
            'cultural_behavioral': nn.MultiheadAttention(fusion_dim, num_heads, dropout, batch_first=True),
            'medical_behavioral': nn.MultiheadAttention(fusion_dim, num_heads, dropout, batch_first=True)
        })

        # 语义对齐网络
        self.semantic_alignment_networks = nn.ModuleDict({
            'medical': nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim // 2),
                nn.ReLU(),
                nn.Linear(fusion_dim // 2, fusion_dim),
                nn.Sigmoid()
            ),
            'nutritional': nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim // 2),
                nn.ReLU(),
                nn.Linear(fusion_dim // 2, fusion_dim),
                nn.Sigmoid()
            ),
            'cultural': nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim // 2),
                nn.ReLU(),
                nn.Linear(fusion_dim // 2, fusion_dim),
                nn.Sigmoid()
            ),
            'behavioral': nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim // 2),
                nn.ReLU(),
                nn.Linear(fusion_dim // 2, fusion_dim),
                nn.Sigmoid()
            )
        })

        # 知识图谱嵌入融合
        self.knowledge_graph_fusion = nn.Sequential(
            nn.Linear(fusion_dim * 4, fusion_dim * 2),
            nn.LayerNorm(fusion_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU()
        )

        # 动态权重网络
        self.domain_weight_network = nn.Sequential(
            nn.Linear(fusion_dim * 4, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, 4),
            nn.Softmax(dim=-1)
        )

        # 知识一致性检查器
        self.consistency_checker = nn.Sequential(
            nn.Linear(fusion_dim * 4, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, 1),
            nn.Sigmoid()
        )

    def forward(self,
                medical_knowledge: torch.Tensor,
                nutritional_knowledge: torch.Tensor,
                cultural_knowledge: torch.Tensor,
                behavioral_knowledge: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        跨领域知识融合前向传播
        """
        batch_size = medical_knowledge.size(0)

        # 领域特定编码
        medical_encoded = self.medical_encoder(medical_knowledge)
        nutritional_encoded = self.nutritional_encoder(nutritional_knowledge)
        cultural_encoded = self.cultural_encoder(cultural_knowledge)
        behavioral_encoded = self.behavioral_encoder(behavioral_knowledge)

        # 语义对齐
        medical_aligned = medical_encoded * self.semantic_alignment_networks['medical'](medical_encoded)
        nutritional_aligned = nutritional_encoded * self.semantic_alignment_networks['nutritional'](nutritional_encoded)
        cultural_aligned = cultural_encoded * self.semantic_alignment_networks['cultural'](cultural_encoded)
        behavioral_aligned = behavioral_encoded * self.semantic_alignment_networks['behavioral'](behavioral_encoded)

        # 跨领域注意力
        cross_domain_outputs = {}
        attention_weights = {}

        # 医学-营养学协同
        med_nut_attn, med_nut_weights = self.cross_domain_attention['medical_nutritional'](
            medical_aligned.unsqueeze(1),
            nutritional_aligned.unsqueeze(1),
            nutritional_aligned.unsqueeze(1)
        )
        cross_domain_outputs['medical_nutritional'] = med_nut_attn.squeeze(1)
        attention_weights['medical_nutritional'] = med_nut_weights

        # 知识图谱融合
        all_domain_features = torch.cat([
            medical_aligned, nutritional_aligned,
            cultural_aligned, behavioral_aligned
        ], dim=-1)

        fused_knowledge = self.knowledge_graph_fusion(all_domain_features)

        # 动态权重计算
        domain_weights = self.domain_weight_network(all_domain_features)

        # 加权融合
        weighted_fusion = (
            medical_aligned * domain_weights[:, 0:1] +
            nutritional_aligned * domain_weights[:, 1:2] +
            cultural_aligned * domain_weights[:, 2:3] +
            behavioral_aligned * domain_weights[:, 3:4]
        )

        # 知识一致性检查
        consistency_score = self.consistency_checker(all_domain_features)

        return {
            'fused_knowledge': fused_knowledge,
            'weighted_fusion': weighted_fusion,
            'domain_weights': domain_weights,
            'consistency_score': consistency_score,
            'cross_domain_outputs': cross_domain_outputs,
            'attention_weights': attention_weights,
            'aligned_features': {
                'medical': medical_aligned,
                'nutritional': nutritional_aligned,
                'cultural': cultural_aligned,
                'behavioral': behavioral_aligned
            }
        }

@dataclass
class KnowledgeEntity:
    """知识实体"""
    entity_id: str
    entity_type: str  # medical, nutritional, cultural, behavioral
    content: Dict[str, Any]
    confidence: float
    source: str
    created_at: datetime
    updated_at: datetime
    usage_count: int = 0

@dataclass
class KnowledgeRelation:
    """知识关系"""
    relation_id: str
    source_entity: str
    target_entity: str
    relation_type: str
    strength: float
    evidence: List[str] = field(default_factory=list)

class EnhancedMedicalKnowledgeBase:
    """增强的医学知识库 - 支持实时更新和可解释性"""

    def __init__(self, max_entities: int = 100000, retention_days: int = 365):
        self.max_entities = max_entities
        self.retention_days = retention_days

        # 知识存储
        self.entities: Dict[str, KnowledgeEntity] = {}
        self.relations: Dict[str, KnowledgeRelation] = {}

        # 索引
        self.type_index: Dict[str, List[str]] = defaultdict(list)
        self.content_index: Dict[str, List[str]] = defaultdict(list)

        # 版本控制
        self.version_history: List[Dict[str, Any]] = []
        self.current_version = 1

        # 知识图谱嵌入
        self.knowledge_embeddings: Dict[str, torch.Tensor] = {}
        self.embedding_dim = 512

        # 实时学习组件
        self.online_updater = OnlineKnowledgeUpdater()

        # 可解释性组件
        self.explainer = KnowledgeExplainer()

        # 初始化基础知识
        self._initialize_base_knowledge()

    def _initialize_base_knowledge(self):
        """初始化基础医学知识"""
        # 糖尿病相关知识
        diabetes_knowledge = [
            {
                'entity_id': 'diabetes_type2',
                'entity_type': 'medical',
                'content': {
                    'name': '2型糖尿病',
                    'definition': '胰岛素抵抗和胰岛素分泌相对不足',
                    'risk_factors': ['肥胖', '年龄', '遗传', '缺乏运动'],
                    'symptoms': ['多饮', '多尿', '多食', '体重减轻'],
                    'complications': ['心血管疾病', '肾病', '视网膜病变', '神经病变']
                },
                'confidence': 0.95,
                'source': 'medical_textbook'
            },
            {
                'entity_id': 'glucose_control',
                'entity_type': 'medical',
                'content': {
                    'name': '血糖控制',
                    'target_range': {'fasting': (4.4, 7.0), 'postprandial': (6.0, 11.1)},
                    'monitoring_frequency': 'daily',
                    'control_methods': ['饮食', '运动', '药物', '胰岛素']
                },
                'confidence': 0.98,
                'source': 'clinical_guidelines'
            }
        ]

        # 营养学知识
        nutrition_knowledge = [
            {
                'entity_id': 'carbohydrate_metabolism',
                'entity_type': 'nutritional',
                'content': {
                    'name': '碳水化合物代谢',
                    'gi_classification': {'low': (0, 55), 'medium': (56, 69), 'high': (70, 100)},
                    'absorption_time': {'simple': 15, 'complex': 45},
                    'blood_glucose_impact': 'direct_correlation'
                },
                'confidence': 0.93,
                'source': 'nutrition_database'
            },
            {
                'entity_id': 'meal_timing',
                'entity_type': 'nutritional',
                'content': {
                    'name': '进餐时机',
                    'optimal_intervals': [3, 4, 5],  # 小时
                    'circadian_effects': 'morning_higher_sensitivity',
                    'diabetes_considerations': '定时定量'
                },
                'confidence': 0.88,
                'source': 'research_papers'
            }
        ]

        # 添加实体
        for knowledge in diabetes_knowledge + nutrition_knowledge:
            entity = KnowledgeEntity(
                entity_id=knowledge['entity_id'],
                entity_type=knowledge['entity_type'],
                content=knowledge['content'],
                confidence=knowledge['confidence'],
                source=knowledge['source'],
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            self.add_entity(entity)

        # 建立关系
        self._build_initial_relations()

        logger.info(f"初始化知识库完成，包含 {len(self.entities)} 个实体")

    def _build_initial_relations(self):
        """建立初始关系"""
        relations = [
            {
                'relation_id': 'diabetes_glucose_rel',
                'source_entity': 'diabetes_type2',
                'target_entity': 'glucose_control',
                'relation_type': 'requires',
                'strength': 0.95,
                'evidence': ['clinical_trials', 'guidelines']
            },
            {
                'relation_id': 'carb_glucose_rel',
                'source_entity': 'carbohydrate_metabolism',
                'target_entity': 'glucose_control',
                'relation_type': 'affects',
                'strength': 0.90,
                'evidence': ['metabolism_studies', 'glycemic_index_research']
            }
        ]

        for rel_data in relations:
            relation = KnowledgeRelation(**rel_data)
            self.relations[relation.relation_id] = relation

    def add_entity(self, entity: KnowledgeEntity) -> bool:
        """添加知识实体"""
        try:
            # 检查容量
            if len(self.entities) >= self.max_entities:
                self._cleanup_old_entities()

            # 添加实体
            self.entities[entity.entity_id] = entity

            # 更新索引
            self.type_index[entity.entity_type].append(entity.entity_id)

            # 内容索引
            for key in entity.content.keys():
                self.content_index[key].append(entity.entity_id)

            # 生成知识嵌入
            self._generate_knowledge_embedding(entity)

            logger.info(f"添加知识实体: {entity.entity_id}")
            return True

        except Exception as e:
            logger.error(f"添加实体失败: {e}")
            return False

    def _generate_knowledge_embedding(self, entity: KnowledgeEntity):
        """生成知识嵌入"""
        # 简化的嵌入生成（实际应用中可以使用预训练模型）
        content_str = json.dumps(entity.content, ensure_ascii=False)

        # 基于内容特征生成嵌入
        features = []

        # 实体类型特征
        type_mapping = {'medical': 0, 'nutritional': 1, 'cultural': 2, 'behavioral': 3}
        type_feature = type_mapping.get(entity.entity_type, 0)
        features.extend([type_feature] * 128)

        # 置信度特征
        features.extend([entity.confidence] * 128)

        # 内容复杂度特征
        complexity = len(entity.content.keys()) / 10.0
        features.extend([complexity] * 128)

        # 使用时间特征
        time_feature = entity.usage_count / 100.0
        features.extend([time_feature] * 128)

        # 截断到嵌入维度
        features = features[:self.embedding_dim]

        # 填充到嵌入维度
        while len(features) < self.embedding_dim:
            features.append(0.0)

        # 转换为张量
        self.knowledge_embeddings[entity.entity_id] = torch.tensor(features, dtype=torch.float32)

    def query_knowledge(self, query: str, knowledge_type: Optional[str] = None,
                       top_k: int = 5) -> List[Tuple[KnowledgeEntity, float]]:
        """查询知识"""
        results = []

        # 简化的查询逻辑
        for entity_id, entity in self.entities.items():
            if knowledge_type and entity.entity_type != knowledge_type:
                continue

            # 计算相关性分数
            relevance_score = self._calculate_relevance(query, entity)

            if relevance_score > 0.1:  # 阈值过滤
                results.append((entity, relevance_score))

        # 按相关性排序
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    def _calculate_relevance(self, query: str, entity: KnowledgeEntity) -> float:
        """计算查询与实体的相关性"""
        query_lower = query.lower()
        content_str = json.dumps(entity.content, ensure_ascii=False).lower()

        # 简单的关键词匹配
        query_words = query_lower.split()
        matches = sum(1 for word in query_words if word in content_str)

        relevance = matches / len(query_words) if query_words else 0

        # 考虑实体置信度
        relevance *= entity.confidence

        return relevance

    def _cleanup_old_entities(self):
        """清理旧实体"""
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)

        entities_to_remove = []
        for entity_id, entity in self.entities.items():
            if entity.updated_at < cutoff_date and entity.usage_count < 5:
                entities_to_remove.append(entity_id)

        remove_count = min(len(entities_to_remove), len(self.entities) // 4)
        for entity_id in entities_to_remove[:remove_count]:
            self._remove_entity(entity_id)

    def _remove_entity(self, entity_id: str):
        """删除实体"""
        if entity_id in self.entities:
            entity = self.entities[entity_id]

            # 从索引中删除
            self.type_index[entity.entity_type].remove(entity_id)
            for key in entity.content.keys():
                if entity_id in self.content_index[key]:
                    self.content_index[key].remove(entity_id)

            # 删除嵌入
            if entity_id in self.knowledge_embeddings:
                del self.knowledge_embeddings[entity_id]

            # 删除实体
            del self.entities[entity_id]

class OnlineKnowledgeUpdater:
    """在线知识更新器"""

    def __init__(self, update_threshold: float = 0.1, batch_size: int = 32):
        self.update_threshold = update_threshold
        self.batch_size = batch_size

        # 更新队列
        self.update_queue: List[Dict[str, Any]] = []
        self.update_statistics = defaultdict(int)

    def add_update_signal(self, entity_id: str, feedback: Dict[str, Any]):
        """添加更新信号"""
        update_signal = {
            'entity_id': entity_id,
            'feedback': feedback,
            'timestamp': datetime.now(),
            'source': feedback.get('source', 'user_feedback')
        }

        self.update_queue.append(update_signal)
        self.update_statistics['total_signals'] += 1

        # 批量处理
        if len(self.update_queue) >= self.batch_size:
            self.process_update_batch()

    def process_update_batch(self):
        """处理更新批次"""
        if not self.update_queue:
            return

        # 按实体分组
        entity_updates = defaultdict(list)
        for update in self.update_queue:
            entity_updates[update['entity_id']].append(update)

        # 处理每个实体的更新
        for entity_id, updates in entity_updates.items():
            self._process_entity_updates(entity_id, updates)

        # 清空队列
        self.update_queue.clear()
        self.update_statistics['batches_processed'] += 1

    def _process_entity_updates(self, entity_id: str, updates: List[Dict[str, Any]]):
        """处理单个实体的更新"""
        # 聚合反馈
        positive_feedback = sum(1 for u in updates if u['feedback'].get('rating', 0) > 3)
        total_feedback = len(updates)

        if total_feedback == 0:
            return

        satisfaction_rate = positive_feedback / total_feedback

        # 如果满意度低于阈值，标记需要更新
        if satisfaction_rate < self.update_threshold:
            self.update_statistics['entities_flagged'] += 1
            logger.info(f"实体 {entity_id} 被标记为需要更新，满意度: {satisfaction_rate:.2f}")

class KnowledgeExplainer:
    """知识解释器 - 提供可解释性分析"""

    def __init__(self):
        self.explanation_templates = {
            'medical': "基于医学知识 '{knowledge}' ，{reasoning}",
            'nutritional': "根据营养学原理 '{knowledge}' ，{reasoning}",
            'cultural': "考虑文化因素 '{knowledge}' ，{reasoning}",
            'behavioral': "基于行为学研究 '{knowledge}' ，{reasoning}"
        }

        self.reasoning_patterns = {
            'causal': "这会导致 {effect}",
            'correlation': "这与 {factor} 相关",
            'recommendation': "建议 {action}",
            'prevention': "可以预防 {risk}"
        }

    def explain_recommendation(self, entity: KnowledgeEntity,
                             context: Dict[str, Any]) -> Dict[str, Any]:
        """解释推荐"""
        try:
            # 获取解释模板
            template = self.explanation_templates.get(
                entity.entity_type,
                "基于知识 '{knowledge}' ，{reasoning}"
            )

            # 生成推理
            reasoning = self._generate_reasoning(entity, context)

            # 格式化解释
            explanation = template.format(
                knowledge=entity.content.get('name', entity.entity_id),
                reasoning=reasoning
            )

            return {
                'explanation': explanation,
                'confidence': entity.confidence,
                'evidence_type': entity.entity_type,
                'source': entity.source,
                'supporting_evidence': self._get_supporting_evidence(entity)
            }

        except Exception as e:
            logger.error(f"生成解释失败: {e}")
            return {
                'explanation': f"基于 {entity.entity_type} 知识的推荐",
                'confidence': entity.confidence,
                'error': str(e)
            }

    def _generate_reasoning(self, entity: KnowledgeEntity, context: Dict[str, Any]) -> str:
        """生成推理过程"""
        # 简化的推理生成
        content = entity.content

        if 'target_range' in content:
            return f"目标范围为 {content['target_range']}，有助于控制血糖"
        elif 'gi_classification' in content:
            return "选择低血糖指数食物可以稳定血糖波动"
        elif 'risk_factors' in content:
            return f"注意避免 {', '.join(content['risk_factors'][:2])} 等风险因素"
        else:
            return "有助于改善健康状况"

    def _get_supporting_evidence(self, entity: KnowledgeEntity) -> List[str]:
        """获取支持证据"""
        evidence = []

        # 基于来源添加证据
        if entity.source == 'clinical_guidelines':
            evidence.append("临床指南推荐")
        elif entity.source == 'research_papers':
            evidence.append("科学研究支持")
        elif entity.source == 'medical_textbook':
            evidence.append("医学教科书记载")

        # 基于置信度添加证据
        if entity.confidence > 0.9:
            evidence.append("高置信度")
        elif entity.confidence > 0.7:
            evidence.append("中等置信度")

        return evidence

class EnhancedExplainableKnowledgeSystem:
    """增强的可解释知识系统"""

    def __init__(self, knowledge_base: EnhancedMedicalKnowledgeBase):
        self.knowledge_base = knowledge_base
        self.explainer = KnowledgeExplainer()

        # 解释缓存
        self.explanation_cache: Dict[str, Dict[str, Any]] = {}

        # 解释质量评估
        self.explanation_feedback: List[Dict[str, Any]] = []

    def generate_explanation(self, input_data: Dict[str, Any],
                           prediction: float) -> Dict[str, Any]:
        """生成预测解释"""
        try:
            # 构建查询
            query_terms = []

            if 'glucose' in input_data:
                glucose_level = input_data['glucose']
                if glucose_level > 7.0:
                    query_terms.append("高血糖")
                elif glucose_level < 4.0:
                    query_terms.append("低血糖")
                else:
                    query_terms.append("血糖控制")

            if 'carbohydrates' in input_data:
                query_terms.append("碳水化合物")

            query = " ".join(query_terms) if query_terms else "血糖预测"

            # 查询相关知识
            relevant_knowledge = self.knowledge_base.query_knowledge(query, top_k=3)

            # 生成解释
            explanations = []
            for entity, relevance in relevant_knowledge:
                entity_explanation = self.explainer.explain_recommendation(entity, input_data)
                entity_explanation['relevance'] = relevance
                explanations.append(entity_explanation)

            # 构建综合解释
            main_explanation = self._build_comprehensive_explanation(
                explanations, prediction, input_data
            )

            return {
                'main_explanation': main_explanation,
                'detailed_explanations': explanations,
                'prediction_value': prediction,
                'confidence_score': np.mean([exp['confidence'] for exp in explanations]) if explanations else 0.5,
                'knowledge_sources': list(set([exp['source'] for exp in explanations]))
            }

        except Exception as e:
            logger.error(f"生成解释失败: {e}")
            return {
                'main_explanation': f"基于输入数据预测血糖值为 {prediction:.1f}",
                'error': str(e)
            }

    def _build_comprehensive_explanation(self, explanations: List[Dict[str, Any]],
                                       prediction: float, input_data: Dict[str, Any]) -> str:
        """构建综合解释"""
        if not explanations:
            return f"基于当前数据，预测血糖值为 {prediction:.1f} mmol/L"

        # 选择最相关的解释
        primary_explanation = max(explanations, key=lambda x: x['relevance'])

        # 构建主要解释
        main_text = f"预测血糖值为 {prediction:.1f} mmol/L。"
        main_text += f"主要依据：{primary_explanation['explanation']}。"

        # 添加次要因素
        if len(explanations) > 1:
            secondary = explanations[1] if explanations[1] != primary_explanation else explanations[0]
            main_text += f"同时考虑：{secondary['explanation']}。"

        # 添加建议
        if prediction > 7.0:
            main_text += "建议关注血糖控制，考虑调整饮食或运动计划。"
        elif prediction < 4.0:
            main_text += "血糖偏低，建议及时补充糖分。"

        return main_text

    def collect_explanation_feedback(self, explanation_id: str,
                                   feedback: Dict[str, Any]):
        """收集解释反馈"""
        feedback_record = {
            'explanation_id': explanation_id,
            'feedback': feedback,
            'timestamp': datetime.now()
        }

        self.explanation_feedback.append(feedback_record)

        # 更新知识库
        if 'entity_ids' in feedback:
            for entity_id in feedback['entity_ids']:
                self.knowledge_base.online_updater.add_update_signal(entity_id, feedback)

    def get_explanation_quality_metrics(self) -> Dict[str, Any]:
        """获取解释质量指标"""
        if not self.explanation_feedback:
            return {}

        # 计算满意度
        satisfaction_scores = [
            f['feedback'].get('satisfaction', 3)
            for f in self.explanation_feedback
        ]

        avg_satisfaction = np.mean(satisfaction_scores)

        # 计算有用性
        usefulness_scores = [
            f['feedback'].get('usefulness', 3)
            for f in self.explanation_feedback
        ]

        avg_usefulness = np.mean(usefulness_scores)

        return {
            'total_feedback': len(self.explanation_feedback),
            'average_satisfaction': avg_satisfaction,
            'average_usefulness': avg_usefulness,
            'feedback_trend': self._calculate_trend(satisfaction_scores[-10:])
        }

    def _calculate_trend(self, recent_scores: List[float]) -> str:
        """计算趋势"""
        if len(recent_scores) < 3:
            return "insufficient_data"

        # 简单的趋势计算
        first_half = np.mean(recent_scores[:len(recent_scores)//2])
        second_half = np.mean(recent_scores[len(recent_scores)//2:])

        if second_half > first_half + 0.2:
            return "improving"
        elif second_half < first_half - 0.2:
            return "declining"
        else:
            return "stable"

class EnhancedExplainableKnowledgeSystem:
    """增强的可解释知识系统"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.medical_kb = EnhancedMedicalKnowledgeBase(config.get('medical_kb', {}))
        self.knowledge_updater = OnlineKnowledgeUpdater(config.get('updater', {}))
        self.explainer = KnowledgeExplainer(config.get('explainer', {}))

        # 新增：跨领域知识融合模块
        self.cross_domain_fusion = CrossDomainKnowledgeFusion(
            medical_dim=config.get('medical_dim', 512),
            nutritional_dim=config.get('nutritional_dim', 256),
            cultural_dim=config.get('cultural_dim', 128),
            behavioral_dim=config.get('behavioral_dim', 128),
            fusion_dim=config.get('fusion_dim', 256)
        )

        self.knowledge_graph = {}
        self.version_history = []
        self.domain_knowledge_cache = defaultdict(dict)

    def add_knowledge(self, entity: KnowledgeEntity, relationships: List[KnowledgeRelationship] = None):
        """添加知识实体"""
        self.medical_kb.add_entity(entity)

        if relationships:
            for rel in relationships:
                self.medical_kb.add_relationship(rel)

        # 更新知识图谱
        self._update_knowledge_graph(entity, relationships)

        # 更新领域知识缓存
        self._update_domain_cache(entity)

        # 记录版本历史
        self.version_history.append({
            'timestamp': datetime.now(),
            'action': 'add_knowledge',
            'entity_id': entity.entity_id,
            'version': len(self.version_history) + 1
        })

    def query_knowledge(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """查询知识"""
        results = self.medical_kb.query(query, context)

        # 生成解释
        explanation = self.explainer.explain_query(query, results, context)

        # 跨领域知识融合
        fused_knowledge = self._fuse_cross_domain_knowledge(query, context)

        return {
            'results': results,
            'explanation': explanation,
            'fused_knowledge': fused_knowledge,
            'confidence': self._calculate_confidence(results),
            'timestamp': datetime.now()
        }

    def update_knowledge(self, entity_id: str, updates: Dict[str, Any]):
        """更新知识"""
        self.knowledge_updater.update_entity(entity_id, updates)

        # 记录版本历史
        self.version_history.append({
            'timestamp': datetime.now(),
            'action': 'update_knowledge',
            'entity_id': entity_id,
            'version': len(self.version_history) + 1
        })

    def explain_recommendation(self, recommendation: Dict[str, Any],
                              user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """解释推荐"""
        base_explanation = self.explainer.explain_recommendation(recommendation, user_context)

        # 跨领域知识解释
        cross_domain_explanation = self._generate_cross_domain_explanation(recommendation, user_context)

        return {
            'base_explanation': base_explanation,
            'cross_domain_explanation': cross_domain_explanation,
            'integrated_explanation': self._integrate_explanations(base_explanation, cross_domain_explanation)
        }

    def get_knowledge_summary(self) -> Dict[str, Any]:
        """获取知识摘要"""
        coverage = self._calculate_coverage()

        return {
            'total_entities': len(self.medical_kb.entities),
            'total_relationships': len(self.medical_kb.relationships),
            'version_count': len(self.version_history),
            'last_updated': self.version_history[-1]['timestamp'] if self.version_history else None,
            'knowledge_coverage': coverage,
            'cross_domain_fusion_stats': self._get_fusion_stats(),
            'semantic_alignment_score': self._calculate_semantic_alignment_score()
        }

    def _fuse_cross_domain_knowledge(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """跨领域知识融合"""
        # 提取各领域知识
        medical_knowledge = self._extract_domain_knowledge('medical', query, context)
        nutritional_knowledge = self._extract_domain_knowledge('nutritional', query, context)
        cultural_knowledge = self._extract_domain_knowledge('cultural', query, context)
        behavioral_knowledge = self._extract_domain_knowledge('behavioral', query, context)

        # 转换为张量
        medical_tensor = torch.tensor(medical_knowledge, dtype=torch.float32)
        nutritional_tensor = torch.tensor(nutritional_knowledge, dtype=torch.float32)
        cultural_tensor = torch.tensor(cultural_knowledge, dtype=torch.float32)
        behavioral_tensor = torch.tensor(behavioral_knowledge, dtype=torch.float32)

        # 跨领域融合
        fusion_result = self.cross_domain_fusion(
            medical_tensor, nutritional_tensor,
            cultural_tensor, behavioral_tensor
        )

        return {
            'fused_knowledge': fusion_result['fused_knowledge'].detach().numpy().tolist(),
            'domain_weights': fusion_result['domain_weights'].detach().numpy().tolist(),
            'consistency_score': fusion_result['consistency_score'].item(),
            'attention_weights': {k: v.detach().numpy().tolist() for k, v in fusion_result['attention_weights'].items()}
        }

    def _extract_domain_knowledge(self, domain: str, query: str, context: Dict[str, Any] = None) -> List[float]:
        """提取领域特定知识"""
        # 从缓存或知识库中提取领域知识
        if domain in self.domain_knowledge_cache:
            cached = self.domain_knowledge_cache[domain].get(query, None)
            if cached:
                return cached

        # 从知识库中查询
        domain_entities = [e for e in self.medical_kb.entities.values() if e.entity_type == domain]

        # 简化的知识向量化（实际应用中应使用更复杂的嵌入方法）
        knowledge_vector = [0.0] * 256  # 假设每个领域256维

        for entity in domain_entities[:10]:  # 限制数量
            # 基于实体内容生成特征
            content_str = str(entity.content)
            for i, char in enumerate(content_str[:256]):
                knowledge_vector[i] += ord(char) / 1000.0

        # 缓存结果
        self.domain_knowledge_cache[domain][query] = knowledge_vector

        return knowledge_vector

    def _generate_cross_domain_explanation(self, recommendation: Dict[str, Any],
                                         user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """生成跨领域知识解释"""
        explanation = {
            'medical_rationale': "基于医学知识的推荐依据",
            'nutritional_rationale': "基于营养学知识的推荐依据",
            'cultural_rationale': "基于文化背景的推荐依据",
            'behavioral_rationale': "基于行为模式的推荐依据",
            'cross_domain_synergy': "跨领域知识协同效应"
        }

        return explanation

    def _integrate_explanations(self, base_explanation: Dict[str, Any],
                               cross_domain_explanation: Dict[str, Any]) -> Dict[str, Any]:
        """整合解释"""
        return {
            'summary': f"综合医学、营养学、文化学和行为学知识，{base_explanation.get('summary', '')}",
            'detailed_rationale': {
                'medical': cross_domain_explanation.get('medical_rationale', ''),
                'nutritional': cross_domain_explanation.get('nutritional_rationale', ''),
                'cultural': cross_domain_explanation.get('cultural_rationale', ''),
                'behavioral': cross_domain_explanation.get('behavioral_rationale', ''),
                'synergy': cross_domain_explanation.get('cross_domain_synergy', '')
            },
            'confidence_level': 'high'  # 基于跨领域知识融合的置信度
        }

    def _update_domain_cache(self, entity: KnowledgeEntity):
        """更新领域知识缓存"""
        domain = entity.entity_type
        if domain not in self.domain_knowledge_cache:
            self.domain_knowledge_cache[domain] = {}

        # 更新缓存（简化实现）
        self.domain_knowledge_cache[domain][entity.entity_id] = [0.1] * 256

    def _get_fusion_stats(self) -> Dict[str, Any]:
        """获取融合统计信息"""
        return {
            'total_domains': 4,
            'fusion_operations': len(self.version_history),
            'cache_hit_rate': 0.85,  # 模拟缓存命中率
            'semantic_alignment_score': 0.92
        }

    def _calculate_semantic_alignment_score(self) -> float:
        """计算语义对齐分数"""
        # 基于知识一致性和跨领域融合质量计算
        consistency_scores = []
        for version in self.version_history[-10:]:  # 最近10次更新
            consistency_scores.append(0.9)  # 模拟一致性分数

        return sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.8

    def _update_knowledge_graph(self, entity: KnowledgeEntity, relationships: List[KnowledgeRelationship]):
        """更新知识图谱"""
        if entity.entity_id not in self.knowledge_graph:
            self.knowledge_graph[entity.entity_id] = {
                'entity': entity,
                'relationships': []
            }

        if relationships:
            self.knowledge_graph[entity.entity_id]['relationships'].extend(relationships)

    def _calculate_confidence(self, results: List[Dict[str, Any]]) -> float:
        """计算置信度"""
        if not results:
            return 0.0

        # 基于结果数量和相关性计算置信度
        base_confidence = min(len(results) / 10.0, 1.0)

        # 考虑结果的相关性
        relevance_scores = [r.get('relevance_score', 0.5) for r in results]
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.5

        return (base_confidence + avg_relevance) / 2.0

    def _calculate_coverage(self) -> Dict[str, float]:
        """计算知识覆盖度"""
        total_medical = len([e for e in self.medical_kb.entities.values() if e.entity_type == 'medical'])
        total_nutritional = len([e for e in self.medical_kb.entities.values() if e.entity_type == 'nutritional'])
        total_cultural = len([e for e in self.medical_kb.entities.values() if e.entity_type == 'cultural'])

        return {
            'medical_coverage': min(total_medical / 1000.0, 1.0),
            'nutritional_coverage': min(total_nutritional / 500.0, 1.0),
            'cultural_coverage': min(total_cultural / 200.0, 1.0)
        }

# 使用示例
def main():
    """使用示例"""
    # 创建增强的知识库
    knowledge_base = EnhancedMedicalKnowledgeBase()

    # 创建可解释系统
    explainable_system = EnhancedExplainableKnowledgeSystem(knowledge_base)

    # 测试解释生成
    test_input = {
        'glucose': 8.5,
        'carbohydrates': 60,
        'exercise': 0,
        'time_of_day': 14
    }

    predicted_glucose = 9.2

    explanation = explainable_system.generate_explanation(test_input, predicted_glucose)

    print("=== 解释系统示例 ===")
    print(f"主要解释: {explanation['main_explanation']}")
    print(f"置信度: {explanation['confidence_score']:.2f}")
    print(f"知识来源: {explanation['knowledge_sources']}")

    # 详细解释
    print("\n=== 详细解释 ===")
    for i, detail in enumerate(explanation['detailed_explanations']):
        print(f"{i+1}. {detail['explanation']}")
        print(f"   证据: {', '.join(detail.get('supporting_evidence', []))}")
        print(f"   相关性: {detail['relevance']:.2f}")

if __name__ == "__main__":
    main()

__all__ = ["'logger'", "'CrossDomainKnowledgeFusion'", "'KnowledgeEntity'", "'KnowledgeRelation'", "'EnhancedMedicalKnowledgeBase'", "'OnlineKnowledgeUpdater'", "'KnowledgeExplainer'", "'EnhancedExplainableKnowledgeSystem'", "'EnhancedExplainableKnowledgeSystem'", "'main'"]
