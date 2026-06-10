"""
个性化知识图谱 - 构建患者个性化的健康知识图谱
"""

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from collections import defaultdict
import json
import time
from enum import Enum

logger = logging.getLogger(__name__)

class EntityType(Enum):
    """实体类型"""
    PATIENT = "patient"
    GLUCOSE = "glucose"
    MEAL = "meal"
    EXERCISE = "exercise"
    MEDICATION = "medication"
    SYMPTOM = "symptom"
    FOOD = "food"
    ACTIVITY = "activity"

class RelationType(Enum):
    """关系类型"""
    CAUSES = "causes"
    AFFECTS = "affects"
    PRECEDES = "precedes"
    FOLLOWS = "follows"
    CORRELATES = "correlates"
    CONTAINS = "contains"
    INTERACTS = "interacts"

@dataclass
class Entity:
    """知识图谱实体"""
    entity_id: str
    entity_type: EntityType
    attributes: Dict[str, Any]
    embedding: Optional[torch.Tensor] = None

    def __post_init__(self):
        if self.embedding is None:
            self.embedding = torch.randn(128)  # 默认嵌入维度

@dataclass
class Relation:
    """知识图谱关系"""
    relation_id: str
    relation_type: RelationType
    source_entity: str
    target_entity: str
    weight: float = 1.0
    attributes: Dict[str, Any] = None

    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}

class GraphEmbedding(nn.Module):
    """图嵌入模型"""

    def __init__(self, num_entities: int, num_relations: int,
                 embedding_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_entities = num_entities
        self.num_relations = num_relations

        # 实体嵌入
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)

        # 关系嵌入
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        # 图神经网络层
        self.gnn_layers = nn.ModuleList([
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        ])

    def forward(self, entity_ids: torch.Tensor, relation_ids: torch.Tensor = None):
        """前向传播"""
        entity_embeds = self.entity_embeddings(entity_ids)

        if relation_ids is not None:
            relation_embeds = self.relation_embeddings(relation_ids)
            return entity_embeds, relation_embeds

        return entity_embeds

class PersonalKnowledgeGraph:
    """个性化知识图谱"""

    def __init__(self, patient_id: str):
        self.patient_id = patient_id
        self.graph = nx.DiGraph()
        self.entities = {}
        self.relations = {}
        self.entity_counter = 0
        self.relation_counter = 0

        # 嵌入模型
        self.embedding_model = None
        self.entity_type_map = {}
        self.relation_type_map = {}

        # 初始化患者节点
        self._initialize_patient_node()

        logger.info(f"个性化知识图谱初始化完成，患者ID: {patient_id}")

    def _initialize_patient_node(self):
        """初始化患者节点"""
        patient_entity = Entity(
            entity_id=f"patient_{self.patient_id}",
            entity_type=EntityType.PATIENT,
            attributes={
                'patient_id': self.patient_id,
                'created_at': time.time()
            }
        )

        self.add_entity(patient_entity)

    def add_entity(self, entity: Entity) -> str:
        """添加实体"""
        if entity.entity_id not in self.entities:
            self.entities[entity.entity_id] = entity
            self.graph.add_node(entity.entity_id, **entity.attributes)

            # 更新实体类型映射
            if entity.entity_type not in self.entity_type_map:
                self.entity_type_map[entity.entity_type] = []
            self.entity_type_map[entity.entity_type].append(entity.entity_id)

            logger.debug(f"添加实体: {entity.entity_id}")

        return entity.entity_id

    def add_relation(self, relation: Relation) -> str:
        """添加关系"""
        if relation.relation_id not in self.relations:
            # 确保源实体和目标实体存在
            if relation.source_entity not in self.entities:
                logger.warning(f"源实体不存在: {relation.source_entity}")
                return None

            if relation.target_entity not in self.entities:
                logger.warning(f"目标实体不存在: {relation.target_entity}")
                return None

            self.relations[relation.relation_id] = relation
            self.graph.add_edge(
                relation.source_entity,
                relation.target_entity,
                relation_type=relation.relation_type.value,
                weight=relation.weight,
                **relation.attributes
            )

            # 更新关系类型映射
            if relation.relation_type not in self.relation_type_map:
                self.relation_type_map[relation.relation_type] = []
            self.relation_type_map[relation.relation_type].append(relation.relation_id)

            logger.debug(f"添加关系: {relation.relation_id}")

        return relation.relation_id

    def build_knowledge_graph(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """构建患者个性化知识图谱"""
        logger.info("开始构建个性化知识图谱...")

        # 添加血糖数据实体和关系
        self._add_glucose_entities(patient_data.get('glucose_data', []))

        # 添加膳食数据实体和关系
        self._add_meal_entities(patient_data.get('meal_data', []))

        # 添加运动数据实体和关系
        self._add_exercise_entities(patient_data.get('exercise_data', []))

        # 添加药物数据实体和关系
        self._add_medication_entities(patient_data.get('medication_data', []))

        # 添加症状数据实体和关系
        self._add_symptom_entities(patient_data.get('symptom_data', []))

        # 构建实体间的关系
        self._build_entity_relationships()

        # 训练图嵌入
        self._train_graph_embeddings()

        # 生成图谱统计信息
        statistics = self._generate_graph_statistics()

        logger.info(f"知识图谱构建完成，实体数: {len(self.entities)}, 关系数: {len(self.relations)}")

        return statistics

    def _add_glucose_entities(self, glucose_data: List[Dict[str, Any]]):
        """添加血糖实体"""
        for i, glucose_record in enumerate(glucose_data):
            entity_id = f"glucose_{i}_{self.patient_id}"

            entity = Entity(
                entity_id=entity_id,
                entity_type=EntityType.GLUCOSE,
                attributes={
                    'value': glucose_record.get('value', 0),
                    'timestamp': glucose_record.get('timestamp', time.time()),
                    'type': glucose_record.get('type', 'unknown'),
                    'context': glucose_record.get('context', {})
                }
            )

            self.add_entity(entity)

            # 连接到患者
            relation = Relation(
                relation_id=f"patient_glucose_{i}",
                relation_type=RelationType.AFFECTS,
                source_entity=f"patient_{self.patient_id}",
                target_entity=entity_id,
                weight=1.0
            )

            self.add_relation(relation)

    def _add_meal_entities(self, meal_data: List[Dict[str, Any]]):
        """添加膳食实体"""
        for i, meal_record in enumerate(meal_data):
            # 添加餐次实体
            meal_entity_id = f"meal_{i}_{self.patient_id}"

            meal_entity = Entity(
                entity_id=meal_entity_id,
                entity_type=EntityType.MEAL,
                attributes={
                    'meal_type': meal_record.get('meal_type', 'unknown'),
                    'timestamp': meal_record.get('timestamp', time.time()),
                    'total_calories': meal_record.get('total_calories', 0),
                    'total_carbs': meal_record.get('total_carbs', 0),
                    'gi_value': meal_record.get('gi_value', 0)
                }
            )

            self.add_entity(meal_entity)

            # 连接到患者
            relation = Relation(
                relation_id=f"patient_meal_{i}",
                relation_type=RelationType.AFFECTS,
                source_entity=f"patient_{self.patient_id}",
                target_entity=meal_entity_id,
                weight=1.0
            )

            self.add_relation(relation)

            # 添加食物实体
            for j, food_item in enumerate(meal_record.get('food_items', [])):
                food_entity_id = f"food_{i}_{j}_{self.patient_id}"

                food_entity = Entity(
                    entity_id=food_entity_id,
                    entity_type=EntityType.FOOD,
                    attributes={
                        'name': food_item.get('name', 'unknown'),
                        'quantity': food_item.get('quantity', 0),
                        'calories': food_item.get('calories', 0),
                        'carbs': food_item.get('carbs', 0),
                        'gi': food_item.get('gi', 0)
                    }
                )

                self.add_entity(food_entity)

                # 食物属于餐次
                relation = Relation(
                    relation_id=f"meal_food_{i}_{j}",
                    relation_type=RelationType.CONTAINS,
                    source_entity=meal_entity_id,
                    target_entity=food_entity_id,
                    weight=1.0
                )

                self.add_relation(relation)

    def _add_exercise_entities(self, exercise_data: List[Dict[str, Any]]):
        """添加运动实体"""
        for i, exercise_record in enumerate(exercise_data):
            entity_id = f"exercise_{i}_{self.patient_id}"

            entity = Entity(
                entity_id=entity_id,
                entity_type=EntityType.EXERCISE,
                attributes={
                    'exercise_type': exercise_record.get('type', 'unknown'),
                    'duration': exercise_record.get('duration', 0),
                    'intensity': exercise_record.get('intensity', 'unknown'),
                    'timestamp': exercise_record.get('timestamp', time.time()),
                    'calories_burned': exercise_record.get('calories_burned', 0)
                }
            )

            self.add_entity(entity)

            # 连接到患者
            relation = Relation(
                relation_id=f"patient_exercise_{i}",
                relation_type=RelationType.AFFECTS,
                source_entity=f"patient_{self.patient_id}",
                target_entity=entity_id,
                weight=1.0
            )

            self.add_relation(relation)

    def _add_medication_entities(self, medication_data: List[Dict[str, Any]]):
        """添加药物实体"""
        for i, med_record in enumerate(medication_data):
            entity_id = f"medication_{i}_{self.patient_id}"

            entity = Entity(
                entity_id=entity_id,
                entity_type=EntityType.MEDICATION,
                attributes={
                    'name': med_record.get('name', 'unknown'),
                    'dosage': med_record.get('dosage', 0),
                    'frequency': med_record.get('frequency', 'unknown'),
                    'timestamp': med_record.get('timestamp', time.time())
                }
            )

            self.add_entity(entity)

            # 连接到患者
            relation = Relation(
                relation_id=f"patient_medication_{i}",
                relation_type=RelationType.AFFECTS,
                source_entity=f"patient_{self.patient_id}",
                target_entity=entity_id,
                weight=1.0
            )

            self.add_relation(relation)

    def _add_symptom_entities(self, symptom_data: List[Dict[str, Any]]):
        """添加症状实体"""
        for i, symptom_record in enumerate(symptom_data):
            entity_id = f"symptom_{i}_{self.patient_id}"

            entity = Entity(
                entity_id=entity_id,
                entity_type=EntityType.SYMPTOM,
                attributes={
                    'symptom_type': symptom_record.get('type', 'unknown'),
                    'severity': symptom_record.get('severity', 0),
                    'timestamp': symptom_record.get('timestamp', time.time()),
                    'description': symptom_record.get('description', '')
                }
            )

            self.add_entity(entity)

            # 连接到患者
            relation = Relation(
                relation_id=f"patient_symptom_{i}",
                relation_type=RelationType.AFFECTS,
                source_entity=f"patient_{self.patient_id}",
                target_entity=entity_id,
                weight=1.0
            )

            self.add_relation(relation)

    def _build_entity_relationships(self):
        """构建实体间的关系"""
        # 血糖与膳食的关系
        self._build_glucose_meal_relationships()

        # 血糖与运动的关系
        self._build_glucose_exercise_relationships()

        # 血糖与药物的关系
        self._build_glucose_medication_relationships()

        # 血糖与症状的关系
        self._build_glucose_symptom_relationships()

    def _build_glucose_meal_relationships(self):
        """构建血糖与膳食的关系"""
        glucose_entities = [eid for eid in self.entities.keys() if eid.startswith('glucose_')]
        meal_entities = [eid for eid in self.entities.keys() if eid.startswith('meal_')]

        for glucose_id in glucose_entities:
            glucose_entity = self.entities[glucose_id]
            glucose_time = glucose_entity.attributes.get('timestamp', 0)

            for meal_id in meal_entities:
                meal_entity = self.entities[meal_id]
                meal_time = meal_entity.attributes.get('timestamp', 0)

                # 如果餐次在血糖测量前2小时内，建立因果关系
                if 0 < glucose_time - meal_time < 7200:  # 2小时
                    relation = Relation(
                        relation_id=f"meal_glucose_{meal_id}_{glucose_id}",
                        relation_type=RelationType.CAUSES,
                        source_entity=meal_id,
                        target_entity=glucose_id,
                        weight=0.8
                    )

                    self.add_relation(relation)

    def _build_glucose_exercise_relationships(self):
        """构建血糖与运动的关系"""
        glucose_entities = [eid for eid in self.entities.keys() if eid.startswith('glucose_')]
        exercise_entities = [eid for eid in self.entities.keys() if eid.startswith('exercise_')]

        for glucose_id in glucose_entities:
            glucose_entity = self.entities[glucose_id]
            glucose_time = glucose_entity.attributes.get('timestamp', 0)

            for exercise_id in exercise_entities:
                exercise_entity = self.entities[exercise_id]
                exercise_time = exercise_entity.attributes.get('timestamp', 0)

                # 如果运动在血糖测量前1小时内，建立影响关系
                if 0 < glucose_time - exercise_time < 3600:  # 1小时
                    relation = Relation(
                        relation_id=f"exercise_glucose_{exercise_id}_{glucose_id}",
                        relation_type=RelationType.AFFECTS,
                        source_entity=exercise_id,
                        target_entity=glucose_id,
                        weight=0.6
                    )

                    self.add_relation(relation)

    def _build_glucose_medication_relationships(self):
        """构建血糖与药物的关系"""
        glucose_entities = [eid for eid in self.entities.keys() if eid.startswith('glucose_')]
        medication_entities = [eid for eid in self.entities.keys() if eid.startswith('medication_')]

        for glucose_id in glucose_entities:
            for med_id in medication_entities:
                # 药物对血糖有持续影响
                relation = Relation(
                    relation_id=f"medication_glucose_{med_id}_{glucose_id}",
                    relation_type=RelationType.AFFECTS,
                    source_entity=med_id,
                    target_entity=glucose_id,
                    weight=0.7
                )

                self.add_relation(relation)

    def _build_glucose_symptom_relationships(self):
        """构建血糖与症状的关系"""
        glucose_entities = [eid for eid in self.entities.keys() if eid.startswith('glucose_')]
        symptom_entities = [eid for eid in self.entities.keys() if eid.startswith('symptom_')]

        for glucose_id in glucose_entities:
            glucose_entity = self.entities[glucose_id]
            glucose_value = glucose_entity.attributes.get('value', 0)

            for symptom_id in symptom_entities:
                symptom_entity = self.entities[symptom_id]
                symptom_time = symptom_entity.attributes.get('timestamp', 0)
                glucose_time = glucose_entity.attributes.get('timestamp', 0)

                # 如果症状和血糖测量时间接近，建立相关关系
                if abs(glucose_time - symptom_time) < 1800:  # 30分钟
                    relation = Relation(
                        relation_id=f"glucose_symptom_{glucose_id}_{symptom_id}",
                        relation_type=RelationType.CORRELATES,
                        source_entity=glucose_id,
                        target_entity=symptom_id,
                        weight=0.9
                    )

                    self.add_relation(relation)

    def _train_graph_embeddings(self):
        """训练图嵌入"""
        if len(self.entities) == 0:
            return

        # 创建实体ID映射
        entity_id_map = {eid: idx for idx, eid in enumerate(self.entities.keys())}

        # 创建关系ID映射
        relation_id_map = {rid: idx for idx, rid in enumerate(self.relations.keys())}

        # 初始化嵌入模型
        self.embedding_model = GraphEmbedding(
            num_entities=len(self.entities),
            num_relations=len(self.relations)
        )

        # 简化的训练过程
        optimizer = torch.optim.Adam(self.embedding_model.parameters(), lr=0.01)

        for epoch in range(50):  # 简化训练
            optimizer.zero_grad()

            # 随机采样实体和关系
            entity_ids = torch.randint(0, len(self.entities), (32,))
            relation_ids = torch.randint(0, len(self.relations), (32,))

            # 前向传播
            entity_embeds, relation_embeds = self.embedding_model(entity_ids, relation_ids)

            # 简化的损失函数
            loss = torch.mean(torch.norm(entity_embeds, dim=1))

            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                logger.debug(f"嵌入训练轮次 {epoch}, 损失: {loss.item():.4f}")

        # 更新实体嵌入
        with torch.no_grad():
            for eid, idx in entity_id_map.items():
                entity_embed = self.embedding_model.entity_embeddings(torch.tensor(idx))
                self.entities[eid].embedding = entity_embed

        logger.info("图嵌入训练完成")

    def _generate_graph_statistics(self) -> Dict[str, Any]:
        """生成图谱统计信息"""
        statistics = {
            'num_entities': len(self.entities),
            'num_relations': len(self.relations),
            'entity_types': {et.value: len(entities) for et, entities in self.entity_type_map.items()},
            'relation_types': {rt.value: len(relations) for rt, relations in self.relation_type_map.items()},
            'graph_density': nx.density(self.graph),
            'average_clustering': nx.average_clustering(self.graph.to_undirected()),
            'connected_components': nx.number_connected_components(self.graph.to_undirected())
        }

        return statistics

    def query_knowledge_graph(self, query: str) -> Dict[str, Any]:
        """查询知识图谱"""
        # 简化的查询实现
        if query.startswith("glucose"):
            return self._query_glucose_patterns()
        elif query.startswith("meal"):
            return self._query_meal_patterns()
        elif query.startswith("exercise"):
            return self._query_exercise_patterns()
        else:
            return {'error': 'Unsupported query type'}

    def _query_glucose_patterns(self) -> Dict[str, Any]:
        """查询血糖模式"""
        glucose_entities = [eid for eid in self.entities.keys() if eid.startswith('glucose_')]

        patterns = {
            'total_readings': len(glucose_entities),
            'average_value': 0,
            'high_glucose_count': 0,
            'low_glucose_count': 0
        }

        if glucose_entities:
            values = [self.entities[eid].attributes.get('value', 0) for eid in glucose_entities]
            patterns['average_value'] = np.mean(values)
            patterns['high_glucose_count'] = sum(1 for v in values if v > 10)
            patterns['low_glucose_count'] = sum(1 for v in values if v < 4)

        return patterns

    def _query_meal_patterns(self) -> Dict[str, Any]:
        """查询膳食模式"""
        meal_entities = [eid for eid in self.entities.keys() if eid.startswith('meal_')]

        patterns = {
            'total_meals': len(meal_entities),
            'meal_types': {},
            'average_calories': 0
        }

        if meal_entities:
            meal_types = [self.entities[eid].attributes.get('meal_type', 'unknown') for eid in meal_entities]
            calories = [self.entities[eid].attributes.get('total_calories', 0) for eid in meal_entities]

            patterns['meal_types'] = {mt: meal_types.count(mt) for mt in set(meal_types)}
            patterns['average_calories'] = np.mean(calories)

        return patterns

    def _query_exercise_patterns(self) -> Dict[str, Any]:
        """查询运动模式"""
        exercise_entities = [eid for eid in self.entities.keys() if eid.startswith('exercise_')]

        patterns = {
            'total_exercises': len(exercise_entities),
            'exercise_types': {},
            'average_duration': 0
        }

        if exercise_entities:
            exercise_types = [self.entities[eid].attributes.get('exercise_type', 'unknown') for eid in exercise_entities]
            durations = [self.entities[eid].attributes.get('duration', 0) for eid in exercise_entities]

            patterns['exercise_types'] = {et: exercise_types.count(et) for et in set(exercise_types)}
            patterns['average_duration'] = np.mean(durations)

        return patterns

    def get_graph_visualization_data(self) -> Dict[str, Any]:
        """获取图谱可视化数据"""
        nodes = []
        edges = []

        # 节点数据
        for eid, entity in self.entities.items():
            nodes.append({
                'id': eid,
                'type': entity.entity_type.value,
                'attributes': entity.attributes
            })

        # 边数据
        for rid, relation in self.relations.items():
            edges.append({
                'id': rid,
                'source': relation.source_entity,
                'target': relation.target_entity,
                'type': relation.relation_type.value,
                'weight': relation.weight
            })

        return {
            'nodes': nodes,
            'edges': edges,
            'statistics': self._generate_graph_statistics()
        }

    def save_knowledge_graph(self, filepath: str):
        """保存知识图谱"""
        graph_data = {
            'patient_id': self.patient_id,
            'entities': {eid: {
                'entity_type': entity.entity_type.value,
                'attributes': entity.attributes,
                'embedding': entity.embedding.tolist() if entity.embedding is not None else None
            } for eid, entity in self.entities.items()},
            'relations': {rid: {
                'relation_type': relation.relation_type.value,
                'source_entity': relation.source_entity,
                'target_entity': relation.target_entity,
                'weight': relation.weight,
                'attributes': relation.attributes
            } for rid, relation in self.relations.items()},
            'statistics': self._generate_graph_statistics()
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)

        logger.info(f"知识图谱已保存到: {filepath}")

    def load_knowledge_graph(self, filepath: str):
        """加载知识图谱"""
        with open(filepath, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)

        # 重建实体
        for eid, entity_data in graph_data['entities'].items():
            entity = Entity(
                entity_id=eid,
                entity_type=EntityType(entity_data['entity_type']),
                attributes=entity_data['attributes'],
                embedding=torch.tensor(entity_data['embedding']) if entity_data['embedding'] else None
            )
            self.entities[eid] = entity

        # 重建关系
        for rid, relation_data in graph_data['relations'].items():
            relation = Relation(
                relation_id=rid,
                relation_type=RelationType(relation_data['relation_type']),
                source_entity=relation_data['source_entity'],
                target_entity=relation_data['target_entity'],
                weight=relation_data['weight'],
                attributes=relation_data['attributes']
            )
            self.relations[rid] = relation

        # 重建图结构
        self.graph = nx.DiGraph()
        for eid in self.entities.keys():
            self.graph.add_node(eid, **self.entities[eid].attributes)

        for rid, relation in self.relations.items():
            self.graph.add_edge(
                relation.source_entity,
                relation.target_entity,
                relation_type=relation.relation_type.value,
                weight=relation.weight,
                **relation.attributes
            )

        logger.info(f"知识图谱已从{filepath}加载")
