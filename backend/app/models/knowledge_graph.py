"""
多维度个性化健康营养知识图谱核心模块
基于Neo4j的知识图谱构建和管理系统
"""

import logging
import os
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import numpy as np
from neo4j import GraphDatabase
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# 优先加载.env中的Neo4j配置（若存在）
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    # 可选依赖，不阻断运行
    pass

@dataclass
class Entity:
    """知识图谱实体"""
    id: str
    name: str
    entity_type: str  # Food, Nutrient, Disease, User, Recipe, etc.
    properties: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

@dataclass
class Relation:
    """知识图谱关系"""
    id: str
    source_entity: str
    target_entity: str
    relation_type: str  # CONTAINS, BENEFICIAL_FOR, HARMFUL_FOR, etc.
    properties: Dict[str, Any]
    weight: float = 1.0
    confidence: float = 1.0
    created_at: Optional[datetime] = None

@dataclass
class KnowledgeGraphConfig:
    """知识图谱配置"""
    neo4j_uri: str = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_password: str = os.environ.get("NEO4J_PASSWORD", "password")
    embedding_dim: int = 768
    max_entities: int = 100000
    max_relations: int = 500000
    similarity_threshold: float = 0.7
    confidence_threshold: float = 0.5

class KnowledgeGraphEmbedding(nn.Module):
    """知识图谱嵌入模型"""

    def __init__(self, config: KnowledgeGraphConfig):
        super().__init__()
        self.config = config
        self.embedding_dim = config.embedding_dim

        # 实体嵌入层
        self.entity_embeddings = nn.Embedding(
            config.max_entities,
            config.embedding_dim
        )

        # 关系嵌入层
        self.relation_embeddings = nn.Embedding(
            config.max_relations,
            config.embedding_dim
        )

        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=config.embedding_dim,
            num_heads=8,
            batch_first=True
        )

        # 图神经网络层
        self.gnn_layers = nn.ModuleList([
            nn.Linear(config.embedding_dim, config.embedding_dim)
            for _ in range(3)
        ])

        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(config.embedding_dim)

    def forward(self, entity_ids: torch.Tensor, relation_ids: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 获取实体和关系嵌入
        entity_emb = self.entity_embeddings(entity_ids)
        relation_emb = self.relation_embeddings(relation_ids)

        # 注意力机制
        attended_emb, _ = self.attention(entity_emb, entity_emb, entity_emb)

        # 图神经网络处理
        for gnn_layer in self.gnn_layers:
            attended_emb = gnn_layer(attended_emb)
            attended_emb = torch.relu(attended_emb)
            attended_emb = self.dropout(attended_emb)

        # 层归一化
        output = self.layer_norm(attended_emb)

        return output

class MultiModalProcessor:
    """多模态数据处理器"""

    def __init__(self, config: KnowledgeGraphConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 文本处理器 - 添加离线模式和错误处理
        try:
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese', local_files_only=True)
            self.text_model = AutoModel.from_pretrained('bert-base-chinese', local_files_only=True).to(self.device)
        except Exception as e:
            logger.warning(f"无法加载BERT模型，使用简化文本处理: {e}")
            self.tokenizer = None
            self.text_model = None

        # 图像处理器（如果需要）
        # from torchvision.models import ResNet50_Weights
        # self.image_model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)

    def process_text(self, text: str) -> np.ndarray:
        """处理文本数据"""
        if self.tokenizer is None or self.text_model is None:
            # 使用简化的文本处理
            return self._simple_text_embedding(text)

        try:
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.text_model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

            return embedding.squeeze()
        except Exception as e:
            logger.error(f"文本处理失败: {e}")
            return self._simple_text_embedding(text)

    def _simple_text_embedding(self, text: str) -> np.ndarray:
        """简化的文本嵌入方法"""
        # 基于字符频率的简单嵌入
        char_freq = {}
        for char in text:
            char_freq[char] = char_freq.get(char, 0) + 1

        # 创建固定长度的嵌入向量
        embedding = np.zeros(self.config.embedding_dim)
        for i, (char, freq) in enumerate(char_freq.items()):
            if i < self.config.embedding_dim:
                embedding[i] = freq / len(text)  # 归一化频率

        return embedding

    def process_food_image(self, image_path: str) -> np.ndarray:
        """处理食物图像（预留接口）"""
        # 这里可以集成ResNet50等图像处理模型
        # 暂时返回零向量
        return np.zeros(self.config.embedding_dim)

    def extract_nutrition_facts(self, text: str) -> Dict[str, float]:
        """从文本中提取营养信息"""
        nutrition_facts = {}

        # 简化的营养信息提取
        keywords = {
            '卡路里': ['卡路里', '热量', 'kcal', 'cal'],
            '蛋白质': ['蛋白质', 'protein'],
            '碳水化合物': ['碳水化合物', '碳水', 'carb'],
            '脂肪': ['脂肪', 'fat'],
            '纤维': ['纤维', 'fiber'],
            '糖': ['糖', 'sugar'],
            '钠': ['钠', 'sodium'],
            '维生素C': ['维生素C', '维C', 'vitamin C']
        }

        for nutrient, keywords_list in keywords.items():
            for keyword in keywords_list:
                if keyword in text.lower():
                    # 简单的数值提取（实际应用中需要更复杂的NLP）
                    import re
                    pattern = rf'{keyword}[:\s]*(\d+(?:\.\d+)?)'
                    match = re.search(pattern, text.lower())
                    if match:
                        nutrition_facts[nutrient] = float(match.group(1))

        return nutrition_facts

class KnowledgeGraphBuilder:
    """知识图谱构建器"""

    def __init__(self, config: KnowledgeGraphConfig):
        self.config = config
        self.driver = None
        self.multimodal_processor = None
        self.embedding_model = None

        # 实体和关系映射
        self.entity_map: Dict[str, str] = {}
        self.relation_map: Dict[str, str] = {}

        # 尝试连接Neo4j，如果失败则使用内存模式（带重试）
        connected = self._connect_with_retry()
        if connected:
            # 初始化处理器
            self.multimodal_processor = MultiModalProcessor(config)
            self.embedding_model = KnowledgeGraphEmbedding(config)
            # 初始化图谱
            self._initialize_graph()
        else:
            self.driver = None
            self._memory_entities: Dict[str, Entity] = {}
            self._memory_relations: Dict[str, Relation] = {}

    def _initialize_graph(self):
        """初始化知识图谱"""
        if self.driver is None:
            logger.info("使用内存模式，跳过Neo4j初始化")
            return

        try:
            with self.driver.session() as session:
                # 创建索引
                session.run("CREATE INDEX entity_id_index IF NOT EXISTS FOR (e:Entity) ON (e.id)")
                session.run("CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type)")
                session.run("CREATE INDEX relation_type_index IF NOT EXISTS FOR ()-[r:RELATION]-() ON (r.type)")

                logger.info("知识图谱初始化完成")
        except Exception as e:
            logger.error(f"知识图谱初始化失败: {e}")

    def _connect_with_retry(self, retries: int = 3, delay_seconds: float = 2.0) -> bool:
        """带重试的连接建立与健康探测"""
        for attempt in range(1, retries + 1):
            try:
                self.driver = GraphDatabase.driver(
                    self.config.neo4j_uri,
                    auth=(self.config.neo4j_user, self.config.neo4j_password)
                )
                # 健康探针
                with self.driver.session() as session:
                    session.run("RETURN 1")
                logger.info("Neo4j连接成功")
                return True
            except Exception as e:
                logger.warning(
                    f"Neo4j连接失败 (第{attempt}/{retries}次): {e}. "
                    f"{('等待重试...' if attempt < retries else '将回退为内存模式')}"
                )
                try:
                    time.sleep(delay_seconds)
                except Exception:
                    pass
        logger.warning("Neo4j连接失败，使用内存模式")
        return False

    def health_probe(self) -> Dict[str, Any]:
        """健康探针：返回连接/认证/延迟等简要状态，用于监控与自愈决策"""
        status = {"connected": False, "latency_ms": None, "error": None}
        if self.driver is None:
            status["error"] = "driver_unavailable"
            return status
        try:
            start = time.time()
            with self.driver.session() as session:
                result = session.run("RETURN 1 AS ok").single()
            status["connected"] = bool(result and result.get("ok") == 1)
            status["latency_ms"] = int((time.time() - start) * 1000)
            return status
        except Exception as e:
            status["error"] = str(e)
            return status

    def _ensure_driver(self) -> bool:
        """确保driver可用；不可用时尝试重连，失败则降级"""
        if self.driver is not None:
            try:
                with self.driver.session() as session:
                    session.run("RETURN 1")
                return True
            except Exception:
                # 连接失效，尝试重连
                pass
        if self._connect_with_retry():
            return True
        # 重连失败，降级为内存模式
        self.driver = None
        if not hasattr(self, "_memory_entities"):
            self._memory_entities = {}
            self._memory_relations = {}
        logger.warning("Neo4j不可用，已自动降级为内存模式")
        return False

    def add_entity(self, entity: Entity) -> bool:
        """添加实体到知识图谱"""
        try:
            if self.driver is None:
                # 内存模式
                if entity.id in self._memory_entities:
                    logger.warning(f"实体 {entity.id} 已存在")
                    return False

                # 生成嵌入
                if entity.embedding is None and self.multimodal_processor:
                    entity.embedding = self.multimodal_processor.process_text(entity.name)

                self._memory_entities[entity.id] = entity
                self.entity_map[entity.id] = entity.name
                logger.info(f"实体添加成功（内存模式）: {entity.name}")
                return True

            with self.driver.session() as session:
                # 生成嵌入
                if entity.embedding is None and self.multimodal_processor:
                    entity.embedding = self.multimodal_processor.process_text(entity.name)

                # 创建实体节点
                query = """
                MERGE (e:Entity {id: $id})
                SET e.name = $name,
                    e.type = $type,
                    e.properties = $properties,
                    e.embedding = $embedding,
                    e.created_at = datetime(),
                    e.updated_at = datetime()
                """

                session.run(query,
                    id=entity.id,
                    name=entity.name,
                    type=entity.entity_type,
                    properties=json.dumps(entity.properties),
                    embedding=entity.embedding.tolist() if entity.embedding is not None else None
                )

                self.entity_map[entity.id] = entity.name
                logger.info(f"实体添加成功: {entity.name}")
                return True

        except Exception as e:
            logger.error(f"实体添加失败: {e}")
            return False

    def add_relation(self, relation: Relation) -> bool:
        """添加关系到知识图谱"""
        try:
            if self.driver is None:
                # 内存模式
                relation_key = f"{relation.source_entity}-{relation.relation_type}-{relation.target_entity}"
                if relation_key in self._memory_relations:
                    logger.warning(f"关系 {relation_key} 已存在")
                    return False

                self._memory_relations[relation_key] = relation
                logger.info(f"关系添加成功（内存模式）: {relation.source_entity} -> {relation.target_entity}")
                return True

            with self.driver.session() as session:
                query = """
                MATCH (source:Entity {id: $source_id})
                MATCH (target:Entity {id: $target_id})
                MERGE (source)-[r:RELATION {type: $relation_type}]->(target)
                SET r.properties = $properties,
                    r.weight = $weight,
                    r.confidence = $confidence,
                    r.created_at = datetime()
                """

                session.run(query,
                    source_id=relation.source_entity,
                    target_id=relation.target_entity,
                    relation_type=relation.relation_type,
                    properties=json.dumps(relation.properties),
                    weight=relation.weight,
                    confidence=relation.confidence
                )

                logger.info(f"关系添加成功: {relation.source_entity} -> {relation.target_entity}")
                return True

        except Exception as e:
            logger.error(f"关系添加失败: {e}")
            return False

    def query_entities(self, entity_type: str, limit: int = 100) -> List[Dict]:
        """查询实体"""
        try:
            if self.driver is None:
                # 内存模式
                entities = []
                for entity in self._memory_entities.values():
                    if entity.entity_type == entity_type:
                        entities.append({
                            'id': entity.id,
                            'name': entity.name,
                            'properties': entity.properties
                        })
                        if len(entities) >= limit:
                            break
                return entities

            with self.driver.session() as session:
                query = """
                MATCH (e:Entity)
                WHERE e.type = $entity_type
                RETURN e.id as id, e.name as name, e.properties as properties
                LIMIT $limit
                """

                result = session.run(query, entity_type=entity_type, limit=limit)
                return [record.data() for record in result]

        except Exception as e:
            logger.error(f"实体查询失败: {e}")
            return []

    def find_related_entities(self, entity_id: str, relation_type: str = None) -> List[Dict]:
        """查找相关实体"""
        try:
            with self.driver.session() as session:
                if relation_type:
                    query = """
                    MATCH (source:Entity {id: $entity_id})-[r:RELATION {type: $relation_type}]->(target:Entity)
                    RETURN target.id as id, target.name as name, target.type as type, r.weight as weight
                    """
                    result = session.run(query, entity_id=entity_id, relation_type=relation_type)
                else:
                    query = """
                    MATCH (source:Entity {id: $entity_id})-[r:RELATION]->(target:Entity)
                    RETURN target.id as id, target.name as name, target.type as type, r.weight as weight
                    """
                    result = session.run(query, entity_id=entity_id)

                return [record.data() for record in result]

        except Exception as e:
            logger.error(f"相关实体查询失败: {e}")
            return []

    def semantic_search(self, query_text: str, entity_type: str = None, limit: int = 10) -> List[Dict]:
        """语义搜索"""
        try:
            # 生成查询嵌入
            query_embedding = self.multimodal_processor.process_text(query_text)

            with self.driver.session() as session:
                if entity_type:
                    query = """
                    MATCH (e:Entity)
                    WHERE e.type = $entity_type
                    RETURN e.id as id, e.name as name, e.embedding as embedding, e.properties as properties
                    """
                    result = session.run(query, entity_type=entity_type)
                else:
                    query = """
                    MATCH (e:Entity)
                    RETURN e.id as id, e.name as name, e.embedding as embedding, e.properties as properties
                    """
                    result = session.run(query)

                entities = []
                for record in result:
                    entity_embedding = np.array(record['embedding'])
                    similarity = cosine_similarity(
                        query_embedding.reshape(1, -1),
                        entity_embedding.reshape(1, -1)
                    )[0][0]

                    if similarity >= self.config.similarity_threshold:
                        entities.append({
                            'id': record['id'],
                            'name': record['name'],
                            'similarity': similarity,
                            'properties': record['properties']
                        })

                # 按相似度排序
                entities.sort(key=lambda x: x['similarity'], reverse=True)
                return entities[:limit]

        except Exception as e:
            logger.error(f"语义搜索失败: {e}")
            return []

    def build_nutrition_knowledge_graph(self, nutrition_data: List[Dict]) -> bool:
        """构建营养知识图谱"""
        try:
            logger.info("开始构建营养知识图谱...")

            # 添加食物实体
            for food_data in nutrition_data:
                food_entity = Entity(
                    id=f"food_{food_data.get('id', '')}",
                    name=food_data.get('name', ''),
                    entity_type='Food',
                    properties={
                        'calories': food_data.get('calories', 0),
                        'protein': food_data.get('protein', 0),
                        'carbs': food_data.get('carbs', 0),
                        'fat': food_data.get('fat', 0),
                        'fiber': food_data.get('fiber', 0),
                        'sugar': food_data.get('sugar', 0),
                        'sodium': food_data.get('sodium', 0),
                        'gi_index': food_data.get('gi_index', 50)
                    }
                )
                self.add_entity(food_entity)

            # 添加营养素实体
            nutrients = ['蛋白质', '碳水化合物', '脂肪', '纤维', '维生素C', '钙', '铁']
            for nutrient in nutrients:
                nutrient_entity = Entity(
                    id=f"nutrient_{nutrient}",
                    name=nutrient,
                    entity_type='Nutrient',
                    properties={'category': 'macronutrient' if nutrient in ['蛋白质', '碳水化合物', '脂肪'] else 'micronutrient'}
                )
                self.add_entity(nutrient_entity)

            # 添加疾病实体
            diseases = ['糖尿病', '高血压', '高血脂', '肥胖', '心血管疾病']
            for disease in diseases:
                disease_entity = Entity(
                    id=f"disease_{disease}",
                    name=disease,
                    entity_type='Disease',
                    properties={'category': 'chronic_disease'}
                )
                self.add_entity(disease_entity)

            # 添加关系
            self._add_nutrition_relations(nutrition_data)

            logger.info("营养知识图谱构建完成")
            return True

        except Exception as e:
            logger.error(f"营养知识图谱构建失败: {e}")
            return False

    def _add_nutrition_relations(self, nutrition_data: List[Dict]):
        """添加营养关系"""
        try:
            for food_data in nutrition_data:
                food_id = f"food_{food_data.get('id', '')}"

                # 食物-营养素关系
                nutrients = ['蛋白质', '碳水化合物', '脂肪', '纤维']
                for nutrient in nutrients:
                    nutrient_id = f"nutrient_{nutrient}"
                    relation = Relation(
                        id=f"{food_id}_contains_{nutrient_id}",
                        source_entity=food_id,
                        target_entity=nutrient_id,
                        relation_type='CONTAINS',
                        properties={'amount': food_data.get(nutrient.lower(), 0)},
                        weight=food_data.get(nutrient.lower(), 0) / 100.0,
                        confidence=0.9
                    )
                    self.add_relation(relation)

                # 食物-疾病关系（基于GI指数）
                gi_index = food_data.get('gi_index', 50)
                if gi_index > 70:
                    # 高GI食物对糖尿病有害
                    diabetes_id = "disease_糖尿病"
                    relation = Relation(
                        id=f"{food_id}_harmful_for_{diabetes_id}",
                        source_entity=food_id,
                        target_entity=diabetes_id,
                        relation_type='HARMFUL_FOR',
                        properties={'gi_index': gi_index},
                        weight=0.8,
                        confidence=0.7
                    )
                    self.add_relation(relation)
                elif gi_index < 55:
                    # 低GI食物对糖尿病有益
                    diabetes_id = "disease_糖尿病"
                    relation = Relation(
                        id=f"{food_id}_beneficial_for_{diabetes_id}",
                        source_entity=food_id,
                        target_entity=diabetes_id,
                        relation_type='BENEFICIAL_FOR',
                        properties={'gi_index': gi_index},
                        weight=0.8,
                        confidence=0.7
                    )
                    self.add_relation(relation)

        except Exception as e:
            logger.error(f"营养关系添加失败: {e}")

    def close(self):
        """关闭数据库连接"""
        if self.driver:
            self.driver.close()

def create_knowledge_graph_builder(config: KnowledgeGraphConfig = None) -> KnowledgeGraphBuilder:
    """创建知识图谱构建器"""
    if config is None:
        config = KnowledgeGraphConfig()
    return KnowledgeGraphBuilder(config)

if __name__ == "__main__":
    # 测试知识图谱构建
    config = KnowledgeGraphConfig()
    kg_builder = create_knowledge_graph_builder(config)

    # 示例营养数据
    sample_nutrition_data = [
        {
            'id': '1',
            'name': '燕麦',
            'calories': 389,
            'protein': 16.9,
            'carbs': 66.2,
            'fat': 6.9,
            'fiber': 10.6,
            'gi_index': 55
        },
        {
            'id': '2',
            'name': '白米饭',
            'calories': 130,
            'protein': 2.7,
            'carbs': 28.2,
            'fat': 0.3,
            'fiber': 0.4,
            'gi_index': 83
        }
    ]

    # 构建知识图谱
    success = kg_builder.build_nutrition_knowledge_graph(sample_nutrition_data)

    if success:
        print("知识图谱构建成功！")

        # 测试查询
        foods = kg_builder.query_entities('Food')
        print(f"食物实体数量: {len(foods)}")

        # 测试语义搜索
        results = kg_builder.semantic_search('低糖食物', 'Food')
        print(f"语义搜索结果: {results}")

    kg_builder.close()
