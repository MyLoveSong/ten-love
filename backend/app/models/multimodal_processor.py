"""
多模态数据处理模块
整合文本、图像、结构化数据的多模态知识抽取
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import cv2
import requests
from io import BytesIO
import re
import jieba
from transformers import (
    AutoTokenizer, AutoModel,
    CLIPProcessor, CLIPModel,
    pipeline
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

logger = logging.getLogger(__name__)

@dataclass
class MultiModalData:
    """多模态数据"""
    text: Optional[str] = None
    image: Optional[np.ndarray] = None
    structured_data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ExtractedEntity:
    """抽取的实体"""
    text: str
    entity_type: str
    confidence: float
    start_pos: int
    end_pos: int
    properties: Dict[str, Any]

@dataclass
class ExtractedRelation:
    """抽取的关系"""
    source_entity: str
    target_entity: str
    relation_type: str
    confidence: float
    context: str
    properties: Dict[str, Any]

class TextProcessor:
    """文本处理器"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 中文NLP模型
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
        self.model = AutoModel.from_pretrained('bert-base-chinese').to(self.device)

        # 实体识别和关系抽取pipeline
        self.ner_pipeline = pipeline(
            "ner",
            model="ckiplab/bert-base-chinese-ner",
            tokenizer="ckiplab/bert-base-chinese-ner",
            device=0 if torch.cuda.is_available() else -1
        )

        # 中文分词
        jieba.setLogLevel(logging.INFO)

        # 营养相关关键词
        self.nutrition_keywords = {
            'food': ['食物', '食材', '菜品', '菜肴', '食品', '营养品'],
            'nutrient': ['蛋白质', '碳水化合物', '脂肪', '纤维', '维生素', '矿物质', '卡路里', '热量'],
            'disease': ['糖尿病', '高血压', '高血脂', '肥胖', '心血管疾病', '癌症'],
            'cooking_method': ['蒸', '煮', '炒', '炸', '烤', '炖', '煎', '凉拌'],
            'taste': ['甜', '咸', '酸', '辣', '苦', '鲜', '香']
        }

    def extract_entities(self, text: str) -> List[ExtractedEntity]:
        """从文本中抽取实体"""
        try:
            entities = []

            # 使用NER模型
            ner_results = self.ner_pipeline(text)

            for result in ner_results:
                entity = ExtractedEntity(
                    text=result['word'],
                    entity_type=result['entity'],
                    confidence=result['score'],
                    start_pos=result['start'],
                    end_pos=result['end'],
                    properties={}
                )
                entities.append(entity)

            # 基于关键词的实体抽取
            keyword_entities = self._extract_keyword_entities(text)
            entities.extend(keyword_entities)

            return entities

        except Exception as e:
            logger.error(f"实体抽取失败: {e}")
            return []

    def _extract_keyword_entities(self, text: str) -> List[ExtractedEntity]:
        """基于关键词抽取实体"""
        entities = []

        for entity_type, keywords in self.nutrition_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    start_pos = text.find(keyword)
                    end_pos = start_pos + len(keyword)

                    entity = ExtractedEntity(
                        text=keyword,
                        entity_type=entity_type,
                        confidence=0.8,
                        start_pos=start_pos,
                        end_pos=end_pos,
                        properties={'extraction_method': 'keyword'}
                    )
                    entities.append(entity)

        return entities

    def extract_relations(self, text: str, entities: List[ExtractedEntity]) -> List[ExtractedRelation]:
        """从文本中抽取关系"""
        try:
            relations = []

            # 基于规则的关系抽取
            relations.extend(self._extract_rule_based_relations(text, entities))

            # 基于模式的关系抽取
            relations.extend(self._extract_pattern_based_relations(text))

            return relations

        except Exception as e:
            logger.error(f"关系抽取失败: {e}")
            return []

    def _extract_rule_based_relations(self, text: str, entities: List[ExtractedEntity]) -> List[ExtractedRelation]:
        """基于规则的关系抽取"""
        relations = []

        # 定义关系模式
        relation_patterns = {
            'CONTAINS': [r'含有', r'包含', r'富含', r'含有'],
            'BENEFICIAL_FOR': [r'有益于', r'有助于', r'对.*好', r'适合'],
            'HARMFUL_FOR': [r'不利于', r'有害于', r'对.*不好', r'不适合'],
            'CAUSES': [r'导致', r'引起', r'造成'],
            'PREVENTS': [r'预防', r'防止', r'避免']
        }

        for relation_type, patterns in relation_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    # 查找匹配的实体对
                    for i, entity1 in enumerate(entities):
                        for j, entity2 in enumerate(entities):
                            if i != j and self._entities_in_context(text, entity1, entity2, match.start(), match.end()):
                                relation = ExtractedRelation(
                                    source_entity=entity1.text,
                                    target_entity=entity2.text,
                                    relation_type=relation_type,
                                    confidence=0.7,
                                    context=text[max(0, match.start()-20):match.end()+20],
                                    properties={'extraction_method': 'rule_based'}
                                )
                                relations.append(relation)

        return relations

    def _extract_pattern_based_relations(self, text: str) -> List[ExtractedRelation]:
        """基于模式的关系抽取"""
        relations = []

        # 营养含量模式
        nutrition_pattern = r'(\w+)[含有|包含|富含](\d+(?:\.\d+)?)\s*(克|g|毫克|mg|微克|μg)'
        matches = re.finditer(nutrition_pattern, text)

        for match in matches:
            food_name = match.group(1)
            amount = float(match.group(2))
            unit = match.group(3)

            relation = ExtractedRelation(
                source_entity=food_name,
                target_entity='营养成分',
                relation_type='CONTAINS',
                confidence=0.9,
                context=match.group(0),
                properties={
                    'amount': amount,
                    'unit': unit,
                    'extraction_method': 'pattern_based'
                }
            )
            relations.append(relation)

        return relations

    def _entities_in_context(self, text: str, entity1: ExtractedEntity, entity2: ExtractedEntity,
                           start: int, end: int, max_distance: int = 50) -> bool:
        """检查两个实体是否在上下文中"""
        distance1 = min(abs(entity1.start_pos - start), abs(entity1.end_pos - end))
        distance2 = min(abs(entity2.start_pos - start), abs(entity2.end_pos - end))

        return distance1 <= max_distance and distance2 <= max_distance

    def generate_text_embedding(self, text: str) -> np.ndarray:
        """生成文本嵌入"""
        try:
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

            return embedding.squeeze()

        except Exception as e:
            logger.error(f"文本嵌入生成失败: {e}")
            return np.zeros(768)

class ImageProcessor:
    """图像处理器"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # CLIP模型用于图文匹配
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)

        # 食物识别模型（预留）
        # from torchvision.models import ResNet50_Weights
        # self.food_classifier = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)

        # 食物相关文本模板
        self.food_templates = [
            "一盘{food}",
            "美味的{food}",
            "营养丰富的{food}",
            "健康的{food}",
            "新鲜的{food}"
        ]

        # 常见食物列表
        self.common_foods = [
            '米饭', '面条', '馒头', '包子', '饺子',
            '鸡肉', '猪肉', '牛肉', '鱼肉', '虾',
            '白菜', '萝卜', '土豆', '西红柿', '黄瓜',
            '苹果', '香蕉', '橙子', '葡萄', '草莓'
        ]

    def process_image(self, image_path: str) -> Dict[str, Any]:
        """处理图像"""
        try:
            # 加载图像
            if image_path.startswith('http'):
                response = requests.get(image_path)
                image = Image.open(BytesIO(response.content))
            else:
                image = Image.open(image_path)

            # 图像预处理
            processed_image = self._preprocess_image(image)

            # 食物识别
            food_recognition = self._recognize_food(image)

            # 营养分析
            nutrition_analysis = self._analyze_nutrition(image)

            return {
                'processed_image': processed_image,
                'food_recognition': food_recognition,
                'nutrition_analysis': nutrition_analysis,
                'image_features': self._extract_image_features(image)
            }

        except Exception as e:
            logger.error(f"图像处理失败: {e}")
            return {}

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """图像预处理"""
        try:
            # 调整大小
            image = image.resize((224, 224))

            # 转换为numpy数组
            image_array = np.array(image)

            # 归一化
            image_array = image_array.astype(np.float32) / 255.0

            return image_array

        except Exception as e:
            logger.error(f"图像预处理失败: {e}")
            return np.zeros((224, 224, 3))

    def _recognize_food(self, image: Image.Image) -> Dict[str, Any]:
        """食物识别"""
        try:
            # 使用CLIP进行食物识别
            food_candidates = []

            for food in self.common_foods:
                for template in self.food_templates:
                    text = template.format(food=food)
                    food_candidates.append(text)

            # 计算图像和文本的相似度
            inputs = self.clip_processor(
                text=food_candidates,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)

            # 找到最可能的食物
            best_idx = probs.argmax().item()
            best_food = food_candidates[best_idx]
            confidence = probs[0][best_idx].item()

            return {
                'recognized_food': best_food,
                'confidence': confidence,
                'all_candidates': list(zip(food_candidates, probs[0].tolist()))
            }

        except Exception as e:
            logger.error(f"食物识别失败: {e}")
            return {'recognized_food': '未知食物', 'confidence': 0.0}

    def _analyze_nutrition(self, image: Image.Image) -> Dict[str, Any]:
        """营养分析（基于图像）"""
        try:
            # 简化的营养分析
            # 实际应用中可以使用更复杂的模型

            # 基于颜色分析
            image_array = np.array(image)

            # 分析主要颜色
            dominant_colors = self._get_dominant_colors(image_array)

            # 基于颜色推断营养
            nutrition_estimate = self._estimate_nutrition_from_colors(dominant_colors)

            return nutrition_estimate

        except Exception as e:
            logger.error(f"营养分析失败: {e}")
            return {}

    def _get_dominant_colors(self, image_array: np.ndarray) -> List[Tuple[int, int, int]]:
        """获取主要颜色"""
        try:
            # 重塑图像数组
            pixels = image_array.reshape(-1, 3)

            # 使用K-means聚类找到主要颜色
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=5, random_state=42)
            kmeans.fit(pixels)

            colors = kmeans.cluster_centers_.astype(int)
            return [tuple(color) for color in colors]

        except Exception as e:
            logger.error(f"颜色分析失败: {e}")
            return [(128, 128, 128)]

    def _estimate_nutrition_from_colors(self, colors: List[Tuple[int, int, int]]) -> Dict[str, Any]:
        """基于颜色估计营养"""
        nutrition_estimate = {
            'calories': 0,
            'protein': 0,
            'carbs': 0,
            'fat': 0,
            'fiber': 0
        }

        for r, g, b in colors:
            # 简化的颜色-营养映射
            if r > 200 and g < 100 and b < 100:  # 红色 - 蛋白质
                nutrition_estimate['protein'] += 5
            elif r < 100 and g > 200 and b < 100:  # 绿色 - 纤维
                nutrition_estimate['fiber'] += 3
            elif r > 150 and g > 150 and b < 100:  # 黄色 - 碳水化合物
                nutrition_estimate['carbs'] += 4
            elif r < 100 and g < 100 and b > 200:  # 蓝色 - 低卡路里
                nutrition_estimate['calories'] += 2

        return nutrition_estimate

    def _extract_image_features(self, image: Image.Image) -> np.ndarray:
        """提取图像特征"""
        try:
            # 使用CLIP提取图像特征
            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                image_features = image_features.cpu().numpy()

            return image_features.squeeze()

        except Exception as e:
            logger.error(f"图像特征提取失败: {e}")
            return np.zeros(512)

class StructuredDataProcessor:
    """结构化数据处理器"""

    def __init__(self):
        self.nutrition_units = {
            'calories': ['kcal', 'cal', '卡路里', '热量'],
            'protein': ['g', '克', '蛋白质'],
            'carbs': ['g', '克', '碳水化合物', '碳水'],
            'fat': ['g', '克', '脂肪'],
            'fiber': ['g', '克', '纤维'],
            'sugar': ['g', '克', '糖'],
            'sodium': ['mg', '毫克', '钠']
        }

    def process_nutrition_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理营养数据"""
        try:
            processed_data = {}

            for key, value in data.items():
                if isinstance(value, str):
                    # 尝试提取数值
                    numeric_value = self._extract_numeric_value(value)
                    if numeric_value is not None:
                        processed_data[key] = numeric_value
                    else:
                        processed_data[key] = value
                else:
                    processed_data[key] = value

            # 标准化单位
            processed_data = self._standardize_units(processed_data)

            return processed_data

        except Exception as e:
            logger.error(f"营养数据处理失败: {e}")
            return data

    def _extract_numeric_value(self, text: str) -> Optional[float]:
        """从文本中提取数值"""
        try:
            # 匹配数字模式
            pattern = r'(\d+(?:\.\d+)?)'
            match = re.search(pattern, text)

            if match:
                return float(match.group(1))

            return None

        except Exception as e:
            logger.error(f"数值提取失败: {e}")
            return None

    def _standardize_units(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """标准化单位"""
        try:
            standardized_data = {}

            for key, value in data.items():
                if isinstance(value, str):
                    # 检查是否包含单位信息
                    for nutrient, units in self.nutrition_units.items():
                        if any(unit in value.lower() for unit in units):
                            # 提取数值
                            numeric_value = self._extract_numeric_value(value)
                            if numeric_value is not None:
                                standardized_data[nutrient] = numeric_value
                            break
                    else:
                        standardized_data[key] = value
                else:
                    standardized_data[key] = value

            return standardized_data

        except Exception as e:
            logger.error(f"单位标准化失败: {e}")
            return data

class MultiModalProcessor:
    """多模态处理器"""

    def __init__(self):
        self.text_processor = TextProcessor()
        self.image_processor = ImageProcessor()
        self.structured_processor = StructuredDataProcessor()

    def process_multimodal_data(self, data: MultiModalData) -> Dict[str, Any]:
        """处理多模态数据"""
        try:
            result = {
                'entities': [],
                'relations': [],
                'embeddings': {},
                'processed_data': {}
            }

            # 处理文本数据
            if data.text:
                text_result = self._process_text_data(data.text)
                result['entities'].extend(text_result['entities'])
                result['relations'].extend(text_result['relations'])
                result['embeddings']['text'] = text_result['embedding']

            # 处理图像数据
            if data.image is not None:
                image_result = self._process_image_data(data.image)
                result['embeddings']['image'] = image_result['features']
                result['processed_data']['image'] = image_result

            # 处理结构化数据
            if data.structured_data:
                structured_result = self.structured_processor.process_nutrition_data(data.structured_data)
                result['processed_data']['structured'] = structured_result

            # 融合多模态信息
            result['fused_embedding'] = self._fuse_embeddings(result['embeddings'])

            return result

        except Exception as e:
            logger.error(f"多模态数据处理失败: {e}")
            return {}

    def _process_text_data(self, text: str) -> Dict[str, Any]:
        """处理文本数据"""
        try:
            entities = self.text_processor.extract_entities(text)
            relations = self.text_processor.extract_relations(text, entities)
            embedding = self.text_processor.generate_text_embedding(text)

            return {
                'entities': entities,
                'relations': relations,
                'embedding': embedding
            }

        except Exception as e:
            logger.error(f"文本数据处理失败: {e}")
            return {'entities': [], 'relations': [], 'embedding': np.zeros(768)}

    def _process_image_data(self, image: Union[str, np.ndarray]) -> Dict[str, Any]:
        """处理图像数据"""
        try:
            if isinstance(image, str):
                return self.image_processor.process_image(image)
            else:
                # 处理numpy数组
                pil_image = Image.fromarray(image.astype(np.uint8))
                return self.image_processor.process_image(pil_image)

        except Exception as e:
            logger.error(f"图像数据处理失败: {e}")
            return {}

    def _fuse_embeddings(self, embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        """融合多模态嵌入"""
        try:
            if not embeddings:
                return np.zeros(768)

            # 简单的加权平均融合
            weights = {'text': 0.6, 'image': 0.4}

            fused_embedding = np.zeros(768)
            total_weight = 0

            for modality, embedding in embeddings.items():
                if modality in weights:
                    weight = weights[modality]
                    fused_embedding += embedding * weight
                    total_weight += weight

            if total_weight > 0:
                fused_embedding /= total_weight

            return fused_embedding

        except Exception as e:
            logger.error(f"嵌入融合失败: {e}")
            return np.zeros(768)

def create_multimodal_processor() -> MultiModalProcessor:
    """创建多模态处理器"""
    return MultiModalProcessor()

if __name__ == "__main__":
    # 测试多模态处理器
    processor = create_multimodal_processor()

    # 测试文本处理
    text_data = MultiModalData(
        text="燕麦富含蛋白质和纤维，有助于控制血糖，适合糖尿病患者食用。"
    )

    result = processor.process_multimodal_data(text_data)

    print("文本处理结果:")
    print(f"实体数量: {len(result['entities'])}")
    for entity in result['entities']:
        print(f"- {entity.text} ({entity.entity_type}): {entity.confidence:.2f}")

    print(f"关系数量: {len(result['relations'])}")
    for relation in result['relations']:
        print(f"- {relation.source_entity} -> {relation.target_entity} ({relation.relation_type})")

    print("多模态处理器测试完成！")
