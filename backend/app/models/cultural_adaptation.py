"""
饮食文化知识图谱和地域偏好适配系统
实现跨文化知识蒸馏和动态文化调整机制

新增：
1) TransE 嵌入与三元组存储（DRY/模块化，供后续蒸馏与API复用）
2) 将菜系与宗教禁忌规则编码到知识图谱（便于文化适配逻辑量化）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class TripleStore:
    """轻量级三元组存储，支持实体/关系注册与查询"""
    def __init__(self):
        self.entities: Dict[str, int] = {}
        self.relations: Dict[str, int] = {}
        self.entity_list: List[str] = []
        self.relation_list: List[str] = []
        self.triples: List[Tuple[str, str, str]] = []
        self._triple_set: Set[Tuple[str, str, str]] = set()
        # 简单同义词规范化（可通过MCP外部扩展）
        self.synonyms: Dict[str, str] = {
            '辣椒': '辣椒', '花椒': '花椒', '郫县豆瓣': '豆瓣酱',
            'halal': '清真', 'vegetarian': '素食', 'vegan': '纯素',
            '白切鸡': '白切鸡', '清蒸鱼': '清蒸鱼', '宫保鸡丁': '宫保鸡丁',
            'oily': '偏油', 'spicy': '偏辣', 'salty': '偏咸', 'sweet': '偏甜',
        }

    def _normalize(self, name: str) -> str:
        norm = (name or '').strip()
        # 统一大小写/空白与简单映射
        if norm.lower() in self.synonyms:
            return self.synonyms[norm.lower()]
        if norm in self.synonyms:
            return self.synonyms[norm]
        return norm

    def _ensure_entity(self, name: str) -> int:
        name = self._normalize(name)
        if name not in self.entities:
            self.entities[name] = len(self.entity_list)
            self.entity_list.append(name)
        return self.entities[name]

    def _ensure_relation(self, name: str) -> int:
        name = self._normalize(name)
        if name not in self.relations:
            self.relations[name] = len(self.relation_list)
            self.relation_list.append(name)
        return self.relations[name]

    def add_triple(self, h: str, r: str, t: str):
        h_n = self._normalize(h)
        r_n = self._normalize(r)
        t_n = self._normalize(t)
        if (h_n, r_n, t_n) in self._triple_set:
            return
        self._ensure_entity(h_n)
        self._ensure_relation(r_n)
        self._ensure_entity(t_n)
        self.triples.append((h_n, r_n, t_n))
        self._triple_set.add((h_n, r_n, t_n))

class TransEEmbedder(nn.Module):
    """TransE 嵌入器（L1/L2可选，简化实现，满足ES模块复用）"""
    def __init__(self, num_entities: int, num_relations: int, emb_dim: int = 128, margin: float = 1.0, p: int = 1):
        super().__init__()
        self.emb_dim = emb_dim
        self.margin = margin
        self.p = p  # 1 or 2
        self.entity_emb = nn.Embedding(num_entities, emb_dim)
        self.relation_emb = nn.Embedding(num_relations, emb_dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def score(self, h_idx: torch.Tensor, r_idx: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        h = self.entity_emb(h_idx)
        r = self.relation_emb(r_idx)
        t = self.entity_emb(t_idx)
        diff = h + r - t
        if self.p == 1:
            return torch.norm(diff, p=1, dim=-1)
        return torch.norm(diff, p=2, dim=-1)

    def loss(self, pos: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], neg: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        pos_score = self.score(*pos)
        neg_score = self.score(*neg)
        # margin ranking: max(0, margin + pos - neg)；分数越小越好
        return torch.relu(self.margin + pos_score - neg_score).mean()

class RotatEEmbedder(nn.Module):
    """RotatE 嵌入器（复数空间旋转，适合一对多/多对一关系）"""
    def __init__(self, num_entities: int, num_relations: int, emb_dim: int = 128, margin: float = 6.0):
        super().__init__()
        assert emb_dim % 2 == 0, "RotatE要求嵌入维度为偶数用于复数表示"
        self.emb_dim = emb_dim
        self.margin = margin
        self.entity_emb = nn.Embedding(num_entities, emb_dim)
        self.relation_emb = nn.Embedding(num_relations, emb_dim)
        # 初始化（相位均匀，模长受控）
        nn.init.uniform_(self.entity_emb.weight, a=-0.5, b=0.5)
        nn.init.uniform_(self.relation_emb.weight, a=-3.1416, b=3.1416)  # 相位

    def _split_complex(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        real, imag = x.chunk(2, dim=-1)
        return real, imag

    def score(self, h_idx: torch.Tensor, r_idx: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        # 实部/虚部
        h_r, h_i = self._split_complex(self.entity_emb(h_idx))
        t_r, t_i = self._split_complex(self.entity_emb(t_idx))
        phase = self.relation_emb(r_idx)
        r_cos, r_sin = torch.cos(phase[..., : self.emb_dim // 2]), torch.sin(phase[..., : self.emb_dim // 2])
        # 旋转：h ⊙ r  ≈  t
        rot_r = h_r * r_cos - h_i * r_sin
        rot_i = h_r * r_sin + h_i * r_cos
        diff_r = rot_r - t_r
        diff_i = rot_i - t_i
        # L2范数
        return torch.sqrt((diff_r ** 2 + diff_i ** 2).sum(dim=-1) + 1e-9)

    def loss(self, pos: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], neg: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        pos_score = self.score(*pos)
        neg_score = self.score(*neg)
        return torch.relu(self.margin + pos_score - neg_score).mean()

class CuisineType(Enum):
    """菜系类型枚举"""
    SICHUAN = "川菜"
    CANTONESE = "粤菜"
    SHANDONG = "鲁菜"
    WESTERN = "西餐"
    MEDITERRANEAN = "地中海"
    JAPANESE = "日式"
    INDIAN = "印度"

class ReligionType(Enum):
    """宗教类型枚举"""
    HALAL = "清真"
    VEGETARIAN = "素食"
    HINDU = "印度教"
    BUDDHIST = "佛教"
    CHRISTIAN = "基督教"
    NONE = "无限制"

@dataclass
class CulturalPreference:
    """文化偏好数据结构 - 增强版，包含更多文化和个人特征"""
    cuisine_type: CuisineType
    religion: ReligionType
    spice_tolerance: float
    salt_preference: float
    sweet_preference: float
    oil_preference: float
    dietary_restrictions: Set[str]
    regional_ingredients: List[str]
    traditional_medicine: bool
    family_meal_pattern: str
    festival_foods: List[str]
    health_beliefs: List[str]
    personal_features: Dict[str, float]

class CulturalPersonalCrossAttention(nn.Module):
    """文化-个人交叉注意力模块 - 4-head交叉注意力"""

    def __init__(self,
                 cultural_dim: int = 36,
                 personal_dim: int = 32,
                 hidden_dim: int = 128,
                 num_heads: int = 4):
        super().__init__()

        self.cultural_dim = cultural_dim
        self.personal_dim = personal_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # 文化特征投影层
        self.cultural_projection = nn.Linear(cultural_dim, hidden_dim)

        # 个人特征投影层
        self.personal_projection = nn.Linear(personal_dim, hidden_dim)

        # 4-head交叉注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        # 特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, cultural_features: torch.Tensor, personal_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播 - 文化特征作为Q，个人特征作为K/V"""
        batch_size = cultural_features.size(0)

        # 投影到隐藏维度
        cultural_proj = self.cultural_projection(cultural_features)  # [batch, hidden_dim]
        personal_proj = self.personal_projection(personal_features)  # [batch, hidden_dim]

        # 添加序列维度用于注意力计算
        cultural_q = cultural_proj.unsqueeze(1)  # [batch, 1, hidden_dim]
        personal_kv = personal_proj.unsqueeze(1)  # [batch, 1, hidden_dim]

        # 交叉注意力计算
        attn_output, attn_weights = self.cross_attention(
            query=cultural_q,
            key=personal_kv,
            value=personal_kv
        )

        # 移除序列维度
        attn_output = attn_output.squeeze(1)  # [batch, hidden_dim]

        # 残差连接和层归一化
        attn_output = self.layer_norm(attn_output + cultural_proj)

        # 特征融合
        combined_features = torch.cat([attn_output, personal_proj], dim=-1)
        fused_features = self.fusion_layer(combined_features)

        return {
            'cross_attention_output': attn_output,
            'attention_weights': attn_weights,
            'fused_features': fused_features,
            'cultural_projected': cultural_proj,
            'personal_projected': personal_proj
        }

class TemporalSmoothLoss(nn.Module):
    """时间平滑损失模块 - 权重变化L2正则化"""

    def __init__(self, lambda_smooth: float = 0.01):
        super().__init__()
        self.lambda_smooth = lambda_smooth

    def forward(self,
                current_weights: torch.Tensor,
                previous_weights: torch.Tensor) -> torch.Tensor:
        """计算时间平滑损失"""
        if previous_weights is None:
            return torch.tensor(0.0, device=current_weights.device)

        # L2正则化权重变化
        weight_diff = current_weights - previous_weights
        smooth_loss = torch.mean(torch.norm(weight_diff, p=2, dim=-1))

        return self.lambda_smooth * smooth_loss

class DynamicCulturalDriftWindow(nn.Module):
    """动态文化漂移窗口 - 30天滑动KG"""

    def __init__(self,
                 window_size: int = 30,
                 cultural_dim: int = 36,
                 hidden_dim: int = 128):
        super().__init__()

        self.window_size = window_size
        self.cultural_dim = cultural_dim
        self.hidden_dim = hidden_dim

        # 时间编码器
        self.temporal_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )

        # 文化特征时间编码器
        self.cultural_temporal_encoder = nn.Sequential(
            nn.Linear(cultural_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, cultural_dim)
        )

        # 滑动窗口聚合器
        self.window_aggregator = nn.Sequential(
            nn.Linear(cultural_dim * window_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, cultural_dim)
        )

    def forward(self,
                cultural_features: torch.Tensor,
                timestamps: torch.Tensor,
                historical_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """前向传播 - 处理30天滑动窗口"""
        batch_size = cultural_features.size(0)

        # 时间编码
        time_encoded = self.temporal_encoder(timestamps.unsqueeze(-1))

        # 确保维度匹配 - 将时间编码压缩到与文化特征相同的维度
        if time_encoded.dim() == 3:
            time_encoded = time_encoded.squeeze(1)  # [batch, 1, hidden_dim] -> [batch, hidden_dim]
        elif time_encoded.dim() == 2 and time_encoded.size(1) != cultural_features.size(1):
            # 如果时间编码的第二个维度与文化特征不匹配，进行适配
            time_encoded = time_encoded[:, :cultural_features.size(1)]

        # 文化特征与时间融合
        cultural_with_time = torch.cat([cultural_features, time_encoded], dim=-1)
        temporal_cultural = self.cultural_temporal_encoder(cultural_with_time)

        # 如果有历史特征，进行滑动窗口聚合
        if historical_features is not None:
            # 确保历史特征维度正确
            if historical_features.size(-1) != self.cultural_dim:
                historical_features = F.adaptive_avg_pool1d(
                    historical_features.unsqueeze(1),
                    self.cultural_dim
                ).squeeze(1)

            # 滑动窗口聚合
            windowed_features = self.window_aggregator(
                historical_features.view(batch_size, -1)
            )

            # 融合当前和历史特征
            final_features = 0.7 * temporal_cultural + 0.3 * windowed_features
        else:
            final_features = temporal_cultural

        return {
            'temporal_cultural_features': final_features,
            'time_encoded': time_encoded,
            'window_aggregated': windowed_features if historical_features is not None else None
        }

class CulturalAdaptationModule(nn.Module):
    """文化适配模块 - 增强版，支持动态文化漂移窗口和交叉注意力"""

    def __init__(self,
                 feature_dim: int = 512,
                 cultural_dim: int = 64,
                 personal_dim: int = 32,
                 use_cross_attention: bool = True,
                 use_temporal_drift: bool = True):
        super().__init__()

        self.feature_dim = feature_dim
        self.cultural_dim = cultural_dim
        self.personal_dim = personal_dim
        self.use_cross_attention = use_cross_attention
        self.use_temporal_drift = use_temporal_drift

        # 文化特征编码器 - 完整版，支持36维文化特征
        self.cultural_encoder = nn.Sequential(
            nn.Linear(36, cultural_dim),  # 完整版文化特征维度
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(cultural_dim, cultural_dim)
        )

        # 个人特征编码器 - 完整版，支持32维个人特征
        self.personal_encoder = nn.Sequential(
            nn.Linear(32, personal_dim),  # 完整版个人特征维度
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(personal_dim, personal_dim)
        )

        # 交叉注意力模块
        if use_cross_attention:
            self.cross_attention = CulturalPersonalCrossAttention(
                cultural_dim=36,
                personal_dim=32,
                hidden_dim=128,
                num_heads=4
            )

        # 动态文化漂移窗口
        if use_temporal_drift:
            self.temporal_drift = DynamicCulturalDriftWindow(
                window_size=30,
                cultural_dim=36,
                hidden_dim=128
            )
            self.temporal_smooth_loss = TemporalSmoothLoss(lambda_smooth=0.01)

        # 特征融合网络
        if use_cross_attention:
            self.fusion_network = nn.Sequential(
                nn.Linear(128, 128),  # 交叉注意力输出维度
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, cultural_dim + personal_dim)
            )
        else:
            # 传统拼接方式
            self.cultural_personal_fusion = nn.Sequential(
                nn.Linear(cultural_dim + personal_dim, (cultural_dim + personal_dim) // 2),
                nn.ReLU(),
                nn.Linear((cultural_dim + personal_dim) // 2, cultural_dim + personal_dim)
            )

        # 动态权重生成器 - 完整版，支持融合特征输入
        self.weight_generator = nn.Sequential(
            nn.Linear(cultural_dim + personal_dim, 128),  # 融合特征维度
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim),
            nn.Tanh()
        )

    def forward(self,
                food_features: torch.Tensor,
                cultural_features: torch.Tensor,
                personal_features: torch.Tensor = None,
                timestamps: torch.Tensor = None,
                historical_features: torch.Tensor = None,
                previous_weights: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """文化适配前向传播 - 增强版，支持动态文化漂移窗口和交叉注意力"""

        # 1. 动态文化漂移窗口处理
        if self.use_temporal_drift and timestamps is not None:
            temporal_result = self.temporal_drift(
                cultural_features, timestamps, historical_features
            )
            cultural_features = temporal_result['temporal_cultural_features']

        # 2. 编码文化特征
        cultural_encoded = self.cultural_encoder(cultural_features)

        # 3. 编码个人特征
        if personal_features is not None:
            personal_encoded = self.personal_encoder(personal_features)
        else:
            personal_encoded = torch.zeros(food_features.size(0), self.personal_dim, device=food_features.device)

        # 4. 交叉注意力或传统融合
        if self.use_cross_attention:
            # 使用交叉注意力机制
            cross_attn_result = self.cross_attention(cultural_features, personal_features)
            fused_features = self.fusion_network(cross_attn_result['fused_features'])

            # 确保维度匹配
            if fused_features.size(-1) != self.cultural_dim + self.personal_dim:
                fused_features = torch.cat([
                    cross_attn_result['cross_attention_output'],
                    personal_encoded
                ], dim=-1)
        else:
            # 传统拼接方式
            combined_features = torch.cat([cultural_encoded, personal_encoded], dim=-1)
            fused_features = self.cultural_personal_fusion(combined_features)

        # 5. 生成动态权重
        adaptation_weights = self.weight_generator(fused_features)

        # 6. 计算时间平滑损失
        temporal_smooth_loss = torch.tensor(0.0, device=food_features.device)
        if self.use_temporal_drift and previous_weights is not None:
            temporal_smooth_loss = self.temporal_smooth_loss(adaptation_weights, previous_weights)

        # 7. 应用文化适配权重
        adapted_features = food_features * adaptation_weights

        # 8. 构建返回结果
        result = {
            'adapted_features': adapted_features,
            'cultural_encoded': cultural_encoded,
            'personal_encoded': personal_encoded,
            'fused_features': fused_features,
            'adaptation_weights': adaptation_weights,
            'temporal_smooth_loss': temporal_smooth_loss
        }

        # 添加交叉注意力相关信息
        if self.use_cross_attention:
            result.update({
                'cross_attention_output': cross_attn_result['cross_attention_output'],
                'attention_weights': cross_attn_result['attention_weights'],
                'cultural_projected': cross_attn_result['cultural_projected'],
                'personal_projected': cross_attn_result['personal_projected']
            })

        # 添加时间漂移相关信息
        if self.use_temporal_drift and timestamps is not None:
            result.update({
                'temporal_cultural_features': temporal_result['temporal_cultural_features'],
                'time_encoded': temporal_result['time_encoded'],
                'window_aggregated': temporal_result['window_aggregated']
            })

        return result

class CookingPenaltyModule(nn.Module):
    """烹饪方式惩罚模块"""

    def __init__(self, penalty_weights: Dict[str, float] = None):
        super().__init__()
        # 烹饪方式惩罚权重
        self.penalty_weights = penalty_weights or {
            'steamed': 0.0,      # 蒸 - 无惩罚
            'boiled': 0.1,       # 煮 - 轻微惩罚
            'stir_fried': 0.3,   # 炒 - 中等惩罚
            'deep_fried': 0.8,   # 炸 - 高惩罚
            'roasted': 0.4,      # 烤 - 中等惩罚
            'grilled': 0.2,      # 烤 - 轻微惩罚
        }

    def forward(self, food_features: torch.Tensor, cooking_methods: List[str]) -> torch.Tensor:
        """计算烹饪方式惩罚"""
        penalties = torch.zeros(food_features.size(0), device=food_features.device)

        for i, method in enumerate(cooking_methods):
            if i < len(penalties):
                penalty = self.penalty_weights.get(method, 0.5)  # 默认中等惩罚
                penalties[i] = penalty

        return penalties

class TabooFilter(nn.Module):
    """禁忌过滤模块"""

    def __init__(self):
        super().__init__()
        # 宗教和饮食禁忌
        self.taboo_rules = {
            'halal': {
                'forbidden': ['pork', 'alcohol', 'non_halal_meat'],
                'allowed': ['halal_meat', 'fish', 'vegetables', 'grains']
            },
            'vegetarian': {
                'forbidden': ['meat', 'fish', 'poultry', 'seafood'],
                'allowed': ['vegetables', 'fruits', 'grains', 'dairy', 'eggs']
            },
            'vegan': {
                'forbidden': ['meat', 'fish', 'poultry', 'seafood', 'dairy', 'eggs'],
                'allowed': ['vegetables', 'fruits', 'grains', 'nuts', 'seeds']
            },
            'low_salt': {
                'forbidden': ['high_sodium_foods', 'processed_foods', 'pickled_foods'],
                'allowed': ['fresh_vegetables', 'fruits', 'lean_meat', 'grains']
            }
        }

    def filter_foods(self, food_list: List[str], dietary_restrictions: List[str]) -> List[str]:
        """过滤禁忌食物"""
        filtered_foods = []

        for food in food_list:
            is_allowed = True

            for restriction in dietary_restrictions:
                if restriction in self.taboo_rules:
                    forbidden = self.taboo_rules[restriction]['forbidden']
                    if any(forbidden_item in food.lower() for forbidden_item in forbidden):
                        is_allowed = False
                        break

            if is_allowed:
                filtered_foods.append(food)

        return filtered_foods

class TrainableCultureSigma(nn.Module):
    """可训练文化约束权重模块 - 完整版，支持36维文化特征"""

    def __init__(self, num_cultural_features: int = 36):
        super().__init__()
        # 可训练的文化约束权重
        self.culture_sigma = nn.Parameter(torch.ones(num_cultural_features) * 0.1)
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(num_cultural_features, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_cultural_features),
            nn.Softplus()  # 确保输出为正
        )

    def forward(self, cultural_features: torch.Tensor) -> torch.Tensor:
        """计算动态文化约束权重"""
        # 基础权重
        base_weights = torch.sigmoid(self.culture_sigma)

        # 不确定性估计
        uncertainty = self.uncertainty_estimator(cultural_features)

        # 动态调整权重
        dynamic_weights = base_weights * (1 + uncertainty)

        return dynamic_weights

class CulturalAdaptationService:
    """文化适配服务 - 增强版，包含烹饪惩罚、禁忌过滤和可训练权重"""

    def __init__(self):
        self.cultural_module = CulturalAdaptationModule()
        self.cooking_penalty = CookingPenaltyModule()
        self.taboo_filter = TabooFilter()
        self.culture_sigma = TrainableCultureSigma()
        # 知识图谱（三元组 + 嵌入器）
        self.triples = TripleStore()
        # 嵌入器占位，延后初始化；默认使用TransE，可随时切换到RotatE
        self._embedder_name: Optional[str] = 'transe'
        self.transe: Optional[TransEEmbedder] = None
        self.rotate: Optional[RotatEEmbedder] = None

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cultural_module.to(self.device)
        self.cooking_penalty.to(self.device)
        self.culture_sigma.to(self.device)

        # 文化偏好数据库 - 完整版，包含详细的文化和个人特征
        self.cultural_database = {
            "四川": CulturalPreference(
                cuisine_type=CuisineType.SICHUAN,
                religion=ReligionType.NONE,
                spice_tolerance=0.9,
                salt_preference=0.7,
                sweet_preference=0.3,
                oil_preference=0.8,
                dietary_restrictions=set(),
                regional_ingredients=["花椒", "辣椒", "豆瓣酱", "泡菜", "腊肉", "香肠", "豆瓣", "郫县豆瓣"],
                traditional_medicine=True,
                family_meal_pattern="communal",
                festival_foods=["火锅", "麻婆豆腐", "宫保鸡丁", "回锅肉", "水煮鱼", "夫妻肺片", "腊肉", "香肠", "泡菜"],
                health_beliefs=["温热性食物", "祛湿", "补气养血", "阴阳平衡", "四季调养", "食疗养生"],
                personal_features={
                    'age_group': 0.6, 'gender': 0.5, 'education_level': 0.7,
                    'income_level': 0.6, 'activity_level': 0.5, 'health_condition': 0.4,
                    'lifestyle_factors': 0.6, 'psychological_factors': 0.5,
                    'social_support': 0.8, 'health_literacy': 0.7,
                    'cultural_identity_score': 0.9, 'acculturation_level': 0.5,
                    'metabolic_rate': 0.7, 'insulin_sensitivity': 0.6,
                    'genetic_background': 0.8, 'socioeconomic_status': 0.6,
                    'occupation_type': 0.5, 'living_environment': 0.7,
                    'dietary_restrictions': 0.2, 'meal_timing': 0.6,
                    'cooking_frequency': 0.8, 'food_shopping_habits': 0.7,
                    'seasonal_adaptation': 0.8, 'festival_participation': 0.9,
                    'traditional_medicine_usage': 0.8, 'health_checkup_frequency': 0.5,
                    'exercise_preferences': 0.6, 'stress_management': 0.7,
                    'sleep_patterns': 0.4, 'work_life_balance': 0.6
                }
            ),
            "广东": CulturalPreference(
                cuisine_type=CuisineType.CANTONESE,
                religion=ReligionType.NONE,
                spice_tolerance=0.2,
                salt_preference=0.4,
                sweet_preference=0.6,
                oil_preference=0.3,
                dietary_restrictions=set(),
                regional_ingredients=["生抽", "蚝油", "虾米", "海鲜", "鲍鱼", "鱼翅", "燕窝", "花胶"],
                traditional_medicine=True,
                family_meal_pattern="individual",
                festival_foods=["白切鸡", "烧鹅", "点心", "早茶", "煲汤", "糖水", "月饼", "年糕"],
                health_beliefs=["清淡饮食", "清热解毒", "滋阴润燥", "健脾开胃", "汤水养生"],
                personal_features={
                    'age_group': 0.5, 'gender': 0.5, 'education_level': 0.8,
                    'income_level': 0.8, 'activity_level': 0.6, 'health_condition': 0.5,
                    'lifestyle_factors': 0.7, 'psychological_factors': 0.6,
                    'social_support': 0.7, 'health_literacy': 0.8,
                    'cultural_identity_score': 0.8, 'acculturation_level': 0.7,
                    'metabolic_rate': 0.6, 'insulin_sensitivity': 0.7,
                    'genetic_background': 0.7, 'socioeconomic_status': 0.8,
                    'occupation_type': 0.7, 'living_environment': 0.8,
                    'dietary_restrictions': 0.1, 'meal_timing': 0.8,
                    'cooking_frequency': 0.9, 'food_shopping_habits': 0.8,
                    'seasonal_adaptation': 0.7, 'festival_participation': 0.8,
                    'traditional_medicine_usage': 0.7, 'health_checkup_frequency': 0.7,
                    'exercise_preferences': 0.7, 'stress_management': 0.8,
                    'sleep_patterns': 0.7, 'work_life_balance': 0.7
                }
            ),
            "地中海": CulturalPreference(
                cuisine_type=CuisineType.MEDITERRANEAN,
                religion=ReligionType.NONE,
                spice_tolerance=0.4,
                salt_preference=0.3,
                sweet_preference=0.4,
                oil_preference=0.9,
                dietary_restrictions=set(),
                regional_ingredients=["橄榄油", "鱼类", "坚果", "蔬菜", "全谷物", "红酒", "奶酪", "香草"],
                traditional_medicine=True,
                family_meal_pattern="communal",
                festival_foods=["海鲜", "橄榄", "无花果", "葡萄", "面包", "奶酪", "红酒"],
                health_beliefs=["地中海饮食", "心血管健康", "抗炎饮食", "长寿理念"],
                personal_features={
                    'age_group': 0.6, 'gender': 0.5, 'education_level': 0.7,
                    'income_level': 0.7, 'activity_level': 0.8, 'health_condition': 0.6,
                    'lifestyle_factors': 0.8, 'psychological_factors': 0.7,
                    'social_support': 0.8, 'health_literacy': 0.7,
                    'cultural_identity_score': 0.7, 'acculturation_level': 0.8,
                    'metabolic_rate': 0.8, 'insulin_sensitivity': 0.8,
                    'genetic_background': 0.6, 'socioeconomic_status': 0.7,
                    'occupation_type': 0.6, 'living_environment': 0.8,
                    'dietary_restrictions': 0.1, 'meal_timing': 0.7,
                    'cooking_frequency': 0.8, 'food_shopping_habits': 0.8,
                    'seasonal_adaptation': 0.9, 'festival_participation': 0.8,
                    'traditional_medicine_usage': 0.6, 'health_checkup_frequency': 0.6,
                    'exercise_preferences': 0.8, 'stress_management': 0.8,
                    'sleep_patterns': 0.7, 'work_life_balance': 0.8
                }
            ),
            "日式": CulturalPreference(
                cuisine_type=CuisineType.JAPANESE,
                religion=ReligionType.NONE,
                spice_tolerance=0.3,
                salt_preference=0.6,
                sweet_preference=0.4,
                oil_preference=0.2,
                dietary_restrictions=set(),
                regional_ingredients=["味噌", "酱油", "海藻", "鱼类", "米饭", "豆腐", "绿茶", "芥末"],
                traditional_medicine=True,
                family_meal_pattern="individual",
                festival_foods=["寿司", "味噌汤", "绿茶", "天妇罗", "拉面", "和牛", "清酒"],
                health_beliefs=["清淡饮食", "长寿理念", "平衡营养", "精致文化"],
                personal_features={
                    'age_group': 0.7, 'gender': 0.5, 'education_level': 0.9,
                    'income_level': 0.8, 'activity_level': 0.6, 'health_condition': 0.7,
                    'lifestyle_factors': 0.8, 'psychological_factors': 0.6,
                    'social_support': 0.6, 'health_literacy': 0.9,
                    'cultural_identity_score': 0.9, 'acculturation_level': 0.5,
                    'metabolic_rate': 0.6, 'insulin_sensitivity': 0.7,
                    'genetic_background': 0.8, 'socioeconomic_status': 0.8,
                    'occupation_type': 0.8, 'living_environment': 0.8,
                    'dietary_restrictions': 0.2, 'meal_timing': 0.8,
                    'cooking_frequency': 0.7, 'food_shopping_habits': 0.8,
                    'seasonal_adaptation': 0.9, 'festival_participation': 0.8,
                    'traditional_medicine_usage': 0.7, 'health_checkup_frequency': 0.9,
                    'exercise_preferences': 0.6, 'stress_management': 0.6,
                    'sleep_patterns': 0.5, 'work_life_balance': 0.5
                }
            ),
            "印度": CulturalPreference(
                cuisine_type=CuisineType.INDIAN,
                religion=ReligionType.HINDU,
                spice_tolerance=0.9,
                salt_preference=0.5,
                sweet_preference=0.7,
                oil_preference=0.6,
                dietary_restrictions=set(),
                regional_ingredients=["咖喱", "香料", "豆类", "米饭", "酸奶", "酥油", "姜黄", "孜然"],
                traditional_medicine=True,
                family_meal_pattern="communal",
                festival_foods=["咖喱", "香料", "素食", "甜点", "奶茶", "薄饼", "酸奶"],
                health_beliefs=["阿育吠陀", "素食主义", "自然疗法", "香料养生"],
                personal_features={
                    'age_group': 0.5, 'gender': 0.5, 'education_level': 0.6,
                    'income_level': 0.5, 'activity_level': 0.7, 'health_condition': 0.5,
                    'lifestyle_factors': 0.6, 'psychological_factors': 0.7,
                    'social_support': 0.8, 'health_literacy': 0.6,
                    'cultural_identity_score': 0.9, 'acculturation_level': 0.6,
                    'metabolic_rate': 0.7, 'insulin_sensitivity': 0.6,
                    'genetic_background': 0.8, 'socioeconomic_status': 0.5,
                    'occupation_type': 0.6, 'living_environment': 0.6,
                    'dietary_restrictions': 0.8, 'meal_timing': 0.7,
                    'cooking_frequency': 0.9, 'food_shopping_habits': 0.7,
                    'seasonal_adaptation': 0.8, 'festival_participation': 0.9,
                    'traditional_medicine_usage': 0.9, 'health_checkup_frequency': 0.4,
                    'exercise_preferences': 0.7, 'stress_management': 0.8,
                    'sleep_patterns': 0.6, 'work_life_balance': 0.6
                }
            )
        }

        # 将菜系与文化规则编码到三元组
        self._initialize_cultural_kg_triples()
        # 初始化嵌入器（默认TransE，可切换RotatE）
        self._initialize_kg_embedder()

    def encode_cultural_features(self, cultural_pref: CulturalPreference) -> torch.Tensor:
        """编码文化特征 - 完整版，包含详细的文化特征"""
        features = [
            # 基础文化特征 (15维)
            float(cultural_pref.cuisine_type.value == "川菜"),
            float(cultural_pref.cuisine_type.value == "粤菜"),
            float(cultural_pref.cuisine_type.value == "地中海"),
            float(cultural_pref.cuisine_type.value == "日式"),
            float(cultural_pref.cuisine_type.value == "印度"),
            cultural_pref.spice_tolerance,
            cultural_pref.salt_preference,
            cultural_pref.sweet_preference,
            cultural_pref.oil_preference,
            float(cultural_pref.traditional_medicine),
            float(cultural_pref.family_meal_pattern == "communal"),
            float(cultural_pref.family_meal_pattern == "individual"),
            len(cultural_pref.dietary_restrictions) / 10.0,
            len(cultural_pref.regional_ingredients) / 10.0,
            len(cultural_pref.festival_foods) / 10.0,

            # 扩展文化特征 (20维)
            len(cultural_pref.health_beliefs) / 10.0,
            float(cultural_pref.religion == ReligionType.HALAL),
            float(cultural_pref.religion == ReligionType.VEGETARIAN),
            float(cultural_pref.religion == ReligionType.HINDU),
            float(cultural_pref.religion == ReligionType.NONE),
            float('火锅' in cultural_pref.festival_foods),  # 特定食物偏好
            float('海鲜' in cultural_pref.festival_foods),
            float('素食' in cultural_pref.festival_foods),
            float('汤水' in cultural_pref.health_beliefs),
            float('阿育吠陀' in cultural_pref.health_beliefs),
            float('地中海饮食' in cultural_pref.health_beliefs),
            float('清淡饮食' in cultural_pref.health_beliefs),
            float('温热性食物' in cultural_pref.health_beliefs),
            float('橄榄油' in cultural_pref.regional_ingredients),
            float('花椒' in cultural_pref.regional_ingredients),
            float('味噌' in cultural_pref.regional_ingredients),
            float('咖喱' in cultural_pref.regional_ingredients),
            float('生抽' in cultural_pref.regional_ingredients),
            float('鱼类' in cultural_pref.regional_ingredients),
            float('香料' in cultural_pref.regional_ingredients),
            float('豆腐' in cultural_pref.regional_ingredients)
        ]
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    def encode_personal_features(self, cultural_pref: CulturalPreference) -> torch.Tensor:
        """编码个人特征 - 完整版，包含详细的个人特征"""
        personal_features = cultural_pref.personal_features
        features = [
            # 基础个人特征 (12维)
            personal_features.get('age_group', 0.5),
            personal_features.get('gender', 0.5),
            personal_features.get('education_level', 0.5),
            personal_features.get('income_level', 0.5),
            personal_features.get('activity_level', 0.5),
            personal_features.get('health_condition', 0.5),
            personal_features.get('lifestyle_factors', 0.5),
            personal_features.get('psychological_factors', 0.5),
            personal_features.get('social_support', 0.5),
            personal_features.get('health_literacy', 0.5),
            personal_features.get('cultural_identity_score', 0.5),
            personal_features.get('acculturation_level', 0.5),

            # 扩展个人特征 (20维)
            personal_features.get('metabolic_rate', 0.5),
            personal_features.get('insulin_sensitivity', 0.5),
            personal_features.get('genetic_background', 0.5),
            personal_features.get('socioeconomic_status', 0.5),
            personal_features.get('occupation_type', 0.5),
            personal_features.get('living_environment', 0.5),
            personal_features.get('dietary_restrictions', 0.5),
            personal_features.get('meal_timing', 0.5),
            personal_features.get('cooking_frequency', 0.5),
            personal_features.get('food_shopping_habits', 0.5),
            personal_features.get('seasonal_adaptation', 0.5),
            personal_features.get('festival_participation', 0.5),
            personal_features.get('traditional_medicine_usage', 0.5),
            personal_features.get('health_checkup_frequency', 0.5),
            personal_features.get('exercise_preferences', 0.5),
            personal_features.get('stress_management', 0.5),
            personal_features.get('sleep_patterns', 0.5),
            personal_features.get('work_life_balance', 0.5),
            personal_features.get('family_size', 0.5),
            personal_features.get('social_network_size', 0.5)
        ]
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    def adapt_meal_recommendation(self,
                                 food_features: torch.Tensor,
                                 user_region: str,
                                 cooking_methods: List[str] = None,
                                 dietary_restrictions: List[str] = None,
                                 timestamps: torch.Tensor = None,
                                 historical_features: torch.Tensor = None,
                                 previous_weights: torch.Tensor = None) -> Dict:
        """适配膳食推荐 - 增强版，支持动态文化漂移窗口和交叉注意力"""
        cultural_pref = self.cultural_database.get(user_region, self.cultural_database["广东"])
        cultural_features = self.encode_cultural_features(cultural_pref)
        personal_features = self.encode_personal_features(cultural_pref)

        cultural_features = cultural_features.to(self.device)
        personal_features = personal_features.to(self.device)
        food_features = food_features.to(self.device)

        # 基础文化适配 - 支持动态文化漂移窗口和交叉注意力
        adaptation_results = self.cultural_module(
            food_features,
            cultural_features,
            personal_features,
            timestamps=timestamps,
            historical_features=historical_features,
            previous_weights=previous_weights
        )

        # 烹饪方式惩罚
        cooking_penalties = torch.zeros(food_features.size(0), device=self.device)
        if cooking_methods:
            cooking_penalties = self.cooking_penalty(food_features, cooking_methods)

        # 可训练文化约束权重
        culture_weights = self.culture_sigma(cultural_features)

        # 应用惩罚和权重
        penalized_features = adaptation_results['adapted_features'] * (1 - cooking_penalties.unsqueeze(-1))
        # 确保权重维度匹配
        if culture_weights.size(0) != penalized_features.size(0):
            # 如果权重数量不匹配，使用平均权重
            avg_weight = torch.mean(culture_weights)
            weighted_features = penalized_features * avg_weight
        else:
            weighted_features = penalized_features * culture_weights.unsqueeze(-1)

        # 禁忌过滤
        filtered_foods = []
        if dietary_restrictions:
            # 模拟食物列表
            food_list = ['宫保鸡丁', '白切鸡', '蒸蛋', '烤羊肉', '清蒸鱼', '红烧肉']
            filtered_foods = self.taboo_filter.filter_foods(food_list, dietary_restrictions)

        # 构建增强的返回结果
        result = {
            **adaptation_results,
            'adapted_features': weighted_features,
            'cooking_penalties': cooking_penalties,
            'culture_weights': culture_weights,
            'filtered_foods': filtered_foods,
            'cultural_preference': cultural_pref,
            'region': user_region,
            'temporal_smooth_loss': adaptation_results.get('temporal_smooth_loss', torch.tensor(0.0)),
            'cultural_info': {
                'cuisine_type': cultural_pref.cuisine_type.value,
                'spice_tolerance': cultural_pref.spice_tolerance,
                'salt_preference': cultural_pref.salt_preference,
                'oil_preference': cultural_pref.oil_preference,
                'traditional_medicine': cultural_pref.traditional_medicine,
                'family_meal_pattern': cultural_pref.family_meal_pattern,
                'festival_foods': cultural_pref.festival_foods,
                'health_beliefs': cultural_pref.health_beliefs
            },
            'personal_info': cultural_pref.personal_features,
            'enhancement_metrics': {
                'cooking_penalty_applied': len(cooking_methods) if cooking_methods else 0,
                'taboo_filtered_count': len(filtered_foods),
                'culture_weight_variance': torch.var(culture_weights).item(),
                'total_enhancement_score': torch.mean(weighted_features).item(),
                'temporal_smooth_loss': adaptation_results.get('temporal_smooth_loss', torch.tensor(0.0)).item()
            }
        }

        # 添加交叉注意力相关信息
        if 'cross_attention_output' in adaptation_results:
            result.update({
                'cross_attention_output': adaptation_results['cross_attention_output'],
                'attention_weights': adaptation_results['attention_weights'],
                'cultural_projected': adaptation_results['cultural_projected'],
                'personal_projected': adaptation_results['personal_projected']
            })
            result['enhancement_metrics']['cross_attention_score'] = torch.mean(
                adaptation_results['cross_attention_output']
            ).item()

        # 添加时间漂移相关信息
        if 'temporal_cultural_features' in adaptation_results:
            result.update({
                'temporal_cultural_features': adaptation_results['temporal_cultural_features'],
                'time_encoded': adaptation_results['time_encoded'],
                'window_aggregated': adaptation_results['window_aggregated']
            })
            result['enhancement_metrics']['temporal_drift_score'] = torch.mean(
                adaptation_results['temporal_cultural_features']
            ).item()

        return result

    # ===== 外部知识注入（MCP/全网搜索挂钩） =====
    def upsert_external_knowledge(self, entity: str, relation: str, target: str) -> bool:
        """将外部获取的知识三元组写入KG并更新TransE空间（轻量增量）。"""
        try:
            # 写入三元组
            self.triples.add_triple(entity, relation, target)
            # 若新加入实体/关系导致尺寸变化，需要重新初始化嵌入（简化处理）
            self._initialize_kg_embedder()
            return True
        except Exception:
            return False

    def batch_upsert_external_knowledge(self, triples: List[Tuple[str, str, str]]) -> int:
        """批量写入外部三元组。返回成功写入数量。"""
        success = 0
        for h, r, t in triples:
            if self.upsert_external_knowledge(h, r, t):
                success += 1
        return success

    # ===== 知识图谱（三元组/TransE） =====
    def _initialize_cultural_kg_triples(self) -> None:
        """基于内置文化库与禁忌，构造“食物-营养-文化/禁忌”三元组"""
        # 基础关系词表
        REL_USES = "uses_ingredient"
        REL_BELONGS = "belongs_to_cuisine"
        REL_FORBIDDEN = "forbidden_for"
        REL_PREFERENCE = "has_preference"

        # 1) 将区域->菜系 加入图谱
        for region, pref in self.cultural_database.items():
            cuisine = pref.cuisine_type.value
            self.triples.add_triple(region, REL_BELONGS, cuisine)
            # 偏好（辣/盐/甜/油）
            if pref.spice_tolerance > 0.7:
                self.triples.add_triple(region, REL_PREFERENCE, "spicy")
            if pref.salt_preference > 0.6:
                self.triples.add_triple(region, REL_PREFERENCE, "salty")
            if pref.sweet_preference > 0.6:
                self.triples.add_triple(region, REL_PREFERENCE, "sweet")
            if pref.oil_preference > 0.6:
                self.triples.add_triple(region, REL_PREFERENCE, "oily")
            # 特色食材/节庆食物
            for ingr in pref.regional_ingredients[:10]:
                self.triples.add_triple(cuisine, REL_USES, ingr)
            for fv in pref.festival_foods[:10]:
                self.triples.add_triple(cuisine, REL_USES, fv)

        # 2) 将禁忌规则加入图谱
        for rule_name, rule in self.taboo_filter.taboo_rules.items():
            for forb in rule.get('forbidden', []):
                self.triples.add_triple(forb, REL_FORBIDDEN, rule_name)

    def _initialize_kg_embedder(self) -> None:
        if self._embedder_name == 'rotate':
            self.rotate = RotatEEmbedder(
                num_entities=len(self.triples.entity_list),
                num_relations=len(self.triples.relation_list),
                emb_dim=128,
                margin=6.0
            ).to(self.device)
            self.transe = None
        else:
            self.transe = TransEEmbedder(
                num_entities=len(self.triples.entity_list),
                num_relations=len(self.triples.relation_list),
                emb_dim=128,
                margin=1.0,
                p=1
            ).to(self.device)
            self.rotate = None

    def switch_embedder(self, name: str = 'rotate') -> None:
        """切换嵌入器类型：'transe' 或 'rotate'，保持接口兼容"""
        assert name in ('transe', 'rotate')
        self._embedder_name = name
        self._initialize_kg_embedder()

    def get_transe_score(self, h: str, r: str, t: str) -> float:
        """查询(h, r, t)的打分（越小越好）；自动选择当前嵌入器"""
        try:
            h_idx = torch.tensor([self.triples.entities[h]], device=self.device)
            r_idx = torch.tensor([self.triples.relations[r]], device=self.device)
            t_idx = torch.tensor([self.triples.entities[t]], device=self.device)
        except KeyError:
            return float('inf')
        with torch.no_grad():
            if self._embedder_name == 'rotate' and self.rotate is not None:
                score = self.rotate.score(h_idx, r_idx, t_idx).item()
            elif self.transe is not None:
                score = self.transe.score(h_idx, r_idx, t_idx).item()
            else:
                score = float('inf')
        return float(score)

    def cultural_compatibility(self, region: str, food_or_ingredient: str) -> float:
        """基于TransE与三元组估计文化适配度（0-1，越高越适配）"""
        # 结果缓存，避免重复计算
        if not hasattr(self, '_compat_cache'):
            self._compat_cache = {}
        key = (region, food_or_ingredient)
        if key in self._compat_cache:
            return self._compat_cache[key]
        if self.transe is None:
            self._compat_cache[key] = 0.5
            return 0.5
        rel = "belongs_to_cuisine"
        # 估计item归属最相关菜系的距离，简化为区域→菜系→食材/食物两跳的一致性
        # 使用现有关系：region belongs_to_cuisine cuisine；cuisine uses_ingredient item
        best = 10.0
        for (h, r, t) in self.triples.triples:
            if h == region and r == rel:
                cuisine = t
                # cuisine uses_ingredient item
                s1 = self.get_transe_score(region, rel, cuisine)
                s2 = self.get_transe_score(cuisine, "uses_ingredient", food_or_ingredient)
                best = min(best, s1 + s2)
        if best == 10.0:
            self._compat_cache[key] = 0.5
            return 0.5
        # 距离到分数的映射（经验缩放）
        compat = max(0.0, min(1.0, 1.0 - best / 10.0))
        self._compat_cache[key] = compat
        return compat

    def batch_compatibility(self, region: str, items: List[str]) -> List[Tuple[str, float]]:
        """批量计算文化适配度，返回[(item, score)]"""
        results: List[Tuple[str, float]] = []
        for it in items:
            results.append((it, self.cultural_compatibility(region, it)))
        return results

def create_cultural_adaptation_service() -> CulturalAdaptationService:
    """创建文化适配服务"""
    return CulturalAdaptationService()

if __name__ == "__main__":
    service = create_cultural_adaptation_service()

    # 测试
    food_features = torch.randn(1, 512)
    result = service.adapt_meal_recommendation(food_features, "四川")

    print("文化适配结果:")
    print(f"适配特征形状: {result['adapted_features'].shape}")
    print(f"用户地区: {result['region']}")
    print("饮食文化适配系统创建成功！")