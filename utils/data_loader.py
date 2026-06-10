#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage2 数据加载器
衔接Stage1特征,构建多模态推荐数据集
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging
import sys

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

logger = logging.getLogger(__name__)


class Stage2RecommendationDataset(Dataset):
    """
    Stage2推荐数据集

    数据格式:
    - user_profile: 用户档案 (从Stage1继承)
    - cultural_features: 文化特征 (8维)
    - item_features: 食物多模态特征
        - image_features: 图像特征 (ResNet提取)
        - text_features: 文本特征 (BERT提取)
        - nutrition_profile: 营养档案 (10维)
    - labels: 用户-食物交互标签
    - stage1_features: Stage1输出特征 (可选)
    """

    def __init__(
        self,
        data_path: Path,
        stage1_model_path: Optional[Path] = None,
        use_stage1_features: bool = True,
        max_items_per_user: int = 100,
        negative_sampling_ratio: float = 8.0,  # 从4.0提升到8.0
        hard_negative_ratio: float = 0.5,  # 新增：困难负样本比例
        use_hard_negatives: bool = True,     # 新增：是否使用困难负样本
        cultural_sampler: Optional[Any] = None,  # CulturalNegativeSampler实例
        stage1_device: Optional[str] = None,
        max_samples: Optional[int] = None,  # 新增：限制样本数量（用于快速测试）
        user_profile_dim: int = 128,  # 新增：期望的user_profile维度
        sequence_mode: bool = False,  # 新增：是否为序列模式（BERT4Rec）
        max_seq_len: int = 50,  # 新增：最大序列长度
    ):
        """
        Args:
            data_path: 数据文件路径 (JSON格式)
            stage1_model_path: Stage1模型路径 (用于特征提取)
            use_stage1_features: 是否使用Stage1特征
            max_items_per_user: 每个用户的最大候选食物数
            negative_sampling_ratio: 负采样比例 (从4提升到8)
            hard_negative_ratio: 困难负样本比例 (0-1之间，0.5表示50%困难+50%随机)
            use_hard_negatives: 是否启用hard negative mining
            cultural_sampler: CulturalNegativeSampler实例 (如果启用)
            max_samples: 限制样本数量（用于快速测试，None表示不限制）
            user_profile_dim: 期望的user_profile维度（用于维度适配）
        """
        self.data_path = data_path
        self.use_stage1_features = use_stage1_features
        self.max_items_per_user = max_items_per_user
        self.user_profile_dim = user_profile_dim
        self.negative_sampling_ratio = negative_sampling_ratio
        self.hard_negative_ratio = hard_negative_ratio
        self.use_hard_negatives = use_hard_negatives
        self.cultural_sampler = cultural_sampler
        self.stage1_device = self._resolve_stage1_device(stage1_device)
        # 保存max_samples以便在_load_data中使用
        self._max_samples_for_loading = max_samples
        self._max_stage1_repeat = 3
        self.stage1_input_dim: Optional[int] = None
        # 序列模式设置（用于BERT4Rec）
        self.sequence_mode = sequence_mode
        self.max_seq_len = max_seq_len
        self._stage1_pad_logged = False
        self._stage1_trim_logged = False
        self._stage1_cache_hits = 0
        self._stage1_cache_misses = 0
        self._stage1_cache_flush_interval = 128
        self._stage1_cache_dirty = False
        self._stage1_cache_updates = 0
        self._stage1_cache_dir = (project_root / 'stage2' / 'results' / 'cache')
        self.stage1_cache_path = self._stage1_cache_dir / 'stage1_features.pt'
        self.stage1_feature_cache: Optional[Dict[str, torch.Tensor]] = None
        if use_stage1_features:
            self._stage1_cache_dir.mkdir(parents=True, exist_ok=True)
            self.stage1_feature_cache = self._load_stage1_feature_cache()

        # Stage1桥接数据 (用于难例重采样)
        self.stage1_bridge_data = self._load_stage1_bridge_data()
        self.stage1_cluster_weights = self._extract_stage1_cluster_weights()

        # 加载数据（_load_data内部已经处理了max_samples限制）
        logger.info(f"Loading data from {self.data_path}...")
        self.data = self._load_data()
        logger.info(f"Data loaded: {len(self.data)} samples")

        logger.info("Building sampling index pool...")
        self.sample_index_pool = self._build_sampling_index_pool()
        logger.info(f"Sampling index pool built: {len(self.sample_index_pool)} indices")
        if len(self.sample_index_pool) > len(self.data) and len(self.data) > 0:
            logger.info(
                "Stage1 cluster resampling applied: %d → %d effective samples (x%.2f)",
                len(self.data),
                len(self.sample_index_pool),
                len(self.sample_index_pool) / len(self.data)
            )
        elif len(self.sample_index_pool) == 0:
            logger.warning("Stage2 dataset is empty after Stage1 resampling logic.")

        # 加载Stage1模型 (如果需要)
        self.stage1_model = None
        if use_stage1_features and stage1_model_path is not None:
            logger.info(f"Stage1 features enabled. Loading model from {stage1_model_path}...")
            try:
                self.stage1_model, self.stage1_input_dim = self._load_stage1_model(stage1_model_path)
                logger.info("Stage1 model loaded successfully.")
            except FileNotFoundError:
                logger.warning("Stage1 model not found at %s. Stage1 features disabled.", stage1_model_path)
                self.use_stage1_features = False
            except Exception as exc:
                logger.warning("Failed to load Stage1 model (%s): %s. Stage1 features disabled.", stage1_model_path, exc)
                self.use_stage1_features = False
        else:
            logger.info("Stage1 features disabled (use_stage1_features=%s, stage1_model_path=%s)", use_stage1_features, stage1_model_path)

        # 当前训练轮次（用于课程学习）
        self._current_epoch = 0

        # 明确说明设备信息
        device_info = f"stage1_device={self.stage1_device.type}"
        if not use_stage1_features:
            device_info += " (Stage1 features disabled, this does not affect training device)"

        logger.info(
            f"Stage2RecommendationDataset initialized: "
            f"{len(self.data)} samples, "
            f"use_stage1_features={use_stage1_features}, "
            f"cultural_sampler={'enabled' if cultural_sampler is not None else 'disabled'}, "
            f"{device_info}"
        )

    def set_epoch(self, epoch: int):
        """
        设置当前训练轮次（用于课程学习）

        Args:
            epoch: 当前训练轮次
        """
        self._current_epoch = epoch

    def _load_data(self) -> List[Dict[str, Any]]:
        """加载数据文件"""
        # Check if this is a batch directory (has batch_index.json)
        if self.data_path.is_dir():
            batch_index_path = self.data_path / "batch_index.json"
            if batch_index_path.exists():
                logger.info(f"Detected batch directory: {self.data_path}")
                with open(batch_index_path, 'r', encoding='utf-8') as f:
                    batch_index = json.load(f)
                batch_files = [self.data_path / bf for bf in batch_index['batch_files']]
                # If running under DDP, only load the batch files assigned to this rank.
                # This avoids each process loading all shards into memory (which causes OOM).
                try:
                    import os
                    rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', 0)))
                    world_size = int(os.environ.get('WORLD_SIZE', 1))
                except Exception:
                    rank, world_size = 0, 1
                if world_size > 1:
                    original_count = len(batch_files)
                    batch_files = [bf for i, bf in enumerate(batch_files) if (i - rank) % world_size == 0]
                    logger.info(f"Distributed mode detected (rank={rank}, world_size={world_size}). "
                                f"Loading {len(batch_files)}/{original_count} batch files for this rank.")

                # For training, we only need to return a placeholder since batch training is handled elsewhere
                # For validation/testing, we should load all batches
                max_samples = getattr(self, '_max_samples_for_loading', None)
                if max_samples is not None and max_samples > 0:
                    # Load subset from assigned batch files with streaming to avoid loading entire files
                    logger.info(f"Loading subset from batch directory with max_samples={max_samples} (streaming mode)")
                    data = []
                    for batch_file in batch_files:
                        if batch_file.exists():
                            # Stream load only the required number of samples from this file
                            remaining = max_samples - len(data)
                            if remaining <= 0:
                                break
                            batch_data = self._load_single_file_streaming(batch_file, remaining)
                            data.extend(batch_data)
                    data = data[:max_samples]
                    logger.info(f"Loaded {len(data)} samples from batch directory (streaming)")
                else:
                    # For full loading, concatenate all batches
                    logger.info("Loading full batch directory...")
                    data = []
                    for batch_file in batch_files:
                        if batch_file.exists():
                            batch_data = self._load_single_file(batch_file)
                            data.extend(batch_data)
                    logger.info(f"Loaded {len(data)} samples from all batches")
                return data
            else:
                raise ValueError(f"Directory {self.data_path} exists but no batch_index.json found")

        if self.data_path.suffix == '.json':
            # 对于大文件，尝试流式读取（如果设置了max_samples）
            max_samples = getattr(self, '_max_samples_for_loading', None)
            if max_samples is not None and max_samples > 0:
                logger.info(f"Loading data with max_samples limit: {max_samples}...")
                data = []
                try:
                    # 尝试使用ijson进行流式解析
                    import ijson
                    logger.info("Using ijson for streaming JSON parsing...")
                    with open(self.data_path, 'rb') as f:
                        parser = ijson.items(f, 'item')
                        for i, item in enumerate(parser):
                            if i >= max_samples:
                                break
                            data.append(item)
                    logger.info(f"Loaded {len(data)} samples using ijson streaming parser")
                except ImportError:
                    # ijson不可用，回退到完整加载（会很慢）
                    logger.warning("ijson not available. Loading full file (this will be slow for large files)...")
                    logger.warning("Consider installing ijson: pip install ijson")
                    with open(self.data_path, 'r', encoding='utf-8') as f:
                        full_data = json.load(f)
                        data = full_data[:max_samples]
                        logger.info(f"Loaded {len(data)} samples from {len(full_data)} total samples")
                except Exception as e:
                    # 如果ijson解析失败，回退到完整加载
                    logger.warning(f"ijson parsing failed ({e}), falling back to full JSON load...")
                    with open(self.data_path, 'r', encoding='utf-8') as f:
                        full_data = json.load(f)
                        data = full_data[:max_samples]
                        logger.info(f"Loaded {len(data)} samples from {len(full_data)} total samples")
            else:
                # 没有限制，尝试使用流式解析（对于大文件更高效）
                logger.info("Loading full data file (using streaming parser for large files)...")
                data = []
                try:
                    # 尝试使用ijson进行流式解析（即使没有max_samples限制）
                    import ijson
                    logger.info("Using ijson for streaming JSON parsing...")
                    with open(self.data_path, 'rb') as f:
                        parser = ijson.items(f, 'item')
                        for item in parser:
                            data.append(item)
                    logger.info(f"Loaded {len(data)} samples using ijson streaming parser")
                except ImportError:
                    # ijson不可用，回退到完整加载
                    logger.warning("ijson not available. Loading full file (this will be slow for large files)...")
                    logger.warning("Consider installing ijson: pip install ijson")
                    with open(self.data_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    logger.info(f"Loaded {len(data)} samples")
                except Exception as e:
                    # 如果ijson解析失败，尝试作为JSONL格式处理
                    logger.warning(f"ijson parsing failed ({e}), trying JSONL format...")
                    data = []
                    try:
                        with open(self.data_path, 'r', encoding='utf-8') as f:
                            for line_num, line in enumerate(f, 1):
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    item = json.loads(line)
                                    data.append(item)
                                except json.JSONDecodeError as e:
                                    logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                                    continue
                        logger.info(f"Loaded {len(data)} samples from JSONL file")
                    except Exception as e2:
                        # 如果JSONL也失败，回退到完整JSON加载
                        logger.warning(f"JSONL parsing also failed ({e2}), falling back to full JSON load...")
                        with open(self.data_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        logger.info(f"Loaded {len(data)} samples")
        elif self.data_path.suffix == '.csv':
            df = pd.read_csv(self.data_path)
            data = df.to_dict('records')
        else:
            raise ValueError(f"Unsupported data format: {self.data_path.suffix}")

        return data

    def _load_single_file_streaming(self, file_path: Path, max_samples: int) -> List[Dict[str, Any]]:
        """流式加载单个数据文件，只读取所需数量的样本"""
        data = []
        try:
            # Simple approach: read the file content and find complete JSON objects
            logger.info(f"Stream loading up to {max_samples} samples from {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Find complete JSON objects by counting braces
            objects = []
            start_pos = 1  # Skip the opening '['
            brace_count = 0
            in_string = False
            string_char = None

            i = 1
            while i < len(content) - 1:  # Skip closing ']'
                char = content[i]

                # Handle strings
                if not in_string and (char == '"' or char == "'"):
                    in_string = True
                    string_char = char
                elif in_string and char == string_char and content[i-1] != '\\':
                    in_string = False
                    string_char = None
                elif not in_string:
                    if char == '{':
                        if brace_count == 0:
                            start_pos = i
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            # Found a complete object
                            obj_str = content[start_pos:i+1]
                            objects.append(obj_str)
                            if len(objects) >= max_samples:
                                break

                i += 1

            # Parse the objects
            for obj_str in objects:
                try:
                    item = json.loads(obj_str)
                    data.append(item)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON object: {e}")
                    continue

            logger.info(f"Stream loaded {len(data)} samples from {file_path}")
            return data
        except Exception as e:
            # Fallback to regular loading with limit
            logger.warning(f"Stream loading failed ({e}), falling back to limited regular load")
            return self._load_single_file(file_path, max_samples)

    def _load_single_file(self, file_path: Path, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """加载单个数据文件"""
        if file_path.suffix in ['.json', '']:
            # Try JSON array first, then JSONL
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    full_data = json.load(f)
                    if isinstance(full_data, list):
                        data = full_data[:max_samples] if max_samples else full_data
                        logger.info(f"Loaded {len(data)} samples from JSON array {file_path}")
                        return data
                    else:
                        return [full_data]
            except json.JSONDecodeError:
                # Try JSONL format
                data = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                            data.append(item)
                            if max_samples and len(data) >= max_samples:
                                break
                        except json.JSONDecodeError as e:
                            logger.warning(f"Skipping invalid JSON at line {line_num} in {file_path}: {e}")
                            continue
                logger.info(f"Loaded {len(data)} samples from JSONL file {file_path}")
                return data
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def _load_stage1_bridge_data(self) -> Optional[Dict[str, Any]]:
        """加载Stage1→Stage2桥接数据"""
        bridge_path = Path('stage2/configs/stage1_bridge_data.json')
        if not bridge_path.exists():
            return None
        try:
            with bridge_path.open('r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("Failed to load Stage1 bridge data (%s): %s", bridge_path, exc)
            return None

    def _extract_stage1_cluster_weights(self) -> Dict[str, float]:
        """提取Stage1失败聚类的权重"""
        if not self.stage1_bridge_data:
            return {}

        failure_patterns = self.stage1_bridge_data.get('failure_patterns', {})
        cluster_patterns = failure_patterns.get('cluster_patterns', {})
        target_types = {'high_protein_high_calorie', 'high_calorie_high_carb'}

        weights = {}
        total = 0.0
        for pattern in cluster_patterns.values():
            cluster_type = pattern.get('type')
            if cluster_type not in target_types:
                continue
            count = max(float(pattern.get('count', 0) or 0.0), 0.0)
            if count <= 0:
                continue
            weights[cluster_type] = weights.get(cluster_type, 0.0) + count
            total += count

        if total == 0:
            return {}

        return {k: v / total for k, v in weights.items()}

    def _build_sampling_index_pool(self) -> List[int]:
        """根据Stage1聚类权重构建采样索引池"""
        base_indices = list(range(len(self.data)))
        if not base_indices:
            return []
        if not self.stage1_cluster_weights:
            return base_indices

        index_pool: List[int] = []
        for idx, sample in enumerate(self.data):
            cluster_counts = self._count_stage1_clusters_in_sample(sample)
            stage1_score = 0.0
            for cluster_type, count in cluster_counts.items():
                if count <= 0:
                    continue
                weight = self.stage1_cluster_weights.get(cluster_type, 0.0)
                if weight == 0:
                    continue
                normalized_density = min(
                    count / max(1, self.max_items_per_user),
                    1.0
                )
                stage1_score += weight * normalized_density

            repeat_factor = self._determine_repeat_factor(stage1_score)
            index_pool.extend([idx] * repeat_factor)

        return index_pool or base_indices

    def _determine_repeat_factor(self, stage1_score: float) -> int:
        """根据Stage1难例得分决定重采样次数"""
        if stage1_score >= 0.8:
            return self._max_stage1_repeat
        if stage1_score >= 0.4:
            return min(self._max_stage1_repeat, 2)
        return 1

    def _count_stage1_clusters_in_sample(self, sample: Dict[str, Any]) -> Dict[str, int]:
        """统计样本中命中Stage1重点聚类的数量"""
        counts = {
            'high_protein_high_calorie': 0,
            'high_calorie_high_carb': 0
        }
        for food_item in sample.get('candidate_items', []):
            cluster_type = self._detect_stage1_cluster(food_item)
            if cluster_type in counts:
                counts[cluster_type] += 1
        return counts

    def _detect_stage1_cluster(self, food_item: Dict[str, Any]) -> Optional[str]:
        """根据营养特征判断所属的Stage1聚类"""
        calories = float(food_item.get('calories') or food_item.get('energy', 0.0) or 0.0)
        protein = float(food_item.get('protein') or food_item.get('proteins', 0.0) or 0.0)
        carbs = float(food_item.get('carbohydrate') or food_item.get('carbs', 0.0) or 0.0)
        fat = float(food_item.get('fat', 0.0) or 0.0)
        sodium = float(food_item.get('sodium', 0.0) or 0.0)

        if (
            protein >= 20 and protein <= 32 and
            calories >= 250 and calories <= 500 and
            600 <= sodium <= 1000
        ):
            return 'high_protein_high_calorie'

        if (
            carbs >= 25 and carbs <= 75 and
            calories >= 200 and calories <= 550 and
            fat >= 10 and fat <= 35
        ):
            return 'high_calorie_high_carb'

        return None

    def _load_stage1_model(self, model_path: Path) -> Tuple[torch.nn.Module, int]:
        """加载Stage1模型用于特征提取"""
        logger.info(f"Loading Stage1 model from {model_path}...")
        from system.projects.nutrition.src.cultural_finetune import CulturalStage1Trainer

        logger.info(f"Loading checkpoint from {model_path}...")
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        logger.info("Checkpoint loaded. Creating Stage1 trainer...")
        config = checkpoint.get('config', {}) if isinstance(checkpoint, dict) else {}

        trainer = CulturalStage1Trainer(
            input_dim=config.get('input_dim', 256),
            hidden_dim=config.get('hidden_dim', 128),
            num_experts=config.get('num_experts', 4),
            expert_dim=config.get('expert_dim', 64),
            lora_rank=config.get('lora_rank', 16),
            lora_alpha=config.get('lora_alpha', 32.0),
            device=self.stage1_device.type
        )

        try:
            trainer.load_lora(model_path)
        except Exception as exc:
            logger.warning("Stage1 LoRA weights failed to load from %s: %s", model_path, exc)

        trainer.model.eval()
        trainer.model.to(self.stage1_device)
        input_dim = config.get('input_dim', getattr(trainer.model, 'input_dim', 256))
        logger.info(
            "Stage1 model loaded from %s (device=%s, input_dim=%d)",
            model_path,
            self.stage1_device.type,
            input_dim
        )

        return trainer.model, input_dim

    def _resolve_stage1_device(self, stage1_device: Optional[str]) -> torch.device:
        """解析Stage1特征推理所使用的设备"""
        requested = (stage1_device or 'cpu').lower()
        if requested == 'auto':
            requested = 'cuda' if torch.cuda.is_available() else 'cpu'
        if requested.startswith('cuda') and not torch.cuda.is_available():
            logger.warning("CUDA requested for Stage1 features but unavailable; falling back to CPU.")
            requested = 'cpu'
        try:
            device = torch.device(requested)
        except (TypeError, ValueError):
            logger.warning("Invalid stage1_device '%s'; defaulting to CPU.", requested)
            device = torch.device('cpu')
        return device

    def _prepare_stage1_input(self, user_profile: torch.Tensor) -> torch.Tensor:
        """调整user_profile维度以匹配Stage1模型输入"""
        if self.stage1_input_dim is None:
            return user_profile

        current_dim = user_profile.shape[-1]
        if current_dim == self.stage1_input_dim:
            return user_profile

        if current_dim < self.stage1_input_dim:
            if not self._stage1_pad_logged:
                logger.warning(
                    "User profile dim (%d) < Stage1 input dim (%d); applying zero-padding bridge.",
                    current_dim,
                    self.stage1_input_dim
                )
                self._stage1_pad_logged = True
            pad_width = self.stage1_input_dim - current_dim
            return F.pad(user_profile, (0, pad_width), value=0.0)

        if not self._stage1_trim_logged:
            logger.warning(
                "User profile dim (%d) > Stage1 input dim (%d); truncating extra features.",
                current_dim,
                self.stage1_input_dim
            )
            self._stage1_trim_logged = True
        return user_profile[..., :self.stage1_input_dim]

    def _load_stage1_feature_cache(self) -> Dict[str, torch.Tensor]:
        """加载缓存的Stage1特征,减少重复推理"""
        if not self.stage1_cache_path.exists():
            return {}
        try:
            payload = torch.load(self.stage1_cache_path, map_location='cpu', weights_only=False)
            if isinstance(payload, dict):
                cache = {}
                for key, value in payload.items():
                    if isinstance(value, torch.Tensor):
                        cache[str(key)] = value.detach().cpu()
                    else:
                        cache[str(key)] = torch.tensor(value, dtype=torch.float32)
                logger.info(
                    "Stage1 feature cache loaded: %d entries from %s",
                    len(cache),
                    self.stage1_cache_path
                )
                return cache
        except Exception as exc:
            logger.warning(
                "Failed to load Stage1 feature cache (%s): %s",
                self.stage1_cache_path,
                exc
            )
        return {}

    def _fetch_stage1_feature(self, cache_key: Optional[str]) -> Optional[torch.Tensor]:
        """读取缓存的Stage1特征"""
        if not cache_key or not self.stage1_feature_cache:
            return None
        cached = self.stage1_feature_cache.get(cache_key)
        if cached is None:
            self._stage1_cache_misses += 1
            return None
        self._stage1_cache_hits += 1
        return cached.clone()

    def _update_stage1_cache(self, cache_key: Optional[str], features: torch.Tensor):
        """写入Stage1特征缓存并定期落盘"""
        if not cache_key or self.stage1_feature_cache is None:
            return
        self.stage1_feature_cache[cache_key] = features.detach().cpu()
        self._stage1_cache_dirty = True
        self._stage1_cache_updates += 1
        if self._stage1_cache_updates % self._stage1_cache_flush_interval == 0:
            self._flush_stage1_cache()

    def _flush_stage1_cache(self, force: bool = False):
        """将Stage1特征缓存保存到磁盘"""
        if self.stage1_feature_cache is None:
            return
        if not self._stage1_cache_dirty and not force:
            return
        try:
            torch.save(self.stage1_feature_cache, self.stage1_cache_path)
            self._stage1_cache_dirty = False
            if force:
                logger.info(
                    "Stage1 feature cache flushed: hits=%d, misses=%d, path=%s",
                    self._stage1_cache_hits,
                    self._stage1_cache_misses,
                    self.stage1_cache_path
                )
        except Exception as exc:
            logger.warning("Failed to flush Stage1 feature cache: %s", exc)

    def _extract_cultural_features(self, user_data: Dict[str, Any]) -> np.ndarray:
        """
        提取文化特征 (8维)

        特征维度:
        0. 地域编码 (0-1)
        1. 语言偏好 (0-1)
        2. 饮食习惯 (0-1)
        3. 宗教信仰 (0-1)
        4. 健康状态 (0-1)
        5. 年龄归一化 (0-1)
        6. BMI归一化 (0-1)
        7. 糖尿病类型 (0-1)
        """
        cultural_features = np.zeros(8, dtype=np.float32)

        # 地域编码 (中国5大区域)
        region_map = {'华北': 0.2, '东北': 0.4, '华东': 0.6, '华南': 0.8, '西部': 1.0}
        cultural_features[0] = region_map.get(user_data.get('region', '华东'), 0.6)

        # 语言偏好 (方言影响)
        language_map = {'普通话': 0.5, '粤语': 0.7, '闽南语': 0.9, '其他': 0.3}
        cultural_features[1] = language_map.get(user_data.get('language', '普通话'), 0.5)

        # 饮食习惯 (甜/咸/辣/清淡)
        taste_map = {'清淡': 0.2, '偏咸': 0.5, '偏甜': 0.7, '偏辣': 0.9}
        cultural_features[2] = taste_map.get(user_data.get('taste_preference', '清淡'), 0.2)

        # 宗教信仰 (影响禁忌食物)
        religion_map = {'无': 0.0, '佛教': 0.3, '伊斯兰': 0.7, '基督教': 0.5}
        cultural_features[3] = religion_map.get(user_data.get('religion', '无'), 0.0)

        # 健康状态 (糖尿病严重程度)（确保是数值类型）
        hba1c = user_data.get('hba1c', 7.0)
        try:
            hba1c = float(hba1c) if hba1c is not None else 7.0
        except (ValueError, TypeError):
            hba1c = 7.0
        cultural_features[4] = min(max((hba1c - 5.0) / 9.0, 0.0), 1.0)  # 归一化到[0, 1]

        # 年龄归一化（确保是数值类型）
        age = user_data.get('age', 50)
        try:
            age = float(age) if age is not None else 50.0
        except (ValueError, TypeError):
            age = 50.0
        cultural_features[5] = min(max((age - 18) / 72, 0.0), 1.0)  # 18-90岁

        # BMI归一化（确保是数值类型）
        bmi = user_data.get('bmi', 24.0)
        try:
            bmi = float(bmi) if bmi is not None else 24.0
        except (ValueError, TypeError):
            bmi = 24.0
        cultural_features[6] = min(max((bmi - 15) / 25, 0.0), 1.0)  # 15-40

        # 糖尿病类型
        dm_type = user_data.get('diabetes_type', 'T2DM')
        cultural_features[7] = 0.3 if dm_type == 'T1DM' else 0.7

        return cultural_features

    def _extract_nutrition_profile(self, food_data: Dict[str, Any]) -> np.ndarray:
        """
        提取营养档案 (10维)

        维度:
        [carbs, protein, fat, fiber, gi, gl, sodium, potassium, calories, cultural_score]
        """
        nutrition_profile = np.zeros(10, dtype=np.float32)

        nutrition_profile[0] = food_data.get('carbohydrate', 30.0)  # 碳水(g)
        nutrition_profile[1] = food_data.get('protein', 20.0)       # 蛋白质(g)
        nutrition_profile[2] = food_data.get('fat', 15.0)           # 脂肪(g)
        nutrition_profile[3] = food_data.get('fiber', 8.0)          # 膳食纤维(g)
        nutrition_profile[4] = food_data.get('glycemic_index', 50.0) # GI
        nutrition_profile[5] = food_data.get('glycemic_load', 8.0)   # GL
        nutrition_profile[6] = food_data.get('sodium', 500.0)       # 钠(mg)
        nutrition_profile[7] = food_data.get('potassium', 800.0)    # 钾(mg)
        nutrition_profile[8] = food_data.get('calories', 500.0)     # 能量(kcal)
        nutrition_profile[9] = food_data.get('cultural_score', 0.8) # 文化适配度

        return nutrition_profile

    def _extract_user_cultural_group(self, user_data: Dict[str, Any]) -> str:
        """
        从用户数据中提取文化组标签

        Returns:
            cultural_group: 文化组字符串 (例如 'east_asian', 'south_asian' 等)
        """
        # 根据地域信息推断文化组
        region = user_data.get('region', '华东')

        # 地域到文化组的映射
        region_to_cultural = {
            '华北': 'east_asian',
            '东北': 'east_asian',
            '华东': 'east_asian',
            '华南': 'east_asian',
            '西部': 'east_asian'  # 中国西部也属于东亚文化
        }

        cultural_group = region_to_cultural.get(region, 'east_asian')

        # 如果有明确的文化组标签，优先使用
        if 'cultural_group' in user_data:
            cultural_group = user_data['cultural_group']

        return cultural_group

    def _extract_item_cultural_group(self, food_item: Dict[str, Any]) -> str:
        """
        从食物item中提取文化组标签

        Returns:
            cultural_group: 文化组字符串
        """
        # 优先使用item中的文化组标签
        if 'cultural_group' in food_item:
            return food_item['cultural_group']

        # 根据文化适配度推断
        cultural_score = food_item.get('cultural_score', 0.8)

        # 如果有地域信息，使用地域推断
        if 'region' in food_item:
            region = food_item['region']
            region_to_cultural = {
                '华北': 'east_asian',
                '东北': 'east_asian',
                '华东': 'east_asian',
                '华南': 'east_asian',
                '西部': 'east_asian'
            }
            return region_to_cultural.get(region, 'east_asian')

        # 默认返回东亚文化组
        return 'east_asian'

    def _compute_item_similarity(
        self,
        item1: Dict[str, Any],
        item2: Dict[str, Any]
    ) -> float:
        """
        计算两个食物item之间的相似度（用于hard negative mining）

        基于营养特征的余弦相似度
        """
        nutrition1 = self._extract_nutrition_profile(item1)
        nutrition2 = self._extract_nutrition_profile(item2)

        # 归一化
        norm1 = np.linalg.norm(nutrition1)
        norm2 = np.linalg.norm(nutrition2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = np.dot(nutrition1, nutrition2) / (norm1 * norm2)
        return float(similarity)

    def _sample_hard_negatives(
        self,
        positive_items: List[Dict[str, Any]],
        all_items: List[Dict[str, Any]],
        num_hard: int
    ) -> List[Dict[str, Any]]:
        """
        采样困难负样本（与正样本相似但用户未交互的样本）

        Args:
            positive_items: 用户交互的正样本
            all_items: 所有候选items
            num_hard: 需要采样的困难负样本数量

        Returns:
            hard_negatives: 困难负样本列表
        """
        if not positive_items or not all_items:
            return []

        # 计算每个候选item与正样本的平均相似度
        item_similarities = []
        for item in all_items:
            # 跳过正样本
            if item in positive_items:
                continue

            # 计算与所有正样本的平均相似度
            similarities = [
                self._compute_item_similarity(item, pos_item)
                for pos_item in positive_items
            ]
            avg_similarity = np.mean(similarities) if similarities else 0.0
            item_similarities.append((item, avg_similarity))

        if not item_similarities:
            return []

        # 按相似度降序排序，选择最相似的（最困难的）
        item_similarities.sort(key=lambda x: x[1], reverse=True)

        # 采样top-num_hard作为困难负样本
        hard_negatives = [item for item, _ in item_similarities[:num_hard]]

        return hard_negatives

    def __len__(self) -> int:
        return len(self.sample_index_pool)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本

        Returns:
            sample: 包含所有输入特征和标签的字典
        """
        real_idx = self.sample_index_pool[idx]
        item = self.data[real_idx]
        cache_key = str(item.get('user_id')) if item.get('user_id') is not None else None

        # 用户档案 (Stage1格式)
        # 注意：数据中的user_profile可能是64维，但模型期望128维
        raw_user_profile = item.get('user_profile', [])
        if isinstance(raw_user_profile, (list, np.ndarray)):
            user_profile_array = np.array(raw_user_profile, dtype=np.float32)
        else:
            user_profile_array = np.zeros(self.user_profile_dim, dtype=np.float32)

        # 如果维度不匹配，进行填充或截断
        if len(user_profile_array) < self.user_profile_dim:
            # 填充到期望维度
            padded = np.zeros(self.user_profile_dim, dtype=np.float32)
            padded[:len(user_profile_array)] = user_profile_array
            user_profile_array = padded
        elif len(user_profile_array) > self.user_profile_dim:
            # 截断到期望维度
            user_profile_array = user_profile_array[:self.user_profile_dim]

        user_profile = torch.tensor(user_profile_array, dtype=torch.float32)

        # 如果是序列模式，为BERT4Rec生成序列数据
        if self.sequence_mode:
            # 从用户历史中创建序列
            user_history = item.get('interaction_history', [])
            if not user_history:
                # 如果没有历史，创建一个简单的序列
                user_history = [0] * min(10, self.max_seq_len - 1)  # padding with 0s

            # 截断或填充到max_seq_len
            if len(user_history) > self.max_seq_len - 1:
                user_history = user_history[:self.max_seq_len - 1]
            elif len(user_history) < self.max_seq_len - 1:
                user_history.extend([0] * (self.max_seq_len - 1 - len(user_history)))

            # 创建输入序列和标签
            input_sequence = user_history.copy()
            labels = user_history[1:] + [0]  # 下一个item作为标签，最后一个用0填充

            # 转换为tensor
            item_sequences = torch.tensor(input_sequence, dtype=torch.long)
            attention_mask = torch.ones(self.max_seq_len, dtype=torch.long)
            sequence_labels = torch.tensor(labels, dtype=torch.long)

            # 文化特征保持不变
            cultural_features = torch.zeros(8, dtype=torch.float32)  # 简化为8维

            # 为兼容性创建其他tensor
            num_items = 50
            item_image_features = torch.zeros(num_items, 2048, dtype=torch.float32)
            item_text_features = torch.zeros(num_items, 768, dtype=torch.float32)
            nutrition_profile = torch.zeros(num_items, 10, dtype=torch.float32)

            return {
                'user_profile': user_profile,
                'cultural_features': cultural_features,
                'item_sequences': item_sequences.unsqueeze(0),  # [1, seq_len]
                'attention_mask': attention_mask.unsqueeze(0),  # [1, seq_len]
                'labels': sequence_labels.unsqueeze(0),  # [1, seq_len]
                'item_image_features': item_image_features,
                'item_text_features': item_text_features,
                'nutrition_profile': nutrition_profile,
                'num_items': torch.tensor(num_items, dtype=torch.long),
                'metadata': {'sequence_mode': True}
            }

        # 文化特征
        user_data = item.get('user_data', {})
        cultural_features = torch.tensor(
            self._extract_cultural_features(user_data),
            dtype=torch.float32
        )

        # 食物特征 (多模态)
        candidate_items = item.get('candidate_items', [])
        num_items = min(len(candidate_items), self.max_items_per_user)

        # 初始化
        image_features = torch.zeros(num_items, 2048, dtype=torch.float32)
        text_features = torch.zeros(num_items, 768, dtype=torch.float32)
        nutrition_profiles = torch.zeros(num_items, 10, dtype=torch.float32)
        candidate_metadata = []

        # 填充特征
        for i, food_item in enumerate(candidate_items[:num_items]):
            # 图像特征 (假设已预提取)
            if 'image_features' in food_item:
                image_features[i] = torch.tensor(food_item['image_features'], dtype=torch.float32)

            # 文本特征 (假设已预提取)
            if 'text_features' in food_item:
                text_features[i] = torch.tensor(food_item['text_features'], dtype=torch.float32)

            # 营养档案
            nutrition_profiles[i] = torch.tensor(
                self._extract_nutrition_profile(food_item),
                dtype=torch.float32
            )

            # 记录候选元信息用于推理/评估
            candidate_metadata.append({
                'index': i,
                'food_id': food_item.get('food_id', i),
                'name': food_item.get('name') or food_item.get('food_name') or f"food_{food_item.get('food_id', i)}",
                'cultural_group': self._extract_item_cultural_group(food_item)
            })

        # 标签（确保长度与num_items匹配）
        raw_labels = item.get('labels', [])
        if isinstance(raw_labels, (list, np.ndarray)):
            # 截取或填充到num_items长度
            if len(raw_labels) > num_items:
                labels = torch.tensor(raw_labels[:num_items], dtype=torch.float32)
            elif len(raw_labels) < num_items:
                # 如果标签数量少于num_items，用0填充
                padded_labels = np.zeros(num_items, dtype=np.float32)
                padded_labels[:len(raw_labels)] = raw_labels
                labels = torch.tensor(padded_labels, dtype=torch.float32)
            else:
                labels = torch.tensor(raw_labels, dtype=torch.float32)
        else:
            # 如果没有标签，创建全零标签
            labels = torch.zeros(num_items, dtype=torch.float32)

        # 如果使用CulturalNegativeSampler，生成负样本
        if self.cultural_sampler is not None:
            # 提取正样本索引
            positive_indices = torch.where(labels > 0.5)[0].tolist()

            if len(positive_indices) > 0:
                # 提取用户文化组
                user_cultural_group = self._extract_user_cultural_group(user_data)

                # 提取所有items的文化组标签
                item_cultural_groups = [
                    self._extract_item_cultural_group(food_item)
                    for food_item in candidate_items[:num_items]
                ]

                # 为每个正样本生成负样本
                # 注意：采样器期望batch输入，我们为单个样本创建batch_size=1的调用
                all_negative_indices = []
                all_difficulty_labels = []

                for pos_idx in positive_indices:
                    # 准备采样器输入
                    positive_item_tensor = torch.tensor([pos_idx], dtype=torch.long)
                    item_pool = torch.arange(num_items, dtype=torch.long)

                    # 调用采样器 (batch_size=1)
                    negative_items, difficulty_labels = self.cultural_sampler.sample(
                        user_cultural_group=[user_cultural_group],
                        positive_items=positive_item_tensor,
                        item_pool=item_pool,
                        item_cultural_groups=item_cultural_groups,
                        item_nutrition=nutrition_profiles,
                        current_epoch=getattr(self, '_current_epoch', 0)  # 从外部设置epoch
                    )

                    # 采样器返回 [batch, num_negatives]，我们取第一个batch
                    negative_indices = negative_items[0].tolist()  # [num_negatives]
                    difficulty = difficulty_labels[0].tolist()  # [num_negatives]

                    all_negative_indices.extend(negative_indices)
                    all_difficulty_labels.extend(difficulty)

                # 将负样本添加到数据中
                # 注意：这里我们只是记录负样本索引，不改变原始数据结构
                # 训练器可以根据需要使用这些负样本
                negative_sample_info = {
                    'negative_indices': torch.tensor(all_negative_indices, dtype=torch.long) if all_negative_indices else torch.tensor([], dtype=torch.long),
                    'difficulty_labels': torch.tensor(all_difficulty_labels, dtype=torch.long) if all_difficulty_labels else torch.tensor([], dtype=torch.long),
                    'positive_indices': torch.tensor(positive_indices, dtype=torch.long)
                }
            else:
                negative_sample_info = {
                    'negative_indices': torch.tensor([], dtype=torch.long),
                    'difficulty_labels': torch.tensor([], dtype=torch.long),
                    'positive_indices': torch.tensor([], dtype=torch.long)
                }
        else:
            negative_sample_info = None

        # Stage1特征 (如果可用)
        stage1_features = None
        if self.use_stage1_features:
            stage1_features = self._fetch_stage1_feature(cache_key)
        if self.use_stage1_features and stage1_features is None and self.stage1_model is not None:
            with torch.no_grad():
                prepared_input = self._prepare_stage1_input(user_profile)
                if self.stage1_input_dim is not None and prepared_input.shape[-1] != self.stage1_input_dim:
                    raise RuntimeError(
                        f"Stage1 bridge failed: got dim {prepared_input.shape[-1]}, expected {self.stage1_input_dim}"
                    )
                stage1_input = prepared_input.to(self.stage1_device)
                stage1_output = self.stage1_model(stage1_input.unsqueeze(0))
                feature_tensor = (
                    stage1_output['distilled_output'].squeeze(0)
                    if isinstance(stage1_output, dict)
                    else stage1_output.squeeze(0)
                )
                stage1_features = feature_tensor.detach().cpu()
                self._update_stage1_cache(cache_key, stage1_features)

        sample = {
            'user_profile': user_profile,
            'cultural_features': cultural_features,
            'item_image_features': image_features,
            'item_text_features': text_features,
            'nutrition_profile': nutrition_profiles,
            'labels': labels,
            'num_items': torch.tensor(num_items, dtype=torch.long),
            'metadata': {
                'user_id': item.get('user_id'),
                'user_cultural_group': self._extract_user_cultural_group(user_data),
                'candidate_metadata': candidate_metadata
            }
        }

        if stage1_features is not None:
            sample['stage1_features'] = stage1_features

        if negative_sample_info is not None:
            sample['negative_samples'] = negative_sample_info

        return sample

    def __del__(self):
        """析构时确保缓存写回磁盘"""
        try:
            self._flush_stage1_cache(force=True)
        except Exception:
            pass


class Stage2DataLoader:
    """
    Stage2数据加载器封装

    功能:
    - 批量加载训练/验证/测试数据
    - 动态负采样
    - 数据增强
    """

    def __init__(
        self,
        train_data_path: Path,
        val_data_path: Optional[Path] = None,
        test_data_path: Optional[Path] = None,
        stage1_model_path: Optional[Path] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        use_stage1_features: bool = True,
        max_items_per_user: int = 100,
        negative_sampling_ratio: float = 4.0,
        sequence_mode: bool = False,
        max_seq_len: int = 50,
        hard_negative_ratio: float = 0.0,
        use_hard_negatives: bool = False,
        use_cultural_negative_sampling: bool = False,
        num_negatives: int = 12,
        negative_sampling_ratios: Optional[Dict[str, float]] = None,
        curriculum_schedule: Optional[Dict[str, List[int]]] = None,
        stage1_device: Optional[str] = None,
        max_samples: Optional[int] = None,  # 新增：限制样本数量（用于快速测试）
        user_profile_dim: int = 128  # 新增：期望的user_profile维度
    ):
        """
        Args:
            train_data_path: 训练数据路径
            val_data_path: 验证数据路径
            test_data_path: 测试数据路径
            stage1_model_path: Stage1模型路径
            batch_size: 批大小
            num_workers: 数据加载线程数
            use_stage1_features: 是否使用Stage1特征
            max_items_per_user: 每位用户的候选上限
            negative_sampling_ratio: 负采样比例
            hard_negative_ratio: 困难负样本比例
            use_hard_negatives: 是否启用困难负样本
            use_cultural_negative_sampling: 是否使用CulturalNegativeSampler
            num_negatives: 每个正样本对应的负样本数
            negative_sampling_ratios: 负样本类型比例 {'easy': 0.15, 'hard': 0.70, 'extreme': 0.15}
            curriculum_schedule: 课程学习时间表
            stage1_device: Stage1特征提取所使用的设备 (cpu / cuda / auto)
            max_samples: 限制样本数量（用于快速测试，None表示不限制）
            user_profile_dim: 期望的user_profile维度（用于维度适配）
        """
        self.batch_size = batch_size
        self.user_profile_dim = user_profile_dim
        self.stage1_device = self._resolve_device(stage1_device)
        self.sequence_mode = sequence_mode
        self.max_seq_len = max_seq_len
        self.use_stage1_features = use_stage1_features  # 保存以便日志使用
        self.num_workers = self._determine_num_workers(num_workers, use_stage1_features)
        # 分布式采样器（由外部DDP设置）
        self.train_sampler = None
        self.val_sampler = None
        self.test_sampler = None

        # #region agent log: H3 - Stage2DataLoader.__init__ enter
        try:
            import json as _aj3, time as _at3, pathlib as _ap3
            _payload3 = {
                "sessionId": "stage2-train-debug",
                "runId": "pre-fix",
                "hypothesisId": "H3",
                "location": "data_loader.py:Stage2DataLoader.__init__:enter",
                "message": "enter_dataloader_init",
                "data": {
                    "train_data_path": str(train_data_path),
                    "use_stage1_features": bool(use_stage1_features),
                    "num_workers": int(num_workers),
                },
                "timestamp": int(_at3.time() * 1000),
            }
            _log_path3 = _ap3.Path("/home/data/xzy/.cursor/debug.log")
            with _log_path3.open("a", encoding="utf-8") as _f3:
                _f3.write(_aj3.dumps(_payload3, ensure_ascii=False) + "\n")
        except Exception:
            pass
        # #endregion agent log

        # Initialize CulturalNegativeSampler if enabled
        self.cultural_sampler = None
        if use_cultural_negative_sampling:
            from stage2.optimizations.modules.cultural_negative_sampler import CulturalNegativeSampler

            # Convert curriculum_schedule format if needed
            curriculum_dict = None
            if curriculum_schedule:
                curriculum_dict = {
                    k: tuple(v) if isinstance(v, list) else v
                    for k, v in curriculum_schedule.items()
                    if not k.startswith('_')  # Filter out comment fields
                }

            clean_sampling_ratios = self._sanitize_sampling_ratios(negative_sampling_ratios)

            self.cultural_sampler = CulturalNegativeSampler(
                sampling_ratios=clean_sampling_ratios,
                num_negatives=num_negatives,
                curriculum_schedule=curriculum_dict
            )
            self._check_sampling_bias(clean_sampling_ratios)
            logger.info(
                "CulturalNegativeSampler initialized: num_negatives=%d, ratios=%s",
                num_negatives,
                clean_sampling_ratios
            )

        # 创建数据集
        self.train_dataset = Stage2RecommendationDataset(
            train_data_path,
            stage1_model_path,
            use_stage1_features,
            max_items_per_user=max_items_per_user,
            negative_sampling_ratio=negative_sampling_ratio,
            sequence_mode=self.sequence_mode,
            max_seq_len=self.max_seq_len,
            hard_negative_ratio=hard_negative_ratio,
            use_hard_negatives=use_hard_negatives,
            cultural_sampler=self.cultural_sampler,
            stage1_device=self.stage1_device if use_stage1_features else 'cpu',
            max_samples=max_samples,
            user_profile_dim=user_profile_dim
        )

        # #region agent log: H4 - after train_dataset
        try:
            import json as _aj4, time as _at4, pathlib as _ap4
            _payload4 = {
                "sessionId": "stage2-train-debug",
                "runId": "pre-fix",
                "hypothesisId": "H4",
                "location": "data_loader.py:Stage2DataLoader.__init__:after_train_dataset",
                "message": "after_train_dataset",
                "data": {
                    "train_size": int(len(self.train_dataset)) if hasattr(self, "train_dataset") else None
                },
                "timestamp": int(_at4.time() * 1000),
            }
            _log_path4 = _ap4.Path("/home/data/xzy/.cursor/debug.log")
            with _log_path4.open("a", encoding="utf-8") as _f4:
                _f4.write(_aj4.dumps(_payload4, ensure_ascii=False) + "\n")
        except Exception:
            pass
        # #endregion agent log

        self.val_dataset = None
        if val_data_path is not None:
            # 验证集也限制样本数量（如果指定），但通常使用更小的比例
            val_max_samples = max_samples // 10 if max_samples else None
            self.val_dataset = Stage2RecommendationDataset(
                val_data_path,
                stage1_model_path,
                use_stage1_features,
                max_items_per_user=max_items_per_user,
                negative_sampling_ratio=negative_sampling_ratio,
                sequence_mode=self.sequence_mode,
                max_seq_len=self.max_seq_len,
                hard_negative_ratio=hard_negative_ratio,
                use_hard_negatives=use_hard_negatives,
                cultural_sampler=None,  # 验证集不使用cultural sampler
                stage1_device=self.stage1_device if use_stage1_features else 'cpu',
                max_samples=val_max_samples,
                user_profile_dim=user_profile_dim
            )

        self.test_dataset = None
        if test_data_path is not None:
            # 测试集也限制样本数量（如果指定），但通常使用更小的比例
            test_max_samples = max_samples // 10 if max_samples else None
            self.test_dataset = Stage2RecommendationDataset(
                test_data_path,
                stage1_model_path,
                use_stage1_features,
                max_items_per_user=max_items_per_user,
                negative_sampling_ratio=negative_sampling_ratio,
                sequence_mode=self.sequence_mode,
                max_seq_len=self.max_seq_len,
                hard_negative_ratio=hard_negative_ratio,
                use_hard_negatives=use_hard_negatives,
                cultural_sampler=None,  # 测试集不使用cultural sampler
                stage1_device=self.stage1_device if use_stage1_features else 'cpu',
                max_samples=test_max_samples,
                user_profile_dim=user_profile_dim
            )

        # #region agent log: H5 - after val/test datasets
        try:
            import json as _aj5, time as _at5, pathlib as _ap5
            _payload5 = {
                "sessionId": "stage2-train-debug",
                "runId": "pre-fix",
                "hypothesisId": "H5",
                "location": "data_loader.py:Stage2DataLoader.__init__:after_val_test",
                "message": "after_val_test",
                "data": {
                    "val_size": int(len(self.val_dataset)) if self.val_dataset is not None else 0,
                    "test_size": int(len(self.test_dataset)) if self.test_dataset is not None else 0,
                },
                "timestamp": int(_at5.time() * 1000),
            }
            _log_path5 = _ap5.Path("/home/data/xzy/.cursor/debug.log")
            with _log_path5.open("a", encoding="utf-8") as _f5:
                _f5.write(_aj5.dumps(_payload5, ensure_ascii=False) + "\n")
        except Exception:
            pass
        # #endregion agent log

        # 明确说明设备信息
        device_info = f"stage1_device={self.stage1_device}"
        if not self.use_stage1_features:
            device_info += " (Stage1 features disabled, training will use GPU via DDP)"
        else:
            device_info += f" (for Stage1 feature extraction only, training device will be set by trainer)"

        logger.info(
            f"Stage2DataLoader initialized: "
            f"train_size={len(self.train_dataset)}, "
            f"val_size={len(self.val_dataset) if self.val_dataset else 0}, "
            f"test_size={len(self.test_dataset) if self.test_dataset else 0}, "
            f"{device_info}"
        )

    @staticmethod
    def _sanitize_sampling_ratios(
        ratios: Optional[Dict[str, float]]
    ) -> Dict[str, float]:
        """
        过滤/归一化负采样比例,避免极端配置导致训练偏差
        """
        default = {'easy': 0.15, 'hard': 0.70, 'extreme': 0.15}
        clean = {
            k: float(v)
            for k, v in (ratios or default).items()
            if not k.startswith('_')
        }
        if not clean:
            clean = default.copy()
        # 下限,防止将某类样本完全排除
        min_floor = 0.05
        clean = {k: max(v, min_floor) for k, v in clean.items()}
        total = sum(clean.values())
        if total == 0:
            clean = default.copy()
            total = sum(clean.values())
        if abs(total - 1.0) > 0.05:
            logger.warning(
                "Negative sampling ratios sum to %.2f (expected 1.0); renormalizing.",
                total
            )
        clean = {k: v / total for k, v in clean.items()}
        return clean

    @staticmethod
    def _check_sampling_bias(ratios: Dict[str, float]):
        """
        检查负样本比例是否过度偏向某一难度,必要时记录警告
        """
        hard_ratio = ratios.get('hard', 0.0)
        easy_ratio = ratios.get('easy', 0.0)
        if hard_ratio > 0.75:
            logger.warning(
                "Hard negative ratio %.2f is very high; guideline compliance may drop.",
                hard_ratio
            )
        if easy_ratio < 0.1:
            logger.warning(
                "Easy negative ratio %.2f is very low; model may overfit hard cases.",
                easy_ratio
            )

    @staticmethod
    def _resolve_device(requested: Optional[str]) -> str:
        """解析设备字符串"""
        if requested is None or requested.lower() == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        normalized = requested.lower()
        if normalized.startswith('cuda') and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable; falling back to CPU.")
            return 'cpu'
        return normalized

    def _determine_num_workers(self, num_workers: int, use_stage1_features: bool) -> int:
        """确保Stage1特征在主线程推理,避免缓存与模型重复加载"""
        if use_stage1_features and num_workers > 0:
            logger.warning(
                "Stage1 feature extraction runs on the main thread for cache coherence; "
                "forcing num_workers from %d to 0.",
                num_workers
            )
            return 0
        return num_workers

    def set_epoch(self, epoch: int):
        """
        设置当前训练轮次（用于课程学习）

        Args:
            epoch: 当前训练轮次
        """
        if self.train_dataset is not None:
            self.train_dataset.set_epoch(epoch)

    def get_train_loader(self, shuffle: bool = True, sampler: Optional[Any] = None) -> DataLoader:
        """获取训练数据加载器"""
        sampler_to_use = sampler if sampler is not None else self.train_sampler
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle if sampler_to_use is None else False,
            sampler=sampler_to_use,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn
        )

    def get_val_loader(self, sampler: Optional[Any] = None) -> Optional[DataLoader]:
        """获取验证数据加载器"""
        if self.val_dataset is None:
            return None

        sampler_to_use = sampler if sampler is not None else self.val_sampler
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False if sampler_to_use is None else False,
            sampler=sampler_to_use,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn
        )

    def get_test_loader(self, sampler: Optional[Any] = None) -> Optional[DataLoader]:
        """获取测试数据加载器"""
        if self.test_dataset is None:
            return None

        sampler_to_use = sampler if sampler is not None else self.test_sampler
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False if sampler_to_use is None else False,
            sampler=sampler_to_use,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn
        )

    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        自定义批处理函数 (处理变长序列)
        """
        # 找到最大item数量
        max_items = max(item['num_items'].item() for item in batch)
        batch_size = len(batch)

        # 初始化批张量
        collated = {
            'user_profile': torch.stack([item['user_profile'] for item in batch]),
            'cultural_features': torch.stack([item['cultural_features'] for item in batch]),
            'item_image_features': torch.zeros(batch_size, max_items, 2048),
            'item_text_features': torch.zeros(batch_size, max_items, 768),
            'nutrition_profile': torch.zeros(batch_size, max_items, 10),
            'labels': torch.zeros(batch_size, max_items),
        }

        # 填充数据
        for i, item in enumerate(batch):
            num_items = item['num_items'].item()
            # 确保item中的tensor长度与num_items匹配
            item_labels = item['labels']
            if item_labels.shape[0] != num_items:
                # 如果长度不匹配，截取或填充
                if item_labels.shape[0] > num_items:
                    item_labels = item_labels[:num_items]
                else:
                    # 填充到num_items长度
                    padded = torch.zeros(num_items, dtype=item_labels.dtype)
                    padded[:item_labels.shape[0]] = item_labels
                    item_labels = padded

            collated['item_image_features'][i, :num_items] = item['item_image_features']
            collated['item_text_features'][i, :num_items] = item['item_text_features']
            collated['nutrition_profile'][i, :num_items] = item['nutrition_profile']
            collated['labels'][i, :num_items] = item_labels

        collated['num_items'] = torch.tensor([item['num_items'].item() for item in batch], dtype=torch.long)

        # Stage1特征 (如果存在)
        if 'stage1_features' in batch[0]:
            collated['stage1_features'] = torch.stack([item['stage1_features'] for item in batch])

        # 负样本信息 (如果存在，作为列表存储，因为长度可能不同)
        if 'negative_samples' in batch[0]:
            collated['negative_samples'] = [item.get('negative_samples', None) for item in batch]

        # 元信息（用于评估/推理导出）
        collated['metadata'] = [item.get('metadata', {}) for item in batch]

        return collated
