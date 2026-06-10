"""
数据处理工具模块
提供数据加载、预处理等核心功能
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter
from torch.utils.data import Dataset, DataLoader, random_split, Subset

logger = logging.getLogger(__name__)


class CGMDataset(Dataset):
    """连续血糖监测(CGM)数据集"""

    def __init__(self, data_path: str, sequence_length: int = 24,
                 prediction_horizon: int = 6, transform: Optional[callable] = None):
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.transform = transform

        self.data = self._load_data()
        self.samples = self._prepare_samples()
        logger.info(f"CGM数据集初始化完成，共{len(self.samples)}个样本")

    def _load_data(self) -> pd.DataFrame:
        """加载数据"""
        try:
            if os.path.isdir(self.data_path):
                all_dfs = []
                for file in os.listdir(self.data_path):
                    if file.endswith('.csv'):
                        df = pd.read_csv(os.path.join(self.data_path, file))
                        all_dfs.append(df)
                if not all_dfs:
                    raise ValueError(f"在{self.data_path}中未找到CSV文件")
                return pd.concat(all_dfs, ignore_index=True)
            else:
                return pd.read_csv(self.data_path)
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            raise

    def _prepare_samples(self) -> List[Dict]:
        """准备样本"""
        samples = []
        grouped = self.data.groupby('patient_id') if 'patient_id' in self.data.columns else [(None, self.data)]

        for patient_id, patient_data in grouped:
            if 'timestamp' in patient_data.columns:
                patient_data = patient_data.sort_values('timestamp')

            glucose = patient_data['glucose'].values if 'glucose' in patient_data.columns else patient_data.iloc[:, 0].values

            for i in range(len(glucose) - self.sequence_length - self.prediction_horizon + 1):
                samples.append({
                    'input_sequence': glucose[i:i+self.sequence_length],
                    'target_sequence': glucose[i+self.sequence_length:i+self.sequence_length+self.prediction_horizon],
                    'patient_id': patient_id
                })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        result = {
            'input_sequence': torch.FloatTensor(sample['input_sequence']).unsqueeze(-1),
            'target_sequence': torch.FloatTensor(sample['target_sequence']),
            'patient_id': sample['patient_id']
        }
        return self.transform(result) if self.transform else result


class CulturalPreferenceDataset(Dataset):
    """文化饮食偏好数据集"""

    def __init__(self, data_path: str, region_embedding: bool = True,
                 transform: Optional[callable] = None):
        self.data_path = data_path
        self.region_embedding = region_embedding
        self.transform = transform
        self.data = self._load_data()
        self.region_to_idx = self._create_region_mapping()

    def _load_data(self) -> List[Dict]:
        try:
            if self.data_path.endswith('.json'):
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            elif self.data_path.endswith('.csv'):
                return pd.read_csv(self.data_path).to_dict('records')
        except Exception as e:
            logger.error(f"文化饮食偏好数据加载失败: {e}")
            raise

    def _create_region_mapping(self) -> Dict[str, int]:
        regions = {item.get('region') for item in self.data if 'region' in item}
        return {r: i for i, r in enumerate(sorted(regions)) if r}

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        features = {}
        if self.region_embedding and 'region' in item:
            features['region_idx'] = torch.LongTensor([self.region_to_idx.get(item['region'], 0)])

        prefs = [item.get(k, 0.0) for k in ['carbs_preference', 'protein_preference', 'fat_preference', 'spicy_level', 'sweet_level']]
        features['preference'] = torch.FloatTensor(prefs)
        features['acceptance_score'] = torch.FloatTensor([item.get('acceptance_score', item.get('satisfaction_score', 0.0))])
        return self.transform(features) if self.transform else features


class FeedbackDataset(Dataset):
    """用户反馈数据集"""

    def __init__(self, data_path: str, max_samples: int = 1000, transform: Optional[callable] = None):
        self.data_path = data_path
        self.max_samples = max_samples
        self.transform = transform
        self.data = self._load_data()

    def _load_data(self) -> List[Dict]:
        try:
            if os.path.isdir(self.data_path):
                files = sorted([f for f in os.listdir(self.data_path) if f.endswith('.json')], reverse=True)
                all_data = []
                for file in files[:self.max_samples]:
                    with open(os.path.join(self.data_path, file)) as f:
                        all_data.append(json.load(f))
                return all_data
            elif self.data_path.endswith('.json'):
                data = json.load(open(self.data_path))
                return data if isinstance(data, list) else [data]
            elif self.data_path.endswith('.csv'):
                return pd.read_csv(self.data_path).head(self.max_samples).to_dict('records')
        except Exception as e:
            logger.error(f"用户反馈数据加载失败: {e}")
            raise

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        features = {k: v for k, v in item.items() if k in ['feedback_type', 'recommendation', 'user_reaction', 'context']}
        if 'rating' in item:
            features['rating'] = torch.FloatTensor([item['rating']])
        return self.transform(features) if self.transform else features


def download_dataset(dataset_name: str, output_dir: str, **kwargs) -> str:
    """下载数据集"""
    raise NotImplementedError("下载功能需要单独实现")


def create_data_loaders(dataset: Dataset, batch_size: int = 32, train_ratio: float = 0.8,
    val_ratio: float = 0.1, test_ratio: float = 0.1, num_workers: int = 4, seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建训练、验证和测试数据加载器"""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "比例总和必须为1"

    torch.manual_seed(seed)
    np.random.seed(seed)

    n = len(dataset)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    test_size = n - train_size - val_size

    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed))

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    )


def normalize_sequence(sequence: torch.Tensor, method: str = 'minmax') -> Tuple[torch.Tensor, Dict]:
    """归一化序列数据"""
    if method == 'minmax':
        min_v, max_v = torch.min(sequence).item(), torch.max(sequence).item()
        if max_v == min_v:
            return torch.zeros_like(sequence), {'min_val': min_v, 'max_val': max_v}
        return (sequence - min_v) / (max_v - min_v), {'min_val': min_v, 'max_val': max_v}
    elif method == 'zscore':
        mean_v, std_v = torch.mean(sequence).item(), torch.std(sequence).item()
        if std_v == 0:
            return torch.zeros_like(sequence), {'mean_val': mean_v, 'std_val': std_v}
        return (sequence - mean_v) / std_v, {'mean_val': mean_v, 'std_val': std_v}
    raise ValueError(f"不支持的归一化方法: {method}")


def denormalize_sequence(seq: torch.Tensor, params: Dict, method: str = 'minmax') -> torch.Tensor:
    """反归一化序列"""
    if method == 'minmax':
        return seq * (params['max_val'] - params['min_val']) + params['min_val']
    elif method == 'zscore':
        return seq * params['std_val'] + params['mean_val']
    raise ValueError(f"不支持的归一化方法: {method}")


def fill_missing_values(sequence: torch.Tensor, method: str = 'linear', max_gap: int = 5) -> torch.Tensor:
    """填充缺失值"""
    filled = sequence.clone()
    mask = torch.isnan(filled) if torch.isnan(sequence).any() else (filled == -1)
    if not mask.any():
        return filled

    arr = filled.numpy()
    m = mask.numpy()

    if method == 'linear':
        for i in range(len(arr)):
            if m[i]:
                left, right = -1, -1
                for j in range(i-1, -1, -1):
                    if not m[j] and i-j <= max_gap:
                        left = j
                        break
                for j in range(i+1, len(arr)):
                    if not m[j] and j-i <= max_gap:
                        right = j
                        break
                if left >= 0 and right >= 0:
                    arr[i] = arr[left] + (arr[right] - arr[left]) * (i - left) / (right - left)
                elif left >= 0:
                    arr[i] = arr[left]
                elif right >= 0:
                    arr[i] = arr[right]
    elif method == 'forward':
        last_valid, last_idx = None, -1
        for i in range(len(arr)):
            if not m[i]:
                last_valid, last_idx = arr[i], i
            elif last_valid is not None and i - last_idx <= max_gap:
                arr[i] = last_valid
    elif method == 'mean':
        valid = arr[~m]
        if len(valid) > 0:
            arr[m] = np.mean(valid)

    return torch.from_numpy(arr).float()


def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    try:
        if config_path.endswith('.json'):
            with open(config_path) as f:
                return json.load(f)
        elif config_path.endswith(('.yaml', '.yml')):
            import yaml
            with open(config_path) as f:
                return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"配置加载失败: {e}")
        raise


def save_config(config: Dict, config_path: str) -> None:
    """保存配置文件"""
    try:
        os.makedirs(os.path.dirname(config_path) or '.', exist_ok=True)
        if config_path.endswith('.json'):
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        elif config_path.endswith(('.yaml', '.yml')):
            import yaml
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
    except Exception as e:
        logger.error(f"配置保存失败: {e}")
        raise
