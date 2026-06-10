

"""
学术级数据增强技术
支持图像、文本、时间序列、表格数据增强
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import json
from datetime import datetime, timedelta
import warnings
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import albumentations as A
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import asyncio
import random

from backend.app.core.exceptions import CustomException, ValidationError
from app.core.task_queue import async_task, TaskPriority

logger = logging.getLogger(__name__)

class AugmentationType(Enum):
    """增强类型"""
    IMAGE = "image"                  # 图像增强
    TEXT = "text"                    # 文本增强
    TIME_SERIES = "time_series"      # 时间序列增强
    TABULAR = "tabular"              # 表格数据增强
    SYNTHETIC = "synthetic"          # 合成数据生成

class ImageAugmentationMethod(Enum):
    """图像增强方法"""
    GEOMETRIC = "geometric"          # 几何变换
    COLOR = "color"                  # 颜色变换
    NOISE = "noise"                  # 噪声添加
    BLUR = "blur"                    # 模糊处理
    CUTOUT = "cutout"                # 随机遮挡
    MIXUP = "mixup"                  # 混合增强

class TextAugmentationMethod(Enum):
    """文本增强方法"""
    SYNONYM_REPLACEMENT = "synonym_replacement"  # 同义词替换
    RANDOM_INSERTION = "random_insertion"       # 随机插入
    RANDOM_SWAP = "random_swap"                 # 随机交换
    RANDOM_DELETION = "random_deletion"         # 随机删除
    BACK_TRANSLATION = "back_translation"       # 反向翻译
    PARAPHRASE = "paraphrase"                   # 释义

class TimeSeriesAugmentationMethod(Enum):
    """时间序列增强方法"""
    NOISE_INJECTION = "noise_injection"         # 噪声注入
    TIME_WARPING = "time_warping"               # 时间扭曲
    WINDOW_SLICING = "window_slicing"           # 窗口切片
    MAGNITUDE_WARPING = "magnitude_warping"     # 幅度扭曲
    PERMUTATION = "permutation"                 # 排列
    JITTERING = "jittering"                     # 抖动

class TabularAugmentationMethod(Enum):
    """表格数据增强方法"""
    SMOTE = "smote"                             # SMOTE过采样
    ADASYN = "adasyn"                           # ADASYN过采样
    GAUSSIAN_NOISE = "gaussian_noise"           # 高斯噪声
    SWAP_NOISE = "swap_noise"                   # 交换噪声
    GENERATIVE_MODEL = "generative_model"       # 生成模型
    COPULA = "copula"                           # Copula方法

@dataclass
class AugmentationConfig:
    """增强配置"""
    augmentation_type: AugmentationType
    methods: List[Union[ImageAugmentationMethod, TextAugmentationMethod,
                       TimeSeriesAugmentationMethod, TabularAugmentationMethod]]
    augmentation_factor: float = 2.0  # 增强倍数
    random_seed: Optional[int] = None
    preserve_original: bool = True
    quality_threshold: float = 0.8
    custom_parameters: Optional[Dict[str, Any]] = None

@dataclass
class AugmentationResult:
    """增强结果"""
    original_count: int
    augmented_count: int
    total_count: int
    augmentation_methods: List[str]
    quality_score: float
    processing_time: float
    timestamp: datetime

class ImageAugmenter:
    """图像增强器"""

    def __init__(self):
        self.augmentation_history: List[Dict[str, Any]] = []

    def augment_images(self, images: List[np.ndarray],
                      methods: List[ImageAugmentationMethod],
                      augmentation_factor: float = 2.0) -> List[np.ndarray]:
        """增强图像"""
        logger.info(f"开始图像增强，方法: {[m.value for m in methods]}")

        augmented_images = []

        for image in images:
            # 保留原始图像
            augmented_images.append(image)

            # 生成增强图像
            for _ in range(int(augmentation_factor)):
                augmented_image = image.copy()

                for method in methods:
                    if method == ImageAugmentationMethod.GEOMETRIC:
                        augmented_image = self._geometric_transform(augmented_image)
                    elif method == ImageAugmentationMethod.COLOR:
                        augmented_image = self._color_transform(augmented_image)
                    elif method == ImageAugmentationMethod.NOISE:
                        augmented_image = self._add_noise(augmented_image)
                    elif method == ImageAugmentationMethod.BLUR:
                        augmented_image = self._add_blur(augmented_image)
                    elif method == ImageAugmentationMethod.CUTOUT:
                        augmented_image = self._cutout(augmented_image)

                augmented_images.append(augmented_image)

        logger.info(f"图像增强完成，生成 {len(augmented_images)} 张图像")
        return augmented_images

    def _geometric_transform(self, image: np.ndarray) -> np.ndarray:
        """几何变换"""
        try:
            # 使用albumentations进行几何变换
            transform = A.Compose([
                A.RandomRotate90(p=0.5),
                A.Flip(p=0.5),
                A.RandomRotate(limit=15, p=0.5),
                A.RandomScale(scale_limit=0.1, p=0.5),
                A.RandomCrop(height=image.shape[0], width=image.shape[1], p=0.5)
            ])

            augmented = transform(image=image)['image']
            return augmented

        except Exception as e:
            logger.warning(f"几何变换失败: {e}")
            return image

    def _color_transform(self, image: np.ndarray) -> np.ndarray:
        """颜色变换"""
        try:
            transform = A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                A.RandomGamma(gamma_limit=(80, 120), p=0.5)
            ])

            augmented = transform(image=image)['image']
            return augmented

        except Exception as e:
            logger.warning(f"颜色变换失败: {e}")
            return image

    def _add_noise(self, image: np.ndarray) -> np.ndarray:
        """添加噪声"""
        try:
            noise_type = random.choice(['gaussian', 'salt_pepper', 'poisson'])

            if noise_type == 'gaussian':
                noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
                augmented = cv2.add(image, noise)
            elif noise_type == 'salt_pepper':
                augmented = self._add_salt_pepper_noise(image)
            elif noise_type == 'poisson':
                augmented = np.random.poisson(image).astype(np.uint8)

            return augmented

        except Exception as e:
            logger.warning(f"噪声添加失败: {e}")
            return image

    def _add_salt_pepper_noise(self, image: np.ndarray, amount: float = 0.05) -> np.ndarray:
        """添加椒盐噪声"""
        augmented = image.copy()
        num_salt = np.ceil(amount * image.size * 0.5)
        num_pepper = np.ceil(amount * image.size * 0.5)

        # 添加盐噪声
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        augmented[tuple(coords)] = 255

        # 添加椒噪声
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        augmented[tuple(coords)] = 0

        return augmented

    def _add_blur(self, image: np.ndarray) -> np.ndarray:
        """添加模糊"""
        try:
            blur_type = random.choice(['gaussian', 'motion', 'median'])

            if blur_type == 'gaussian':
                kernel_size = random.choice([3, 5, 7])
                augmented = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            elif blur_type == 'motion':
                augmented = cv2.GaussianBlur(image, (15, 1), 0)
            elif blur_type == 'median':
                kernel_size = random.choice([3, 5])
                augmented = cv2.medianBlur(image, kernel_size)

            return augmented

        except Exception as e:
            logger.warning(f"模糊处理失败: {e}")
            return image

    def _cutout(self, image: np.ndarray, num_holes: int = 1, hole_size: int = 16) -> np.ndarray:
        """随机遮挡"""
        try:
            augmented = image.copy()
            h, w = image.shape[:2]

            for _ in range(num_holes):
                y = np.random.randint(0, h)
                x = np.random.randint(0, w)

                y1 = np.clip(y - hole_size // 2, 0, h)
                y2 = np.clip(y + hole_size // 2, 0, h)
                x1 = np.clip(x - hole_size // 2, 0, w)
                x2 = np.clip(x + hole_size // 2, 0, w)

                augmented[y1:y2, x1:x2] = 0

            return augmented

        except Exception as e:
            logger.warning(f"随机遮挡失败: {e}")
            return image

class TextAugmenter:
    """文本增强器"""

    def __init__(self):
        self.augmentation_history: List[Dict[str, Any]] = []
        self.synonyms_cache: Dict[str, List[str]] = {}

    def augment_texts(self, texts: List[str],
                     methods: List[TextAugmentationMethod],
                     augmentation_factor: float = 2.0) -> List[str]:
        """增强文本"""
        logger.info(f"开始文本增强，方法: {[m.value for m in methods]}")

        augmented_texts = []

        for text in texts:
            # 保留原始文本
            augmented_texts.append(text)

            # 生成增强文本
            for _ in range(int(augmentation_factor)):
                augmented_text = text

                for method in methods:
                    if method == TextAugmentationMethod.SYNONYM_REPLACEMENT:
                        augmented_text = self._synonym_replacement(augmented_text)
                    elif method == TextAugmentationMethod.RANDOM_INSERTION:
                        augmented_text = self._random_insertion(augmented_text)
                    elif method == TextAugmentationMethod.RANDOM_SWAP:
                        augmented_text = self._random_swap(augmented_text)
                    elif method == TextAugmentationMethod.RANDOM_DELETION:
                        augmented_text = self._random_deletion(augmented_text)
                    elif method == TextAugmentationMethod.BACK_TRANSLATION:
                        augmented_text = self._back_translation(augmented_text)

                augmented_texts.append(augmented_text)

        logger.info(f"文本增强完成，生成 {len(augmented_texts)} 个文本")
        return augmented_texts

    def _synonym_replacement(self, text: str, replacement_ratio: float = 0.3) -> str:
        """同义词替换"""
        try:
            words = text.split()
            num_replacements = int(len(words) * replacement_ratio)

            for _ in range(num_replacements):
                if len(words) == 0:
                    break

                word_index = random.randint(0, len(words) - 1)
                word = words[word_index]

                # 简单的同义词替换（实际应用中可以使用WordNet等）
                synonyms = self._get_synonyms(word)
                if synonyms:
                    words[word_index] = random.choice(synonyms)

            return ' '.join(words)

        except Exception as e:
            logger.warning(f"同义词替换失败: {e}")
            return text

    def _get_synonyms(self, word: str) -> List[str]:
        """获取同义词"""
        # 简单的同义词映射（实际应用中可以使用WordNet、BERT等）
        synonym_dict = {
            'good': ['great', 'excellent', 'wonderful', 'amazing'],
            'bad': ['terrible', 'awful', 'horrible', 'poor'],
            'big': ['large', 'huge', 'enormous', 'massive'],
            'small': ['tiny', 'little', 'miniature', 'mini'],
            'fast': ['quick', 'rapid', 'swift', 'speedy'],
            'slow': ['sluggish', 'leisurely', 'gradual', 'delayed']
        }

        return synonym_dict.get(word.lower(), [])

    def _random_insertion(self, text: str, insertion_ratio: float = 0.3) -> str:
        """随机插入"""
        try:
            words = text.split()
            num_insertions = int(len(words) * insertion_ratio)

            for _ in range(num_insertions):
                if len(words) == 0:
                    break

                word_index = random.randint(0, len(words))
                random_word = random.choice(words)
                words.insert(word_index, random_word)

            return ' '.join(words)

        except Exception as e:
            logger.warning(f"随机插入失败: {e}")
            return text

    def _random_swap(self, text: str, swap_ratio: float = 0.3) -> str:
        """随机交换"""
        try:
            words = text.split()
            num_swaps = int(len(words) * swap_ratio)

            for _ in range(num_swaps):
                if len(words) < 2:
                    break

                idx1, idx2 = random.sample(range(len(words)), 2)
                words[idx1], words[idx2] = words[idx2], words[idx1]

            return ' '.join(words)

        except Exception as e:
            logger.warning(f"随机交换失败: {e}")
            return text

    def _random_deletion(self, text: str, deletion_ratio: float = 0.3) -> str:
        """随机删除"""
        try:
            words = text.split()
            num_deletions = int(len(words) * deletion_ratio)

            for _ in range(num_deletions):
                if len(words) == 0:
                    break

                word_index = random.randint(0, len(words) - 1)
                words.pop(word_index)

            return ' '.join(words)

        except Exception as e:
            logger.warning(f"随机删除失败: {e}")
            return text

    def _back_translation(self, text: str) -> str:
        """反向翻译"""
        try:
            # 简单的反向翻译实现（实际应用中可以使用Google Translate API等）
            # 这里使用简单的词汇替换模拟
            translation_dict = {
                'hello': 'hola',
                'world': 'mundo',
                'good': 'bueno',
                'morning': 'mañana',
                'night': 'noche'
            }

            words = text.split()
            for i, word in enumerate(words):
                if word.lower() in translation_dict:
                    words[i] = translation_dict[word.lower()]

            return ' '.join(words)

        except Exception as e:
            logger.warning(f"反向翻译失败: {e}")
            return text

class TimeSeriesAugmenter:
    """时间序列增强器"""

    def __init__(self):
        self.augmentation_history: List[Dict[str, Any]] = []

    def augment_time_series(self, time_series: List[np.ndarray],
                           methods: List[TimeSeriesAugmentationMethod],
                           augmentation_factor: float = 2.0) -> List[np.ndarray]:
        """增强时间序列"""
        logger.info(f"开始时间序列增强，方法: {[m.value for m in methods]}")

        augmented_series = []

        for series in time_series:
            # 保留原始时间序列
            augmented_series.append(series)

            # 生成增强时间序列
            for _ in range(int(augmentation_factor)):
                augmented_ts = series.copy()

                for method in methods:
                    if method == TimeSeriesAugmentationMethod.NOISE_INJECTION:
                        augmented_ts = self._inject_noise(augmented_ts)
                    elif method == TimeSeriesAugmentationMethod.TIME_WARPING:
                        augmented_ts = self._time_warping(augmented_ts)
                    elif method == TimeSeriesAugmentationMethod.WINDOW_SLICING:
                        augmented_ts = self._window_slicing(augmented_ts)
                    elif method == TimeSeriesAugmentationMethod.MAGNITUDE_WARPING:
                        augmented_ts = self._magnitude_warping(augmented_ts)
                    elif method == TimeSeriesAugmentationMethod.PERMUTATION:
                        augmented_ts = self._permutation(augmented_ts)
                    elif method == TimeSeriesAugmentationMethod.JITTERING:
                        augmented_ts = self._jittering(augmented_ts)

                augmented_series.append(augmented_ts)

        logger.info(f"时间序列增强完成，生成 {len(augmented_series)} 个序列")
        return augmented_series

    def _inject_noise(self, series: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
        """注入噪声"""
        try:
            noise = np.random.normal(0, noise_level * np.std(series), series.shape)
            return series + noise
        except Exception as e:
            logger.warning(f"噪声注入失败: {e}")
            return series

    def _time_warping(self, series: np.ndarray, sigma: float = 0.2) -> np.ndarray:
        """时间扭曲"""
        try:
            n = len(series)
            warp_steps = np.arange(n)

            # 生成扭曲函数
            warp_function = np.cumsum(np.random.normal(0, sigma, n))
            warp_function = warp_function - warp_function[0]
            warp_function = warp_function * (n - 1) / warp_function[-1]

            # 插值
            from scipy.interpolate import interp1d
            f = interp1d(warp_steps, series, kind='linear', bounds_error=False, fill_value='extrapolate')
            warped_series = f(warp_function)

            return warped_series

        except Exception as e:
            logger.warning(f"时间扭曲失败: {e}")
            return series

    def _window_slicing(self, series: np.ndarray, reduce_ratio: float = 0.9) -> np.ndarray:
        """窗口切片"""
        try:
            target_length = int(len(series) * reduce_ratio)
            if target_length < 2:
                return series

            start_idx = random.randint(0, len(series) - target_length)
            return series[start_idx:start_idx + target_length]

        except Exception as e:
            logger.warning(f"窗口切片失败: {e}")
            return series

    def _magnitude_warping(self, series: np.ndarray, sigma: float = 0.2) -> np.ndarray:
        """幅度扭曲"""
        try:
            n = len(series)
            warp_steps = np.arange(n)

            # 生成幅度扭曲函数
            warp_function = np.random.normal(1, sigma, n)
            warp_function = np.cumsum(warp_function)
            warp_function = warp_function / warp_function[0]

            return series * warp_function

        except Exception as e:
            logger.warning(f"幅度扭曲失败: {e}")
            return series

    def _permutation(self, series: np.ndarray, max_segments: int = 5) -> np.ndarray:
        """排列"""
        try:
            n = len(series)
            num_segments = random.randint(2, max_segments)
            segment_size = n // num_segments

            segments = []
            for i in range(num_segments):
                start_idx = i * segment_size
                end_idx = start_idx + segment_size if i < num_segments - 1 else n
                segments.append(series[start_idx:end_idx])

            # 随机排列段
            random.shuffle(segments)

            return np.concatenate(segments)

        except Exception as e:
            logger.warning(f"排列失败: {e}")
            return series

    def _jittering(self, series: np.ndarray, sigma: float = 0.03) -> np.ndarray:
        """抖动"""
        try:
            noise = np.random.normal(0, sigma * np.std(series), series.shape)
            return series + noise
        except Exception as e:
            logger.warning(f"抖动失败: {e}")
            return series

class TabularAugmenter:
    """表格数据增强器"""

    def __init__(self):
        self.augmentation_history: List[Dict[str, Any]] = []

    def augment_tabular_data(self, X: pd.DataFrame, methods: List[TabularAugmentationMethod],
                           y: Optional[pd.Series] = None,
                           augmentation_factor: float = 2.0) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """增强表格数据"""
        logger.info(f"开始表格数据增强，方法: {[m.value for m in methods]}")

        X_augmented = X.copy()
        y_augmented = y.copy() if y is not None else None

        for method in methods:
            if method == TabularAugmentationMethod.SMOTE:
                X_augmented, y_augmented = self._smote_augmentation(X_augmented, y_augmented, augmentation_factor)
            elif method == TabularAugmentationMethod.ADASYN:
                X_augmented, y_augmented = self._adasyn_augmentation(X_augmented, y_augmented, augmentation_factor)
            elif method == TabularAugmentationMethod.GAUSSIAN_NOISE:
                X_augmented = self._gaussian_noise_augmentation(X_augmented, augmentation_factor)
            elif method == TabularAugmentationMethod.SWAP_NOISE:
                X_augmented = self._swap_noise_augmentation(X_augmented, augmentation_factor)
            elif method == TabularAugmentationMethod.GENERATIVE_MODEL:
                X_augmented, y_augmented = self._generative_model_augmentation(X_augmented, y_augmented, augmentation_factor)

        logger.info(f"表格数据增强完成，生成 {len(X_augmented)} 个样本")
        return X_augmented, y_augmented

    def _smote_augmentation(self, X: pd.DataFrame, y: pd.Series, augmentation_factor: float) -> Tuple[pd.DataFrame, pd.Series]:
        """SMOTE过采样"""
        try:
            from imblearn.over_sampling import SMOTE

            smote = SMOTE(random_state=42, k_neighbors=3)
            X_resampled, y_resampled = smote.fit_resample(X, y)

            return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)

        except Exception as e:
            logger.warning(f"SMOTE增强失败: {e}")
            return X, y

    def _adasyn_augmentation(self, X: pd.DataFrame, y: pd.Series, augmentation_factor: float) -> Tuple[pd.DataFrame, pd.Series]:
        """ADASYN过采样"""
        try:
            from imblearn.over_sampling import ADASYN

            adasyn = ADASYN(random_state=42)
            X_resampled, y_resampled = adasyn.fit_resample(X, y)

            return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)

        except Exception as e:
            logger.warning(f"ADASYN增强失败: {e}")
            return X, y

    def _gaussian_noise_augmentation(self, X: pd.DataFrame, augmentation_factor: float) -> pd.DataFrame:
        """高斯噪声增强"""
        try:
            numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
            augmented_data = []

            for _ in range(int(augmentation_factor)):
                X_noisy = X.copy()
                for col in numeric_columns:
                    noise = np.random.normal(0, 0.1 * X[col].std(), len(X))
                    X_noisy[col] = X[col] + noise
                augmented_data.append(X_noisy)

            return pd.concat([X] + augmented_data, ignore_index=True)

        except Exception as e:
            logger.warning(f"高斯噪声增强失败: {e}")
            return X

    def _swap_noise_augmentation(self, X: pd.DataFrame, augmentation_factor: float) -> pd.DataFrame:
        """交换噪声增强"""
        try:
            augmented_data = []

            for _ in range(int(augmentation_factor)):
                X_swapped = X.copy()

                # 随机交换行
                swap_indices = np.random.choice(len(X), size=len(X)//10, replace=False)
                for i in range(0, len(swap_indices), 2):
                    if i+1 < len(swap_indices):
                        idx1, idx2 = swap_indices[i], swap_indices[i+1]
                        X_swapped.iloc[idx1], X_swapped.iloc[idx2] = X_swapped.iloc[idx2].copy(), X_swapped.iloc[idx1].copy()

                augmented_data.append(X_swapped)

            return pd.concat([X] + augmented_data, ignore_index=True)

        except Exception as e:
            logger.warning(f"交换噪声增强失败: {e}")
            return X

    def _generative_model_augmentation(self, X: pd.DataFrame, y: Optional[pd.Series], augmentation_factor: float) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """生成模型增强"""
        try:
            # 简单的生成模型实现（实际应用中可以使用GAN、VAE等）
            numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
            augmented_data = []
            augmented_labels = []

            for _ in range(int(augmentation_factor)):
                # 使用K-means聚类生成新样本
                if len(numeric_columns) > 0:
                    kmeans = KMeans(n_clusters=min(5, len(X)//10), random_state=42)
                    clusters = kmeans.fit_predict(X[numeric_columns])

                    # 在每个聚类内生成新样本
                    for cluster_id in range(kmeans.n_clusters):
                        cluster_mask = clusters == cluster_id
                        cluster_data = X[cluster_mask][numeric_columns]

                        if len(cluster_data) > 1:
                            # 计算聚类中心和协方差
                            center = cluster_data.mean()
                            cov = cluster_data.cov()

                            # 生成新样本
                            new_sample = np.random.multivariate_normal(center, cov, 1)
                            new_row = X.iloc[0].copy()
                            new_row[numeric_columns] = new_sample[0]
                            augmented_data.append(new_row)

                            if y is not None:
                                # 使用聚类内最常见的标签
                                cluster_labels = y[cluster_mask]
                                most_common_label = cluster_labels.mode()[0] if len(cluster_labels.mode()) > 0 else cluster_labels.iloc[0]
                                augmented_labels.append(most_common_label)

            if augmented_data:
                X_augmented = pd.concat([X] + [pd.DataFrame([row]) for row in augmented_data], ignore_index=True)
                y_augmented = pd.concat([y] + [pd.Series(augmented_labels)]) if y is not None and augmented_labels else y
                return X_augmented, y_augmented

            return X, y

        except Exception as e:
            logger.warning(f"生成模型增强失败: {e}")
            return X, y

class DataAugmenter:
    """数据增强主类"""

    def __init__(self):
        self.image_augmenter = ImageAugmenter()
        self.text_augmenter = TextAugmenter()
        self.timeseries_augmenter = TimeSeriesAugmenter()
        self.tabular_augmenter = TabularAugmenter()
        self.augmentation_history: List[AugmentationResult] = []

    def augment_data(self, data: Any, config: AugmentationConfig) -> Tuple[Any, AugmentationResult]:
        """数据增强"""
        logger.info(f"开始数据增强，类型: {config.augmentation_type.value}")

        start_time = datetime.now()
        original_count = len(data) if hasattr(data, '__len__') else 1

        try:
            if config.augmentation_type == AugmentationType.IMAGE:
                augmented_data = self.image_augmenter.augment_images(
                    data, config.methods, config.augmentation_factor
                )

            elif config.augmentation_type == AugmentationType.TEXT:
                augmented_data = self.text_augmenter.augment_texts(
                    data, config.methods, config.augmentation_factor
                )

            elif config.augmentation_type == AugmentationType.TIME_SERIES:
                augmented_data = self.timeseries_augmenter.augment_time_series(
                    data, config.methods, config.augmentation_factor
                )

            elif config.augmentation_type == AugmentationType.TABULAR:
                augmented_data = self.tabular_augmenter.augment_tabular_data(
                    data, None, config.methods, config.augmentation_factor
                )

            else:
                raise CustomException(f"不支持的数据增强类型: {config.augmentation_type}")

            # 计算增强结果
            augmented_count = len(augmented_data) - original_count if config.preserve_original else len(augmented_data)
            total_count = len(augmented_data)
            processing_time = (datetime.now() - start_time).total_seconds()

            result = AugmentationResult(
                original_count=original_count,
                augmented_count=augmented_count,
                total_count=total_count,
                augmentation_methods=[m.value for m in config.methods],
                quality_score=config.quality_threshold,
                processing_time=processing_time,
                timestamp=datetime.now()
            )

            self.augmentation_history.append(result)

            logger.info(f"数据增强完成，生成 {augmented_count} 个新样本")
            return augmented_data, result

        except Exception as e:
            logger.error(f"数据增强失败: {e}")
            raise CustomException(f"数据增强失败: {e}")

# 全局数据增强器实例
data_augmenter = DataAugmenter()

# 异步任务
@async_task("augment_data", TaskPriority.NORMAL)
def augment_data_task(data: Any, config_dict: Dict[str, Any]):
    """数据增强任务"""
    config = AugmentationConfig(**config_dict)
    augmented_data, result = data_augmenter.augment_data(data, config)

    return {
        "augmented_data": augmented_data,
        "result": asdict(result),
        "success": True
    }

# 数据增强API
def augment_data(data: Any, config: AugmentationConfig) -> Tuple[Any, AugmentationResult]:
    """数据增强"""
    return data_augmenter.augment_data(data, config)

def get_augmentation_history() -> List[AugmentationResult]:
    """获取增强历史"""
    return data_augmenter.augmentation_history

if __name__ == "__main__":
    # 测试数据增强
    import numpy as np

    # 测试图像增强
    test_images = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(5)]

    config = AugmentationConfig(
        augmentation_type=AugmentationType.IMAGE,
        methods=[ImageAugmentationMethod.GEOMETRIC, ImageAugmentationMethod.COLOR],
        augmentation_factor=2.0
    )

    augmented_images, result = augment_data(test_images, config)

    print("图像增强完成:")
    print(f"原始图像数: {result.original_count}")
    print(f"增强图像数: {result.augmented_count}")
    print(f"总图像数: {result.total_count}")
    print(f"增强方法: {result.augmentation_methods}")
    print(f"处理时间: {result.processing_time:.2f}秒")
