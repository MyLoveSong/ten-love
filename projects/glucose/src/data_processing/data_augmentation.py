"""
数据增强模块
提供葡萄糖预测的时间序列数据增强方法
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class GlucoseDataAugmenter:
    """葡萄糖时间序列数据增强器"""

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler()

    def augment_data(self, data: np.ndarray, targets: np.ndarray,
                    augmentation_factor: float = 2.0,
                    methods: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """应用多种增强方法到葡萄糖数据"""
        if methods is None:
            methods = ['noise', 'scaling', 'time_warping', 'magnitude_warping']

        try:
            augmented_data = [data]
            augmented_targets = [targets]

            samples_to_generate = int(len(data) * (augmentation_factor - 1))

            for method in methods:
                if method == 'noise':
                    aug_data, aug_targets = self._add_gaussian_noise(
                        data, targets, samples_to_generate // len(methods)
                    )
                elif method == 'scaling':
                    aug_data, aug_targets = self._apply_scaling(
                        data, targets, samples_to_generate // len(methods)
                    )
                elif method == 'time_warping':
                    aug_data, aug_targets = self._apply_time_warping(
                        data, targets, samples_to_generate // len(methods)
                    )
                elif method == 'magnitude_warping':
                    aug_data, aug_targets = self._apply_magnitude_warping(
                        data, targets, samples_to_generate // len(methods)
                    )
                else:
                    continue

                augmented_data.append(aug_data)
                augmented_targets.append(aug_targets)

            final_data = np.concatenate(augmented_data, axis=0)
            final_targets = np.concatenate(augmented_targets, axis=0)

            logger.info(f"数据增强完成: {len(data)} -> {len(final_data)} samples")
            return final_data, final_targets

        except Exception as e:
            raise ValueError(f"数据增强失败: {str(e)}")

    def _add_gaussian_noise(self, data: np.ndarray, targets: np.ndarray,
                           num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """添加高斯噪声"""
        indices = np.random.choice(len(data), num_samples, replace=True)
        selected_data = data[indices]
        selected_targets = targets[indices]

        noise_levels = np.random.uniform(0.01, 0.05, num_samples)

        augmented_data = []
        for i, noise_level in enumerate(noise_levels):
            noise = np.random.normal(0, noise_level, selected_data[i].shape)
            augmented_sample = selected_data[i] + noise
            augmented_sample = np.clip(augmented_sample, 0.0, 1.0)
            augmented_data.append(augmented_sample)

        return np.array(augmented_data), selected_targets

    def _apply_scaling(self, data: np.ndarray, targets: np.ndarray,
                      num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """应用随机缩放"""
        indices = np.random.choice(len(data), num_samples, replace=True)
        selected_data = data[indices]
        selected_targets = targets[indices]

        scale_factors = np.random.uniform(0.9, 1.1, num_samples)

        augmented_data = []
        augmented_targets = []

        for i, scale_factor in enumerate(scale_factors):
            scaled_data = selected_data[i] * scale_factor
            scaled_targets = selected_targets[i] * scale_factor

            scaled_data = np.clip(scaled_data, 0.0, 1.0)
            scaled_targets = np.clip(scaled_targets, 0.0, 1.0)

            augmented_data.append(scaled_data)
            augmented_targets.append(scaled_targets)

        return np.array(augmented_data), np.array(augmented_targets)

    def _apply_time_warping(self, data: np.ndarray, targets: np.ndarray,
                           num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """应用时间扭曲"""
        indices = np.random.choice(len(data), num_samples, replace=True)
        selected_data = data[indices]
        selected_targets = targets[indices]

        augmented_data = []

        for sample in selected_data:
            seq_len = sample.shape[0]
            warp_factor = np.random.uniform(0.8, 1.2)

            original_indices = np.arange(seq_len)
            warped_indices = np.linspace(0, seq_len - 1, int(seq_len * warp_factor))

            if len(sample.shape) == 1:
                warped_sample = np.interp(original_indices, warped_indices,
                                        np.interp(warped_indices, original_indices, sample))
            else:
                warped_sample = np.zeros_like(sample)
                for feature_idx in range(sample.shape[1]):
                    warped_sample[:, feature_idx] = np.interp(
                        original_indices, warped_indices,
                        np.interp(warped_indices, original_indices, sample[:, feature_idx])
                    )

            augmented_data.append(warped_sample)

        return np.array(augmented_data), selected_targets

    def _apply_magnitude_warping(self, data: np.ndarray, targets: np.ndarray,
                                num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """应用幅度扭曲"""
        indices = np.random.choice(len(data), num_samples, replace=True)
        selected_data = data[indices]
        selected_targets = targets[indices]

        augmented_data = []

        for sample in selected_data:
            seq_len = sample.shape[0]

            knot_points = np.random.uniform(0.8, 1.2, max(3, seq_len // 4))
            knot_indices = np.linspace(0, seq_len - 1, len(knot_points))

            warp_curve = np.interp(np.arange(seq_len), knot_indices, knot_points)

            if len(sample.shape) == 1:
                warped_sample = sample * warp_curve
            else:
                warped_sample = sample * warp_curve.reshape(-1, 1)

            warped_sample = np.clip(warped_sample, 0.0, 1.0)
            augmented_data.append(warped_sample)

        return np.array(augmented_data), selected_targets
