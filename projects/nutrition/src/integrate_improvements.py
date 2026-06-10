#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集成所有改进组件到主模型
将细粒度特征提取、多尺度融合、分层校准、温度缩放等组件集成到主模型
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# 添加项目根目录
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入新组件
try:
    from .fine_grained_feature_extractor import FineGrainedFeatureExtractor
    from .multi_scale_feature_fusion import MultiScaleFeatureFusion
    from .hierarchical_calibration_trainer import HierarchicalCalibrationHead
    from .temperature_scaling_calibrator import TemperatureScaling
    from .adversarial_training_enhancer import HighHealthDishAdversarialTrainer
except ImportError as e:
    print(f"导入错误: {e}")
    raise

# 导入现有模型组件
try:
    from .final_enhanced_health_taste_model import FinalEnhancedHealthTasteModel
except ImportError:
    from final_enhanced_health_taste_model import FinalEnhancedHealthTasteModel


class IntegratedHealthTasteModel(nn.Module):
    """集成所有改进组件的健康-口味模型"""

    def __init__(self,
                 base_model: nn.Module,
                 use_fine_grained_features: bool = True,
                 use_multi_scale_fusion: bool = True,
                 use_hierarchical_calibration: bool = True,
                 use_temperature_scaling: bool = True,
                 fine_grained_feature_dim: int = 36,
                 multi_scale_hidden_dims: list = [128, 64, 32],
                 device: str = 'cpu'):
        super().__init__()

        self.base_model = base_model
        self.use_fine_grained_features = use_fine_grained_features
        self.use_multi_scale_fusion = use_multi_scale_fusion
        self.use_hierarchical_calibration = use_hierarchical_calibration
        self.use_temperature_scaling = use_temperature_scaling
        self.device = device

        # 细粒度特征提取器
        if use_fine_grained_features:
            self.fine_grained_extractor = FineGrainedFeatureExtractor()
            self.fine_grained_proj = nn.Linear(fine_grained_feature_dim, 64)
        else:
            self.fine_grained_extractor = None
            self.fine_grained_proj = None

        # 多尺度特征融合
        if use_multi_scale_fusion:
            # 获取基础模型的隐藏维度
            base_hidden_dim = getattr(base_model, 'hidden_dim', 256)
            self.multi_scale_fusion = MultiScaleFeatureFusion(
                input_dim=base_hidden_dim + (64 if use_fine_grained_features else 0),
                hidden_dims=multi_scale_hidden_dims,
                num_attention_heads=8
            )
        else:
            self.multi_scale_fusion = None

        # 分层校准头
        if use_hierarchical_calibration:
            base_hidden_dim = getattr(base_model, 'hidden_dim', 256)
            fusion_output_dim = multi_scale_hidden_dims[-1] if use_multi_scale_fusion else base_hidden_dim
            self.hierarchical_calibrator = HierarchicalCalibrationHead(
                input_dim=fusion_output_dim,
                hidden_dims=[64, 32, 16]
            )
        else:
            self.hierarchical_calibrator = None

        # 温度缩放
        if use_temperature_scaling:
            self.temperature_scaler = TemperatureScaling(initial_temperature=1.0)
        else:
            self.temperature_scaler = None

    def extract_fine_grained_features(self, dish_names: list, nutrition_data: list) -> torch.Tensor:
        """提取细粒度特征"""
        if not self.use_fine_grained_features or self.fine_grained_extractor is None:
            return None

        features_list = []
        for dish_name, nutrition in zip(dish_names, nutrition_data):
            features = self.fine_grained_extractor.extract_all(dish_name, nutrition)
            # 转换为tensor
            feature_vector = torch.tensor([
                features.get(f'cooking_{k}', 0) for k in ['steam', 'boil', 'fry', 'deep_fry',
                'roast', 'braise', 'stir_fry', 'grill', 'raw']
            ] + [
                features.get(f'ingredient_{k}', 0) for k in ['meat', 'vegetable', 'grain',
                'dairy', 'nut', 'spice']
            ] + [
                features.get(f'ratio_{k}', 0) for k in ['protein_ratio', 'carbs_ratio',
                'fat_ratio', 'protein_fat_ratio', 'fiber_carbs_ratio', 'energy_density',
                'protein_quality', 'healthy_fat_ratio']
            ] + [
                features.get(f'regional_{k}', 0) for k in ['sichuan', 'cantonese', 'shandong',
                'jiangsu', 'zhejiang', 'fujian', 'hunan', 'anhui']
            ] + [
                features.get(f'seasonal_{k}', 0) for k in ['spring', 'summer', 'autumn', 'winter']
            ] + [features.get('composite_health_score', 0)], dtype=torch.float32)
            features_list.append(feature_vector)

        return torch.stack(features_list).to(self.device)

    def forward(self,
                region_ids: torch.Tensor,
                cuisine_ids: torch.Tensor,
                preferences: torch.Tensor,
                nutrition_features: torch.Tensor,
                dish_names: list = None,
                return_intermediate: bool = False) -> dict:
        """
        前向传播

        Args:
            region_ids: 地域ID
            cuisine_ids: 菜系ID
            preferences: 偏好特征
            nutrition_features: 营养特征
            dish_names: 菜品名称列表（用于细粒度特征提取）
            return_intermediate: 是否返回中间结果
        """
        # 基础模型前向传播
        base_output = self.base_model(
            region_ids=region_ids,
            cuisine_ids=cuisine_ids,
            preferences=preferences,
            nutrition_features=nutrition_features
        )

        # 获取基础特征（从base_model的中间层）
        # 这里需要根据实际模型结构调整
        if hasattr(self.base_model, 'feature_fusion'):
            # 尝试获取融合后的特征
            # 这需要修改base_model来暴露中间特征
            base_features = base_output.get('fused_features', None)
        else:
            base_features = None

        # 提取细粒度特征
        fine_grained_features = None
        if self.use_fine_grained_features and dish_names is not None:
            # 准备营养数据字典
            nutrition_dicts = []
            for i in range(len(dish_names)):
                nutrition_dicts.append({
                    'calories': float(nutrition_features[i, 0]),
                    'protein': float(nutrition_features[i, 1]),
                    'carbs': float(nutrition_features[i, 2]),
                    'fat': float(nutrition_features[i, 3]),
                    'fiber': float(nutrition_features[i, 4]),
                    'sodium': float(nutrition_features[i, 5]),
                    'sugar': 0.0  # 如果nutrition_features没有sugar字段
                })

            fine_grained_features = self.extract_fine_grained_features(dish_names, nutrition_dicts)
            fine_grained_features = self.fine_grained_proj(fine_grained_features)

        # 多尺度特征融合
        if self.use_multi_scale_fusion and base_features is not None:
            if fine_grained_features is not None:
                # 拼接基础特征和细粒度特征
                combined_features = torch.cat([base_features, fine_grained_features], dim=1)
            else:
                combined_features = base_features

            fused_features = self.multi_scale_fusion(combined_features)
        else:
            fused_features = base_features if base_features is not None else None

        # 获取基础预测
        base_taste = base_output.get('taste_pred', base_output.get('base_taste'))
        base_health = base_output.get('health_pred', base_output.get('base_health'))

        # 分层校准
        if self.use_hierarchical_calibration and fused_features is not None:
            calibrated_taste = self.hierarchical_calibrator(fused_features, base_taste.unsqueeze(1)).squeeze(1)
            calibrated_health = self.hierarchical_calibrator(fused_features, base_health.unsqueeze(1)).squeeze(1)
        else:
            calibrated_taste = base_taste
            calibrated_health = base_health

        # 温度缩放
        if self.use_temperature_scaling:
            # 温度缩放应用于logit空间
            # 这里简化处理，直接对预测值进行缩放
            taste_logit = torch.logit(calibrated_taste.clamp(1e-6, 1-1e-6))
            health_logit = torch.logit(calibrated_health.clamp(1e-6, 1-1e-6))

            scaled_taste_logit = self.temperature_scaler(taste_logit)
            scaled_health_logit = self.temperature_scaler(health_logit)

            final_taste = torch.sigmoid(scaled_taste_logit)
            final_health = torch.sigmoid(scaled_health_logit)
        else:
            final_taste = calibrated_taste
            final_health = calibrated_health

        results = {
            'taste_pred': final_taste,
            'health_pred': final_health,
            'base_taste': base_taste,
            'base_health': base_health,
        }

        if return_intermediate:
            results.update({
                'calibrated_taste': calibrated_taste,
                'calibrated_health': calibrated_health,
                'fine_grained_features': fine_grained_features,
                'fused_features': fused_features
            })

        return results


def create_integrated_model(base_model_path: str = None,
                           use_all_improvements: bool = True,
                           device: str = 'cpu') -> IntegratedHealthTasteModel:
    """
    创建集成所有改进的模型

    Args:
        base_model_path: 基础模型路径（可选）
        use_all_improvements: 是否使用所有改进
        device: 设备
    """
    # 创建或加载基础模型
    if base_model_path and Path(base_model_path).exists():
        base_model = torch.load(base_model_path, map_location=device, weights_only=False)
        if isinstance(base_model, dict):
            # 如果是checkpoint，需要重建模型
            base_model = FinalEnhancedHealthTasteModel()
            base_model.load_state_dict(base_model.get('model_state_dict', base_model))
    else:
        base_model = FinalEnhancedHealthTasteModel()

    # 创建集成模型
    integrated_model = IntegratedHealthTasteModel(
        base_model=base_model,
        use_fine_grained_features=use_all_improvements,
        use_multi_scale_fusion=use_all_improvements,
        use_hierarchical_calibration=use_all_improvements,
        use_temperature_scaling=use_all_improvements,
        device=device
    )

    return integrated_model


if __name__ == "__main__":
    # 测试集成模型
    print("创建集成模型...")
    model = create_integrated_model(use_all_improvements=True, device='cpu')
    print("[OK] 集成模型创建成功")
    print(f"  - 细粒度特征: {model.use_fine_grained_features}")
    print(f"  - 多尺度融合: {model.use_multi_scale_fusion}")
    print(f"  - 分层校准: {model.use_hierarchical_calibration}")
    print(f"  - 温度缩放: {model.use_temperature_scaling}")
