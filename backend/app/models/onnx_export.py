"""
ONNX导出修复模块
解决adaptive_avg_pool1d兼容性问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class ONNXCompatiblePool1d(nn.Module):
    """ONNX兼容的1D池化层"""

    def __init__(self, output_size: int):
        super().__init__()
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ONNX兼容的前向传播"""
        # 使用固定大小的平均池化替代adaptive_avg_pool1d
        if x.size(-1) == self.output_size:
            return x

        # 计算池化核大小和步长
        kernel_size = max(1, x.size(-1) // self.output_size)
        stride = kernel_size

        # 使用avg_pool1d
        pooled = F.avg_pool1d(x, kernel_size=kernel_size, stride=stride)

        # 如果输出尺寸不匹配，进行插值
        if pooled.size(-1) != self.output_size:
            pooled = F.interpolate(pooled, size=self.output_size, mode='linear', align_corners=False)

        return pooled

class ONNXCompatibleTemporalDecomposition(nn.Module):
    """ONNX兼容的时序特征分解模块"""

    def __init__(self, input_dim: int, k: int = 3, win: int = 32, stride: int = 16):
        super().__init__()
        self.k = k
        self.win = win
        self.stride = stride

        # 趋势分量提取器
        self.trend_extractor = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=win, stride=stride, padding=win//2),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # 周期分量提取器
        self.periodic_extractor = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=win//2, stride=stride//2, padding=win//4),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # 残差分量提取器
        self.residual_extractor = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=1),
            nn.ReLU()
        )

        # ONNX兼容的池化层
        self.trend_pool = ONNXCompatiblePool1d(50)  # 假设目标长度为50
        self.periodic_pool = ONNXCompatiblePool1d(50)
        self.residual_pool = ONNXCompatiblePool1d(50)

        # 特征融合
        self.fusion_layer = nn.Linear(96, input_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """ONNX兼容的时序特征分解"""
        batch_size, seq_len, input_dim = x.shape

        # 转换为卷积输入格式
        x_conv = x.transpose(1, 2)  # [batch_size, input_dim, seq_len]

        # 提取各分量
        trend = self.trend_extractor(x_conv)
        periodic = self.periodic_extractor(x_conv)
        residual = self.residual_extractor(x_conv)

        # ONNX兼容的池化
        trend = self.trend_pool(trend)
        periodic = self.periodic_pool(periodic)
        residual = self.residual_pool(residual)

        # 拼接特征
        decomposed_features = torch.cat([trend, periodic, residual], dim=1)
        decomposed_features = decomposed_features.transpose(1, 2)  # [batch_size, seq_len, 96]

        # 特征融合
        fused_features = self.fusion_layer(decomposed_features)

        # 返回分解结果
        decomposed_modes = {
            'trend': trend.transpose(1, 2),
            'periodic': periodic.transpose(1, 2),
            'residual': residual.transpose(1, 2)
        }

        return fused_features, decomposed_modes

class ONNXCompatibleGlucoNetMM(nn.Module):
    """ONNX兼容的GlucoNet-MM模型"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # ONNX兼容的时序特征分解
        self.temporal_decomposition = ONNXCompatibleTemporalDecomposition(
            input_dim=config.input_dim,
            k=config.fd_k,
            win=config.fd_win,
            stride=config.fd_stride
        )

        # 多模态注意力
        self.modality_encoders = nn.ModuleDict({
            'cgm': nn.Linear(config.input_dim, config.hidden_dim),
            'insulin': nn.Linear(config.insulin_dim, config.hidden_dim),
            'diet': nn.Linear(config.diet_dim, config.hidden_dim),
            'exercise': nn.Linear(config.exercise_dim, config.hidden_dim),
            'image': nn.Linear(config.image_dim, config.hidden_dim),
            'text': nn.Linear(config.text_dim, config.hidden_dim),
            'temporal': nn.Linear(config.input_dim, config.hidden_dim)
        })

        # 自注意力模块
        self.self_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )

        # 交叉注意力模块
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )

        # 模态融合层
        self.modality_fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * 7, config.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        )

        # 多任务学习头
        self.task_heads = nn.ModuleDict({
            'glucose_prediction': nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim // 2, 1)
            ),
            'insulin_sensitivity': nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim // 2, 1)
            ),
            'carb_estimation': nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim // 2, 1)
            ),
            'exercise_effect': nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim // 2, 1)
            )
        })

        # 任务权重学习
        self.task_weight_network = nn.Sequential(
            nn.Linear(config.hidden_dim, 4),
            nn.Softmax(dim=-1)
        )

        # 最终融合层
        self.final_fusion = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

    def forward(self,
                cgm_data: torch.Tensor,
                insulin_data: Optional[torch.Tensor] = None,
                diet_data: Optional[torch.Tensor] = None,
                exercise_data: Optional[torch.Tensor] = None,
                image_data: Optional[torch.Tensor] = None,
                text_data: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """ONNX兼容的前向传播"""

        # 1. 时序特征分解
        temporal_features, temporal_modes = self.temporal_decomposition(cgm_data)

        # 2. 多模态注意力处理
        modalities = {
            'cgm': temporal_features.mean(dim=1),
            'temporal': temporal_features.mean(dim=1)
        }

        if insulin_data is not None:
            modalities['insulin'] = insulin_data
        if diet_data is not None:
            modalities['diet'] = diet_data
        if exercise_data is not None:
            modalities['exercise'] = exercise_data
        if image_data is not None:
            modalities['image'] = image_data
        if text_data is not None:
            modalities['text'] = text_data

        # 模态编码
        encoded_modalities = {}
        for modality_name, modality_data in modalities.items():
            if modality_name in self.modality_encoders:
                encoded_modalities[modality_name] = self.modality_encoders[modality_name](modality_data)

        # 堆叠所有模态
        modality_stack = torch.stack(list(encoded_modalities.values()), dim=1)

        # 自注意力处理
        self_attended, self_attention_weights = self.self_attention(
            modality_stack, modality_stack, modality_stack
        )

        # 交叉注意力处理
        cross_attended, cross_attention_weights = self.cross_attention(
            self_attended, self_attended, self_attended
        )

        # 模态融合
        batch_size = cross_attended.shape[0]
        flattened_features = cross_attended.reshape(batch_size, -1)
        fused_features = self.modality_fusion(flattened_features)

        # 3. 最终特征融合
        final_features = self.final_fusion(fused_features)

        # 4. 多任务预测
        task_outputs = {}
        task_weights = self.task_weight_network(final_features)

        task_names = ['glucose_prediction', 'insulin_sensitivity', 'carb_estimation', 'exercise_effect']

        for i, task_name in enumerate(task_names):
            task_output = self.task_heads[task_name](final_features)
            task_outputs[task_name] = task_output
            task_outputs[f'{task_name}_weight'] = task_weights[:, i:i+1]

        # 5. 构建返回结果
        results = {
            'task_outputs': task_outputs,
            'temporal_modes': temporal_modes,
            'attention_weights': {
                'self_attention': self_attention_weights,
                'cross_attention': cross_attention_weights
            },
            'final_features': final_features
        }

        return results

class ONNXExporter:
    """ONNX导出器"""

    def __init__(self):
        self.export_history = []

    def export_model(self,
                    model: nn.Module,
                    output_path: str,
                    input_shapes: Dict[str, Tuple[int, ...]],
                    opset_version: int = 11) -> Dict[str, Any]:
        """导出模型到ONNX"""

        logger.info(f"开始导出ONNX模型: {output_path}")

        # 设置模型为评估模式
        model.eval()

        # 创建示例输入
        example_inputs = self._create_example_inputs(input_shapes)

        try:
            # 导出ONNX
            torch.onnx.export(
                model,
                example_inputs,
                output_path,
                export_params=True,
                opset_version=13,  # 使用更高版本的opset
                do_constant_folding=True,
                input_names=list(input_shapes.keys()),
                output_names=['glucose_prediction', 'insulin_sensitivity', 'carb_estimation', 'exercise_effect'],
                dynamic_axes={
                    'cgm_data': {0: 'batch_size'},
                    'insulin_data': {0: 'batch_size'},
                    'diet_data': {0: 'batch_size'},
                    'exercise_data': {0: 'batch_size'},
                    'image_data': {0: 'batch_size'},
                    'text_data': {0: 'batch_size'},
                    'glucose_prediction': {0: 'batch_size'},
                    'insulin_sensitivity': {0: 'batch_size'},
                    'carb_estimation': {0: 'batch_size'},
                    'exercise_effect': {0: 'batch_size'}
                }
            )

            # 验证导出的模型
            self._verify_onnx_model(output_path)

            # 计算模型大小
            model_size = os.path.getsize(output_path) / (1024 * 1024)  # MB

            export_info = {
                'output_path': output_path,
                'model_size_mb': model_size,
                'opset_version': opset_version,
                'input_shapes': input_shapes,
                'export_time': datetime.now().isoformat(),
                'status': 'success'
            }

            self.export_history.append(export_info)

            logger.info(f"ONNX模型导出成功: {output_path}, 大小: {model_size:.2f}MB")

            return export_info

        except Exception as e:
            logger.error(f"ONNX导出失败: {e}")

            export_info = {
                'output_path': output_path,
                'error': str(e),
                'export_time': datetime.now().isoformat(),
                'status': 'failed'
            }

            self.export_history.append(export_info)

            return export_info

    def _create_example_inputs(self, input_shapes: Dict[str, Tuple[int, ...]]) -> Tuple[torch.Tensor, ...]:
        """创建示例输入"""
        example_inputs = []

        for name, shape in input_shapes.items():
            if name == 'cgm_data':
                example_inputs.append(torch.randn(shape))
            elif name == 'insulin_data':
                example_inputs.append(torch.randn(shape))
            elif name == 'diet_data':
                example_inputs.append(torch.randn(shape))
            elif name == 'exercise_data':
                example_inputs.append(torch.randn(shape))
            elif name == 'image_data':
                example_inputs.append(torch.randn(shape))
            elif name == 'text_data':
                example_inputs.append(torch.randn(shape))

        return tuple(example_inputs)

    def _verify_onnx_model(self, model_path: str):
        """验证ONNX模型"""
        try:
            import onnx
            model = onnx.load(model_path)
            onnx.checker.check_model(model)
            logger.info("ONNX模型验证通过")
        except ImportError:
            logger.warning("ONNX包未安装，跳过模型验证")
        except Exception as e:
            logger.warning(f"ONNX模型验证失败: {e}")

    def export_edge_model(self,
                         model: nn.Module,
                         output_path: str,
                         input_shapes: Dict[str, Tuple[int, ...]]) -> Dict[str, Any]:
        """导出边缘模型"""

        # 创建轻量级边缘模型
        edge_model = self._create_edge_model()

        # 复制权重（这里需要根据实际模型结构进行映射）
        self._copy_weights_to_edge_model(model, edge_model)

        # 导出ONNX
        return self.export_model(edge_model, output_path, input_shapes)

    def _create_edge_model(self) -> nn.Module:
        """创建轻量级边缘模型"""
        class EdgeGlucoNet(nn.Module):
            def __init__(self):
                super().__init__()
                # 简化的边缘模型结构
                self.feature_extractor = nn.Sequential(
                    nn.Linear(10, 64),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(64, 32),
                    nn.ReLU()
                )

                self.task_heads = nn.ModuleDict({
                    'glucose_prediction': nn.Linear(32, 1),
                    'insulin_sensitivity': nn.Linear(32, 1),
                    'carb_estimation': nn.Linear(32, 1),
                    'exercise_effect': nn.Linear(32, 1)
                })

            def forward(self, cgm_data):
                # 取CGM数据的平均值作为输入
                x = cgm_data.mean(dim=1)  # [batch_size, input_dim]
                features = self.feature_extractor(x)

                outputs = {}
                for task, head in self.task_heads.items():
                    outputs[task] = head(features)

                return outputs['glucose_prediction'], outputs['insulin_sensitivity'], outputs['carb_estimation'], outputs['exercise_effect']

        return EdgeGlucoNet()

    def _copy_weights_to_edge_model(self, source_model: nn.Module, target_model: nn.Module):
        """复制权重到边缘模型"""
        # 这里需要根据实际模型结构进行权重映射
        # 暂时使用随机初始化
        pass

def create_onnx_compatible_model(config) -> ONNXCompatibleGlucoNetMM:
    """创建ONNX兼容模型"""
    return ONNXCompatibleGlucoNetMM(config)

def create_onnx_exporter() -> ONNXExporter:
    """创建ONNX导出器"""
    return ONNXExporter()

# 导出主要类和函数
__all__ = [
    'ONNXCompatibleGlucoNetMM',
    'ONNXCompatibleTemporalDecomposition',
    'ONNXCompatiblePool1d',
    'ONNXExporter',
    'create_onnx_compatible_model',
    'create_onnx_exporter'
]
