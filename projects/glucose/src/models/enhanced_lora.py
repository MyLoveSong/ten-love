"""
增强型LoRA实现模块
基于最新研究和最佳实践的低秩适配实现
"""

import os
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any, Set
from pathlib import Path
import copy

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LoRAConfig:
    """LoRA配置类"""

    def __init__(
        self,
        r: int = 8,
        alpha: float = 16,
        dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
        bias: str = "none",
        modules_to_save: Optional[List[str]] = None,
        init_lora_weights: bool = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        use_lokr: bool = False,
        use_adaptive: bool = False,
        rank_pattern: Optional[Dict[str, int]] = None,
        alpha_pattern: Optional[Dict[str, float]] = None,
    ):
        """
        初始化LoRA配置

        Args:
            r: LoRA秩
            alpha: LoRA缩放因子
            dropout: Dropout比率
            target_modules: 目标模块列表
            bias: 偏置处理方式，可选['none', 'all', 'lora_only']
            modules_to_save: 需要完整保存的模块列表
            init_lora_weights: 是否初始化LoRA权重
            use_rslora: 是否使用Rank-Stabilized LoRA (RSLoRA)
            use_dora: 是否使用Weight-Decomposed Low-Rank Adaptation (DoRA)
            use_lokr: 是否使用Low-Rank Kronecker Product (LoKr)
            use_adaptive: 是否使用自适应秩
            rank_pattern: 模块特定的秩配置
            alpha_pattern: 模块特定的alpha配置
        """
        self.r = r
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules
        self.bias = bias
        self.modules_to_save = modules_to_save or []
        self.init_lora_weights = init_lora_weights

        # 高级LoRA变体
        self.use_rslora = use_rslora
        self.use_dora = use_dora
        self.use_lokr = use_lokr
        self.use_adaptive = use_adaptive

        # 模块特定配置
        self.rank_pattern = rank_pattern or {}
        self.alpha_pattern = alpha_pattern or {}

        # 验证配置
        self._validate_config()

    def _validate_config(self):
        """验证配置有效性"""
        if self.r <= 0:
            raise ValueError(f"LoRA秩必须为正数，当前值: {self.r}")

        if self.alpha <= 0:
            raise ValueError(f"LoRA缩放因子必须为正数，当前值: {self.alpha}")

        if self.bias not in ["none", "all", "lora_only"]:
            raise ValueError(f"偏置处理方式必须为 'none', 'all' 或 'lora_only'，当前值: {self.bias}")

        # 验证高级变体的互斥性
        advanced_variants = [self.use_rslora, self.use_dora, self.use_lokr]
        if sum(advanced_variants) > 1:
            raise ValueError("RSLoRA, DoRA 和 LoKr 是互斥的，只能启用一个")

class EnhancedLoRALayer(nn.Module):
    """增强型LoRA层"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: LoRAConfig,
        layer_name: str = ""
    ):
        """
        初始化增强型LoRA层

        Args:
            in_features: 输入特征维度
            out_features: 输出特征维度
            config: LoRA配置
            layer_name: 层名称，用于模块特定配置
        """
        super().__init__()

        # 确定该层的秩和alpha
        self.r = config.rank_pattern.get(layer_name, config.r)
        self.alpha = config.alpha_pattern.get(layer_name, config.alpha)
        self.scaling = self.alpha / self.r
        self.layer_name = layer_name

        # 基础LoRA参数
        self.lora_A = nn.Parameter(torch.zeros((self.r, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, self.r)))

        # Dropout层
        self.dropout = nn.Dropout(config.dropout)

        # 高级LoRA变体参数
        if config.use_rslora:
            # RSLoRA: 添加正则化参数
            self.lora_E = nn.Parameter(torch.ones(1))

        if config.use_dora:
            # DoRA: 添加方向调制参数
            self.lora_D = nn.Parameter(torch.zeros(out_features, 1))
            self.original_weights = None

        if config.use_lokr:
            # LoKr: 使用Kronecker乘积分解
            self.lora_A = nn.Parameter(torch.zeros((self.r, int(math.sqrt(in_features)))))
            self.lora_B = nn.Parameter(torch.zeros((int(math.sqrt(out_features)), self.r)))
            self.lora_C = nn.Parameter(torch.zeros((self.r, int(math.sqrt(in_features)))))
            self.lora_D = nn.Parameter(torch.zeros((int(math.sqrt(out_features)), self.r)))

        # 初始化权重
        if config.init_lora_weights:
            self.reset_parameters()

        # 保存配置
        self.config = config

    def reset_parameters(self):
        """重置参数"""
        if self.config.use_lokr:
            # LoKr初始化
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            nn.init.kaiming_uniform_(self.lora_C, a=math.sqrt(5))
            nn.init.zeros_(self.lora_D)
        else:
            # 标准LoRA初始化
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

            # DoRA初始化
            if self.config.use_dora and hasattr(self, 'lora_D'):
                nn.init.zeros_(self.lora_D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量

        Returns:
            LoRA适配结果
        """
        # 应用Dropout
        x = self.dropout(x)

        if self.config.use_lokr:
            # LoKr计算
            x_reshaped = x.view(-1, int(math.sqrt(x.shape[-1])), int(math.sqrt(x.shape[-1])))

            # 第一个低秩矩阵
            temp1 = torch.matmul(self.lora_A, x_reshaped)
            temp1 = torch.matmul(self.lora_B, temp1)

            # 第二个低秩矩阵
            temp2 = torch.matmul(self.lora_C, x_reshaped.transpose(-1, -2))
            temp2 = torch.matmul(self.lora_D, temp2)

            # 组合结果
            result = (temp1 + temp2.transpose(-1, -2)).view(x.shape[0], -1)

        else:
            # 标准LoRA计算: B·(A·x)
            result = (self.lora_B @ self.lora_A) @ x.T

            # RSLoRA: 添加正则化
            if self.config.use_rslora and hasattr(self, 'lora_E'):
                norm_x = torch.norm(x, dim=1, keepdim=True)
                norm_factor = torch.clamp(self.lora_E * norm_x, min=1e-6)
                result = result / norm_factor.T

            result = result.T

        # 应用缩放
        return result * self.scaling

    def get_delta_weight(self) -> torch.Tensor:
        """
        获取权重增量

        Returns:
            权重增量矩阵
        """
        if self.config.use_lokr:
            # LoKr权重增量计算较为复杂，需要重建完整权重矩阵
            raise NotImplementedError("LoKr权重增量计算尚未实现")
        else:
            # 标准LoRA权重增量
            delta = (self.lora_B @ self.lora_A) * self.scaling

            # DoRA: 应用方向调制
            if self.config.use_dora and hasattr(self, 'lora_D') and self.original_weights is not None:
                # 计算原始权重的范数
                W_norm = torch.norm(self.original_weights, dim=1, keepdim=True)
                # 计算原始权重的方向
                W_direction = self.original_weights / (W_norm + 1e-6)
                # 应用方向调制
                delta = delta * (1 - self.lora_D @ W_direction)

            return delta

class EnhancedLoRALinear(nn.Module):
    """增强型LoRA线性层，替代nn.Linear"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: LoRAConfig,
        bias: bool = True,
        layer_name: str = "",
        original_module: Optional[nn.Module] = None
    ):
        """
        初始化增强型LoRA线性层

        Args:
            in_features: 输入特征维度
            out_features: 输出特征维度
            config: LoRA配置
            bias: 是否使用偏置
            layer_name: 层名称，用于模块特定配置
            original_module: 原始模块，用于权重初始化
        """
        super().__init__()

        # 原始线性层
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # 如果提供了原始模块，复制其权重
        if original_module is not None:
            self.linear.weight.data.copy_(original_module.weight.data)
            if bias and hasattr(original_module, 'bias') and original_module.bias is not None:
                self.linear.bias.data.copy_(original_module.bias.data)

        # LoRA层
        self.lora = EnhancedLoRALayer(
            in_features=in_features,
            out_features=out_features,
            config=config,
            layer_name=layer_name
        )

        # 如果使用DoRA，保存原始权重
        if config.use_dora:
            self.lora.original_weights = self.linear.weight.data.clone()

        # 是否启用LoRA
        self.enable_lora = True

        # 偏置处理
        self.bias_mode = config.bias
        if self.bias_mode == "lora_only":
            # 为LoRA添加可训练偏置
            self.lora_bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量

        Returns:
            输出张量
        """
        # 原始线性层输出
        result = self.linear(x)

        # 添加LoRA适配
        if self.enable_lora:
            result = result + self.lora(x)

            # 处理LoRA偏置
            if self.bias_mode == "lora_only":
                result = result + self.lora_bias

        return result

    def merge_weights(self):
        """合并LoRA权重到原始权重"""
        if not self.enable_lora:
            return

        # 获取权重增量
        delta_weight = self.lora.get_delta_weight()

        # 合并权重
        self.linear.weight.data += delta_weight

        # 合并偏置
        if self.bias_mode == "lora_only" and hasattr(self, 'lora_bias'):
            self.linear.bias.data += self.lora_bias.data

        # 禁用LoRA
        self.enable_lora = False

    def unmerge_weights(self):
        """分离LoRA权重"""
        # 这个操作需要保存原始权重，目前简单实现为重新启用LoRA
        self.enable_lora = True

def apply_enhanced_lora(
    model: nn.Module,
    config: LoRAConfig,
    adapter_name: str = "default"
) -> nn.Module:
    """
    将增强型LoRA应用到模型

    Args:
        model: 原始模型
        config: LoRA配置
        adapter_name: 适配器名称

    Returns:
        应用LoRA后的模型
    """
    # 创建模型副本
    model_copy = copy.deepcopy(model)

    # 获取目标模块名称
    target_module_names = config.target_modules
    if target_module_names is None:
        # 默认应用于所有线性层
        target_module_names = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                target_module_names.append(name)

    # 递归替换线性层
    _replace_with_enhanced_lora_recursive(
        model_copy,
        config=config,
        target_module_names=target_module_names,
        adapter_name=adapter_name,
        prefix=""
    )

    # 为模型添加适配器信息
    if not hasattr(model_copy, "_lora_adapters"):
        model_copy._lora_adapters = {}

    model_copy._lora_adapters[adapter_name] = {
        "config": config,
        "target_modules": target_module_names
    }

    return model_copy

def _replace_with_enhanced_lora_recursive(
    module: nn.Module,
    config: LoRAConfig,
    target_module_names: List[str],
    adapter_name: str,
    prefix: str
):
    """
    递归替换线性层为增强型LoRA层

    Args:
        module: 当前模块
        config: LoRA配置
        target_module_names: 目标模块名称列表
        adapter_name: 适配器名称
        prefix: 当前模块路径前缀
    """
    for name, child in module.named_children():
        child_path = f"{prefix}.{name}" if prefix else name

        # 检查是否是目标模块
        is_target = child_path in target_module_names or any(
            child_path.startswith(t) for t in target_module_names
        )

        if is_target and isinstance(child, nn.Linear):
            # 替换为增强型LoRA线性层
            enhanced_lora_layer = EnhancedLoRALinear(
                in_features=child.in_features,
                out_features=child.out_features,
                config=config,
                bias=child.bias is not None,
                layer_name=child_path,
                original_module=child
            )

            # 替换层
            setattr(module, name, enhanced_lora_layer)
            logger.info(f"已将 {child_path} 替换为增强型LoRA层")
        else:
            # 递归处理子模块
            _replace_with_enhanced_lora_recursive(
                child,
                config=config,
                target_module_names=target_module_names,
                adapter_name=adapter_name,
                prefix=child_path
            )

def get_trainable_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """
    获取模型中所有可训练的LoRA参数

    Args:
        model: 应用了LoRA的模型

    Returns:
        可训练参数列表
    """
    params = []

    for module in model.modules():
        if isinstance(module, EnhancedLoRALinear):
            # 添加LoRA参数
            params.extend([module.lora.lora_A, module.lora.lora_B])

            # 添加高级LoRA变体参数
            if hasattr(module.lora, 'lora_E'):
                params.append(module.lora.lora_E)

            if hasattr(module.lora, 'lora_D'):
                params.append(module.lora.lora_D)

            if hasattr(module.lora, 'lora_C'):
                params.append(module.lora.lora_C)

            # 添加LoRA偏置
            if module.bias_mode == "lora_only" and hasattr(module, 'lora_bias'):
                params.append(module.lora_bias)

    return params

def freeze_non_lora_parameters(model: nn.Module):
    """
    冻结模型中所有非LoRA参数

    Args:
        model: 应用了LoRA的模型
    """
    for name, param in model.named_parameters():
        if not any(x in name for x in ['lora_A', 'lora_B', 'lora_E', 'lora_D', 'lora_C', 'lora_bias']):
            param.requires_grad = False

    # 打印可训练参数数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"可训练参数: {trainable_params:,d} / {total_params:,d} ({trainable_params/total_params:.2%})")

def create_enhanced_lora_optimizer(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    optimizer_type: str = "adamw"
) -> torch.optim.Optimizer:
    """
    为LoRA参数创建优化器

    Args:
        model: 应用了LoRA的模型
        lr: 学习率
        weight_decay: 权重衰减
        optimizer_type: 优化器类型，可选['adamw', 'adam', 'sgd']

    Returns:
        优化器
    """
    # 获取LoRA参数
    lora_params = get_trainable_lora_parameters(model)

    # 创建优化器
    if optimizer_type.lower() == "adamw":
        optimizer = torch.optim.AdamW(lora_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == "adam":
        optimizer = torch.optim.Adam(lora_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == "sgd":
        optimizer = torch.optim.SGD(lora_params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"不支持的优化器类型: {optimizer_type}")

    return optimizer

def save_enhanced_lora_weights(
    model: nn.Module,
    save_path: str,
    adapter_name: str = "default",
    metadata: Optional[Dict[str, Any]] = None
):
    """
    保存模型中的LoRA权重

    Args:
        model: 应用了LoRA的模型
        save_path: 保存路径
        adapter_name: 适配器名称
        metadata: 元数据
    """
    try:
        lora_state_dict = {}
        lora_config = None

        # 提取LoRA配置
        if hasattr(model, "_lora_adapters") and adapter_name in model._lora_adapters:
            lora_config = model._lora_adapters[adapter_name]["config"]

        # 提取所有LoRA层的权重
        for name, module in model.named_modules():
            if isinstance(module, EnhancedLoRALinear):
                lora_state_dict[f"{name}.lora_A"] = module.lora.lora_A.data.cpu()
                lora_state_dict[f"{name}.lora_B"] = module.lora.lora_B.data.cpu()

                # 保存高级LoRA变体参数
                if hasattr(module.lora, 'lora_E'):
                    lora_state_dict[f"{name}.lora_E"] = module.lora.lora_E.data.cpu()

                if hasattr(module.lora, 'lora_D'):
                    lora_state_dict[f"{name}.lora_D"] = module.lora.lora_D.data.cpu()

                if hasattr(module.lora, 'lora_C'):
                    lora_state_dict[f"{name}.lora_C"] = module.lora.lora_C.data.cpu()
                    lora_state_dict[f"{name}.lora_D"] = module.lora.lora_D.data.cpu()

                # 保存LoRA偏置
                if module.bias_mode == "lora_only" and hasattr(module, 'lora_bias'):
                    lora_state_dict[f"{name}.lora_bias"] = module.lora_bias.data.cpu()

                # 保存配置信息
                lora_state_dict[f"{name}.alpha"] = module.lora.alpha
                lora_state_dict[f"{name}.r"] = module.lora.r

        # 准备保存数据
        save_data = {
            'lora_state_dict': lora_state_dict,
            'metadata': metadata or {},
            'adapter_name': adapter_name
        }

        # 如果有配置，也保存
        if lora_config:
            save_data['config'] = {
                'r': lora_config.r,
                'alpha': lora_config.alpha,
                'dropout': lora_config.dropout,
                'bias': lora_config.bias,
                'use_rslora': lora_config.use_rslora,
                'use_dora': lora_config.use_dora,
                'use_lokr': lora_config.use_lokr,
                'use_adaptive': lora_config.use_adaptive
            }

        # 创建目录
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 保存权重
        torch.save(save_data, save_path)
        logger.info(f"增强型LoRA权重已保存到: {save_path}")

    except Exception as e:
        logger.error(f"保存增强型LoRA权重失败: {e}")
        raise

def load_enhanced_lora_weights(
    model: nn.Module,
    weights_path: str,
    adapter_name: Optional[str] = None,
    target_modules: Optional[List[str]] = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    加载LoRA权重到模型

    Args:
        model: 目标模型
        weights_path: 权重文件路径
        adapter_name: 适配器名称，如果为None则使用保存的名称
        target_modules: 目标模块列表，如果为None则加载所有

    Returns:
        加载权重后的模型和元数据
    """
    try:
        # 加载权重文件
        checkpoint = torch.load(weights_path, map_location='cpu')
        lora_state_dict = checkpoint['lora_state_dict']
        metadata = checkpoint.get('metadata', {})

        # 获取适配器名称
        if adapter_name is None:
            adapter_name = checkpoint.get('adapter_name', 'default')

        # 加载权重到模型
        for name, module in model.named_modules():
            if not isinstance(module, EnhancedLoRALinear):
                continue

            if target_modules is not None and name not in target_modules:
                continue

            # 加载A矩阵
            if f"{name}.lora_A" in lora_state_dict:
                module.lora.lora_A.data.copy_(lora_state_dict[f"{name}.lora_A"])

            # 加载B矩阵
            if f"{name}.lora_B" in lora_state_dict:
                module.lora.lora_B.data.copy_(lora_state_dict[f"{name}.lora_B"])

            # 加载高级LoRA变体参数
            if f"{name}.lora_E" in lora_state_dict and hasattr(module.lora, 'lora_E'):
                module.lora.lora_E.data.copy_(lora_state_dict[f"{name}.lora_E"])

            if f"{name}.lora_D" in lora_state_dict and hasattr(module.lora, 'lora_D'):
                module.lora.lora_D.data.copy_(lora_state_dict[f"{name}.lora_D"])

            if f"{name}.lora_C" in lora_state_dict and hasattr(module.lora, 'lora_C'):
                module.lora.lora_C.data.copy_(lora_state_dict[f"{name}.lora_C"])

            # 加载LoRA偏置
            if f"{name}.lora_bias" in lora_state_dict and hasattr(module, 'lora_bias'):
                module.lora_bias.data.copy_(lora_state_dict[f"{name}.lora_bias"])

            # 加载配置
            if f"{name}.alpha" in lora_state_dict:
                module.lora.alpha = lora_state_dict[f"{name}.alpha"]
                module.lora.scaling = module.lora.alpha / module.lora.r

            if f"{name}.r" in lora_state_dict:
                module.lora.r = lora_state_dict[f"{name}.r"]

        logger.info(f"增强型LoRA权重已从 {weights_path} 加载")
        return model, metadata

    except Exception as e:
        logger.error(f"加载增强型LoRA权重失败: {e}")
        raise
