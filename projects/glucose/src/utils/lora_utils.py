"""
LoRA工具模块
提供低秩适配(Low-Rank Adaptation)相关功能
"""

import os
import logging
import math
import copy
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class LoRALayer(nn.Module):
    """LoRA适配层"""

    def __init__(self, in_features: int, out_features: int, rank: int = 8,
                 alpha: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x) @ (self.lora_A.T @ self.lora_B.T) * self.scaling


class LoRALinear(nn.Module):
    """LoRA线性层"""

    def __init__(self, in_features: int, out_features: int, rank: int = 8,
                 alpha: float = 1.0, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.lora = LoRALayer(in_features, out_features, rank, alpha, dropout)
        self.enable_lora = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.linear(x)
        if self.enable_lora:
            result = result + self.lora(x)
        return result

    def merge_weights(self):
        if self.enable_lora:
            self.linear.weight.data += (self.lora_B @ self.lora_A) * self.lora.scaling
            self.enable_lora = False

    def unmerge_weights(self):
        self.enable_lora = True


def replace_linear_with_lora(model: nn.Module, target_modules: Optional[List[str]] = None,
    rank: int = 8, alpha: float = 1.0, dropout: float = 0.0,
    exclude_modules: Optional[List[str]] = None) -> nn.Module:
    """将模型中的线性层替换为LoRA线性层"""
    exclude_modules = exclude_modules or []
    model_copy = copy.deepcopy(model)

    def replace_recursive(module: nn.Module, prefix: str = ""):
        for name, child in module.named_children():
            child_path = f"{prefix}.{name}" if prefix else name
            if child_path in exclude_modules:
                continue

            is_target = target_modules is None or child_path in target_modules or \
                any(child_path.startswith(t) for t in target_modules)

            if is_target and isinstance(child, nn.Linear):
                lora_layer = LoRALinear(child.in_features, child.out_features,
                    rank=rank, alpha=alpha, dropout=dropout, bias=child.bias is not None)
                lora_layer.linear.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    lora_layer.linear.bias.data.copy_(child.bias.data)
                setattr(module, name, lora_layer)
            else:
                replace_recursive(child, child_path)

    replace_recursive(model_copy)
    return model_copy


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """合并模型中所有LoRA权重"""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge_weights()
    return model


def unmerge_lora_weights(model: nn.Module) -> nn.Module:
    """分离模型中所有LoRA权重"""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.unmerge_weights()
    return model


def save_lora_weights(model: nn.Module, save_path: str, metadata: Optional[Dict] = None):
    """保存LoRA权重"""
    lora_state_dict = {}

    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state_dict[f"{name}.lora_A"] = module.lora.lora_A.data.cpu()
            lora_state_dict[f"{name}.lora_B"] = module.lora.lora_B.data.cpu()
            lora_state_dict[f"{name}.alpha"] = module.lora.alpha
            lora_state_dict[f"{name}.rank"] = module.lora.rank

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    torch.save({'lora_state_dict': lora_state_dict, 'metadata': metadata or {}}, save_path)
    logger.info(f"LoRA权重已保存到: {save_path}")


def load_lora_weights(model: nn.Module, weights_path: str,
                      target_modules: Optional[List[str]] = None) -> tuple:
    """加载LoRA权重"""
    checkpoint = torch.load(weights_path, map_location='cpu')
    lora_state_dict = checkpoint['lora_state_dict']
    metadata = checkpoint.get('metadata', {})

    for name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        if target_modules is not None and name not in target_modules:
            continue

        if f"{name}.lora_A" in lora_state_dict:
            module.lora.lora_A.data.copy_(lora_state_dict[f"{name}.lora_A"])
        if f"{name}.lora_B" in lora_state_dict:
            module.lora.lora_B.data.copy_(lora_state_dict[f"{name}.lora_B"])
        if f"{name}.alpha" in lora_state_dict:
            module.lora.alpha = lora_state_dict[f"{name}.alpha"]
            module.lora.scaling = module.lora.alpha / module.lora.rank

    logger.info(f"LoRA权重已从 {weights_path} 加载")
    return model, metadata


def get_trainable_params(model: nn.Module) -> List[nn.Parameter]:
    """获取所有LoRA可训练参数"""
    params = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            params.extend([module.lora.lora_A, module.lora.lora_B])
    return params


def freeze_non_lora_params(model: nn.Module):
    """冻结非LoRA参数"""
    for name, param in model.named_parameters():
        if '.lora_A' not in name and '.lora_B' not in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"可训练参数: {trainable:,d} / {total:,d} ({trainable/total:.2%})")


def create_lora_optimizer(model: nn.Module, lr: float = 1e-3, weight_decay: float = 0.01):
    """为LoRA参数创建优化器"""
    return torch.optim.AdamW(get_trainable_params(model), lr=lr, weight_decay=weight_decay)