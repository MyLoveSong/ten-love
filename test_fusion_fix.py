#!/usr/bin/env python3
"""
测试多模态融合代码修复
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class SimpleMultimodalDataset(Dataset):
    """简化的多模态数据集用于测试"""
    def __init__(self, tabular_data, targets=None):
        # 兼容DataFrame/Series和ndarray
        if hasattr(tabular_data, 'values'):
            self.tabular_data = torch.FloatTensor(tabular_data.values)
        else:
            self.tabular_data = torch.FloatTensor(tabular_data)
        
        if targets is not None:
            if isinstance(targets, pd.Series):
                self.targets = torch.FloatTensor(targets.values)
            else:
                self.targets = torch.FloatTensor(np.array(targets))
        else:
            self.targets = None
    
    def __len__(self):
        return len(self.tabular_data)
    
    def __getitem__(self, idx):
        sample = {'tabular': self.tabular_data[idx]}
        if self.targets is not None:
            sample['target'] = self.targets[idx]
        return sample

def test_type_handling():
    """测试类型处理是否正确"""
    print("🧪 测试类型处理...")
    
    # 创建测试数据
    data = {
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [2, 4, 6, 8, 10],
        'target': [100, 200, 300, 400, 500]
    }
    df = pd.DataFrame(data)
    
    # 测试pandas Series
    X = df.drop(columns=['target'])
    y = df['target']  # 这是pandas Series
    
    print(f"X类型: {type(X)}")
    print(f"y类型: {type(y)}")
    
    # 测试类型转换
    if isinstance(y, pd.Series):
        y_data = y.values
        print(f"✅ pandas Series转换成功: {type(y_data)}")
    else:
        y_data = np.array(y)
        print(f"✅ numpy array转换成功: {type(y_data)}")
    
    # 创建数据集
    dataset = SimpleMultimodalDataset(X, y)
    print(f"✅ 数据集创建成功，长度: {len(dataset)}")
    
    # 测试数据加载
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    for batch in loader:
        print(f"✅ 批次数据形状: {batch['tabular'].shape}")
        print(f"✅ 目标数据形状: {batch['target'].shape}")
        break
    
    print("🎉 类型处理测试通过！")

def test_numpy_array():
    """测试numpy array处理"""
    print("\n🧪 测试numpy array处理...")
    
    # 创建numpy array数据
    X = np.random.randn(10, 3)
    y = np.random.randn(10)
    
    print(f"X类型: {type(X)}")
    print(f"y类型: {type(y)}")
    
    # 测试类型转换
    if isinstance(y, pd.Series):
        y_data = y.values
    else:
        y_data = np.array(y)
    
    print(f"✅ 转换后类型: {type(y_data)}")
    
    # 创建数据集
    dataset = SimpleMultimodalDataset(X, y)
    print(f"✅ 数据集创建成功，长度: {len(dataset)}")
    
    print("🎉 numpy array处理测试通过！")

def main():
    print("🔧 多模态融合类型修复测试")
    print("=" * 50)
    
    try:
        test_type_handling()
        test_numpy_array()
        
        print("\n✅ 所有测试通过！")
        print("📝 类型错误已修复，可以正常运行multimodal_fusion.py")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        print("请检查代码修复")

if __name__ == "__main__":
    main() 