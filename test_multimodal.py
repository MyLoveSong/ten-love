import torch
import torch.nn as nn
import numpy as np

class SimpleMultiModalModel(nn.Module):
    """简化的多模态模型，不依赖外部包"""
    def __init__(self, numerical_dim=17, feature_dim=64):
        super(SimpleMultiModalModel, self).__init__()
        
        # 数值特征提取器
        self.numerical_extractor = nn.Sequential(
            nn.Linear(numerical_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, feature_dim)
        )
        
        # 图像特征提取器（模拟）
        self.image_extractor = nn.Sequential(
            nn.Linear(3 * 224 * 224, 256),  # 假设图像大小为224x224
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, feature_dim)
        )
        
        # 文本特征提取器（模拟）
        self.text_extractor = nn.Sequential(
            nn.Linear(128, 256),  # 假设文本编码为128维
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, feature_dim)
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, numerical_data, image_data=None, text_data=None):
        # 提取数值特征
        numerical_features = self.numerical_extractor(numerical_data)
        
        # 提取图像特征
        if image_data is not None:
            image_flat = image_data.view(image_data.size(0), -1)
            image_features = self.image_extractor(image_flat)
        else:
            # 确保在同一设备上
            image_features = torch.zeros(numerical_data.size(0), numerical_features.size(1), device=numerical_data.device)
        
        # 提取文本特征
        if text_data is not None:
            if isinstance(text_data, list):
                # 如果是文本列表，创建随机特征
                text_features = torch.randn(len(text_data), 128, device=numerical_data.device)
            else:
                text_features = text_data
            text_features = self.text_extractor(text_features)
        else:
            # 确保在同一设备上
            text_features = torch.zeros(numerical_data.size(0), numerical_features.size(1), device=numerical_data.device)
        
        # 融合所有特征
        combined_features = torch.cat([numerical_features, image_features, text_features], dim=1)
        
        # 输出预测
        output = self.fusion(combined_features)
        
        return output

def test_simple_multimodal():
    """测试简化的多模态模型"""
    print("🧪 测试简化多模态模型...")
    
    # 创建模型
    model = SimpleMultiModalModel()
    
    # 创建示例数据
    batch_size = 3
    numerical_data = torch.randn(batch_size, 17)
    image_data = torch.randn(batch_size, 3, 224, 224)
    
    # 创建文本描述并模拟文本特征
    text_descriptions = ["食物描述1", "食物描述2", "食物描述3"]
    # 为文本描述创建随机特征向量
    text_features = torch.randn(batch_size, 128)
    
    print(f"📊 输入数据形状:")
    print(f"   数值特征: {numerical_data.shape}")
    print(f"   图像特征: {image_data.shape}")
    print(f"   文本描述: {len(text_descriptions)} 条")
    
    # 前向传播 - 使用文本特征而不是原始文本
    with torch.no_grad():
        output = model(numerical_data, image_data, text_features)
    
    print(f"🎯 输出形状: {output.shape}")
    print(f"📈 预测值: {output.squeeze().numpy()}")
    
    # 测试只有部分模态的情况
    print("\n测试只有数值特征的情况:")
    with torch.no_grad():
        output_num_only = model(numerical_data)
    print(f"🎯 输出形状: {output_num_only.shape}")
    
    print("✅ 简化多模态模型测试完成！")
    
    return model

if __name__ == "__main__":
    test_simple_multimodal() 