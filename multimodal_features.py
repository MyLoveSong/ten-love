import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image

# 可选依赖导入
try:
    import cv2
    cv2_available = True
except ImportError:
    cv2_available = False
    print("OpenCV未安装，图像处理功能将受限")

try:
    from transformers import AutoTokenizer, AutoModel
    transformers_available = True
except ImportError:
    transformers_available = False
    print("Transformers未安装，文本处理功能将受限")

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class ImageFeatureExtractor(nn.Module):
    """图像特征提取器 - 基于轻量级CNN"""
    def __init__(self, feature_dim=512):
        super(ImageFeatureExtractor, self).__init__()
        
        # 轻量级CNN架构
        self.conv_layers = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # 第二层卷积
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # 第三层卷积
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # 第四层卷积
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # 特征映射层
        self.feature_mapping = nn.Sequential(
            nn.Linear(256, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
    def forward(self, x):
        # x shape: (batch_size, 3, height, width)
        conv_features = self.conv_layers(x)
        conv_features = conv_features.view(conv_features.size(0), -1)
        features = self.feature_mapping(conv_features)
        return features

class TextFeatureExtractor(nn.Module):
    """文本特征提取器 - 基于BERT或简单编码"""
    def __init__(self, model_name='bert-base-chinese', feature_dim=512):
        super(TextFeatureExtractor, self).__init__()
        
        if transformers_available:
            # 加载预训练BERT模型
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert = AutoModel.from_pretrained(model_name)
            
            # 冻结BERT参数（可选）
            for param in self.bert.parameters():
                param.requires_grad = False
                
            # 特征映射层
            self.feature_mapping = nn.Sequential(
                nn.Linear(768, feature_dim),  # BERT输出维度为768
                nn.ReLU(),
                nn.Dropout(0.3)
            )
        else:
            # 简单的文本编码器（当transformers不可用时）
            self.simple_encoder = nn.Sequential(
                nn.Linear(128, 256),  # 假设文本编码为128维
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, feature_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
        
    def forward(self, text_inputs):
        if transformers_available:
            # 使用BERT
            with torch.no_grad():
                bert_outputs = self.bert(**text_inputs)
            
            # 使用[CLS]标记的输出作为句子表示
            cls_output = bert_outputs.last_hidden_state[:, 0, :]
            features = self.feature_mapping(cls_output)
        else:
            # 使用简单编码器
            if isinstance(text_inputs, dict):
                # 如果是BERT格式的输入，转换为简单编码
                text_features = torch.randn(text_inputs['input_ids'].size(0), 128)
            else:
                # 直接使用输入
                text_features = torch.randn(text_inputs.size(0), 128)
            features = self.simple_encoder(text_features)
        
        return features
    
    def tokenize_text(self, texts):
        """文本预处理"""
        if transformers_available:
            return self.tokenizer(texts, padding=True, truncation=True, 
                                 max_length=128, return_tensors='pt')
        else:
            # 简单的文本编码（模拟）
            batch_size = len(texts) if isinstance(texts, list) else 1
            return torch.randn(batch_size, 128)

class NumericalFeatureExtractor(nn.Module):
    """数值特征提取器"""
    def __init__(self, input_dim, feature_dim=512):
        super(NumericalFeatureExtractor, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
    def forward(self, x):
        return self.feature_extractor(x)

class MultiModalFusion(nn.Module):
    """多模态特征融合模块"""
    def __init__(self, feature_dim=512, fusion_dim=256, num_modalities=3):
        super(MultiModalFusion, self).__init__()
        
        self.feature_dim = feature_dim
        self.fusion_dim = fusion_dim
        self.num_modalities = num_modalities
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim, 
            num_heads=8, 
            dropout=0.1,
            batch_first=True
        )
        
        # 特征融合层
        self.fusion_layers = nn.Sequential(
            nn.Linear(feature_dim * num_modalities, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 模态权重学习
        self.modality_weights = nn.Parameter(torch.ones(num_modalities))
        
    def forward(self, features_list):
        """
        features_list: 包含各个模态特征的列表
        """
        # 确保所有特征维度一致
        batch_size = features_list[0].size(0)
        
        # 应用模态权重
        weighted_features = []
        for i, features in enumerate(features_list):
            weight = F.softmax(self.modality_weights, dim=0)[i]
            weighted_features.append(features * weight)
        
        # 拼接特征
        concatenated = torch.cat(weighted_features, dim=1)
        
        # 注意力融合
        if len(features_list) > 1:
            # 重塑为序列形式用于注意力
            features_seq = torch.stack(features_list, dim=1)  # (batch, num_modalities, feature_dim)
            attended_features, _ = self.attention(features_seq, features_seq, features_seq)
            attended_features = attended_features.mean(dim=1)  # 平均池化
        else:
            attended_features = features_list[0]
        
        # 最终融合
        fused_features = self.fusion_layers(concatenated)
        
        return fused_features, attended_features

class MultiModalGluFormer(nn.Module):
    """多模态GluFormer模型"""
    def __init__(self, numerical_dim, feature_dim=512, hidden_size=128, num_layers=2):
        super(MultiModalGluFormer, self).__init__()
        
        # 特征提取器
        self.image_extractor = ImageFeatureExtractor(feature_dim)
        self.text_extractor = TextFeatureExtractor(feature_dim=feature_dim)
        self.numerical_extractor = NumericalFeatureExtractor(numerical_dim, feature_dim)
        
        # 多模态融合
        self.fusion = MultiModalFusion(feature_dim, hidden_size, num_modalities=3)
        
        # LSTM-GRU融合层
        self.lstm = nn.LSTM(feature_dim, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.gru = nn.GRU(feature_dim, hidden_size, num_layers,
                         batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        
        # 输出层
        self.output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, numerical_data, image_data=None, text_data=None):
        """
        numerical_data: 数值特征 (batch_size, numerical_dim)
        image_data: 图像数据 (batch_size, 3, height, width) 或 None
        text_data: 文本数据 (batch_size,) 或 None
        """
        features_list = []
        
        # 提取数值特征
        numerical_features = self.numerical_extractor(numerical_data)
        features_list.append(numerical_features)
        
        # 提取图像特征（如果有）
        if image_data is not None:
            image_features = self.image_extractor(image_data)
            features_list.append(image_features)
        else:
            # 如果没有图像数据，使用零向量
            batch_size = numerical_data.size(0)
            image_features = torch.zeros(batch_size, numerical_features.size(1)).to(numerical_data.device)
            features_list.append(image_features)
        
        # 提取文本特征（如果有）
        if text_data is not None:
            text_features = self.text_extractor(text_data)
            features_list.append(text_features)
        else:
            # 如果没有文本数据，使用零向量
            batch_size = numerical_data.size(0)
            text_features = torch.zeros(batch_size, numerical_features.size(1)).to(numerical_data.device)
            features_list.append(text_features)
        
        # 多模态融合
        fused_features, attended_features = self.fusion(features_list)
        
        # 序列处理（这里简化为单步预测）
        # 在实际应用中，您可能需要处理时间序列数据
        lstm_out, _ = self.lstm(attended_features.unsqueeze(1))
        gru_out, _ = self.gru(attended_features.unsqueeze(1))
        
        # 融合LSTM和GRU输出
        combined_features = (lstm_out + gru_out).squeeze(1)
        
        # 输出预测
        output = self.output(combined_features)
        
        return output

def preprocess_image(image_path, target_size=(224, 224)):
    """图像预处理"""
    try:
        # 读取图像
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
            
        # 调整大小
        image = image.resize(target_size)
        
        # 转换为numpy数组并归一化
        image_array = np.array(image) / 255.0
        
        # 转换为tensor
        image_tensor = torch.FloatTensor(image_array).permute(2, 0, 1)  # (C, H, W)
        
        return image_tensor
    except Exception as e:
        print(f"图像预处理错误: {e}")
        # 返回默认图像
        return torch.zeros(3, target_size[0], target_size[1])

def create_sample_data():
    """创建示例数据"""
    # 数值特征
    numerical_data = np.random.randn(10, 17)  # 17个数值特征
    
    # 示例图像路径（实际使用时替换为真实路径）
    image_paths = ["sample_food1.jpg", "sample_food2.jpg"]  # 示例路径
    
    # 示例文本描述
    text_descriptions = [
        "今天吃了一碗米饭和青菜",
        "早餐吃了面包和牛奶",
        "午餐有鱼和蔬菜"
    ]
    
    return numerical_data, image_paths, text_descriptions

def test_multimodal_model():
    """测试多模态模型"""
    print("🧪 测试多模态GluFormer模型...")
    
    # 检查依赖可用性
    print(f"📦 依赖检查:")
    print(f"   OpenCV: {'✅ 可用' if cv2_available else '❌ 不可用'}")
    print(f"   Transformers: {'✅ 可用' if transformers_available else '❌ 不可用'}")
    
    # 创建模型
    numerical_dim = 17  # 根据您的数据调整
    model = MultiModalGluFormer(numerical_dim)
    
    # 创建示例数据
    batch_size = 3
    numerical_data = torch.randn(batch_size, numerical_dim)
    
    # 模拟图像数据
    image_data = torch.randn(batch_size, 3, 224, 224)
    
    # 模拟文本数据
    text_descriptions = ["食物描述1", "食物描述2", "食物描述3"]
    
    print(f"📊 输入数据形状:")
    print(f"   数值特征: {numerical_data.shape}")
    print(f"   图像特征: {image_data.shape}")
    print(f"   文本描述: {len(text_descriptions)} 条")
    
    # 前向传播
    with torch.no_grad():
        output = model(numerical_data, image_data, text_descriptions)
    
    print(f"🎯 输出形状: {output.shape}")
    print(f"📈 预测值: {output.squeeze().numpy()}")
    
    return model

def visualize_feature_importance(model, feature_names):
    """可视化特征重要性"""
    # 这里可以添加特征重要性分析
    # 例如使用SHAP或LIME
    pass

if __name__ == "__main__":
    print("🎯 多模态特征提取与融合模块")
    print("=" * 50)
    
    # 测试模型
    model = test_multimodal_model()
    
    print("\n✅ 多模态模型测试完成！")
    print("📝 下一步可以:")
    print("   1. 集成到GluFormer训练流程中")
    print("   2. 添加真实图像和文本数据")
    print("   3. 实现端到端训练") 