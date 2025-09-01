import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 检查是否有GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

class MultimodalDataset(Dataset):
    """多模态数据集"""
    def __init__(self, tabular_data, image_features=None, text_features=None, sequence_data=None, targets=None):
        # 兼容DataFrame/Series和ndarray
        if hasattr(tabular_data, 'values'):
            self.tabular_data = torch.FloatTensor(tabular_data.values)
        else:
            self.tabular_data = torch.FloatTensor(tabular_data)
        if image_features is not None:
            self.image_features = torch.FloatTensor(image_features)
        else:
            self.image_features = None
        if text_features is not None:
            self.text_features = torch.FloatTensor(text_features)
        else:
            self.text_features = None
        if sequence_data is not None:
            self.sequence_data = torch.FloatTensor(sequence_data)
        else:
            self.sequence_data = None
        if targets is not None:
            if hasattr(targets, 'values'):
                self.targets = torch.FloatTensor(targets.values)
            else:
                self.targets = torch.FloatTensor(targets)
        else:
            self.targets = None
    def __len__(self):
        return len(self.tabular_data)
    def __getitem__(self, idx):
        sample = {'tabular': self.tabular_data[idx]}
        if self.image_features is not None:
            sample['image'] = self.image_features[idx]
        if self.text_features is not None:
            sample['text'] = self.text_features[idx]
        if self.sequence_data is not None:
            sample['sequence'] = self.sequence_data[idx]
        if self.targets is not None:
            sample['target'] = self.targets[idx]
        return sample

class MultimodalFusionModel(nn.Module):
    """多模态融合模型"""
    def __init__(self, tabular_dim, image_dim=64, text_dim=32, sequence_dim=16, hidden_dim=128):
        super(MultimodalFusionModel, self).__init__()
        
        # 表格数据处理
        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim//2)
        )
        
        # 图像特征处理
        self.image_encoder = nn.Sequential(
            nn.Linear(image_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.2)
        ) if image_dim > 0 else None
        
        # 文本特征处理
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.2)
        ) if text_dim > 0 else None
        
        # 时序数据处理
        self.sequence_encoder = nn.LSTM(
            input_size=sequence_dim,
            hidden_size=hidden_dim//2,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        ) if sequence_dim > 0 else None
        
        # 融合层
        total_features = hidden_dim//2
        if image_dim > 0:
            total_features += hidden_dim//2
        if text_dim > 0:
            total_features += hidden_dim//2
        if sequence_dim > 0:
            total_features += hidden_dim//2
            
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)
        )
        
    def forward(self, tabular, image=None, text=None, sequence=None):
        # 编码表格数据
        tabular_features = self.tabular_encoder(tabular)
        
        # 融合所有模态
        features = [tabular_features]
        
        if self.image_encoder and image is not None:
            image_features = self.image_encoder(image)
            features.append(image_features)
            
        if self.text_encoder and text is not None:
            text_features = self.text_encoder(text)
            features.append(text_features)
            
        if self.sequence_encoder and sequence is not None:
            sequence_output, _ = self.sequence_encoder(sequence)
            sequence_features = sequence_output[:, -1, :]  # 取最后一个时间步
            features.append(sequence_features)
        
        # 拼接所有特征
        combined_features = torch.cat(features, dim=1)
        
        # 输出预测
        output = self.fusion_layer(combined_features)
        return output.squeeze()

def check_and_fix_nan(arr, name):
    arr = np.array(arr)
    if np.isnan(arr).any() or np.isinf(arr).any():
        print(f"警告：{name} 含有NaN或inf，将用0替换！")
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr

def generate_multimodal_data(df, n_samples):
    """生成模拟的多模态数据，并做标准化"""
    print("生成模拟多模态数据...")
    image_features = np.random.normal(0, 1, (n_samples, 64))
    image_features = StandardScaler().fit_transform(image_features)
    image_features = check_and_fix_nan(image_features, 'image_features')
    text_features = np.random.normal(0, 1, (n_samples, 32))
    text_features = StandardScaler().fit_transform(text_features)
    text_features = check_and_fix_nan(text_features, 'text_features')
    sequence_data = np.random.normal(0, 1, (n_samples, 24, 16))
    sequence_data = sequence_data.reshape(-1, 16)
    sequence_data = StandardScaler().fit_transform(sequence_data)
    sequence_data = sequence_data.reshape(n_samples, 24, 16)
    sequence_data = check_and_fix_nan(sequence_data, 'sequence_data')
    return image_features, text_features, sequence_data

def train_multimodal_model(model, train_loader, val_loader, epochs=50, lr=0.0005):
    """训练多模态模型，出现nan提前终止"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            tabular = batch['tabular'].to(device)
            image = batch.get('image', None)
            text = batch.get('text', None)
            sequence = batch.get('sequence', None)
            if image is not None:
                image = image.to(device)
            if text is not None:
                text = text.to(device)
            if sequence is not None:
                sequence = sequence.to(device)
            outputs = model(tabular, image, text, sequence)
            loss = criterion(outputs, batch['target'].to(device))
            if torch.isnan(loss):
                print(f"训练过程中出现NaN，提前终止。Epoch: {epoch+1}")
                return train_losses, val_losses
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                tabular = batch['tabular'].to(device)
                image = batch.get('image', None)
                text = batch.get('text', None)
                sequence = batch.get('sequence', None)
                if image is not None:
                    image = image.to(device)
                if text is not None:
                    text = text.to(device)
                if sequence is not None:
                    sequence = sequence.to(device)
                outputs = model(tabular, image, text, sequence)
                loss = criterion(outputs, batch['target'].to(device))
                if torch.isnan(loss):
                    print(f"验证过程中出现NaN，提前终止。Epoch: {epoch+1}")
                    return train_losses, val_losses
                val_loss += loss.item()
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        scheduler.step(val_losses[-1])
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')
    return train_losses, val_losses

def multimodal_fusion_main():
    """多模态融合主函数"""
    print("=== 开始多模态数据融合与高级建模 ===")
    
    # 读取特征工程后的数据
    df = pd.read_csv('处理过的数据集_特征工程版.csv')
    print(f"读取数据: {len(df)} 条记录")
    
    # 准备表格数据
    target_col = 'blood_glucose'
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    X_scaled = check_and_fix_nan(X_scaled, 'tabular')
    y = check_and_fix_nan(y, 'target')
    
    # 生成模拟多模态数据
    image_features, text_features, sequence_data = generate_multimodal_data(df, len(df))
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    X_train = check_and_fix_nan(X_train, 'X_train')
    X_test = check_and_fix_nan(X_test, 'X_test')
    y_train = check_and_fix_nan(y_train, 'y_train')
    y_test = check_and_fix_nan(y_test, 'y_test')
    
    # 创建数据集
    # 确保y_train和y_test是正确的格式
    if isinstance(y_train, pd.Series):
        y_train_data = y_train.values
    else:
        y_train_data = np.array(y_train)
    
    if isinstance(y_test, pd.Series):
        y_test_data = y_test.values
    else:
        y_test_data = np.array(y_test)
    
    train_dataset = MultimodalDataset(
        X_train, 
        image_features[:len(X_train)], 
        text_features[:len(X_train)], 
        sequence_data[:len(X_train)],
        y_train_data
    )
    test_dataset = MultimodalDataset(
        X_test, 
        image_features[len(X_train):], 
        text_features[len(X_train):], 
        sequence_data[len(X_train):],
        y_test_data
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 创建模型
    model = MultimodalFusionModel(
        tabular_dim=X.shape[1],
        image_dim=64,
        text_dim=32,
        sequence_dim=16
    ).to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 训练模型
    print("开始训练多模态融合模型...")
    train_losses, val_losses = train_multimodal_model(model, train_loader, test_loader, epochs=30)
    
    # 评估模型
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            tabular = batch['tabular'].to(device)
            image = batch.get('image', None)
            text = batch.get('text', None)
            sequence = batch.get('sequence', None)
            
            if image is not None:
                image = image.to(device)
            if text is not None:
                text = text.to(device)
            if sequence is not None:
                sequence = sequence.to(device)
            
            outputs = model(tabular, image, text, sequence)
            predictions.extend(outputs.cpu().numpy())
            targets.extend(batch['target'].numpy())
    
    # 计算指标
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    
    print(f"\n=== 多模态融合模型结果 ===")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R²: {r2:.6f}")
    
    # 保存模型
    torch.save(model.state_dict(), 'multimodal_fusion_model.pth')
    print("多模态融合模型已保存为 multimodal_fusion_model.pth")
    
    # 绘制训练曲线
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(targets, predictions, alpha=0.5)
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Prediction vs True')
    plt.tight_layout()
    plt.savefig('multimodal_results.png')
    plt.close()
    
    print("结果图已保存为 multimodal_results.png")
    
    return model, mse, mae, r2

if __name__ == "__main__":
    multimodal_fusion_main() 