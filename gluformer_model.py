import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class GluFormerDataset(Dataset):
    """血糖预测数据集类"""
    def __init__(self, X, y, sequence_length=10):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.X) - self.sequence_length
        
    def __getitem__(self, idx):
        # 创建序列数据
        X_seq = self.X[idx:idx + self.sequence_length]
        y_target = self.y[idx + self.sequence_length]
        return X_seq, y_target

class CrossAttention(nn.Module):
    """交叉注意力机制"""
    def __init__(self, hidden_size, num_heads=8):
        super(CrossAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x1, x2):
        batch_size = x1.size(0)
        
        # 计算Q, K, V
        Q = self.query(x1).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        K = self.key(x2).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        V = self.value(x2).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        
        # 计算注意力权重
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_size)
        attention_weights = F.softmax(scores, dim=-1)
        
        # 应用注意力
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        
        return self.output(context)

class GluFormer(nn.Module):
    """GluFormer: 基于LSTM+GRU融合的血糖预测模型"""
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(GluFormer, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层 - 处理长期依赖
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # GRU层 - 处理短期依赖
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # 交叉注意力机制
        self.cross_attention = CrossAttention(hidden_size)
        
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 输出层
        self.output = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1)
        )
        
        # 注意力权重可视化
        self.attention_weights = None
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # LSTM处理
        lstm_out, (lstm_hidden, lstm_cell) = self.lstm(x)
        
        # GRU处理
        gru_out, gru_hidden = self.gru(x)
        
        # 交叉注意力融合
        attended_lstm = self.cross_attention(lstm_out, gru_out)
        attended_gru = self.cross_attention(gru_out, lstm_out)
        
        # 特征融合
        fused_features = torch.cat([attended_lstm, attended_gru], dim=-1)
        fused = self.fusion(fused_features)
        
        # 取最后一个时间步的输出
        output = self.output(fused[:, -1, :])
        
        return output

class GluFormerTrainer:
    """GluFormer训练器"""
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
                
                predictions.extend(outputs.cpu().numpy())
                targets.extend(batch_y.cpu().numpy())
                
        return total_loss / len(val_loader), np.array(predictions), np.array(targets)

def load_and_preprocess_data():
    """加载和预处理数据"""
    print("正在加载数据...")
    
    # 加载数据 - 使用更大的增强版数据集
    data = pd.read_csv("处理过的数据集_完整_增强版.csv")
    print(f"数据加载完成，共 {len(data)} 条记录")
    
    # 数据预处理
    print("正在预处理数据...")
    
    # 编码分类变量
    if 'gender' in data.columns:
        # 处理可能的空值
        data['gender'] = data['gender'].fillna('未知')
        data['gender'] = LabelEncoder().fit_transform(data['gender'].astype(str))
    
    # 确保数值类型
    numeric_columns = ['age', 'BMI', 'blood_pressure', 'fasting_glucose', 
                      'postprandial_glucose', 'HbA1c', 'insulin', 'cholesterol',
                      'LDL', 'HDL', 'triglycerides', 'physical_activity', 
                      'sleep_quality', 'stress_level', 'diabetes_type', 
                      'pregnant', 'blood_glucose']
    
    for col in numeric_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # 处理缺失值
    data = data.fillna(data.mean())
    
    # 特征选择 - 排除不需要的列
    exclude_cols = ['blood_glucose', 'patient_id', 'data_source']
    feature_cols = [col for col in data.columns if col not in exclude_cols]
    target_col = 'blood_glucose'
    
    print(f"特征数量: {len(feature_cols)}")
    print(f"目标变量: {target_col}")
    
    # 数据标准化
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X = scaler_X.fit_transform(data[feature_cols])
    y = scaler_y.fit_transform(data[[target_col]]).flatten()
    
    # 保存特征列名和缩放器，以便后续使用
    import pickle
    with open('feature_cols.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)
    with open('scaler_X.pkl', 'wb') as f:
        pickle.dump(scaler_X, f)
    with open('scaler_y.pkl', 'wb') as f:
        pickle.dump(scaler_y, f)
    
    return X, y, scaler_X, scaler_y, feature_cols

def create_sequences(X, y, sequence_length=10):
    """创建序列数据"""
    X_sequences = []
    y_sequences = []
    
    for i in range(len(X) - sequence_length):
        X_sequences.append(X[i:i + sequence_length])
        y_sequences.append(y[i + sequence_length])
    
    return np.array(X_sequences), np.array(y_sequences)

def train_gluformer():
    """训练GluFormer模型"""
    print("🚀 开始训练GluFormer模型...")
    
    # 加载数据
    X, y, scaler_X, scaler_y, feature_cols = load_and_preprocess_data()
    
    # 创建序列数据
    sequence_length = 10
    X_seq, y_seq = create_sequences(X, y, sequence_length)
    
    print(f"序列数据形状: X={X_seq.shape}, y={y_seq.shape}")
    
    # 数据划分
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42, shuffle=False
    )
    
    # 创建数据加载器
    train_dataset = GluFormerDataset(X_train, y_train, sequence_length)
    test_dataset = GluFormerDataset(X_test, y_test, sequence_length)
    
    batch_size = 64  # 增加批量大小以加速训练
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化模型
    input_size = X.shape[1]
    hidden_size = 128
    num_layers = 2
    dropout = 0.3  # 增加dropout以减少过拟合
    
    model = GluFormer(input_size=input_size, hidden_size=hidden_size, 
                      num_layers=num_layers, dropout=dropout)
    trainer = GluFormerTrainer(model)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"模型配置: hidden_size={hidden_size}, num_layers={num_layers}, dropout={dropout}")
    
    # 训练模型
    epochs = 200  # 增加最大训练轮数
    train_losses = []
    val_losses = []
    
    # 早停参数
    patience = 20
    min_delta = 0.001
    best_val_loss = float('inf')
    counter = 0
    best_model_state = None
    
    print("开始训练...")
    for epoch in range(epochs):
        train_loss = trainer.train_epoch(train_loader)
        val_loss, val_pred, val_true = trainer.validate(test_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # 学习率调度
        trainer.scheduler.step(val_loss)
        
        # 打印进度
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f}")
        
        # 早停检查
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            counter = 0
            # 保存最佳模型状态
            best_model_state = model.state_dict().copy()
            print(f"Epoch {epoch:3d}: 验证损失改善，保存最佳模型 (损失: {best_val_loss:.6f})")
        else:
            counter += 1
            if counter >= patience:
                print(f"早停: {patience} 个epoch内验证损失未改善")
                break
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("已加载最佳模型状态")
    
    # 最终评估
    print("\n模型评估...")
    final_val_loss, final_pred, final_true = trainer.validate(test_loader)
    
    # 反标准化预测结果
    final_pred_original = scaler_y.inverse_transform(final_pred.reshape(-1, 1)).flatten()
    final_true_original = scaler_y.inverse_transform(final_true.reshape(-1, 1)).flatten()
    
    # 计算指标
    mae = mean_absolute_error(final_true_original, final_pred_original)
    rmse = np.sqrt(mean_squared_error(final_true_original, final_pred_original))
    r2 = r2_score(final_true_original, final_pred_original)
    
    print(f"\n最终评估结果:")
    print(f"   MAE: {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   R²: {r2:.4f}")
    
    # 保存模型
    torch.save(model.state_dict(), 'trained_gluformer_model.pth')
    print("模型已保存到 'trained_gluformer_model.pth'")
    
    # 可视化结果
    visualize_results(train_losses, val_losses, final_true_original, final_pred_original)
    
    return model, scaler_X, scaler_y, feature_cols

def visualize_results(train_losses, val_losses, true_values, pred_values):
    """可视化训练结果"""
    plt.figure(figsize=(20, 16))
    
    # 1. 训练损失曲线
    plt.subplot(2, 3, 1)
    plt.plot(train_losses, label='训练损失', color='blue', linewidth=2)
    plt.plot(val_losses, label='验证损失', color='red', linewidth=2)
    plt.title('训练过程损失曲线', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 2. 预测vs真实值散点图
    plt.subplot(2, 3, 2)
    plt.scatter(true_values, pred_values, alpha=0.6, color='green', s=50, edgecolor='white')
    min_val = min(true_values.min(), pred_values.min())
    max_val = max(true_values.max(), pred_values.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.title('预测值 vs 真实值', fontsize=14, fontweight='bold')
    plt.xlabel('真实血糖值', fontsize=12)
    plt.ylabel('预测血糖值', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 3. 预测结果时间序列
    plt.subplot(2, 3, 3)
    # 仅显示前100个样本点，以避免过度拥挤
    sample_size = min(100, len(true_values))
    indices = np.arange(sample_size)
    plt.plot(indices, true_values[:sample_size], label='真实值', color='blue', linewidth=2)
    plt.plot(indices, pred_values[:sample_size], label='预测值', color='red', linewidth=2)
    plt.title('血糖预测结果 (前100个样本)', fontsize=14, fontweight='bold')
    plt.xlabel('样本索引', fontsize=12)
    plt.ylabel('血糖值', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 4. 误差分布
    plt.subplot(2, 3, 4)
    errors = pred_values - true_values
    sns.histplot(errors, bins=30, kde=True, color='orange', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
    plt.title('预测误差分布', fontsize=14, fontweight='bold')
    plt.xlabel('预测误差', fontsize=12)
    plt.ylabel('频次', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 5. 残差图
    plt.subplot(2, 3, 5)
    plt.scatter(pred_values, errors, alpha=0.6, color='purple', s=50, edgecolor='white')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    plt.title('残差图', fontsize=14, fontweight='bold')
    plt.xlabel('预测值', fontsize=12)
    plt.ylabel('残差 (预测值-真实值)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 6. 误差累积分布
    plt.subplot(2, 3, 6)
    abs_errors = np.abs(errors)
    sorted_errors = np.sort(abs_errors)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    plt.plot(sorted_errors, cumulative, 'b-', linewidth=2)
    plt.title('误差累积分布', fontsize=14, fontweight='bold')
    plt.xlabel('绝对误差', fontsize=12)
    plt.ylabel('累积概率', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加整体标题
    plt.suptitle('GluFormer 血糖预测模型评估结果', fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig('glucose_prediction_results.png', dpi=300, bbox_inches='tight')
    
    # 另外生成一个散点图，专门用于论文或报告
    plt.figure(figsize=(10, 8))
    plt.scatter(true_values, pred_values, alpha=0.7, color='#1f77b4', s=60, edgecolor='white')
    min_val = min(true_values.min(), pred_values.min())
    max_val = max(true_values.max(), pred_values.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    # 添加回归方程和R²值
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(true_values, pred_values)
    r2 = float(r_value) * float(r_value)  # 明确转换为float类型
    equation = f"y = {slope:.4f}x + {intercept:.4f}"
    plt.annotate(f"{equation}\nR² = {r2:.4f}", 
                xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
                fontsize=12, ha='left', va='top')
    
    plt.title('真实血糖值 vs 预测血糖值', fontsize=16, fontweight='bold')
    plt.xlabel('真实血糖值 (mmol/L)', fontsize=14)
    plt.ylabel('预测血糖值 (mmol/L)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('glucose_prediction_scatter.png', dpi=300, bbox_inches='tight')
    
    print("可视化结果已保存为 'glucose_prediction_results.png' 和 'glucose_prediction_scatter.png'")
    plt.show()

if __name__ == "__main__":
    print("GluFormer血糖预测模型训练")
    print("=" * 50)
    
    # 训练模型
    model, scaler_X, scaler_y, feature_cols = train_gluformer()
    
    print("\n训练完成！模型已保存。")
    print("结果图表已保存为 'gluformer_results.png'") 