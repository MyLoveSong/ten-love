import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# 1. 数据加载
data = pd.read_csv("2.csv")

# 2. 编码与类型转换
if 'gender' in data.columns:
    data['gender'] = LabelEncoder().fit_transform(data['gender'])
if 'diabetes_type' in data.columns:
    data['diabetes_type'] = data['diabetes_type'].astype(int)
if 'pregnant' in data.columns:
    data['pregnant'] = data['pregnant'].astype(int)

# 3. 特征选择（根据数据列动态调整）
exclude_cols = ['blood_glucose']
feature_cols = [col for col in data.columns if col not in exclude_cols]
target_col = 'blood_glucose'

# 4. 数据缩放
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X = scaler_X.fit_transform(data[feature_cols])
y = scaler_y.fit_transform(data[[target_col]])

# 5. 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转为Tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# 6. 构建MLP模型
class MLPRegressor(nn.Module):
    def __init__(self, input_size):
        super(MLPRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.model(x)

# 初始化模型
model = MLPRegressor(input_size=X.shape[1])
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 7. 训练模型
epochs = 300
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = loss_fn(output, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# 8. 预测与反缩放
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor).numpy()

y_pred_inv = scaler_y.inverse_transform(y_pred)
y_test_inv = scaler_y.inverse_transform(y_test)

# 9. 可视化结果
plt.figure(figsize=(10, 6))
plt.plot(y_test_inv, label='True', linestyle='-', marker='o')
plt.plot(y_pred_inv, label='Predicted', linestyle='--', marker='x')
plt.title('Blood Glucose Prediction (MLP)')
plt.xlabel('Sample Index')
plt.ylabel('Blood Glucose')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 10. 误差指标
mae = mean_absolute_error(y_test_inv, y_pred_inv)
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
print(f"\nMAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
