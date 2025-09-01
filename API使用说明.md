# 糖尿病多模态血糖预测API使用说明

## 概述

这是一个基于深度学习的多模态血糖预测与健康建议系统，集成了：
- 多模态深度学习模型（表格数据 + 图像 + 文本 + 时序）
- DeepSeek大模型API（个性化健康建议生成）
- FastAPI Web服务框架

## 服务地址

- **API文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/health
- **根路径**: http://localhost:8000/

## 主要接口

### 1. 健康检查
```
GET /health
```
返回服务状态和模型加载情况。

### 2. 血糖预测
```
POST /predict
```

**请求参数**:
```json
{
    "age": 45.0,                    // 年龄（必填）
    "BMI": 25.5,                    // BMI（必填）
    "blood_pressure": 130.0,        // 血压（必填）
    "fasting_glucose": 6.8,         // 空腹血糖（必填）
    "postprandial_glucose": 8.5,    // 餐后血糖（可选）
    "HbA1c": 6.2,                   // 糖化血红蛋白（可选）
    "insulin": 15.0,                // 胰岛素（可选）
    "cholesterol": 200.0,           // 胆固醇（可选）
    "LDL": 120.0,                   // 低密度脂蛋白（可选）
    "HDL": 50.0,                    // 高密度脂蛋白（可选）
    "triglycerides": 150.0,         // 甘油三酯（可选）
    "physical_activity": 3,         // 体力活动等级 1-5（可选，默认3）
    "sleep_quality": 4,             // 睡眠质量等级 1-5（可选，默认3）
    "stress_level": 2,              // 压力水平等级 1-5（可选，默认3）
    "gender": 1,                    // 性别 1=男性, 2=女性（可选，默认1）
    "diabetes_type": 1,             // 糖尿病类型 1=正常, 2=糖尿病（可选，默认1）
    "pregnant": 0,                  // 是否怀孕 0=否, 1=是（可选，默认0）
    "image_features": [0.1, 0.2, ...],           // 64维图像特征（可选）
    "text_features": [0.1, 0.2, ...],            // 32维文本特征（可选）
    "sequence_features": [[0.1, 0.2, ...], ...]  // 24x16时序特征（可选）
}
```

**响应结果**:
```json
{
    "predicted_glucose": 7.2,       // 预测血糖值
    "risk_level": "轻度升高",        // 风险等级
    "confidence": 0.95,             // 预测置信度
    "health_advice": "根据您的血糖水平...",  // AI生成的健康建议
    "recommendations": [            // 建议列表
        "减少精制碳水化合物摄入",
        "增加有氧运动",
        "控制体重",
        "定期监测血糖"
    ]
}
```

### 3. 批量预测
```
POST /batch_predict
```
支持批量预测，输入为预测请求的数组。

## 使用示例

### Python调用示例
```python
import requests

# 准备数据
data = {
    "age": 45.0,
    "BMI": 25.5,
    "blood_pressure": 130.0,
    "fasting_glucose": 6.8,
    "physical_activity": 3,
    "sleep_quality": 4,
    "stress_level": 2
}

# 发送请求
response = requests.post("http://localhost:8000/predict", json=data)
result = response.json()

print(f"预测血糖: {result['predicted_glucose']:.2f} mmol/L")
print(f"风险等级: {result['risk_level']}")
print(f"AI建议: {result['health_advice']}")
```

### JavaScript调用示例
```javascript
const data = {
    age: 45.0,
    BMI: 25.5,
    blood_pressure: 130.0,
    fasting_glucose: 6.8,
    physical_activity: 3,
    sleep_quality: 4,
    stress_level: 2
};

fetch('http://localhost:8000/predict', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify(data)
})
.then(response => response.json())
.then(result => {
    console.log(`预测血糖: ${result.predicted_glucose} mmol/L`);
    console.log(`风险等级: ${result.risk_level}`);
    console.log(`AI建议: ${result.health_advice}`);
});
```

## 风险等级说明

- **正常**: < 6.1 mmol/L
- **轻度升高**: 6.1 - 7.8 mmol/L
- **中度升高**: 7.8 - 11.1 mmol/L
- **重度升高**: > 11.1 mmol/L

## 启动服务

```bash
python api_deployment.py
```

## 测试服务

```bash
python test_api.py
```

## 注意事项

1. 首次启动时会自动加载模型，可能需要几秒钟时间
2. 确保所有必需字段都已填写
3. 可选字段如果不提供，会使用默认值
4. 多模态特征（图像、文本、时序）为可选，不提供时仅使用表格数据
5. AI健康建议需要网络连接以调用DeepSeek API

## 技术特点

- **多模态融合**: 支持表格、图像、文本、时序数据的联合建模
- **深度学习**: 基于LSTM+Transformer的先进架构
- **智能建议**: 集成大模型API，生成个性化健康建议
- **RESTful API**: 标准化的Web服务接口
- **自动文档**: 内置Swagger UI文档
- **批量处理**: 支持单条和批量预测 