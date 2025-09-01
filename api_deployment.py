import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import requests
import json
import uvicorn
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 重新定义模型结构（与训练时一致）
class MultimodalFusionModel(nn.Module):
    def __init__(self, tabular_dim, image_dim=64, text_dim=32, sequence_dim=16, hidden_dim=128):
        super(MultimodalFusionModel, self).__init__()
        
        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim//2)
        )
        
        self.image_encoder = nn.Sequential(
            nn.Linear(image_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.2)
        ) if image_dim > 0 else None
        
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.2)
        ) if text_dim > 0 else None
        
        self.sequence_encoder = nn.LSTM(
            input_size=sequence_dim,
            hidden_size=hidden_dim//2,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        ) if sequence_dim > 0 else None
        
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
        tabular_features = self.tabular_encoder(tabular)
        features = [tabular_features]
        
        if self.image_encoder and image is not None:
            image_features = self.image_encoder(image)
            features.append(image_features)
            
        if self.text_encoder and text is not None:
            text_features = self.text_encoder(text)
            features.append(text_features)
            
        if self.sequence_encoder and sequence is not None:
            sequence_output, _ = self.sequence_encoder(sequence)
            sequence_features = sequence_output[:, -1, :]
            features.append(sequence_features)
        
        combined_features = torch.cat(features, dim=1)
        output = self.fusion_layer(combined_features)
        return output.squeeze()

# 数据模型定义
class PredictionRequest(BaseModel):
    # 基础特征
    age: float
    BMI: float
    blood_pressure: float
    fasting_glucose: float
    postprandial_glucose: Optional[float] = None
    HbA1c: Optional[float] = None
    insulin: Optional[float] = None
    cholesterol: Optional[float] = None
    LDL: Optional[float] = None
    HDL: Optional[float] = None
    triglycerides: Optional[float] = None
    
    # 生活方式特征
    physical_activity: Optional[int] = 3  # 1-5等级
    sleep_quality: Optional[int] = 3      # 1-5等级
    stress_level: Optional[int] = 3       # 1-5等级
    
    # 分类特征
    gender: Optional[int] = 1             # 1=男性, 2=女性
    diabetes_type: Optional[int] = 1      # 1=正常, 2=糖尿病
    pregnant: Optional[int] = 0           # 0=否, 1=是
    
    # 多模态特征（可选）
    image_features: Optional[List[float]] = None  # 64维图像特征
    text_features: Optional[List[float]] = None   # 32维文本特征
    sequence_features: Optional[List[List[float]]] = None  # 24x16时序特征

class PredictionResponse(BaseModel):
    predicted_glucose: float
    risk_level: str
    confidence: float
    health_advice: str
    recommendations: List[str]

# 全局变量
app = FastAPI(
    title="糖尿病多模态血糖预测API",
    description="基于深度学习的多模态血糖预测与健康建议系统",
    version="1.0.0"
)

model = None
scaler = None
feature_columns = None

# DeepSeek API配置
DEEPSEEK_API_KEY = "sk-4b072acfba2440eeb028a56903cfec6a"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

def load_model():
    """加载训练好的模型"""
    global model, scaler, feature_columns
    
    print("正在加载模型...")
    
    # 读取特征工程后的数据来获取特征列
    df = pd.read_csv('处理过的数据集_特征工程版.csv')
    feature_columns = [col for col in df.columns if col != 'blood_glucose']
    
    # 创建模型实例
    model = MultimodalFusionModel(tabular_dim=len(feature_columns))
    
    # 加载模型权重
    model.load_state_dict(torch.load('multimodal_fusion_model.pth', map_location='cpu'))
    model.eval()
    
    # 创建标准化器
    scaler = StandardScaler()
    X = df[feature_columns]
    scaler.fit(X)
    
    print("模型加载完成！")

def generate_health_advice(glucose_level: float, risk_level: str, user_data: dict) -> str:
    """使用DeepSeek API生成个性化健康建议"""
    try:
        prompt = f"""
        作为一位专业的糖尿病管理专家，请为以下患者提供个性化的血糖管理建议：

        患者信息：
        - 年龄: {user_data.get('age', '未知')}岁
        - BMI: {user_data.get('BMI', '未知')}
        - 血压: {user_data.get('blood_pressure', '未知')}mmHg
        - 空腹血糖: {user_data.get('fasting_glucose', '未知')}mmol/L
        - 预测血糖: {glucose_level:.2f}mmol/L
        - 风险等级: {risk_level}
        - 体力活动: {user_data.get('physical_activity', '未知')}级
        - 睡眠质量: {user_data.get('sleep_quality', '未知')}级
        - 压力水平: {user_data.get('stress_level', '未知')}级

        请提供：
        1. 血糖水平分析
        2. 饮食建议（具体食物推荐）
        3. 运动建议（运动类型、时长、频率）
        4. 生活方式改善建议
        5. 需要就医的预警信号

        请用中文回答，语言要通俗易懂，建议要具体可操作。
        """
        
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return f"AI建议生成失败，错误码: {response.status_code}。建议咨询专业医生。"
            
    except Exception as e:
        print(f"DeepSeek API调用失败: {e}")
        return "AI建议生成失败，建议咨询专业医生。"

def get_risk_level(glucose_level: float) -> str:
    """根据血糖水平判断风险等级"""
    if glucose_level < 6.1:
        return "正常"
    elif glucose_level < 7.8:
        return "轻度升高"
    elif glucose_level < 11.1:
        return "中度升高"
    else:
        return "重度升高"

def get_recommendations(risk_level: str) -> List[str]:
    """根据风险等级生成建议列表"""
    recommendations = {
        "正常": [
            "继续保持健康的生活方式",
            "定期监测血糖",
            "均衡饮食，适量运动"
        ],
        "轻度升高": [
            "减少精制碳水化合物摄入",
            "增加有氧运动",
            "控制体重",
            "定期监测血糖"
        ],
        "中度升高": [
            "立即调整饮食结构",
            "增加运动频率",
            "考虑咨询营养师",
            "密切监测血糖变化"
        ],
        "重度升高": [
            "立即就医咨询",
            "严格饮食控制",
            "增加运动量",
            "可能需要药物治疗"
        ]
    }
    return recommendations.get(risk_level, ["建议咨询专业医生"])

@app.on_event("startup")
async def startup_event():
    """服务启动时加载模型"""
    load_model()

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "糖尿病多模态血糖预测API服务",
        "version": "1.0.0",
        "status": "运行中"
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict_glucose(request: PredictionRequest):
    """血糖预测接口"""
    try:
        # 准备输入数据
        input_data = {
            'age': request.age,
            'BMI': request.BMI,
            'blood_pressure': request.blood_pressure,
            'fasting_glucose': request.fasting_glucose,
            'postprandial_glucose': request.postprandial_glucose or 0,
            'HbA1c': request.HbA1c or 0,
            'insulin': request.insulin or 0,
            'cholesterol': request.cholesterol or 0,
            'LDL': request.LDL or 0,
            'HDL': request.HDL or 0,
            'triglycerides': request.triglycerides or 0,
            'physical_activity': request.physical_activity or 3,
            'sleep_quality': request.sleep_quality or 3,
            'stress_level': request.stress_level or 3,
            'gender': request.gender or 1,
            'diabetes_type': request.diabetes_type or 1,
            'pregnant': request.pregnant or 0
        }
        
        # 转换为DataFrame并标准化
        df_input = pd.DataFrame([input_data])
        df_scaled = pd.DataFrame(scaler.transform(df_input), columns=feature_columns)
        
        # 转换为tensor
        tabular_tensor = torch.FloatTensor(df_scaled.values)
        
        # 准备多模态特征
        image_tensor = None
        text_tensor = None
        sequence_tensor = None
        
        if request.image_features:
            image_tensor = torch.FloatTensor([request.image_features])
        if request.text_features:
            text_tensor = torch.FloatTensor([request.text_features])
        if request.sequence_features:
            sequence_tensor = torch.FloatTensor([request.sequence_features])
        
        # 预测
        with torch.no_grad():
            prediction = model(tabular_tensor, image_tensor, text_tensor, sequence_tensor)
            predicted_glucose = prediction.item()
        
        # 判断风险等级
        risk_level = get_risk_level(predicted_glucose)
        
        # 生成建议
        recommendations = get_recommendations(risk_level)
        
        # 生成AI健康建议
        health_advice = generate_health_advice(predicted_glucose, risk_level, input_data)
        
        # 计算置信度（简化版本）
        confidence = 0.95 if abs(predicted_glucose - request.fasting_glucose) < 2 else 0.85
        
        return PredictionResponse(
            predicted_glucose=predicted_glucose,
            risk_level=risk_level,
            confidence=confidence,
            health_advice=health_advice,
            recommendations=recommendations
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")

@app.post("/batch_predict")
async def batch_predict(requests: List[PredictionRequest]):
    """批量预测接口"""
    results = []
    for request in requests:
        try:
            result = await predict_glucose(request)
            results.append(result.dict())
        except Exception as e:
            results.append({"error": str(e)})
    return {"results": results}

if __name__ == "__main__":
    print("启动糖尿病多模态血糖预测API服务...")
    print("API文档地址: http://localhost:8000/docs")
    print("健康检查地址: http://localhost:8000/health")
    uvicorn.run(app, host="0.0.0.0", port=8000) 