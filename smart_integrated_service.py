#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能集成服务：自动检查并加载已训练的模型，或训练新模型
"""

import asyncio
import threading
import time
import os
import sys
import pickle
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import torch
import numpy as np
import pandas as pd
import json
import base64
from PIL import Image
import io
from typing import Optional, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# 导入我们的模型
from gluformer_model import GluFormer, load_and_preprocess_data, GluFormerTrainer
from multimodal_features import MultiModalGluFormer, preprocess_image

app = FastAPI(
    title="血糖预测与智能饮食推荐系统 - 智能版",
    description="自动检查并加载已训练模型，或训练新模型的智能系统",
    version="3.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
model = None
scaler_X = None
scaler_y = None
feature_cols = None
multimodal_model = None
training_status = "检查中..."
training_progress = 0
model_ready = False
model_source = "未知"

# 模型文件路径
MODEL_FILES = {
    'model': 'trained_gluformer_model.pth',
    'scaler_X': 'scaler_X.pkl',
    'scaler_y': 'scaler_y.pkl',
    'feature_cols': 'feature_cols.pkl'
}

class GlucosePredictionRequest:
    """血糖预测请求模型"""
    def __init__(self, data: Dict[str, Any]):
        self.gender = data.get('gender', 0)
        self.age = data.get('age', 45)
        self.bmi = data.get('bmi', 24.5)
        self.blood_pressure = data.get('blood_pressure', 120)
        self.fasting_glucose = data.get('fasting_glucose', 5.6)
        self.postprandial_glucose = data.get('postprandial_glucose', 7.8)
        self.hba1c = data.get('hba1c', 6.2)
        self.insulin = data.get('insulin', 15)
        self.cholesterol = data.get('cholesterol', 180)
        self.ldl = data.get('ldl', 110)
        self.hdl = data.get('hdl', 50)
        self.triglycerides = data.get('triglycerides', 150)
        self.physical_activity = data.get('physical_activity', 2)
        self.sleep_quality = data.get('sleep_quality', 3)
        self.stress_level = data.get('stress_level', 2)
        self.diabetes_type = data.get('diabetes_type', 2)
        self.pregnant = data.get('pregnant', 0)

class FoodRecommendation:
    """食物推荐类"""
    def __init__(self):
        # 食物数据库（简化版）
        self.food_database = {
            "低GI食物": {
                "蔬菜": ["菠菜", "西兰花", "胡萝卜", "黄瓜", "西红柿"],
                "水果": ["苹果", "梨", "橙子", "草莓", "蓝莓"],
                "蛋白质": ["鸡胸肉", "鱼肉", "豆腐", "鸡蛋"],
                "谷物": ["燕麦", "糙米", "全麦面包"]
            },
            "中GI食物": {
                "谷物": ["白米", "面条", "土豆"],
                "水果": ["香蕉", "葡萄", "芒果"]
            },
            "高GI食物": {
                "甜食": ["糖果", "蛋糕", "冰淇淋"],
                "饮料": ["可乐", "果汁", "奶茶"]
            }
        }
        
        # 文化标签
        self.cultural_tags = {
            "清真": ["羊肉", "牛肉", "鸡肉"],
            "素食": ["豆腐", "蔬菜", "水果"],
            "川菜": ["辣椒", "花椒", "豆瓣酱"],
            "粤菜": ["蒸鱼", "白切鸡", "点心"]
        }
    
    def get_recommendations(self, glucose_level: float, cultural_preferences: Optional[List[str]] = None) -> Dict[str, Any]:
        """根据血糖水平和文化偏好推荐食物"""
        recommendations = {
            "breakfast": [],
            "lunch": [],
            "dinner": [],
            "snacks": [],
            "avoid": []
        }
        
        # 根据血糖水平调整推荐
        if glucose_level < 5.0:
            # 血糖偏低，推荐适量碳水化合物
            recommendations["breakfast"] = ["燕麦粥", "全麦面包", "牛奶"]
            recommendations["snacks"] = ["苹果", "香蕉"]
        elif glucose_level < 7.0:
            # 血糖正常，推荐均衡饮食
            recommendations["breakfast"] = ["鸡蛋", "全麦面包", "牛奶"]
            recommendations["lunch"] = ["鸡胸肉", "糙米", "西兰花"]
            recommendations["dinner"] = ["鱼肉", "蔬菜沙拉"]
        else:
            # 血糖偏高，推荐低GI食物
            recommendations["breakfast"] = ["燕麦", "鸡蛋", "牛奶"]
            recommendations["lunch"] = ["鸡胸肉", "蔬菜", "豆腐"]
            recommendations["dinner"] = ["鱼肉", "蔬菜汤"]
            recommendations["avoid"] = ["白米饭", "甜食", "果汁"]
        
        # 应用文化偏好
        if cultural_preferences:
            for meal in ["breakfast", "lunch", "dinner"]:
                cultural_foods = []
                for pref in cultural_preferences:
                    if pref in self.cultural_tags:
                        cultural_foods.extend(self.cultural_tags[pref])
                recommendations[meal].extend(cultural_foods[:2])  # 添加2个文化相关食物
        
        return recommendations

def check_existing_model():
    """检查是否存在已训练的模型"""
    global model_source
    
    # 检查所有必需的文件是否存在
    all_files_exist = all(os.path.exists(MODEL_FILES[key]) for key in MODEL_FILES.keys())
    
    if all_files_exist:
        # 检查文件大小，确保不是空文件
        file_sizes = {key: os.path.getsize(MODEL_FILES[key]) for key in MODEL_FILES.keys()}
        if all(size > 100 for size in file_sizes.values()):  # 文件大小至少100字节
            model_source = "已训练模型"
            return True
    
    model_source = "需要训练"
    return False

def load_existing_model():
    """加载已训练的模型"""
    global model, scaler_X, scaler_y, feature_cols, training_status, training_progress, model_ready
    
    try:
        training_status = "正在加载已训练模型..."
        training_progress = 20
        
        # 加载模型
        input_size = 17  # 默认输入大小
        model = GluFormer(input_size=input_size)
        model.load_state_dict(torch.load(MODEL_FILES['model']))
        model.eval()  # 设置为评估模式
        
        training_progress = 40
        
        # 加载预处理器
        with open(MODEL_FILES['scaler_X'], 'rb') as f:
            scaler_X = pickle.load(f)
        
        with open(MODEL_FILES['scaler_y'], 'rb') as f:
            scaler_y = pickle.load(f)
        
        with open(MODEL_FILES['feature_cols'], 'rb') as f:
            feature_cols = pickle.load(f)
        
        training_progress = 80
        
        # 初始化多模态模型
        multimodal_model = MultiModalGluFormer(numerical_dim=input_size)
        
        training_status = "模型加载完成！"
        training_progress = 100
        model_ready = True
        
        print("✅ 已训练模型加载完成！")
        return True
        
    except Exception as e:
        print(f"❌ 加载已训练模型失败: {e}")
        return False

def train_model_in_background():
    """在后台训练模型"""
    global model, scaler_X, scaler_y, feature_cols, training_status, training_progress, model_ready
    
    try:
        training_status = "正在加载数据..."
        training_progress = 10
        
        # 加载数据
        data_file = "处理过的数据集_完整_增强版.csv"
        if not os.path.exists(data_file):
            data_file = "2.csv"  # 备用数据文件
        
        if not os.path.exists(data_file):
            training_status = "错误：找不到数据文件"
            return
        
        training_status = "正在预处理数据..."
        training_progress = 20
        
        # 加载和预处理数据
        data = pd.read_csv(data_file)
        print(f"加载数据: {len(data)} 条记录")
        
        # 数据预处理
        if 'blood_glucose' in data.columns:
            y = data['blood_glucose']
            X = data.drop(columns=['blood_glucose', 'patient_id', 'data_source'], errors='ignore')
        else:
            training_status = "错误：数据格式不正确"
            return
        
        # 处理缺失值
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())
        
        training_status = "正在创建模型..."
        training_progress = 30
        
        # 创建模型
        input_size = len(X.columns)
        model = GluFormer(input_size=input_size)
        
        training_status = "正在训练模型..."
        training_progress = 40
        
        # 创建训练器
        trainer = GluFormerTrainer(model)
        
        # 划分训练集和验证集
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 创建数据加载器
        from torch.utils.data import DataLoader, TensorDataset
        
        # 转换为张量
        X_train_tensor = torch.FloatTensor(np.array(X_train))
        y_train_tensor = torch.FloatTensor(np.array(y_train))
        X_val_tensor = torch.FloatTensor(np.array(X_val))
        y_val_tensor = torch.FloatTensor(np.array(y_val))
        
        # 创建数据集
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # 训练模型
        epochs = 50
        for epoch in range(epochs):
            # 训练一个epoch
            train_loss = trainer.train_epoch(train_loader)
            
            # 验证
            val_loss, predictions, targets = trainer.validate(val_loader)
            
            # 更新进度
            progress = 40 + int((epoch + 1) / epochs * 50)
            training_progress = min(progress, 90)
            training_status = f"训练中... Epoch {epoch+1}/{epochs}, 训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}"
            
            print(f"Epoch {epoch+1}/{epochs}: 训练损失={train_loss:.4f}, 验证损失={val_loss:.4f}")
        
        training_status = "正在保存模型..."
        training_progress = 95
        
        # 保存模型
        torch.save(model.state_dict(), MODEL_FILES['model'])
        
        # 保存预处理器
        from sklearn.preprocessing import StandardScaler
        scaler_X = StandardScaler()
        scaler_X.fit(np.array(X_train))
        scaler_y = StandardScaler()
        scaler_y.fit(np.array(y_train).reshape(-1, 1))
        
        feature_cols = list(X.columns)
        
        # 保存预处理器到文件
        with open(MODEL_FILES['scaler_X'], 'wb') as f:
            pickle.dump(scaler_X, f)
        
        with open(MODEL_FILES['scaler_y'], 'wb') as f:
            pickle.dump(scaler_y, f)
        
        with open(MODEL_FILES['feature_cols'], 'wb') as f:
            pickle.dump(feature_cols, f)
        
        training_status = "训练完成！"
        training_progress = 100
        model_ready = True
        
        print("✅ 模型训练完成并保存！")
        
    except Exception as e:
        training_status = f"训练失败: {str(e)}"
        print(f"❌ 训练失败: {e}")

def initialize_models():
    """初始化模型（智能检查）"""
    global training_status, training_progress, model_ready
    
    print("🔍 检查已训练模型...")
    
    if check_existing_model():
        print("📁 发现已训练模型，正在加载...")
        if load_existing_model():
            print("✅ 模型加载成功！")
            return
        else:
            print("⚠️ 模型加载失败，将重新训练...")
    
    print("🚀 开始训练新模型...")
    
    # 在后台线程中训练模型
    training_thread = threading.Thread(target=train_model_in_background)
    training_thread.daemon = True
    training_thread.start()

# 启动时初始化模型
@app.on_event("startup")
async def start_initialization():
    """启动时初始化模型"""
    global training_status
    
    print("🚀 启动智能集成服务...")
    
    # 在后台线程中初始化模型
    init_thread = threading.Thread(target=initialize_models)
    init_thread.daemon = True
    init_thread.start()

class SmartGlucosePredictionService:
    """
    封装本地血糖预测服务，便于主程序直接调用。
    """
    def __init__(self):
        # 加载模型和预处理器
        from gluformer_model import GluFormer
        import pickle
        import torch
        import numpy as np
        # 加载特征列
        with open('feature_cols.pkl', 'rb') as f:
            self.feature_cols = pickle.load(f)
        # 加载scaler
        with open('scaler_X.pkl', 'rb') as f:
            self.scaler_X = pickle.load(f)
        with open('scaler_y.pkl', 'rb') as f:
            self.scaler_y = pickle.load(f)
        # 加载模型
        self.model = GluFormer(input_size=len(self.feature_cols))
        self.model.load_state_dict(torch.load('trained_gluformer_model.pth'))
        self.model.eval()

    def predict_glucose(self, input_dict: dict) -> float:
        """
        输入：input_dict为特征字典，输出预测血糖值（float）
        """
        import numpy as np
        import torch
        # 构造特征向量
        X = np.array([[input_dict.get(col, 0) for col in self.feature_cols]])
        X_scaled = self.scaler_X.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            y_pred = self.model(X_tensor)
        y_pred = y_pred.numpy().flatten()
        # 反归一化
        y_pred_inv = self.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        return float(y_pred_inv[0])

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "血糖预测与智能饮食推荐系统 - 智能版",
        "version": "3.0.0",
        "status": "running",
        "training_status": training_status,
        "model_ready": model_ready,
        "model_source": model_source
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "multimodal_model_loaded": multimodal_model is not None,
        "training_status": training_status,
        "training_progress": training_progress,
        "model_ready": model_ready,
        "model_source": model_source
    }

@app.get("/training/status")
async def get_training_status():
    """获取训练状态"""
    return {
        "status": training_status,
        "progress": training_progress,
        "model_ready": model_ready,
        "model_source": model_source
    }

@app.post("/predict/glucose")
async def predict_glucose(request_data: Dict[str, Any]):
    """血糖预测接口"""
    global model_ready, model
    
    if not model_ready:
        raise HTTPException(status_code=503, detail="模型正在初始化中，请稍后再试")
    if model is None:
        raise HTTPException(status_code=500, detail="模型未加载，请稍后再试")
    try:
        # 解析请求数据
        prediction_request = GlucosePredictionRequest(request_data)
        # 构建特征向量
        features = np.array([
            prediction_request.gender,
            prediction_request.age,
            prediction_request.bmi,
            prediction_request.blood_pressure,
            prediction_request.fasting_glucose,
            prediction_request.postprandial_glucose,
            prediction_request.hba1c,
            prediction_request.insulin,
            prediction_request.cholesterol,
            prediction_request.ldl,
            prediction_request.hdl,
            prediction_request.triglycerides,
            prediction_request.physical_activity,
            prediction_request.sleep_quality,
            prediction_request.stress_level,
            prediction_request.diabetes_type,
            prediction_request.pregnant
        ]).reshape(1, -1)
        # 数据预处理
        if scaler_X is not None:
            features = scaler_X.transform(features)
        # 模型预测
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features)
            prediction = model(features_tensor)
            predicted_glucose = prediction.item()
        # 反标准化
        if scaler_y is not None:
            predicted_glucose = scaler_y.inverse_transform([[predicted_glucose]])[0][0]
        # 血糖水平评估
        glucose_status = "正常"
        if predicted_glucose < 3.9:
            glucose_status = "偏低"
        elif predicted_glucose > 6.1:
            glucose_status = "偏高"
        return {
            "predicted_glucose": round(predicted_glucose, 2),
            "glucose_status": glucose_status,
            "confidence": 0.85,
            "model_source": model_source,
            "recommendations": {
                "immediate": "建议监测血糖变化",
                "long_term": "保持规律作息和饮食"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")

@app.post("/recommend/food")
async def recommend_food(
    glucose_level: float,
    cultural_preferences: Optional[List[str]] = Form(None),
    meal_type: Optional[str] = Form("lunch")
):
    """食物推荐接口"""
    try:
        # 初始化推荐器
        recommender = FoodRecommendation()
        
        # 获取推荐
        recommendations = recommender.get_recommendations(glucose_level, cultural_preferences)
        
        # 根据餐次类型返回相应推荐
        meal_recommendations = recommendations.get(meal_type or "lunch", [])
        
        return {
            "glucose_level": glucose_level,
            "meal_type": meal_type,
            "cultural_preferences": cultural_preferences or [],
            "recommended_foods": meal_recommendations,
            "all_recommendations": recommendations,
            "nutritional_advice": {
                "carbohydrates": "适量摄入",
                "protein": "充足摄入",
                "fiber": "增加摄入",
                "sugar": "限制摄入"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推荐失败: {str(e)}")

@app.get("/food/database")
async def get_food_database():
    """获取食物数据库"""
    recommender = FoodRecommendation()
    return {
        "food_categories": recommender.food_database,
        "cultural_tags": recommender.cultural_tags
    }

@app.post("/retrain")
async def retrain_model():
    """重新训练模型"""
    global model_ready, training_status, training_progress
    
    if model_ready:
        # 删除现有模型文件
        for file_path in MODEL_FILES.values():
            if os.path.exists(file_path):
                os.remove(file_path)
        
        # 重置状态
        model_ready = False
        training_status = "准备重新训练..."
        training_progress = 0
        
        # 开始重新训练
        training_thread = threading.Thread(target=train_model_in_background)
        training_thread.daemon = True
        training_thread.start()
        
        return {"message": "开始重新训练模型", "status": "training_started"}
    else:
        raise HTTPException(status_code=400, detail="模型正在训练中，无法重新训练")

if __name__ == "__main__":
    print("🚀 启动血糖预测智能集成服务...")
    print("🔍 系统将自动检查已训练模型")
    print("📊 如果没有模型，将自动训练新模型")
    print("🌐 API服务将在 http://localhost:8000 启动")
    print("📖 API文档将在 http://localhost:8000/docs 可用")
    print("📊 训练状态可在 http://localhost:8000/training/status 查看")
    print("🔄 重新训练可在 http://localhost:8000/retrain 触发")
    print("⏹️  按 Ctrl+C 停止服务")
    
    uvicorn.run(
        "smart_integrated_service:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    ) 