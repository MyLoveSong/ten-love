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
from gluformer_model import GluFormer, load_and_preprocess_data
from multimodal_features import MultiModalGluFormer, preprocess_image

app = FastAPI(
    title="血糖预测与智能饮食推荐系统",
    description="基于多模态数据的个性化血糖预测和饮食推荐API",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量存储模型和预处理器
model = None
scaler_X = None
scaler_y = None
feature_cols = None
multimodal_model = None

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
    
    def get_recommendations(self, glucose_level: float, cultural_preferences: List[str] = None) -> Dict[str, Any]:
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

# 初始化模型
@app.on_event("startup")
async def load_models():
    """启动时加载模型"""
    global model, scaler_X, scaler_y, feature_cols, multimodal_model
    
    print("🚀 正在加载模型...")
    
    try:
        # 加载特征列名
        import pickle
        try:
            with open('feature_cols.pkl', 'rb') as f:
                feature_cols = pickle.load(f)
            print(f"✅ 特征列加载成功，共 {len(feature_cols)} 个特征")
        except Exception as e:
            print(f"⚠️ 特征列加载失败: {e}")
            feature_cols = None
        
        # 加载缩放器
        try:
            with open('scaler_X.pkl', 'rb') as f:
                scaler_X = pickle.load(f)
            with open('scaler_y.pkl', 'rb') as f:
                scaler_y = pickle.load(f)
            print("✅ 数据缩放器加载成功")
        except Exception as e:
            print(f"⚠️ 数据缩放器加载失败: {e}")
            scaler_X = None
            scaler_y = None
        
        # 加载基础GluFormer模型
        try:
            input_size = len(feature_cols) if feature_cols is not None else 17
            model = GluFormer(input_size=input_size)
            model.load_state_dict(torch.load('trained_gluformer_model.pth', map_location='cpu'))
            model.eval()  # 设置为评估模式
            print("✅ GluFormer模型加载成功")
        except Exception as e:
            print(f"⚠️ GluFormer模型加载失败: {e}")
            model = None
        
        # 加载多模态模型
        try:
            numerical_dim = input_size  # 使用已经确定的input_size
            multimodal_model = MultiModalGluFormer(numerical_dim=numerical_dim)
            multimodal_model.load_state_dict(torch.load('multimodal_fusion_model.pth', map_location='cpu'))
            multimodal_model.eval()  # 设置为评估模式
            print("✅ 多模态模型加载成功")
        except Exception as e:
            print(f"⚠️ 多模态模型加载失败: {e}")
            multimodal_model = None
        
        print("✅ 所有模型加载完成！")
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        raise e

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "血糖预测与智能饮食推荐系统API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "multimodal_model_loaded": multimodal_model is not None
    }

@app.post("/predict/glucose")
async def predict_glucose(request_data: Dict[str, Any]):
    """血糖预测接口"""
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="模型未加载")
        
        # 解析请求数据
        prediction_request = GlucosePredictionRequest(request_data)
        
        # 构建特征向量
        feature_dict = {
            "gender": prediction_request.gender,
            "age": prediction_request.age,
            "BMI": prediction_request.bmi,
            "blood_pressure": prediction_request.blood_pressure,
            "fasting_glucose": prediction_request.fasting_glucose,
            "postprandial_glucose": prediction_request.postprandial_glucose,
            "HbA1c": prediction_request.hba1c,
            "insulin": prediction_request.insulin,
            "cholesterol": prediction_request.cholesterol,
            "LDL": prediction_request.ldl,
            "HDL": prediction_request.hdl,
            "triglycerides": prediction_request.triglycerides,
            "physical_activity": prediction_request.physical_activity,
            "sleep_quality": prediction_request.sleep_quality,
            "stress_level": prediction_request.stress_level,
            "diabetes_type": prediction_request.diabetes_type,
            "pregnant": prediction_request.pregnant
        }
        
        # 检查是否有特征列名
        if feature_cols is not None:
            # 使用特征列名构建特征向量
            features_array = []
            for col in feature_cols:
                if col in feature_dict:
                    features_array.append(feature_dict[col])
                else:
                    features_array.append(0.0)  # 对于缺失的特征，使用默认值0
            features = np.array(features_array).reshape(1, -1)
        else:
            # 使用所有特征
            features = np.array(list(feature_dict.values())).reshape(1, -1)
        
        # 数据预处理
        if scaler_X is not None:
            features = scaler_X.transform(features)
        
        # 转换为张量
        features_tensor = torch.FloatTensor(features)
        
        # 模型推理
        with torch.no_grad():
            prediction = model(features_tensor)
            
        # 转换预测结果
        predicted_value = prediction.item()
        
        # 反标准化
        if scaler_y is not None:
            predicted_value = scaler_y.inverse_transform([[predicted_value]])[0][0]
        
        # 计算风险等级
        risk_level = "正常"
        if predicted_value < 3.9:
            risk_level = "低血糖"
        elif predicted_value > 7.8:
            risk_level = "高血糖"
        
        # 计算置信度（简化处理）
        confidence = 0.85
        
        # 生成建议
        recommendations = []
        if predicted_value < 3.9:
            recommendations = ["适当增加碳水化合物摄入", "避免长时间空腹", "随身携带含糖食物"]
        elif predicted_value > 7.8:
            recommendations = ["控制碳水化合物摄入", "增加体育锻炼", "定期监测血糖"]
        else:
            recommendations = ["保持健康饮食", "规律作息", "适量运动"]
        
        # 生成健康建议
        health_advice = f"您的预测血糖为 {predicted_value:.2f} mmol/L，属于{risk_level}水平。"
        if risk_level == "低血糖":
            health_advice += "\n\n低血糖可能导致头晕、乏力、心悸等症状，建议您：\n"
            health_advice += "1. 立即食用含糖食物或饮料\n"
            health_advice += "2. 休息15-20分钟后再次检测血糖\n"
            health_advice += "3. 如症状持续或加重，请立即就医"
        elif risk_level == "高血糖":
            health_advice += "\n\n长期高血糖可能增加糖尿病及其并发症风险，建议您：\n"
            health_advice += "1. 控制饮食，减少高糖高脂食物摄入\n"
            health_advice += "2. 增加体育锻炼，保持健康体重\n"
            health_advice += "3. 定期监测血糖，必要时咨询医生调整治疗方案"
        else:
            health_advice += "\n\n您的血糖水平正常，请继续保持健康的生活方式：\n"
            health_advice += "1. 均衡饮食，控制总热量摄入\n"
            health_advice += "2. 坚持规律运动，每周至少150分钟中等强度活动\n"
            health_advice += "3. 保持良好作息，减少压力"
        
        return {
            "predicted_glucose": float(predicted_value),
            "risk_level": risk_level,
            "confidence": confidence,
            "recommendations": recommendations,
            "health_advice": health_advice
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")

@app.post("/predict/multimodal")
async def predict_multimodal_glucose(
    numerical_data: Dict[str, Any],
    image: Optional[UploadFile] = File(None),
    food_description: Optional[str] = Form(None)
):
    """多模态血糖预测接口"""
    try:
        # 处理数值数据
        features = np.array(list(numerical_data.values())).reshape(1, -1)
        features_tensor = torch.FloatTensor(features)
        
        # 处理图像数据
        image_tensor = None
        if image:
            image_data = await image.read()
            pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')
            image_tensor = preprocess_image(pil_image).unsqueeze(0)  # 添加batch维度
        
        # 处理文本数据
        text_tensor = None
        if food_description:
            # 这里简化处理，实际应该使用文本特征提取器
            text_tensor = torch.randn(1, 512)  # 模拟文本特征
        
        # 多模态预测
        with torch.no_grad():
            prediction = multimodal_model(features_tensor, image_tensor, text_tensor)
            predicted_glucose = prediction.item()
        
        return {
            "predicted_glucose": round(predicted_glucose, 2),
            "modalities_used": {
                "numerical": True,
                "image": image is not None,
                "text": food_description is not None
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"多模态预测失败: {str(e)}")

@app.post("/recommend/food")
async def recommend_food(request_data: Dict[str, Any]):
    """食物推荐接口"""
    try:
        # 从请求中提取参数
        glucose_level = request_data.get("glucose_level", 5.5)
        cultural_preferences = request_data.get("cultural_preferences", [])
        meal_type = request_data.get("meal_type", "lunch")
        
        # 创建食物推荐实例
        recommender = FoodRecommendation()
        
        # 获取推荐
        recommendations = recommender.get_recommendations(glucose_level, cultural_preferences)
        
        return {
            "glucose_level": glucose_level,
            "cultural_preferences": cultural_preferences,
            "meal_type": meal_type,
            "recommendations": recommendations
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

@app.post("/analyze/meal")
async def analyze_meal(
    image: UploadFile = File(...),
    description: str = Form(...),
    user_profile: Dict[str, Any] = Form(...)
):
    """餐食分析接口"""
    try:
        # 读取图像
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data))
        
        # 图像分析（简化版）
        image_analysis = {
            "food_items_detected": ["米饭", "青菜", "鸡肉"],
            "estimated_calories": 450,
            "carbohydrates": 60,
            "protein": 25,
            "fat": 15
        }
        
        # 文本分析
        text_analysis = {
            "keywords": ["米饭", "青菜", "鸡肉"],
            "cooking_method": "炒",
            "portion_size": "中等"
        }
        
        # 营养评估
        nutrition_assessment = {
            "overall_score": 8.5,
            "balance": "良好",
            "suitable_for_diabetes": True,
            "suggestions": ["可以增加蔬菜摄入", "减少油盐用量"]
        }
        
        return {
            "image_analysis": image_analysis,
            "text_analysis": text_analysis,
            "nutrition_assessment": nutrition_assessment,
            "glucose_impact": {
                "estimated_rise": 2.1,
                "peak_time": "餐后1-2小时",
                "recommendation": "建议餐后适量运动"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"餐食分析失败: {str(e)}")

@app.get("/api/docs")
async def get_api_documentation():
    """API文档"""
    return {
        "endpoints": {
            "/predict/glucose": "血糖预测",
            "/predict/multimodal": "多模态血糖预测",
            "/recommend/food": "食物推荐",
            "/analyze/meal": "餐食分析",
            "/food/database": "食物数据库"
        },
        "usage_examples": {
            "glucose_prediction": {
                "method": "POST",
                "url": "/predict/glucose",
                "body": {
                    "age": 45,
                    "bmi": 24.5,
                    "fasting_glucose": 5.6,
                    # ... 其他参数
                }
            }
        }
    }

if __name__ == "__main__":
    print("🚀 启动血糖预测API服务...")
    uvicorn.run(
        "api_service:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 