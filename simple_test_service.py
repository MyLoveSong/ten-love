#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化测试服务：快速验证API功能
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from typing import Dict, Any, Optional, List

app = FastAPI(
    title="血糖预测测试服务",
    description="简化版API服务，用于快速测试",
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

# 模拟模型状态
model_ready = True
model_source = "模拟模型"

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "血糖预测测试服务",
        "version": "1.0.0",
        "status": "running",
        "model_ready": model_ready
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "model_ready": model_ready,
        "model_source": model_source
    }

@app.get("/training/status")
async def get_training_status():
    """获取训练状态"""
    return {
        "status": "模型已就绪",
        "progress": 100,
        "model_ready": model_ready,
        "model_source": model_source
    }

@app.post("/predict/glucose")
async def predict_glucose(request_data: Dict[str, Any]):
    """血糖预测接口（模拟）"""
    try:
        # 提取关键特征
        age = request_data.get('age', 45)
        bmi = request_data.get('bmi', 24.5)
        fasting_glucose = request_data.get('fasting_glucose', 5.6)
        hba1c = request_data.get('hba1c', 6.2)
        
        # 简单的模拟预测逻辑
        base_glucose = fasting_glucose
        age_factor = (age - 45) * 0.01  # 年龄影响
        bmi_factor = (bmi - 24.5) * 0.05  # BMI影响
        hba1c_factor = (hba1c - 6.2) * 0.3  # HbA1c影响
        
        predicted_glucose = base_glucose + age_factor + bmi_factor + hba1c_factor
        predicted_glucose = max(3.0, min(12.0, predicted_glucose))  # 限制范围
        
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
    cultural_preferences: Optional[List[str]] = None,
    meal_type: Optional[str] = "lunch"
):
    """食物推荐接口（模拟）"""
    try:
        # 基础推荐
        base_recommendations = {
            "breakfast": ["燕麦", "鸡蛋", "牛奶"],
            "lunch": ["鸡胸肉", "蔬菜", "糙米"],
            "dinner": ["鱼肉", "蔬菜汤"],
            "snacks": ["苹果", "坚果"],
            "avoid": []
        }
        
        # 根据血糖水平调整
        if glucose_level > 7.0:
            base_recommendations["avoid"] = ["白米饭", "甜食", "果汁"]
            base_recommendations["lunch"] = ["鸡胸肉", "蔬菜", "豆腐"]
        elif glucose_level < 4.0:
            base_recommendations["snacks"].extend(["香蕉", "全麦面包"])
        
        # 应用文化偏好
        if cultural_preferences:
            if "清真" in cultural_preferences:
                base_recommendations["lunch"].append("羊肉")
            if "素食" in cultural_preferences:
                base_recommendations["lunch"] = ["豆腐", "蔬菜", "糙米"]
        
        # 获取指定餐次的推荐
        meal_recommendations = base_recommendations.get(meal_type or "lunch", [])
        
        return {
            "glucose_level": glucose_level,
            "meal_type": meal_type or "lunch",
            "cultural_preferences": cultural_preferences or [],
            "recommended_foods": meal_recommendations,
            "all_recommendations": base_recommendations,
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
    return {
        "food_categories": {
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
        },
        "cultural_tags": {
            "清真": ["羊肉", "牛肉", "鸡肉"],
            "素食": ["豆腐", "蔬菜", "水果"],
            "川菜": ["辣椒", "花椒", "豆瓣酱"],
            "粤菜": ["蒸鱼", "白切鸡", "点心"]
        }
    }

if __name__ == "__main__":
    print("🚀 启动血糖预测测试服务...")
    print("🌐 API服务将在 http://localhost:8000 启动")
    print("📖 API文档将在 http://localhost:8000/docs 可用")
    print("⏹️  按 Ctrl+C 停止服务")
    
    uvicorn.run(
        "simple_test_service:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    ) 