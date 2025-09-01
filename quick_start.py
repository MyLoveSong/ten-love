#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
血糖预测与智能饮食推荐系统 - 快速启动脚本
"""

import os
import sys
import subprocess
import importlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def check_dependencies():
    """检查依赖包是否安装"""
    print("🔍 检查依赖包...")
    
    required_packages = [
        'torch', 'pandas', 'numpy', 'sklearn', 
        'matplotlib', 'seaborn', 'fastapi', 'uvicorn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 需要安装以下包:")
        for package in missing_packages:
            print(f"   pip install {package}")
        return False
    
    print("✅ 所有依赖包已安装！")
    return True

def check_data_files():
    """检查数据文件是否存在"""
    print("\n📊 检查数据文件...")
    
    data_files = [
        "2.csv",
        "数据集1.csv", 
        "数据集2.csv",
        "数据集3.csv",
        "数据集4.csv"
    ]
    
    existing_files = []
    missing_files = []
    
    for file in data_files:
        if os.path.exists(file):
            print(f"✅ {file}")
            existing_files.append(file)
        else:
            print(f"❌ {file} - 文件不存在")
            missing_files.append(file)
    
    if existing_files:
        print(f"\n📈 找到 {len(existing_files)} 个数据文件")
        return existing_files[0]  # 返回第一个可用的数据文件
    
    print("❌ 未找到任何数据文件")
    return None

def analyze_data(data_file):
    """分析数据文件"""
    print(f"\n📊 分析数据文件: {data_file}")
    
    try:
        data = pd.read_csv(data_file)
        print(f"✅ 数据加载成功")
        print(f"   行数: {len(data)}")
        print(f"   列数: {len(data.columns)}")
        print(f"   列名: {list(data.columns)}")
        
        # 显示前几行数据
        print(f"\n📋 数据预览:")
        print(data.head())
        
        # 基本统计信息
        print(f"\n📈 基本统计信息:")
        print(data.describe())
        
        return data
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return None

def run_basic_model():
    """运行基础模型"""
    print("\n🚀 运行基础血糖预测模型...")
    
    try:
        # 检查blood2.py是否存在
        if os.path.exists("blood2.py"):
            print("✅ 找到基础模型文件 blood2.py")
            
            # 运行模型
            result = subprocess.run([sys.executable, "blood2.py"], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ 基础模型运行成功！")
                print("📊 输出结果:")
                print(result.stdout)
            else:
                print("❌ 基础模型运行失败:")
                print(result.stderr)
        else:
            print("❌ 未找到 blood2.py 文件")
            
    except Exception as e:
        print(f"❌ 运行基础模型时出错: {e}")

def run_gluformer_model():
    """运行GluFormer模型"""
    print("\n🧠 运行GluFormer模型...")
    
    try:
        if os.path.exists("gluformer_model.py"):
            print("✅ 找到GluFormer模型文件")
            
            # 检查是否有数据文件
            data_file = check_data_files()
            if not data_file:
                print("❌ 没有可用的数据文件，跳过GluFormer训练")
                return
            
            print("🎯 开始训练GluFormer模型...")
            print("⏳ 这可能需要几分钟时间...")
            
            # 运行模型训练
            result = subprocess.run([sys.executable, "gluformer_model.py"], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ GluFormer模型训练成功！")
                print("📊 训练输出:")
                print(result.stdout)
            else:
                print("❌ GluFormer模型训练失败:")
                print(result.stderr)
        else:
            print("❌ 未找到 gluformer_model.py 文件")
            
    except Exception as e:
        print(f"❌ 运行GluFormer模型时出错: {e}")

def test_api_service():
    """测试API服务"""
    print("\n🌐 测试API服务...")
    
    try:
        if os.path.exists("api_service.py"):
            print("✅ 找到API服务文件")
            print("🚀 启动API服务...")
            print("📝 服务将在 http://localhost:8000 启动")
            print("📖 API文档将在 http://localhost:8000/docs 可用")
            print("⏹️  按 Ctrl+C 停止服务")
            
            # 启动API服务
            subprocess.run([sys.executable, "api_service.py"])
            
        else:
            print("❌ 未找到 api_service.py 文件")
            
    except KeyboardInterrupt:
        print("\n⏹️  API服务已停止")
    except Exception as e:
        print(f"❌ 启动API服务时出错: {e}")

def create_sample_data():
    """创建示例数据（如果没有真实数据）"""
    print("\n📝 创建示例数据...")
    
    # 创建示例数据
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'gender': np.random.choice([0, 1], n_samples),
        'age': np.random.normal(50, 15, n_samples).astype(int),
        'BMI': np.random.normal(25, 5, n_samples),
        'blood_pressure': np.random.normal(120, 20, n_samples).astype(int),
        'fasting_glucose': np.random.normal(6.0, 1.5, n_samples),
        'postprandial_glucose': np.random.normal(8.5, 2.0, n_samples),
        'HbA1c': np.random.normal(6.5, 1.0, n_samples),
        'insulin': np.random.normal(18, 8, n_samples),
        'cholesterol': np.random.normal(200, 40, n_samples),
        'LDL': np.random.normal(120, 30, n_samples),
        'HDL': np.random.normal(55, 15, n_samples),
        'triglycerides': np.random.normal(150, 50, n_samples),
        'physical_activity': np.random.choice([1, 2, 3], n_samples),
        'sleep_quality': np.random.choice([1, 2, 3, 4], n_samples),
        'stress_level': np.random.choice([1, 2, 3], n_samples),
        'diabetes_type': np.random.choice([1, 2], n_samples),
        'pregnant': np.zeros(n_samples),
        'blood_glucose': np.random.normal(8.0, 2.0, n_samples)
    }
    
    # 确保数据在合理范围内
    data['age'] = np.clip(data['age'], 20, 80)
    data['BMI'] = np.clip(data['BMI'], 18, 40)
    data['blood_pressure'] = np.clip(data['blood_pressure'], 90, 180)
    data['fasting_glucose'] = np.clip(data['fasting_glucose'], 3.0, 15.0)
    data['postprandial_glucose'] = np.clip(data['postprandial_glucose'], 4.0, 20.0)
    data['HbA1c'] = np.clip(data['HbA1c'], 4.0, 12.0)
    data['blood_glucose'] = np.clip(data['blood_glucose'], 3.0, 20.0)
    
    df = pd.DataFrame(data)
    df.to_csv("sample_data.csv", index=False)
    
    print("✅ 示例数据已创建: sample_data.csv")
    print(f"   包含 {len(df)} 条记录")
    print(f"   包含 {len(df.columns)} 个特征")
    
    return df

def show_project_status():
    """显示项目状态"""
    print("\n📊 项目状态概览")
    print("=" * 50)
    
    # 检查文件
    files = [
        ("全流程说明.txt", "项目规划文档"),
        ("清单.md", "数据需求清单"),
        ("blood2.py", "基础模型"),
        ("gluformer_model.py", "GluFormer模型"),
        ("multimodal_features.py", "多模态特征提取"),
        ("api_service.py", "API服务"),
        ("项目进度与下一步计划.md", "进度跟踪")
    ]
    
    for file, description in files:
        if os.path.exists(file):
            print(f"✅ {file} - {description}")
        else:
            print(f"❌ {file} - {description} (缺失)")
    
    print("\n🎯 下一步建议:")
    print("1. 运行 GluFormer 模型训练")
    print("2. 测试 API 服务功能")
    print("3. 收集多模态数据")
    print("4. 开发前端界面")

def main():
    """主函数"""
    print("🎯 血糖预测与智能饮食推荐系统 - 快速启动")
    print("=" * 60)
    
    # 检查依赖
    if not check_dependencies():
        print("\n❌ 请先安装缺失的依赖包")
        return
    
    # 检查数据文件
    data_file = check_data_files()
    if not data_file:
        print("\n📝 未找到数据文件，创建示例数据...")
        data = create_sample_data()
        data_file = "sample_data.csv"
    
    # 分析数据
    data = analyze_data(data_file)
    
    # 显示项目状态
    show_project_status()
    
    # 用户选择
    print("\n🎮 请选择要执行的操作:")
    print("1. 运行基础模型 (blood2.py)")
    print("2. 运行GluFormer模型训练")
    print("3. 启动API服务")
    print("4. 查看项目文档")
    print("5. 退出")
    
    while True:
        try:
            choice = input("\n请输入选择 (1-5): ").strip()
            
            if choice == "1":
                run_basic_model()
            elif choice == "2":
                run_gluformer_model()
            elif choice == "3":
                test_api_service()
            elif choice == "4":
                print("\n📚 项目文档:")
                print("- 全流程说明.txt: 详细的项目实施指南")
                print("- 清单.md: 数据采集需求清单")
                print("- 项目进度与下一步计划.md: 当前进度和计划")
                print("- gluformer_model.py: GluFormer模型实现")
                print("- multimodal_features.py: 多模态特征提取")
                print("- api_service.py: API服务实现")
            elif choice == "5":
                print("👋 再见！")
                break
            else:
                print("❌ 无效选择，请输入 1-5")
                
        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")

if __name__ == "__main__":
    main() 