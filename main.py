#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
血糖预测与智能饮食推荐系统 - 主程序
Blood Glucose Prediction and Intelligent Diet Recommendation System - Main Program

功能特性：
1. 智能血糖预测服务
2. 多模态特征融合
3. 时空注意力机制
4. 多尺度特征提取
5. 学术级演示模式
6. 用户交互界面

作者：AI Assistant
版本：2.0
日期：2024
"""

import os
import sys
import time
import traceback
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
import torch
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

def print_banner():
    """打印系统横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║    血糖预测与智能饮食推荐系统 - Blood Glucose Prediction System              ║
║                                                                              ║
║    🎯 智能预测 | 🧠 多模态融合 | ⚡ 实时服务 | 📊 学术级演示                ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def print_menu():
    """打印主菜单"""
    menu = """
📋 主菜单 - 请选择功能：

1️⃣  启动智能血糖预测服务
2️⃣  运行学术级演示模式
3️⃣  多模态特征融合测试
4️⃣  时空注意力机制测试
5️⃣  多尺度特征提取测试
6️⃣  查看系统状态和模型信息
7️⃣  运行完整测试套件
8️⃣  查看操作指南
9️⃣  退出系统

请输入选项 (1-9): """
    return input(menu)

def check_system_status():
    """检查系统状态"""
    print("\n🔍 系统状态检查...")
    
    # 检查必要文件
    required_files = [
        'smart_integrated_service.py',
        'academic_demo.py',
        'multimodal_fusion.py',
        'gluformer_model.py',
        'trained_gluformer_model.pth',
        'scaler_X.pkl',
        'scaler_y.pkl'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ 缺少必要文件: {missing_files}")
        return False
    
    # 检查模型文件大小
    model_file = 'trained_gluformer_model.pth'
    if os.path.exists(model_file):
        size_mb = os.path.getsize(model_file) / (1024 * 1024)
        print(f"✅ 模型文件存在: {model_file} ({size_mb:.1f} MB)")
    
    # 检查数据集
    dataset_files = [
        '处理过的数据集_完整_增强版.csv',
        '处理过的数据集_测试集_增强版.csv'
    ]
    
    for dataset in dataset_files:
        if os.path.exists(dataset):
            size_mb = os.path.getsize(dataset) / (1024 * 1024)
            print(f"✅ 数据集存在: {dataset} ({size_mb:.1f} MB)")
    
    print("✅ 系统状态检查完成")
    return True

def run_smart_service():
    """启动智能血糖预测服务"""
    print("\n🚀 启动智能血糖预测服务...")
    try:
        from smart_integrated_service import SmartGlucosePredictionService
        
        service = SmartGlucosePredictionService()
        print("✅ 服务启动成功！")
        
        feature_cols = service.feature_cols
        print("\n请选择输入方式：")
        print("1. 使用系统演示数据（默认）")
        print("2. 手动输入自己的特征数据")
        mode = input("请输入选项（1或2，回车默认1）：").strip()
        if mode == '2':
            print("\n请依次输入以下特征（直接回车则用默认值1.0）：")
            sample_data = {}
            for col in feature_cols:
                val = input(f"  {col}: ").strip()
                if val == '':
                    sample_data[col] = 1.0
                else:
                    try:
                        sample_data[col] = float(val)
                    except Exception:
                        print("  ⚠️ 输入无效，已用默认值1.0")
                        sample_data[col] = 1.0
            print("\n已采集自定义输入数据：")
            print(sample_data)
            print("\n正在预测...")
            prediction = service.predict_glucose(sample_data)
            print(f"🎯 预测结果: {prediction:.2f} mg/dL")
            print("（本次预测基于您的自定义输入）")
        else:
            # 构造演示数据
            sample_data = {col: 1.0 for col in feature_cols}
            for key in ["glucose", "fasting_glucose", "postprandial_glucose", "insulin", "carbohydrates", "physical_activity", "stress_level", "age", "bmi", "blood_pressure"]:
                if key in sample_data:
                    sample_data[key] = 6.0 if "glucose" in key else 25.0
            print("\n系统将用演示数据进行预测，仅用于功能演示，不代表真实血糖水平。")
            prediction = service.predict_glucose(sample_data)
            print(f"🎯 预测结果: {prediction:.2f} mg/dL")
            print("（本次预测基于系统演示数据）")
        return True
        
    except Exception as e:
        print(f"❌ 服务启动失败: {str(e)}")
        traceback.print_exc()
        return False

def run_academic_demo():
    """运行学术级演示"""
    print("\n🎓 启动学术级演示模式...")
    try:
        import subprocess
        result = subprocess.run([sys.executable, 'academic_demo.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 学术演示运行成功！")
            print(result.stdout)
        else:
            print("❌ 学术演示运行失败:")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ 启动学术演示失败: {str(e)}")
        traceback.print_exc()

def run_multimodal_test():
    """运行多模态融合测试"""
    print("\n🔬 运行多模态特征融合测试...")
    try:
        import subprocess
        result = subprocess.run([sys.executable, 'test_multimodal.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 多模态测试通过！")
            print(result.stdout)
        else:
            print("❌ 多模态测试失败:")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ 多模态测试失败: {str(e)}")
        traceback.print_exc()

def run_attention_test():
    """运行时空注意力测试"""
    print("\n🧠 运行时空注意力机制测试...")
    try:
        import subprocess
        result = subprocess.run([sys.executable, 'test_spatiotemporal_attention.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 时空注意力测试通过！")
            print(result.stdout)
        else:
            print("❌ 时空注意力测试失败:")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ 时空注意力测试失败: {str(e)}")
        traceback.print_exc()

def run_multiscale_test():
    """运行多尺度特征提取测试"""
    print("\n📏 运行多尺度特征提取测试...")
    try:
        import subprocess
        result = subprocess.run([sys.executable, 'test_multiscale_feature_extractor.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 多尺度特征提取测试通过！")
            print(result.stdout)
        else:
            print("❌ 多尺度特征提取测试失败:")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ 多尺度特征提取测试失败: {str(e)}")
        traceback.print_exc()

def run_full_test_suite():
    """运行完整测试套件"""
    print("\n🧪 运行完整测试套件...")
    
    tests = [
        ('多模态融合', 'test_multimodal.py'),
        ('时空注意力', 'test_spatiotemporal_attention.py'),
        ('多尺度特征提取', 'test_multiscale_feature_extractor.py'),
        ('智能服务', 'test_smart_api.py')
    ]
    
    results = []
    
    # 导入测试模块
    import importlib.util
    import sys
    
    for test_name, test_file in tests:
        print(f"\n🔍 运行 {test_name} 测试...")
        try:
            # 尝试导入测试模块
            module_name = test_file.replace('.py', '')
            
            try:
                # 尝试直接导入
                test_module = __import__(module_name)
            except ImportError:
                # 如果直接导入失败，尝试使用spec导入
                spec = importlib.util.spec_from_file_location(module_name, test_file)
                if spec is None:
                    raise ImportError(f"无法加载模块: {test_file}")
                test_module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = test_module
                if spec.loader:
                    spec.loader.exec_module(test_module)
                else:
                    raise ImportError(f"模块加载器为空: {test_file}")
            
            # 运行测试函数
            if hasattr(test_module, 'main'):
                test_module.main()
                success = True
            elif hasattr(test_module, f'test_{module_name}'):
                getattr(test_module, f'test_{module_name}')()
                success = True
            else:
                # 尝试找到任何以test_开头的函数
                test_functions = [f for f in dir(test_module) if f.startswith('test_')]
                if test_functions:
                    getattr(test_module, test_functions[0])()
                    success = True
                else:
                    raise AttributeError(f"在{test_file}中找不到测试函数")
            
            print(f"✅ {test_name} 测试通过")
            results.append((test_name, True))
        except Exception as e:
            print(f"❌ {test_name} 测试失败")
            print(f"   错误: {str(e)}")
            results.append((test_name, False))
    
    # 输出测试结果摘要
    print("\n📊 测试结果总结:")
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"  {test_name}: {status}")
    
    # 计算通过率
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\n🎯 总体结果: {passed}/{total} 测试通过")
    
    return passed == total

def show_guide():
    """显示操作指南"""
    guide = """
📚 操作指南：

1️⃣ 智能血糖预测服务
   - 提供实时血糖预测
   - 支持多模态数据输入
   - 自动模型加载和缓存

2️⃣ 学术级演示模式
   - 展示完整的学术级功能
   - 包含实验设计和结果分析
   - 适合论文和报告展示

3️⃣ 多模态特征融合测试
   - 验证多模态数据融合效果
   - 测试不同融合策略
   - 评估融合性能

4️⃣ 时空注意力机制测试
   - 验证时空注意力机制
   - 测试不同注意力配置
   - 评估注意力权重分布

5️⃣ 多尺度特征提取测试
   - 验证多尺度特征提取
   - 测试不同窗口大小
   - 评估特征提取效果

6️⃣ 系统状态检查
   - 检查必要文件完整性
   - 验证模型和数据状态
   - 诊断系统问题

7️⃣ 完整测试套件
   - 运行所有模块测试
   - 验证系统整体功能
   - 生成测试报告

💡 提示：
- 首次使用建议先运行系统状态检查
- 学术演示适合展示给导师或评委
- 测试套件确保系统稳定性
    """
    print(guide)

def main():
    """主函数"""
    print_banner()
    
    # 检查系统状态
    if not check_system_status():
        print("\n⚠️  系统状态检查发现问题，建议先解决后再继续")
        input("按回车键继续...")
    
    while True:
        try:
            choice = print_menu()
            
            if choice == '1':
                run_smart_service()
            elif choice == '2':
                run_academic_demo()
            elif choice == '3':
                run_multimodal_test()
            elif choice == '4':
                run_attention_test()
            elif choice == '5':
                run_multiscale_test()
            elif choice == '6':
                check_system_status()
            elif choice == '7':
                run_full_test_suite()
            elif choice == '8':
                show_guide()
            elif choice == '9':
                print("\n👋 感谢使用血糖预测系统！再见！")
                break
            else:
                print("❌ 无效选项，请输入 1-9")
            
            input("\n按回车键返回主菜单...")
            
        except KeyboardInterrupt:
            print("\n\n👋 用户中断，退出系统")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {str(e)}")
            traceback.print_exc()
            input("按回车键继续...")

if __name__ == "__main__":
    main() 