#!/usr/bin/env python3
"""
依赖安装脚本
用于安装多模态血糖预测系统所需的依赖包
"""

import subprocess
import sys
import os

def install_package(package):
    """安装单个包"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} 安装成功")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ {package} 安装失败")
        return False

def main():
    print("🔧 多模态血糖预测系统 - 依赖安装")
    print("=" * 50)
    
    # 核心依赖（必需）
    core_packages = [
        "torch>=1.9.0",
        "numpy>=1.21.0", 
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "Pillow>=8.3.0"
    ]
    
    # 可选依赖
    optional_packages = [
        "opencv-python>=4.5.0",
        "transformers>=4.20.0",
        "torchvision>=0.10.0",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "requests>=2.25.0",
        "joblib>=1.1.0"
    ]
    
    print("📦 安装核心依赖...")
    core_success = True
    for package in core_packages:
        if not install_package(package):
            core_success = False
    
    if not core_success:
        print("❌ 核心依赖安装失败，请检查网络连接或手动安装")
        return
    
    print("\n📦 安装可选依赖...")
    optional_success = True
    for package in optional_packages:
        if not install_package(package):
            optional_success = False
            print(f"⚠️  {package} 安装失败，但系统仍可运行（功能受限）")
    
    print("\n" + "=" * 50)
    if core_success:
        print("✅ 核心依赖安装完成！系统可以运行")
        if optional_success:
            print("✅ 所有依赖安装完成！系统功能完整")
        else:
            print("⚠️  部分可选依赖安装失败，但核心功能可用")
    else:
        print("❌ 依赖安装失败，请手动安装")
    
    print("\n📝 下一步:")
    print("   1. 运行 python test_multimodal.py 测试模型")
    print("   2. 运行 python multimodal_features.py 查看完整功能")
    print("   3. 运行 python api_service.py 启动API服务")

if __name__ == "__main__":
    main() 