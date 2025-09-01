#!/usr/bin/env python3
"""
快速修复导入问题
"""

import subprocess
import sys
import os

def check_and_install_packages():
    """检查并安装缺失的包"""
    print("🔧 检查并修复导入问题...")
    print("=" * 50)
    
    # 正确的包名映射
    package_mapping = {
        'cv2': 'opencv-python',
        'transformers': 'transformers',
        'torch': 'torch',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
        'PIL': 'Pillow'
    }
    
    missing_packages = []
    
    # 检查每个包
    for import_name, package_name in package_mapping.items():
        try:
            if import_name == 'cv2':
                import cv2
            elif import_name == 'transformers':
                import transformers
            elif import_name == 'torch':
                import torch
            elif import_name == 'numpy':
                import numpy
            elif import_name == 'pandas':
                import pandas
            elif import_name == 'sklearn':
                import sklearn
            elif import_name == 'matplotlib':
                import matplotlib
            elif import_name == 'PIL':
                from PIL import Image
                
            print(f"✅ {import_name} 已安装")
        except ImportError:
            print(f"❌ {import_name} 未安装")
            missing_packages.append((import_name, package_name))
    
    if not missing_packages:
        print("\n🎉 所有依赖都已安装！")
        return True
    
    print(f"\n📦 需要安装 {len(missing_packages)} 个包:")
    for import_name, package_name in missing_packages:
        print(f"   {import_name} -> pip install {package_name}")
    
    # 询问是否自动安装
    print("\n是否自动安装缺失的包？(y/n): ", end="")
    try:
        choice = input().lower().strip()
        if choice in ['y', 'yes', '是']:
            return install_missing_packages(missing_packages)
        else:
            print("请手动运行以下命令:")
            for import_name, package_name in missing_packages:
                print(f"pip install {package_name}")
            return False
    except KeyboardInterrupt:
        print("\n取消安装")
        return False

def install_missing_packages(missing_packages):
    """安装缺失的包"""
    print("\n📦 开始安装...")
    success_count = 0
    
    for import_name, package_name in missing_packages:
        try:
            print(f"正在安装 {package_name}...", end=" ")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("✅ 成功")
            success_count += 1
        except subprocess.CalledProcessError:
            print("❌ 失败")
    
    print(f"\n📊 安装结果: {success_count}/{len(missing_packages)} 成功")
    
    if success_count == len(missing_packages):
        print("🎉 所有包安装成功！")
        return True
    else:
        print("⚠️  部分包安装失败，请手动安装")
        return False

def test_imports():
    """测试导入是否正常"""
    print("\n🧪 测试导入...")
    
    try:
        import torch
        print("✅ torch 导入成功")
    except ImportError:
        print("❌ torch 导入失败")
    
    try:
        import cv2
        print("✅ cv2 导入成功")
    except ImportError:
        print("❌ cv2 导入失败")
    
    try:
        import transformers
        print("✅ transformers 导入成功")
    except ImportError:
        print("❌ transformers 导入失败")
    
    try:
        import numpy as np
        print("✅ numpy 导入成功")
    except ImportError:
        print("❌ numpy 导入失败")

def main():
    print("🔧 多模态血糖预测系统 - 导入修复工具")
    print("=" * 50)
    
    # 检查并安装包
    success = check_and_install_packages()
    
    if success:
        # 测试导入
        test_imports()
        
        print("\n📝 下一步:")
        print("   1. 运行 python test_multimodal.py 测试模型")
        print("   2. 运行 python multimodal_features.py 查看完整功能")
        print("   3. 如果仍有问题，请查看 README_依赖说明.md")
    else:
        print("\n📝 手动安装命令:")
        print("   pip install opencv-python  # 安装OpenCV")
        print("   pip install transformers   # 安装Transformers")
        print("   pip install torch          # 安装PyTorch")

if __name__ == "__main__":
    main() 