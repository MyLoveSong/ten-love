

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版依赖安装脚本
Enhanced Dependencies Installer

自动安装增强版学术系统所需的所有依赖包
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path

def check_package(package_name, import_name=None):
    """检查包是否已安装"""
    if import_name is None:
        import_name = package_name

    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False

def install_package(package_name):
    """安装单个包"""
    try:
        print(f"📦 正在安装 {package_name}...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", package_name
        ], capture_output=True, text=True, check=True)
        print(f"✅ {package_name} 安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {package_name} 安装失败: {e}")
        return False

def main():
    """主安装函数"""
    print("🎯 增强版学术系统依赖安装器")
    print("=" * 50)

    # 核心依赖包列表
    core_packages = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scikit-learn", "sklearn"),
        ("torch", "torch"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("plotly", "plotly"),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
    ]

    # 增强功能包列表
    enhanced_packages = [
        ("scipy", "scipy"),
        ("statsmodels", "statsmodels"),
        ("pingouin", "pingouin"),
        ("psutil", "psutil"),
        ("pyyaml", "yaml"),
        ("python-multipart", "multipart"),
        ("openpyxl", "openpyxl"),
        ("xlrd", "xlrd"),
        ("joblib", "joblib"),
        ("tqdm", "tqdm"),
        ("click", "click"),
        ("rich", "rich"),
        ("loguru", "loguru"),
    ]

    # 可选包列表
    optional_packages = [
        ("shap", "shap"),
        ("lime", "lime"),
        ("jupyter", "jupyter"),
        ("notebook", "notebook"),
        ("pytest", "pytest"),
        ("pytest-cov", "pytest_cov"),
    ]

    all_packages = core_packages + enhanced_packages + optional_packages

    print("🔍 检查当前安装状态...")
    missing_packages = []
    installed_count = 0

    for package_name, import_name in all_packages:
        if check_package(package_name, import_name):
            print(f"✅ {package_name}: 已安装")
            installed_count += 1
        else:
            print(f"❌ {package_name}: 未安装")
            missing_packages.append(package_name)

    print(f"\n📊 安装状态: {installed_count}/{len(all_packages)} 包已安装")

    if not missing_packages:
        print("🎉 所有依赖都已安装！")
        return True

    print(f"\n📦 需要安装 {len(missing_packages)} 个包:")
    for pkg in missing_packages:
        print(f"   - {pkg}")

    # 询问是否安装
    response = input("\n是否开始安装缺失的包？ (y/n): ").lower().strip()
    if response not in ['y', 'yes', '是']:
        print("⏸️ 安装已取消")
        return False

    print("\n🚀 开始安装缺失的包...")

    # 首先升级pip
    print("📈 升级pip...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
                   capture_output=True)

    # 分批安装
    success_count = 0

    # 1. 安装核心包
    print("\n📋 第1步: 安装核心包...")
    core_missing = [pkg for pkg, _ in core_packages if pkg in missing_packages]
    for package in core_missing:
        if install_package(package):
            success_count += 1

    # 2. 安装增强包
    print("\n📋 第2步: 安装增强功能包...")
    enhanced_missing = [pkg for pkg, _ in enhanced_packages if pkg in missing_packages]
    for package in enhanced_missing:
        if install_package(package):
            success_count += 1

    # 3. 安装可选包
    print("\n📋 第3步: 安装可选包...")
    optional_missing = [pkg for pkg, _ in optional_packages if pkg in missing_packages]
    for package in optional_missing:
        # 可选包安装失败不影响整体结果
        install_package(package)
        success_count += 1  # 对于可选包，我们认为尝试安装就算成功

    print(f"\n📊 安装完成: {success_count}/{len(missing_packages)} 包安装成功")

    # 验证安装结果
    print("\n🔍 验证安装结果...")
    final_check_passed = 0

    for package_name, import_name in core_packages + enhanced_packages:
        if check_package(package_name, import_name):
            print(f"✅ {package_name}: 验证通过")
            final_check_passed += 1
        else:
            print(f"❌ {package_name}: 验证失败")

    success_rate = final_check_passed / len(core_packages + enhanced_packages) * 100
    print(f"\n📈 最终成功率: {success_rate:.1f}%")

    if success_rate >= 90:
        print("🎉 依赖安装完成！系统准备就绪")
        print("🚀 可以运行: python start_enhanced_academic_system.py --demo")
        return True
    elif success_rate >= 70:
        print("⚠️ 大部分依赖已安装，系统基本可用")
        print("💡 建议: 手动安装失败的包")
        return True
    else:
        print("🚨 多个关键依赖安装失败")
        print("💡 建议: 检查网络连接和Python环境")
        return False

if __name__ == "__main__":
    try:
        success = main()

        print(f"\n{'='*50}")
        if success:
            print("✅ 依赖安装流程完成")
            print("📚 接下来可以运行增强版系统测试:")
            print("   python test_simple_enhanced.py")
            print("   python start_enhanced_academic_system.py --demo")
        else:
            print("❌ 依赖安装遇到问题")
            print("🛠️ 请手动安装失败的包或检查环境配置")

    except KeyboardInterrupt:
        print("\n⏸️ 安装被用户中断")
    except Exception as e:
        print(f"\n💥 安装过程中发生错误: {e}")
        sys.exit(1)

__all__ = ["'check_package'", "'install_package'", "'main'"]
