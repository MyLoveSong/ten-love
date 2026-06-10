#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查训练环境
验证Python版本、PyTorch、CUDA、依赖包等是否满足训练要求
"""

import sys
import platform
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    print(f"Python版本: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("  [FAIL] Python版本过低，需要Python 3.8+")
        return False
    else:
        print("  [OK] Python版本符合要求")
        return True

def check_pytorch():
    """检查PyTorch"""
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")

        # 检查CUDA
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"  [OK] CUDA可用")
            print(f"  CUDA版本: {torch.version.cuda}")
            print(f"  GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print(f"  [WARN] CUDA不可用，将使用CPU训练（速度较慢）")

        return True
    except ImportError:
        print("  [FAIL] PyTorch未安装")
        return False

def check_dependencies():
    """检查依赖包"""
    required_packages = {
        'numpy': 'numpy',
        'json': 'json (内置)',
        'pathlib': 'pathlib (内置)',
        'dataclasses': 'dataclasses (内置)',
        'typing': 'typing (内置)'
    }

    optional_packages = {
        'sklearn': 'scikit-learn (用于Platt Scaling)',
        'torch': 'PyTorch'
    }

    print("\n检查依赖包:")
    all_ok = True

    for module, name in required_packages.items():
        try:
            __import__(module)
            print(f"  [OK] {name}")
        except ImportError:
            print(f"  [FAIL] {name} 未安装")
            all_ok = False

    for module, name in optional_packages.items():
        try:
            __import__(module)
            print(f"  [OK] {name} (可选)")
        except ImportError:
            print(f"  [WARN] {name} 未安装（可选）")

    return all_ok

def check_data_files():
    """检查数据文件"""
    print("\n检查数据文件:")
    data_path = Path("stage1/data/high_health_dishes_expanded.json")

    if data_path.exists():
        size_mb = data_path.stat().st_size / 1024 / 1024
        print(f"  [OK] 训练数据文件存在: {data_path}")
        print(f"  文件大小: {size_mb:.2f} MB")

        # 检查文件内容
        try:
            import json
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"  样本数量: {data.get('total_samples', 0):,}")
            return True
        except Exception as e:
            print(f"  [FAIL] 数据文件格式错误: {e}")
            return False
    else:
        print(f"  [FAIL] 训练数据文件不存在: {data_path}")
        return False

def check_output_directory():
    """检查输出目录"""
    print("\n检查输出目录:")
    output_dir = Path("stage1/outputs")

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"  [OK] 创建输出目录: {output_dir}")
    else:
        print(f"  [OK] 输出目录存在: {output_dir}")

    return True

def check_model_files():
    """检查模型文件"""
    print("\n检查模型文件:")

    required_files = [
        "stage1/integrate_improvements.py",
        "stage1/train_integrated_model.py",
        "stage1/evaluate_integrated_model.py",
        "stage1/fine_grained_feature_extractor.py",
        "stage1/multi_scale_feature_fusion.py",
        "stage1/hierarchical_calibration_trainer.py",
        "stage1/temperature_scaling_calibrator.py",
        "stage1/adversarial_training_enhancer.py",
        "stage1/advanced_data_augmentation.py"
    ]

    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"  [OK] {path.name}")
        else:
            print(f"  [FAIL] {path.name} 不存在")
            all_exist = False

    return all_exist

def check_system_resources():
    """检查系统资源"""
    print("\n检查系统资源:")

    import psutil
    try:
        # CPU信息
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=1)
        print(f"  CPU核心数: {cpu_count}")
        print(f"  CPU使用率: {cpu_percent:.1f}%")

        # 内存信息
        memory = psutil.virtual_memory()
        memory_gb = memory.total / 1024 / 1024 / 1024
        memory_available_gb = memory.available / 1024 / 1024 / 1024
        print(f"  总内存: {memory_gb:.2f} GB")
        print(f"  可用内存: {memory_available_gb:.2f} GB")

        if memory_available_gb < 4:
            print(f"  [WARN] 可用内存较少，可能影响训练")
        else:
            print(f"  [OK] 内存充足")

        return True
    except ImportError:
        print(f"  [WARN] psutil未安装，无法检查系统资源")
        return True  # 不是必需的

def main():
    """主函数"""
    print("=" * 60)
    print("训练环境检查")
    print("=" * 60)

    print(f"\n系统信息:")
    print(f"  操作系统: {platform.system()} {platform.release()}")
    print(f"  架构: {platform.machine()}")

    checks = {
        "Python版本": check_python_version,
        "PyTorch": check_pytorch,
        "依赖包": check_dependencies,
        "数据文件": check_data_files,
        "输出目录": check_output_directory,
        "模型文件": check_model_files,
        "系统资源": check_system_resources
    }

    results = {}
    for name, check_func in checks.items():
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"  [ERROR] 检查失败: {e}")
            results[name] = False

    # 总结
    print("\n" + "=" * 60)
    print("检查总结")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "[OK]" if result else "[FAIL]"
        print(f"{status} {name}")

    print(f"\n通过: {passed}/{total}")

    if passed == total:
        print("\n[OK] 训练环境检查通过，可以开始训练")
        return 0
    else:
        print("\n[FAIL] 部分检查未通过，请修复后重试")
        return 1

if __name__ == "__main__":
    exit(main())
