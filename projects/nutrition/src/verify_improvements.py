#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证改进效果脚本
检查所有改进模块是否正常工作，验证是否达到预期效果
"""

import json
import sys
from pathlib import Path
import importlib.util
import traceback

# 添加项目根目录
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def check_file_exists(filepath: str) -> tuple:
    """检查文件是否存在"""
    path = Path(filepath)
    exists = path.exists()
    size = path.stat().st_size if exists else 0
    return exists, size

def check_module_import(module_path: str, class_name: str = None) -> tuple:
    """检查模块是否可以导入"""
    try:
        spec = importlib.util.spec_from_file_location("module", module_path)
        if spec is None:
            return False, "无法加载模块规范"

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if class_name:
            if hasattr(module, class_name):
                return True, f"成功导入类和模块"
            else:
                return False, f"模块存在但缺少类: {class_name}"
        else:
            return True, "成功导入模块"
    except Exception as e:
        return False, f"导入错误: {str(e)}"

def check_data_file() -> dict:
    """检查数据文件"""
    data_path = Path("stage1/data/high_health_dishes_expanded.json")
    result = {
        "exists": False,
        "size_mb": 0,
        "sample_count": 0,
        "valid": False,
        "error": None
    }

    if not data_path.exists():
        result["error"] = "数据文件不存在"
        return result

    result["exists"] = True
    result["size_mb"] = data_path.stat().st_size / 1024 / 1024

    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        result["sample_count"] = data.get("total_samples", 0)
        result["valid"] = True
    except Exception as e:
        result["error"] = str(e)

    return result

def verify_modules() -> dict:
    """验证所有模块"""
    modules_to_check = [
        {
            "name": "扩展高分菜品训练集",
            "path": "stage1/expand_high_health_dishes.py",
            "class": None
        },
        {
            "name": "分层校准训练器",
            "path": "stage1/hierarchical_calibration_trainer.py",
            "class": "HierarchicalCalibrationHead"
        },
        {
            "name": "温度缩放校准器",
            "path": "stage1/temperature_scaling_calibrator.py",
            "class": "TemperatureScaling"
        },
        {
            "name": "多尺度特征融合",
            "path": "stage1/multi_scale_feature_fusion.py",
            "class": "MultiScaleFeatureFusion"
        },
        {
            "name": "高级数据增强",
            "path": "stage1/advanced_data_augmentation.py",
            "class": "MixupAugmentation"
        },
        {
            "name": "细粒度特征提取器",
            "path": "stage1/fine_grained_feature_extractor.py",
            "class": "FineGrainedFeatureExtractor"
        },
        {
            "name": "系统化消融实验",
            "path": "stage1/systematic_ablation_experiments.py",
            "class": "AblationExperimentDesigner"
        },
        {
            "name": "对抗训练增强器",
            "path": "stage1/adversarial_training_enhancer.py",
            "class": "FGM"
        }
    ]

    results = {}
    for module_info in modules_to_check:
        name = module_info["name"]
        path = module_info["path"]
        class_name = module_info["class"]

        exists, size = check_file_exists(path)
        if exists:
            import_ok, import_msg = check_module_import(path, class_name)
            results[name] = {
                "file_exists": True,
                "file_size_kb": size / 1024,
                "import_success": import_ok,
                "import_message": import_msg
            }
        else:
            results[name] = {
                "file_exists": False,
                "file_size_kb": 0,
                "import_success": False,
                "import_message": "文件不存在"
            }

    return results

def check_expected_vs_actual() -> dict:
    """对比预期效果和实际效果"""
    expectations = {
        "训练样本数量": {
            "expected": "从6种扩展到20+种",
            "expected_min": 20,
            "unit": "种"
        },
        "数据增强样本": {
            "expected": "35,000+个样本",
            "expected_min": 35000,
            "unit": "个"
        },
        "特征维度": {
            "expected": "36维细粒度特征",
            "expected_value": 36,
            "unit": "维"
        },
        "模块完整性": {
            "expected": "8个核心模块全部实现",
            "expected_value": 8,
            "unit": "个"
        }
    }

    # 检查数据文件
    data_result = check_data_file()

    # 检查模块
    module_results = verify_modules()
    module_count = sum(1 for r in module_results.values() if r["import_success"])

    # 检查特征维度
    try:
        from .fine_grained_feature_extractor import FineGrainedFeatureExtractor
        extractor = FineGrainedFeatureExtractor()
        feature_dim = extractor.get_feature_dimension()
    except:
        feature_dim = 0

    actual_results = {
        "训练样本数量": {
            "actual": data_result.get("sample_count", 0),
            "status": "OK" if data_result.get("sample_count", 0) >= 20 else "FAIL"
        },
        "数据增强样本": {
            "actual": data_result.get("sample_count", 0),
            "status": "OK" if data_result.get("sample_count", 0) >= 35000 else "FAIL"
        },
        "特征维度": {
            "actual": feature_dim,
            "status": "OK" if feature_dim == 36 else "FAIL"
        },
        "模块完整性": {
            "actual": module_count,
            "status": "OK" if module_count == 8 else "FAIL"
        }
    }

    comparison = {}
    for key in expectations:
        exp = expectations[key]
        act = actual_results[key]
        comparison[key] = {
            "expected": exp.get("expected", ""),
            "actual": act["actual"],
            "status": "OK" if act["status"] == "OK" else "FAIL",
            "unit": exp.get("unit", "")
        }

    return comparison

def main():
    """主函数"""
    print("=" * 60)
    print("Stage1模型改进效果验证")
    print("=" * 60)

    # 1. 检查数据文件
    print("\n[1/3] 检查数据文件...")
    data_result = check_data_file()
    if data_result["exists"]:
        print(f"  [OK] 数据文件存在")
        print(f"  - 文件大小: {data_result['size_mb']:.2f} MB")
        print(f"  - 样本数量: {data_result['sample_count']:,} 个")
        if data_result["sample_count"] >= 35000:
            print(f"  [OK] 达到预期: >= 35,000个样本")
        else:
            print(f"  [FAIL] 未达预期: 需要 >= 35,000个样本")
    else:
        print(f"  [FAIL] 数据文件不存在: {data_result.get('error', '未知错误')}")

    # 2. 验证模块
    print("\n[2/3] 验证模块...")
    module_results = verify_modules()
    success_count = 0
    for name, result in module_results.items():
        if result["import_success"]:
            print(f"  [OK] {name}: 导入成功")
            success_count += 1
        else:
            print(f"  [FAIL] {name}: {result['import_message']}")

    print(f"\n  模块完整性: {success_count}/8 ({success_count*100//8}%)")
    if success_count == 8:
        print(f"  [OK] 所有模块均可正常导入")
    else:
        print(f"  [FAIL] 部分模块存在问题")

    # 3. 对比预期效果
    print("\n[3/3] 对比预期效果...")
    comparison = check_expected_vs_actual()
    for key, result in comparison.items():
        status_icon = "[OK]" if result["status"] == "✓" else "[FAIL]"
        expected = result["expected"]
        actual = result["actual"]
        unit = result["unit"]
        print(f"  {status_icon} {key}:")
        print(f"    预期: {expected}")
        print(f"    实际: {actual:,} {unit}")

    # 总结
    print("\n" + "=" * 60)
    print("验证总结")
    print("=" * 60)

    all_passed = (
        data_result.get("sample_count", 0) >= 35000 and
        success_count == 8 and
        all(r["status"] == "OK" for r in comparison.values())
    )

    if all_passed:
        print("[OK] 所有检查通过，达到预期效果！")
        return 0
    else:
        print("[FAIL] 部分检查未通过，需要进一步优化")
        return 1

if __name__ == "__main__":
    exit(main())
