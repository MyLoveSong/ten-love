#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行修复后的测试
"""

import sys
import os
import traceback

def run_test(test_name, module_name):
    """运行指定的测试模块"""
    print(f"\n🧪 运行 {test_name} 测试...")
    try:
        # 尝试导入测试模块
        __import__(module_name)
        module = sys.modules[module_name]
        
        # 运行测试函数
        if hasattr(module, 'main'):
            module.main()
        elif hasattr(module, f'test_{module_name}'):
            getattr(module, f'test_{module_name}')()
        else:
            # 尝试找到任何以test_开头的函数
            test_functions = [f for f in dir(module) if f.startswith('test_')]
            if test_functions:
                getattr(module, test_functions[0])()
            else:
                raise AttributeError(f"在{module_name}中找不到测试函数")
        
        print(f"✅ {test_name} 测试通过")
        return True
    except Exception as e:
        print(f"❌ {test_name} 测试失败")
        print(f"错误: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("🧪 运行修复后的测试...")
    
    # 确保当前目录在Python路径中
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # 定义测试
    tests = [
        ('多模态融合', 'test_multimodal'),
        ('智能服务', 'test_smart_api')
    ]
    
    # 运行测试
    results = []
    for test_name, module_name in tests:
        success = run_test(test_name, module_name)
        results.append((test_name, success))
    
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

if __name__ == "__main__":
    main() 