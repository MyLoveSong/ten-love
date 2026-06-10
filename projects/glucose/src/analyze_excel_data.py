#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
分析Excel数据文件
检查添加数据1.xlsx和添加数据2.xlsx的结构和内容
判断是否能整理成与现有数据集相同的格式
"""

import pandas as pd
import json
import os
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_excel_file(file_path, file_name):
    """分析Excel文件"""
    logger.info(f"📊 分析文件: {file_name}")

    try:
        # 读取Excel文件
        df = pd.read_excel(file_path)

        logger.info(f"✅ 文件读取成功: {file_name}")
        logger.info(f"📊 数据形状: {df.shape}")
        logger.info(f"📊 列名: {list(df.columns)}")

        # 显示前几行数据
        logger.info(f"📊 前5行数据:")
        print(df.head())

        # 检查数据类型
        logger.info(f"📊 数据类型:")
        print(df.dtypes)

        # 检查缺失值
        logger.info(f"📊 缺失值统计:")
        print(df.isnull().sum())

        # 检查数值列的基本统计
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            logger.info(f"📊 数值列统计:")
            print(df[numeric_cols].describe())

        return df

    except Exception as e:
        logger.error(f"❌ 读取文件失败: {file_name}, 错误: {e}")
        return None

def check_data_compatibility(df, file_name):
    """检查数据兼容性"""
    logger.info(f"🔍 检查数据兼容性: {file_name}")

    # 现有数据集需要的字段
    required_fields = [
        'food_name', 'calories', 'protein', 'carbs', 'fat', 'fiber',
        'sugar', 'sodium', 'nutrition_score'
    ]

    # 可选字段
    optional_fields = [
        'vitamin_a', 'vitamin_c', 'calcium', 'iron', 'potassium',
        'cholesterol', 'acceptance_score', 'cultural_relevance',
        'health_score', 'taste_score', 'texture_score', 'aroma_score', 'visual_score'
    ]

    available_fields = list(df.columns)

    logger.info(f"📊 可用字段: {available_fields}")

    # 检查必需字段
    missing_required = [field for field in required_fields if field not in available_fields]
    available_required = [field for field in required_fields if field in available_fields]

    logger.info(f"✅ 可用的必需字段: {available_required}")
    if missing_required:
        logger.warning(f"⚠️ 缺失的必需字段: {missing_required}")

    # 检查可选字段
    available_optional = [field for field in optional_fields if field in available_fields]
    logger.info(f"✅ 可用的可选字段: {available_optional}")

    # 计算兼容性评分
    compatibility_score = len(available_required) / len(required_fields)

    logger.info(f"📊 兼容性评分: {compatibility_score:.2f}")

    if compatibility_score >= 0.8:
        logger.info("✅ 数据兼容性良好，可以用于实验")
        return True
    elif compatibility_score >= 0.5:
        logger.info("⚠️ 数据兼容性一般，需要补充字段")
        return True
    else:
        logger.warning("❌ 数据兼容性较差，不建议使用")
        return False

def suggest_data_transformation(df, file_name):
    """建议数据转换方案"""
    logger.info(f"💡 建议数据转换方案: {file_name}")

    suggestions = []

    # 检查是否有营养相关字段
    nutrition_keywords = ['calorie', 'protein', 'carb', 'fat', 'fiber', 'sugar', 'sodium']
    available_nutrition = []

    for col in df.columns:
        col_lower = col.lower()
        for keyword in nutrition_keywords:
            if keyword in col_lower:
                available_nutrition.append(col)
                break

    if available_nutrition:
        suggestions.append(f"✅ 发现营养相关字段: {available_nutrition}")
    else:
        suggestions.append("⚠️ 未发现明显的营养相关字段")

    # 检查是否有食物名称字段
    name_keywords = ['name', 'food', 'item', 'product', 'dish']
    name_fields = []

    for col in df.columns:
        col_lower = col.lower()
        for keyword in name_keywords:
            if keyword in col_lower:
                name_fields.append(col)
                break

    if name_fields:
        suggestions.append(f"✅ 发现食物名称字段: {name_fields}")
    else:
        suggestions.append("⚠️ 未发现明显的食物名称字段")

    # 检查数值字段
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        suggestions.append(f"✅ 发现数值字段: {list(numeric_cols)}")

    # 输出建议
    for suggestion in suggestions:
        logger.info(suggestion)

    return suggestions

def main():
    """主函数"""
    print("\n" + "="*80)
    print("🔍 Excel数据分析工具")
    print("检查添加数据1.xlsx和添加数据2.xlsx的结构和兼容性")
    print("="*80 + "\n")

    # 文件路径
    file1_path = "D:\\b比赛\\创新创业\\集成系统\\添加数据1.xlsx"
    file2_path = "D:\\b比赛\\创新创业\\集成系统\\添加数据2.xlsx"

    results = {}

    # 分析第一个文件
    if os.path.exists(file1_path):
        logger.info("📊 开始分析添加数据1.xlsx")
        df1 = analyze_excel_file(file1_path, "添加数据1.xlsx")
        if df1 is not None:
            results['file1'] = {
                'dataframe': df1,
                'compatible': check_data_compatibility(df1, "添加数据1.xlsx"),
                'suggestions': suggest_data_transformation(df1, "添加数据1.xlsx")
            }
    else:
        logger.error(f"❌ 文件不存在: {file1_path}")

    print("\n" + "-"*60 + "\n")

    # 分析第二个文件
    if os.path.exists(file2_path):
        logger.info("📊 开始分析添加数据2.xlsx")
        df2 = analyze_excel_file(file2_path, "添加数据2.xlsx")
        if df2 is not None:
            results['file2'] = {
                'dataframe': df2,
                'compatible': check_data_compatibility(df2, "添加数据2.xlsx"),
                'suggestions': suggest_data_transformation(df2, "添加数据2.xlsx")
            }
    else:
        logger.error(f"❌ 文件不存在: {file2_path}")

    # 总结分析结果
    print("\n" + "="*80)
    print("📊 分析结果总结")
    print("="*80)

    for file_key, result in results.items():
        file_name = "添加数据1.xlsx" if file_key == 'file1' else "添加数据2.xlsx"
        print(f"\n📁 {file_name}:")
        print(f"  兼容性: {'✅ 兼容' if result['compatible'] else '❌ 不兼容'}")
        print(f"  数据形状: {result['dataframe'].shape}")
        print(f"  字段数量: {len(result['dataframe'].columns)}")

    # 最终建议
    print("\n💡 最终建议:")
    compatible_files = [key for key, result in results.items() if result['compatible']]

    if len(compatible_files) > 0:
        print("✅ 以下文件可以用于实验:")
        for file_key in compatible_files:
            file_name = "添加数据1.xlsx" if file_key == 'file1' else "添加数据2.xlsx"
            print(f"  - {file_name}")
        print("\n🚀 建议进行数据格式转换和集成")
    else:
        print("❌ 当前文件格式与实验要求不兼容")
        print("💡 建议检查数据字段和格式")

    print("\n" + "="*80)
    print("✅ 分析完成")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
