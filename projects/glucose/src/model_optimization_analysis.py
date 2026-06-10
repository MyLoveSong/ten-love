#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型优化分析工具
分析R²为负值的原因并提供优化方案
"""

import os
import sys
import json
import logging
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelOptimizationAnalyzer:
    """模型优化分析器"""

    def __init__(self):
        """初始化分析器"""
        self.analysis_results = {}

    def analyze_negative_r2_causes(self) -> Dict[str, Any]:
        """分析R²为负值的原因"""
        logger.info("🔍 分析R²为负值的原因...")

        causes = {
            "数据质量问题": {
                "样本数量不足": "35个样本对于复杂模型可能不够",
                "数据分布不均": "可能存在类别不平衡",
                "特征相关性低": "特征与目标变量相关性可能较低",
                "噪声数据": "可能存在异常值或错误标签"
            },
            "模型复杂度问题": {
                "过拟合": "模型参数过多(192K)相对于样本数(35)",
                "欠拟合": "模型可能无法捕捉数据模式",
                "特征维度不匹配": "不同特征源的维度差异"
            },
            "训练策略问题": {
                "学习率过高": "可能导致训练不稳定",
                "批次大小不当": "批次大小(32)接近样本数(35)",
                "正则化不足": "缺乏有效的正则化机制",
                "早停过早": "可能过早停止训练"
            },
            "特征工程问题": {
                "特征缩放": "不同特征量级差异大",
                "特征选择": "可能存在冗余或无关特征",
                "特征交互": "缺乏特征间的有效交互"
            }
        }

        self.analysis_results["causes"] = causes
        return causes

    def generate_optimization_solutions(self) -> Dict[str, List[str]]:
        """生成优化解决方案"""
        logger.info("💡 生成优化解决方案...")

        solutions = {
            "数据增强策略": [
                "增加训练样本数量 - 收集更多真实数据",
                "数据增强技术 - 使用SMOTE、数据插值等方法",
                "合成数据生成 - 基于现有数据生成合成样本",
                "跨域数据迁移 - 利用其他相关数据集"
            ],
            "模型架构优化": [
                "简化模型结构 - 减少参数数量",
                "使用预训练模型 - 利用迁移学习",
                "集成学习方法 - 使用多个简单模型",
                "正则化技术 - 添加Dropout、L1/L2正则化"
            ],
            "训练策略改进": [
                "调整学习率 - 使用更小的学习率",
                "改进批次策略 - 使用更小的批次大小",
                "数据标准化 - 对特征进行标准化处理",
                "交叉验证 - 使用K折交叉验证"
            ],
            "特征工程优化": [
                "特征选择 - 选择最重要的特征",
                "特征变换 - 使用PCA、特征组合等",
                "特征交互 - 创建特征间的交互项",
                "领域知识 - 利用营养学领域知识"
            ]
        }

        self.analysis_results["solutions"] = solutions
        return solutions

    def create_optimized_model_architecture(self) -> Dict[str, Any]:
        """创建优化后的模型架构"""
        logger.info("🏗️ 设计优化后的模型架构...")

        optimized_architecture = {
            "简化模型": {
                "description": "减少参数数量，避免过拟合",
                "layers": [
                    "输入层: 特征标准化",
                    "隐藏层1: 64个神经元 + ReLU + Dropout(0.3)",
                    "隐藏层2: 32个神经元 + ReLU + Dropout(0.3)",
                    "输出层: 1个神经元"
                ],
                "parameters": "约10K参数 (vs 192K)",
                "regularization": ["Dropout", "L2正则化", "早停"]
            },
            "集成模型": {
                "description": "使用多个简单模型集成",
                "components": [
                    "营养特征模型: 专门处理营养数据",
                    "文化特征模型: 专门处理文化数据",
                    "偏好特征模型: 专门处理偏好数据",
                    "融合层: 加权平均或投票机制"
                ],
                "advantages": ["降低过拟合风险", "提高泛化能力", "可解释性强"]
            },
            "预训练模型": {
                "description": "使用预训练的营养预测模型",
                "approach": [
                    "使用公开的营养预测模型作为基础",
                    "微调最后几层以适应文化适应任务",
                    "冻结大部分参数，只训练少量参数"
                ],
                "benefits": ["减少训练数据需求", "提高模型性能", "加速收敛"]
            }
        }

        self.analysis_results["architecture"] = optimized_architecture
        return optimized_architecture

    def design_data_augmentation_strategy(self) -> Dict[str, Any]:
        """设计数据增强策略"""
        logger.info("📊 设计数据增强策略...")

        augmentation_strategy = {
            "数据收集策略": {
                "目标样本数": "500-1000个样本",
                "数据来源": [
                    "扩展Nutritionix API查询",
                    "增加Spoonacular食谱数据",
                    "收集用户反馈数据",
                    "使用公开营养数据库"
                ],
                "数据平衡": "确保各菜系样本数量均衡"
            },
            "数据增强技术": {
                "SMOTE": "合成少数类过采样技术",
                "数据插值": "在相似样本间插值",
                "噪声注入": "添加适量噪声增加鲁棒性",
                "特征变换": "对现有特征进行变换"
            },
            "质量控制": {
                "数据验证": "验证营养数据的准确性",
                "异常值检测": "识别和处理异常值",
                "一致性检查": "确保数据格式一致",
                "专家审核": "营养学专家审核数据质量"
            }
        }

        self.analysis_results["augmentation"] = augmentation_strategy
        return augmentation_strategy

    def create_optimization_roadmap(self) -> Dict[str, List[str]]:
        """创建优化路线图"""
        logger.info("🗺️ 创建优化路线图...")

        roadmap = {
            "第一阶段 - 数据优化": [
                "1. 收集更多训练数据 (目标: 500+样本)",
                "2. 实施数据标准化和特征工程",
                "3. 进行数据质量检查和清洗",
                "4. 创建训练/验证/测试集划分"
            ],
            "第二阶段 - 模型优化": [
                "1. 简化模型架构 (减少参数)",
                "2. 实施正则化技术",
                "3. 使用交叉验证评估",
                "4. 调整超参数"
            ],
            "第三阶段 - 训练优化": [
                "1. 使用更小的学习率",
                "2. 实施学习率调度",
                "3. 使用早停机制",
                "4. 监控训练过程"
            ],
            "第四阶段 - 评估优化": [
                "1. 使用多个评估指标",
                "2. 进行模型解释性分析",
                "3. 测试模型泛化能力",
                "4. 与基线模型比较"
            ]
        }

        self.analysis_results["roadmap"] = roadmap
        return roadmap

    def generate_immediate_actions(self) -> List[Dict[str, str]]:
        """生成立即可执行的行动"""
        logger.info("⚡ 生成立即可执行的行动...")

        immediate_actions = [
            {
                "action": "数据标准化",
                "description": "对所有特征进行标准化处理",
                "priority": "高",
                "estimated_time": "30分钟"
            },
            {
                "action": "简化模型架构",
                "description": "将模型参数从192K减少到10K以下",
                "priority": "高",
                "estimated_time": "1小时"
            },
            {
                "action": "调整训练参数",
                "description": "降低学习率，减小批次大小",
                "priority": "高",
                "estimated_time": "15分钟"
            },
            {
                "action": "增加数据收集",
                "description": "收集更多真实营养数据",
                "priority": "中",
                "estimated_time": "2小时"
            },
            {
                "action": "实施交叉验证",
                "description": "使用5折交叉验证评估模型",
                "priority": "中",
                "estimated_time": "45分钟"
            }
        ]

        self.analysis_results["immediate_actions"] = immediate_actions
        return immediate_actions

    def save_analysis_report(self) -> None:
        """保存分析报告"""
        output_dir = Path("outputs/model_optimization_analysis")
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "optimization_analysis_report.json", "w", encoding="utf-8") as f:
            json.dump(self.analysis_results, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"📋 分析报告已保存: {output_dir / 'optimization_analysis_report.json'}")

    def run_complete_analysis(self) -> Dict[str, Any]:
        """运行完整分析"""
        logger.info("🚀 开始完整的模型优化分析...")

        # 运行所有分析
        self.analyze_negative_r2_causes()
        self.generate_optimization_solutions()
        self.create_optimized_model_architecture()
        self.design_data_augmentation_strategy()
        self.create_optimization_roadmap()
        self.generate_immediate_actions()

        # 保存分析报告
        self.save_analysis_report()

        logger.info("✅ 模型优化分析完成！")
        return self.analysis_results


def main():
    """主函数"""
    print("\n" + "="*80)
    print("🔍 模型优化分析工具")
    print("分析R²为负值的原因并提供优化方案")
    print("="*80 + "\n")

    analyzer = ModelOptimizationAnalyzer()
    results = analyzer.run_complete_analysis()

    # 输出关键发现
    print("\n📊 关键发现:")
    print("1. 数据样本不足 (35个样本 vs 192K参数)")
    print("2. 模型过于复杂，容易过拟合")
    print("3. 缺乏数据标准化和特征工程")
    print("4. 训练策略需要优化")

    print("\n💡 立即可执行的优化:")
    for action in results["immediate_actions"][:3]:
        print(f"   • {action['action']}: {action['description']}")

    print("\n" + "="*80)
    print("✅ 模型优化分析完成！")
    print("📁 详细报告已保存到: outputs/model_optimization_analysis/")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
