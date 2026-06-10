

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目功能演示脚本
展示省创项目申请表中的所有核心功能
"""

import asyncio
import json
import torch
import numpy as np
from datetime import datetime
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def demo_gluformer_prediction():
    """演示GluFormer血糖预测功能"""
    print("\n" + "="*60)
    print("🧠 演示1: GluFormer血糖预测模型")
    print("="*60)

    try:
        from academic_integrated_system import GluFormer

        # 创建GluFormer模型
        gluformer = GluFormer(input_dim=10, hidden_dim=128)

        # 模拟血糖时序数据
        batch_size = 1
        seq_length = 10
        input_dim = 10

        # 创建模拟输入数据
        mock_data = torch.randn(batch_size, seq_length, input_dim)

        # 进行预测
        with torch.no_grad():
            prediction = gluformer(mock_data)

        print(f"✅ GluFormer模型预测结果: {prediction.item():.2f} mmol/L")
        print(f"📊 模型架构: LSTM-GRU融合 + 交叉注意力机制")
        print(f"🎯 创新点: 长期记忆能力与短期动态捕捉相结合")

    except Exception as e:
        print(f"❌ GluFormer演示失败: {e}")

async def demo_cultural_adaptation():
    """演示文化适配功能"""
    print("\n" + "="*60)
    print("🌍 演示2: 文化适配膳食推荐")
    print("="*60)

    try:
        from models.cultural_adaptation import CulturalDietaryRecommendationEngine, CulturalProfile

        # 创建文化适配引擎
        engine = CulturalDietaryRecommendationEngine()

        # 创建用户文化档案
        user_profile = CulturalProfile(
            region='中国',
            cuisine_type='川菜',
            flavor_preferences=['麻辣', '重油', '重盐'],
            cooking_methods=['炒', '炸', '炖'],
            dietary_restrictions=[],
            religious_restrictions=[]
        )

        # 注册用户
        engine.register_user_cultural_profile('demo_user', user_profile)

        # 生成推荐
        health_profile = {
            'max_gi': 60,
            'max_carbs': 40,
            'max_calories': 500
        }

        recommendations = engine.generate_cultural_recommendations(
            'demo_user', health_profile, 'lunch'
        )

        print(f"✅ 文化适配推荐生成成功")
        print(f"👤 用户文化档案: {recommendations['cultural_profile']['cuisine_type']}")
        print(f"🎯 文化适配评分: {recommendations['cultural_adaptation_score']:.2f}")
        print(f"🍽️ 推荐食物数量: {len(recommendations['recommendations'])}")

        # 显示前3个推荐
        for i, rec in enumerate(recommendations['recommendations'][:3]):
            print(f"   {i+1}. {rec['food_name']} (文化匹配度: {rec['cultural_match']:.2f})")

    except Exception as e:
        print(f"❌ 文化适配演示失败: {e}")

async def demo_mixture_of_experts():
    """演示混合专家系统"""
    print("\n" + "="*60)
    print("👥 演示3: 混合专家系统 (MoE)")
    print("="*60)

    try:
        from models.mixture_of_experts import MERLSystem, ExpertConfig

        # 创建专家配置
        expert_configs = [
            ExpertConfig(
                expert_type='health_indicator',
                input_dim=20,
                hidden_dim=128,
                output_dim=64
            ),
            ExpertConfig(
                expert_type='medical_diagnosis',
                input_dim=20,
                hidden_dim=128,
                output_dim=64
            ),
            ExpertConfig(
                expert_type='lifestyle_assessment',
                input_dim=20,
                hidden_dim=128,
                output_dim=64
            )
        ]

        # 创建MERL系统
        merl_system = MERLSystem(
            input_dim=20,
            action_dim=5,
            expert_configs=expert_configs
        )

        # 模拟输入
        input_features = torch.randn(1, 20)

        # 获取系统输出
        with torch.no_grad():
            output = merl_system(input_features)

        print(f"✅ 混合专家系统运行成功")
        print(f"🎯 推荐动作: {torch.argmax(output['policy_probs'], dim=-1).item()}")
        print(f"📊 价值估计: {output['value'].item():.3f}")
        print(f"⚖️ 专家权重分布:")

        gate_weights = output['gate_weights'][0]
        expert_names = ['健康指标专家', '医学诊断专家', '生活习惯专家']
        for i, (name, weight) in enumerate(zip(expert_names, gate_weights)):
            print(f"   - {name}: {weight:.3f}")

    except Exception as e:
        print(f"❌ 混合专家系统演示失败: {e}")

async def demo_knowledge_distillation():
    """演示知识蒸馏框架"""
    print("\n" + "="*60)
    print("🧠 演示4: 跨域知识蒸馏框架")
    print("="*60)

    try:
        from models.knowledge_distillation import MedicalKnowledgeBase, ExplainableKnowledgeSystem

        # 创建医学知识库
        knowledge_base = MedicalKnowledgeBase()

        # 创建可解释性系统
        explainable_system = ExplainableKnowledgeSystem(knowledge_base)

        # 模拟患者特征
        patient_features = {
            'glucose': 8.5,
            'hba1c': 7.2,
            'carbohydrates': 60,
            'fiber': 15,
            'protein': 25,
            'exercise_duration': 20,
            'sleep_hours': 5,
            'stress_level': 4
        }

        # 生成解释
        explanation = explainable_system.generate_explanation(patient_features, 0.8)

        print(f"✅ 知识蒸馏框架运行成功")
        print(f"📊 可解释性评分: {explanation['neural_analysis']['explainability_score']:.3f}")
        print(f"🔍 适用规则数量: {len(explanation['applicable_rules'])}")

        # 显示领域贡献
        print(f"📈 领域贡献分析:")
        for domain, contribution in explanation['neural_analysis']['domain_contributions'].items():
            print(f"   - {domain}: {contribution:.3f}")

        # 显示前3个适用规则
        print(f"📋 适用规则:")
        for i, rule in enumerate(explanation['applicable_rules'][:3]):
            print(f"   {i+1}. {rule['rule_id']}: {rule['condition']} -> {rule['conclusion']}")

    except Exception as e:
        print(f"❌ 知识蒸馏演示失败: {e}")

async def demo_enhanced_system():
    """演示增强集成系统"""
    print("\n" + "="*60)
    print("🚀 演示5: 增强学术级集成系统")
    print("="*60)

    try:
        from academic_integrated_system import AcademicIntegratedSystem

        # 创建系统实例
        system = AcademicIntegratedSystem()

        # 获取系统概要
        summary = system.get_system_summary()

        print(f"✅ 增强集成系统初始化成功")
        print(f"📊 系统版本: {summary['version']}")
        print(f"🎯 核心能力数量: {len(summary['capabilities'])}")
        print(f"🌐 API接口数量: {len(summary['api_endpoints'])}")

        # 显示组件状态
        print(f"🔧 核心组件状态:")
        for category, components in summary['components'].items():
            print(f"   {category}:")
            for comp_name, status in components.items():
                status_icon = "✅" if status else "❌"
                print(f"      {status_icon} {comp_name}")

        # 显示核心能力
        print(f"🎯 核心能力:")
        for capability in summary['capabilities']:
            print(f"   • {capability}")

    except Exception as e:
        print(f"❌ 增强系统演示失败: {e}")

async def demo_data_enhancement():
    """演示数据增强功能"""
    print("\n" + "="*60)
    print("📈 演示6: 动态数据增强与自监督学习")
    print("="*60)

    try:
        from app.data_integration.data_augmentation import DataEnhancementPipeline, AugmentationConfig

        # 创建数据增强配置
        config = AugmentationConfig(
            noise_level=0.05,
            augmentation_ratio=2.0,
            synthetic_samples=100
        )

        # 创建数据增强流水线
        pipeline = DataEnhancementPipeline(config)

        # 模拟血糖数据
        import pandas as pd
        np.random.seed(42)
        glucose_data = pd.DataFrame({
            'glucose': np.random.normal(120, 20, 50),
            'timestamp': pd.date_range('2024-01-01', periods=50, freq='H'),
            'carbohydrates': np.random.normal(50, 15, 50),
            'exercise': np.random.normal(30, 10, 50)
        })

        # 数据增强
        enhanced_datasets = pipeline.enhance_dataset(glucose_data)

        print(f"✅ 数据增强流水线运行成功")
        print(f"📊 原始数据量: {len(glucose_data)}")
        print(f"📈 增强后数据量: {len(enhanced_datasets['final'])}")
        print(f"🎯 增强倍数: {len(enhanced_datasets['final']) / len(glucose_data):.1f}x")

        # 显示增强方法
        print(f"🔧 使用的增强方法:")
        for method, dataset in enhanced_datasets.items():
            if method != 'final':
                print(f"   - {method}: {len(dataset)} 样本")

    except Exception as e:
        print(f"❌ 数据增强演示失败: {e}")

async def main():
    """主演示函数"""
    print("🎉 省创项目功能演示开始")
    print("📋 项目: 基于多模态大模型的糖尿病个性化营养膳食方案")
    print("🏆 创新点: GluFormer + 文化适配 + 混合专家 + 知识蒸馏")

    # 运行所有演示
    await demo_gluformer_prediction()
    await demo_cultural_adaptation()
    await demo_mixture_of_experts()
    await demo_knowledge_distillation()
    await demo_data_enhancement()
    await demo_enhanced_system()

    print("\n" + "="*60)
    print("🎊 所有功能演示完成！")
    print("="*60)
    print("📝 项目特色总结:")
    print("   ✅ 创新点一: GluFormer LSTM-GRU融合血糖预测")
    print("   ✅ 创新点二: 医学知识增强的混合专家决策系统")
    print("   ✅ 创新点三: 文化适配的个性化膳食生成引擎")
    print("   ✅ 技术路线: 多源数据融合 → 时序建模 → 跨域协同优化")
    print("   ✅ 系统集成: 工作流 + 数据增强 + 模型适应 + 反馈优化")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())

__all__ = ["'logger'"]
