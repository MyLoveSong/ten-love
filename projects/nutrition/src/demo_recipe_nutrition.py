#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
食谱-营养功能演示脚本
展示完整的营养评估、健康评分和推荐功能
"""

import sys
from pathlib import Path
import json
import numpy as np
import torch
from datetime import datetime
from typing import Dict, List, Any

# 添加项目路径
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入核心模块
try:
    from .enhanced_health_multitask_model import (
        NutritionDatabase,
        HealthTasteMultiTaskModel,
        HealthTasteMultiTaskTrainer,
        TrainingConfig
    )
    from .authoritative_nrf93_hei2015_model import (
        AuthoritativeNutritionDatabase,
        demonstrate_authoritative_standards
    )
    from .nutrition_driven_multitask_model import (
        NutritionDrivenDataGenerator,
        NutritionDrivenMultiTaskModel
    )
except ImportError as e:
    print(f"导入错误: {e}")
    print("使用简化版本...")
    from .simple_health_multitask_demo import (
        SimpleNutritionDatabase,
        SimpleHealthTasteModel
    )


class RecipeNutritionDemo:
    """食谱-营养功能演示类"""

    def __init__(self):
        self.output_dir = project_root / 'stage1' / 'outputs' / 'recipe_nutrition_demo'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化数据库
        try:
            self.nutrition_db = NutritionDatabase()
            self.auth_db = AuthoritativeNutritionDatabase()
            self.use_authoritative = True
        except:
            self.nutrition_db = SimpleNutritionDatabase()
            self.use_authoritative = False
            print("使用简化营养数据库")

    def demo_nutrition_profiles(self):
        """演示1: 查看菜品营养档案"""
        print("\n" + "="*60)
        print("[Step1] 菜品营养档案")
        print("="*60)

        test_dishes = [
            "清蒸鲈鱼", "蒸蛋羹", "清炒时蔬",
            "宫保鸡丁", "麻婆豆腐", "红烧肉"
        ]

        results = {}

        for dish in test_dishes:
            try:
                nutrition = self.nutrition_db.get_nutrition_profile(dish)
                health_score = nutrition.calculate_health_score()

                print(f"\n{dish}:")
                print(f"  营养成分 (每100g):")
                print(f"    - 卡路里: {nutrition.calories:.1f} kcal")
                print(f"    - 蛋白质: {nutrition.protein_g:.1f} g")
                print(f"    - 脂肪: {nutrition.fat_g:.1f} g")
                print(f"    - 钠: {nutrition.sodium_mg:.1f} mg")
                print(f"    - 糖: {nutrition.sugar_g:.1f} g")
                print(f"    - 膳食纤维: {nutrition.fiber_g:.1f} g")
                print(f"  健康评分: {health_score:.4f}")

                results[dish] = {
                    'nutrition': {
                        'calories': nutrition.calories,
                        'protein_g': nutrition.protein_g,
                        'fat_g': nutrition.fat_g,
                        'sodium_mg': nutrition.sodium_mg,
                        'sugar_g': nutrition.sugar_g,
                        'fiber_g': nutrition.fiber_g
                    },
                    'health_score': float(health_score)
                }
            except Exception as e:
                print(f"  错误: {e}")

        return results

    def demo_authoritative_scores(self):
        """演示2: 权威标准评分 (NRF9.3 + HEI-2015)"""
        print("\n" + "="*60)
        print("[Step2] 权威标准健康评分")
        print("="*60)

        if not self.use_authoritative:
            print("权威标准功能不可用，跳过此演示")
            return {}

        test_dishes = [
            "清蒸鲈鱼", "蒸蛋羹", "清炒时蔬",
            "宫保鸡丁", "麻婆豆腐", "红烧肉"
        ]

        results = {}

        for dish in test_dishes:
            try:
                auth_results = self.auth_db.get_authoritative_scores(dish)

                print(f"\n{dish}:")
                print(f"  NRF9.3评分: {auth_results['nrf93_results']['nrf93_normalized']:.4f}")
                print(f"    - 等级: {auth_results['nrf93_results']['health_grade']}")
                print(f"    - 原始分: {auth_results['nrf93_results']['nrf93_raw']:.1f}")

                print(f"  HEI-2015评分: {auth_results['hei2015_results']['hei_normalized']:.4f}")
                print(f"    - 等级: {auth_results['hei2015_results']['health_grade']}")

                print(f"  综合健康评分: {auth_results['composite_health_score']:.4f}")
                print(f"  健康分类: {auth_results['health_classification']}")

                if auth_results['clinical_recommendations']:
                    print(f"  临床建议:")
                    for rec in auth_results['clinical_recommendations'][:3]:  # 显示前3条
                        print(f"    - {rec}")

                results[dish] = {
                    'nrf93': {
                        'normalized': float(auth_results['nrf93_results']['nrf93_normalized']),
                        'raw': float(auth_results['nrf93_results']['nrf93_raw']),
                        'grade': auth_results['nrf93_results']['health_grade']
                    },
                    'hei2015': {
                        'normalized': float(auth_results['hei2015_results']['hei_normalized']),
                        'grade': auth_results['hei2015_results']['health_grade']
                    },
                    'composite_score': float(auth_results['composite_health_score']),
                    'classification': auth_results['health_classification'],
                    'recommendations': auth_results['clinical_recommendations']
                }
            except Exception as e:
                print(f"  错误: {e}")

        return results

    def demo_health_taste_prediction(self):
        """演示3: 健康-口味多任务预测"""
        print("\n" + "="*60)
        print("[Step3] 健康-口味多任务预测")
        print("="*60)

        # 创建模型（未训练状态，仅演示架构）
        try:
            model = HealthTasteMultiTaskModel(
                num_regions=7,
                num_cuisines=8,
                cultural_feature_dim=64,
                nutrition_feature_dim=32,
                hidden_dim=128
            )
            model.eval()
            print("[OK] 模型创建成功")
        except:
            model = SimpleHealthTasteModel()
            model.eval()
            print("[OK] 使用简化模型")

        # 地域和菜系映射
        regions = ["华北", "华南", "华东", "华西", "东北", "西北", "西南"]
        cuisines = ["川菜", "粤菜", "鲁菜", "苏菜", "浙菜", "闽菜", "湘菜", "徽菜"]
        region_to_id = {region: i for i, region in enumerate(regions)}
        cuisine_to_id = {cuisine: i for i, cuisine in enumerate(cuisines)}

        # 测试案例
        test_cases = {
            "清蒸鲈鱼": {"region": "华南", "cuisine": "粤菜"},
            "蒸蛋羹": {"region": "华南", "cuisine": "粤菜"},
            "红烧肉": {"region": "华东", "cuisine": "苏菜"},
            "宫保鸡丁": {"region": "华西", "cuisine": "川菜"},
        }

        results = {}

        with torch.no_grad():
            for dish_name, case_info in test_cases.items():
                try:
                    # 准备输入
                    region_id = torch.LongTensor([region_to_id[case_info["region"]]])
                    cuisine_id = torch.LongTensor([cuisine_to_id[case_info["cuisine"]]])
                    preferences = torch.FloatTensor([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])

                    # 营养特征
                    nutrition = self.nutrition_db.get_nutrition_profile(dish_name)
                    nutrition_features = torch.FloatTensor([[
                        nutrition.sodium_mg / 2000.0,
                        nutrition.fat_g / 70.0,
                        nutrition.sugar_g / 50.0,
                        nutrition.fiber_g / 25.0,
                        nutrition.protein_g / 50.0,
                        nutrition.calories / 400.0,
                    ]])

                    # 预测
                    taste_pred, health_pred = model(region_id, cuisine_id, preferences, nutrition_features)

                    print(f"\n{dish_name} ({case_info['region']} - {case_info['cuisine']}):")
                    print(f"  口味预测: {taste_pred.item():.4f}")
                    print(f"  健康预测: {health_pred.item():.4f}")
                    print(f"  营养健康分: {nutrition.calculate_health_score():.4f}")

                    results[dish_name] = {
                        'taste_prediction': float(taste_pred.item()),
                        'health_prediction': float(health_pred.item()),
                        'nutrition_health_score': float(nutrition.calculate_health_score()),
                        'region': case_info['region'],
                        'cuisine': case_info['cuisine']
                    }
                except Exception as e:
                    print(f"  错误: {e}")

        return results

    def demo_recipe_recommendation(self):
        """演示4: 个性化食谱推荐"""
        print("\n" + "="*60)
        print("[Step4] 个性化食谱推荐")
        print("="*60)

        # 模拟用户档案
        user_profiles = [
            {
                'name': '健康追求者',
                'region': '华南',
                'preferences': [0.3, 0.4, 0.3, 0.5, 0.2, 0.6],  # 偏好清淡
                'health_priority': 0.8,  # 健康优先
                'taste_priority': 0.2
            },
            {
                'name': '口味优先者',
                'region': '华西',
                'preferences': [0.8, 0.6, 0.7, 0.5, 0.3, 0.4],  # 偏好重口味
                'health_priority': 0.3,
                'taste_priority': 0.7
            },
            {
                'name': '平衡型用户',
                'region': '华东',
                'preferences': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  # 中性偏好
                'health_priority': 0.5,
                'taste_priority': 0.5
            }
        ]

        all_dishes = [
            "清蒸鲈鱼", "蒸蛋羹", "清炒时蔬",
            "宫保鸡丁", "麻婆豆腐", "红烧肉"
        ]

        results = {}

        for user in user_profiles:
            print(f"\nUser {user['name']} ({user['region']}):")
            print(f"  健康权重: {user['health_priority']:.1f}, 口味权重: {user['taste_priority']:.1f}")

            recommendations = []

            for dish in all_dishes:
                try:
                    nutrition = self.nutrition_db.get_nutrition_profile(dish)
                    health_score = nutrition.calculate_health_score()

                    # 简化的口味评分（基于地域匹配）
                    if user['region'] == '华南' and dish in ['清蒸鲈鱼', '蒸蛋羹']:
                        taste_score = 0.8
                    elif user['region'] == '华西' and dish in ['宫保鸡丁', '麻婆豆腐']:
                        taste_score = 0.75
                    else:
                        taste_score = 0.6

                    # 综合推荐分数
                    recommendation_score = (
                        user['health_priority'] * health_score +
                        user['taste_priority'] * taste_score
                    )

                    recommendations.append({
                        'dish': dish,
                        'health_score': health_score,
                        'taste_score': taste_score,
                        'recommendation_score': recommendation_score
                    })
                except Exception as e:
                    continue

            # 排序并显示Top 3
            recommendations.sort(key=lambda x: x['recommendation_score'], reverse=True)

            print(f"  [Top3] 推荐:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"    {i}. {rec['dish']}: 推荐分={rec['recommendation_score']:.4f} "
                      f"(健康={rec['health_score']:.3f}, 口味={rec['taste_score']:.3f})")

            results[user['name']] = {
                'profile': user,
                'recommendations': recommendations[:3]
            }

        return results

    def generate_summary_report(self, all_results: Dict[str, Any]):
        """生成总结报告"""
        print("\n" + "="*60)
        print("[Report] 生成演示报告")
        print("="*60)

        report = {
            'timestamp': datetime.now().isoformat(),
            'demo_summary': {
                'nutrition_profiles': 'done',
                'authoritative_scores': 'done' if self.use_authoritative else 'simplified',
                'health_taste_prediction': 'done',
                'recipe_recommendation': 'done'
            },
            'results': all_results,
            'key_features': [
                '营养成分档案管理',
                '权威标准健康评分 (NRF9.3 + HEI-2015)',
                '健康-口味多任务预测',
                '个性化食谱推荐',
                '临床营养建议'
            ],
            'supported_dishes': [
                "清蒸鲈鱼", "蒸蛋羹", "清炒时蔬",
                "宫保鸡丁", "麻婆豆腐", "红烧肉"
            ]
        }

        # 保存报告
        report_path = self.output_dir / 'recipe_nutrition_demo_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        print(f"[OK] 报告已保存: {report_path}")

        # 打印总结
        print("\n" + "="*60)
        print("[Summary] 演示总结")
        print("="*60)
        print(f"[OK] 营养档案: {len(all_results.get('nutrition_profiles', {}))} 个菜品")
        if self.use_authoritative:
            print(f"[OK] 权威评分: {len(all_results.get('authoritative_scores', {}))} 个菜品")
        print(f"[OK] 模型预测: {len(all_results.get('predictions', {}))} 个案例")
        print(f"[OK] 个性化推荐: {len(all_results.get('recommendations', {}))} 个用户")

        return report

    def run_full_demo(self):
        """运行完整演示"""
        print("\n" + "="*60)
        print("[Demo] 食谱-营养功能完整演示")
        print("="*60)
        print(f"输出目录: {self.output_dir}")

        all_results = {}

        # 1. 营养档案演示
        all_results['nutrition_profiles'] = self.demo_nutrition_profiles()

        # 2. 权威标准评分
        if self.use_authoritative:
            all_results['authoritative_scores'] = self.demo_authoritative_scores()

        # 3. 健康-口味预测
        all_results['predictions'] = self.demo_health_taste_prediction()

        # 4. 个性化推荐
        all_results['recommendations'] = self.demo_recipe_recommendation()

        # 5. 生成报告
        report = self.generate_summary_report(all_results)

        print("\n" + "="*60)
        print("[Done] 演示完成！")
        print("="*60)
        print("\n[Info] 核心功能:")
        print("  - 营养成分档案管理")
        print("  - 权威标准健康评分 (NRF9.3 + HEI-2015)")
        print("  - 健康-口味多任务预测")
        print("  - 个性化食谱推荐")
        print("  - 临床营养建议")
        print(f"\n[Report] 详细报告: {self.output_dir / 'recipe_nutrition_demo_report.json'}")


def main():
    """主函数"""
    demo = RecipeNutritionDemo()
    demo.run_full_demo()


if __name__ == "__main__":
    main()
