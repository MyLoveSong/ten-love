

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库使用示例
演示如何使用数据库管理器和各种功能
"""

import sys
import os
import logging
from datetime import datetime, timedelta
import json

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.models import initialize_database, SessionLocal
from database.database_manager import DatabaseManager
from database.migrations import MigrationManager
from database.config import get_current_database_config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def demo_user_management():
    """演示用户管理功能"""
    print("\n" + "="*60)
    print("用户管理功能演示")
    print("="*60)

    # 创建数据库会话
    session = SessionLocal()
    db_manager = DatabaseManager(session)

    try:
        # 1. 创建用户
        print("\n1. 创建用户")
        user1 = db_manager.create_user(
            username="demo_user1",
            email="demo1@example.com",
            password_hash="hashed_password_1",
            role="patient",
            profile_data={"full_name": "演示用户1", "age": 35, "gender": "female"}
        )
        print(f"创建用户: {user1.username} (ID: {user1.user_id})")

        user2 = db_manager.create_user(
            username="demo_researcher",
            email="researcher@example.com",
            password_hash="hashed_password_2",
            role="researcher",
            profile_data={"full_name": "演示研究员", "department": "研究部"}
        )
        print(f"创建用户: {user2.username} (ID: {user2.user_id})")

        # 2. 查询用户
        print("\n2. 查询用户")
        found_user = db_manager.get_user_by_username("demo_user1")
        print(f"查找用户: {found_user.username if found_user else '未找到'}")

        # 3. 按角色查询用户
        print("\n3. 按角色查询用户")
        patients = db_manager.get_users_by_role("patient")
        print(f"患者用户数量: {len(patients)}")
        for patient in patients:
            print(f"  - {patient.username} ({patient.email})")

        # 4. 更新用户最后登录时间
        print("\n4. 更新用户登录时间")
        db_manager.update_user_last_login(user1.user_id)
        updated_user = db_manager.get_user_by_id(user1.user_id)
        print(f"用户 {updated_user.username} 最后登录: {updated_user.last_login}")

    finally:
        session.close()

def demo_cultural_profiles():
    """演示文化档案管理功能"""
    print("\n" + "="*60)
    print("文化档案管理功能演示")
    print("="*60)

    session = SessionLocal()
    db_manager = DatabaseManager(session)

    try:
        # 获取用户
        user = db_manager.get_user_by_username("demo_user1")
        if not user:
            print("未找到演示用户，跳过文化档案演示")
            return

        # 1. 创建文化档案
        print("\n1. 创建文化档案")
        profile = db_manager.create_cultural_profile(
            user_id=user.user_id,
            region="中国",
            cuisine_type="川菜",
            flavor_preferences=["辣", "麻", "香"],
            cooking_methods=["炒", "煮", "蒸"],
            dietary_restrictions=["素食"],
            religious_restrictions=[],
            spice_tolerance="high"
        )
        print(f"创建文化档案: 用户{user.username}, 地区{profile.region}, 菜系{profile.cuisine_type}")

        # 2. 查询文化档案
        print("\n2. 查询文化档案")
        found_profile = db_manager.get_cultural_profile(user.user_id)
        if found_profile:
            print(f"文化档案: 地区={found_profile.region}, 菜系={found_profile.cuisine_type}")
            print(f"口味偏好: {found_profile.flavor_preferences}")
            print(f"烹饪方式: {found_profile.cooking_methods}")
            print(f"饮食限制: {found_profile.dietary_restrictions}")

        # 3. 更新文化档案
        print("\n3. 更新文化档案")
        updated_profile = db_manager.update_cultural_profile(
            user.user_id,
            spice_tolerance="medium",
            flavor_preferences=["辣", "香", "鲜"]
        )
        if updated_profile:
            print(f"更新后的口味偏好: {updated_profile.flavor_preferences}")
            print(f"更新后的辣度耐受: {updated_profile.spice_tolerance}")

    finally:
        session.close()

def demo_glucose_predictions():
    """演示血糖预测管理功能"""
    print("\n" + "="*60)
    print("血糖预测管理功能演示")
    print("="*60)

    session = SessionLocal()
    db_manager = DatabaseManager(session)

    try:
        # 获取用户
        user = db_manager.get_user_by_username("demo_user1")
        if not user:
            print("未找到演示用户，跳过血糖预测演示")
            return

        # 1. 创建血糖预测记录
        print("\n1. 创建血糖预测记录")
        predictions = []
        for i in range(5):
            prediction = db_manager.create_glucose_prediction(
                user_id=user.user_id,
                predicted_glucose=120.0 + i * 10,
                confidence=0.85 + i * 0.02,
                model_type="GluFormer",
                model_version="1.0.0",
                input_features={
                    "glucose_history": [100, 110, 120],
                    "meal_carbs": 50 + i * 10,
                    "exercise_minutes": 30 - i * 5
                },
                context_data={
                    "meal_type": "breakfast" if i < 2 else "lunch",
                    "time_of_day": "morning" if i < 2 else "afternoon"
                }
            )
            predictions.append(prediction)
            print(f"预测 {i+1}: {prediction.predicted_glucose} mg/dL (置信度: {prediction.confidence:.2f})")

        # 2. 查询用户预测历史
        print("\n2. 查询用户预测历史")
        user_predictions = db_manager.get_user_glucose_predictions(user.user_id, limit=10)
        print(f"用户 {user.username} 的预测记录数量: {len(user_predictions)}")
        for pred in user_predictions[:3]:  # 显示前3条
            print(f"  - 预测值: {pred.predicted_glucose} mg/dL, 模型: {pred.model_type}, 时间: {pred.prediction_time}")

        # 3. 更新预测反馈
        print("\n3. 更新预测反馈")
        if predictions:
            first_prediction = predictions[0]
            actual_glucose = 125.0  # 模拟实际测量值
            accuracy_score = 1.0 - abs(first_prediction.predicted_glucose - actual_glucose) / actual_glucose

            updated_prediction = db_manager.update_glucose_prediction_feedback(
                first_prediction.id,
                actual_glucose,
                accuracy_score
            )
            if updated_prediction:
                print(f"更新反馈: 预测值={updated_prediction.predicted_glucose}, 实际值={updated_prediction.actual_glucose}")
                print(f"准确度评分: {updated_prediction.accuracy_score:.3f}")

        # 4. 获取模型性能统计
        print("\n4. 获取模型性能统计")
        performance_stats = db_manager.get_model_performance_stats("GluFormer", days=7)
        print(f"GluFormer模型性能统计:")
        print(f"  - 预测次数: {performance_stats['count']}")
        print(f"  - 准确率: {performance_stats['accuracy']:.2%}")
        print(f"  - 平均绝对误差: {performance_stats['mae']:.2f} mg/dL")
        print(f"  - 均方根误差: {performance_stats['rmse']:.2f} mg/dL")

    finally:
        session.close()

def demo_food_items():
    """演示食物项目管理功能"""
    print("\n" + "="*60)
    print("食物项目管理功能演示")
    print("="*60)

    session = SessionLocal()
    db_manager = DatabaseManager(session)

    try:
        # 1. 创建食物项目
        print("\n1. 创建食物项目")
        foods = [
            {
                "name": "宫保鸡丁",
                "name_en": "Kung Pao Chicken",
                "category": "主菜",
                "subcategory": "川菜",
                "nutrition": {"calories": 280, "carbohydrates": 15, "protein": 25, "fat": 12},
                "cultural_tags": ["Chinese", "Sichuan", "Spicy"],
                "cooking_methods": ["炒"],
                "flavor_profile": {"taste": "spicy", "texture": "tender"},
                "glycemic_index": 45,
                "cultural_region": "China",
                "is_vegetarian": False,
                "is_halal": True,
                "is_vegan": False
            },
            {
                "name": "麻婆豆腐",
                "name_en": "Mapo Tofu",
                "category": "主菜",
                "subcategory": "川菜",
                "nutrition": {"calories": 180, "carbohydrates": 8, "protein": 15, "fat": 10},
                "cultural_tags": ["Chinese", "Sichuan", "Vegetarian"],
                "cooking_methods": ["炒", "煮"],
                "flavor_profile": {"taste": "spicy", "texture": "soft"},
                "glycemic_index": 25,
                "cultural_region": "China",
                "is_vegetarian": True,
                "is_halal": True,
                "is_vegan": True
            }
        ]

        created_foods = []
        for food_data in foods:
            food_item = db_manager.create_food_item(**food_data)
            created_foods.append(food_item)
            print(f"创建食物: {food_item.name} (ID: {food_item.id})")

        # 2. 搜索食物项目
        print("\n2. 搜索食物项目")

        # 按名称搜索
        search_results = db_manager.search_food_items(query="豆腐")
        print(f"搜索'豆腐'的结果: {len(search_results)} 个")
        for food in search_results:
            print(f"  - {food.name} ({food.category})")

        # 按类别搜索
        chinese_foods = db_manager.search_food_items(category="主菜")
        print(f"主菜类别: {len(chinese_foods)} 个")

        # 按素食搜索
        vegetarian_foods = db_manager.search_food_items(is_vegetarian=True)
        print(f"素食食物: {len(vegetarian_foods)} 个")

        # 按低GI搜索
        low_gi_foods = db_manager.search_food_items(max_gi=50)
        print(f"低GI食物 (≤50): {len(low_gi_foods)} 个")

        # 3. 按文化标签搜索
        print("\n3. 按文化标签搜索")
        sichuan_foods = db_manager.get_food_items_by_cultural_tags(["Sichuan"])
        print(f"川菜食物: {len(sichuan_foods)} 个")
        for food in sichuan_foods:
            print(f"  - {food.name}: {food.cultural_tags}")

    finally:
        session.close()

def demo_cultural_recommendations():
    """演示文化推荐管理功能"""
    print("\n" + "="*60)
    print("文化推荐管理功能演示")
    print("="*60)

    session = SessionLocal()
    db_manager = DatabaseManager(session)

    try:
        # 获取用户
        user = db_manager.get_user_by_username("demo_user1")
        if not user:
            print("未找到演示用户，跳过文化推荐演示")
            return

        # 1. 创建文化推荐记录
        print("\n1. 创建文化推荐记录")
        recommendations = []
        meal_types = ["breakfast", "lunch", "dinner"]

        for i, meal_type in enumerate(meal_types):
            recommendation = db_manager.create_cultural_recommendation(
                user_id=user.user_id,
                meal_type=meal_type,
                recommendations={
                    "main_dish": f"推荐主菜{i+1}",
                    "side_dish": f"推荐配菜{i+1}",
                    "beverage": "推荐饮品"
                },
                cultural_adaptation_score=0.85 + i * 0.05,
                health_constraints={
                    "max_calories": 500 + i * 100,
                    "max_carbs": 50 + i * 20,
                    "dietary_restrictions": ["素食"]
                },
                nutritional_analysis={
                    "total_calories": 450 + i * 80,
                    "total_carbs": 45 + i * 15,
                    "protein": 20 + i * 5,
                    "fat": 15 + i * 3
                }
            )
            recommendations.append(recommendation)
            print(f"创建{meal_type}推荐: 适配度{recommendation.cultural_adaptation_score:.2f}")

        # 2. 查询用户推荐历史
        print("\n2. 查询用户推荐历史")
        user_recommendations = db_manager.get_user_recommendations(user.user_id, limit=10)
        print(f"用户 {user.username} 的推荐记录数量: {len(user_recommendations)}")

        # 3. 更新推荐反馈
        print("\n3. 更新推荐反馈")
        if recommendations:
            first_recommendation = recommendations[0]
            updated_rec = db_manager.update_recommendation_feedback(
                first_recommendation.id,
                user_satisfaction=4,  # 1-5分
                feedback_notes="味道很好，符合我的口味偏好"
            )
            if updated_rec:
                print(f"更新推荐反馈: 满意度={updated_rec.user_satisfaction}, 备注={updated_rec.feedback_notes}")

    finally:
        session.close()

def demo_system_metrics():
    """演示系统指标管理功能"""
    print("\n" + "="*60)
    print("系统指标管理功能演示")
    print("="*60)

    session = SessionLocal()
    db_manager = DatabaseManager(session)

    try:
        # 1. 记录系统指标
        print("\n1. 记录系统指标")
        metrics = [
            ("execution_time", 1.2, "seconds", "workflow_001", "node_001"),
            ("success_rate", 0.95, "percentage", "workflow_001", None),
            ("memory_usage", 512, "MB", None, None),
            ("cpu_usage", 45.5, "percentage", None, None),
            ("prediction_accuracy", 0.88, "percentage", "glucose_prediction", None)
        ]

        for metric_type, value, unit, workflow_id, node_id in metrics:
            metric = db_manager.record_system_metric(
                metric_type=metric_type,
                metric_value=value,
                metric_unit=unit,
                workflow_id=workflow_id,
                node_id=node_id,
                additional_data={"source": "demo"},
                tags=["demo", "test"]
            )
            print(f"记录指标: {metric_type}={value} {unit}")

        # 2. 查询系统指标
        print("\n2. 查询系统指标")
        recent_metrics = db_manager.get_system_metrics(limit=10)
        print(f"最近指标记录: {len(recent_metrics)} 条")
        for metric in recent_metrics[:5]:  # 显示前5条
            print(f"  - {metric.metric_type}: {metric.metric_value} {metric.metric_unit} ({metric.timestamp})")

        # 3. 获取指标统计
        print("\n3. 获取指标统计")
        execution_time_stats = db_manager.get_metric_statistics("execution_time", days=7)
        print(f"执行时间统计:")
        print(f"  - 记录数: {execution_time_stats['count']}")
        print(f"  - 平均值: {execution_time_stats['avg']:.2f} 秒")
        print(f"  - 最小值: {execution_time_stats['min']:.2f} 秒")
        print(f"  - 最大值: {execution_time_stats['max']:.2f} 秒")
        print(f"  - 标准差: {execution_time_stats['std']:.2f} 秒")

    finally:
        session.close()

def demo_feedback_management():
    """演示反馈管理功能"""
    print("\n" + "="*60)
    print("反馈管理功能演示")
    print("="*60)

    session = SessionLocal()
    db_manager = DatabaseManager(session)

    try:
        # 获取用户
        user = db_manager.get_user_by_username("demo_user1")
        if not user:
            print("未找到演示用户，跳过反馈管理演示")
            return

        # 1. 创建反馈记录
        print("\n1. 创建反馈记录")
        feedbacks = []
        for i in range(3):
            feedback = db_manager.create_feedback_record(
                user_id=user.user_id,
                prediction_id=f"pred_{i+1}",
                actual_glucose=120.0 + i * 5,
                satisfaction_score=4 - i,  # 4, 3, 2
                accuracy_rating=5 - i,     # 5, 4, 3
                recommendation_helpfulness=4 - i,  # 4, 3, 2
                additional_info={"meal_type": f"meal_{i+1}"},
                feedback_type="prediction"
            )
            feedbacks.append(feedback)
            print(f"创建反馈: 满意度={feedback.satisfaction_score}, 准确度={feedback.accuracy_rating}")

        # 2. 查询用户反馈
        print("\n2. 查询用户反馈")
        user_feedbacks = db_manager.get_user_feedback(user.user_id, limit=10)
        print(f"用户 {user.username} 的反馈记录数量: {len(user_feedbacks)}")

        # 3. 获取反馈统计
        print("\n3. 获取反馈统计")
        feedback_stats = db_manager.get_feedback_statistics(days=30)
        print(f"反馈统计:")
        print(f"  - 总反馈数: {feedback_stats['count']}")
        print(f"  - 平均满意度: {feedback_stats['avg_satisfaction']:.2f}")
        print(f"  - 平均准确度: {feedback_stats['avg_accuracy']:.2f}")
        print(f"  - 满意度分布: {feedback_stats['satisfaction_distribution']}")

    finally:
        session.close()

def demo_database_statistics():
    """演示数据库统计功能"""
    print("\n" + "="*60)
    print("数据库统计功能演示")
    print("="*60)

    session = SessionLocal()
    db_manager = DatabaseManager(session)

    try:
        # 获取数据库统计信息
        print("\n数据库统计信息:")
        stats = db_manager.get_database_stats()
        for table, count in stats.items():
            print(f"  {table}: {count} 条记录")

        # 清理旧数据（演示）
        print("\n清理旧数据:")
        cleanup_result = db_manager.cleanup_old_data(days=1)  # 清理1天前的数据
        print(f"清理结果: {cleanup_result}")

    finally:
        session.close()

def main():
    """主函数"""
    print("数据库功能演示开始")
    print("="*60)

    try:
        # 初始化数据库
        config = get_current_database_config()
        print(f"使用数据库: {config.db_type}")

        engine, SessionLocal = initialize_database(
            config.get_database_url(),
            config.echo
        )

        # 运行迁移
        session = SessionLocal()
        migration_manager = MigrationManager(session)
        migration_manager.initialize_database()
        migration_manager.upgrade_database()
        session.close()

        # 运行各种演示
        demo_user_management()
        demo_cultural_profiles()
        demo_glucose_predictions()
        demo_food_items()
        demo_cultural_recommendations()
        demo_system_metrics()
        demo_feedback_management()
        demo_database_statistics()

        print("\n" + "="*60)
        print("数据库功能演示完成！")
        print("="*60)

        print("\n下一步:")
        print("1. 启动系统: python academic_integrated_system.py")
        print("2. 访问API文档: http://localhost:8000/docs")
        print("3. 测试数据库API: http://localhost:8000/database/stats")
        print("4. 查看用户列表: http://localhost:8000/users")
        print("5. 搜索食物: http://localhost:8000/food-items?query=豆腐")

    except Exception as e:
        logger.error(f"演示过程中发生错误: {e}")
        print(f"\n❌ 演示失败: {e}")

if __name__ == "__main__":
    main()

__all__ = ["'logger'", "'demo_user_management'", "'demo_cultural_profiles'", "'demo_glucose_predictions'", "'demo_food_items'", "'demo_cultural_recommendations'", "'demo_system_metrics'", "'demo_feedback_management'", "'demo_database_statistics'", "'main'"]
