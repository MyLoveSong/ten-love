

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库初始化脚本
用于创建数据库、运行迁移和初始化基础数据
"""

import sys
import os
import logging
from datetime import datetime
import json

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.models import initialize_database, SessionLocal
from database.database_manager import DatabaseManager
from database.migrations import MigrationManager
from database.config import get_current_database_config, config_manager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def init_database():
    """初始化数据库"""
    try:
        # 获取数据库配置
        config = get_current_database_config()
        logger.info(f"使用数据库配置: {config.db_type}")

        # 验证配置
        if not config_manager.validate_config(config):
            logger.error("数据库配置验证失败")
            return False

        # 初始化数据库
        engine, SessionLocal = initialize_database(
            config.get_database_url(),
            config.echo
        )

        # 创建会话
        session = SessionLocal()

        try:
            # 运行迁移
            migration_manager = MigrationManager(session)
            if not migration_manager.initialize_database():
                logger.error("数据库迁移初始化失败")
                return False

            if not migration_manager.upgrade_database():
                logger.error("数据库升级失败")
                return False

            # 创建数据库管理器
            db_manager = DatabaseManager(session)

            # 初始化基础数据
            if not init_base_data(db_manager):
                logger.error("基础数据初始化失败")
                return False

            # 显示数据库状态
            show_database_status(db_manager, migration_manager)

            logger.info("数据库初始化完成")
            return True

        finally:
            session.close()

    except Exception as e:
        logger.error(f"数据库初始化失败: {e}")
        return False

def init_base_data(db_manager: DatabaseManager):
    """初始化基础数据"""
    try:
        logger.info("开始初始化基础数据...")

        # 创建默认管理员用户
        admin_user = db_manager.get_user_by_username("admin")
        if not admin_user:
            admin_user = db_manager.create_user(
                username="admin",
                email="admin@academic-system.com",
                password_hash="admin_password_hash",  # 实际应用中应该使用安全的哈希
                role="admin",
                profile_data={"full_name": "系统管理员", "department": "IT"}
            )
            logger.info("创建默认管理员用户")

        # 创建默认研究员用户
        researcher_user = db_manager.get_user_by_username("researcher")
        if not researcher_user:
            researcher_user = db_manager.create_user(
                username="researcher",
                email="researcher@academic-system.com",
                password_hash="researcher_password_hash",
                role="researcher",
                profile_data={"full_name": "研究员", "department": "研究部"}
            )
            logger.info("创建默认研究员用户")

        # 创建默认患者用户
        patient_user = db_manager.get_user_by_username("patient")
        if not patient_user:
            patient_user = db_manager.create_user(
                username="patient",
                email="patient@academic-system.com",
                password_hash="patient_password_hash",
                role="patient",
                profile_data={"full_name": "测试患者", "age": 45, "gender": "male"}
            )
            logger.info("创建默认患者用户")

        # 初始化食物数据库
        if not init_food_database(db_manager):
            logger.warning("食物数据库初始化失败")

        # 创建示例文化档案
        if not init_cultural_profiles(db_manager):
            logger.warning("文化档案初始化失败")

        # 创建示例工作流定义
        if not init_workflow_definitions(db_manager):
            logger.warning("工作流定义初始化失败")

        logger.info("基础数据初始化完成")
        return True

    except Exception as e:
        logger.error(f"基础数据初始化失败: {e}")
        return False

def init_food_database(db_manager: DatabaseManager):
    """初始化食物数据库"""
    try:
        # 检查是否已有食物数据
        existing_foods = db_manager.search_food_items("米饭")
        if existing_foods:
            logger.info("食物数据库已存在，跳过初始化")
            return True

        # 创建基础食物项目
        foods = [
            {
                "name": "米饭",
                "name_en": "Rice",
                "category": "主食",
                "subcategory": "谷物",
                "nutrition": {"calories": 130, "carbohydrates": 28, "protein": 2.7, "fat": 0.3},
                "cultural_tags": ["Chinese", "Asian", "Staple"],
                "cooking_methods": ["蒸", "煮"],
                "flavor_profile": {"taste": "neutral", "texture": "soft"},
                "glycemic_index": 73,
                "cultural_region": "Asia",
                "is_vegetarian": True,
                "is_halal": True,
                "is_vegan": True
            },
            {
                "name": "鸡胸肉",
                "name_en": "Chicken Breast",
                "category": "蛋白质",
                "subcategory": "肉类",
                "nutrition": {"calories": 165, "carbohydrates": 0, "protein": 31, "fat": 3.6},
                "cultural_tags": ["Global", "Protein"],
                "cooking_methods": ["烤", "煮", "炒"],
                "flavor_profile": {"taste": "mild", "texture": "tender"},
                "glycemic_index": 0,
                "cultural_region": "Global",
                "is_vegetarian": False,
                "is_halal": True,
                "is_vegan": False
            },
            {
                "name": "豆腐",
                "name_en": "Tofu",
                "category": "蛋白质",
                "subcategory": "豆制品",
                "nutrition": {"calories": 76, "carbohydrates": 1.9, "protein": 8, "fat": 4.8},
                "cultural_tags": ["Chinese", "Vegetarian", "Protein"],
                "cooking_methods": ["炒", "煮", "炸"],
                "flavor_profile": {"taste": "neutral", "texture": "soft"},
                "glycemic_index": 15,
                "cultural_region": "Asia",
                "is_vegetarian": True,
                "is_halal": True,
                "is_vegan": True
            },
            {
                "name": "蔬菜沙拉",
                "name_en": "Vegetable Salad",
                "category": "蔬菜",
                "subcategory": "生菜",
                "nutrition": {"calories": 100, "carbohydrates": 10, "protein": 5, "fat": 5},
                "cultural_tags": ["Global", "Healthy", "Vegetarian"],
                "cooking_methods": ["生食"],
                "flavor_profile": {"taste": "fresh", "texture": "crispy"},
                "glycemic_index": 20,
                "cultural_region": "Global",
                "is_vegetarian": True,
                "is_halal": True,
                "is_vegan": True
            }
        ]

        for food_data in foods:
            db_manager.create_food_item(**food_data)

        logger.info(f"创建了 {len(foods)} 个基础食物项目")
        return True

    except Exception as e:
        logger.error(f"食物数据库初始化失败: {e}")
        return False

def init_cultural_profiles(db_manager: DatabaseManager):
    """初始化文化档案"""
    try:
        # 获取患者用户
        patient_user = db_manager.get_user_by_username("patient")
        if not patient_user:
            logger.warning("未找到患者用户，跳过文化档案初始化")
            return True

        # 检查是否已有文化档案
        existing_profile = db_manager.get_cultural_profile(patient_user.user_id)
        if existing_profile:
            logger.info("文化档案已存在，跳过初始化")
            return True

        # 创建示例文化档案
        profile = db_manager.create_cultural_profile(
            user_id=patient_user.user_id,
            region="中国",
            cuisine_type="川菜",
            flavor_preferences=["辣", "麻", "香"],
            cooking_methods=["炒", "煮", "蒸"],
            dietary_restrictions=["素食"],
            religious_restrictions=[],
            spice_tolerance="high"
        )

        logger.info("创建示例文化档案")
        return True

    except Exception as e:
        logger.error(f"文化档案初始化失败: {e}")
        return False

def init_workflow_definitions(db_manager: DatabaseManager):
    """初始化工作流定义"""
    try:
        # 获取管理员用户
        admin_user = db_manager.get_user_by_username("admin")
        if not admin_user:
            logger.warning("未找到管理员用户，跳过工作流定义初始化")
            return True

        # 检查是否已有工作流定义
        existing_workflows = db_manager.get_workflow_executions(limit=1)
        if existing_workflows:
            logger.info("工作流定义已存在，跳过初始化")
            return True

        # 创建血糖预测工作流定义
        glucose_workflow = db_manager.create_workflow_definition(
            name="血糖预测工作流",
            description="从CGM数据到血糖预测的端到端工作流",
            version="1.0.0",
            definition={
                "nodes": [
                    {"id": "data_source", "type": "data_source", "name": "CGM数据源"},
                    {"id": "preprocessing", "type": "preprocessing", "name": "数据预处理"},
                    {"id": "prediction", "type": "model", "name": "血糖预测"},
                    {"id": "output", "type": "output", "name": "结果输出"}
                ],
                "edges": [
                    {"source": "data_source", "target": "preprocessing"},
                    {"source": "preprocessing", "target": "prediction"},
                    {"source": "prediction", "target": "output"}
                ]
            },
            created_by=admin_user.user_id,
            tags=["glucose", "prediction", "health"],
            category="health_monitoring"
        )

        # 创建文化推荐工作流定义
        cultural_workflow = db_manager.create_workflow_definition(
            name="文化适配膳食推荐工作流",
            description="结合用户文化和健康档案生成个性化膳食推荐",
            version="1.0.0",
            definition={
                "nodes": [
                    {"id": "cultural_profile", "type": "data_source", "name": "文化档案"},
                    {"id": "health_profile", "type": "data_source", "name": "健康档案"},
                    {"id": "recommendation_engine", "type": "model", "name": "推荐引擎"},
                    {"id": "output", "type": "output", "name": "推荐结果"}
                ],
                "edges": [
                    {"source": "cultural_profile", "target": "recommendation_engine"},
                    {"source": "health_profile", "target": "recommendation_engine"},
                    {"source": "recommendation_engine", "target": "output"}
                ]
            },
            created_by=admin_user.user_id,
            tags=["cultural", "recommendation", "diet"],
            category="recommendation"
        )

        logger.info("创建示例工作流定义")
        return True

    except Exception as e:
        logger.error(f"工作流定义初始化失败: {e}")
        return False

def show_database_status(db_manager: DatabaseManager, migration_manager: MigrationManager):
    """显示数据库状态"""
    try:
        print("\n" + "="*60)
        print("数据库初始化状态报告")
        print("="*60)

        # 数据库统计信息
        stats = db_manager.get_database_stats()
        print("\n📊 数据库统计信息:")
        for table, count in stats.items():
            print(f"  {table}: {count} 条记录")

        # 迁移状态
        migration_status = migration_manager.get_migration_status()
        print(f"\n🔄 迁移状态:")
        print(f"  当前版本: {migration_status.get('current_version', 'N/A')}")
        print(f"  已应用迁移: {len(migration_status.get('applied_migrations', []))}")
        print(f"  待处理迁移: {len(migration_status.get('pending_migrations', []))}")

        # 架构一致性
        schema_consistency = migration_status.get('schema_consistency', {})
        if schema_consistency.get('consistent', False):
            print(f"  ✅ 架构一致性: 正常")
        else:
            print(f"  ❌ 架构一致性: 异常")
            if schema_consistency.get('missing_tables'):
                print(f"    缺失表: {schema_consistency['missing_tables']}")
            if schema_consistency.get('extra_tables'):
                print(f"    多余表: {schema_consistency['extra_tables']}")

        # 用户信息
        users = db_manager.get_users_by_role("admin") + db_manager.get_users_by_role("researcher") + db_manager.get_users_by_role("patient")
        print(f"\n👥 用户信息:")
        for user in users:
            print(f"  {user.username} ({user.role}) - {user.email}")

        # 食物项目数量
        food_count = len(db_manager.search_food_items(""))
        print(f"\n🍽️ 食物数据库: {food_count} 个项目")

        print("\n" + "="*60)
        print("数据库初始化完成！")
        print("="*60)

    except Exception as e:
        logger.error(f"显示数据库状态失败: {e}")

def main():
    """主函数"""
    print("开始初始化数据库...")

    if init_database():
        print("\n✅ 数据库初始化成功！")
        print("\n下一步:")
        print("1. 启动系统: python academic_integrated_system.py")
        print("2. 访问API文档: http://localhost:8000/docs")
        print("3. 访问GraphQL: http://localhost:8000/api/graphql")
        return 0
    else:
        print("\n❌ 数据库初始化失败！")
        print("请检查错误日志并重试")
        return 1

if __name__ == "__main__":
    exit(main())

__all__ = ["'logger'", "'init_database'", "'init_base_data'", "'init_food_database'", "'init_cultural_profiles'", "'init_workflow_definitions'", "'show_database_status'", "'main'"]
