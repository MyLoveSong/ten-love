

"""
数据库迁移脚本 - 添加CookLikeHOC相关表
"""

import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from pathlib import Path
import sys

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.app.models.recipe_models import Base
from app.backend.app.core.config import DATABASE_URL

logger = logging.getLogger(__name__)

def create_tables():
    """创建CookLikeHOC相关表"""
    try:
        # 创建数据库引擎
        engine = create_engine(DATABASE_URL)

        # 创建所有表
        Base.metadata.create_all(bind=engine)

        logger.info("CookLikeHOC相关表创建成功")
        print("✅ CookLikeHOC相关表创建成功")

        # 验证表是否创建成功
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name IN ('recipes', 'ingredient_nutrition', 'recipe_recommendations', 'user_recipe_preferences')
            """))
            tables = [row[0] for row in result]

            print(f"📋 已创建的表: {', '.join(tables)}")

            # 检查表结构
            for table_name in tables:
                result = conn.execute(text(f"PRAGMA table_info({table_name})"))
                columns = [row[1] for row in result]
                print(f"   {table_name}: {', '.join(columns)}")

        return True

    except Exception as e:
        logger.error(f"创建表失败: {e}")
        print(f"❌ 创建表失败: {e}")
        return False

def drop_tables():
    """删除CookLikeHOC相关表"""
    try:
        engine = create_engine(DATABASE_URL)

        # 删除表
        Base.metadata.drop_all(bind=engine)

        logger.info("CookLikeHOC相关表删除成功")
        print("✅ CookLikeHOC相关表删除成功")

        return True

    except Exception as e:
        logger.error(f"删除表失败: {e}")
        print(f"❌ 删除表失败: {e}")
        return False

def check_tables():
    """检查表是否存在"""
    try:
        engine = create_engine(DATABASE_URL)

        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name IN ('recipes', 'ingredient_nutrition', 'recipe_recommendations', 'user_recipe_preferences')
            """))
            tables = [row[0] for row in result]

            print(f"📋 存在的表: {', '.join(tables) if tables else '无'}")

            # 检查每个表的记录数
            for table_name in tables:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                count = result.scalar()
                print(f"   {table_name}: {count} 条记录")

        return True

    except Exception as e:
        logger.error(f"检查表失败: {e}")
        print(f"❌ 检查表失败: {e}")
        return False

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='CookLikeHOC数据库迁移工具')
    parser.add_argument('--action', choices=['create', 'drop', 'check'],
                       default='create', help='执行的操作')

    args = parser.parse_args()

    print("🍽️ CookLikeHOC数据库迁移工具")
    print("=" * 50)

    if args.action == 'create':
        success = create_tables()
    elif args.action == 'drop':
        success = drop_tables()
    elif args.action == 'check':
        success = check_tables()

    if success:
        print("\n✅ 操作完成")
    else:
        print("\n❌ 操作失败")
        sys.exit(1)

if __name__ == "__main__":
    main()

__all__ = ["'project_root'", "'logger'", "'create_tables'", "'drop_tables'", "'check_tables'", "'main'"]
