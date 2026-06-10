"""
数据库模型定义
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import Optional, Dict, Any

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库模型定义
使用SQLAlchemy ORM定义数据表结构
支持SQLite开发环境和PostgreSQL生产环境
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, JSON, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID
Base = declarative_base()
import logging
import uuid

logger = logging.getLogger(__name__)

Base = declarative_base()

class User(Base):
    """用户表"""
    __tablename__ = 'users'

    user_id = Column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(200), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False, default='guest', index=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    last_login = Column(DateTime)
    profile_data = Column(JSON)  # 用户配置数据

    # 关系
    glucose_predictions = relationship("GlucosePrediction", back_populates="user", cascade="all, delete-orphan")
    cultural_profiles = relationship("CulturalProfile", back_populates="user", cascade="all, delete-orphan")
    feedback_records = relationship("FeedbackRecord", back_populates="user", cascade="all, delete-orphan")
    workflow_executions = relationship("WorkflowExecution", back_populates="user", cascade="all, delete-orphan")
    experiment_results = relationship("ExperimentResult", back_populates="creator", cascade="all, delete-orphan")
    data_augmentation_records = relationship("DataAugmentationRecord", back_populates="creator", cascade="all, delete-orphan")

class CulturalProfile(Base):
    """文化档案表"""
    __tablename__ = 'cultural_profiles'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(50), ForeignKey('users.user_id', ondelete='CASCADE'), nullable=False, index=True)
    region = Column(String(100), nullable=False, index=True)
    cuisine_type = Column(String(100), nullable=False, index=True)
    flavor_preferences = Column(JSON)  # 存储列表
    cooking_methods = Column(JSON)  # 存储列表
    dietary_restrictions = Column(JSON)  # 存储列表
    religious_restrictions = Column(JSON)  # 存储列表
    spice_tolerance = Column(String(20), default='medium')  # low, medium, high
    meal_timing_preferences = Column(JSON)  # 用餐时间偏好
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 关系
    user = relationship("User", back_populates="cultural_profiles")

    # 索引
    __table_args__ = (
        Index('idx_cultural_user_region', 'user_id', 'region'),
    )

class GlucosePrediction(Base):
    """血糖预测记录表"""
    __tablename__ = 'glucose_predictions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(50), ForeignKey('users.user_id', ondelete='CASCADE'), nullable=False, index=True)
    predicted_glucose = Column(Float, nullable=False, index=True)
    actual_glucose = Column(Float, index=True)  # 实际测量值，用于反馈
    confidence = Column(Float, nullable=False)
    model_type = Column(String(100), nullable=False, index=True)
    model_version = Column(String(50), nullable=False)
    input_features = Column(JSON)  # 输入特征
    prediction_time = Column(DateTime, default=datetime.utcnow, index=True)
    feedback_time = Column(DateTime, index=True)  # 反馈时间
    accuracy_score = Column(Float)  # 预测准确度评分
    context_data = Column(JSON)  # 上下文数据（餐前餐后、运动等）

    # 关系
    user = relationship("User", back_populates="glucose_predictions")

    # 索引
    __table_args__ = (
        Index('idx_glucose_user_time', 'user_id', 'prediction_time'),
        Index('idx_glucose_model_time', 'model_type', 'prediction_time'),
        Index('idx_glucose_confidence', 'confidence'),
        Index('idx_glucose_accuracy', 'accuracy_score'),
        Index('idx_glucose_feedback', 'feedback_time'),
    )

class FoodItem(Base):
    """食物项目表"""
    __tablename__ = 'food_items'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), nullable=False, index=True)
    name_en = Column(String(200), index=True)  # 英文名称
    category = Column(String(100), nullable=False, index=True)
    subcategory = Column(String(100), index=True)
    nutrition = Column(JSON)  # 营养成分
    cultural_tags = Column(JSON)  # 文化标签
    cooking_methods = Column(JSON)  # 烹饪方式
    flavor_profile = Column(JSON)  # 口味特征
    glycemic_index = Column(Float, index=True)
    glycemic_load = Column(Float)
    cultural_region = Column(String(100), index=True)
    allergen_info = Column(JSON)  # 过敏原信息
    is_vegetarian = Column(Boolean, default=False, index=True)
    is_halal = Column(Boolean, default=False, index=True)
    is_vegan = Column(Boolean, default=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 索引
    __table_args__ = (
        Index('idx_food_category_region', 'category', 'cultural_region'),
        Index('idx_food_gi_vegetarian', 'glycemic_index', 'is_vegetarian'),
        Index('idx_food_name_search', 'name'),
        Index('idx_food_name_en_search', 'name_en'),
        Index('idx_food_subcategory', 'subcategory'),
        Index('idx_food_dietary', 'is_vegetarian', 'is_halal', 'is_vegan'),
        Index('idx_food_gi_range', 'glycemic_index'),
    )

class Recipe(Base):
    """菜谱表 - 集成CookLikeHOC数据"""
    __tablename__ = 'recipes'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False, index=True)
    category = Column(String(50), nullable=False, index=True)  # 主食、凉拌、卤菜等
    description = Column(Text)
    ingredients = Column(JSON)  # 食材列表
    cooking_method = Column(String(100))  # 烹饪方式
    cultural_tags = Column(JSON)  # 文化标签
    nutritional_info = Column(JSON)  # 营养信息
    health_tags = Column(JSON)  # 健康标签
    difficulty_level = Column(Integer, default=1)  # 难度等级 1-5
    cooking_time = Column(Integer)  # 烹饪时间(分钟)
    servings = Column(Integer, default=1)  # 份数
    source = Column(String(100), default="CookLikeHOC")  # 来源
    image_url = Column(String(500))  # 图片URL
    instructions = Column(Text)  # 制作步骤
    tips = Column(Text)  # 制作小贴士
    is_active = Column(Boolean, default=True)  # 是否启用
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 关系
    recommendations = relationship("RecipeRecommendation", back_populates="recipe", cascade="all, delete-orphan")

    # 索引
    __table_args__ = (
        Index('idx_recipe_category_active', 'category', 'is_active'),
        Index('idx_recipe_cooking_time', 'cooking_time'),
        Index('idx_recipe_difficulty', 'difficulty_level'),
    )

class RecipeRecommendation(Base):
    """菜谱推荐记录表"""
    __tablename__ = 'recipe_recommendations'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(50), ForeignKey('users.user_id', ondelete='CASCADE'), nullable=False, index=True)
    recipe_id = Column(Integer, ForeignKey('recipes.id', ondelete='CASCADE'), nullable=False, index=True)
    meal_type = Column(String(20), nullable=False)  # breakfast, lunch, dinner, snack
    recommendation_score = Column(Float, nullable=False)  # 推荐分数
    health_score = Column(Float)  # 健康分数
    cultural_score = Column(Float)  # 文化适配分数
    nutritional_score = Column(Float)  # 营养分数
    user_satisfaction = Column(Integer)  # 用户满意度 1-5
    feedback_notes = Column(Text)  # 反馈备注
    recommendation_reason = Column(JSON)  # 推荐理由
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 关系
    user = relationship("User")
    recipe = relationship("Recipe", back_populates="recommendations")

    # 索引
    __table_args__ = (
        Index('idx_recipe_rec_user_meal', 'user_id', 'meal_type'),
        Index('idx_recipe_rec_score', 'recommendation_score'),
    )

class UserRecipePreference(Base):
    """用户菜谱偏好表"""
    __tablename__ = 'user_recipe_preferences'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(50), ForeignKey('users.user_id', ondelete='CASCADE'), nullable=False, index=True)
    preferred_categories = Column(JSON)  # 偏好菜谱类别
    preferred_ingredients = Column(JSON)  # 偏好食材
    dietary_restrictions = Column(JSON)  # 饮食限制
    health_goals = Column(JSON)  # 健康目标
    cultural_background = Column(JSON)  # 文化背景
    cooking_skill_level = Column(Integer, default=1)  # 烹饪技能等级
    available_cooking_time = Column(Integer)  # 可用烹饪时间(分钟)
    kitchen_equipment = Column(JSON)  # 厨房设备
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 关系
    user = relationship("User")

    # 索引
    __table_args__ = (
        Index('idx_user_pref_user_id', 'user_id'),
    )

class CulturalRecommendation(Base):
    """文化推荐记录表"""
    __tablename__ = 'cultural_recommendations'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(50), ForeignKey('users.user_id', ondelete='CASCADE'), nullable=False, index=True)
    meal_type = Column(String(50), nullable=False, index=True)  # breakfast, lunch, dinner, snack
    recommendations = Column(JSON)  # 推荐结果
    cultural_adaptation_score = Column(Float, index=True)
    health_constraints = Column(JSON)  # 健康约束
    nutritional_analysis = Column(JSON)  # 营养分析
    user_satisfaction = Column(Integer)  # 用户满意度 1-5
    feedback_notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    served_at = Column(DateTime)  # 实际用餐时间

    # 关系
    user = relationship("User")

    # 索引
    __table_args__ = (
        Index('idx_cultural_rec_user_meal', 'user_id', 'meal_type'),
        Index('idx_cultural_rec_score_time', 'cultural_adaptation_score', 'created_at'),
    )

class WorkflowDefinition(Base):
    """工作流定义表"""
    __tablename__ = 'workflow_definitions'

    workflow_id = Column(String(100), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(200), nullable=False, index=True)
    description = Column(Text)
    version = Column(String(50), nullable=False, index=True)
    definition = Column(JSON)  # 工作流定义JSON
    created_by = Column(String(50), ForeignKey('users.user_id', ondelete='SET NULL'))
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True, index=True)
    tags = Column(JSON)  # 标签列表
    category = Column(String(100), index=True)  # 工作流分类
    complexity_score = Column(Float)  # 复杂度评分

    # 关系
    creator = relationship("User")
    executions = relationship("WorkflowExecution", back_populates="workflow", cascade="all, delete-orphan")

    # 索引
    __table_args__ = (
        Index('idx_workflow_creator_active', 'created_by', 'is_active'),
        Index('idx_workflow_category_version', 'category', 'version'),
    )

class WorkflowExecution(Base):
    """工作流执行记录表"""
    __tablename__ = 'workflow_executions'

    execution_id = Column(String(100), primary_key=True, default=lambda: str(uuid.uuid4()))
    workflow_id = Column(String(100), ForeignKey('workflow_definitions.workflow_id', ondelete='CASCADE'), nullable=False, index=True)
    user_id = Column(String(50), ForeignKey('users.user_id', ondelete='CASCADE'), nullable=False, index=True)
    status = Column(String(50), nullable=False, index=True)  # running, completed, failed, cancelled
    start_time = Column(DateTime, default=datetime.utcnow, index=True)
    end_time = Column(DateTime, index=True)
    execution_time = Column(Float)  # 执行时间（秒）
    execution_context = Column(JSON)  # 执行上下文
    node_executions = Column(JSON)  # 节点执行详情
    error_details = Column(Text)
    performance_metrics = Column(JSON)  # 性能指标
    resource_usage = Column(JSON)  # 资源使用情况

    # 关系
    workflow = relationship("WorkflowDefinition", back_populates="executions")
    user = relationship("User", back_populates="workflow_executions")

    # 索引
    __table_args__ = (
        Index('idx_execution_workflow_status', 'workflow_id', 'status'),
        Index('idx_execution_user_time', 'user_id', 'start_time'),
        Index('idx_execution_status_time', 'status', 'start_time'),
    )

class SystemMetrics(Base):
    """系统指标表"""
    __tablename__ = 'system_metrics'

    id = Column(Integer, primary_key=True, autoincrement=True)
    metric_type = Column(String(100), nullable=False, index=True)  # execution_time, success_rate, etc.
    metric_value = Column(Float, nullable=False, index=True)
    metric_unit = Column(String(20))  # seconds, percentage, count, etc.
    workflow_id = Column(String(100), index=True)  # 可选，关联特定工作流
    node_id = Column(String(100), index=True)  # 可选，关联特定节点
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    additional_data = Column(JSON)  # 额外数据
    tags = Column(JSON)  # 标签

    # 索引
    __table_args__ = (
        Index('idx_metrics_type_time', 'metric_type', 'timestamp'),
        Index('idx_metrics_workflow_time', 'workflow_id', 'timestamp'),
    )

class Alert(Base):
    """告警记录表"""
    __tablename__ = 'alerts'

    alert_id = Column(String(100), primary_key=True, default=lambda: str(uuid.uuid4()))
    rule_id = Column(String(100), nullable=False, index=True)
    alert_level = Column(String(50), nullable=False, index=True)  # info, warning, error, critical
    message = Column(Text, nullable=False)
    metric_value = Column(Float, nullable=False)
    threshold = Column(Float, nullable=False)
    workflow_id = Column(String(100), index=True)
    node_id = Column(String(100), index=True)
    triggered_at = Column(DateTime, default=datetime.utcnow, index=True)
    resolved_at = Column(DateTime, index=True)
    resolved_by = Column(String(50))
    resolution_note = Column(Text)
    additional_context = Column(JSON)
    notification_sent = Column(Boolean, default=False)

    # 索引
    __table_args__ = (
        Index('idx_alert_level_time', 'alert_level', 'triggered_at'),
        Index('idx_alert_workflow_resolved', 'workflow_id', 'resolved_at'),
    )

class FeedbackRecord(Base):
    """反馈记录表"""
    __tablename__ = 'feedback_records'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(50), ForeignKey('users.user_id', ondelete='CASCADE'), nullable=False, index=True)
    prediction_id = Column(String(100), nullable=False, index=True)
    actual_glucose = Column(Float, nullable=False)
    satisfaction_score = Column(Integer, nullable=False, index=True)  # 1-5分
    accuracy_rating = Column(Integer)  # 1-5分
    recommendation_helpfulness = Column(Integer)  # 1-5分
    additional_info = Column(JSON)
    feedback_time = Column(DateTime, default=datetime.utcnow, index=True)
    feedback_type = Column(String(50), index=True)  # prediction, recommendation, system

    # 关系
    user = relationship("User", back_populates="feedback_records")

    # 索引
    __table_args__ = (
        Index('idx_feedback_user_time', 'user_id', 'feedback_time'),
        Index('idx_feedback_type_score', 'feedback_type', 'satisfaction_score'),
    )

class ExperimentResult(Base):
    """实验结果表"""
    __tablename__ = 'experiment_results'

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(String(100), nullable=False, index=True)
    experiment_name = Column(String(200), nullable=False, index=True)
    model_name = Column(String(200), nullable=False, index=True)
    model_version = Column(String(50), nullable=False)
    accuracy = Column(Float, index=True)
    f1_score = Column(Float, index=True)
    auc = Column(Float, index=True)
    precision = Column(Float)
    recall = Column(Float)
    mae = Column(Float)  # Mean Absolute Error
    rmse = Column(Float)  # Root Mean Square Error
    parameters = Column(JSON)  # 实验参数
    dataset_info = Column(JSON)  # 数据集信息
    training_time = Column(Float)  # 训练时间
    inference_time = Column(Float)  # 推理时间
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    created_by = Column(String(50), ForeignKey('users.user_id', ondelete='SET NULL'))
    experiment_notes = Column(Text)

    # 关系
    creator = relationship("User", back_populates="experiment_results")

    # 索引
    __table_args__ = (
        Index('idx_experiment_model_time', 'model_name', 'created_at'),
        Index('idx_experiment_accuracy_time', 'accuracy', 'created_at'),
    )

class DataAugmentationRecord(Base):
    """数据增强记录表"""
    __tablename__ = 'data_augmentation_records'

    id = Column(Integer, primary_key=True, autoincrement=True)
    original_samples = Column(Integer, nullable=False)
    augmented_samples = Column(Integer, nullable=False)
    augmentation_method = Column(String(100), nullable=False, index=True)  # noise, gan, etc.
    augmentation_parameters = Column(JSON)  # 增强参数
    quality_metrics = Column(JSON)  # 质量评估指标
    data_diversity_score = Column(Float)  # 数据多样性评分
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    created_by = Column(String(50), ForeignKey('users.user_id', ondelete='SET NULL'))
    dataset_name = Column(String(200), index=True)
    processing_time = Column(Float)  # 处理时间

    # 关系
    creator = relationship("User", back_populates="data_augmentation_records")

    # 索引
    __table_args__ = (
        Index('idx_augmentation_method_time', 'augmentation_method', 'created_at'),
        Index('idx_augmentation_dataset_time', 'dataset_name', 'created_at'),
    )

# 创建数据库引擎和会话
def create_database_engine(database_url: str = "sqlite:///academic_system.db", echo: bool = False):
    """创建数据库引擎"""
    engine = create_engine(
        database_url,
        echo=echo,  # 设置为True可以看到SQL语句
        pool_pre_ping=True,
        pool_recycle=300,
        # PostgreSQL特定配置
        connect_args={"check_same_thread": False} if "sqlite" in database_url else {}
    )
    return engine

def create_session_factory(engine):
    """创建会话工厂"""
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal

def create_tables(engine):
    """创建所有表"""
    Base.metadata.create_all(bind=engine)
    logger.info("数据库表创建完成")

# 数据库依赖注入
def get_db():
    """获取数据库会话"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 全局变量
engine = None
SessionLocal = None

def initialize_database(database_url: str = "sqlite:///academic_system.db", echo: bool = False):
    """初始化数据库"""
    global engine, SessionLocal

    engine = create_database_engine(database_url, echo)
    SessionLocal = create_session_factory(engine)
    create_tables(engine)

    logger.info(f"数据库初始化完成: {database_url}")
    return engine, SessionLocal

if __name__ == "__main__":
    # 测试数据库初始化
    engine, SessionLocal = initialize_database()
    print("数据库模型定义完成")

__all__ = ["'logger'", "'Base'", "'User'", "'CulturalProfile'", "'GlucosePrediction'", "'FoodItem'", "'Recipe'", "'RecipeRecommendation'", "'UserRecipePreference'", "'CulturalRecommendation'", "'WorkflowDefinition'", "'WorkflowExecution'", "'SystemMetrics'", "'Alert'", "'FeedbackRecord'", "'ExperimentResult'", "'DataAugmentationRecord'", "'create_database_engine'", "'create_session_factory'", "'create_tables'", "'get_db'", "'engine'", "'SessionLocal'", "'initialize_database'"]
