"""
数据库管理器
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
import json

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库管理器
提供数据库操作的统一接口，支持SQLite和PostgreSQL
"""

import logging
logger = logging.getLogger(__name__)
from sqlalchemy.orm import Session
from sqlalchemy import text, func, desc, asc
from datetime import datetime, timedelta
import json

from .models import (
    Base, User, CulturalProfile, GlucosePrediction, FoodItem,
    CulturalRecommendation, WorkflowDefinition, WorkflowExecution,
    SystemMetrics, Alert, FeedbackRecord, ExperimentResult, DataAugmentationRecord
)
from backend.app.models.recipe_models import Recipe, RecipeRecommendation, UserRecipePreference, IngredientNutrition

logger = logging.getLogger(__name__)

class DatabaseManager:
    """数据库管理器"""

    def __init__(self, session: Session):
        self.session = session

    # ==================== 用户管理 ====================

    def create_user(self, username: str, email: str, password_hash: str,
                   role: str = 'guest', profile_data: Optional[Dict] = None) -> User:
        """创建用户"""
        user = User(
            username=username,
            email=email,
            password_hash=password_hash,
            role=role,
            profile_data=profile_data or {}
        )
        self.session.add(user)
        self.session.commit()
        self.session.refresh(user)
        logger.info(f"创建用户: {username}")
        return user

    def get_user_by_username(self, username: str) -> Optional[User]:
        """根据用户名获取用户"""
        return self.session.query(User).filter(User.username == username).first()

    def get_user_by_email(self, email: str) -> Optional[User]:
        """根据邮箱获取用户"""
        return self.session.query(User).filter(User.email == email).first()

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """根据ID获取用户"""
        return self.session.query(User).filter(User.user_id == user_id).first()

    def update_user_last_login(self, user_id: str):
        """更新用户最后登录时间"""
        user = self.get_user_by_id(user_id)
        if user:
            user.last_login = datetime.utcnow()
            self.session.commit()

    def get_users_by_role(self, role: str) -> List[User]:
        """根据角色获取用户列表"""
        return self.session.query(User).filter(User.role == role).all()

    # ==================== 文化档案管理 ====================

    def create_cultural_profile(self, user_id: str, region: str, cuisine_type: str,
                               flavor_preferences: List[str], cooking_methods: List[str],
                               dietary_restrictions: List[str], religious_restrictions: List[str],
                               spice_tolerance: str = 'medium') -> CulturalProfile:
        """创建文化档案"""
        profile = CulturalProfile(
            user_id=user_id,
            region=region,
            cuisine_type=cuisine_type,
            flavor_preferences=flavor_preferences,
            cooking_methods=cooking_methods,
            dietary_restrictions=dietary_restrictions,
            religious_restrictions=religious_restrictions,
            spice_tolerance=spice_tolerance
        )
        self.session.add(profile)
        self.session.commit()
        self.session.refresh(profile)
        logger.info(f"创建文化档案: 用户{user_id}, 地区{region}")
        return profile

    def get_cultural_profile(self, user_id: str) -> Optional[CulturalProfile]:
        """获取用户文化档案"""
        return self.session.query(CulturalProfile).filter(
            CulturalProfile.user_id == user_id
        ).first()

    def update_cultural_profile(self, user_id: str, **kwargs) -> Optional[CulturalProfile]:
        """更新文化档案"""
        profile = self.get_cultural_profile(user_id)
        if profile:
            for key, value in kwargs.items():
                if hasattr(profile, key):
                    setattr(profile, key, value)
            profile.updated_at = datetime.utcnow()
            self.session.commit()
            self.session.refresh(profile)
        return profile

    # ==================== 血糖预测管理 ====================

    def create_glucose_prediction(self, user_id: str, predicted_glucose: float,
                                 confidence: float, model_type: str, model_version: str,
                                 input_features: Dict, context_data: Optional[Dict] = None) -> GlucosePrediction:
        """创建血糖预测记录"""
        prediction = GlucosePrediction(
            user_id=user_id,
            predicted_glucose=predicted_glucose,
            confidence=confidence,
            model_type=model_type,
            model_version=model_version,
            input_features=input_features,
            context_data=context_data or {}
        )
        self.session.add(prediction)
        self.session.commit()
        self.session.refresh(prediction)
        logger.info(f"创建血糖预测: 用户{user_id}, 预测值{predicted_glucose}")
        return prediction

    def update_glucose_prediction_feedback(self, prediction_id: int, actual_glucose: float,
                                          accuracy_score: Optional[float] = None) -> Optional[GlucosePrediction]:
        """更新血糖预测反馈"""
        prediction = self.session.query(GlucosePrediction).filter(
            GlucosePrediction.id == prediction_id
        ).first()
        if prediction:
            prediction.actual_glucose = actual_glucose
            prediction.feedback_time = datetime.utcnow()
            if accuracy_score is not None:
                prediction.accuracy_score = accuracy_score
            self.session.commit()
            self.session.refresh(prediction)
        return prediction

    def get_user_glucose_predictions(self, user_id: str, limit: int = 100,
                                    start_date: Optional[datetime] = None,
                                    end_date: Optional[datetime] = None) -> List[GlucosePrediction]:
        """获取用户血糖预测历史"""
        query = self.session.query(GlucosePrediction).filter(
            GlucosePrediction.user_id == user_id
        )

        if start_date:
            query = query.filter(GlucosePrediction.prediction_time >= start_date)
        if end_date:
            query = query.filter(GlucosePrediction.prediction_time <= end_date)

        return query.order_by(desc(GlucosePrediction.prediction_time)).limit(limit).all()

    def get_model_performance_stats(self, model_type: str, days: int = 30) -> Dict[str, Any]:
        """获取模型性能统计"""
        start_date = datetime.utcnow() - timedelta(days=days)

        query = self.session.query(GlucosePrediction).filter(
            GlucosePrediction.model_type == model_type,
            GlucosePrediction.prediction_time >= start_date,
            GlucosePrediction.actual_glucose.isnot(None)
        )

        predictions = query.all()

        if not predictions:
            return {"count": 0, "accuracy": 0, "mae": 0, "rmse": 0}

        # 计算统计指标
        count = len(predictions)
        errors = [abs(p.predicted_glucose - p.actual_glucose) for p in predictions]
        mae = sum(errors) / count
        rmse = (sum(e**2 for e in errors) / count) ** 0.5

        # 计算准确率（误差在20mg/dL以内）
        accurate_count = sum(1 for e in errors if e <= 20)
        accuracy = accurate_count / count

        return {
            "count": count,
            "accuracy": accuracy,
            "mae": mae,
            "rmse": rmse,
            "avg_confidence": sum(p.confidence for p in predictions) / count
        }

    # ==================== 食物项目管理 ====================

    def create_food_item(self, name: str, category: str, nutrition: Dict,
                        cultural_tags: List[str], cooking_methods: List[str],
                        flavor_profile: Dict, glycemic_index: Optional[float] = None,
                        cultural_region: Optional[str] = None) -> FoodItem:
        """创建食物项目"""
        food_item = FoodItem(
            name=name,
            category=category,
            nutrition=nutrition,
            cultural_tags=cultural_tags,
            cooking_methods=cooking_methods,
            flavor_profile=flavor_profile,
            glycemic_index=glycemic_index,
            cultural_region=cultural_region
        )
        self.session.add(food_item)
        self.session.commit()
        self.session.refresh(food_item)
        logger.info(f"创建食物项目: {name}")
        return food_item

    def search_food_items(self, query: str, category: Optional[str] = None,
                         cultural_region: Optional[str] = None,
                         is_vegetarian: Optional[bool] = None,
                         max_gi: Optional[float] = None) -> List[FoodItem]:
        """搜索食物项目"""
        search_query = self.session.query(FoodItem)

        if query:
            search_query = search_query.filter(
                FoodItem.name.contains(query) | FoodItem.name_en.contains(query)
            )

        if category:
            search_query = search_query.filter(FoodItem.category == category)

        if cultural_region:
            search_query = search_query.filter(FoodItem.cultural_region == cultural_region)

        if is_vegetarian is not None:
            search_query = search_query.filter(FoodItem.is_vegetarian == is_vegetarian)

        if max_gi is not None:
            search_query = search_query.filter(FoodItem.glycemic_index <= max_gi)

        return search_query.all()

    def get_food_items_by_cultural_tags(self, tags: List[str]) -> List[FoodItem]:
        """根据文化标签获取食物项目"""
        # 使用JSON查询（SQLite和PostgreSQL都支持）
        query = self.session.query(FoodItem)
        for tag in tags:
            query = query.filter(FoodItem.cultural_tags.contains([tag]))
        return query.all()

    # ==================== 文化推荐管理 ====================

    def create_cultural_recommendation(self, user_id: str, meal_type: str,
                                     recommendations: Dict, cultural_adaptation_score: float,
                                     health_constraints: Dict,
                                     nutritional_analysis: Optional[Dict] = None) -> CulturalRecommendation:
        """创建文化推荐记录"""
        recommendation = CulturalRecommendation(
            user_id=user_id,
            meal_type=meal_type,
            recommendations=recommendations,
            cultural_adaptation_score=cultural_adaptation_score,
            health_constraints=health_constraints,
            nutritional_analysis=nutritional_analysis or {}
        )
        self.session.add(recommendation)
        self.session.commit()
        self.session.refresh(recommendation)
        logger.info(f"创建文化推荐: 用户{user_id}, 餐型{meal_type}")
        return recommendation

    def get_user_recommendations(self, user_id: str, meal_type: Optional[str] = None,
                                limit: int = 50) -> List[CulturalRecommendation]:
        """获取用户推荐历史"""
        query = self.session.query(CulturalRecommendation).filter(
            CulturalRecommendation.user_id == user_id
        )

        if meal_type:
            query = query.filter(CulturalRecommendation.meal_type == meal_type)

        return query.order_by(desc(CulturalRecommendation.created_at)).limit(limit).all()

    def update_recommendation_feedback(self, recommendation_id: int,
                                     user_satisfaction: int,
                                     feedback_notes: Optional[str] = None) -> Optional[CulturalRecommendation]:
        """更新推荐反馈"""
        recommendation = self.session.query(CulturalRecommendation).filter(
            CulturalRecommendation.id == recommendation_id
        ).first()
        if recommendation:
            recommendation.user_satisfaction = user_satisfaction
            if feedback_notes:
                recommendation.feedback_notes = feedback_notes
            self.session.commit()
            self.session.refresh(recommendation)
        return recommendation

    # ==================== 工作流管理 ====================

    def create_workflow_definition(self, name: str, description: str, version: str,
                                  definition: Dict, created_by: str,
                                  tags: Optional[List[str]] = None,
                                  category: Optional[str] = None) -> WorkflowDefinition:
        """创建工作流定义"""
        workflow = WorkflowDefinition(
            name=name,
            description=description,
            version=version,
            definition=definition,
            created_by=created_by,
            tags=tags or [],
            category=category
        )
        self.session.add(workflow)
        self.session.commit()
        self.session.refresh(workflow)
        logger.info(f"创建工作流定义: {name}")
        return workflow

    def create_workflow_execution(self, workflow_id: str, user_id: str,
                                 execution_context: Dict) -> WorkflowExecution:
        """创建工作流执行记录"""
        execution = WorkflowExecution(
            workflow_id=workflow_id,
            user_id=user_id,
            status='running',
            execution_context=execution_context
        )
        self.session.add(execution)
        self.session.commit()
        self.session.refresh(execution)
        logger.info(f"创建工作流执行: {execution.execution_id}")
        return execution

    def update_workflow_execution(self, execution_id: str, status: str,
                                 node_executions: Optional[Dict] = None,
                                 error_details: Optional[str] = None,
                                 performance_metrics: Optional[Dict] = None) -> Optional[WorkflowExecution]:
        """更新工作流执行状态"""
        execution = self.session.query(WorkflowExecution).filter(
            WorkflowExecution.execution_id == execution_id
        ).first()
        if execution:
            execution.status = status
            if status in ['completed', 'failed', 'cancelled']:
                execution.end_time = datetime.utcnow()
                if execution.start_time:
                    execution.execution_time = (execution.end_time - execution.start_time).total_seconds()

            if node_executions:
                execution.node_executions = node_executions
            if error_details:
                execution.error_details = error_details
            if performance_metrics:
                execution.performance_metrics = performance_metrics

            self.session.commit()
            self.session.refresh(execution)
        return execution

    def get_workflow_executions(self, workflow_id: Optional[str] = None,
                               user_id: Optional[str] = None,
                               status: Optional[str] = None,
                               limit: int = 100) -> List[WorkflowExecution]:
        """获取工作流执行记录"""
        query = self.session.query(WorkflowExecution)

        if workflow_id:
            query = query.filter(WorkflowExecution.workflow_id == workflow_id)
        if user_id:
            query = query.filter(WorkflowExecution.user_id == user_id)
        if status:
            query = query.filter(WorkflowExecution.status == status)

        return query.order_by(desc(WorkflowExecution.start_time)).limit(limit).all()

    # ==================== 系统指标管理 ====================

    def record_system_metric(self, metric_type: str, metric_value: float,
                           metric_unit: str, workflow_id: Optional[str] = None,
                           node_id: Optional[str] = None,
                           additional_data: Optional[Dict] = None,
                           tags: Optional[List[str]] = None) -> SystemMetrics:
        """记录系统指标"""
        metric = SystemMetrics(
            metric_type=metric_type,
            metric_value=metric_value,
            metric_unit=metric_unit,
            workflow_id=workflow_id,
            node_id=node_id,
            additional_data=additional_data or {},
            tags=tags or []
        )
        self.session.add(metric)
        self.session.commit()
        self.session.refresh(metric)
        return metric

    def get_system_metrics(self, metric_type: Optional[str] = None,
                          workflow_id: Optional[str] = None,
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None,
                          limit: int = 1000) -> List[SystemMetrics]:
        """获取系统指标"""
        query = self.session.query(SystemMetrics)

        if metric_type:
            query = query.filter(SystemMetrics.metric_type == metric_type)
        if workflow_id:
            query = query.filter(SystemMetrics.workflow_id == workflow_id)
        if start_date:
            query = query.filter(SystemMetrics.timestamp >= start_date)
        if end_date:
            query = query.filter(SystemMetrics.timestamp <= end_date)

        return query.order_by(desc(SystemMetrics.timestamp)).limit(limit).all()

    def get_metric_statistics(self, metric_type: str, days: int = 7) -> Dict[str, Any]:
        """获取指标统计信息"""
        start_date = datetime.utcnow() - timedelta(days=days)

        metrics = self.session.query(SystemMetrics).filter(
            SystemMetrics.metric_type == metric_type,
            SystemMetrics.timestamp >= start_date
        ).all()

        if not metrics:
            return {"count": 0, "avg": 0, "min": 0, "max": 0, "std": 0}

        values = [m.metric_value for m in metrics]
        count = len(values)
        avg = sum(values) / count
        min_val = min(values)
        max_val = max(values)
        std = (sum((v - avg) ** 2 for v in values) / count) ** 0.5

        return {
            "count": count,
            "avg": avg,
            "min": min_val,
            "max": max_val,
            "std": std
        }

    # ==================== 告警管理 ====================

    def create_alert(self, rule_id: str, alert_level: str, message: str,
                    metric_value: float, threshold: float,
                    workflow_id: Optional[str] = None,
                    node_id: Optional[str] = None,
                    additional_context: Optional[Dict] = None) -> Alert:
        """创建告警"""
        alert = Alert(
            rule_id=rule_id,
            alert_level=alert_level,
            message=message,
            metric_value=metric_value,
            threshold=threshold,
            workflow_id=workflow_id,
            node_id=node_id,
            additional_context=additional_context or {}
        )
        self.session.add(alert)
        self.session.commit()
        self.session.refresh(alert)
        logger.warning(f"创建告警: {alert_level} - {message}")
        return alert

    def resolve_alert(self, alert_id: str, resolved_by: str, resolution_note: str) -> Optional[Alert]:
        """解决告警"""
        alert = self.session.query(Alert).filter(Alert.alert_id == alert_id).first()
        if alert:
            alert.resolved_at = datetime.utcnow()
            alert.resolved_by = resolved_by
            alert.resolution_note = resolution_note
            self.session.commit()
            self.session.refresh(alert)
        return alert

    def get_active_alerts(self, alert_level: Optional[str] = None,
                         workflow_id: Optional[str] = None) -> List[Alert]:
        """获取活跃告警"""
        query = self.session.query(Alert).filter(Alert.resolved_at.is_(None))

        if alert_level:
            query = query.filter(Alert.alert_level == alert_level)
        if workflow_id:
            query = query.filter(Alert.workflow_id == workflow_id)

        return query.order_by(desc(Alert.triggered_at)).all()

    # ==================== 反馈管理 ====================

    def create_feedback_record(self, user_id: str, prediction_id: str,
                              actual_glucose: float, satisfaction_score: int,
                              accuracy_rating: Optional[int] = None,
                              recommendation_helpfulness: Optional[int] = None,
                              additional_info: Optional[Dict] = None,
                              feedback_type: str = 'prediction') -> FeedbackRecord:
        """创建反馈记录"""
        feedback = FeedbackRecord(
            user_id=user_id,
            prediction_id=prediction_id,
            actual_glucose=actual_glucose,
            satisfaction_score=satisfaction_score,
            accuracy_rating=accuracy_rating,
            recommendation_helpfulness=recommendation_helpfulness,
            additional_info=additional_info or {},
            feedback_type=feedback_type
        )
        self.session.add(feedback)
        self.session.commit()
        self.session.refresh(feedback)
        logger.info(f"创建反馈记录: 用户{user_id}, 类型{feedback_type}")
        return feedback

    def get_user_feedback(self, user_id: str, feedback_type: Optional[str] = None,
                         limit: int = 100) -> List[FeedbackRecord]:
        """获取用户反馈记录"""
        query = self.session.query(FeedbackRecord).filter(
            FeedbackRecord.user_id == user_id
        )

        if feedback_type:
            query = query.filter(FeedbackRecord.feedback_type == feedback_type)

        return query.order_by(desc(FeedbackRecord.feedback_time)).limit(limit).all()

    def get_feedback_statistics(self, days: int = 30) -> Dict[str, Any]:
        """获取反馈统计信息"""
        start_date = datetime.utcnow() - timedelta(days=days)

        feedbacks = self.session.query(FeedbackRecord).filter(
            FeedbackRecord.feedback_time >= start_date
        ).all()

        if not feedbacks:
            return {"count": 0, "avg_satisfaction": 0, "avg_accuracy": 0}

        count = len(feedbacks)
        avg_satisfaction = sum(f.satisfaction_score for f in feedbacks) / count
        accuracy_ratings = [f.accuracy_rating for f in feedbacks if f.accuracy_rating is not None]
        avg_accuracy = sum(accuracy_ratings) / len(accuracy_ratings) if accuracy_ratings else 0

        return {
            "count": count,
            "avg_satisfaction": avg_satisfaction,
            "avg_accuracy": avg_accuracy,
            "satisfaction_distribution": {
                str(i): sum(1 for f in feedbacks if f.satisfaction_score == i)
                for i in range(1, 6)
            }
        }

    # ==================== 实验管理 ====================

    def create_experiment_result(self, experiment_id: str, experiment_name: str,
                                model_name: str, model_version: str,
                                accuracy: Optional[float] = None,
                                f1_score: Optional[float] = None,
                                auc: Optional[float] = None,
                                precision: Optional[float] = None,
                                recall: Optional[float] = None,
                                mae: Optional[float] = None,
                                rmse: Optional[float] = None,
                                parameters: Optional[Dict] = None,
                                dataset_info: Optional[Dict] = None,
                                training_time: Optional[float] = None,
                                inference_time: Optional[float] = None,
                                created_by: Optional[str] = None,
                                experiment_notes: Optional[str] = None) -> ExperimentResult:
        """创建实验结果记录"""
        result = ExperimentResult(
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            model_name=model_name,
            model_version=model_version,
            accuracy=accuracy,
            f1_score=f1_score,
            auc=auc,
            precision=precision,
            recall=recall,
            mae=mae,
            rmse=rmse,
            parameters=parameters or {},
            dataset_info=dataset_info or {},
            training_time=training_time,
            inference_time=inference_time,
            created_by=created_by,
            experiment_notes=experiment_notes
        )
        self.session.add(result)
        self.session.commit()
        self.session.refresh(result)
        logger.info(f"创建实验结果: {experiment_name}")
        return result

    def get_experiment_results(self, model_name: Optional[str] = None,
                              experiment_id: Optional[str] = None,
                              created_by: Optional[str] = None,
                              limit: int = 100) -> List[ExperimentResult]:
        """获取实验结果"""
        query = self.session.query(ExperimentResult)

        if model_name:
            query = query.filter(ExperimentResult.model_name == model_name)
        if experiment_id:
            query = query.filter(ExperimentResult.experiment_id == experiment_id)
        if created_by:
            query = query.filter(ExperimentResult.created_by == created_by)

        return query.order_by(desc(ExperimentResult.created_at)).limit(limit).all()

    def get_model_performance_comparison(self, model_names: List[str], days: int = 30) -> Dict[str, Any]:
        """获取模型性能对比"""
        start_date = datetime.utcnow() - timedelta(days=days)

        results = {}
        for model_name in model_names:
            experiments = self.session.query(ExperimentResult).filter(
                ExperimentResult.model_name == model_name,
                ExperimentResult.created_at >= start_date
            ).all()

            if experiments:
                results[model_name] = {
                    "count": len(experiments),
                    "avg_accuracy": sum(e.accuracy for e in experiments if e.accuracy) / len([e for e in experiments if e.accuracy]),
                    "avg_f1": sum(e.f1_score for e in experiments if e.f1_score) / len([e for e in experiments if e.f1_score]),
                    "avg_training_time": sum(e.training_time for e in experiments if e.training_time) / len([e for e in experiments if e.training_time]),
                    "latest_experiment": experiments[0].created_at.isoformat() if experiments else None
                }
            else:
                results[model_name] = {"count": 0, "message": "无实验数据"}

        return results

    # ==================== 数据增强管理 ====================

    def create_data_augmentation_record(self, original_samples: int, augmented_samples: int,
                                       augmentation_method: str, augmentation_parameters: Dict,
                                       quality_metrics: Dict, data_diversity_score: Optional[float] = None,
                                       dataset_name: Optional[str] = None,
                                       processing_time: Optional[float] = None,
                                       created_by: Optional[str] = None) -> DataAugmentationRecord:
        """创建数据增强记录"""
        record = DataAugmentationRecord(
            original_samples=original_samples,
            augmented_samples=augmented_samples,
            augmentation_method=augmentation_method,
            augmentation_parameters=augmentation_parameters,
            quality_metrics=quality_metrics,
            data_diversity_score=data_diversity_score,
            dataset_name=dataset_name,
            processing_time=processing_time,
            created_by=created_by
        )
        self.session.add(record)
        self.session.commit()
        self.session.refresh(record)
        logger.info(f"创建数据增强记录: {augmentation_method}")
        return record

    def get_data_augmentation_records(self, augmentation_method: Optional[str] = None,
                                     dataset_name: Optional[str] = None,
                                     created_by: Optional[str] = None,
                                     limit: int = 100) -> List[DataAugmentationRecord]:
        """获取数据增强记录"""
        query = self.session.query(DataAugmentationRecord)

        if augmentation_method:
            query = query.filter(DataAugmentationRecord.augmentation_method == augmentation_method)
        if dataset_name:
            query = query.filter(DataAugmentationRecord.dataset_name == dataset_name)
        if created_by:
            query = query.filter(DataAugmentationRecord.created_by == created_by)

        return query.order_by(desc(DataAugmentationRecord.created_at)).limit(limit).all()

    def get_augmentation_statistics(self, days: int = 30) -> Dict[str, Any]:
        """获取数据增强统计信息"""
        start_date = datetime.utcnow() - timedelta(days=days)

        records = self.session.query(DataAugmentationRecord).filter(
            DataAugmentationRecord.created_at >= start_date
        ).all()

        if not records:
            return {"count": 0, "total_original": 0, "total_augmented": 0, "methods": {}}

        total_original = sum(r.original_samples for r in records)
        total_augmented = sum(r.augmented_samples for r in records)

        methods = {}
        for record in records:
            method = record.augmentation_method
            if method not in methods:
                methods[method] = {"count": 0, "total_original": 0, "total_augmented": 0}
            methods[method]["count"] += 1
            methods[method]["total_original"] += record.original_samples
            methods[method]["total_augmented"] += record.augmented_samples

        return {
            "count": len(records),
            "total_original": total_original,
            "total_augmented": total_augmented,
            "augmentation_ratio": total_augmented / total_original if total_original > 0 else 0,
            "methods": methods
        }

    # ==================== 通用查询方法 ====================

    def get_database_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        stats = {}

        # 各表记录数
        tables = [User, CulturalProfile, GlucosePrediction, FoodItem,
                 CulturalRecommendation, WorkflowDefinition, WorkflowExecution,
                 SystemMetrics, Alert, FeedbackRecord, ExperimentResult, DataAugmentationRecord]

        for table in tables:
            count = self.session.query(table).count()
            stats[table.__tablename__] = count

        # 活跃用户数（最近30天有活动）
        active_users = self.session.query(User).filter(
            User.last_login >= datetime.utcnow() - timedelta(days=30)
        ).count()
        stats['active_users_30d'] = active_users

        # 今日预测数
        today_predictions = self.session.query(GlucosePrediction).filter(
            func.date(GlucosePrediction.prediction_time) == func.date('now')
        ).count()
        stats['predictions_today'] = today_predictions

        # 活跃告警数
        active_alerts = self.session.query(Alert).filter(
            Alert.resolved_at.is_(None)
        ).count()
        stats['active_alerts'] = active_alerts

        return stats

    def cleanup_old_data(self, days: int = 90):
        """清理旧数据"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        # 清理旧的系统指标
        old_metrics = self.session.query(SystemMetrics).filter(
            SystemMetrics.timestamp < cutoff_date
        ).delete()

        # 清理已解决的旧告警
        old_alerts = self.session.query(Alert).filter(
            Alert.resolved_at.isnot(None),
            Alert.resolved_at < cutoff_date
        ).delete()

        self.session.commit()

        logger.info(f"清理完成: 删除了{old_metrics}条旧指标记录，{old_alerts}条旧告警记录")
        return {"cleaned_metrics": old_metrics, "cleaned_alerts": old_alerts}

    # ==================== 菜谱管理 ====================

    def create_recipe(self, name: str, category: str, description: str = None,
                     ingredients: List[Dict] = None, cooking_method: str = None,
                     cultural_tags: List[str] = None, health_tags: List[str] = None,
                     difficulty_level: int = 1, cooking_time: int = None,
                     servings: int = 1, source: str = "CookLikeHOC",
                     image_url: str = None, instructions: str = None,
                     tips: str = None) -> Recipe:
        """创建菜谱"""

        recipe = Recipe(
            name=name,
            category=category,
            description=description,
            ingredients=ingredients or [],
            cooking_method=cooking_method,
            cultural_tags=cultural_tags or [],
            health_tags=health_tags or [],
            difficulty_level=difficulty_level,
            cooking_time=cooking_time,
            servings=servings,
            source=source,
            image_url=image_url,
            instructions=instructions,
            tips=tips
        )
        self.session.add(recipe)
        self.session.commit()
        self.session.refresh(recipe)
        logger.info(f"创建菜谱: {name}")
        return recipe

    def get_recipe_by_id(self, recipe_id: int) -> Optional[Recipe]:
        """根据ID获取菜谱"""
        from .models import Recipe
        return self.session.query(Recipe).filter(
            Recipe.id == recipe_id,
            Recipe.is_active == True
        ).first()

    def get_recipes(self, category: str = None, health_tags: List[str] = None,
                   cultural_tags: List[str] = None, max_cooking_time: int = None,
                   max_difficulty: int = None, keyword: str = None,
                   limit: int = 20, offset: int = 0) -> List[Recipe]:
        """获取菜谱列表"""
        from .models import Recipe
        from sqlalchemy import or_

        query = self.session.query(Recipe).filter(Recipe.is_active == True)

        if category:
            query = query.filter(Recipe.category == category)

        if health_tags:
            for tag in health_tags:
                query = query.filter(Recipe.health_tags.contains([tag]))

        if cultural_tags:
            for tag in cultural_tags:
                query = query.filter(Recipe.cultural_tags.contains([tag]))

        if max_cooking_time:
            query = query.filter(Recipe.cooking_time <= max_cooking_time)

        if max_difficulty:
            query = query.filter(Recipe.difficulty_level <= max_difficulty)

        if keyword:
            query = query.filter(
                or_(
                    Recipe.name.contains(keyword),
                    Recipe.description.contains(keyword),
                    Recipe.ingredients.contains([{"name": keyword}])
                )
            )

        return query.order_by(desc(Recipe.created_at)).offset(offset).limit(limit).all()

    def create_recipe_recommendation(self, user_id: str, recipe_id: int,
                                   meal_type: str, recommendation_score: float,
                                   health_score: float = None, cultural_score: float = None,
                                   nutritional_score: float = None,
                                   recommendation_reason: Dict = None) -> RecipeRecommendation:
        """创建菜谱推荐记录"""

        recommendation = RecipeRecommendation(
            user_id=user_id,
            recipe_id=recipe_id,
            meal_type=meal_type,
            recommendation_score=recommendation_score,
            health_score=health_score,
            cultural_score=cultural_score,
            nutritional_score=nutritional_score,
            recommendation_reason=recommendation_reason or {}
        )
        self.session.add(recommendation)
        self.session.commit()
        self.session.refresh(recommendation)
        logger.info(f"创建菜谱推荐: 用户{user_id}, 菜谱{recipe_id}")
        return recommendation

    def get_user_recipe_preferences(self, user_id: str) -> Optional[UserRecipePreference]:
        """获取用户菜谱偏好"""
        return self.session.query(UserRecipePreference).filter(
            UserRecipePreference.user_id == user_id
        ).first()

    def update_user_recipe_preferences(self, user_id: str, preferences_data: Dict) -> UserRecipePreference:
        """更新用户菜谱偏好"""

        preferences = self.get_user_recipe_preferences(user_id)
        if not preferences:
            preferences = UserRecipePreference(user_id=user_id)
            self.session.add(preferences)

        # 更新字段
        for key, value in preferences_data.items():
            if hasattr(preferences, key):
                setattr(preferences, key, value)

        self.session.commit()
        self.session.refresh(preferences)
        logger.info(f"更新用户菜谱偏好: {user_id}")
        return preferences

    def get_user_recipe_recommendations(self, user_id: str, limit: int = 20) -> List[RecipeRecommendation]:
        """获取用户菜谱推荐历史"""
        return self.session.query(RecipeRecommendation).filter(
            RecipeRecommendation.user_id == user_id
        ).order_by(desc(RecipeRecommendation.created_at)).limit(limit).all()

    def update_recipe_recommendation_feedback(self, recommendation_id: int,
                                            user_satisfaction: int,
                                            feedback_notes: str = None) -> Optional[RecipeRecommendation]:
        """更新菜谱推荐反馈"""

        recommendation = self.session.query(RecipeRecommendation).filter(
            RecipeRecommendation.id == recommendation_id
        ).first()

        if recommendation:
            recommendation.user_satisfaction = user_satisfaction
            recommendation.feedback_notes = feedback_notes
            self.session.commit()
            self.session.refresh(recommendation)
            logger.info(f"更新菜谱推荐反馈: {recommendation_id}")

        return recommendation

if __name__ == "__main__":
    # 测试数据库管理器
    from .models import initialize_database

    engine, SessionLocal = initialize_database()
    session = SessionLocal()

    try:
        db_manager = DatabaseManager(session)

        # 测试创建用户
        user = db_manager.create_user("test_user", "test@example.com", "hashed_password", "researcher")
        print(f"创建用户: {user.username}")

        # 测试获取统计信息
        stats = db_manager.get_database_stats()
        print(f"数据库统计: {stats}")

    finally:
        session.close()

__all__ = ["'logger'", "'DatabaseManager'"]
