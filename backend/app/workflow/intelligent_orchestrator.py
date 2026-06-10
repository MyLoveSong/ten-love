"""
智能编排器模块
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import asyncio
import json

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能工作流编排引擎
增强版：支持智能决策、动态调整、自适应优化
"""

import asyncio
import json
import logging
logger = logging.getLogger(__name__)
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

class DecisionType(Enum):
    """决策类型"""
    ROUTING = "routing"           # 路由决策
    PRIORITY = "priority"          # 优先级决策
    RESOURCE = "resource"         # 资源分配决策
    QUALITY = "quality"           # 质量控制决策

class WorkflowState(Enum):
    """工作流状态"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ADAPTING = "adapting"  # 动态调整中

@dataclass
class IntelligentRule:
    """智能规则"""
    rule_id: str
    name: str
    condition: str
    action: str
    priority: int = 1
    confidence_threshold: float = 0.8
    enabled: bool = True
    learning_enabled: bool = True
    execution_count: int = 0
    success_rate: float = 0.0

@dataclass
class AdaptiveConfig:
    """自适应配置"""
    learning_rate: float = 0.1
    exploration_rate: float = 0.2
    memory_size: int = 1000
    update_frequency: int = 100  # 每100次执行更新一次
    performance_window: int = 24  # 24小时性能窗口

class IntelligentDecisionEngine:
    """智能决策引擎"""

    def __init__(self):
        self.rules: Dict[str, IntelligentRule] = {}
        self.decision_models: Dict[str, Any] = {}
        self.performance_history: List[Dict[str, Any]] = []
        self.adaptive_config = AdaptiveConfig()
        self.learning_enabled = True

        # 初始化机器学习模型
        self._initialize_models()

        # 加载预训练规则
        self._load_default_rules()

    def _initialize_models(self):
        """初始化机器学习模型"""
        try:
            # 路由决策模型
            self.decision_models['routing'] = DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )

            # 性能预测模型
            self.decision_models['performance'] = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                random_state=42
            )

            # 资源优化模型
            self.decision_models['resource'] = RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                random_state=42
            )

            logger.info("智能决策模型初始化完成")
        except Exception as e:
            logger.error(f"模型初始化失败: {e}")
            self.learning_enabled = False

    def _load_default_rules(self):
        """加载默认智能规则"""
        default_rules = [
            IntelligentRule(
                rule_id="glucose_priority",
                name="血糖优先级规则",
                condition="glucose_level > 180 OR glucose_level < 70",
                action="increase_priority",
                priority=1,
                confidence_threshold=0.9
            ),
            IntelligentRule(
                rule_id="data_quality",
                name="数据质量规则",
                condition="data_quality_score < 0.8",
                action="trigger_cleaning",
                priority=2,
                confidence_threshold=0.8
            ),
            IntelligentRule(
                rule_id="resource_optimization",
                name="资源优化规则",
                condition="cpu_usage > 80 OR memory_usage > 85",
                action="scale_resources",
                priority=3,
                confidence_threshold=0.7
            ),
            IntelligentRule(
                rule_id="adaptive_routing",
                name="自适应路由规则",
                condition="execution_time > avg_time * 1.5",
                action="reroute_workflow",
                priority=4,
                confidence_threshold=0.8
            )
        ]

        for rule in default_rules:
            self.rules[rule.rule_id] = rule

        logger.info(f"加载了 {len(default_rules)} 个默认智能规则")

    async def make_intelligent_decision(self, context: Dict[str, Any],
                                      decision_type: DecisionType) -> Dict[str, Any]:
        """智能决策"""
        try:
            # 收集决策特征
            features = self._extract_features(context, decision_type)

            # 规则引擎决策
            rule_decision = await self._rule_based_decision(context, decision_type)

            # 机器学习决策
            ml_decision = await self._ml_based_decision(features, decision_type)

            # 融合决策结果
            final_decision = self._fuse_decisions(rule_decision, ml_decision, decision_type)

            # 记录决策历史
            self._record_decision(context, final_decision, decision_type)

            return final_decision

        except Exception as e:
            logger.error(f"智能决策失败: {e}")
            return self._fallback_decision(decision_type)

    def _extract_features(self, context: Dict[str, Any], decision_type: DecisionType) -> np.ndarray:
        """提取决策特征"""
        features = []

        # 基础特征
        features.extend([
            context.get('glucose_level', 0),
            context.get('data_quality_score', 1.0),
            context.get('cpu_usage', 0),
            context.get('memory_usage', 0),
            context.get('execution_time', 0),
            context.get('queue_length', 0),
            context.get('user_priority', 1),
            context.get('workflow_complexity', 1)
        ])

        # 时间特征
        current_hour = datetime.now().hour
        features.extend([
            np.sin(2 * np.pi * current_hour / 24),  # 小时的正弦编码
            np.cos(2 * np.pi * current_hour / 24)   # 小时的余弦编码
        ])

        # 历史性能特征
        if self.performance_history:
            recent_performance = self.performance_history[-10:]  # 最近10次
            avg_performance = np.mean([p.get('success_rate', 0) for p in recent_performance])
            features.append(avg_performance)
        else:
            features.append(0.5)  # 默认值

        return np.array(features).reshape(1, -1)

    async def _rule_based_decision(self, context: Dict[str, Any],
                                 decision_type: DecisionType) -> Dict[str, Any]:
        """基于规则的决策"""
        applicable_rules = []

        for rule in self.rules.values():
            if not rule.enabled:
                continue

            # 评估规则条件
            if self._evaluate_condition(rule.condition, context):
                confidence = self._calculate_rule_confidence(rule, context)
                if confidence >= rule.confidence_threshold:
                    applicable_rules.append((rule, confidence))

        if not applicable_rules:
            return {"action": "no_action", "confidence": 0.0}

        # 选择最高优先级的规则
        best_rule, confidence = max(applicable_rules, key=lambda x: (x[0].priority, x[1]))

        return {
            "action": best_rule.action,
            "confidence": confidence,
            "rule_id": best_rule.rule_id,
            "reasoning": f"规则 {best_rule.name} 触发"
        }

    async def _ml_based_decision(self, features: np.ndarray,
                               decision_type: DecisionType) -> Dict[str, Any]:
        """基于机器学习的决策"""
        if not self.learning_enabled or decision_type.value not in self.decision_models:
            return {"action": "no_action", "confidence": 0.0}

        try:
            model = self.decision_models[decision_type.value]

            if decision_type == DecisionType.ROUTING:
                # 分类决策
                prediction = model.predict(features)[0]
                probabilities = model.predict_proba(features)[0]
                confidence = np.max(probabilities)

                return {
                    "action": f"route_to_{prediction}",
                    "confidence": confidence,
                    "probabilities": probabilities.tolist()
                }
            else:
                # 回归决策
                prediction = model.predict(features)[0]
                confidence = min(1.0, max(0.0, 1.0 - abs(prediction - 0.5) * 2))

                return {
                    "action": f"adjust_{decision_type.value}",
                    "value": prediction,
                    "confidence": confidence
                }

        except Exception as e:
            logger.error(f"ML决策失败: {e}")
            return {"action": "no_action", "confidence": 0.0}

    def _fuse_decisions(self, rule_decision: Dict[str, Any],
                       ml_decision: Dict[str, Any],
                       decision_type: DecisionType) -> Dict[str, Any]:
        """融合决策结果"""
        # 权重配置
        rule_weight = 0.6
        ml_weight = 0.4

        # 计算融合置信度
        rule_conf = rule_decision.get('confidence', 0.0)
        ml_conf = ml_decision.get('confidence', 0.0)
        fused_confidence = rule_weight * rule_conf + ml_weight * ml_conf

        # 选择最佳决策
        if rule_conf > ml_conf:
            final_decision = rule_decision.copy()
            final_decision['fusion_method'] = 'rule_based'
        else:
            final_decision = ml_decision.copy()
            final_decision['fusion_method'] = 'ml_based'

        final_decision['confidence'] = fused_confidence
        final_decision['rule_confidence'] = rule_conf
        final_decision['ml_confidence'] = ml_conf

        return final_decision

    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """评估规则条件"""
        try:
            # 简单的条件评估器
            # 实际应用中可以使用更复杂的表达式解析器
            condition = condition.replace('glucose_level', str(context.get('glucose_level', 0)))
            condition = condition.replace('data_quality_score', str(context.get('data_quality_score', 1.0)))
            condition = condition.replace('cpu_usage', str(context.get('cpu_usage', 0)))
            condition = condition.replace('memory_usage', str(context.get('memory_usage', 0)))
            condition = condition.replace('execution_time', str(context.get('execution_time', 0)))
            condition = condition.replace('avg_time', str(context.get('avg_execution_time', 1.0)))

            return eval(condition)
        except Exception as e:
            logger.error(f"条件评估失败: {e}")
            return False

    def _calculate_rule_confidence(self, rule: IntelligentRule, context: Dict[str, Any]) -> float:
        """计算规则置信度"""
        # 基于历史成功率
        base_confidence = rule.success_rate

        # 基于执行次数调整
        execution_factor = min(1.0, rule.execution_count / 100)

        # 基于上下文匹配度
        context_factor = self._calculate_context_match(rule, context)

        return base_confidence * execution_factor * context_factor

    def _calculate_context_match(self, rule: IntelligentRule, context: Dict[str, Any]) -> float:
        """计算上下文匹配度"""
        # 简化的匹配度计算
        match_score = 0.0
        total_factors = 0

        if 'glucose_level' in context:
            glucose = context['glucose_level']
            if glucose > 180 or glucose < 70:
                match_score += 1.0
            total_factors += 1

        if 'data_quality_score' in context:
            quality = context['data_quality_score']
            if quality < 0.8:
                match_score += 1.0
            total_factors += 1

        return match_score / max(total_factors, 1)

    def _record_decision(self, context: Dict[str, Any], decision: Dict[str, Any],
                        decision_type: DecisionType):
        """记录决策历史"""
        record = {
            'timestamp': datetime.now(),
            'decision_type': decision_type.value,
            'context': context.copy(),
            'decision': decision.copy(),
            'success': True  # 将在后续执行中更新
        }

        self.performance_history.append(record)

        # 保持历史记录大小
        if len(self.performance_history) > self.adaptive_config.memory_size:
            self.performance_history = self.performance_history[-self.adaptive_config.memory_size:]

    def _fallback_decision(self, decision_type: DecisionType) -> Dict[str, Any]:
        """回退决策"""
        fallback_actions = {
            DecisionType.ROUTING: {"action": "default_route", "confidence": 0.5},
            DecisionType.PRIORITY: {"action": "normal_priority", "confidence": 0.5},
            DecisionType.RESOURCE: {"action": "default_allocation", "confidence": 0.5},
            DecisionType.QUALITY: {"action": "standard_quality", "confidence": 0.5}
        }

        return fallback_actions.get(decision_type, {"action": "no_action", "confidence": 0.0})

    def update_rule_performance(self, rule_id: str, success: bool):
        """更新规则性能"""
        if rule_id in self.rules:
            rule = self.rules[rule_id]
            rule.execution_count += 1

            # 更新成功率
            if success:
                rule.success_rate = (rule.success_rate * (rule.execution_count - 1) + 1) / rule.execution_count
            else:
                rule.success_rate = (rule.success_rate * (rule.execution_count - 1)) / rule.execution_count

    async def adaptive_learning(self):
        """自适应学习"""
        if not self.learning_enabled or len(self.performance_history) < 10:
            return

        try:
            # 准备训练数据
            X, y = self._prepare_training_data()

            if len(X) < 5:
                return

            # 训练模型
            for decision_type, model in self.decision_models.items():
                if decision_type == 'routing':
                    model.fit(X, y['routing'])
                else:
                    model.fit(X, y[decision_type])

            logger.info("自适应学习完成")

        except Exception as e:
            logger.error(f"自适应学习失败: {e}")

    def _prepare_training_data(self) -> tuple:
        """准备训练数据"""
        X = []
        y = {'routing': [], 'performance': [], 'resource': []}

        for record in self.performance_history[-100:]:  # 最近100条记录
            context = record['context']
            decision = record['decision']

            # 提取特征
            features = self._extract_features(context, DecisionType.ROUTING)
            X.append(features[0])

            # 提取标签
            if 'routing' in decision.get('action', ''):
                y['routing'].append(1)
            else:
                y['routing'].append(0)

            y['performance'].append(context.get('execution_time', 1.0))
            y['resource'].append(context.get('cpu_usage', 0))

        return np.array(X), y

class IntelligentWorkflowOrchestrator:
    """智能工作流编排器"""

    def __init__(self):
        self.workflows: Dict[str, Dict[str, Any]] = {}
        self.active_executions: Dict[str, Dict[str, Any]] = {}
        self.decision_engine = IntelligentDecisionEngine()
        self.adaptive_scheduler = None
        self.performance_optimizer = None

        # 启动自适应学习任务
        self._start_adaptive_learning()

    def _start_adaptive_learning(self):
        """启动自适应学习任务"""
        # 延迟启动，避免在初始化时创建异步任务
        self._learning_task_started = False

    async def _ensure_learning_task(self):
        """确保学习任务正在运行"""
        if not self._learning_task_started:
            self._learning_task_started = True
            asyncio.create_task(self._learning_loop())

    async def _learning_loop(self):
        """学习循环"""
        while True:
            try:
                await self.decision_engine.adaptive_learning()
                await asyncio.sleep(3600)  # 每小时学习一次
            except Exception as e:
                logger.error(f"自适应学习任务失败: {e}")
                await asyncio.sleep(300)  # 5分钟后重试

    async def register_intelligent_workflow(self, workflow_config: Dict[str, Any]):
        """注册智能工作流"""
        workflow_id = workflow_config['workflow_id']

        # 增强工作流配置
        enhanced_config = {
            **workflow_config,
            'intelligent_features': {
                'adaptive_routing': True,
                'dynamic_scaling': True,
                'quality_monitoring': True,
                'performance_optimization': True
            },
            'learning_config': {
                'learning_enabled': True,
                'feedback_collection': True,
                'performance_tracking': True
            }
        }

        self.workflows[workflow_id] = enhanced_config
        logger.info(f"注册智能工作流: {workflow_id}")

    async def execute_intelligent_workflow(self, workflow_id: str,
                                        input_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行智能工作流"""
        if workflow_id not in self.workflows:
            raise ValueError(f"工作流 {workflow_id} 不存在")

        execution_id = f"exec_{workflow_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        try:
            # 创建执行上下文
            execution_context = {
                'workflow_id': workflow_id,
                'execution_id': execution_id,
                'input_data': input_data,
                'start_time': datetime.now(),
                'state': WorkflowState.RUNNING,
                'performance_metrics': {},
                'adaptive_adjustments': []
            }

            self.active_executions[execution_id] = execution_context

            # 智能执行工作流
            result = await self._intelligent_execution(execution_context)

            # 更新执行状态
            execution_context['state'] = WorkflowState.COMPLETED
            execution_context['end_time'] = datetime.now()
            execution_context['execution_time'] = (
                execution_context['end_time'] - execution_context['start_time']
            ).total_seconds()

            return result

        except Exception as e:
            logger.error(f"智能工作流执行失败: {e}")
            execution_context['state'] = WorkflowState.FAILED
            execution_context['error'] = str(e)
            raise

    async def _intelligent_execution(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """智能执行工作流"""
        workflow_config = self.workflows[context['workflow_id']]
        nodes = workflow_config['definition']['nodes']
        edges = workflow_config['definition']['edges']

        # 构建执行图
        execution_graph = self._build_execution_graph(nodes, edges)

        # 智能节点执行
        results = {}
        execution_order = self._determine_execution_order(execution_graph, context)

        for node_id in execution_order:
            node_config = next(n for n in nodes if n['id'] == node_id)

            # 智能决策
            decision_context = {
                'node_id': node_id,
                'node_type': node_config['type'],
                'current_results': results,
                'workflow_context': context
            }

            # 获取智能决策
            routing_decision = await self.decision_engine.make_intelligent_decision(
                decision_context, DecisionType.ROUTING
            )

            # 执行节点
            node_result = await self._execute_intelligent_node(
                node_config, context, results, routing_decision
            )

            results[node_id] = node_result

            # 动态调整
            if routing_decision.get('action') == 'reroute_workflow':
                await self._dynamic_workflow_adjustment(context, node_id, results)

        return {
            'execution_id': context['execution_id'],
            'status': 'completed',
            'results': results,
            'performance_metrics': context['performance_metrics'],
            'adaptive_adjustments': context['adaptive_adjustments']
        }

    def _build_execution_graph(self, nodes: List[Dict], edges: List[Dict]) -> Dict[str, List[str]]:
        """构建执行图"""
        graph = {node['id']: [] for node in nodes}

        for edge in edges:
            source = edge['source']
            target = edge['target']
            graph[source].append(target)

        return graph

    def _determine_execution_order(self, graph: Dict[str, List[str]],
                                 context: Dict[str, Any]) -> List[str]:
        """确定执行顺序"""
        # 拓扑排序
        in_degree = {node: 0 for node in graph}
        for node in graph:
            for neighbor in graph[node]:
                in_degree[neighbor] += 1

        queue = [node for node, degree in in_degree.items() if degree == 0]
        execution_order = []

        while queue:
            current = queue.pop(0)
            execution_order.append(current)

            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return execution_order

    async def _execute_intelligent_node(self, node_config: Dict[str, Any],
                                     context: Dict[str, Any],
                                     previous_results: Dict[str, Any],
                                     decision: Dict[str, Any]) -> Dict[str, Any]:
        """执行智能节点"""
        node_id = node_config['id']
        node_type = node_config['type']

        # 根据节点类型执行
        if node_type == 'data_source':
            return await self._execute_data_source_node(node_config, context)
        elif node_type == 'preprocessing':
            return await self._execute_preprocessing_node(node_config, context, previous_results)
        elif node_type == 'model':
            return await self._execute_model_node(node_config, context, previous_results, decision)
        elif node_type == 'output':
            return await self._execute_output_node(node_config, context, previous_results)
        else:
            return {'error': f'未知节点类型: {node_type}'}

    async def _execute_data_source_node(self, node_config: Dict[str, Any],
                                       context: Dict[str, Any]) -> Dict[str, Any]:
        """执行数据源节点"""
        # 模拟数据源执行
        return {
            'node_id': node_config['id'],
            'data': context['input_data'],
            'timestamp': datetime.now().isoformat(),
            'quality_score': 0.95
        }

    async def _execute_preprocessing_node(self, node_config: Dict[str, Any],
                                        context: Dict[str, Any],
                                        previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """执行预处理节点"""
        # 模拟预处理执行
        return {
            'node_id': node_config['id'],
            'processed_data': 'preprocessed_data',
            'timestamp': datetime.now().isoformat(),
            'quality_score': 0.98
        }

    async def _execute_model_node(self, node_config: Dict[str, Any],
                                context: Dict[str, Any],
                                previous_results: Dict[str, Any],
                                decision: Dict[str, Any]) -> Dict[str, Any]:
        """执行模型节点"""
        # 根据智能决策调整模型参数
        model_params = self._adjust_model_parameters(decision)

        # 模拟模型执行
        return {
            'node_id': node_config['id'],
            'prediction': 'model_prediction',
            'confidence': decision.get('confidence', 0.8),
            'model_params': model_params,
            'timestamp': datetime.now().isoformat()
        }

    async def _execute_output_node(self, node_config: Dict[str, Any],
                                 context: Dict[str, Any],
                                 previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """执行输出节点"""
        return {
            'node_id': node_config['id'],
            'final_result': previous_results,
            'timestamp': datetime.now().isoformat(),
            'status': 'completed'
        }

    def _adjust_model_parameters(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """根据决策调整模型参数"""
        base_params = {
            'learning_rate': 0.01,
            'batch_size': 32,
            'epochs': 100
        }

        # 根据决策调整参数
        if decision.get('action') == 'increase_priority':
            base_params['learning_rate'] *= 1.2
            base_params['batch_size'] = min(64, base_params['batch_size'] * 2)

        return base_params

    async def _dynamic_workflow_adjustment(self, context: Dict[str, Any],
                                         current_node: str,
                                         results: Dict[str, Any]):
        """动态工作流调整"""
        context['state'] = WorkflowState.ADAPTING

        # 记录调整
        adjustment = {
            'timestamp': datetime.now(),
            'node_id': current_node,
            'adjustment_type': 'dynamic_rerouting',
            'reason': 'performance_optimization'
        }

        context['adaptive_adjustments'].append(adjustment)

        logger.info(f"动态调整工作流: {context['execution_id']} at node {current_node}")

# 创建全局智能编排器实例
intelligent_orchestrator = IntelligentWorkflowOrchestrator()

logger.info("智能工作流编排器已创建")

__all__ = ["'logger'", "'DecisionType'", "'WorkflowState'", "'IntelligentRule'", "'AdaptiveConfig'", "'IntelligentDecisionEngine'", "'IntelligentWorkflowOrchestrator'", "'intelligent_orchestrator'"]
