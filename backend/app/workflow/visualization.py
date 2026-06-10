

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工作流可视化模块
实现工作流的图形化界面展示和监控
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class NodeType(Enum):
    """节点类型"""
    INPUT = "input"
    PROCESS = "process"
    DECISION = "decision"
    OUTPUT = "output"
    API_CALL = "api_call"
    MODEL_INFERENCE = "model_inference"
    DATA_TRANSFORM = "data_transform"

class NodeStatus(Enum):
    """节点状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class WorkflowNode:
    """工作流节点"""
    node_id: str
    name: str
    node_type: NodeType
    description: str = ""
    position: Tuple[int, int] = (0, 0)
    config: Dict[str, Any] = field(default_factory=dict)
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    status: NodeStatus = NodeStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkflowEdge:
    """工作流边"""
    edge_id: str
    source_node: str
    target_node: str
    condition: Optional[str] = None
    data_mapping: Dict[str, str] = field(default_factory=dict)

@dataclass
class WorkflowDefinition:
    """工作流定义"""
    workflow_id: str
    name: str
    description: str
    version: str
    nodes: List[WorkflowNode] = field(default_factory=list)
    edges: List[WorkflowEdge] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""
    tags: List[str] = field(default_factory=list)

@dataclass
class WorkflowExecution:
    """工作流执行实例"""
    execution_id: str
    workflow_id: str
    status: str
    start_time: datetime
    end_time: Optional[datetime] = None
    node_executions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    execution_context: Dict[str, Any] = field(default_factory=dict)
    error_details: Optional[str] = None

class WorkflowVisualizationEngine:
    """工作流可视化引擎"""

    def __init__(self):
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self.templates: Dict[str, WorkflowDefinition] = {}

        self._initialize_templates()

    def _initialize_templates(self):
        """初始化工作流模板"""
        # 血糖预测工作流模板
        glucose_prediction_template = self._create_glucose_prediction_template()
        self.templates["glucose_prediction"] = glucose_prediction_template

        # 文化适配推荐工作流模板
        cultural_recommendation_template = self._create_cultural_recommendation_template()
        self.templates["cultural_recommendation"] = cultural_recommendation_template

        # 数据增强工作流模板
        data_enhancement_template = self._create_data_enhancement_template()
        self.templates["data_enhancement"] = data_enhancement_template

        logger.info(f"工作流模板初始化完成，包含 {len(self.templates)} 个模板")

    def _create_glucose_prediction_template(self) -> WorkflowDefinition:
        """创建血糖预测工作流模板"""
        nodes = [
            WorkflowNode(
                node_id="input_data",
                name="数据输入",
                node_type=NodeType.INPUT,
                description="接收患者血糖相关数据",
                position=(100, 100),
                config={"required_fields": ["glucose", "carbohydrates", "exercise"]}
            ),
            WorkflowNode(
                node_id="data_validation",
                name="数据验证",
                node_type=NodeType.PROCESS,
                description="验证输入数据的完整性和有效性",
                position=(300, 100),
                config={"validation_rules": ["glucose > 0", "carbohydrates >= 0"]}
            ),
            WorkflowNode(
                node_id="feature_engineering",
                name="特征工程",
                node_type=NodeType.DATA_TRANSFORM,
                description="提取和转换特征",
                position=(500, 100),
                config={"features": ["time_features", "nutrition_features", "activity_features"]}
            ),
            WorkflowNode(
                node_id="gluformer_prediction",
                name="GluFormer预测",
                node_type=NodeType.MODEL_INFERENCE,
                description="使用GluFormer模型进行血糖预测",
                position=(700, 100),
                config={"model_name": "GluFormer", "confidence_threshold": 0.8}
            ),
            WorkflowNode(
                node_id="result_validation",
                name="结果验证",
                node_type=NodeType.DECISION,
                description="验证预测结果的合理性",
                position=(900, 100),
                config={"validation_range": [50, 400]}
            ),
            WorkflowNode(
                node_id="output_result",
                name="输出结果",
                node_type=NodeType.OUTPUT,
                description="返回预测结果",
                position=(1100, 100),
                config={"output_format": "json"}
            )
        ]

        edges = [
            WorkflowEdge("edge_1", "input_data", "data_validation"),
            WorkflowEdge("edge_2", "data_validation", "feature_engineering"),
            WorkflowEdge("edge_3", "feature_engineering", "gluformer_prediction"),
            WorkflowEdge("edge_4", "gluformer_prediction", "result_validation"),
            WorkflowEdge("edge_5", "result_validation", "output_result", condition="valid")
        ]

        return WorkflowDefinition(
            workflow_id="glucose_prediction_template",
            name="血糖预测工作流",
            description="基于GluFormer模型的血糖预测流程",
            version="1.0",
            nodes=nodes,
            edges=edges,
            tags=["healthcare", "prediction", "gluformer"]
        )

    def _create_cultural_recommendation_template(self) -> WorkflowDefinition:
        """创建文化适配推荐工作流模板"""
        nodes = [
            WorkflowNode(
                node_id="user_profile_input",
                name="用户档案输入",
                node_type=NodeType.INPUT,
                description="接收用户文化档案和健康信息",
                position=(100, 100)
            ),
            WorkflowNode(
                node_id="cultural_analysis",
                name="文化分析",
                node_type=NodeType.PROCESS,
                description="分析用户的饮食文化偏好",
                position=(300, 100)
            ),
            WorkflowNode(
                node_id="health_constraint_check",
                name="健康约束检查",
                node_type=NodeType.DECISION,
                description="检查健康相关约束条件",
                position=(500, 100)
            ),
            WorkflowNode(
                node_id="recommendation_generation",
                name="推荐生成",
                node_type=NodeType.MODEL_INFERENCE,
                description="生成文化适配的膳食推荐",
                position=(700, 100)
            ),
            WorkflowNode(
                node_id="cultural_scoring",
                name="文化评分",
                node_type=NodeType.PROCESS,
                description="对推荐结果进行文化适配评分",
                position=(900, 100)
            ),
            WorkflowNode(
                node_id="recommendation_output",
                name="推荐输出",
                node_type=NodeType.OUTPUT,
                description="输出最终推荐结果",
                position=(1100, 100)
            )
        ]

        edges = [
            WorkflowEdge("edge_1", "user_profile_input", "cultural_analysis"),
            WorkflowEdge("edge_2", "cultural_analysis", "health_constraint_check"),
            WorkflowEdge("edge_3", "health_constraint_check", "recommendation_generation", condition="constraints_met"),
            WorkflowEdge("edge_4", "recommendation_generation", "cultural_scoring"),
            WorkflowEdge("edge_5", "cultural_scoring", "recommendation_output")
        ]

        return WorkflowDefinition(
            workflow_id="cultural_recommendation_template",
            name="文化适配推荐工作流",
            description="基于文化偏好的个性化膳食推荐流程",
            version="1.0",
            nodes=nodes,
            edges=edges,
            tags=["cultural", "recommendation", "personalization"]
        )

    def _create_data_enhancement_template(self) -> WorkflowDefinition:
        """创建数据增强工作流模板"""
        nodes = [
            WorkflowNode(
                node_id="raw_data_input",
                name="原始数据输入",
                node_type=NodeType.INPUT,
                description="接收原始数据集",
                position=(100, 100)
            ),
            WorkflowNode(
                node_id="data_quality_check",
                name="数据质量检查",
                node_type=NodeType.PROCESS,
                description="检查数据质量和完整性",
                position=(300, 100)
            ),
            WorkflowNode(
                node_id="augmentation_strategy",
                name="增强策略选择",
                node_type=NodeType.DECISION,
                description="根据数据特征选择增强策略",
                position=(500, 100)
            ),
            WorkflowNode(
                node_id="noise_augmentation",
                name="噪声增强",
                node_type=NodeType.PROCESS,
                description="添加噪声进行数据增强",
                position=(600, 200)
            ),
            WorkflowNode(
                node_id="gan_augmentation",
                name="GAN增强",
                node_type=NodeType.MODEL_INFERENCE,
                description="使用GAN生成合成数据",
                position=(700, 300)
            ),
            WorkflowNode(
                node_id="data_merging",
                name="数据合并",
                node_type=NodeType.PROCESS,
                description="合并原始数据和增强数据",
                position=(900, 200)
            ),
            WorkflowNode(
                node_id="enhanced_data_output",
                name="增强数据输出",
                node_type=NodeType.OUTPUT,
                description="输出增强后的数据集",
                position=(1100, 200)
            )
        ]

        edges = [
            WorkflowEdge("edge_1", "raw_data_input", "data_quality_check"),
            WorkflowEdge("edge_2", "data_quality_check", "augmentation_strategy"),
            WorkflowEdge("edge_3", "augmentation_strategy", "noise_augmentation", condition="use_noise"),
            WorkflowEdge("edge_4", "augmentation_strategy", "gan_augmentation", condition="use_gan"),
            WorkflowEdge("edge_5", "noise_augmentation", "data_merging"),
            WorkflowEdge("edge_6", "gan_augmentation", "data_merging"),
            WorkflowEdge("edge_7", "data_merging", "enhanced_data_output")
        ]

        return WorkflowDefinition(
            workflow_id="data_enhancement_template",
            name="数据增强工作流",
            description="智能数据增强处理流程",
            version="1.0",
            nodes=nodes,
            edges=edges,
            tags=["data_augmentation", "enhancement", "ml"]
        )

    def create_workflow_from_template(self, template_name: str, workflow_name: str,
                                    created_by: str = "") -> WorkflowDefinition:
        """基于模板创建工作流"""
        if template_name not in self.templates:
            raise ValueError(f"模板 {template_name} 不存在")

        template = self.templates[template_name]
        workflow_id = str(uuid.uuid4())

        # 深拷贝模板
        new_workflow = WorkflowDefinition(
            workflow_id=workflow_id,
            name=workflow_name,
            description=template.description,
            version="1.0",
            nodes=template.nodes.copy(),
            edges=template.edges.copy(),
            created_by=created_by,
            tags=template.tags.copy()
        )

        self.workflows[workflow_id] = new_workflow
        logger.info(f"基于模板 {template_name} 创建工作流: {workflow_name}")

        return new_workflow

    def get_workflow_visualization_data(self, workflow_id: str) -> Dict[str, Any]:
        """获取工作流可视化数据"""
        if workflow_id not in self.workflows:
            raise ValueError(f"工作流 {workflow_id} 不存在")

        workflow = self.workflows[workflow_id]

        # 转换为研究可视化数据格式
        nodes_data = []
        for node in workflow.nodes:
            node_data = {
                "id": node.node_id,
                "label": node.name,
                "type": node.node_type.value,
                "description": node.description,
                "position": {"x": node.position[0], "y": node.position[1]},
                "status": node.status.value,
                "config": node.config,
                "metrics": node.metrics
            }

            # 添加执行信息
            if node.start_time:
                node_data["start_time"] = node.start_time.isoformat()
            if node.end_time:
                node_data["end_time"] = node.end_time.isoformat()
                node_data["duration"] = (node.end_time - node.start_time).total_seconds()
            if node.error_message:
                node_data["error"] = node.error_message

            nodes_data.append(node_data)

        edges_data = []
        for edge in workflow.edges:
            edge_data = {
                "id": edge.edge_id,
                "source": edge.source_node,
                "target": edge.target_node,
                "condition": edge.condition,
                "data_mapping": edge.data_mapping
            }
            edges_data.append(edge_data)

        return {
            "workflow_id": workflow_id,
            "name": workflow.name,
            "description": workflow.description,
            "version": workflow.version,
            "nodes": nodes_data,
            "edges": edges_data,
            "created_at": workflow.created_at.isoformat(),
            "created_by": workflow.created_by,
            "tags": workflow.tags
        }

    def get_workflow_execution_status(self, execution_id: str) -> Dict[str, Any]:
        """获取工作流执行状态"""
        if execution_id not in self.executions:
            raise ValueError(f"执行实例 {execution_id} 不存在")

        execution = self.executions[execution_id]

        status_data = {
            "execution_id": execution_id,
            "workflow_id": execution.workflow_id,
            "status": execution.status,
            "start_time": execution.start_time.isoformat(),
            "node_executions": execution.node_executions,
            "execution_context": execution.execution_context
        }

        if execution.end_time:
            status_data["end_time"] = execution.end_time.isoformat()
            status_data["total_duration"] = (execution.end_time - execution.start_time).total_seconds()

        if execution.error_details:
            status_data["error_details"] = execution.error_details

        return status_data

    def get_templates_list(self) -> List[Dict[str, Any]]:
        """获取工作流模板列表"""
        templates_list = []
        for template_name, template in self.templates.items():
            template_info = {
                "template_name": template_name,
                "name": template.name,
                "description": template.description,
                "version": template.version,
                "tags": template.tags,
                "node_count": len(template.nodes),
                "edge_count": len(template.edges)
            }
            templates_list.append(template_info)

        return templates_list

    def export_workflow_definition(self, workflow_id: str) -> str:
        """导出工作流定义为JSON"""
        if workflow_id not in self.workflows:
            raise ValueError(f"工作流 {workflow_id} 不存在")

        workflow = self.workflows[workflow_id]

        # 转换为可序列化的格式
        workflow_dict = {
            "workflow_id": workflow.workflow_id,
            "name": workflow.name,
            "description": workflow.description,
            "version": workflow.version,
            "created_at": workflow.created_at.isoformat(),
            "created_by": workflow.created_by,
            "tags": workflow.tags,
            "nodes": [
                {
                    "node_id": node.node_id,
                    "name": node.name,
                    "node_type": node.node_type.value,
                    "description": node.description,
                    "position": node.position,
                    "config": node.config,
                    "inputs": node.inputs,
                    "outputs": node.outputs
                }
                for node in workflow.nodes
            ],
            "edges": [
                {
                    "edge_id": edge.edge_id,
                    "source_node": edge.source_node,
                    "target_node": edge.target_node,
                    "condition": edge.condition,
                    "data_mapping": edge.data_mapping
                }
                for edge in workflow.edges
            ]
        }

        return json.dumps(workflow_dict, indent=2, ensure_ascii=False)

# 创建全局可视化引擎实例
workflow_viz_engine = WorkflowVisualizationEngine()

logger.info("工作流可视化引擎已创建")

__all__ = ["'logger'", "'NodeType'", "'NodeStatus'", "'WorkflowNode'", "'WorkflowEdge'", "'WorkflowDefinition'", "'WorkflowExecution'", "'WorkflowVisualizationEngine'", "'workflow_viz_engine'"]
