"""
工作流端点
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

"""
增强版工作流API端点
支持智能决策、预测性监控、跨系统集成
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime, timedelta
import asyncio

# 导入增强模块
from backend.app.workflow.intelligent_orchestrator import intelligent_orchestrator, DecisionType
from backend.app.workflow.enhanced_monitoring import enhanced_monitor, AlertSeverity, PredictionType
from backend.app.workflow.enhanced_integration import cross_system_integrator, custom_api_manager

logger = logging.getLogger(__name__)
router = APIRouter()

class WorkflowTriggerRequest(BaseModel):
    workflow_id: str
    parameters: Optional[Dict[str, Any]] = None
    intelligent_routing: bool = True
    adaptive_optimization: bool = True

class IntelligentWorkflowRequest(BaseModel):
    workflow_config: Dict[str, Any]
    user_id: str
    intelligent_features: Optional[Dict[str, Any]] = None
    learning_enabled: bool = True

class IntegrationDataRequest(BaseModel):
    integration_id: str
    data: Dict[str, Any]
    data_format: Optional[str] = "json"

class CustomAPIRequest(BaseModel):
    api_id: str
    request_data: Dict[str, Any]

class PredictiveAlertRequest(BaseModel):
    prediction_type: str
    confidence_threshold: float = 0.7
    prediction_window: int = 60

@router.post("/trigger")
async def trigger_workflow(request: WorkflowTriggerRequest, background_tasks: BackgroundTasks):
    """触发智能工作流"""
    try:
        # 使用智能编排器执行工作流
        execution_context = {
            'workflow_id': request.workflow_id,
            'parameters': request.parameters or {},
            'intelligent_routing': request.intelligent_routing,
            'adaptive_optimization': request.adaptive_optimization,
            'user_id': 'system',  # 可以从认证中获取
            'timestamp': datetime.now()
        }

        # 异步执行工作流
        result = await intelligent_orchestrator.execute_intelligent_workflow(
            request.workflow_id, execution_context
        )

        # 后台任务：更新监控数据
        background_tasks.add_task(
            enhanced_monitor.feed_workflow_data,
            request.workflow_id,
            {
                'execution_time': result.get('execution_time', 0),
                'success_rate': 1.0 if result.get('status') == 'completed' else 0.0,
                'adaptive_adjustments': result.get('adaptive_adjustments', [])
            }
        )

        logger.info(f"智能工作流触发完成: {request.workflow_id}")
        return result

    except Exception as e:
        logger.error(f"智能工作流触发失败: {e}")
        raise HTTPException(status_code=500, detail=f"工作流服务暂时不可用: {str(e)}")

@router.post("/intelligent/register")
async def register_intelligent_workflow(request: IntelligentWorkflowRequest):
    """注册智能工作流"""
    try:
        workflow_config = {
            'workflow_id': f"intelligent_{request.user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'name': request.workflow_config.get('name', '智能工作流'),
            'description': request.workflow_config.get('description', ''),
            'definition': request.workflow_config.get('definition', {}),
            'intelligent_features': request.intelligent_features or {},
            'learning_enabled': request.learning_enabled
        }

        await intelligent_orchestrator.register_intelligent_workflow(workflow_config)

        return {
            "workflow_id": workflow_config['workflow_id'],
            "status": "registered",
            "intelligent_features": workflow_config['intelligent_features'],
            "message": "智能工作流注册成功"
        }

    except Exception as e:
        logger.error(f"智能工作流注册失败: {e}")
        raise HTTPException(status_code=500, detail=f"注册失败: {str(e)}")

@router.post("/intelligent/decision")
async def make_intelligent_decision(
    context: Dict[str, Any],
    decision_type: str = "routing"
):
    """智能决策"""
    try:
        decision_type_enum = DecisionType(decision_type)
        decision_result = await intelligent_orchestrator.decision_engine.make_intelligent_decision(
            context, decision_type_enum
        )

        return {
            "decision": decision_result,
            "timestamp": datetime.now().isoformat(),
            "context": context
        }

    except Exception as e:
        logger.error(f"智能决策失败: {e}")
        raise HTTPException(status_code=500, detail=f"决策失败: {str(e)}")

@router.get("/monitoring/predictive-alerts")
async def get_predictive_alerts(hours: int = 24):
    """获取预测性告警"""
    try:
        alerts = enhanced_monitor.get_predictive_alerts(hours)
        dashboard_data = enhanced_monitor.get_monitoring_dashboard_data()

        return {
            "alerts": alerts,
            "dashboard": dashboard_data,
            "summary": {
                "total_alerts": len(alerts),
                "critical_alerts": len([a for a in alerts if a['severity'] == 'critical']),
                "auto_heal_success_rate": dashboard_data['auto_heal']['success_rate']
            }
        }

    except Exception as e:
        logger.error(f"获取预测性告警失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取告警失败: {str(e)}")

@router.post("/monitoring/feed-data")
async def feed_monitoring_data(
    stream_type: str,
    data: Dict[str, Any],
    background_tasks: BackgroundTasks
):
    """输入监控数据"""
    try:
        from app.app.workflow.enhanced_monitoring import DataStreamType

        stream_type_enum = DataStreamType(stream_type)

        # 异步处理数据流
        background_tasks.add_task(
            enhanced_monitor.data_stream_monitor.feed_data_stream,
            stream_type_enum,
            data
        )

        return {
            "status": "accepted",
            "stream_type": stream_type,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"输入监控数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"数据输入失败: {str(e)}")

@router.get("/monitoring/heal-statistics")
async def get_heal_statistics():
    """获取自愈统计"""
    try:
        statistics = enhanced_monitor.get_heal_action_statistics()
        return {
            "heal_actions": statistics,
            "summary": {
                "total_actions": len(statistics),
                "most_used_action": max(statistics.items(), key=lambda x: x[1]['execution_count'])[0] if statistics else None
            }
        }

    except Exception as e:
        logger.error(f"获取自愈统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取统计失败: {str(e)}")

@router.post("/integration/send-data")
async def send_integration_data(request: IntegrationDataRequest):
    """发送数据到集成系统"""
    try:
        result = await cross_system_integrator.send_data_to_integration(
            request.integration_id,
            request.data
        )

        return {
            "status": "sent",
            "integration_id": request.integration_id,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"发送集成数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"发送失败: {str(e)}")

@router.get("/integration/status")
async def get_integration_status():
    """获取集成状态"""
    try:
        status = cross_system_integrator.get_integration_status()
        return {
            "integrations": status,
            "summary": {
                "total_integrations": len(status),
                "healthy_integrations": len([s for s in status.values() if s['status'] == 'healthy']),
                "enabled_integrations": len([s for s in status.values() if s['enabled']])
            }
        }

    except Exception as e:
        logger.error(f"获取集成状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取状态失败: {str(e)}")

@router.post("/integration/register")
async def register_integration(config: Dict[str, Any]):
    """注册新集成"""
    try:
        from app.app.workflow.enhanced_integration import IntegrationConfig, IntegrationType, ProtocolType, DataFormat

        integration_config = IntegrationConfig(
            integration_id=config['integration_id'],
            name=config['name'],
            integration_type=IntegrationType(config['integration_type']),
            protocol=ProtocolType(config['protocol']),
            endpoint=config['endpoint'],
            authentication=config.get('authentication', {}),
            data_format=DataFormat(config.get('data_format', 'json')),
            timeout=config.get('timeout', 30),
            enabled=config.get('enabled', True)
        )

        await cross_system_integrator.register_integration(integration_config)

        return {
            "status": "registered",
            "integration_id": config['integration_id'],
            "message": "集成注册成功"
        }

    except Exception as e:
        logger.error(f"注册集成失败: {e}")
        raise HTTPException(status_code=500, detail=f"注册失败: {str(e)}")

@router.post("/custom-api/execute")
async def execute_custom_api(request: CustomAPIRequest):
    """执行自定义API"""
    try:
        result = await custom_api_manager.execute_custom_api(
            request.api_id,
            request.request_data
        )

        return {
            "api_id": request.api_id,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"执行自定义API失败: {e}")
        raise HTTPException(status_code=500, detail=f"执行失败: {str(e)}")

@router.get("/custom-api/list")
async def get_custom_api_list():
    """获取自定义API列表"""
    try:
        api_list = custom_api_manager.get_custom_api_list()
        return {
            "apis": api_list,
            "total": len(api_list)
        }

    except Exception as e:
        logger.error(f"获取自定义API列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取列表失败: {str(e)}")

@router.post("/custom-api/register")
async def register_custom_api(api_definition: Dict[str, Any]):
    """注册自定义API"""
    try:
        from app.app.workflow.enhanced_integration import CustomAPIDefinition

        custom_api = CustomAPIDefinition(
            api_id=api_definition['api_id'],
            name=api_definition['name'],
            description=api_definition['description'],
            endpoint=api_definition['endpoint'],
            method=api_definition['method'],
            parameters=api_definition.get('parameters', []),
            response_schema=api_definition.get('response_schema', {}),
            authentication_required=api_definition.get('authentication_required', True),
            rate_limit=api_definition.get('rate_limit'),
            timeout=api_definition.get('timeout', 30),
            enabled=api_definition.get('enabled', True)
        )

        # 创建默认处理器
        async def default_handler(request_data: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "status": "success",
                "message": f"自定义API {custom_api.name} 执行成功",
                "data": request_data
            }

        await custom_api_manager.register_custom_api(custom_api, default_handler)

        return {
            "status": "registered",
            "api_id": api_definition['api_id'],
            "message": "自定义API注册成功"
        }

    except Exception as e:
        logger.error(f"注册自定义API失败: {e}")
        raise HTTPException(status_code=500, detail=f"注册失败: {str(e)}")

@router.get("/intelligent/learning-status")
async def get_learning_status():
    """获取学习状态"""
    try:
        decision_engine = intelligent_orchestrator.decision_engine

        return {
            "learning_enabled": decision_engine.learning_enabled,
            "rules_count": len(decision_engine.rules),
            "performance_history_size": len(decision_engine.performance_history),
            "models_status": {
                model_name: "trained" if hasattr(model, 'fit') else "not_trained"
                for model_name, model in decision_engine.decision_models.items()
            },
            "adaptive_config": {
                "learning_rate": decision_engine.adaptive_config.learning_rate,
                "exploration_rate": decision_engine.adaptive_config.exploration_rate,
                "memory_size": decision_engine.adaptive_config.memory_size
            }
        }

    except Exception as e:
        logger.error(f"获取学习状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取状态失败: {str(e)}")

@router.post("/intelligent/update-rule-performance")
async def update_rule_performance(rule_id: str, success: bool):
    """更新规则性能"""
    try:
        intelligent_orchestrator.decision_engine.update_rule_performance(rule_id, success)

        return {
            "status": "updated",
            "rule_id": rule_id,
            "success": success,
            "message": "规则性能更新成功"
        }

    except Exception as e:
        logger.error(f"更新规则性能失败: {e}")
        raise HTTPException(status_code=500, detail=f"更新失败: {str(e)}")

__all__ = ["'logger'", "'router'", "'WorkflowTriggerRequest'", "'IntelligentWorkflowRequest'", "'IntegrationDataRequest'", "'CustomAPIRequest'", "'PredictiveAlertRequest'"]
