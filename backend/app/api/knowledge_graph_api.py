"""
多维度个性化健康营养知识图谱API
集成知识图谱、多模态处理和个性化推荐功能
"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
import json

from app.services.integrated_diabetes_service import (
    create_integrated_diabetes_service,
    IntegratedDiabetesService
)

logger = logging.getLogger(__name__)

# 创建API路由器
router = APIRouter(prefix="/api/v3/kg", tags=["Knowledge Graph"])

# 全局服务实例
diabetes_service: Optional[IntegratedDiabetesService] = None

def get_diabetes_service() -> IntegratedDiabetesService:
    """获取糖尿病服务实例"""
    global diabetes_service
    if diabetes_service is None:
        diabetes_service = create_integrated_diabetes_service(
            use_ssa_optimization=True,
            use_knowledge_graph=True
        )
    return diabetes_service

# Pydantic模型
class KnowledgeGraphQueryRequest(BaseModel):
    query: str
    query_type: str = "semantic_search"  # semantic_search, entity_query
    entity_type: Optional[str] = None
    limit: int = 10

class MultimodalProcessRequest(BaseModel):
    text: Optional[str] = None
    structured_data: Optional[Dict[str, Any]] = None

class KGRecommendationRequest(BaseModel):
    user_id: str
    recommendation_type: str = "food"  # food, recipe, exercise
    num_recommendations: int = 10

class KnowledgeGraphStatsResponse(BaseModel):
    status: str
    entity_counts: Optional[Dict[str, int]] = None
    total_entities: Optional[int] = None
    kg_enabled: bool
    multimodal_enabled: bool
    error_message: Optional[str] = None

class KGQueryResponse(BaseModel):
    status: str
    query: str
    query_type: str
    results: List[Dict[str, Any]]
    count: int
    error_message: Optional[str] = None

class MultimodalProcessResponse(BaseModel):
    status: str
    entities: List[Dict[str, Any]]
    relations: List[Dict[str, Any]]
    processed_data: Dict[str, Any]
    fused_embedding: List[float]
    error_message: Optional[str] = None

class KGRecommendationResponse(BaseModel):
    status: str
    recommendations: List[Dict[str, Any]]
    reasoning_summary: str
    total_score: float
    error_message: Optional[str] = None

# API端点
@router.get("/stats", response_model=KnowledgeGraphStatsResponse)
async def get_knowledge_graph_stats():
    """获取知识图谱统计信息"""
    try:
        service = get_diabetes_service()
        stats = service.get_knowledge_graph_stats()

        return KnowledgeGraphStatsResponse(**stats)

    except Exception as e:
        logger.error(f"获取知识图谱统计失败: {e}")
        return KnowledgeGraphStatsResponse(
            status="error",
            kg_enabled=False,
            multimodal_enabled=False,
            error_message=str(e)
        )

@router.post("/query", response_model=KGQueryResponse)
async def query_knowledge_graph(request: KnowledgeGraphQueryRequest):
    """查询知识图谱"""
    try:
        service = get_diabetes_service()
        result = service.query_knowledge_graph(
            query=request.query,
            query_type=request.query_type,
            entity_type=request.entity_type,
            limit=request.limit
        )

        return KGQueryResponse(**result)

    except Exception as e:
        logger.error(f"知识图谱查询失败: {e}")
        return KGQueryResponse(
            status="error",
            query=request.query,
            query_type=request.query_type,
            results=[],
            count=0,
            error_message=str(e)
        )

@router.post("/process-multimodal", response_model=MultimodalProcessResponse)
async def process_multimodal_data(request: MultimodalProcessRequest):
    """处理多模态数据"""
    try:
        service = get_diabetes_service()
        result = service.process_multimodal_data(
            text=request.text,
            structured_data=request.structured_data
        )

        return MultimodalProcessResponse(**result)

    except Exception as e:
        logger.error(f"多模态数据处理失败: {e}")
        return MultimodalProcessResponse(
            status="error",
            entities=[],
            relations=[],
            processed_data={},
            fused_embedding=[],
            error_message=str(e)
        )

@router.post("/process-image")
async def process_image_data(file: UploadFile = File(...)):
    """处理图像数据"""
    try:
        service = get_diabetes_service()

        # 读取图像文件
        image_content = await file.read()

        # 这里可以添加图像处理逻辑
        # 暂时返回基本信息
        return {
            "status": "success",
            "filename": file.filename,
            "content_type": file.content_type,
            "size": len(image_content),
            "message": "图像处理功能开发中"
        }

    except Exception as e:
        logger.error(f"图像处理失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/recommend", response_model=KGRecommendationResponse)
async def get_kg_based_recommendation(request: KGRecommendationRequest):
    """基于知识图谱的个性化推荐"""
    try:
        service = get_diabetes_service()
        result = service.get_kg_based_recommendation(
            user_id=request.user_id,
            recommendation_type=request.recommendation_type,
            num_recommendations=request.num_recommendations
        )

        return KGRecommendationResponse(**result)

    except Exception as e:
        logger.error(f"知识图谱推荐失败: {e}")
        return KGRecommendationResponse(
            status="error",
            recommendations=[],
            reasoning_summary="",
            total_score=0.0,
            error_message=str(e)
        )

@router.get("/entities/{entity_type}")
async def get_entities_by_type(entity_type: str, limit: int = 100):
    """根据类型获取实体"""
    try:
        service = get_diabetes_service()
        entities = service.kg_builder.query_entities(entity_type, limit)

        return {
            "status": "success",
            "entity_type": entity_type,
            "entities": entities,
            "count": len(entities)
        }

    except Exception as e:
        logger.error(f"实体查询失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/entities/{entity_id}/related")
async def get_related_entities(entity_id: str, relation_type: Optional[str] = None):
    """获取相关实体"""
    try:
        service = get_diabetes_service()
        related_entities = service.kg_builder.find_related_entities(entity_id, relation_type)

        return {
            "status": "success",
            "entity_id": entity_id,
            "relation_type": relation_type,
            "related_entities": related_entities,
            "count": len(related_entities)
        }

    except Exception as e:
        logger.error(f"相关实体查询失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/build-nutrition-kg")
async def build_nutrition_knowledge_graph(nutrition_data: List[Dict[str, Any]]):
    """构建营养知识图谱"""
    try:
        service = get_diabetes_service()

        if not service.use_kg:
            raise HTTPException(status_code=400, detail="知识图谱服务未启用")

        success = service.kg_builder.build_nutrition_knowledge_graph(nutrition_data)

        if success:
            return {
                "status": "success",
                "message": "营养知识图谱构建成功",
                "data_count": len(nutrition_data)
            }
        else:
            return {
                "status": "error",
                "message": "营养知识图谱构建失败"
            }

    except Exception as e:
        logger.error(f"营养知识图谱构建失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/health")
async def health_check():
    """健康检查"""
    try:
        service = get_diabetes_service()

        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "features": [
                "多维度个性化健康营养知识图谱",
                "基于知识图谱的个性化推荐",
                "多模态数据处理",
                "语义搜索和知识推理",
                "麻雀搜索算法超参数优化",
                "TCN-GRU混合神经网络",
                "血糖预测精度提升",
                "文化适配分析",
                "可解释性分析"
            ],
            "kg_enabled": service.use_kg,
            "ssa_enabled": service.use_ssa_model,
            "multimodal_enabled": service.multimodal_processor is not None
        }

    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 示例数据端点
@router.get("/sample-data")
async def get_sample_nutrition_data():
    """获取示例营养数据"""
    sample_data = [
        {
            "id": "1",
            "name": "燕麦",
            "calories": 389,
            "protein": 16.9,
            "carbs": 66.2,
            "fat": 6.9,
            "fiber": 10.6,
            "gi_index": 55,
            "category": "谷物"
        },
        {
            "id": "2",
            "name": "鸡胸肉",
            "calories": 165,
            "protein": 31.0,
            "carbs": 0,
            "fat": 3.6,
            "fiber": 0,
            "gi_index": 0,
            "category": "肉类"
        },
        {
            "id": "3",
            "name": "西兰花",
            "calories": 34,
            "protein": 2.8,
            "carbs": 6.6,
            "fat": 0.4,
            "fiber": 2.6,
            "gi_index": 15,
            "category": "蔬菜"
        },
        {
            "id": "4",
            "name": "苹果",
            "calories": 52,
            "protein": 0.3,
            "carbs": 13.8,
            "fat": 0.2,
            "fiber": 2.4,
            "gi_index": 36,
            "category": "水果"
        },
        {
            "id": "5",
            "name": "白米饭",
            "calories": 130,
            "protein": 2.7,
            "carbs": 28.2,
            "fat": 0.3,
            "fiber": 0.4,
            "gi_index": 83,
            "category": "谷物"
        }
    ]

    return {
        "status": "success",
        "sample_data": sample_data,
        "count": len(sample_data),
        "description": "示例营养数据，可用于构建知识图谱"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(router, host="0.0.0.0", port=8002)
