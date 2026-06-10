"""
文化适应端点
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

"""
文化适配API端点
集成CookLikeHOC菜谱推荐功能
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Path
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import logging

from app.core.database import get_db
from app.database.database_manager import DatabaseManager
from app.services.ai_service import ai_service
from app.models.cultural_adaptation import CulturalAdaptationService

# 数据库依赖注入
def get_database_manager(db_session = Depends(get_db)) -> DatabaseManager:
    """获取数据库管理器"""
    return DatabaseManager(db_session)

logger = logging.getLogger(__name__)
router = APIRouter()

class CulturalRecommendationRequest(BaseModel):
    cultural_id: str
    user_preferences: Optional[Dict[str, Any]] = None
    dietary_restrictions: Optional[list] = None

class RecipeRecommendationRequest(BaseModel):
    """菜谱推荐请求"""
    user_id: str = Field(..., description="用户ID")
    health_profile: Dict[str, Any] = Field(..., description="健康档案")
    cultural_profile: Dict[str, Any] = Field(..., description="文化档案")
    meal_type: str = Field("lunch", description="餐型")
    limit: int = Field(10, description="推荐数量")

class UserPreferencesRequest(BaseModel):
    """用户偏好请求"""
    user_id: str = Field(..., description="用户ID")
    preferred_categories: Optional[List[str]] = Field(None, description="偏好菜谱类别")
    preferred_ingredients: Optional[List[str]] = Field(None, description="偏好食材")
    dietary_restrictions: Optional[List[str]] = Field(None, description="饮食限制")
    health_goals: Optional[List[str]] = Field(None, description="健康目标")
    cultural_background: Optional[Dict[str, Any]] = Field(None, description="文化背景")
    cooking_skill_level: Optional[int] = Field(1, description="烹饪技能等级")
    available_cooking_time: Optional[int] = Field(None, description="可用烹饪时间")

class CulturalCompatibilityRequest(BaseModel):
    region: str = Field(..., description="用户地区/菜系区域名称，如 四川/广东/地中海/日式/印度")
    item: str = Field(..., description="评估对象：食材、节庆食物或菜品关键词，如 花椒/清蒸鱼/宫保鸡丁")

class CulturalCompatibilityResponse(BaseModel):
    region: str
    item: str
    compatibility: float = Field(..., ge=0.0, le=1.0, description="文化适配度(0-1，越高越适配)")
    debug: Optional[Dict[str, Any]] = None

@router.post("/compatibility", response_model=CulturalCompatibilityResponse)
async def get_cultural_compatibility(request: CulturalCompatibilityRequest):
    """基于知识图谱TransE嵌入的文化相容性打分"""
    try:
        service = CulturalAdaptationService()
        score = service.cultural_compatibility(request.region, request.item)
        return CulturalCompatibilityResponse(
            region=request.region,
            item=request.item,
            compatibility=score,
            debug={
                "note": "score来源于TransE(h+r≈t)距离映射",
            }
        )
    except Exception as e:
        logger.error(f"文化相容性计算失败: {e}")
        raise HTTPException(status_code=500, detail="文化相容性服务暂时不可用")

class CulturalCompatibilityBulkRequest(BaseModel):
    region: str
    items: List[str]

class CulturalCompatibilityBulkResponse(BaseModel):
    region: str
    results: List[CulturalCompatibilityResponse]

@router.post("/compatibility/bulk", response_model=CulturalCompatibilityBulkResponse)
async def get_cultural_compatibility_bulk(request: CulturalCompatibilityBulkRequest):
    """批量文化相容性打分"""
    try:
        service = CulturalAdaptationService()
        results: List[CulturalCompatibilityResponse] = []
        for item in request.items:
            score = service.cultural_compatibility(request.region, item)
            results.append(CulturalCompatibilityResponse(region=request.region, item=item, compatibility=score))
        return CulturalCompatibilityBulkResponse(region=request.region, results=results)
    except Exception as e:
        logger.error(f"批量文化相容性计算失败: {e}")
        raise HTTPException(status_code=500, detail="批量文化相容性服务暂时不可用")

class KGUpsertTriplesRequest(BaseModel):
    triples: List[List[str]] = Field(..., description="三元组数组[[head, relation, tail], ...]")

class KGUpsertTriplesResponse(BaseModel):
    success: int
    total: int

@router.post("/kg/upsert", response_model=KGUpsertTriplesResponse)
async def kg_upsert_triples(request: KGUpsertTriplesRequest):
    """批量写入三元组到文化知识图谱，触发TransE增量更新"""
    try:
        service = CulturalAdaptationService()
        triples_tuples = [(h, r, t) for h, r, t in request.triples if len([h, r, t]) == 3]
        ok = service.batch_upsert_external_knowledge(triples_tuples)
        return KGUpsertTriplesResponse(success=ok, total=len(request.triples))
    except Exception as e:
        logger.error(f"KG三元组写入失败: {e}")
        raise HTTPException(status_code=500, detail="KG写入服务暂时不可用")

@router.post("/recommendations")
async def get_cultural_recommendations(request: CulturalRecommendationRequest):
    """获取文化适配推荐"""
    try:
        # 模拟文化推荐
        recommendations = {
            "cultural_id": request.cultural_id,
            "dietary_recommendations": [
                "建议选择传统食物",
                "注意营养均衡",
                "适量运动"
            ],
            "cultural_adaptations": {
                "meal_times": {"breakfast": 8, "lunch": 13, "dinner": 19},
                "food_preferences": ["传统", "健康", "均衡"]
            }
        }

        logger.info(f"文化推荐完成: {request.cultural_id}")
        return recommendations

    except Exception as e:
        logger.error(f"文化推荐失败: {e}")
        raise HTTPException(status_code=500, detail="文化推荐服务暂时不可用")

# ==================== 菜谱相关端点 ====================

@router.get("/recipes")
async def get_recipes(
    category: Optional[str] = Query(None, description="菜谱类别"),
    health_tags: Optional[List[str]] = Query(None, description="健康标签"),
    cultural_tags: Optional[List[str]] = Query(None, description="文化标签"),
    max_cooking_time: Optional[int] = Query(None, description="最大烹饪时间"),
    max_difficulty: Optional[int] = Query(None, description="最大难度等级"),
    keyword: Optional[str] = Query(None, description="关键词搜索"),
    limit: int = Query(20, ge=1, le=100, description="每页数量"),
    offset: int = Query(0, ge=0, description="偏移量"),
    db: DatabaseManager = Depends(get_database_manager)
):
    """获取菜谱列表"""
    try:
        recipes = db.get_recipes(
            category=category,
            health_tags=health_tags,
            cultural_tags=cultural_tags,
            max_cooking_time=max_cooking_time,
            max_difficulty=max_difficulty,
            keyword=keyword,
            limit=limit,
            offset=offset
        )

        return {
            "success": True,
            "data": [
                {
                    "id": recipe.id,
                    "name": recipe.name,
                    "category": recipe.category,
                    "description": recipe.description,
                    "ingredients": recipe.ingredients,
                    "cooking_method": recipe.cooking_method,
                    "cultural_tags": recipe.cultural_tags,
                    "health_tags": recipe.health_tags,
                    "difficulty_level": recipe.difficulty_level,
                    "cooking_time": recipe.cooking_time,
                    "servings": recipe.servings,
                    "image_url": recipe.image_url,
                    "created_at": recipe.created_at.isoformat() if recipe.created_at else None
                }
                for recipe in recipes
            ],
            "total": len(recipes),
            "limit": limit,
            "offset": offset
        }

    except Exception as e:
        logger.error(f"获取菜谱列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取菜谱列表失败: {str(e)}")

@router.get("/recipes/{recipe_id}")
async def get_recipe_detail(
    recipe_id: int = Path(..., description="菜谱ID"),
    db: DatabaseManager = Depends(get_database_manager)
):
    """获取菜谱详情"""
    try:
        recipe = db.get_recipe_by_id(recipe_id)

        if not recipe:
            raise HTTPException(status_code=404, detail="菜谱不存在")

        return {
            "success": True,
            "data": {
                "id": recipe.id,
                "name": recipe.name,
                "category": recipe.category,
                "description": recipe.description,
                "ingredients": recipe.ingredients,
                "cooking_method": recipe.cooking_method,
                "cultural_tags": recipe.cultural_tags,
                "health_tags": recipe.health_tags,
                "difficulty_level": recipe.difficulty_level,
                "cooking_time": recipe.cooking_time,
                "servings": recipe.servings,
                "source": recipe.source,
                "image_url": recipe.image_url,
                "instructions": recipe.instructions,
                "tips": recipe.tips,
                "created_at": recipe.created_at.isoformat() if recipe.created_at else None
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取菜谱详情失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取菜谱详情失败: {str(e)}")

@router.post("/recipes/recommend")
async def recommend_recipes(
    request: RecipeRecommendationRequest,
    db: DatabaseManager = Depends(get_database_manager)
):
    """个性化菜谱推荐"""
    try:
        # 简化的推荐逻辑
        recipes = db.get_recipes(limit=request.limit)

        # 基于健康档案和文化档案进行简单筛选
        filtered_recipes = []
        for recipe in recipes:
            # 健康标签匹配
            health_match = True
            if request.health_profile.get('diabetes'):
                # 糖尿病用户偏好低糖食物
                if '低糖' not in (recipe.health_tags or []):
                    health_match = False

            # 文化标签匹配
            cultural_match = True
            user_cultural_tags = request.cultural_profile.get('cultural_tags', [])
            if user_cultural_tags:
                recipe_cultural_tags = recipe.cultural_tags or []
                cultural_match = any(tag in recipe_cultural_tags for tag in user_cultural_tags)

            if health_match and cultural_match:
                filtered_recipes.append(recipe)

        # 生成推荐结果
        recommendations = []
        for recipe in filtered_recipes[:request.limit]:
            recommendation = {
                "recipe": {
                    "id": recipe.id,
                    "name": recipe.name,
                    "category": recipe.category,
                    "description": recipe.description,
                    "cooking_method": recipe.cooking_method,
                    "cultural_tags": recipe.cultural_tags,
                    "health_tags": recipe.health_tags,
                    "difficulty_level": recipe.difficulty_level,
                    "cooking_time": recipe.cooking_time,
                    "servings": recipe.servings,
                    "image_url": recipe.image_url
                },
                "recommendation_score": 0.8,  # 简化评分
                "health_score": 0.8,
                "cultural_score": 0.7,
                "nutritional_score": 0.7,
                "recommendation_reason": {
                    "reasons": ["符合您的健康需求", "符合您的饮食文化偏好"],
                    "scores": {"health": 0.8, "cultural": 0.7, "nutritional": 0.7}
                }
            }
            recommendations.append(recommendation)

            # 记录推荐
            db.create_recipe_recommendation(
                user_id=request.user_id,
                recipe_id=recipe.id,
                meal_type=request.meal_type,
                recommendation_score=0.8,
                health_score=0.8,
                cultural_score=0.7,
                nutritional_score=0.7,
                recommendation_reason=recommendation["recommendation_reason"]
            )

        return {
            "success": True,
            "data": recommendations,
            "message": f"为您推荐了{len(recommendations)}个菜谱"
        }

    except Exception as e:
        logger.error(f"菜谱推荐失败: {e}")
        raise HTTPException(status_code=500, detail=f"菜谱推荐失败: {str(e)}")

@router.post("/recipes/preferences")
async def update_user_preferences(
    request: UserPreferencesRequest,
    db: DatabaseManager = Depends(get_database_manager)
):
    """更新用户偏好"""
    try:
        preferences_data = request.dict(exclude={'user_id'})
        preferences = db.update_user_recipe_preferences(request.user_id, preferences_data)

        return {
            "success": True,
            "data": {
                "user_id": preferences.user_id,
                "preferred_categories": preferences.preferred_categories,
                "preferred_ingredients": preferences.preferred_ingredients,
                "dietary_restrictions": preferences.dietary_restrictions,
                "health_goals": preferences.health_goals,
                "cultural_background": preferences.cultural_background,
                "cooking_skill_level": preferences.cooking_skill_level,
                "available_cooking_time": preferences.available_cooking_time
            },
            "message": "用户偏好更新成功"
        }

    except Exception as e:
        logger.error(f"更新用户偏好失败: {e}")
        raise HTTPException(status_code=500, detail=f"更新用户偏好失败: {str(e)}")

@router.get("/recipes/users/{user_id}/preferences")
async def get_user_preferences(
    user_id: str = Path(..., description="用户ID"),
    db: DatabaseManager = Depends(get_database_manager)
):
    """获取用户偏好"""
    try:
        preferences = db.get_user_recipe_preferences(user_id)

        if not preferences:
            return {
                "success": True,
                "data": None,
                "message": "用户偏好不存在"
            }

        return {
            "success": True,
            "data": {
                "user_id": preferences.user_id,
                "preferred_categories": preferences.preferred_categories,
                "preferred_ingredients": preferences.preferred_ingredients,
                "dietary_restrictions": preferences.dietary_restrictions,
                "health_goals": preferences.health_goals,
                "cultural_background": preferences.cultural_background,
                "cooking_skill_level": preferences.cooking_skill_level,
                "available_cooking_time": preferences.available_cooking_time
            }
        }

    except Exception as e:
        logger.error(f"获取用户偏好失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取用户偏好失败: {str(e)}")

@router.get("/recipes/categories")
async def get_recipe_categories():
    """获取菜谱类别列表"""
    categories = [
        {"name": "主食", "description": "米饭、面条、包子等主食类"},
        {"name": "凉拌", "description": "凉拌菜、沙拉等"},
        {"name": "卤菜", "description": "卤制菜品"},
        {"name": "早餐", "description": "早餐类食物"},
        {"name": "汤", "description": "各种汤品"},
        {"name": "炒菜", "description": "炒制菜品"},
        {"name": "炖菜", "description": "炖制菜品"},
        {"name": "炸品", "description": "油炸食品"},
        {"name": "烤类", "description": "烤制食品"},
        {"name": "烫菜", "description": "烫制菜品"},
        {"name": "煮锅", "description": "煮制菜品"},
        {"name": "砂锅菜", "description": "砂锅菜品"},
        {"name": "蒸菜", "description": "蒸制菜品"},
        {"name": "配料", "description": "调料、配料等"},
        {"name": "饮品", "description": "各种饮品"}
    ]

    return {
        "success": True,
        "data": categories
    }

@router.get("/recipes/health-tags")
async def get_health_tags():
    """获取健康标签列表"""
    health_tags = [
        {"name": "高蛋白", "description": "富含蛋白质"},
        {"name": "高纤维", "description": "富含膳食纤维"},
        {"name": "低脂肪", "description": "低脂肪含量"},
        {"name": "低糖", "description": "低糖分含量"},
        {"name": "低钠", "description": "低钠盐含量"},
        {"name": "低卡路里", "description": "低热量"},
        {"name": "富含维生素", "description": "富含各种维生素"},
        {"name": "富含矿物质", "description": "富含矿物质"},
        {"name": "抗氧化", "description": "具有抗氧化作用"},
        {"name": "易消化", "description": "容易消化吸收"}
    ]

    return {
        "success": True,
        "data": health_tags
    }

@router.post("/recipes/recommend/ai")
async def recommend_recipes_with_ai(
    request: RecipeRecommendationRequest,
    db: DatabaseManager = Depends(get_database_manager)
):
    """使用AI进行个性化菜谱推荐"""
    try:
        # 构建用户档案
        user_profile = {
            "age": request.health_profile.get("age", 30),
            "gender": request.health_profile.get("gender", "未知"),
            "occupation": request.health_profile.get("occupation", "未知")
        }

        # 调用AI服务生成推荐
        ai_result = await ai_service.generate_recipe_recommendations(
            user_profile=user_profile,
            health_profile=request.health_profile,
            cultural_profile=request.cultural_profile,
            meal_type=request.meal_type
        )

        if ai_result["success"]:
            # AI推荐成功，处理推荐结果
            recommendations = []
            for ai_rec in ai_result["data"][:request.limit]:
                # 查找匹配的菜谱
                recipes = db.get_recipes(
                    keyword=ai_rec.get("recipe_name", ""),
                    limit=1
                )

                if recipes:
                    recipe = recipes[0]
                    recommendation = {
                        "recipe": {
                            "id": recipe.id,
                            "name": recipe.name,
                            "category": recipe.category,
                            "description": recipe.description,
                            "cooking_method": recipe.cooking_method,
                            "cultural_tags": recipe.cultural_tags,
                            "health_tags": recipe.health_tags,
                            "difficulty_level": recipe.difficulty_level,
                            "cooking_time": recipe.cooking_time,
                            "servings": recipe.servings,
                            "image_url": recipe.image_url
                        },
                        "recommendation_score": 0.9,  # AI推荐分数更高
                        "health_score": 0.8,
                        "cultural_score": 0.8,
                        "nutritional_score": 0.8,
                        "recommendation_reason": {
                            "reasons": ai_rec.get("recommendation_reason", "AI智能推荐"),
                            "ai_analysis": ai_rec,
                            "scores": {"health": 0.8, "cultural": 0.8, "nutritional": 0.8}
                        },
                        "ai_enhanced": True
                    }
                    recommendations.append(recommendation)

                    # 记录推荐
                    db.create_recipe_recommendation(
                        user_id=request.user_id,
                        recipe_id=recipe.id,
                        meal_type=request.meal_type,
                        recommendation_score=0.9,
                        health_score=0.8,
                        cultural_score=0.8,
                        nutritional_score=0.8,
                        recommendation_reason=recommendation["recommendation_reason"]
                    )

            return {
                "success": True,
                "data": recommendations,
                "message": f"AI为您推荐了{len(recommendations)}个菜谱",
                "ai_source": "qwen-ai",
                "fallback_used": ai_result.get("fallback", False)
            }
        else:
            # AI推荐失败，使用传统推荐
            logger.warning(f"AI推荐失败，使用传统推荐: {ai_result.get('error')}")
            return await recommend_recipes(request, db)

    except Exception as e:
        logger.error(f"AI菜谱推荐失败: {e}")
        # 降级到传统推荐
        return await recommend_recipes(request, db)

__all__ = ["'get_database_manager'", "'logger'", "'router'", "'CulturalRecommendationRequest'", "'RecipeRecommendationRequest'", "'UserPreferencesRequest'"]
