

"""
CookLikeHOC菜谱API端点
提供菜谱管理、推荐、营养分析等API接口
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import logging

from ..database.database_manager import get_db
from ..services.recipe_service import RecipeService
from ..models.recipe_models import RecipeDataValidator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/recipes", tags=["菜谱管理"])

# ==================== 请求/响应模型 ====================

class RecipeCreateRequest(BaseModel):
    """创建菜谱请求"""
    name: str = Field(..., description="菜谱名称")
    category: str = Field(..., description="菜谱类别")
    description: Optional[str] = Field(None, description="菜谱描述")
    ingredients: List[Dict[str, Any]] = Field(..., description="食材列表")
    cooking_method: Optional[str] = Field(None, description="烹饪方式")
    cultural_tags: Optional[List[str]] = Field(None, description="文化标签")
    difficulty_level: Optional[int] = Field(1, description="难度等级")
    cooking_time: Optional[int] = Field(None, description="烹饪时间(分钟)")
    servings: Optional[int] = Field(1, description="份数")
    instructions: Optional[str] = Field(None, description="制作步骤")
    tips: Optional[str] = Field(None, description="制作小贴士")
    image_url: Optional[str] = Field(None, description="图片URL")

class RecipeUpdateRequest(BaseModel):
    """更新菜谱请求"""
    name: Optional[str] = Field(None, description="菜谱名称")
    category: Optional[str] = Field(None, description="菜谱类别")
    description: Optional[str] = Field(None, description="菜谱描述")
    ingredients: Optional[List[Dict[str, Any]]] = Field(None, description="食材列表")
    cooking_method: Optional[str] = Field(None, description="烹饪方式")
    cultural_tags: Optional[List[str]] = Field(None, description="文化标签")
    difficulty_level: Optional[int] = Field(None, description="难度等级")
    cooking_time: Optional[int] = Field(None, description="烹饪时间(分钟)")
    servings: Optional[int] = Field(None, description="份数")
    instructions: Optional[str] = Field(None, description="制作步骤")
    tips: Optional[str] = Field(None, description="制作小贴士")
    image_url: Optional[str] = Field(None, description="图片URL")

class RecipeRecommendationRequest(BaseModel):
    """菜谱推荐请求"""
    user_id: str = Field(..., description="用户ID")
    health_profile: Dict[str, Any] = Field(..., description="健康档案")
    cultural_profile: Dict[str, Any] = Field(..., description="文化档案")
    meal_type: str = Field("lunch", description="餐型")
    limit: int = Field(10, description="推荐数量")

class RecipeFilters(BaseModel):
    """菜谱筛选条件"""
    category: Optional[str] = Field(None, description="菜谱类别")
    health_tags: Optional[List[str]] = Field(None, description="健康标签")
    cultural_tags: Optional[List[str]] = Field(None, description="文化标签")
    max_cooking_time: Optional[int] = Field(None, description="最大烹饪时间")
    max_difficulty: Optional[int] = Field(None, description="最大难度等级")
    keyword: Optional[str] = Field(None, description="关键词搜索")

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
    kitchen_equipment: Optional[List[str]] = Field(None, description="厨房设备")

class RecommendationFeedbackRequest(BaseModel):
    """推荐反馈请求"""
    recommendation_id: int = Field(..., description="推荐ID")
    user_satisfaction: int = Field(..., ge=1, le=5, description="用户满意度(1-5)")
    feedback_notes: Optional[str] = Field(None, description="反馈备注")

# ==================== API端点 ====================

@router.get("/", summary="获取菜谱列表")
async def get_recipes(
    category: Optional[str] = Query(None, description="菜谱类别"),
    health_tags: Optional[List[str]] = Query(None, description="健康标签"),
    cultural_tags: Optional[List[str]] = Query(None, description="文化标签"),
    max_cooking_time: Optional[int] = Query(None, description="最大烹饪时间"),
    max_difficulty: Optional[int] = Query(None, description="最大难度等级"),
    keyword: Optional[str] = Query(None, description="关键词搜索"),
    limit: int = Query(20, ge=1, le=100, description="每页数量"),
    offset: int = Query(0, ge=0, description="偏移量"),
    order_by: str = Query("created_at", description="排序字段"),
    order_direction: str = Query("desc", description="排序方向"),
    db: Session = Depends(get_db)
):
    """获取菜谱列表"""
    try:
        filters = {
            'category': category,
            'health_tags': health_tags,
            'cultural_tags': cultural_tags,
            'max_cooking_time': max_cooking_time,
            'max_difficulty': max_difficulty,
            'keyword': keyword
        }

        # 移除None值
        filters = {k: v for k, v in filters.items() if v is not None}

        recipe_service = RecipeService(db)
        recipes = recipe_service.get_recipes(filters, limit, offset, order_by, order_direction)

        return {
            "success": True,
            "data": [recipe.to_dict() for recipe in recipes],
            "total": len(recipes),
            "limit": limit,
            "offset": offset
        }

    except Exception as e:
        logger.error(f"获取菜谱列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取菜谱列表失败: {str(e)}")

@router.get("/{recipe_id}", summary="获取菜谱详情")
async def get_recipe_detail(
    recipe_id: int = Path(..., description="菜谱ID"),
    db: Session = Depends(get_db)
):
    """获取菜谱详情"""
    try:
        recipe_service = RecipeService(db)
        recipe = recipe_service.get_recipe_by_id(recipe_id)

        if not recipe:
            raise HTTPException(status_code=404, detail="菜谱不存在")

        return {
            "success": True,
            "data": recipe.to_dict()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取菜谱详情失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取菜谱详情失败: {str(e)}")

@router.post("/", summary="创建菜谱")
async def create_recipe(
    recipe_data: RecipeCreateRequest,
    db: Session = Depends(get_db)
):
    """创建新菜谱"""
    try:
        recipe_service = RecipeService(db)
        recipe = recipe_service.create_recipe(recipe_data.dict())

        return {
            "success": True,
            "data": recipe.to_dict(),
            "message": "菜谱创建成功"
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"创建菜谱失败: {e}")
        raise HTTPException(status_code=500, detail=f"创建菜谱失败: {str(e)}")

@router.put("/{recipe_id}", summary="更新菜谱")
async def update_recipe(
    recipe_id: int = Path(..., description="菜谱ID"),
    update_data: RecipeUpdateRequest = Body(...),
    db: Session = Depends(get_db)
):
    """更新菜谱"""
    try:
        recipe_service = RecipeService(db)

        # 移除None值
        update_dict = {k: v for k, v in update_data.dict().items() if v is not None}

        recipe = recipe_service.update_recipe(recipe_id, update_dict)

        if not recipe:
            raise HTTPException(status_code=404, detail="菜谱不存在")

        return {
            "success": True,
            "data": recipe.to_dict(),
            "message": "菜谱更新成功"
        }

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"更新菜谱失败: {e}")
        raise HTTPException(status_code=500, detail=f"更新菜谱失败: {str(e)}")

@router.delete("/{recipe_id}", summary="删除菜谱")
async def delete_recipe(
    recipe_id: int = Path(..., description="菜谱ID"),
    db: Session = Depends(get_db)
):
    """删除菜谱"""
    try:
        recipe_service = RecipeService(db)
        success = recipe_service.delete_recipe(recipe_id)

        if not success:
            raise HTTPException(status_code=404, detail="菜谱不存在")

        return {
            "success": True,
            "message": "菜谱删除成功"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除菜谱失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除菜谱失败: {str(e)}")

@router.post("/recommend", summary="个性化菜谱推荐")
async def recommend_recipes(
    request: RecipeRecommendationRequest,
    db: Session = Depends(get_db)
):
    """个性化菜谱推荐"""
    try:
        recipe_service = RecipeService(db)
        recommendations = recipe_service.generate_personalized_recommendations(
            user_id=request.user_id,
            health_profile=request.health_profile,
            cultural_profile=request.cultural_profile,
            meal_type=request.meal_type,
            limit=request.limit
        )

        return {
            "success": True,
            "data": recommendations,
            "message": f"为您推荐了{len(recommendations)}个菜谱"
        }

    except Exception as e:
        logger.error(f"菜谱推荐失败: {e}")
        raise HTTPException(status_code=500, detail=f"菜谱推荐失败: {str(e)}")

@router.get("/{recipe_id}/nutrition", summary="分析菜谱营养")
async def analyze_recipe_nutrition(
    recipe_id: int = Path(..., description="菜谱ID"),
    user_profile: Optional[Dict[str, Any]] = Body(None, description="用户档案"),
    db: Session = Depends(get_db)
):
    """分析菜谱营养信息"""
    try:
        recipe_service = RecipeService(db)
        recipe = recipe_service.get_recipe_by_id(recipe_id)

        if not recipe:
            raise HTTPException(status_code=404, detail="菜谱不存在")

        # 获取营养信息
        nutritional_info = recipe.nutritional_info or {}
        health_tags = recipe.health_tags or []

        # 基于用户档案的个性化分析
        analysis = {
            "recipe_id": recipe_id,
            "recipe_name": recipe.name,
            "nutritional_info": nutritional_info,
            "health_tags": health_tags,
            "analysis": {
                "calorie_level": "中等" if nutritional_info.get('calories', 0) < 400 else "较高",
                "protein_level": "高" if nutritional_info.get('protein', 0) > 20 else "中等",
                "sugar_level": "低" if nutritional_info.get('sugar', 0) < 10 else "中等",
                "sodium_level": "低" if nutritional_info.get('sodium', 0) < 300 else "中等"
            }
        }

        # 如果有用户档案，提供个性化建议
        if user_profile:
            suggestions = []

            if user_profile.get('diabetes') and nutritional_info.get('sugar', 0) > 15:
                suggestions.append("建议减少糖分摄入")

            if user_profile.get('hypertension') and nutritional_info.get('sodium', 0) > 400:
                suggestions.append("建议减少钠盐摄入")

            if user_profile.get('weight_loss') and nutritional_info.get('calories', 0) > 500:
                suggestions.append("建议控制热量摄入")

            analysis["personalized_suggestions"] = suggestions

        return {
            "success": True,
            "data": analysis
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"分析菜谱营养失败: {e}")
        raise HTTPException(status_code=500, detail=f"分析菜谱营养失败: {str(e)}")

@router.get("/ingredients/search", summary="搜索食材")
async def search_ingredients(
    keyword: str = Query(..., description="搜索关键词"),
    db: Session = Depends(get_db)
):
    """搜索食材"""
    try:
        recipe_service = RecipeService(db)
        ingredients = recipe_service.search_ingredients(keyword)

        return {
            "success": True,
            "data": [ingredient.to_dict() for ingredient in ingredients],
            "total": len(ingredients)
        }

    except Exception as e:
        logger.error(f"搜索食材失败: {e}")
        raise HTTPException(status_code=500, detail=f"搜索食材失败: {str(e)}")

@router.get("/ingredients/{ingredient_name}/nutrition", summary="获取食材营养信息")
async def get_ingredient_nutrition(
    ingredient_name: str = Path(..., description="食材名称"),
    db: Session = Depends(get_db)
):
    """获取食材营养信息"""
    try:
        recipe_service = RecipeService(db)
        nutrition = recipe_service.get_ingredient_nutrition(ingredient_name)

        if not nutrition:
            raise HTTPException(status_code=404, detail="食材营养信息不存在")

        return {
            "success": True,
            "data": nutrition
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取食材营养信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取食材营养信息失败: {str(e)}")

@router.post("/preferences", summary="更新用户偏好")
async def update_user_preferences(
    request: UserPreferencesRequest,
    db: Session = Depends(get_db)
):
    """更新用户偏好"""
    try:
        recipe_service = RecipeService(db)
        preferences = recipe_service.update_user_preferences(
            user_id=request.user_id,
            preferences_data=request.dict(exclude={'user_id'})
        )

        return {
            "success": True,
            "data": preferences.to_dict(),
            "message": "用户偏好更新成功"
        }

    except Exception as e:
        logger.error(f"更新用户偏好失败: {e}")
        raise HTTPException(status_code=500, detail=f"更新用户偏好失败: {str(e)}")

@router.get("/users/{user_id}/preferences", summary="获取用户偏好")
async def get_user_preferences(
    user_id: str = Path(..., description="用户ID"),
    db: Session = Depends(get_db)
):
    """获取用户偏好"""
    try:
        recipe_service = RecipeService(db)
        preferences = recipe_service.get_user_preferences(user_id)

        if not preferences:
            return {
                "success": True,
                "data": None,
                "message": "用户偏好不存在"
            }

        return {
            "success": True,
            "data": preferences.to_dict()
        }

    except Exception as e:
        logger.error(f"获取用户偏好失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取用户偏好失败: {str(e)}")

@router.get("/users/{user_id}/recommendations", summary="获取用户推荐历史")
async def get_user_recommendation_history(
    user_id: str = Path(..., description="用户ID"),
    limit: int = Query(20, ge=1, le=100, description="数量限制"),
    db: Session = Depends(get_db)
):
    """获取用户推荐历史"""
    try:
        recipe_service = RecipeService(db)
        recommendations = recipe_service.get_user_recommendation_history(user_id, limit)

        return {
            "success": True,
            "data": [rec.to_dict() for rec in recommendations],
            "total": len(recommendations)
        }

    except Exception as e:
        logger.error(f"获取用户推荐历史失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取用户推荐历史失败: {str(e)}")

@router.post("/recommendations/feedback", summary="提交推荐反馈")
async def submit_recommendation_feedback(
    request: RecommendationFeedbackRequest,
    db: Session = Depends(get_db)
):
    """提交推荐反馈"""
    try:
        recipe_service = RecipeService(db)
        recommendation = recipe_service.update_recommendation_feedback(
            recommendation_id=request.recommendation_id,
            user_satisfaction=request.user_satisfaction,
            feedback_notes=request.feedback_notes
        )

        if not recommendation:
            raise HTTPException(status_code=404, detail="推荐记录不存在")

        return {
            "success": True,
            "data": recommendation.to_dict(),
            "message": "反馈提交成功"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"提交推荐反馈失败: {e}")
        raise HTTPException(status_code=500, detail=f"提交推荐反馈失败: {str(e)}")

@router.get("/categories", summary="获取菜谱类别列表")
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

@router.get("/health-tags", summary="获取健康标签列表")
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

__all__ = ["'logger'", "'router'", "'RecipeCreateRequest'", "'RecipeUpdateRequest'", "'RecipeRecommendationRequest'", "'RecipeFilters'", "'UserPreferencesRequest'", "'RecommendationFeedbackRequest'"]
