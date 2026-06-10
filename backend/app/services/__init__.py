# Backend services

"""services 包入口

避免在导入包时强依赖可选模块（如第三方DB/ORM），通过惰性/可选导入降低耦合。
"""
from .ai_service import AIService
try:
    from .recipe_service import RecipeService  # 可选依赖：SQLAlchemy
except Exception:
    RecipeService = None

__all__ = ['AIService', 'RecipeService']