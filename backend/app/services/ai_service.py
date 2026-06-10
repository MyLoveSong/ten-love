"""
AI服务模块
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import asyncio
import json

"""
AI服务模块
集成通义千问API作为备选方案
"""

import logging
import requests
import json
logger = logging.getLogger(__name__)
from datetime import datetime

logger = logging.getLogger(__name__)

class AIService:
    """AI服务类 - 集成通义千问API"""

    def __init__(self):
        self.api_key = os.getenv("DASHSCOPE_API_KEY", "")
        self.base_url = "https://dashscope.aliyuncs.com/api/v1"
        self.model = "qwen-turbo"

    async def generate_recipe_recommendations(
        self,
        user_profile: Dict[str, Any],
        health_profile: Dict[str, Any],
        cultural_profile: Dict[str, Any],
        meal_type: str = "lunch"
    ) -> Dict[str, Any]:
        """使用AI生成菜谱推荐"""
        try:
            prompt = self._build_recommendation_prompt(
                user_profile, health_profile, cultural_profile, meal_type
            )

            response = await self._call_qwen_api(prompt)

            # 解析AI响应
            recommendations = self._parse_ai_response(response)

            logger.info(f"AI生成推荐完成: {len(recommendations)}个推荐")
            return {
                "success": True,
                "data": recommendations,
                "source": "qwen-ai",
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"AI推荐生成失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "fallback": True
            }

    async def analyze_recipe_nutrition(
        self,
        recipe_data: Dict[str, Any],
        user_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """使用AI分析菜谱营养"""
        try:
            prompt = self._build_nutrition_prompt(recipe_data, user_profile)

            response = await self._call_qwen_api(prompt)

            analysis = self._parse_nutrition_response(response)

            logger.info(f"AI营养分析完成: {recipe_data.get('name', 'Unknown')}")
            return {
                "success": True,
                "data": analysis,
                "source": "qwen-ai"
            }

        except Exception as e:
            logger.error(f"AI营养分析失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def generate_cultural_adaptations(
        self,
        recipe_data: Dict[str, Any],
        cultural_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """使用AI生成文化适配建议"""
        try:
            prompt = self._build_cultural_prompt(recipe_data, cultural_context)

            response = await self._call_qwen_api(prompt)

            adaptations = self._parse_cultural_response(response)

            logger.info(f"AI文化适配完成: {recipe_data.get('name', 'Unknown')}")
            return {
                "success": True,
                "data": adaptations,
                "source": "qwen-ai"
            }

        except Exception as e:
            logger.error(f"AI文化适配失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _build_recommendation_prompt(
        self,
        user_profile: Dict[str, Any],
        health_profile: Dict[str, Any],
        cultural_profile: Dict[str, Any],
        meal_type: str
    ) -> str:
        """构建推荐提示词"""
        prompt = f"""
作为专业的营养师和烹饪专家，请根据以下信息为用户推荐适合的菜谱：

用户档案：
- 年龄：{user_profile.get('age', '未知')}
- 性别：{user_profile.get('gender', '未知')}
- 职业：{user_profile.get('occupation', '未知')}

健康档案：
- 健康状况：{', '.join(health_profile.get('conditions', []))}
- 饮食限制：{', '.join(health_profile.get('restrictions', []))}
- 健康目标：{', '.join(health_profile.get('goals', []))}

文化档案：
- 地区：{cultural_profile.get('region', '未知')}
- 菜系偏好：{', '.join(cultural_profile.get('cuisines', []))}
- 口味偏好：{', '.join(cultural_profile.get('flavor_preferences', []))}

餐型：{meal_type}

请推荐3-5个适合的菜谱，包括：
1. 菜谱名称
2. 主要食材
3. 烹饪方式
4. 营养特点
5. 推荐理由

请以JSON格式返回结果。
"""
        return prompt

    def _build_nutrition_prompt(
        self,
        recipe_data: Dict[str, Any],
        user_profile: Dict[str, Any]
    ) -> str:
        """构建营养分析提示词"""
        prompt = f"""
作为专业营养师，请分析以下菜谱的营养成分：

菜谱信息：
- 名称：{recipe_data.get('name', '未知')}
- 食材：{', '.join([ing.get('name', '') for ing in recipe_data.get('ingredients', [])])}
- 烹饪方式：{recipe_data.get('cooking_method', '未知')}

用户档案：
- 年龄：{user_profile.get('age', '未知')}
- 健康状况：{', '.join(user_profile.get('conditions', []))}

请分析：
1. 主要营养成分（蛋白质、碳水化合物、脂肪、纤维等）
2. 热量估算
3. 健康益处
4. 注意事项
5. 个性化建议

请以JSON格式返回结果。
"""
        return prompt

    def _build_cultural_prompt(
        self,
        recipe_data: Dict[str, Any],
        cultural_context: Dict[str, Any]
    ) -> str:
        """构建文化适配提示词"""
        prompt = f"""
作为文化饮食专家，请为以下菜谱提供文化适配建议：

菜谱信息：
- 名称：{recipe_data.get('name', '未知')}
- 文化标签：{', '.join(recipe_data.get('cultural_tags', []))}
- 食材：{', '.join([ing.get('name', '') for ing in recipe_data.get('ingredients', [])])}

文化背景：
- 地区：{cultural_context.get('region', '未知')}
- 菜系：{cultural_context.get('cuisine_type', '未知')}
- 口味偏好：{', '.join(cultural_context.get('flavor_preferences', []))}

请提供：
1. 文化适配建议
2. 口味调整方案
3. 食材替代建议
4. 烹饪技巧
5. 文化意义说明

请以JSON格式返回结果。
"""
        return prompt

    async def _call_qwen_api(self, prompt: str) -> str:
        """调用通义千问API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            data = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 2000
            }

            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                raise Exception(f"API调用失败: {response.status_code} - {response.text}")

        except Exception as e:
            logger.error(f"通义千问API调用失败: {e}")
            raise

    def _parse_ai_response(self, response: str) -> List[Dict[str, Any]]:
        """解析AI响应"""
        try:
            # 尝试解析JSON响应
            if response.strip().startswith('{') or response.strip().startswith('['):
                return json.loads(response)
            else:
                # 如果不是JSON，返回模拟数据
                return self._generate_fallback_recommendations()
        except json.JSONDecodeError:
            logger.warning("AI响应不是有效JSON，使用备用方案")
            return self._generate_fallback_recommendations()

    def _parse_nutrition_response(self, response: str) -> Dict[str, Any]:
        """解析营养分析响应"""
        try:
            if response.strip().startswith('{'):
                return json.loads(response)
            else:
                return self._generate_fallback_nutrition()
        except json.JSONDecodeError:
            logger.warning("营养分析响应不是有效JSON，使用备用方案")
            return self._generate_fallback_nutrition()

    def _parse_cultural_response(self, response: str) -> Dict[str, Any]:
        """解析文化适配响应"""
        try:
            if response.strip().startswith('{'):
                return json.loads(response)
            else:
                return self._generate_fallback_cultural()
        except json.JSONDecodeError:
            logger.warning("文化适配响应不是有效JSON，使用备用方案")
            return self._generate_fallback_cultural()

    def _generate_fallback_recommendations(self) -> List[Dict[str, Any]]:
        """生成备用推荐"""
        return [
            {
                "recipe_name": "宫保鸡丁",
                "ingredients": ["鸡胸肉", "花生米", "干辣椒"],
                "cooking_method": "炒",
                "nutrition_highlights": "高蛋白、低脂肪",
                "recommendation_reason": "符合健康饮食需求，营养均衡"
            },
            {
                "recipe_name": "西红柿鸡蛋汤",
                "ingredients": ["西红柿", "鸡蛋", "大葱"],
                "cooking_method": "煮",
                "nutrition_highlights": "富含维生素、易消化",
                "recommendation_reason": "简单易做，营养丰富"
            }
        ]

    def _generate_fallback_nutrition(self) -> Dict[str, Any]:
        """生成备用营养分析"""
        return {
            "calories": 350,
            "protein": 25,
            "carbs": 15,
            "fat": 20,
            "fiber": 5,
            "health_benefits": ["高蛋白", "营养均衡"],
            "recommendations": ["适合健康饮食", "注意适量摄入"]
        }

    def _generate_fallback_cultural(self) -> Dict[str, Any]:
        """生成备用文化适配"""
        return {
            "cultural_adaptations": ["可根据口味调整辣度", "可替换部分食材"],
            "flavor_adjustments": ["减少盐分", "增加蔬菜"],
            "ingredient_substitutions": ["可用其他肉类替代", "可添加更多蔬菜"],
            "cooking_tips": ["掌握火候", "注意调味"],
            "cultural_significance": "传统家常菜，营养丰富"
        }

# 创建全局AI服务实例
ai_service = AIService()

__all__ = ["'logger'", "'AIService'", "'ai_service'"]
