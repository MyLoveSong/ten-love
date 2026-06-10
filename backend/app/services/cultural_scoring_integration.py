"""
文化评分集成模块
提供Stage1模型与患者个性化服务的集成功能
包含错误处理、缓存、监控等增强功能
"""

import sys
import logging
import time
import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
from functools import lru_cache
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.services.cultural_scoring_service import get_cultural_scoring_service

logger = logging.getLogger(__name__)


class ModelLoadError(Exception):
    """模型加载错误"""
    pass


class PredictionTimeoutError(Exception):
    """预测超时错误"""
    pass


class PredictionError(Exception):
    """预测错误"""
    pass


@dataclass
class CulturalScoringMetrics:
    """文化评分监控指标"""
    call_count: int = 0
    success_count: int = 0
    fallback_count: int = 0
    timeout_count: int = 0
    error_count: int = 0
    total_duration: float = 0.0
    durations: list = field(default_factory=list)

    def record_call(self, duration: float, success: bool, used_fallback: bool,
                   timeout: bool = False, error: bool = False):
        """记录调用指标"""
        self.call_count += 1
        if success:
            self.success_count += 1
        if used_fallback:
            self.fallback_count += 1
        if timeout:
            self.timeout_count += 1
        if error:
            self.error_count += 1
        self.total_duration += duration
        self.durations.append(duration)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        if self.call_count == 0:
            return {
                'total_calls': 0,
                'success_rate': 0.0,
                'fallback_rate': 0.0,
                'timeout_rate': 0.0,
                'error_rate': 0.0,
                'avg_duration': 0.0,
                'p95_duration': 0.0
            }

        avg_duration = self.total_duration / self.call_count
        success_rate = self.success_count / self.call_count
        fallback_rate = self.fallback_count / self.call_count
        timeout_rate = self.timeout_count / self.call_count
        error_rate = self.error_count / self.call_count

        p95_duration = np.percentile(self.durations, 95) if self.durations else 0.0

        return {
            'total_calls': self.call_count,
            'success_rate': success_rate,
            'fallback_rate': fallback_rate,
            'timeout_rate': timeout_rate,
            'error_rate': error_rate,
            'avg_duration': avg_duration,
            'p95_duration': p95_duration,
            'min_duration': min(self.durations) if self.durations else 0.0,
            'max_duration': max(self.durations) if self.durations else 0.0
        }


class CulturalScoreCache:
    """文化评分缓存管理器"""

    def __init__(self, maxsize: int = 1000, ttl: int = 3600):
        self.cache: Dict[str, Tuple[float, float]] = {}
        self.maxsize = maxsize
        self.ttl = ttl

    def _generate_cache_key(self, food_info: Dict, cultural_profile: Dict) -> str:
        """生成缓存键"""
        key_data = {
            'food': sorted(food_info.items()),
            'cultural': sorted(cultural_profile.items())
        }
        key_str = json.dumps(key_data, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(key_str.encode('utf-8')).hexdigest()

    def get(self, food_info: Dict, cultural_profile: Dict) -> Optional[float]:
        """获取缓存值"""
        key = self._generate_cache_key(food_info, cultural_profile)
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            else:
                # TTL过期，删除缓存
                del self.cache[key]
        return None

    def set(self, food_info: Dict, cultural_profile: Dict, score: float):
        """设置缓存值"""
        key = self._generate_cache_key(food_info, cultural_profile)

        # 如果缓存已满，删除最旧的条目
        if len(self.cache) >= self.maxsize and key not in self.cache:
            oldest_key = min(self.cache.keys(),
                           key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]

        self.cache[key] = (score, time.time())

    def clear(self):
        """清空缓存"""
        self.cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return {
            'cache_size': len(self.cache),
            'maxsize': self.maxsize,
            'ttl': self.ttl
        }


class CulturalScoringIntegration:
    """
    文化评分集成服务
    提供Stage1模型与患者个性化服务的集成，包含完整的错误处理、缓存、监控
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化文化评分集成服务

        Args:
            config: 配置字典，包含权重、缓存、超时等配置
        """
        # 默认配置
        self.config = {
            'scoring_weights': {
                'cultural': 0.4,
                'health': 0.4,
                'nutritional': 0.2,
                'fallback_cultural_score': 0.8
            },
            'feature_flags': {
                'use_stage1_model': True,
                'stage1_model_rollout_percentage': 100,
                'enable_cache': True,
                'enable_metrics': True
            },
            'cache_config': {
                'maxsize': 1000,
                'ttl': 3600
            },
            'timeout_config': {
                'cultural_scoring_timeout': 5.0
            }
        }

        # 合并用户配置
        if config:
            self._merge_config(config)

        # 初始化文化评分服务
        self.cultural_scoring_service = get_cultural_scoring_service()

        # 初始化缓存
        if self.config['feature_flags']['enable_cache']:
            cache_cfg = self.config['cache_config']
            self.cache = CulturalScoreCache(
                maxsize=cache_cfg['maxsize'],
                ttl=cache_cfg['ttl']
            )
        else:
            self.cache = None

        # 初始化监控指标
        if self.config['feature_flags']['enable_metrics']:
            self.metrics = CulturalScoringMetrics()
        else:
            self.metrics = None

        logger.info("文化评分集成服务初始化完成")

    def _merge_config(self, user_config: Dict[str, Any]):
        """合并用户配置到默认配置"""
        def deep_merge(base: dict, update: dict):
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value

        deep_merge(self.config, user_config)

    def _should_use_stage1_model(self, user_id: str) -> bool:
        """判断是否应该使用Stage1模型（灰度发布）"""
        if not self.config['feature_flags']['use_stage1_model']:
            return False

        rollout_percentage = self.config['feature_flags']['stage1_model_rollout_percentage']
        if rollout_percentage >= 100:
            return True

        # 基于用户ID的灰度策略
        user_hash = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        return (user_hash % 100) < rollout_percentage

    def _calculate_rule_based_cultural_score(
        self,
        recipe: Any,
        cultural_profile: Dict[str, Any]
    ) -> float:
        """规则基础的文化分数计算（降级方案）"""
        score = self.config['scoring_weights']['fallback_cultural_score']

        # 提取菜品文化标签
        if hasattr(recipe, 'cultural_tags'):
            recipe_cultural_tags = recipe.cultural_tags or []
        elif isinstance(recipe, dict):
            recipe_cultural_tags = recipe.get('cultural_tags', [])
        else:
            recipe_cultural_tags = []

        # 提取用户文化偏好
        user_cultural_tags = cultural_profile.get('cultural_tags', [])

        if user_cultural_tags:
            match_count = sum(1 for tag in user_cultural_tags if tag in recipe_cultural_tags)
            if match_count > 0:
                match_ratio = match_count / len(user_cultural_tags)
                score = 0.5 + match_ratio * 0.5

        return min(score, 1.0)

    def calculate_cultural_score_with_fallback(
        self,
        recipe: Any,
        nutritional_info: Dict[str, Any],
        cultural_profile: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> float:
        """
        带完整降级策略的文化评分计算

        Args:
            recipe: Recipe对象或包含菜品信息的字典
            nutritional_info: 营养信息字典
            cultural_profile: 用户文化偏好配置
            user_id: 用户ID（用于灰度发布）

        Returns:
            float: 文化适配分数 [0, 1]
        """
        start_time = time.time()
        success = False
        used_fallback = False
        timeout = False
        error = False

        try:
            # 检查是否应该使用Stage1模型
            if user_id and not self._should_use_stage1_model(user_id):
                logger.debug(f"用户 {user_id} 不在灰度范围内，使用规则计算")
                used_fallback = True
                score = self._calculate_rule_based_cultural_score(recipe, cultural_profile)
                success = True
                return score

            # 检查缓存
            if self.cache:
                food_info = {
                    'name': recipe.get('name') if isinstance(recipe, dict) else getattr(recipe, 'name', ''),
                    'nutritional_info': nutritional_info
                }
                cached_score = self.cache.get(food_info, cultural_profile)
                if cached_score is not None:
                    logger.debug("使用缓存的文化评分")
                    success = True
                    return cached_score

            # 检查模型是否可用
            if not self.cultural_scoring_service.is_available():
                logger.warning("Stage1模型不可用，使用规则计算")
                used_fallback = True
                score = self._calculate_rule_based_cultural_score(recipe, cultural_profile)
                success = True
                return score

            # 尝试使用Stage1模型（带超时保护）
            timeout_seconds = self.config['timeout_config']['cultural_scoring_timeout']

            try:
                # 使用线程超时（如果可用）
                import threading

                result_container = {'score': None, 'exception': None}

                def predict_with_timeout():
                    try:
                        result_container['score'] = self.cultural_scoring_service.calculate_cultural_score(
                            recipe=recipe,
                            nutritional_info=nutritional_info,
                            cultural_profile=cultural_profile
                        )
                    except Exception as e:
                        result_container['exception'] = e

                thread = threading.Thread(target=predict_with_timeout)
                thread.daemon = True
                thread.start()
                thread.join(timeout=timeout_seconds)

                if thread.is_alive():
                    # 超时
                    logger.warning(f"Stage1模型预测超时（>{timeout_seconds}秒），使用规则计算")
                    timeout = True
                    used_fallback = True
                    score = self._calculate_rule_based_cultural_score(recipe, cultural_profile)
                elif result_container['exception']:
                    # 预测出错
                    raise result_container['exception']
                else:
                    # 成功
                    score = result_container['score']
                    success = True

                    # 保存到缓存
                    if self.cache:
                        food_info = {
                            'name': recipe.get('name') if isinstance(recipe, dict) else getattr(recipe, 'name', ''),
                            'nutritional_info': nutritional_info
                        }
                        self.cache.set(food_info, cultural_profile, score)

                    return score

            except (ModelLoadError, PredictionError) as e:
                logger.warning(f"Stage1模型预测失败，使用规则计算: {e}")
                used_fallback = True
                score = self._calculate_rule_based_cultural_score(recipe, cultural_profile)
                success = True
                return score

        except Exception as e:
            logger.error(f"文化评分计算出现意外错误: {e}", exc_info=True)
            error = True
            used_fallback = True
            score = self.config['scoring_weights']['fallback_cultural_score']
            success = False

        finally:
            # 记录监控指标
            if self.metrics:
                duration = time.time() - start_time
                self.metrics.record_call(
                    duration=duration,
                    success=success,
                    used_fallback=used_fallback,
                    timeout=timeout,
                    error=error
                )

        return score

    def get_metrics(self) -> Optional[Dict[str, Any]]:
        """获取监控指标"""
        if self.metrics:
            return self.metrics.get_stats()
        return None

    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """获取缓存统计信息"""
        if self.cache:
            return self.cache.get_stats()
        return None


def get_cultural_scoring_integration(config: Optional[Dict[str, Any]] = None) -> CulturalScoringIntegration:
    """获取文化评分集成服务单例"""
    if not hasattr(get_cultural_scoring_integration, '_instance'):
        get_cultural_scoring_integration._instance = CulturalScoringIntegration(config)
    return get_cultural_scoring_integration._instance
