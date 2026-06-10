#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据源管理器
管理和配置各种外部数据源
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class DataSourceManager:
    """数据源管理器 - 单一职责原则"""

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "configs" / "data_sources.yaml"

        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"✅ 数据源配置加载成功: {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'cgm_data_sources': {},
            'nutrition_apis': {},
            'quality_standards': {},
            'cache_settings': {'max_age_days': 7},
            'api_settings': {'timeout_seconds': 60, 'max_retries': 3}
        }

    def get_cgm_sources(self) -> Dict[str, Any]:
        """获取CGM数据源配置"""
        return self.config.get('cgm_data_sources', {})

    def get_nutrition_apis(self) -> Dict[str, Any]:
        """获取营养API配置"""
        return self.config.get('nutrition_apis', {})

    def get_quality_standards(self) -> Dict[str, Any]:
        """获取数据质量标准"""
        return self.config.get('quality_standards', {})

    def get_cache_settings(self) -> Dict[str, Any]:
        """获取缓存设置"""
        return self.config.get('cache_settings', {})

    def get_api_settings(self) -> Dict[str, Any]:
        """获取API设置"""
        return self.config.get('api_settings', {})

    def get_available_sources(self, source_type: str = 'all') -> List[str]:
        """获取可用的数据源列表"""
        sources = []

        if source_type in ['all', 'cgm']:
            sources.extend(self.get_cgm_sources().keys())

        if source_type in ['all', 'nutrition']:
            sources.extend(self.get_nutrition_apis().keys())

        return sources

    def get_source_info(self, source_name: str) -> Optional[Dict[str, Any]]:
        """获取特定数据源的信息"""
        # 先在CGM数据源中查找
        cgm_sources = self.get_cgm_sources()
        if source_name in cgm_sources:
            return cgm_sources[source_name]

        # 再在营养API中查找
        nutrition_apis = self.get_nutrition_apis()
        if source_name in nutrition_apis:
            return nutrition_apis[source_name]

        return None

    def is_source_available(self, source_name: str) -> bool:
        """检查数据源是否可用"""
        source_info = self.get_source_info(source_name)
        if not source_info:
            return False

        # 检查是否需要认证
        if source_info.get('requires_auth', False):
            # 这里可以添加认证检查逻辑
            logger.warning(f"数据源 {source_name} 需要认证")
            return False

        return True

    def get_recommended_sources(self, data_type: str = 'cgm') -> List[str]:
        """获取推荐的数据源"""
        if data_type == 'cgm':
            sources = self.get_cgm_sources()
            # 优先推荐不需要认证且数据量适中的源
            recommended = []
            for name, info in sources.items():
                if not info.get('requires_auth', False):
                    if info.get('size') in ['small', 'medium', 'large']:
                        recommended.append(name)
            return recommended

        elif data_type == 'nutrition':
            apis = self.get_nutrition_apis()
            # 优先推荐免费API
            recommended = []
            for name, info in apis.items():
                if not info.get('key_required', True):
                    recommended.append(name)
            return recommended

        return []

    def validate_source_config(self, source_name: str) -> Dict[str, Any]:
        """验证数据源配置"""
        source_info = self.get_source_info(source_name)
        if not source_info:
            return {'valid': False, 'error': f'数据源 {source_name} 不存在'}

        validation_result = {'valid': True, 'warnings': [], 'info': source_info}

        # 检查必需字段
        required_fields = ['url', 'description']
        for field in required_fields:
            if field not in source_info:
                validation_result['valid'] = False
                validation_result['error'] = f'缺少必需字段: {field}'
                return validation_result

        # 检查认证要求
        if source_info.get('requires_auth', False):
            validation_result['warnings'].append('此数据源需要认证')

        # 检查速率限制
        if 'rate_limit' in source_info and source_info['rate_limit']:
            validation_result['warnings'].append(f'速率限制: {source_info["rate_limit"]}')

        return validation_result

    def add_custom_source(self, source_name: str, source_config: Dict[str, Any], source_type: str = 'cgm') -> bool:
        """添加自定义数据源"""
        try:
            if source_type == 'cgm':
                self.config['cgm_data_sources'][source_name] = source_config
            elif source_type == 'nutrition':
                self.config['nutrition_apis'][source_name] = source_config
            else:
                raise ValueError(f"不支持的数据源类型: {source_type}")

            # 保存配置
            self._save_config()
            logger.info(f"✅ 自定义数据源 {source_name} 添加成功")
            return True

        except Exception as e:
            logger.error(f"添加自定义数据源失败: {e}")
            return False

    def _save_config(self):
        """保存配置到文件"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            logger.error(f"保存配置文件失败: {e}")

    def get_usage_statistics(self) -> Dict[str, Any]:
        """获取数据源使用统计"""
        stats = {
            'total_cgm_sources': len(self.get_cgm_sources()),
            'total_nutrition_apis': len(self.get_nutrition_apis()),
            'available_cgm_sources': len([s for s in self.get_cgm_sources().keys() if self.is_source_available(s)]),
            'available_nutrition_apis': len([s for s in self.get_nutrition_apis().keys() if self.is_source_available(s)]),
            'recommended_cgm': self.get_recommended_sources('cgm'),
            'recommended_nutrition': self.get_recommended_sources('nutrition')
        }
        return stats


def main():
    """主函数 - 演示数据源管理器功能"""
    import argparse

    parser = argparse.ArgumentParser(description="数据源管理器")
    parser.add_argument("--list", action="store_true", help="列出所有数据源")
    parser.add_argument("--info", type=str, help="显示特定数据源信息")
    parser.add_argument("--validate", type=str, help="验证数据源配置")
    parser.add_argument("--stats", action="store_true", help="显示使用统计")
    parser.add_argument("--recommend", choices=['cgm', 'nutrition'], help="获取推荐数据源")

    args = parser.parse_args()

    # 创建数据源管理器
    manager = DataSourceManager()

    if args.list:
        print("📊 可用的CGM数据源:")
        for source in manager.get_cgm_sources().keys():
            available = "✅" if manager.is_source_available(source) else "❌"
            print(f"  {available} {source}")

        print("\n🍎 可用的营养API:")
        for api in manager.get_nutrition_apis().keys():
            available = "✅" if manager.is_source_available(api) else "❌"
            print(f"  {available} {api}")

    elif args.info:
        info = manager.get_source_info(args.info)
        if info:
            print(f"📋 数据源信息: {args.info}")
            for key, value in info.items():
                print(f"  {key}: {value}")
        else:
            print(f"❌ 数据源 {args.info} 不存在")

    elif args.validate:
        result = manager.validate_source_config(args.validate)
        if result['valid']:
            print(f"✅ 数据源 {args.validate} 配置有效")
            if result['warnings']:
                print("⚠️ 警告:")
                for warning in result['warnings']:
                    print(f"  - {warning}")
        else:
            print(f"❌ 数据源 {args.validate} 配置无效: {result.get('error', '未知错误')}")

    elif args.stats:
        stats = manager.get_usage_statistics()
        print("📈 数据源使用统计:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    elif args.recommend:
        recommended = manager.get_recommended_sources(args.recommend)
        print(f"🎯 推荐的{args.recommend}数据源:")
        for source in recommended:
            print(f"  - {source}")

    else:
        print("请使用 --help 查看可用选项")


if __name__ == "__main__":
    main()
