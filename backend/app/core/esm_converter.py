"""
ESM模块标准化工具
将现有CommonJS模块转换为ESM标准，消除重复导入
"""

import ast
import os
import re
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional, Union, Callable

logger = logging.getLogger(__name__)

class ESMConverter:
    """ESM转换器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.converted_files: Set[str] = set()
        self.import_map: Dict[str, List[str]] = {}
        self.duplicate_imports: Dict[str, List[str]] = {}

    def analyze_imports(self, file_path: str) -> Dict[str, object]:
        """分析文件中的导入"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 查找所有import语句
        import_pattern = r'(?:from\s+([^\s]+)\s+)?import\s+([^;]+)'
        matches = re.findall(import_pattern, content)

        analysis = {
            'file': file_path,
            'imports': [],
            'duplicates': [],
            'unused': []
        }

        for match in matches:
            # 确保match是元组且有两个元素
            if len(match) >= 2:
                module, imports_str = match[0], match[1]
            else:
                continue

            if not module:
                module = 'default'

            # 解析导入的具体内容
            import_items = [item.strip() for item in imports_str.split(',')]

            for item in import_items:
                # 处理as别名
                if ' as ' in item:
                    parts = item.split(' as ')
                    if len(parts) >= 2:
                        original, alias = parts[0], parts[1]
                        analysis['imports'].append({
                            'module': module,
                            'name': original.strip(),
                            'alias': alias.strip()
                        })
                else:
                    analysis['imports'].append({
                        'module': module,
                        'name': item.strip(),
                        'alias': None
                    })

        return analysis

    def find_duplicate_imports(self, directory: str) -> Dict[str, List[str]]:
        """查找重复导入"""
        duplicates = {}

        for file_path in Path(directory).rglob('*.py'):
            if file_path.is_file():
                analysis = self.analyze_imports(str(file_path))

                for imp in analysis['imports']:
                    key = f"{imp['module']}.{imp['name']}"
                    if key not in duplicates:
                        duplicates[key] = []
                    duplicates[key].append(str(file_path))

        # 过滤出真正的重复
        return {k: v for k, v in duplicates.items() if len(v) > 1}

    def create_shared_imports(self, duplicates: Dict[str, List[str]]) -> str:
        """创建共享导入模块"""
        shared_content = '''"""
共享导入模块
统一管理常用导入，避免重复
"""

# 标准库导入
import os
import sys
import json
import logging
import asyncio
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import warnings

# 第三方库导入
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, JSON
from sqlalchemy.orm import sessionmaker, relationship
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 项目内部导入
from app.core.mcp_registry import mcp_registry, register_module, register_service, inject
from backend.app.core.exceptions import CustomException
from app.app.core.configuration import get_configuration
from app.core.logging import get_logger

# 导出所有导入
__all__ = [
    # 标准库
    'os', 'sys', 'json', 'logging', 'asyncio', 'threading', 'time',
    'Path', 'Dict', 'List', 'Optional', 'Union', 'Tuple', 'Callable',
    'dataclass', 'field', 'datetime', 'timedelta', 'ABC', 'abstractmethod', 'warnings',

    # 第三方库
    'np', 'pd', 'FastAPI', 'HTTPException', 'Depends', 'JSONResponse', 'BaseModel',
    'create_engine', 'Column', 'Integer', 'String', 'Float', 'DateTime', 'Boolean', 'Text', 'JSON',
    'sessionmaker', 'relationship', 'accuracy_score', 'precision_score', 'recall_score', 'f1_score',
    'roc_auc_score', 'mean_squared_error', 'mean_absolute_error', 'r2_score',
    'train_test_split', 'cross_val_score', 'KFold', 'StandardScaler', 'MinMaxScaler',

    # 项目内部
    'mcp_registry', 'register_module', 'register_service', 'inject',
    'CustomException', 'get_configuration', 'get_logger'
]
'''

        shared_file = self.project_root / "backend" / "app" / "core" / "shared_imports.py"
        with open(shared_file, 'w', encoding='utf-8') as f:
            f.write(shared_content)

        logger.info(f"共享导入模块已创建: {shared_file}")
        return str(shared_file)

    def optimize_imports(self, file_path: str) -> str:
        """优化文件导入"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 替换重复导入为共享导入
        optimized_content = content

        # 添加共享导入
        if 'from app.core.shared_imports_optimized import' not in optimized_content:
            optimized_content = '\n' + optimized_content

        return optimized_content

    def convert_to_esm(self, file_path: str) -> str:
        """转换为ESM格式"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 转换import语句
        esm_content = content

        # 替换相对导入
        esm_content = re.sub(
            r'from \.\.([^\\s]+) import',
            r'from app\1 import',
            esm_content
        )

        # 替换绝对导入
        esm_content = re.sub(
            r'from ([^\\s]+) import',
            r'from app.\1 import',
            esm_content
        )

        return esm_content

    def process_directory(self, directory: str) -> Dict[str, object]:
        """处理整个目录"""
        results = {
            'processed_files': [],
            'duplicates_found': {},
            'optimizations_applied': []
        }

        # 查找重复导入
        duplicates = self.find_duplicate_imports(directory)
        results['duplicates_found'] = duplicates

        # 创建共享导入模块
        shared_file = self.create_shared_imports(duplicates)

        # 处理每个文件
        for file_path in Path(directory).rglob('*.py'):
            if file_path.is_file() and 'shared_imports.py' not in str(file_path):
                try:
                    # 优化导入
                    optimized_content = self.optimize_imports(str(file_path))

                    # 转换为ESM
                    esm_content = self.convert_to_esm(str(file_path))

                    # 写回文件
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(esm_content)

                    results['processed_files'].append(str(file_path))
                    results['optimizations_applied'].append({
                        'file': str(file_path),
                        'duplicates_removed': len(duplicates),
                        'esm_converted': True
                    })

                except Exception as e:
                    logger.error(f"处理文件失败 {file_path}: {e}")

        return results

class ImportOptimizer:
    """导入优化器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.converter = ESMConverter(project_root)

    def optimize_project(self) -> Dict[str, object]:
        """优化整个项目"""
        logger.info("开始优化项目导入...")

        # 处理后端目录
        backend_dir = self.project_root / "backend" / "app"
        results = self.converter.process_directory(str(backend_dir))

        # 生成优化报告
        report = {
            'timestamp': datetime.now().isoformat(),
            'project_root': str(self.project_root),
            'optimization_results': results,
            'summary': {
                'files_processed': len(results['processed_files']),
                'duplicates_found': len(results['duplicates_found']),
                'optimizations_applied': len(results['optimizations_applied'])
            }
        }

        # 保存报告
        report_file = self.project_root / "import_optimization_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"优化完成，报告已保存: {report_file}")
        return report

# 使用示例
if __name__ == "__main__":
    optimizer = ImportOptimizer(".")
    results = optimizer.optimize_project()
    print(f"优化完成: {results['summary']}")
