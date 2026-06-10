"""
共享函数模块
消除重复代码，遵循DRY原则
"""

import os
import re
import ast
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from enum import Enum
import json
import hashlib
from sqlalchemy.orm import Session
from backend.app.core.database import SessionLocal

logger = logging.getLogger(__name__)

class Environment(Enum):
    """环境枚举"""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"

def _extract_functions(tree: ast.AST) -> List[Dict[str, Any]]:
    """提取函数定义"""
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_code = ast.unparse(node)
            functions.append({
                'name': node.name,
                'code': func_code,
                'lines': len(func_code.split('\n'))
            })
    return functions

def _extract_imports(content: str) -> List[str]:
    """提取导入语句"""
    imports = []
    import_pattern = r'import\s+([^\n]+)'
    for match in re.finditer(import_pattern, content):
        imports.append(match.group(1).strip())

    from_pattern = r'from\s+([^\s]+)\s+import\s+([^\n]+)'
    for match in re.finditer(from_pattern, content):
        module = match.group(1)
        items = match.group(2)
        imports.append(f'{module}.{items}')
    return imports

def _calculate_similarity(occurrences: List[Dict[str, Any]]) -> float:
    """计算代码相似度"""
    if len(occurrences) < 2:
        return 0.0

    codes = [occ['code'] for occ in occurrences]
    words1 = set(codes[0].split())
    words2 = set(codes[1].split())
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    return intersection / union if union > 0 else 0.0

def _get_current_environment() -> Environment:
    """获取当前环境"""
    env_name = os.getenv('APP_ENVIRONMENT', 'development').lower()
    try:
        return Environment(env_name)
    except ValueError:
        return Environment.DEVELOPMENT

def get_db():
    """获取数据库会话"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

__all__ = [
    '_extract_functions',
    '_extract_imports',
    '_calculate_similarity',
    '_get_current_environment',
    'get_db',
    'Environment'
]
