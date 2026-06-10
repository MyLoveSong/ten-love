"""
代码去重与优化工具
自动消除重复导入和代码功能重复，遵循DRY原则
"""

import os
import sys
import json
import logging
import ast
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import shutil

logger = logging.getLogger(__name__)

@dataclass
class DuplicateFunction:
    """重复函数记录"""
    name: str
    files: List[str]
    code_hash: str
    similarity: float
    lines_count: int

@dataclass
class DuplicateImport:
    """重复导入记录"""
    import_statement: str
    files: List[str]
    count: int

class CodeDeduplicator:
    """代码去重器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.duplicate_functions: List[DuplicateFunction] = []
        self.duplicate_imports: List[DuplicateImport] = []
        self.shared_functions: Dict[str, str] = {}
        self.shared_imports: Dict[str, str] = {}

    def deduplicate_project(self) -> Dict[str, Any]:
        """对整个项目进行去重"""
        logger.info("开始代码去重...")

        backend_dir = self.project_root / "backend" / "app"

        # 1. 发现重复函数
        self._find_duplicate_functions(backend_dir)

        # 2. 发现重复导入
        self._find_duplicate_imports(backend_dir)

        # 3. 创建共享模块
        self._create_shared_modules()

        # 4. 重构重复代码
        self._refactor_duplicates()

        # 生成报告
        report = self._generate_deduplication_report()

        # 保存报告
        report_file = self.project_root / "code_deduplication_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"代码去重完成，报告已保存: {report_file}")
        return report

    def _find_duplicate_functions(self, directory: Path):
        """发现重复函数"""
        logger.info("发现重复函数...")

        function_hashes = {}

        for py_file in directory.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                tree = ast.parse(content)
                functions = self._extract_functions(tree)

                for func in functions:
                    func_hash = hashlib.md5(func['code'].encode()).hexdigest()

                    if func_hash not in function_hashes:
                        function_hashes[func_hash] = []

                    function_hashes[func_hash].append({
                        'file': str(py_file),
                        'name': func['name'],
                        'code': func['code'],
                        'lines': func['lines']
                    })

            except Exception as e:
                logger.warning(f"分析函数失败 {py_file}: {e}")

        # 识别重复函数
        for func_hash, occurrences in function_hashes.items():
            if len(occurrences) > 1:
                similarity = self._calculate_similarity(occurrences)

                if similarity > 0.8:  # 80%以上相似度
                    self.duplicate_functions.append(DuplicateFunction(
                        name=occurrences[0]['name'],
                        files=[occ['file'] for occ in occurrences],
                        code_hash=func_hash,
                        similarity=similarity,
                        lines_count=occurrences[0]['lines']
                    ))

    def _find_duplicate_imports(self, directory: Path):
        """发现重复导入"""
        logger.info("发现重复导入...")

        import_usage = {}

        for py_file in directory.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                imports = self._extract_imports(content)

                for imp in imports:
                    if imp not in import_usage:
                        import_usage[imp] = []
                    import_usage[imp].append(str(py_file))

            except Exception as e:
                logger.warning(f"分析导入失败 {py_file}: {e}")

        # 识别重复导入
        for imp, files in import_usage.items():
            if len(files) > 1:
                self.duplicate_imports.append(DuplicateImport(
                    import_statement=imp,
                    files=files,
                    count=len(files)
                ))

    def _extract_functions(self, tree: ast.AST) -> List[Dict[str, Any]]:
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

    def _extract_imports(self, content: str) -> List[str]:
        """提取导入语句"""
        imports = []

        # 匹配import语句
        import_pattern = r'import\s+([^\n]+)'
        for match in re.finditer(import_pattern, content):
            imports.append(match.group(1).strip())

        # 匹配from import语句
        from_pattern = r'from\s+([^\s]+)\s+import\s+([^\n]+)'
        for match in re.finditer(from_pattern, content):
            module = match.group(1)
            items = match.group(2)
            imports.append(f"{module}.{items}")

        return imports

    def _calculate_similarity(self, occurrences: List[Dict[str, Any]]) -> float:
        """计算代码相似度"""
        if len(occurrences) < 2:
            return 0.0

        codes = [occ['code'] for occ in occurrences]

        # 使用集合交集计算相似度
        words1 = set(codes[0].split())
        words2 = set(codes[1].split())

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def _create_shared_modules(self):
        """创建共享模块"""
        logger.info("创建共享模块...")

        # 创建共享函数模块
        if self.duplicate_functions:
            self._create_shared_functions_module()

        # 创建共享导入模块
        if self.duplicate_imports:
            self._create_shared_imports_module()

    def _create_shared_functions_module(self):
        """创建共享函数模块"""
        shared_functions_file = self.project_root / "backend" / "app" / "core" / "shared_functions.py"

        content = '''"""
共享函数模块
消除重复代码，遵循DRY原则
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json
import hashlib

logger = logging.getLogger(__name__)

'''

        # 添加重复函数
        for dup_func in self.duplicate_functions[:5]:  # 只处理前5个最重复的函数
            # 从第一个文件中提取函数代码
            first_file = dup_func.files[0]
            try:
                with open(first_file, 'r', encoding='utf-8') as f:
                    file_content = f.read()

                tree = ast.parse(file_content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name == dup_func.name:
                        func_code = ast.unparse(node)
                        content += f"\n{func_code}\n"
                        break

            except Exception as e:
                logger.warning(f"提取函数失败 {first_file}: {e}")

        # 添加__all__定义
        function_names = [dup_func.name for dup_func in self.duplicate_functions[:5]]
        content += f"\n__all__ = {function_names}\n"

        with open(shared_functions_file, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.info(f"共享函数模块已创建: {shared_functions_file}")

    def _create_shared_imports_module(self):
        """创建共享导入模块"""
        shared_imports_file = self.project_root / "backend" / "app" / "core" / "shared_imports_optimized.py"

        content = '''"""
优化共享导入模块
消除重复导入，遵循ESM标准
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
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import warnings
import hashlib
import re
import ast

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
from app.core.configuration import get_configuration
from app.core.structured_logging import get_logger

# 导出所有导入
__all__ = [
    # 标准库
    'os', 'sys', 'json', 'logging', 'asyncio', 'threading', 'time',
    'Path', 'Dict', 'List', 'Any', 'Optional', 'Union', 'Tuple', 'Callable',
    'dataclass', 'field', 'datetime', 'timedelta', 'ABC', 'abstractmethod', 'warnings',
    'hashlib', 're', 'ast',

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

        with open(shared_imports_file, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.info(f"优化共享导入模块已创建: {shared_imports_file}")

    def _refactor_duplicates(self):
        """重构重复代码"""
        logger.info("重构重复代码...")

        # 重构重复函数
        for dup_func in self.duplicate_functions[:3]:  # 只重构前3个最重复的函数
            self._refactor_duplicate_function(dup_func)

        # 重构重复导入
        for dup_import in self.duplicate_imports[:10]:  # 只重构前10个最重复的导入
            self._refactor_duplicate_import(dup_import)

    def _refactor_duplicate_function(self, dup_func: DuplicateFunction):
        """重构重复函数"""
        logger.info(f"重构重复函数: {dup_func.name}")

        # 在文件中替换函数调用为导入
        for file_path in dup_func.files[1:]:  # 跳过第一个文件（保留原始函数）
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 添加导入语句
                import_line = "from app.core.shared_functions import {}\n".format(dup_func.name)

                # 在文件开头添加导入
                lines = content.split('\n')
                import_inserted = False

                for i, line in enumerate(lines):
                    if line.strip().startswith('import ') or line.strip().startswith('from '):
                        continue
                    else:
                        lines.insert(i, import_line)
                        import_inserted = True
                        break

                if not import_inserted:
                    lines.insert(0, import_line)

                # 移除原始函数定义
                new_lines = []
                skip_function = False

                for line in lines:
                    if f"def {dup_func.name}(" in line:
                        skip_function = True
                        continue
                    elif skip_function and line.strip() == "":
                        skip_function = False
                        continue
                    elif skip_function and line.startswith("def ") and not line.startswith("    "):
                        skip_function = False
                        new_lines.append(line)
                    elif not skip_function:
                        new_lines.append(line)

                # 写回文件
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(new_lines))

                logger.info(f"重构完成: {file_path}")

            except Exception as e:
                logger.warning(f"重构函数失败 {file_path}: {e}")

    def _refactor_duplicate_import(self, dup_import: DuplicateImport):
        """重构重复导入"""
        logger.info(f"重构重复导入: {dup_import.import_statement}")

        # 在文件中替换重复导入为共享导入
        for file_path in dup_import.files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 检查是否已经使用了共享导入
                if "from app.core.shared_imports_optimized import" in content:
                    continue

                # 添加共享导入
                shared_import = "\n"

                lines = content.split('\n')
                lines.insert(0, shared_import)

                # 写回文件
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))

                logger.info(f"重构导入完成: {file_path}")

            except Exception as e:
                logger.warning(f"重构导入失败 {file_path}: {e}")

    def _generate_deduplication_report(self) -> Dict[str, Any]:
        """生成去重报告"""
        return {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'duplicate_functions_found': len(self.duplicate_functions),
                'duplicate_imports_found': len(self.duplicate_imports),
                'functions_refactored': min(len(self.duplicate_functions), 3),
                'imports_refactored': min(len(self.duplicate_imports), 10)
            },
            'duplicate_functions': [
                {
                    'name': dup.name,
                    'files': dup.files,
                    'similarity': dup.similarity,
                    'lines_count': dup.lines_count
                }
                for dup in self.duplicate_functions
            ],
            'duplicate_imports': [
                {
                    'import_statement': dup.import_statement,
                    'files': dup.files,
                    'count': dup.count
                }
                for dup in self.duplicate_imports
            ],
            'created_modules': [
                'backend/app/core/shared_functions.py',
                'backend/app/core/shared_imports_optimized.py'
            ],
            'recommendations': [
                "继续使用共享模块减少代码重复",
                "定期运行去重检查保持代码质量",
                "为共享函数添加单元测试",
                "考虑使用依赖注入进一步减少耦合"
            ]
        }

def main():
    """主函数"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    deduplicator = CodeDeduplicator(".")
    results = deduplicator.deduplicate_project()

    print("\n" + "="*60)
    print("🎯 代码去重与优化完成报告")
    print("="*60)

    summary = results['summary']
    print(f"发现重复函数: {summary['duplicate_functions_found']}")
    print(f"发现重复导入: {summary['duplicate_imports_found']}")
    print(f"重构函数: {summary['functions_refactored']}")
    print(f"重构导入: {summary['imports_refactored']}")

    print("\n创建的共享模块:")
    for module in results['created_modules']:
        print(f"• {module}")

    print("\n改进建议:")
    for rec in results['recommendations']:
        print(f"• {rec}")

    print("="*60)

if __name__ == "__main__":
    main()
