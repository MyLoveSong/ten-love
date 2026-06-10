"""
批量导入修复工具
系统性地修复所有导入错误
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple

class ImportFixer:
    """导入修复器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.fixes_applied = 0
        self.files_processed = 0

    def fix_all_imports(self) -> Dict[str, int]:
        """修复所有导入错误"""
        print("🔧 开始批量修复导入错误...")

        # 修复模式映射
        fix_patterns = [
            # 标准库导入修复
            (r'from app\.typing import', 'from typing import'),
            (r'from app\.datetime import', 'from datetime import'),
            (r'from app\.pathlib import', 'from pathlib import'),
            (r'from app\.enum import', 'from enum import'),
            (r'from app\.logging import', 'import logging'),
            (r'from app\.json import', 'import json'),
            (r'from app\.os import', 'import os'),
            (r'from app\.sys import', 'import sys'),
            (r'from app\.asyncio import', 'import asyncio'),
            (r'from app\.threading import', 'import threading'),
            (r'from app\.time import', 'import time'),
            (r'from app\.hashlib import', 'import hashlib'),
            (r'from app\.warnings import', 'import warnings'),
            (r'from app\.abc import', 'from abc import'),
            (r'from app\.dataclasses import', 'from dataclasses import'),
            (r'from app\.collections import', 'from collections import'),
            (r'from app\.functools import', 'from functools import'),
            (r'from app\.itertools import', 'from itertools import'),
            (r'from app\.contextlib import', 'from contextlib import'),

            # 第三方库导入修复
            (r'from app\.pydantic import', 'from pydantic import'),
            (r'from app\.fastapi import', 'from fastapi import'),
            (r'from app\.sqlalchemy import', 'from sqlalchemy import'),
            (r'from app\.numpy import', 'import numpy'),
            (r'from app\.pandas import', 'import pandas'),
            (r'from app\.sklearn import', 'from sklearn import'),
            (r'from app\.torch import', 'import torch'),
            (r'from app\.requests import', 'import requests'),
            (r'from app\.aiohttp import', 'import aiohttp'),
            (r'from app\.redis import', 'import redis'),
            (r'from app\.cryptography import', 'from cryptography import'),

            # 项目内部导入修复
            (r'from app\.core\.shared_imports_optimized import \*', ''),
            (r'from app\.core\.config import', 'from backend.app.core.config import'),
            (r'from app\.core\.database import', 'from backend.app.core.database import'),
            (r'from app\.core\.exceptions import', 'from backend.app.core.exceptions import'),
            (r'from app\.core\.cache import', 'from backend.app.core.cache import'),
            (r'from app\.core\.observability import', 'from backend.app.core.observability import'),
            (r'from app\.core\.performance import', 'from backend.app.core.performance import'),
            (r'from app\.core\.middleware import', 'from backend.app.core.middleware import'),
            (r'from app\.database import', 'from backend.app.database import'),
            (r'from app\.services import', 'from backend.app.services import'),
            (r'from app\.models import', 'from backend.app.models import'),
            (r'from app\.api import', 'from backend.app.api import'),
            (r'from app\.workflow import', 'from backend.app.workflow import'),
        ]

        # 处理所有Python文件
        for py_file in self.project_root.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue

            self.files_processed += 1
            file_fixes = self._fix_file_imports(py_file, fix_patterns)
            self.fixes_applied += file_fixes

            if file_fixes > 0:
                print(f"✅ 修复 {py_file.name}: {file_fixes} 个导入")

        print(f"🎯 批量修复完成: {self.fixes_applied} 个修复, {self.files_processed} 个文件")

        return {
            'fixes_applied': self.fixes_applied,
            'files_processed': self.files_processed
        }

    def _should_skip_file(self, file_path: Path) -> bool:
        """判断是否应该跳过文件"""
        skip_patterns = [
            '__pycache__',
            '.git',
            'node_modules',
            'venv',
            'env',
            '.pytest_cache',
            'migrations',
            'alembic'
        ]

        return any(pattern in str(file_path) for pattern in skip_patterns)

    def _fix_file_imports(self, file_path: Path, fix_patterns: List[Tuple[str, str]]) -> int:
        """修复单个文件的导入"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content
            fixes_count = 0

            for pattern, replacement in fix_patterns:
                if re.search(pattern, content):
                    content = re.sub(pattern, replacement, content)
                    fixes_count += 1

            # 清理重复的空行
            content = re.sub(r'\n\n\n+', '\n\n', content)

            # 如果有修改，写回文件
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)

            return fixes_count

        except Exception as e:
            print(f"❌ 修复文件失败 {file_path}: {e}")
            return 0

def main():
    """主函数"""
    fixer = ImportFixer(".")
    results = fixer.fix_all_imports()

    print("\n" + "="*60)
    print("🎯 批量导入修复报告")
    print("="*60)
    print(f"修复数量: {results['fixes_applied']}")
    print(f"处理文件: {results['files_processed']}")
    print("="*60)

if __name__ == "__main__":
    main()
