"""
ESM标准合规性检查器
检查项目是否符合ESM模块标准，消除重复导入和代码功能重复
"""

import os
import sys
import json
import logging
import ast
import re
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import hashlib

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from backend.app.core.shared_functions import _extract_functions, _extract_imports, _calculate_similarity
except ImportError:
    # 如果导入失败，使用本地实现
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

logger = logging.getLogger(__name__)

@dataclass
class ESMViolation:
    """ESM违规记录"""
    file_path: str
    violation_type: str
    line_number: int
    description: str
    severity: str  # 'error', 'warning', 'info'
    suggestion: str = ""

@dataclass
class DuplicateCode:
    """重复代码记录"""
    file1: str
    file2: str
    code_hash: str
    similarity: float
    lines_count: int
    suggestion: str = ""

class ESMComplianceChecker:
    """ESM合规性检查器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.violations: List[ESMViolation] = []
        self.duplicate_code: List[DuplicateCode] = []
        self.import_map: Dict[str, List[str]] = {}
        self.function_map: Dict[str, List[str]] = {}

    def check_project_compliance(self) -> Dict[str, Any]:
        """检查整个项目的ESM合规性"""
        logger.info("开始ESM合规性检查...")

        # 检查后端目录
        backend_dir = self.project_root / "backend" / "app"

        # 1. 检查导入语句
        self._check_import_statements(backend_dir)

        # 2. 检查重复导入
        self._check_duplicate_imports(backend_dir)

        # 3. 检查重复代码
        self._check_duplicate_code(backend_dir)

        # 4. 检查模块结构
        self._check_module_structure(backend_dir)

        # 5. 检查命名规范
        self._check_naming_conventions(backend_dir)

        # 生成报告
        report = self._generate_compliance_report()

        # 保存报告
        report_file = self.project_root / "esm_compliance_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"ESM合规性检查完成，报告已保存: {report_file}")
        return report

    def _check_import_statements(self, directory: Path):
        """检查导入语句"""
        logger.info("检查导入语句...")

        for py_file in directory.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                lines = content.split('\n')

                for i, line in enumerate(lines, 1):
                    line = line.strip()

                    # 检查相对导入
                    if line.startswith('from .') and not line.startswith('from ..'):
                        self.violations.append(ESMViolation(
                            file_path=str(py_file),
                            violation_type="relative_import",
                            line_number=i,
                            description=f"使用相对导入: {line}",
                            severity="warning",
                            suggestion="考虑使用绝对导入以提高可读性"
                        ))

                    # 检查通配符导入
                    if 'import *' in line:
                        self.violations.append(ESMViolation(
                            file_path=str(py_file),
                            violation_type="wildcard_import",
                            line_number=i,
                            description=f"使用通配符导入: {line}",
                            severity="error",
                            suggestion="避免使用通配符导入，明确指定导入的内容"
                        ))

                    # 检查循环导入
                    if line.startswith('import ') or line.startswith('from '):
                        self._check_circular_import(py_file, line, i)

            except Exception as e:
                logger.warning(f"检查导入语句失败 {py_file}: {e}")

    def _check_circular_import(self, file_path: Path, import_line: str, line_number: int):
        """检查循环导入"""
        # 提取导入的模块
        if import_line.startswith('from '):
            match = re.match(r'from\s+([^\s]+)\s+import', import_line)
            if match:
                imported_module = match.group(1)
                current_module = self._get_module_name(file_path)

                # 检查是否导入自己
                if imported_module == current_module:
                    self.violations.append(ESMViolation(
                        file_path=str(file_path),
                        violation_type="self_import",
                        line_number=line_number,
                        description=f"模块导入自己: {import_line}",
                        severity="error",
                        suggestion="移除自导入语句"
                    ))

    def _check_duplicate_imports(self, directory: Path):
        """检查重复导入"""
        logger.info("检查重复导入...")

        import_usage = {}

        for py_file in directory.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 提取所有导入
                imports = self._extract_imports(content)

                for imp in imports:
                    if imp not in import_usage:
                        import_usage[imp] = []
                    import_usage[imp].append(str(py_file))

            except Exception as e:
                logger.warning(f"检查重复导入失败 {py_file}: {e}")

        # 查找重复导入
        for imp, files in import_usage.items():
            if len(files) > 1:
                self.violations.append(ESMViolation(
                    file_path=files[0],
                    violation_type="duplicate_import",
                    line_number=0,
                    description=f"重复导入 '{imp}' 在 {len(files)} 个文件中",
                    severity="warning",
                    suggestion=f"考虑将 '{imp}' 移到共享模块中"
                ))

    def _check_duplicate_code(self, directory: Path):
        """检查重复代码"""
        logger.info("检查重复代码...")

        function_hashes = {}

        for py_file in directory.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 解析AST
                tree = ast.parse(content)

                # 提取函数
                functions = self._extract_functions(tree)

                for func in functions:
                    func_hash = hashlib.md5(func['code'].encode()).hexdigest()

                    if func_hash not in function_hashes:
                        function_hashes[func_hash] = []

                    function_hashes[func_hash].append({
                        'file': str(py_file),
                        'name': func['name'],
                        'lines': func['lines'],
                        'code': func['code']
                    })

            except Exception as e:
                logger.warning(f"检查重复代码失败 {py_file}: {e}")

        # 查找重复函数
        for func_hash, occurrences in function_hashes.items():
            if len(occurrences) > 1:
                # 计算相似度
                similarity = self._calculate_similarity(occurrences)

                if similarity > 0.8:  # 80%以上相似度
                    self.duplicate_code.append(DuplicateCode(
                        file1=occurrences[0]['file'],
                        file2=occurrences[1]['file'],
                        code_hash=func_hash,
                        similarity=similarity,
                        lines_count=occurrences[0]['lines'],
                        suggestion=f"函数 '{occurrences[0]['name']}' 在多个文件中重复，考虑提取到共享模块"
                    ))

    def _check_module_structure(self, directory: Path):
        """检查模块结构"""
        logger.info("检查模块结构...")

        for py_file in directory.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 检查是否有__all__定义
                if not re.search(r'__all__\s*=', content):
                    self.violations.append(ESMViolation(
                        file_path=str(py_file),
                        violation_type="missing_all",
                        line_number=1,
                        description="模块缺少__all__定义",
                        severity="info",
                        suggestion="添加__all__列表明确导出内容"
                    ))

                # 检查模块文档字符串
                if not content.strip().startswith('"""') and not content.strip().startswith("'''"):
                    self.violations.append(ESMViolation(
                        file_path=str(py_file),
                        violation_type="missing_docstring",
                        line_number=1,
                        description="模块缺少文档字符串",
                        severity="warning",
                        suggestion="添加模块文档字符串"
                    ))

            except Exception as e:
                logger.warning(f"检查模块结构失败 {py_file}: {e}")

    def _check_naming_conventions(self, directory: Path):
        """检查命名规范"""
        logger.info("检查命名规范...")

        for py_file in directory.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                tree = ast.parse(content)

                for node in ast.walk(tree):
                    # 检查类名（应该使用PascalCase）
                    if isinstance(node, ast.ClassDef):
                        if not re.match(r'^[A-Z][a-zA-Z0-9]*$', node.name):
                            self.violations.append(ESMViolation(
                                file_path=str(py_file),
                                violation_type="class_naming",
                                line_number=node.lineno,
                                description=f"类名不符合PascalCase规范: {node.name}",
                                severity="warning",
                                suggestion="类名应该使用PascalCase（如：MyClass）"
                            ))

                    # 检查函数名（应该使用snake_case）
                    elif isinstance(node, ast.FunctionDef):
                        if not re.match(r'^[a-z][a-z0-9_]*$', node.name):
                            self.violations.append(ESMViolation(
                                file_path=str(py_file),
                                violation_type="function_naming",
                                line_number=node.lineno,
                                description=f"函数名不符合snake_case规范: {node.name}",
                                severity="warning",
                                suggestion="函数名应该使用snake_case（如：my_function）"
                            ))

            except Exception as e:
                logger.warning(f"检查命名规范失败 {py_file}: {e}")

    def _get_module_name(self, file_path: Path) -> str:
        """获取模块名"""
        relative_path = file_path.relative_to(self.project_root)
        return str(relative_path).replace('/', '.').replace('\\', '.').replace('.py', '')

    def _generate_compliance_report(self) -> Dict[str, Any]:
        """生成合规性报告"""
        # 按严重程度分组
        errors = [v for v in self.violations if v.severity == 'error']
        warnings = [v for v in self.violations if v.severity == 'warning']
        infos = [v for v in self.violations if v.severity == 'info']

        # 按类型分组
        violation_types = {}
        for violation in self.violations:
            if violation.violation_type not in violation_types:
                violation_types[violation.violation_type] = []
            violation_types[violation.violation_type].append(violation)

        return {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_violations': len(self.violations),
                'errors': len(errors),
                'warnings': len(warnings),
                'infos': len(infos),
                'duplicate_code_blocks': len(self.duplicate_code)
            },
            'violations_by_severity': {
                'errors': [
                    {
                        'file': v.file_path,
                        'line': v.line_number,
                        'description': v.description,
                        'suggestion': v.suggestion
                    }
                    for v in errors
                ],
                'warnings': [
                    {
                        'file': v.file_path,
                        'line': v.line_number,
                        'description': v.description,
                        'suggestion': v.suggestion
                    }
                    for v in warnings
                ],
                'infos': [
                    {
                        'file': v.file_path,
                        'line': v.line_number,
                        'description': v.description,
                        'suggestion': v.suggestion
                    }
                    for v in infos
                ]
            },
            'violations_by_type': {
                violation_type: [
                    {
                        'file': v.file_path,
                        'line': v.line_number,
                        'description': v.description,
                        'severity': v.severity,
                        'suggestion': v.suggestion
                    }
                    for v in violations
                ]
                for violation_type, violations in violation_types.items()
            },
            'duplicate_code': [
                {
                    'file1': dup.file1,
                    'file2': dup.file2,
                    'similarity': dup.similarity,
                    'lines_count': dup.lines_count,
                    'suggestion': dup.suggestion
                }
                for dup in self.duplicate_code
            ],
            'recommendations': self._generate_recommendations()
        }

    def _generate_recommendations(self) -> List[str]:
        """生成改进建议"""
        recommendations = []

        if len(self.violations) > 0:
            recommendations.append(f"发现 {len(self.violations)} 个ESM合规性问题，建议优先修复错误级别的问题")

        if len(self.duplicate_code) > 0:
            recommendations.append(f"发现 {len(self.duplicate_code)} 个重复代码块，建议提取到共享模块")

        # 基于违规类型生成具体建议
        violation_types = set(v.violation_type for v in self.violations)

        if 'wildcard_import' in violation_types:
            recommendations.append("避免使用通配符导入（import *），明确指定导入的内容")

        if 'duplicate_import' in violation_types:
            recommendations.append("创建共享导入模块，消除重复导入")

        if 'missing_all' in violation_types:
            recommendations.append("为所有模块添加__all__定义，明确导出内容")

        if 'missing_docstring' in violation_types:
            recommendations.append("为所有模块添加文档字符串，提高代码可读性")

        return recommendations

def main():
    """主函数"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    checker = ESMComplianceChecker(".")
    results = checker.check_project_compliance()

    print("\n" + "="*60)
    print("🎯 ESM标准合规性检查完成报告")
    print("="*60)

    summary = results['summary']
    print(f"总违规数: {summary['total_violations']}")
    print(f"错误: {summary['errors']}")
    print(f"警告: {summary['warnings']}")
    print(f"信息: {summary['infos']}")
    print(f"重复代码块: {summary['duplicate_code_blocks']}")

    print("\n改进建议:")
    for rec in results['recommendations']:
        print(f"• {rec}")

    print("="*60)

if __name__ == "__main__":
    main()
