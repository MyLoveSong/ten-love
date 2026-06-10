

"""
MCP增强模块注册器
充分利用MCP架构，实现智能模块发现和自动注册
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
import importlib
import inspect
import ast
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ModuleInfo:
    """模块信息"""
    name: str
    path: str
    version: str = "1.0.0"
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    size: int = 0
    complexity: int = 0
    last_modified: Optional[datetime] = None

class MCPModuleDiscovery:
    """MCP模块发现器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.discovered_modules: Dict[str, ModuleInfo] = {}
        self.module_graph: Dict[str, Set[str]] = {}

    def discover_modules(self, directory: str = None) -> Dict[str, ModuleInfo]:
        """发现项目中的所有模块"""
        if directory is None:
            directory = str(self.project_root / "backend" / "app")

        logger.info(f"开始发现模块: {directory}")

        for py_file in Path(directory).rglob("*.py"):
            if py_file.name == "__init__.py":
                continue

            try:
                module_info = self._analyze_module(py_file)
                if module_info:
                    self.discovered_modules[module_info.name] = module_info
                    logger.info(f"发现模块: {module_info.name}")
            except Exception as e:
                logger.warning(f"分析模块失败 {py_file}: {e}")

        # 构建模块依赖图
        self._build_dependency_graph()

        return self.discovered_modules

    def _analyze_module(self, file_path: Path) -> Optional[ModuleInfo]:
        """分析单个模块文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 解析AST
            tree = ast.parse(content)

            # 获取模块名
            module_name = self._get_module_name(file_path)

            # 分析导入
            imports = self._extract_imports(tree)

            # 分析导出
            exports = self._extract_exports(tree)

            # 分析依赖
            dependencies = self._extract_dependencies(imports)

            # 计算复杂度
            complexity = self._calculate_complexity(tree)

            # 获取文件信息
            stat = file_path.stat()

            return ModuleInfo(
                name=module_name,
                path=str(file_path),
                description=self._extract_docstring(tree),
                dependencies=dependencies,
                exports=exports,
                imports=imports,
                tags=self._extract_tags(tree),
                size=stat.st_size,
                complexity=complexity,
                last_modified=datetime.fromtimestamp(stat.st_mtime)
            )

        except Exception as e:
            logger.error(f"分析模块失败 {file_path}: {e}")
            return None

    def _get_module_name(self, file_path: Path) -> str:
        """获取模块名"""
        relative_path = file_path.relative_to(self.project_root)
        return str(relative_path).replace('/', '.').replace('\\', '.').replace('.py', '')

    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """提取导入语句"""
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")

        return imports

    def _extract_exports(self, tree: ast.AST) -> List[str]:
        """提取导出内容"""
        exports = []

        # 查找__all__定义
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == '__all__':
                        if isinstance(node.value, ast.List):
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Constant):
                                    exports.append(elt.value)

        # 如果没有__all__，查找顶级函数和类
        if not exports:
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    exports.append(node.name)

        return exports

    def _extract_dependencies(self, imports: List[str]) -> List[str]:
        """提取项目内部依赖"""
        dependencies = []

        for imp in imports:
            # 检查是否是项目内部模块
            if imp.startswith('backend.app.') or imp.startswith('app.'):
                # 提取模块名
                parts = imp.split('.')
                if len(parts) >= 3:
                    module_name = '.'.join(parts[:3])  # backend.app.module
                    if module_name not in dependencies:
                        dependencies.append(module_name)

        return dependencies

    def _extract_docstring(self, tree: ast.AST) -> str:
        """提取模块文档字符串"""
        if tree.body and isinstance(tree.body[0], ast.Expr):
            if isinstance(tree.body[0].value, ast.Constant):
                return tree.body[0].value.value
        return ""

    def _extract_tags(self, tree: ast.AST) -> List[str]:
        """提取模块标签"""
        tags = []

        # 基于文件名和内容推断标签
        docstring = self._extract_docstring(tree)

        if 'experiment' in docstring.lower():
            tags.append('experiments')
        if 'data' in docstring.lower():
            tags.append('data')
        if 'model' in docstring.lower():
            tags.append('modeling')
        if 'api' in docstring.lower():
            tags.append('api')
        if 'core' in docstring.lower():
            tags.append('core')

        return tags

    def _calculate_complexity(self, tree: ast.AST) -> int:
        """计算模块复杂度"""
        complexity = 0

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                complexity += 1
            elif isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
                complexity += 1

        return complexity

    def _build_dependency_graph(self):
        """构建模块依赖图"""
        self.module_graph = {}

        for module_name, module_info in self.discovered_modules.items():
            self.module_graph[module_name] = set(module_info.dependencies)

    def find_circular_dependencies(self) -> List[List[str]]:
        """查找循环依赖"""
        visited = set()
        rec_stack = set()
        cycles = []

        def dfs(node, path):
            if node in rec_stack:
                # 找到循环
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:] + [node])
                return

            if node in visited:
                return

            visited.add(node)
            rec_stack.add(node)

            for neighbor in self.module_graph.get(node, set()):
                dfs(neighbor, path + [node])

            rec_stack.remove(node)

        for module in self.discovered_modules:
            if module not in visited:
                dfs(module, [])

        return cycles

    def suggest_module_consolidation(self) -> List[Dict[str, Any]]:
        """建议模块合并"""
        suggestions = []

        # 查找小模块
        small_modules = [
            name for name, info in self.discovered_modules.items()
            if info.complexity < 3 and info.size < 1000
        ]

        if len(small_modules) > 1:
            suggestions.append({
                'type': 'consolidation',
                'modules': small_modules,
                'reason': '多个小模块可以合并以提高内聚性',
                'action': '考虑将这些模块合并为一个更大的模块'
            })

        return suggestions

    def generate_discovery_report(self) -> Dict[str, Any]:
        """生成发现报告"""
        return {
            'timestamp': datetime.now().isoformat(),
            'total_modules': len(self.discovered_modules),
            'modules': {
                name: {
                    'path': info.path,
                    'version': info.version,
                    'description': info.description,
                    'dependencies': info.dependencies,
                    'exports': info.exports,
                    'tags': info.tags,
                    'size': info.size,
                    'complexity': info.complexity,
                    'last_modified': info.last_modified.isoformat() if info.last_modified else None
                }
                for name, info in self.discovered_modules.items()
            },
            'circular_dependencies': self.find_circular_dependencies(),
            'consolidation_suggestions': self.suggest_module_consolidation()
        }

class MCPAutoRegistrar:
    """MCP自动注册器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.discovery = MCPModuleDiscovery(project_root)
        self.registry_file = self.project_root / "mcp_auto_registry.json"

    def auto_register_all_modules(self) -> Dict[str, Any]:
        """自动注册所有发现的模块"""
        logger.info("开始自动注册模块...")

        # 发现模块
        modules = self.discovery.discover_modules()

        # 生成注册报告
        report = self.discovery.generate_discovery_report()

        # 保存报告
        with open(self.registry_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"自动注册完成，报告已保存: {self.registry_file}")

        return {
            'status': 'success',
            'modules_discovered': len(modules),
            'report_file': str(self.registry_file),
            'summary': {
                'total_modules': report['total_modules'],
                'circular_dependencies': len(report['circular_dependencies']),
                'consolidation_suggestions': len(report['consolidation_suggestions'])
            }
        }

def main():
    """主函数"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    registrar = MCPAutoRegistrar(".")
    results = registrar.auto_register_all_modules()

    print("\n" + "="*60)
    print("🎯 MCP自动模块发现与注册完成报告")
    print("="*60)
    print(f"状态: {results['status']}")
    print(f"发现模块: {results['modules_discovered']}")
    print(f"报告文件: {results['report_file']}")

    summary = results['summary']
    print(f"总模块数: {summary['total_modules']}")
    print(f"循环依赖: {summary['circular_dependencies']}")
    print(f"合并建议: {summary['consolidation_suggestions']}")

    print("="*60)

if __name__ == "__main__":
    main()
