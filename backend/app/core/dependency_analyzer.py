"""
依赖分析与优化工具
分析模块依赖关系，优化导入结构
"""

import os
import sys
import json
import logging
import ast
import re
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

"""
模块依赖分析器
分析模块间的依赖关系，识别循环依赖和优化建议
"""

import ast
import json
import logging
logger = logging.getLogger(__name__)
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, deque
import networkx as nx

logger = logging.getLogger(__name__)

class DependencyAnalyzer:
    """依赖分析器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.graph = nx.DiGraph()
        self.modules: Dict[str, Dict[str, object]] = {}
        self.imports: Dict[str, List[str]] = defaultdict(list)
        self.exports: Dict[str, List[str]] = defaultdict(list)

    def analyze_file(self, file_path: Path) -> Dict[str, object]:
        """分析单个文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content)

            analysis = {
                'file': str(file_path),
                'imports': [],
                'exports': [],
                'classes': [],
                'functions': [],
                'variables': []
            }

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis['imports'].append({
                            'type': 'import',
                            'module': alias.name,
                            'alias': alias.asname
                        })

                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        analysis['imports'].append({
                            'type': 'from_import',
                            'module': module,
                            'name': alias.name,
                            'alias': alias.asname
                        })

                elif isinstance(node, ast.ClassDef):
                    analysis['classes'].append({
                        'name': node.name,
                        'bases': [base.id if hasattr(base, 'id') else str(base) for base in node.bases],
                        'line': node.lineno
                    })

                elif isinstance(node, ast.FunctionDef):
                    analysis['functions'].append({
                        'name': node.name,
                        'args': [arg.arg for arg in node.args.args],
                        'line': node.lineno
                    })

            return analysis

        except Exception as e:
            logger.error(f"分析文件失败 {file_path}: {e}")
            return {'file': str(file_path), 'error': str(e)}

    def build_dependency_graph(self, directory: str) -> None:
        """构建依赖图"""
        logger.info(f"构建依赖图: {directory}")

        for file_path in Path(directory).rglob('*.py'):
            if file_path.is_file():
                analysis = self.analyze_file(file_path)

                if 'error' in analysis:
                    continue

                module_name = self._get_module_name(file_path)
                self.modules[module_name] = analysis

                # 添加节点
                self.graph.add_node(module_name, **analysis)

                # 添加边（依赖关系）
                for imp in analysis['imports']:
                    if imp['type'] == 'from_import' and imp['module']:
                        dep_module = self._resolve_module_name(imp['module'], file_path)
                        if dep_module:
                            self.graph.add_edge(module_name, dep_module)
                            self.imports[module_name].append(dep_module)

    def _get_module_name(self, file_path: Path) -> str:
        """获取模块名"""
        relative_path = file_path.relative_to(self.project_root)
        return str(relative_path).replace('/', '.').replace('\\', '.').replace('.py', '')

    def _resolve_module_name(self, module_name: str, current_file: Path) -> Optional[str]:
        """解析模块名"""
        if module_name.startswith('.'):
            # 相对导入
            current_dir = current_file.parent
            if module_name == '.':
                return self._get_module_name(current_file)
            elif module_name.startswith('..'):
                # 上级目录
                levels = len(module_name) - len(module_name.lstrip('.'))
                for _ in range(levels):
                    current_dir = current_dir.parent
                return self._get_module_name(current_dir)
        else:
            # 绝对导入
            return module_name

    def find_circular_dependencies(self) -> List[List[str]]:
        """查找循环依赖"""
        try:
            cycles = list(nx.simple_cycles(self.graph))
            return cycles
        except Exception as e:
            logger.error(f"查找循环依赖失败: {e}")
            return []

    def find_unused_imports(self) -> Dict[str, List[str]]:
        """查找未使用的导入"""
        unused = {}

        for module_name, analysis in self.modules.items():
            imports = analysis['imports']
            exports = analysis['exports']
            classes = [cls['name'] for cls in analysis['classes']]
            functions = [func['name'] for func in analysis['functions']]

            all_used = set(exports + classes + functions)
            unused_imports = []

            for imp in imports:
                if imp['alias']:
                    if imp['alias'] not in all_used:
                        unused_imports.append(imp['alias'])
                else:
                    if imp.get('name', imp['module']) not in all_used:
                        unused_imports.append(imp.get('name', imp['module']))

            if unused_imports:
                unused[module_name] = unused_imports

        return unused

    def suggest_optimizations(self) -> Dict[str, object]:
        """建议优化方案"""
        suggestions = {
            'circular_dependencies': self.find_circular_dependencies(),
            'unused_imports': self.find_unused_imports(),
            'module_consolidation': self._suggest_module_consolidation(),
            'dependency_injection': self._suggest_dependency_injection()
        }

        return suggestions

    def _suggest_module_consolidation(self) -> List[Dict[str, object]]:
        """建议模块合并"""
        suggestions = []

        # 查找相似的小模块
        small_modules = [
            name for name, analysis in self.modules.items()
            if len(analysis['classes']) + len(analysis['functions']) < 3
        ]

        if len(small_modules) > 1:
            suggestions.append({
                'type': 'consolidation',
                'modules': small_modules,
                'reason': '多个小模块可以合并以提高内聚性',
                'action': '考虑将这些模块合并为一个更大的模块'
            })

        return suggestions

    def _suggest_dependency_injection(self) -> List[Dict[str, object]]:
        """建议依赖注入"""
        suggestions = []

        # 查找直接实例化的依赖
        for module_name, analysis in self.modules.items():
            for cls in analysis['classes']:
                for base in cls['bases']:
                    if base in self.modules:
                        suggestions.append({
                            'type': 'dependency_injection',
                            'module': module_name,
                            'class': cls['name'],
                            'dependency': base,
                            'reason': '直接继承可能导致紧耦合',
                            'action': '考虑使用依赖注入模式'
                        })

        return suggestions

    def generate_report(self) -> Dict[str, object]:
        """生成分析报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'project_root': str(self.project_root),
            'summary': {
                'total_modules': len(self.modules),
                'total_dependencies': self.graph.number_of_edges(),
                'circular_dependencies': len(self.find_circular_dependencies()),
                'unused_imports': len(self.find_unused_imports())
            },
            'modules': self.modules,
            'dependency_graph': {
                'nodes': list(self.graph.nodes()),
                'edges': list(self.graph.edges())
            },
            'optimizations': self.suggest_optimizations()
        }

        return report

    def export_graph(self, output_file: str) -> None:
        """导出依赖图"""
        try:
            # 使用GraphML格式导出
            nx.write_graphml(self.graph, output_file)
            logger.info(f"依赖图已导出: {output_file}")
        except Exception as e:
            logger.error(f"导出依赖图失败: {e}")

class ModuleOptimizer:
    """模块优化器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.analyzer = DependencyAnalyzer(project_root)

    def optimize_project(self) -> Dict[str, object]:
        """优化整个项目"""
        logger.info("开始模块优化...")

        # 构建依赖图
        backend_dir = self.project_root / "backend" / "app"
        self.analyzer.build_dependency_graph(str(backend_dir))

        # 生成报告
        report = self.analyzer.generate_report()

        # 导出依赖图
        graph_file = self.project_root / "dependency_graph.graphml"
        self.analyzer.export_graph(str(graph_file))

        # 保存报告
        report_file = self.project_root / "module_optimization_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"优化完成，报告已保存: {report_file}")
        return report

# 使用示例
if __name__ == "__main__":
    optimizer = ModuleOptimizer(".")
    results = optimizer.optimize_project()
    print(f"优化完成: {results['summary']}")

__all__ = ["'logger'", "'DependencyAnalyzer'", "'ModuleOptimizer'"]
