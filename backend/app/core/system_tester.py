"""
全方位系统测试脚本
确保项目能够正常运行
"""

import os
import sys
import json
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import subprocess
import importlib

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

class SystemTester:
    """系统测试器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.test_results = {}
        self.failed_tests = []
        self.passed_tests = []

    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        logger.info("开始全方位系统测试...")

        # 1. 语法检查
        self._test_syntax()

        # 2. 导入测试
        self._test_imports()

        # 3. 核心模块测试
        self._test_core_modules()

        # 4. 数据库测试
        self._test_database()

        # 5. API测试
        self._test_api()

        # 6. 优化工具测试
        self._test_optimization_tools()

        # 生成测试报告
        report = self._generate_test_report()

        # 保存报告
        report_file = self.project_root / "system_test_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"系统测试完成，报告已保存: {report_file}")
        return report

    def _test_syntax(self):
        """测试语法"""
        logger.info("测试Python语法...")

        backend_dir = self.project_root / "backend" / "app"
        syntax_errors = []

        for py_file in backend_dir.rglob("*.py"):
            try:
                # 编译检查
                compile(open(py_file, 'r', encoding='utf-8').read(), str(py_file), 'exec')
                self.passed_tests.append(f"语法检查: {py_file.name}")
            except SyntaxError as e:
                syntax_errors.append({
                    'file': str(py_file),
                    'error': str(e),
                    'line': e.lineno
                })
                self.failed_tests.append(f"语法错误: {py_file.name} - {e}")
            except Exception as e:
                self.failed_tests.append(f"语法检查失败: {py_file.name} - {e}")

        self.test_results['syntax'] = {
            'passed': len(self.passed_tests),
            'failed': len(syntax_errors),
            'errors': syntax_errors
        }

        logger.info(f"语法测试完成: {len(self.passed_tests)} 通过, {len(syntax_errors)} 失败")

    def _test_imports(self):
        """测试导入"""
        logger.info("测试模块导入...")

        # 测试核心模块导入
        core_modules = [
            'backend.app.core.mcp_registry',
            'backend.app.core.esm_converter',
            'backend.app.core.dependency_analyzer',
            'backend.app.core.project_optimizer',
            'backend.app.core.code_quality_monitor'
        ]

        import_errors = []

        for module_name in core_modules:
            try:
                importlib.import_module(module_name)
                self.passed_tests.append(f"导入测试: {module_name}")
            except Exception as e:
                import_errors.append({
                    'module': module_name,
                    'error': str(e)
                })
                self.failed_tests.append(f"导入失败: {module_name} - {e}")

        self.test_results['imports'] = {
            'passed': len([t for t in self.passed_tests if '导入测试' in t]),
            'failed': len(import_errors),
            'errors': import_errors
        }

        logger.info(f"导入测试完成: {len(core_modules) - len(import_errors)} 通过, {len(import_errors)} 失败")

    def _test_core_modules(self):
        """测试核心模块功能"""
        logger.info("测试核心模块功能...")

        core_tests = []

        # 测试MCP注册中心
        try:
            from backend.app.core.mcp_registry import mcp_registry
            registry = mcp_registry
            core_tests.append("MCP注册中心: 正常")
            self.passed_tests.append("核心模块: MCP注册中心")
        except Exception as e:
            core_tests.append(f"MCP注册中心: 失败 - {e}")
            self.failed_tests.append(f"核心模块: MCP注册中心 - {e}")

        # 测试ESM转换器
        try:
            from backend.app.core.esm_converter import ESMConverter
            converter = ESMConverter(str(self.project_root))
            core_tests.append("ESM转换器: 正常")
            self.passed_tests.append("核心模块: ESM转换器")
        except Exception as e:
            core_tests.append(f"ESM转换器: 失败 - {e}")
            self.failed_tests.append(f"核心模块: ESM转换器 - {e}")

        # 测试依赖分析器
        try:
            from backend.app.core.dependency_analyzer import DependencyAnalyzer
            analyzer = DependencyAnalyzer(str(self.project_root))
            core_tests.append("依赖分析器: 正常")
            self.passed_tests.append("核心模块: 依赖分析器")
        except Exception as e:
            core_tests.append(f"依赖分析器: 失败 - {e}")
            self.failed_tests.append(f"核心模块: 依赖分析器 - {e}")

        self.test_results['core_modules'] = {
            'tests': core_tests,
            'passed': len([t for t in self.passed_tests if '核心模块' in t]),
            'failed': len([t for t in self.failed_tests if '核心模块' in t])
        }

        logger.info(f"核心模块测试完成: {len(core_tests)} 项测试")

    def _test_database(self):
        """测试数据库"""
        logger.info("测试数据库连接...")

        db_tests = []

        try:
            from backend.app.database import DatabaseManager
            db_manager = DatabaseManager()
            db_tests.append("数据库管理器: 正常")
            self.passed_tests.append("数据库: 管理器")
        except Exception as e:
            db_tests.append(f"数据库管理器: 失败 - {e}")
            self.failed_tests.append(f"数据库: 管理器 - {e}")

        try:
            from backend.app.database.models import Base, User
            db_tests.append("数据库模型: 正常")
            self.passed_tests.append("数据库: 模型")
        except Exception as e:
            db_tests.append(f"数据库模型: 失败 - {e}")
            self.failed_tests.append(f"数据库: 模型 - {e}")

        self.test_results['database'] = {
            'tests': db_tests,
            'passed': len([t for t in self.passed_tests if '数据库' in t]),
            'failed': len([t for t in self.failed_tests if '数据库' in t])
        }

        logger.info(f"数据库测试完成: {len(db_tests)} 项测试")

    def _test_api(self):
        """测试API"""
        logger.info("测试API端点...")

        api_tests = []

        try:
            from backend.app.api.v1.api import api_router
            api_tests.append("API路由器: 正常")
            self.passed_tests.append("API: 路由器")
        except Exception as e:
            api_tests.append(f"API路由器: 失败 - {e}")
            self.failed_tests.append(f"API: 路由器 - {e}")

        # 测试主要端点
        endpoints = [
            'backend.app.api.v1.endpoints.experiments',
            'backend.app.api.v1.endpoints.glucose',
            'backend.app.api.v1.endpoints.health'
        ]

        for endpoint in endpoints:
            try:
                importlib.import_module(endpoint)
                api_tests.append(f"端点 {endpoint.split('.')[-1]}: 正常")
                self.passed_tests.append(f"API: 端点 {endpoint.split('.')[-1]}")
            except Exception as e:
                api_tests.append(f"端点 {endpoint.split('.')[-1]}: 失败 - {e}")
                self.failed_tests.append(f"API: 端点 {endpoint.split('.')[-1]} - {e}")

        self.test_results['api'] = {
            'tests': api_tests,
            'passed': len([t for t in self.passed_tests if 'API' in t]),
            'failed': len([t for t in self.failed_tests if 'API' in t])
        }

        logger.info(f"API测试完成: {len(api_tests)} 项测试")

    def _test_optimization_tools(self):
        """测试优化工具"""
        logger.info("测试优化工具...")

        tool_tests = []

        # 测试项目优化器
        try:
            from backend.app.core.project_optimizer import ProjectOptimizer
            optimizer = ProjectOptimizer()
            tool_tests.append("项目优化器: 正常")
            self.passed_tests.append("优化工具: 项目优化器")
        except Exception as e:
            tool_tests.append(f"项目优化器: 失败 - {e}")
            self.failed_tests.append(f"优化工具: 项目优化器 - {e}")

        # 测试代码质量监控器
        try:
            from backend.app.core.code_quality_monitor import CodeQualityMonitor
            monitor = CodeQualityMonitor(str(self.project_root))
            tool_tests.append("代码质量监控器: 正常")
            self.passed_tests.append("优化工具: 代码质量监控器")
        except Exception as e:
            tool_tests.append(f"代码质量监控器: 失败 - {e}")
            self.failed_tests.append(f"优化工具: 代码质量监控器 - {e}")

        # 测试ESM合规性修复器
        try:
            from backend.app.core.esm_compliance_fixer import ESMComplianceFixer
            fixer = ESMComplianceFixer(str(self.project_root))
            tool_tests.append("ESM合规性修复器: 正常")
            self.passed_tests.append("优化工具: ESM合规性修复器")
        except Exception as e:
            tool_tests.append(f"ESM合规性修复器: 失败 - {e}")
            self.failed_tests.append(f"优化工具: ESM合规性修复器 - {e}")

        self.test_results['optimization_tools'] = {
            'tests': tool_tests,
            'passed': len([t for t in self.passed_tests if '优化工具' in t]),
            'failed': len([t for t in self.failed_tests if '优化工具' in t])
        }

        logger.info(f"优化工具测试完成: {len(tool_tests)} 项测试")

    def _generate_test_report(self) -> Dict[str, Any]:
        """生成测试报告"""
        total_passed = len(self.passed_tests)
        total_failed = len(self.failed_tests)
        total_tests = total_passed + total_failed

        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0

        return {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_tests': total_tests,
                'passed': total_passed,
                'failed': total_failed,
                'success_rate': success_rate
            },
            'test_results': self.test_results,
            'passed_tests': self.passed_tests,
            'failed_tests': self.failed_tests,
            'recommendations': self._generate_recommendations()
        }

    def _generate_recommendations(self) -> List[str]:
        """生成改进建议"""
        recommendations = []

        if len(self.failed_tests) > 0:
            recommendations.append(f"修复 {len(self.failed_tests)} 个失败的测试")

        if 'syntax' in self.test_results and self.test_results['syntax']['failed'] > 0:
            recommendations.append("优先修复语法错误")

        if 'imports' in self.test_results and self.test_results['imports']['failed'] > 0:
            recommendations.append("检查并修复导入错误")

        if 'database' in self.test_results and self.test_results['database']['failed'] > 0:
            recommendations.append("检查数据库配置和连接")

        recommendations.extend([
            "定期运行系统测试确保稳定性",
            "为关键模块添加单元测试",
            "建立持续集成流程"
        ])

        return recommendations

def main():
    """主函数"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    tester = SystemTester(".")
    results = tester.run_all_tests()

    print("\n" + "="*80)
    print("🎯 全方位系统测试报告")
    print("="*80)

    summary = results['summary']
    print(f"总测试数: {summary['total_tests']}")
    print(f"通过: {summary['passed']}")
    print(f"失败: {summary['failed']}")
    print(f"成功率: {summary['success_rate']:.1f}%")

    if results['failed_tests']:
        print("\n❌ 失败的测试:")
        for test in results['failed_tests'][:10]:  # 只显示前10个
            print(f"• {test}")
        if len(results['failed_tests']) > 10:
            print(f"... 还有 {len(results['failed_tests']) - 10} 个失败测试")

    print("\n✅ 通过的测试:")
    for test in results['passed_tests'][:10]:  # 只显示前10个
        print(f"• {test}")
    if len(results['passed_tests']) > 10:
        print(f"... 还有 {len(results['passed_tests']) - 10} 个通过测试")

    print("\n💡 改进建议:")
    for rec in results['recommendations']:
        print(f"• {rec}")

    print("="*80)

    # 返回测试结果状态
    return summary['success_rate'] >= 80

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
