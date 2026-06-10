

"""
项目优化执行器
统一执行所有优化任务，包括MCP注册、ESM转换、依赖分析
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# 使用标准库logging避免冲突
import logging as std_logging
logger = std_logging.getLogger(__name__)

# 延迟导入避免循环依赖
def get_mcp_registry():
    try:
        from backend.app.core.mcp_registry import mcp_registry
        return mcp_registry
    except ImportError:
        return None

def get_import_optimizer():
    try:
        from backend.app.core.esm_converter import ImportOptimizer
        return ImportOptimizer
    except ImportError:
        return None

def get_module_optimizer():
    try:
        from app.backend.app.core.dependency_analyzer import ModuleOptimizer
        return ModuleOptimizer
    except ImportError:
        return None

class ProjectOptimizer:
    """项目优化器"""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.results = {}

    def register_core_modules(self) -> Dict[str, object]:
        """注册核心模块到MCP"""
        logger.info("注册核心模块...")

        mcp_registry = get_mcp_registry()
        if not mcp_registry:
            return {'status': 'error', 'message': 'MCP注册中心不可用'}

        # 注册实验模块
        mcp_registry.register_module(
            name="experiments",
            version="1.0.0",
            description="实验设计与验证模块",
            dependencies=["core"],
            exports=["enhanced_kfold_train", "enhanced_compare_models", "comprehensive_evaluator"],
            tags=["experiments", "validation", "statistics"]
        )

        # 注册数据处理模块
        mcp_registry.register_module(
            name="data_processing",
            version="1.0.0",
            description="数据处理与增强模块",
            dependencies=["core"],
            exports=["DataCleaner", "DataEnhancer", "pipeline_registry"],
            tags=["data", "processing", "pipeline"]
        )

        # 注册模型模块
        mcp_registry.register_module(
            name="modeling",
            version="1.0.0",
            description="机器学习模型模块",
            dependencies=["core", "data_processing"],
            exports=["model_registry", "train", "predict"],
            tags=["ml", "models", "training"]
        )

        # 注册API模块
        mcp_registry.register_module(
            name="api",
            version="1.0.0",
            description="API端点模块",
            dependencies=["core", "experiments", "modeling"],
            exports=["router", "endpoints"],
            tags=["api", "endpoints", "rest"]
        )

        # 导出注册表
        registry_file = self.project_root / "mcp_registry.json"
        mcp_registry.export_registry(str(registry_file))

        result = {
            'modules_registered': len(mcp_registry._modules),
            'registry_file': str(registry_file),
            'status': 'success'
        }

        logger.info(f"核心模块注册完成: {result}")
        return result

    def optimize_imports(self) -> Dict[str, object]:
        """优化导入"""
        logger.info("优化项目导入...")

        ImportOptimizerClass = get_import_optimizer()
        if not ImportOptimizerClass:
            return {'status': 'error', 'message': '导入优化器不可用'}

        optimizer = ImportOptimizerClass(str(self.project_root))
        results = optimizer.optimize_project()

        logger.info(f"导入优化完成: {results['summary']}")
        return results

    def analyze_dependencies(self) -> Dict[str, object]:
        """分析依赖关系"""
        logger.info("分析模块依赖...")

        ModuleOptimizerClass = get_module_optimizer()
        if not ModuleOptimizerClass:
            return {'status': 'error', 'message': '模块优化器不可用'}

        analyzer = ModuleOptimizerClass(str(self.project_root))
        results = analyzer.optimize_project()

        logger.info(f"依赖分析完成: {results['summary']}")
        return results

    def optimize_pnpm_config(self) -> Dict[str, object]:
        """优化pnpm配置"""
        logger.info("优化pnpm配置...")

        # 更新.pnpmrc配置
        pnpmrc_content = '''# PNPM 配置文件 - MCP架构优化版
# 企业级包管理配置

# 存储路径
store-dir = .pnpm-store

# 网络配置
registry = https://registry.npmjs.org/
fetch-retries = 3
fetch-retry-factor = 10
fetch-retry-mintimeout = 10000
fetch-retry-maxtimeout = 60000

# 安装配置
prefer-offline = true
prefer-frozen-lockfile = true
save-exact = true
save-prefix = ""

# 工作空间配置
link-workspace-packages = true
prefer-workspace-packages = true

# 安全配置
audit-level = moderate
fund = false

# 日志配置
loglevel = info
reporter = default

# 缓存配置
cache-dir = .pnpm-cache
cache = true

# 性能配置
network-concurrency = 16
child-concurrency = 5

# MCP架构特定配置
mcp-modules = true
mcp-components = true
mcp-plugins = true

# ESM支持
esm-support = true
module-resolution = node

# 依赖去重
dedupe-peer-dependents = true
auto-install-peers = true
'''

        pnpmrc_file = self.project_root / ".pnpmrc"
        with open(pnpmrc_file, 'w', encoding='utf-8') as f:
            f.write(pnpmrc_content)

        # 更新package.json脚本
        package_json_file = self.project_root / "package.json"
        if package_json_file.exists():
            with open(package_json_file, 'r', encoding='utf-8') as f:
                package_data = json.load(f)

            # 添加优化脚本
            if 'scripts' not in package_data:
                package_data['scripts'] = {}

            package_data['scripts'].update({
                "optimize": "python backend/app/core/project_optimizer.py",
                "analyze-deps": "python backend/app/core/dependency_analyzer.py",
                "convert-esm": "python backend/app/core/esm_converter.py",
                "mcp-register": "python backend/app/core/mcp_registry.py",
                "dedupe": "pnpm dedupe",
                "audit": "pnpm audit --audit-level moderate"
            })

            with open(package_json_file, 'w', encoding='utf-8') as f:
                json.dump(package_data, f, indent=2, ensure_ascii=False)

        result = {
            'pnpmrc_updated': True,
            'package_json_updated': True,
            'status': 'success'
        }

        logger.info(f"pnpm配置优化完成: {result}")
        return result

    def run_full_optimization(self) -> Dict[str, object]:
        """运行完整优化"""
        logger.info("开始完整项目优化...")

        start_time = datetime.now()

        try:
            # 1. 注册核心模块
            self.results['mcp_registration'] = self.register_core_modules()

            # 2. 优化导入
            self.results['import_optimization'] = self.optimize_imports()

            # 3. 分析依赖
            self.results['dependency_analysis'] = self.analyze_dependencies()

            # 4. 优化pnpm配置
            self.results['pnpm_optimization'] = self.optimize_pnpm_config()

            # 生成总结报告
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            summary = {
                'timestamp': datetime.now().isoformat(),
                'duration_seconds': duration,
                'status': 'success',
                'optimizations_completed': len(self.results),
                'summary': {
                    'modules_registered': self.results.get('mcp_registration', {}).get('modules_registered', 0),
                    'files_processed': self.results.get('import_optimization', {}).get('summary', {}).get('files_processed', 0),
                    'duplicates_found': self.results.get('import_optimization', {}).get('summary', {}).get('duplicates_found', 0),
                    'circular_dependencies': self.results.get('dependency_analysis', {}).get('summary', {}).get('circular_dependencies', 0)
                }
            }

            # 保存完整报告
            report_file = self.project_root / "project_optimization_report.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'summary': summary,
                    'details': self.results
                }, f, indent=2, ensure_ascii=False)

            logger.info(f"完整优化完成，报告已保存: {report_file}")
            return summary

        except Exception as e:
            logger.error(f"优化过程中出现错误: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'error': str(e)
            }

def main():
    """主函数"""
    std_logging.basicConfig(
        level=std_logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    optimizer = ProjectOptimizer()
    results = optimizer.run_full_optimization()

    print("\n" + "="*50)
    print("🎯 项目优化完成报告")
    print("="*50)
    print(f"状态: {results['status']}")
    print(f"耗时: {results.get('duration_seconds', 0):.2f} 秒")
    print(f"完成优化: {results.get('optimizations_completed', 0)} 项")

    if results['status'] == 'success':
        summary = results['summary']
        print(f"注册模块: {summary['modules_registered']}")
        print(f"处理文件: {summary['files_processed']}")
        print(f"发现重复: {summary['duplicates_found']}")
        print(f"循环依赖: {summary['circular_dependencies']}")

    print("="*50)

if __name__ == "__main__":
    main()

__all__ = ["'project_root'", "'logger'", "'get_mcp_registry'", "'get_import_optimizer'", "'get_module_optimizer'", "'ProjectOptimizer'", "'main'"]
