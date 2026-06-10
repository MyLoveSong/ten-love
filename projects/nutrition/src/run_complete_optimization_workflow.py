#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的优化工作流程

1. 数据充足性评估
2. 实施优化方案
3. 验证优化效果
"""

import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime
import json
import logging

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_step(step_name: str, script_path: str, description: str) -> dict:
    """运行单个步骤"""
    print(f"\n{'='*60}")
    print(f"步骤: {step_name}")
    print(f"描述: {description}")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=str(project_root / "stage1"),
            capture_output=True,
            text=True,
            timeout=300
        )

        duration = time.time() - start_time

        if result.returncode == 0:
            print(f"[成功] {step_name} 完成 (耗时: {duration:.1f}秒)")
            return {"status": "success", "duration": duration, "output": result.stdout}
        else:
            print(f"[失败] {step_name} 失败 (耗时: {duration:.1f}秒)")
            if result.stderr:
                error_lines = result.stderr.strip().split('\n')
                if error_lines:
                    print(f"   错误: {error_lines[-1][:150]}")
            return {"status": "failed", "duration": duration, "error": result.stderr}

    except Exception as e:
        duration = time.time() - start_time
        print(f"[异常] {step_name} 异常: {e}")
        return {"status": "error", "duration": duration, "error": str(e)}


def main():
    """主函数"""
    print("=" * 60)
    print("完整优化工作流程")
    print("=" * 60)
    print("包括: 数据评估 → 实施优化 → 验证效果")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    results = {}

    # 步骤1: 数据充足性评估
    results["data_evaluation"] = run_step(
        "数据充足性评估",
        "evaluate_data_adequacy.py",
        "评估当前数据量是否足够，是否需要采集更多数据"
    )

    # 步骤2: 实施优化方案
    results["implementation"] = run_step(
        "实施优化方案",
        "implement_optimizations.py",
        "使用生成的训练数据和优化策略创建实施计划"
    )

    # 步骤3: 验证优化效果
    results["verification"] = run_step(
        "验证优化效果",
        "run_optimized_verification.py",
        "运行完整验证流程确认优化效果"
    )

    # 生成综合报告
    print("\n" + "=" * 60)
    print("工作流程总结")
    print("=" * 60)

    total_time = sum(r.get("duration", 0) for r in results.values())
    success_count = sum(1 for r in results.values() if r.get("status") == "success")

    print(f"总步骤数: {len(results)}")
    print(f"成功步骤: {success_count}")
    print(f"总耗时: {total_time:.1f}秒")

    # 保存报告
    output_dir = project_root / "stage1" / "outputs" / "optimization_workflow"
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / f"workflow_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "workflow_results": results,
            "summary": {
                "total_steps": len(results),
                "successful_steps": success_count,
                "total_duration": total_time
            }
        }, f, indent=2, ensure_ascii=False)

    print(f"\n工作流程报告已保存: {report_path}")

    if success_count == len(results):
        print("\n[完成] 所有步骤完成！")
    else:
        print(f"\n[警告] 部分步骤失败，请检查日志")


if __name__ == "__main__":
    main()
