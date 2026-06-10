"""
代码质量监控仪表板
实时监控项目代码质量，提供可视化报告
"""

import os
import sys
import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import subprocess
import threading
import time

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

class CodeQualityMonitor:
    """代码质量监控器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.metrics = {}
        self.alerts = []
        self.monitoring_active = False
        self.monitor_thread = None

    def start_monitoring(self, interval: int = 300):
        """开始监控"""
        logger.info("启动代码质量监控...")
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()

    def stop_monitoring(self):
        """停止监控"""
        logger.info("停止代码质量监控...")
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join()

    def _monitoring_loop(self, interval: int):
        """监控循环"""
        while self.monitoring_active:
            try:
                self._collect_metrics()
                self._check_alerts()
                self._generate_report()
                time.sleep(interval)
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
                time.sleep(60)  # 错误后等待1分钟再继续

    def _collect_metrics(self):
        """收集指标"""
        logger.info("收集代码质量指标...")

        backend_dir = self.project_root / "backend" / "app"

        metrics = {
            'timestamp': datetime.now().isoformat(),
            'file_count': 0,
            'line_count': 0,
            'function_count': 0,
            'class_count': 0,
            'import_count': 0,
            'duplicate_imports': 0,
            'missing_docstrings': 0,
            'complexity_score': 0,
            'test_coverage': 0,
            'esm_violations': 0,
            'mcp_modules': 0
        }

        # 统计文件
        for py_file in backend_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                lines = content.split('\n')
                metrics['file_count'] += 1
                metrics['line_count'] += len(lines)

                # 统计函数和类
                metrics['function_count'] += content.count('def ')
                metrics['class_count'] += content.count('class ')
                metrics['import_count'] += content.count('import ')

                # 检查缺失文档字符串
                if not ('"""' in content[:200] or "'''" in content[:200]):
                    metrics['missing_docstrings'] += 1

            except Exception as e:
                logger.warning(f"统计文件失败 {py_file}: {e}")

        # 计算复杂度分数
        metrics['complexity_score'] = self._calculate_complexity_score(metrics)

        # 获取MCP模块数
        metrics['mcp_modules'] = self._count_mcp_modules()

        # 获取ESM违规数
        metrics['esm_violations'] = self._count_esm_violations()

        self.metrics = metrics
        logger.info(f"收集完成: {metrics['file_count']} 文件, {metrics['line_count']} 行")

    def _calculate_complexity_score(self, metrics: Dict[str, Any]) -> float:
        """计算复杂度分数"""
        if metrics['file_count'] == 0:
            return 0.0

        # 基于文件数、函数数、类数计算复杂度
        complexity = (
            metrics['function_count'] * 0.3 +
            metrics['class_count'] * 0.5 +
            metrics['line_count'] / metrics['file_count'] * 0.2
        )

        # 归一化到0-100分
        return min(100, max(0, 100 - complexity / 10))

    def _count_mcp_modules(self) -> int:
        """统计MCP模块数"""
        try:
            registry_file = self.project_root / "mcp_auto_registry.json"
            if registry_file.exists():
                with open(registry_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data.get('total_modules', 0)
        except Exception as e:
            logger.warning(f"统计MCP模块失败: {e}")
        return 0

    def _count_esm_violations(self) -> int:
        """统计ESM违规数"""
        try:
            compliance_file = self.project_root / "esm_compliance_report.json"
            if compliance_file.exists():
                with open(compliance_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data.get('summary', {}).get('total_violations', 0)
        except Exception as e:
            logger.warning(f"统计ESM违规失败: {e}")
        return 0

    def _check_alerts(self):
        """检查告警"""
        alerts = []

        if self.metrics.get('esm_violations', 0) > 100:
            alerts.append({
                'type': 'esm_violations',
                'severity': 'warning',
                'message': f"ESM违规数过多: {self.metrics['esm_violations']}",
                'timestamp': datetime.now().isoformat()
            })

        if self.metrics.get('missing_docstrings', 0) > 50:
            alerts.append({
                'type': 'missing_docstrings',
                'severity': 'info',
                'message': f"缺失文档字符串的文件过多: {self.metrics['missing_docstrings']}",
                'timestamp': datetime.now().isoformat()
            })

        if self.metrics.get('complexity_score', 0) < 50:
            alerts.append({
                'type': 'complexity',
                'severity': 'warning',
                'message': f"代码复杂度分数过低: {self.metrics['complexity_score']:.1f}",
                'timestamp': datetime.now().isoformat()
            })

        self.alerts = alerts

    def _generate_report(self):
        """生成报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'metrics': self.metrics,
            'alerts': self.alerts,
            'status': self._calculate_overall_status()
        }

        # 保存报告
        report_file = self.project_root / "code_quality_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"代码质量报告已生成: {report_file}")

    def _calculate_overall_status(self) -> str:
        """计算整体状态"""
        if not self.metrics:
            return 'unknown'

        score = 0

        # ESM合规性 (30%)
        esm_violations = self.metrics.get('esm_violations', 0)
        if esm_violations < 50:
            score += 30
        elif esm_violations < 200:
            score += 20
        else:
            score += 10

        # 复杂度分数 (30%)
        complexity = self.metrics.get('complexity_score', 0)
        score += complexity * 0.3

        # 文档完整性 (20%)
        missing_docs = self.metrics.get('missing_docstrings', 0)
        total_files = self.metrics.get('file_count', 1)
        doc_coverage = (total_files - missing_docs) / total_files
        score += doc_coverage * 20

        # MCP模块数 (20%)
        mcp_modules = self.metrics.get('mcp_modules', 0)
        if mcp_modules > 100:
            score += 20
        elif mcp_modules > 50:
            score += 15
        else:
            score += 10

        if score >= 80:
            return 'excellent'
        elif score >= 60:
            return 'good'
        elif score >= 40:
            return 'fair'
        else:
            return 'poor'

    def get_current_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        return {
            'timestamp': datetime.now().isoformat(),
            'monitoring_active': self.monitoring_active,
            'metrics': self.metrics,
            'alerts': self.alerts,
            'status': self._calculate_overall_status()
        }

    def generate_dashboard_html(self) -> str:
        """生成仪表板HTML"""
        status = self.get_current_status()

        html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>代码质量监控仪表板</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .dashboard {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
        }}
        .metric-label {{
            color: #666;
            font-size: 0.9em;
        }}
        .status-badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            color: white;
            font-weight: bold;
            margin-top: 10px;
        }}
        .status-excellent {{ background-color: #4CAF50; }}
        .status-good {{ background-color: #8BC34A; }}
        .status-fair {{ background-color: #FF9800; }}
        .status-poor {{ background-color: #F44336; }}
        .alerts {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .alert {{
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
            border-left: 4px solid;
        }}
        .alert-warning {{ background-color: #fff3cd; border-color: #ffc107; }}
        .alert-info {{ background-color: #d1ecf1; border-color: #17a2b8; }}
        .alert-error {{ background-color: #f8d7da; border-color: #dc3545; }}
        .refresh-btn {{
            background: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin-bottom: 20px;
        }}
        .refresh-btn:hover {{
            background: #0056b3;
        }}
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>🎯 代码质量监控仪表板</h1>
            <p>智能健康监测集成系统 - 实时代码质量监控</p>
            <button class="refresh-btn" onclick="location.reload()">🔄 刷新数据</button>
        </div>

        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{status['metrics'].get('file_count', 0)}</div>
                <div class="metric-label">Python文件数</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{status['metrics'].get('line_count', 0):,}</div>
                <div class="metric-label">代码行数</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{status['metrics'].get('function_count', 0)}</div>
                <div class="metric-label">函数数量</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{status['metrics'].get('class_count', 0)}</div>
                <div class="metric-label">类数量</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{status['metrics'].get('mcp_modules', 0)}</div>
                <div class="metric-label">MCP模块数</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{status['metrics'].get('complexity_score', 0):.1f}</div>
                <div class="metric-label">复杂度分数</div>
            </div>
        </div>

        <div class="alerts">
            <h3>📊 质量状态</h3>
            <div class="status-badge status-{status['status']}">
                {status['status'].upper()}
            </div>

            <h3>⚠️ 告警信息</h3>
            {self._generate_alerts_html(status['alerts'])}
        </div>

        <div style="text-align: center; margin-top: 20px; color: #666;">
            <p>最后更新: {status['timestamp']}</p>
            <p>监控状态: {'🟢 活跃' if status['monitoring_active'] else '🔴 停止'}</p>
        </div>
    </div>

    <script>
        // 自动刷新页面
        setTimeout(() => {{
            location.reload();
        }}, 300000); // 5分钟刷新一次
    </script>
</body>
</html>
        """

        return html

    def _generate_alerts_html(self, alerts: List[Dict[str, Any]]) -> str:
        """生成告警HTML"""
        if not alerts:
            return '<p style="color: #28a745;">✅ 无告警信息</p>'

        html = ""
        for alert in alerts:
            severity_class = f"alert-{alert['severity']}"
            html += f"""
            <div class="alert {severity_class}">
                <strong>{alert['type'].upper()}</strong>: {alert['message']}
                <br><small>{alert['timestamp']}</small>
            </div>
            """

        return html

    def save_dashboard(self):
        """保存仪表板"""
        html_content = self.generate_dashboard_html()
        dashboard_file = self.project_root / "code_quality_dashboard.html"

        with open(dashboard_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"仪表板已保存: {dashboard_file}")
        return str(dashboard_file)

def main():
    """主函数"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    monitor = CodeQualityMonitor(".")

    # 收集一次指标
    monitor._collect_metrics()
    monitor._check_alerts()

    # 生成报告
    monitor._generate_report()

    # 保存仪表板
    dashboard_file = monitor.save_dashboard()

    # 显示状态
    status = monitor.get_current_status()

    print("\n" + "="*60)
    print("🎯 代码质量监控仪表板")
    print("="*60)
    print(f"文件数: {status['metrics'].get('file_count', 0)}")
    print(f"代码行数: {status['metrics'].get('line_count', 0):,}")
    print(f"函数数: {status['metrics'].get('function_count', 0)}")
    print(f"类数: {status['metrics'].get('class_count', 0)}")
    print(f"MCP模块: {status['metrics'].get('mcp_modules', 0)}")
    print(f"复杂度分数: {status['metrics'].get('complexity_score', 0):.1f}")
    print(f"整体状态: {status['status'].upper()}")
    print(f"告警数: {len(status['alerts'])}")
    print(f"仪表板文件: {dashboard_file}")
    print("="*60)

    # 启动持续监控
    print("\n启动持续监控...")
    monitor.start_monitoring(interval=300)  # 5分钟间隔

    try:
        while True:
            time.sleep(60)
            current_status = monitor.get_current_status()
            print(f"监控中... 状态: {current_status['status']}, 告警: {len(current_status['alerts'])}")
    except KeyboardInterrupt:
        print("\n停止监控...")
        monitor.stop_monitoring()

if __name__ == "__main__":
    main()
