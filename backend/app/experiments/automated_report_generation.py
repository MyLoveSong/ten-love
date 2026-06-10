

"""
自动化报告生成模块
支持多种格式的实验报告生成，包括PDF、HTML、Markdown等
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
import logging

try:
    from jinja2 import Environment, FileSystemLoader, Template
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    logging.warning("Jinja2 not available, using basic template rendering")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("Matplotlib not available, charts will be skipped")

logger = logging.getLogger(__name__)

@dataclass
class ReportSection:
    """报告章节"""
    title: str
    content: str
    charts: List[Dict[str, Any]] = None
    tables: List[Dict[str, Any]] = None

@dataclass
class ExperimentReport:
    """实验报告"""
    experiment_id: str
    experiment_name: str
    author: str
    timestamp: str
    sections: List[ReportSection]
    metadata: Dict[str, Any]
    summary: Dict[str, Any]

class ReportGenerator:
    """报告生成器"""

    def __init__(self, templates_dir: str = "report_templates", output_dir: str = "reports"):
        self.templates_dir = Path(templates_dir)
        self.output_dir = Path(output_dir)
        self.templates_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

        self.logger = logging.getLogger(__name__)

        # 初始化模板环境
        if JINJA2_AVAILABLE:
            self.jinja_env = Environment(
                loader=FileSystemLoader(str(self.templates_dir)),
                autoescape=True
            )
        else:
            self.jinja_env = None

        # 创建默认模板
        self._create_default_templates()

    def generate_experiment_report(
        self,
        experiment_data: Dict[str, Any],
        template_name: str = "default",
        format: str = "html",
        include_charts: bool = True
    ) -> str:
        """生成实验报告"""

        # 解析实验数据
        report_data = self._parse_experiment_data(experiment_data)

        # 生成图表
        if include_charts and MATPLOTLIB_AVAILABLE:
            charts = self._generate_charts(report_data)
            report_data['charts'] = charts

        # 选择模板
        template = self._get_template(template_name, format)

        # 渲染报告
        if self.jinja_env and template:
            rendered_content = template.render(**report_data)
        else:
            rendered_content = self._render_basic_template(report_data, format)

        # 保存报告
        output_file = self._save_report(rendered_content, report_data['experiment_id'], format)

        self.logger.info(f"报告已生成: {output_file}")
        return str(output_file)

    def generate_comparison_report(
        self,
        comparison_data: Dict[str, Any],
        template_name: str = "comparison",
        format: str = "html"
    ) -> str:
        """生成模型比较报告"""

        report_data = self._parse_comparison_data(comparison_data)

        # 生成比较图表
        if MATPLOTLIB_AVAILABLE:
            charts = self._generate_comparison_charts(report_data)
            report_data['charts'] = charts

        template = self._get_template(template_name, format)

        if self.jinja_env and template:
            rendered_content = template.render(**report_data)
        else:
            rendered_content = self._render_basic_template(report_data, format)

        output_file = self._save_report(rendered_content, f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}", format)

        self.logger.info(f"比较报告已生成: {output_file}")
        return str(output_file)

    def generate_batch_report(
        self,
        batch_data: List[Dict[str, Any]],
        template_name: str = "batch",
        format: str = "html"
    ) -> str:
        """生成批量实验报告"""

        report_data = self._parse_batch_data(batch_data)

        # 生成批量分析图表
        if MATPLOTLIB_AVAILABLE:
            charts = self._generate_batch_charts(report_data)
            report_data['charts'] = charts

        template = self._get_template(template_name, format)

        if self.jinja_env and template:
            rendered_content = template.render(**report_data)
        else:
            rendered_content = self._render_basic_template(report_data, format)

        output_file = self._save_report(rendered_content, f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}", format)

        self.logger.info(f"批量报告已生成: {output_file}")
        return str(output_file)

    def _parse_experiment_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """解析实验数据"""

        return {
            'experiment_id': data.get('experiment_id', 'unknown'),
            'experiment_name': data.get('experiment_name', 'Unnamed Experiment'),
            'author': data.get('author', 'Unknown'),
            'timestamp': data.get('timestamp', datetime.now().isoformat()),
            'model_name': data.get('model_name', 'Unknown Model'),
            'metrics': data.get('metrics', {}),
            'scores': data.get('scores', []),
            'mean_score': data.get('mean', 0),
            'std_score': data.get('std', 0),
            'confidence_interval': data.get('ci95', {}),
            'bootstrap_ci': data.get('bootstrap_ci', None),
            'detailed_results': data.get('detailed_results', []),
            'statistical_tests': data.get('statistical_tests', {}),
            'metadata': data.get('metadata', {})
        }

    def _parse_comparison_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """解析比较数据"""

        return {
            'comparison_id': data.get('comparison_id', f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            'timestamp': datetime.now().isoformat(),
            'models': data.get('results', {}),
            'best_model': data.get('best', 'Unknown'),
            'statistical_comparison': data.get('statistical_comparison', {}),
            'significance_tests': data.get('significance', {}),
            'summary': data.get('summary', {})
        }

    def _parse_batch_data(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """解析批量数据"""

        return {
            'batch_id': f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'experiments': data,
            'total_experiments': len(data),
            'summary_stats': self._calculate_batch_summary(data)
        }

    def _calculate_batch_summary(self, experiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算批量实验摘要统计"""

        if not experiments:
            return {}

        scores = [exp.get('mean', 0) for exp in experiments]
        model_names = [exp.get('model_name', 'Unknown') for exp in experiments]

        return {
            'mean_score': sum(scores) / len(scores),
            'std_score': (sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores))**0.5,
            'min_score': min(scores),
            'max_score': max(scores),
            'best_model': model_names[scores.index(max(scores))],
            'worst_model': model_names[scores.index(min(scores))],
            'model_counts': {name: model_names.count(name) for name in set(model_names)}
        }

    def _generate_charts(self, report_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成图表"""

        charts = []

        try:
            # 分数分布图
            if 'scores' in report_data and report_data['scores']:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(report_data['scores'], bins=10, alpha=0.7, edgecolor='black')
                ax.set_title('Score Distribution')
                ax.set_xlabel('Score')
                ax.set_ylabel('Frequency')

                chart_path = self.output_dir / f"{report_data['experiment_id']}_scores.png"
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()

                charts.append({
                    'type': 'histogram',
                    'title': 'Score Distribution',
                    'path': str(chart_path)
                })

            # 置信区间图
            if 'confidence_interval' in report_data and report_data['confidence_interval']:
                ci = report_data['confidence_interval']
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.errorbar([0], [report_data['mean_score']],
                           yerr=[[report_data['mean_score'] - ci['low']],
                                 [ci['high'] - report_data['mean_score']]],
                           fmt='o', capsize=5, capthick=2)
                ax.set_title('Confidence Interval')
                ax.set_xlabel('Experiment')
                ax.set_ylabel('Score')
                ax.set_xlim(-0.5, 0.5)

                chart_path = self.output_dir / f"{report_data['experiment_id']}_ci.png"
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()

                charts.append({
                    'type': 'errorbar',
                    'title': 'Confidence Interval',
                    'path': str(chart_path)
                })

        except Exception as e:
            self.logger.warning(f"图表生成失败: {e}")

        return charts

    def _generate_comparison_charts(self, report_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成比较图表"""

        charts = []

        try:
            # 模型性能比较图
            models = report_data['models']
            model_names = list(models.keys())
            scores = [models[name].get('mean', 0) for name in model_names]
            errors = [models[name].get('std', 0) for name in model_names]

            fig, ax = plt.subplots(figsize=(12, 6))
            bars = ax.bar(model_names, scores, yerr=errors, capsize=5, alpha=0.7)
            ax.set_title('Model Performance Comparison')
            ax.set_xlabel('Models')
            ax.set_ylabel('Score')
            ax.tick_params(axis='x', rotation=45)

            # 高亮最佳模型
            best_idx = scores.index(max(scores))
            bars[best_idx].set_color('gold')

            chart_path = self.output_dir / f"{report_data['comparison_id']}_comparison.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()

            charts.append({
                'type': 'bar',
                'title': 'Model Performance Comparison',
                'path': str(chart_path)
            })

        except Exception as e:
            self.logger.warning(f"比较图表生成失败: {e}")

        return charts

    def _generate_batch_charts(self, report_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成批量图表"""

        charts = []

        try:
            # 批量实验趋势图
            experiments = report_data['experiments']
            scores = [exp.get('mean', 0) for exp in experiments]
            timestamps = [exp.get('timestamp', '') for exp in experiments]

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(range(len(scores)), scores, marker='o', linewidth=2, markersize=6)
            ax.set_title('Batch Experiment Trend')
            ax.set_xlabel('Experiment Index')
            ax.set_ylabel('Score')
            ax.grid(True, alpha=0.3)

            chart_path = self.output_dir / f"{report_data['batch_id']}_trend.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()

            charts.append({
                'type': 'line',
                'title': 'Batch Experiment Trend',
                'path': str(chart_path)
            })

        except Exception as e:
            self.logger.warning(f"批量图表生成失败: {e}")

        return charts

    def _get_template(self, template_name: str, format: str) -> Optional[Template]:
        """获取模板"""

        if not self.jinja_env:
            return None

        template_file = f"{template_name}_{format}.html"
        try:
            return self.jinja_env.get_template(template_file)
        except Exception:
            # 尝试默认模板
            try:
                return self.jinja_env.get_template(f"default_{format}.html")
            except Exception:
                return None

    def _render_basic_template(self, data: Dict[str, Any], format: str) -> str:
        """基础模板渲染"""

        if format == "html":
            return self._render_basic_html(data)
        elif format == "markdown":
            return self._render_basic_markdown(data)
        else:
            return json.dumps(data, indent=2, ensure_ascii=False)

    def _render_basic_html(self, data: Dict[str, Any]) -> str:
        """基础HTML渲染"""

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{data.get('experiment_name', 'Experiment Report')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ margin: 10px 0; }}
                .chart {{ text-align: center; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>{data.get('experiment_name', 'Experiment Report')}</h1>
            <p><strong>实验ID:</strong> {data.get('experiment_id', 'Unknown')}</p>
            <p><strong>作者:</strong> {data.get('author', 'Unknown')}</p>
            <p><strong>时间:</strong> {data.get('timestamp', 'Unknown')}</p>

            <h2>实验结果</h2>
            <div class="metric"><strong>平均分数:</strong> {data.get('mean_score', 0):.4f}</div>
            <div class="metric"><strong>标准差:</strong> {data.get('std_score', 0):.4f}</div>

            {self._render_metrics_table(data.get('metrics', {}))}

            {self._render_charts_html(data.get('charts', []))}
        </body>
        </html>
        """

        return html

    def _render_basic_markdown(self, data: Dict[str, Any]) -> str:
        """基础Markdown渲染"""

        md = f"""# {data.get('experiment_name', 'Experiment Report')}

## 基本信息
- **实验ID:** {data.get('experiment_id', 'Unknown')}
- **作者:** {data.get('author', 'Unknown')}
- **时间:** {data.get('timestamp', 'Unknown')}

## 实验结果
- **平均分数:** {data.get('mean_score', 0):.4f}
- **标准差:** {data.get('std_score', 0):.4f}

{self._render_metrics_markdown(data.get('metrics', {}))}

{self._render_charts_markdown(data.get('charts', []))}
"""

        return md

    def _render_metrics_table(self, metrics: Dict[str, Any]) -> str:
        """渲染指标表格"""

        if not metrics:
            return ""

        html = "<h2>详细指标</h2><table><tr><th>指标</th><th>值</th></tr>"
        for key, value in metrics.items():
            html += f"<tr><td>{key}</td><td>{value}</td></tr>"
        html += "</table>"

        return html

    def _render_metrics_markdown(self, metrics: Dict[str, Any]) -> str:
        """渲染指标Markdown"""

        if not metrics:
            return ""

        md = "## 详细指标\n\n| 指标 | 值 |\n|------|-----|\n"
        for key, value in metrics.items():
            md += f"| {key} | {value} |\n"

        return md

    def _render_charts_html(self, charts: List[Dict[str, Any]]) -> str:
        """渲染图表HTML"""

        if not charts:
            return ""

        html = "<h2>图表</h2>"
        for chart in charts:
            html += f"""
            <div class="chart">
                <h3>{chart.get('title', 'Chart')}</h3>
                <img src="{chart.get('path', '')}" alt="{chart.get('title', 'Chart')}" style="max-width: 100%;">
            </div>
            """

        return html

    def _render_charts_markdown(self, charts: List[Dict[str, Any]]) -> str:
        """渲染图表Markdown"""

        if not charts:
            return ""

        md = "## 图表\n\n"
        for chart in charts:
            md += f"### {chart.get('title', 'Chart')}\n\n"
            md += f"![{chart.get('title', 'Chart')}]({chart.get('path', '')})\n\n"

        return md

    def _save_report(self, content: str, experiment_id: str, format: str) -> Path:
        """保存报告"""

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{experiment_id}_{timestamp}.{format}"
        output_file = self.output_dir / filename

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)

        return output_file

    def _create_default_templates(self):
        """创建默认模板"""

        # HTML模板
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ experiment_name }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1 { color: #333; }
        h2 { color: #666; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .metric { margin: 10px 0; }
        .chart { text-align: center; margin: 20px 0; }
    </style>
</head>
<body>
    <h1>{{ experiment_name }}</h1>
    <p><strong>实验ID:</strong> {{ experiment_id }}</p>
    <p><strong>作者:</strong> {{ author }}</p>
    <p><strong>时间:</strong> {{ timestamp }}</p>

    <h2>实验结果</h2>
    <div class="metric"><strong>平均分数:</strong> {{ "%.4f"|format(mean_score) }}</div>
    <div class="metric"><strong>标准差:</strong> {{ "%.4f"|format(std_score) }}</div>

    {% if charts %}
    <h2>图表</h2>
    {% for chart in charts %}
    <div class="chart">
        <h3>{{ chart.title }}</h3>
        <img src="{{ chart.path }}" alt="{{ chart.title }}" style="max-width: 100%;">
    </div>
    {% endfor %}
    {% endif %}
</body>
</html>
"""

        template_file = self.templates_dir / "default_html.html"
        with open(template_file, 'w', encoding='utf-8') as f:
            f.write(html_template)

# 全局报告生成器实例
report_generator = ReportGenerator()

__all__ = ["'logger'", "'ReportSection'", "'ExperimentReport'", "'ReportGenerator'", "'report_generator'"]
