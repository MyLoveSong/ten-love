

"""
系统可解释性和透明度增强模块
基于用户建议的改进方向设计
实现可解释AI模型、可视化决策路径和决策审计
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
from collections import deque
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import hashlib

logger = logging.getLogger(__name__)

class ExplainableAIModel(nn.Module):
    """
    可解释AI模型
    基于用户建议的改进方向设计
    使用LIME、SHAP等技术提供模型解释
    """

    def __init__(self,
                 input_dim: int = 256,
                 hidden_dim: int = 128,
                 num_classes: int = 5,
                 explanation_methods: List[str] = ['lime', 'shap', 'gradient']):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.explanation_methods = explanation_methods

        # 主模型
        self.main_model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

        # 特征重要性网络
        self.feature_importance = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Softmax(dim=-1)
        )

        # 注意力机制
        self.attention_layer = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # 解释生成器
        self.explanation_generator = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 解释历史
        self.explanation_history = deque(maxlen=1000)

    def forward(self,
                x: torch.Tensor,
                return_explanations: bool = True) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 主模型预测
        main_output = self.main_model(x)

        # 特征重要性
        feature_importance = self.feature_importance(x)

        # 注意力权重
        x_reshaped = x.unsqueeze(1)  # 添加序列维度
        attended_output, attention_weights = self.attention_layer(x_reshaped, x_reshaped, x_reshaped)

        # 生成解释
        if return_explanations:
            explanation_input = torch.cat([attended_output.squeeze(1), x], dim=-1)
            explanation_score = self.explanation_generator(explanation_input)
        else:
            explanation_score = torch.tensor(0.0)

        return {
            'prediction': main_output,
            'feature_importance': feature_importance,
            'attention_weights': attention_weights,
            'explanation_score': explanation_score,
            'attended_features': attended_output
        }

    def explain_prediction(self,
                          x: torch.Tensor,
                          method: str = 'lime') -> Dict[str, Any]:
        """解释预测结果"""
        with torch.no_grad():
            output = self.forward(x, return_explanations=True)

        if method == 'lime':
            return self._lime_explanation(x, output)
        elif method == 'shap':
            return self._shap_explanation(x, output)
        elif method == 'gradient':
            return self._gradient_explanation(x, output)
        else:
            return self._default_explanation(x, output)

    def _lime_explanation(self,
                         x: torch.Tensor,
                         output: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """LIME解释"""
        # 简化的LIME实现
        feature_importance = output['feature_importance'].squeeze().numpy()

        # 选择最重要的特征
        top_features = np.argsort(feature_importance)[-10:]  # 前10个重要特征

        explanation = {
            'method': 'LIME',
            'top_features': top_features.tolist(),
            'feature_importance': feature_importance[top_features].tolist(),
            'explanation_text': f"模型预测主要基于前{len(top_features)}个特征",
            'confidence': output['explanation_score'].item()
        }

        return explanation

    def _shap_explanation(self,
                         x: torch.Tensor,
                         output: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """SHAP解释"""
        # 简化的SHAP实现
        feature_importance = output['feature_importance'].squeeze().numpy()

        # 计算SHAP值（简化）
        shap_values = feature_importance * x.squeeze().numpy()

        explanation = {
            'method': 'SHAP',
            'shap_values': shap_values.tolist(),
            'feature_importance': feature_importance.tolist(),
            'explanation_text': "SHAP值显示了每个特征对预测的贡献",
            'confidence': output['explanation_score'].item()
        }

        return explanation

    def _gradient_explanation(self,
                             x: torch.Tensor,
                             output: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """梯度解释"""
        # 计算梯度
        x.requires_grad_(True)
        prediction = self.main_model(x)

        # 计算梯度
        gradients = torch.autograd.grad(
            outputs=prediction.sum(),
            inputs=x,
            create_graph=True
        )[0]

        explanation = {
            'method': 'Gradient',
            'gradients': gradients.squeeze().detach().numpy().tolist(),
            'feature_importance': gradients.abs().squeeze().detach().numpy().tolist(),
            'explanation_text': "梯度显示了输入变化对预测的影响",
            'confidence': output['explanation_score'].item()
        }

        return explanation

    def _default_explanation(self,
                            x: torch.Tensor,
                            output: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """默认解释"""
        feature_importance = output['feature_importance'].squeeze().numpy()

        explanation = {
            'method': 'Default',
            'feature_importance': feature_importance.tolist(),
            'attention_weights': output['attention_weights'].squeeze().detach().numpy().tolist(),
            'explanation_text': "基于特征重要性和注意力权重的解释",
            'confidence': output['explanation_score'].item()
        }

        return explanation

class DecisionPathVisualizer(nn.Module):
    """
    决策路径可视化器
    基于用户建议的改进方向设计
    可视化决策路径和决策过程
    """

    def __init__(self,
                 input_dim: int = 256,
                 hidden_dim: int = 128,
                 num_decision_nodes: int = 10):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_decision_nodes = num_decision_nodes

        # 决策节点网络
        self.decision_nodes = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            ) for _ in range(num_decision_nodes)
        ])

        # 决策路径编码器
        self.path_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 路径选择器
        self.path_selector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_decision_nodes),
            nn.Softmax(dim=-1)
        )

        # 决策历史
        self.decision_history = deque(maxlen=1000)

    def forward(self,
                x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 编码输入
        encoded_input = self.path_encoder(x)

        # 计算每个决策节点的输出
        node_outputs = []
        for node in self.decision_nodes:
            node_output = node(x)
            node_outputs.append(node_output)

        # 路径选择
        path_weights = self.path_selector(encoded_input)

        # 加权决策
        weighted_decision = torch.sum(
            torch.stack(node_outputs) * path_weights.unsqueeze(-1), dim=0
        )

        return {
            'node_outputs': node_outputs,
            'path_weights': path_weights,
            'weighted_decision': weighted_decision,
            'encoded_input': encoded_input
        }

    def visualize_decision_path(self,
                               x: torch.Tensor,
                               decision_id: str = "decision_001") -> Dict[str, Any]:
        """可视化决策路径"""
        with torch.no_grad():
            output = self.forward(x)

        # 创建决策路径图
        decision_path = {
            'decision_id': decision_id,
            'timestamp': datetime.now(),
            'input_features': x.squeeze().numpy().tolist(),
            'node_outputs': [node_output.item() for node_output in output['node_outputs']],
            'path_weights': output['path_weights'].squeeze().numpy().tolist(),
            'final_decision': output['weighted_decision'].item(),
            'decision_path': self._trace_decision_path(output)
        }

        # 记录决策历史
        self.decision_history.append(decision_path)

        return decision_path

    def _trace_decision_path(self,
                             output: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]:
        """追踪决策路径"""
        path_weights = output['path_weights'].squeeze().numpy()
        node_outputs = [node_output.item() for node_output in output['node_outputs']]

        # 按权重排序节点
        sorted_indices = np.argsort(path_weights)[::-1]

        decision_path = []
        for i, idx in enumerate(sorted_indices):
            decision_path.append({
                'node_id': f'node_{idx}',
                'weight': path_weights[idx],
                'output': node_outputs[idx],
                'rank': i + 1,
                'contribution': path_weights[idx] * node_outputs[idx]
            })

        return decision_path

    def generate_decision_flowchart(self,
                                   decision_path: Dict[str, Any]) -> str:
        """生成决策流程图"""
        flowchart = f"""
        graph TD
            A[输入特征] --> B[决策节点1]
            A --> C[决策节点2]
            A --> D[决策节点3]
            A --> E[决策节点4]
            A --> F[决策节点5]

            B --> G[路径权重1: {decision_path['path_weights'][0]:.3f}]
            C --> H[路径权重2: {decision_path['path_weights'][1]:.3f}]
            D --> I[路径权重3: {decision_path['path_weights'][2]:.3f}]
            E --> J[路径权重4: {decision_path['path_weights'][3]:.3f}]
            F --> K[路径权重5: {decision_path['path_weights'][4]:.3f}]

            G --> L[最终决策: {decision_path['final_decision']:.3f}]
            H --> L
            I --> L
            J --> L
            K --> L
        """

        return flowchart

    def get_decision_statistics(self) -> Dict[str, Any]:
        """获取决策统计信息"""
        if not self.decision_history:
            return {}

        # 统计路径权重
        all_path_weights = [decision['path_weights'] for decision in self.decision_history]
        avg_path_weights = np.mean(all_path_weights, axis=0)

        # 统计决策分布
        all_decisions = [decision['final_decision'] for decision in self.decision_history]

        return {
            'total_decisions': len(self.decision_history),
            'average_path_weights': avg_path_weights.tolist(),
            'decision_statistics': {
                'mean': np.mean(all_decisions),
                'std': np.std(all_decisions),
                'min': np.min(all_decisions),
                'max': np.max(all_decisions)
            },
            'most_used_nodes': self._get_most_used_nodes()
        }

    def _get_most_used_nodes(self) -> List[Dict[str, Any]]:
        """获取最常用的决策节点"""
        node_usage = [0] * self.num_decision_nodes

        for decision in self.decision_history:
            path_weights = decision['path_weights']
            for i, weight in enumerate(path_weights):
                node_usage[i] += weight

        # 排序
        sorted_nodes = sorted(enumerate(node_usage), key=lambda x: x[1], reverse=True)

        return [
            {
                'node_id': f'node_{idx}',
                'usage_score': usage,
                'rank': i + 1
            }
            for i, (idx, usage) in enumerate(sorted_nodes)
        ]

class DecisionAuditSystem(nn.Module):
    """
    决策审计系统
    基于用户建议的改进方向设计
    提供决策审计和回溯分析
    """

    def __init__(self,
                 audit_categories: List[str] = ['input', 'process', 'output', 'feedback'],
                 max_audit_records: int = 10000):
        super().__init__()

        self.audit_categories = audit_categories
        self.max_audit_records = max_audit_records

        # 审计记录存储
        self.audit_records = deque(maxlen=max_audit_records)

        # 审计分析器
        self.audit_analyzer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, len(audit_categories)),
            nn.Softmax(dim=-1)
        )

        # 异常检测器
        self.anomaly_detector = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # 审计规则
        self.audit_rules = {
            'input_validation': self._validate_input,
            'process_consistency': self._check_process_consistency,
            'output_reasonableness': self._check_output_reasonableness,
            'feedback_alignment': self._check_feedback_alignment
        }

    def audit_decision(self,
                       decision_id: str,
                       input_data: Dict[str, Any],
                       decision_process: Dict[str, Any],
                       output_result: Dict[str, Any],
                       user_feedback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """审计决策"""
        # 创建审计记录
        audit_record = {
            'decision_id': decision_id,
            'timestamp': datetime.now(),
            'input_data': input_data,
            'decision_process': decision_process,
            'output_result': output_result,
            'user_feedback': user_feedback,
            'audit_results': {}
        }

        # 执行审计规则
        for rule_name, rule_function in self.audit_rules.items():
            try:
                audit_result = rule_function(audit_record)
                audit_record['audit_results'][rule_name] = audit_result
            except Exception as e:
                audit_record['audit_results'][rule_name] = {
                    'status': 'error',
                    'message': str(e)
                }

        # 计算审计分数
        audit_score = self._calculate_audit_score(audit_record['audit_results'])
        audit_record['audit_score'] = audit_score

        # 检测异常
        anomaly_score = self._detect_anomaly(audit_record)
        audit_record['anomaly_score'] = anomaly_score

        # 存储审计记录
        self.audit_records.append(audit_record)

        return audit_record

    def _validate_input(self,
                       audit_record: Dict[str, Any]) -> Dict[str, Any]:
        """验证输入数据"""
        input_data = audit_record['input_data']

        # 检查必要字段
        required_fields = ['glucose', 'blood_pressure_systolic', 'heart_rate']
        missing_fields = [field for field in required_fields if field not in input_data]

        # 检查数据范围
        range_checks = []
        if 'glucose' in input_data:
            glucose = input_data['glucose']
            if glucose < 0 or glucose > 50:
                range_checks.append(f"血糖值{glucose}超出正常范围")

        if 'blood_pressure_systolic' in input_data:
            bp = input_data['blood_pressure_systolic']
            if bp < 50 or bp > 250:
                range_checks.append(f"收缩压{bp}超出正常范围")

        return {
            'status': 'pass' if not missing_fields and not range_checks else 'fail',
            'missing_fields': missing_fields,
            'range_issues': range_checks,
            'validation_score': 1.0 if not missing_fields and not range_checks else 0.5
        }

    def _check_process_consistency(self,
                                   audit_record: Dict[str, Any]) -> Dict[str, Any]:
        """检查决策过程一致性"""
        decision_process = audit_record['decision_process']

        # 检查决策步骤
        required_steps = ['data_preprocessing', 'feature_extraction', 'model_prediction', 'post_processing']
        missing_steps = [step for step in required_steps if step not in decision_process]

        # 检查时间戳
        timestamps = decision_process.get('timestamps', {})
        time_consistency = True
        if len(timestamps) > 1:
            times = list(timestamps.values())
            for i in range(1, len(times)):
                if times[i] < times[i-1]:
                    time_consistency = False
                    break

        return {
            'status': 'pass' if not missing_steps and time_consistency else 'fail',
            'missing_steps': missing_steps,
            'time_consistency': time_consistency,
            'consistency_score': 1.0 if not missing_steps and time_consistency else 0.7
        }

    def _check_output_reasonableness(self,
                                    audit_record: Dict[str, Any]) -> Dict[str, Any]:
        """检查输出合理性"""
        output_result = audit_record['output_result']
        input_data = audit_record['input_data']

        # 检查输出与输入的一致性
        consistency_checks = []

        if 'glucose' in input_data and 'predicted_glucose' in output_result:
            input_glucose = input_data['glucose']
            predicted_glucose = output_result['predicted_glucose']
            if abs(predicted_glucose - input_glucose) > 10:
                consistency_checks.append(f"预测血糖{predicted_glucose}与输入血糖{input_glucose}差异过大")

        # 检查输出范围
        range_checks = []
        if 'predicted_glucose' in output_result:
            pred_glucose = output_result['predicted_glucose']
            if pred_glucose < 0 or pred_glucose > 50:
                range_checks.append(f"预测血糖{pred_glucose}超出合理范围")

        return {
            'status': 'pass' if not consistency_checks and not range_checks else 'fail',
            'consistency_issues': consistency_checks,
            'range_issues': range_checks,
            'reasonableness_score': 1.0 if not consistency_checks and not range_checks else 0.6
        }

    def _check_feedback_alignment(self,
                                 audit_record: Dict[str, Any]) -> Dict[str, Any]:
        """检查反馈对齐"""
        user_feedback = audit_record.get('user_feedback')
        output_result = audit_record['output_result']

        if user_feedback is None:
            return {
                'status': 'no_feedback',
                'alignment_score': 0.5
            }

        # 检查反馈与输出的对齐
        alignment_score = 0.5

        if 'satisfaction' in user_feedback:
            satisfaction = user_feedback['satisfaction']
            if satisfaction > 4:  # 高满意度
                alignment_score += 0.3
            elif satisfaction < 2:  # 低满意度
                alignment_score -= 0.3

        if 'effectiveness' in user_feedback:
            effectiveness = user_feedback['effectiveness']
            if effectiveness > 4:  # 高有效性
                alignment_score += 0.2
            elif effectiveness < 2:  # 低有效性
                alignment_score -= 0.2

        return {
            'status': 'pass' if alignment_score > 0.6 else 'fail',
            'alignment_score': max(0, min(1, alignment_score)),
            'feedback_quality': 'high' if alignment_score > 0.7 else 'medium' if alignment_score > 0.4 else 'low'
        }

    def _calculate_audit_score(self,
                              audit_results: Dict[str, Dict[str, Any]]) -> float:
        """计算审计分数"""
        scores = []
        for result in audit_results.values():
            if 'validation_score' in result:
                scores.append(result['validation_score'])
            elif 'consistency_score' in result:
                scores.append(result['consistency_score'])
            elif 'reasonableness_score' in result:
                scores.append(result['reasonableness_score'])
            elif 'alignment_score' in result:
                scores.append(result['alignment_score'])

        return np.mean(scores) if scores else 0.5

    def _detect_anomaly(self,
                        audit_record: Dict[str, Any]) -> float:
        """检测异常"""
        # 简化的异常检测
        features = []

        # 提取特征
        input_data = audit_record['input_data']
        features.extend([
            input_data.get('glucose', 0),
            input_data.get('blood_pressure_systolic', 0),
            input_data.get('heart_rate', 0)
        ])

        output_result = audit_record['output_result']
        features.extend([
            output_result.get('predicted_glucose', 0),
            output_result.get('confidence', 0)
        ])

        # 填充到256维
        while len(features) < 256:
            features.append(0.0)

        # 转换为张量
        feature_tensor = torch.tensor(features[:256], dtype=torch.float32).unsqueeze(0)

        # 计算异常分数
        with torch.no_grad():
            anomaly_score = self.anomaly_detector(feature_tensor).item()

        return anomaly_score

    def generate_audit_report(self,
                             start_time: Optional[datetime] = None,
                             end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """生成审计报告"""
        # 过滤审计记录
        filtered_records = self.audit_records

        if start_time:
            filtered_records = [record for record in filtered_records if record['timestamp'] >= start_time]

        if end_time:
            filtered_records = [record for record in filtered_records if record['timestamp'] <= end_time]

        if not filtered_records:
            return {'error': 'No audit records found'}

        # 统计信息
        total_decisions = len(filtered_records)
        audit_scores = [record['audit_score'] for record in filtered_records]
        anomaly_scores = [record['anomaly_score'] for record in filtered_records]

        # 审计结果统计
        audit_results_stats = {}
        for record in filtered_records:
            for rule_name, result in record['audit_results'].items():
                if rule_name not in audit_results_stats:
                    audit_results_stats[rule_name] = {'pass': 0, 'fail': 0, 'error': 0}

                status = result.get('status', 'unknown')
                if status in audit_results_stats[rule_name]:
                    audit_results_stats[rule_name][status] += 1

        return {
            'total_decisions': total_decisions,
            'audit_score_statistics': {
                'mean': np.mean(audit_scores),
                'std': np.std(audit_scores),
                'min': np.min(audit_scores),
                'max': np.max(audit_scores)
            },
            'anomaly_score_statistics': {
                'mean': np.mean(anomaly_scores),
                'std': np.std(anomaly_scores),
                'min': np.min(anomaly_scores),
                'max': np.max(anomaly_scores)
            },
            'audit_results_stats': audit_results_stats,
            'high_anomaly_decisions': [
                record['decision_id'] for record in filtered_records
                if record['anomaly_score'] > 0.8
            ],
            'low_audit_score_decisions': [
                record['decision_id'] for record in filtered_records
                if record['audit_score'] < 0.5
            ]
        }

    def trace_decision_history(self,
                               decision_id: str) -> Optional[Dict[str, Any]]:
        """追踪决策历史"""
        for record in self.audit_records:
            if record['decision_id'] == decision_id:
                return record

        return None

class ExplainabilitySystem(nn.Module):
    """
    可解释性系统
    整合所有可解释性模块
    """

    def __init__(self,
                 input_dim: int = 256,
                 hidden_dim: int = 128,
                 num_classes: int = 5,
                 num_decision_nodes: int = 10):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # 可解释AI模型
        self.explainable_model = ExplainableAIModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes
        )

        # 决策路径可视化器
        self.decision_visualizer = DecisionPathVisualizer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_decision_nodes=num_decision_nodes
        )

        # 决策审计系统
        self.audit_system = DecisionAuditSystem()

        # 解释生成器
        self.explanation_generator = nn.Sequential(
            nn.Linear(hidden_dim + 256, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 解释历史
        self.explanation_history = deque(maxlen=1000)

    def forward(self,
                x: torch.Tensor,
                decision_id: str = "decision_001",
                return_explanations: bool = True) -> Dict[str, Any]:
        """前向传播"""
        # 可解释AI模型预测
        explainable_output = self.explainable_model(x, return_explanations)

        # 决策路径可视化
        decision_path = self.decision_visualizer.visualize_decision_path(x, decision_id)

        # 生成综合解释
        if return_explanations:
            explanation_input = torch.cat([
                explainable_output['attended_features'].squeeze(1),
                x
            ], dim=-1)
            explanation_score = self.explanation_generator(explanation_input)
        else:
            explanation_score = torch.tensor(0.0)

        # 创建综合解释
        comprehensive_explanation = self._create_comprehensive_explanation(
            explainable_output, decision_path, explanation_score.item()
        )

        # 记录解释历史
        self._record_explanation(decision_id, comprehensive_explanation)

        return {
            'prediction': explainable_output['prediction'],
            'explainable_output': explainable_output,
            'decision_path': decision_path,
            'comprehensive_explanation': comprehensive_explanation,
            'explanation_score': explanation_score.item()
        }

    def _create_comprehensive_explanation(self,
                                         explainable_output: Dict[str, torch.Tensor],
                                         decision_path: Dict[str, Any],
                                         explanation_score: float) -> Dict[str, Any]:
        """创建综合解释"""
        # 提取关键信息
        feature_importance = explainable_output['feature_importance'].squeeze().numpy()
        attention_weights = explainable_output['attention_weights'].squeeze().detach().numpy()

        # 生成解释文本
        explanation_text = self._generate_explanation_text(
            feature_importance, attention_weights, decision_path
        )

        # 创建可视化数据
        visualization_data = self._create_visualization_data(
            feature_importance, attention_weights, decision_path
        )

        return {
            'explanation_text': explanation_text,
            'feature_importance': feature_importance.tolist(),
            'attention_weights': attention_weights.tolist(),
            'decision_path': decision_path,
            'visualization_data': visualization_data,
            'explanation_score': explanation_score,
            'confidence_level': self._calculate_confidence_level(explanation_score),
            'timestamp': datetime.now()
        }

    def _generate_explanation_text(self,
                                   feature_importance: np.ndarray,
                                   attention_weights: np.ndarray,
                                   decision_path: Dict[str, Any]) -> str:
        """生成解释文本"""
        # 找到最重要的特征
        top_features = np.argsort(feature_importance)[-5:]

        # 找到最重要的决策节点
        top_nodes = sorted(decision_path['decision_path'], key=lambda x: x['contribution'], reverse=True)[:3]

        explanation = f"""
        模型预测解释：

        1. 主要影响因素：
        """

        for i, feature_idx in enumerate(top_features):
            explanation += f"   - 特征{feature_idx}: 重要性 {feature_importance[feature_idx]:.3f}\n"

        explanation += f"""
        2. 决策路径：
        """

        for i, node in enumerate(top_nodes):
            explanation += f"   - {node['node_id']}: 贡献度 {node['contribution']:.3f}\n"

        explanation += f"""
        3. 最终决策: {decision_path['final_decision']:.3f}
        4. 解释置信度: {self._calculate_confidence_level(decision_path['final_decision'])}
        """

        return explanation

    def _create_visualization_data(self,
                                   feature_importance: np.ndarray,
                                   attention_weights: np.ndarray,
                                   decision_path: Dict[str, Any]) -> Dict[str, Any]:
        """创建可视化数据"""
        return {
            'feature_importance_chart': {
                'labels': [f'Feature_{i}' for i in range(len(feature_importance))],
                'values': feature_importance.tolist()
            },
            'attention_weights_chart': {
                'labels': [f'Attention_{i}' for i in range(len(attention_weights))],
                'values': attention_weights.tolist()
            },
            'decision_path_chart': {
                'nodes': [node['node_id'] for node in decision_path['decision_path']],
                'weights': [node['weight'] for node in decision_path['decision_path']],
                'contributions': [node['contribution'] for node in decision_path['decision_path']]
            }
        }

    def _calculate_confidence_level(self, score: float) -> str:
        """计算置信度级别"""
        if score > 0.8:
            return 'high'
        elif score > 0.6:
            return 'medium'
        elif score > 0.4:
            return 'low'
        else:
            return 'very_low'

    def _record_explanation(self,
                           decision_id: str,
                           explanation: Dict[str, Any]):
        """记录解释"""
        explanation_record = {
            'decision_id': decision_id,
            'explanation': explanation,
            'timestamp': datetime.now()
        }
        self.explanation_history.append(explanation_record)

    def audit_decision(self,
                       decision_id: str,
                       input_data: Dict[str, Any],
                       decision_process: Dict[str, Any],
                       output_result: Dict[str, Any],
                       user_feedback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """审计决策"""
        return self.audit_system.audit_decision(
            decision_id, input_data, decision_process, output_result, user_feedback
        )

    def get_explainability_report(self) -> Dict[str, Any]:
        """获取可解释性报告"""
        decision_stats = self.decision_visualizer.get_decision_statistics()
        audit_report = self.audit_system.generate_audit_report()

        return {
            'decision_statistics': decision_stats,
            'audit_report': audit_report,
            'explanation_history_count': len(self.explanation_history),
            'system_explainability_score': self._calculate_system_explainability_score()
        }

    def _calculate_system_explainability_score(self) -> float:
        """计算系统可解释性分数"""
        # 基于决策统计和审计报告计算
        decision_stats = self.decision_visualizer.get_decision_statistics()
        audit_report = self.audit_system.generate_audit_report()

        score = 0.5  # 基础分数

        # 基于决策统计
        if decision_stats and 'decision_statistics' in decision_stats:
            decision_stats_data = decision_stats['decision_statistics']
            if decision_stats_data['std'] < 0.2:  # 低标准差表示一致性高
                score += 0.2

        # 基于审计报告
        if audit_report and 'audit_score_statistics' in audit_report:
            audit_stats = audit_report['audit_score_statistics']
            if audit_stats['mean'] > 0.7:  # 高审计分数
                score += 0.3

        return min(1.0, score)

# 使用示例
def main():
    """使用示例"""
    # 创建可解释性系统
    system = ExplainabilitySystem()

    # 模拟输入数据
    input_data = torch.randn(1, 256)

    # 生成解释
    result = system.forward(input_data, "decision_001")

    print("预测结果:", result['prediction'])
    print("综合解释:", result['comprehensive_explanation'])
    print("可解释性报告:", system.get_explainability_report())

if __name__ == "__main__":
    main()

__all__ = ["'logger'", "'ExplainableAIModel'", "'DecisionPathVisualizer'", "'DecisionAuditSystem'", "'ExplainabilitySystem'", "'main'"]
