"""
GlucoNet-MM集成服务
将GlucoNet-MM架构与现有糖尿病管理系统深度整合
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from datetime import datetime, timedelta
import json
from dataclasses import dataclass, asdict

# 导入GlucoNet-MM核心模块
from app.models.gluconet import (
    GlucoNet, GlucoNetConfig, create_gluconet_model,
    ModalityType, TaskType
)

# 导入现有系统模块
from app.models.ssa_optimizer import SSA, SSACallback
from app.models.fd_encoder import FDEcomposer
from app.models.kd_distillation import EdgeStudent, KnowledgeDistillationTrainer
from app.models.interpretability import InterpretabilityAnalyzer

logger = logging.getLogger(__name__)

@dataclass
class GlucoNetMMInput:
    """GlucoNet-MM输入数据结构"""
    cgm_data: torch.Tensor
    insulin_data: Optional[torch.Tensor] = None
    diet_data: Optional[torch.Tensor] = None
    exercise_data: Optional[torch.Tensor] = None
    image_data: Optional[torch.Tensor] = None
    text_data: Optional[torch.Tensor] = None
    clinical_strategy_id: Optional[torch.Tensor] = None
    patient_profile: Optional[torch.Tensor] = None

@dataclass
class GlucoNetMMOutput:
    """GlucoNet-MM输出数据结构"""
    glucose_prediction: float
    insulin_sensitivity: float
    carb_estimation: float
    exercise_effect: float
    task_weights: Dict[str, float]
    temporal_modes: Dict[str, torch.Tensor]
    spatial_modes: Dict[str, torch.Tensor]
    attention_weights: Dict[str, torch.Tensor]
    strategy_weights: Optional[torch.Tensor]
    distillation_loss: float
    interpretability_score: float
    clinical_insights: List[str]

class GlucoNetMMOptimizer:
    """GlucoNet-MM优化器"""

    def __init__(self, config: GlucoNetMMConfig):
        self.config = config
        self.optimization_history = []

    def optimize_hyperparameters(self,
                                model: GlucoNetMM,
                                train_data: List[GlucoNetMMInput],
                                val_data: List[GlucoNetMMInput]) -> Dict[str, Any]:
        """使用SSA优化超参数"""

        def objective_function(params: np.ndarray) -> float:
            """目标函数：验证集MAE"""
            # 更新模型参数
            self._update_model_parameters(model, params)

            # 在验证集上评估
            total_loss = 0.0
            for data in val_data:
                with torch.no_grad():
                    results = model(
                        cgm_data=data.cgm_data,
                        insulin_data=data.insulin_data,
                        diet_data=data.diet_data,
                        exercise_data=data.exercise_data,
                        image_data=data.image_data,
                        text_data=data.text_data,
                        clinical_strategy_id=data.clinical_strategy_id,
                        patient_profile=data.patient_profile
                    )

                    # 计算血糖预测MAE
                    glucose_pred = results['task_outputs']['glucose_prediction']
                    # 这里需要真实标签，暂时用随机值
                    true_glucose = torch.randn_like(glucose_pred)
                    mae = torch.mean(torch.abs(glucose_pred - true_glucose))
                    total_loss += mae.item()

            return total_loss / len(val_data)

        # SSA优化
        ssa_config = {
            'dim': 8,  # 优化8个超参数
            'pop': 20,
            'max_iter': 50,
            'lb': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            'ub': [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
        }

        ssa = SSA(ssa_config, objective_function)
        best_params, best_fitness, optimization_info = ssa.optimize()

        # 记录优化历史
        self.optimization_history.append({
            'timestamp': datetime.now().isoformat(),
            'best_params': best_params.tolist(),
            'best_fitness': best_fitness,
            'convergence_curve': optimization_info.get('convergence_curve', [])
        })

        return {
            'best_params': best_params,
            'best_fitness': best_fitness,
            'optimization_info': optimization_info
        }

    def _update_model_parameters(self, model: GlucoNetMM, params: np.ndarray):
        """更新模型参数"""
        # 这里可以根据params更新模型的超参数
        # 例如学习率、dropout率等
        pass

class GlucoNetMMService:
    """GlucoNet-MM集成服务"""

    def __init__(self,
                 config: Optional[GlucoNetConfig] = None,
                 use_ssa_optimization: bool = True,
                 use_knowledge_distillation: bool = True,
                 use_interpretability: bool = True):

        # 初始化配置
        self.config = config or GlucoNetConfig()

        # 初始化模型
        self.model = create_gluconet_model(self.config)

        # 初始化优化器
        if use_ssa_optimization:
            self.optimizer = GlucoNetMMOptimizer(self.config)
        else:
            self.optimizer = None

        # 初始化知识蒸馏
        if use_knowledge_distillation:
            self.kd_trainer = KnowledgeDistillationTrainer(
                teacher_model=self.model,
                student_hidden_dim=self.config.student_hidden_dim,
                temperature=self.config.temperature,
                alpha=self.config.alpha
            )
        else:
            self.kd_trainer = None

        # 初始化可解释性分析
        if use_interpretability:
            self.interpretability_analyzer = InterpretabilityAnalyzer(
                model=self.model,
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            )
        else:
            self.interpretability_analyzer = None

        logger.info("GlucoNet-MM集成服务初始化完成")

    def predict(self, input_data: GlucoNetMMInput) -> GlucoNetMMOutput:
        """预测接口"""
        self.model.eval()

        with torch.no_grad():
            results = self.model(
                cgm_data=input_data.cgm_data,
                insulin_data=input_data.insulin_data,
                diet_data=input_data.diet_data,
                exercise_data=input_data.exercise_data,
                image_data=input_data.image_data,
                text_data=input_data.text_data,
                clinical_strategy_id=input_data.clinical_strategy_id,
                patient_profile=input_data.patient_profile
            )

        # 提取任务输出
        task_outputs = results['task_outputs']

        # 计算可解释性分数
        interpretability_score = 0.0
        clinical_insights = []

        if self.interpretability_analyzer:
            try:
                # 生成归因热图
                heatmaps = self.interpretability_analyzer.generate_attribution_heatmap(
                    input_data.cgm_data[:, :, 0]  # 取第一个特征
                )

                # 计算可解释性分数
                interpretability_score = self._calculate_interpretability_score(heatmaps)

                # 生成临床洞察
                clinical_insights = self._generate_clinical_insights(
                    results['temporal_modes'],
                    results['spatial_modes'],
                    task_outputs
                )

            except Exception as e:
                logger.warning(f"可解释性分析失败: {e}")

        # 构建输出
        output = GlucoNetMMOutput(
            glucose_prediction=task_outputs['glucose_prediction'].item(),
            insulin_sensitivity=task_outputs['insulin_sensitivity'].item(),
            carb_estimation=task_outputs['carb_estimation'].item(),
            exercise_effect=task_outputs['exercise_effect'].item(),
            task_weights={
                'glucose_prediction': task_outputs['glucose_prediction_weight'].item(),
                'insulin_sensitivity': task_outputs['insulin_sensitivity_weight'].item(),
                'carb_estimation': task_outputs['carb_estimation_weight'].item(),
                'exercise_effect': task_outputs['exercise_effect_weight'].item()
            },
            temporal_modes=results['temporal_modes'],
            spatial_modes=results['spatial_modes'],
            attention_weights=results['attention_weights'],
            strategy_weights=results['strategy_weights'],
            distillation_loss=results['distillation_loss'].item(),
            interpretability_score=interpretability_score,
            clinical_insights=clinical_insights
        )

        return output

    def train(self,
              train_data: List[GlucoNetMMInput],
              val_data: List[GlucoNetMMInput],
              epochs: int = 100,
              learning_rate: float = 0.001) -> Dict[str, Any]:
        """训练接口"""

        # 设置优化器
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        # 训练历史
        train_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'glucose_mae': [],
            'insulin_mae': [],
            'carb_mae': [],
            'exercise_mae': []
        }

        best_val_loss = float('inf')

        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0

            for data in train_data:
                optimizer.zero_grad()

                results = self.model(
                    cgm_data=data.cgm_data,
                    insulin_data=data.insulin_data,
                    diet_data=data.diet_data,
                    exercise_data=data.exercise_data,
                    image_data=data.image_data,
                    text_data=data.text_data,
                    clinical_strategy_id=data.clinical_strategy_id,
                    patient_profile=data.patient_profile
                )

                # 计算损失
                loss = self._calculate_total_loss(results, data)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            val_metrics = {'glucose_mae': 0.0, 'insulin_mae': 0.0, 'carb_mae': 0.0, 'exercise_mae': 0.0}

            with torch.no_grad():
                for data in val_data:
                    results = self.model(
                        cgm_data=data.cgm_data,
                        insulin_data=data.insulin_data,
                        diet_data=data.diet_data,
                        exercise_data=data.exercise_data,
                        image_data=data.image_data,
                        text_data=data.text_data,
                        clinical_strategy_id=data.clinical_strategy_id,
                        patient_profile=data.patient_profile
                    )

                    loss = self._calculate_total_loss(results, data)
                    val_loss += loss.item()

                    # 计算各任务指标
                    task_outputs = results['task_outputs']
                    for task_name in ['glucose_prediction', 'insulin_sensitivity', 'carb_estimation', 'exercise_effect']:
                        if task_name in task_outputs:
                            # 这里需要真实标签，暂时用随机值
                            true_values = torch.randn_like(task_outputs[task_name])
                            mae = torch.mean(torch.abs(task_outputs[task_name] - true_values))
                            val_metrics[f'{task_name.split("_")[0]}_mae'] += mae.item()

            # 记录历史
            avg_train_loss = train_loss / len(train_data)
            avg_val_loss = val_loss / len(val_data)

            train_history['epoch'].append(epoch)
            train_history['train_loss'].append(avg_train_loss)
            train_history['val_loss'].append(avg_val_loss)
            train_history['glucose_mae'].append(val_metrics['glucose_mae'] / len(val_data))
            train_history['insulin_mae'].append(val_metrics['insulin_mae'] / len(val_data))
            train_history['carb_mae'].append(val_metrics['carb_mae'] / len(val_data))
            train_history['exercise_mae'].append(val_metrics['exercise_mae'] / len(val_data))

            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'best_gluconet_mm_model.pth')

            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

        return train_history

    def optimize_hyperparameters(self,
                                train_data: List[GlucoNetMMInput],
                                val_data: List[GlucoNetMMInput]) -> Dict[str, Any]:
        """超参数优化"""
        if self.optimizer is None:
            raise ValueError("优化器未初始化")

        return self.optimizer.optimize_hyperparameters(
            self.model, train_data, val_data
        )

    def export_edge_model(self, output_path: str) -> Dict[str, Any]:
        """导出边缘模型"""
        if self.kd_trainer is None:
            raise ValueError("知识蒸馏训练器未初始化")

        # 导出学生模型
        edge_model_info = self.kd_trainer.export_student_model(output_path)

        return edge_model_info

    def generate_clinical_report(self,
                               input_data: GlucoNetMMInput,
                               output_path: Optional[str] = None) -> Dict[str, Any]:
        """生成临床报告"""

        # 获取预测结果
        prediction = self.predict(input_data)

        # 生成报告
        report = {
            'timestamp': datetime.now().isoformat(),
            'predictions': {
                'glucose_prediction': prediction.glucose_prediction,
                'insulin_sensitivity': prediction.insulin_sensitivity,
                'carb_estimation': prediction.carb_estimation,
                'exercise_effect': prediction.exercise_effect
            },
            'task_weights': prediction.task_weights,
            'interpretability_score': prediction.interpretability_score,
            'clinical_insights': prediction.clinical_insights,
            'risk_assessment': self._assess_risk_level(prediction.glucose_prediction),
            'recommendations': self._generate_recommendations(prediction)
        }

        # 保存报告
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

        return report

    def _calculate_total_loss(self, results: Dict[str, Any], data: GlucoNetMMInput) -> torch.Tensor:
        """计算总损失"""
        task_outputs = results['task_outputs']
        distillation_loss = results['distillation_loss']

        # 任务损失（这里需要真实标签，暂时用随机值）
        task_loss = 0.0
        for task_name in ['glucose_prediction', 'insulin_sensitivity', 'carb_estimation', 'exercise_effect']:
            if task_name in task_outputs:
                true_values = torch.randn_like(task_outputs[task_name])
                mse_loss = F.mse_loss(task_outputs[task_name], true_values)
                task_loss += mse_loss

        # 总损失
        total_loss = task_loss + 0.1 * distillation_loss

        return total_loss

    def _calculate_interpretability_score(self, heatmaps: Dict[str, np.ndarray]) -> float:
        """计算可解释性分数"""
        # 基于热图的方差计算可解释性分数
        scores = []
        for mode, heatmap in heatmaps.items():
            variance = np.var(heatmap)
            scores.append(variance)

        return np.mean(scores)

    def _generate_clinical_insights(self,
                                  temporal_modes: Dict[str, torch.Tensor],
                                  spatial_modes: Dict[str, torch.Tensor],
                                  task_outputs: Dict[str, torch.Tensor]) -> List[str]:
        """生成临床洞察"""
        insights = []

        # 基于时序模态的洞察
        if 'trend' in temporal_modes:
            trend_mean = temporal_modes['trend'].mean().item()
            if trend_mean > 0.5:
                insights.append("血糖呈上升趋势，建议调整饮食和运动计划")
            elif trend_mean < -0.5:
                insights.append("血糖呈下降趋势，注意低血糖风险")
            else:
                insights.append("血糖变化相对平稳，继续保持当前管理方案")

        # 基于空间模态的洞察
        if 'physiological' in spatial_modes:
            phys_mean = spatial_modes['physiological'].mean().item()
            if phys_mean > 0.7:
                insights.append("生理指标良好，代谢状态稳定")
            else:
                insights.append("生理指标需要关注，建议咨询医生")

        return insights

    def _assess_risk_level(self, glucose_prediction: float) -> str:
        """评估风险等级"""
        if glucose_prediction < 3.9:
            return "低血糖风险"
        elif glucose_prediction > 10.0:
            return "高血糖风险"
        elif glucose_prediction > 7.8:
            return "血糖偏高"
        else:
            return "血糖正常"

    def _generate_recommendations(self, prediction: GlucoNetMMOutput) -> List[str]:
        """生成建议"""
        recommendations = []

        # 基于血糖预测的建议
        if prediction.glucose_prediction > 7.8:
            recommendations.append("建议减少碳水化合物摄入")
            recommendations.append("增加餐后运动")
        elif prediction.glucose_prediction < 3.9:
            recommendations.append("注意低血糖风险，准备糖类食物")

        # 基于胰岛素敏感性的建议
        if prediction.insulin_sensitivity < 0.5:
            recommendations.append("胰岛素敏感性较低，建议增加运动")

        # 基于运动效果的建议
        if prediction.exercise_effect > 0.7:
            recommendations.append("运动效果良好，继续保持")
        else:
            recommendations.append("建议调整运动强度和时长")

        return recommendations

def create_gluconet_mm_service(config: Optional[GlucoNetConfig] = None,
                            use_ssa_optimization: bool = True,
                            use_knowledge_distillation: bool = True,
                            use_interpretability: bool = True) -> GlucoNetMMService:
    """创建GlucoNet-MM服务"""
    return GlucoNetMMService(
        config=config,
        use_ssa_optimization=use_ssa_optimization,
        use_knowledge_distillation=use_knowledge_distillation,
        use_interpretability=use_interpretability
    )

# 导出主要类和函数
__all__ = [
    'GlucoNetMMService',
    'GlucoNetMMOptimizer',
    'GlucoNetMMInput',
    'GlucoNetMMOutput',
    'create_gluconet_mm_service'
]
