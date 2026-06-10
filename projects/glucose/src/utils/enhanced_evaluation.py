"""
增强评估系统
实现多指标评估和早停机制
"""

import json
import logging
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationMetrics:
    """评估指标集合（不可变数据结构）"""
    mae: float = 0.0
    mse: float = 0.0
    rmse: float = 0.0
    r2: float = 0.0
    overall_score: float = 0.0


class EarlyStoppingCallback:
    """早停回调"""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4,
                 restore_best_weights: bool = True, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        self.best_score: Optional[float] = None
        self.best_weights: Optional[Dict[str, torch.Tensor]] = None
        self.wait = 0
        self.history: deque = deque(maxlen=patience * 2)

    def __call__(self, current_score: float, model: nn.Module) -> bool:
        """检查是否应该早停"""
        self.history.append(current_score)

        if self.best_score is None:
            self.best_score = current_score
            if self.restore_best_weights:
                self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            return False

        if self._is_improvement(current_score):
            self.best_score = current_score
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.wait += 1

        if self.wait >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            logger.info(f"早停触发！最佳分数: {self.best_score:.6f}")
            return True
        return False

    def _is_improvement(self, current_score: float) -> bool:
        if self.mode == 'min':
            return current_score < (self.best_score - self.min_delta)
        return current_score > (self.best_score + self.min_delta)

    def get_best_score(self) -> float:
        return self.best_score if self.best_score is not None else float('inf')


class EnhancedEvaluator:
    """增强评估器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config: Dict[str, Any] = config or {'patience': 15, 'min_delta': 1e-4}
        self.evaluation_history: List[Dict] = []

    def create_early_stopping(self, patience: int = 10, mode: str = 'min') -> EarlyStoppingCallback:
        """创建早停回调"""
        return EarlyStoppingCallback(
            patience=patience,
            min_delta=self.config.get('min_delta', 1e-4),
            mode=mode
        )

    def evaluate_model(self, model: nn.Module, data_loader: torch.utils.data.DataLoader,
                      device: torch.device, epoch: int = 0) -> EvaluationMetrics:
        """评估模型"""
        model.eval()
        all_predictions: List[float] = []
        all_targets: List[float] = []
        all_losses: List[float] = []
        criterion = nn.MSELoss()

        with torch.no_grad():
            for batch in data_loader:
                if len(batch) == 4:
                    region_ids, cuisine_ids, preferences, targets = batch
                    region_ids = region_ids.to(device)
                    cuisine_ids = cuisine_ids.to(device)
                    preferences = preferences.to(device)
                    targets = targets.to(device)
                    predictions = model(region_ids, cuisine_ids, preferences)
                else:
                    inputs, targets = batch
                    if isinstance(inputs, dict):
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                    else:
                        inputs = inputs.to(device)
                    targets = targets.to(device)
                    predictions = model(inputs)

                loss = criterion(predictions, targets)
                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_targets.extend(targets.cpu().numpy().flatten())
                all_losses.append(loss.item())

        predictions_arr = np.array(all_predictions)
        targets_arr = np.array(all_targets)
        metrics = self._calculate_metrics(predictions_arr, targets_arr, all_losses)

        self.evaluation_history.append({'epoch': epoch, 'metrics': metrics})
        return metrics

    def _calculate_metrics(self, predictions: np.ndarray, targets: np.ndarray,
                           losses: List[float]) -> EvaluationMetrics:
        """计算评估指标"""
        mae = mean_absolute_error(targets, predictions)
        mse = mean_squared_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        mae_score = max(0.0, 1 - mae / 0.5)
        r2_score_val = max(0.0, r2)
        overall_score = 0.6 * r2_score_val + 0.4 * mae_score
        return EvaluationMetrics(
            mae=float(mae),
            mse=float(mse),
            rmse=float(np.sqrt(mse)),
            r2=float(r2),
            overall_score=float(overall_score)
        )

    def check_convergence(self, window_size: int = 5, threshold: float = 1e-4) -> bool:
        """检查模型是否收敛"""
        if len(self.evaluation_history) < window_size:
            return False
        recent_scores = [e['metrics'].overall_score for e in self.evaluation_history[-window_size:]]
        return np.std(recent_scores) < threshold

    def get_best_metrics(self) -> Optional[EvaluationMetrics]:
        """获取最佳指标"""
        if not self.evaluation_history:
            return None
        return max(self.evaluation_history, key=lambda x: x['metrics'].overall_score)['metrics']

    def save_evaluation_report(self, output_dir: Path, model_name: str = "model"):
        """保存评估报告"""
        output_dir.mkdir(parents=True, exist_ok=True)

        best = self.get_best_metrics()
        report = {
            'model_name': model_name,
            'total_epochs': len(self.evaluation_history),
            'best_metrics': asdict(best) if best else None,
            'converged': self.check_convergence(),
            'history': [
                {'epoch': e['epoch'], 'mae': float(e['metrics'].mae),
                 'r2': float(e['metrics'].r2), 'overall': float(e['metrics'].overall_score)}
                for e in self.evaluation_history
            ]
        }

        report_path = output_dir / f"{model_name}_evaluation_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"评估报告已保存: {report_path}")
