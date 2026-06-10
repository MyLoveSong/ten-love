"""experiment_runner模块\n\n模块描述\n"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from app.modeling.model_registry import train as train_model, predict as predict_model
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
)
from scipy import stats

# 导入增强的评估和统计模块
from app.experiments.comprehensive_evaluation_metrics import comprehensive_evaluator
from app.experiments.advanced_statistical_validation import statistical_validator
from app.experiments.enhanced_experiment_config import config_manager

@dataclass
class RunResult:
    model_id: str
    metrics: Dict[str, Any]

def enhanced_kfold_train(
    model_name: str,
    data: Dict[str, List[Any]],
    target: List[Any],
    k: int = 5,
    params: Optional[Dict[str, Any]] = None,
    experiment_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """增强的KFold训练，支持高级统计验证和配置管理"""
    from sklearn.model_selection import KFold

    X = pd.DataFrame(data)
    y = pd.Series(target)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    scores: List[float] = []
    detailed_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        tm = train_model(model_name, X_tr.to_dict(orient="list"), y_tr.tolist(), params)
        y_pred = tm.estimator.predict(X_te)

        # 使用综合评估器
        if _is_classification(y):
            metrics = comprehensive_evaluator.evaluate_classification(y_te, y_pred)
            scores.append(metrics.f1_score)  # 使用F1分数作为主要指标
        else:
            metrics = comprehensive_evaluator.evaluate_regression(y_te, y_pred)
            scores.append(metrics.r2)

        detailed_results.append({
            'fold': fold_idx + 1,
            'metrics': asdict(metrics),
            'predictions': y_pred.tolist(),
            'true_values': y_te.tolist()
        })

    # Bootstrap置信区间
    bootstrap_result = statistical_validator.bootstrap_confidence_interval(
        scores, np.mean, n_bootstrap=1000
    )

    # 创建实验配置记录
    if experiment_config:
        config = config_manager.create_experiment_config(
            experiment_name=f"KFold_{model_name}",
            description=f"KFold交叉验证实验，k={k}",
            author=experiment_config.get('author', 'system'),
            model_config={'name': model_name, 'params': params},
            data_config={'features': list(data.keys()), 'samples': len(target)},
            training_config={'k_folds': k, 'random_seed': 42},
            evaluation_config={'metric': 'f1_score' if _is_classification(y) else 'r2'}
        )
        experiment_id = config.experiment_id
    else:
        experiment_id = None

    return {
        "k": k,
        "scores": scores,
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "ci95": _ci95(scores),
        "bootstrap_ci": bootstrap_result.confidence_interval,
        "bootstrap_bias": bootstrap_result.bias,
        "detailed_results": detailed_results,
        "experiment_id": experiment_id
    }

def multi_run(model_name: str, data: Dict[str, List[Any]], target: List[Any], runs: int = 5, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    X = pd.DataFrame(data)
    y = pd.Series(target)
    scores: List[float] = []
    for _ in range(max(1, runs)):
        tm = train_model(model_name, X.to_dict(orient="list"), y.tolist(), params)
        y_pred = tm.estimator.predict(X)
        if _is_classification(y):
            scores.append(float(accuracy_score(y, y_pred)))
        else:
            scores.append(float(r2_score(y, y_pred)))
    details = {}
    # 附加常用指标（单次全量评估，作为参考）
    try:
        if _is_classification(y):
            details = _classification_metrics(y, tm.estimator.predict(X), y)
        else:
            details = _regression_metrics(y, tm.estimator.predict(X), y)
    except Exception:
        pass
    return {
        "runs": runs,
        "scores": scores,
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "ci95": _ci95(scores),
        "metrics_sample": details,
    }

def enhanced_compare_models(
    model_names: List[str],
    data: Dict[str, List[Any]],
    target: List[Any],
    runs: int = 5,
    experiment_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """增强的模型比较，支持高级统计验证"""
    results: Dict[str, Any] = {}

    # 运行多次实验
    for name in model_names:
        mr = multi_run(name, data, target, runs=runs)
        results[name] = mr

    # 选最优
    best = max(results.items(), key=lambda kv: kv[1]["mean"]) if results else None

    if not best:
        return {"results": results, "best": None, "significance": {}}

    best_name, best_val = best
    best_scores = best_val.get("scores", [])

    # 使用高级统计验证
    model_scores = {name: val.get("scores", []) for name, val in results.items()}
    statistical_comparison = statistical_validator.model_comparison_statistical_test(
        model_scores, test_type="permutation"
    )

    # 创建实验配置记录
    if experiment_config:
        config = config_manager.create_experiment_config(
            experiment_name=f"Model_Comparison_{len(model_names)}_models",
            description=f"比较{len(model_names)}个模型的性能",
            author=experiment_config.get('author', 'system'),
            model_config={'models': model_names, 'runs': runs},
            data_config={'features': list(data.keys()), 'samples': len(target)},
            training_config={'multi_run': runs},
            evaluation_config={'comparison_method': 'permutation_test'}
        )
        experiment_id = config.experiment_id
    else:
        experiment_id = None

    return {
        "results": results,
        "best": best_name,
        "statistical_comparison": statistical_comparison,
        "experiment_id": experiment_id
    }

# 保持向后兼容性
def kfold_train(model_name: str, data: Dict[str, List[Any]], target: List[Any], k: int = 5, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """向后兼容的KFold训练"""
    return enhanced_kfold_train(model_name, data, target, k, params)

def compare_models(model_names: List[str], data: Dict[str, List[Any]], target: List[Any]) -> Dict[str, Any]:
    """向后兼容的模型比较"""
    return enhanced_compare_models(model_names, data, target)

def _is_classification(y: pd.Series) -> bool:
    uniq = np.unique(y)
    return (len(uniq) <= 20) and (set(uniq).issubset({0, 1} | set(range(100))))

def _classification_metrics(y_true: pd.Series, y_pred: np.ndarray, y_for_labels: pd.Series) -> Dict[str, Any]:
    avg = "binary" if set(np.unique(y_for_labels)).issubset({0, 1}) else "macro"
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
    }
    try:
        if avg == "binary":
            # 若可得概率，计算AUC，回退为 None
            out["roc_auc"] = None
    except Exception:
        pass
    return out

def _regression_metrics(y_true: pd.Series, y_pred: np.ndarray, _: pd.Series) -> Dict[str, Any]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    return {
        "mse": float(mse),
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }

def _ci95(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=float)
    if arr.size < 2:
        return {"low": float(arr.mean() if arr.size else 0.0), "high": float(arr.mean() if arr.size else 0.0)}
    mean = arr.mean()
    std = arr.std(ddof=1)
    se = std / np.sqrt(arr.size)
    margin = 1.96 * se
    return {"low": float(mean - margin), "high": float(mean + margin)}

__all__ = ["'RunResult'", "'enhanced_kfold_train'", "'multi_run'", "'enhanced_compare_models'", "'kfold_train'", "'compare_models'"]
