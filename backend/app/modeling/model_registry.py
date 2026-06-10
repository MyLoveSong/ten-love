"""model_registry模块\n\n模块描述\n"""

from typing import Callable, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import uuid
import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.metrics import accuracy_score, r2_score
    _SKLEARN_AVAILABLE = True
except Exception:
    _SKLEARN_AVAILABLE = False

_registry: Dict[str, Callable[..., Any]] = {}
_models: Dict[str, "TrainedModel"] = {}

@dataclass
class TrainedModel:
    model_id: str
    model_name: str
    task_type: str  # "classification" | "regression"
    estimator: Any
    metrics: Dict[str, Any]
    feature_names: Optional[list]

def register_model(name: str, builder: Callable[..., Any]) -> None:
    _registry[name] = builder

def get_model_builder(name: str) -> Optional[Callable[..., Any]]:
    return _registry.get(name)

def _to_xy(data: Dict[str, list], target: Optional[list]) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    X = pd.DataFrame(data)
    y = pd.Series(target) if target is not None else None
    return X, y

def train(model_name: str, data: Dict[str, list], target: list, params: Optional[Dict[str, Any]] = None) -> TrainedModel:
    if not _SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn 未安装")
    builder = get_model_builder(model_name)
    if builder is None:
        raise ValueError(f"未注册模型: {model_name}")
    params = params or {}
    estimator = builder(**params)
    X, y = _to_xy(data, target)
    estimator.fit(X, y)

    # 评估（粗略）
    try:
        if hasattr(estimator, "predict_proba") or hasattr(estimator, "predict"):
            y_pred = estimator.predict(X)
            if len(np.unique(y)) <= 20 and set(np.unique(y)).issubset({0, 1} | set(range(100))):
                metric_val = accuracy_score(y, y_pred)
                task_type = "classification"
                metrics = {"accuracy_train": float(metric_val)}
            else:
                metric_val = r2_score(y, y_pred)
                task_type = "regression"
                metrics = {"r2_train": float(metric_val)}
        else:
            task_type = "unknown"
            metrics = {}
    except Exception:
        task_type = "unknown"
        metrics = {}

    model_id = str(uuid.uuid4())
    tm = TrainedModel(
        model_id=model_id,
        model_name=model_name,
        task_type=task_type,
        estimator=estimator,
        metrics=metrics,
        feature_names=list(X.columns)
    )
    _models[model_id] = tm
    return tm

def predict(model_id: str, data: Dict[str, list]) -> Dict[str, Any]:
    if model_id not in _models:
        raise ValueError("模型不存在")
    tm = _models[model_id]
    X = pd.DataFrame(data)
    preds = tm.estimator.predict(X)
    return {"model_id": model_id, "predictions": preds.tolist()}

def explain(model_id: str) -> Dict[str, Any]:
    if model_id not in _models:
        raise ValueError("模型不存在")
    tm = _models[model_id]
    est = tm.estimator
    importance: Dict[str, float] = {}
    if hasattr(est, "feature_importances_"):
        vals = getattr(est, "feature_importances_")
        importance = {f: float(v) for f, v in zip(tm.feature_names or [], vals)}
    elif hasattr(est, "coef_"):
        coef = getattr(est, "coef_")
        coef_vec = coef[0] if getattr(coef, "ndim", 1) > 1 else coef
        importance = {f: float(v) for f, v in zip(tm.feature_names or [], coef_vec)}
    return {"model_id": model_id, "importance": importance}

# 预注册常用模型
def _lr(**kwargs):
    return LogisticRegression(max_iter=kwargs.pop("max_iter", 200), **kwargs)

def _rf_classifier(**kwargs):
    return RandomForestClassifier(**kwargs)

def _rf_regressor(**kwargs):
    return RandomForestRegressor(**kwargs)

def _svc(**kwargs):
    return SVC(probability=True, **kwargs)

def _svr(**kwargs):
    return SVR(**kwargs)

if _SKLEARN_AVAILABLE:
    register_model("logistic_regression", _lr)
    register_model("rf_classifier", _rf_classifier)
    register_model("rf_regressor", _rf_regressor)
    register_model("svc", _svc)
    register_model("svr", _svr)

__all__ = ["'TrainedModel'", "'register_model'", "'get_model_builder'", "'train'", "'predict'", "'explain'"]
