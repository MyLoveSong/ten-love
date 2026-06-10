#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
外部验证与多基线评估脚本（增量实现，独立可运行）

功能：
- 载入“论文版”数据切分（train/val/test/external）CSV
- 训练基线模型：LinearRegression、GradientBoosting（优先XGBoost/LightGBM，退化到sklearn GBDT）、MLP
- 评估：MAE/MSE/RMSE/ECE/Brier/Acc(阈值0.5) + 95%CI（自助法）
- 可选：导出校准曲线与误差分布图
- 输出：JSON报告 + 可选LaTeX表 + PNG图
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor

try:
    import xgboost as xgb  # type: ignore
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import lightgbm as lgb  # type: ignore
    HAS_LGB = True
except Exception:
    HAS_LGB = False

from analysis.source_aware_split_dataset import load_source_aware_split_windows

try:
    from .utils.metrics_extra import compute_ece, compute_brier
except ImportError:  # pragma: no cover - direct script execution
    from utils.metrics_extra import compute_ece, compute_brier


logger = logging.getLogger("external_eval")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_split(csv_path: Path, feature_cols: List[str], target_col: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    X = df[feature_cols].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)
    return X, y


def bootstrap_ci(values: np.ndarray, alpha: float = 0.05, n_boot: int = 2000, seed: int = 42) -> Tuple[float, float]:
    rng = np.random.RandomState(seed)
    n = len(values)
    samples = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        samples.append(np.mean(values[idx]))
    q_low = np.quantile(samples, alpha / 2)
    q_high = np.quantile(samples, 1 - alpha / 2)
    return float(q_low), float(q_high)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.reshape(-1)
    y_pred = np.clip(y_pred.reshape(-1), 0.0, 1.0)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    ece, _ = compute_ece(y_true, y_pred)
    brier = compute_brier(y_true, y_pred)
    acc = float(np.mean((y_pred >= 0.5) == (y_true >= 0.5)))
    return {"mae": float(mae), "mse": float(mse), "rmse": rmse, "ece": float(ece), "brier": float(brier), "accuracy": acc}


def evaluate_glucose_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    diff = y_true - y_pred
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    per_horizon = {}
    if y_true.ndim == 2:
        for idx in range(y_true.shape[1]):
            step_diff = y_true[:, idx] - y_pred[:, idx]
            step_mse = float(np.mean(step_diff ** 2))
            per_horizon[f"t+{idx + 1}"] = {
                "mae": float(np.mean(np.abs(step_diff))),
                "rmse": float(np.sqrt(step_mse)),
            }
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "per_horizon": per_horizon,
    }


def fit_linear(X_tr: np.ndarray, y_tr: np.ndarray):
    model = LinearRegression()
    model.fit(X_tr, y_tr)
    return model


def fit_gbm(X_tr: np.ndarray, y_tr: np.ndarray):
    if HAS_XGB:
        model = xgb.XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=42
        )
        if y_tr.ndim > 1 and y_tr.shape[1] > 1:
            model = MultiOutputRegressor(model)
        model.fit(X_tr, y_tr)
        return model
    if HAS_LGB:
        model = lgb.LGBMRegressor(
            n_estimators=600, num_leaves=63, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=42
        )
        if y_tr.ndim > 1 and y_tr.shape[1] > 1:
            model = MultiOutputRegressor(model)
        model.fit(X_tr, y_tr)
        return model
    # fallback: sklearn GradientBoostingRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    model = GradientBoostingRegressor(random_state=42)
    if y_tr.ndim > 1 and y_tr.shape[1] > 1:
        model = MultiOutputRegressor(model)
    model.fit(X_tr, y_tr)
    return model


def fit_mlp(X_tr: np.ndarray, y_tr: np.ndarray):
    model = MLPRegressor(hidden_layer_sizes=(128, 64), activation="relu", solver="adam", alpha=1e-4, batch_size=256,
                         learning_rate_init=1e-3, max_iter=300, random_state=42, early_stopping=True, n_iter_no_change=15)
    model.fit(X_tr, y_tr)
    return model


def flatten_sequences(sequences: np.ndarray) -> np.ndarray:
    return sequences.reshape(sequences.shape[0], -1)


def inverse_scale(values: np.ndarray, scaler: Dict[str, float]) -> np.ndarray:
    return values * float(scaler["std"]) + float(scaler["mean"])


def limit_windows(values: np.ndarray, max_windows: int | None) -> np.ndarray:
    if max_windows is None:
        return values
    if max_windows <= 0:
        raise ValueError("--max-windows-per-split must be positive")
    return values[:max_windows]


def parse_model_list(models: List[str] | None) -> List[str]:
    selected = models or ["persistence", "linear", "gbm", "mlp"]
    normalized = [model.strip().lower() for model in selected if model.strip()]
    valid = {"persistence", "linear", "gbm", "mlp"}
    invalid = sorted(set(normalized) - valid)
    if invalid:
        raise ValueError(f"Unsupported baseline model(s): {', '.join(invalid)}")
    if not normalized:
        raise ValueError("At least one baseline model is required")
    return normalized


def predict_persistence(X: np.ndarray, output_horizon: int) -> np.ndarray:
    last_observed = X[:, -1:]
    return np.repeat(last_observed, output_horizon, axis=1)


def run_split_manifest_baselines(
    dataset_path: Path,
    manifest_path: Path,
    output_dir: Path,
    input_horizon: int,
    output_horizon: int,
    models: List[str] | None = None,
    max_windows_per_split: int | None = None,
) -> Dict[str, object]:
    split_data = load_source_aware_split_windows(
        dataset_path,
        manifest_path,
        input_horizon=input_horizon,
        output_horizon=output_horizon,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    X_tr = limit_windows(flatten_sequences(np.asarray(split_data["sequences"]["train"], dtype=np.float32)), max_windows_per_split)
    y_tr = limit_windows(np.asarray(split_data["targets"]["train"], dtype=np.float32), max_windows_per_split)
    X_va = limit_windows(flatten_sequences(np.asarray(split_data["sequences"]["val"], dtype=np.float32)), max_windows_per_split)
    y_va = limit_windows(np.asarray(split_data["targets"]["val"], dtype=np.float32), max_windows_per_split)
    X_te = limit_windows(flatten_sequences(np.asarray(split_data["sequences"]["test"], dtype=np.float32)), max_windows_per_split)
    y_te = limit_windows(np.asarray(split_data["targets"]["test"], dtype=np.float32), max_windows_per_split)

    selected_models = parse_model_list(models)
    baselines = {}
    if "persistence" in selected_models:
        baselines["persistence"] = lambda X: predict_persistence(X, output_horizon)
    if "linear" in selected_models:
        logger.info("训练 split-manifest LinearRegression ...")
        baselines["linear"] = fit_linear(X_tr, y_tr).predict
    if "gbm" in selected_models:
        logger.info("训练 split-manifest GBM ...")
        baselines["gbm"] = fit_gbm(X_tr, y_tr).predict
    if "mlp" in selected_models:
        logger.info("训练 split-manifest MLPRegressor ...")
        baselines["mlp"] = fit_mlp(X_tr, y_tr).predict

    def eval_on_split(X: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, object]]:
        out: Dict[str, Dict[str, object]] = {}
        y_orig = inverse_scale(y, split_data["scaler"])
        for name, pred_fn in baselines.items():
            yhat = pred_fn(X)
            yhat_orig = inverse_scale(np.asarray(yhat), split_data["scaler"])
            out[name] = evaluate_glucose_predictions(y_orig, yhat_orig)
        return out

    results: Dict[str, object] = {
        "mode": "source_aware_split_manifest",
        "dataset": str(dataset_path),
        "split_manifest": str(manifest_path),
        "evaluation_scope": "smoke_subset" if max_windows_per_split is not None else "full_split",
        "max_windows_per_split": max_windows_per_split,
        "models": selected_models,
        "split_metadata": split_data["metadata"],
        "scaler": split_data["scaler"],
        "val": eval_on_split(X_va, y_va),
        "test": eval_on_split(X_te, y_te),
    }
    report_path = output_dir / "split_manifest_baseline_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"split-manifest baseline 报告已保存: {report_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="外部验证与多基线评估")
    parser.add_argument("--data-root", type=str, default=None, help="论文版数据根目录，包含 train.csv/val.csv/test.csv/external.csv")
    parser.add_argument("--features", type=str, default=None, help="以逗号分隔的特征列名")
    parser.add_argument("--target", type=str, default="label", help="目标列名，默认 label")
    parser.add_argument("--output", type=str, default="outputs/external_validation", help="输出目录")
    parser.add_argument("--plots", action="store_true", help="是否导出校准/误差图")
    parser.add_argument("--split-manifest", type=str, default=None, help="Source-aware split manifest JSON")
    parser.add_argument("--split-dataset", type=str, default=None, help="Dataset JSON matching --split-manifest")
    parser.add_argument("--input-horizon", type=int, default=12)
    parser.add_argument("--output-horizon", type=int, default=6)
    parser.add_argument("--models", type=str, default=None, help="Comma-separated baselines: persistence,linear,gbm,mlp")
    parser.add_argument("--max-windows-per-split", type=int, default=None, help="Deterministic smoke limit per split")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.split_manifest:
        if not args.split_dataset:
            parser.error("--split-manifest requires --split-dataset")
        run_split_manifest_baselines(
            Path(args.split_dataset),
            Path(args.split_manifest),
            output_dir,
            args.input_horizon,
            args.output_horizon,
            models=args.models.split(",") if args.models else None,
            max_windows_per_split=args.max_windows_per_split,
        )
        return

    if not args.data_root or not args.features:
        parser.error("--data-root and --features are required unless --split-manifest is provided")

    data_root = Path(args.data_root)

    feature_cols = [c.strip() for c in args.features.split(",") if c.strip()]
    target_col = args.target

    # 加载数据切分
    X_tr, y_tr = load_split(data_root / "train.csv", feature_cols, target_col)
    X_va, y_va = load_split(data_root / "val.csv", feature_cols, target_col)
    X_te, y_te = load_split(data_root / "test.csv", feature_cols, target_col)
    X_ex, y_ex = load_split(data_root / "external.csv", feature_cols, target_col)

    # 训练基线
    logger.info("训练 LinearRegression ...")
    m_lin = fit_linear(X_tr, y_tr)
    logger.info("训练 GBM (XGBoost/LightGBM/fallback) ...")
    m_gbm = fit_gbm(X_tr, y_tr)
    logger.info("训练 MLPRegressor ...")
    m_mlp = fit_mlp(X_tr, y_tr)

    baselines = {
        "linear": m_lin.predict,
        "gbm": m_gbm.predict,
        "mlp": m_mlp.predict,
    }

    def eval_on_split(X: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for name, pred_fn in baselines.items():
            yhat = pred_fn(X)
            metrics = evaluate_predictions(y, yhat)
            out[name] = metrics
        return out

    results = {
        "val": eval_on_split(X_va, y_va),
        "test": eval_on_split(X_te, y_te),
        "external": eval_on_split(X_ex, y_ex),
    }

    # 导出 JSON 报告
    report_path = output_dir / "external_validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"✅ 报告已保存: {report_path}")

    # 可选：绘图
    if args.plots:
        try:
            import matplotlib.pyplot as plt
            for split_name, split_res in results.items():
                # 校准曲线：以gbm为例
                if "gbm" in split_res:
                    yhat = baselines["gbm"](X_ex if split_name == "external" else (X_te if split_name == "test" else X_va))
                    ytrue = y_ex if split_name == "external" else (y_te if split_name == "test" else y_va)
                    _, stats = compute_ece(ytrue, yhat)
                    plt.figure(figsize=(4, 4))
                    plt.plot([0, 1], [0, 1], "k--", label="perfect")
                    plt.scatter(stats["bin_confidence"], stats["bin_empirical"], s=40)
                    plt.xlabel("Predicted confidence")
                    plt.ylabel("Empirical")
                    plt.title(f"Calibration ({split_name})")
                    plt.tight_layout()
                    plt.savefig(output_dir / f"calibration_{split_name}.png", dpi=200)
                    plt.close()
        except Exception as e:
            logger.warning(f"绘图失败: {e}")


if __name__ == "__main__":
    main()
