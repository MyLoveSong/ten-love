"""Sync Stage2 glucose training stats into the shared evaluation summary."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _load_training_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"未找到训练结果文件: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"训练结果文件格式无效: {path}")
    return data


def _load_evaluation_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"评估文件格式无效: {path}")
    return data


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _last_history_value(history: Dict[str, Any], key: str) -> Optional[float]:
    values = history.get(key)
    if isinstance(values, list) and values:
        return _safe_float(values[-1])
    return None


def _best_epoch(training_data: Dict[str, Any], history: Dict[str, Any]) -> Optional[int]:
    if isinstance(training_data.get("best_epoch"), (int, float)):
        return int(training_data["best_epoch"])
    epochs = history.get("epochs")
    val_loss = history.get("val_loss")
    if isinstance(val_loss, list) and val_loss:
        best_idx = min(range(len(val_loss)), key=lambda idx: val_loss[idx])
        if isinstance(epochs, list) and len(epochs) == len(val_loss):
            try:
                return int(epochs[best_idx])
            except (TypeError, ValueError):
                return best_idx + 1
        return best_idx + 1
    return None


def _build_training_snapshot(training_data: Dict[str, Any], source: Path) -> Dict[str, Any]:
    history = training_data.get("training_history", {})
    total_epochs = training_data.get("total_epochs")
    if not isinstance(total_epochs, int):
        epochs = history.get("epochs")
        if isinstance(epochs, list) and epochs:
            try:
                total_epochs = int(epochs[-1])
            except (TypeError, ValueError):
                total_epochs = len(epochs)
        else:
            total_epochs = 0

    snapshot: Dict[str, Any] = {
        "best_val_loss": _safe_float(training_data.get("best_val_loss")),
        "best_epoch": _best_epoch(training_data, history) or 0,
        "total_epochs": int(total_epochs),
        "source": str(source).replace("\\", "/"),
        "synced_at": datetime.now(timezone.utc).astimezone().isoformat(),
    }

    last_train_loss = _last_history_value(history, "train_loss")
    last_val_loss = _last_history_value(history, "val_loss")
    last_train_mae = _last_history_value(history, "train_mae")
    last_val_mae = _last_history_value(history, "val_mae")
    last_lr = _last_history_value(history, "learning_rate")

    if last_train_loss is not None:
        snapshot["train_loss_last"] = last_train_loss
    if last_val_loss is not None:
        snapshot["val_loss_last"] = last_val_loss
    if last_train_mae is not None:
        snapshot["train_mae_last"] = last_train_mae
    if last_val_mae is not None:
        snapshot["val_mae_last"] = last_val_mae
    if last_lr is not None:
        snapshot["learning_rate_last"] = last_lr

    return snapshot


def _build_test_snapshot(training_data: Dict[str, Any]) -> Optional[Dict[str, float]]:
    test_results = training_data.get("test_results")
    if not isinstance(test_results, dict):
        return None

    snapshot = {
        "loss": _safe_float(test_results.get("loss")),
        "mae": _safe_float(test_results.get("mae")),
        "rmse": _safe_float(test_results.get("rmse")),
    }

    filtered = {k: v for k, v in snapshot.items() if v is not None}
    return filtered or None


def sync_training_to_evaluation(
    training_path: Path,
    evaluation_path: Path,
    dry_run: bool = False,
    eval_report_path: Optional[Path] = None,
    figures_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Update Stage2 evaluation results with the latest glucose training stats."""
    training_data = _load_training_json(training_path)
    evaluation_data = _load_evaluation_json(evaluation_path)

    training_snapshot = _build_training_snapshot(training_data, training_path)
    test_snapshot = _build_test_snapshot(training_data)

    evaluation_data["training_snapshot"] = training_snapshot
    if test_snapshot:
        evaluation_data["test_snapshot"] = test_snapshot
    if eval_report_path and eval_report_path.exists():
        evaluation_data["stage2_eval_report"] = _load_evaluation_json(eval_report_path)
    if figures_dir and figures_dir.exists():
        evaluation_data["figures"] = sorted(
            [str(p) for p in figures_dir.glob("*.png")]
        )

    if dry_run:
        logger.info("Dry run: 生成的评估内容未写入文件。")
        return evaluation_data

    evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    with evaluation_path.open("w", encoding="utf-8") as f:
        json.dump(evaluation_data, f, indent=2, ensure_ascii=False)

    logger.info(
        "已同步训练结果: %s -> %s",
        training_snapshot["source"],
        str(evaluation_path).replace("\\", "/"),
    )
    return evaluation_data


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将训练结果(training_results.json)写入 Stage2 评估文件。"
    )
    parser.add_argument(
        "--training",
        help="训练结果JSON路径，例如 stage2(xt）/suya_model/final_model/1/training_results.json",
    )
    parser.add_argument(
        "--training-glob",
        help="若路径包含特殊字符，可用glob表达式自动匹配，例如 \"stage2*/suya_model/final_model/1/training_results.json\"。",
    )
    parser.add_argument(
        "--evaluation",
        default="stage2/results/models/evaluation_results.json",
        help="Stage2评估结果文件路径（默认: stage2/results/models/evaluation_results.json）",
    )
    parser.add_argument(
        "--eval-report",
        help="stage2/evaluation/run_eval.py 输出的 report.json 路径",
    )
    parser.add_argument(
        "--figures-dir",
        help="可视化图表目录 (例如 stage2/results/eval_reports/latest/figures)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只预览同步结果，不写入文件。",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.training:
        training_path = Path(args.training)
    elif args.training_glob:
        matches = sorted(Path(".").glob(args.training_glob))
        if not matches:
            raise FileNotFoundError(f"无法匹配到任何 training_results.json: {args.training_glob}")
        training_path = matches[0]
        logger.warning(
            "根据 glob 匹配到训练结果: %s",
            str(training_path).replace("\\", "/"),
        )
    else:
        raise SystemExit("必须提供 --training 或 --training-glob。")
    evaluation_path = Path(args.evaluation)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    eval_report = Path(args.eval_report) if args.eval_report else None
    figures_dir = Path(args.figures_dir) if args.figures_dir else None
    sync_training_to_evaluation(
        training_path,
        evaluation_path,
        dry_run=args.dry_run,
        eval_report_path=eval_report,
        figures_dir=figures_dir,
    )


if __name__ == "__main__":
    main()
