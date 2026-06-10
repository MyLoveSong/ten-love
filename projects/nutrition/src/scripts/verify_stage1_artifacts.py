#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage1 训练产出验证与桥接数据更新
================================

功能:
1. 校验 Stage1 文化适配微调的关键产物 (summary 与 LoRA 权重)
2. 生成结构化验证报告,支持写入 TRAIN/outputs/stage1_cultural
3. 可选: 自动更新 stage2/configs/stage1_bridge_data.json 的性能指标
"""

import argparse
import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional


def compute_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> Optional[str]:
    """计算文件的 SHA256,文件不存在时返回 None"""
    if not file_path.exists():
        return None
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ArtifactStatus:
    path: str
    exists: bool
    size_bytes: int = 0
    sha256: Optional[str] = None

    @classmethod
    def from_path(cls, path: Path) -> "ArtifactStatus":
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        checksum = compute_sha256(path) if exists else None
        return cls(
            path=str(path),
            exists=exists,
            size_bytes=size,
            sha256=checksum,
        )


def load_training_summary(summary_path: Path) -> Dict[str, Any]:
    """加载 Stage1 summary,提取训练指标"""
    if not summary_path.exists():
        raise FileNotFoundError(f"未找到 summary 文件: {summary_path}")

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    result = summary.get("result", {})
    history = result.get("history", [])
    best_epoch = result.get("best_epoch")
    if best_epoch is None and history:
        best_epoch = history[-1].get("epoch")

    def _get_entry(epoch_value: Optional[int]) -> Dict[str, Any]:
        if epoch_value is None:
            return history[-1] if history else {}
        for record in history:
            if record.get("epoch") == epoch_value:
                return record
        return history[-1] if history else {}

    best_entry = _get_entry(best_epoch)
    last_entry = history[-1] if history else {}
    first_entry = history[0] if history else {}

    metrics = {
        "epochs": len(history),
        "best_epoch": best_epoch,
        "best_train_loss": best_entry.get("train_loss"),
        "best_val_loss": best_entry.get("val_loss"),
        "first_train_loss": first_entry.get("train_loss"),
        "first_val_loss": first_entry.get("val_loss"),
        "last_train_loss": last_entry.get("train_loss"),
        "last_val_loss": last_entry.get("val_loss"),
    }
    return metrics


def update_bridge_file(
    bridge_path: Path,
    metrics: Dict[str, Any],
    artifacts: Dict[str, ArtifactStatus],
    verified_at: str,
) -> None:
    """将最新的 Stage1 指标写入 Stage2 桥接文件"""
    bridge: Dict[str, Any] = {}
    if bridge_path.exists():
        with bridge_path.open("r", encoding="utf-8") as f:
            bridge = json.load(f)

    performance = bridge.setdefault("performance_metrics", {})
    performance.update(
        {
            "stage1_best_epoch": metrics.get("best_epoch"),
            "stage1_best_val_loss": metrics.get("best_val_loss"),
            "stage1_best_train_loss": metrics.get("best_train_loss"),
            "stage1_total_epochs": metrics.get("epochs"),
            "stage1_last_val_loss": metrics.get("last_val_loss"),
            "source": "stage1_verification_report",
            "verified_at": verified_at,
        }
    )

    artifacts_section = bridge.setdefault("artifacts", {})
    artifacts_section["stage1_weights"] = asdict(artifacts["weights"])
    artifacts_section["stage1_best_weights"] = asdict(artifacts["best_weights"])
    artifacts_section["stage1_summary"] = {
        "path": str(metrics.get("summary_path")),
        "verified_at": verified_at,
    }

    bridge_path.parent.mkdir(parents=True, exist_ok=True)
    with bridge_path.open("w", encoding="utf-8") as f:
        json.dump(bridge, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="验证 Stage1 产物并可选更新 Stage2 桥接数据")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("TRAIN/outputs/stage1_cultural/stage1_run_summary.json"),
        help="Stage1 训练 summary 文件路径",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("TRAIN/outputs/stage1_cultural/cultural_lora_weights.pt"),
        help="最新 LoRA 权重文件路径",
    )
    parser.add_argument(
        "--best-weights",
        type=Path,
        default=Path("TRAIN/outputs/stage1_cultural/cultural_lora_weights_best.pt"),
        help="最佳指标对应的 LoRA 权重路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("TRAIN/outputs/stage1_cultural/stage1_verification_report.json"),
        help="验证报告输出路径",
    )
    parser.add_argument(
        "--bridge-path",
        type=Path,
        default=Path("stage2/configs/stage1_bridge_data.json"),
        help="Stage2 桥接数据路径",
    )
    parser.add_argument(
        "--update-bridge",
        action="store_true",
        help="验证完成后自动写入 Stage2 桥接文件",
    )

    args = parser.parse_args()

    summary_metrics = load_training_summary(args.summary)
    summary_metrics["summary_path"] = str(args.summary)

    weights_status = ArtifactStatus.from_path(args.weights)
    best_weights_status = ArtifactStatus.from_path(args.best_weights)

    report = {
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary_metrics,
        "artifacts": {
            "weights": asdict(weights_status),
            "best_weights": asdict(best_weights_status),
        },
        "status": "passed"
        if weights_status.exists and best_weights_status.exists
        else "failed",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    if args.update_bridge:
        update_bridge_file(
            args.bridge_path,
            summary_metrics,
            {"weights": weights_status, "best_weights": best_weights_status},
            report["verified_at"],
        )


if __name__ == "__main__":
    main()
