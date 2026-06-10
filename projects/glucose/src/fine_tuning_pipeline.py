#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
End-to-end fine-tuning pipeline for glucose forecasting.

Workflow:
1. 拉取公开CGM数据（Ohio T1DM / Glucose-ML 等）并标准化为统一窗口。
2. 可选地拼接本地样本（默认取 TRAIN/data 内的 JSON/CSV）。
3. 载入开源预训练 checkpoint（本地或 Hugging Face 仓库）。
4. 在 EnhancedGlucosePredictionSystem 中应用 LoRA 或部分层微调。
5. 训练 + 评估 + （可选）按患者 LoRA 个性化。

运行示例：
python TRAIN/fine_tuning_pipeline.py ^
  --public_sources ohio_t1dm glucose_ml_collection ^
  --local_data TRAIN/data/final_dataset/final_training_data.json ^
  --pretrained_repo Lichen/Gluformer ^
  --pretrained_filename gluformer_glucose.pt ^
  --use_lora ^
  --personalize_patients
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

# Ensure project root on sys.path when executed directly
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from data_sources.web_data_collector import CGMDataSource  # noqa: E402
from enhanced_glucose_system import EnhancedGlucosePredictionSystem  # noqa: E402
from core.adapters import LoRAAdapter  # noqa: E402
from run_glucose_training import aggregate_datasets, build_windows  # noqa: E402

try:
    from huggingface_hub import hf_hub_download  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    hf_hub_download = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("fine_tune_pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CGM LoRA fine-tuning pipeline")
    parser.add_argument(
        "--public_sources",
        nargs="+",
        default=["ohio_t1dm", "glucose_ml_collection"],
        help="公开数据源 key，参考 CGMDataSource.data_sources",
    )
    parser.add_argument(
        "--local_data",
        nargs="*",
        default=[],
        help="额外拼接的本地 CSV/JSON（采用 run_glucose_training 聚合逻辑）",
    )
    parser.add_argument("--in_len", type=int, default=12, help="输入窗口长度 (5min granularity)")
    parser.add_argument("--out_len", type=int, default=6, help="预测步数")
    parser.add_argument(
        "--max_windows",
        type=int,
        default=150_000,
        help="限制最终窗口总量，防止显存爆炸（0 表示不限制）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="TRAIN/outputs/fine_tune_pipeline",
        help="保存日志/模型/指标的目录",
    )
    parser.add_argument(
        "--pretrained_checkpoint",
        type=str,
        default="",
        help="本地 checkpoint 路径（.pt/.bin）",
    )
    parser.add_argument(
        "--pretrained_repo",
        type=str,
        default="",
        help="Hugging Face 仓库 ID（如 Lichen/Gluformer）。若提供则优先下载",
    )
    parser.add_argument(
        "--pretrained_filename",
        type=str,
        default="pytorch_model.bin",
        help="从 Hugging Face 拉取的文件名",
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="在训练前对 GluFormer / WaveletGluFormer 应用 LoRA 适配，仅更新低秩参数",
    )
    parser.add_argument(
        "--lora_targets",
        nargs="*",
        default=["gluformer", "wavelet_gluformer"],
        help="需要 LoRA 的模型名称，默认只投向 GluFormer 系列",
    )
    parser.add_argument(
        "--personalize_patients",
        action="store_true",
        help="训练完成后，对拥有充足窗口的患者做 LoRA 个性化微调",
    )
    parser.add_argument(
        "--personalization_min_windows",
        type=int,
        default=120,
        help="单个患者最少窗口数（不足则跳过个性化）",
    )
    parser.add_argument(
        "--personalization_epochs",
        type=int,
        default=5,
        help="每个患者 LoRA 个性化轮数",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="全局验证集比例（传递到 EnhancedGlucosePredictionSystem）",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="全局训练 epoch",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="全局 batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
        help="全局学习率",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="仅检查数据与 checkpoint，跳过训练（用于快速 smoke test）",
    )
    return parser.parse_args()


def fetch_public_data(
    sources: Sequence[str],
    in_len: int,
    out_len: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[str, Dict[str, np.ndarray]]]:
    """Call CGMDataSource for each source and windowize per patient."""
    collector = CGMDataSource()
    sequences_collection: List[np.ndarray] = []
    targets_collection: List[np.ndarray] = []
    patient_windows: Dict[str, Dict[str, np.ndarray]] = {}

    for source in sources:
        try:
            df = collector.fetch_data(source=source)
            collector.validate_data(df)
        except Exception as exc:  # pragma: no cover - best-effort data pull
            logger.warning("Failed to fetch %s: %s", source, exc)
            continue

        seqs, tgts, per_patient = dataframe_to_windows(df, in_len, out_len, prefix=source)
        if seqs.size == 0:
            logger.warning("Source %s produced zero windows", source)
            continue
        sequences_collection.append(seqs)
        targets_collection.append(tgts)
        patient_windows.update(per_patient)
        logger.info(
            "Loaded %s windows from %s (patients=%s)",
            len(seqs),
            source,
            len(per_patient),
        )

    return sequences_collection, targets_collection, patient_windows


def dataframe_to_windows(
    df,
    in_len: int,
    out_len: int,
    prefix: str = "public",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Dict[str, np.ndarray]]]:
    """Convert a standardized CGM dataframe into sliding windows."""
    if df.empty:
        return (
            np.empty((0, in_len, 1), dtype=np.float32),
            np.empty((0, out_len), dtype=np.float32),
            {},
        )

    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")

    glucose_col = None
    for candidate in ["glucose_mg_dl", "glucose", "bg"]:
        if candidate in df.columns:
            glucose_col = candidate
            break
    if glucose_col is None and "glucose_mmol_l" in df.columns:
        df["glucose_mg_dl"] = df["glucose_mmol_l"] * 18.0
        glucose_col = "glucose_mg_dl"

    if glucose_col is None:
        logger.warning("Dataframe missing glucose column, skipping")
        return (
            np.empty((0, in_len, 1), dtype=np.float32),
            np.empty((0, out_len), dtype=np.float32),
            {},
        )

    if "patient_id" not in df.columns:
        df = df.copy()
        df["patient_id"] = [f"{prefix}_{i:06d}" for i in range(len(df))]

    grouped = df.groupby("patient_id", sort=False)
    sequences_list: List[np.ndarray] = []
    targets_list: List[np.ndarray] = []
    per_patient: Dict[str, Dict[str, np.ndarray]] = {}

    for pid, group in grouped:
        glucose_series = group[glucose_col].astype(float).to_numpy()
        seqs, tgts = build_windows(glucose_series.tolist(), in_len, out_len)
        if seqs.size == 0:
            continue
        sequences_list.append(seqs)
        targets_list.append(tgts)
        per_patient[str(pid)] = {
            "sequences": seqs,
            "targets": tgts,
        }

    if not sequences_list:
        return (
            np.empty((0, in_len, 1), dtype=np.float32),
            np.empty((0, out_len), dtype=np.float32),
            {},
        )

    sequences = np.concatenate(sequences_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)
    return sequences, targets, per_patient


def load_local_windows(
    paths: Sequence[str],
    in_len: int,
    out_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Leverage existing aggregation helpers for local CSV/JSON."""
    if not paths:
        return (
            np.empty((0, in_len, 1), dtype=np.float32),
            np.empty((0, out_len), dtype=np.float32),
        )

    resolved_paths = [str(Path(p)) for p in paths]
    sequences, targets = aggregate_datasets(resolved_paths, in_len, out_len)
    return sequences, targets


def resolve_pretrained_checkpoint(
    args: argparse.Namespace,
) -> Optional[Path]:
    """Find or download checkpoint file."""
    if args.pretrained_repo:
        if hf_hub_download is None:
            logger.warning("huggingface_hub not available, cannot download repo checkpoint")
        else:
            try:
                logger.info("Downloading checkpoint %s/%s", args.pretrained_repo, args.pretrained_filename)
                ckpt_path = hf_hub_download(
                    repo_id=args.pretrained_repo,
                    filename=args.pretrained_filename,
                )
                return Path(ckpt_path)
            except Exception as exc:  # pragma: no cover - network side effect
                logger.warning("HF download failed: %s", exc)

    if args.pretrained_checkpoint:
        path = Path(args.pretrained_checkpoint)
        if path.is_file():
            return path
        logger.warning("Provided checkpoint path %s does not exist", path)
    return None


def load_weights_into_models(
    system: EnhancedGlucosePredictionSystem,
    checkpoint_path: Path,
    target_models: Sequence[str],
) -> Dict[str, List[str]]:
    """Attempt to load pretrained weights into selected models."""
    logger.info("Loading pretrained weights from %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(ckpt, dict):
        # unwrap common nesting
        if "model_state_dict" in ckpt:
            ckpt = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]

    load_report: Dict[str, List[str]] = {}

    for model_name in target_models:
        model = system.models.get(model_name)
        if model is None:
            continue
        try:
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            load_report[model_name] = [
                f"missing={len(missing)}",
                f"unexpected={len(unexpected)}",
            ]
            logger.info(
                "Loaded pretrained weights into %s (missing=%d, unexpected=%d)",
                model_name,
                len(missing),
                len(unexpected),
            )
        except Exception as exc:
            logger.warning("Failed to load weights into %s: %s", model_name, exc)

    return load_report


def apply_lora_to_models(
    system: EnhancedGlucosePredictionSystem,
    model_names: Sequence[str],
) -> Dict[str, int]:
    """Wrap selected models with LoRA adapters."""
    if not system.config.get("lora", {}).get("enabled", False):
        logger.info("LoRA disabled in config, skipping adapter application")
        return {}

    adapter = LoRAAdapter(
        rank=system.config["lora"]["rank"],
        alpha=system.config["lora"]["alpha"],
        dropout=system.config["lora"]["dropout"],
        target_modules=system.config["lora"]["target_modules"],
        glucose_specific=system.config["lora"]["glucose_specific"],
    )

    adapted_counts: Dict[str, int] = {}

    for model_name in model_names:
        model = system.models.get(model_name)
        if model is None:
            continue
        adapter.apply_lora(model, module_name=model_name)
        params = adapter.count_parameters(model)
        adapted_counts[model_name] = params.get("lora", 0)
        logger.info(
            "Applied LoRA to %s (LoRA params=%d, total=%d)",
            model_name,
            params.get("lora", 0),
            params.get("total", 0),
        )

    return adapted_counts


def cap_windows(
    sequences: np.ndarray,
    targets: np.ndarray,
    max_windows: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Downsample when exceeding requested cap."""
    if max_windows <= 0 or len(sequences) <= max_windows:
        return sequences, targets
    idx = np.random.permutation(len(sequences))[:max_windows]
    return sequences[idx], targets[idx]


def personalize_by_patient(
    system: EnhancedGlucosePredictionSystem,
    patient_windows: Dict[str, Dict[str, np.ndarray]],
    min_windows: int,
    epochs: int,
) -> List[Dict[str, object]]:
    """Run LoRA personalization for selected patients."""
    if not patient_windows:
        return []

    results: List[Dict[str, object]] = []

    for pid, data in patient_windows.items():
        seqs = data["sequences"]
        tgts = data["targets"]
        if len(seqs) < min_windows:
            continue

        engineered = system._apply_feature_engineering(seqs.copy())  # pylint: disable=protected-access
        result = system.personalize_for_patient(
            patient_id=pid,
            patient_data=engineered,
            patient_targets=tgts,
            epochs=epochs,
        )
        results.append(result)
        logger.info(
            "Personalized patient %s (epochs=%d, samples=%d)",
            pid,
            result["personalization_results"]["epochs_trained"],
            len(seqs),
        )

    return results


def build_system_config(args: argparse.Namespace, output_dir: Path) -> Dict[str, object]:
    """Create config override for EnhancedGlucosePredictionSystem."""
    return {
        "output_dir": str(output_dir),
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "val_split": args.val_split,
            "patience": max(8, args.epochs // 4),
            "gradient_clip": 1.0,
            "lr_scheduler": {
                "type": "cosine_warmup",
                "warmup_epochs": min(5, args.epochs // 4),
                "warmup_start_factor": 0.2,
                "min_lr": 1e-5,
            },
            "early_stopping": {
                "patience": max(5, args.epochs // 5),
                "min_delta": 5e-4,
                "burn_in_epochs": 3,
                "cooldown": 1,
            },
        },
        "lora": {
            "enabled": args.use_lora or args.personalize_patients,
            "rank": 8,
            "alpha": 16.0,
            "dropout": 0.05,
            "target_modules": ["self_attention", "prediction", "output"],
            "glucose_specific": True,
            "learning_rate": 1e-3,
            "weight_decay": 0.01,
            "personalization": {
                "enabled": args.personalize_patients,
                "max_epochs": args.personalization_epochs,
                "validation_split": 0.2,
                "early_stopping_patience": 3,
                "glucose_metrics": True,
            },
        },
        "rare_event_augmentation": {
            "enabled": True,
            "event_types": ["hypoglycemia", "hyperglycemia", "rapid_drop", "rapid_rise"],
        },
        "monitoring": {
            "enabled": True,
            "window_size": 500,
            "drift_threshold": 0.05,
        },
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / f"finetune_log_{datetime.now():%Y%m%d_%H%M%S}.txt"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(file_handler)

    sequences_parts, targets_parts, patient_windows = fetch_public_data(
        args.public_sources,
        args.in_len,
        args.out_len,
    )

    local_sequences, local_targets = load_local_windows(args.local_data, args.in_len, args.out_len)
    if local_sequences.size:
        sequences_parts.append(local_sequences)
        targets_parts.append(local_targets)

    if not sequences_parts:
        logger.error("No training data loaded. Please verify data sources.")
        return 1

    sequences = np.concatenate(sequences_parts, axis=0)
    targets = np.concatenate(targets_parts, axis=0)

    if args.max_windows > 0 and len(sequences) > args.max_windows:
        sequences, targets = cap_windows(sequences, targets, args.max_windows)
        logger.info("Down-sampled to %d windows per --max_windows", args.max_windows)

    logger.info("Final dataset: sequences=%s, targets=%s", sequences.shape, targets.shape)

    system_config = build_system_config(args, output_dir)
    system = EnhancedGlucosePredictionSystem(system_config)

    if args.dry_run:
        logger.info("Dry run requested; data preparation successful, exiting before training.")
        return 0

    data_dict = system.prepare_data(sequences, targets)
    system.create_models()

    ckpt_path = resolve_pretrained_checkpoint(args)
    load_report = {}
    if ckpt_path:
        load_report = load_weights_into_models(system, ckpt_path, args.lora_targets)

    lora_report = {}
    if args.use_lora:
        lora_report = apply_lora_to_models(system, args.lora_targets)

    training_results = system.train_models(data_dict)
    system.create_ensemble(data_dict)
    system.setup_monitoring(data_dict["train_sequences"])
    ensemble_metrics = system.evaluate_ensemble(
        data_dict["val_sequences"],
        data_dict["val_targets"],
    )
    system.save_system_state()
    system.is_trained = True

    personalization_reports: List[Dict[str, object]] = []
    if args.personalize_patients:
        system.setup_personalization()
        personalization_reports = personalize_by_patient(
            system,
            patient_windows,
            args.personalization_min_windows,
            args.personalization_epochs,
        )

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_windows": int(len(sequences)),
        "train_windows": int(len(data_dict["train_sequences"])),
        "val_windows": int(len(data_dict["val_sequences"])),
        "public_sources": list(args.public_sources),
        "local_files": list(args.local_data),
        "pretrained_checkpoint": str(ckpt_path) if ckpt_path else "",
        "load_report": load_report,
        "lora_report": lora_report,
        "ensemble_metrics": ensemble_metrics,
        "training_results": training_results,
        "personalization": personalization_reports,
        "config": system_config,
    }

    summary_path = output_dir / "finetune_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    logger.info("Fine-tuning pipeline finished. Summary saved to %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
