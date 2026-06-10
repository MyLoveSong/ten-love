"""
Data loader for glucose datasets (CSV/JSON) with 12->6 windowing.
- Handles multiple datasets
- Forward-fill and interpolation for missing values
- Optional patient grouping if `patient_id` is available
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


def _to_numpy_array(values: List[float]) -> np.ndarray:
    return np.array(values, dtype=float)


def _forward_fill(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    mask = np.isnan(arr)
    if mask.all():
        return np.zeros_like(arr)
    idx = np.where(~mask, np.arange(mask.shape[0]), 0)
    np.maximum.accumulate(idx, out=idx)
    return arr[idx]


def _interpolate_nan(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    n = len(arr)
    x = np.arange(n)
    mask = np.isfinite(arr)
    if mask.sum() < 2:
        return _forward_fill(arr)
    return np.interp(x, x[mask], arr[mask])


def _window_series(series: np.ndarray, in_len: int, out_len: int) -> Tuple[np.ndarray, np.ndarray]:
    X, Y = [], []
    total = in_len + out_len
    for i in range(0, len(series) - total + 1):
        x = series[i : i + in_len]
        y = series[i + in_len : i + total]
        X.append(x.reshape(in_len, 1))
        Y.append(y)
    if not X:
        return np.empty((0, in_len, 1)), np.empty((0, out_len))
    return np.stack(X, axis=0), np.stack(Y, axis=0)


def _extract_glucose(record: Dict, glucose_keys: List[str]) -> Optional[float]:
    for k in glucose_keys:
        if k in record and record[k] is not None:
            try:
                return float(record[k])
            except Exception:
                continue
    return None


def _iter_json_records(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        return data["data"]
    if isinstance(data, list):
        return data
    # Fallback: unknown format
    return []


def _iter_csv_records(path: Path) -> List[Dict]:
    # Lightweight CSV reader without pandas
    import csv

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def load_datasets(
    dataset_paths: List[Union[str, Path]],
    glucose_keys: Optional[List[str]] = None,
    patient_id_keys: Optional[List[str]] = None,
    input_len: int = 12,
    output_len: int = 6,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """
    Load multiple datasets and build 12->6 windows.

    Returns:
        sequences: (N, input_len, 1)
        targets:   (N, output_len)
        by_patient: mapping patient_id -> (sequences, targets)
    """
    glucose_keys = glucose_keys or [
        "glucose", "bg", "blood_glucose", "cgms", "value"
    ]
    patient_id_keys = patient_id_keys or ["patient_id", "pid", "subject_id", "user_id"]

    all_sequences: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    by_patient: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for p in dataset_paths:
        path = Path(p)
        if not path.exists():
            logger.warning(f"Dataset not found: {path}")
            continue
        try:
            if path.suffix.lower() in (".json", ".ndjson"):
                records = _iter_json_records(path)
            elif path.suffix.lower() in (".csv", ".tsv"):
                records = _iter_csv_records(path)
            else:
                logger.warning(f"Unsupported file type: {path.suffix} ({path})")
                continue

            # Group by patient if available
            groups: Dict[str, List[float]] = {}
            current_series: List[float] = []

            for rec in records:
                pid = None
                for k in patient_id_keys:
                    if k in rec and rec[k]:
                        pid = str(rec[k])
                        break

                g = _extract_glucose(rec, glucose_keys)
                current_series.append(np.nan if g is None else g)

                key = pid or "__global__"
                # We'll aggregate globally first; patient grouping built after

            series = _to_numpy_array(current_series)
            series = _interpolate_nan(series)
            seq, tgt = _window_series(series, input_len, output_len)
            if seq.size:
                all_sequences.append(seq)
                all_targets.append(tgt)

            # Patient-wise if available
            if any(k in records[0] for k in patient_id_keys) if records else False:
                bucket: Dict[str, List[float]] = {}
                for rec in records:
                    pid = None
                    for k in patient_id_keys:
                        if k in rec and rec[k]:
                            pid = str(rec[k])
                            break
                    g = _extract_glucose(rec, glucose_keys)
                    key = pid or "__global__"
                    bucket.setdefault(key, []).append(np.nan if g is None else g)
                for pid, vals in bucket.items():
                    s = _interpolate_nan(_to_numpy_array(vals))
                    x, y = _window_series(s, input_len, output_len)
                    if x.size:
                        by_patient[pid] = (
                            x if pid not in by_patient else np.concatenate([by_patient[pid][0], x], axis=0),
                            y if pid not in by_patient else np.concatenate([by_patient[pid][1], y], axis=0),
                        )

            logger.info(f"Loaded {path} -> windows: {0 if not seq.size else len(seq)}")
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
            continue

    if all_sequences:
        sequences = np.concatenate(all_sequences, axis=0)
        targets = np.concatenate(all_targets, axis=0)
    else:
        sequences = np.empty((0, input_len, 1))
        targets = np.empty((0, output_len))

    return sequences, targets, by_patient
