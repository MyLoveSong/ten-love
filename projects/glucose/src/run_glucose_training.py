import os
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, DefaultDict
from collections import defaultdict

import numpy as np
import torch

from enhanced_glucose_system import EnhancedGlucosePredictionSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


ID_CANDIDATE_KEYS = [
    'patient_id', 'pid', 'user_id', 'subject', 'subject_id', 'patient', 'id'
]


def _select_best_numeric_series(candidates: Dict[str, List[float]]) -> List[float]:
    """Select best glucose-like series among numeric candidates."""
    if not candidates:
        return []
    best_key = None
    best_score = -1.0
    for key, vals in candidates.items():
        if len(vals) == 0:
            continue
        arr = np.array(vals, dtype=np.float32)
        q1, q3 = np.percentile(arr, [25, 75])
        med = float(np.median(arr))
        in_range = np.mean((arr >= 40) & (arr <= 400))
        var = float(np.var(arr))
        score = in_range * 2.0
        if 70 <= med <= 180:
            score += 1.0
        score += min(var / 1000.0, 2.0)
        lowered = key.lower()
        if any(k in lowered for k in ['glucose', 'bg', 'blood_glucose']):
            score += 1.0
        if score > best_score:
            best_score = score
            best_key = key
    return candidates.get(best_key, [])


essential_value_keys = [
    'glucose', 'bg', 'blood_glucose', 'value', 'glucose_value'
]


def load_csv_series(path: Path, value_keys: List[str]) -> List[float]:
    import csv
    series: List[float] = []
    candidates: Dict[str, List[float]] = {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                if val in (None, ""):
                    continue
                try:
                    v = float(val)
                except Exception:
                    continue
                if key not in candidates:
                    candidates[key] = []
                candidates[key].append(v)
    for k in value_keys:
        if k in candidates and len(candidates[k]) > 0:
            series = candidates[k]
            break
    if not series:
        series = _select_best_numeric_series(candidates)
    return series


def load_csv_by_patient(path: Path) -> DefaultDict[str, List[float]]:
    import csv
    groups: DefaultDict[str, List[float]] = defaultdict(list)
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        id_key = None
        # detect id key
        header = reader.fieldnames or []
        lowered = [h.lower() for h in header]
        for cand in ID_CANDIDATE_KEYS:
            if cand in lowered:
                id_key = header[lowered.index(cand)]
                break
        for row in reader:
            # get id
            pid = row.get(id_key) if id_key else None
            pid = str(pid) if pid not in (None, "") else "unknown"
            # get glucose numeric candidates
            best_val = None
            for key, val in row.items():
                if val in (None, ""):
                    continue
                try:
                    v = float(val)
                except Exception:
                    continue
                # pick first plausible in range
                if 40 <= v <= 400:
                    best_val = v
                    break
            if best_val is not None:
                groups[pid].append(best_val)
    return groups


def load_json_series(path: Path, value_keys: List[str]) -> List[float]:
    series: List[float] = []
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict):
        for k in ['records', 'data', 'items']:
            if k in data and isinstance(data[k], list):
                data = data[k]
                break
    candidates: Dict[str, List[float]] = {}
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                for key, val in item.items():
                    if isinstance(val, (int, float)):
                        candidates.setdefault(key, []).append(float(val))
                    elif isinstance(val, str):
                        try:
                            v = float(val)
                            candidates.setdefault(key, []).append(v)
                        except Exception:
                            continue
            elif isinstance(item, (int, float)):
                candidates.setdefault('value', []).append(float(item))
    for k in value_keys:
        if k in candidates and len(candidates[k]) > 0:
            series = candidates[k]
            break
    if not series:
        series = _select_best_numeric_series(candidates)
    return series


def load_json_by_patient(path: Path) -> DefaultDict[str, List[float]]:
    groups: DefaultDict[str, List[float]] = defaultdict(list)
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict):
        for k in ['records', 'data', 'items']:
            if k in data and isinstance(data[k], list):
                data = data[k]
                break
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            # detect id key once
            pid = None
            for cand in ID_CANDIDATE_KEYS:
                if cand in item:
                    pid = item[cand]
                    break
            pid = str(pid) if pid not in (None, "") else "unknown"
            # pick first plausible glucose value
            best_val = None
            for key, val in item.items():
                if isinstance(val, (int, float)):
                    v = float(val)
                elif isinstance(val, str):
                    try:
                        v = float(val)
                    except Exception:
                        continue
                else:
                    continue
                if 40 <= v <= 400:
                    best_val = v
                    break
            if best_val is not None:
                groups[pid].append(best_val)
    return groups


def build_windows(series: List[float], in_len: int = 12, out_len: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.array(series, dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, in_len, 1), dtype=np.float32), np.empty((0, out_len), dtype=np.float32)
    if np.isnan(arr).any():
        mask = np.isnan(arr)
        idx = np.where(~mask, np.arange(mask.size), 0)
        np.maximum.accumulate(idx, out=idx)
        arr = arr[idx]
    X, Y = [], []
    total = len(arr)
    win = in_len + out_len
    for i in range(0, total - win + 1):
        x = arr[i:i+in_len]
        y = arr[i+in_len:i+in_len+out_len]
        if np.any(np.isnan(x)) or np.any(np.isnan(y)):
            continue
        X.append(x.reshape(in_len, 1))
        Y.append(y)
    if len(X) == 0:
        return np.empty((0, in_len, 1), dtype=np.float32), np.empty((0, out_len), dtype=np.float32)
    return np.stack(X), np.stack(Y)


def aggregate_datasets(paths: List[str], in_len: int, out_len: int) -> Tuple[np.ndarray, np.ndarray]:
    sequences_list: List[np.ndarray] = []
    targets_list: List[np.ndarray] = []
    total_loaded = 0
    for p in paths:
        path = Path(p)
        if not path.exists():
            logger.warning(f"Path not found: {path}")
            continue
        try:
            if path.suffix.lower() in ['.csv', '.tsv']:
                series = load_csv_series(path, essential_value_keys)
            elif path.suffix.lower() in ['.json']:
                series = load_json_series(path, essential_value_keys)
            else:
                logger.warning(f"Unsupported format: {path.suffix} at {path}")
                continue
            if len(series) < in_len + out_len:
                logger.warning(f"Too short or no numeric glucose-like series in {path}, length={len(series)}")
                continue
            X, Y = build_windows(series, in_len, out_len)
            if len(X) > 0:
                sequences_list.append(X)
                targets_list.append(Y)
                total_loaded += len(X)
                logger.info(f"Loaded {len(X)} windows from {path}")
            else:
                logger.warning(f"No windows from {path} after slicing")
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
    if not sequences_list:
        return np.empty((0, in_len, 1), dtype=np.float32), np.empty((0, out_len), dtype=np.float32)
    sequences = np.concatenate(sequences_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)
    logger.info(f"Aggregated windows: {len(sequences)}")
    return sequences, targets


def aggregate_by_patient(paths: List[str], in_len: int, out_len: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    per_patient: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    all_X: List[np.ndarray] = []
    all_Y: List[np.ndarray] = []
    patient_series: DefaultDict[str, List[float]] = defaultdict(list)

    for p in paths:
        path = Path(p)
        if not path.exists():
            logger.warning(f"Path not found: {path}")
            continue
        try:
            if path.suffix.lower() in ['.csv', '.tsv']:
                groups = load_csv_by_patient(path)
            elif path.suffix.lower() in ['.json']:
                groups = load_json_by_patient(path)
            else:
                continue
            for pid, series in groups.items():
                if len(series) >= in_len + out_len:
                    patient_series[pid].extend(series)
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")

    for pid, series in patient_series.items():
        X, Y = build_windows(series, in_len, out_len)
        if len(X) > 0:
            per_patient[pid] = (X, Y)
            all_X.append(X)
            all_Y.append(Y)

    if not all_X:
        return np.empty((0, in_len, 1), dtype=np.float32), np.empty((0, out_len), dtype=np.float32), {}

    sequences = np.concatenate(all_X, axis=0)
    targets = np.concatenate(all_Y, axis=0)
    logger.info(f"Aggregated by patient: patients={len(per_patient)}, windows={len(sequences)}")
    return sequences, targets, per_patient


def get_default_config(output_dir: Path) -> Dict[str, Any]:
    return {
        'models': {
            'lstm': {'input_dim': 1, 'hidden_dim': 64, 'output_dim': 6, 'dropout': 0.1},
            'transformer': {'input_dim': 1, 'hidden_dim': 64, 'output_dim': 6, 'dropout': 0.1},
            'gluformer': {'input_dim': 1, 'hidden_dim': 64, 'output_dim': 6, 'dropout': 0.1},
            'wavelet_gluformer': {'input_dim': 1, 'hidden_dim': 64, 'output_dim': 6, 'dropout': 0.1, 'wavelet': 'db4', 'wavelet_levels': 3}
        },
        'training': {
            'epochs': 10,
            'batch_size': 128,
            'learning_rate': 0.001,
            'patience': 3,
            'val_split': 0.2,
            'gradient_clip': 1.0,
            'lr_scheduler': {
                'type': 'cosine_warmup',
                'warmup_epochs': 3,
                'warmup_start_factor': 0.3,
                'min_lr': 1e-5
            },
            'early_stopping': {
                'patience': 3,
                'min_delta': 5e-4,
                'burn_in_epochs': 3,
                'cooldown': 1
            }
        },
        'ensemble': {'strategy': 'weighted', 'auto_weight_update': True},
        'augmentation': {'enabled': True, 'factor': 1.2, 'methods': ['noise', 'scaling']},
        'rare_event_augmentation': {'enabled': False, 'factor': 1.2, 'methods': ['smote', 'noise_injection'], 'event_types': ['hypoglycemia', 'hyperglycemia']},
        'feature_engineering': {
            'enabled': True,
            'features': ['delta', 'rolling_mean'],
            'rolling_window': 4,
            'normalize': True
        },
        'lora': {
            'enabled': True, 'rank': 4, 'alpha': 8.0, 'dropout': 0.05,
            'target_modules': ['lstm', 'gru', 'self_attention', 'cross_attention', 'prediction_heads'],
            'glucose_specific': True, 'learning_rate': 0.001, 'weight_decay': 0.01,
            'personalization': {'enabled': True, 'early_stopping_patience': 3, 'validation_split': 0.2, 'max_epochs': 5, 'glucose_metrics': True}
        },
        'personalized_moe': {'enabled': True, 'lora_rank': 4, 'lora_alpha': 8.0, 'personalization_weight': 0.05},
        'anomaly_detection': {'enabled': True, 'sequence_length': 12, 'normal_threshold': 0.1, 'anomaly_threshold': 0.3, 'critical_threshold': 0.5},
        'monitoring': {'enabled': True, 'window_size': 200, 'drift_threshold': 0.05},
        'output_dir': str(output_dir).replace('\\', '/'),
        'save_models': True, 'save_plots': False
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='+', required=True, help='List of CSV/JSON data files')
    parser.add_argument('--in_len', type=int, default=12)
    parser.add_argument('--out_len', type=int, default=6)
    parser.add_argument('--run_lora', action='store_true')
    parser.add_argument('--use_patient_ids', action='store_true', help='Group personalization by real patient IDs if available')
    args = parser.parse_args()

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path('TRAIN/outputs') / f'exp_{ts}'
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info('Loading datasets...')
    if args.use_patient_ids:
        sequences, targets, per_patient = aggregate_by_patient(args.inputs, args.in_len, args.out_len)
    else:
        sequences, targets = aggregate_datasets(args.inputs, args.in_len, args.out_len)
        per_patient = {}
    logger.info(f'Total windows: {len(sequences)}')

    if len(sequences) == 0:
        logger.error('No training windows constructed. Abort.')
        return 1

    config = get_default_config(out_dir)
    system = EnhancedGlucosePredictionSystem(config)

    # Base training
    logger.info('Starting base training...')
    base_results = system.train_complete_system(sequences, targets)

    # Save base metrics
    metrics_path = out_dir / 'base_training_results.json'
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(base_results, f, ensure_ascii=False, indent=2)

    # Optional LoRA personalization
    if args.run_lora and config['lora']['personalization']['enabled']:
        logger.info('Setting up LoRA personalization...')
        system.setup_personalization()

        def to_native(obj):
            import numpy as _np
            import torch as _torch
            if isinstance(obj, dict):
                return {k: to_native(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [to_native(v) for v in obj]
            if isinstance(obj, (_np.integer,)):
                return int(obj)
            if isinstance(obj, (_np.floating,)):
                return float(obj)
            if isinstance(obj, _np.ndarray):
                return obj.tolist()
            if isinstance(obj, _torch.Tensor):
                return obj.detach().cpu().tolist()
            return obj

        lora_report: Dict[str, Any] = {}
        if args.use_patient_ids and per_patient:
            for pid, (X, Y) in per_patient.items():
                if len(X) < 16:
                    continue
                logger.info(f'Personalizing real patient {pid} with {len(X)} samples...')
                res = system.personalize_for_patient(str(pid), X, Y, epochs=config['lora']['personalization']['max_epochs'])
                lora_report[str(pid)] = res
        else:
            # Fallback to pseudo groups
            n_groups = 20
            idx = np.arange(len(sequences))
            groups: List[np.ndarray] = np.array_split(idx, n_groups)
            for gid, gidx in enumerate(groups):
                if len(gidx) < 16:
                    continue
                pid = f'patient_{gid:02d}'
                pdata = sequences[gidx]
                ptgt = targets[gidx]
                logger.info(f'Personalizing {pid} with {len(pdata)} samples...')
                res = system.personalize_for_patient(pid, pdata, ptgt, epochs=config['lora']['personalization']['max_epochs'])
                lora_report[pid] = res

        with open(out_dir / 'lora_personalization_results.json', 'w', encoding='utf-8') as f:
            json.dump(to_native(lora_report), f, ensure_ascii=False, indent=2)

    logger.info(f'All done. Results at: {out_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
