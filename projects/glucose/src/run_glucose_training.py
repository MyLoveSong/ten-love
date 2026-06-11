import os

DEFAULT_CUBLAS_WORKSPACE_CONFIG = ':4096:8'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', DEFAULT_CUBLAS_WORKSPACE_CONFIG)

import json
import argparse
import logging
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, DefaultDict
from collections import defaultdict

import numpy as np
import torch

from analysis.source_aware_split_dataset import load_source_aware_split_windows
from enhanced_glucose_system import EnhancedGlucosePredictionSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


ID_CANDIDATE_KEYS = [
    'patient_id', 'pid', 'user_id', 'subject', 'subject_id', 'patient', 'id'
]


def set_reproducible_seed(seed: int) -> Dict[str, Any]:
    if seed < 0:
        raise ValueError('--seed must be non-negative')

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    torch_cuda_seed = None
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch_cuda_seed = seed

    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    return {
        'value': seed,
        'python_random': seed,
        'python_hash_seed_env': os.environ.get('PYTHONHASHSEED'),
        'python_hash_seed_scope': 'runtime_environment_record',
        'numpy': seed,
        'torch': seed,
        'torch_cuda': torch_cuda_seed,
        'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'deterministic_algorithms': True,
        'deterministic_warn_only': True,
        'cudnn_benchmark': False,
        'cudnn_deterministic': True,
    }


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


def split_to_tensor_dict(split_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    tensor_dict = {
        'train_sequences': torch.FloatTensor(np.asarray(split_data['sequences']['train'], dtype=np.float32)),
        'train_targets': torch.FloatTensor(np.asarray(split_data['targets']['train'], dtype=np.float32)),
        'val_sequences': torch.FloatTensor(np.asarray(split_data['sequences']['val'], dtype=np.float32)),
        'val_targets': torch.FloatTensor(np.asarray(split_data['targets']['val'], dtype=np.float32)),
        'test_sequences': torch.FloatTensor(np.asarray(split_data['sequences']['test'], dtype=np.float32)),
        'test_targets': torch.FloatTensor(np.asarray(split_data['targets']['test'], dtype=np.float32)),
    }
    for key, value in tensor_dict.items():
        if value.numel() == 0:
            raise ValueError(f"Split-aware training requires non-empty {key}")
    return tensor_dict


def inverse_scale_array(values: np.ndarray, scaler: Dict[str, float]) -> np.ndarray:
    return values * float(scaler["std"]) + float(scaler["mean"])


def evaluate_inverse_scaled_predictions(
    y_true_scaled: np.ndarray,
    y_pred_scaled: np.ndarray,
    scaler: Dict[str, float],
) -> Dict[str, Any]:
    y_true = inverse_scale_array(np.asarray(y_true_scaled, dtype=np.float64), scaler)
    y_pred = inverse_scale_array(np.asarray(y_pred_scaled, dtype=np.float64), scaler)
    diff = y_true - y_pred
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    per_horizon: Dict[str, Dict[str, float]] = {}
    if y_true.ndim == 2:
        for idx in range(y_true.shape[1]):
            step_diff = y_true[:, idx] - y_pred[:, idx]
            step_mse = float(np.mean(step_diff ** 2))
            per_horizon[f"t+{idx + 1}"] = {
                "mae": float(np.mean(np.abs(step_diff))),
                "rmse": float(np.sqrt(step_mse)),
            }

    return {
        "unit": "mg/dL",
        "scale_scope": scaler.get("scope", "train_sequences_only"),
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "per_horizon": per_horizon,
    }


def evaluate_inverse_scaled_ensemble(
    system: EnhancedGlucosePredictionSystem,
    sequences: torch.Tensor,
    targets: torch.Tensor,
    scaler: Dict[str, float],
) -> Dict[str, Any]:
    device = getattr(system, "device", None)
    eval_sequences = sequences.to(device) if device is not None else sequences
    with torch.no_grad():
        predictions = system.ensemble.predict(eval_sequences)
    return evaluate_inverse_scaled_predictions(
        targets.detach().cpu().numpy(),
        predictions.detach().cpu().numpy(),
        scaler,
    )


def limit_split_windows(split_data: Dict[str, Any], max_windows_per_split: int | None) -> Dict[str, Any]:
    if max_windows_per_split is None:
        return split_data
    if max_windows_per_split <= 0:
        raise ValueError('--max_windows_per_split must be positive')

    limited = dict(split_data)
    for section in ('sequences', 'targets', 'raw_sequences', 'raw_targets'):
        if section in split_data:
            limited[section] = {
                partition: values[:max_windows_per_split]
                for partition, values in split_data[section].items()
            }
    metadata = dict(split_data.get('metadata', {}))
    metadata['training_scope'] = 'smoke_subset'
    metadata['max_windows_per_split'] = max_windows_per_split
    metadata['effective_partitions'] = {
        partition: {
            'window_count': len(limited['sequences'][partition]),
            'target_count': len(limited['targets'][partition]),
        }
        for partition in ('train', 'val', 'test')
    }
    limited['metadata'] = metadata
    return limited


def parse_training_models(models_arg: str | None, available_models: Dict[str, Any]) -> List[str] | None:
    if not models_arg:
        return None
    requested = [name.strip() for name in models_arg.split(',') if name.strip()]
    if not requested:
        raise ValueError('--models must name at least one model')
    invalid = sorted(set(requested) - set(available_models))
    if invalid:
        raise ValueError(f"Unsupported model(s): {', '.join(invalid)}")
    return requested


def train_with_split_manifest(
    dataset_path: Path,
    manifest_path: Path,
    in_len: int,
    out_len: int,
    out_dir: Path,
    max_windows_per_split: int | None = None,
    epochs: int | None = None,
    model_names: List[str] | None = None,
    seed: int = 42,
) -> Dict[str, Any]:
    seed_record = set_reproducible_seed(seed)
    logger.info("Loading source-aware split manifest...")
    split_data = load_source_aware_split_windows(
        dataset_path,
        manifest_path,
        input_horizon=in_len,
        output_horizon=out_len,
    )
    split_data = limit_split_windows(split_data, max_windows_per_split)
    data_dict = split_to_tensor_dict(split_data)

    config = get_default_config(out_dir)
    config['seed'] = seed_record
    if epochs is not None:
        if epochs <= 0:
            raise ValueError('--epochs must be positive')
        config['training']['epochs'] = epochs
        config['training']['patience'] = min(config['training']['patience'], epochs)
        config['training']['lr_scheduler']['warmup_epochs'] = min(
            config['training']['lr_scheduler']['warmup_epochs'],
            epochs,
        )
        config['training']['early_stopping']['burn_in_epochs'] = min(
            config['training']['early_stopping']['burn_in_epochs'],
            epochs,
        )
    if model_names is not None:
        config['models'] = {name: config['models'][name] for name in model_names}
    config['augmentation']['enabled'] = False
    config['rare_event_augmentation']['enabled'] = False
    config['feature_engineering']['enabled'] = False
    config['monitoring']['enabled'] = False
    config['anomaly_detection']['enabled'] = False

    system = EnhancedGlucosePredictionSystem(config)
    system._update_model_input_dims(int(data_dict['train_sequences'].shape[-1]))
    system.create_models()
    training_results = system.train_models(data_dict)
    system.create_ensemble(data_dict)
    system.setup_monitoring(data_dict['train_sequences'])
    test_metrics = system.evaluate_ensemble(data_dict['test_sequences'], data_dict['test_targets'])
    val_metrics_inverse_scaled = evaluate_inverse_scaled_ensemble(
        system,
        data_dict['val_sequences'],
        data_dict['val_targets'],
        split_data['scaler'],
    )
    test_metrics_inverse_scaled = evaluate_inverse_scaled_ensemble(
        system,
        data_dict['test_sequences'],
        data_dict['test_targets'],
        split_data['scaler'],
    )
    system.save_system_state()
    system.is_trained = True

    results = {
        'training_mode': 'source_aware_split_manifest',
        'split_manifest': str(manifest_path),
        'dataset': str(dataset_path),
        'split_metadata': split_data['metadata'],
        'scaler': split_data['scaler'],
        'training_scope': 'smoke_subset' if max_windows_per_split is not None else 'full_split',
        'max_windows_per_split': max_windows_per_split,
        'epochs': config['training']['epochs'],
        'seed': seed_record,
        'models': list(config['models'].keys()),
        'individual_models': training_results,
        'test_metrics': test_metrics,
        'val_metrics_inverse_scaled': val_metrics_inverse_scaled,
        'test_metrics_inverse_scaled': test_metrics_inverse_scaled,
        'ensemble_weights': system.ensemble.get_model_weights(),
        'training_completed': True,
        'timestamp': datetime.now().isoformat(),
    }
    metrics_path = out_dir / 'split_manifest_training_results.json'
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(to_native(results), f, ensure_ascii=False, indent=2)
    logger.info(f"Split-aware results saved: {metrics_path}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='+', default=[], help='List of CSV/JSON data files')
    parser.add_argument('--split_manifest', type=str, default=None, help='Source-aware split manifest JSON')
    parser.add_argument('--split_dataset', type=str, default=None, help='Dataset JSON matching --split_manifest')
    parser.add_argument('--in_len', type=int, default=12)
    parser.add_argument('--out_len', type=int, default=6)
    parser.add_argument('--max_windows_per_split', type=int, default=None, help='Deterministic smoke limit per split')
    parser.add_argument('--epochs', type=int, default=None, help='Override training epochs for split-manifest runs')
    parser.add_argument('--models', type=str, default=None, help='Comma-separated model names for split-manifest runs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for split-manifest runs')
    parser.add_argument('--run_lora', action='store_true')
    parser.add_argument('--use_patient_ids', action='store_true', help='Group personalization by real patient IDs if available')
    args = parser.parse_args()

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path('TRAIN/outputs') / f'exp_{ts}'
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.split_manifest:
        dataset_arg = args.split_dataset or (args.inputs[0] if args.inputs else None)
        if not dataset_arg:
            parser.error('--split_manifest requires --split_dataset or one --inputs path')
        train_with_split_manifest(
            Path(dataset_arg),
            Path(args.split_manifest),
            args.in_len,
            args.out_len,
            out_dir,
            max_windows_per_split=args.max_windows_per_split,
            epochs=args.epochs,
            model_names=parse_training_models(args.models, get_default_config(out_dir)['models']),
            seed=args.seed,
        )
        logger.info(f'All done. Results at: {out_dir}')
        return 0

    if not args.inputs:
        parser.error('--inputs is required unless --split_manifest is provided')

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
