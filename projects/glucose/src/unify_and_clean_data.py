import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_ROOT = Path('TRAIN/data')
OUTPUT_DIR = DATA_ROOT / 'cleaned_dataset'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ID_KEYS = ['patient_id', 'pid', 'user_id', 'subject', 'subject_id', 'patient', 'id']
GLU_KEYS = ['glucose', 'bg', 'blood_glucose', 'value', 'glucose_value']
TIME_KEYS = ['timestamp', 'time', 'datetime', 'date']


def read_json(path: Path) -> List[Dict[str, Any]]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            for k in ['records', 'data', 'items', 'rows']:
                if k in data and isinstance(data[k], list):
                    return data[k]
            return []
        if isinstance(data, list):
            return data
        return []
    except Exception as e:
        logger.warning(f'JSON read failed: {path} - {e}')
        return []


def read_csv(path: Path) -> List[Dict[str, Any]]:
    import csv
    rows: List[Dict[str, Any]] = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except Exception as e:
        logger.warning(f'CSV read failed: {path} - {e}')
    return rows


def coerce_float(x: Any) -> Optional[float]:
    if x in (None, "", "NaN"):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return float(str(x).strip())
    except Exception:
        return None


def pick_patient_id(rec: Dict[str, Any]) -> str:
    for k in ID_KEYS:
        if k in rec and rec[k] not in (None, ""):
            return str(rec[k])
    return 'unknown'


def pick_glucose(rec: Dict[str, Any]) -> Optional[float]:
    # prefer explicit keys
    for k in GLU_KEYS:
        if k in rec:
            v = coerce_float(rec[k])
            if v is not None:
                return v
    # fallback: find any plausible numeric value
    best = None
    for k, v in rec.items():
        num = coerce_float(v)
        if num is None:
            continue
        if 30 <= num <= 500:
            best = num
            break
    return best


def pick_timestamp(rec: Dict[str, Any]) -> Optional[str]:
    for k in TIME_KEYS:
        if k in rec and rec[k] not in (None, ""):
            return str(rec[k])
    return None


def clean_series(values: List[float]) -> List[float]:
    if not values:
        return []
    arr = np.array(values, dtype=np.float32)
    # clamp to physiological range
    arr = np.clip(arr, 40, 400)
    # forward fill NaN (if any)
    if np.isnan(arr).any():
        mask = np.isnan(arr)
        idx = np.where(~mask, np.arange(mask.size), 0)
        np.maximum.accumulate(idx, out=idx)
        arr = arr[idx]
    # simple spike removal via median filter (window 3)
    med = np.copy(arr)
    for i in range(1, len(arr)-1):
        window = np.array([arr[i-1], arr[i], arr[i+1]])
        med[i] = float(np.median(window))
    # blend 70% original + 30% median-filtered to keep trends
    arr = 0.7 * arr + 0.3 * med
    return arr.astype(np.float32).tolist()


def unify(root: Path) -> Tuple[Path, Path]:
    """Scan root, unify to CSV/JSON with columns: patient_id,timestamp,glucose"""
    logger.info(f'Scanning data under {root} ...')
    records: List[Dict[str, Any]] = []
    for path in root.rglob('*'):
        if not path.is_file():
            continue
        suf = path.suffix.lower()
        if suf not in ['.json', '.csv', '.tsv']:
            continue
        rows: List[Dict[str, Any]] = []
        if suf == '.json':
            rows = read_json(path)
        else:
            rows = read_csv(path)
        if not rows:
            continue
        count_before = len(records)
        for r in rows:
            pid = pick_patient_id(r)
            g = pick_glucose(r)
            if g is None:
                continue
            ts = pick_timestamp(r)
            records.append({'patient_id': pid, 'timestamp': ts, 'glucose': g})
        logger.info(f'Loaded {len(records) - count_before} recs from {path}')

    if not records:
        raise RuntimeError('No valid glucose records found.')

    # group by patient and clean series order-preserving (no timestamp sorting since unknown)
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        grouped[r['patient_id']].append(r)

    cleaned: List[Dict[str, Any]] = []
    for pid, recs in grouped.items():
        series = [r['glucose'] for r in recs]
        series_clean = clean_series(series)
        for i, r in enumerate(recs[:len(series_clean)]):
            cleaned.append({'patient_id': pid, 'timestamp': r['timestamp'], 'glucose': float(series_clean[i])})

    # write CSV
    csv_path = OUTPUT_DIR / 'unified_cleaned_glucose.csv'
    import csv
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['patient_id', 'timestamp', 'glucose'])
        writer.writeheader()
        writer.writerows(cleaned)

    # write JSON
    json_path = OUTPUT_DIR / 'unified_cleaned_glucose.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({'records': cleaned}, f, ensure_ascii=False, indent=2)

    logger.info(f'Unified CSV: {csv_path}')
    logger.info(f'Unified JSON: {json_path}')
    return csv_path, json_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=str(DATA_ROOT), help='Root data directory to scan')
    args = parser.parse_args()

    csv_path, json_path = unify(Path(args.root))
    print(str(csv_path))
    print(str(json_path))


if __name__ == '__main__':
    main()
