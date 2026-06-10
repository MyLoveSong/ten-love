import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import math

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # 脚本将尝试在无pandas时用简化CSV解析


LOW_THRESHOLD = 70.0
HIGH_THRESHOLD = 180.0


def parse_timestamp(ts: Any) -> Optional[datetime]:
    if ts is None or (isinstance(ts, float) and math.isnan(ts)):
        return None
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
        "%m/%d/%Y %H:%M",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M",
    ):
        try:
            return datetime.strptime(str(ts), fmt)
        except Exception:
            continue
    try:
        # ISO 8601 fallback
        return datetime.fromisoformat(str(ts))
    except Exception:
        return None


def read_csv_safely(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if pd is not None:
        try:
            df = pd.read_csv(path)
            rows = df.to_dict(orient="records")
            return rows
        except Exception:
            pass
    # fallback: naive CSV
    import csv
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def read_json_safely(path: str) -> List[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # 尝试常见容器键
            for key in ("data", "records", "items", "samples", "training_data"):
                if key in data and isinstance(data[key], list):
                    return data[key]
            return [data]
    except Exception:
        return []
    return []


def normalize_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    keys = {k.lower(): k for k in rec.keys()}
    get = lambda *alts: next((rec[keys[a]] for a in alts if a in keys), None)

    ts = get("timestamp", "time", "datetime")
    ts_parsed = parse_timestamp(ts)

    glu = get("glucose", "bg", "cgm", "value")
    try:
        glucose = float(glu) if glu is not None and str(glu) != "" else None
    except Exception:
        glucose = None

    pid = get("patient_id", "pid", "user_id", "subject_id")
    meal = get("meal", "food", "carb", "carbs", "meal_flag")
    exercise = get("exercise", "activity", "sport", "exercise_flag")
    sleep = get("sleep", "sleep_flag")

    def to_bool(x: Any) -> Optional[int]:
        if x is None:
            return None
        s = str(x).strip().lower()
        if s in ("1", "true", "yes", "y", "t"): return 1
        if s in ("0", "false", "no", "n", "f"): return 0
        try:
            v = float(s)
            return 1 if v != 0.0 else 0
        except Exception:
            return None

    return {
        "timestamp": ts_parsed,
        "glucose": glucose,
        "patient_id": str(pid) if pid is not None else None,
        "meal": to_bool(meal),
        "exercise": to_bool(exercise),
        "sleep": to_bool(sleep),
        "raw": rec,
    }


def group_by_patient(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        pid = r.get("patient_id") or "unknown"
        buckets[pid].append(r)
    for pid, arr in buckets.items():
        arr.sort(key=lambda x: x.get("timestamp") or datetime.min)
    return buckets


def compute_basic_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    glucose_vals = [r["glucose"] for r in records if r.get("glucose") is not None]
    missing_rate = (total - len(glucose_vals)) / total if total else 0.0
    low_events = sum(1 for v in glucose_vals if v < LOW_THRESHOLD)
    high_events = sum(1 for v in glucose_vals if v > HIGH_THRESHOLD)
    extreme_rate = (low_events + high_events) / len(glucose_vals) if glucose_vals else 0.0

    # 事件覆盖
    def cov(name: str) -> float:
        vals = [r[name] for r in records if r.get(name) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    event_cov = {
        "meal": cov("meal"),
        "exercise": cov("exercise"),
        "sleep": cov("sleep"),
    }

    # 覆盖时长
    ts = [r["timestamp"] for r in records if r.get("timestamp") is not None]
    hours = 0.0
    if ts:
        hours = max(0.0, (max(ts) - min(ts)).total_seconds() / 3600.0)

    return {
        "num_samples": total,
        "missing_rate": missing_rate,
        "extreme_rate": extreme_rate,
        "low_events": low_events,
        "high_events": high_events,
        "event_coverage": event_cov,
        "coverage_hours": hours,
        "coverage_days": hours / 24.0,
    }


def count_trainable_slices(patient_series: List[Dict[str, Any]], in_len: int, out_len: int) -> int:
    # 以有效glucose和有序时间为基准的滑窗计数
    seq = [r for r in patient_series if r.get("glucose") is not None]
    n = len(seq)
    need = in_len + out_len
    return max(0, n - need + 1)


def scan_data_dir(data_dir: str) -> List[Tuple[str, List[Dict[str, Any]]]]:
    collected: List[Tuple[str, List[Dict[str, Any]]]] = []
    for root, _dirs, files in os.walk(data_dir):
        for fn in files:
            path = os.path.join(root, fn)
            lower = fn.lower()
            try:
                if lower.endswith(".csv"):
                    rows = read_csv_safely(path)
                    recs = [normalize_record(r) for r in rows]
                elif lower.endswith(".json"):
                    rows = read_json_safely(path)
                    recs = [normalize_record(r) for r in rows]
                else:
                    continue
                # 至少有glucose或timestamp之一才算
                recs = [r for r in recs if (r.get("glucose") is not None) or (r.get("timestamp") is not None)]
                if recs:
                    collected.append((path, recs))
            except Exception:
                continue
    return collected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--window-in", type=int, default=12)
    parser.add_argument("--window-out", type=int, default=6)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    datasets = scan_data_dir(args.data_dir)

    global_stats = {
        "files_scanned": len(datasets),
        "total_samples": 0,
        "estimated_patients": 0,
        "coverage_hours": 0.0,
        "coverage_days": 0.0,
        "missing_rate": 0.0,
        "extreme_rate": 0.0,
        "low_events": 0,
        "high_events": 0,
        "event_coverage": {"meal": 0.0, "exercise": 0.0, "sleep": 0.0},
        "trainable_slices": 0,
        "window_in": args.window_in,
        "window_out": args.window_out,
        "per_file": [],
    }

    # 汇总
    total_weight = 0
    event_sum = {"meal": 0.0, "exercise": 0.0, "sleep": 0.0}
    patients_global: set = set()

    for path, recs in datasets:
        stats = compute_basic_stats(recs)
        buckets = group_by_patient(recs)
        patients = list(buckets.keys())
        if len(patients) == 1 and patients[0] == "unknown":
            est_patients = 1
        else:
            est_patients = len(patients)

        # 训练切片统计
        slices = sum(count_trainable_slices(buckets[pid], args.window_in, args.window_out) for pid in buckets)

        global_stats["per_file"].append({
            "file": path,
            "num_samples": stats["num_samples"],
            "coverage_hours": stats["coverage_hours"],
            "coverage_days": stats["coverage_days"],
            "missing_rate": stats["missing_rate"],
            "extreme_rate": stats["extreme_rate"],
            "low_events": stats["low_events"],
            "high_events": stats["high_events"],
            "event_coverage": stats["event_coverage"],
            "estimated_patients": est_patients,
            "trainable_slices": slices,
        })

        global_stats["total_samples"] += stats["num_samples"]
        global_stats["coverage_hours"] += stats["coverage_hours"]
        global_stats["coverage_days"] += stats["coverage_days"]
        global_stats["low_events"] += stats["low_events"]
        global_stats["high_events"] += stats["high_events"]
        global_stats["trainable_slices"] += slices

        # 缺失率与极端率做样本数加权平均
        w = max(1, stats["num_samples"])  # 防止0
        total_weight += w
        global_stats["missing_rate"] += stats["missing_rate"] * w
        global_stats["extreme_rate"] += stats["extreme_rate"] * w
        for k in event_sum:
            event_sum[k] += stats["event_coverage"].get(k, 0.0) * w

        for pid in buckets:
            patients_global.add(pid)

    if total_weight > 0:
        global_stats["missing_rate"] /= total_weight
        global_stats["extreme_rate"] /= total_weight
        for k in event_sum:
            global_stats["event_coverage"][k] = event_sum[k] / total_weight

    # 估算患者数量
    if len(patients_global) == 1 and next(iter(patients_global)) == "unknown":
        global_stats["estimated_patients"] = 1
    else:
        # 去除unknown计数
        filtered = {p for p in patients_global if p != "unknown"}
        global_stats["estimated_patients"] = len(filtered) if filtered else 1

    out_path = args.out or os.path.join(args.data_dir, "sample_audit_summary.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(global_stats, f, ensure_ascii=False, indent=2, default=str)

    # 同时输出一个精简摘要，便于快速查看
    brief = {
        "files_scanned": global_stats["files_scanned"],
        "total_samples": global_stats["total_samples"],
        "coverage_hours": round(global_stats["coverage_hours"], 2),
        "coverage_days": round(global_stats["coverage_days"], 2),
        "estimated_patients": global_stats["estimated_patients"],
        "missing_rate": round(global_stats["missing_rate"], 4),
        "extreme_rate": round(global_stats["extreme_rate"], 4),
        "low_events": global_stats["low_events"],
        "high_events": global_stats["high_events"],
        "event_coverage": {k: round(v, 4) for k, v in global_stats["event_coverage"].items()},
        "trainable_slices": global_stats["trainable_slices"],
        "window": [global_stats["window_in"], global_stats["window_out"]],
        "report": out_path,
    }
    print(json.dumps(brief, ensure_ascii=False))


if __name__ == "__main__":
    main()
