import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PARTITIONS = ("train", "val", "test")


def parse_timestamp(value: Any) -> datetime:
    if value is None:
        raise ValueError("missing timestamp")
    text = str(value).strip()
    if not text:
        raise ValueError("missing timestamp")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
    raise ValueError(f"invalid timestamp: {value!r}")


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def stable_hash(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def load_records(path: Path) -> List[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict) and isinstance(data.get("records"), list):
        records = data["records"]
    else:
        raise ValueError("input JSON must be a list or contain a records list")
    if not all(isinstance(record, Mapping) for record in records):
        raise ValueError("all records must be JSON objects")
    return records


def allocate_partition_counts(
    group_count: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Dict[str, int]:
    ratios = [train_ratio, val_ratio, test_ratio]
    if group_count < 0:
        raise ValueError("group_count must be non-negative")
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("split ratios must be non-negative")
    if sum(ratios) <= 0:
        raise ValueError("at least one split ratio must be positive")

    normalized = [ratio / sum(ratios) for ratio in ratios]
    raw = [group_count * ratio for ratio in normalized]
    counts = [math.floor(value) for value in raw]
    remaining = group_count - sum(counts)
    order = sorted(range(len(raw)), key=lambda idx: (raw[idx] - counts[idx], normalized[idx]), reverse=True)
    for idx in order[:remaining]:
        counts[idx] += 1

    positive = [idx for idx, ratio in enumerate(ratios) if ratio > 0]
    if group_count >= len(positive):
        for idx in positive:
            if counts[idx] == 0:
                donor = max((j for j in positive if counts[j] > 1), key=lambda j: counts[j])
                counts[donor] -= 1
                counts[idx] = 1

    return dict(zip(PARTITIONS, counts))


def assign_partitions(group_keys: Sequence[Tuple[str, str]], seed: int, counts: Mapping[str, int]) -> Dict[Tuple[str, str], str]:
    keys = list(group_keys)
    random.Random(seed).shuffle(keys)

    assignments: Dict[Tuple[str, str], str] = {}
    cursor = 0
    for partition in PARTITIONS:
        count = counts[partition]
        for key in keys[cursor:cursor + count]:
            assignments[key] = partition
        cursor += count
    return assignments


def build_groups(
    records: Iterable[Mapping[str, Any]],
    input_horizon: int,
    output_horizon: int,
) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], Counter]:
    if input_horizon <= 0 or output_horizon <= 0:
        raise ValueError("input_horizon and output_horizon must be positive")

    duplicate_keys = Counter()
    seen_keys = set()
    buckets: Dict[Tuple[str, str], List[datetime]] = defaultdict(list)
    source_counts: Counter = Counter()

    for idx, record in enumerate(records):
        source = str(record.get("source") or "").strip()
        patient_id = str(record.get("patient_id") or "").strip()
        if not source:
            raise ValueError(f"record {idx} missing source")
        if not patient_id:
            raise ValueError(f"record {idx} missing patient_id")
        timestamp = parse_timestamp(record.get("timestamp"))
        key = (source, patient_id, timestamp.isoformat())
        if key in seen_keys:
            duplicate_keys[(source, patient_id, timestamp.isoformat())] += 1
        seen_keys.add(key)
        buckets[(source, patient_id)].append(timestamp)
        source_counts[source] += 1

    if duplicate_keys:
        raise ValueError(f"duplicate source+patient+timestamp groups: {len(duplicate_keys)}")

    window_size = input_horizon + output_horizon
    groups: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for group_key, timestamps in buckets.items():
        timestamps_sorted = sorted(timestamps)
        record_count = len(timestamps_sorted)
        window_count = max(0, record_count - window_size + 1)
        groups[group_key] = {
            "source": group_key[0],
            "record_count": record_count,
            "timestamp_min": timestamps_sorted[0].isoformat(),
            "timestamp_max": timestamps_sorted[-1].isoformat(),
            "window_count": window_count,
        }
    return groups, source_counts


def build_split_manifest(
    input_path: Path,
    output_path: Optional[Path] = None,
    *,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    input_horizon: int = 12,
    output_horizon: int = 6,
    generated_at: Optional[str] = None,
    dataset_label: Optional[str] = None,
) -> Dict[str, Any]:
    path = Path(input_path)
    records = load_records(path)
    groups, source_record_counts = build_groups(records, input_horizon, output_horizon)
    sorted_group_keys = sorted(groups)
    counts = allocate_partition_counts(len(sorted_group_keys), train_ratio, val_ratio, test_ratio)
    assignments = assign_partitions(sorted_group_keys, seed, counts)

    partition_stats = {
        partition: {
            "group_count": 0,
            "record_count": 0,
            "window_count": 0,
        }
        for partition in PARTITIONS
    }
    group_entries = []
    source_group_counts: Counter = Counter()

    for source, patient_id in sorted_group_keys:
        group = groups[(source, patient_id)]
        partition = assignments[(source, patient_id)]
        partition_stats[partition]["group_count"] += 1
        partition_stats[partition]["record_count"] += group["record_count"]
        partition_stats[partition]["window_count"] += group["window_count"]
        source_group_counts[source] += 1
        group_entries.append(
            {
                "group_hash": stable_hash(source, patient_id),
                "source": source,
                "partition": partition,
                "record_count": group["record_count"],
                "window_count": group["window_count"],
                "timestamp_min": group["timestamp_min"],
                "timestamp_max": group["timestamp_max"],
            }
        )

    manifest = {
        "schema_version": "glucose-source-aware-split-manifest/v1",
        "generated_at": generated_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "dataset": {
            "path": dataset_label or str(path),
            "sha256": sha256_file(path),
            "record_count": len(records),
            "source_count": len(source_record_counts),
            "source_patient_group_count": len(sorted_group_keys),
        },
        "split_policy": {
            "type": "group_disjoint",
            "group_key": "source + patient_id",
            "seed": seed,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "input_horizon": input_horizon,
            "output_horizon": output_horizon,
            "window_boundary": "windows are counted after group partition assignment; no group crosses partitions",
        },
        "normalization_scope": "train_only_required",
        "privacy_boundary": "no row-level glucose values or raw patient identifiers are included",
        "sources": [
            {
                "source": source,
                "group_count": source_group_counts[source],
                "record_count": source_record_counts[source],
            }
            for source in sorted(source_record_counts)
        ],
        "partitions": partition_stats,
        "groups": group_entries,
    }

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a source-aware glucose split manifest without row-level values.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--input-horizon", type=int, default=12)
    parser.add_argument("--output-horizon", type=int, default=6)
    parser.add_argument("--dataset-label", default=None)
    args = parser.parse_args()

    build_split_manifest(
        args.input,
        args.output,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        input_horizon=args.input_horizon,
        output_horizon=args.output_horizon,
        dataset_label=args.dataset_label,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
