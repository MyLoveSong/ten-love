import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

from analysis.source_aware_split_manifest import load_records, parse_timestamp, sha256_file, stable_hash


PARTITIONS = ("train", "val", "test")
VALUE_KEYS = ("glucose_mg_dl", "glucose", "blood_glucose", "bg", "value")


def load_manifest(path: Path) -> Mapping[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, Mapping):
        raise ValueError("split manifest must be a JSON object")
    return manifest


def extract_glucose_value(record: Mapping[str, Any], value_key: Optional[str] = None) -> float:
    keys = (value_key,) if value_key else VALUE_KEYS
    for key in keys:
        if key and key in record and record[key] not in (None, ""):
            try:
                return float(record[key])
            except (TypeError, ValueError):
                continue
    raise ValueError("record has no usable glucose value")


def build_manifest_partition_map(manifest: Mapping[str, Any]) -> Dict[str, str]:
    groups = manifest.get("groups")
    if not isinstance(groups, list):
        raise ValueError("split manifest missing groups list")

    partitions: Dict[str, str] = {}
    for idx, group in enumerate(groups):
        if not isinstance(group, Mapping):
            raise ValueError(f"manifest group {idx} is not an object")
        group_hash = str(group.get("group_hash") or "")
        partition = str(group.get("partition") or "")
        if not group_hash:
            raise ValueError(f"manifest group {idx} missing group_hash")
        if partition not in PARTITIONS:
            raise ValueError(f"manifest group {idx} has invalid partition {partition!r}")
        partitions[group_hash] = partition
    return partitions


def build_partitioned_series(
    records: Iterable[Mapping[str, Any]],
    group_partitions: Mapping[str, str],
    value_key: Optional[str] = None,
) -> Dict[str, List[List[Tuple[str, float]]]]:
    buckets: MutableMapping[str, List[Tuple[str, float]]] = defaultdict(list)
    for idx, record in enumerate(records):
        source = str(record.get("source") or "").strip()
        patient_id = str(record.get("patient_id") or "").strip()
        if not source:
            raise ValueError(f"record {idx} missing source")
        if not patient_id:
            raise ValueError(f"record {idx} missing patient_id")
        group_hash = stable_hash(source, patient_id)
        if group_hash not in group_partitions:
            raise ValueError(f"record {idx} group is absent from split manifest")
        timestamp = parse_timestamp(record.get("timestamp")).isoformat()
        value = extract_glucose_value(record, value_key)
        buckets[group_hash].append((timestamp, value))

    partitioned: Dict[str, List[List[Tuple[str, float]]]] = {partition: [] for partition in PARTITIONS}
    for group_hash, series in buckets.items():
        partition = group_partitions[group_hash]
        partitioned[partition].append(sorted(series, key=lambda item: item[0]))
    return partitioned


def make_windows(series: List[Tuple[str, float]], input_horizon: int, output_horizon: int) -> Tuple[List[List[List[float]]], List[List[float]]]:
    window_size = input_horizon + output_horizon
    sequences: List[List[List[float]]] = []
    targets: List[List[float]] = []
    values = [value for _timestamp, value in series]
    for start in range(0, len(values) - window_size + 1):
        input_values = values[start:start + input_horizon]
        target_values = values[start + input_horizon:start + window_size]
        sequences.append([[value] for value in input_values])
        targets.append(target_values)
    return sequences, targets


def build_raw_split_windows(
    partitioned_series: Mapping[str, List[List[Tuple[str, float]]]],
    input_horizon: int,
    output_horizon: int,
) -> Tuple[Dict[str, List[List[List[float]]]], Dict[str, List[List[float]]]]:
    split_sequences = {partition: [] for partition in PARTITIONS}
    split_targets = {partition: [] for partition in PARTITIONS}
    for partition in PARTITIONS:
        for series in partitioned_series.get(partition, []):
            sequences, targets = make_windows(series, input_horizon, output_horizon)
            split_sequences[partition].extend(sequences)
            split_targets[partition].extend(targets)
    return split_sequences, split_targets


def flatten_sequences(sequences: Iterable[List[List[float]]]) -> List[float]:
    values: List[float] = []
    for sequence in sequences:
        for row in sequence:
            values.extend(row)
    return values


def fit_train_sequence_scaler(train_sequences: List[List[List[float]]]) -> Dict[str, float]:
    values = flatten_sequences(train_sequences)
    if not values:
        raise ValueError("cannot fit scaler without train windows")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = variance ** 0.5
    if std < 1e-12:
        std = 1.0
    return {
        "mean": mean,
        "std": std,
        "scope": "train_sequences_only",
    }


def scale_sequences(sequences: List[List[List[float]]], scaler: Mapping[str, float]) -> List[List[List[float]]]:
    mean = float(scaler["mean"])
    std = float(scaler["std"])
    return [
        [[(value - mean) / std for value in row] for row in sequence]
        for sequence in sequences
    ]


def scale_targets(targets: List[List[float]], scaler: Mapping[str, float]) -> List[List[float]]:
    mean = float(scaler["mean"])
    std = float(scaler["std"])
    return [[(value - mean) / std for value in target] for target in targets]


def summarize_partitions(
    sequences: Mapping[str, List[List[List[float]]]],
    targets: Mapping[str, List[List[float]]],
) -> Dict[str, Dict[str, int]]:
    return {
        partition: {
            "window_count": len(sequences.get(partition, [])),
            "target_count": len(targets.get(partition, [])),
        }
        for partition in PARTITIONS
    }


def load_source_aware_split_windows(
    dataset_path: Path,
    manifest_path: Path,
    *,
    input_horizon: Optional[int] = None,
    output_horizon: Optional[int] = None,
    value_key: Optional[str] = None,
    verify_hash: bool = True,
) -> Dict[str, Any]:
    dataset = Path(dataset_path)
    manifest_file = Path(manifest_path)
    manifest = load_manifest(manifest_file)

    expected_hash = manifest.get("dataset", {}).get("sha256") if isinstance(manifest.get("dataset"), Mapping) else None
    if verify_hash and expected_hash:
        observed_hash = sha256_file(dataset)
        if observed_hash != expected_hash:
            raise ValueError("dataset SHA-256 does not match split manifest")

    split_policy = manifest.get("split_policy", {})
    if not isinstance(split_policy, Mapping):
        split_policy = {}
    in_len = input_horizon or int(split_policy.get("input_horizon", 12))
    out_len = output_horizon or int(split_policy.get("output_horizon", 6))

    records = load_records(dataset)
    group_partitions = build_manifest_partition_map(manifest)
    partitioned_series = build_partitioned_series(records, group_partitions, value_key=value_key)
    raw_sequences, raw_targets = build_raw_split_windows(partitioned_series, in_len, out_len)
    scaler = fit_train_sequence_scaler(raw_sequences["train"])
    scaled_sequences = {partition: scale_sequences(raw_sequences[partition], scaler) for partition in PARTITIONS}
    scaled_targets = {partition: scale_targets(raw_targets[partition], scaler) for partition in PARTITIONS}

    return {
        "sequences": scaled_sequences,
        "targets": scaled_targets,
        "raw_sequences": raw_sequences,
        "raw_targets": raw_targets,
        "scaler": scaler,
        "metadata": {
            "dataset_path": str(dataset),
            "manifest_path": str(manifest_file),
            "group_key": "source + patient_id",
            "input_horizon": in_len,
            "output_horizon": out_len,
            "normalization_scope": "train_sequences_only",
            "partitions": summarize_partitions(raw_sequences, raw_targets),
        },
    }
