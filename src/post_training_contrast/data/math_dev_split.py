from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path


def build_type_level_key(record: dict) -> tuple[str, str]:
    return (
        str(record.get("type", "unknown")),
        str(record.get("level", "unknown")),
    )


def load_jsonl_records(path: str | Path) -> list[dict]:
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl_records(records: list[dict], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _group_records_by_stratum(records: list[dict]) -> dict[tuple[str, str], list[dict]]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        groups[build_type_level_key(record)].append(record)
    return groups


def _allocate_sample_counts(
    groups: dict[tuple[str, str], list[dict]],
    sample_size: int,
    seed: int,
) -> dict[tuple[str, str], int]:
    if sample_size < 0:
        raise ValueError("sample_size 不能小于 0。")
    total = sum(len(items) for items in groups.values())
    if sample_size > total:
        raise ValueError("sample_size 不能超过 records 总数。")

    rng = random.Random(seed)
    strata = sorted(groups)
    allocation = {key: 0 for key in strata}
    if sample_size == 0 or not strata:
        return allocation

    guaranteed = min(sample_size, len(strata))
    if sample_size >= len(strata):
        for key in strata:
            allocation[key] = 1

    remaining = sample_size - guaranteed
    residual_sizes = {
        key: len(groups[key]) - allocation[key]
        for key in strata
    }
    residual_total = sum(residual_sizes.values())
    if remaining <= 0 or residual_total <= 0:
        return allocation

    fractional_parts: list[tuple[float, float, tuple[str, str]]] = []
    for key in strata:
        exact = remaining * residual_sizes[key] / residual_total
        whole = int(exact)
        allocation[key] += whole
        fractional_parts.append((exact - whole, rng.random(), key))

    leftover = sample_size - sum(allocation.values())
    for _, _, key in sorted(fractional_parts, reverse=True)[:leftover]:
        allocation[key] += 1

    return allocation


def sample_stratified_subset(
    records: list[dict],
    sample_size: int,
    seed: int,
) -> list[dict]:
    groups = _group_records_by_stratum(records)
    allocation = _allocate_sample_counts(groups, sample_size, seed)
    rng = random.Random(seed)

    selected: list[dict] = []
    for key in sorted(groups):
        bucket = list(groups[key])
        rng.shuffle(bucket)
        selected.extend(bucket[: allocation[key]])
    rng.shuffle(selected)
    return selected


def split_math_train_records(
    records: list[dict],
    dev_size: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    dev_records = sample_stratified_subset(records, sample_size=dev_size, seed=seed)
    dev_ids = {record["id"] for record in dev_records}
    train_records = [record for record in records if record["id"] not in dev_ids]
    return train_records, dev_records
