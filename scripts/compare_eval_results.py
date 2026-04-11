"""
compare_eval_results.py - 对比两份 Math500 评测 JSON 的逐题差异。

默认用于比较当前工作区里的 GRPO / GSPO 结果：
    python3 scripts/compare_eval_results.py
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_LEFT_PATH = "outputs/eval/GRPO/math500.json"
DEFAULT_RIGHT_PATH = "outputs/eval/gspo/math500.json"
DEFAULT_DATASET_PATH = "datasets/processed/math500_test/test.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/eval/comparisons/grpo_vs_gspo_math500"


SUMMARY_KEYS = [
    "dataset_name",
    "num_samples",
    "accuracy",
    "format_fail_rate",
    "truncation_rate",
    "mean_length",
    "median_length",
    "p90_length",
    "num_correct",
    "num_format_fail",
    "num_truncated",
]


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_eval_details(path: str | Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[str]]:
    data = _load_json(path)
    details = data.get("details")
    if not isinstance(details, list):
        raise ValueError(f"{path} 缺少 details 列表")

    by_id: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for index, item in enumerate(details):
        item_id = item.get("id") or f"index-{index}"
        if item_id in by_id:
            raise ValueError(f"{path} 存在重复 id: {item_id}")
        by_id[item_id] = item
        order.append(item_id)
    return data, by_id, order


def _load_dataset_records(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    dataset_path = Path(path)
    if not dataset_path.exists():
        return {}

    records: dict[str, dict[str, Any]] = {}
    with open(dataset_path, "r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record_id = record.get("id") or f"index-{index}"
            records[record_id] = record
    return records


def _summary_snapshot(data: dict[str, Any]) -> dict[str, Any]:
    return {key: data.get(key) for key in SUMMARY_KEYS if key in data}


def _model_view(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "is_correct": bool(item.get("is_correct", False)),
        "format_failed": bool(item.get("format_failed", False)),
        "is_truncated": bool(item.get("is_truncated", False)),
        "response_length": item.get("response_length"),
        "parsed_prediction": item.get("parsed_prediction", ""),
        "raw_prediction": item.get("raw_prediction", ""),
    }


def _comparison_item(
    item_id: str,
    left_item: dict[str, Any],
    right_item: dict[str, Any],
    left_name: str,
    right_name: str,
    dataset_records: dict[str, dict[str, Any]],
    include_solution: bool,
) -> dict[str, Any]:
    dataset_record = dataset_records.get(item_id, {})
    output = {
        "id": item_id,
        "unique_id": dataset_record.get("unique_id"),
        "level": left_item.get("level", right_item.get("level")),
        "subject": left_item.get("subject", right_item.get("subject")),
        "problem": dataset_record.get("problem"),
        "ground_truth": left_item.get("ground_truth", right_item.get("ground_truth")),
        left_name: _model_view(left_item),
        right_name: _model_view(right_item),
    }
    if include_solution:
        output["solution"] = dataset_record.get("solution")
    return output


def _empty_slice_stats() -> dict[str, int]:
    return {"total": 0, "left_correct": 0, "right_correct": 0, "diff": 0}


def _add_slice_stat(
    table: dict[str, dict[str, int]],
    key: Any,
    left_correct: bool,
    right_correct: bool,
) -> None:
    label = str(key) if key is not None else "unknown"
    row = table[label]
    row["total"] += 1
    row["left_correct"] += int(left_correct)
    row["right_correct"] += int(right_correct)
    row["diff"] = row["right_correct"] - row["left_correct"]


def _finalize_slice_stats(table: dict[str, dict[str, int]]) -> dict[str, dict[str, Any]]:
    finalized: dict[str, dict[str, Any]] = {}
    for key in sorted(table, key=lambda value: (str(value))):
        row = table[key]
        total = max(1, row["total"])
        finalized[key] = {
            **row,
            "left_accuracy": row["left_correct"] / total,
            "right_accuracy": row["right_correct"] / total,
        }
    return finalized


def compare_eval_results(
    left_path: str | Path = DEFAULT_LEFT_PATH,
    right_path: str | Path = DEFAULT_RIGHT_PATH,
    left_name: str = "grpo",
    right_name: str = "gspo",
    dataset_path: str | Path | None = DEFAULT_DATASET_PATH,
    include_solution: bool = False,
) -> dict[str, Any]:
    left_data, left_by_id, left_order = _load_eval_details(left_path)
    right_data, right_by_id, _ = _load_eval_details(right_path)
    dataset_records = _load_dataset_records(dataset_path)

    left_ids = set(left_by_id)
    right_ids = set(right_by_id)
    common_ids = [item_id for item_id in left_order if item_id in right_ids]
    missing_from_left = sorted(right_ids - left_ids)
    missing_from_right = sorted(left_ids - right_ids)

    categories: dict[str, list[dict[str, Any]]] = {
        "both_correct": [],
        "both_wrong": [],
        "left_only_correct": [],
        "right_only_correct": [],
    }
    by_level: dict[str, dict[str, int]] = defaultdict(_empty_slice_stats)
    by_subject: dict[str, dict[str, int]] = defaultdict(_empty_slice_stats)

    for item_id in common_ids:
        left_item = left_by_id[item_id]
        right_item = right_by_id[item_id]
        left_correct = bool(left_item.get("is_correct", False))
        right_correct = bool(right_item.get("is_correct", False))
        if left_correct and right_correct:
            category = "both_correct"
        elif left_correct and not right_correct:
            category = "left_only_correct"
        elif right_correct and not left_correct:
            category = "right_only_correct"
        else:
            category = "both_wrong"

        categories[category].append(
            _comparison_item(
                item_id=item_id,
                left_item=left_item,
                right_item=right_item,
                left_name=left_name,
                right_name=right_name,
                dataset_records=dataset_records,
                include_solution=include_solution,
            )
        )
        _add_slice_stat(
            by_level,
            left_item.get("level", right_item.get("level")),
            left_correct,
            right_correct,
        )
        _add_slice_stat(
            by_subject,
            left_item.get("subject", right_item.get("subject")),
            left_correct,
            right_correct,
        )

    counts = {name: len(items) for name, items in categories.items()}
    counts["total_common"] = len(common_ids)
    counts["different_correctness"] = (
        counts["left_only_correct"] + counts["right_only_correct"]
    )

    return {
        "left_name": left_name,
        "right_name": right_name,
        "left_path": str(left_path),
        "right_path": str(right_path),
        "dataset_path": str(dataset_path) if dataset_path is not None else None,
        "left_summary": _summary_snapshot(left_data),
        "right_summary": _summary_snapshot(right_data),
        "counts": counts,
        "missing_from_left": missing_from_left,
        "missing_from_right": missing_from_right,
        "by_level": _finalize_slice_stats(by_level),
        "by_subject": _finalize_slice_stats(by_subject),
        "categories": categories,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")


def _write_summary_markdown(path: Path, report: dict[str, Any]) -> None:
    left = report["left_name"]
    right = report["right_name"]
    counts = report["counts"]
    lines = [
        f"# {left.upper()} vs {right.upper()} Eval Comparison",
        "",
        "## Overall",
        "",
        f"- Common samples: {counts['total_common']}",
        f"- Both correct: {counts['both_correct']}",
        f"- Both wrong: {counts['both_wrong']}",
        f"- {left} only correct: {counts['left_only_correct']}",
        f"- {right} only correct: {counts['right_only_correct']}",
        f"- Different correctness: {counts['different_correctness']}",
        "",
        "## Summary Metrics",
        "",
        f"| Metric | {left} | {right} | Delta ({right}-{left}) |",
        "|---|---:|---:|---:|",
    ]
    for key in (
        "accuracy",
        "format_fail_rate",
        "truncation_rate",
        "mean_length",
        "median_length",
        "p90_length",
        "num_correct",
        "num_format_fail",
        "num_truncated",
    ):
        left_value = report["left_summary"].get(key)
        right_value = report["right_summary"].get(key)
        delta = (
            right_value - left_value
            if isinstance(left_value, (int, float)) and isinstance(right_value, (int, float))
            else ""
        )
        lines.append(f"| {key} | {left_value} | {right_value} | {delta} |")

    lines.extend(["", "## By Level", "", f"| Level | Total | {left} correct | {right} correct | Delta |", "|---|---:|---:|---:|---:|"])
    for key, row in report["by_level"].items():
        lines.append(
            f"| {key} | {row['total']} | {row['left_correct']} | {row['right_correct']} | {row['diff']} |"
        )

    lines.extend(["", "## By Subject", "", f"| Subject | Total | {left} correct | {right} correct | Delta |", "|---|---:|---:|---:|---:|"])
    for key, row in report["by_subject"].items():
        lines.append(
            f"| {key} | {row['total']} | {row['left_correct']} | {row['right_correct']} | {row['diff']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_comparison_outputs(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary = {key: value for key, value in report.items() if key != "categories"}
    summary_path = output_path / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path = output_path / "summary.md"
    _write_summary_markdown(markdown_path, report)

    category_paths: dict[str, str] = {}
    for category, rows in report["categories"].items():
        path = output_path / f"{category}.jsonl"
        _write_jsonl(path, rows)
        category_paths[category] = str(path)

    return {
        "summary_json": str(summary_path),
        "summary_md": str(markdown_path),
        **category_paths,
    }


def _print_report(report: dict[str, Any], output_paths: dict[str, str]) -> None:
    left = report["left_name"]
    right = report["right_name"]
    counts = report["counts"]
    print(f"{left} vs {right}")
    print(f"  common samples       : {counts['total_common']}")
    print(f"  both correct         : {counts['both_correct']}")
    print(f"  both wrong           : {counts['both_wrong']}")
    print(f"  {left} only correct  : {counts['left_only_correct']}")
    print(f"  {right} only correct : {counts['right_only_correct']}")
    print(f"  different correctness: {counts['different_correctness']}")
    print("\nBy level delta (right-left):")
    for key, row in report["by_level"].items():
        print(f"  Level {key}: {row['diff']:+d} ({row['left_correct']} -> {row['right_correct']}, n={row['total']})")
    print("\nBy subject delta (right-left):")
    for key, row in report["by_subject"].items():
        print(f"  {key}: {row['diff']:+d} ({row['left_correct']} -> {row['right_correct']}, n={row['total']})")
    print("\nWritten:")
    for label, path in output_paths.items():
        print(f"  {label}: {path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two evaluator JSON outputs per problem.")
    parser.add_argument("--left", default=DEFAULT_LEFT_PATH, help="左侧评测 JSON，例如 GRPO")
    parser.add_argument("--right", default=DEFAULT_RIGHT_PATH, help="右侧评测 JSON，例如 GSPO")
    parser.add_argument("--left-name", default="grpo")
    parser.add_argument("--right-name", default="gspo")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--include-solution",
        action="store_true",
        help="在逐题 JSONL 中包含标准解答文本，文件会更大",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    report = compare_eval_results(
        left_path=args.left,
        right_path=args.right,
        left_name=args.left_name,
        right_name=args.right_name,
        dataset_path=args.dataset_path,
        include_solution=args.include_solution,
    )
    output_paths = write_comparison_outputs(report, args.output_dir)
    _print_report(report, output_paths)


if __name__ == "__main__":
    main()
