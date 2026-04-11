"""
analyze_eval_badcases.py - 分析评测 JSON 中的错误样本原因。

示例：
    python3 scripts/analyze_eval_badcases.py --input outputs/eval/gspo/math500.json --name gspo
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.post_training_contrast.math_utils import symbolic_equals


DEFAULT_OUTPUT_ROOT = "outputs/eval/badcases"


def _strip_degree_unit(text: str) -> str:
    cleaned = str(text).strip()
    cleaned = re.sub(r"\^\s*\{?\\circ\}?", "", cleaned)
    cleaned = cleaned.replace("°", "")
    return cleaned.strip()


def _single_variable_equation_rhs(text: str) -> str | None:
    cleaned = str(text).strip()
    match = re.fullmatch(r"([A-Za-z])\s*=\s*(.+)", cleaned)
    if match:
        return match.group(2).strip()
    match = re.fullmatch(r"(.+)\s*=\s*([A-Za-z])", cleaned)
    if match:
        return match.group(1).strip()
    return None


def _has_latex_frac_shorthand(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text))
    return (
        re.search(
            r"\\frac(?:[A-Za-z0-9](?:[A-Za-z0-9]|\{)|\{[^{}]+\}[A-Za-z0-9])",
            compact,
        )
        is not None
    )


def _is_degree_unit_false_negative(predicted: str, ground_truth: str) -> bool:
    if "\\circ" not in ground_truth and "°" not in ground_truth:
        return False
    return symbolic_equals(_strip_degree_unit(predicted), _strip_degree_unit(ground_truth))


def _is_equation_rhs_false_negative(predicted: str, ground_truth: str) -> bool:
    rhs = _single_variable_equation_rhs(ground_truth)
    if rhs is None:
        return False
    return symbolic_equals(predicted, rhs)


def _is_latex_frac_shorthand_false_negative(predicted: str, ground_truth: str) -> bool:
    if not _has_latex_frac_shorthand(predicted) and not _has_latex_frac_shorthand(ground_truth):
        return False
    return symbolic_equals(predicted, ground_truth)


def classify_detail(detail: dict[str, Any]) -> str:
    """返回错误样本的主要原因分类。正确样本返回 correct。"""
    if bool(detail.get("is_correct", False)):
        return "correct"

    predicted = str(detail.get("parsed_prediction") or "").strip()
    ground_truth = str(detail.get("ground_truth") or "").strip()
    if bool(detail.get("format_failed", False)) or not predicted:
        return "format_or_parse_fail"

    if _is_degree_unit_false_negative(predicted, ground_truth):
        return "likely_false_negative_degree_unit"
    if _is_equation_rhs_false_negative(predicted, ground_truth):
        return "likely_false_negative_equation_rhs"
    if _is_latex_frac_shorthand_false_negative(predicted, ground_truth):
        return "likely_false_negative_latex_frac_shorthand"

    if bool(detail.get("is_truncated", False)):
        return "truncated_with_parsed_answer"
    return "wrong_answer_parsed"


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
            records[record.get("id") or f"index-{index}"] = record
    return records


def _badcase_row(
    detail: dict[str, Any],
    category: str,
    dataset_record: dict[str, Any],
    include_solution: bool,
) -> dict[str, Any]:
    row = {
        "category": category,
        "id": detail.get("id"),
        "unique_id": dataset_record.get("unique_id"),
        "level": detail.get("level"),
        "subject": detail.get("subject"),
        "problem": dataset_record.get("problem"),
        "ground_truth": detail.get("ground_truth"),
        "parsed_prediction": detail.get("parsed_prediction"),
        "format_failed": bool(detail.get("format_failed", False)),
        "is_truncated": bool(detail.get("is_truncated", False)),
        "response_length": detail.get("response_length"),
        "raw_prediction": detail.get("raw_prediction", ""),
    }
    if include_solution:
        row["solution"] = dataset_record.get("solution")
    return row


def analyze_eval_badcases(
    input_path: str | Path,
    dataset_path: str | Path | None = "datasets/processed/math500_test/test.jsonl",
    include_solution: bool = False,
) -> dict[str, Any]:
    data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    details = data.get("details")
    if not isinstance(details, list):
        raise ValueError(f"{input_path} 缺少 details 列表")

    dataset_records = _load_dataset_records(dataset_path)
    counts: Counter[str] = Counter()
    by_category: dict[str, list[dict[str, Any]]] = {}
    by_level: dict[str, Counter[str]] = {}
    by_subject: dict[str, Counter[str]] = {}

    for detail in details:
        category = classify_detail(detail)
        counts[category] += 1
        if category != "correct":
            detail_id = detail.get("id")
            row = _badcase_row(
                detail=detail,
                category=category,
                dataset_record=dataset_records.get(detail_id, {}),
                include_solution=include_solution,
            )
            by_category.setdefault(category, []).append(row)
            by_level.setdefault(str(detail.get("level")), Counter())[category] += 1
            by_subject.setdefault(str(detail.get("subject")), Counter())[category] += 1

    summary = {
        "input_path": str(input_path),
        "dataset_name": data.get("dataset_name"),
        "num_samples": data.get("num_samples"),
        "accuracy": data.get("accuracy"),
        "format_fail_rate": data.get("format_fail_rate"),
        "truncation_rate": data.get("truncation_rate"),
        "mean_length": data.get("mean_length"),
        "median_length": data.get("median_length"),
        "p90_length": data.get("p90_length"),
        "counts": dict(counts),
        "by_level": {key: dict(value) for key, value in sorted(by_level.items())},
        "by_subject": {key: dict(value) for key, value in sorted(by_subject.items())},
    }
    return {"summary": summary, "by_category": by_category}


def write_badcase_outputs(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    summary_path = target / "summary.json"
    summary_path.write_text(
        json.dumps(report["summary"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths = {"summary": str(summary_path)}
    for category, rows in report["by_category"].items():
        path = target / f"{category}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                json.dump(row, f, ensure_ascii=False)
                f.write("\n")
        paths[category] = str(path)
    return paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze badcases from evaluator JSON details.")
    parser.add_argument("--input", required=True, help="评测 JSON 路径")
    parser.add_argument("--name", required=True, help="模型/实验名称，用于输出目录")
    parser.add_argument("--dataset-path", default="datasets/processed/math500_test/test.jsonl")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--include-solution", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    report = analyze_eval_badcases(
        input_path=args.input,
        dataset_path=args.dataset_path,
        include_solution=args.include_solution,
    )
    paths = write_badcase_outputs(report, Path(args.output_root) / args.name)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print("\nWritten:")
    for key, path in paths.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
