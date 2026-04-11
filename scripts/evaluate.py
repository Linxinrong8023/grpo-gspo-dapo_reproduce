"""
scripts/evaluate.py - 评测入口。

用法（在线生成）:
    python scripts/evaluate.py --config configs/eval_math500.json

用法（离线加载已有预测）:
    python scripts/evaluate.py \\
        --dataset-path datasets/math500.jsonl \\
        --dataset-name math500 \\
        --predictions-file outputs/eval/predictions.jsonl \\
        --output-path outputs/eval/result.json
"""

import logging
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 触发依赖自愈补丁 (Fix for outlines/vllm/pyairports)
import src.post_training_contrast

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

from src.post_training_contrast.config import load_eval_config
from src.post_training_contrast.evaluator import run_evaluation


def main() -> None:
    config = load_eval_config()
    summary = run_evaluation(config)
    print(
        "\n"
        f"✓ Accuracy: {summary['accuracy']:.2%}  (n={summary['num_samples']})\n"
        f"  Format fail : {summary.get('format_fail_rate', 0.0):.2%}\n"
        f"  Truncation  : {summary.get('truncation_rate', 0.0):.2%}\n"
        f"  Mean length : {summary.get('mean_length', 0.0):.1f}\n"
        f"  P90 length  : {summary.get('p90_length', 0.0):.1f}\n"
        f"  Results     : {config.output_path}"
    )


if __name__ == "__main__":
    main()
