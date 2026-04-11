"""
scripts/plot.py - 可视化入口。

用法（训练曲线）:
    python scripts/plot.py training --summary outputs/train/grpo_math/train_summary.json

用法（评测结果）:
    python scripts/plot.py eval --result outputs/eval/result.json

用法（多算法对比）:
    python scripts/plot.py compare \\
        --grpo outputs/eval/grpo_result.json \\
        --dapo outputs/eval/dapo_result.json \\
        --gspo outputs/eval/gspo_result.json
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.post_training_contrast.visualization import (
    plot_algorithm_comparison,
    plot_eval_summary,
    plot_training_curves,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="可视化工具")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 训练曲线子命令
    p_train = subparsers.add_parser("training", help="画训练曲线")
    p_train.add_argument("--summary", required=True, help="train_summary.json 路径")
    p_train.add_argument("--save", default=None, help="图片保存路径")
    p_train.add_argument("--no-show", action="store_true")

    # 评测结果子命令
    p_eval = subparsers.add_parser("eval", help="画评测结果")
    p_eval.add_argument("--result", required=True, help="评测结果 JSON 路径")
    p_eval.add_argument("--save", default=None)
    p_eval.add_argument("--no-show", action="store_true")

    # 多算法对比子命令
    p_cmp = subparsers.add_parser("compare", help="多算法准确率对比")
    p_cmp.add_argument("--grpo", default=None, help="GRPO 评测结果 JSON")
    p_cmp.add_argument("--gspo", default=None, help="GSPO 评测结果 JSON")
    p_cmp.add_argument("--dapo", default=None, help="DAPO 评测结果 JSON")
    p_cmp.add_argument("--save", default=None)
    p_cmp.add_argument("--no-show", action="store_true")

    args = parser.parse_args()

    if args.command == "training":
        save_file = args.save or "outputs/figures/training_curve.png"
        plot_training_curves(args.summary, save_path=save_file, show=not args.no_show)

    elif args.command == "eval":
        save_file = args.save or "outputs/figures/eval_summary.png"
        plot_eval_summary(args.result, save_path=save_file, show=not args.no_show)

    elif args.command == "compare":
        save_file = args.save or "outputs/figures/algorithm_comparison.png"
        result_paths = {}
        for algo in ("grpo", "gspo", "dapo"):
            path = getattr(args, algo)
            if path:
                result_paths[algo.upper()] = path
        if not result_paths:
            parser.error("至少提供一个算法的评测结果（--grpo / --gspo / --dapo）")
        plot_algorithm_comparison(result_paths, save_path=save_file, show=not args.no_show)


if __name__ == "__main__":
    main()
