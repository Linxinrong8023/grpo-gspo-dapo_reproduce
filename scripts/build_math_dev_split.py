import argparse
import os
import sys
from collections import Counter
from rich.traceback import install

install()
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.post_training_contrast.data.math_dev_split import build_type_level_key
from src.post_training_contrast.data.math_dev_split import load_jsonl_records
from src.post_training_contrast.data.math_dev_split import save_jsonl_records
from src.post_training_contrast.data.math_dev_split import split_math_train_records


def summarize_split(records: list[dict], split_name: str) -> None:
    """功能：
    - 打印当前切分的基础统计
    - 方便快速检查分层结果是否合理
    """
    type_counter = Counter(record["type"] for record in records)
    level_counter = Counter(record["level"] for record in records)
    combo_counter = Counter(build_type_level_key(record) for record in records)

    print(f"\n[{split_name}] 样本数: {len(records)}")
    print(f"[{split_name}] type 分布: {dict(sorted(type_counter.items()))}")
    print(f"[{split_name}] level 分布: {dict(sorted(level_counter.items()))}")
    print(f"[{split_name}] 分层组合数: {len(combo_counter)}")


def main() -> None:
    """功能：
    - 从完整的 MATH full 集切出固定的 dev 集
    - 保存真正训练用的 train 集和 dev 集
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="datasets/processed/math_full/full.jsonl",
        help="完整 MATH full 的输入路径",
    )
    parser.add_argument(
        "--output-train",
        default="datasets/processed/math_train/train.jsonl",
        help="切分后的训练集输出路径",
    )
    parser.add_argument(
        "--output-dev",
        default="datasets/processed/math_dev/dev.jsonl",
        help="切分后的 dev 集输出路径",
    )
    parser.add_argument(
        "--dev-size",
        type=int,
        default=500,
        help="dev 集样本数",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="固定切分随机种子",
    )
    args = parser.parse_args()

    records = load_jsonl_records(args.input)
    train_records, dev_records = split_math_train_records(
        records=records,
        dev_size=args.dev_size,
        seed=args.seed,
    )

    save_jsonl_records(train_records, args.output_train)
    save_jsonl_records(dev_records, args.output_dev)

    print(f"已写出 train: {args.output_train}")
    print(f"已写出 dev  : {args.output_dev}")
    print(f"seed={args.seed}, dev_size={args.dev_size}")

    summarize_split(train_records, "train")
    summarize_split(dev_records, "dev")


if __name__ == "__main__":
    main()
