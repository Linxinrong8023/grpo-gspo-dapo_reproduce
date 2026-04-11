import argparse
import os
import sys
from collections import Counter


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.post_training_contrast.data.math_dev_split import build_type_level_key
from src.post_training_contrast.data.math_dev_split import load_jsonl_records
from src.post_training_contrast.data.math_dev_split import sample_stratified_subset
from src.post_training_contrast.data.math_dev_split import save_jsonl_records
from src.post_training_contrast.data.math_dev_split import split_math_train_records


def summarize(records: list[dict], split_name: str) -> None:
    type_counter = Counter(record["type"] for record in records)
    level_counter = Counter(record["level"] for record in records)
    combo_counter = Counter(build_type_level_key(record) for record in records)

    print(f"\n[{split_name}] 样本数: {len(records)}")
    print(f"[{split_name}] type 分布: {dict(sorted(type_counter.items()))}")
    print(f"[{split_name}] level 分布: {dict(sorted(level_counter.items()))}")
    print(f"[{split_name}] 分层组合数: {len(combo_counter)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 MATH 7500 full 集重新构建分层的 3000 train + 500 val 划分。"
    )
    parser.add_argument(
        "--input",
        default="datasets/processed/math_full/full.jsonl",
        help="完整 7500 条 MATH full 的路径",
    )
    parser.add_argument(
        "--output-train",
        default="datasets/processed/math_train/train_3000_stratified.jsonl",
        help="3000 条分层训练集输出路径",
    )
    parser.add_argument(
        "--output-val",
        default="datasets/processed/math_dev/dev_500_stratified.jsonl",
        help="500 条分层验证集输出路径",
    )
    parser.add_argument(
        "--output-train-pool",
        default="datasets/processed/math_train/train_pool_7000_after_val.jsonl",
        help="切掉验证集后剩余 7000 条训练池输出路径",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=3000,
        help="训练子集大小",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=500,
        help="验证集大小",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="固定随机种子",
    )
    args = parser.parse_args()

    records = load_jsonl_records(args.input)
    train_pool, val_records = split_math_train_records(
        records=records,
        dev_size=args.val_size,
        seed=args.seed,
    )
    train_records = sample_stratified_subset(
        train_pool,
        sample_size=args.train_size,
        seed=args.seed,
    )

    save_jsonl_records(train_pool, args.output_train_pool)
    save_jsonl_records(train_records, args.output_train)
    save_jsonl_records(val_records, args.output_val)

    print(f"已写出 train_pool: {args.output_train_pool}")
    print(f"已写出 train     : {args.output_train}")
    print(f"已写出 val       : {args.output_val}")
    print(f"seed={args.seed}, train_size={args.train_size}, val_size={args.val_size}")

    summarize(train_pool, "train_pool")
    summarize(train_records, "train")
    summarize(val_records, "val")


if __name__ == "__main__":
    main()
