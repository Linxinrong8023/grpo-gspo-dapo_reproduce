"""
Prepare datasets for post-training contrast experiments.

This script converts raw datasets into standardized JSONL format:
  - MATH (train): 7 topic parquets aggregated -> train.jsonl  (for training)
  - MATH500 (test): already JSONL, reformatted for consistency
  - GSM8K (test):   parquet -> test.jsonl  (for evaluation)

Output directory: datasets/processed/
"""

import json
import os
import re
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "datasets", "raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "datasets", "processed")

MATH_TOPICS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def write_jsonl(data, filepath):
    """Write a list of dicts to a JSONL file."""
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  ✅ Written {len(data)} samples -> {filepath}")


def extract_boxed_answer(solution):
    """Extract the answer from \\boxed{...} in a MATH solution.

    Handles nested braces properly.
    """
    # Find all \boxed occurrences and take the last one
    idx = solution.rfind("\\boxed{")
    if idx == -1:
        return ""
    # Find matching closing brace
    depth = 0
    start = idx + len("\\boxed{")
    for i in range(start, len(solution)):
        if solution[i] == "{":
            depth += 1
        elif solution[i] == "}":
            if depth == 0:
                return solution[start:i]
            depth -= 1
    return ""


def extract_gsm8k_answer(answer_text):
    """Extract the final numerical answer from GSM8K chain-of-thought.

    GSM8K answers have step-by-step reasoning followed by '#### <answer>'.
    """
    match = re.search(r"####\s*(.+)", answer_text)
    if match:
        return match.group(1).strip().replace(",", "")
    return answer_text.strip()


def prepare_math_train():
    """Convert MATH train parquets (all 7 topics) to a single full.jsonl.

    Source: dataset/math/{topic}/train-00000-of-00001.parquet
    Fields: problem, level, type, solution
    Output fields: problem, level, type, solution, answer, id
    """
    print("\n📐 Processing MATH training set (all topics)...")

    all_records = []
    global_idx = 0

    for topic in MATH_TOPICS:
        parquet_path = os.path.join(DATASET_DIR, "math", topic, "train-00000-of-00001.parquet")
        df = pd.read_parquet(parquet_path)
        print(f"  📁 {topic}: {len(df)} samples")

        for _, row in df.iterrows():
            solution = row["solution"]
            answer = extract_boxed_answer(solution)

            record = {
                "problem": row["problem"],
                "level": row["level"],
                "type": row["type"],
                "solution": solution,
                "answer": answer,
                "id": f"math-train-{global_idx}",
            }
            all_records.append(record)
            global_idx += 1

    out_dir = os.path.join(OUTPUT_DIR, "math_full")
    ensure_dir(out_dir)
    write_jsonl(all_records, os.path.join(out_dir, "full.jsonl"))

    # Print stats
    type_counts = {}
    level_counts = {}
    for r in all_records:
        t = r["type"]
        l = r["level"]
        type_counts[t] = type_counts.get(t, 0) + 1
        level_counts[l] = level_counts.get(l, 0) + 1
    print(f"  Type distribution: {json.dumps(type_counts, indent=4)}")
    print(f"  Level distribution: {json.dumps(dict(sorted(level_counts.items())), indent=4)}")
    return all_records


def prepare_math500_test():
    """Copy and standardize MATH500 test set.

    Source: dataset/math500/test.jsonl
    Fields: problem, solution, answer, subject, level, unique_id
    Output fields: problem, solution, answer, subject, level, unique_id, id
    """
    print("\n📐 Processing MATH500 test set...")
    src_path = os.path.join(DATASET_DIR, "math500", "test.jsonl")

    records = []
    with open(src_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "id" not in item:
                item["id"] = f"math500-test-{idx}"
            records.append(item)

    out_dir = os.path.join(OUTPUT_DIR, "math500_test")
    ensure_dir(out_dir)
    write_jsonl(records, os.path.join(out_dir, "test.jsonl"))

    # Print stats
    subjects = {}
    levels = {}
    for r in records:
        subj = r.get("subject", "unknown")
        level = r.get("level", "unknown")
        subjects[subj] = subjects.get(subj, 0) + 1
        levels[level] = levels.get(level, 0) + 1
    print(f"  Subject distribution: {json.dumps(subjects, indent=4)}")
    print(f"  Level distribution: {json.dumps(dict(sorted(levels.items())), indent=4)}")
    return records


def prepare_gsm8k_test():
    """Convert GSM8K test parquet to JSONL.

    Source: dataset/gsm8k/main/test-00000-of-00001.parquet
    Fields: question, answer (with chain-of-thought and #### final_answer)
    Output fields: question, answer (final numerical answer), cot (chain-of-thought), id
    """
    print("\n🧮 Processing GSM8K test set...")
    parquet_path = os.path.join(DATASET_DIR, "gsm8k", "main", "test-00000-of-00001.parquet")
    df = pd.read_parquet(parquet_path)
    print(f"  Loaded {len(df)} samples from parquet")

    records = []
    for idx, row in df.iterrows():
        full_answer = row["answer"]
        final_answer = extract_gsm8k_answer(full_answer)
        cot = full_answer.split("####")[0].strip() if "####" in full_answer else full_answer

        record = {
            "question": row["question"],
            "answer": final_answer,
            "cot": cot,
            "id": f"gsm8k-test-{idx}",
        }
        records.append(record)

    out_dir = os.path.join(OUTPUT_DIR, "gsm8k_test")
    ensure_dir(out_dir)
    write_jsonl(records, os.path.join(out_dir, "test.jsonl"))
    return records


def main():
    print("=" * 60)
    print("Dataset Preparation for Post-Training Contrast")
    print("=" * 60)

    ensure_dir(OUTPUT_DIR)

    math_train = prepare_math_train()
    math500_test = prepare_math500_test()
    gsm8k_test = prepare_gsm8k_test()

    print("\n" + "=" * 60)
    print("📊 Summary")
    print("=" * 60)
    print(f"  MATH   full  : {len(math_train):>6} samples -> datasets/processed/math_full/full.jsonl")
    print(f"  MATH500 test : {len(math500_test):>6} samples -> datasets/processed/math500_test/test.jsonl")
    print(f"  GSM8K   test : {len(gsm8k_test):>6} samples -> datasets/processed/gsm8k_test/test.jsonl")
    print("=" * 60)
    print("✅ All datasets prepared successfully!")


if __name__ == "__main__":
    main()
