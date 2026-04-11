"""
post_train_eval.py - 训练后处理入口。

职责：
  1. 读取 train_summary.json
  2. 解析 checkpoint（best / final / stepNNN）
  3. 如有需要，自动 merge LoRA adapter
  4. 调用现有 evaluator 跑 Math500
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import src.post_training_contrast  # noqa: F401
from src.post_training_contrast.config import EvalConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_MATH500_PATH = "datasets/processed/math500_test/test.jsonl"


def run_evaluation(config: EvalConfig) -> dict:
    from src.post_training_contrast.evaluator import run_evaluation as _run_evaluation

    return _run_evaluation(config)


def _load_train_summary(train_output_dir: Path) -> dict:
    summary_path = train_output_dir / "train_summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_base_model_path(summary: dict) -> str:
    trainer_config = summary.get("trainer_config") or {}
    model_path = trainer_config.get("model") or summary.get("model")
    if not model_path:
        raise ValueError("train_summary.json 缺少 base model 路径。")
    return str(model_path)


def _resolve_checkpoint_dir(train_output_dir: Path, checkpoint: str) -> Path:
    normalized = (checkpoint or "best").strip().lower()
    if normalized == "best":
        checkpoint_dir = train_output_dir / "checkpoint_best"
    elif normalized == "final":
        checkpoint_dir = train_output_dir / "checkpoint_final"
    elif normalized.startswith("checkpoint_step"):
        checkpoint_dir = train_output_dir / normalized
    elif normalized.startswith("step"):
        checkpoint_dir = train_output_dir / f"checkpoint_{normalized}"
    else:
        checkpoint_dir = train_output_dir / checkpoint

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"找不到 checkpoint: {checkpoint_dir}")
    return checkpoint_dir


def _is_lora_adapter_checkpoint(checkpoint_dir: Path) -> bool:
    return (checkpoint_dir / "adapter_config.json").exists()


def merge_lora_checkpoint(
    base_model_path: str,
    adapter_dir: Path,
    merged_dir: Path,
) -> Path:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Merging LoRA checkpoint %s onto %s", adapter_dir, base_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    merged = PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(merged_dir)

    has_adapter_tokenizer = any(
        (adapter_dir / filename).exists()
        for filename in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
    )
    tokenizer_source = str(adapter_dir) if has_adapter_tokenizer else base_model_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=True,
    )
    tokenizer.save_pretrained(merged_dir)
    logger.info("Merged checkpoint saved → %s", merged_dir)
    return merged_dir


def run_post_train_eval(
    train_output_dir: str | Path,
    checkpoint: str = "best",
    dataset_path: str = DEFAULT_MATH500_PATH,
    dataset_name: str = "math500",
    output_path: str | None = None,
    merged_dir: str | Path | None = None,
    max_samples: int | None = None,
    batch_size: int = 4,
    max_new_tokens: int = 2048,
    device: str = "auto",
    dtype: str = "auto",
    use_vllm: bool = False,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
) -> dict:
    train_output_dir = Path(train_output_dir)
    summary = _load_train_summary(train_output_dir)
    base_model_path = _resolve_base_model_path(summary)
    checkpoint_dir = _resolve_checkpoint_dir(train_output_dir, checkpoint)

    model_path = checkpoint_dir
    merged_from_lora = _is_lora_adapter_checkpoint(checkpoint_dir)
    merged_model_path: str | None = None
    if merged_from_lora:
        merged_dir = (
            Path(merged_dir)
            if merged_dir is not None
            else checkpoint_dir.parent / f"{checkpoint_dir.name}_merged"
        )
        model_path = merge_lora_checkpoint(
            base_model_path=base_model_path,
            adapter_dir=checkpoint_dir,
            merged_dir=merged_dir,
        )
        merged_model_path = str(model_path)

    resolved_output_path = output_path or str(
        train_output_dir / "eval" / checkpoint_dir.name / f"{dataset_name}.json"
    )
    metadata = {
        "train_output_dir": str(train_output_dir),
        "requested_checkpoint": checkpoint,
        "resolved_checkpoint_dir": str(checkpoint_dir),
        "base_model_path": base_model_path,
        "model_path": str(model_path),
        "merged_from_lora": merged_from_lora,
        "merged_model_path": merged_model_path,
    }
    eval_config = EvalConfig(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        output_path=resolved_output_path,
        model=str(model_path),
        max_samples=max_samples,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        device=device,
        dtype=dtype,
        use_vllm=use_vllm,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        metadata=metadata,
    )
    logger.info(
        "Running post-train eval: checkpoint=%s  model=%s  dataset=%s",
        checkpoint_dir.name,
        model_path,
        dataset_path,
    )
    return run_evaluation(eval_config)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="训练后自动 merge 并评测 Math500")
    parser.add_argument("--train-output-dir", required=True, help="训练输出目录")
    parser.add_argument(
        "--checkpoint",
        default="best",
        help="best / final / stepNNN / checkpoint_stepNNN",
    )
    parser.add_argument("--dataset-path", default=DEFAULT_MATH500_PATH)
    parser.add_argument("--dataset-name", default="math500")
    parser.add_argument("--output-path")
    parser.add_argument(
        "--merged-dir",
        help="LoRA checkpoint 合并后的完整模型保存目录；建议指向数据盘以避免占满系统盘",
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    summary = run_post_train_eval(
        train_output_dir=args.train_output_dir,
        checkpoint=args.checkpoint,
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        output_path=args.output_path,
        merged_dir=args.merged_dir,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        dtype=args.dtype,
        use_vllm=args.use_vllm,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    print(
        "\n"
        f"✓ Accuracy: {summary['accuracy']:.2%}  (n={summary['num_samples']})\n"
        f"  Format fail : {summary.get('format_fail_rate', 0.0):.2%}\n"
        f"  Truncation  : {summary.get('truncation_rate', 0.0):.2%}\n"
        f"  Mean length : {summary.get('mean_length', 0.0):.1f}\n"
        f"  P90 length  : {summary.get('p90_length', 0.0):.1f}\n"
        f"  Results     : {summary.get('eval_config', {}).get('output_path', '')}"
    )


if __name__ == "__main__":
    main()
