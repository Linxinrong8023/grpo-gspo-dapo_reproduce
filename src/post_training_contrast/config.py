"""
config.py - 所有训练和评测的配置 dataclass。

用法：
    train_config = load_train_config()   # scripts/train.py 里调用
    eval_config  = load_eval_config()    # scripts/evaluate.py 里调用

设计原则：
- 所有配置字段都有类型注解和合理默认值
- CLI 参数覆盖 JSON 文件，JSON 文件覆盖默认值
- 只要 load_*_config() 成功返回，config 就是合法的，无需后续再做 key 检查
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, asdict, fields
from datetime import datetime
from pathlib import Path


# ── 配置 dataclass ──────────────────────────────────────────────


@dataclass
class TrainConfig:
    """训练配置。所有字段都可以从 JSON 文件或 CLI 参数注入。"""

    # 必填
    model: str = ""
    dataset_path: str = ""
    dataset_name: str = "math"
    output_dir: str = "outputs/run"
    output_root: str | None = None
    run_id: str | None = None
    timestamp_output_dir: bool = True
    algorithm_name: str = "grpo"

    # 数据
    seed: int = 42
    num_epochs: int = 1
    shuffle_prompts_each_epoch: bool = True
    prompt_batch_size: int = 4
    max_samples: int | None = None
    max_steps: int | None = None

    # 训练超参
    group_size: int = 8
    num_policy_updates: int = 1
    mini_batch_size: int = 8
    shuffle_minibatches: bool = True
    learning_rate: float = 5e-6
    max_grad_norm: float = 1.0
    lr_schedule: str = "cosine"
    lr_warmup_steps: int = 0
    gradient_accumulation_steps: int = (
        1  # 累积多少个 mini_batch 才做一次 optimizer.step()
    )

    # 生成参数
    max_new_tokens: int = 1536
    temperature: float = 0.7
    top_p: float = 0.95
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.3
    max_model_len: int | None = None

    # 硬件
    device: str = "auto"
    dtype: str = "auto"
    rollout_dtype: str = "auto"
    attn_implementation: str | None = "auto"

    # 输出与 checkpoint
    checkpoint_interval: int = 0
    save_best_checkpoint: bool = True
    best_checkpoint_metric: str = "val_accuracy"
    early_stopping_patience: int | None = 8
    reload_rollout_each_step: bool = True
    single_gpu_safe_mode: bool = True
    vllm_sleep_during_update: bool = False
    keep_vllm_resident_on_lora: bool = True

    # 训练中验证集评测
    eval_dataset_path: str | None = None  # 验证集路径，None = 不评测
    eval_interval: int = 50  # 每隔多少步评测一次
    eval_max_new_tokens: int = 2048
    max_eval_samples: int | None = None
    val_backend: str = "vllm"
    val_batch_size: int = 16

    # 通用 RL 参数
    clip_epsilon: float = 0.2
    kl_beta: float = 0.0
    track_policy_entropy: bool = True
    answer_reward_correct: float = 1.0
    answer_reward_incorrect: float = 0.0
    format_penalty_missing_think_close: float = -0.3
    format_penalty_missing_answer_tag: float = -0.15

    # DAPO 专属
    dapo_use_dynamic_sampling: bool = False
    dapo_use_overlong_penalty: bool = False
    dapo_clip_epsilon_low: float = 0.2
    dapo_clip_epsilon_high: float = 0.28
    dapo_gen_prompt_batch_size: int | None = None
    dapo_max_num_gen_batches: int = 6
    dapo_min_mixed_prompt_batch_size: int = 24

    # LoRA（use_lora=False 时以下所有字段均被忽略）
    use_lora: bool = False
    lora_r: int = 16  # LoRA 秩，越大拟合能力越强，显存也越多
    lora_alpha: int = 32  # 缩放系数，通常 = 2 * lora_r
    lora_dropout: float = 0.05
    lora_target_modules: list | None = None  # None = 自动选 attention+MLP 层

    # WandB 监控（wandb_enabled=False 时其他字段均被忽略）
    wandb_enabled: bool = False
    wandb_project: str = "post_training_contrast"
    wandb_run_name: str | None = None  # None = WandB 自动生成随机名称
    wandb_tags: list | None = None  # 标签，方便过滤，如 ["grpo", "1.5b"]

    # 训练诊断样本（只保存异常 response 的短 preview，便于长跑后复盘）
    dump_diagnostic_samples: bool = True
    diagnostic_sample_limit_per_step: int = 3
    diagnostic_response_max_chars: int = 2000


@dataclass
class EvalConfig:
    """评测配置。"""

    dataset_path: str = ""
    dataset_name: str = "math500"
    output_path: str = "outputs/eval/result.json"

    # 生成方式：model（在线生成）或 predictions_file（离线预测）
    model: str | None = None
    predictions_file: str | None = None

    max_samples: int | None = None
    batch_size: int = 1
    max_new_tokens: int = 2048
    device: str = "auto"
    dtype: str = "auto"

    # vLLM 加速选项
    use_vllm: bool = False
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9

    # 仅用于结果追溯，不参与评分逻辑
    metadata: dict | None = None


# ── 内部工具函数 ─────────────────────────────────────────────────


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _merge_configs(file_config: dict, cli_overrides: dict) -> dict:
    """CLI 参数覆盖 JSON 文件中的同名字段。"""
    merged = dict(file_config)
    for key, value in cli_overrides.items():
        if value is not None:
            merged[key] = value
    return merged


def _from_dict(cls, data: dict):
    """只取 dataclass 已声明的字段，忽略多余的 JSON key。"""
    valid_keys = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in valid_keys})


def _make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _apply_timestamp_output_dir(config: TrainConfig) -> TrainConfig:
    """把训练输出根目录扩展为带时间戳的 run 目录。"""
    output_root = config.output_root or config.output_dir
    config.output_root = output_root
    if config.timestamp_output_dir:
        run_id = config.run_id or _make_run_id()
        config.run_id = run_id
        config.output_dir = str(Path(output_root) / run_id)
    else:
        config.run_id = None
        config.output_dir = output_root
    return config


class _FlushingFileHandler(logging.FileHandler):
    """每条日志写完立刻刷盘，防止进程被 SIGKILL 时丢失缓冲区内容。"""

    def emit(self, record):
        super().emit(record)
        self.flush()


def setup_run_logging(output_dir: str) -> None:
    """在 output_dir 创建 train.log，同时输出到终端。每次调用重新配置。

    使用 FlushingFileHandler 确保每条日志立刻写入磁盘，
    即使进程被 OOM Killer 无声杀掉（SIGKILL），已打印的日志也不会丢失。
    """
    import sys

    log_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "train.log"

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    # StreamHandler 也强制 flush（防止终端输出缓冲）
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)

    file_handler = _FlushingFileHandler(str(log_file), encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[stream_handler, file_handler],
        force=True,
    )
    logging.getLogger(__name__).info("Logging initialized → %s", log_file)


# ── 公开接口 ─────────────────────────────────────────────────────


def load_train_config() -> TrainConfig:
    """从 --config JSON 文件 + CLI 参数解析 TrainConfig。CLI 优先级更高。"""
    parser = argparse.ArgumentParser(description="RL 训练")
    parser.add_argument("--config", required=True, help="训练配置文件路径")
    parser.add_argument("--model")
    parser.add_argument("--dataset-path")
    parser.add_argument("--dataset-name")
    parser.add_argument("--output-dir")
    parser.add_argument("--algorithm-name")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--prompt-batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--max-grad-norm", type=float)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument(
        "--no-timestamp-output-dir",
        action="store_false",
        dest="timestamp_output_dir",
        default=None,
        help="直接使用 output_dir，不自动追加时间戳 run 子目录",
    )
    parser.add_argument(
        "--attn-implementation",
        help="HF attention backend, e.g. auto, flash_attention_2, sdpa, eager, or none",
    )
    args = parser.parse_args()

    file_config = _load_json(args.config)
    cli_overrides = {
        "model": args.model,
        "dataset_path": args.dataset_path,
        "dataset_name": args.dataset_name,
        "output_dir": args.output_dir,
        "algorithm_name": args.algorithm_name,
        "max_samples": args.max_samples,
        "max_steps": args.max_steps,
        "prompt_batch_size": args.prompt_batch_size,
        "learning_rate": args.learning_rate,
        "max_grad_norm": args.max_grad_norm,
        "checkpoint_interval": args.checkpoint_interval,
        "attn_implementation": args.attn_implementation,
        "timestamp_output_dir": args.timestamp_output_dir,
    }
    merged = _merge_configs(file_config, cli_overrides)
    if args.attn_implementation is not None and args.attn_implementation.lower() in {
        "none",
        "null",
        "default",
    }:
        merged["attn_implementation"] = None

    # 检查必填项
    required = ("model", "dataset_path", "dataset_name", "output_dir", "algorithm_name")
    missing = [k for k in required if not merged.get(k)]
    if missing:
        parser.error(f"缺少必要配置项：{', '.join(missing)}")

    config = _apply_timestamp_output_dir(_from_dict(TrainConfig, merged))
    setup_run_logging(config.output_dir)
    return config


def load_eval_config() -> EvalConfig:
    """从 --config JSON 文件 + CLI 参数解析 EvalConfig。"""
    parser = argparse.ArgumentParser(description="评测")
    parser.add_argument("--config", help="评测配置文件路径")
    parser.add_argument("--dataset-path")
    parser.add_argument("--dataset-name")
    parser.add_argument("--output-path")
    parser.add_argument("--model")
    parser.add_argument("--predictions-file")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--device")
    parser.add_argument("--dtype")
    parser.add_argument(
        "--use-vllm", action="store_true", help="使用 vLLM 引擎进行推理（大幅加速）"
    )
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float)
    args = parser.parse_args()

    file_config: dict = {}
    if args.config:
        file_config = _load_json(args.config)

    cli_overrides = {
        "dataset_path": args.dataset_path,
        "dataset_name": args.dataset_name,
        "output_path": args.output_path,
        "model": args.model,
        "predictions_file": args.predictions_file,
        "max_samples": args.max_samples,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "device": args.device,
        "dtype": args.dtype,
        "use_vllm": args.use_vllm if args.use_vllm else None,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    merged = _merge_configs(file_config, cli_overrides)

    if not merged.get("model") and not merged.get("predictions_file"):
        parser.error("必须提供 --model 或 --predictions-file")

    return _from_dict(EvalConfig, merged)
