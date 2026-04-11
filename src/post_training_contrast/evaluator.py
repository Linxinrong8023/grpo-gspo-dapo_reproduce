"""
evaluator.py - 评测器，输出三个论文核心指标。

三个指标（对齐 GRPO / DAPO / GSPO 论文标准）：
  1. accuracy          — Greedy pass@1，写进论文主表格
  2. mean_length       — 平均回复 token 数，证明模型没有啰嗦骗分
  3. format_fail_rate  — 答案提取失败率，证明格式没被训崩

额外输出：按 MATH Level 1-5 切片的 accuracy（分析难度泛化性）。

公开接口：run_evaluation(config: EvalConfig) -> dict
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from pathlib import Path

import torch

from src.post_training_contrast.config import EvalConfig
from src.post_training_contrast.math_utils import batch_evaluate_rewards
from src.post_training_contrast.math_utils import parse_final_answer
from src.post_training_contrast.reward import answers_match

logger = logging.getLogger(__name__)


# ── Prompt 构造 ──────────────────────────────────────────────────


def build_user_message(record: dict, dataset_name: str) -> str:
    """根据数据集类型构建用户提问的纯文本内容（不含任何模型 chat 格式）。

    ⚠️  这个函数必须和 trainer.py 里 rollout 使用的完全一致！
        任何修改都要两边同步，否则 eval 分布和训练分布不匹配。
    """
    question = record.get("problem", record.get("question", ""))
    name = dataset_name.lower()

    if name in ("gsm8k",):
        return (
            f"{question}\n\n"
            "Please reason step by step inside <think>...</think> tags, "
            "then write your final numeric answer after ####."
        )
    # math / math500 / 其他默认走 boxed 格式
    return (
        f"{question}\n\n"
        "Reason step by step in the <think> block. "
        "After closing </think>, immediately write exactly one final answer line: "
        "\\boxed{answer}. "
        "Do not put the final boxed answer inside <think>."
    )


def format_chat_prompt(user_message: str, tokenizer) -> str:
    """用 tokenizer 自带的 chat template 把用户消息包装成模型能识别的对话格式。

    使用 <think> 推理格式：
      - 系统提示词告诉模型先思考再作答
      - 在 assistant 开头强制追加 <think>\n，让模型直接进入推理模式
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful and harmless assistant. "
                "You should think step-by-step inside <think>...</think> tags "
                "before providing your final answer."
            ),
        },
        {"role": "user", "content": user_message},
    ]
    # tokenize=False → 返回纯文本字符串；add_generation_prompt=True → 末尾加上 assistant 开头
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # 强制以 <think>\n 开头，让模型跳过犹豫、直接进入推理模式
    # （这与 DeepSeek-R1 / QwQ 的 forced-think 技巧完全一致）
    return prompt + "<think>\n"


def build_eval_prompt(record: dict, dataset_name: str, tokenizer=None) -> str:
    """统一入口：构建完整 prompt。

    - 有 tokenizer 时 → 自动适配模型的 chat template（推荐）
    - 无 tokenizer 时 → 返回纯文本用户消息（向后兼容 / 测试用）
    """
    user_msg = build_user_message(record, dataset_name)
    if tokenizer is not None:
        return format_chat_prompt(user_msg, tokenizer)
    return user_msg


# ── 单题判分 ─────────────────────────────────────────────────────


def score_prediction(
    record: dict,
    prediction: str,
    response_length: int,
    dataset_name: str,
    is_truncated: bool = False,
    parsed_prediction: str | None = None,
    is_correct: bool | None = None,
) -> dict:
    """对单条预测做判分，同时记录格式失败和 token 长度。

    Parameters
    ----------
    response_length : int
        实际生成的 token 数（不含 prompt）。
    """
    parsed = (
        parsed_prediction
        if parsed_prediction is not None
        else parse_final_answer(prediction, dataset_name)
    )
    format_failed = parsed is None or parsed.strip() == ""
    # 强制转换成原生 bool，防止 SymPy 引擎返回 BooleanFalse/BooleanTrue 导致后续报错
    if format_failed:
        correct = False
    elif is_correct is not None:
        correct = bool(is_correct)
    else:
        correct = bool(answers_match(parsed, record["answer"]))

    return {
        "id": record.get("id", ""),
        "is_correct": correct,
        "format_failed": format_failed,
        "response_length": response_length,
        "is_truncated": is_truncated,
        "parsed_prediction": parsed or "",
        "ground_truth": record["answer"],
        "level": record.get("level"),  # MATH Level 1-5
        "subject": record.get("subject"),  # Algebra / Geometry / ...
        "raw_prediction": prediction,
    }


# ── 汇总统计（三大指标 + 切片）────────────────────────────────────


def summarize_scores(
    scored_records: list[dict],
    dataset_name: str,
) -> dict:
    """把逐题判分汇总成三大核心指标和 Level 切片。"""
    n = len(scored_records)
    if n == 0:
        return {"dataset_name": dataset_name, "num_samples": 0}

    num_correct = sum(1 for s in scored_records if s["is_correct"])
    num_format_fail = sum(1 for s in scored_records if s["format_failed"])
    num_truncated = sum(1 for s in scored_records if s.get("is_truncated", False))
    total_length = sum(s["response_length"] for s in scored_records)
    lengths = sorted(s["response_length"] for s in scored_records)

    # ── 三大核心指标 ──
    accuracy = num_correct / n
    format_fail_rate = num_format_fail / n
    truncation_rate = num_truncated / n
    mean_length = total_length / n
    mid = n // 2
    median_length = lengths[mid] if n % 2 else (lengths[mid - 1] + lengths[mid]) / 2
    p90_index = min(n - 1, math.ceil(0.9 * n) - 1)
    p90_length = lengths[p90_index]

    # ── 按 Level / subject 切片（各维度的 accuracy）──
    level_counters: dict = defaultdict(lambda: {"total": 0, "correct": 0})
    subject_counters: dict = defaultdict(lambda: {"total": 0, "correct": 0})
    for s in scored_records:
        if s["level"] is not None:
            lv = str(s["level"])
            level_counters[lv]["total"] += 1
            level_counters[lv]["correct"] += int(s["is_correct"])
        if s["subject"] is not None:
            subj = str(s["subject"])
            subject_counters[subj]["total"] += 1
            subject_counters[subj]["correct"] += int(s["is_correct"])

    def _to_acc(counter_dict: dict) -> dict:
        return {
            k: {
                "accuracy": v["correct"] / v["total"] if v["total"] else 0.0,
                "correct": v["correct"],
                "total": v["total"],
            }
            for k, v in sorted(counter_dict.items())
        }

    return {
        "dataset_name": dataset_name,
        "num_samples": n,
        # ── 三大核心指标 ──────────────────────────────────────────
        "accuracy": round(accuracy, 4),
        "format_fail_rate": round(format_fail_rate, 4),
        "truncation_rate": round(truncation_rate, 4),
        "mean_length": round(mean_length, 1),
        "median_length": round(median_length, 1),
        "p90_length": round(p90_length, 1),
        # ── 计数明细 ──────────────────────────────────────────────
        "num_correct": num_correct,
        "num_format_fail": num_format_fail,
        "num_truncated": num_truncated,
        # ── 切片 ──────────────────────────────────────────────────
        "by_level": _to_acc(level_counters),
        "by_subject": _to_acc(subject_counters),
        # ── 每题详情（可用于 error analysis）─────────────────────
        "details": scored_records,
    }


def _eval_config_snapshot(config: EvalConfig) -> dict:
    """保存评测配置的可 JSON 序列化快照，便于结果文件自解释。"""
    return {
        "dataset_path": config.dataset_path,
        "dataset_name": config.dataset_name,
        "output_path": config.output_path,
        "model": config.model,
        "predictions_file": config.predictions_file,
        "max_samples": config.max_samples,
        "batch_size": config.batch_size,
        "max_new_tokens": config.max_new_tokens,
        "device": config.device,
        "dtype": config.dtype,
        "use_vllm": config.use_vllm,
        "tensor_parallel_size": config.tensor_parallel_size,
        "gpu_memory_utilization": config.gpu_memory_utilization,
    }


# ── 模型推理（HuggingFace Greedy 解码）──────────────────────────


def _resolve_device_dtype(config: EvalConfig) -> tuple[str, str]:
    cuda_ok = torch.cuda.is_available()
    mps_ok = (
        getattr(getattr(torch, "backends", None), "mps", None)
        and torch.backends.mps.is_available()
    )
    bf16_ok = cuda_ok and torch.cuda.is_bf16_supported()

    device = (
        ("cuda" if cuda_ok else "mps" if mps_ok else "cpu")
        if config.device == "auto"
        else config.device
    )
    if config.dtype == "auto":
        dtype = (
            "bfloat16"
            if (device == "cuda" and bf16_ok)
            else "float16"
            if device == "cuda"
            else "float32"
        )
    else:
        dtype = config.dtype
    return device, dtype


def _generate_with_hf(
    records: list[dict],
    config: EvalConfig,
    dataset_name: str,
) -> tuple[list[str], list[int], list[bool]]:
    """Greedy 解码生成全部预测，同时返回每条的实际 token 数。

    Returns
    -------
    predictions : list[str]
        每条记录解码后的文本。
    response_lengths : list[int]
        每条记录实际生成的 token 数（不含 prompt，不含 padding）。
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm

    device, dtype_name = _resolve_device_dtype(config)
    logger.info(
        "Loading model %s  device=%s  dtype=%s", config.model, device, dtype_name
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs: dict = {
        "trust_remote_code": True,
        "torch_dtype": getattr(torch, dtype_name),
        "low_cpu_mem_usage": True,
    }
    if device == "cuda":
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(config.model, **model_kwargs)
    if device in {"cpu", "mps"}:
        model.to(device)
    model.eval()

    predictions: list[str] = []
    response_lengths: list[int] = []
    truncated_flags: list[bool] = []
    eos_id = tokenizer.eos_token_id

    pbar = tqdm(total=len(records), desc="Generating (greedy)")
    for i in range(0, len(records), config.batch_size):
        batch = records[i : i + config.batch_size]
        prompts = [
            build_eval_prompt(r, dataset_name, tokenizer=tokenizer) for r in batch
        ]
        model_device = next(model.parameters()).device
        inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(model_device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=config.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )

        new_tokens = out[:, inputs["input_ids"].shape[1] :]  # shape: [batch, gen_len]

        # 计算每条序列的真实 token 数（到第一个 EOS 为止）
        for seq in new_tokens:
            eos_positions = (seq == eos_id).nonzero(as_tuple=True)[0]
            length = (
                int(eos_positions[0].item()) + 1 if len(eos_positions) > 0 else len(seq)
            )
            response_lengths.append(length)
            truncated_flags.append(
                len(eos_positions) == 0 and length >= config.max_new_tokens
            )

        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        predictions.extend(decoded)
        pbar.update(len(batch))

    pbar.close()
    return predictions, response_lengths, truncated_flags


def _generate_with_vllm(
    records: list[dict],
    config: EvalConfig,
    dataset_name: str,
) -> tuple[list[str], list[int], list[bool]]:
    """使用 vLLM 进行高速推理。"""
    from .rollout import VllmRolloutEngine
    from transformers import AutoTokenizer

    logger.info("Initializing vLLM Engine for evaluation...")
    tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=True)
    rollout_engine = VllmRolloutEngine(
        model_name_or_path=config.model,
        max_new_tokens=config.max_new_tokens,
        temperature=0.0,  # Greedy decoding for eval
        top_p=1.0,
        tensor_parallel_size=config.tensor_parallel_size,
        gpu_memory_utilization=config.gpu_memory_utilization,
        trust_remote_code=True,
        dtype=config.dtype if config.dtype != "auto" else "bfloat16",
    )
    predictions: list[str] = []
    response_lengths: list[int] = []
    truncated_flags: list[bool] = []
    batch_size = max(1, config.batch_size)

    try:
        logger.info("Starting vLLM text generation...")
        for start_idx in range(0, len(records), batch_size):
            batch_records = records[start_idx : start_idx + batch_size]
            prompts = [
                build_eval_prompt(record, dataset_name, tokenizer=tokenizer)
                for record in batch_records
            ]
            rollout_batch = rollout_engine.generate(
                prompts=prompts,
                num_samples_per_prompt=1,
                return_logprobs=False,
            )
            for sample in rollout_batch.samples:
                predictions.append(sample.response_text)
                response_lengths.append(len(sample.response_token_ids))
                truncated_flags.append(sample.is_truncated)
    finally:
        rollout_engine.release()

    return predictions, response_lengths, truncated_flags


# ── 内部工具 ─────────────────────────────────────────────────────


def _guess_dataset_name(dataset_path: str) -> str:
    """从数据集路径推断 dataset_name，用于未显式指定时的 fallback。

    推断规则（按优先级）：
      - 路径中包含 'gsm8k'  → 'gsm8k'
      - 路径中包含 'math500' → 'math500'
      - 否则               → 'math'（默认）
    """
    path_lower = dataset_path.lower()
    if "gsm8k" in path_lower:
        return "gsm8k"
    if "math500" in path_lower:
        return "math500"
    return "math"


# ── 公开入口 ─────────────────────────────────────────────────────


def run_evaluation(config: EvalConfig) -> dict:
    """评测主流程：加载数据 → Greedy 推理 → 三指标判分 → 保存结果。

    Returns
    -------
    dict
        包含 accuracy / format_fail_rate / truncation_rate /
        mean_length / median_length / p90_length / by_level /
        by_subject / details 的汇总结果。
    """
    # ── 1. 加载数据 ──────────────────────────────────────────────
    records: list[dict] = []
    with open(config.dataset_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "id" not in item:
                item["id"] = f"sample-{idx}"
            records.append(item)

    if config.max_samples is not None:
        records = records[: config.max_samples]
    logger.info("Loaded %d problems from %s", len(records), config.dataset_path)
    dataset_name = config.dataset_name or _guess_dataset_name(config.dataset_path)

    # ── 2. 获取预测 ──────────────────────────────────────────────
    if config.predictions_file:
        logger.info("Loading predictions from %s", config.predictions_file)
        with open(config.predictions_file, "r", encoding="utf-8") as f:
            pred_records = [json.loads(line) for line in f if line.strip()]
        predictions = [r["prediction"] for r in pred_records]
        # 离线文件可能没有 length，退 fallback 到字符数 / 4
        response_lengths = [
            r.get("response_length", len(r["prediction"]) // 4) for r in pred_records
        ]
        truncated_flags = [bool(r.get("is_truncated", False)) for r in pred_records]
    elif config.model:
        if getattr(config, "use_vllm", False):
            predictions, response_lengths, truncated_flags = _generate_with_vllm(
                records, config, dataset_name
            )
        else:
            predictions, response_lengths, truncated_flags = _generate_with_hf(
                records, config, dataset_name
            )
    else:
        raise ValueError("必须提供 model 或 predictions_file。")

    assert len(predictions) == len(records), "预测数量和样本数量不一致"

    # ── 3. 逐题判分 ──────────────────────────────────────────────
    parsed_predictions = [
        parse_final_answer(prediction, dataset_name) for prediction in predictions
    ]
    correctness_scores = batch_evaluate_rewards(
        parsed_predictions,
        [record["answer"] for record in records],
    )
    scored = [
        score_prediction(
            r,
            p,
            length,
            dataset_name,
            is_truncated=is_truncated,
            parsed_prediction=parsed,
            is_correct=bool(score),
        )
        for r, p, length, is_truncated, parsed, score in zip(
            records,
            predictions,
            response_lengths,
            truncated_flags,
            parsed_predictions,
            correctness_scores,
        )
    ]

    # ── 4. 汇总三大指标 ──────────────────────────────────────────
    summary = summarize_scores(scored, dataset_name)
    summary["eval_config"] = _eval_config_snapshot(config)
    if config.metadata is not None:
        summary["metadata"] = config.metadata
    logger.info(
        "dataset=%s  n=%d  accuracy=%.4f  format_fail=%.4f  truncation=%.4f  mean_length=%.1f  p90_length=%.1f",
        summary["dataset_name"],
        summary["num_samples"],
        summary["accuracy"],
        summary["format_fail_rate"],
        summary["truncation_rate"],
        summary["mean_length"],
        summary["p90_length"],
    )

    # ── 5. 保存结果 ──────────────────────────────────────────────
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("Results saved → %s", output_path)

    return summary
