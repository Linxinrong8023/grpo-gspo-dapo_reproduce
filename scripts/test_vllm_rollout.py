import argparse
import json
import os
import sys
from pathlib import Path

# Ensure PROJECT_ROOT is in sys.path before importing from src
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.post_training_contrast.rollout.vllm_rollout import VllmRolloutEngine

"""
单独测试 vLLM rollout 的入口脚本。
"""


def load_json_config(config_path: str) -> dict:
    """功能：
    - 读取 rollout 测试配置
    - 当前统一使用 JSON，减少依赖和心智负担
    """
    with open(config_path, "r", encoding="utf-8") as file:
        return json.load(file)


def merge_cli_args_with_config(args: argparse.Namespace, config: dict) -> dict:
    """功能：
    - 合并配置文件和命令行参数
    - 命令行参数优先级更高
    """
    merged = dict(config)
    for key, value in vars(args).items():
        if key == "config":
            continue
        if value is not None:
            merged[key] = value
    return merged


def normalize_prompt_inputs(prompt: str | None, prompt_file: str | None) -> list[str]:
    """功能：
    - 统一整理 rollout 测试时要发给 vLLM 的 prompt 列表
    - 支持直接传一个 prompt，也支持从文本文件逐行读取
    """
    if prompt:
        return [prompt]

    prompts = []
    with open(prompt_file, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                prompts.append(line)
    return prompts


def build_rollout_engine(config: dict) -> VllmRolloutEngine:
    """功能：
    - 按测试配置构造一个最小 rollout engine
    - 只打开 vLLM 单测真正需要的字段
    """
    return VllmRolloutEngine(
        model_name_or_path=config["model"],
        max_new_tokens=config.get("max_new_tokens", 64),
        temperature=config.get("temperature", 0.0),
        top_p=config.get("top_p", 1.0),
        tensor_parallel_size=config.get("tensor_parallel_size", 1),
        gpu_memory_utilization=config.get("gpu_memory_utilization", 0.9),
        max_model_len=config.get("max_model_len"),
        trust_remote_code=True,
        dtype=config.get("dtype", "auto"),
    )


def main():
    parser = argparse.ArgumentParser(description="单独测试 vLLM rollout")
    parser.add_argument("--config", default=None, help="rollout 测试配置文件路径")
    parser.add_argument("--model", default=None, help="模型路径")
    parser.add_argument("--prompt", default=None, help="直接传一个 prompt")
    parser.add_argument("--prompt-file", default=None, help="按行读取 prompt 的文本文件")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    args = parser.parse_args()

    config = load_json_config(args.config) if args.config else {}
    final_args = merge_cli_args_with_config(args, config)

    if not final_args.get("model"):
        parser.error("必须提供 --model 或在配置文件中写 model。")
    if not final_args.get("prompt") and not final_args.get("prompt_file"):
        parser.error("必须提供 --prompt 或 --prompt-file。")

    prompts = normalize_prompt_inputs(
        prompt=final_args.get("prompt"),
        prompt_file=final_args.get("prompt_file"),
    )
    rollout = build_rollout_engine(final_args)
    rollout_batch = rollout.generate(
        prompts=prompts,
        num_samples_per_prompt=final_args.get("num_samples", 1),
    )

    print(f"num_prompts={len(prompts)}")
    print(f"num_samples={len(rollout_batch.samples)}")
    print(f"group_size={rollout_batch.group_size}")

    for index, sample in enumerate(rollout_batch.samples, start=1):
        print(f"\n===== sample {index} =====")
        print(f"group_id={sample.group_id}")
        print(f"sample_id_within_group={sample.sample_id_within_group}")
        print(f"finish_reason={sample.finish_reason}")
        print(f"is_truncated={sample.is_truncated}")
        print(f"response_token_count={len(sample.response_token_ids)}")
        print(f"logprob_count={len(sample.response_logprobs)}")
        print("response_text:")
        print(sample.response_text)


if __name__ == "__main__":
    main()
