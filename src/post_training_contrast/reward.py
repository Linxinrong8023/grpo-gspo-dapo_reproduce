"""
reward.py - RL 训练中的奖励函数。

只依赖 math_utils，无其他内部依赖。
被 trainer.py 在每次 rollout 后调用，计算 sample 级 reward。
"""

from __future__ import annotations

from dataclasses import dataclass

from src.post_training_contrast.math_utils import batch_evaluate_rewards
from src.post_training_contrast.math_utils import (
    normalize_answer,
    parse_final_answer,
    symbolic_equals,
)


@dataclass(frozen=True)
class RewardBreakdown:
    """训练阶段单条样本的 reward 拆解。"""

    answer_reward: float
    format_reward: float
    total_reward: float
    is_correct: bool
    parse_failed: bool
    has_think_close: bool
    has_answer_tag: bool


def answers_match(predicted: str, ground_truth: str) -> bool:
    """判断预测答案和标准答案是否等价。

    比较流程：
      1. 轻量归一化后字符串精确匹配
      2. 符号等价比较（数值 / 元组 / SymPy）
    """
    norm_pred = normalize_answer(predicted)
    norm_gt = normalize_answer(ground_truth)

    if not norm_pred or not norm_gt:
        return False
    if norm_pred == norm_gt:
        return True
    return symbolic_equals(predicted, ground_truth)


def compute_binary_reward(
    response: str,
    ground_truth_answer: str,
    dataset_name: str = "math",
) -> float:
    """从模型输出里解析最终答案，根据是否正确返回 1.0 / 0.0。

    Parameters
    ----------
    response : str
        模型的完整输出文本。
    ground_truth_answer : str
        标准答案字符串。
    dataset_name : str
        数据集名称，影响答案提取策略（math/math500 vs gsm8k）。
    """
    predicted = parse_final_answer(response, dataset_name)
    return 1.0 if answers_match(predicted, ground_truth_answer) else 0.0


def _has_answer_tag(response: str, dataset_name: str) -> bool:
    """判断闭合 </think> 后是否显式出现了数据集要求的答案标签。"""
    answer_segment = _answer_segment_after_think(response)
    if not answer_segment:
        return False
    if dataset_name.lower() == "gsm8k":
        return "####" in answer_segment
    return "\\boxed{" in answer_segment


def _answer_segment_after_think(response: str) -> str:
    think_end = response.rfind("</think>")
    if think_end == -1:
        return ""
    return response[think_end + len("</think>"):].strip()


def _parse_training_final_answer(response: str, dataset_name: str) -> str:
    """训练使用 forced-<think> prompt；未生成 </think> 时不能解析最终答案。"""
    if "</think>" not in response:
        return ""
    return parse_final_answer(response, dataset_name)


def _as_penalty(value: float) -> float:
    return -abs(value)


def _build_reward_breakdown(
    response: str,
    predicted: str,
    is_correct: bool,
    dataset_name: str,
    answer_correct_reward: float,
    answer_incorrect_reward: float,
    missing_think_close_penalty: float,
    missing_answer_tag_penalty: float,
) -> RewardBreakdown:
    has_think_close = "</think>" in response
    has_answer_tag = _has_answer_tag(response, dataset_name)

    answer_reward = answer_correct_reward if is_correct else answer_incorrect_reward
    format_reward = 0.0
    if not has_think_close:
        format_reward += _as_penalty(missing_think_close_penalty)
    elif not has_answer_tag:
        format_reward += _as_penalty(missing_answer_tag_penalty)

    return RewardBreakdown(
        answer_reward=answer_reward,
        format_reward=format_reward,
        total_reward=answer_reward + format_reward,
        is_correct=is_correct,
        parse_failed=not predicted,
        has_think_close=has_think_close,
        has_answer_tag=has_answer_tag,
    )


def compute_reward_breakdown(
    response: str,
    ground_truth_answer: str,
    dataset_name: str = "math",
    answer_correct_reward: float = 1.0,
    answer_incorrect_reward: float = 0.0,
    missing_think_close_penalty: float = -0.3,
    missing_answer_tag_penalty: float = -0.15,
) -> RewardBreakdown:
    """计算训练阶段使用的答案奖励、格式奖励和总奖励。

    格式项采用 penalty shaping：
    - 格式正确时不额外加分，避免把格式当成可刷的正奖励；
    - 格式错误时不看答案对错直接扣分，避免训练早期的格式冷启动问题。
    """
    predicted = _parse_training_final_answer(response, dataset_name)
    is_correct = answers_match(predicted, ground_truth_answer)
    return _build_reward_breakdown(
        response=response,
        predicted=predicted,
        is_correct=is_correct,
        dataset_name=dataset_name,
        answer_correct_reward=answer_correct_reward,
        answer_incorrect_reward=answer_incorrect_reward,
        missing_think_close_penalty=missing_think_close_penalty,
        missing_answer_tag_penalty=missing_answer_tag_penalty,
    )


def compute_batch_reward_breakdowns(
    responses,
    ground_truth_answers,
    dataset_name: str = "math",
    answer_correct_reward: float = 1.0,
    answer_incorrect_reward: float = 0.0,
    missing_think_close_penalty: float = -0.3,
    missing_answer_tag_penalty: float = -0.15,
) -> list[RewardBreakdown]:
    """批量解析与判分；当前始终使用单线程 reward 判分。"""
    response_list = list(responses)
    answer_list = list(ground_truth_answers)
    if len(response_list) != len(answer_list):
        raise ValueError("responses 和 ground_truth_answers 长度必须一致")

    predicted_answers = [
        _parse_training_final_answer(response, dataset_name)
        for response in response_list
    ]
    correctness_scores = batch_evaluate_rewards(
        predicted_answers,
        answer_list,
    )

    return [
        _build_reward_breakdown(
            response=response,
            predicted=predicted,
            is_correct=bool(score),
            dataset_name=dataset_name,
            answer_correct_reward=answer_correct_reward,
            answer_incorrect_reward=answer_incorrect_reward,
            missing_think_close_penalty=missing_think_close_penalty,
            missing_answer_tag_penalty=missing_answer_tag_penalty,
        )
        for response, predicted, score in zip(
            response_list,
            predicted_answers,
            correctness_scores,
        )
    ]
