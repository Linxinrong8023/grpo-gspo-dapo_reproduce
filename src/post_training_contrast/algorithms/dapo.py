"""
algorithms/dapo.py - DAPO（Decoupled Clip and Dynamic sAmpling Policy Optimization）算法。

论文：DAPO: An Open-Source LLM Reinforcement Learning System at Scale

核心特性（相比 GRPO）：
  1. Decoupled Clipping：下界和上界用不同的 epsilon（放宽上界保留更大探索空间）
  2. Dynamic Sampling：过滤全对或全错的 group，只保留 0 < acc < 1 的 group
  3. Overlong Reward Penalty：对超出 soft/hard 长度限制的 response 施加线性惩罚
"""

from __future__ import annotations

import torch

from src.post_training_contrast.algorithms.base import (
    Algorithm,
    PolicyLossBatch,
    PolicyLossOutput,
    build_loss_output,
    compute_clip_fraction,
    compute_sequence_ratio,
    masked_mean,
    validate_policy_loss_batch,
)


# ── DAPO 核心 objective ──────────────────────────────────────────


def compute_dapo_loss(
    batch: PolicyLossBatch,
    clip_epsilon_low: float = 0.2,
    clip_epsilon_high: float = 0.28,
) -> PolicyLossOutput:
    """DAPO 的 Decoupled Clipping token-level objective。

    与 GRPO 的区别：lower bound 和 upper bound 的 epsilon 分开设置。
    论文默认 clip_epsilon_low=0.2, clip_epsilon_high=0.28。
    loss 用全局 token 均值（与 GRPO 的 per-seq mean 不同，DAPO 论文明确指定）。
    DAPO 论文移除了 reference KL penalty，因此这里不向 loss 加 KL 项。
    """
    validate_policy_loss_batch(batch)

    token_advantages = batch.advantages.unsqueeze(1).expand_as(batch.logprobs)
    log_ratio = batch.logprobs - batch.old_logprobs
    ratio = torch.exp(log_ratio)
    clipped_ratio = ratio.clamp(1.0 - clip_epsilon_low, 1.0 + clip_epsilon_high)

    surrogate = ratio * token_advantages
    clipped_surrogate = clipped_ratio * token_advantages
    per_token_objective = torch.minimum(surrogate, clipped_surrogate)

    # DAPO 用全局 token 均值（所有 token 等权）
    loss = -masked_mean(per_token_objective, batch.token_mask)

    return build_loss_output(
        batch=batch,
        loss=loss,
        clip_fraction=compute_clip_fraction(ratio, clipped_ratio, batch.token_mask),
        mean_token_ratio=(ratio * batch.token_mask).sum() / batch.token_mask.sum().clamp_min(1.0),
        sequence_ratio_mean=compute_sequence_ratio(batch).mean(),
    )


# ── DAPO 专属预处理 ──────────────────────────────────────────────


def build_dynamic_sampling_mask(rewards: torch.Tensor, group_size: int) -> torch.Tensor:
    """过滤全对或全错的 group，保留 0 < accuracy < 1 的 group。

    DAPO 论文要求：只有一个 group 内同时有正确和错误答案时，
    该 group 的优势值才有意义（normalized advantage 不会退化为 0）。

    返回形状与 rewards 相同的 bool/float mask（1 = 保留，0 = 过滤）。
    整组被保留或整组被丢弃，不会破坏 group 完整性。
    """
    if rewards.ndim != 1:
        raise ValueError("rewards 必须是一维张量。")
    if rewards.shape[0] % group_size != 0:
        raise ValueError("rewards 数量必须能被 group_size 整除。")

    grouped = rewards.view(-1, group_size)
    correct_count = grouped.gt(0).sum(dim=1)
    # 保留 0 < correct_count < group_size 的 group
    keep_group = correct_count.gt(0) & correct_count.lt(group_size)
    return keep_group.unsqueeze(1).expand(-1, group_size).reshape(-1)


def apply_overlong_reward_penalty(
    base_rewards: torch.Tensor,
    response_lengths: torch.Tensor,
    max_response_length: int,
    cache_length: int,
) -> torch.Tensor:
    """对超长 response 施加 soft overlong punishment。

    区间划分：
      - length ≤ safe_length              无惩罚
      - safe_length < length ≤ max_len    线性从 0 惩罚到 -1（soft zone）
      - length > max_len                  固定惩罚 -1（hard zone）

    其中 safe_length = max_response_length - cache_length。
    """
    if base_rewards.shape != response_lengths.shape:
        raise ValueError("base_rewards 和 response_lengths 的形状必须一致。")

    response_lengths = response_lengths.to(base_rewards.dtype)
    safe_length = max_response_length - cache_length

    penalties = torch.zeros_like(base_rewards)

    # Soft zone：超出 safe_length 的比例线性化为负值
    soft_mask = (response_lengths > safe_length) & (response_lengths <= max_response_length)
    hard_mask = response_lengths > max_response_length

    penalties[soft_mask] = -(
        (response_lengths[soft_mask] - safe_length) / cache_length
    )
    penalties[hard_mask] = -1.0

    return base_rewards + penalties


# ── DAPO Algorithm 类 ────────────────────────────────────────────


class DAPOAlgorithm(Algorithm):
    """DAPO 算法实现。

    在 prepare_batch 中处理 overlong penalty 和 dynamic sampling，
    这两步必须在 advantage 计算之前完成。

    DAPO 论文明确移除了 reference model KL penalty，因此：
      - make_policy_loss_batch 强制 ref_logprobs=None，屏蔽上游误传
      - compute_loss 不加任何 KL 惩罚项
    """

    name = "dapo"

    def make_policy_loss_batch(
        self,
        rollout_batch,
        current_logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor | None,
    ) -> PolicyLossBatch:
        """DAPO 不使用 reference model，强制 ref_logprobs=None。

        即使上游（trainer）误传了 ref_logprobs，也在此处丢弃，
        确保 DAPO 的 loss 计算路径与 ref model 完全解耦。
        """
        if ref_logprobs is not None:
            # 防御性警告：正常情况下 DAPO 的 reference_policy=None，不会产生 ref_logprobs
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "[DAPOAlgorithm] ref_logprobs was passed but DAPO does not use a "
                "reference model; ignoring ref_logprobs to avoid incorrect KL computation."
            )
        return PolicyLossBatch(
            logprobs=current_logprobs,
            old_logprobs=rollout_batch.old_logprobs,
            advantages=rollout_batch.advantages,
            token_mask=rollout_batch.token_mask,
            ref_logprobs=None,  # DAPO 明确不使用 reference model
        )

    def prepare_batch(self, rollout_batch, config):
        """应用 DAPO 专属的 batch 预处理。

        执行顺序（必须）：
          1. Overlong reward penalty（先修正 reward）
          2. Dynamic sampling filter（再过滤 group）

        注意：dynamic sampling 过滤整组，保留的 group 仍然是完整的 group_size，
        后续 compute_group_relative_advantages 可以正确计算。
        """
        prepared = rollout_batch

        if config.dapo_use_overlong_penalty and prepared.rewards is not None:
            cache_length = max(1, config.max_new_tokens // 4)
            prepared.rewards = apply_overlong_reward_penalty(
                base_rewards=prepared.rewards,
                response_lengths=prepared.response_lengths,
                max_response_length=config.max_new_tokens,
                cache_length=cache_length,
            )

        dynamic_sampling_rewards = (
            prepared.answer_rewards
            if prepared.answer_rewards is not None
            else prepared.rewards
        )
        if config.dapo_use_dynamic_sampling and dynamic_sampling_rewards is not None:
            keep_mask = build_dynamic_sampling_mask(
                dynamic_sampling_rewards,
                group_size=prepared.group_size,
            )
            keep_indices = keep_mask.nonzero(as_tuple=False).view(-1).tolist()
            prepared = prepared.slice(keep_indices)

        return prepared

    def compute_loss(self, batch: PolicyLossBatch, config) -> PolicyLossOutput:
        return compute_dapo_loss(
            batch=batch,
            clip_epsilon_low=config.dapo_clip_epsilon_low,
            clip_epsilon_high=config.dapo_clip_epsilon_high,
        )
