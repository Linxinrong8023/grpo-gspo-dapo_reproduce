"""
algorithms/grpo.py - GRPO（Group Relative Policy Optimization）算法。

论文：DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models

核心：token-level clipped surrogate objective，每条序列等权（先 per-seq 均值再 batch 均值）。
"""

from __future__ import annotations

import torch

from src.post_training_contrast.algorithms.base import (
    Algorithm,
    PolicyLossBatch,
    PolicyLossOutput,
    build_loss_output,
    compute_clip_fraction,
    compute_reference_kl,
    compute_sequence_ratio,
    masked_mean,
    validate_policy_loss_batch,
)


def compute_grpo_loss(
    batch: PolicyLossBatch,
    clip_epsilon: float = 0.2,
    kl_beta: float = 0.0,
) -> PolicyLossOutput:
    """GRPO token-level clipped objective。

    Loss = -(1/G) * Σᵢ (1/|oᵢ|) * Σₜ min(r·Â, clip(r,1-ε,1+ε)·Â)

    等权对待每条序列（masked_mean dim=1），而非全局 token 均值。
    """
    validate_policy_loss_batch(batch)

    token_advantages = batch.advantages.unsqueeze(1).expand_as(batch.logprobs)
    log_ratio = batch.logprobs - batch.old_logprobs
    ratio = torch.exp(log_ratio)
    clipped_ratio = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)

    surrogate = ratio * token_advantages
    clipped_surrogate = clipped_ratio * token_advantages
    per_token_objective = torch.minimum(surrogate, clipped_surrogate)

    # 先 per-seq 均值，再 batch 均值（= 每条序列等权）
    loss = -masked_mean(per_token_objective, batch.token_mask, dim=1).mean()

    if kl_beta > 0.0 and batch.ref_logprobs is not None:
        loss = loss + kl_beta * compute_reference_kl(batch)

    return build_loss_output(
        batch=batch,
        loss=loss,
        clip_fraction=compute_clip_fraction(ratio, clipped_ratio, batch.token_mask),
        mean_token_ratio=(ratio * batch.token_mask).sum() / batch.token_mask.sum().clamp_min(1.0),
        sequence_ratio_mean=compute_sequence_ratio(batch).mean(),
    )


class GRPOAlgorithm(Algorithm):
    """GRPO 算法实现。"""

    name = "grpo"

    def compute_loss(self, batch: PolicyLossBatch, config) -> PolicyLossOutput:
        return compute_grpo_loss(
            batch=batch,
            clip_epsilon=config.clip_epsilon,
            kl_beta=config.kl_beta,
        )
