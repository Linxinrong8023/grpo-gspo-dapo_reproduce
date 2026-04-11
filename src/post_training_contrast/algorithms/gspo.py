"""
algorithms/gspo.py - GSPO（Group Sequence Policy Optimization）算法。

核心：sequence-level importance ratio + clipping，reward 和 clipping 都对齐到完整序列层。
"""

from __future__ import annotations

import torch

from src.post_training_contrast.algorithms.base import (
    Algorithm,
    PolicyLossBatch,
    PolicyLossOutput,
    build_loss_output,
    compute_reference_kl,
    compute_sequence_ratio,
    validate_policy_loss_batch,
)


def compute_gspo_loss(
    batch: PolicyLossBatch,
    clip_epsilon: float = 0.2,
    kl_beta: float = 0.0,
) -> PolicyLossOutput:
    """GSPO sequence-level clipped objective。

    ratio 是整条序列的 importance ratio（长度归一化后 exp），
    clipping 也在序列层面做，不是 token 层面。
    """
    validate_policy_loss_batch(batch)

    sequence_ratio = compute_sequence_ratio(batch)
    clipped_ratio = sequence_ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)

    surrogate = sequence_ratio * batch.advantages
    clipped_surrogate = clipped_ratio * batch.advantages
    sequence_objective = torch.minimum(surrogate, clipped_surrogate)

    loss = -sequence_objective.mean()

    if kl_beta > 0.0 and batch.ref_logprobs is not None:
        loss = loss + kl_beta * compute_reference_kl(batch)

    clip_fraction = ((sequence_ratio - clipped_ratio).abs() > 1e-8).to(sequence_ratio.dtype).mean()

    return build_loss_output(
        batch=batch,
        loss=loss,
        clip_fraction=clip_fraction,
        mean_token_ratio=torch.exp(batch.logprobs - batch.old_logprobs)
        .mul(batch.token_mask)
        .sum()
        / batch.token_mask.sum().clamp_min(1.0),
        sequence_ratio_mean=sequence_ratio.mean(),
    )


class GSPOAlgorithm(Algorithm):
    """GSPO 算法实现。"""

    name = "gspo"

    def compute_loss(self, batch: PolicyLossBatch, config) -> PolicyLossOutput:
        return compute_gspo_loss(
            batch=batch,
            clip_epsilon=config.clip_epsilon,
            kl_beta=config.kl_beta,
        )
