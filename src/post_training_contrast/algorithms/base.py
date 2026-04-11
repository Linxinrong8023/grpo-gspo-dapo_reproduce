"""
algorithms/base.py - 共享张量原语、数据结构和 Algorithm 基类。

包含：
  - PolicyLossBatch / PolicyLossOutput    输入/输出数据结构
  - masked_mean / compute_* 系列          共享张量运算
  - compute_group_relative_advantages     GRPO/GSPO/DAPO 通用优势计算
  - build_loss_output                     统一整理返回字段
  - Algorithm                             各算法的基类（Strategy Pattern）
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


# ── 数据结构 ─────────────────────────────────────────────────────


@dataclass
class PolicyLossBatch:
    """一次策略优化所需的最小张量输入。"""

    logprobs: torch.Tensor         # [B, seq_len] 当前策略 logprob
    old_logprobs: torch.Tensor     # [B, seq_len] rollout 时的 logprob
    advantages: torch.Tensor       # [B]           组内相对优势
    token_mask: torch.Tensor       # [B, seq_len] response 有效 token 标记
    ref_logprobs: torch.Tensor | None = None  # [B, seq_len] 参考策略 logprob


@dataclass
class PolicyLossOutput:
    """算法 loss 和常用诊断指标。"""

    loss: torch.Tensor
    clip_fraction: torch.Tensor
    approx_kl: torch.Tensor
    mean_advantage: torch.Tensor
    mean_token_ratio: torch.Tensor
    sequence_ratio_mean: torch.Tensor
    valid_token_count: torch.Tensor
    valid_sequence_count: torch.Tensor
    mean_ref_kl: torch.Tensor


# ── 输入验证 ─────────────────────────────────────────────────────


def validate_policy_loss_batch(batch: PolicyLossBatch) -> None:
    """检查算法输入张量的基本形状是否一致。"""
    if batch.logprobs.shape != batch.old_logprobs.shape:
        raise ValueError("logprobs 和 old_logprobs 的形状必须一致。")
    if batch.logprobs.shape != batch.token_mask.shape:
        raise ValueError("logprobs 和 token_mask 的形状必须一致。")
    if batch.logprobs.ndim != 2:
        raise ValueError("logprobs 必须是 [batch, seq_len] 二维张量。")
    if batch.advantages.ndim != 1:
        raise ValueError("advantages 必须是 [batch] 一维张量。")
    if batch.advantages.shape[0] != batch.logprobs.shape[0]:
        raise ValueError("advantages 的 batch 维必须和 logprobs 一致。")
    if batch.ref_logprobs is not None and batch.ref_logprobs.shape != batch.logprobs.shape:
        raise ValueError("ref_logprobs 的形状必须和 logprobs 一致。")


# ── 共享张量运算 ─────────────────────────────────────────────────


def masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """对有效 token（mask=1）求加权平均。支持全局平均和按维度平均。"""
    mask = mask.to(values.dtype)
    weighted = values * mask
    num = weighted.sum() if dim is None else weighted.sum(dim=dim)
    den = mask.sum() if dim is None else mask.sum(dim=dim)
    return num / den.clamp_min(eps)


def compute_sequence_ratio(batch: PolicyLossBatch) -> torch.Tensor:
    """计算 GSPO 的 sequence-level importance ratio（长度归一化后 exp）。"""
    avg_log_ratio = masked_mean(
        batch.logprobs - batch.old_logprobs,
        batch.token_mask,
        dim=1,
    )
    return torch.exp(avg_log_ratio)


def compute_clip_fraction(
    ratio: torch.Tensor,
    clipped_ratio: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """统计有效位置里有多少比例触发了 clipping。"""
    was_clipped = (ratio - clipped_ratio).abs() > 1e-8
    return masked_mean(was_clipped.to(ratio.dtype), mask)


def compute_approx_kl(batch: PolicyLossBatch, estimator: str = "k3") -> torch.Tensor:
    """计算当前策略相对旧策略的近似 KL（用于诊断，非惩罚项）。"""
    log_ratio = batch.logprobs - batch.old_logprobs
    if estimator == "k2":
        kl_values = 0.5 * log_ratio.square()
    elif estimator == "k3":
        ratio = torch.exp(log_ratio)
        kl_values = ratio - log_ratio - 1.0
    else:
        raise ValueError("approx_kl 只支持 k2 或 k3。")
    return masked_mean(kl_values, batch.token_mask)


def compute_reference_kl(batch: PolicyLossBatch) -> torch.Tensor:
    """用 k3 estimator 计算 KL(π || π_ref)。ref_logprobs 为 None 时返回 0。"""
    if batch.ref_logprobs is None:
        return batch.logprobs.new_tensor(0.0)
    # log(π_ref/π)，k3 estimator: exp(r) - r - 1 ≈ KL(π||π_ref)
    log_ratio = batch.ref_logprobs - batch.logprobs
    kl = torch.exp(log_ratio) - log_ratio - 1.0
    return masked_mean(kl, batch.token_mask)


def compute_group_relative_advantages(
    rewards: torch.Tensor,
    group_size: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """GRPO/GSPO/DAPO 通用的组内相对优势计算。

    同组内做 (r - mean) / std 归一化，输出 shape 与 rewards 相同。
    """
    if rewards.ndim != 1:
        raise ValueError("rewards 必须是一维张量。")
    if rewards.shape[0] % group_size != 0:
        raise ValueError("rewards 数量必须能被 group_size 整除。")
    grouped = rewards.view(-1, group_size)
    normed = (grouped - grouped.mean(dim=1, keepdim=True)) / (
        grouped.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
    )
    return normed.reshape(-1)


def build_loss_output(
    batch: PolicyLossBatch,
    loss: torch.Tensor,
    clip_fraction: torch.Tensor,
    mean_token_ratio: torch.Tensor,
    sequence_ratio_mean: torch.Tensor,
    approx_kl_estimator: str = "k3",
) -> PolicyLossOutput:
    """统一整理各算法共有的输出字段。"""
    return PolicyLossOutput(
        loss=loss,
        clip_fraction=clip_fraction,
        approx_kl=compute_approx_kl(batch, estimator=approx_kl_estimator),
        mean_advantage=batch.advantages.mean(),
        mean_token_ratio=mean_token_ratio,
        sequence_ratio_mean=sequence_ratio_mean,
        valid_token_count=batch.token_mask.sum(),
        valid_sequence_count=batch.token_mask.sum(dim=1).gt(0).sum(),
        mean_ref_kl=compute_reference_kl(batch),
    )


# ── Algorithm 基类 ───────────────────────────────────────────────


class Algorithm:
    """RL 算法策略基类（Strategy Pattern）。

    trainer 只通过这个接口与算法交互，不感知具体 objective 细节。
    子类重写 prepare_batch（可选）和 compute_loss（必须）。
    """

    name: str = "base"

    def prepare_batch(self, rollout_batch, config):
        """训练前的 batch 预处理（过滤、reward shaping 等）。默认无操作。"""
        return rollout_batch

    def make_policy_loss_batch(
        self,
        rollout_batch,
        current_logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor | None,
    ) -> PolicyLossBatch:
        """标准转换：rollout batch → loss 所需的最小输入结构。"""
        return PolicyLossBatch(
            logprobs=current_logprobs,
            old_logprobs=rollout_batch.old_logprobs,
            advantages=rollout_batch.advantages,
            token_mask=rollout_batch.token_mask,
            ref_logprobs=ref_logprobs,
        )

    def compute_loss(self, batch: PolicyLossBatch, config) -> PolicyLossOutput:
        raise NotImplementedError
