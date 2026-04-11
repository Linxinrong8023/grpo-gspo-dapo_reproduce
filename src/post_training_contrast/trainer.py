"""
trainer.py - RL 训练的完整实现。
测试同步
按这个文件从上到下阅读，就能理解整个训练流程：
  1. RlTrainer 类          — 单步核心逻辑（rollout → reward → advantage → update）
  2. run_training(config)  — 完整训练流程（加载模型 → 循环 → 保存 checkpoint）

依赖关系：
  trainer.py → config / rollout / policy / algorithms / reward / math_utils
"""

from __future__ import annotations

import gc
import json
import logging
import math
import random
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LambdaLR

from src.post_training_contrast.algorithms import (
    compute_group_relative_advantages,
    resolve_algorithm,
)
from src.post_training_contrast.config import TrainConfig
from src.post_training_contrast.evaluator import build_eval_prompt
from src.post_training_contrast.math_utils import parse_final_answer
from src.post_training_contrast.policy import CausalLmPolicy
from src.post_training_contrast.reward import RewardBreakdown
from src.post_training_contrast.reward import compute_batch_reward_breakdowns
from src.post_training_contrast.rollout import (
    RolloutBatch,
    VllmRolloutEngine,
)

logger = logging.getLogger(__name__)


# ── 单步训练结果 ─────────────────────────────────────────────────


@dataclass
class TrainerStepResult:
    """一次训练步的返回值：训练诊断 + rollout 统计。"""

    algorithm_name: str
    accuracy: float
    truncation_rate: float
    parse_fail_rate: float
    mean_response_length: float
    did_update_actor: bool
    num_policy_updates: int
    num_rollout_samples: int
    num_kept_samples: int
    update_losses: list[float]
    mini_batch_losses: list[float]
    lr: float = 0.0
    skipped_step: bool = False
    skip_reason: str | None = None
    update_step: int | None = None
    extra_metrics: dict[str, int | float | bool | str] = field(default_factory=dict)
    # RL 诊断指标
    answer_reward_mean: float = 0.0
    format_reward_mean: float = 0.0
    total_reward_mean: float = 0.0
    approx_kl: float = 0.0       # KL(π_new || π_old)，策略漂移程度
    ref_kl: float = 0.0          # KL(π || π_ref)，与参考策略的偏离
    clip_fraction: float = 0.0   # 触发 clipping 的 token 比例
    policy_entropy: float = 0.0  # 策略熵，接近 0 = mode collapse
    group_all_correct_rate: float = 0.0
    group_all_wrong_rate: float = 0.0
    group_mixed_answer_rate: float = 0.0
    group_signal_rate: float = 0.0
    group_total_reward_std_mean: float = 0.0
    diagnostic_samples: list[dict] = field(default_factory=list)


@dataclass
class RolloutSyncState:
    """维护 rollout policy 的同步方式。"""

    mode: str
    model_name_or_path: str
    snapshot_dir: Path
    adapter_dir: Path | None = None
    adapter_revision: int = 1


@dataclass
class RewardComputationResult:
    """一次 rollout 的 reward 与监控统计。"""

    answer_rewards: torch.Tensor
    format_rewards: torch.Tensor
    total_rewards: torch.Tensor
    accuracy: float
    parse_fail_rate: float
    group_all_correct_rate: float
    group_all_wrong_rate: float
    group_mixed_answer_rate: float
    reward_breakdowns: list[RewardBreakdown]


@dataclass
class PolicyUpdateResult:
    """一次 policy update 阶段的聚合结果。"""

    did_update_actor: bool
    update_losses: list[float]
    mini_batch_losses: list[float]
    approx_kl: float
    ref_kl: float
    clip_fraction: float
    policy_entropy: float


def _preview_text(text: str, max_chars: int) -> str:
    """截断长文本 preview，避免诊断 JSONL 过大。"""
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _collect_diagnostic_samples(
    rollout_batch: RolloutBatch,
    reward_breakdowns: list[RewardBreakdown],
    dataset_name: str,
    config: TrainConfig,
) -> list[dict]:
    """收集少量解析失败/截断样本，用于长训练后的错误分析。"""
    if not config.dump_diagnostic_samples or config.diagnostic_sample_limit_per_step <= 0:
        return []
    if not rollout_batch.records:
        return []

    parse_fail_count = 0
    truncated_count = 0
    limit = config.diagnostic_sample_limit_per_step
    max_chars = config.diagnostic_response_max_chars
    diagnostics: list[dict] = []

    for record, sample, breakdown in zip(
        rollout_batch.records, rollout_batch.samples, reward_breakdowns
    ):
        reasons: list[str] = []
        if breakdown.parse_failed and parse_fail_count < limit:
            reasons.append("parse_failed")
            parse_fail_count += 1
        if sample.is_truncated and truncated_count < limit:
            reasons.append("truncated")
            truncated_count += 1
        if not reasons:
            continue

        parsed_prediction = parse_final_answer(sample.response_text, dataset_name) or ""
        diagnostics.append(
            {
                "record_id": record.get("id") or record.get("unique_id") or "",
                "group_id": sample.group_id,
                "sample_id_within_group": sample.sample_id_within_group,
                "reasons": reasons,
                "finish_reason": sample.finish_reason,
                "stop_reason": sample.stop_reason,
                "response_length": len(sample.response_token_ids),
                "ground_truth": record.get("answer", ""),
                "parsed_prediction": parsed_prediction,
                "is_correct": breakdown.is_correct,
                "parse_failed": breakdown.parse_failed,
                "has_think_close": breakdown.has_think_close,
                "has_answer_tag": breakdown.has_answer_tag,
                "prompt_preview": _preview_text(sample.prompt, max_chars),
                "response_preview": _preview_text(sample.response_text, max_chars),
            }
        )

        if parse_fail_count >= limit and truncated_count >= limit:
            break

    return diagnostics


# ── RlTrainer ────────────────────────────────────────────────────


def _build_minibatch_indices(
    num_samples: int, mini_batch_size: int, shuffle: bool
) -> list[list[int]]:
    """把样本索引切成若干 minibatch，optionally 打乱顺序。"""
    if num_samples == 0:
        return []
    effective = mini_batch_size if mini_batch_size > 0 else num_samples
    indices = (
        torch.randperm(num_samples).tolist() if shuffle else list(range(num_samples))
    )
    return [indices[i : i + effective] for i in range(0, num_samples, effective)]


def _iter_prompt_batches(
    records: list[dict],
    prompt_batch_size: int,
    num_epochs: int,
    shuffle_prompts_each_epoch: bool,
    seed: int,
    max_steps: int | None,
):
    """按 epoch 产出 prompt 级 batch，max_steps 始终是全局上限。"""
    if prompt_batch_size < 1:
        raise ValueError("prompt_batch_size 至少为 1。")
    if num_epochs < 1:
        raise ValueError("num_epochs 至少为 1。")

    global_step = 0
    for epoch_idx in range(1, num_epochs + 1):
        epoch_records = list(records)
        if shuffle_prompts_each_epoch:
            random.Random(seed + epoch_idx - 1).shuffle(epoch_records)

        for start_idx in range(0, len(epoch_records), prompt_batch_size):
            yield epoch_idx, epoch_records[start_idx : start_idx + prompt_batch_size]
            global_step += 1
            if max_steps is not None and global_step >= max_steps:
                return


def _align_seq_len(tensor: torch.Tensor, target_len: int) -> torch.Tensor:
    """对齐张量的 seq_len 维度（dim=1）到 target_len。

    compute_logprobs 返回的宽度按 mini-batch 内最长 response 计算，
    但 old_logprobs / token_mask 是按整个 rollout batch 最长 response padding 的。
    两者宽度可能不一致，这里统一对齐。
    """
    current_len = tensor.shape[1]
    if current_len == target_len:
        return tensor
    if current_len < target_len:
        # 右侧补零
        pad = tensor.new_zeros(tensor.shape[0], target_len - current_len)
        return torch.cat([tensor, pad], dim=1)
    # 截断（理论上不应该发生，但防御性处理）
    return tensor[:, :target_len]


def _compute_answer_group_metrics(
    answer_rewards: torch.Tensor,
    group_size: int,
) -> tuple[float, float, float]:
    """基于答案奖励统计每个 prompt 是否全对、全错或混合。"""
    if answer_rewards.numel() == 0:
        return 0.0, 0.0, 0.0

    grouped = answer_rewards.view(-1, group_size)
    all_correct = grouped.gt(0).all(dim=1)
    all_wrong = grouped.le(0).all(dim=1)
    mixed = ~(all_correct | all_wrong)
    return (
        float(all_correct.float().mean()),
        float(all_wrong.float().mean()),
        float(mixed.float().mean()),
    )


def _compute_group_signal_metrics(
    total_rewards: torch.Tensor,
    group_size: int,
) -> tuple[float, float]:
    """统计实际进入 advantage 的总 reward 是否存在组内方差。"""
    if total_rewards.numel() == 0:
        return 0.0, 0.0

    grouped = total_rewards.view(-1, group_size)
    std = grouped.std(dim=1, unbiased=False)
    return float(std.gt(1e-8).float().mean()), float(std.mean())


def _is_cuda_oom_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return (
        "cuda out of memory" in message
        or "cuda error: out of memory" in message
        or "cublas_status_alloc_failed" in message
    )


def _clear_optimizer_grads(optimizer: torch.optim.Optimizer) -> None:
    try:
        optimizer.zero_grad(set_to_none=True)
    except TypeError:
        optimizer.zero_grad()


def _cleanup_cuda_after_fallback() -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    torch.cuda.empty_cache()
    try:
        torch.cuda.synchronize()
    except Exception:
        pass


def _materialize_prepared_batch_for_update(
    prepared_cpu: RolloutBatch,
    device: torch.device,
) -> RolloutBatch:
    """保留 CPU baseline，按需创建一次 device copy 给 policy update 使用。"""
    prepared_device = RolloutBatch(
        prompts=prepared_cpu.prompts,
        samples=prepared_cpu.samples,
        group_size=prepared_cpu.group_size,
        records=prepared_cpu.records,
        rewards=None if prepared_cpu.rewards is None else prepared_cpu.rewards.to(device),
        answer_rewards=(
            None
            if prepared_cpu.answer_rewards is None
            else prepared_cpu.answer_rewards.to(device)
        ),
        format_rewards=(
            None
            if prepared_cpu.format_rewards is None
            else prepared_cpu.format_rewards.to(device)
        ),
        advantages=(
            None if prepared_cpu.advantages is None else prepared_cpu.advantages.to(device)
        ),
        old_logprobs=(
            None
            if prepared_cpu.old_logprobs is None
            else prepared_cpu.old_logprobs.to(device)
        ),
        token_mask=(
            None if prepared_cpu.token_mask is None else prepared_cpu.token_mask.to(device)
        ),
        response_lengths=(
            None
            if prepared_cpu.response_lengths is None
            else prepared_cpu.response_lengths.to(device)
        ),
        truncation_mask=(
            None
            if prepared_cpu.truncation_mask is None
            else prepared_cpu.truncation_mask.to(device)
        ),
    )
    return prepared_device


class RlTrainer:
    """RL 训练器：单步 rollout → reward → advantage → μ 次 policy update。

    actor 和 reference policy 共用 CausalLmPolicy 接口；
    算法差异通过 Algorithm 子类（GRPO / GSPO / DAPO）封装。
    """

    def __init__(
        self,
        rollout_engine: VllmRolloutEngine,
        actor_policy: CausalLmPolicy,
        optimizer: torch.optim.Optimizer,
        tokenizer=None,
        reference_policy: CausalLmPolicy | None = None,
    ) -> None:
        self.rollout_engine = rollout_engine
        self.actor_policy = actor_policy
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.reference_policy = reference_policy

    def _run_policy_update(
        self,
        prepared: RolloutBatch,
        algorithm,
        config: TrainConfig,
        update_actor: bool,
    ) -> PolicyUpdateResult:
        """执行 Step 5：在已 materialize 的 batch 上跑 μ 次 policy update。"""
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(
                "[GPU memory] allocated=%.2f GB  reserved=%.2f GB", alloc, reserved
            )

        update_losses: list[float] = []
        mini_batch_losses: list[float] = []
        did_update_actor = False
        accum_steps = max(1, config.gradient_accumulation_steps)
        all_approx_kl: list[float] = []
        all_ref_kl: list[float] = []
        all_clip_fraction: list[float] = []
        all_entropy: list[float] = []

        for update_idx in range(config.num_policy_updates):
            epoch_losses: list[float] = []
            batches = _build_minibatch_indices(
                num_samples=len(prepared.samples),
                mini_batch_size=config.mini_batch_size,
                shuffle=config.shuffle_minibatches,
            )

            if update_actor:
                _clear_optimizer_grads(self.optimizer)

            for batch_idx, indices in enumerate(batches):
                mini_batch = prepared.slice(indices)
                logger.debug(
                    "[train_step]   update=%d  batch=%d/%d  samples=%d",
                    update_idx + 1,
                    batch_idx + 1,
                    len(batches),
                    len(indices),
                )

                ref_logprobs = None
                if self.reference_policy is not None:
                    logger.info("[train_step]   Computing reference logprobs...")
                    with torch.no_grad():
                        ref_output = self.reference_policy.compute_logprobs(mini_batch)
                        ref_logprobs = ref_output.logprobs.float()
                    logger.info("[train_step]   Reference logprobs done.")

                logger.info(
                    "[train_step]   Computing actor logprobs (update=%d, batch=%d/%d)...",
                    update_idx + 1,
                    batch_idx + 1,
                    len(batches),
                )
                actor_output = self.actor_policy.compute_logprobs(
                    mini_batch,
                    compute_entropy=config.track_policy_entropy,
                )
                current_logprobs = actor_output.logprobs.float()
                actor_entropy = actor_output.entropy
                logger.info("[train_step]   Actor logprobs done.")

                if actor_entropy is not None:
                    mask = mini_batch.token_mask
                    if mask is not None:
                        aligned_entropy = _align_seq_len(actor_entropy, mask.shape[1])
                        valid = mask.sum().clamp_min(1)
                        mean_ent = float((aligned_entropy * mask).sum() / valid)
                    else:
                        mean_ent = float(actor_entropy.mean())
                    all_entropy.append(mean_ent)

                target_len = mini_batch.old_logprobs.shape[1]
                current_logprobs = _align_seq_len(current_logprobs, target_len)
                if ref_logprobs is not None:
                    ref_logprobs = _align_seq_len(ref_logprobs, target_len)

                policy_loss_batch = algorithm.make_policy_loss_batch(
                    mini_batch, current_logprobs, ref_logprobs
                )
                algo_output = algorithm.compute_loss(policy_loss_batch, config)

                loss_val = float(algo_output.loss.detach())
                epoch_losses.append(loss_val)
                mini_batch_losses.append(loss_val)
                logger.info("[train_step]   loss=%.6f", loss_val)
                all_approx_kl.append(float(algo_output.approx_kl.detach()))
                all_ref_kl.append(float(algo_output.mean_ref_kl.detach()))
                all_clip_fraction.append(float(algo_output.clip_fraction.detach()))

                if update_actor:
                    scaled_loss = algo_output.loss / accum_steps
                    logger.info("[train_step]   Running backward()...")
                    scaled_loss.backward()
                    logger.info("[train_step]   Backward done.")

                    is_last_in_epoch = (batch_idx + 1) == len(batches)
                    is_accum_boundary = (batch_idx + 1) % accum_steps == 0
                    if is_accum_boundary or is_last_in_epoch:
                        if config.max_grad_norm > 0:
                            params = [
                                p
                                for group in self.optimizer.param_groups
                                for p in group["params"]
                                if p.grad is not None
                            ]
                            torch.nn.utils.clip_grad_norm_(params, config.max_grad_norm)
                        self.optimizer.step()
                        _clear_optimizer_grads(self.optimizer)
                        did_update_actor = True
                        logger.info("[train_step]   Optimizer step done.")

            if epoch_losses:
                update_losses.append(sum(epoch_losses) / len(epoch_losses))

        def _safe_mean(values: list[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        return PolicyUpdateResult(
            did_update_actor=did_update_actor,
            update_losses=update_losses,
            mini_batch_losses=mini_batch_losses,
            approx_kl=_safe_mean(all_approx_kl),
            ref_kl=_safe_mean(all_ref_kl),
            clip_fraction=_safe_mean(all_clip_fraction),
            policy_entropy=_safe_mean(all_entropy),
        )

    def train_step(
        self,
        records: list[dict],
        dataset_name: str,
        config: TrainConfig,
        update_actor: bool = True,
    ) -> TrainerStepResult:
        """执行一次完整训练步。

        流程：
          1. 用当前 rollout engine 对 prompts 采样 G 条 response
          2. 计算每条 response 的 binary reward
          3. （DAPO）overlong penalty + dynamic sampling
          4. 组内 advantage 归一化
          5. μ 次 policy update（每次都重新算 current logprobs）
        """
        if config.group_size < 1:
            raise ValueError("group_size 至少为 1。")
        if config.num_policy_updates < 1:
            raise ValueError("num_policy_updates 至少为 1。")

        algorithm = resolve_algorithm(config.algorithm_name)

        # ── Step 1：Rollout ──────────────────────────────────────
        logger.info(
            "[train_step] Step 1/5: Rollout starting (%d prompts × %d)...",
            len(records),
            config.group_size,
        )
        prompts = [
            build_eval_prompt(r, dataset_name, tokenizer=self.tokenizer)
            for r in records
        ]
        rollout_batch = self.rollout_engine.generate(
            prompts,
            num_samples_per_prompt=config.group_size,
            sampling_overrides={"max_tokens": config.max_new_tokens},
        )
        rollout_batch.records = self._expand_records(records, config.group_size)
        logger.info(
            "[train_step] Step 1/5: Rollout done. %d samples generated.",
            len(rollout_batch.samples),
        )
        num_rollout_samples = len(rollout_batch.samples)
        self._lower_rollout_engine_memory_before_update(config)

        # ── Step 2：Reward ───────────────────────────────────────
        logger.info("[train_step] Step 2/5: Computing rewards...")
        reward_result = self._compute_rewards(rollout_batch, dataset_name, config)
        self._attach_rewards(rollout_batch, reward_result)

        logger.info(
            "[train_step] Step 2/5: Rewards done. acc=%.4f  total=%.4f  format=%.4f  parse_fail=%.4f",
            reward_result.accuracy,
            float(reward_result.total_rewards.mean()) if len(reward_result.total_rewards) else 0.0,
            float(reward_result.format_rewards.mean()) if len(reward_result.format_rewards) else 0.0,
            reward_result.parse_fail_rate,
        )

        # 诊断指标（在 prepare_batch 之前基于全量 rollout 计算）
        accuracy = reward_result.accuracy
        parse_fail_rate = reward_result.parse_fail_rate
        truncation_rate = (
            float(rollout_batch.truncation_mask.mean())
            if rollout_batch.truncation_mask is not None
            and len(rollout_batch.truncation_mask)
            else 0.0
        )
        mean_response_length = (
            float(rollout_batch.response_lengths.float().mean())
            if rollout_batch.response_lengths is not None
            and len(rollout_batch.response_lengths)
            else 0.0
        )
        answer_reward_mean = (
            float(reward_result.answer_rewards.mean())
            if len(reward_result.answer_rewards)
            else 0.0
        )
        format_reward_mean = (
            float(reward_result.format_rewards.mean())
            if len(reward_result.format_rewards)
            else 0.0
        )
        total_reward_mean = (
            float(reward_result.total_rewards.mean())
            if len(reward_result.total_rewards)
            else 0.0
        )
        diagnostic_samples = _collect_diagnostic_samples(
            rollout_batch=rollout_batch,
            reward_breakdowns=reward_result.reward_breakdowns,
            dataset_name=dataset_name,
            config=config,
        )

        # ── Step 3：算法专属预处理（DAPO: penalty + filter）────
        logger.info("[train_step] Step 3/5: Algorithm prepare_batch...")
        prepared_cpu = algorithm.prepare_batch(rollout_batch, config)
        if prepared_cpu.rewards is None:
            prepared_cpu.rewards = reward_result.total_rewards

        # ── Step 4：Advantage 归一化 ─────────────────────────────
        logger.info(
            "[train_step] Step 4/5: Computing advantages (%d samples)...",
            len(prepared_cpu.samples),
        )
        if len(prepared_cpu.samples):
            prepared_cpu.advantages = compute_group_relative_advantages(
                prepared_cpu.rewards, group_size=prepared_cpu.group_size
            )
        else:
            prepared_cpu.advantages = torch.zeros(0, dtype=torch.float32)

        group_signal_rate, group_total_reward_std_mean = _compute_group_signal_metrics(
            prepared_cpu.rewards,
            prepared_cpu.group_size,
        )

        # ── Step 5：μ 次 policy update（支持梯度累积）───────────
        logger.info(
            "[train_step] Step 5/5: Policy update starting "
            "(updates=%d, mini_batch=%d, accum=%d)...",
            config.num_policy_updates,
            config.mini_batch_size,
            config.gradient_accumulation_steps,
        )
        device = next(self.actor_policy.model.parameters()).device
        policy_update = self._run_policy_update_with_oom_retry(
            prepared_cpu=prepared_cpu,
            algorithm=algorithm,
            config=config,
            update_actor=update_actor,
            device=device,
        )

        return TrainerStepResult(
            algorithm_name=algorithm.name,
            accuracy=accuracy,
            truncation_rate=truncation_rate,
            parse_fail_rate=parse_fail_rate,
            mean_response_length=mean_response_length,
            did_update_actor=policy_update.did_update_actor,
            num_policy_updates=config.num_policy_updates,
            num_rollout_samples=num_rollout_samples,
            num_kept_samples=len(prepared_cpu.samples),
            update_losses=policy_update.update_losses,
            mini_batch_losses=policy_update.mini_batch_losses,
            answer_reward_mean=answer_reward_mean,
            format_reward_mean=format_reward_mean,
            total_reward_mean=total_reward_mean,
            approx_kl=policy_update.approx_kl,
            ref_kl=policy_update.ref_kl,
            clip_fraction=policy_update.clip_fraction,
            policy_entropy=policy_update.policy_entropy,
            group_all_correct_rate=reward_result.group_all_correct_rate,
            group_all_wrong_rate=reward_result.group_all_wrong_rate,
            group_mixed_answer_rate=reward_result.group_mixed_answer_rate,
            group_signal_rate=group_signal_rate,
            group_total_reward_std_mean=group_total_reward_std_mean,
            diagnostic_samples=diagnostic_samples,
        )

    def _run_policy_update_with_oom_retry(
        self,
        *,
        prepared_cpu: RolloutBatch,
        algorithm,
        config: TrainConfig,
        update_actor: bool,
        device,
    ) -> PolicyUpdateResult:
        prepared_for_update = _materialize_prepared_batch_for_update(prepared_cpu, device)
        try:
            return self._run_policy_update(
                prepared=prepared_for_update,
                algorithm=algorithm,
                config=config,
                update_actor=update_actor,
            )
        except RuntimeError as exc:
            if not _is_cuda_oom_error(exc):
                raise
            logger.warning(
                "[train_step] rollout_engine_transition=fallback_release_retry: CUDA OOM during policy update, releasing resident vLLM and retrying once."
            )
            _clear_optimizer_grads(self.optimizer)
            self.rollout_engine.release(reason="fallback_release_retry")
            logger.info(
                "[train_step] next_rollout_transition=fresh_rebuild_after_fallback"
            )
            _cleanup_cuda_after_fallback()
            prepared_for_update = _materialize_prepared_batch_for_update(
                prepared_cpu, device
            )
            return self._run_policy_update(
                prepared=prepared_for_update,
                algorithm=algorithm,
                config=config,
                update_actor=update_actor,
            )

    def _lower_rollout_engine_memory_before_update(self, config: TrainConfig) -> None:
        keep_vllm_resident_on_lora = (
            config.use_lora and config.keep_vllm_resident_on_lora
        )
        if keep_vllm_resident_on_lora:
            logger.info(
                "[train_step] rollout_engine_transition=resident: keeping vLLM engine resident for LoRA update; single_gpu_safe_mode will not release vLLM on this path."
            )
        elif config.single_gpu_safe_mode:
            logger.info(
                "[train_step] Single-GPU safe mode: lowering rollout engine memory before update..."
            )
            self.rollout_engine.reduce_memory_usage(prefer_sleep=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            gc.collect()
            logger.info("[train_step] Rollout engine memory reduced.")
        else:
            logger.info("[train_step] Releasing vLLM engine to free GPU memory...")
            self.rollout_engine.release(reason="standard_release")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            gc.collect()
            logger.info("[train_step] vLLM released.")

    def _expand_records(self, records: list[dict], group_size: int) -> list[dict]:
        """把每个 prompt 的 record 展开到 sample 级别（与 rollout sample 一一对应）。"""
        expanded = []
        for record in records:
            expanded.extend([record] * group_size)
        return expanded

    def _compute_rewards(
        self,
        rollout_batch: RolloutBatch,
        dataset_name: str,
        config: TrainConfig,
    ) -> RewardComputationResult:
        """计算 sample 级答案奖励、格式奖励和总奖励。"""
        reward_breakdowns = compute_batch_reward_breakdowns(
            responses=[sample.response_text for sample in rollout_batch.samples],
            ground_truth_answers=[record["answer"] for record in rollout_batch.records],
            dataset_name=dataset_name,
            answer_correct_reward=config.answer_reward_correct,
            answer_incorrect_reward=config.answer_reward_incorrect,
            missing_think_close_penalty=config.format_penalty_missing_think_close,
            missing_answer_tag_penalty=config.format_penalty_missing_answer_tag,
        )
        answer_rewards = [breakdown.answer_reward for breakdown in reward_breakdowns]
        format_rewards = [breakdown.format_reward for breakdown in reward_breakdowns]
        total_rewards = [breakdown.total_reward for breakdown in reward_breakdowns]
        parse_fail = sum(int(breakdown.parse_failed) for breakdown in reward_breakdowns)
        correct = sum(int(breakdown.is_correct) for breakdown in reward_breakdowns)

        answer_rewards_tensor = torch.tensor(answer_rewards, dtype=torch.float32)
        format_rewards_tensor = torch.tensor(format_rewards, dtype=torch.float32)
        total_rewards_tensor = torch.tensor(total_rewards, dtype=torch.float32)
        parse_fail_rate = (
            parse_fail / len(rollout_batch.samples) if rollout_batch.samples else 0.0
        )
        accuracy = correct / len(rollout_batch.samples) if rollout_batch.samples else 0.0
        (
            group_all_correct_rate,
            group_all_wrong_rate,
            group_mixed_answer_rate,
        ) = _compute_answer_group_metrics(
            answer_rewards_tensor,
            rollout_batch.group_size,
        )
        return RewardComputationResult(
            answer_rewards=answer_rewards_tensor,
            format_rewards=format_rewards_tensor,
            total_rewards=total_rewards_tensor,
            accuracy=accuracy,
            parse_fail_rate=parse_fail_rate,
            group_all_correct_rate=group_all_correct_rate,
            group_all_wrong_rate=group_all_wrong_rate,
            group_mixed_answer_rate=group_mixed_answer_rate,
            reward_breakdowns=reward_breakdowns,
        )

    def _attach_rewards(
        self,
        rollout_batch: RolloutBatch,
        reward_result: RewardComputationResult,
    ) -> None:
        rollout_batch.answer_rewards = reward_result.answer_rewards
        rollout_batch.format_rewards = reward_result.format_rewards
        rollout_batch.rewards = reward_result.total_rewards


# ── 完整训练流程 ─────────────────────────────────────────────────


def _init_wandb(config: TrainConfig) -> bool:
    """初始化 WandB run。失败时降级为无监控模式（不中断训练）。"""
    if not config.wandb_enabled:
        return False
    try:
        import wandb
        from dataclasses import asdict

        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            tags=config.wandb_tags or [],
            config=asdict(config),
        )
        logger.info(
            "WandB run initialized: %s/%s", config.wandb_project, config.wandb_run_name
        )
        return True
    except Exception as e:
        logger.warning("WandB 初始化失败（将跳过监控）: %s", e)
        return False


def _wandb_log(metrics: dict, step: int, use_wandb: bool) -> None:
    """记录一步的指标。WandB 未初始化时静默忽略。"""
    if not use_wandb:
        return
    try:
        import wandb

        wandb.log(metrics, step=step)
    except Exception:
        pass


def _smooth_wandb_train_metrics(
    history: list[dict[str, float]],
    current_metrics: dict[str, float],
    window_size: int = 20,
) -> dict[str, float]:
    """对 W&B train 指标做 rolling mean；只影响展示，不改变 summary raw 数据。"""
    history.append(current_metrics)
    del history[:-window_size]

    smoothed: dict[str, float] = {}
    for key in current_metrics:
        values = [
            metrics[key]
            for metrics in history
            if key in metrics and math.isfinite(metrics[key])
        ]
        smoothed[key] = sum(values) / len(values) if values else float("nan")
    return smoothed


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_total_steps: int,
    schedule_name: str = "cosine",
) -> LambdaLR:
    """构建 warmup + 调度器。

    支持：
      - cosine
      - constant_with_warmup
    """

    normalized_name = (schedule_name or "cosine").lower()

    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return float(step) / max(1, num_warmup_steps)
        if normalized_name == "constant_with_warmup":
            return 1.0
        if normalized_name == "cosine":
            progress = float(step - num_warmup_steps) / max(
                1, num_total_steps - num_warmup_steps
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        raise ValueError(f"不支持的 lr_schedule: {schedule_name}")

    return LambdaLR(optimizer, lr_lambda)


def _resolve_attn_implementation(requested: str | None) -> str | None:
    """Resolve the HF attention backend without making flash-attn a hard dependency."""
    if requested is None:
        return None
    normalized = requested.lower()
    if normalized in {"none", "null", "default"}:
        return None
    if normalized != "auto":
        return requested
    if find_spec("flash_attn") is not None:
        return "flash_attention_2"
    return None


def _load_actor(config: TrainConfig):
    """Actor 模型加载。如果配置了 LoRA ，会在全量模型基础上加载轻量适配层。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cuda_ok = torch.cuda.is_available()
    mps_ok = (
        getattr(getattr(torch, "backends", None), "mps", None)
        and torch.backends.mps.is_available()
    )
    bf16_ok = cuda_ok and torch.cuda.is_bf16_supported()

    if config.device == "auto":
        device = "cuda" if cuda_ok else "mps" if mps_ok else "cpu"
    else:
        device = config.device

    if config.dtype == "auto":
        dtype_name = (
            "bfloat16"
            if (device == "cuda" and bf16_ok)
            else "float16"
            if device == "cuda"
            else "float32"
        )
    else:
        dtype_name = config.dtype

    attn_implementation = _resolve_attn_implementation(config.attn_implementation)
    logger.info(
        "Loading actor model %s  device=%s  dtype=%s  requested_attn=%s  resolved_attn=%s",
        config.model,
        device,
        dtype_name,
        config.attn_implementation or "default",
        attn_implementation or "default",
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": getattr(torch, dtype_name),
        "low_cpu_mem_usage": True,
    }
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation
    actor_model = AutoModelForCausalLM.from_pretrained(config.model, **model_kwargs)
    actor_model.to(device)
    actor_model.train()
    logger.info(
        "Actor attention implementation: %s",
        getattr(getattr(actor_model, "config", None), "_attn_implementation", "unknown"),
    )

    # 核心救命代码：开启梯度检查点，用时间换空间
    # 将前向传播 Activation 的显存消耗由 O(N_layers) 降至 O(1)，直接省出几十 GB！
    actor_model.gradient_checkpointing_enable()

    if config.use_lora:
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError as e:
            raise RuntimeError(
                "use_lora=True 需要安装 peft 库：pip install peft"
            ) from e

        # Qwen / LLaMA / Mistral 通用默认 target modules
        target_modules = config.lora_target_modules or [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        actor_model = get_peft_model(actor_model, lora_cfg)
        trainable, total = actor_model.get_nb_trainable_parameters()
        logger.info(
            "LoRA applied: r=%d  alpha=%d  trainable_params=%.3f%% (%s / %s)",
            config.lora_r,
            config.lora_alpha,
            100 * trainable / total,
            f"{trainable:,}",
            f"{total:,}",
        )

    return actor_model, tokenizer, device, dtype_name


def _load_reference(config: TrainConfig, device: str, dtype_name: str):
    """加载冻结的参考模型（KL 惩罚中使用的 π_ref）。"""
    from transformers import AutoModelForCausalLM

    attn_implementation = _resolve_attn_implementation(config.attn_implementation)
    logger.info(
        "Loading reference model (frozen) %s  requested_attn=%s  resolved_attn=%s",
        config.model,
        config.attn_implementation or "default",
        attn_implementation or "default",
    )
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": getattr(torch, dtype_name),
        "low_cpu_mem_usage": True,
    }
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation
    ref_model = AutoModelForCausalLM.from_pretrained(config.model, **model_kwargs)
    ref_model.to(device)
    ref_model.eval()
    logger.info(
        "Reference attention implementation: %s",
        getattr(getattr(ref_model, "config", None), "_attn_implementation", "unknown"),
    )
    for param in ref_model.parameters():
        param.requires_grad_(False)
    return ref_model


def _build_rollout_engine(
    config: TrainConfig,
    rollout_state: RolloutSyncState,
) -> VllmRolloutEngine:
    rollout_dtype = (
        config.rollout_dtype if config.rollout_dtype != "auto" else config.dtype
    )
    enable_lora = rollout_state.mode == "lora_adapter"
    return VllmRolloutEngine(
        model_name_or_path=rollout_state.model_name_or_path,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        tensor_parallel_size=config.tensor_parallel_size,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_model_len=config.max_model_len,
        trust_remote_code=True,
        dtype=rollout_dtype,
        enable_lora=enable_lora,
        lora_adapter_path=(
            str(rollout_state.adapter_dir)
            if enable_lora and rollout_state.adapter_dir is not None
            else None
        ),
        lora_adapter_name=f"rollout_adapter_step_{rollout_state.adapter_revision}",
        lora_adapter_id=rollout_state.adapter_revision,
    )


def _init_rollout_sync_state(
    config: TrainConfig,
    actor_model,
    tokenizer,
    output_dir: Path,
) -> RolloutSyncState:
    state = RolloutSyncState(
        mode="lora_adapter" if config.use_lora else "full_model",
        model_name_or_path=config.model,
        snapshot_dir=output_dir / "rollout_actor",
        adapter_dir=output_dir / "rollout_adapter" if config.use_lora else None,
        adapter_revision=1,
    )
    if state.mode == "lora_adapter" and state.adapter_dir is not None:
        _save_lora_adapter_for_rollout(actor_model, tokenizer, state.adapter_dir)
    return state


def _save_actor_for_rollout(
    actor_model,
    tokenizer,
    snapshot_dir: Path,
    use_lora: bool,
) -> None:
    """vLLM 只能加载合并后的完整权重。LoRA 开启时必须先 merge 再保存。

    merge_and_unload() 不修改原来的 PeftModel (actor_model)，
    返回一个新的标准 HF 模型对象，训练可以继续进行。
    """
    if use_lora:
        import gc

        logger.info("Merging LoRA adapter for rollout snapshot...")
        merged = actor_model.merge_and_unload()
        merged.save_pretrained(snapshot_dir)
        del merged
        gc.collect()
        torch.cuda.empty_cache()
    else:
        actor_model.save_pretrained(snapshot_dir)
    tokenizer.save_pretrained(snapshot_dir)


def _save_lora_adapter_for_rollout(
    actor_model,
    tokenizer,
    adapter_dir: Path,
) -> None:
    """只保存 LoRA adapter，供 vLLM base+adapter rollout 模式使用。"""
    adapter_dir.mkdir(parents=True, exist_ok=True)
    # 我们没有改 tokenizer / 词表，显式关闭 PEFT 的 embedding auto-detect，
    # 避免 save_pretrained() 在离线机器上去探测 base model config。
    actor_model.save_pretrained(adapter_dir, save_embedding_layers=False)
    tokenizer.save_pretrained(adapter_dir)


def _refresh_rollout_engine(
    trainer: RlTrainer,
    actor_model,
    tokenizer,
    config: TrainConfig,
    rollout_state: RolloutSyncState,
) -> None:
    """刷新下一步 rollout 使用的 old policy。"""
    if rollout_state.mode == "lora_adapter":
        if rollout_state.adapter_dir is None:
            raise ValueError("lora_adapter 模式缺少 adapter_dir。")
        _save_lora_adapter_for_rollout(actor_model, tokenizer, rollout_state.adapter_dir)
        rollout_state.adapter_revision += 1
        trainer.rollout_engine.refresh_lora_adapter(
            adapter_path=str(rollout_state.adapter_dir),
            adapter_id=rollout_state.adapter_revision,
            adapter_name=f"rollout_adapter_step_{rollout_state.adapter_revision}",
        )
        logger.info(
            "Rollout policy refreshed from LoRA adapter %s (revision=%d).",
            rollout_state.adapter_dir,
            rollout_state.adapter_revision,
        )
        return

    trainer.rollout_engine.release()
    logger.info("Old vLLM instance released.")
    _save_actor_for_rollout(
        actor_model,
        tokenizer,
        rollout_state.snapshot_dir,
        config.use_lora,
    )
    rollout_state.model_name_or_path = str(rollout_state.snapshot_dir)
    trainer.rollout_engine = _build_rollout_engine(config, rollout_state)
    logger.info("Rollout engine refreshed from %s", rollout_state.snapshot_dir)


def _switch_rollout_engine_to_snapshot_fallback(
    trainer: RlTrainer,
    actor_model,
    tokenizer,
    config: TrainConfig,
    rollout_state: RolloutSyncState,
) -> None:
    """LoRA adapter rollout 不可用时，退回 merged snapshot 重建路径。"""
    logger.warning(
        "Switching rollout engine to merged snapshot fallback mode."
    )
    trainer.rollout_engine.release()
    _save_actor_for_rollout(
        actor_model,
        tokenizer,
        rollout_state.snapshot_dir,
        use_lora=config.use_lora,
    )
    rollout_state.mode = "full_model"
    rollout_state.model_name_or_path = str(rollout_state.snapshot_dir)
    trainer.rollout_engine = _build_rollout_engine(config, rollout_state)


def _validate_with_hf(
    actor_model,
    tokenizer,
    eval_records: list[dict],
    dataset_name: str,
    max_eval_samples: int | None = None,
    max_new_tokens: int = 2048,
) -> float:
    """训练中期在验证集上做一次快速评测。

    用 HF model.generate() greedy 解码，直接反映 LoRA 最新权重。
    返回 accuracy (0~1)。
    """
    samples = (
        eval_records
        if max_eval_samples is None
        else eval_records[:max_eval_samples]
    )
    if not samples:
        return 0.0

    was_training = actor_model.training
    actor_model.eval()
    device = next(actor_model.parameters()).device
    responses: list[str] = []
    answers: list[str] = []

    for record in samples:
        prompt = build_eval_prompt(record, dataset_name, tokenizer=tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = actor_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        # 只解码新生成的 token
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        responses.append(response)
        answers.append(record["answer"])

    if was_training:
        actor_model.train()

    breakdowns = compute_batch_reward_breakdowns(
        responses=responses,
        ground_truth_answers=answers,
        dataset_name=dataset_name,
    )
    correct = sum(int(breakdown.is_correct) for breakdown in breakdowns)
    return correct / len(samples)


def _validate_with_vllm(
    rollout_engine: VllmRolloutEngine,
    tokenizer,
    eval_records: list[dict],
    dataset_name: str,
    batch_size: int,
    max_eval_samples: int | None = None,
    max_new_tokens: int = 2048,
) -> float:
    """训练中验证：复用 rollout engine 做 greedy 批量验证。"""
    samples = (
        eval_records
        if max_eval_samples is None
        else eval_records[:max_eval_samples]
    )
    if not samples:
        return 0.0

    correct = 0
    total = 0
    batch_size = max(1, batch_size)
    for start_idx in range(0, len(samples), batch_size):
        batch_records = samples[start_idx : start_idx + batch_size]
        prompts = [
            build_eval_prompt(record, dataset_name, tokenizer=tokenizer)
            for record in batch_records
        ]
        rollout_batch = rollout_engine.generate(
            prompts=prompts,
            num_samples_per_prompt=1,
            sampling_overrides={
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": max_new_tokens,
            },
            return_logprobs=False,
        )
        breakdowns = compute_batch_reward_breakdowns(
            responses=[sample.response_text for sample in rollout_batch.samples],
            ground_truth_answers=[record["answer"] for record in batch_records],
            dataset_name=dataset_name,
        )
        correct += sum(int(breakdown.is_correct) for breakdown in breakdowns)
        total += len(breakdowns)

    return correct / max(1, total)


def _run_validation(
    actor_model,
    tokenizer,
    rollout_engine: VllmRolloutEngine,
    eval_records: list[dict],
    dataset_name: str,
    config: TrainConfig,
) -> float:
    backend = (config.val_backend or "hf").lower()
    if backend == "vllm":
        return _validate_with_vllm(
            rollout_engine=rollout_engine,
            tokenizer=tokenizer,
            eval_records=eval_records,
            dataset_name=dataset_name,
            batch_size=config.val_batch_size,
            max_eval_samples=config.max_eval_samples,
            max_new_tokens=config.eval_max_new_tokens,
        )
    if backend == "hf":
        return _validate_with_hf(
            actor_model=actor_model,
            tokenizer=tokenizer,
            eval_records=eval_records,
            dataset_name=dataset_name,
            max_eval_samples=config.max_eval_samples,
            max_new_tokens=config.eval_max_new_tokens,
        )
    raise ValueError(f"不支持的 val_backend: {config.val_backend}")


def run_training(config: TrainConfig) -> dict:
    """RL 训练主流程。

    流程（从上到下线性读）：
      1. 加载 actor 模型 + tokenizer
      2. 加载冻结 reference 模型
      3. 创建 optimizer + LR scheduler
      4. 创建 rollout engine（vLLM）
      5. 加载训练数据 → 切 prompt batches
      6. 逐步训练，记录日志，保存 checkpoint
      7. 训练结束后保存最终模型 + train_summary.json
    """
    if config.algorithm_name.lower() == "dapo" and config.dapo_use_dynamic_sampling:
        from src.post_training_contrast.dapo_trainer import run_dapo_dynamic_training

        return run_dapo_dynamic_training(config)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_samples_path: Path | None = None
    if config.dump_diagnostic_samples:
        diagnostic_dir = output_dir / "diagnostics"
        diagnostic_dir.mkdir(parents=True, exist_ok=True)
        diagnostic_samples_path = diagnostic_dir / "train_diagnostic_samples.jsonl"
        diagnostic_samples_path.write_text("", encoding="utf-8")
        logger.info("Training diagnostic samples → %s", diagnostic_samples_path)

    if not config.reload_rollout_each_step:
        logger.warning(
            "reload_rollout_each_step=False 会让 old policy 逐步过期；已强制改为 True。"
        )
        config.reload_rollout_each_step = True
    if config.vllm_sleep_during_update:
        logger.info(
            "vllm_sleep_during_update=True is ignored on vLLM 0.5.4; "
            "the training path no longer uses sleep/wake during update."
        )

    # ── 1. 加载模型 ──────────────────────────────────────────────
    actor_model, tokenizer, device, dtype_name = _load_actor(config)
    actor_policy = CausalLmPolicy(
        model=actor_model, pad_token_id=tokenizer.pad_token_id
    )
    ref_model = None
    if config.use_lora:
        ref_policy = CausalLmPolicy(
            model=actor_model,
            pad_token_id=tokenizer.pad_token_id,
            disable_adapter=True,
            enforce_eval_mode=True,
        )
    else:
        ref_model = _load_reference(config, device, dtype_name)
        ref_policy = CausalLmPolicy(
            model=ref_model,
            pad_token_id=tokenizer.pad_token_id,
        )

    # ── 2. Optimizer + Scheduler ─────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in actor_model.parameters() if p.requires_grad],
        lr=config.learning_rate,
    )

    # ── 3. 加载数据 ──────────────────────────────────────────────
    records: list[dict] = []
    with open(config.dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if config.max_samples is not None:
        records = records[: config.max_samples]

    batches = list(
        _iter_prompt_batches(
            records=records,
            prompt_batch_size=config.prompt_batch_size,
            num_epochs=config.num_epochs,
            shuffle_prompts_each_epoch=config.shuffle_prompts_each_epoch,
            seed=config.seed,
            max_steps=config.max_steps,
        )
    )
    num_total_steps = len(batches)
    first_epoch_steps = (
        math.ceil(len(records) / config.prompt_batch_size) if records else 0
    )
    logger.info(
        "Training: algo=%s  planned_steps=%d  first_epoch_steps=%d  prompts=%d  prompt_batch_size=%d  group_size=%d  epochs=%d  shuffle_prompts_each_epoch=%s",
        config.algorithm_name,
        num_total_steps,
        first_epoch_steps,
        len(records),
        config.prompt_batch_size,
        config.group_size,
        config.num_epochs,
        config.shuffle_prompts_each_epoch,
    )
    logger.info(
        "Training diagnostics note: approx_kl and clip_fraction are internal optimization diagnostics; prioritize val_accuracy and final Math500 for model selection."
    )

    best_metric_name = (config.best_checkpoint_metric or "val_accuracy").lower()
    if best_metric_name != "val_accuracy":
        raise ValueError(
            f"当前只支持 best_checkpoint_metric=val_accuracy，收到: {config.best_checkpoint_metric}"
        )

    num_warmup = config.lr_warmup_steps or (
        max(1, num_total_steps // 10) if num_total_steps else 0
    )
    scheduler = _build_lr_scheduler(
        optimizer,
        num_warmup,
        max(1, num_total_steps),
        schedule_name=config.lr_schedule,
    )

    # ── 4. WandB ─────────────────────────────────────────────────
    use_wandb = _init_wandb(config)

    # ── 5. Trainer ───────────────────────────────────────────────
    rollout_state = _init_rollout_sync_state(config, actor_model, tokenizer, output_dir)
    rollout_engine = _build_rollout_engine(config, rollout_state)
    if torch.cuda.is_available():
        total_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
        budget_gib = total_gib * config.gpu_memory_utilization
        logger.info(
            "vLLM rollout mode=%s  gpu_memory_utilization=%.2f  total-budget-cap≈%.2f GiB",
            "adapter hot update" if rollout_state.mode == "lora_adapter" else "full rebuild",
            config.gpu_memory_utilization,
            budget_gib,
        )
    else:
        logger.info(
            "vLLM rollout mode=%s  gpu_memory_utilization=%.2f",
            "adapter hot update" if rollout_state.mode == "lora_adapter" else "full rebuild",
            config.gpu_memory_utilization,
        )
    trainer = RlTrainer(
        rollout_engine=rollout_engine,
        actor_policy=actor_policy,
        optimizer=optimizer,
        tokenizer=tokenizer,
        reference_policy=ref_policy,
    )

    # ── 加载验证集（如果配置了）──────────────────────────────────
    eval_records: list[dict] = []
    if config.eval_dataset_path:
        with open(config.eval_dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    eval_records.append(json.loads(line))
        logger.info(
            "Validation dataset loaded: %d samples from %s",
            len(eval_records),
            config.eval_dataset_path,
        )
    early_stopping_enabled = (
        bool(eval_records)
        and config.eval_interval > 0
        and config.early_stopping_patience is not None
    )
    if config.early_stopping_patience is not None and not early_stopping_enabled:
        logger.warning(
            "early_stopping_patience=%s but validation is disabled; early stopping will be ignored.",
            config.early_stopping_patience,
        )

    from src.post_training_contrast.training_runtime import (
        TrainingStepRequest,
        run_training_event_loop,
    )

    def _standard_step_requests():
        for step_idx, (epoch_idx, batch_records) in enumerate(batches, start=1):
            yield TrainingStepRequest(
                step_idx=step_idx,
                epoch_idx=epoch_idx,
                run=lambda batch_records=batch_records: trainer.train_step(
                    records=batch_records,
                    dataset_name=config.dataset_name,
                    config=config,
                    update_actor=True,
                ),
            )

    return run_training_event_loop(
        config=config,
        output_dir=output_dir,
        actor_model=actor_model,
        tokenizer=tokenizer,
        trainer=trainer,
        scheduler=scheduler,
        rollout_state=rollout_state,
        eval_records=eval_records,
        diagnostic_samples_path=diagnostic_samples_path,
        num_total_steps=num_total_steps,
        first_epoch_steps=first_epoch_steps,
        best_metric_name=best_metric_name,
        use_wandb=use_wandb,
        step_requests=_standard_step_requests(),
        get_stop_reason=lambda: "completed",
    )
