"""DAPO strict dynamic sampling trainer.

This module keeps DAPO candidate-pool sampling out of the shared GRPO/GSPO
trainer path. It only owns prompt selection, mixed-group filtering, recycle
pool handling, and skipped-step decisions; reward, advantage, and policy update
logic are still inherited from :mod:`trainer`.
"""

from __future__ import annotations

import json
import logging
import math
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from src.post_training_contrast.algorithms import (
    compute_group_relative_advantages,
    resolve_algorithm,
)
from src.post_training_contrast.config import TrainConfig
from src.post_training_contrast.evaluator import build_eval_prompt
from src.post_training_contrast.policy import CausalLmPolicy
from src.post_training_contrast.rollout import RolloutBatch
from src.post_training_contrast.trainer import (
    RlTrainer,
    TrainerStepResult,
    _build_lr_scheduler,
    _build_rollout_engine,
    _collect_diagnostic_samples,
    _compute_group_signal_metrics,
    _init_rollout_sync_state,
    _init_wandb,
    _load_actor,
    _load_reference,
)
from src.post_training_contrast.training_runtime import (
    TrainingStepRequest,
    run_training_event_loop,
)

logger = logging.getLogger(__name__)


def _record_key(record: dict) -> str:
    """Stable prompt identity for avoiding same-step retries."""
    for key in ("id", "uid", "problem_id"):
        if key in record:
            return f"{key}:{record[key]}"
    if "problem" in record:
        return f"problem:{record['problem']}"
    return json.dumps(record, sort_keys=True, ensure_ascii=False)


@dataclass
class CandidateBatch:
    records: list[dict]
    recycled_count: int
    epoch: int


class DapoCandidatePool:
    """Prompt source for DAPO strict dynamic sampling.

    Rejected prompts are never returned again inside the same train step. After
    the step finishes, callers can recycle them into this pool so future train
    steps can retry them after the actor has had a chance to update.
    """

    def __init__(self, records: list[dict], config: TrainConfig) -> None:
        self.seed = config.seed
        self.shuffle = config.shuffle_prompts_each_epoch
        self.max_draws = len(records) * max(1, config.num_epochs)
        self.drawn = 0
        self.recycle_round = 0
        self._fresh: deque[tuple[int, dict]] = deque()
        self._recycle: deque[tuple[int, dict]] = deque()

        for epoch_idx in range(1, max(1, config.num_epochs) + 1):
            epoch_records = list(records)
            if self.shuffle:
                random.Random(config.seed + epoch_idx - 1).shuffle(epoch_records)
            for record in epoch_records:
                self._fresh.append((epoch_idx, record))

    def has_candidates(self) -> bool:
        return self.drawn < self.max_draws and bool(self._fresh or self._recycle)

    def take(self, batch_size: int, exclude_keys: set[str]) -> CandidateBatch:
        records: list[dict] = []
        recycled_count = 0
        epochs: list[int] = []
        batch_size = max(1, batch_size)

        while len(records) < batch_size and self.has_candidates():
            prefer_recycle = bool(self._recycle) and (
                not self._fresh or len(records) % 2 == 0
            )
            item, source = self._pop_candidate(prefer_recycle, exclude_keys)
            if item is None:
                break
            epoch_idx, record = item
            records.append(record)
            epochs.append(epoch_idx)
            recycled_count += int(source == "recycle")
            self.drawn += 1

        return CandidateBatch(
            records=records,
            recycled_count=recycled_count,
            epoch=epochs[0] if epochs else 0,
        )

    def recycle(self, records: Iterable[dict]) -> None:
        recycled = list(records)
        if not recycled:
            return
        self.recycle_round += 1
        random.Random(self.seed + 100_000 + self.recycle_round).shuffle(recycled)
        for record in recycled:
            self._recycle.append((0, record))

    def _pop_candidate(
        self,
        prefer_recycle: bool,
        exclude_keys: set[str],
    ) -> tuple[tuple[int, dict] | None, str | None]:
        sources = ("recycle", "fresh") if prefer_recycle else ("fresh", "recycle")
        for source in sources:
            queue = self._recycle if source == "recycle" else self._fresh
            item = self._pop_from_queue(queue, exclude_keys)
            if item is not None:
                return item, source
        return None, None

    @staticmethod
    def _pop_from_queue(
        queue: deque[tuple[int, dict]],
        exclude_keys: set[str],
    ) -> tuple[int, dict] | None:
        for _ in range(len(queue)):
            item = queue.popleft()
            if _record_key(item[1]) in exclude_keys:
                queue.append(item)
                continue
            return item
        return None


class DapoDynamicTrainer(RlTrainer):
    """Trainer entrypoint for DAPO strict dynamic sampling."""

    def train_step_from_pool(
        self,
        *,
        candidate_pool: DapoCandidatePool,
        dataset_name: str,
        config: TrainConfig,
        update_actor: bool = True,
    ) -> TrainerStepResult:
        if config.algorithm_name.lower() != "dapo":
            raise ValueError("DapoDynamicTrainer 只支持 algorithm_name='dapo'。")
        if not config.dapo_use_dynamic_sampling:
            raise ValueError("DapoDynamicTrainer 需要 dapo_use_dynamic_sampling=True。")

        group_size = config.group_size
        target_prompt_count = config.prompt_batch_size
        target_sample_count = target_prompt_count * group_size
        min_mixed_prompt_count = config.dapo_min_mixed_prompt_batch_size
        gen_prompt_batch_size = (
            config.dapo_gen_prompt_batch_size or config.prompt_batch_size
        )
        max_gen_batches = max(1, config.dapo_max_num_gen_batches)

        qualified_batches: list[RolloutBatch] = []
        rejected_records: list[dict] = []
        seen_this_step: set[str] = set()
        num_gen_batches = 0
        candidate_prompt_count = 0
        recycled_prompt_count = 0
        mixed_prompt_count = 0
        all_correct_prompt_count = 0
        all_wrong_prompt_count = 0
        correct_count = 0.0
        parse_fail_count = 0.0
        answer_reward_sum = 0.0
        format_reward_sum = 0.0
        total_reward_sum = 0.0
        truncation_sum = 0.0
        response_length_sum = 0.0
        candidate_sample_count = 0

        while (
            mixed_prompt_count < target_prompt_count
            and num_gen_batches < max_gen_batches
            and candidate_pool.has_candidates()
        ):
            candidate_batch = candidate_pool.take(
                gen_prompt_batch_size,
                exclude_keys=seen_this_step,
            )
            if not candidate_batch.records:
                break
            for record in candidate_batch.records:
                seen_this_step.add(_record_key(record))
            num_gen_batches += 1
            candidate_prompt_count += len(candidate_batch.records)
            recycled_prompt_count += candidate_batch.recycled_count

            rollout_batch = self._rollout_candidate_records(
                candidate_batch.records,
                dataset_name,
                config,
            )
            reward_result = self._compute_rewards(rollout_batch, dataset_name, config)
            self._attach_rewards(rollout_batch, reward_result)
            (
                mixed_batch,
                rejected_batch_records,
                all_correct_count,
                all_wrong_count,
            ) = self._split_mixed_groups(rollout_batch)

            if len(mixed_batch.samples):
                qualified_batches.append(mixed_batch)
                mixed_prompt_count += len(mixed_batch.samples) // group_size
            rejected_records.extend(rejected_batch_records)
            all_correct_prompt_count += all_correct_count
            all_wrong_prompt_count += all_wrong_count

            sample_count = len(rollout_batch.samples)
            candidate_sample_count += sample_count
            correct_count += reward_result.accuracy * sample_count
            answer_reward_sum += float(reward_result.answer_rewards.sum())
            format_reward_sum += float(reward_result.format_rewards.sum())
            total_reward_sum += float(reward_result.total_rewards.sum())
            parse_fail_count += reward_result.parse_fail_rate * sample_count
            if rollout_batch.truncation_mask is not None:
                truncation_sum += float(rollout_batch.truncation_mask.sum())
            if rollout_batch.response_lengths is not None:
                response_length_sum += float(rollout_batch.response_lengths.sum())

        candidate_pool.recycle(rejected_records)

        qualified = self._concat_rollout_batches(qualified_batches, group_size)
        sampled_mixed_prompt_count = mixed_prompt_count
        overflow_mixed_prompt_count = 0
        overflow_records: list[dict] = []
        if len(qualified.samples) > target_sample_count:
            overflow_start = target_sample_count
            overflow_records = self._group_records_from_batch(
                qualified,
                start_sample_idx=overflow_start,
            )
            overflow_mixed_prompt_count = len(overflow_records)
            qualified = qualified.slice(list(range(target_sample_count)))
            mixed_prompt_count = target_prompt_count
            if overflow_records:
                candidate_pool.recycle(overflow_records)

        pool_exhausted = (
            mixed_prompt_count < target_prompt_count
            and not candidate_pool.has_candidates()
        )
        should_update = mixed_prompt_count >= target_prompt_count or (
            (num_gen_batches >= max_gen_batches or pool_exhausted)
            and mixed_prompt_count >= min_mixed_prompt_count
        )
        if not should_update:
            logger.warning(
                "[DAPO dynamic sampling] skipped update: mixed_prompts=%d target=%d min=%d gen_batches=%d candidates=%d",
                mixed_prompt_count,
                target_prompt_count,
                min_mixed_prompt_count,
                num_gen_batches,
                candidate_prompt_count,
            )
            self._lower_rollout_engine_memory_before_update(config)
            return self._build_skipped_result(
                config=config,
                candidate_sample_count=candidate_sample_count,
                kept_sample_count=len(qualified.samples),
                candidate_prompt_count=candidate_prompt_count,
                mixed_prompt_count=mixed_prompt_count,
                rejected_prompt_count=len(rejected_records),
                recycled_prompt_count=recycled_prompt_count,
                num_gen_batches=num_gen_batches,
                answer_reward_sum=answer_reward_sum,
                format_reward_sum=format_reward_sum,
                total_reward_sum=total_reward_sum,
                parse_fail_count=parse_fail_count,
                truncation_sum=truncation_sum,
                response_length_sum=response_length_sum,
                correct_count=correct_count,
                all_correct_prompt_count=all_correct_prompt_count,
                all_wrong_prompt_count=all_wrong_prompt_count,
            )

        reward_result = self._compute_rewards(qualified, dataset_name, config)
        self._attach_rewards(qualified, reward_result)
        self._lower_rollout_engine_memory_before_update(config)

        algorithm = resolve_algorithm("dapo")
        prepared_cpu = algorithm.prepare_batch(qualified, config)
        if prepared_cpu.rewards is None:
            prepared_cpu.rewards = reward_result.total_rewards
        if len(prepared_cpu.samples):
            prepared_cpu.advantages = compute_group_relative_advantages(
                prepared_cpu.rewards,
                group_size=prepared_cpu.group_size,
            )
        else:
            prepared_cpu.advantages = torch.zeros(0, dtype=torch.float32)

        group_signal_rate, group_total_reward_std_mean = _compute_group_signal_metrics(
            prepared_cpu.rewards,
            prepared_cpu.group_size,
        )
        device = next(self.actor_policy.model.parameters()).device
        policy_update = self._run_policy_update_with_oom_retry(
            prepared_cpu=prepared_cpu,
            algorithm=algorithm,
            config=config,
            update_actor=update_actor,
            device=device,
        )
        diagnostic_samples = _collect_diagnostic_samples(
            rollout_batch=qualified,
            reward_breakdowns=reward_result.reward_breakdowns,
            dataset_name=dataset_name,
            config=config,
        )

        return TrainerStepResult(
            algorithm_name="dapo",
            accuracy=self._safe_mean(correct_count, candidate_sample_count),
            truncation_rate=self._safe_mean(truncation_sum, candidate_sample_count),
            parse_fail_rate=self._safe_mean(parse_fail_count, candidate_sample_count),
            mean_response_length=self._safe_mean(
                response_length_sum,
                candidate_sample_count,
            ),
            did_update_actor=policy_update.did_update_actor,
            num_policy_updates=config.num_policy_updates,
            num_rollout_samples=candidate_sample_count,
            num_kept_samples=len(prepared_cpu.samples),
            update_losses=policy_update.update_losses,
            mini_batch_losses=policy_update.mini_batch_losses,
            answer_reward_mean=self._safe_mean(answer_reward_sum, candidate_sample_count),
            format_reward_mean=self._safe_mean(format_reward_sum, candidate_sample_count),
            total_reward_mean=self._safe_mean(total_reward_sum, candidate_sample_count),
            approx_kl=policy_update.approx_kl,
            ref_kl=policy_update.ref_kl,
            clip_fraction=policy_update.clip_fraction,
            policy_entropy=policy_update.policy_entropy,
            group_all_correct_rate=self._safe_mean(
                all_correct_prompt_count,
                candidate_prompt_count,
            ),
            group_all_wrong_rate=self._safe_mean(
                all_wrong_prompt_count,
                candidate_prompt_count,
            ),
            group_mixed_answer_rate=self._safe_mean(
                mixed_prompt_count,
                candidate_prompt_count,
            ),
            group_signal_rate=group_signal_rate,
            group_total_reward_std_mean=group_total_reward_std_mean,
            diagnostic_samples=diagnostic_samples,
            skipped_step=False,
            extra_metrics={
                "dapo_num_gen_batches": num_gen_batches,
                "dapo_candidate_prompt_count": candidate_prompt_count,
                "dapo_mixed_prompt_count": mixed_prompt_count,
                "dapo_sampled_mixed_prompt_count": sampled_mixed_prompt_count,
                "dapo_overflow_mixed_prompt_count": overflow_mixed_prompt_count,
                "dapo_rejected_prompt_count": len(rejected_records),
                "dapo_recycled_prompt_count": recycled_prompt_count,
                "dapo_sampling_acceptance_rate": self._safe_mean(
                    sampled_mixed_prompt_count,
                    candidate_prompt_count,
                ),
            },
        )

    def _rollout_candidate_records(
        self,
        records: list[dict],
        dataset_name: str,
        config: TrainConfig,
    ) -> RolloutBatch:
        prompts = [
            build_eval_prompt(record, dataset_name, tokenizer=self.tokenizer)
            for record in records
        ]
        rollout_batch = self.rollout_engine.generate(
            prompts,
            num_samples_per_prompt=config.group_size,
            sampling_overrides={"max_tokens": config.max_new_tokens},
        )
        rollout_batch.records = self._expand_records(records, config.group_size)
        return rollout_batch

    def _split_mixed_groups(
        self,
        rollout_batch: RolloutBatch,
    ) -> tuple[RolloutBatch, list[dict], int, int]:
        if rollout_batch.answer_rewards is None:
            raise ValueError("DAPO dynamic sampling 需要 answer_rewards。")
        group_size = rollout_batch.group_size
        grouped_rewards = rollout_batch.answer_rewards.view(-1, group_size)
        keep_indices: list[int] = []
        rejected_records: list[dict] = []
        all_correct_count = 0
        all_wrong_count = 0
        for group_idx, rewards in enumerate(grouped_rewards):
            start = group_idx * group_size
            end = start + group_size
            has_correct = bool(rewards.gt(0).any())
            has_wrong = bool(rewards.le(0).any())
            if has_correct and has_wrong:
                keep_indices.extend(range(start, end))
                continue
            all_correct_count += int(has_correct and not has_wrong)
            all_wrong_count += int(has_wrong and not has_correct)
            if rollout_batch.records is not None:
                rejected_records.append(rollout_batch.records[start])
        mixed_batch = (
            rollout_batch.slice(keep_indices)
            if keep_indices
            else RolloutBatch.from_samples([], [], group_size, records=[])
        )
        return mixed_batch, rejected_records, all_correct_count, all_wrong_count

    @staticmethod
    def _concat_rollout_batches(
        batches: list[RolloutBatch],
        group_size: int,
    ) -> RolloutBatch:
        samples = []
        records = []
        for batch in batches:
            samples.extend(batch.samples)
            if batch.records is not None:
                records.extend(batch.records)
        return RolloutBatch.from_samples(
            prompts=[],
            samples=samples,
            group_size=group_size,
            records=records,
        )

    @staticmethod
    def _group_records_from_batch(
        batch: RolloutBatch,
        *,
        start_sample_idx: int,
    ) -> list[dict]:
        if batch.records is None:
            return []
        group_size = batch.group_size
        records: list[dict] = []
        for sample_idx in range(start_sample_idx, len(batch.samples), group_size):
            records.append(batch.records[sample_idx])
        return records

    def _build_skipped_result(
        self,
        *,
        config: TrainConfig,
        candidate_sample_count: int,
        kept_sample_count: int,
        candidate_prompt_count: int,
        mixed_prompt_count: int,
        rejected_prompt_count: int,
        recycled_prompt_count: int,
        num_gen_batches: int,
        answer_reward_sum: float,
        format_reward_sum: float,
        total_reward_sum: float,
        parse_fail_count: float,
        truncation_sum: float,
        response_length_sum: float,
        correct_count: float,
        all_correct_prompt_count: int,
        all_wrong_prompt_count: int,
    ) -> TrainerStepResult:
        return TrainerStepResult(
            algorithm_name="dapo",
            accuracy=self._safe_mean(correct_count, candidate_sample_count),
            truncation_rate=self._safe_mean(truncation_sum, candidate_sample_count),
            parse_fail_rate=self._safe_mean(parse_fail_count, candidate_sample_count),
            mean_response_length=self._safe_mean(
                response_length_sum,
                candidate_sample_count,
            ),
            did_update_actor=False,
            num_policy_updates=config.num_policy_updates,
            num_rollout_samples=candidate_sample_count,
            num_kept_samples=kept_sample_count,
            update_losses=[],
            mini_batch_losses=[],
            answer_reward_mean=self._safe_mean(answer_reward_sum, candidate_sample_count),
            format_reward_mean=self._safe_mean(format_reward_sum, candidate_sample_count),
            total_reward_mean=self._safe_mean(total_reward_sum, candidate_sample_count),
            group_all_correct_rate=self._safe_mean(
                all_correct_prompt_count,
                candidate_prompt_count,
            ),
            group_all_wrong_rate=self._safe_mean(
                all_wrong_prompt_count,
                candidate_prompt_count,
            ),
            group_mixed_answer_rate=self._safe_mean(
                mixed_prompt_count,
                candidate_prompt_count,
            ),
            skipped_step=True,
            skip_reason="dapo_dynamic_sampling_insufficient_mixed_prompts",
            extra_metrics={
                "dapo_num_gen_batches": num_gen_batches,
                "dapo_candidate_prompt_count": candidate_prompt_count,
                "dapo_mixed_prompt_count": mixed_prompt_count,
                "dapo_sampled_mixed_prompt_count": mixed_prompt_count,
                "dapo_overflow_mixed_prompt_count": 0,
                "dapo_rejected_prompt_count": rejected_prompt_count,
                "dapo_recycled_prompt_count": recycled_prompt_count,
                "dapo_sampling_acceptance_rate": self._safe_mean(
                    mixed_prompt_count,
                    candidate_prompt_count,
                ),
            },
        )

    @staticmethod
    def _safe_mean(total: float, count: int) -> float:
        return float(total / count) if count else 0.0


def run_dapo_dynamic_training(config: TrainConfig) -> dict:
    """DAPO strict dynamic sampling 的完整训练入口。"""
    if config.algorithm_name.lower() != "dapo" or not config.dapo_use_dynamic_sampling:
        raise ValueError("run_dapo_dynamic_training 只支持 DAPO dynamic sampling。")

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

    actor_model, tokenizer, device, dtype_name = _load_actor(config)
    actor_policy = CausalLmPolicy(
        model=actor_model, pad_token_id=tokenizer.pad_token_id
    )
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

    optimizer = torch.optim.AdamW(
        [p for p in actor_model.parameters() if p.requires_grad],
        lr=config.learning_rate,
    )

    records: list[dict] = []
    with open(config.dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if config.max_samples is not None:
        records = records[: config.max_samples]

    gen_prompt_batch_size = config.dapo_gen_prompt_batch_size or config.prompt_batch_size
    num_total_steps = (
        config.max_steps
        if config.max_steps is not None
        else math.ceil(
            len(records) * max(1, config.num_epochs) / max(1, config.prompt_batch_size)
        )
    )
    first_epoch_steps = math.ceil(len(records) / max(1, gen_prompt_batch_size))
    logger.info(
        "DAPO dynamic training: planned_steps=%d  first_epoch_steps=%d  prompts=%d  prompt_batch_size=%d  gen_prompt_batch_size=%d  group_size=%d  epochs=%d",
        num_total_steps,
        first_epoch_steps,
        len(records),
        config.prompt_batch_size,
        gen_prompt_batch_size,
        config.group_size,
        config.num_epochs,
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
    use_wandb = _init_wandb(config)

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

    trainer = DapoDynamicTrainer(
        rollout_engine=rollout_engine,
        actor_policy=actor_policy,
        optimizer=optimizer,
        tokenizer=tokenizer,
        reference_policy=ref_policy,
    )
    candidate_pool = DapoCandidatePool(records, config)

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

    stop_reason = "completed"
    update_step_idx = 0

    def _dapo_step_requests():
        nonlocal stop_reason, update_step_idx
        step_idx = 0
        while True:
            if config.max_steps is not None and update_step_idx >= config.max_steps:
                stop_reason = f"max_steps={config.max_steps} reached"
                return
            if not candidate_pool.has_candidates():
                stop_reason = "dapo_candidate_pool_exhausted"
                return

            step_idx += 1
            epoch_idx = (
                math.ceil(candidate_pool.drawn / max(1, len(records)))
                if records
                else 0
            )
            epoch_idx = min(max(1, epoch_idx), max(1, config.num_epochs))

            def _run_step() -> TrainerStepResult:
                nonlocal update_step_idx
                result = trainer.train_step_from_pool(
                    candidate_pool=candidate_pool,
                    dataset_name=config.dataset_name,
                    config=config,
                    update_actor=True,
                )
                if result.did_update_actor:
                    update_step_idx += 1
                return result

            yield TrainingStepRequest(
                step_idx=step_idx,
                epoch_idx=epoch_idx,
                run=_run_step,
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
        step_requests=_dapo_step_requests(),
        get_stop_reason=lambda: stop_reason,
    )
