"""Shared training-loop runtime for RL trainers."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from src.post_training_contrast.config import TrainConfig
from src.post_training_contrast.rollout import RolloutLoRAAdapterError
from src.post_training_contrast.trainer import (
    RlTrainer,
    RolloutSyncState,
    TrainerStepResult,
    _refresh_rollout_engine,
    _run_validation,
    _smooth_wandb_train_metrics,
    _switch_rollout_engine_to_snapshot_fallback,
    _wandb_log,
)

logger = logging.getLogger(__name__)


RAW_WANDB_TRAIN_METRIC_NAMES = {
    "train/skipped_step",
    "train/update_step",
    "train/dapo_num_gen_batches",
    "train/dapo_mixed_prompt_count",
    "train/dapo_rejected_prompt_count",
}


@dataclass(frozen=True)
class TrainingStepRequest:
    step_idx: int
    epoch_idx: int
    run: Callable[[], TrainerStepResult]


def _step_summary_from_result(
    *,
    step_idx: int,
    update_step_idx: int,
    epoch_idx: int,
    result: TrainerStepResult,
    num_diagnostic_samples: int,
    current_lr: float,
) -> dict:
    payload = {
        "step": step_idx,
        "update_step": update_step_idx,
        "epoch": epoch_idx,
        "algorithm_name": result.algorithm_name,
        "skipped_step": result.skipped_step,
        "skip_reason": result.skip_reason,
        "accuracy": result.accuracy,
        "answer_reward_mean": result.answer_reward_mean,
        "format_reward_mean": result.format_reward_mean,
        "total_reward_mean": result.total_reward_mean,
        "truncation_rate": result.truncation_rate,
        "parse_fail_rate": result.parse_fail_rate,
        "mean_response_length": result.mean_response_length,
        "num_rollout_samples": result.num_rollout_samples,
        "num_kept_samples": result.num_kept_samples,
        "num_diagnostic_samples": num_diagnostic_samples,
        "num_policy_updates": result.num_policy_updates,
        "group_all_correct_rate": result.group_all_correct_rate,
        "group_all_wrong_rate": result.group_all_wrong_rate,
        "group_mixed_answer_rate": result.group_mixed_answer_rate,
        "group_signal_rate": result.group_signal_rate,
        "group_total_reward_std_mean": result.group_total_reward_std_mean,
        "approx_kl": result.approx_kl,
        "ref_kl": result.ref_kl,
        "clip_fraction": result.clip_fraction,
        "policy_entropy": result.policy_entropy,
        "update_losses": result.update_losses,
        "mini_batch_losses": result.mini_batch_losses,
        "lr": current_lr,
    }
    payload.update(result.extra_metrics)
    return payload


def _wandb_train_metrics_from_result(
    result: TrainerStepResult,
    update_step_idx: int,
) -> dict[str, float]:
    metrics = {
        "train/accuracy": result.accuracy,
        "train/answer_reward_mean": result.answer_reward_mean,
        "train/format_reward_mean": result.format_reward_mean,
        "train/truncation_rate": result.truncation_rate,
        "train/parse_fail_rate": result.parse_fail_rate,
        "train/mean_response_length": result.mean_response_length,
        "train/group_all_wrong_rate": result.group_all_wrong_rate,
        "train/loss": result.update_losses[-1]
        if result.update_losses
        else float("nan"),
        "train/approx_kl": result.approx_kl,
        "train/ref_kl": result.ref_kl,
        "train/clip_fraction": result.clip_fraction,
        "train/policy_entropy": result.policy_entropy,
        "train/skipped_step": float(result.skipped_step),
        "train/update_step": update_step_idx,
    }
    dapo_wandb_metrics = {
        "train/dapo_num_gen_batches": "dapo_num_gen_batches",
        "train/dapo_mixed_prompt_count": "dapo_mixed_prompt_count",
        "train/dapo_rejected_prompt_count": "dapo_rejected_prompt_count",
        "train/dapo_sampling_acceptance_rate": "dapo_sampling_acceptance_rate",
    }
    for wandb_name, extra_metric_name in dapo_wandb_metrics.items():
        metric_value = result.extra_metrics.get(extra_metric_name)
        if isinstance(metric_value, (int, float)):
            metrics[wandb_name] = metric_value
    return metrics


def run_training_event_loop(
    *,
    config: TrainConfig,
    output_dir: Path,
    actor_model,
    tokenizer,
    trainer: RlTrainer,
    scheduler,
    rollout_state: RolloutSyncState,
    eval_records: list[dict],
    diagnostic_samples_path: Path | None,
    num_total_steps: int,
    first_epoch_steps: int,
    best_metric_name: str,
    use_wandb: bool,
    step_requests: Iterable[TrainingStepRequest],
    get_stop_reason: Callable[[], str],
) -> dict:
    step_summaries: list[dict] = []
    wandb_train_metric_history: list[dict[str, float]] = []
    best_val_accuracy: float | None = None
    best_step: int | None = None
    best_checkpoint_path: str | None = None
    non_improving_evals = 0
    stopped_early = False
    stop_reason = "completed"
    update_step_idx = 0

    early_stopping_enabled = (
        bool(eval_records)
        and config.eval_interval > 0
        and config.early_stopping_patience is not None
    )

    for request in step_requests:
        step_idx = request.step_idx
        epoch_idx = request.epoch_idx
        try:
            result = request.run()
        except RolloutLoRAAdapterError as exc:
            if rollout_state.mode != "lora_adapter":
                raise
            logger.warning(
                "LoRA adapter rollout failed at step %d: %s", step_idx, exc
            )
            _switch_rollout_engine_to_snapshot_fallback(
                trainer,
                actor_model,
                tokenizer,
                config,
                rollout_state,
            )
            result = request.run()

        if result.did_update_actor:
            scheduler.step()
            update_step_idx += 1
        current_lr = scheduler.get_last_lr()[0]
        result.lr = current_lr
        result.update_step = update_step_idx

        num_diagnostic_samples = 0
        if diagnostic_samples_path is not None and result.diagnostic_samples:
            with open(diagnostic_samples_path, "a", encoding="utf-8") as f:
                for sample in result.diagnostic_samples:
                    payload = {
                        "step": step_idx,
                        "epoch": epoch_idx,
                        **sample,
                    }
                    json.dump(payload, f, ensure_ascii=False)
                    f.write("\n")
                    num_diagnostic_samples += 1

        logger.info(
            "step=%d/%d  update_step=%d  algo=%s  acc=%.4f  kept=%d/%d  loss=%.4f  lr=%.2e  skipped=%s",
            step_idx,
            num_total_steps,
            update_step_idx,
            result.algorithm_name,
            result.accuracy,
            result.num_kept_samples,
            result.num_rollout_samples,
            (result.update_losses[-1] if result.update_losses else float("nan")),
            current_lr,
            result.skipped_step,
        )

        step_summaries.append(
            _step_summary_from_result(
                step_idx=step_idx,
                update_step_idx=update_step_idx,
                epoch_idx=epoch_idx,
                result=result,
                num_diagnostic_samples=num_diagnostic_samples,
                current_lr=current_lr,
            )
        )

        train_metrics_for_wandb = _wandb_train_metrics_from_result(
            result,
            update_step_idx,
        )
        smoothed_train_metrics = _smooth_wandb_train_metrics(
            wandb_train_metric_history,
            train_metrics_for_wandb,
            window_size=20,
        )
        for metric_name in RAW_WANDB_TRAIN_METRIC_NAMES:
            if metric_name in train_metrics_for_wandb:
                smoothed_train_metrics[metric_name] = train_metrics_for_wandb[metric_name]
        smoothed_train_metrics["train/lr"] = current_lr
        _wandb_log(
            step=step_idx,
            use_wandb=use_wandb,
            metrics=smoothed_train_metrics,
        )

        if (
            result.did_update_actor
            and config.checkpoint_interval > 0
            and update_step_idx % config.checkpoint_interval == 0
        ):
            ckpt_dir = output_dir / f"checkpoint_step{update_step_idx}"
            actor_model.save_pretrained(ckpt_dir, save_embedding_layers=False)
            tokenizer.save_pretrained(ckpt_dir)
            logger.info("Checkpoint saved → %s", ckpt_dir)

        needs_validation = (
            result.did_update_actor
            and bool(eval_records)
            and config.eval_interval > 0
            and update_step_idx % config.eval_interval == 0
        )
        rollout_synced_this_step = False
        if (
            needs_validation
            and config.reload_rollout_each_step
            and (config.val_backend or "hf").lower() == "vllm"
        ):
            _refresh_rollout_engine(
                trainer,
                actor_model,
                tokenizer,
                config,
                rollout_state=rollout_state,
            )
            rollout_synced_this_step = True

        if needs_validation:
            if (config.val_backend or "hf").lower() == "hf":
                trainer.rollout_engine.reduce_memory_usage(prefer_sleep=False)
            logger.info(
                "[Val] Running validation at update_step %d with backend=%s...",
                update_step_idx,
                config.val_backend,
            )
            try:
                val_acc = _run_validation(
                    actor_model=actor_model,
                    tokenizer=tokenizer,
                    rollout_engine=trainer.rollout_engine,
                    eval_records=eval_records,
                    dataset_name=config.dataset_name,
                    config=config,
                )
            except RolloutLoRAAdapterError as exc:
                if rollout_state.mode != "lora_adapter":
                    raise
                logger.warning(
                    "LoRA adapter validation failed at step %d: %s", step_idx, exc
                )
                _switch_rollout_engine_to_snapshot_fallback(
                    trainer,
                    actor_model,
                    tokenizer,
                    config,
                    rollout_state,
                )
                rollout_synced_this_step = True
                val_acc = _run_validation(
                    actor_model=actor_model,
                    tokenizer=tokenizer,
                    rollout_engine=trainer.rollout_engine,
                    eval_records=eval_records,
                    dataset_name=config.dataset_name,
                    config=config,
                )
            logger.info("[Val] step=%d  accuracy=%.4f", step_idx, val_acc)
            step_summaries[-1]["val_accuracy"] = val_acc
            _wandb_log(
                step=step_idx,
                use_wandb=use_wandb,
                metrics={"val/accuracy": val_acc},
            )

            improved = best_val_accuracy is None or val_acc > best_val_accuracy
            if improved:
                best_val_accuracy = val_acc
                best_step = update_step_idx
                non_improving_evals = 0
                if config.save_best_checkpoint:
                    best_ckpt_dir = output_dir / "checkpoint_best"
                    actor_model.save_pretrained(
                        best_ckpt_dir, save_embedding_layers=False
                    )
                    tokenizer.save_pretrained(best_ckpt_dir)
                    best_checkpoint_path = str(best_ckpt_dir)
                    logger.info(
                        "Best checkpoint updated → %s  metric=%s  step=%d  value=%.4f",
                        best_ckpt_dir,
                        best_metric_name,
                        update_step_idx,
                        val_acc,
                    )
            elif early_stopping_enabled:
                non_improving_evals += 1
                logger.info(
                    "Early stopping monitor: metric=%s  best=%.4f  current=%.4f  patience=%d/%d",
                    best_metric_name,
                    best_val_accuracy,
                    val_acc,
                    non_improving_evals,
                    config.early_stopping_patience,
                )
                if non_improving_evals >= config.early_stopping_patience:
                    early_stopping_can_trigger = update_step_idx >= first_epoch_steps
                    if early_stopping_can_trigger:
                        stopped_early = True
                        stop_reason = (
                            f"early_stopping_patience={config.early_stopping_patience} exhausted "
                            f"on {best_metric_name} at update_step {update_step_idx}"
                        )
                        logger.info("Early stopping triggered: %s", stop_reason)
                    else:
                        logger.info(
                            "Early stopping deferred until first epoch completes: update_step=%d  first_epoch_steps=%d",
                            update_step_idx,
                            first_epoch_steps,
                        )

        if stopped_early:
            break

        if (
            result.did_update_actor
            and config.reload_rollout_each_step
            and not rollout_synced_this_step
        ):
            _refresh_rollout_engine(
                trainer,
                actor_model,
                tokenizer,
                config,
                rollout_state=rollout_state,
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        stop_reason = get_stop_reason()

    final_ckpt = output_dir / "checkpoint_final"
    actor_model.save_pretrained(final_ckpt, save_embedding_layers=False)
    tokenizer.save_pretrained(final_ckpt)
    if config.use_lora:
        logger.info(
            "Final LoRA adapter saved → %s\n"
            "  合并命令（如需完整权重用于部署/评测）：\n"
            '  python -c "from peft import PeftModel; '
            "from transformers import AutoModelForCausalLM; "
            "m = AutoModelForCausalLM.from_pretrained('%s'); "
            "m = PeftModel.from_pretrained(m, '%s').merge_and_unload(); "
            "m.save_pretrained('%s/checkpoint_final_merged')\"",
            final_ckpt,
            config.model,
            final_ckpt,
            config.output_dir,
        )
    else:
        logger.info("Final model saved → %s", final_ckpt)

    summary = {
        "dataset_path": config.dataset_path,
        "dataset_name": config.dataset_name,
        "model": config.model,
        "algorithm_name": config.algorithm_name,
        "output_root": config.output_root or config.output_dir,
        "output_dir": config.output_dir,
        "run_id": config.run_id,
        "planned_num_steps": num_total_steps,
        "num_steps": len(step_summaries),
        "num_update_steps": update_step_idx,
        "num_skipped_steps": sum(
            1 for step in step_summaries if step.get("skipped_step")
        ),
        "skip_reasons": sorted(
            {
                step.get("skip_reason")
                for step in step_summaries
                if step.get("skip_reason")
            }
        ),
        "best_checkpoint_metric": best_metric_name,
        "best_val_accuracy": best_val_accuracy,
        "best_step": best_step,
        "best_checkpoint_path": best_checkpoint_path,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "diagnostic_samples_path": (
            str(diagnostic_samples_path) if diagnostic_samples_path is not None else None
        ),
        "trainer_config": asdict(config),
        "steps": step_summaries,
    }
    summary_path = output_dir / "train_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("Training summary → %s", summary_path)

    if use_wandb:
        try:
            import wandb

            wandb.finish()
        except Exception:
            pass

    return summary
