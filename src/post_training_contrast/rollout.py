"""
rollout.py - vLLM rollout 生成引擎和批数据结构。

包含：
  - RolloutSample     单条生成结果
  - RolloutBatch      一次 rollout 后的批数据（训练主循环围绕它运转）
  - VllmRolloutEngine vLLM 封装（支持 fake client 注入，方便测试）
"""

from __future__ import annotations

import gc
import inspect
import logging
import os
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


class RolloutLoRAAdapterError(RuntimeError):
    """vLLM LoRA adapter 路径不可用时抛出，trainer 可自动切到 merged snapshot fallback。"""


def _stabilize_cuda_memory_before_vllm_init() -> None:
    """在初始化 vLLM 前做一次轻量 CUDA 清理，降低 profiling 抖动。"""
    gc.collect()
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        torch.cuda.synchronize()
    except Exception:
        pass


def _is_vllm_memory_profiling_error(exc: BaseException) -> bool:
    """判断是否命中 vLLM 0.5.4 初始化时的 memory profiling 断言。"""
    return "Error in memory profiling" in str(exc)


def _is_vllm_kv_cache_error(exc: BaseException) -> bool:
    message = str(exc)
    return (
        "No available memory for the cache blocks" in message
        or "maximum number of tokens that can be stored in KV cache" in message
    )


def _log_vllm_cache_capacity(llm) -> None:
    """打印 vLLM 初始化后的 KV cache 容量信息。"""
    llm_engine = getattr(llm, "llm_engine", None)
    cache_config = getattr(llm_engine, "cache_config", None)
    if cache_config is None:
        return

    num_gpu_blocks = getattr(cache_config, "num_gpu_blocks", None)
    block_size = getattr(cache_config, "block_size", None)
    if num_gpu_blocks is None or block_size is None:
        return

    kv_cache_token_capacity = num_gpu_blocks * block_size
    logger.info(
        "vLLM KV cache ready: num_gpu_blocks=%s  block_size=%s  kv_cache_token_capacity≈%s",
        num_gpu_blocks,
        block_size,
        kv_cache_token_capacity,
    )


def _get_vllm_cache_capacity(llm) -> int | None:
    """尽力读取 vLLM KV cache token 容量；取不到时返回 None。"""
    llm_engine = getattr(llm, "llm_engine", None)
    cache_config = getattr(llm_engine, "cache_config", None)
    if cache_config is None:
        return None

    num_gpu_blocks = getattr(cache_config, "num_gpu_blocks", None)
    block_size = getattr(cache_config, "block_size", None)
    if num_gpu_blocks is None or block_size is None:
        return None
    return int(num_gpu_blocks * block_size)


# ── 数据结构 ─────────────────────────────────────────────────────


@dataclass(eq=True)
class RolloutSample:
    """单条 rollout 结果。"""

    prompt: str
    prompt_token_ids: list[int]
    response_text: str
    response_token_ids: list[int]
    response_logprobs: list[float]
    finish_reason: str | None
    stop_reason: int | str | None
    is_truncated: bool
    group_id: int = 0
    sample_id_within_group: int = 0


def _pad_token_level_values(
    token_lists: list[list[float]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """把变长 token 列表右填充成二维张量，同时返回 token mask。"""
    if not token_lists:
        empty = torch.zeros((0, 0), dtype=torch.float32)
        return empty, empty

    max_length = max(len(t) for t in token_lists)
    padded, mask = [], []
    for tokens in token_lists:
        pad = max_length - len(tokens)
        padded.append(tokens + [0.0] * pad)
        mask.append([1.0] * len(tokens) + [0.0] * pad)

    return (
        torch.tensor(padded, dtype=torch.float32),
        torch.tensor(mask, dtype=torch.float32),
    )


@dataclass
class RolloutBatch:
    """一次 rollout 后的统一批数据。

    训练主循环里所有的 reward 计算、过滤、minibatch 切分都基于它。
    """

    prompts: list[str]
    samples: list[RolloutSample]
    group_size: int
    records: list[dict] | None = None
    rewards: torch.Tensor | None = None
    answer_rewards: torch.Tensor | None = None
    format_rewards: torch.Tensor | None = None
    advantages: torch.Tensor | None = None
    old_logprobs: torch.Tensor | None = None
    token_mask: torch.Tensor | None = None
    response_lengths: torch.Tensor | None = None
    truncation_mask: torch.Tensor | None = None

    @classmethod
    def from_samples(
        cls,
        prompts: list[str],
        samples: list[RolloutSample],
        group_size: int,
        records: list[dict] | None = None,
    ) -> RolloutBatch:
        """从 RolloutSample 列表直接构造批数据，自动补齐所有张量。"""
        old_logprobs, token_mask = _pad_token_level_values(
            [s.response_logprobs for s in samples]
        )
        response_lengths = torch.tensor(
            [len(s.response_token_ids) for s in samples], dtype=torch.long
        )
        truncation_mask = torch.tensor(
            [1.0 if s.is_truncated else 0.0 for s in samples], dtype=torch.float32
        )
        return cls(
            prompts=prompts,
            samples=samples,
            group_size=group_size,
            records=records,
            old_logprobs=old_logprobs,
            token_mask=token_mask,
            response_lengths=response_lengths,
            truncation_mask=truncation_mask,
        )

    def slice(self, indices: list[int]) -> RolloutBatch:
        """取出一个子批次，所有张量和 records 一起对齐切片。"""
        # 自动检测数据张量所在设备，确保 idx 与之一致
        device = None
        for t in (self.rewards, self.advantages, self.old_logprobs, self.token_mask):
            if t is not None:
                device = t.device
                break
        idx = torch.tensor(indices, dtype=torch.long, device=device)
        return RolloutBatch(
            prompts=self.prompts,
            samples=[self.samples[i] for i in indices],
            group_size=self.group_size,
            records=None if self.records is None else [self.records[i] for i in indices],
            rewards=None if self.rewards is None else self.rewards.index_select(0, idx),
            answer_rewards=None if self.answer_rewards is None else self.answer_rewards.index_select(0, idx),
            format_rewards=None if self.format_rewards is None else self.format_rewards.index_select(0, idx),
            advantages=None if self.advantages is None else self.advantages.index_select(0, idx),
            old_logprobs=None if self.old_logprobs is None else self.old_logprobs.index_select(0, idx),
            token_mask=None if self.token_mask is None else self.token_mask.index_select(0, idx),
            response_lengths=None if self.response_lengths is None else self.response_lengths.index_select(0, idx),
            truncation_mask=None if self.truncation_mask is None else self.truncation_mask.index_select(0, idx),
        )


# ── vLLM 引擎 ────────────────────────────────────────────────────


def _extract_selected_logprobs(
    completion_logprobs, response_token_ids: list[int]
) -> list[float]:
    """从 vLLM 返回的逐位置 logprobs 里取出被采样 token 的 logprob。"""
    if completion_logprobs is None:
        return []
    selected = []
    for pos_logprobs, token_id in zip(completion_logprobs, response_token_ids):
        selected.append(float(pos_logprobs[token_id].logprob))
    return selected


class VllmRolloutEngine:
    """用 vLLM 做 rollout 生成。

    支持真实 vLLM 和测试时注入的 fake client（通过 llm 参数）。
    """

    def __init__(
        self,
        model_name_or_path: str,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 1.0,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
        trust_remote_code: bool = True,
        dtype: str = "auto",
        enable_lora: bool = False,
        lora_adapter_path: str | None = None,
        lora_adapter_name: str = "rollout_adapter",
        lora_adapter_id: int = 1,
        llm=None,
        sampling_params_factory=None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.trust_remote_code = trust_remote_code
        self.dtype = dtype
        self.enable_lora = enable_lora
        self.lora_adapter_path = lora_adapter_path
        self.lora_adapter_name = lora_adapter_name
        self.lora_adapter_id = lora_adapter_id
        self._llm = llm
        self._sampling_params_factory = sampling_params_factory
        self._is_sleeping = False
        self._last_release_reason: str | None = None

    def _build_llm(self):
        """惰性创建真实的 vLLM LLM 实例。"""
        if self._llm is not None:
            self._wake_if_needed()
            return self._llm

        os.environ.setdefault("VLLM_CACHE_ROOT", "/tmp/vllm")
        os.environ.setdefault("VLLM_CONFIG_ROOT", "/tmp/vllm_config")
        os.environ.setdefault("VLLM_RPC_BASE_PATH", "/tmp")

        try:
            from vllm import LLM
        except ImportError as e:
            raise RuntimeError("当前环境没有安装 vllm，无法启动真实 rollout。") from e

        kwargs = {
            "model": self.model_name_or_path,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "trust_remote_code": self.trust_remote_code,
            "dtype": self.dtype,
            "enable_lora": self.enable_lora,
        }
        if self.max_model_len is not None:
            kwargs["max_model_len"] = self.max_model_len

        init_mode = (
            "fresh_rebuild_after_fallback"
            if self._last_release_reason == "fallback_release_retry"
            else "resident_rebuild"
            if self._last_release_reason
            else "cold_init"
        )
        logger.info(
            "Initializing vLLM engine [%s]: model=%s  tp=%d  gpu_mem=%.2f  dtype=%s",
            init_mode,
            self.model_name_or_path,
            self.tensor_parallel_size,
            self.gpu_memory_utilization,
            self.dtype,
        )
        _stabilize_cuda_memory_before_vllm_init()
        try:
            self._llm = LLM(**kwargs)
        except Exception as exc:
            if _is_vllm_kv_cache_error(exc):
                logger.warning(
                    "vLLM KV cache initialization failed: %s. "
                    "Try lowering max_model_len or increasing gpu_memory_utilization.",
                    exc,
                )
                raise
            if _is_vllm_memory_profiling_error(exc):
                logger.warning(
                    "vLLM init hit memory profiling instability; cleaning CUDA state and retrying once."
                )
                _stabilize_cuda_memory_before_vllm_init()
                self._llm = LLM(**kwargs)
            else:
                raise
        self._last_release_reason = None
        logger.info("vLLM engine ready.")
        _log_vllm_cache_capacity(self._llm)
        return self._llm

    def _build_lora_request(self):
        if not self.enable_lora or not self.lora_adapter_path:
            return None

        try:
            from vllm.lora.request import LoRARequest
        except ImportError:
            try:
                from vllm import LoRARequest
            except ImportError as e:
                raise RolloutLoRAAdapterError(
                    "当前 vLLM 环境缺少 LoRARequest，无法走 adapter rollout。"
                ) from e

        kwargs: dict = {}
        try:
            if "load_inplace" in inspect.signature(LoRARequest).parameters:
                kwargs["load_inplace"] = True
        except (TypeError, ValueError):
            pass

        return LoRARequest(
            self.lora_adapter_name,
            self.lora_adapter_id,
            self.lora_adapter_path,
            **kwargs,
        )

    def _build_sampling_params(
        self,
        num_samples_per_prompt: int,
        overrides: dict | None = None,
        return_logprobs: bool = True,
    ):
        kwargs = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_new_tokens,
            "n": num_samples_per_prompt,
        }
        if return_logprobs:
            kwargs["logprobs"] = 1
        if overrides:
            kwargs.update(overrides)
        if self._sampling_params_factory is not None:
            return self._sampling_params_factory(**kwargs)
        try:
            from vllm import SamplingParams
        except ImportError as e:
            raise RuntimeError("当前环境没有安装 vllm，无法创建 SamplingParams。") from e
        return SamplingParams(**kwargs)

    def generate(
        self,
        prompts: list[str],
        num_samples_per_prompt: int = 1,
        sampling_overrides: dict | None = None,
        return_logprobs: bool = True,
    ) -> RolloutBatch:
        """按 prompt 组采样 rollout 样本，返回 RolloutBatch。"""
        logger.info(
            "Rollout: %d prompts × %d samples = %d total",
            len(prompts), num_samples_per_prompt, len(prompts) * num_samples_per_prompt,
        )
        llm = self._build_llm()
        params = self._build_sampling_params(
            num_samples_per_prompt,
            sampling_overrides,
            return_logprobs=return_logprobs,
        )
        lora_request = self._build_lora_request()
        try:
            if lora_request is None:
                outputs = llm.generate(prompts, params)
            else:
                outputs = llm.generate(prompts, params, lora_request=lora_request)
        except TypeError as e:
            if lora_request is None:
                raise
            raise RolloutLoRAAdapterError(
                "当前 vLLM 版本不支持在 generate() 中传 lora_request。"
            ) from e
        except Exception as e:
            if lora_request is None:
                raise
            raise RolloutLoRAAdapterError(
                f"LoRA adapter rollout 失败：{e}"
            ) from e

        samples: list[RolloutSample] = []
        prompt_lengths: list[int] = []
        response_lengths: list[int] = []
        for group_id, request_output in enumerate(outputs):
            completions = list(getattr(request_output, "outputs", []) or [])
            prompt_len = len(list(getattr(request_output, "prompt_token_ids", []) or []))
            for sid, completion in enumerate(completions):
                finish_reason = getattr(completion, "finish_reason", None)
                response_len = len(completion.token_ids)
                prompt_lengths.append(prompt_len)
                response_lengths.append(response_len)
                samples.append(
                    RolloutSample(
                        prompt=request_output.prompt,
                        prompt_token_ids=list(getattr(request_output, "prompt_token_ids", []) or []),
                        response_text=completion.text,
                        response_token_ids=list(completion.token_ids),
                        response_logprobs=_extract_selected_logprobs(
                            completion.logprobs, list(completion.token_ids)
                        ),
                        finish_reason=finish_reason,
                        stop_reason=getattr(completion, "stop_reason", None),
                        is_truncated=finish_reason == "length",
                        group_id=group_id,
                        sample_id_within_group=sid,
                    )
                )

        total_sequences = len(samples)
        mean_prompt_tokens = (
            sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0.0
        )
        mean_response_length = (
            sum(response_lengths) / len(response_lengths) if response_lengths else 0.0
        )
        approx_active_tokens = sum(
            prompt_len + response_len
            for prompt_len, response_len in zip(prompt_lengths, response_lengths)
        )
        kv_cache_token_capacity = _get_vllm_cache_capacity(llm)
        if kv_cache_token_capacity:
            approx_kv_fill_ratio = approx_active_tokens / max(1, kv_cache_token_capacity)
            logger.info(
                "Rollout stats: num_prompts=%d  group_size=%d  total_sequences=%d  mean_prompt_tokens=%.1f  mean_response_length=%.1f  approx_active_tokens=%d  kv_cache_token_capacity≈%d  approx_kv_fill_ratio≈%.4f",
                len(prompts),
                num_samples_per_prompt,
                total_sequences,
                mean_prompt_tokens,
                mean_response_length,
                approx_active_tokens,
                kv_cache_token_capacity,
                approx_kv_fill_ratio,
            )
        else:
            logger.info(
                "Rollout stats: num_prompts=%d  group_size=%d  total_sequences=%d  mean_prompt_tokens=%.1f  mean_response_length=%.1f  approx_active_tokens=%d",
                len(prompts),
                num_samples_per_prompt,
                total_sequences,
                mean_prompt_tokens,
                mean_response_length,
                approx_active_tokens,
            )
        logger.info("Rollout done: %d samples generated.", len(samples))
        return RolloutBatch.from_samples(
            prompts=prompts,
            samples=samples,
            group_size=num_samples_per_prompt,
        )

    def refresh_lora_adapter(
        self,
        adapter_path: str,
        adapter_id: int,
        adapter_name: str | None = None,
    ) -> None:
        """更新下一次 rollout 要挂载的 LoRA adapter 信息。"""
        self.enable_lora = True
        self.lora_adapter_path = adapter_path
        self.lora_adapter_id = adapter_id
        if adapter_name is not None:
            self.lora_adapter_name = adapter_name

    def _wake_if_needed(self) -> None:
        if not self._is_sleeping or self._llm is None:
            return

        for target in (self._llm, getattr(self._llm, "llm_engine", None)):
            if target is None:
                continue
            for method_name in ("wake_up", "wake"):
                wake_fn = getattr(target, method_name, None)
                if callable(wake_fn):
                    try:
                        wake_fn()
                        self._is_sleeping = False
                        logger.info("vLLM engine resumed from sleep.")
                        return
                    except TypeError:
                        continue
                    except Exception as e:
                        logger.warning("Failed to wake vLLM engine: %s", e)
                        self.release()
                        return

        self.release()

    def reduce_memory_usage(self, prefer_sleep: bool = False) -> None:
        """尽量降低 rollout 阶段之外的显存占用。"""
        if prefer_sleep and self._llm is not None:
            for target in (self._llm, getattr(self._llm, "llm_engine", None)):
                if target is None:
                    continue
                sleep_fn = getattr(target, "sleep", None)
                if not callable(sleep_fn):
                    continue
                try:
                    sleep_fn(level=1)
                except TypeError:
                    try:
                        sleep_fn()
                    except Exception:
                        continue
                except Exception:
                    continue
                self._is_sleeping = True
                logger.info("vLLM engine moved to sleep mode.")
                return

        self.release()

    def release(self, reason: str | None = None) -> None:
        """显式释放 vLLM 实例，避免 GPU OOM（在 reload_rollout_each_step=True 时调用）。"""
        if reason is not None:
            self._last_release_reason = reason
        if self._llm is not None:
            # vLLM 初始化了全局分布式环境，必须显式销毁才能彻底释放 40G 的 KV Block
            try:
                from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
                destroy_model_parallel()
                destroy_distributed_environment()
            except ImportError:
                pass
            except Exception as e:
                logger.warning(f"Failed to destroy vllm distributed env: {e}")

            del self._llm
            self._llm = None
            self._is_sleeping = False

            import gc
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:
                pass
