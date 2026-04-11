"""
policy.py - Actor / Reference policy：用 causal LM 计算 response 的逐 token logprob。

只依赖 PyTorch，不依赖任何项目内部模块。
被 trainer.py 调用，actor 和 reference 共用同一个类。
"""

from __future__ import annotations

from contextlib import ExitStack
from typing import NamedTuple

import torch


class PolicyOutput(NamedTuple):
    """compute_logprobs 的返回值。"""

    logprobs: torch.Tensor   # [batch, max_response_len]
    entropy: torch.Tensor    # [batch, max_response_len]  策略熵


def _build_padded_input_ids(
    token_id_sequences: list[list[int]],
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """把变长 token 序列右填充成批量输入，同时返回 attention mask。"""
    max_length = max(len(ids) for ids in token_id_sequences)
    input_ids, attention_mask = [], []
    for ids in token_id_sequences:
        pad = max_length - len(ids)
        input_ids.append(ids + [pad_token_id] * pad)
        attention_mask.append([1] * len(ids) + [0] * pad)
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attention_mask, dtype=torch.long),
    )


class CausalLmPolicy:
    """用 causal LM 计算 rollout response 的逐 token logprob。

    actor 和 reference policy 共用这个类：
      - actor_policy  = CausalLmPolicy(model=actor_model, ...)    # model.train()
      - ref_policy    = CausalLmPolicy(model=ref_model, ...)      # model.eval() + no grad
    """

    def __init__(
        self,
        model,
        pad_token_id: int,
        disable_adapter: bool = False,
        enforce_eval_mode: bool = False,
    ) -> None:
        self.model = model
        self.pad_token_id = pad_token_id
        self.disable_adapter = disable_adapter
        self.enforce_eval_mode = enforce_eval_mode

    def compute_logprobs(self, rollout_data, compute_entropy: bool = False) -> PolicyOutput:
        """计算每条 rollout response 的逐 token logprob 和策略熵。

        Returns
        -------
        PolicyOutput
            logprobs: [batch, max_response_len] 补零张量（padding 位置为 0）。
            entropy:  [batch, max_response_len] 策略熵（padding 位置为 0）。
        """
        samples = (
            rollout_data.samples
            if hasattr(rollout_data, "samples")
            else rollout_data
        )
        # prompt + response 拼接成完整序列，再前向传播
        full_sequences = [s.prompt_token_ids + s.response_token_ids for s in samples]
        input_ids, attention_mask = _build_padded_input_ids(full_sequences, self.pad_token_id)

        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        was_training = self.model.training
        if self.enforce_eval_mode and was_training:
            self.model.eval()

        try:
            with ExitStack() as stack:
                if self.disable_adapter:
                    disable_adapter = getattr(self.model, "disable_adapter", None)
                    if disable_adapter is None:
                        raise RuntimeError("当前模型不支持 disable_adapter()，无法计算 LoRA reference logprobs。")
                    stack.enter_context(disable_adapter())
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
        finally:
            if self.enforce_eval_mode and was_training:
                self.model.train()

        logits = outputs.logits[:, :-1, :]           # [B, L-1, vocab]
        target_ids = input_ids[:, 1:]                # [B, L-1]

        # F.cross_entropy 是 fused kernel，不会创建 [B, L, vocab] 的中间张量
        # 比 log_softmax + gather 节省 ~5-9 GB 显存
        import torch.nn.functional as F
        token_logprobs = -F.cross_entropy(
            logits.transpose(1, 2),
            target_ids,
            reduction="none",
        )                                            # [B, L-1]

        # 策略熵：分块计算，每次只处理一个 batch 样本，避免一次性创建巨大张量
        token_entropy = None
        if compute_entropy:
            entropy_list = []
            for sample_logits in logits.unbind(dim=0):
                lp = torch.log_softmax(sample_logits, dim=-1)  # [L-1, vocab]
                ent = -(lp.exp() * lp).sum(dim=-1)         # [L-1]
                entropy_list.append(ent)
                del lp
            token_entropy = torch.stack(entropy_list, dim=0)  # [B, L-1]

        del logits  # 释放最大的张量

        # 只取 response 部分
        batch_size = len(samples)
        max_resp_len = max(len(s.response_token_ids) for s in samples)
        response_logprobs = token_logprobs.new_zeros((batch_size, max_resp_len))
        response_entropy = None
        if token_entropy is not None:
            response_entropy = token_entropy.new_zeros((batch_size, max_resp_len))

        for i, sample in enumerate(samples):
            p_len = len(sample.prompt_token_ids)
            r_len = len(sample.response_token_ids)
            start = p_len - 1
            response_logprobs[i, :r_len] = token_logprobs[i, start : start + r_len]
            if response_entropy is not None:
                response_entropy[i, :r_len] = token_entropy[i, start : start + r_len]

        return PolicyOutput(logprobs=response_logprobs, entropy=response_entropy)
