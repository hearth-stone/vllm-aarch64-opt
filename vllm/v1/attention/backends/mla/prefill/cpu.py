# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU 平台的 MLA prefill backend。

使用 PyTorch 的 ``scaled_dot_product_attention`` 在 CPU 上实现变长（varlen）
注意力，替代 CUDA 上的 ``flash_attn_varlen_func``。

在 v0.21 的 MLA 架构里，prefill 计算被解耦为独立的 ``MLAPrefillBackend``：
``MLACommonImpl.forward_mha`` / ``_compute_prefill_context`` 调用
``prefill_metadata.prefill_backend.run_prefill_new_tokens`` /
``run_prefill_context_chunk``。本模块为 CPU 提供该后端，让 CPU 路径可以
直接复用父类 ``forward_mha`` 流程，不必再 monkey-patch 模块级符号。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import torch
import torch.nn.functional as F

from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.platforms.interface import DeviceCapability
    from vllm.v1.attention.backends.mla.prefill.selector import (
        MLAPrefillSelectorConfig,
    )


class CPUPrefillBackend(MLAPrefillBackend):
    """CPU 平台的 MLA prefill backend（PyTorch SDPA 实现）。"""

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]

    @staticmethod
    def get_name() -> str:
        return "CPU"

    @classmethod
    def supports_compute_capability(cls, device_capability: "DeviceCapability") -> bool:
        # CPU 不依赖 GPU compute capability。
        return True

    @classmethod
    def is_available(cls) -> bool:
        return True

    @classmethod
    def validate_configuration(
        cls,
        device_capability: "DeviceCapability",
        selector_config: "MLAPrefillSelectorConfig",
    ) -> list[str]:
        invalid_reasons: list[str] = []
        if not cls.supports_dtype(selector_config.dtype):
            invalid_reasons.append(f"dtype {selector_config.dtype} not supported")
        return invalid_reasons

    def __init__(
        self,
        num_heads: int,
        scale: float,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        vllm_config: "VllmConfig",
    ) -> None:
        super().__init__(
            num_heads=num_heads,
            scale=scale,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            vllm_config=vllm_config,
        )

    def _flash_attn_varlen_diff_headdims(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: float | None = None,
        causal: bool = True,
        return_softmax_lse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """使用 PyTorch SDPA 实现变长多头注意力（替代 flash_attn_varlen_func）。

        Args:
            q: shape = [total_q_tokens, num_heads, qk_head_dim]
            k: shape = [total_k_tokens, num_heads, qk_head_dim]
            v: shape = [total_k_tokens, num_heads, v_head_dim]
            cu_seqlens_q: 累积 query 序列长度，shape = [batch_size + 1]
            cu_seqlens_k: 累积 key 序列长度，shape = [batch_size + 1]
            max_seqlen_q: query 最大序列长度
            max_seqlen_k: key 最大序列长度
            softmax_scale: 注意力缩放因子
            causal: 是否使用因果掩码
            return_softmax_lse: 是否返回 log-sum-exp

        Returns:
            注意力输出，shape = [total_q_tokens, num_heads, v_head_dim]。
            若 ``return_softmax_lse=True``，额外返回 lse，shape =
            [num_heads, total_q_tokens]。
        """
        if softmax_scale is None:
            softmax_scale = self.scale

        batch_size = cu_seqlens_q.shape[0] - 1
        total_q_tokens = q.shape[0]
        num_heads = q.shape[1]
        v_head_dim = v.shape[2]
        qk_head_dim = q.shape[2]

        output = torch.zeros(
            total_q_tokens,
            num_heads,
            v_head_dim,
            dtype=q.dtype,
            device=q.device,
        )

        # lse: [num_heads, total_q_tokens]，初始化为 -inf
        lse = (
            torch.full(
                (num_heads, total_q_tokens),
                float("-inf"),
                dtype=torch.float32,
                device=q.device,
            )
            if return_softmax_lse
            else None
        )

        cu_seqlens_q_cpu = cu_seqlens_q.cpu().numpy()
        cu_seqlens_k_cpu = cu_seqlens_k.cpu().numpy()

        for i in range(batch_size):
            q_start = int(cu_seqlens_q_cpu[i])
            q_end = int(cu_seqlens_q_cpu[i + 1])
            k_start = int(cu_seqlens_k_cpu[i])
            k_end = int(cu_seqlens_k_cpu[i + 1])

            if q_end <= q_start or k_end <= k_start:
                continue

            # q_i: [1, num_heads, seq_q, qk_head_dim]
            q_i = q[q_start:q_end].transpose(0, 1).unsqueeze(0)
            # k_i: [1, num_heads, seq_k, qk_head_dim]
            k_i = k[k_start:k_end].transpose(0, 1).unsqueeze(0)
            # v_i: [1, num_heads, seq_k, v_head_dim]
            v_i = v[k_start:k_end].transpose(0, 1).unsqueeze(0)

            # 如果 q_head_dim != v_head_dim，pad v 到 qk_head_dim
            if qk_head_dim != v_head_dim:
                v_i = F.pad(v_i, [0, qk_head_dim - v_head_dim], value=0.0)

            if return_softmax_lse:
                # 手动算 attention 以同时拿到 LSE
                attn_scores = torch.matmul(q_i, k_i.transpose(-2, -1)) * softmax_scale

                if causal:
                    seq_q = q_end - q_start
                    seq_k = k_end - k_start
                    q_idx = torch.arange(seq_q, device=q.device).unsqueeze(1)
                    k_idx = torch.arange(seq_k, device=q.device).unsqueeze(0)
                    causal_mask = q_idx >= k_idx
                    attn_scores = attn_scores.masked_fill(
                        ~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
                    )

                lse_i = torch.logsumexp(attn_scores, dim=-1)
                assert lse is not None
                lse[:, q_start:q_end] = lse_i.squeeze(0)

                attn_weights = torch.softmax(attn_scores, dim=-1)
                output_i = torch.matmul(attn_weights, v_i)
            else:
                output_i = F.scaled_dot_product_attention(
                    q_i,
                    k_i,
                    v_i,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=causal,
                    scale=softmax_scale,
                )

            # 截到 v_head_dim 并转回 [seq_q, num_heads, v_head_dim]
            output[q_start:q_end] = output_i[0, :, :, :v_head_dim].transpose(0, 1)

        if return_softmax_lse:
            assert lse is not None
            return output, lse
        return output

    def run_prefill_new_tokens(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self._flash_attn_varlen_diff_headdims(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=self._prefill_metadata.query_start_loc,
            cu_seqlens_k=self._prefill_metadata.query_start_loc,
            max_seqlen_q=self._prefill_metadata.max_query_len,
            max_seqlen_k=self._prefill_metadata.max_query_len,
            softmax_scale=self.scale,
            causal=True,
            return_softmax_lse=return_softmax_lse,
        )

    def run_prefill_context_chunk(
        self,
        chunk_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._prefill_metadata.chunked_context is None:
            raise AssertionError("prefill_metadata.chunked_context is None")
        result = self._flash_attn_varlen_diff_headdims(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=self._prefill_metadata.query_start_loc,
            cu_seqlens_k=self._prefill_metadata.chunked_context.cu_seq_lens[chunk_idx],
            max_seqlen_q=self._prefill_metadata.max_query_len,
            max_seqlen_k=self._prefill_metadata.chunked_context.max_seq_lens[chunk_idx],
            softmax_scale=self.scale,
            causal=False,  # context 不加 causal mask
            return_softmax_lse=True,
        )
        assert isinstance(result, tuple)
        return result
