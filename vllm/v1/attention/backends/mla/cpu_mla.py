# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU MLA (Multi-head Latent Attention) 后端实现。

参考 DeepSeek V3 的 MLA 注意力机制，在 CPU 上实现 Prefill（MHA 路径）
和 Decode（MQA 路径）两种计算模式。

MLA 核心思路：
- KV Cache 存储压缩后的潜在向量 kv_c（kv_lora_rank 维）和 k_pe（qk_rope_head_dim 维）
- Prefill（MHA）：将 kv_c 通过 kv_b_proj 上投影为完整的 k/v，再做标准多头注意力
- Decode（MQA）：将 q_nope 通过 W_UK_T 投影到潜在空间，直接与 kv_c 做 MQA 注意力，
  再将输出通过 W_UV 上投影回 v_head_dim 空间
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch
import torch.nn.functional as F

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonDecodeMetadata,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
)
from vllm.v1.attention.backend import (
    AttentionLayer,
    AttentionType,
)

logger = init_logger(__name__)


class CPUMLABackend(MLACommonBackend):
    """CPU 平台的 MLA 注意力后端。"""

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto"]

    @staticmethod
    def get_name() -> str:
        return "CPU_MLA"

    @staticmethod
    def get_impl_cls() -> type[CPUMLAImpl]:
        return CPUMLAImpl

    @staticmethod
    def get_builder_cls() -> type[CPUMLAMetadataBuilder]:
        return CPUMLAMetadataBuilder

    @classmethod
    def get_supported_dtypes(cls) -> list[torch.dtype]:
        return cls.supported_dtypes

    @classmethod
    def get_supported_kv_cache_dtypes(cls) -> list[CacheDType]:
        return cls.supported_kv_cache_dtypes

    @classmethod
    def supports_combination(
        cls,
        head_size: int | None = None,
        dtype: torch.dtype | None = None,
        kv_cache_dtype: CacheDType | None = None,
        block_size: int | None = None,
        use_mla: bool = True,
        has_sink: bool = False,
        use_sparse: bool = False,
        device_capability=None,
        attn_type: str | None = None,
    ) -> str | None:
        if use_sparse:
            return "CPU MLA does not support sparse attention"
        return None


@dataclass
class CPUMLADecodeMetadata(MLACommonDecodeMetadata):
    """CPU MLA Decode 阶段的元数据。"""

    pass


@dataclass
class CPUMLAMetadata(MLACommonMetadata[CPUMLADecodeMetadata]):
    """CPU MLA 的注意力元数据。"""

    pass


class CPUMLAMetadataBuilder(MLACommonMetadataBuilder[CPUMLAMetadata]):
    """CPU MLA 元数据构建器。"""

    def __init__(
        self,
        kv_cache_spec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(
            kv_cache_spec,
            layer_names,
            vllm_config,
            device,
            CPUMLAMetadata,
        )

    def _build_decode(
        self,
        block_table_tensor: torch.Tensor,
        seq_lens_device: torch.Tensor,
        max_seq_len: int,
        query_start_loc_cpu: torch.Tensor,
        query_start_loc_device: torch.Tensor,
        num_decode_tokens: int,
        dcp_tot_seq_lens_device: torch.Tensor | None,
    ) -> CPUMLADecodeMetadata:
        return CPUMLADecodeMetadata(
            block_table=block_table_tensor,
            seq_lens=seq_lens_device,
            dcp_tot_seq_lens=dcp_tot_seq_lens_device,
        )


class CPUMLAImpl(MLACommonImpl[CPUMLAMetadata]):
    """CPU MLA 注意力实现。

    Prefill 阶段（MHA 路径）：
        使用 PyTorch SDPA 实现标准多头注意力，kv_c 通过 kv_b_proj 上投影为完整 k/v。

    Decode 阶段（MQA 路径）：
        将 q_nope 投影到潜在空间（W_UK_T），直接与 kv_c cache 做 MQA 注意力，
        输出再通过 W_UV 上投影回 v_head_dim 空间。
        这等价于 DeepSeek V3 论文中的 "absorption" 技巧，避免了 decode 时的 kv_b_proj。
    """

    # CPU 不支持返回 LSE（log-sum-exp），decode 路径直接返回 None
    can_return_lse_for_decode: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA 专用参数
        **mla_args,
    ) -> None:
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "CPUMLAImpl 仅支持 DECODER 注意力类型，"
                "不支持 encoder self-attention 和 encoder/decoder cross-attention"
            )

        if alibi_slopes is not None:
            raise NotImplementedError("CPUMLAImpl 不支持 alibi_slopes")

        if sliding_window is not None:
            raise NotImplementedError("CPUMLAImpl 不支持 sliding_window")

        if logits_soft_cap is not None:
            raise NotImplementedError("CPUMLAImpl 不支持 logits_soft_cap")

        # MLACommonImpl.__init__ 在 else 分支（FlashAttention 路径）中会检查
        # flash_attn_varlen_func 是否可用，CPU 上不可用会抛出 RuntimeError。
        # 通过临时 patch 绕过这个检查，之后再重新设置为 CPU 实现。
        import vllm.model_executor.layers.attention.mla_attention as _mla_mod

        _orig_fa_func = _mla_mod.flash_attn_varlen_func
        # 提供一个占位函数，使父类 __init__ 不会因 None 检查而报错
        _mla_mod.flash_attn_varlen_func = lambda *args, **kwargs: None

        try:
            super().__init__(
                num_heads,
                head_size,
                scale,
                num_kv_heads,
                alibi_slopes,
                sliding_window,
                kv_cache_dtype,
                logits_soft_cap,
                attn_type,
                kv_sharing_target_layer_name,
                **mla_args,
            )
        finally:
            # 恢复原始值
            _mla_mod.flash_attn_varlen_func = _orig_fa_func

        # 覆盖父类在 __init__ 中设置的 FlashAttention prefill 方法，
        # 改为使用 CPU 的 SDPA 实现
        self._run_prefill_new_tokens = self._cpu_run_prefill_new_tokens
        self._run_prefill_context_chunk = self._cpu_run_prefill_context_chunk
        # CPU 不需要 pad v（SDPA 支持不同的 q/v head dim）
        self._pad_v = False

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
        **kwargs,
    ):
        """使用 PyTorch SDPA 实现变长多头注意力（替代 flash_attn_varlen_func）。

        Args:
            q: shape = [total_tokens, num_heads, qk_head_dim]
            k: shape = [total_tokens, num_heads, qk_head_dim]
            v: shape = [total_tokens, num_heads, v_head_dim]
            cu_seqlens_q: 累积 query 序列长度，shape = [batch_size + 1]
            cu_seqlens_k: 累积 key 序列长度，shape = [batch_size + 1]
            max_seqlen_q: query 最大序列长度
            max_seqlen_k: key 最大序列长度
            softmax_scale: 注意力缩放因子
            causal: 是否使用因果掩码
            return_softmax_lse: 是否返回 log-sum-exp

        Returns:
            注意力输出，shape = [total_tokens, num_heads, v_head_dim]
            如果 return_softmax_lse=True，还返回 lse，shape = [num_heads, total_tokens]
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
            q_start = cu_seqlens_q_cpu[i]
            q_end = cu_seqlens_q_cpu[i + 1]
            k_start = cu_seqlens_k_cpu[i]
            k_end = cu_seqlens_k_cpu[i + 1]

            if q_end <= q_start or k_end <= k_start:
                continue

            # q_i: [1, num_heads, seq_q, qk_head_dim]
            q_i = q[q_start:q_end].transpose(0, 1).unsqueeze(0)
            # k_i: [1, num_heads, seq_k, qk_head_dim]
            k_i = k[k_start:k_end].transpose(0, 1).unsqueeze(0)
            # v_i: [1, num_heads, seq_k, v_head_dim]
            v_i = v[k_start:k_end].transpose(0, 1).unsqueeze(0)

            # 如果 q_head_dim != v_head_dim，需要 pad v
            if qk_head_dim != v_head_dim:
                v_i = F.pad(v_i, [0, qk_head_dim - v_head_dim], value=0.0)

            if return_softmax_lse:
                # 手动计算注意力分数以获取 LSE
                # attn_scores: [1, num_heads, seq_q, seq_k]
                attn_scores = torch.matmul(q_i, k_i.transpose(-2, -1)) * softmax_scale

                if causal:
                    seq_q = q_end - q_start
                    seq_k = k_end - k_start
                    # 因果掩码：query 位置 t 只能看到 key 位置 <= t 的 token
                    # 对于 prefill，query 和 key 共享同一序列，
                    # query[t] 对应序列位置 t，key[s] 对应序列位置 s
                    q_idx = torch.arange(seq_q, device=q.device).unsqueeze(1)
                    k_idx = torch.arange(seq_k, device=q.device).unsqueeze(0)
                    causal_mask = q_idx >= k_idx
                    attn_scores = attn_scores.masked_fill(
                        ~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
                    )

                # lse_i: [1, num_heads, seq_q]
                lse_i = torch.logsumexp(attn_scores, dim=-1)
                # 存入 lse: [num_heads, total_q_tokens]
                lse[:, q_start:q_end] = lse_i.squeeze(0)

                # 计算注意力输出
                attn_weights = torch.softmax(attn_scores, dim=-1)
                output_i = torch.matmul(attn_weights, v_i)
            else:
                # 使用 SDPA 计算注意力（更高效）
                output_i = F.scaled_dot_product_attention(
                    q_i,
                    k_i,
                    v_i,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=causal,
                    scale=softmax_scale,
                )

            # 截取 v_head_dim 维度，转换回 [seq_q, num_heads, v_head_dim]
            output[q_start:q_end] = output_i[0, :, :, :v_head_dim].transpose(0, 1)

        if return_softmax_lse:
            return output, lse

        return output

    def _cpu_run_prefill_new_tokens(
        self,
        prefill,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool,
    ):
        """CPU Prefill 新 token 的注意力计算（因果注意力）。"""
        return self._flash_attn_varlen_diff_headdims(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=prefill.query_start_loc,
            cu_seqlens_k=prefill.query_start_loc,
            max_seqlen_q=prefill.max_query_len,
            max_seqlen_k=prefill.max_query_len,
            softmax_scale=self.scale,
            causal=True,
            return_softmax_lse=return_softmax_lse,
        )

    def _cpu_run_prefill_context_chunk(
        self,
        prefill,
        chunk_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        """CPU Prefill 上下文 chunk 的注意力计算（非因果注意力）。"""
        if prefill.chunked_context is None:
            raise AssertionError("prefill.chunked_context is None")
        return self._flash_attn_varlen_diff_headdims(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=prefill.query_start_loc,
            cu_seqlens_k=prefill.chunked_context.cu_seq_lens[chunk_idx],
            max_seqlen_q=prefill.max_query_len,
            max_seqlen_k=prefill.chunked_context.max_seq_lens[chunk_idx],
            softmax_scale=self.scale,
            causal=False,
            return_softmax_lse=True,
        )

    @staticmethod
    def _cpu_merge_attn_states(
        prefix_output: torch.Tensor,
        prefix_lse: torch.Tensor,
        suffix_output: torch.Tensor,
        suffix_lse: torch.Tensor,
    ) -> torch.Tensor:
        """纯 PyTorch 实现 LSE merge，合并两段注意力输出。

        数学公式（参考 https://www.arxiv.org/pdf/2501.01005 Section 2.2）：
            p_scale = exp(p_lse - max_lse) / (
                exp(p_lse - max_lse) + exp(s_lse - max_lse)
            )
            s_scale = 1 - p_scale
            output = p_scale * prefix_output + s_scale * suffix_output

        Args:
            prefix_output: shape = [num_tokens, num_heads, head_size]
            prefix_lse: shape = [num_heads, num_tokens]
            suffix_output: shape = [num_tokens, num_heads, head_size]
            suffix_lse: shape = [num_heads, num_tokens]

        Returns:
            merged output, shape = [num_tokens, num_heads, head_size]
        """
        # prefix_lse / suffix_lse: [num_heads, num_tokens]
        # 转置为 [num_tokens, num_heads, 1] 方便广播
        p_lse = prefix_lse.transpose(0, 1).unsqueeze(-1).float()
        s_lse = suffix_lse.transpose(0, 1).unsqueeze(-1).float()

        # 数值稳定：减去 max
        max_lse = torch.maximum(p_lse, s_lse)
        p_se = torch.exp(p_lse - max_lse)
        s_se = torch.exp(s_lse - max_lse)
        out_se = p_se + s_se

        p_scale = p_se / out_se
        s_scale = s_se / out_se

        merged = (p_scale * prefix_output.float() + s_scale * suffix_output.float()).to(
            prefix_output.dtype
        )
        return merged

    def _cpu_compute_prefill_context(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: CPUMLAMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """从 KV cache 中 gather context token，计算 prefill context 注意力。

        纯 PyTorch 实现，替代父类中依赖 CUDA 算子的 _compute_prefill_context。

        Args:
            q: shape = [num_prefill_tokens, num_heads, qk_head_dim]
            kv_c_and_k_pe_cache: shape =
                [num_blocks, block_size, kv_lora_rank + qk_rope_head_dim]
            attn_metadata: 注意力元数据

        Returns:
            (context_output, context_lse):
                context_output: shape = [num_prefill_tokens, num_heads, v_head_dim]
                context_lse: shape = [num_heads, num_prefill_tokens]
        """
        assert attn_metadata.prefill is not None
        prefill_metadata = attn_metadata.prefill
        if prefill_metadata.chunked_context is None:
            raise AssertionError("prefill_metadata.chunked_context is None")

        block_size = kv_c_and_k_pe_cache.shape[1]
        block_table = prefill_metadata.block_table  # [num_prefills, max_blocks]
        num_prefills = block_table.shape[0]

        # 从 chunked_context 中获取每个序列的 context 长度
        # cu_seq_lens[0]: [num_prefills + 1]，
        # 第一个 chunk 的累积长度即为各序列 context 长度
        cu_seq_lens_first = prefill_metadata.chunked_context.cu_seq_lens[0]
        context_lens = (
            cu_seq_lens_first[1:] - cu_seq_lens_first[:-1]
        ).cpu()  # [num_prefills]

        # 逐序列 gather context token，拼接后做批量注意力
        # 构造 cu_seqlens_k（context 长度的累积和）
        cu_seqlens_k = torch.zeros(num_prefills + 1, dtype=torch.int32, device=q.device)
        for i in range(num_prefills):
            cu_seqlens_k[i + 1] = cu_seqlens_k[i] + int(context_lens[i])

        total_context_tokens = int(cu_seqlens_k[-1].item())
        if total_context_tokens == 0:
            # 没有 context token，返回零输出和 -inf LSE
            num_tokens = q.shape[0]
            num_heads = q.shape[1]
            context_output = torch.zeros(
                num_tokens,
                num_heads,
                self.v_head_dim,
                dtype=q.dtype,
                device=q.device,
            )
            context_lse = torch.full(
                (num_heads, num_tokens),
                float("-inf"),
                dtype=torch.float32,
                device=q.device,
            )
            return context_output, context_lse

        # gather 所有 context token 的 kv_c 和 k_pe
        gathered_kv = torch.empty(
            total_context_tokens,
            self.kv_lora_rank + self.qk_rope_head_dim,
            dtype=kv_c_and_k_pe_cache.dtype,
            device=q.device,
        )
        for i in range(num_prefills):
            ctx_len = int(context_lens[i])
            if ctx_len == 0:
                continue
            dst_start = int(cu_seqlens_k[i].item())
            gathered_kv[dst_start : dst_start + ctx_len] = self._gather_kv_cache(
                kv_c_and_k_pe_cache, block_table[i], ctx_len, block_size
            )

        # 分离 kv_c 和 k_pe
        kv_c_ctx = gathered_kv[:, : self.kv_lora_rank]
        k_pe_ctx = gathered_kv[:, self.kv_lora_rank :].unsqueeze(1)

        # 通过 kv_b_proj 上投影
        kv_nope = self._linear(self.kv_b_proj, kv_c_ctx).view(
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope_ctx, v_ctx = kv_nope.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        k_ctx = self._concat_k_nope_k_pe(k_nope_ctx, k_pe_ctx)

        # 使用 query 的 cu_seqlens（prefill query 的累积长度）
        cu_seqlens_q = prefill_metadata.query_start_loc

        # 计算 context 注意力（非因果）
        context_output, context_lse = self._flash_attn_varlen_diff_headdims(
            q=q,
            k=k_ctx,
            v=v_ctx,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=prefill_metadata.max_query_len,
            max_seqlen_k=int(context_lens.max().item()),
            softmax_scale=self.scale,
            causal=False,
            return_softmax_lse=True,
        )
        return context_output, context_lse

    def forward_mha(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: CPUMLAMetadata,
        k_scale: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Prefill 阶段的 MHA 前向计算。

        将 kv_c_normed 通过 kv_b_proj 上投影为完整的 k/v，
        再与 q 做标准多头注意力。有历史 context 时，从 KV cache 中
        gather context token 并用 LSE merge 合并两段注意力输出。

        Args:
            q: shape = [num_prefill_tokens, num_heads, qk_head_dim]
            kv_c_normed: shape = [num_prefill_tokens, kv_lora_rank]
            k_pe: shape = [num_prefill_tokens, 1, qk_rope_head_dim]
            kv_c_and_k_pe_cache: KV cache 张量
            attn_metadata: 注意力元数据
            k_scale: KV cache 缩放因子
            output: 输出张量，shape = [num_prefill_tokens, num_heads * v_head_dim]
        """
        if attn_metadata.prefill is None:
            raise AssertionError("attn_metadata.prefill is None")

        prefill_metadata = attn_metadata.prefill
        has_context = prefill_metadata.chunked_context is not None

        # 通过 kv_b_proj 将压缩的 kv_c 上投影为完整的 k_nope 和 v
        # kv_nope: [num_tokens, num_heads, qk_nope_head_dim + v_head_dim]
        kv_nope = self._linear(self.kv_b_proj, kv_c_normed).view(
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # 拼接 k_nope 和 k_pe 得到完整的 k
        # k: [num_tokens, num_heads, qk_head_dim]
        k = self._concat_k_nope_k_pe(k_nope, k_pe)

        output_prefill = self._run_prefill_new_tokens(
            prefill=prefill_metadata,
            q=q,
            k=k,
            v=v,
            return_softmax_lse=has_context,
        )

        if has_context:
            # 有历史 context：从 KV cache gather context token，计算 context 注意力，
            # 再用 LSE merge 合并 suffix（新 token）和 prefix（context）的输出
            suffix_output, suffix_lse = output_prefill
            suffix_output = suffix_output[..., : self.v_head_dim]

            context_output, context_lse = self._cpu_compute_prefill_context(
                q, kv_c_and_k_pe_cache, attn_metadata
            )

            # LSE merge：合并 context（prefix）和新 token（suffix）的注意力输出
            merged = self._cpu_merge_attn_states(
                prefix_output=context_output,
                prefix_lse=context_lse,
                suffix_output=suffix_output,
                suffix_lse=suffix_lse,
            )
            output.copy_(merged.flatten(start_dim=-2))
        else:
            output_prefill = output_prefill[..., : v.shape[-1]].flatten(start_dim=-2)
            output.copy_(output_prefill)

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: CPUMLAMetadata,
        layer: AttentionLayer | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Decode 阶段的 MQA 前向计算（"absorption" 技巧）。

        利用 MLA 的 absorption 特性，将 q_nope 投影到潜在空间后，
        直接与 kv_c cache 做 MQA 注意力，避免了 decode 时的 kv_b_proj 展开。

        计算流程：
            1. q_l = q_nope @ W_UK_T  (投影到潜在空间，shape: [B, N, L])
            2. 从 KV cache 中读取 kv_c 和 k_pe
            3. 计算注意力分数：
               attn_score = q_l @ kv_c.T + q_pe @ k_pe.T  (在潜在空间做 MQA)
            4. 输出 = softmax(attn_score) @ kv_c  (shape: [B, N, L])
            5. 上层调用 _v_up_proj 将输出从 L 维投影回 v_head_dim 维

        Args:
            q: 可以是 (q_nope_projected, q_pe) 的元组，或拼接后的张量
               - q_nope_projected: [B, N, kv_lora_rank]（已经过 W_UK_T 投影）
               - q_pe: [B, N, qk_rope_head_dim]
            kv_c_and_k_pe_cache: KV cache，shape = [num_blocks, block_size, head_size]
               其中 head_size = kv_lora_rank + qk_rope_head_dim
            attn_metadata: 注意力元数据
            layer: 注意力层对象（可选）

        Returns:
            (output, lse): output shape = [B, N, kv_lora_rank]，lse = None
        """
        if kv_c_and_k_pe_cache.numel() <= 0:
            raise AssertionError("kv_c_and_k_pe_cache is empty")
        if attn_metadata.decode is None:
            raise AssertionError("attn_metadata.decode is None")

        decode_metadata = attn_metadata.decode

        # 解析 q
        if isinstance(q, tuple):
            q_nope_proj, q_pe = q
        else:
            # q 是拼接后的张量 [B, N, kv_lora_rank + qk_rope_head_dim]
            q_nope_proj, q_pe = torch.split(
                q, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )

        batch_size = q_nope_proj.shape[0]
        num_heads = q_nope_proj.shape[1]

        # 从 KV cache 中收集每个请求的 kv_c 和 k_pe
        # kv_c_and_k_pe_cache: [num_blocks, block_size, kv_lora_rank + qk_rope_head_dim]
        block_table = decode_metadata.block_table  # [B, max_blocks]
        seq_lens = decode_metadata.seq_lens  # [B]
        seq_lens_cpu = seq_lens.cpu()

        # 输出张量：[B, N, kv_lora_rank]
        output = torch.zeros(
            batch_size,
            num_heads,
            self.kv_lora_rank,
            dtype=q_nope_proj.dtype,
            device=q_nope_proj.device,
        )

        block_size = kv_c_and_k_pe_cache.shape[1]

        # 优化 2：将 q_nope_proj 和 q_pe 沿特征维拼接为 [B, N, L + R]，
        # KV cache 本就是 [L + R] 拼接存储，这样可以一次 matmul 完成
        # q_l @ kv_c.T + q_pe @ k_pe.T = q_cat @ kv_cat.T
        # q_cat: [B, N, L + R]
        q_cat = torch.cat([q_nope_proj, q_pe], dim=-1)

        for b in range(batch_size):
            seq_len = int(seq_lens_cpu[b].item())
            if seq_len == 0:
                continue

            # 优化 1：向量化 _gather_kv_cache，一次 index_select 替代逐 block for 循环
            # gathered_kv: [seq_len, kv_lora_rank + qk_rope_head_dim]
            gathered_kv = self._gather_kv_cache(
                kv_c_and_k_pe_cache,
                block_table[b],
                seq_len,
                block_size,
            )

            # 合并的注意力分数 gemm：[N, L + R] @ [L + R, seq_len] -> [N, seq_len]
            attn_scores = torch.mm(q_cat[b], gathered_kv.t()) * self.scale

            # softmax
            attn_weights = F.softmax(attn_scores, dim=-1)

            # 加权求和：[N, seq_len] @ [seq_len, kv_lora_rank] -> [N, kv_lora_rank]
            # 只取 kv_c 部分（前 kv_lora_rank 列），避免引入 k_pe
            output[b] = torch.mm(attn_weights, gathered_kv[:, : self.kv_lora_rank])

        return output, None

    def _gather_kv_cache(
        self,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_len: int,
        block_size: int,
    ) -> torch.Tensor:
        """从分页 KV cache 中收集指定序列的所有 token。

        优化：使用 index_select 一次性 gather 所有需要的 block，避免逐 block
        的 Python for 循环和 .item() 同步开销。对长序列（num_blocks 较多）
        提升明显。

        Args:
            kv_cache: shape = [num_blocks, block_size, head_size]
            block_table: 该请求的 block 索引，shape = [max_blocks]
            seq_len: 序列长度
            block_size: 每个 block 的 token 数

        Returns:
            gathered: shape = [seq_len, head_size]，连续存储
        """
        head_size = kv_cache.shape[2]
        # 向上取整，覆盖 seq_len 所需的全部 block
        num_blocks = (seq_len + block_size - 1) // block_size

        # 一次 index_select：[num_blocks, block_size, head_size]
        block_ids = block_table[:num_blocks]
        gathered = kv_cache.index_select(0, block_ids)

        # reshape 为 [num_blocks * block_size, head_size] 后截断到 seq_len
        # 返回 contiguous 张量，便于后续 gemm 高效访存
        return gathered.reshape(num_blocks * block_size, head_size)[:seq_len]

    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        """覆盖父类方法，使用纯 PyTorch 实现替代 CUDA 算子 concat_and_cache_mla。"""
        if kv_cache.numel() == 0:
            return
        self._write_kv_cache_cpu(
            kv_c_normed,
            k_pe.squeeze(1),
            kv_cache,
            slot_mapping.flatten(),
        )

    @staticmethod
    def _write_kv_cache_cpu(
        kv_c: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """将 kv_c 和 k_pe 拼接后按 slot_mapping 写入分页 KV cache。

        CPU 纯 PyTorch 实现，替代 ops.concat_and_cache_mla。

        Args:
            kv_c: shape = [num_tokens, kv_lora_rank]
            k_pe: shape = [num_tokens, qk_rope_head_dim]
            kv_cache: shape = [num_blocks, block_size, head_size]
            slot_mapping: shape = [num_tokens]，每个 token 对应的 flat slot 索引
        """
        kv_combined = torch.cat([kv_c, k_pe], dim=-1)
        block_size = kv_cache.shape[1]
        for i in range(slot_mapping.shape[0]):
            slot = int(slot_mapping[i].item())
            if slot < 0:
                continue
            kv_cache[slot // block_size, slot % block_size] = kv_combined[i]

    @staticmethod
    def _linear(layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        """通过 `cpu_linear` 执行 Linear，走 oneDNN / sgl 的 CPU 加速路径。

        `dispatch_cpu_unquantized_gemm` 会为每个 Linear 构建加速用的
        `cpu_linear`（oneDNN / sgl kernel 封装）。直接走 `cpu_linear` 可以
        绕过 `Linear.forward` 内部的 dispatch 逻辑，减少一层间接调用开销。

        :param layer: Linear 层（含 cpu_linear 属性）
        :param x: 输入张量
        :return: 线性变换输出
        """
        bias = getattr(layer, "bias", None)
        if getattr(layer, "skip_bias_add", False):
            bias = None
        return layer.cpu_linear(x, layer.weight, bias)
