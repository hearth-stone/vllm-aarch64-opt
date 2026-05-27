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

v0.21 适配说明：
- v0.21 引入了 ``MLAPrefillBackend`` 抽象，prefill 计算被解耦到独立后端类。
  本模块依赖兄弟模块 ``mla.prefill.cpu.CPUPrefillBackend`` 提供 PyTorch SDPA
  varlen 实现，因此 ``CPUMLAImpl.__init__`` 不再需要 monkey-patch
  ``flash_attn_varlen_func`` 模块符号。
- 父类 ``MLACommonImpl._compute_prefill_context`` 依赖 CUDA 算子
  ``ops.gather_and_maybe_dequant_cache`` 从分页 cache gather context token，
  本模块使用纯 PyTorch ``index_select`` 路径覆盖。
- 父类 ``forward_mha`` 中合并 prefix/suffix 输出的 ``merge_attn_states`` 在
  非 CUDA 平台会 fallback 到 Triton（CPU 上不可用），因此 ``forward_mha``
  完整覆盖，使用纯 PyTorch ``_cpu_merge_attn_states``。
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
    def get_impl_cls() -> type["CPUMLAImpl"]:
        return CPUMLAImpl

    @staticmethod
    def get_builder_cls() -> type["CPUMLAMetadataBuilder"]:
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
        ``forward_mha`` 调用 ``prefill_metadata.prefill_backend.run_prefill_new_tokens``
        （由 :class:`CPUPrefillBackend` 提供 PyTorch SDPA 实现）；有历史 context
        时再调用 ``_compute_prefill_context``（本类覆盖）从 KV cache gather
        context token，最后通过 ``_cpu_merge_attn_states`` 用 LSE merge
        合并两段输出。

    Decode 阶段（MQA 路径）：
        利用 MLA 的 absorption 技巧，将 q_nope 投影到潜在空间（W_UK_T）后，
        直接与 kv_c cache 做 MQA 注意力，输出再通过 W_UV 上投影回 v_head_dim
        空间，避免了 decode 时的 kv_b_proj 展开。
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

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        """在权重被清空前提取 kv_b_proj 权重构建 W_UK_T 和 W_UV。

        CPU 路径下，``dispatch_cpu_unquantized_gemm`` 可能将 ``kv_b_proj.weight``
        打包进 ``cpu_linear`` 并将 ``layer.weight`` 替换为空 tensor（例如 sgl
        kernel 路径）。该方法在 ``layer.weight`` 已被清空时通过对 ``cpu_linear``
        喂入单位矩阵恢复原始权重，再构建 absorption 路径所需的 W_UK_T / W_UV。

        v0.21 中的 layer 级 ``MLAAttention.process_weights_after_loading`` 也
        会构建 W_UK_T / W_UV 但仅在权重未被清空时正常工作。本方法主要服务于
        独立测试场景（``tests/v1/attention/test_cpu_mla*.py`` 直接构造
        ``CPUMLAImpl`` 时手动调用本方法）。
        """
        kv_b_weight = self.kv_b_proj.weight
        if kv_b_weight.numel() == 0:
            # weight 已被清空，通过 cpu_linear 用单位矩阵探测恢复原始权重
            eye = torch.eye(
                self.kv_lora_rank,
                dtype=act_dtype,
                device=kv_b_weight.device,
            )
            kv_b_proj_weight = self.kv_b_proj.quant_method.apply(
                self.kv_b_proj, eye, bias=None
            ).to(act_dtype)
        else:
            kv_b_proj_weight = kv_b_weight.to(act_dtype).T

        if kv_b_proj_weight.shape != (
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
        ):
            raise AssertionError(
                f"{kv_b_proj_weight.shape=}, "
                f"{self.kv_lora_rank=}, "
                f"{self.num_heads=}, "
                f"{self.qk_nope_head_dim=}, "
                f"{self.v_head_dim=}"
            )

        kv_b_proj_weight = kv_b_proj_weight.view(
            self.kv_lora_rank,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        w_uk, w_uv = kv_b_proj_weight.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        # W_UK_T: [num_heads, qk_nope_head_dim, kv_lora_rank]
        self.W_UK_T = w_uk.permute(1, 2, 0).contiguous()
        # W_UV: [num_heads, kv_lora_rank, v_head_dim]
        self.W_UV = w_uv.transpose(0, 1).contiguous()

        logger.debug(
            "process_weights_after_loading: W_UK_T=%s, W_UV=%s",
            tuple(self.W_UK_T.shape),
            tuple(self.W_UV.shape),
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

        merged = (
            p_scale * prefix_output.float() + s_scale * suffix_output.float()
        ).to(prefix_output.dtype)
        return merged

    def _compute_prefill_context(  # type: ignore[override]
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: CPUMLAMetadata,
        k_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """从 KV cache 中 gather context token，计算 prefill context 注意力。

        纯 PyTorch 实现，替代父类中依赖 CUDA 算子
        ``ops.gather_and_maybe_dequant_cache`` 的 ``_compute_prefill_context``。

        Args:
            q: shape = [num_prefill_tokens, num_heads, qk_head_dim]
            kv_c_and_k_pe_cache: shape =
                [num_blocks, block_size, kv_lora_rank + qk_rope_head_dim]
            attn_metadata: 注意力元数据
            k_scale: KV cache 缩放因子（CPU 路径仅支持未量化的 KV cache，本参数
                被忽略，保留参数仅为与父类签名一致）

        Returns:
            (context_output, context_lse):
                context_output: shape = [num_prefill_tokens, num_heads, v_head_dim]
                context_lse: shape = [num_heads, num_prefill_tokens]
        """
        del k_scale  # 未量化 KV cache 不需要 dequant scale
        if attn_metadata.prefill is None:
            raise AssertionError("attn_metadata.prefill is None")
        prefill_metadata = attn_metadata.prefill
        if prefill_metadata.chunked_context is None:
            raise AssertionError("prefill_metadata.chunked_context is None")
        if prefill_metadata.prefill_backend is None:
            raise AssertionError("prefill_metadata.prefill_backend is None")

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

        # 直接调用 CPU prefill backend 的 varlen 实现，传入手动 gather 出来的
        # context cu_seqlens_k（与父类委托给 ``run_prefill_context_chunk`` 的
        # 区别在于：父类的 chunk_idx 路径用的是 ``chunked_context.cu_seq_lens``，
        # 这里我们重算了一份只覆盖实际 context 的版本）。
        prefill_backend = prefill_metadata.prefill_backend
        # 使用 backend 的 _flash_attn_varlen_diff_headdims 直接传 cu_seqlens
        context_output, context_lse = prefill_backend._flash_attn_varlen_diff_headdims(
            q=q,
            k=k_ctx,
            v=v_ctx,
            cu_seqlens_q=prefill_metadata.query_start_loc,
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

        将 kv_c_normed 通过 kv_b_proj 上投影为完整的 k/v，再与 q 做标准多头
        注意力（委托给 ``prefill_backend.run_prefill_new_tokens``）。
        有历史 context 时，调用 ``_compute_prefill_context`` 从 KV cache
        gather context token 并用 ``_cpu_merge_attn_states`` LSE merge
        合并两段注意力输出。

        Args:
            q: shape = [num_prefill_tokens, num_heads, qk_head_dim]
            kv_c_normed: shape = [num_prefill_tokens, kv_lora_rank]
            k_pe: shape = [num_prefill_tokens, 1, qk_rope_head_dim]
            kv_c_and_k_pe_cache: KV cache 张量
            attn_metadata: 注意力元数据
            k_scale: KV cache 缩放因子（CPU 路径忽略）
            output: 输出张量，shape = [num_prefill_tokens, num_heads * v_head_dim]
        """
        if attn_metadata.prefill is None:
            raise AssertionError("attn_metadata.prefill is None")

        prefill_metadata = attn_metadata.prefill
        if prefill_metadata.prefill_backend is None:
            raise AssertionError("prefill_metadata.prefill_backend is None")

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

        output_prefill = prefill_metadata.prefill_backend.run_prefill_new_tokens(
            q=q,
            k=k,
            v=v,
            return_softmax_lse=has_context,
        )

        if has_context:
            # 有历史 context：从 KV cache gather context token，计算 context 注意力，
            # 再用 LSE merge 合并 suffix（新 token）和 prefix（context）的输出
            assert isinstance(output_prefill, tuple)
            suffix_output, suffix_lse = output_prefill
            suffix_output = suffix_output[..., : self.v_head_dim]

            context_output, context_lse = self._compute_prefill_context(
                q, kv_c_and_k_pe_cache, attn_metadata, k_scale
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
            assert isinstance(output_prefill, torch.Tensor)
            output_no_lse = output_prefill[..., : v.shape[-1]].flatten(start_dim=-2)
            output.copy_(output_no_lse)

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

    @staticmethod
    def _gather_kv_cache(
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

        CPU 纯 PyTorch 实现，替代 ``ops.concat_and_cache_mla``。

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
        """通过 ``cpu_linear`` 执行 Linear，走 oneDNN / sgl 的 CPU 加速路径。

        ``dispatch_cpu_unquantized_gemm`` 会为每个 Linear 构建加速用的
        ``cpu_linear``（oneDNN / sgl kernel 封装）。直接走 ``cpu_linear``
        可以绕过 ``Linear.forward`` 内部的 dispatch 逻辑，减少一层间接调用
        开销；同时也支持 ``layer.weight`` 已被 dispatch 阶段清空、原始
        权重被打包进 ``cpu_linear`` 的情形。

        若 layer 没有 ``cpu_linear`` 属性（例如测试中用的 ``_MockKvBProj``），
        则回退为标准 ``torch.nn.functional.linear`` 调用，行为与
        ``Linear.forward`` 一致。

        :param layer: Linear 层（可能含 ``cpu_linear`` 属性）
        :param x: 输入张量
        :return: 线性变换输出
        """
        bias = getattr(layer, "bias", None)
        if getattr(layer, "skip_bias_add", False):
            bias = None
        cpu_linear = getattr(layer, "cpu_linear", None)
        if cpu_linear is not None:
            return cpu_linear(x, layer.weight, bias)
        # Fallback: 直接调用 layer（兼容测试 mock 与未走 cpu dispatch 的环境）
        out = layer(x)
        if isinstance(out, tuple):
            out = out[0]
        return out
