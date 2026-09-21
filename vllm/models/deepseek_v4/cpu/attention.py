# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 sparse MLA for CPU.

The class follows the v0.28 attention data flow, but every fallback in this
module is implemented with regular Torch CPU operators.  Fused Arm operators
are prepared lazily after checkpoint loading and never import a GPU backend.
"""

import os
from typing import Any, cast

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.utils import extract_layer_index
from vllm.model_executor.utils import replace_parameter, set_weight_attrs
from vllm.models.deepseek_v4.common.rope import build_deepseek_v4_rope
from vllm.utils.cpu_resource_utils import parse_id_list
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)
from vllm.v1.utils import record_function_or_nullcontext

from .ops import (
    apply_rope_tail,
    compress_and_store,
    gather_paged_cache,
    rms_norm,
    save_compressor_states,
    sparse_mla_reference,
    write_paged_cache,
)
from .sparse_mla import (
    DeepseekV4CPUCompressorBackend,
    DeepseekV4CPUCompressorMetadata,
    DeepseekV4CPUSparseMLABackend,
    DeepseekV4CPUSparseMLAMetadata,
    DeepseekV4CPUSWABackend,
    DeepseekV4CPUSWAMetadata,
)

logger = init_logger(__name__)


def _linear(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    output = module(x)
    return output[0] if isinstance(output, tuple) else output


def _compressed_prefill_ranges(
    positions: torch.Tensor,
    req_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    compress_ratio: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build fused indexer ranges without per-token Python synchronization."""
    compressed_lens = torch.div(seq_lens, compress_ratio, rounding_mode="floor").to(
        torch.int32
    )
    cu_seq_lens = torch.cat((compressed_lens.new_zeros(1), compressed_lens.cumsum(0)))
    starts = cu_seq_lens[req_ids.to(torch.long)]
    ends = starts + torch.div(
        positions.to(torch.int64) + 1,
        compress_ratio,
        rounding_mode="floor",
    ).to(torch.int32)
    return cu_seq_lens, starts, ends


def _mark_joint_weight(module: nn.Module) -> None:
    module._cpu_keep_raw_weight = True


def _cpu_core_ids() -> tuple[int, ...]:
    binding = envs.VLLM_CPU_OMP_THREADS_BIND
    rank = get_tensor_model_parallel_rank()
    if binding not in ("auto", "nobind"):
        rank_bindings = binding.split("|")
        if rank < len(rank_bindings):
            ids = tuple(parse_id_list(rank_bindings[rank]))
            if ids:
                return ids
    if hasattr(os, "sched_getaffinity"):
        return tuple(sorted(os.sched_getaffinity(0)))
    return tuple(range(torch.get_num_threads()))


class _CPUCacheLayer(nn.Module, AttentionLayerBase):
    def __init__(
        self,
        *,
        head_dim: int,
        dtype: torch.dtype,
        block_size: int,
        prefix: str,
        backend_cls: type[AttentionBackend],
        compress_ratio: int = 1,
        sliding_window: int | None = None,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.dtype = dtype
        self.block_size = block_size
        self.prefix = prefix
        self.backend_cls = backend_cls
        self.compress_ratio = compress_ratio
        self.sliding_window = sliding_window
        self.kv_cache = torch.tensor([])
        context = get_current_vllm_config().compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        context[prefix] = self

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.backend_cls

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        if self.sliding_window is not None:
            return SlidingWindowMLASpec(
                block_size=self.block_size,
                num_kv_heads=1,
                head_size=self.head_dim,
                dtype=self.dtype,
                sliding_window=self.sliding_window,
                alignment=max(1, self.head_dim * self.dtype.itemsize),
            )
        return MLAAttentionSpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            compress_ratio=self.compress_ratio,
            alignment=max(1, self.head_dim * self.dtype.itemsize),
            model_version="deepseek_v4",
        )

    def forward(self): ...


class DeepseekV4CPUCompressor(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        compress_ratio: int,
        hidden_size: int,
        head_dim: int,
        prefix: str,
        output_cache_prefix: str,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.overlap = compress_ratio == 4
        self.coff = 2 if self.overlap else 1
        self.prefix = prefix
        self.output_cache_prefix = output_cache_prefix
        self.norm_eps = config.rms_norm_eps

        self.ape = nn.Parameter(
            torch.empty(compress_ratio, self.coff * head_dim, dtype=torch.float32),
            requires_grad=False,
        )
        self.fused_wkv_wgate = MergedColumnParallelLinear(
            hidden_size,
            [self.coff * head_dim, self.coff * head_dim],
            bias=False,
            return_bias=False,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.fused_wkv_wgate",
        )
        _mark_joint_weight(self.fused_wkv_wgate)
        self.norm = RMSNorm(head_dim, self.norm_eps)
        state_block_size = 4 if compress_ratio == 4 else 8
        self.state_cache = _CPUCacheLayer(
            head_dim=2 * self.coff * head_dim,
            dtype=torch.float32,
            block_size=state_block_size,
            prefix=f"{prefix}.state_cache",
            backend_cls=DeepseekV4CPUCompressorBackend,
            sliding_window=self.coff * compress_ratio,
        )
        self._static_forward_context = (
            vllm_config.compilation_config.static_forward_context
        )

    def project(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return _linear(self.fused_wkv_wgate, hidden_states).float()

    def forward(
        self,
        projected_kv_score: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> None:
        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict):
            return
        state_metadata = cast(
            DeepseekV4CPUCompressorMetadata,
            metadata[self.state_cache.prefix],
        )
        output_metadata = cast(
            DeepseekV4CPUSparseMLAMetadata,
            metadata[self.output_cache_prefix],
        )
        kv, score = projected_kv_score.split(
            [self.coff * self.head_dim, self.coff * self.head_dim], dim=-1
        )
        save_compressor_states(
            kv,
            score,
            self.ape,
            positions,
            self.state_cache.kv_cache,
            state_metadata.slot_mapping,
            self.compress_ratio,
        )
        output_layer = self._static_forward_context[self.output_cache_prefix]
        compress_and_store(
            state_cache=self.state_cache.kv_cache,
            state_metadata=state_metadata,
            output_cache=output_layer.kv_cache,
            output_metadata=output_metadata,
            positions=positions,
            norm_weight=self.norm.weight,
            norm_eps=self.norm_eps,
            cos_sin_cache=rotary_emb.cos_sin_cache,
            head_dim=self.head_dim,
            rope_dim=self.rope_dim,
            compress_ratio=self.compress_ratio,
            overlap=self.overlap,
        )


class DeepseekV4CPUIndexer(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        hidden_size: int,
        q_lora_rank: int,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig,
        compress_ratio: int,
        prefix: str,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.prefix = prefix
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.topk = config.index_topk
        self.compress_ratio = compress_ratio
        self.softmax_scale = self.head_dim**-0.5
        self.head_scale = self.n_heads**-0.5

        self.wq_b = ReplicatedLinear(
            q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        self.weights_proj = ReplicatedLinear(
            hidden_size,
            self.n_heads,
            bias=False,
            return_bias=False,
            quant_config=None,
            prefix=f"{prefix}.weights_proj",
        )
        _mark_joint_weight(self.wq_b)
        _mark_joint_weight(self.weights_proj)
        self.k_cache = _CPUCacheLayer(
            head_dim=self.head_dim,
            dtype=torch.bfloat16,
            block_size=cache_config.block_size,
            prefix=f"{prefix}.k_cache",
            backend_cls=DeepseekV4CPUSparseMLABackend,
            compress_ratio=compress_ratio,
        )
        self.compressor = DeepseekV4CPUCompressor(
            vllm_config=vllm_config,
            compress_ratio=compress_ratio,
            hidden_size=hidden_size,
            head_dim=self.head_dim,
            prefix=f"{prefix}.compressor",
            output_cache_prefix=self.k_cache.prefix,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        projected_kv_score: torch.Tensor,
        projected_weights: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> torch.Tensor:
        self.compressor(projected_kv_score, positions, rotary_emb)
        q = _linear(self.wq_b, qr).reshape(-1, self.n_heads, self.head_dim)
        q = apply_rope_tail(q, positions, rotary_emb.cos_sin_cache, self.rope_dim)
        weights = projected_weights.float() * self.softmax_scale * self.head_scale

        metadata_dict = get_forward_context().attn_metadata
        if not isinstance(metadata_dict, dict):
            return torch.full(
                (q.shape[0], self.topk),
                -1,
                dtype=torch.int32,
                device=q.device,
            )
        metadata = cast(
            DeepseekV4CPUSparseMLAMetadata,
            metadata_dict[self.k_cache.prefix],
        )
        output = torch.full(
            (q.shape[0], self.topk),
            -1,
            dtype=torch.int32,
            device=q.device,
        )
        for token in range(q.shape[0]):
            count = int((int(positions[token].item()) + 1) // self.compress_ratio)
            if count <= 0:
                continue
            req = int(metadata.req_id_per_token[token].item())
            logical = torch.arange(count, device=q.device)
            reqs = torch.full_like(logical, req)
            block_numbers = metadata.block_table[
                reqs, logical // metadata.storage_block_size
            ].to(torch.long)
            slots = (
                block_numbers * metadata.storage_block_size
                + logical % metadata.storage_block_size
            )
            keys = gather_paged_cache(self.k_cache.kv_cache, slots).float()
            scores = torch.zeros(count, dtype=torch.float32, device=q.device)
            for head in range(self.n_heads):
                scores.add_(
                    torch.relu(keys @ q[token, head].float()) * weights[token, head]
                )
            take = min(self.topk, count)
            output[token, :take] = torch.topk(scores, take, sorted=False).indices.to(
                torch.int32
            )
        return output


class DeepseekV4CPUAttention(nn.Module, AttentionLayerBase):
    backend_cls = DeepseekV4CPUSparseMLABackend

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        return num_heads

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        cache_config = vllm_config.cache_config
        assert cache_config is not None
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        layer_id = extract_layer_index(prefix)

        self.prefix = prefix
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_local_heads = self.n_heads // tp_size
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.head_dim = config.head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.nope_dim = self.head_dim - self.rope_dim
        self.n_groups = config.o_groups
        self.n_local_groups = self.n_groups // tp_size
        self.heads_per_group = self.n_local_heads // self.n_local_groups
        self.window_size = config.sliding_window
        self.compress_ratio = (
            max(1, config.compress_ratios[layer_id])
            if layer_id < config.num_hidden_layers
            else 1
        )
        self.eps = config.rms_norm_eps
        self.scale = self.head_dim**-0.5
        self.topk_indices_buffer = topk_indices_buffer
        self.layer_id = layer_id
        self.is_mtp_block = layer_id >= config.num_hidden_layers
        self.max_model_len = vllm_config.model_config.max_model_len
        self.max_num_batched_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        self._fused_input_weights: Any | None = None
        self._fused_post_weights: Any | None = None
        self._fused_inv_woa_weights: Any | None = None
        self._fused_inv_woa_cos_sin_cache: torch.Tensor | None = None
        self._fused_wo_b_w8a8: Any | None = None
        self._fused_ops: dict[str, Any] | None = None
        self._fused_disabled_reason: str | None = None
        self._weights_prepared = False
        self._cores = _cpu_core_ids()

        self.attn_sink = nn.Parameter(
            torch.full((self.n_local_heads,), -float("inf"), dtype=torch.float32),
            requires_grad=False,
        )

        def load_attn_sink(
            parameter: torch.nn.Parameter, loaded_weight: torch.Tensor
        ) -> None:
            rank = get_tensor_model_parallel_rank()
            start = rank * self.n_local_heads
            local = loaded_weight[start : start + self.n_local_heads]
            parameter.fill_(-float("inf"))
            parameter[: local.shape[0]].copy_(local)

        set_weight_attrs(self.attn_sink, {"weight_loader": load_attn_sink})
        self.fused_wqa_wkv = MergedColumnParallelLinear(
            self.hidden_size,
            [self.q_lora_rank, self.head_dim],
            bias=False,
            # The checkpoint keeps WQ_A/WKV_A in BF16. The synthetic fused
            # module name does not match its compressed-tensors ignore rules.
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.fused_wqa_wkv",
        )
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
            return_bias=False,
            # Grouped WO_A remains BF16; only WO_B is a W8A8 target.
            quant_config=None,
            prefix=f"{prefix}.wo_a",
        )
        self.wo_a.is_bmm = True
        self.wo_a.bmm_batch_size = self.n_local_groups
        self.wo_b = RowParallelLinear(
            self.n_groups * self.o_lora_rank,
            self.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wo_b",
        )
        for module in (self.fused_wqa_wkv, self.wq_b, self.wo_a):
            _mark_joint_weight(module)
        self.wo_b._cpu_fused_cpp_linear_enabled = True
        self.wo_b._cpu_fused_cpp_linear_required = envs.VLLM_CPU_FUSED_CPP_STRICT
        if not self.is_mtp_block:
            for module in (self.wq_b, self.wo_b):
                module._cpu_fused_cpp_joint_int8_owned = True
        else:
            # The real checkpoint MTP block is intentionally outside the 107
            # full-model W8A8 projection targets.  Materialize its two INT8
            # linears once as BF16 and keep the ordinary vLLM call sites.
            for module in (self.wq_b, self.wo_b):
                module._cpu_int8_dequantize_to_bf16 = True

        self.rotary_emb = build_deepseek_v4_rope(
            config,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_dim,
            max_position_embeddings=config.max_position_embeddings,
            compress_ratio=self.compress_ratio,
        )
        self.indexer_rotary_emb = self.rotary_emb
        self.kv_cache = torch.tensor([])
        context = vllm_config.compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        context[prefix] = self

        self.swa_cache_layer = _CPUCacheLayer(
            head_dim=self.head_dim,
            dtype=torch.bfloat16,
            block_size=64,
            prefix=f"{prefix}.swa_cache",
            backend_cls=DeepseekV4CPUSWABackend,
            sliding_window=self.window_size,
        )
        self.compressor: DeepseekV4CPUCompressor | None = None
        if self.compress_ratio > 1:
            self.compressor = DeepseekV4CPUCompressor(
                vllm_config=vllm_config,
                compress_ratio=self.compress_ratio,
                hidden_size=self.hidden_size,
                head_dim=self.head_dim,
                prefix=f"{prefix}.compressor",
                output_cache_prefix=prefix,
            )

        self.indexer: DeepseekV4CPUIndexer | None = None
        if self.compress_ratio == 4:
            self.indexer = DeepseekV4CPUIndexer(
                vllm_config=vllm_config,
                hidden_size=self.hidden_size,
                q_lora_rank=self.q_lora_rank,
                quant_config=quant_config,
                cache_config=cache_config,
                compress_ratio=self.compress_ratio,
                prefix=f"{prefix}.indexer",
            )
            if not self.is_mtp_block:
                self.indexer.wq_b._cpu_fused_cpp_joint_int8_owned = True

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.backend_cls

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        if self.compress_ratio <= 1:
            return None
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=torch.bfloat16,
            compress_ratio=self.compress_ratio,
            alignment=self.head_dim * torch.bfloat16.itemsize,
            model_version="deepseek_v4",
        )

    def _input_projections(
        self, hidden_states: torch.Tensor
    ) -> tuple[
        torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None
    ]:
        qr_kv = _linear(self.fused_wqa_wkv, hidden_states)
        main_score = (
            self.compressor.project(hidden_states)
            if self.compressor is not None
            else None
        )
        index_score = None
        index_weights = None
        if self.indexer is not None:
            index_score = self.indexer.compressor.project(hidden_states)
            index_weights = _linear(self.indexer.weights_proj, hidden_states)
        return qr_kv, main_score, index_score, index_weights

    def _prepare_q_kv(
        self,
        qr: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        swa_metadata: DeepseekV4CPUSWAMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = _linear(self.wq_b, qr).reshape(-1, self.n_local_heads, self.head_dim)
        q = rms_norm(q, None, self.eps)
        q = apply_rope_tail(q, positions, self.rotary_emb.cos_sin_cache, self.rope_dim)
        kv = apply_rope_tail(
            kv, positions, self.rotary_emb.cos_sin_cache, self.rope_dim
        )
        write_paged_cache(
            self.swa_cache_layer.kv_cache,
            kv,
            swa_metadata.slot_mapping,
        )
        return q, kv

    @staticmethod
    def _logical_slots(
        logical: torch.Tensor,
        req: int,
        metadata: DeepseekV4CPUSparseMLAMetadata,
    ) -> torch.Tensor:
        reqs = torch.full_like(logical, req)
        blocks = metadata.block_table[reqs, logical // metadata.storage_block_size].to(
            torch.long
        )
        return (
            blocks * metadata.storage_block_size + logical % metadata.storage_block_size
        )

    def _candidate_rows(
        self,
        token: int,
        positions: torch.Tensor,
        swa_metadata: DeepseekV4CPUSWAMetadata,
        sparse_metadata: DeepseekV4CPUSparseMLAMetadata | None,
        indexer_topk: torch.Tensor | None,
    ) -> torch.Tensor:
        if token < swa_metadata.num_decode_tokens:
            swa_slots = swa_metadata.decode_swa_indices[token]
            swa_len = int(swa_metadata.decode_swa_lens[token].item())
        else:
            assert swa_metadata.prefill_swa_indices is not None
            assert swa_metadata.prefill_swa_lens is not None
            local = token - swa_metadata.num_decode_tokens
            swa_slots = swa_metadata.prefill_swa_indices[local]
            swa_len = int(swa_metadata.prefill_swa_lens[local].item())
        rows = [
            gather_paged_cache(
                self.swa_cache_layer.kv_cache,
                swa_slots[:swa_len].to(torch.long),
            )
        ]
        if sparse_metadata is not None and self.compress_ratio > 1:
            count = int((int(positions[token].item()) + 1) // self.compress_ratio)
            if self.compress_ratio == 4 and indexer_topk is not None:
                logical = indexer_topk[token]
                logical = logical[logical >= 0].to(torch.long)
                logical = logical[logical < count]
            else:
                logical = torch.arange(count, device=positions.device)
            if logical.numel():
                req = int(sparse_metadata.req_id_per_token[token].item())
                slots = self._logical_slots(logical, req, sparse_metadata)
                rows.insert(0, gather_paged_cache(self.kv_cache, slots))
        return torch.cat(rows, dim=0)

    def _execute_fused_input(
        self, hidden_states: torch.Tensor
    ) -> (
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor | None,
        ]
        | None
    ):
        if self._fused_input_weights is None or self._fused_ops is None:
            if envs.VLLM_CPU_FUSED_CPP_STRICT:
                raise RuntimeError("required fused_cpp attention input weights missing")
            return None
        try:
            with record_function_or_nullcontext(
                "vllm::deepseek_v4_attention/input_fused_cpp"
            ):
                return self._fused_ops["input_normed"](
                    hidden_states,
                    self._fused_input_weights,
                    self.q_norm.weight,
                    self.kv_norm.weight,
                    self.q_lora_rank,
                    self.head_dim,
                    self.eps,
                    self._cores,
                )
        except (RuntimeError, TypeError, ValueError) as exc:
            if envs.VLLM_CPU_FUSED_CPP_STRICT:
                raise RuntimeError("required fused_cpp attention input failed") from exc
            logger.warning_once(
                "DeepSeek V4 fused_cpp input stage failed (%s); using Torch.", exc
            )
            return None

    def _compressor_state(
        self,
        compressor: DeepseekV4CPUCompressor,
        metadata: dict[str, Any],
    ) -> Any:
        assert self._fused_ops is not None
        state = cast(
            DeepseekV4CPUCompressorMetadata,
            metadata[compressor.state_cache.prefix],
        )
        output = cast(
            DeepseekV4CPUSparseMLAMetadata,
            metadata[compressor.output_cache_prefix],
        )
        output_layer = compressor._static_forward_context[
            compressor.output_cache_prefix
        ]
        return self._fused_ops["CompressorState"](
            ape=compressor.ape,
            state_cache=compressor.state_cache.kv_cache,
            state_slot_mapping=state.slot_mapping,
            token_to_req_indices=state.token_to_req_indices,
            block_table=state.block_table,
            kv_cache=output_layer.kv_cache,
            kv_slot_mapping=output.slot_mapping,
            norm_weight=compressor.norm.weight,
            compress_ratio=compressor.compress_ratio,
            rms_norm_eps=compressor.norm_eps,
        )

    def _c4_prefill_metadata(
        self,
        positions: torch.Tensor,
        sparse_metadata: DeepseekV4CPUSparseMLAMetadata,
        swa_metadata: DeepseekV4CPUSWAMetadata,
    ) -> Any | None:
        if self._fused_ops is None or self.indexer is None:
            return None
        if (
            swa_metadata.num_decode_tokens
            or swa_metadata.num_prefill_tokens != positions.shape[0]
        ):
            return None
        cu_seq_lens = sparse_metadata.c4a_prefill_cu_seq_lens
        starts = sparse_metadata.c4a_prefill_starts
        ends = sparse_metadata.c4a_prefill_ends
        if cu_seq_lens is None or starts is None or ends is None:
            cu_seq_lens, starts, ends = _compressed_prefill_ranges(
                positions,
                sparse_metadata.req_id_per_token,
                swa_metadata.seq_lens,
                self.compress_ratio,
            )
        return self._fused_ops["Prefill"](
            cu_seq_lens=cu_seq_lens,
            cu_seqlen_ks=starts,
            cu_seqlen_ke=ends,
            block_table=sparse_metadata.block_table,
            topk_tokens=self.indexer.topk,
        )

    def _execute_fused_post(
        self,
        qr: torch.Tensor,
        kv: torch.Tensor,
        main_score: torch.Tensor | None,
        index_score: torch.Tensor | None,
        index_weights: torch.Tensor | None,
        positions: torch.Tensor,
        metadata: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor | None] | None:
        if self._fused_post_weights is None or self._fused_ops is None:
            if envs.VLLM_CPU_FUSED_CPP_STRICT and not self.is_mtp_block:
                raise RuntimeError("required fused_cpp post-GEMM weights missing")
            return None
        swa_metadata = cast(
            DeepseekV4CPUSWAMetadata,
            metadata[self.swa_cache_layer.prefix],
        )
        sparse_metadata = (
            cast(DeepseekV4CPUSparseMLAMetadata, metadata[self.prefix])
            if self.compress_ratio > 1
            else None
        )
        prefill = None
        if self.indexer is not None:
            assert sparse_metadata is not None
            indexer_meta = cast(
                DeepseekV4CPUSparseMLAMetadata,
                metadata[self.indexer.k_cache.prefix],
            )
            prefill = self._c4_prefill_metadata(positions, indexer_meta, swa_metadata)
            if prefill is None:
                if envs.VLLM_CPU_FUSED_CPP_STRICT:
                    raise RuntimeError(
                        "required fused_cpp C4A post stage supports "
                        "prefill-only batches"
                    )
                return None
        try:
            swa = self._fused_ops["SWAState"](
                kv_cache=self.swa_cache_layer.kv_cache,
                slot_mapping=swa_metadata.slot_mapping,
            )
            main_compressor = (
                self._compressor_state(self.compressor, metadata)
                if self.compressor is not None
                else None
            )
            indexer_compressor = (
                self._compressor_state(self.indexer.compressor, metadata)
                if self.indexer is not None
                else None
            )
            inputs = self._fused_ops["PostInputs"](
                qr=qr,
                kv=kv,
                positions=positions.to(torch.int64),
                main_wq_b_weight=getattr(self.wq_b, "weight", torch.empty(0)),
                main_cos_sin_cache=self.rotary_emb.cos_sin_cache,
                swa=swa,
                main_head_dim=self.head_dim,
                q_eps=self.eps,
                kv_score=main_score,
                indexer_kv_score=index_score,
                indexer_weights=index_weights,
                indexer_wq_b_weight=(
                    getattr(self.indexer.wq_b, "weight", None)
                    if self.indexer is not None
                    else None
                ),
                indexer_cos_sin_cache=(
                    self.indexer_rotary_emb.cos_sin_cache
                    if self.indexer is not None
                    else None
                ),
                mla_compressor=main_compressor,
                indexer_compressor=indexer_compressor,
                topk_indices_buffer=(
                    self.topk_indices_buffer if self.indexer is not None else None
                ),
                prefill=prefill,
                prepared_weights=self._fused_post_weights,
            )
            with record_function_or_nullcontext(
                "vllm::deepseek_v4_attention/post_fused_cpp"
            ):
                return self._fused_ops["post"](inputs, self._fused_post_weights)
        except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
            if envs.VLLM_CPU_FUSED_CPP_STRICT:
                raise RuntimeError("required fused_cpp post-GEMM stage failed") from exc
            logger.warning_once(
                "DeepSeek V4 fused_cpp post stage failed (%s); using Torch.", exc
            )
            return None

    def _run_fused_sparse(
        self,
        q: torch.Tensor,
        key_rows: list[torch.Tensor],
        out: torch.Tensor,
    ) -> bool:
        if self._fused_ops is None:
            if envs.VLLM_CPU_FUSED_CPP_STRICT:
                raise RuntimeError("required fused_cpp sparse MLA op is unavailable")
            return False
        total = sum(rows.shape[0] for rows in key_rows)
        if total == 0:
            out.zero_()
            return True
        width = max(rows.shape[0] for rows in key_rows)
        kv = torch.cat(key_rows, dim=0).unsqueeze(1)
        indices = torch.full(
            (len(key_rows), 1, width),
            -1,
            dtype=torch.int32,
            device=q.device,
        )
        lengths = torch.empty(len(key_rows), dtype=torch.int32, device=q.device)
        offset = 0
        for token, rows in enumerate(key_rows):
            count = rows.shape[0]
            lengths[token] = count
            indices[token, 0, :count] = torch.arange(
                offset, offset + count, dtype=torch.int32, device=q.device
            )
            offset += count
        try:
            self._fused_ops["sparse"](
                q,
                kv,
                indices,
                self.scale,
                d_v=self.head_dim,
                attn_sink=self.attn_sink,
                topk_length=lengths,
                out=out,
                return_stats=False,
            )
            return True
        except (RuntimeError, TypeError, ValueError) as exc:
            if envs.VLLM_CPU_FUSED_CPP_STRICT:
                raise RuntimeError("required fused_cpp sparse MLA failed") from exc
            logger.warning_once(
                "DeepSeek V4 fused_cpp sparse MLA failed (%s); using Torch.", exc
            )
            return False

    def _run_fused_prefill(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        sparse_metadata: DeepseekV4CPUSparseMLAMetadata | None,
        swa_metadata: DeepseekV4CPUSWAMetadata,
        indexer_topk: torch.Tensor | None,
        out: torch.Tensor,
    ) -> bool:
        all_prefill = (
            not swa_metadata.num_decodes
            and swa_metadata.num_prefill_tokens == q.shape[0]
        )
        if not all_prefill:
            return False
        if self._fused_ops is None or not self._fused_ops["prefill_available"]:
            if envs.VLLM_CPU_FUSED_CPP_STRICT:
                raise RuntimeError("required fused_cpp sparse prefill is unavailable")
            return False
        seq_lens = cast(torch.Tensor, swa_metadata.prefill_seq_lens)
        gather_lens = cast(torch.Tensor, swa_metadata.prefill_gather_lens)
        num_reqs = seq_lens.shape[0]
        has_compressed = sparse_metadata is not None and self.compress_ratio > 1
        N = cdiv(self.max_model_len, self.compress_ratio) if has_compressed else 0
        M = N + self.window_size + self.max_num_batched_tokens
        gathered = torch.empty(
            num_reqs, M, self.head_dim, dtype=torch.bfloat16, device=q.device
        )
        compressed_cache = (
            self.kv_cache if has_compressed else self.swa_cache_layer.kv_cache
        )
        compressed_lens = (
            torch.div(seq_lens, self.compress_ratio, rounding_mode="floor")
            if has_compressed
            else seq_lens
        )
        compressed_table = (
            sparse_metadata.block_table
            if sparse_metadata is not None
            else swa_metadata.block_table
        )
        compressed_block_size = (
            sparse_metadata.storage_block_size if sparse_metadata is not None else 1
        )
        try:
            self._fused_ops["dual_gather"](
                gathered,
                compressed_cache,
                compressed_lens,
                compressed_table,
                compressed_block_size,
                0,
                has_compressed,
                self.swa_cache_layer.kv_cache,
                seq_lens,
                gather_lens,
                swa_metadata.block_table,
                swa_metadata.block_size,
                N,
            )
            if self.compress_ratio == 4:
                assert indexer_topk is not None
                topk_indices = indexer_topk[: q.shape[0]]
                topk = topk_indices.shape[-1]
            elif self.compress_ratio == 128:
                assert sparse_metadata is not None
                topk_indices = cast(
                    torch.Tensor, sparse_metadata.c128a_prefill_topk_indices
                )
                topk = topk_indices.shape[-1]
            else:
                topk_indices = torch.empty(
                    q.shape[0], 0, dtype=torch.int32, device=q.device
                )
                topk = 0
            combined, lengths = self._fused_ops["combine"](
                topk_indices,
                swa_metadata.query_start_loc,
                seq_lens,
                gather_lens,
                self.window_size,
                self.compress_ratio,
                topk,
                M,
                N,
            )
            with record_function_or_nullcontext(
                "vllm::deepseek_v4_attention/sparse_fused_cpp"
            ):
                self._fused_ops["sparse"](
                    q,
                    gathered.view(-1, 1, self.head_dim),
                    combined.unsqueeze(1),
                    self.scale,
                    d_v=self.head_dim,
                    attn_sink=self.attn_sink,
                    topk_length=lengths,
                    out=out,
                    return_stats=False,
                )
            return True
        except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
            if envs.VLLM_CPU_FUSED_CPP_STRICT:
                raise RuntimeError("required fused_cpp sparse prefill failed") from exc
            logger.warning_once(
                "DeepSeek V4 fused_cpp sparse prefill failed (%s); using Torch.",
                exc,
            )
            return False

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if self._fused_inv_woa_weights is not None and self._fused_ops is not None:
            assert self._fused_inv_woa_cos_sin_cache is not None
            with record_function_or_nullcontext(
                "vllm::deepseek_v4_attention/output_inv_rope_woa_fused_cpp"
            ):
                z = self._fused_ops["inv_woa"](
                    o,
                    positions,
                    self._fused_inv_woa_cos_sin_cache,
                    self._fused_inv_woa_weights,
                    core_ids=self._cores,
                )
        else:
            if envs.VLLM_CPU_FUSED_CPP_STRICT:
                raise RuntimeError("required fused inverse-RoPE WO_A weights missing")
            o = apply_rope_tail(
                o,
                positions,
                self.rotary_emb.cos_sin_cache,
                self.rope_dim,
                inverse=True,
            )
            grouped = o.reshape(
                o.shape[0],
                self.n_local_groups,
                self.heads_per_group * self.head_dim,
            )
            weight = self.wo_a.weight.reshape(
                self.n_local_groups,
                self.o_lora_rank,
                self.heads_per_group * self.head_dim,
            )
            z = torch.einsum("bgr,gdr->bgd", grouped.float(), weight.float()).to(
                o.dtype
            )
        z_flat = z.flatten(1)
        if self._fused_wo_b_w8a8 is not None:
            assert self._fused_ops is not None
            with record_function_or_nullcontext(
                "vllm::deepseek_v4_attention/output_wo_b_w8a8"
            ):
                output = self._fused_ops["wo_b_w8a8"](
                    z_flat,
                    self._fused_wo_b_w8a8,
                    num_threads=len(self._cores),
                )
                if self.wo_b.reduce_results and self.wo_b.tp_size > 1:
                    output = tensor_model_parallel_all_reduce(output)
                return output
        with record_function_or_nullcontext("vllm::deepseek_v4_attention/output_wo_b"):
            return _linear(self.wo_b, z_flat)

    def process_weights_after_loading(self, dtype: torch.dtype | None = None) -> None:
        """Prepare joint fused_cpp attention weights after all loads complete."""

        del dtype
        # v0.28 invokes deferred-attention hooks before the model-level hook.
        # Keep this idempotent because both paths can visit the same module.
        if self._weights_prepared:
            return

        joint_int8_modules = [self.wq_b, self.wo_b]
        if self.indexer is not None:
            joint_int8_modules.append(self.indexer.wq_b)
        requires_fused_w8a8 = any(
            getattr(module, "_cpu_fused_cpp_joint_int8_owned", False)
            and isinstance((weight := getattr(module, "weight", None)), torch.Tensor)
            and weight.dtype == torch.int8
            and weight.numel() > 0
            for module in joint_int8_modules
        )

        def fail_or_fallback(reason: str, cause: Exception | None = None) -> None:
            self._fused_disabled_reason = reason
            if envs.VLLM_CPU_FUSED_CPP_STRICT or requires_fused_w8a8:
                raise RuntimeError(reason) from cause

        try:
            from fused_cpp import (
                deepseek_v4_attn_gemm_fused as attn_gemm_ops,
            )
            from fused_cpp.deepseek_v4_inv_rope_woa import (
                _HAS_DEEPSEEK_V4_INV_ROPE_WOA,
                deepseek_v4_inv_rope_grouped_woa,
                prepare_deepseek_v4_inv_rope_woa,
            )
            from fused_cpp.deepseek_v4_post_gemm_stage import (
                _HAS_DEEPSEEK_V4_POST_GEMM_C128A_PREPACKED,
                _HAS_DEEPSEEK_V4_POST_GEMM_DENSE_PREPACKED,
                _HAS_DEEPSEEK_V4_POST_GEMM_PROJECTED,
                _HAS_DEEPSEEK_V4_POST_GEMM_STAGE_PREPACKED,
                CompressorState,
                PostGemmStageInputs,
                SparseIndexerPrefillMetadata,
                SWACacheState,
                post_gemm_parallel_stage_cpp_prepacked,
                prepare_deepseek_v4_post_gemm_w8a8_quantized_weights,
                prepare_deepseek_v4_post_gemm_weights,
            )
            from fused_cpp.deepseek_v4_prefill_cache import (
                _HAS_DEEPSEEK_V4_PREFILL_CACHE_OPS,
                combine_topk_swa_indices_cpp,
                dequantize_and_gather_dual_k_cache_cpp,
            )
            from fused_cpp.deepseek_v4_w8a8 import (
                _HAS_DEEPSEEK_V4_W8A8,
                deepseek_v4_wo_b_w8a8,
                prepare_deepseek_v4_w8a8_linear_quantized_weight,
            )
            from fused_cpp.sparse_mla import flash_mla_sparse_fwd
        except (ImportError, AttributeError) as exc:
            fail_or_fallback(f"fused_cpp import failed: {exc}", exc)
            return

        if not attn_gemm_ops._HAS_DEEPSEEK_V4_ATTN_GEMM_FUSED:
            fail_or_fallback("attention input fused kernel is unavailable")
            return
        self._fused_ops = {
            "input": attn_gemm_ops.deepseek_v4_attn_gemm_fused_prepacked,
            "input_normed": (
                attn_gemm_ops.deepseek_v4_attn_gemm_fused_prepacked_normed
            ),
            "post": post_gemm_parallel_stage_cpp_prepacked,
            "PostInputs": PostGemmStageInputs,
            "SWAState": SWACacheState,
            "CompressorState": CompressorState,
            "Prefill": SparseIndexerPrefillMetadata,
            "inv_woa": deepseek_v4_inv_rope_grouped_woa,
            "dual_gather": dequantize_and_gather_dual_k_cache_cpp,
            "combine": combine_topk_swa_indices_cpp,
            "sparse": flash_mla_sparse_fwd,
            "wo_b_w8a8": deepseek_v4_wo_b_w8a8,
            "post_flags": (
                _HAS_DEEPSEEK_V4_POST_GEMM_DENSE_PREPACKED,
                _HAS_DEEPSEEK_V4_POST_GEMM_C128A_PREPACKED,
                _HAS_DEEPSEEK_V4_POST_GEMM_STAGE_PREPACKED,
                _HAS_DEEPSEEK_V4_POST_GEMM_PROJECTED,
            ),
            "prefill_available": _HAS_DEEPSEEK_V4_PREFILL_CACHE_OPS,
        }

        def kt(module: nn.Module, name: str) -> torch.Tensor:
            weight = getattr(module, "weight", None)
            if (
                not isinstance(weight, torch.Tensor)
                or weight.dtype != torch.bfloat16
                or weight.device.type != "cpu"
                or weight.ndim != 2
                or weight.numel() == 0
            ):
                raise RuntimeError(f"{name} must retain a CPU BF16 [N, K] weight")
            return weight.detach().t().contiguous()

        try:
            self._fused_input_weights = (
                attn_gemm_ops.prepare_deepseek_v4_attn_gemm_weights(
                    kt(self.fused_wqa_wkv, "fused_wqa_wkv"),
                    (
                        kt(
                            self.compressor.fused_wkv_wgate,
                            "compressor projection",
                        )
                        if self.compressor is not None
                        else None
                    ),
                    (
                        kt(
                            self.indexer.compressor.fused_wkv_wgate,
                            "indexer compressor projection",
                        )
                        if self.indexer is not None
                        else None
                    ),
                    (
                        kt(
                            self.indexer.weights_proj,
                            "indexer weights projection",
                        )
                        if self.indexer is not None
                        else None
                    ),
                )
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            fail_or_fallback(f"input preparation failed: {exc}", exc)
            return

        main_weight = getattr(self.wq_b, "weight", None)
        indexer_weight = (
            getattr(self.indexer.wq_b, "weight", None)
            if self.indexer is not None
            else None
        )
        if not self.is_mtp_block and isinstance(main_weight, torch.Tensor):
            if main_weight.dtype == torch.int8:
                if (
                    not _HAS_DEEPSEEK_V4_W8A8
                    or not _HAS_DEEPSEEK_V4_POST_GEMM_PROJECTED
                ):
                    raise RuntimeError(
                        "required DeepSeek V4 W8A8 post stage is unavailable"
                    )
                main_scale = getattr(self.wq_b, "weight_scale", None)
                indexer_scale = (
                    getattr(self.indexer.wq_b, "weight_scale", None)
                    if self.indexer is not None
                    else None
                )
                if not isinstance(main_scale, torch.Tensor):
                    raise RuntimeError("main wq_b W8A8 scale is unavailable")
                self._fused_post_weights = (
                    prepare_deepseek_v4_post_gemm_w8a8_quantized_weights(
                        main_weight,
                        main_scale.reshape(-1).float().contiguous(),
                        indexer_weight,
                        (
                            indexer_scale.reshape(-1).float().contiguous()
                            if isinstance(indexer_scale, torch.Tensor)
                            else None
                        ),
                    )
                )
            elif main_weight.dtype == torch.bfloat16:
                self._fused_post_weights = prepare_deepseek_v4_post_gemm_weights(
                    main_weight.contiguous(),
                    (
                        indexer_weight.contiguous()
                        if isinstance(indexer_weight, torch.Tensor)
                        else None
                    ),
                )
        elif not self.is_mtp_block:
            raise RuntimeError("required main wq_b source weight is unavailable")

        if _HAS_DEEPSEEK_V4_INV_ROPE_WOA:
            wo_a_weight = getattr(self.wo_a, "weight", None)
            if (
                isinstance(wo_a_weight, torch.Tensor)
                and wo_a_weight.dtype == torch.bfloat16
            ):
                self._fused_inv_woa_weights = prepare_deepseek_v4_inv_rope_woa(
                    wo_a_weight.contiguous(),
                    n_groups=self.n_local_groups,
                    heads_per_group=self.heads_per_group,
                    head_dim=self.head_dim,
                    rope_dim=self.rope_dim,
                    backend="arm_sve_bf16",
                )
                self._fused_inv_woa_cos_sin_cache = (
                    self.rotary_emb.cos_sin_cache.detach().float().contiguous()
                )
        if envs.VLLM_CPU_FUSED_CPP_STRICT and self._fused_inv_woa_weights is None:
            raise RuntimeError("required inverse-RoPE grouped WO_A is unavailable")

        wo_b_weight = getattr(self.wo_b, "weight", None)
        if (
            not self.is_mtp_block
            and isinstance(wo_b_weight, torch.Tensor)
            and wo_b_weight.dtype == torch.int8
        ):
            if not _HAS_DEEPSEEK_V4_W8A8:
                raise RuntimeError(
                    "required DeepSeek V4 WO_B W8A8 backend is unavailable"
                )
            wo_b_scale = getattr(self.wo_b, "weight_scale", None)
            if not isinstance(wo_b_scale, torch.Tensor):
                raise RuntimeError("WO_B W8A8 scale is unavailable")
            self._fused_wo_b_w8a8 = prepare_deepseek_v4_w8a8_linear_quantized_weight(
                wo_b_weight,
                wo_b_scale.reshape(-1).float().contiguous(),
            )

        if envs.VLLM_CPU_FUSED_CPP_STRICT and not self.is_mtp_block:
            dense, c128a, c4a, projected = self._fused_ops["post_flags"]
            post_available = (
                dense
                if self.compress_ratio <= 1
                else (c4a if self.compress_ratio == 4 else c128a)
            )
            if not post_available or self._fused_post_weights is None:
                raise RuntimeError(
                    "required DeepSeek V4 post-GEMM stage is unavailable"
                )
            if not self._fused_ops["prefill_available"]:
                raise RuntimeError(
                    "required DeepSeek V4 prefill cache ops are unavailable"
                )

        def release(module: nn.Module, names: tuple[str, ...] = ("weight",)) -> None:
            for name in names:
                value = getattr(module, name, None)
                if isinstance(value, torch.Tensor):
                    replace_parameter(
                        module,
                        name,
                        torch.empty(0, dtype=value.dtype, device=value.device),
                    )

        # Required paths own their packed objects and can release all raw
        # projection tensors.  W8A8 targets are always required, independent
        # of the global strict switch.
        if envs.VLLM_CPU_FUSED_CPP_STRICT:
            release(self.fused_wqa_wkv)
            if self.compressor is not None:
                release(self.compressor.fused_wkv_wgate)
            if self.indexer is not None:
                release(self.indexer.compressor.fused_wkv_wgate)
                release(self.indexer.weights_proj)
            release(self.wo_a)
            if self._fused_post_weights is not None:
                release(self.wq_b)
                if self.indexer is not None:
                    release(self.indexer.wq_b)
        if self._fused_wo_b_w8a8 is not None:
            release(self.wo_b, ("weight", "weight_scale"))

        logger.info_once(
            "Prepared fused_cpp DeepSeek V4 CPU attention layer "
            "(variant=%s, W8A8_post=%s, W8A8_wo_b=%s, threads=%d).",
            "c4a"
            if self.indexer is not None
            else ("c128a" if self.compressor is not None else "dense"),
            main_weight is not None and main_weight.dtype == torch.int8,
            self._fused_wo_b_w8a8 is not None,
            len(self._cores),
        )
        self._weights_prepared = True

    def _forward_impl(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del llama_4_scaling
        fused_input = self._execute_fused_input(hidden_states)
        if fused_input is not None:
            qr, kv, main_score, index_score, index_weights = fused_input
        else:
            qr_kv, main_score, index_score, index_weights = self._input_projections(
                hidden_states
            )
            qr, kv = qr_kv.split((self.q_lora_rank, self.head_dim), dim=-1)
            qr = rms_norm(qr, self.q_norm.weight, self.eps)
            kv = rms_norm(kv, self.kv_norm.weight, self.eps)

        metadata_dict = get_forward_context().attn_metadata
        if not isinstance(metadata_dict, dict):
            o = hidden_states.new_zeros(
                hidden_states.shape[0], self.n_local_heads, self.head_dim
            )
            return self._o_proj(o, positions)
        swa_metadata = cast(
            DeepseekV4CPUSWAMetadata,
            metadata_dict[self.swa_cache_layer.prefix],
        )
        sparse_metadata = (
            cast(DeepseekV4CPUSparseMLAMetadata, metadata_dict[self.prefix])
            if self.compress_ratio > 1
            else None
        )
        fused_post = self._execute_fused_post(
            qr,
            kv,
            main_score,
            index_score,
            index_weights,
            positions,
            metadata_dict,
        )
        if fused_post is not None:
            q, indexer_topk = fused_post
        else:
            q, _ = self._prepare_q_kv(qr, kv, positions, swa_metadata)
            if self.compressor is not None:
                assert main_score is not None
                self.compressor(main_score, positions, self.rotary_emb)
            indexer_topk = None
            if self.indexer is not None:
                assert index_score is not None and index_weights is not None
                indexer_topk = self.indexer(
                    hidden_states,
                    qr,
                    index_score,
                    index_weights,
                    positions,
                    self.indexer_rotary_emb,
                )

        o = torch.empty_like(q)
        if self._run_fused_prefill(
            q,
            positions,
            sparse_metadata,
            swa_metadata,
            indexer_topk,
            o,
        ):
            return self._o_proj(o, positions)

        candidate_rows = [
            self._candidate_rows(
                token,
                positions,
                swa_metadata,
                sparse_metadata,
                indexer_topk,
            )
            for token in range(q.shape[0])
        ]
        if not self._run_fused_sparse(q, candidate_rows, o):
            sparse_mla_reference(q, candidate_rows, self.scale, self.attn_sink, o)
        return self._o_proj(o, positions)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with record_function_or_nullcontext("vllm::deepseek_v4_attention"):
            return self._forward_impl(positions, hidden_states, llama_4_scaling)


__all__ = [
    "DeepseekV4CPUAttention",
    "DeepseekV4CPUCompressor",
    "DeepseekV4CPUIndexer",
]
