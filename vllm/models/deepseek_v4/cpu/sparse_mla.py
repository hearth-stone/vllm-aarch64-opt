# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only DeepSeek V4 sparse-MLA metadata and cache backends."""

from dataclasses import dataclass
from typing import Any, ClassVar, cast

import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import AttentionSpec, SlidingWindowMLASpec

_C128A_ALIGNMENT = 128


def _logical_to_slot(
    positions: torch.Tensor,
    req_ids: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    positions = positions.to(torch.long)
    req_ids = req_ids.to(torch.long)
    blocks = positions // block_size
    offsets = positions % block_size
    block_numbers = block_table[req_ids, blocks].to(torch.long)
    return block_numbers * block_size + offsets


def compressed_slot_mapping_torch(
    cm: CommonAttentionMetadata,
    block_size: int,
    compress_ratio: int,
    out: torch.Tensor,
    req_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pure-Torch equivalent of the compressed-slot Triton helper."""

    out.fill_(-1)
    if cm.positions is not None:
        positions = cm.positions[: cm.num_actual_tokens].to(torch.long)
        if req_ids is None:
            query_lens = torch.diff(cm.query_start_loc_cpu).to(torch.long)
            req_ids = torch.repeat_interleave(
                torch.arange(cm.num_reqs, device=positions.device),
                query_lens.to(device=positions.device),
            )[: cm.num_actual_tokens]
        else:
            req_ids = req_ids[: cm.num_actual_tokens].to(
                device=positions.device, dtype=torch.long
            )
        valid = (
            (positions >= 0)
            & ((positions + 1).remainder(compress_ratio) == 0)
            & (cm.slot_mapping[: cm.num_actual_tokens] >= 0)
        )
        compressed = torch.div(positions, compress_ratio, rounding_mode="floor")
        block_indices = torch.div(compressed, block_size, rounding_mode="floor")
        block_offsets = compressed.remainder(block_size)
        valid_indices = valid.nonzero(as_tuple=False).flatten()
        if valid_indices.numel():
            physical_blocks = cm.block_table_tensor[
                req_ids[valid_indices], block_indices[valid_indices]
            ].to(torch.long)
            out[valid_indices] = (
                physical_blocks * block_size + block_offsets[valid_indices]
            )
        return out[: cm.num_actual_tokens]

    qsl = cm.query_start_loc_cpu.tolist()
    seq_lens = cm.seq_lens.to("cpu").tolist()
    block_table = cm.block_table_tensor
    for req in range(cm.num_reqs):
        query_start, query_end = qsl[req], qsl[req + 1]
        query_len = query_end - query_start
        first_position = int(seq_lens[req]) - query_len
        for token_offset in range(query_len):
            position = first_position + token_offset
            if position < 0 or (position + 1) % compress_ratio:
                continue
            compressed_position = position // compress_ratio
            block_index = compressed_position // block_size
            block_offset = compressed_position % block_size
            block_number = int(block_table[req, block_index].item())
            out[query_start + token_offset] = block_number * block_size + block_offset
    return out[: cm.num_actual_tokens]


@dataclass
class DeepseekV4CPUSparseMLAMetadata(AttentionMetadata):
    num_reqs: int
    max_query_len: int
    max_seq_len: int
    num_actual_tokens: int
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    req_id_per_token: torch.Tensor
    block_size: int
    storage_block_size: int
    topk_tokens: int
    c128a_global_decode_topk_indices: torch.Tensor | None = None
    c128a_decode_topk_lens: torch.Tensor | None = None
    c128a_prefill_topk_indices: torch.Tensor | None = None
    c4a_prefill_cu_seq_lens: torch.Tensor | None = None
    c4a_prefill_starts: torch.Tensor | None = None
    c4a_prefill_ends: torch.Tensor | None = None


class DeepseekV4CPUSparseMLAMetadataBuilder(
    AttentionMetadataBuilder[DeepseekV4CPUSparseMLAMetadata]
):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)
        self.topk_tokens = vllm_config.model_config.hf_config.index_topk
        self.compress_ratio = int(getattr(kv_cache_spec, "compress_ratio", 1))
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.req_ids = torch.empty(max_tokens, dtype=torch.int32, device=device)
        self.compressed_slots = torch.empty(
            max_tokens, dtype=torch.int64, device=device
        )
        max_compressed = cdiv(
            vllm_config.model_config.max_model_len,
            max(self.compress_ratio, 1),
        )
        self.c128_width = cdiv(max_compressed, _C128A_ALIGNMENT) * _C128A_ALIGNMENT

    def _build_c128a(
        self,
        cm: CommonAttentionMetadata,
        req_ids: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if self.compress_ratio != 128 or cm.positions is None:
            return None, None, None
        num_decodes, _, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                cm, decode_threshold=self.reorder_batch_threshold or 1
            )
        )
        positions = cm.positions[: cm.num_actual_tokens].to(torch.long)
        counts = torch.div(
            positions + 1, self.compress_ratio, rounding_mode="floor"
        ).clamp(max=self.c128_width)

        global_decode = None
        decode_lens = None
        if num_decode_tokens:
            global_decode = torch.full(
                (num_decode_tokens, 1, self.c128_width),
                -1,
                dtype=torch.int32,
                device=self.device,
            )
            decode_lens = counts[:num_decode_tokens].to(torch.int32)
            for token in range(num_decode_tokens):
                count = int(decode_lens[token].item())
                if count == 0 or cm.slot_mapping[token] < 0:
                    decode_lens[token] = 0
                    continue
                logical = torch.arange(count, device=self.device)
                token_reqs = req_ids[token].expand(count)
                slots = _logical_to_slot(
                    logical,
                    token_reqs,
                    cm.block_table_tensor[:num_decodes],
                    int(getattr(self.kv_cache_spec, "storage_block_size", 1)),
                )
                global_decode[token, 0, :count] = slots.to(torch.int32)

        prefill_local = None
        if num_prefill_tokens:
            prefill_local = torch.full(
                (num_prefill_tokens, self.c128_width),
                -1,
                dtype=torch.int32,
                device=self.device,
            )
            for local_token, count_tensor in enumerate(counts[num_decode_tokens:]):
                count = int(count_tensor.item())
                prefill_local[local_token, :count] = torch.arange(
                    count, dtype=torch.int32, device=self.device
                )
        return global_decode, decode_lens, prefill_local

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV4CPUSparseMLAMetadata:
        del common_prefix_len, fast_build
        cm = common_attn_metadata
        req_ids = cm.token_to_req_indices(self.req_ids)
        slots = cm.slot_mapping
        if self.compress_ratio > 1:
            storage_block_size = int(
                getattr(self.kv_cache_spec, "storage_block_size", 1)
            )
            slots = compressed_slot_mapping_torch(
                cm,
                storage_block_size,
                self.compress_ratio,
                self.compressed_slots,
                req_ids,
            )
        c128_decode, c128_lens, c128_prefill = self._build_c128a(cm, req_ids)
        c4_cu_lens = c4_starts = c4_ends = None
        if self.compress_ratio == 4 and cm.positions is not None:
            positions = cm.positions[: cm.num_actual_tokens]
            compressed_lens = torch.div(
                cm.seq_lens, self.compress_ratio, rounding_mode="floor"
            ).to(torch.int32)
            c4_cu_lens = torch.cat(
                (compressed_lens.new_zeros(1), compressed_lens.cumsum(0))
            )
            c4_starts = c4_cu_lens[req_ids.to(torch.long)]
            c4_ends = c4_starts + torch.div(
                positions.to(torch.int64) + 1,
                self.compress_ratio,
                rounding_mode="floor",
            ).to(torch.int32)
        return DeepseekV4CPUSparseMLAMetadata(
            num_reqs=cm.num_reqs,
            max_query_len=cm.max_query_len,
            max_seq_len=cm.max_seq_len,
            num_actual_tokens=cm.num_actual_tokens,
            query_start_loc=cm.query_start_loc,
            slot_mapping=slots,
            block_table=cm.block_table_tensor,
            req_id_per_token=req_ids,
            block_size=int(self.kv_cache_spec.block_size),
            storage_block_size=int(
                getattr(self.kv_cache_spec, "storage_block_size", 1)
            ),
            topk_tokens=self.topk_tokens,
            c128a_global_decode_topk_indices=c128_decode,
            c128a_decode_topk_lens=c128_lens,
            c128a_prefill_topk_indices=c128_prefill,
            c4a_prefill_cu_seq_lens=c4_cu_lens,
            c4a_prefill_starts=c4_starts,
            c4a_prefill_ends=c4_ends,
        )


class DeepseekV4CPUSparseMLABackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto"]

    @staticmethod
    def get_name() -> str:
        return "CPU_MLA_SPARSE_DSV4"

    @staticmethod
    def get_builder_cls() -> type[DeepseekV4CPUSparseMLAMetadataBuilder]:
        return DeepseekV4CPUSparseMLAMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type[Any]:
        raise NotImplementedError("DeepSeek V4 CPU attention runs in its model layer")

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(1)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [128, 512]

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_sink(cls) -> bool:
        return True

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        del num_kv_heads, cache_dtype_str
        return num_blocks, block_size, head_size


@dataclass
class DeepseekV4CPUSWAMetadata(AttentionMetadata):
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    block_size: int
    seq_lens: torch.Tensor
    query_start_loc: torch.Tensor
    query_start_loc_cpu: torch.Tensor
    is_valid_token: torch.Tensor
    token_to_req_indices: torch.Tensor
    decode_swa_indices: torch.Tensor
    decode_swa_lens: torch.Tensor
    prefill_swa_indices: torch.Tensor | None
    prefill_swa_lens: torch.Tensor | None
    num_decodes: int
    num_prefills: int
    num_decode_tokens: int
    num_prefill_tokens: int
    max_decode_query_len: int
    prefill_seq_lens: torch.Tensor | None = None
    prefill_seq_lens_cpu: torch.Tensor | None = None
    prefill_gather_lens: torch.Tensor | None = None
    prefill_query_lens_cpu: torch.Tensor | None = None
    prefill_window_size: int = 0
    prefill_max_model_len: int = 0
    prefill_max_num_batched_tokens: int = 0

    def get_prefill_chunk_plan(
        self, compress_ratio: int, prefill_chunk_size: int
    ) -> list[tuple[int, int, int, int]]:
        if self.num_prefills == 0:
            return []
        assert self.prefill_seq_lens_cpu is not None
        assert self.prefill_query_lens_cpu is not None
        plan: list[tuple[int, int, int, int]] = []
        for start in range(0, self.num_prefills, prefill_chunk_size):
            end = min(start + prefill_chunk_size, self.num_prefills)
            seq = self.prefill_seq_lens_cpu[start:end]
            query = self.prefill_query_lens_cpu[start:end]
            gather = query + torch.clamp(
                seq - query, min=0, max=self.prefill_window_size - 1
            )
            compressed = (
                torch.zeros_like(seq)
                if compress_ratio <= 1
                else torch.div(seq, compress_ratio, rounding_mode="floor")
            )
            plan.append(
                (
                    start,
                    end,
                    int(compressed.max().item()),
                    int(gather.max().item()),
                )
            )
        return plan


class DeepseekV4CPUSWAMetadataBuilder(
    AttentionMetadataBuilder[DeepseekV4CPUSWAMetadata]
):
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)
        self.window_size = int(cast(SlidingWindowMLASpec, kv_cache_spec).sliding_window)
        self.block_size = int(kv_cache_spec.block_size)
        self.max_model_len = vllm_config.model_config.max_model_len
        self.max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.req_ids = torch.empty(self.max_tokens, dtype=torch.int32, device=device)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV4CPUSWAMetadata:
        del common_prefix_len, fast_build
        cm = common_attn_metadata
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                cm, decode_threshold=self.reorder_batch_threshold or 1
            )
        )
        req_ids = cm.token_to_req_indices(self.req_ids)
        valid = cm.slot_mapping[: cm.num_actual_tokens] >= 0
        # Strict fused prefill consumes seq/gather lengths directly.  Building
        # a [prefill_tokens, window] index matrix here would allocate and fill
        # tens of millions of elements on every request even though the fused
        # dual-cache gather never reads it.  Keep explicit rows only for
        # decode; non-strict reference mode retains the full metadata.
        indexed_tokens = (
            num_decode_tokens
            if envs.VLLM_CPU_FUSED_CPP_STRICT
            else cm.num_actual_tokens
        )
        indices = torch.full(
            (indexed_tokens, self.window_size),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        lens = torch.zeros(indexed_tokens, dtype=torch.int32, device=self.device)
        qsl = cm.query_start_loc_cpu.tolist()
        seq_lens_cpu = (
            cm.seq_lens_cpu_upper_bound
            if cm.seq_lens_cpu_upper_bound is not None
            else cm.seq_lens.to("cpu")
        )
        seq_lens_list = seq_lens_cpu.tolist()
        for token in range(indexed_tokens):
            if not bool(valid[token]):
                continue
            req = int(req_ids[token].item())
            query_len = qsl[req + 1] - qsl[req]
            position = int(seq_lens_list[req]) - query_len + token - qsl[req]
            start = max(0, position - self.window_size + 1)
            logical = torch.arange(start, position + 1, device=self.device)
            token_reqs = torch.full_like(logical, req)
            slots = _logical_to_slot(
                logical,
                token_reqs,
                cm.block_table_tensor,
                self.block_size,
            )
            count = slots.numel()
            indices[token, :count] = slots.to(torch.int32)
            lens[token] = count

        prefill_seq_lens = cm.seq_lens[num_decodes:] if num_prefills else None
        prefill_seq_lens_cpu = seq_lens_cpu[num_decodes:] if num_prefills else None
        prefill_query_lens_cpu = None
        prefill_gather_lens = None
        if num_prefills:
            prefill_query_lens_cpu = (
                cm.query_start_loc_cpu[num_decodes + 1 : num_decodes + num_prefills + 1]
                - cm.query_start_loc_cpu[num_decodes : num_decodes + num_prefills]
            ).to(torch.int32)
            assert prefill_seq_lens is not None
            query_lens = cm.query_start_loc[1:] - cm.query_start_loc[:-1]
            prefill_query_lens = query_lens[num_decodes:]
            prefill_gather_lens = prefill_query_lens + torch.clamp(
                prefill_seq_lens - prefill_query_lens,
                min=0,
                max=self.window_size - 1,
            )

        return DeepseekV4CPUSWAMetadata(
            block_table=cm.block_table_tensor,
            slot_mapping=cm.slot_mapping,
            block_size=self.block_size,
            seq_lens=cm.seq_lens,
            query_start_loc=cm.query_start_loc,
            query_start_loc_cpu=cm.query_start_loc_cpu,
            is_valid_token=valid,
            token_to_req_indices=req_ids,
            decode_swa_indices=indices[:num_decode_tokens],
            decode_swa_lens=lens[:num_decode_tokens],
            prefill_swa_indices=(
                indices[num_decode_tokens:]
                if num_prefill_tokens and not envs.VLLM_CPU_FUSED_CPP_STRICT
                else None
            ),
            prefill_swa_lens=(
                lens[num_decode_tokens:]
                if num_prefill_tokens and not envs.VLLM_CPU_FUSED_CPP_STRICT
                else None
            ),
            num_decodes=num_decodes,
            num_prefills=num_prefills,
            num_decode_tokens=num_decode_tokens,
            num_prefill_tokens=num_prefill_tokens,
            max_decode_query_len=min(
                cm.max_query_len, self.reorder_batch_threshold or 1
            ),
            prefill_seq_lens=prefill_seq_lens,
            prefill_seq_lens_cpu=prefill_seq_lens_cpu,
            prefill_gather_lens=prefill_gather_lens,
            prefill_query_lens_cpu=prefill_query_lens_cpu,
            prefill_window_size=self.window_size,
            prefill_max_model_len=self.max_model_len,
            prefill_max_num_batched_tokens=self.max_tokens,
        )


class DeepseekV4CPUSWABackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "CPU_DEEPSEEK_V4_SWA"

    @staticmethod
    def get_builder_cls() -> type[DeepseekV4CPUSWAMetadataBuilder]:
        return DeepseekV4CPUSWAMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type[Any]:
        raise NotImplementedError("DeepSeek V4 CPU SWA runs in its model layer")

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(1)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [512]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        del num_kv_heads, cache_dtype_str
        return num_blocks, block_size, head_size


@dataclass
class DeepseekV4CPUCompressorMetadata(AttentionMetadata):
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    block_size: int
    token_to_req_indices: torch.Tensor


class DeepseekV4CPUCompressorMetadataBuilder(
    AttentionMetadataBuilder[DeepseekV4CPUCompressorMetadata]
):
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.block_size = int(kv_cache_spec.block_size)
        self.req_ids = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.int32,
            device=device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV4CPUCompressorMetadata:
        del common_prefix_len, fast_build
        cm = common_attn_metadata
        return DeepseekV4CPUCompressorMetadata(
            block_table=cm.block_table_tensor.clamp(min=0),
            slot_mapping=cm.slot_mapping,
            block_size=self.block_size,
            token_to_req_indices=cm.token_to_req_indices(self.req_ids),
        )


class DeepseekV4CPUCompressorBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "CPU_DEEPSEEK_V4_COMPRESSOR"

    @staticmethod
    def get_builder_cls() -> type[DeepseekV4CPUCompressorMetadataBuilder]:
        return DeepseekV4CPUCompressorMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type[Any]:
        raise NotImplementedError("DeepSeek V4 compressor is model-owned")

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(1)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [256, 1024, 2048]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        del num_kv_heads, cache_dtype_str
        return num_blocks, block_size, head_size


__all__ = [
    "DeepseekV4CPUCompressorBackend",
    "DeepseekV4CPUCompressorMetadata",
    "DeepseekV4CPUSWABackend",
    "DeepseekV4CPUSWAMetadata",
    "DeepseekV4CPUSparseMLABackend",
    "DeepseekV4CPUSparseMLAMetadata",
    "compressed_slot_mapping_torch",
]
