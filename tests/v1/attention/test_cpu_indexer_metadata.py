# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the CPU branch of ``DeepseekV32IndexerMetadataBuilder``
(M2b part 2b).

The builder lives in ``vllm/v1/attention/backends/mla/indexer.py`` and
holds a stack of GPU-only call sites (``num_compute_units``,
``_prepare_uniform_decode_kernel``, ``get_compressed_slot_mapping``,
``_build_prefill_chunk_metadata_kernel``). These tests exercise each
call site through the public ``__init__`` / ``build`` surface on a
``torch.device('cpu')`` device and check that the constructed
``DeepseekV32IndexerMetadata`` has the right shapes and contents.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.mla.compressor_utils import (
    get_compressed_slot_mapping,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
    DeepseekV32IndexerMetadataBuilder,
    build_prefill_chunk_metadata,
)
from vllm.v1.kv_cache_interface import MLAAttentionSpec


pytestmark = pytest.mark.cpu_test


def _make_vllm_config(
    max_num_batched_tokens: int = 64,
    max_num_seqs: int = 8,
    max_model_len: int = 256,
):
    cfg = MagicMock()
    cfg.scheduler_config.max_num_batched_tokens = max_num_batched_tokens
    cfg.scheduler_config.max_num_seqs = max_num_seqs
    cfg.speculative_config = None
    cfg.attention_config.use_fp4_indexer_cache = False
    cfg.model_config.max_model_len = max_model_len
    return cfg


def _make_kv_cache_spec(block_size: int = 16, compress_ratio: int = 4):
    return MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        compress_ratio=compress_ratio,
    )


def _make_builder(compress_ratio: int = 4):
    return DeepseekV32IndexerMetadataBuilder(
        kv_cache_spec=_make_kv_cache_spec(compress_ratio=compress_ratio),
        layer_names=["x"],
        vllm_config=_make_vllm_config(),
        device=torch.device("cpu"),
    )


# ---------------------------------------------------------------------------
# get_compressed_slot_mapping CPU port
# ---------------------------------------------------------------------------


def test_get_compressed_slot_mapping_cpu_matches_expected_for_compress_ratio_2():
    """compress_ratio=2: only odd absolute positions (where (pos+1) % 2 == 0)
    map to compressed slots; even positions stay -1."""
    seq_lens = torch.tensor([4, 6], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 4, 10], dtype=torch.int32)
    block_table = torch.tensor([[10, 0], [20, 0]], dtype=torch.int32)

    out = get_compressed_slot_mapping(
        num_tokens=10,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        block_table=block_table,
        block_size=4,
        compress_ratio=2,
    )

    # req 0: positions 0..3 → valid at pos=1 (compressed=0) and pos=3 (compressed=1)
    #   → slot ids 10*4+0=40, 10*4+1=41
    # req 1: positions 0..5 → valid at pos=1,3,5 (compressed 0,1,2)
    #   → slot ids 80, 81, 82
    expected = [-1, 40, -1, 41, -1, 80, -1, 81, -1, 82]
    assert out.tolist() == expected


def test_get_compressed_slot_mapping_cpu_uses_out_buffer():
    """When ``out`` is provided the function returns a view into it
    and fills the entire buffer with -1 first."""
    out_buf = torch.full((20,), 999, dtype=torch.int64)
    seq_lens = torch.tensor([4], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 4], dtype=torch.int32)
    block_table = torch.tensor([[7]], dtype=torch.int32)

    result = get_compressed_slot_mapping(
        num_tokens=4,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        block_table=block_table,
        block_size=4,
        compress_ratio=4,  # only pos=3 is valid → slot 7*4+0 = 28
        out=out_buf,
    )

    assert result.data_ptr() == out_buf.data_ptr()
    assert result.shape == (4,)
    assert result.tolist() == [-1, -1, -1, 28]
    # Tail of the buffer is reset to -1, not left as 999.
    assert torch.all(out_buf[4:] == -1)


# ---------------------------------------------------------------------------
# build_prefill_chunk_metadata CPU port
# ---------------------------------------------------------------------------


def test_build_prefill_chunk_metadata_cpu_matches_expected():
    """Two prefill requests with seq_lens=[4, 6] and full query_lens=[4, 6].
    compress_ratio=2 → compressed seq_lens=[2, 3]. cu_seq_lens=[0, 2, 5].
    For each query token at position p (0..query_len-1), the valid
    compressed key range is [cu_seq_lens[r], cu_seq_lens[r] + (p+1)//2].
    """
    device = torch.device("cpu")
    compress_ratio = 2
    # query_start_loc is per-request rebased to 0 (matches what
    # ``DeepseekV32IndexerMetadataBuilder.build`` passes after the
    # per-chunk rebase ``query_start_loc - query_start_loc[start_idx]``).
    query_start_loc = torch.tensor([0, 4, 10], dtype=torch.int32, device=device)
    uncompressed_seq_lens = torch.tensor([4, 6], dtype=torch.int32, device=device)
    compressed_seq_lens_cpu = torch.tensor([2, 3], dtype=torch.int32)
    compressed_seq_lens = compressed_seq_lens_cpu.clone()
    block_table = torch.tensor(
        [[10, 11], [20, 21]], dtype=torch.int32, device=device
    )

    meta = build_prefill_chunk_metadata(
        start_idx=0,
        end_idx=2,
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc,
        uncompressed_seq_lens=uncompressed_seq_lens,
        compressed_seq_lens=compressed_seq_lens,
        compressed_seq_lens_cpu=compressed_seq_lens_cpu,
        block_table=block_table,
        compress_ratio=compress_ratio,
    )

    assert meta is not None
    # cu_seq_lens = cumsum of compressed seq lens, so [0, 2, 5].
    assert meta.cu_seq_lens.tolist() == [0, 2, 5]
    # Per-token cu_seqlen_ks = chunk-flat start of the request's K range.
    # req 0 has 4 tokens → ks=[0,0,0,0]. req 1 has 6 tokens → ks=[2,2,2,2,2,2].
    expected_ks = [0, 0, 0, 0, 2, 2, 2, 2, 2, 2]
    assert meta.cu_seqlen_ks.tolist() == expected_ks
    # Per-token cu_seqlen_ke = ks + (pos+1) // compress_ratio. With
    # full prefill, start_pos = 0 and i = pos. compress_ratio=2 →
    # (p+1)//2 = [0, 1, 1, 2] for req 0 and [0, 1, 1, 2, 2, 3] for req 1.
    expected_ke = [0, 1, 1, 2, 2, 3, 3, 4, 4, 5]
    assert meta.cu_seqlen_ke.tolist() == expected_ke
    # token_to_seq covers the chunk-flat compressed K range:
    # [0:2] → 0, [2:5] → 1.
    assert meta.token_to_seq.tolist() == [0, 0, 1, 1, 1]
    assert meta.total_seq_lens == 5
    assert meta.num_reqs == 2


def test_build_prefill_chunk_metadata_cpu_returns_none_when_total_zero():
    """When all compressed seq lens in the chunk are 0, the function
    should return None (matches GPU behavior)."""
    compressed_seq_lens_cpu = torch.tensor([0], dtype=torch.int32)
    out = build_prefill_chunk_metadata(
        start_idx=0,
        end_idx=1,
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 4], dtype=torch.int32),
        uncompressed_seq_lens=torch.tensor([4], dtype=torch.int32),
        compressed_seq_lens=compressed_seq_lens_cpu.clone(),
        compressed_seq_lens_cpu=compressed_seq_lens_cpu,
        block_table=torch.tensor([[0]], dtype=torch.int32),
        compress_ratio=8,
    )
    assert out is None


# ---------------------------------------------------------------------------
# DeepseekV32IndexerMetadataBuilder.__init__ on CPU
# ---------------------------------------------------------------------------


def test_indexer_metadata_builder_init_does_not_raise_on_cpu():
    """The GPU path calls ``num_compute_units(self.device.index)`` in
    ``__init__`` which raises NotImplementedError on CPU. The CPU branch
    must skip that and finish constructing the builder cleanly."""
    builder = _make_builder(compress_ratio=4)
    assert builder.num_sms == 0
    # All buffers should live on CPU.
    assert builder.decode_lens_buffer.device.type == "cpu"
    assert builder.expanded_block_table_buffer.device.type == "cpu"
    assert builder.scheduler_metadata_buffer.device.type == "cpu"
    # compress_ratio = 4 means the compressed_slot_mapping_buffer is
    # allocated.
    assert builder.compressed_slot_mapping_buffer.device.type == "cpu"


def test_indexer_metadata_builder_init_does_not_allocate_extra_buffers_for_v32():
    """For DeepSeek V3.2 (compress_ratio=1), ``compressed_slot_mapping_buffer``
    and ``expanded_seq_lens_buffer`` are not allocated (gated by
    ``self.compress_ratio > 1``). Verify the CPU path matches that gating."""
    builder = _make_builder(compress_ratio=1)
    assert builder.compress_ratio == 1
    assert not hasattr(builder, "compressed_slot_mapping_buffer")
    assert not hasattr(builder, "expanded_seq_lens_buffer")


# ---------------------------------------------------------------------------
# DeepseekV32IndexerMetadataBuilder.build() on CPU
# ---------------------------------------------------------------------------


def _make_common_attn_metadata(
    seq_lens_list: list[int],
    query_lens_list: list[int],
    block_table: torch.Tensor,
):
    """Build a CommonAttentionMetadata for CPU tests."""
    num_reqs = len(seq_lens_list)
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32)
    query_start_loc_cpu = torch.zeros(num_reqs + 1, dtype=torch.int32)
    query_start_loc_cpu[1:] = torch.cumsum(
        torch.tensor(query_lens_list, dtype=torch.int32), dim=0
    )
    num_tokens = int(query_start_loc_cpu[-1].item())
    return CommonAttentionMetadata(
        query_start_loc=query_start_loc_cpu.clone(),
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        num_reqs=num_reqs,
        num_actual_tokens=num_tokens,
        max_query_len=int(max(query_lens_list)),
        max_seq_len=int(max(seq_lens_list)),
        block_table_tensor=block_table,
        slot_mapping=torch.zeros(num_tokens, dtype=torch.int64),
        seq_lens_cpu_upper_bound=seq_lens.clone(),
    )


def test_indexer_metadata_builder_build_decode_only_compress_ratio_4():
    """Pure decode batch (next_n=1) with compress_ratio=4: builder
    should emit a DeepseekV32IndexerMetadata whose decode field has
    seq_lens converted to compressed (// 4) and shape (B, 1)."""
    builder = _make_builder(compress_ratio=4)
    block_table = torch.tensor(
        [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]] * 3,
        dtype=torch.int32,
    )
    common = _make_common_attn_metadata(
        seq_lens_list=[16, 32, 8],
        query_lens_list=[1, 1, 1],
        block_table=block_table,
    )
    meta = builder.build(common_prefix_len=0, common_attn_metadata=common)

    assert isinstance(meta, DeepseekV32IndexerMetadata)
    assert meta.num_decodes == 3
    assert meta.num_decode_tokens == 3
    assert meta.num_prefills == 0
    assert meta.prefill is None
    assert meta.decode is not None
    # Compressed seq lens: 16//4=4, 32//4=8, 8//4=2. Shape is (B, 1) for next_n=1.
    assert meta.decode.seq_lens.shape == (3, 1)
    assert meta.decode.seq_lens.flatten().tolist() == [4, 8, 2]
    # slot_mapping for compress_ratio>1 is always rewritten by
    # ``get_compressed_slot_mapping``: only the last token of each
    # multiple-of-compress_ratio block is valid; for query_len=1 at
    # position seq_len-1, only req 0 (pos=15, (15+1)%4==0) and req 1
    # (pos=31, (31+1)%4==0) are valid; req 2 (pos=7, (7+1)%4==0) is
    # also valid.
    sm = meta.slot_mapping
    # All three positions are multiples-of-4 - 1, so all valid.
    assert (sm[:3] >= 0).all()


def test_indexer_metadata_builder_build_prefill_only_compress_ratio_4():
    """Pure prefill batch with compress_ratio=4: builder should emit
    prefill chunks with token_start/token_end covering the full prefill
    and cu_seqlen_ks/ke matching the per-request offsets."""
    builder = _make_builder(compress_ratio=4)
    # 1 request, seq=8 (compressed=2), full prefill (query_len=8).
    block_table = torch.tensor([[100, 101, 102, 103]], dtype=torch.int32)
    common = _make_common_attn_metadata(
        seq_lens_list=[8],
        query_lens_list=[8],
        block_table=block_table,
    )
    meta = builder.build(common_prefix_len=0, common_attn_metadata=common)

    assert meta.num_prefills == 1
    assert meta.num_prefill_tokens == 8
    assert meta.num_decodes == 0
    assert meta.decode is None
    assert meta.prefill is not None
    assert len(meta.prefill.chunks) == 1
    chunk = meta.prefill.chunks[0]
    # 1 request, compressed_seq_len = 8 // 4 = 2.
    assert chunk.total_seq_lens == 2
    assert chunk.cu_seq_lens.tolist() == [0, 2]
    # Per-token (compress_ratio=4, query_len=8, start_pos=0):
    # ke[t] = ks + (t+1)//4 = [0, 0, 0, 1, 1, 1, 1, 2]
    expected_ke = [0, 0, 0, 1, 1, 1, 1, 2]
    assert chunk.cu_seqlen_ke.tolist() == expected_ke
    # ks[t] = 0 (only 1 request, so all tokens have ks = cu_seq_lens[0] = 0).
    assert chunk.cu_seqlen_ks.tolist() == [0] * 8
    assert chunk.num_reqs == 1
    assert chunk.token_start == 0
    assert chunk.token_end == 8
