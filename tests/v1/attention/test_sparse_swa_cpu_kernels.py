# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for the CPU torch ports of the two Triton kernels in
``vllm/v1/attention/backends/mla/sparse_swa.py`` (M3.2 part 12).

Background
----------
``DeepseekSparseSWAMetadataBuilder.build()`` calls two Triton kernels via
the ``kernel[(grid,)](...)`` indexing form:

* ``_compute_prefill_metadata_kernel`` (sparse_swa.py:430)
* ``_compute_swa_indices_and_lens_kernel`` (sparse_swa.py:333)

On CPU ``triton.jit`` is replaced by a no-op decorator (per
``vllm/triton_utils.py``) so the wrapped function stays a plain Python
function. The grid-indexing form then raises::

    TypeError: 'function' object is not subscriptable

This is the first error after part 11 unblocked the SWA builder
``pin_memory()`` issue. Part 12 ports both kernels to vectorized torch
in two module-level helpers:

* ``_cpu_compute_prefill_gather_lens``
* ``_cpu_compute_swa_indices_and_lens``

These tests drive each helper directly with hand-checked inputs and
compare against a Python reference that mirrors the Triton math
literally. The reference implementation is intentionally *not* the
helper under test — it is an independent loop-based formulation, so a
regression in either one is caught.
"""

from __future__ import annotations

import pytest
import torch

from vllm.v1.attention.backends.mla.sparse_swa import (
    _cpu_compute_prefill_gather_lens,
    _cpu_compute_swa_indices_and_lens,
)


pytestmark = pytest.mark.cpu_test


# ---------------------------------------------------------------------------
# _cpu_compute_prefill_gather_lens
# ---------------------------------------------------------------------------


def _ref_prefill_gather_lens(
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_prefills: int,
    num_decodes: int,
    window_size: int,
) -> torch.Tensor:
    """Independent Python loop reference, mirroring the Triton kernel
    formula at sparse_swa.py:451-475."""
    out = torch.empty(num_prefills, dtype=torch.int32)
    for i in range(num_prefills):
        d = num_decodes
        seq_len = int(seq_lens[d + i].item())
        qsl_start = int(query_start_loc[d + i].item())
        qsl_end = int(query_start_loc[d + i + 1].item())
        query_len = qsl_end - qsl_start
        prefix_len = seq_len - query_len
        gather_len = query_len + min(prefix_len, window_size - 1)
        out[i] = gather_len
    return out


def test_prefill_gather_lens_short_prefix_below_window() -> None:
    """Prefix smaller than (window - 1) -> gather_len = seq_len."""
    seq_lens = torch.tensor([100, 200, 50, 60], dtype=torch.int32)
    # query lengths: 100, 8, 50, 60 (decodes are the first 0)
    query_start_loc = torch.tensor([0, 100, 108, 158, 218], dtype=torch.int32)
    out = _cpu_compute_prefill_gather_lens(
        seq_lens, query_start_loc, num_prefills=4, num_decodes=0,
        window_size=4096,
    )
    ref = _ref_prefill_gather_lens(
        seq_lens, query_start_loc, 4, 0, 4096,
    )
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_prefill_gather_lens_long_prefix_clamped_to_window() -> None:
    """Prefix exceeds (window - 1) -> gather_len = query_len + (window - 1)."""
    # seq_len 8000, query_len 1000, prefix 7000, window 4096
    # gather_len = 1000 + min(7000, 4095) = 1000 + 4095 = 5095
    seq_lens = torch.tensor([8000], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 1000], dtype=torch.int32)
    out = _cpu_compute_prefill_gather_lens(
        seq_lens, query_start_loc, num_prefills=1, num_decodes=0,
        window_size=4096,
    )
    assert out.tolist() == [5095]


def test_prefill_gather_lens_with_decode_offset() -> None:
    """Decodes occupy the front; prefills start at offset = num_decodes."""
    seq_lens = torch.tensor([10, 20, 100, 4000], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 1, 2, 80, 4060], dtype=torch.int32)
    # num_decodes=2 (first two requests are decodes)
    # prefill 0: seq=100, qlen=78, pre=22, w-1=4095 -> gather=78+22=100
    # prefill 1: seq=4000, qlen=3980, pre=20, w-1=4095 -> gather=3980+20=4000
    out = _cpu_compute_prefill_gather_lens(
        seq_lens, query_start_loc, num_prefills=2, num_decodes=2,
        window_size=4096,
    )
    ref = _ref_prefill_gather_lens(
        seq_lens, query_start_loc, 2, 2, 4096,
    )
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert out.tolist() == [100, 4000]


def test_prefill_gather_lens_dtype_is_int32() -> None:
    seq_lens = torch.tensor([100], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 100], dtype=torch.int32)
    out = _cpu_compute_prefill_gather_lens(
        seq_lens, query_start_loc, 1, 0, 4096,
    )
    assert out.dtype == torch.int32


# ---------------------------------------------------------------------------
# _cpu_compute_swa_indices_and_lens
# ---------------------------------------------------------------------------


def _ref_swa_indices_and_lens(
    num_decode_tokens: int,
    window_size: int,
    block_size: int,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    is_valid_token: torch.Tensor,
    block_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent Python loop reference. Returns (swa_indices_flat,
    swa_lens) where swa_indices_flat is shape ``[num_decode_tokens, W]``
    int32 with -1 padding past swa_len."""
    W = window_size
    bs = block_size
    swa_lens = torch.zeros(num_decode_tokens, dtype=torch.int32)
    swa_idx = torch.full(
        (num_decode_tokens, W), -1, dtype=torch.int32
    )
    for t in range(num_decode_tokens):
        if not bool(is_valid_token[t].item()):
            continue
        r = int(token_to_req_indices[t].item())
        qs = int(query_start_loc[r].item())
        qe = int(query_start_loc[r + 1].item())
        qlen = qe - qs
        pre = int(seq_lens[r].item()) - qlen
        pos = pre + t - qs
        start = max(pos - W + 1, 0)
        end = pos + 1
        L = end - start
        swa_lens[t] = L
        for o in range(W):
            if o < L:
                po = start + o
                bn = int(block_table[r, po // bs].item())
                swa_idx[t, o] = bn * bs + (po % bs)
    return swa_idx, swa_lens


def _make_block_table(num_reqs: int, max_blocks: int, base: int = 100) -> torch.Tensor:
    """Each req gets a contiguous run of distinct block ids so slot id
    arithmetic is easy to verify."""
    bt = torch.empty(num_reqs, max_blocks, dtype=torch.int32)
    for r in range(num_reqs):
        for b in range(max_blocks):
            bt[r, b] = base + r * max_blocks + b
    return bt


def _run_helper(
    num_decode_tokens: int,
    window_size: int,
    block_size: int,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    is_valid_token: torch.Tensor,
    block_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drive the helper using buffers shaped like
    ``DeepseekSparseSWAMetadataBuilder``'s state."""
    max_tokens = max(num_decode_tokens, 1)
    swa_indices_buf = torch.full(
        (max_tokens, 1, window_size), -1, dtype=torch.int32
    )
    swa_lens_buf = torch.zeros(max_tokens, dtype=torch.int32)
    _cpu_compute_swa_indices_and_lens(
        swa_indices_buf,
        swa_lens_buf,
        num_decode_tokens,
        window_size,
        block_size,
        query_start_loc,
        seq_lens,
        token_to_req_indices,
        is_valid_token,
        block_table,
    )
    return (
        swa_indices_buf[:num_decode_tokens, 0, :],
        swa_lens_buf[:num_decode_tokens],
    )


def test_swa_indices_single_decode_token_within_window() -> None:
    """One req, decode token at pos < window: swa covers [0, pos+1)."""
    # window=4, block_size=4, one req with seq_len=3, query_len=1
    # decode token: t=0, r=0; qs=0, qe=1, qlen=1, pre=2, pos=2+0-0=2
    # start=max(2-4+1,0)=0, end=3, swa_len=3
    # pos_offsets 0,1,2 -> block_indices 0,0,0 -> block_numbers all 100
    # slots = 100*4+0, 100*4+1, 100*4+2 = 400, 401, 402
    block_table = _make_block_table(1, 4)
    out_idx, out_lens = _run_helper(
        num_decode_tokens=1,
        window_size=4,
        block_size=4,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([3], dtype=torch.int32),
        token_to_req_indices=torch.tensor([0], dtype=torch.int32),
        is_valid_token=torch.tensor([True]),
        block_table=block_table,
    )
    assert out_lens.tolist() == [3]
    assert out_idx[0].tolist() == [400, 401, 402, -1]


def test_swa_indices_window_truncates_long_history() -> None:
    """Pos beyond window: swa_len = window, slots come from later blocks."""
    # window=4, block_size=4, one req with seq_len=10, query_len=1
    # t=0, qs=0, qe=1, qlen=1, pre=9, pos=9, start=max(9-3,0)=6, end=10
    # swa_len=4, pos_offsets 6,7,8,9 -> block_indices 1,1,2,2
    # block_table[0,1]=101, block_table[0,2]=102
    # slots: 101*4+2=406, 101*4+3=407, 102*4+0=408, 102*4+1=409
    block_table = _make_block_table(1, 4)
    out_idx, out_lens = _run_helper(
        num_decode_tokens=1,
        window_size=4,
        block_size=4,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([10], dtype=torch.int32),
        token_to_req_indices=torch.tensor([0], dtype=torch.int32),
        is_valid_token=torch.tensor([True]),
        block_table=block_table,
    )
    assert out_lens.tolist() == [4]
    assert out_idx[0].tolist() == [406, 407, 408, 409]


def test_swa_indices_invalid_token_zero_length() -> None:
    """Invalid tokens get swa_len=0, swa_indices row stays -1."""
    block_table = _make_block_table(2, 4)
    out_idx, out_lens = _run_helper(
        num_decode_tokens=2,
        window_size=4,
        block_size=4,
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        seq_lens=torch.tensor([3, 5], dtype=torch.int32),
        token_to_req_indices=torch.tensor([0, 1], dtype=torch.int32),
        is_valid_token=torch.tensor([True, False]),
        block_table=block_table,
    )
    assert out_lens.tolist() == [3, 0]
    # Invalid row is all -1.
    assert out_idx[1].tolist() == [-1, -1, -1, -1]


def test_swa_indices_multi_request_against_reference() -> None:
    """Mixed decode tokens across requests: full vector vs python ref.

    DeepseekV4-Flash-BF16 has window=4096 and block_size=128 but the
    helper math is independent of those magnitudes; we use small values
    so the python reference loop is fast.
    """
    torch.manual_seed(0)
    num_reqs = 5
    max_blocks = 8
    block_size = 4
    window = 6
    block_table = _make_block_table(num_reqs, max_blocks)

    # Construct a heterogeneous batch: each request decodes 1 token,
    # mixed seq_lens, one invalid mid-batch.
    seq_lens = torch.tensor([3, 9, 15, 1, 6], dtype=torch.int32)
    # query_lens = [1, 1, 1, 1, 1]; cumsum starts at 0
    query_start_loc = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32)
    token_to_req = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32)
    is_valid = torch.tensor([True, True, True, False, True])

    out_idx, out_lens = _run_helper(
        num_decode_tokens=5,
        window_size=window,
        block_size=block_size,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        token_to_req_indices=token_to_req,
        is_valid_token=is_valid,
        block_table=block_table,
    )
    ref_idx, ref_lens = _ref_swa_indices_and_lens(
        5, window, block_size,
        query_start_loc, seq_lens, token_to_req, is_valid, block_table,
    )
    # Invalid token row is allowed to differ (helper writes -1, ref also
    # writes -1), but be strict because we *do* fully clear the slab.
    torch.testing.assert_close(out_lens, ref_lens, rtol=0, atol=0)
    torch.testing.assert_close(out_idx, ref_idx, rtol=0, atol=0)


def test_swa_indices_writes_int32_dtype() -> None:
    """Buffer dtype is int32 — check the in-place write keeps it."""
    block_table = _make_block_table(1, 4)
    swa_indices_buf = torch.full((1, 1, 4), -1, dtype=torch.int32)
    swa_lens_buf = torch.zeros(1, dtype=torch.int32)
    _cpu_compute_swa_indices_and_lens(
        swa_indices_buf,
        swa_lens_buf,
        1,
        4,
        4,
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([True]),
        block_table,
    )
    assert swa_indices_buf.dtype == torch.int32
    assert swa_lens_buf.dtype == torch.int32


def test_swa_indices_empty_returns_no_op() -> None:
    """num_decode_tokens=0 is a valid no-op: buffers untouched."""
    block_table = _make_block_table(1, 4)
    swa_indices_buf = torch.full((1, 1, 4), -1, dtype=torch.int32)
    swa_lens_buf = torch.zeros(1, dtype=torch.int32)
    _cpu_compute_swa_indices_and_lens(
        swa_indices_buf,
        swa_lens_buf,
        0,
        4,
        4,
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([], dtype=torch.int32),
        torch.tensor([], dtype=torch.int32),
        torch.tensor([], dtype=torch.bool),
        block_table,
    )
    # Helper must not raise. Buffers stay at sentinel state.
    assert swa_indices_buf.tolist() == [[[-1, -1, -1, -1]]]
    assert swa_lens_buf.tolist() == [0]
