# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for ``cpu_forward_decode`` on page-padded SWA / MLA caches.

Background
==========

vLLM's KV cache allocator (``vllm/v1/worker/gpu_model_runner.py:6750``)
builds the per-layer cache as a ``torch.as_strided`` view over a raw
byte tensor whenever ``kv_cache_spec.page_size_padded is not None``.
That happens on the DSV4 ARM CPU bf16 path because the SWA group's
compressor-state cache (fp32, large state_dim) drives the SWA group's
max page size **above** the full-MLA group's max page size, triggering
the cross-platform "pad full-MLA up" branch added in M3.2 part 9
(commit ``200500e5c``).

Result: ``swa_kv_cache.stride()`` becomes ``(page_padded, head_dim, 1)``
with ``page_padded > block_size * head_dim``. The tensor is **not
contiguous**, and any ``view(-1, head_dim)`` on it raises
``RuntimeError: view size is not compatible with input tensor's size
and stride``.

``cpu_forward_decode`` originally did

.. code-block:: python

    def _to_blocked(cache):
        return cache.unsqueeze(-2)
    blocked_swa = _to_blocked(swa_k_cache)
    rocm_ref_sparse_attn_decode(blocked_k=blocked_swa, ...)

and then ``rocm_ref_sparse_attn_decode``'s inner ``process_scope`` did

.. code-block:: python

    cur_blocked_k.view(-1, d_qk).index_select(0, fixed_indices.view(-1))

which crashes on the page-padded layout.

The fix gathers the cache by ``(block_idx, within_block)`` advanced
indexing in the CPU helper and feeds a contiguous gathered tensor to
the v0.22 CPU sparse attention helper. Advanced indexing on
``[num_blocks, block_size, head_dim]`` works on any stride layout, so
this is page-padded-safe.
"""

from __future__ import annotations

import inspect
import re

import pytest
import torch

from vllm.models.deepseek_v4 import cpu as cpu_mod

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_comments(src: str) -> str:
    return "\n".join(re.sub(r"#.*$", "", line) for line in src.splitlines())


def _make_padded_cache(
    num_blocks: int,
    block_size: int,
    head_dim: int,
    extra_pad_elems: int,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Build a page-padded ``[num_blocks, block_size, head_dim]`` cache.

    Mirrors what ``gpu_model_runner.py:6763`` produces when
    ``page_size_padded is not None``: the storage layout has
    ``stride[0] == page_padded > block_size * head_dim``, and the tensor
    is non-contiguous.

    The padding bytes are filled with garbage so any test that
    accidentally reads them via ``view(-1, head_dim)`` would see noise
    instead of zeros (helps catch silent fall-throughs).
    """
    page_padded = block_size * head_dim + extra_pad_elems
    raw = torch.randn(num_blocks * page_padded, dtype=dtype)
    cache = torch.as_strided(
        raw,
        size=(num_blocks, block_size, head_dim),
        stride=(page_padded, head_dim, 1),
    )
    assert not cache.is_contiguous(), "synthetic cache must be non-contiguous"
    return cache


def _make_contig_cache(
    num_blocks: int,
    block_size: int,
    head_dim: int,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Plain contiguous cache — sanity baseline (no page padding)."""
    return torch.randn(num_blocks, block_size, head_dim, dtype=dtype)


# ---------------------------------------------------------------------------
# Source-inspection guards
# ---------------------------------------------------------------------------


def test_cpu_forward_decode_does_not_use_unsqueeze_to_blocked() -> None:
    """The original buggy ``cache.unsqueeze(-2)`` pattern must be gone.

    ``unsqueeze`` does not allocate; it just inserts a size-1 dim into the
    same stride layout. On a page-padded (non-contiguous) cache the
    downstream ``view(-1, head_dim)`` then raises. The fix replaces this
    with a 2-D advanced-indexing gather.
    """
    src = _strip_comments(inspect.getsource(cpu_mod.cpu_forward_decode))
    # The exact buggy line was ``return cache.unsqueeze(-2)`` inside a
    # nested ``_to_blocked`` helper. Forbid both forms (with and without
    # the helper).
    assert "unsqueeze(-2)" not in src, (
        "cpu_forward_decode must not call unsqueeze(-2) on the raw cache: "
        "that was the page-padded view(-1, head_dim) crash root cause "
        "(M3.2 part 15 fix)."
    )


def test_cpu_forward_decode_uses_2d_advanced_indexing() -> None:
    """Fix uses ``cache[block_idx, within]`` 2-D advanced indexing.

    This is the ``page_size_padded``-safe gather pattern already in use by
    ``_cpu_save_partial_states`` (line ~509). The two key tokens we look
    for are integer-divide-by-block-size and modulo-by-block-size, which
    together unambiguously identify the 2-D index decomposition.
    """
    src = _strip_comments(inspect.getsource(cpu_mod.cpu_forward_decode))
    assert re.search(r"//\s*block_size", src) is not None, (
        "cpu_forward_decode must compute block_idx via floor-divide by "
        "block_size — required to gather without view(-1, head_dim)."
    )
    assert re.search(r"%\s*block_size", src) is not None, (
        "cpu_forward_decode must compute within-block offset via modulo "
        "by block_size — required to gather without view(-1, head_dim)."
    )


# ---------------------------------------------------------------------------
# Behavioural tests — drive the helper directly
# ---------------------------------------------------------------------------


def _run_cpu_forward_decode_swa_only(cache: torch.Tensor) -> torch.Tensor:
    """Tiny SWA-only invocation: 1 query, 1 head, fixed indices.

    Returns the output tensor (filled in place) so the caller can assert
    on the values.
    """
    num_blocks, block_size, head_dim = cache.shape
    batch = 1
    num_heads = 1

    # Pick three slot ids spread across blocks, plus one -1 sentinel to
    # exercise the invalid-mask path. Slot 0 of block 0 (=0), slot 5 of
    # block 0 (=5), slot 0 of block 1 (=block_size), and -1.
    swa_indices = torch.tensor([[[0, 5, block_size, -1]]], dtype=torch.int64)
    swa_lens = torch.tensor([3], dtype=torch.int64)  # 3 valid out of 4

    q = torch.randn(batch, num_heads, head_dim, dtype=torch.bfloat16)
    output = torch.zeros_like(q)

    cpu_mod.cpu_forward_decode(
        q=q,
        kv_cache=None,
        swa_k_cache=cache,
        swa_only=True,
        topk_indices=None,
        topk_lens=None,
        swa_indices=swa_indices,
        swa_lens=swa_lens,
        attn_sink=None,
        scale=1.0 / (head_dim ** 0.5),
        head_dim=head_dim,
        nope_head_dim=head_dim - 64,
        rope_head_dim=64,
        output=output,
    )
    return output


def test_cpu_forward_decode_runs_on_padded_swa_cache() -> None:
    """The exact production crash: page-padded SWA cache + decode forward.

    Pre-fix: ``RuntimeError: view size is not compatible with input
    tensor's size and stride`` raised inside ``rocm_ref_sparse_attn_decode``
    line 1039. Post-fix: returns finite output.
    """
    torch.manual_seed(0)
    cache = _make_padded_cache(
        num_blocks=4, block_size=64, head_dim=128, extra_pad_elems=200
    )
    out = _run_cpu_forward_decode_swa_only(cache)
    assert out.shape == (1, 1, 128)
    assert torch.isfinite(out).all(), "output must be finite"


def test_cpu_forward_decode_runs_on_contiguous_swa_cache() -> None:
    """Contiguous cache (no page padding) still works — no regression."""
    torch.manual_seed(0)
    cache = _make_contig_cache(num_blocks=4, block_size=64, head_dim=128)
    out = _run_cpu_forward_decode_swa_only(cache)
    assert out.shape == (1, 1, 128)
    assert torch.isfinite(out).all()


def test_cpu_forward_decode_padded_matches_contiguous() -> None:
    """Numerical: padded cache and contiguous cache with the same logical
    contents must produce the same attention output.

    Constructs a contiguous cache and a padded cache that share the same
    block contents (the padding bytes are zero in this test). Asserts
    output parity to within bf16 tolerance.
    """
    torch.manual_seed(0)
    num_blocks, block_size, head_dim = 4, 64, 128
    extra = 200

    # Build padded cache with ZERO padding bytes so logical contents match.
    page_padded = block_size * head_dim + extra
    raw = torch.zeros(num_blocks * page_padded, dtype=torch.bfloat16)
    padded = torch.as_strided(
        raw,
        size=(num_blocks, block_size, head_dim),
        stride=(page_padded, head_dim, 1),
    )
    # Fill the logical region with random data:
    payload = torch.randn(num_blocks, block_size, head_dim, dtype=torch.bfloat16)
    padded.copy_(payload)
    assert not padded.is_contiguous()

    # Contiguous twin with the same logical contents:
    contig = payload.clone()
    assert contig.is_contiguous()

    # Same Q + indices for both runs:
    swa_indices = torch.tensor([[[0, 5, block_size, -1]]], dtype=torch.int64)
    swa_lens = torch.tensor([3], dtype=torch.int64)
    q = torch.randn(1, 1, head_dim, dtype=torch.bfloat16)

    out_padded = torch.zeros_like(q)
    cpu_mod.cpu_forward_decode(
        q=q,
        kv_cache=None,
        swa_k_cache=padded,
        swa_only=True,
        topk_indices=None,
        topk_lens=None,
        swa_indices=swa_indices,
        swa_lens=swa_lens,
        attn_sink=None,
        scale=1.0 / (head_dim ** 0.5),
        head_dim=head_dim,
        nope_head_dim=head_dim - 64,
        rope_head_dim=64,
        output=out_padded,
    )

    out_contig = torch.zeros_like(q)
    cpu_mod.cpu_forward_decode(
        q=q,
        kv_cache=None,
        swa_k_cache=contig,
        swa_only=True,
        topk_indices=None,
        topk_lens=None,
        swa_indices=swa_indices,
        swa_lens=swa_lens,
        attn_sink=None,
        scale=1.0 / (head_dim ** 0.5),
        head_dim=head_dim,
        nope_head_dim=head_dim - 64,
        rope_head_dim=64,
        output=out_contig,
    )

    torch.testing.assert_close(
        out_padded.float(),
        out_contig.float(),
        rtol=5e-3,
        atol=5e-3,
        msg="padded-cache decode must match contiguous-cache decode",
    )


def test_cpu_forward_decode_runs_with_extra_padded_kv_cache() -> None:
    """C4A / C128A path: ``kv_cache`` is also page-paddable on CPU bf16.

    ``DeepseekV4MLAAttention.get_kv_cache_spec`` returns an
    ``MLAAttentionSpec`` that lives in the full-MLA group, which is the
    cohort that part 9 pads up. So both ``swa_k_cache`` AND ``kv_cache``
    can arrive non-contiguous in production. Verify the fix covers both
    arms.
    """
    torch.manual_seed(0)
    num_blocks, block_size, head_dim = 4, 64, 128
    swa_cache = _make_padded_cache(num_blocks, block_size, head_dim, 200)
    kv_cache = _make_padded_cache(num_blocks, block_size, head_dim, 240)

    batch = 1
    swa_indices = torch.tensor([[[0, 5, block_size, -1]]], dtype=torch.int64)
    swa_lens = torch.tensor([3], dtype=torch.int64)
    topk_indices = torch.tensor([[[0, 1, 2, -1]]], dtype=torch.int64)
    topk_lens = torch.tensor([3], dtype=torch.int64)

    q = torch.randn(batch, 1, head_dim, dtype=torch.bfloat16)
    output = torch.zeros_like(q)

    cpu_mod.cpu_forward_decode(
        q=q,
        kv_cache=kv_cache,
        swa_k_cache=swa_cache,
        swa_only=False,
        topk_indices=topk_indices,
        topk_lens=topk_lens,
        swa_indices=swa_indices,
        swa_lens=swa_lens,
        attn_sink=None,
        scale=1.0 / (head_dim ** 0.5),
        head_dim=head_dim,
        nope_head_dim=head_dim - 64,
        rope_head_dim=64,
        output=output,
    )
    assert torch.isfinite(output).all()


def test_cpu_forward_decode_invalid_indices_zero_out() -> None:
    """All-invalid input (every swa_indices == -1, swa_lens == 0) must
    produce zeros — both the padded and the contiguous code path agree.

    This pins down the ``-1`` sentinel handling in the gather/remap step:
    if the fix accidentally clamped negative indices and looked them up
    in cache row 0 without preserving the invalid mask, this test would
    show non-zero output.
    """
    torch.manual_seed(0)
    cache = _make_padded_cache(
        num_blocks=4, block_size=64, head_dim=128, extra_pad_elems=200
    )
    swa_indices = torch.tensor([[[-1, -1, -1, -1]]], dtype=torch.int64)
    swa_lens = torch.tensor([0], dtype=torch.int64)
    q = torch.randn(1, 1, 128, dtype=torch.bfloat16)
    output = torch.full_like(q, 9.0)  # sentinel: must be overwritten with 0

    cpu_mod.cpu_forward_decode(
        q=q,
        kv_cache=None,
        swa_k_cache=cache,
        swa_only=True,
        topk_indices=None,
        topk_lens=None,
        swa_indices=swa_indices,
        swa_lens=swa_lens,
        attn_sink=None,
        scale=1.0 / (128 ** 0.5),
        head_dim=128,
        nope_head_dim=64,
        rope_head_dim=64,
        output=output,
    )
    # No valid neighbours means the CPU sparse helper writes zeros.
    assert torch.all(output == 0), (
        "all-invalid decode must zero output, got max abs "
        f"{output.abs().max().item()}"
    )


def test_padded_cache_view_minus_1_head_dim_actually_fails() -> None:
    """Sanity: confirm the exact production crash on a page-padded cache.

    This is the negative control: shows that without the fix's gather,
    ``cache.unsqueeze(-2).view(-1, head_dim)`` does indeed raise on a
    page-padded layout. If torch ever changes its view rules to accept
    this, the regression test above (which forbids ``unsqueeze(-2)`` in
    the source) would still catch the bug, but this control documents
    *why* we forbid it.
    """
    cache = _make_padded_cache(
        num_blocks=4, block_size=64, head_dim=128, extra_pad_elems=200
    )
    blocked = cache.unsqueeze(-2)  # exact original buggy line
    with pytest.raises(RuntimeError, match="view size is not compatible"):
        blocked.view(-1, 128)
