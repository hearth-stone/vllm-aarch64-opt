# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-torch CPU fallbacks for DeepSeek V4 MLA attention.

DeepSeek V4 (Base / Flash / Pro) is shipped in vLLM as a SM100-tuned
implementation that depends on a stack of CUDA-only fused operators:

* ``fused_q_kv_rmsnorm``
* ``fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert`` (per-head Q-norm +
  GPT-J RoPE on q_pe + GPT-J RoPE on k_pe + UE8M0 FP8 quant + paged cache
  insert, all in one kernel)
* ``fused_inv_rope_fp8_quant`` + ``fp8_einsum`` (inverse RoPE on the MLA
  output then a quantized GEMM through ``wo_a``)
* ``flash_mla_with_kvcache`` (FlashMLA decode kernel with FP8 KV cache,
  attn_sink and sliding-window/topk gather)
* ``flash_mla_sparse_fwd`` (FlashMLA sparse prefill kernel)
* ``dequantize_and_gather_k_cache`` (FP8 KV cache → bf16 dense workspace)

This module is the **CPU-only torch port** of those kernels. It exists so
the ARM CPU bf16 path can run DeepSeek V4 end-to-end without touching
``flashmla`` / ``deep_gemm`` / ``triton`` / ``tilelang``.

Design notes:

1. **Numerical reference.** vLLM already ships a pure-torch reference for
   the *attention* side of these ops in
   ``vllm.v1.attention.ops.rocm_aiter_mla_sparse`` (the ``rocm_ref_*`` and
   ``_apply_*_ref`` family). Those functions are already platform-agnostic
   torch; this module re-uses them directly when possible. The only
   divergence between the ROCm path and CPU is the **KV cache layout**:
   ROCm keeps the cache as packed FP8 + UE8M0 scales (132 B / head with
   ``rocm_dequantize_blocked_k_cache`` doing the unpack); CPU keeps the
   cache as **bf16** (``[num_blocks, block_size, head_dim]``), so the
   dequant step is skipped.

2. **The Q/KV pre-attention fused op has no torch reference yet.** Both
   GPU and ROCm route the Q-head-norm + dual GPT-J RoPE + cache insert
   through ``torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert``
   (a C++ kernel). We provide the torch breakdown here:
   ``cpu_q_kv_rmsnorm`` + ``cpu_qnorm_rope_kv_rope_insert``.

3. **W8A8 / FP8 weights are out of scope.** The test target
   (``huihui-ai/DeepSeek-V4-Flash-BF16``) ships bf16 weights end-to-end;
   ``rocm_inv_rope_einsum`` is reused for the output projection and
   handles both quanted (``weight_scale_inv``) and bf16 ``wo_a`` cases.

4. **Milestone gating.** The helpers required for each milestone are
   tagged in section banners below. M1 helpers are fully implemented;
   M2 / M3 helpers raise ``NotImplementedError`` with pointers to where
   to fill them in (see ``docs/Deep seek v4 ARM CPU 推理待办.md`` and the
   plan at ``vllm-model-executor-models-deepseek-v4-inherited-cat``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, cast

import torch
import torch.nn.functional as F

from vllm.forward_context import get_forward_context
from vllm.v1.attention.backend import MultipleOf
from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseBackend

# These three helpers in ``rocm_aiter_mla_sparse`` are pure torch and
# already do exactly what CPU needs (modulo the FP8 cache layout, which
# we skip on CPU). Importing them on CPU is safe — the file's top-level
# triton imports go through ``vllm.triton_utils`` which provides a no-op
# placeholder when triton is unavailable.
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    _apply_inv_rope_ref,
    rocm_inv_rope_einsum,
)

if TYPE_CHECKING:
    from vllm.models.deepseek_v4.attention import DeepseekV4MLAAttention
    from vllm.v1.attention.backends.mla.flashmla_sparse import (
        FlashMLASparseMetadata,
    )
    from vllm.v1.attention.backends.mla.sparse_swa import (
        DeepseekSparseSWAMetadata,
    )

# ---------------------------------------------------------------------------
# M1: Q / KV pre-attention (RMSNorm + per-head RMSNorm + GPT-J RoPE + cache)
# ---------------------------------------------------------------------------


def _rmsnorm_native(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """Pure-torch RMSNorm matching ``vllm.layers.layernorm.RMSNorm.forward_native``.

    The reduction is done in fp32 for numerical stability and the result is
    cast back to ``x.dtype`` so downstream bf16 GEMMs see the expected dtype.
    """
    orig_dtype = x.dtype
    x_f = x.to(torch.float32)
    var = x_f.pow(2).mean(-1, keepdim=True)
    x_f = x_f * torch.rsqrt(var + eps)
    if weight is not None:
        x_f = x_f * weight.to(torch.float32)
    return x_f.to(orig_dtype)


def cpu_q_kv_rmsnorm(
    qr: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
    kv_lora_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU torch fallback for ``fused_q_kv_rmsnorm``.

    DeepSeek V4 produces ``qr_kv`` from ``fused_wqa_wkv`` and splits it into
    ``qr`` (q_lora_rank) and ``kv`` (kv_lora_rank + qk_rope_head_dim) before
    normalization. ``q_norm.weight`` covers the full ``qr``; ``kv_norm.weight``
    only covers the ``kv_lora_rank`` portion of ``kv`` — the trailing
    ``qk_rope_head_dim`` slice (``k_pe``) is **not** RMS-normalized here
    (it gets RoPE'd later in ``cpu_qnorm_rope_kv_rope_insert``).

    Args:
        qr:        ``[num_tokens, q_lora_rank]``
        kv:        ``[num_tokens, kv_lora_rank + qk_rope_head_dim]``
        q_weight:  ``[q_lora_rank]``
        kv_weight: ``[kv_lora_rank]``
        eps:       RMSNorm epsilon (typically ``config.rms_norm_eps``)
        kv_lora_rank: split offset between kv_c and k_pe inside ``kv``

    Returns:
        ``(qr_normed, kv_normed)`` where ``kv_normed`` has the ``kv_c``
        portion RMS-normalized and the ``k_pe`` portion left as-is.
    """
    qr_normed = _rmsnorm_native(qr, q_weight, eps)
    kv_c = kv[..., :kv_lora_rank]
    k_pe = kv[..., kv_lora_rank:]
    kv_c_normed = _rmsnorm_native(kv_c, kv_weight, eps)
    kv_normed = torch.cat([kv_c_normed, k_pe], dim=-1)
    return qr_normed, kv_normed


def cpu_q_kv_rmsnorm_no_k_pe(
    qr: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU RMSNorm for DeepSeek V4's current KV layout without ``k_pe``.

    Current DeepSeek V4 emits ``kv`` as the full latent vector, so there is no
    trailing RoPE key slice to preserve. Normalize ``qr`` and ``kv``
    independently with their own learned RMSNorm weights.
    """
    return (
        _rmsnorm_native(qr, q_weight, eps),
        _rmsnorm_native(kv, kv_weight, eps),
    )


def _per_head_rmsnorm_no_weight(
    q: torch.Tensor, eps: float
) -> torch.Tensor:
    """Per-head RMSNorm with no learnable weight.

    Equivalent to ``vllm.layers.layernorm.RMSNorm(has_weight=False)`` applied
    along the head_dim axis. DeepSeek V4 uses this on Q after ``wq_b`` and
    before RoPE; the GPU path embeds it inside
    ``fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert``.

    Args:
        q:   ``[num_tokens, num_heads, head_dim]``
        eps: RMSNorm epsilon (typically ``config.rms_norm_eps``)

    Returns:
        ``q_normed`` of the same shape and dtype as ``q``.
    """
    orig_dtype = q.dtype
    q_f = q.to(torch.float32)
    var = q_f.pow(2).mean(-1, keepdim=True)
    q_f = q_f * torch.rsqrt(var + eps)
    return q_f.to(orig_dtype)


def cpu_qnorm_rope_kv_rope_insert(
    q: torch.Tensor,
    kv: torch.Tensor,
    swa_kv_cache_2d: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    rotary_emb: torch.nn.Module,
    eps: float,
    kv_lora_rank: int,
    rope_head_dim: int,
    nope_head_dim: int,
) -> None:
    """CPU torch fallback for ``fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert``.

    Mutates ``q``, ``kv`` and ``swa_kv_cache_2d`` in place. Equivalent to:

      1. **Q side**: per-head RMSNorm (no weight) over ``head_dim``, then
         GPT-J RoPE on the trailing ``rope_head_dim`` slice (``q_pe``).
      2. **KV side**: GPT-J RoPE on the trailing ``rope_head_dim`` slice
         (``k_pe``).
      3. **Cache write**: scatter the (post-RoPE) ``[kv_c | k_pe]`` rows
         into ``swa_kv_cache_2d`` at positions ``slot_mapping``. **No FP8
         quantization** — CPU keeps the cache as bf16. Slots with
         ``slot_mapping == -1`` are skipped (padding for dummy/profile runs).

    Args:
        q:               ``[num_tokens, num_heads, head_dim]`` — modified
                         in place (full-head RMS + RoPE on q_pe).
        kv:              ``[num_tokens, kv_lora_rank + rope_head_dim]`` —
                         modified in place (RoPE on k_pe).
        swa_kv_cache_2d: ``[num_blocks * block_size, kv_lora_rank +
                         rope_head_dim]`` bf16 paged cache, flat-indexed
                         by ``slot_mapping``.
        slot_mapping:    ``[num_tokens]`` int64; ``-1`` = skip.
        positions:       ``[num_tokens]`` int64.
        rotary_emb:      A ``DeepseekV4ScalingRotaryEmbedding`` (or
                         compatible RoPE module) whose ``forward_native``
                         applies forward GPT-J RoPE.
        eps:             RMSNorm epsilon for the per-head Q norm.
        kv_lora_rank:    split offset between kv_c and k_pe in ``kv``
                         and in ``swa_kv_cache_2d``.
        rope_head_dim:   length of the rotary slice at the tail of each row.
        nope_head_dim:   length of the non-rotary slice at the head of
                         each Q row (``head_dim - rope_head_dim``).
    """
    # 1. Q per-head RMSNorm (no weight) over head_dim.
    q.copy_(_per_head_rmsnorm_no_weight(q, eps))

    # 2. GPT-J RoPE on q_pe and k_pe. ``rotary_emb.forward_native`` rotates
    #    *the trailing rotary_dim* of its inputs and broadcasts cos/sin
    #    against a head axis; both query and key must be 3-D
    #    ``[num_tokens, num_heads, head_dim]`` for the broadcast to fire.
    #    DeepSeek V4's KV side is a single-head latent (kv_lora_rank +
    #    qk_rope_head_dim per token), so we add a length-1 head axis on
    #    ``k_pe`` before the call and squeeze it out afterwards.
    q_pe = q[..., nope_head_dim:].contiguous()
    rope_dim = kv.shape[-1] - kv_lora_rank
    k_pe = kv[..., kv_lora_rank:].unsqueeze(1).contiguous()  # [T, 1, rope]
    if rope_dim > 0:
        q_pe_rot, k_pe_rot = rotary_emb.forward_native(
            positions.to(torch.int64), q_pe, k_pe
        )
        q[..., nope_head_dim:] = q_pe_rot
        kv[..., kv_lora_rank:] = k_pe_rot.squeeze(1)

    # 3. Scatter [kv_c | k_pe_rot] into the paged cache at slot_mapping.
    #    No FP8 quant: cache stays bf16.
    if swa_kv_cache_2d.numel() == 0:
        return
    flat_slots = slot_mapping.to(torch.int64).flatten()
    valid = flat_slots >= 0
    if valid.any():
        valid_slots = flat_slots[valid]
        # ``kv`` already holds the post-RoPE bf16 row ([kv_c | k_pe_rot]).
        kv_bs = kv.to(swa_kv_cache_2d.dtype)
        swa_kv_cache_2d[valid_slots] = kv_bs[valid]


# ---------------------------------------------------------------------------
# M1: KV cache gather (replace dequantize_and_gather_k_cache)
# ---------------------------------------------------------------------------


def cpu_dequantize_and_gather_k_cache(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor | None,
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
) -> None:
    """CPU torch fallback for ``dequantize_and_gather_k_cache``.

    The GPU op fuses (1) FP8 → bf16 dequant, (2) per-tile UE8M0 scale
    multiplication, and (3) per-sequence gather into a contiguous bf16
    workspace. CPU caches are stored as bf16 directly, so this function is
    a pure ``index_select`` gather.

    Args:
        out:          ``[chunk_size, M, head_dim]`` bf16, written in place.
                      The destination region is ``out[i, offset:offset +
                      gather_lens[i], :]`` (or ``offset:offset +
                      seq_lens[i]`` if ``gather_lens`` is ``None``).
        k_cache:      ``[num_blocks, block_size, head_dim]`` bf16.
        seq_lens:     ``[chunk_size]`` int — sequence length per request.
        gather_lens:  optional ``[chunk_size]`` int — only the first
                      ``gather_lens[i]`` tokens are gathered (used for the
                      sliding-window cache). When ``None``, gather the
                      full ``seq_lens[i]`` tokens.
        block_table:  ``[chunk_size, max_blocks]`` int — paged block ids.
        block_size:   tokens per block.
        offset:       write offset into ``out`` along the M axis.
    """
    if k_cache.numel() == 0:
        return
    head_dim = k_cache.shape[-1]
    chunk_size = seq_lens.shape[0]
    seq_lens_cpu = seq_lens.to("cpu")
    gather_lens_cpu = gather_lens.to("cpu") if gather_lens is not None else None

    for i in range(chunk_size):
        n_full = int(seq_lens_cpu[i].item())
        if n_full == 0:
            continue
        n_to_gather = (
            int(gather_lens_cpu[i].item()) if gather_lens_cpu is not None else n_full
        )
        if n_to_gather == 0:
            continue

        # Vectorized gather: pull all blocks needed to cover the prefix,
        # flatten, then truncate. Mirrors ``CPUMLAImpl._gather_kv_cache``.
        num_blocks = (n_full + block_size - 1) // block_size
        block_ids = block_table[i, :num_blocks]
        gathered = k_cache.index_select(0, block_ids)
        gathered = gathered.reshape(num_blocks * block_size, head_dim)
        # Take the most recent ``n_to_gather`` tokens (i.e. the last slice
        # of the sequence, which is what the SWA window keeps).
        start = max(0, n_full - n_to_gather)
        out[i, offset : offset + n_to_gather, :] = gathered[start:n_full].to(
            out.dtype
        )


# ---------------------------------------------------------------------------
# M1: Output projection (inverse RoPE + WO_A einsum)
# ---------------------------------------------------------------------------


def cpu_inv_rope_einsum(
    rotary_emb: torch.nn.Module,
    o: torch.Tensor,
    positions: torch.Tensor,
    rope_head_dim: int,
    n_local_groups: int,
    o_lora_rank: int,
    wo_a: torch.nn.Module,
) -> torch.Tensor:
    """CPU torch fallback for the wrapper's ``fused_inv_rope_fp8_quant`` +
    ``deepseek_v4_fp8_einsum`` output projection.

    Delegates to ``rocm_inv_rope_einsum`` which is already platform-
    agnostic torch: it (1) inverse-RoPEs ``o`` via
    ``rotary_emb.forward_native(..., inverse=True)``, (2) handles both
    quanted (``hasattr(wo_a, "weight_scale_inv")``) and bf16 ``wo_a``
    cases by dequantizing the weight on the fly when needed, and (3)
    contracts via ``einsum("tgd,grd->tgr", ...)``.

    The huihui-ai BF16 checkpoint stores ``wo_a.weight`` directly in bf16
    so the dequant branch is a no-op; the function still works correctly
    if a future CPU run-time meets a quantized checkpoint.
    """
    return rocm_inv_rope_einsum(
        rotary_emb=rotary_emb,
        o=o,
        positions=positions,
        rope_head_dim=rope_head_dim,
        n_local_groups=n_local_groups,
        o_lora_rank=o_lora_rank,
        wo_a=wo_a,
    )


# ---------------------------------------------------------------------------
# M1: Decode attention (sparse decode kernel — covers SWA-only too)
# ---------------------------------------------------------------------------


def _cpu_sparse_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor | None,
) -> torch.Tensor:
    """Small torch reference for DeepSeek V4 sparse MLA attention.

    ``indices`` may contain ``-1`` padding. ``attn_sink`` is modeled as one
    extra zero-value key per head with the provided logit bias, matching the
    FlashMLA sparse kernels' denominator effect.
    """
    if kv.ndim == 3:
        assert kv.shape[1] == 1
        kv = kv.squeeze(1)
    assert q.ndim == 3
    assert kv.ndim == 2

    num_tokens, num_heads, head_dim = q.shape
    out = torch.zeros_like(q, dtype=torch.float32)
    indices_2d = indices.reshape(num_tokens, -1).to(torch.long)
    sink = attn_sink[:num_heads].to(torch.float32) if attn_sink is not None else None

    for token_idx in range(num_tokens):
        valid_indices = indices_2d[token_idx]
        valid_indices = valid_indices[valid_indices >= 0]
        q_i = q[token_idx].to(torch.float32)
        if valid_indices.numel() == 0:
            if sink is not None:
                # All probability mass goes to the zero-value sink.
                continue
            continue

        k_i = kv.index_select(0, valid_indices).to(torch.float32)
        logits = torch.matmul(q_i, k_i.T) * scale
        if sink is not None:
            logits = torch.cat([logits, sink[:, None]], dim=-1)
        probs = torch.softmax(logits, dim=-1)
        out[token_idx] = torch.matmul(probs[..., : k_i.shape[0]], k_i)

    return out.to(q.dtype)


def cpu_forward_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor | None,
    swa_k_cache: torch.Tensor,
    swa_only: bool,
    topk_indices: torch.Tensor | None,
    topk_lens: torch.Tensor | None,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    attn_sink: torch.Tensor | None,
    scale: float,
    head_dim: int,
    nope_head_dim: int,
    rope_head_dim: int,
    output: torch.Tensor,
) -> None:
    """CPU torch fallback for ``flash_mla_with_kvcache`` (DeepSeek V4 decode).

    Replaces ``rocm_forward_decode_fallback`` for the CPU bf16 path.
    Mirrors the GPU kernel semantics: dense MLA decode over the SWA window
    (always present), optionally merged with a top-k indexed dense MLA
    decode over the compressed cache (``swa_only=False`` for C4A / C128A).

    The CPU cache is bf16 (no FP8 + UE8M0 packing), so the dequant step
    that ``rocm_forward_decode_fallback`` runs through
    ``rocm_dequantize_blocked_k_cache`` is **skipped** here — we feed the
    bf16 cache straight to ``rocm_ref_sparse_attn_decode`` which expects
    ``[num_blocks, block_size, 1, head_dim]`` bf16.

    Args:
        q:            ``[batch, num_heads, head_dim]`` (the wrapper has
                      already padded to ``padded_heads`` if needed).
        kv_cache:     ``[num_blocks, block_size, head_dim]`` bf16, or
                      ``None`` when ``swa_only`` is True.
        swa_k_cache:  ``[num_blocks, block_size, head_dim]`` bf16.
        swa_only:     If True, skip the top-k branch (M1 / C1 layers).
        topk_indices: ``[batch, 1, topk]`` int (M2/M3 only); ignored
                      when ``swa_only`` is True.
        topk_lens:    ``[batch]`` int (M2/M3 only).
        swa_indices:  ``[batch, swa_topk]`` int — gather indices into
                      ``swa_k_cache`` (already in flat-block layout).
        swa_lens:     ``[batch]`` int — valid length per query.
        attn_sink:    ``[padded_heads]`` fp32 -inf bias, or ``None``.
        scale:        softmax scale.
        head_dim:     full head_dim (kv_lora_rank + rope_head_dim).
        nope_head_dim:    qk_nope_head_dim (unused on CPU; kept for API
                          parity with ``rocm_forward_decode_fallback``).
        rope_head_dim:    qk_rope_head_dim (unused on CPU; kept for API
                          parity).
        output:       ``[batch, num_heads, head_dim]`` bf16, written in place.
    """
    del nope_head_dim, rope_head_dim  # parity-only

    # ---- Pre-gather both caches with 2-D advanced indexing -----------------
    #
    # Why we can't just hand the cache to ``rocm_ref_sparse_attn_decode``:
    # on the CPU bf16 path, ``swa_k_cache`` (and ``kv_cache``) often arrive
    # **non-contiguous**. The KV cache allocator
    # (``vllm/v1/worker/gpu_model_runner.py:6750``) builds them via
    # ``torch.as_strided`` whenever ``kv_cache_spec.page_size_padded is not
    # None`` — which always fires for DSV4-Flash-BF16 because the SWA
    # group's compressor-state cache (fp32, large state_dim) drives the SWA
    # max page size above the full-MLA max, triggering the cross-platform
    # "pad full-MLA up" branch in ``kv_cache_utils.py`` (M3.2 part 9).
    #
    # On a non-contiguous ``[num_blocks, block_size, head_dim]`` cache,
    # ``cache.view(-1, head_dim)`` raises ``RuntimeError: view size is not
    # compatible with input tensor's size and stride``. The original
    # implementation called ``cache.unsqueeze(-2)`` and handed the result
    # to ``rocm_ref_sparse_attn_decode``, which then ran
    # ``cur_blocked_k.view(-1, d_qk).index_select(...)`` internally — that
    # crashes on the page-padded layout.
    #
    # Fix: gather the cache here with the same 2-D index decomposition
    # ``_cpu_save_partial_states`` (line 509-518) uses on the write side:
    # ``block_idx = slot // bs``, ``within = slot % bs``,
    # ``cache[block_idx, within]``. Advanced indexing works on any stride,
    # so it is page-padded-safe. The result is contiguous, so the
    # downstream ``view(-1, d_qk)`` inside ``rocm_ref_sparse_attn_decode``
    # is trivially legal.
    #
    # We then remap ``swa_indices`` / ``topk_indices`` from "global slot
    # ids" to "positions within the gathered tensor" (a plain
    # ``arange`` reshaped per-token), keeping the ``-1`` sentinels intact
    # so ``rocm_ref_sparse_attn_decode``'s ``invalid_mask = cur_indices ==
    # -1`` continues to work.
    #
    # Cost: exactly one ``index_select``-equivalent gather per cache, same
    # work the inner ``index_select`` would have done anyway. No extra
    # full-cache copy.

    def _gather_and_remap(
        cache: torch.Tensor,
        global_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather a paged cache by 2-D advanced indexing and produce a
        ``(blocked_kv, faux_indices)`` pair compatible with
        ``rocm_ref_sparse_attn_decode``.

        Args:
            cache:           ``[num_blocks, block_size, head_dim]`` — may
                             be non-contiguous (``page_size_padded``).
            global_indices:  ``[batch, ..., topk]`` int — global slot ids
                             (``slot = block_idx * block_size + within``);
                             ``-1`` marks invalid entries.

        Returns:
            blocked_kv:    ``[batch, topk, 1, head_dim]`` contiguous bf16.
            faux_indices: ``[batch, ..., topk]`` int. Position
                          ``i * topk + t`` for valid entries, ``-1`` for
                          invalid (preserves the sentinel).
        """
        block_size = cache.shape[1]
        # Promote to int64 for safe ``//`` / ``%`` on bf16/int32 inputs.
        idx_i64 = global_indices.to(torch.int64)
        clamped = torch.clamp_min(idx_i64, 0)
        block_idx = clamped // block_size
        within = clamped % block_size
        # Advanced indexing — page-padded-safe, returns contiguous.
        gathered = cache[block_idx, within]   # [batch, ..., topk, head_dim]

        flat_shape = gathered.shape  # [batch, ..., topk, head_dim]
        # Reshape into ``[batch * ... * topk, 1, head_dim]`` then split off
        # the leading dim back into ``[batch, ..., topk, 1, head_dim]`` so
        # ``rocm_ref_sparse_attn_decode``'s ``view(-1, d_qk)`` sees the
        # leading dims as (num_blocks=batch*..., block_size=topk).
        # Concretely we keep the first dim as ``batch`` and roll the
        # remaining dims into ``topk_total``:
        batch = flat_shape[0]
        head_dim_local = flat_shape[-1]
        topk_total = 1
        for s in flat_shape[1:-1]:
            topk_total *= s
        blocked_kv = gathered.reshape(batch, topk_total, 1, head_dim_local)

        # Build a faux index tensor: position-in-gathered for valid
        # entries, ``-1`` for invalid. Layout matches ``global_indices``.
        pos_per_query = torch.arange(
            topk_total,
            device=idx_i64.device,
            dtype=idx_i64.dtype,
        )  # [topk_total]
        offset_per_batch = (
            torch.arange(batch, device=idx_i64.device, dtype=idx_i64.dtype)
            * topk_total
        ).view(batch, *([1] * (len(flat_shape) - 2)))
        faux = (pos_per_query.view(*([1] * (len(flat_shape) - 2)), topk_total)
                + offset_per_batch).view(*flat_shape[:-1])
        faux_indices = torch.where(
            idx_i64 >= 0, faux, torch.full_like(faux, -1)
        )
        return blocked_kv, faux_indices

    blocked_swa, faux_swa_indices = _gather_and_remap(
        swa_k_cache, swa_indices
    )
    blocked_extra: torch.Tensor | None = None
    faux_topk_indices: torch.Tensor | None = None
    if not swa_only:
        if kv_cache is None:
            raise AssertionError(
                "kv_cache is None but swa_only=False; expected a compressed "
                "KV cache for C4A / C128A decode."
            )
        # ``topk_indices`` arrives as ``[batch, 1, topk]`` (already
        # unsqueezed by the caller). The gather works on any leading
        # shape; the offsets we built above index into the gathered tensor
        # which has matching layout.
        topk_indices_for_gather = (
            topk_indices.squeeze(1) if topk_indices is not None else topk_indices
        )
        blocked_extra, faux_topk_indices = _gather_and_remap(
            kv_cache, topk_indices_for_gather
        )
        # Restore the ``[batch, 1, topk]`` shape that
        # ``rocm_ref_sparse_attn_decode`` expects on
        # ``extra_indices_in_kvcache``.
        faux_topk_indices = faux_topk_indices.unsqueeze(1)

    sink_for_q = attn_sink[: q.shape[1]] if attn_sink is not None else None

    kv_parts = [blocked_swa.reshape(-1, 1, head_dim)]
    # faux_swa_indices already has shape [batch, 1, topk] from decode_swa_indices
    # which is the expected [batch, 1, total_topk] for cat with topk_indices.
    index_parts = [faux_swa_indices]
    if blocked_extra is not None:
        assert faux_topk_indices is not None
        extra_base = kv_parts[0].shape[0]
        adjusted_topk_indices = torch.where(
            faux_topk_indices >= 0,
            faux_topk_indices + extra_base,
            faux_topk_indices,
        )
        kv_parts.append(blocked_extra.reshape(-1, 1, head_dim))
        index_parts.append(adjusted_topk_indices)

    attn_out = _cpu_sparse_attention(
        q=q,
        kv=torch.cat(kv_parts, dim=0),
        indices=torch.cat(index_parts, dim=-1),
        scale=scale,
        attn_sink=sink_for_q,
    )
    output.copy_(attn_out.to(output.dtype))


# ---------------------------------------------------------------------------
# M1: Sparse prefill (also serves SWA-only since prefill always goes through
#     the sparse machinery — top_k=0 / N=0 means "SWA window only")
# ---------------------------------------------------------------------------


def cpu_sparse_attn_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor | None,
    scale: float,
    head_dim: int,
    attn_sink: torch.Tensor | None,
    output: torch.Tensor,
) -> None:
    """CPU torch fallback for ``flash_mla_sparse_fwd``.

    Thin wrapper on ``rocm_ref_sparse_attn_prefill`` (already pure torch).
    Both the SWA-only path (top_k=0, indices = causal+window) and the
    sparse paths (C4A / C128A combined indices) flow through here.

    Mutates ``output`` in place.
    """
    out_chunk = _cpu_sparse_attention(
        q=q,
        kv=kv,
        indices=indices,
        scale=scale,
        attn_sink=attn_sink,
    )
    output.copy_(out_chunk.to(output.dtype))


# ---------------------------------------------------------------------------
# M2: DeepseekCompressor torch fallbacks (state save + fused compress kernel)
# ---------------------------------------------------------------------------


def _cpu_save_partial_states(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    positions: torch.Tensor,
    state_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    compress_ratio: int,
) -> None:
    """CPU torch port of ``_save_partial_states_kernel``.

    Writes the per-token (kv, score+APE) state into the compressor's paged
    state cache. Slots with ``slot_mapping == -1`` are skipped (PAD sentinel
    used by vLLM during dummy / profile runs).

    Layout note: for C4A (overlap=True, coff=2) the kv/score halves are
    each ``2*head_dim`` wide and pack two parallel sub-rows that the fused
    compressor kernel later reads as the two halves of its compress
    window. For C128A (overlap=False, coff=1) they are ``head_dim`` wide.
    This helper is layout-agnostic — it copies whatever width comes in.

    Args:
        kv:           ``[num_tokens, coff * head_dim]`` fp32 — kv state.
        score:        ``[num_tokens, coff * head_dim]`` fp32 — score state.
        ape:          ``[compress_ratio, coff * head_dim]`` fp32 — APE.
        positions:    ``[num_tokens]`` int.
        state_cache:  ``[num_blocks, block_size, 2 * coff * head_dim]`` fp32.
                      Last dim packs ``[kv_state | score_state]`` of equal
                      width (each ``coff * head_dim``).
        slot_mapping: ``[num_tokens]`` int; ``-1`` = skip.
        compress_ratio: per-layer compress ratio (4 for C4A, 128 for C128A).
    """
    if state_cache.numel() == 0:
        return
    flat_slots = slot_mapping.to(torch.int64)
    valid = flat_slots >= 0
    if not bool(valid.any()):
        return

    valid_slots = flat_slots[valid]
    block_size = state_cache.shape[1]
    state_width = state_cache.shape[-1] // 2

    block_idx = valid_slots // block_size
    pos_in_block = valid_slots % block_size

    kv_v = kv[valid].to(state_cache.dtype)
    score_v = score[valid].to(state_cache.dtype)
    ape_rows = (positions[valid].to(torch.int64) % compress_ratio).clamp_min(0)
    ape_v = ape[ape_rows].to(state_cache.dtype)

    state_cache[block_idx, pos_in_block, :state_width] = kv_v
    state_cache[block_idx, pos_in_block, state_width:] = score_v + ape_v


def _gptj_rope_apply_native(
    x: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    position,
    rope_head_dim: int,
) -> torch.Tensor:
    """Apply forward GPT-J style RoPE to the trailing ``rope_head_dim`` of ``x``.

    Mirrors the kernel's register-based RoPE in
    ``_fused_kv_compress_norm_rope_insert_sparse_attn`` (lines 189-208),
    ``_fused_kv_compress_norm_rope_insert_indexer_attn`` (lines 347-366),
    and the indexer-Q kernel ``_fused_indexer_q_rope_quant_kernel`` (lines
    104-119) / ``_fused_indexer_q_rope_mxfp4_kernel`` (lines 243-260).

    Layout assumptions matched verbatim:
      * ``cos_sin_cache[position]`` is ``[rope_head_dim]`` with the FIRST half
        cos values (length ``rope_head_dim // 2``) and the SECOND half sin
        values.
      * GPT-J style means *interleaved* even/odd pairs:
        ``new_even = even * cos - odd * sin`` /
        ``new_odd  = odd * cos  + even * sin`` with ``even = x[..., 0::2]``
        and ``odd = x[..., 1::2]`` over the rotary slice.
      * The rotary slice is the *trailing* ``rope_head_dim`` elements of ``x``;
        the leading ``head_dim - rope_head_dim`` elements pass through.

    Operates entirely in fp32 internally; the caller casts the result.

    Args:
        x:             tensor with trailing dim ``head_dim``. Two shapes are
                       supported:
                         * 1-D ``[head_dim]`` with scalar ``position`` (used by
                           the M2a per-token compressor loop).
                         * Arbitrary leading shape ``[..., head_dim]`` (e.g.
                           ``[T, H, head_dim]`` for indexer Q) with
                           ``position`` either a scalar or a 1-D tensor of
                           length ``T`` (broadcast over leading H).
        cos_sin_cache: ``[max_pos, rope_head_dim]`` fp32 with cos|sin halves.
        position:      python int / 0-D tensor / 1-D LongTensor of positions.
                       Caller is responsible for any compress-boundary
                       quantization (e.g. ``(position // ratio) * ratio``).
        rope_head_dim: number of trailing elements covered by the rotation.

    Returns:
        Same shape as ``x`` in fp32 with the rotated trailing slice.
    """
    head_dim = x.shape[-1]
    nope = head_dim - rope_head_dim
    half = rope_head_dim // 2
    x_f = x.to(torch.float32)
    out = x_f.clone()

    rope = x_f[..., nope:]  # [..., rope_head_dim]
    even = rope[..., 0::2]  # [..., half]
    odd = rope[..., 1::2]  # [..., half]

    cs = cos_sin_cache.to(torch.float32)
    if isinstance(position, torch.Tensor) and position.dim() >= 1:
        # Vectorized path. Gather rows in one shot, then unsqueeze enough
        # singleton dims so cos/sin broadcast against any leading shape
        # between T and head_dim (e.g. [T, H, head_dim] gets [T, 1, half]).
        cs_rows = cs[position]  # [T, rope_head_dim]
        extra_dims = x.dim() - position.dim() - 1
        for _ in range(extra_dims):
            cs_rows = cs_rows.unsqueeze(-2)
        cos_v = cs_rows[..., :half]
        sin_v = cs_rows[..., half : 2 * half]
    else:
        # Scalar path (preserves M2a per-token loop semantics exactly).
        cs_row = cs[position]  # [rope_head_dim]
        cos_v = cs_row[:half]
        sin_v = cs_row[half : 2 * half]

    new_even = even * cos_v - odd * sin_v
    new_odd = odd * cos_v + even * sin_v

    # Re-interleave back into [..., rope_head_dim].
    rotated = torch.empty_like(rope)
    rotated[..., 0::2] = new_even
    rotated[..., 1::2] = new_odd
    out[..., nope:] = rotated
    return out


def cpu_kv_compress_norm_rope_insert(
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    head_dim: int,
    compress_ratio: int,
    overlap: int,
    rope_head_dim: int,
) -> None:
    """CPU torch port of ``_fused_kv_compress_norm_rope_insert_*``.

    GPU has two specialized kernels:

    * ``_fused_kv_compress_norm_rope_insert_sparse_attn`` (head_dim=512, SPA;
      overlap=1, coff=2): compress window of ``2*compress_ratio`` tokens.
      The state-cache row layout stores TWO parallel halves of width
      ``coff*head_dim = 2*head_dim`` each (``[kv_part1 | kv_part2 |
      score_part1 | score_part2]``). The first half of the window reads
      kv from kv_part1 / score from score_part1; the second half reads
      kv_part2 / score_part2. RMSNorm → GPT-J RoPE on rope tail → FP8
      quant of NoPE + bf16 store of RoPE → packed cache (132 B/head).
    * ``_fused_kv_compress_norm_rope_insert_indexer_attn`` (head_dim=128;
      overlap=0, coff=1): compress window of just ``compress_ratio``
      tokens. Single-half row layout (``[kv | score]``, each ``head_dim``
      wide). RMSNorm → GPT-J RoPE on rope tail → FP8 quant whole row →
      132 B/head layout.

    On CPU we drop FP8 quant entirely and store the post-RoPE bf16 row
    directly into a flat ``[num_blocks, block_size, head_dim]`` cache. The
    only structural difference between the two kernels is the quant tail,
    so they collapse to one torch implementation.

    Early-out semantics match the GPU kernel exactly:
      * skip if ``slot_mapping[i] == -1`` (state-cache write skipped);
      * skip if ``(positions[i] + 1) % compress_ratio != 0`` (not a
        compress boundary — only boundary tokens write the K cache);
      * skip if ``kv_slot_mapping[i] == -1`` (target K-cache slot is PAD).

    Args:
        state_cache:         ``[num_blocks, block_size, 2*coff*head_dim]``
                             fp32 paged compressor state cache.
        token_to_req_indices: ``[num_tokens]`` int — request idx per token.
        positions:           ``[num_tokens]`` int.
        slot_mapping:        ``[num_tokens]`` int — STATE cache slot per
                             token.
        block_table:         ``[num_reqs, max_blocks]`` int — paged block
                             ids into ``state_cache``.
        block_size:          tokens per block in ``state_cache``.
        rms_norm_weight:     ``[head_dim]`` (any float).
        rms_norm_eps:        float.
        cos_sin_cache:       ``[max_pos, rope_head_dim]`` fp32 (cos | sin
                             halves).
        kv_cache:            ``[num_blocks, block_size, head_dim]`` bf16
                             paged K cache (the compressor's *output*
                             cache). Written in place at boundary tokens.
        kv_slot_mapping:     ``[num_tokens]`` int — slot per token in
                             ``kv_cache``; ``-1`` = skip.
        head_dim:            cache head_dim (512 for SPA, 128 for indexer).
        compress_ratio:      4 (SPA) or 128 (indexer).
        overlap:             1 for SPA (window is ``2*compress_ratio``);
                             0 for indexer (window is ``compress_ratio``).
        rope_head_dim:       trailing rotary slice (e.g. 64).
    """
    if kv_cache.numel() == 0 or state_cache.numel() == 0:
        return

    num_tokens = positions.shape[0]
    if num_tokens == 0:
        return

    # GPU kernel iterates per-token and early-exits in 3 places. We mirror
    # the same control flow on CPU. Vectorising would be preferable for
    # perf, but only a small fraction of tokens hit the boundary so a
    # python loop over `num_tokens` is fine for correctness-first work.
    pos_cpu = positions.to(torch.int64).to("cpu")
    slot_cpu = slot_mapping.to(torch.int64).to("cpu")
    kv_slot_cpu = kv_slot_mapping.to(torch.int64).to("cpu")
    req_idx_cpu = token_to_req_indices.to(torch.int64).to("cpu")

    state_block_size = state_cache.shape[1]
    state_dim = state_cache.shape[-1]
    state_width = state_dim // 2  # coff * head_dim
    coff = state_width // head_dim
    assert coff in (1, 2), (
        f"DeepseekCompressor state_cache last-dim layout implies coff=1 or 2; "
        f"got state_dim={state_dim}, head_dim={head_dim} → coff={coff}"
    )

    kv_cache_block_size = kv_cache.shape[1]
    window = (1 + overlap) * compress_ratio

    rms_w_f = rms_norm_weight.to(torch.float32)

    for i in range(num_tokens):
        slot_id = int(slot_cpu[i].item())
        if slot_id < 0:
            continue
        position = int(pos_cpu[i].item())
        if (position + 1) % compress_ratio != 0:
            continue
        kv_slot_idx = int(kv_slot_cpu[i].item())
        if kv_slot_idx < 0:
            continue
        req_idx = int(req_idx_cpu[i].item())

        # Gather window rows.
        #
        # For overlap=1 (SPA / coff=2): first ``compress_ratio`` rows
        # take kv from row[0:head_dim] (kv_part1) and score from
        # row[2*head_dim:3*head_dim] (score_part1). The next
        # ``compress_ratio`` rows take kv from row[head_dim:2*head_dim]
        # (kv_part2) and score from row[3*head_dim:4*head_dim]
        # (score_part2). Both halves are parallel "sub-windows" stored at
        # the same physical token position, doubling the effective
        # window without doubling the cache footprint.
        #
        # For overlap=0 (indexer / coff=1): all rows take kv from
        # row[0:head_dim] and score from row[head_dim:2*head_dim].
        start = position - window + 1
        rows_kv: list[torch.Tensor] = []
        rows_score: list[torch.Tensor] = []
        for t in range(window):
            p = start + t
            if p < 0:
                # Out-of-range row: kv = 0, score = -inf (masked off in
                # softmax — kv coefficient ends up zero unless ALL window
                # rows are -inf, in which case softmax becomes 1/N for
                # numerical stability and kv stays 0 → compressed = 0).
                rows_kv.append(
                    torch.zeros(
                        head_dim, dtype=torch.float32, device=state_cache.device
                    )
                )
                rows_score.append(
                    torch.full(
                        (head_dim,),
                        float("-inf"),
                        dtype=torch.float32,
                        device=state_cache.device,
                    )
                )
                continue
            blk = p // state_block_size
            off = p % state_block_size
            blk_no = int(block_table[req_idx, blk].item())
            row = state_cache[blk_no, off]  # [state_dim] fp32
            if overlap > 0 and t >= compress_ratio:
                # Second half of the window: read part2 of kv & score.
                kv_row = row[head_dim : 2 * head_dim].to(torch.float32)
                score_row = row[
                    state_width + head_dim : state_width + 2 * head_dim
                ].to(torch.float32)
            else:
                # First half (or single-half for overlap=0).
                kv_row = row[0:head_dim].to(torch.float32)
                score_row = row[state_width : state_width + head_dim].to(
                    torch.float32
                )
            rows_kv.append(kv_row)
            rows_score.append(score_row)

        kv_stack = torch.stack(rows_kv, dim=0)  # [window, head_dim]
        score_stack = torch.stack(rows_score, dim=0)  # [window, head_dim]

        # If every score row is -inf (e.g. partial-window-at-start with
        # masked first half and no score loaded for the second half on a
        # buffer-OOB GPU read), softmax produces NaN. Replace such
        # all-masked columns with uniform 1/window weights, which when
        # combined with kv_stack=0 still gives compressed=0 — matching
        # the kernel's effective behaviour. The valid-row path is
        # unaffected because in that case at least one score is finite.
        with torch.no_grad():
            all_neg_inf = torch.isneginf(score_stack).all(dim=0, keepdim=True)
        if bool(all_neg_inf.any()):
            score_stack = torch.where(
                all_neg_inf.expand_as(score_stack),
                torch.zeros_like(score_stack),
                score_stack,
            )

        weights = torch.softmax(score_stack, dim=0)  # over window axis
        compressed = (kv_stack * weights).sum(dim=0)  # [head_dim] fp32

        # RMSNorm with weight (fp32 throughout).
        var = compressed.pow(2).mean(-1, keepdim=False)
        rrms = torch.rsqrt(var + rms_norm_eps)
        normed = compressed * rrms * rms_w_f  # [head_dim] fp32

        # Forward GPT-J RoPE on the trailing rope_head_dim, position
        # quantized to the compress boundary (matches kernel's
        # ``compressed_pos = (position // ratio) * ratio``).
        compressed_pos = (position // compress_ratio) * compress_ratio
        rotated = _gptj_rope_apply_native(
            normed, cos_sin_cache, compressed_pos, rope_head_dim
        )

        # Store the post-RoPE bf16 row into the flat CPU cache layout
        # [num_blocks, block_size, head_dim].
        kv_block_idx = kv_slot_idx // kv_cache_block_size
        kv_pos_in_block = kv_slot_idx % kv_cache_block_size
        kv_cache[kv_block_idx, kv_pos_in_block, :] = rotated.to(kv_cache.dtype)


# ---------------------------------------------------------------------------
# M2 stubs — sparse Indexer + topk merge
# ---------------------------------------------------------------------------


_M2_TODO = (
    "Not implemented yet — required for compress_ratio=4 (C4A) layers. "
    "See plan ``vllm-model-executor-models-deepseek-v4-inherited-cat`` "
    "section M2 for the spec."
)


def cpu_indexer_q_rope_quant(
    positions: torch.Tensor,
    index_q: torch.Tensor,
    index_q_cos_sin_cache: torch.Tensor,
    index_weights: torch.Tensor,
    index_weights_softmax_scale: float,
    index_weights_head_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU torch port of ``fused_indexer_q_rope_quant`` (bf16, no quant).

    GPU has two paths (``vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py``):
      * FP8 (``_fused_indexer_q_rope_quant_kernel``):
        ``weights_out = weights * q_scale * softmax_scale * head_scale``
        — per-token Q scale folded into weights because FP8 Q has no scale
        tensor.
      * MXFP4 (``_fused_indexer_q_rope_mxfp4_kernel``):
        ``weights_out = weights * softmax_scale * head_scale`` — per-block
        scales travel with the Q values, so weights carry only the constant
        scales.

    On CPU we keep Q in bf16 and do no quantization, so there is no
    per-token Q scale to fold — we follow the MXFP4 contract:
    ``weights_out = weights * softmax_scale * head_scale``.

    GPT-J interleaved RoPE is applied to the trailing ``rope_dim`` of each
    head; the leading ``head_dim - rope_dim`` (NoPE) passes through
    unchanged. ``rope_dim = index_q_cos_sin_cache.shape[-1]`` (matches the
    kernel's ``INDEX_Q_HALF_ROT_DIM = cos_sin_cache.shape[-1] // 2`` then
    doubled).

    Args:
        positions:                    [T] int64.
        index_q:                      [T, H, head_dim] (bf16/fp32).
        index_q_cos_sin_cache:        [max_pos, rope_dim] (cos|sin halves).
        index_weights:                [T, H] (bf16/fp32).
        index_weights_softmax_scale:  scalar float.
        index_weights_head_scale:     scalar float.

    Returns:
        (q_out, weights_out)
          * q_out: [T, H, head_dim] bf16, RoPE-applied.
          * weights_out: [T, H] fp32 (matches GPU output dtype — see
            ``index_weights_out = torch.empty_like(index_weights, dtype=fp32)``
            in fused_indexer_q.py:329).
    """
    assert positions.dim() == 1
    assert index_q.dim() == 3
    assert index_q_cos_sin_cache.dim() == 2
    assert index_weights.dim() == 2

    rope_dim = index_q_cos_sin_cache.shape[-1]

    # GPT-J RoPE on the trailing rope_dim of [T, H, head_dim]. The helper
    # broadcasts cos/sin from [T, rope_dim/2] over the [H] axis automatically.
    q_rot_f32 = _gptj_rope_apply_native(
        index_q, index_q_cos_sin_cache, positions, rope_dim
    )

    # Match the FP8/MXFP4 kernels' bf16 round-trip on the rotated values
    # (fused_indexer_q.py:123-124, 259-260). Easiest way: cast the whole
    # tensor; the NoPE pass-through is already in fp32 so casting it to
    # bf16 only loses precision once, the same as the FP8 kernel which
    # also stores NoPE through fp8 (we just store bf16 instead).
    q_out = q_rot_f32.to(torch.bfloat16)

    weights_out = (
        index_weights.to(torch.float32)
        * float(index_weights_softmax_scale)
        * float(index_weights_head_scale)
    )

    return q_out, weights_out


def cpu_sparse_attn_indexer_op(
    q_quant: torch.Tensor,
    weights: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_indices_buffer: torch.Tensor,
    topk_tokens: int,
    attn_metadata,
) -> torch.Tensor:
    """CPU torch fallback for ``SparseAttnIndexer.forward``.

    Mirrors the GPU body in ``vllm/model_executor/layers/sparse_attn_indexer.py``
    (``sparse_attn_indexer`` free function) for the bf16 CPU path:

      * For each prefill chunk, gather K from the indexer's paged KV cache
        per request, compute per-token logits ``logits[t, s] =
        sum_h weights[t, h] * (q[t, h] @ k[s])`` for ``s`` within
        ``[cu_seqlen_ks[t], cu_seqlen_ke[t])``, and take ``topk_tokens``
        along the seq axis. Padded slots stay -1.
      * For each decode token, do the same against its paged K rows for
        positions ``[0, seq_len[t])`` (causal, full prefix).

    The CPU path keeps Q in bf16 (no FP8/FP4 quant), and the indexer K
    cache is pure bf16 ``[num_blocks, block_size, head_dim]`` (no scale
    padding); ``DeepseekV4IndexerCache.get_kv_cache_spec`` and
    ``DeepseekV4Indexer.__init__`` produce that layout on CPU.

    Args:
        q_quant:               [num_tokens, n_head, head_dim] bf16 (RoPE'd).
        weights:               [num_tokens, n_head] fp32 (already folded with
                               softmax_scale * head_scale by
                               ``cpu_indexer_q_rope_quant``).
        kv_cache:              [num_blocks, block_size, head_dim] bf16. The
                               compressor's CPU path
                               (``cpu_kv_compress_norm_rope_insert``) writes
                               directly into this tensor since
                               ``skip_k_cache_insert=True`` on the indexer.
        topk_indices_buffer:   [max_tokens, K_max] int32. The indexer writes
                               local block-relative indices per token here;
                               downstream
                               ``compute_global_topk_indices_and_lens``
                               translates them to global slot ids.
        topk_tokens:           int. Number of top entries per token.
        attn_metadata:         ``DeepseekV32IndexerMetadata`` for this layer.

    Returns:
        ``topk_indices_buffer`` (mutated in place).
    """
    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerMetadata,
    )

    assert isinstance(attn_metadata, DeepseekV32IndexerMetadata)

    num_tokens = q_quant.shape[0]
    head_dim = q_quant.shape[-1]
    block_size = kv_cache.shape[1]

    # Reset buffer rows for this forward (mirrors GPU
    # ``topk_indices_buffer[:hidden_states.shape[0]] = -1``).
    topk_indices_buffer[:num_tokens] = -1

    has_decode = attn_metadata.num_decodes > 0
    has_prefill = attn_metadata.num_prefills > 0
    num_decode_tokens = attn_metadata.num_decode_tokens

    # Pre-fold weights into Q to make the per-token logit a plain inner
    # product over heads × head_dim. logits[t, s] =
    #   sum_h w[t, h] * sum_d q[t, h, d] * k[s, d]
    # = (q[t] * w[t, :, None]).sum(0) @ k[s]
    q_w = (q_quant.to(torch.float32) * weights.to(torch.float32).unsqueeze(-1)).sum(
        dim=1
    )  # [num_tokens, head_dim]

    if has_prefill:
        prefill_metadata = attn_metadata.prefill
        assert prefill_metadata is not None
        for chunk in prefill_metadata.chunks:
            token_start = chunk.token_start
            token_end = chunk.token_end
            num_chunk_tokens = token_end - token_start
            if num_chunk_tokens == 0:
                continue

            # Gather K rows for all requests in this chunk into a single
            # contiguous [total_seq_lens, head_dim] tensor.
            cu_seq_lens_cpu = chunk.cu_seq_lens.to("cpu")
            block_table_cpu = chunk.block_table.to("cpu")
            num_reqs = chunk.num_reqs
            total_seq_lens = int(chunk.total_seq_lens)
            k_gathered = torch.empty(
                (total_seq_lens, head_dim),
                dtype=torch.float32,
                device=q_quant.device,
            )
            for r in range(num_reqs):
                ks = int(cu_seq_lens_cpu[r].item())
                ke = int(cu_seq_lens_cpu[r + 1].item())
                seq_len_r = ke - ks
                if seq_len_r == 0:
                    continue
                num_blocks = (seq_len_r + block_size - 1) // block_size
                block_ids = block_table_cpu[r, :num_blocks].to(torch.long)
                gathered = kv_cache.index_select(0, block_ids).reshape(
                    num_blocks * block_size, head_dim
                )
                k_gathered[ks:ke] = gathered[:seq_len_r].to(torch.float32)

            cu_seqlen_ks_cpu = chunk.cu_seqlen_ks.to("cpu")
            cu_seqlen_ke_cpu = chunk.cu_seqlen_ke.to("cpu")
            q_w_chunk = q_w[token_start:token_end]  # [Tc, head_dim]
            # Full logits for the chunk: [Tc, total_seq_lens]. Memory bound
            # by VLLM_SPARSE_INDEXER_MAX_LOGITS_MB on the GPU; on CPU we
            # follow the same chunk split so this stays under that limit.
            logits = F.linear(q_w_chunk, k_gathered)  # [Tc, total_seq_lens]
            for i in range(num_chunk_tokens):
                ks_i = int(cu_seqlen_ks_cpu[i].item())
                ke_i = int(cu_seqlen_ke_cpu[i].item())
                valid_len = ke_i - ks_i
                if valid_len <= 0:
                    continue
                row = logits[i, ks_i:ke_i]
                k_take = min(topk_tokens, valid_len)
                _, idx_local = torch.topk(row, k_take, dim=-1)
                # Indices are RELATIVE to ks_i (matching GPU
                # ``top_k_per_row_prefill`` which writes
                # ``cu_seqlen_ks[t] + col`` style local indices into the
                # buffer; downstream ``compute_global_topk_indices_and_lens``
                # consumes them as cache-relative slots).
                topk_indices_buffer[token_start + i, :k_take] = idx_local.to(
                    torch.int32
                )

    if has_decode:
        decode_metadata = attn_metadata.decode
        assert decode_metadata is not None
        block_table_d = decode_metadata.block_table.to("cpu").to(torch.long)
        seq_lens_d = decode_metadata.seq_lens.to("cpu")
        if seq_lens_d.dim() == 2:
            # native MTP path: (B, next_n) — flatten to per-token contexts.
            seq_lens_per_token = seq_lens_d.reshape(-1)
        else:
            seq_lens_per_token = seq_lens_d
        # Build a per-token expanded block table when MTP padded the
        # decode batch: GPU's ``_prepare_decode_tensors`` already handles
        # this; here we collapse by mirroring the same expansion.
        # decode_lens drives the per-request copy count. For
        # next_n=1 (no spec), this collapses to identity.
        decode_lens_cpu = decode_metadata.decode_lens.to("cpu")
        # If the builder produced per-token rows already (flatten path),
        # block_table_d.shape[0] == num_decode_tokens. Otherwise it is
        # num_decodes and we need to expand.
        if block_table_d.shape[0] == num_decode_tokens:
            per_token_block_table = block_table_d
        else:
            per_token_block_table = torch.repeat_interleave(
                block_table_d, decode_lens_cpu.to(torch.long), dim=0
            )

        for t in range(num_decode_tokens):
            seq_len_t = int(seq_lens_per_token[t].item())
            if seq_len_t <= 0:
                continue
            num_blocks = (seq_len_t + block_size - 1) // block_size
            block_ids = per_token_block_table[t, :num_blocks]
            gathered = kv_cache.index_select(0, block_ids).reshape(
                num_blocks * block_size, head_dim
            )
            k_t = gathered[:seq_len_t].to(torch.float32)  # [seq_len_t, head_dim]
            row = F.linear(q_w[t], k_t)  # [seq_len_t]
            k_take = min(topk_tokens, seq_len_t)
            _, idx_local = torch.topk(row, k_take, dim=-1)
            topk_indices_buffer[t, :k_take] = idx_local.to(torch.int32)

    return topk_indices_buffer


def cpu_compute_global_topk_indices_and_lens(
    topk_indices: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    is_valid_token: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU torch port of ``compute_global_topk_indices_and_lens``.

    Mirrors ``_compute_global_topk_indices_and_lens_kernel`` (cache_utils.py
    :386-434). The kernel does three things per token:

      1. Block-table lookup: turn each local index into a global slot id::

             slot = block_table[req, idx // block_size] * block_size
                    + (idx % block_size)

         applied only where ``idx >= 0`` (sentinel for empty topk slots).

      2. Count valid entries per token (``topk_lens``).

      3. Zero out ``topk_lens`` for padding tokens (``is_valid_token == 0``).

    Args:
        topk_indices:          [T, K] int. Negative values mark empty slots.
        token_to_req_indices:  [T] int. Maps each token to its request row.
        block_table:           [num_reqs, max_blocks_per_seq] int.
        block_size:            int. Cache block size (e.g. 64).
        is_valid_token:        [T] bool/int. Padding mask (1 = real token).

    Returns:
        (global_topk_indices, topk_lens):
          * global_topk_indices: [T, K] int with the same sentinel rules.
          * topk_lens:           [T] int32, 0 where ``is_valid_token == 0``.
    """
    assert topk_indices.dim() == 2
    assert token_to_req_indices.dim() == 1
    assert block_table.dim() == 2
    assert is_valid_token.dim() == 1

    is_valid = topk_indices >= 0
    safe_idx = topk_indices.clamp_min(0)
    block_idx = safe_idx // block_size
    block_off = safe_idx % block_size

    # Gather block numbers from the block table along the per-token req row.
    # block_table is indexed as block_table[req_idx, block_idx] → [T, K].
    req_idx = token_to_req_indices.to(torch.long).unsqueeze(-1).expand_as(block_idx)
    block_numbers = block_table[req_idx, block_idx.to(torch.long)].to(
        topk_indices.dtype
    )

    slot_ids = block_numbers * block_size + block_off
    sentinel = torch.full_like(slot_ids, -1)
    global_topk_indices = torch.where(is_valid, slot_ids, sentinel)

    # topk_lens: count is_valid per row, zeroed for padding tokens.
    counts = is_valid.sum(dim=-1).to(torch.int32)
    topk_lens = torch.where(
        is_valid_token.to(torch.bool),
        counts,
        torch.zeros_like(counts),
    )

    return global_topk_indices, topk_lens


# Mirror of ``vllm.v1.attention.ops.deepseek_v4_ops.cache_utils
# ._SPARSE_PREFILL_TOPK_ALIGNMENT``. FlashMLA Sparse prefill asserts
# ``params.topk % B_TOPK == 0`` and B_TOPK is 64 (h_q=64) or 128 (h_q=128);
# 128 satisfies both. Padding slots stay -1 and ``combined_lens`` caps the
# valid range, so the alignment is a no-op for the math but we keep it for
# layout parity with the GPU path.
_SPARSE_PREFILL_TOPK_ALIGNMENT = 128


def cpu_combine_topk_swa_indices(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
    M: int,
    N: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU torch port of ``combine_topk_swa_indices``.

    Mirrors ``_combine_topk_swa_indices_kernel`` (cache_utils.py:493-563).
    Per query token at absolute position ``pos = seq_len - query_len + i``::

        topk_len  = min((pos + 1) // compress_ratio, topk)
        swa_len   = min(pos + 1, window_size)
        out[: topk_len]                    = topk_indices[token, : topk_len]
                                              + M * batch_idx
        out[topk_len : topk_len + swa_len] = M * batch_idx + N
                                              + (pos - swa_len + 1 - gather_start)
                                              + arange(swa_len)
        out[topk_len + swa_len :]          = -1
        combined_len                       = topk_len + swa_len

    where ``gather_start = seq_len - gather_len`` is the local-buffer offset
    for the SWA portion. Output width is padded to a multiple of 128
    (``_SPARSE_PREFILL_TOPK_ALIGNMENT``) — extra slots stay -1.

    Args:
        topk_indices:    [num_tokens, K] int32.
        query_start_loc: [num_reqs + 1] int. Per-request query offsets,
                         possibly globally rebased; the kernel subtracts
                         ``query_start_loc[0]`` (``base``) to get
                         chunk-local offsets.
        seq_lens:        [num_reqs] int.
        gather_lens:     [num_reqs] int.
        window_size:     int. SWA window size.
        compress_ratio:  int. C4A → 4, C128A → 128. SWA-only callers pass
                         ``topk=0`` to disable the topk slice.
        topk:            int. ``TOP_K`` constant in the kernel.
        M, N:            int. Per-batch index offsets.

    Returns:
        (combined_indices, combined_lens):
          * combined_indices: [num_tokens, padded_topk] int32, padded with -1.
          * combined_lens:    [num_tokens] int32.
    """
    assert topk_indices.dim() == 2
    assert query_start_loc.dim() == 1
    assert seq_lens.dim() == 1
    assert gather_lens.dim() == 1

    num_tokens, K = topk_indices.shape
    num_reqs = seq_lens.shape[0]

    combined_topk = (
        (topk + window_size + _SPARSE_PREFILL_TOPK_ALIGNMENT - 1)
        // _SPARSE_PREFILL_TOPK_ALIGNMENT
        * _SPARSE_PREFILL_TOPK_ALIGNMENT
    )
    combined_indices = torch.full(
        (num_tokens, combined_topk),
        fill_value=-1,
        dtype=torch.int32,
        device=topk_indices.device,
    )
    combined_lens = torch.zeros(
        num_tokens, dtype=torch.int32, device=topk_indices.device
    )
    if num_tokens == 0:
        return combined_indices, combined_lens

    base = int(query_start_loc[0].item())
    qsl = (query_start_loc - base).to(torch.long).tolist()
    seq_lens_l = seq_lens.to(torch.long).tolist()
    gather_lens_l = gather_lens.to(torch.long).tolist()

    topk_indices_i32 = topk_indices.to(torch.int32)

    for batch_idx in range(num_reqs):
        q_start = qsl[batch_idx]
        q_end = qsl[batch_idx + 1]
        query_len = q_end - q_start
        if query_len <= 0:
            continue
        seq_len = seq_lens_l[batch_idx]
        gather_len = gather_lens_l[batch_idx]
        gather_start = seq_len - gather_len
        start_pos = seq_len - query_len

        for i in range(query_len):
            tok = q_start + i
            pos = start_pos + i
            topk_len = (
                min((pos + 1) // compress_ratio, topk)
                if compress_ratio > 0
                else 0
            )
            swa_len = min(pos + 1, window_size)

            if topk_len > 0:
                src = topk_indices_i32[tok, :topk_len]
                combined_indices[tok, :topk_len] = src + (M * batch_idx)
            if swa_len > 0:
                offsets = torch.arange(
                    swa_len, dtype=torch.int32, device=topk_indices.device
                )
                base_off = M * batch_idx + N + (pos - swa_len + 1 - gather_start)
                combined_indices[tok, topk_len : topk_len + swa_len] = (
                    base_off + offsets
                )
            combined_lens[tok] = topk_len + swa_len

    return combined_indices, combined_lens


# ---------------------------------------------------------------------------
# M3 — C128A precomputed indices (compress_ratio=128 path)
# ---------------------------------------------------------------------------


def cpu_build_c128a_topk_metadata(
    positions: torch.Tensor,
    compress_ratio: int,
    num_decode_tokens: int,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    slot_mapping: torch.Tensor,
    global_decode_buffer: torch.Tensor,
    decode_lens_buffer: torch.Tensor,
    prefill_buffer: torch.Tensor,
    max_compressed_tokens: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CPU torch port of ``build_c128a_topk_metadata``.

    Mirrors the Triton kernel in
    ``vllm/v1/attention/backends/mla/flashmla_sparse.py``
    (``_build_c128a_topk_metadata_kernel``).

    Decode tokens (``token_idx < num_decode_tokens``):
      * ``num_compressed = min((position + 1) // compress_ratio,
        max_compressed_tokens)``
      * For each ``offset`` in ``[0, max_compressed_tokens)``:
          ``slot_id = block_table[req, offset // block_size] * block_size
                       + offset % block_size``  if ``offset < num_compressed``
          else ``slot_id = -1``.
        Written into ``global_decode_buffer[token_idx]``.
      * ``decode_lens_buffer[token_idx] = num_compressed`` if
        ``slot_mapping[token_idx] >= 0`` else ``0``.

    Prefill tokens (``token_idx >= num_decode_tokens``):
      * Write ``[0, 1, ..., n-1, -1, -1, ...]`` into
        ``prefill_buffer[token_idx - num_decode_tokens]`` where ``n =
        num_compressed``.

    Buffers are pre-allocated by the metadata builder. Returns slices of
    the buffers (matching the GPU helper's contract).
    """
    num_tokens = int(positions.shape[0])
    num_prefill_tokens = num_tokens - num_decode_tokens

    global_decode = global_decode_buffer[:num_decode_tokens]
    decode_lens = decode_lens_buffer[:num_decode_tokens]
    prefill_local = prefill_buffer[:num_prefill_tokens]

    if num_tokens == 0:
        return global_decode, decode_lens, prefill_local

    pos_int = positions.to(torch.int64)
    # num_compressed[t] = min((pos[t] + 1) // compress_ratio, max_compressed_tokens)
    num_compressed = torch.clamp(
        (pos_int + 1) // compress_ratio,
        max=max_compressed_tokens,
    )

    width = int(global_decode_buffer.shape[1])
    # Sanity: the wire site sets max_compressed_tokens to the buffer
    # stride, so width should match. Tolerate width >= max for safety.
    assert width >= max_compressed_tokens, (
        f"global_decode_buffer width ({width}) < max_compressed_tokens "
        f"({max_compressed_tokens})"
    )
    assert int(prefill_buffer.shape[1]) >= max_compressed_tokens, (
        "prefill_buffer width must be at least max_compressed_tokens"
    )

    # ------------------------------------------------------------------
    # Decode branch — fully vectorized.
    # ------------------------------------------------------------------
    if num_decode_tokens > 0:
        # offset grid: [num_decode_tokens, max_compressed_tokens]
        offsets = torch.arange(
            max_compressed_tokens,
            dtype=torch.int64,
            device=positions.device,
        ).unsqueeze(0).expand(num_decode_tokens, max_compressed_tokens)
        # Per-row validity (offset < num_compressed[t])
        nc_dec = num_compressed[:num_decode_tokens].unsqueeze(1)
        is_valid = offsets < nc_dec  # [num_decode_tokens, max_compressed_tokens]

        block_indices = offsets // block_size
        block_offsets = offsets % block_size

        # Gather block_numbers from block_table (clamp invalid indices to 0
        # so the gather is always in-bounds; we mask with is_valid below).
        bt_max = block_table.shape[1]
        safe_block_indices = torch.where(
            is_valid,
            block_indices,
            torch.zeros_like(block_indices),
        ).clamp(max=bt_max - 1)

        req_idx = token_to_req_indices[:num_decode_tokens].to(torch.int64)
        # block_table[req[t], safe_block_indices[t, o]]
        block_numbers = block_table[
            req_idx.unsqueeze(1).expand_as(safe_block_indices),
            safe_block_indices,
        ]
        slot_ids = block_numbers.to(torch.int64) * block_size + block_offsets
        slot_ids = torch.where(
            is_valid,
            slot_ids,
            torch.full_like(slot_ids, -1),
        )

        # Write into the buffer slice; pad past max_compressed_tokens stays
        # whatever the buffer was initialized with (typically uninitialized;
        # GPU kernel doesn't touch it either).
        global_decode_buffer[:num_decode_tokens, :max_compressed_tokens].copy_(
            slot_ids.to(global_decode_buffer.dtype)
        )

        # decode_lens: num_compressed if slot_mapping[t] >= 0 else 0.
        is_valid_token = slot_mapping[:num_decode_tokens] >= 0
        count = num_compressed[:num_decode_tokens]
        decode_lens_out = torch.where(
            is_valid_token,
            count,
            torch.zeros_like(count),
        ).to(decode_lens_buffer.dtype)
        decode_lens_buffer[:num_decode_tokens].copy_(decode_lens_out)

    # ------------------------------------------------------------------
    # Prefill branch — write [0..n-1, -1, ...].
    # ------------------------------------------------------------------
    if num_prefill_tokens > 0:
        pfx_offsets = torch.arange(
            max_compressed_tokens,
            dtype=torch.int64,
            device=positions.device,
        ).unsqueeze(0).expand(num_prefill_tokens, max_compressed_tokens)
        nc_pfx = num_compressed[num_decode_tokens:].unsqueeze(1)
        local_idx = torch.where(
            pfx_offsets < nc_pfx,
            pfx_offsets,
            torch.full_like(pfx_offsets, -1),
        )
        prefill_buffer[:num_prefill_tokens, :max_compressed_tokens].copy_(
            local_idx.to(prefill_buffer.dtype)
        )

    return global_decode, decode_lens, prefill_local


# ---------------------------------------------------------------------------
# CPU DeepSeek V4 sparse MLA impl
# ---------------------------------------------------------------------------


class DeepseekV4CPUFlashMLASparseBackend(FlashMLASparseBackend):
    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @staticmethod
    def get_name() -> str:
        return "V4_CPU_FLASHMLA_SPARSE"

    @staticmethod
    def get_impl_cls() -> type[DeepseekV4CPUSparseMLAImpl]:
        return DeepseekV4CPUSparseMLAImpl

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [512]

    @classmethod
    def supports_compute_capability(cls, capability) -> bool:
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
        return (num_blocks, block_size, head_size)


class DeepseekV4CPUSparseMLAImpl:
    """CPU sparse MLA implementation for DeepSeek V4's custom MLA wrapper."""

    backend_cls: ClassVar[type[DeepseekV4CPUFlashMLASparseBackend]] = (
        DeepseekV4CPUFlashMLASparseBackend
    )
    PREFILL_CHUNK_SIZE: ClassVar[int] = 4

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        return num_heads

    @classmethod
    def forward_mqa(
        cls,
        layer: DeepseekV4MLAAttention,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        del kv, positions
        assert output.shape == q.shape, (
            f"output buffer shape {output.shape} must match q shape {q.shape}"
        )
        assert output.dtype == q.dtype, (
            f"output buffer dtype {output.dtype} must match q dtype {q.dtype}"
        )

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        if attn_metadata is None:
            output.zero_()
            return

        assert isinstance(attn_metadata, dict)
        flashmla_metadata = cast(
            "FlashMLASparseMetadata | None",
            attn_metadata.get(layer.prefix),
        )
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(layer.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_only = layer.compress_ratio <= 1
        self_kv_cache = layer.kv_cache if not swa_only else None
        swa_k_cache = layer.swa_cache_layer.kv_cache

        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens

        if num_prefills > 0:
            cls._forward_prefill(
                layer=layer,
                q=q[num_decode_tokens:],
                compressed_k_cache=self_kv_cache,
                swa_k_cache=swa_k_cache,
                output=output[num_decode_tokens:],
                attn_metadata=flashmla_metadata,
                swa_metadata=swa_metadata,
            )
        if num_decodes > 0:
            cls._forward_decode(
                layer=layer,
                q=q[:num_decode_tokens],
                kv_cache=self_kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=flashmla_metadata,
                swa_only=swa_only,
                output=output[:num_decode_tokens],
            )

    @classmethod
    def _forward_decode(
        cls,
        layer: DeepseekV4MLAAttention,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_metadata: DeepseekSparseSWAMetadata,
        attn_metadata: FlashMLASparseMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        topk_indices = None
        topk_lens = None
        if not swa_only:
            assert attn_metadata is not None
            assert swa_metadata.is_valid_token is not None
            block_size = attn_metadata.block_size // layer.compress_ratio
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            if layer.compress_ratio == 4:
                assert layer.topk_indices_buffer is not None
                global_indices, topk_lens = (
                    cpu_compute_global_topk_indices_and_lens(
                        layer.topk_indices_buffer[:num_decode_tokens],
                        swa_metadata.token_to_req_indices,
                        attn_metadata.block_table[:num_decodes],
                        block_size,
                        is_valid,
                    )
                )
                topk_indices = global_indices.view(num_decode_tokens, 1, -1)
            else:
                topk_indices = attn_metadata.c128a_global_decode_topk_indices
                topk_lens = attn_metadata.c128a_decode_topk_lens

        assert swa_metadata.decode_swa_indices is not None
        assert swa_metadata.decode_swa_lens is not None
        cpu_forward_decode(
            q=q,
            kv_cache=kv_cache,
            swa_k_cache=layer.swa_cache_layer.kv_cache,
            swa_only=swa_only,
            topk_indices=topk_indices,
            topk_lens=topk_lens,
            swa_indices=swa_metadata.decode_swa_indices,
            swa_lens=swa_metadata.decode_swa_lens,
            attn_sink=layer.attn_sink,
            scale=layer.scale,
            head_dim=layer.head_dim,
            nope_head_dim=layer.nope_head_dim,
            rope_head_dim=layer.rope_head_dim,
            output=output,
        )

    @classmethod
    def _forward_prefill(
        cls,
        layer: DeepseekV4MLAAttention,
        q: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata | None,
        swa_metadata: DeepseekSparseSWAMetadata,
    ) -> None:
        swa_only = attn_metadata is None

        num_prefills = swa_metadata.num_prefills
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens
        num_prefill_tokens = swa_metadata.num_prefill_tokens

        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        assert seq_lens is not None
        assert gather_lens is not None

        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        query_start_loc = swa_metadata.query_start_loc
        assert query_start_loc_cpu is not None
        assert query_start_loc is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        if not swa_only:
            if layer.compress_ratio == 4:
                assert layer.topk_indices_buffer is not None
                topk_indices = layer.topk_indices_buffer[num_decode_tokens:]
                topk_indices = topk_indices[:num_prefill_tokens]
            else:
                assert attn_metadata is not None
                topk_indices = attn_metadata.c128a_prefill_topk_indices
                assert topk_indices is not None
            top_k = topk_indices.shape[-1]
            N = (
                layer.max_model_len + layer.compress_ratio - 1
            ) // layer.compress_ratio
        else:
            assert layer.topk_indices_buffer is not None
            topk_indices = layer.topk_indices_buffer[num_decode_tokens:]
            top_k = 0
            N = 0

        M = N + layer.window_size + layer.max_num_batched_tokens
        chunk_size_const = cls.PREFILL_CHUNK_SIZE
        num_chunks = (num_prefills + chunk_size_const - 1) // chunk_size_const

        kv = torch.empty(
            (chunk_size_const, M, q.shape[-1]),
            dtype=torch.bfloat16,
            device=q.device,
        )
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size_const
            chunk_end = min(chunk_start + chunk_size_const, num_prefills)
            chunk_size = chunk_end - chunk_start
            if not swa_only:
                assert attn_metadata is not None
                assert compressed_k_cache is not None
                block_table = attn_metadata.block_table[num_decodes:]
                cpu_dequantize_and_gather_k_cache(
                    kv[:chunk_size],
                    compressed_k_cache,
                    seq_lens=seq_lens[chunk_start:chunk_end]
                    // layer.compress_ratio,
                    gather_lens=None,
                    block_table=block_table[chunk_start:chunk_end],
                    block_size=attn_metadata.block_size // layer.compress_ratio,
                    offset=0,
                )

            swa_block_table = swa_metadata.block_table[num_decodes:]
            cpu_dequantize_and_gather_k_cache(
                kv[:chunk_size],
                swa_k_cache,
                seq_lens=seq_lens[chunk_start:chunk_end],
                gather_lens=gather_lens[chunk_start:chunk_end],
                block_table=swa_block_table[chunk_start:chunk_end],
                block_size=swa_metadata.block_size,
                offset=N,
            )

            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )

            combined_indices, combined_lens = cpu_combine_topk_swa_indices(
                topk_indices[query_start:query_end],
                query_start_loc[
                    num_decodes + chunk_start : num_decodes + chunk_end + 1
                ],
                seq_lens[chunk_start:chunk_end],
                gather_lens[chunk_start:chunk_end],
                layer.window_size,
                layer.compress_ratio,
                top_k,
                M,
                N,
            )
            q_chunk = q[query_start:query_end]
            kv_view = kv.view(-1, 1, q.shape[-1])
            indices_chunk = combined_indices.unsqueeze(1)
            output_chunk = output[query_start:query_end]
            cpu_sparse_attn_prefill(
                q=q_chunk,
                kv=kv_view,
                indices=indices_chunk,
                topk_length=combined_lens,
                scale=layer.scale,
                head_dim=layer.head_dim,
                attn_sink=layer.attn_sink,
                output=output_chunk,
            )


__all__ = [
    # M1 helpers
    "DeepseekV4CPUFlashMLASparseBackend",
    "DeepseekV4CPUSparseMLAImpl",
    "cpu_q_kv_rmsnorm",
    "cpu_q_kv_rmsnorm_no_k_pe",
    "cpu_qnorm_rope_kv_rope_insert",
    "cpu_dequantize_and_gather_k_cache",
    "cpu_inv_rope_einsum",
    "cpu_forward_decode",
    "cpu_sparse_attn_prefill",
    # M2 helpers
    "cpu_kv_compress_norm_rope_insert",
    "_cpu_save_partial_states",
    "_gptj_rope_apply_native",
    "cpu_indexer_q_rope_quant",
    "cpu_sparse_attn_indexer_op",
    "cpu_compute_global_topk_indices_and_lens",
    "cpu_combine_topk_swa_indices",
    # M3 helpers
    "cpu_build_c128a_topk_metadata",
    # Internal helpers (exposed for tests)
    "_apply_inv_rope_ref",
    "_rmsnorm_native",
    "_per_head_rmsnorm_no_weight",
]
