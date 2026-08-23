# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Portable Torch operators for the DeepSeek V4 CPU implementation."""

import torch


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    x_float = x.float()
    output = x_float * torch.rsqrt(x_float.square().mean(-1, keepdim=True) + eps)
    if weight is not None:
        output *= weight.float()
    return output.to(x.dtype)


def apply_rope_tail(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_dim: int,
    *,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply GPT-J interleaved RoPE to the trailing ``rope_dim`` values."""

    x_float = x.float()
    rope = x_float[..., -rope_dim:]
    even, odd = rope[..., 0::2], rope[..., 1::2]
    rows = cos_sin_cache.index_select(0, positions.reshape(-1).to(torch.long)).float()
    while rows.ndim < x.ndim:
        rows = rows.unsqueeze(-2)
    half = rope_dim // 2
    cos = rows[..., :half]
    sin = rows[..., half : 2 * half]
    if inverse:
        sin = -sin
    rotated = torch.empty_like(rope)
    rotated[..., 0::2] = even * cos - odd * sin
    rotated[..., 1::2] = odd * cos + even * sin
    output = x_float.clone()
    output[..., -rope_dim:] = rotated
    return output.to(x.dtype)


def write_paged_cache(
    cache: torch.Tensor,
    values: torch.Tensor,
    slots: torch.Tensor,
) -> None:
    if cache.numel() == 0 or values.numel() == 0:
        return
    slots = slots[: values.shape[0]].to(torch.long)
    valid = slots >= 0
    if not bool(valid.any()):
        return
    block_size = cache.shape[1]
    selected = slots[valid]
    cache[selected // block_size, selected % block_size] = values[valid].to(cache.dtype)


def gather_paged_cache(
    cache: torch.Tensor,
    slots: torch.Tensor,
) -> torch.Tensor:
    slots = slots.to(torch.long)
    if slots.numel() == 0:
        return cache.new_empty((0, cache.shape[-1]))
    block_size = cache.shape[1]
    return cache[slots // block_size, slots % block_size]


def save_compressor_states(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    positions: torch.Tensor,
    state_cache: torch.Tensor,
    slots: torch.Tensor,
    compress_ratio: int,
) -> None:
    if state_cache.numel() == 0:
        return
    slots = slots[: kv.shape[0]].to(torch.long)
    valid = slots >= 0
    if not bool(valid.any()):
        return
    selected = slots[valid]
    block_size = state_cache.shape[1]
    rows = state_cache[selected // block_size, selected % block_size]
    width = rows.shape[-1] // 2
    rows[:, :width] = kv[valid].float()
    ape_rows = positions[valid].to(torch.long).remainder(compress_ratio)
    rows[:, width:] = score[valid].float() + ape.index_select(0, ape_rows).float()


def compress_and_store(
    *,
    state_cache: torch.Tensor,
    state_metadata,
    output_cache: torch.Tensor,
    output_metadata,
    positions: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    cos_sin_cache: torch.Tensor,
    head_dim: int,
    rope_dim: int,
    compress_ratio: int,
    overlap: bool,
) -> None:
    """Compress boundary windows and write their BF16 rows to paged cache."""

    if state_cache.numel() == 0 or output_cache.numel() == 0:
        return
    state_slots = state_metadata.slot_mapping.to(torch.long)
    output_slots = output_metadata.slot_mapping.to(torch.long)
    req_ids = state_metadata.token_to_req_indices.to(torch.long)
    block_table = state_metadata.block_table
    state_block_size = state_cache.shape[1]
    state_width = state_cache.shape[-1] // 2
    coff = state_width // head_dim
    assert coff == (2 if overlap else 1)
    window = compress_ratio * coff

    for token in range(positions.shape[0]):
        position = int(positions[token].item())
        if (
            state_slots[token] < 0
            or output_slots[token] < 0
            or (position + 1) % compress_ratio
        ):
            continue
        req = int(req_ids[token].item())
        kv_rows: list[torch.Tensor] = []
        score_rows: list[torch.Tensor] = []
        for offset, logical_position in enumerate(
            range(position - window + 1, position + 1)
        ):
            if logical_position < 0:
                kv_rows.append(state_cache.new_zeros(head_dim))
                score_rows.append(state_cache.new_full((head_dim,), float("-inf")))
                continue
            logical_block = logical_position // state_block_size
            block = int(block_table[req, logical_block].item())
            row = state_cache[block, logical_position % state_block_size]
            part = 1 if overlap and offset >= compress_ratio else 0
            kv_rows.append(row[part * head_dim : (part + 1) * head_dim].float())
            score_start = state_width + part * head_dim
            score_rows.append(row[score_start : score_start + head_dim].float())

        kv_stack = torch.stack(kv_rows)
        score_stack = torch.stack(score_rows)
        all_masked = torch.isneginf(score_stack).all(dim=0, keepdim=True)
        score_stack = torch.where(
            all_masked, torch.zeros_like(score_stack), score_stack
        )
        compressed = (kv_stack * torch.softmax(score_stack, dim=0)).sum(dim=0)
        normalized = rms_norm(compressed, norm_weight, norm_eps)
        compressed_position = (position // compress_ratio) * compress_ratio
        rotated = apply_rope_tail(
            normalized.unsqueeze(0),
            positions.new_tensor([compressed_position]),
            cos_sin_cache,
            rope_dim,
        ).squeeze(0)
        write_paged_cache(
            output_cache,
            rotated.unsqueeze(0),
            output_slots[token : token + 1],
        )


def sparse_mla_reference(
    q: torch.Tensor,
    key_rows: list[torch.Tensor],
    scale: float,
    attn_sink: torch.Tensor | None,
    out: torch.Tensor,
) -> None:
    """Reference sparse MLA for per-token candidate key/value rows."""

    num_heads = q.shape[1]
    sink = attn_sink[:num_heads].float() if attn_sink is not None else None
    out.zero_()
    for token, rows in enumerate(key_rows):
        if rows.numel() == 0:
            continue
        q_token = q[token].float()
        rows_float = rows.float()
        logits = q_token @ rows_float.t() * scale
        if sink is not None:
            logits = torch.cat((logits, sink[:, None]), dim=-1)
        probs = torch.softmax(logits, dim=-1)[..., : rows_float.shape[0]]
        out[token].copy_((probs @ rows_float).to(out.dtype))


__all__ = [
    "apply_rope_tail",
    "compress_and_store",
    "gather_paged_cache",
    "rms_norm",
    "save_compressor_states",
    "sparse_mla_reference",
    "write_paged_cache",
]
