# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression for DeepSeek V4 CPU KV split offset in cache insert."""

from __future__ import annotations

import inspect
import re

import pytest
import torch

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def _strip_comments(src: str) -> str:
    out_lines: list[str] = []
    for line in src.splitlines():
        if line.lstrip().startswith("#"):
            continue
        idx = line.find("#")
        if idx >= 0:
            line = line[:idx]
        out_lines.append(line)
    return "\n".join(out_lines)


def _get_fused_qnorm_rope_kv_insert_source() -> str:
    from vllm.models.deepseek_v4.attention import (
        DeepseekV4MultiHeadLatentAttentionWrapper,
    )

    src = inspect.getsource(
        DeepseekV4MultiHeadLatentAttentionWrapper._fused_qnorm_rope_kv_insert
    )
    return _strip_comments(src)


def test_cpu_branch_passes_nope_head_dim_as_split_offset() -> None:
    src = _get_fused_qnorm_rope_kv_insert_source()
    m = re.search(
        r"cpu_qnorm_rope_kv_rope_insert\(\s*(.*?)\s*\)\s*\n\s*if\s+"
        r"self\.n_local_heads",
        src,
        re.DOTALL,
    )
    assert m is not None

    args = [arg.strip() for arg in m.group(1).split(",") if arg.strip()]
    assert len(args) == 10
    assert args[7] == "self.nope_head_dim"
    assert "self.kv_lora_rank" not in m.group(1)


def test_cpu_branch_reviews_swa_cache_for_flat_slot_indexing() -> None:
    src = _get_fused_qnorm_rope_kv_insert_source()
    m = re.search(
        r"if\s+current_platform\.is_cpu\(\):(.*?)\n\s*return\s+q\b",
        src,
        re.DOTALL,
    )
    assert m is not None
    assert re.search(r"\.view\(\s*-1\s*,\s*self\.head_dim\s*\)", m.group(1))


class _FakeRotaryEmb:
    def __init__(self, rotary_dim: int, max_pos: int = 4096) -> None:
        self.rotary_dim = rotary_dim
        self.calls = 0
        t = torch.arange(max_pos, dtype=torch.float32).unsqueeze(1)
        f = torch.arange(rotary_dim // 2, dtype=torch.float32).unsqueeze(0)
        self.cos_cache = torch.cos(0.01 * t * (f + 1))
        self.sin_cache = torch.sin(0.01 * t * (f + 1))

    def forward_native(
        self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls += 1
        if query.shape[-1] != self.rotary_dim or key.shape[-1] != self.rotary_dim:
            raise RuntimeError(
                "must match the size of tensor at non-singleton dimension 2"
            )
        cos = self.cos_cache[positions].repeat_interleave(2, dim=-1).unsqueeze(-2)
        sin = self.sin_cache[positions].repeat_interleave(2, dim=-1).unsqueeze(-2)

        def rotate_gptj(x: torch.Tensor) -> torch.Tensor:
            x1 = x[..., 0::2]
            x2 = x[..., 1::2]
            return torch.stack((-x2, x1), dim=-1).flatten(-2)

        return (
            (query * cos + rotate_gptj(query) * sin).to(query.dtype),
            (key * cos + rotate_gptj(key) * sin).to(key.dtype),
        )


def test_cpu_qnorm_rope_kv_rope_insert_with_correct_split_offset_runs() -> None:
    from vllm.models.deepseek_v4.cpu import cpu_qnorm_rope_kv_rope_insert

    head_dim = 128
    rope_head_dim = 64
    nope_head_dim = head_dim - rope_head_dim
    num_heads = 4
    num_blocks = 2
    block_size = 4
    num_tokens = 3

    torch.manual_seed(0)
    q = torch.randn(num_tokens, num_heads, head_dim, dtype=torch.float32)
    kv = torch.randn(num_tokens, head_dim, dtype=torch.float32)
    kv_pe_pre = kv[..., nope_head_dim:].clone()

    swa_cache = torch.zeros(num_blocks * block_size, head_dim, dtype=torch.float32)
    slot_mapping = torch.tensor([0, 3, 7], dtype=torch.int64)
    positions = torch.tensor([0, 1, 2], dtype=torch.int64)

    cpu_qnorm_rope_kv_rope_insert(
        q,
        kv,
        swa_cache,
        slot_mapping,
        positions,
        _FakeRotaryEmb(rotary_dim=rope_head_dim),
        1e-6,
        nope_head_dim,
        rope_head_dim,
        nope_head_dim,
    )

    kv_pe_post = kv[..., nope_head_dim:]
    assert kv_pe_post.shape == (num_tokens, rope_head_dim)
    assert not torch.allclose(kv_pe_post, kv_pe_pre, atol=1e-6)
    torch.testing.assert_close(swa_cache[0], kv[0])
    torch.testing.assert_close(swa_cache[3], kv[1])
    torch.testing.assert_close(swa_cache[7], kv[2])


def test_cpu_qnorm_rope_kv_rope_insert_with_buggy_split_offset_skips_rope() -> None:
    from vllm.models.deepseek_v4.cpu import cpu_qnorm_rope_kv_rope_insert

    head_dim = 128
    rope_head_dim = 64
    nope_head_dim = head_dim - rope_head_dim
    num_tokens = 2

    q = torch.randn(num_tokens, 4, head_dim, dtype=torch.float32)
    kv = torch.randn(num_tokens, head_dim, dtype=torch.float32)
    kv_pre = kv.clone()
    swa_cache = torch.zeros(8, head_dim, dtype=torch.float32)
    rotary_emb = _FakeRotaryEmb(rotary_dim=rope_head_dim)

    cpu_qnorm_rope_kv_rope_insert(
        q,
        kv,
        swa_cache,
        torch.tensor([0, 1], dtype=torch.int64),
        torch.tensor([0, 1], dtype=torch.int64),
        rotary_emb,
        1e-6,
        head_dim,
        rope_head_dim,
        nope_head_dim,
    )

    assert rotary_emb.calls == 0
    torch.testing.assert_close(kv, kv_pre)
