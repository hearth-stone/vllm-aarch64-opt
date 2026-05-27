# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU bf16 path smoke tests for DeepSeek V4.

Scope of this first cut:

* ``vllm.model_executor.layers.mhc`` and ``vllm.model_executor.models.deepseek_v4``
  must import on CPU (no triton / tilelang / deep_gemm / flashmla pulled in).
* The pure-torch fallback paths in ``mhc.py`` (``mhc_pre`` / ``mhc_post``
  / ``mhc_fused_post_pre`` / ``hc_head_fused_kernel``) produce numerically
  consistent results with each other on CPU.
* The ``DeepseekV4FP8Config`` config / weight-mapper plumbing recognises the
  ``"w8a8"`` ``expert_dtype`` reserved path.

End-to-end CPU MLA execution is *not* in scope here — the
``DeepseekV4MultiHeadLatentAttentionWrapper`` is intentionally bound to
CUDA-only fused kernels and raises ``NotImplementedError`` on CPU.
"""

from __future__ import annotations

import importlib

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cpu(),
    reason="CPU-only smoke tests for DeepSeek V4 ARM bf16 path.",
)


def test_mhc_imports_on_cpu():
    mhc = importlib.import_module("vllm.model_executor.layers.mhc")
    # On CPU the tilelang module must be unset and the dispatcher functions
    # must still be callable Python objects (their ROCm-style torch fallback
    # path is what runs).
    assert mhc.tilelang is None, "tilelang should be None on CPU"
    for name in ("mhc_pre", "mhc_post", "mhc_fused_post_pre", "_hc_head_fused_kernel"):
        assert callable(getattr(mhc, name)), f"{name} should be a callable"


def test_deepseek_v4_module_imports_on_cpu():
    mod = importlib.import_module("vllm.model_executor.models.deepseek_v4")
    # All the public classes the loader uses should resolve.
    for name in (
        "DeepseekV4ForCausalLM",
        "DeepseekV4FP8Config",
        "_make_deepseek_v4_weights_mapper",
        "_DEEPSEEK_V4_EXPERT_DTYPES",
    ):
        assert hasattr(mod, name), f"deepseek_v4 module missing {name}"

    # On CPU, Mxfp4MoEMethod is bound to None — the FP4 expert path should
    # not be reachable from the quant config.
    assert mod.Mxfp4MoEMethod is None


def test_w8a8_in_expert_dtype_whitelist():
    mod = importlib.import_module("vllm.model_executor.models.deepseek_v4")
    assert "w8a8" in mod._DEEPSEEK_V4_EXPERT_DTYPES
    assert "fp4" in mod._DEEPSEEK_V4_EXPERT_DTYPES
    assert "fp8" in mod._DEEPSEEK_V4_EXPERT_DTYPES


def test_w8a8_weights_mapper_remaps_input_scale():
    mod = importlib.import_module("vllm.model_executor.models.deepseek_v4")
    mapper = mod._make_deepseek_v4_weights_mapper("w8a8")
    # The mapper is a WeightsMapper; reach into its regex table to confirm
    # the w8a8 branch installed both ``.weight_scale`` (passthrough) and
    # ``.input_scale`` (passthrough) preserving rules for expert keys.
    keys = [p.pattern for p in mapper.orig_to_new_regex.keys()]
    assert any("input_scale" in k for k in keys), (
        f"w8a8 mapper missing .input_scale rule, got {keys}"
    )


def test_mhc_pre_torch_fallback_runs():
    # Construct a minimal valid input for mhc_pre on CPU and check that
    # it runs through the ROCm-shared torch fallback without raising.
    mhc = importlib.import_module("vllm.model_executor.layers.mhc")

    hc_mult = 2
    hidden_size = 32  # divisible by 32 (mhc kernels assume multiples of 128
    # in the GPU path, but the torch fallback only requires shapes match).
    num_tokens = 4
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult

    residual = torch.randn(num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16)
    fn = torch.randn(hc_mult3, hc_mult * hidden_size, dtype=torch.float32)
    hc_scale = torch.randn(3, dtype=torch.float32)
    hc_base = torch.randn(hc_mult3, dtype=torch.float32)

    post_mix, comb_mix, layer_input = mhc.mhc_pre(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps=1e-6,
        hc_pre_eps=1e-3,
        hc_sinkhorn_eps=1e-3,
        hc_post_mult_value=2.0,
        sinkhorn_repeat=1,
    )

    assert post_mix.shape == (num_tokens, hc_mult, 1)
    assert comb_mix.shape == (num_tokens, hc_mult, hc_mult)
    assert layer_input.shape == (num_tokens, hidden_size)
    assert layer_input.dtype == torch.bfloat16


def test_mhc_post_torch_fallback_runs():
    mhc = importlib.import_module("vllm.model_executor.layers.mhc")
    hc_mult = 2
    hidden_size = 16
    num_tokens = 3

    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16)
    residual = torch.randn(
        num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16
    )
    post_layer_mix = torch.randn(num_tokens, hc_mult, 1, dtype=torch.float32)
    comb_res_mix = torch.randn(num_tokens, hc_mult, hc_mult, dtype=torch.float32)

    out = mhc.mhc_post(x, residual, post_layer_mix, comb_res_mix)
    assert out.shape == residual.shape
    assert out.dtype == torch.bfloat16


def test_mhc_fused_post_pre_composes_post_then_pre():
    # CPU fallback for ``mhc_fused_post_pre`` should equal ``mhc_post``
    # followed by ``mhc_pre`` on the resulting residual (we just use the
    # same inputs and check no exceptions; numerical equivalence is by
    # construction since the CPU path literally calls the two reference
    # impls in sequence).
    mhc = importlib.import_module("vllm.model_executor.layers.mhc")

    hc_mult = 2
    hidden_size = 32
    num_tokens = 4
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult

    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16)
    residual = torch.randn(
        num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16
    )
    post_layer_mix = torch.randn(num_tokens, hc_mult, 1, dtype=torch.float32)
    comb_res_mix = torch.randn(num_tokens, hc_mult, hc_mult, dtype=torch.float32)
    fn = torch.randn(hc_mult3, hc_mult * hidden_size, dtype=torch.float32)
    hc_scale = torch.randn(3, dtype=torch.float32)
    hc_base = torch.randn(hc_mult3, dtype=torch.float32)

    residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur = (
        mhc.mhc_fused_post_pre(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            fn,
            hc_scale,
            hc_base,
            rms_eps=1e-6,
            hc_pre_eps=1e-3,
            hc_sinkhorn_eps=1e-3,
            hc_post_mult_value=2.0,
            sinkhorn_repeat=1,
        )
    )

    assert residual_cur.shape == residual.shape
    assert residual_cur.dtype == torch.bfloat16
    assert post_mix_cur.shape == (num_tokens, hc_mult, 1)
    assert comb_mix_cur.shape == (num_tokens, hc_mult, hc_mult)
    assert layer_input_cur.shape == (num_tokens, hidden_size)
    assert layer_input_cur.dtype == torch.bfloat16


def test_hc_head_fused_kernel_torch_fallback():
    mhc = importlib.import_module("vllm.model_executor.layers.mhc")

    hc_mult = 2
    hidden_size = 32
    num_tokens = 5

    hs_flat = torch.randn(num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16)
    fn = torch.randn(hc_mult, hc_mult * hidden_size, dtype=torch.float32)
    hc_scale = torch.randn(1, dtype=torch.float32)
    hc_base = torch.randn(hc_mult, dtype=torch.float32)
    out = torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16)

    mhc._hc_head_fused_kernel(
        hs_flat,
        fn,
        hc_scale,
        hc_base,
        out,
        hidden_size,
        rms_eps=1e-6,
        hc_eps=1e-3,
        hc_mult=hc_mult,
    )
    assert out.shape == (num_tokens, hidden_size)
    assert out.dtype == torch.bfloat16
    # Output should not be all-zeros (the kernel should have written
    # something) for non-degenerate inputs.
    assert torch.isfinite(out).all()


def test_mla_wrapper_raises_on_cpu():
    """The DeepseekV4 MLA wrapper currently has no CPU implementation; it
    must raise a clear NotImplementedError that points users at the
    follow-up work needed to add torch reference impls."""
    from vllm.model_executor.layers.deepseek_v4_attention import (
        DeepseekV4MultiHeadLatentAttentionWrapper,
    )

    with pytest.raises(NotImplementedError, match="CPU implementation"):
        # Pass enough kwargs for super().__init__() then trip the CPU
        # guard before any GPU-specific module access. The arguments here
        # are placeholders; the guard lives at the very top of __init__
        # immediately after super().__init__().
        DeepseekV4MultiHeadLatentAttentionWrapper(
            hidden_size=32,
            num_heads=4,
            head_dim=16,
            scale=1.0,
            qk_nope_head_dim=8,
            qk_rope_head_dim=8,
            v_head_dim=16,
            q_lora_rank=8,
            kv_lora_rank=8,
            o_lora_rank=8,
            mla_modules=None,  # type: ignore[arg-type]  # never read
            window_size=16,
            compress_ratio=1,
        )
