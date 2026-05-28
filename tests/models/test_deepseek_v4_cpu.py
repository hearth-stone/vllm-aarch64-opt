# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v4.cpu import cpu_sparse_attn_prefill


class _FakeCPUPlatform:
    def is_cuda(self):
        return False

    def is_cpu(self):
        return True

    def is_rocm(self):
        return False


class _FakeCUDAPlatform:
    def is_cuda(self):
        return True

    def is_cpu(self):
        return False

    def is_rocm(self):
        return False


def test_deepseek_v4_aux_streams_are_disabled_on_cpu(monkeypatch):
    from vllm.models.deepseek_v4.nvidia import model as nvidia_model

    def fail_stream():
        raise AssertionError("CPU path must not create CUDA streams")

    monkeypatch.setattr(nvidia_model, "current_platform", _FakeCPUPlatform())
    monkeypatch.setattr(nvidia_model.torch.cuda, "Stream", fail_stream)

    assert nvidia_model._make_deepseek_v4_aux_streams() is None


def test_deepseek_v4_scale_fmt_is_optional_for_bf16_config():
    from vllm.models.deepseek_v4.nvidia import model as nvidia_model

    assert nvidia_model._get_deepseek_v4_scale_fmt(SimpleNamespace()) is None
    assert (
        nvidia_model._get_deepseek_v4_scale_fmt(
            SimpleNamespace(quantization_config={"scale_fmt": "ue8m0"})
        )
        == "ue8m0"
    )


def test_deepseek_v4_decoder_uses_native_forward_off_cuda(monkeypatch):
    from vllm.models.deepseek_v4.nvidia import model as nvidia_model

    monkeypatch.setattr(nvidia_model, "current_platform", _FakeCPUPlatform())
    assert nvidia_model._use_deepseek_v4_native_decoder_forward()

    monkeypatch.setattr(nvidia_model, "current_platform", _FakeCUDAPlatform())
    assert not nvidia_model._use_deepseek_v4_native_decoder_forward()


def test_deepseek_v4_sparse_impl_uses_cpu_fallback(monkeypatch):
    from vllm.models.deepseek_v4 import attention as deepseek_attention
    from vllm.models.deepseek_v4.cpu import DeepseekV4CPUSparseMLAImpl

    monkeypatch.setattr(deepseek_attention, "current_platform", _FakeCPUPlatform())

    impl_cls = deepseek_attention._select_v4_sparse_impl()

    assert impl_cls is DeepseekV4CPUSparseMLAImpl
    assert impl_cls.backend_cls.get_supported_head_sizes() == [512]


def test_deepseek_v4_cpu_linear_output_uses_module_forward_with_empty_weight():
    from vllm.models.deepseek_v4.attention import _linear_output_to_fp32

    class CPUDispatchedLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(0), requires_grad=False)
            self.dispatch_weight = torch.randn((3, 4), dtype=torch.bfloat16)

        def forward(self, hidden_states):
            return torch.nn.functional.linear(hidden_states, self.dispatch_weight)

    layer = CPUDispatchedLinear()
    hidden_states = torch.randn((2, 4), dtype=torch.bfloat16)

    output = _linear_output_to_fp32(layer, hidden_states)

    expected = torch.nn.functional.linear(
        hidden_states,
        layer.dispatch_weight,
    ).to(torch.float32)
    assert output.dtype is torch.float32
    torch.testing.assert_close(output, expected)


def test_deepseek_v4_cpu_mla_block_size_can_group_swa_pages():
    from vllm.models.deepseek_v4.attention import _DEEPSEEK_V4_MLA_BLOCK_SIZE
    from vllm.v1.core.kv_cache_utils import _get_kv_cache_groups_uniform_groups
    from vllm.v1.kv_cache_interface import (
        MLAAttentionSpec,
        SlidingWindowMLASpec,
        UniformTypeKVCacheSpecs,
    )

    full_mla_spec = MLAAttentionSpec(
        block_size=_DEEPSEEK_V4_MLA_BLOCK_SIZE,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        compress_ratio=4,
    )
    default_block_mla_spec = MLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        compress_ratio=4,
    )
    swa_spec = SlidingWindowMLASpec(
        block_size=64,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        sliding_window=4096,
    )
    compressor_state_spec = SlidingWindowMLASpec(
        block_size=4,
        num_kv_heads=1,
        head_size=2048,
        dtype=torch.float32,
        sliding_window=4096,
    )

    assert default_block_mla_spec.page_size_bytes < swa_spec.page_size_bytes
    assert swa_spec.page_size_bytes <= full_mla_spec.page_size_bytes
    assert compressor_state_spec.page_size_bytes <= full_mla_spec.page_size_bytes

    grouped_specs = [
        UniformTypeKVCacheSpecs.from_specs({"full": full_mla_spec}),
        UniformTypeKVCacheSpecs.from_specs({"swa": swa_spec}),
        UniformTypeKVCacheSpecs.from_specs({"compressor": compressor_state_spec}),
    ]
    assert all(spec is not None for spec in grouped_specs)

    groups = _get_kv_cache_groups_uniform_groups(grouped_specs)

    assert [group.layer_names for group in groups] == [
        ["full"],
        ["swa"],
        ["compressor"],
    ]


def test_deepseek_v4_cpu_sparse_impl_dummy_forward_zeroes_output(monkeypatch):
    from vllm.models.deepseek_v4 import cpu as deepseek_cpu

    monkeypatch.setattr(
        deepseek_cpu,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=None),
    )

    q = torch.ones((2, 3, 4), dtype=torch.bfloat16)
    output = torch.empty_like(q)

    deepseek_cpu.DeepseekV4CPUSparseMLAImpl.forward_mqa(
        SimpleNamespace(),
        q,
        q,
        torch.arange(q.shape[0]),
        output,
    )

    assert torch.count_nonzero(output) == 0


def test_deepseek_v4_mega_moe_is_rejected_on_cpu(monkeypatch):
    from vllm.models.deepseek_v4.nvidia import model as nvidia_model

    monkeypatch.setattr(nvidia_model, "current_platform", _FakeCPUPlatform())

    with pytest.raises(NotImplementedError, match="CUDA-only"):
        nvidia_model._check_deepseek_v4_mega_moe_supported(True)

    nvidia_model._check_deepseek_v4_mega_moe_supported(False)


def test_deepseek_v4_quant_config_uses_unquantized_moe_on_cpu(monkeypatch):
    from vllm.models.deepseek_v4 import quant_config as deepseek_quant_config

    class DummyFusedMoE:
        moe_config = object()

    class DummyMoEMethod:
        def __init__(self, moe_config):
            self.moe_config = moe_config

    monkeypatch.setattr(deepseek_quant_config, "current_platform", _FakeCPUPlatform())
    monkeypatch.setattr(deepseek_quant_config, "FusedMoE", DummyFusedMoE)
    monkeypatch.setattr(
        deepseek_quant_config, "UnquantizedFusedMoEMethod", DummyMoEMethod
    )
    monkeypatch.setattr(deepseek_quant_config, "is_layer_skipped", lambda **_: False)

    config = deepseek_quant_config.DeepseekV4FP8Config()
    layer = DummyFusedMoE()

    method = config.get_quant_method(layer, "model.layers.0.mlp.experts")

    assert isinstance(method, DummyMoEMethod)
    assert method.moe_config is layer.moe_config
    assert not config.is_mxfp4_quant("model.layers.0.mlp.experts", layer)


def test_cpu_sparse_attn_prefill_basic_with_attn_sink():
    q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.bfloat16)
    kv = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=torch.bfloat16)
    indices = torch.tensor([[[0, 1, -1]]], dtype=torch.int32)
    attn_sink = torch.tensor([-1000.0, -1000.0], dtype=torch.float32)
    output = torch.empty_like(q)

    cpu_sparse_attn_prefill(
        q=q,
        kv=kv,
        indices=indices,
        topk_length=None,
        scale=1.0,
        head_dim=2,
        attn_sink=attn_sink,
        output=output,
    )

    expected = torch.softmax(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=-1)
    expected = expected @ torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    torch.testing.assert_close(output[0].float(), expected, atol=1e-2, rtol=1e-2)


def test_cpu_q_kv_rmsnorm_matches_native_rmsnorm(default_vllm_config):
    """CPU q/kv RMSNorm should match RMSNorm.forward_native."""
    del default_vllm_config
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.models.deepseek_v4.cpu import cpu_q_kv_rmsnorm

    torch.manual_seed(2)
    num_tokens = 5
    q_lora_rank = 64
    kv_lora_rank = 24
    rope = 8

    qr = torch.randn(num_tokens, q_lora_rank, dtype=torch.bfloat16)
    kv = torch.randn(num_tokens, kv_lora_rank + rope, dtype=torch.bfloat16)
    q_w = torch.randn(q_lora_rank, dtype=torch.bfloat16)
    kv_w = torch.randn(kv_lora_rank, dtype=torch.bfloat16)

    qr_norm_mod = RMSNorm(q_lora_rank, eps=1e-6)
    qr_norm_mod.weight.data.copy_(q_w)
    kv_norm_mod = RMSNorm(kv_lora_rank, eps=1e-6)
    kv_norm_mod.weight.data.copy_(kv_w)

    qr_ref = qr_norm_mod.forward_native(qr.clone())
    kv_c_ref = kv_norm_mod.forward_native(kv[..., :kv_lora_rank].clone())

    qr_out, kv_out = cpu_q_kv_rmsnorm(
        qr,
        kv,
        q_w,
        kv_w,
        eps=1e-6,
        kv_lora_rank=kv_lora_rank,
    )

    torch.testing.assert_close(qr_out, qr_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        kv_out[..., :kv_lora_rank],
        kv_c_ref,
        atol=1e-2,
        rtol=1e-2,
    )
    assert torch.equal(kv_out[..., kv_lora_rank:], kv[..., kv_lora_rank:])


def test_cpu_worker_initializes_workspace_manager():
    """CPUWorker.init_device should initialize workspace before model runner."""
    import inspect
    import re

    from vllm.v1.worker import cpu_worker as cpu_worker_mod

    src = inspect.getsource(cpu_worker_mod.CPUWorker.init_device)
    code_lines = [
        line for line in src.splitlines() if not line.lstrip().startswith("#")
    ]
    code = "\n".join(code_lines)

    assert "init_workspace_manager(" in code
    ws_idx = code.find("init_workspace_manager(")
    runner_idx = code.find("CPUModelRunner(")
    assert ws_idx != -1 and runner_idx != -1
    assert ws_idx < runner_idx

    mod_src = inspect.getsource(cpu_worker_mod)
    assert re.search(
        r"from\s+vllm\.v1\.worker\.workspace\s+import\s+[^\n]*init_workspace_manager",
        mod_src,
    )


def test_unquantized_linear_method_preserves_bmm_weight_on_cpu(monkeypatch):
    """CPU unquantized GEMM dispatch should skip is_bmm linear layers."""
    from unittest.mock import patch

    from vllm.model_executor.layers import linear as linear_mod

    monkeypatch.setattr(linear_mod, "current_platform", _FakeCPUPlatform())

    plain = torch.nn.Linear(8, 16, bias=False, dtype=torch.bfloat16)
    with patch(
        "vllm.model_executor.layers.utils.dispatch_cpu_unquantized_gemm"
    ) as mock_dispatch:
        linear_mod.UnquantizedLinearMethod().process_weights_after_loading(plain)
    assert mock_dispatch.called

    bmm = torch.nn.Linear(8, 16, bias=False, dtype=torch.bfloat16)
    bmm.is_bmm = True
    bmm.bmm_batch_size = 2
    orig_weight = bmm.weight.detach().clone()
    with patch(
        "vllm.model_executor.layers.utils.dispatch_cpu_unquantized_gemm"
    ) as mock_dispatch:
        linear_mod.UnquantizedLinearMethod().process_weights_after_loading(bmm)
    assert not mock_dispatch.called
    assert torch.equal(bmm.weight.data, orig_weight)


def test_cpu_select_experts_supports_dsv4_sqrtsoftplus_with_bias():
    """CPU select_experts should support DSV4 sqrtsoftplus biased routing."""
    from torch.nn import functional as F

    from vllm.model_executor.layers.fused_moe.cpu_fused_moe import select_experts

    torch.manual_seed(0)
    num_tokens = 4
    num_experts = 8
    top_k = 2
    router_logits = torch.randn(num_tokens, num_experts, dtype=torch.float32)
    e_bias = torch.tensor(
        [0.5, -1.0, 0.0, 2.0, -0.5, 0.1, 0.0, -2.0],
        dtype=torch.float32,
    )

    for scoring_func in ("softmax", "sigmoid", "sqrtsoftplus"):
        if scoring_func == "softmax":
            scores = router_logits.softmax(dim=-1)
        elif scoring_func == "sigmoid":
            scores = router_logits.sigmoid()
        else:
            scores = F.softplus(router_logits).sqrt()

        scores_for_choice = scores + e_bias.unsqueeze(0)
        expected_idx = torch.topk(
            scores_for_choice,
            k=top_k,
            dim=-1,
            sorted=False,
        )[1]
        expected_vals = scores.gather(1, expected_idx)
        expected_vals = expected_vals / expected_vals.sum(
            dim=-1,
            keepdim=True,
        )
        expected_vals = (expected_vals * 2.5).to(torch.float32)

        vals, idx = select_experts(
            hidden_states=torch.empty(num_tokens, 16),
            router_logits=router_logits,
            top_k=top_k,
            use_grouped_topk=False,
            renormalize=True,
            scoring_func=scoring_func,
            e_score_correction_bias=e_bias,
            routed_scaling_factor=2.5,
        )

        assert idx.dtype == torch.int32
        assert vals.dtype == torch.float32
        assert idx.shape == (num_tokens, top_k)
        assert vals.shape == (num_tokens, top_k)
        assert torch.equal(
            idx.sort(dim=-1).values,
            expected_idx.to(torch.int32).sort(dim=-1).values,
        )
        torch.testing.assert_close(
            vals.sort(dim=-1).values,
            expected_vals.sort(dim=-1).values,
            rtol=1e-5,
            atol=1e-6,
        )

    with pytest.raises(ValueError, match="Unsupported scoring function"):
        select_experts(
            hidden_states=torch.empty(num_tokens, 16),
            router_logits=router_logits,
            top_k=top_k,
            use_grouped_topk=False,
            renormalize=True,
            scoring_func="not_a_real_scoring_func",
        )


def test_cpu_moe_act_fn_silu_does_not_require_vllm_config():
    """CPU MoE SILU activation should not instantiate a CustomOp."""
    from torch.nn import functional as F

    import vllm.config.vllm as vllm_cfg
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.cpu_fused_moe import _CPU_MOE_ACT_FN

    saved_cfg = vllm_cfg._current_vllm_config
    vllm_cfg._current_vllm_config = None
    vllm_cfg.get_cached_compilation_config.cache_clear()
    try:
        torch.manual_seed(0)
        x = torch.randn(3, 16, dtype=torch.float32)
        for act in (
            MoEActivation.SILU,
            MoEActivation.SWIGLUOAI,
            MoEActivation.GELU,
        ):
            out = _CPU_MOE_ACT_FN[act](x)
            assert out.shape == (3, 8)

        d = x.shape[-1] // 2
        expected = F.silu(x[..., :d]) * x[..., d:]
        torch.testing.assert_close(
            _CPU_MOE_ACT_FN[MoEActivation.SILU](x),
            expected,
            rtol=1e-6,
            atol=1e-7,
        )
    finally:
        vllm_cfg._current_vllm_config = saved_cfg
        vllm_cfg.get_cached_compilation_config.cache_clear()


def test_compressor_kv_score_uses_cpu_linear_output_helper():
    """Compressor side GEMMs should avoid CPU aten::mm.dtype dispatch."""
    import inspect

    from vllm.models.deepseek_v4 import attention as deepseek_attention

    wrapper_cls = deepseek_attention.DeepseekV4MultiHeadLatentAttentionWrapper
    src = inspect.getsource(wrapper_cls.attn_gemm_parallel_execute)
    code_only = "\n".join(
        line for line in src.splitlines() if not line.lstrip().startswith("#")
    )

    assert code_only.count("current_platform.is_cpu()") >= 2
    assert code_only.count("_linear_output_to_fp32(") >= 2


def test_compressor_fused_wkv_wgate_marked_is_bmm():
    """Compressor fused_wkv_wgate should preserve raw weight on CPU."""
    import inspect

    from vllm.models.deepseek_v4 import compressor as deepseek_compressor

    src = inspect.getsource(deepseek_compressor.DeepseekCompressor)
    assert "self.fused_wkv_wgate.is_bmm = True" in src
