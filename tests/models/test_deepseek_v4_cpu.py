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
