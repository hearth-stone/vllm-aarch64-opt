# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v4.cpu import cpu_sparse_attn_prefill

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


class _FakeCPUPlatform:
    def is_cuda(self):
        return False

    def is_cpu(self):
        return True

    def is_rocm(self):
        return False

    def is_xpu(self):
        return False


def test_deepseek_v4_sparse_impl_uses_cpu_fallback(monkeypatch):
    from vllm.models.deepseek_v4 import attention as deepseek_attention
    from vllm.models.deepseek_v4.cpu import DeepseekV4CPUSparseMLAImpl

    monkeypatch.setattr(deepseek_attention, "current_platform", _FakeCPUPlatform())

    impl_cls = deepseek_attention._select_v4_sparse_impl()

    assert impl_cls is DeepseekV4CPUSparseMLAImpl
    assert impl_cls.backend_cls.get_supported_head_sizes() == [512]


def test_deepseek_v4_cpu_compressor_scores_read_raw_weight(monkeypatch):
    from vllm.models.deepseek_v4 import attention as deepseek_attention

    class RawWeightLinear(torch.nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.weight = torch.nn.Parameter(weight, requires_grad=False)

        def forward(self, hidden_states):
            raise AssertionError("CPU compressor score path must not call forward")

    class FakeWeightsProj(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states[:, :2].contiguous(), None

    hidden_states = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    compressor_weight = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    indexer_weight = torch.arange(20, dtype=torch.float32).reshape(5, 4)

    wrapper_cls = deepseek_attention.DeepseekV4MultiHeadLatentAttentionWrapper
    wrapper = wrapper_cls.__new__(wrapper_cls)
    wrapper.aux_stream_list = None
    wrapper.ln_events = [None] * 4
    wrapper.compressor = SimpleNamespace(
        fused_wkv_wgate=RawWeightLinear(compressor_weight)
    )
    wrapper.indexer = SimpleNamespace(
        weights_proj=FakeWeightsProj(),
        compressor=SimpleNamespace(fused_wkv_wgate=RawWeightLinear(indexer_weight)),
    )
    wrapper.fused_wqa_wkv = lambda x: (torch.empty((x.shape[0], 3)), None)

    monkeypatch.setattr(deepseek_attention, "current_platform", _FakeCPUPlatform())

    _, kv_score, indexer_kv_score, indexer_weights = (
        wrapper_cls.attn_gemm_parallel_execute(wrapper, hidden_states)
    )

    assert kv_score.dtype is torch.float32
    assert indexer_kv_score.dtype is torch.float32
    torch.testing.assert_close(
        kv_score,
        torch.mm(hidden_states, compressor_weight.T).to(torch.float32),
    )
    torch.testing.assert_close(
        indexer_kv_score,
        torch.mm(hidden_states, indexer_weight.T).to(torch.float32),
    )
    torch.testing.assert_close(indexer_weights, hidden_states[:, :2])


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


def test_deepseek_v4_decoder_layer_uses_native_forward_on_cpu(monkeypatch):
    """CPU forward must avoid CUDA-only mhc_fused_post_pre path."""
    from vllm.models.deepseek_v4.nvidia import model as nvidia_model

    monkeypatch.setattr(nvidia_model, "current_platform", _FakeCPUPlatform())
    layer = object.__new__(nvidia_model.DeepseekV4DecoderLayer)

    def _forward_native(*args, **kwargs):
        return "native", None, None, None

    def _forward_cuda(*args, **kwargs):
        raise AssertionError("CPU forward must not call _forward_cuda")

    layer._forward_native = _forward_native
    layer._forward_cuda = _forward_cuda

    output, residual, post_mix, res_mix = nvidia_model.DeepseekV4DecoderLayer.forward(
        layer,
        torch.empty(1, 1),
        torch.arange(1),
        None,
    )

    assert output == "native"
    assert residual is None
    assert post_mix is None
    assert res_mix is None


def test_deepseek_v4_aux_streams_are_disabled_on_cpu():
    """CPU model init must not create torch.cuda.Stream objects."""
    import inspect

    from vllm.models.deepseek_v4.amd import model as amd_model
    from vllm.models.deepseek_v4.amd import mtp as amd_mtp
    from vllm.models.deepseek_v4.nvidia import model as nvidia_model
    from vllm.models.deepseek_v4.nvidia import mtp as nvidia_mtp

    sources = [
        inspect.getsource(nvidia_model),
        inspect.getsource(nvidia_mtp),
        inspect.getsource(amd_model),
        inspect.getsource(amd_mtp),
    ]
    for src in sources:
        code = "\n".join(
            line for line in src.splitlines() if not line.lstrip().startswith("#")
        )
        stream_idx = code.find("torch.cuda.Stream()")
        cpu_guard_idx = code.find("current_platform.is_cpu()")

        assert stream_idx != -1
        assert cpu_guard_idx != -1
        assert cpu_guard_idx < stream_idx


def test_deepseek_v4_hc_head_native_matches_torch_reference():
    """CPU hc_head native path should match the v0.21 torch fallback math."""
    from vllm.model_executor.layers.mhc import HCHeadOp

    torch.manual_seed(3)
    batch_size = 2
    seq_len = 3
    hc_mult = 2
    hidden_size = 32

    hidden_states = torch.randn(
        batch_size,
        seq_len,
        hc_mult,
        hidden_size,
        dtype=torch.bfloat16,
    )
    hc_fn = torch.randn(hc_mult, hc_mult * hidden_size, dtype=torch.float32)
    hc_scale = torch.randn(1, dtype=torch.float32)
    hc_base = torch.randn(hc_mult, dtype=torch.float32)

    op = object.__new__(HCHeadOp)
    actual = op.forward_native(
        hidden_states,
        hc_fn,
        hc_scale,
        hc_base,
        rms_norm_eps=1e-6,
        hc_eps=1e-3,
    )

    hs_flat = hidden_states.reshape(-1, hc_mult, hidden_size)
    x = hs_flat.reshape(-1, hc_mult * hidden_size).to(torch.float32)
    mixes = torch.matmul(x, hc_fn.t())
    sqrsum = x.square().sum(dim=-1, keepdim=True)
    rsqrt = torch.rsqrt(sqrsum / (hc_mult * hidden_size) + 1e-6)
    pre_mix = torch.sigmoid(mixes * rsqrt * hc_scale[0] + hc_base) + 1e-3
    expected = torch.sum(
        pre_mix.unsqueeze(-1) * hs_flat.to(torch.float32),
        dim=1,
    ).to(torch.bfloat16)
    expected = expected.reshape(batch_size, seq_len, hidden_size)

    assert actual.shape == (batch_size, seq_len, hidden_size)
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected)


def test_deepseek_v4_hc_head_native_handles_empty_tokens():
    from vllm.model_executor.layers.mhc import HCHeadOp

    hc_mult = 2
    hidden_size = 32
    op = object.__new__(HCHeadOp)
    actual = op.forward_native(
        torch.empty(0, hc_mult, hidden_size, dtype=torch.bfloat16),
        torch.empty(hc_mult, hc_mult * hidden_size, dtype=torch.float32),
        torch.empty(1, dtype=torch.float32),
        torch.empty(hc_mult, dtype=torch.float32),
        rms_norm_eps=1e-6,
        hc_eps=1e-3,
    )

    assert actual.shape == (0, hidden_size)
    assert actual.dtype == torch.bfloat16


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


def test_cpu_q_kv_rmsnorm_matches_native_rmsnorm():
    """CPU q/kv RMSNorm should match RMSNorm.forward_native."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.device import DeviceConfig
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

    with set_current_vllm_config(
        VllmConfig(device_config=DeviceConfig(device="cpu"))
    ):
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


def test_compressor_kv_score_avoids_mm_dtype_overload_on_cpu():
    """Compressor side GEMMs should avoid CPU aten::mm.dtype dispatch."""
    import inspect

    from vllm.models.deepseek_v4 import attention as deepseek_attention

    wrapper_cls = deepseek_attention.DeepseekV4MultiHeadLatentAttentionWrapper
    src = inspect.getsource(wrapper_cls.attn_gemm_parallel_execute)
    code_only = "\n".join(
        line for line in src.splitlines() if not line.lstrip().startswith("#")
    )

    assert code_only.count("current_platform.is_cpu()") >= 2
    assert code_only.count("torch.mm(hidden_states, w).to(torch.float32)") >= 2
    assert code_only.count("out_dtype=torch.float32") >= 2
    assert "_linear_output_to_fp32(" not in code_only


def test_compressor_fused_wkv_wgate_marked_is_bmm():
    """Compressor fused_wkv_wgate should preserve raw weight on CPU."""
    import inspect

    from vllm.models.deepseek_v4 import compressor as deepseek_compressor

    src = inspect.getsource(deepseek_compressor.DeepseekCompressor)
    assert "self.fused_wkv_wgate.is_bmm = True" in src


def test_kv_cache_groups_pad_full_mla_when_swa_exceeds():
    """Full-MLA pages should pad up when an SWA MLA page is larger."""
    from vllm.v1.core.kv_cache_utils import _get_kv_cache_groups_uniform_groups
    from vllm.v1.kv_cache_interface import (
        MLAAttentionSpec,
        SlidingWindowMLASpec,
        UniformTypeKVCacheSpecs,
    )

    full_mla_specs = {
        "layer.0.full": MLAAttentionSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.bfloat16,
        ),
        "layer.1.full": MLAAttentionSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.bfloat16,
        ),
    }
    full_mla_group = UniformTypeKVCacheSpecs.from_specs(full_mla_specs)
    assert full_mla_group is not None
    assert max(full_mla_group.get_page_sizes()) == 18432

    swa_specs = {
        "layer.0.swa": SlidingWindowMLASpec(
            block_size=4,
            num_kv_heads=1,
            head_size=2048,
            dtype=torch.float32,
            sliding_window=8,
        ),
    }
    swa_group = UniformTypeKVCacheSpecs.from_specs(swa_specs)
    assert swa_group is not None
    assert max(swa_group.get_page_sizes()) == 32768

    result = _get_kv_cache_groups_uniform_groups([full_mla_group, swa_group])

    assert len(result) >= 2
    assert {"layer.0.full", "layer.1.full"} <= set(result[0].layer_names)
    for layer_name in ("layer.0.full", "layer.1.full"):
        spec = full_mla_specs[layer_name]
        assert spec.page_size_padded == 32768
        assert spec.page_size_bytes == 32768

    swa_spec = swa_specs["layer.0.swa"]
    assert swa_spec.page_size_padded is None or swa_spec.page_size_padded == 32768


def test_kv_cache_groups_unchanged_when_swa_fits():
    """Full-MLA pages should not pad when all SWA pages already fit."""
    from vllm.v1.core.kv_cache_utils import _get_kv_cache_groups_uniform_groups
    from vllm.v1.kv_cache_interface import (
        MLAAttentionSpec,
        SlidingWindowMLASpec,
        UniformTypeKVCacheSpecs,
    )

    full_mla_specs = {
        "layer.0.full": MLAAttentionSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.bfloat16,
        ),
        "layer.1.full": MLAAttentionSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.bfloat16,
        ),
    }
    full_mla_group = UniformTypeKVCacheSpecs.from_specs(full_mla_specs)
    assert full_mla_group is not None

    swa_specs = {
        "layer.0.swa": SlidingWindowMLASpec(
            block_size=16,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.bfloat16,
            sliding_window=128,
        ),
    }
    swa_group = UniformTypeKVCacheSpecs.from_specs(swa_specs)
    assert swa_group is not None

    _get_kv_cache_groups_uniform_groups([full_mla_group, swa_group])

    for layer_name in ("layer.0.full", "layer.1.full"):
        assert full_mla_specs[layer_name].page_size_padded is None
