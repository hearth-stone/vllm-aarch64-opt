# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused tests for the v0.28 DeepSeek V4 CPU platform package."""

import importlib
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

import vllm._custom_ops as ops
import vllm.envs as envs
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts import fused_cpp_cpu_moe
from vllm.model_executor.layers.fused_moe.experts.fused_cpp_cpu_moe import (
    FusedCppArmExperts,
)
from vllm.model_executor.model_loader.weight_utils import (
    safetensors_weights_iterator,
)
from vllm.models.deepseek_v4.cpu import mhc as cpu_mhc
from vllm.models.deepseek_v4.cpu.attention import (
    DeepseekV4CPUAttention,
    _compressed_prefill_ranges,
)
from vllm.models.deepseek_v4.cpu.mhc import hc_head, mhc_post, mhc_pre
from vllm.models.deepseek_v4.cpu.model import (
    DeepseekV4DecoderLayer,
    DeepseekV4MoE,
    _configure_cpu_router_gate,
)
from vllm.models.deepseek_v4.cpu.ops import (
    gather_paged_cache,
    sparse_mla_reference,
    write_paged_cache,
)
from vllm.models.deepseek_v4.cpu.sparse_mla import (
    DeepseekV4CPUSWAMetadataBuilder,
    compressed_slot_mapping_torch,
)
from vllm.platforms.cpu import CpuPlatform
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def test_cpu_package_does_not_import_platform_gpu_model_modules():
    importlib.import_module("vllm.models.deepseek_v4.cpu")
    forbidden = (
        "vllm.models.deepseek_v4.nvidia.model",
        "vllm.models.deepseek_v4.amd.model",
        "vllm.models.deepseek_v4.xpu.model",
        "vllm.model_executor.kernels.mhc.tilelang",
    )
    assert all(name not in sys.modules for name in forbidden)


def test_cpu_platform_selects_v4_sparse_mla_backend():
    selector = SimpleNamespace(use_sparse=True, use_mla=True)
    path = CpuPlatform.get_attn_backend_cls(
        AttentionBackendEnum.CPU_MLA_SPARSE_DSV4,
        selector,
    )
    assert path.endswith("DeepseekV4CPUSparseMLABackend")


def test_attention_post_load_hook_accepts_dtype_and_is_idempotent():
    attention = object.__new__(DeepseekV4CPUAttention)
    attention._weights_prepared = True
    attention.process_weights_after_loading(torch.bfloat16)


def test_cpu_sqrtsoftplus_routing_matches_v028_reference():
    gating = torch.tensor(
        [[-1.0, 0.5, 2.0, -0.25], [float("nan")] * 4],
        dtype=torch.bfloat16,
    )
    bias = torch.tensor([0.3, -0.2, 0.1, 0.0])
    weights = torch.empty(2, 2, dtype=torch.float32)
    ids = torch.empty(2, 2, dtype=torch.int32)
    source_rows = torch.empty_like(ids)
    ops.topk_hash_softplus_sqrt(
        weights,
        ids,
        source_rows,
        gating,
        renormalize=True,
        routed_scaling_factor=1.5,
        e_score_correction_bias=bias,
        is_padding=torch.tensor([False, True]),
    )

    scores = F.softplus(gating[:1].float()).sqrt()
    expected_ids = torch.topk(
        scores + bias,
        2,
        dim=-1,
        sorted=envs.VLLM_BATCH_INVARIANT,
    ).indices
    expected_weights = scores.gather(1, expected_ids)
    expected_weights = expected_weights / expected_weights.sum(-1, keepdim=True) * 1.5
    torch.testing.assert_close(ids[0], expected_ids[0].to(torch.int32))
    torch.testing.assert_close(weights[0], expected_weights[0])
    assert ids[1].tolist() == [-1, -1]
    assert weights[1].tolist() == [0.0, 0.0]
    assert source_rows.tolist() == [[0, 2], [1, 3]]


def test_all_zero_hash_table_uses_score_routing():
    moe = object.__new__(DeepseekV4MoE)
    moe.prefix = "model.layers.0.ffn"
    table = torch.zeros(8, 2, dtype=torch.int32)
    moe.gate = SimpleNamespace(tid2eid=table)
    router = SimpleNamespace(_hash_indices_table=table)
    moe.experts = SimpleNamespace(router=router)
    assert moe.disable_placeholder_hash_routing()
    assert router._hash_indices_table is None


def test_router_gate_uses_fused_cpp_fp32_contract():
    gate = SimpleNamespace()
    _configure_cpu_router_gate(gate, strict=True)
    assert gate._cpu_fused_cpp_linear_enabled
    assert gate._cpu_fused_cpp_linear_required
    assert gate._cpu_fused_cpp_out_dtype is torch.float32


def test_nonzero_hash_table_remains_enabled():
    moe = object.__new__(DeepseekV4MoE)
    moe.prefix = "model.layers.0.ffn"
    table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    moe.gate = SimpleNamespace(tid2eid=table)
    router = SimpleNamespace(_hash_indices_table=table)
    moe.experts = SimpleNamespace(router=router)
    assert not moe.disable_placeholder_hash_routing()
    assert router._hash_indices_table is table


def test_torch_mhc_reference_shapes_and_head_formula():
    torch.manual_seed(0)
    tokens, hc_mult, hidden = 3, 2, 8
    residual = torch.randn(tokens, hc_mult, hidden, dtype=torch.bfloat16)
    mix = (2 + hc_mult) * hc_mult
    fn = torch.randn(mix, hc_mult * hidden, dtype=torch.float32)
    scale = torch.randn(3, dtype=torch.float32)
    base = torch.randn(mix, dtype=torch.float32)
    post, combine, layer_input = mhc_pre(residual, fn, scale, base, 1e-6, 1e-6, 2.0, 3)
    assert post.shape == (tokens, hc_mult, 1)
    assert combine.shape == (tokens, hc_mult, hc_mult)
    assert layer_input.shape == (tokens, hidden)
    reconstructed = mhc_post(layer_input, residual, post, combine)
    assert reconstructed.shape == residual.shape

    head_fn = fn[:hc_mult]
    head = hc_head(residual, head_fn, scale[:1], base[:hc_mult], 1e-6, 1e-6)
    x = residual.reshape(tokens, -1).float()
    rrms = torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
    gates = torch.sigmoid((x @ head_fn.t()) * rrms * scale[0] + base[:hc_mult])
    expected = ((gates + 1e-6).unsqueeze(-1) * residual.float()).sum(1)
    torch.testing.assert_close(head.float(), expected.to(torch.bfloat16).float())


def test_fused_cpp_mhc_adapter_prepares_once_and_forwards_contract(monkeypatch):
    calls = []
    prepared = object()
    expected = (
        torch.empty(2, 4, 1),
        torch.empty(2, 4, 4),
        torch.empty(2, 8, dtype=torch.bfloat16),
    )

    fake_mhc = SimpleNamespace(
        prepare_mhc_weight=lambda weight, **kwargs: (
            calls.append(("prepare", weight, kwargs)) or prepared
        ),
        mhc_pre_rmsnorm_sve_candidate=lambda *args, **kwargs: (
            calls.append(("pre", args, kwargs)) or expected
        ),
    )
    monkeypatch.setattr(cpu_mhc, "_load_fused_cpp_mhc", lambda: fake_mhc)
    weight = torch.empty(24, 32, dtype=torch.float32)
    assert (
        cpu_mhc.prepare_fused_cpp_mhc_weight(
            weight,
            kind="pre",
            required=True,
        )
        is prepared
    )
    residual = torch.empty(2, 4, 8, dtype=torch.bfloat16)
    scale = torch.empty(3)
    base = torch.empty(24)
    norm_weight = torch.empty(8, dtype=torch.bfloat16)
    actual = cpu_mhc.fused_cpp_mhc_pre_rmsnorm(
        residual,
        prepared,
        scale,
        base,
        norm_weight,
        1e-6,
        1e-6,
        2.0,
        20,
    )
    assert actual is expected
    assert calls[0] == ("prepare", weight, {"kind": "pre"})
    assert calls[1][1][:5] == (residual, prepared, scale, base, norm_weight)
    assert calls[1][2] == {
        "rms_eps": 1e-6,
        "hc_pre_eps": 1e-6,
        "hc_sinkhorn_eps": 1e-6,
        "hc_post_mult_value": 2.0,
        "sinkhorn_repeat": 20,
        "norm_eps": 1e-6,
        "num_threads": torch.get_num_threads(),
    }


def test_decoder_mhc_weights_are_prepared_once(monkeypatch):
    calls = []

    def prepare(weight, *, kind, required):
        calls.append((weight, kind, required))
        return object()

    monkeypatch.setenv("VLLM_CPU_FUSED_CPP_STRICT", "1")
    monkeypatch.setattr(
        "vllm.models.deepseek_v4.cpu.model.prepare_fused_cpp_mhc_weight",
        prepare,
    )
    layer = object.__new__(DeepseekV4DecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.hc_attn_fn = torch.nn.Parameter(torch.empty(24, 32))
    layer.hc_ffn_fn = torch.nn.Parameter(torch.empty(24, 32))
    layer._prepared_hc_attn_fn = None
    layer._prepared_hc_ffn_fn = None

    layer.process_mhc_weights_after_loading()
    layer.process_mhc_weights_after_loading()

    assert len(calls) == 2
    assert [call[1] for call in calls] == ["pre", "pre"]
    assert all(call[2] for call in calls)


def test_compressed_slot_mapping_handles_padding_and_page_lookup():
    cm = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 4], dtype=torch.int32),
        seq_lens=torch.tensor([4], dtype=torch.int32),
        num_reqs=1,
        num_actual_tokens=4,
        max_query_len=4,
        max_seq_len=4,
        block_table_tensor=torch.tensor([[7, 9]], dtype=torch.int32),
        slot_mapping=torch.tensor([0, 1, 2, 3], dtype=torch.int64),
    )
    output = torch.empty(8, dtype=torch.int64)
    actual = compressed_slot_mapping_torch(cm, 2, 2, output)
    assert actual.tolist() == [-1, 14, -1, 15]
    assert output[4:].tolist() == [-1, -1, -1, -1]


def test_compressed_slot_mapping_vectorizes_positions_and_padding():
    cm = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 3, 6], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 3, 6], dtype=torch.int32),
        seq_lens=torch.tensor([3, 3], dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=6,
        max_query_len=3,
        max_seq_len=3,
        block_table_tensor=torch.tensor([[7], [9]], dtype=torch.int32),
        slot_mapping=torch.tensor([0, 1, 2, 3, -1, 5], dtype=torch.int64),
        positions=torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int64),
    )
    output = torch.empty(6, dtype=torch.int64)
    actual = compressed_slot_mapping_torch(cm, 4, 2, output)
    assert actual.tolist() == [-1, 28, -1, -1, -1, -1]


def test_compressed_prefill_ranges_are_vectorized_per_request():
    cu_lens, starts, ends = _compressed_prefill_ranges(
        positions=torch.tensor([0, 1, 4, 5, 6]),
        req_ids=torch.tensor([0, 0, 1, 1, 1], dtype=torch.int32),
        seq_lens=torch.tensor([2, 7], dtype=torch.int32),
        compress_ratio=2,
    )
    assert cu_lens.tolist() == [0, 1, 4]
    assert starts.tolist() == [0, 0, 1, 1, 1]
    assert ends.tolist() == [0, 1, 3, 4, 4]


def test_page_padded_cache_write_and_gather():
    storage = torch.zeros(2, 5, 4, dtype=torch.bfloat16)
    cache = storage[:, :3, :]
    assert not cache.is_contiguous()
    values = torch.arange(12, dtype=torch.float32).reshape(3, 4).to(torch.bfloat16)
    slots = torch.tensor([0, -1, 5])
    write_paged_cache(cache, values, slots)
    gathered = gather_paged_cache(cache, torch.tensor([0, 5]))
    torch.testing.assert_close(gathered, values[[0, 2]])


def test_strict_swa_metadata_skips_unused_prefill_index_matrix(monkeypatch):
    monkeypatch.setenv("VLLM_CPU_FUSED_CPP_STRICT", "1")
    spec = SimpleNamespace(sliding_window=8, block_size=4)
    config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=32),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        speculative_config=None,
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )
    builder = DeepseekV4CPUSWAMetadataBuilder(
        spec, ["layer"], config, torch.device("cpu")
    )
    cm = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 4], dtype=torch.int32),
        seq_lens=torch.tensor([4], dtype=torch.int32),
        num_reqs=1,
        num_actual_tokens=4,
        max_query_len=4,
        max_seq_len=4,
        block_table_tensor=torch.tensor([[2]], dtype=torch.int32),
        slot_mapping=torch.tensor([8, 9, 10, 11], dtype=torch.int64),
        positions=torch.arange(4, dtype=torch.int64),
    )
    metadata = builder.build(0, cm)
    assert metadata.num_decode_tokens == 0
    assert metadata.prefill_swa_indices is None
    assert metadata.prefill_gather_lens.tolist() == [4]


def test_sparse_mla_reference_includes_sink_denominator():
    q = torch.tensor([[[1.0, 0.0]]], dtype=torch.bfloat16)
    keys = [torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.bfloat16)]
    out = torch.empty_like(q)
    sink = torch.tensor([0.0], dtype=torch.float32)
    sparse_mla_reference(q, keys, 1.0, sink, out)
    logits = torch.tensor([1.0, 0.0, 0.0])
    probs = logits.softmax(0)
    expected = probs[0] * keys[0][0].float() + probs[1] * keys[0][1].float()
    torch.testing.assert_close(out[0, 0].float(), expected.to(torch.bfloat16).float())


def test_prepack_iterator_skips_expert_weights_and_scales(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    tensors = {
        "model.layers.0.ffn.experts.0.w1.weight": torch.ones(2, 2),
        "model.layers.0.ffn.experts.0.w1.scale": torch.ones(2),
        "model.layers.0.ffn.shared_experts.w2.weight": torch.ones(2, 2),
        "model.layers.0.attn.wq_b.weight": torch.ones(2, 2),
    }
    save_file(tensors, checkpoint)
    loaded = dict(
        safetensors_weights_iterator(
            [str(checkpoint)],
            use_tqdm_on_load=False,
            skip_prepacked_moe_weights=True,
        )
    )
    assert loaded.keys() == {"model.layers.0.attn.wq_b.weight"}


def test_fused_cpp_experts_reuses_modular_moe_output(monkeypatch):
    runtime = object()
    calls = []

    def execute(hidden_states, prepared, topk_weights, topk_ids, **kwargs):
        calls.append((hidden_states, prepared, topk_weights, topk_ids, kwargs))
        kwargs["out"].fill_(3)
        return kwargs["out"]

    fake_moe = SimpleNamespace(get_default_moe_planner_runtime=lambda: runtime)
    monkeypatch.setattr(fused_cpp_cpu_moe, "_load_fused_cpp_moe", lambda: fake_moe)
    experts = object.__new__(FusedCppArmExperts)
    experts.prepared = object()
    experts.runtime = runtime
    experts.execute = execute
    experts.num_threads = 80
    experts.moe_config = SimpleNamespace(swiglu_limit=10.0)
    hidden = torch.randn(4, 8, dtype=torch.bfloat16)
    output = torch.empty_like(hidden)
    weights = torch.empty(0)
    topk_weights = torch.rand(4, 2)
    topk_ids = torch.zeros(4, 2, dtype=torch.int32)
    experts.apply(
        output,
        hidden,
        weights,
        weights,
        topk_weights,
        topk_ids,
        MoEActivation.SILU,
        8,
        None,
        None,
        None,
        torch.empty(0),
        torch.empty(0),
        None,
        False,
    )
    assert torch.all(output == 3)
    assert calls[0][-1]["out"] is output
    assert calls[0][-1]["num_threads"] == 80
    assert calls[0][-1]["swiglu_limit"] == 10.0


def test_fused_cpp_prepack_key_covers_rank_shapes_quant_and_abi(monkeypatch):
    projection = lambda weight_shape, scale_shape: SimpleNamespace(
        weight=torch.empty(weight_shape, dtype=torch.int8),
        weight_scale=torch.empty(scale_shape, dtype=torch.float32),
    )
    layer = SimpleNamespace(
        layer_name="model.layers.7.ffn.experts",
        w13_weight=torch.empty(8, 16, 4, dtype=torch.int8),
        w2_weight=torch.empty(8, 4, 8, dtype=torch.int8),
        w13_weight_scale=torch.empty(8, 16, 1),
        w2_weight_scale=torch.empty(8, 4, 1),
        _dsv4_shared_experts=SimpleNamespace(
            gate_up_proj=projection((16, 4), (16, 1)),
            down_proj=projection((4, 8), (4, 1)),
        ),
    )
    config = SimpleNamespace(model_config=SimpleNamespace(model="dsv4-test"))
    monkeypatch.setattr(fused_cpp_cpu_moe, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(
        fused_cpp_cpu_moe, "get_tensor_model_parallel_world_size", lambda: 4
    )
    monkeypatch.setattr(fused_cpp_cpu_moe, "get_tensor_model_parallel_rank", lambda: 2)
    monkeypatch.setattr(fused_cpp_cpu_moe, "_layer_id", lambda _: 7)
    key = fused_cpp_cpu_moe._cache_key(layer, "W8A8")
    assert key["backend_abi"] == "fused_cpp-482e03e-plan-v2"
    assert (key["model"], key["tp_size"], key["tp_rank"]) == (
        "dsv4-test",
        4,
        2,
    )
    assert key["routed_w13_shape"] == (8, 16, 4)
    assert key["shared_w2_scale_shape"] == (4, 1)
    assert key["quant_mode"] == "W8A8"
