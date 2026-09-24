# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused tests for the v0.28 DeepSeek V4 CPU platform package."""

import importlib
import sys
from types import SimpleNamespace

import pytest
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
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.model_loader.weight_utils import (
    safetensors_weights_iterator,
)
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.models.deepseek_v4.cpu import mhc as cpu_mhc
from vllm.models.deepseek_v4.cpu.attention import (
    DeepseekV4CPUAttention,
    _compressed_prefill_ranges,
)
from vllm.models.deepseek_v4.cpu.fp8_requant import (
    convert_block_fp8_weight,
    convert_fp8_checkpoint_for_cpu_w8a8,
    is_cpu_w8a8_target,
    make_cpu_w8a8_quantization_config,
)
from vllm.models.deepseek_v4.cpu.mhc import hc_head, mhc_post, mhc_pre
from vllm.models.deepseek_v4.cpu.model import (
    DeepseekV4DecoderLayer,
    DeepseekV4MoE,
    _configure_cpu_router_gate,
)
from vllm.models.deepseek_v4.cpu.mtp import DeepSeekV4MTP
from vllm.models.deepseek_v4.cpu.ops import (
    gather_paged_cache,
    save_compressor_states,
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


def test_attention_joint_int8_requires_fused_cpp_even_when_non_strict(
    monkeypatch,
):
    monkeypatch.setenv("VLLM_CPU_FUSED_CPP_STRICT", "0")
    monkeypatch.setitem(sys.modules, "fused_cpp", None)
    attention = object.__new__(DeepseekV4CPUAttention)
    torch.nn.Module.__init__(attention)
    attention._weights_prepared = False
    attention.indexer = None
    attention.wq_b = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
    attention.wo_b = torch.nn.Module()
    attention.wo_b.register_parameter(
        "weight",
        torch.nn.Parameter(torch.ones(2, 2, dtype=torch.int8), requires_grad=False),
    )
    attention.wo_b._cpu_fused_cpp_joint_int8_owned = True

    with pytest.raises(RuntimeError, match="fused_cpp import failed"):
        attention.process_weights_after_loading()


def test_int8_shared_experts_require_fused_moe_preparation():
    moe = object.__new__(DeepseekV4MoE)
    torch.nn.Module.__init__(moe)
    projection = torch.nn.Module()
    projection.register_parameter(
        "weight",
        torch.nn.Parameter(torch.ones(2, 2, dtype=torch.int8), requires_grad=False),
    )
    moe.shared_experts = SimpleNamespace(
        gate_up_proj=projection,
        down_proj=torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16),
    )

    with pytest.raises(RuntimeError, match="require fused_cpp Plan V2 preparation"):
        moe.ensure_joint_int8_experts_prepared()


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

    def prepare_mhc_weight(weight, **kwargs):
        calls.append(("prepare", weight, kwargs))
        return prepared

    def mhc_pre_rmsnorm(*args, **kwargs):
        calls.append(("pre", args, kwargs))
        return expected

    fake_mhc = SimpleNamespace(
        prepare_mhc_weight=prepare_mhc_weight,
        mhc_pre_rmsnorm_sve_candidate=mhc_pre_rmsnorm,
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


def test_compressor_states_persist_in_page_padded_cache():
    """Later compression must see KV and position-biased scores saved by Torch."""
    storage = torch.full((3, 12), -99.0)
    cache = storage[:, :8].view(3, 2, 4)
    kv = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.bfloat16)
    scores = torch.tensor([[10, 20], [30, 40], [50, 60]], dtype=torch.bfloat16)
    ape = torch.tensor([[0.5, 1.5], [2.5, 3.5]])

    save_compressor_states(
        kv, scores, ape, torch.tensor([3, 4, 5]), cache, torch.tensor([4, -1, 1]), 2
    )

    expected = torch.full_like(storage, -99.0)
    expected[2, :4] = torch.tensor([1, 2, 12.5, 23.5])
    expected[0, 4:8] = torch.tensor([5, 6, 52.5, 63.5])
    torch.testing.assert_close(storage, expected)


def test_strict_c4a_decode_post_returns_to_torch(monkeypatch):
    monkeypatch.setenv("VLLM_CPU_FUSED_CPP_STRICT", "1")
    attention = object.__new__(DeepseekV4CPUAttention)
    attention._fused_post_weights = object()
    attention._fused_ops = {}
    attention.compress_ratio = 4
    attention.prefix = "layers.0.self_attn"
    attention.indexer = SimpleNamespace(k_cache=SimpleNamespace(prefix="indexer.k"))
    attention.swa_cache_layer = SimpleNamespace(prefix="swa")
    metadata = {
        "swa": SimpleNamespace(num_decode_tokens=1, num_prefill_tokens=0),
        "layers.0.self_attn": SimpleNamespace(),
        "indexer.k": SimpleNamespace(),
    }

    result = attention._execute_fused_post(
        torch.empty(1, 4),
        torch.empty(1, 8),
        None,
        None,
        None,
        torch.tensor([8]),
        metadata,
    )

    assert result is None


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


def test_block_fp8_conversion_produces_int8_per_channel_and_bf16():
    source = torch.tensor(
        [
            [1.0, -2.0, 3.0, -4.0],
            [2.0, 1.0, -1.0, -2.0],
            [4.0, -3.0, 2.0, -1.0],
        ],
        dtype=torch.float32,
    ).to(torch.float8_e4m3fn)
    block_scales = torch.tensor([[0.5, 2.0], [1.5, 0.25]])
    expanded = torch.tensor(
        [
            [0.5, 0.5, 2.0, 2.0],
            [0.5, 0.5, 2.0, 2.0],
            [1.5, 1.5, 0.25, 0.25],
        ]
    )
    dequantized = source.float() * expanded

    int8_weight, int8_scale = convert_block_fp8_weight(
        source,
        block_scales,
        to_int8=True,
        block_size=(2, 2),
        rows_per_chunk=2,
    )
    assert int8_scale is not None
    expected_scale = dequantized.abs().amax(dim=1, keepdim=True) / 127
    expected_int8 = (
        dequantized.div(expected_scale).round().clamp(-127, 127).to(torch.int8)
    )
    torch.testing.assert_close(int8_scale, expected_scale)
    torch.testing.assert_close(int8_weight, expected_int8)

    bf16_weight, bf16_scale = convert_block_fp8_weight(
        source,
        block_scales,
        to_int8=False,
        block_size=(2, 2),
        rows_per_chunk=2,
    )
    assert bf16_scale is None
    torch.testing.assert_close(bf16_weight, dequantized.to(torch.bfloat16))


def test_fp8_checkpoint_stream_converts_supported_targets_only():
    fp8 = torch.tensor([[1.0, -2.0], [3.0, 4.0]]).to(torch.float8_e4m3fn)
    scale = torch.tensor([[0.5]], dtype=torch.float32)
    norm = torch.ones(2, dtype=torch.bfloat16)
    converted = dict(
        convert_fp8_checkpoint_for_cpu_w8a8(
            [
                ("layers.0.attn.wq_b.scale", scale),
                ("layers.0.attn.wq_b.weight", fp8),
                ("layers.0.attn.wq_a.weight", fp8),
                ("layers.0.attn.wq_a.scale", scale),
                ("layers.0.attn_norm.weight", norm),
            ],
            block_size=(2, 2),
            rows_per_chunk=2,
        )
    )
    assert converted["layers.0.attn.wq_b.weight"].dtype is torch.int8
    assert converted["layers.0.attn.wq_b.weight_scale"].shape == (2, 1)
    assert converted["layers.0.attn.wq_a.weight"].dtype is torch.bfloat16
    assert "layers.0.attn.wq_a.scale" not in converted
    assert converted["layers.0.attn_norm.weight"] is norm


def test_mtp_loader_converts_only_its_fp8_weight_stream():
    mtp = object.__new__(DeepSeekV4MTP)
    torch.nn.Module.__init__(mtp)
    mtp.config = SimpleNamespace(
        cpu_fp8_to_int8=True,
        cpu_fp8_source_block_size=(2, 2),
        cpu_fp8_conversion_rows_per_chunk=2,
    )
    fp8 = torch.tensor([[1.0, -2.0], [3.0, 4.0]]).to(torch.float8_e4m3fn)
    scale = torch.tensor([[0.5]], dtype=torch.float32)

    converted = dict(
        mtp._prepare_weight_stream(
            [
                ("layers.0.attn.wq_b.scale", scale),
                ("layers.0.attn.wq_b.weight", fp8),
                ("mtp.0.attn.wq_b.scale", scale),
                ("mtp.0.attn.wq_b.weight", fp8),
                ("mtp.0.attn.wq_a.weight", fp8),
                ("mtp.0.attn.wq_a.scale", scale),
                ("mtp.0.ffn.experts.3.w2.scale", scale),
                ("mtp.0.ffn.experts.3.w2.weight", fp8),
            ],
            tp_rank=0,
            tp_size=1,
        )
    )

    assert "layers.0.attn.wq_b.weight" not in converted
    assert converted["mtp.0.attn.wq_b.weight"].dtype is torch.int8
    assert converted["mtp.0.attn.wq_b.weight_scale"].shape == (2, 1)
    assert converted["mtp.0.attn.wq_a.weight"].dtype is torch.bfloat16
    assert "mtp.0.attn.wq_a.scale" not in converted
    assert converted["mtp.0.ffn.experts.3.w2.weight"].dtype is torch.int8
    assert converted["mtp.0.ffn.experts.3.w2.weight_scale"].shape == (2, 1)


def test_pro_cpu_w8a8_targets_and_ignore_config_cover_even_indexers():
    assert is_cpu_w8a8_target("layers.0.ffn.experts.383.w2.weight")
    assert is_cpu_w8a8_target("layers.0.ffn.shared_experts.w3.weight")
    assert is_cpu_w8a8_target("layers.60.attn.indexer.wq_b.weight")
    assert not is_cpu_w8a8_target("layers.60.attn.wq_a.weight")

    config = make_cpu_w8a8_quantization_config(61)
    ignore = config["ignore"]
    assert isinstance(ignore, list)
    assert "layers.60.attn.indexer.weights_proj" in ignore
    assert "layers.60.attn.wq_a" in ignore
    assert "layers.60.attn.wq_b" not in ignore
    assert "mtp.0.attn.wq_b" in ignore


def test_fp8_checkpoint_stream_slices_column_parallel_before_conversion():
    source = torch.arange(32, dtype=torch.float32).reshape(8, 4).to(torch.float8_e4m3fn)
    scale = torch.ones(4, 2, dtype=torch.float32)
    converted = dict(
        convert_fp8_checkpoint_for_cpu_w8a8(
            [
                ("layers.0.ffn.experts.0.w1.scale", scale),
                ("layers.0.ffn.experts.0.w1.weight", source),
            ],
            block_size=(2, 2),
            rows_per_chunk=2,
            tp_rank=2,
            tp_size=4,
        )
    )
    weight = converted["layers.0.ffn.experts.0.w1.weight"]
    weight_scale = converted["layers.0.ffn.experts.0.w1.weight_scale"]
    assert weight.shape == (2, 4)
    assert weight_scale.shape == (2, 1)
    assert weight._vllm_tp_pre_sharded
    assert weight_scale._vllm_tp_pre_sharded

    row_parallel = dict(
        convert_fp8_checkpoint_for_cpu_w8a8(
            [
                ("layers.0.ffn.experts.0.w2.scale", scale),
                ("layers.0.ffn.experts.0.w2.weight", source),
            ],
            block_size=(2, 2),
            rows_per_chunk=2,
            tp_rank=2,
            tp_size=4,
        )
    )["layers.0.ffn.experts.0.w2.weight"]
    assert row_parallel.shape == source.shape
    assert not getattr(row_parallel, "_vllm_tp_pre_sharded", False)

    replicated_indexer = dict(
        convert_fp8_checkpoint_for_cpu_w8a8(
            [
                ("layers.0.attn.indexer.wq_b.scale", scale),
                ("layers.0.attn.indexer.wq_b.weight", source),
            ],
            block_size=(2, 2),
            rows_per_chunk=2,
            tp_rank=2,
            tp_size=4,
        )
    )["layers.0.attn.indexer.wq_b.weight"]
    assert replicated_indexer.shape == source.shape
    assert not getattr(replicated_indexer, "_vllm_tp_pre_sharded", False)


def test_pre_sharded_linear_and_moe_loaders_do_not_slice_twice(monkeypatch):
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1
    )
    linear = ModelWeightParameter(
        data=torch.zeros(2, 3),
        input_dim=1,
        output_dim=0,
        weight_loader=lambda _: None,
    )
    linear.tp_rank = 3
    local = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    local._vllm_tp_pre_sharded = True
    linear.load_column_parallel_weight(local)
    torch.testing.assert_close(linear, local)

    merged = ModelWeightParameter(
        data=torch.zeros(4, 3),
        input_dim=1,
        output_dim=0,
        weight_loader=lambda _: None,
    )
    merged.tp_rank = 3
    merged.load_merged_column_weight(local, shard_offset=2, shard_size=2)
    torch.testing.assert_close(merged[2:], local)
    assert torch.count_nonzero(merged[:2]) == 0

    routed = object.__new__(RoutedExperts)
    routed.moe_config = SimpleNamespace(
        is_act_and_mul=True,
        moe_parallel_config=SimpleNamespace(tp_size=4),
    )
    destination = torch.zeros(4, 3)
    routed._load_w13(
        destination,
        shard_dim=0,
        shard_id="w1",
        loaded_weight=local,
        tp_rank=3,
    )
    torch.testing.assert_close(destination[:2], local)
    assert torch.count_nonzero(destination[2:]) == 0
