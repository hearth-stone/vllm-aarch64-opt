# -*- coding: utf-8 -*-
"""对比测试：CPUMLAImpl vs transformers.DeepseekV3Attention（eager 模式）。

测试目标：
    验证 CPUMLAImpl 的输出与 transformers 库中 DeepseekV3Attention 在 eager
    模式下的输出余弦相似度，确保两者在数值上高度一致。

测试场景：
    - Prefill-only（纯 prefill，无历史 context）
    - Decode-only（纯 decode，query_len == 1）
    - Mixed（prefill + decode 混合批次）

运行方式：
    pytest tests/v1/attention/test_cpu_mla_vs_hf.py -v
    # 或直接运行（输出详细的余弦相似度报告）：
    python tests/v1/attention/test_cpu_mla_vs_hf.py
"""
import argparse
import sys
from dataclasses import dataclass
from typing import Optional

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config.vllm import set_current_vllm_config
from vllm.model_executor.layers.attention.mla_attention import _DecodeConcatQuantFP8
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.kv_cache_interface import MLAAttentionSpec

# ──────────────────────────────────────────────────────────────────────────────
# DeepSeek V3 MLA 超参数（与 DeepSeek-V3/R1 实际配置一致）
# ──────────────────────────────────────────────────────────────────────────────
NUM_HEADS = 8           # 测试用，实际为 128
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_NOPE_HEAD_DIM = 128
V_HEAD_DIM = 128
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM  # 192
HEAD_SIZE = KV_LORA_RANK + QK_ROPE_HEAD_DIM          # 576
BLOCK_SIZE = 16
HIDDEN_SIZE = NUM_HEADS * V_HEAD_DIM                  # 1024（测试用）

# 余弦相似度阈值
COSINE_SIM_THRESHOLD = {
    torch.float32: 0.9999,
    torch.bfloat16: 0.999,
}

# 测试场景配置
TEST_CASES = {
    "prefill_only_single": BatchSpec(seq_lens=[16], query_lens=[16]),
    "prefill_only_batch": BatchSpec(seq_lens=[32, 48], query_lens=[16, 16]),
    "prefill_with_context": BatchSpec(seq_lens=[32, 48], query_lens=[8, 8]),
    "decode_only_single": BatchSpec(seq_lens=[32], query_lens=[1]),
    "decode_only_batch": BatchSpec(seq_lens=[32, 48, 64], query_lens=[1, 1, 1]),
    "mixed_batch": BatchSpec(
        seq_lens=[32, 48, 64, 80],
        query_lens=[1, 1, 8, 8],
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# 构造 DeepseekV3Config（最小化配置，仅包含 MLA 所需字段）
# ──────────────────────────────────────────────────────────────────────────────

def _make_deepseek_v3_config(
    num_heads: int,
    hidden_size: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    qk_nope_head_dim: int,
    v_head_dim: int,
):
    """构造一个最小化的 DeepseekV3Config，用于实例化 DeepseekV3Attention。

    :param num_heads: 注意力头数
    :param hidden_size: 隐藏层维度
    :param kv_lora_rank: KV LoRA 秩
    :param qk_rope_head_dim: RoPE 部分的 head dim
    :param qk_nope_head_dim: NoPE 部分的 head dim
    :param v_head_dim: Value head dim
    :return: DeepseekV3Config 实例
    """
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        pytest.skip(f"transformers 库不可用：{exc}")

    # 使用 AutoConfig 从预置配置构造，再覆盖关键字段
    # 这里直接用 dict 构造，避免依赖网络下载
    from transformers.models.deepseek_v3.configuration_deepseek_v3 import (
        DeepseekV3Config,
    )

    cfg = DeepseekV3Config(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,  # MLA 中 num_kv_heads == num_heads
        q_lora_rank=None,               # 不使用 Q LoRA
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        qk_nope_head_dim=qk_nope_head_dim,
        v_head_dim=v_head_dim,
        attention_bias=False,
        attention_dropout=0.0,
        rope_interleave=False,
        # RoPE 参数：使用 default 类型，rope_theta 任意（测试中不做 RoPE）
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
        },
        # 其余字段填充默认值，不影响 MLA 计算
        intermediate_size=hidden_size * 4,
        num_hidden_layers=1,
        vocab_size=1000,
        max_position_embeddings=4096,
        rms_norm_eps=1e-6,
        # MoE 相关（不使用，填充最小值）
        n_routed_experts=1,
        n_shared_experts=1,
        num_experts_per_tok=1,
        n_group=1,
        topk_group=1,
        moe_intermediate_size=hidden_size,
        first_k_dense_replace=1,
        norm_topk_prob=False,
        routed_scaling_factor=1.0,
    )
    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# 权重容器
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MLAWeights:
    """MLA 所有权重的容器，供两个实现共享。"""

    # Q 投影（无 LoRA，直接投影）
    # shape: [num_heads * qk_head_dim, hidden_size]
    q_proj_weight: torch.Tensor
    # KV 压缩投影
    # shape: [kv_lora_rank + qk_rope_head_dim, hidden_size]
    kv_a_proj_weight: torch.Tensor
    # kv_a_layernorm 权重
    # shape: [kv_lora_rank]
    kv_a_layernorm_weight: torch.Tensor
    # KV 上投影
    # shape: [num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]
    kv_b_proj_weight: torch.Tensor
    # 输出投影
    # shape: [hidden_size, num_heads * v_head_dim]
    o_proj_weight: torch.Tensor


def _make_mla_weights(
    num_heads: int,
    hidden_size: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    qk_nope_head_dim: int,
    v_head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int = 42,
) -> MLAWeights:
    """随机初始化 MLA 权重（两个实现共享同一套权重）。

    :param seed: 随机种子，保证可复现性
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    def _randn(*shape):
        return torch.randn(*shape, dtype=dtype, device=device, generator=gen)

    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    scale = kv_lora_rank ** -0.5

    return MLAWeights(
        q_proj_weight=_randn(num_heads * qk_head_dim, hidden_size) * (hidden_size ** -0.5),
        kv_a_proj_weight=_randn(kv_lora_rank + qk_rope_head_dim, hidden_size) * (hidden_size ** -0.5),
        kv_a_layernorm_weight=torch.ones(kv_lora_rank, dtype=dtype, device=device),
        kv_b_proj_weight=_randn(
            num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank
        ) * scale,
        o_proj_weight=_randn(hidden_size, num_heads * v_head_dim) * ((num_heads * v_head_dim) ** -0.5),
    )


# ──────────────────────────────────────────────────────────────────────────────
# 构造并运行 transformers.DeepseekV3Attention（eager 模式）
# ──────────────────────────────────────────────────────────────────────────────

def _build_hf_attention(
    weights: MLAWeights,
    num_heads: int,
    hidden_size: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    qk_nope_head_dim: int,
    v_head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
):
    """实例化 transformers.DeepseekV3Attention 并加载共享权重。

    :return: 已加载权重的 DeepseekV3Attention 实例（eval 模式）
    """
    try:
        from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
            DeepseekV3Attention,
        )
    except ImportError as exc:
        pytest.skip(f"transformers DeepseekV3Attention 不可用：{exc}")

    cfg = _make_deepseek_v3_config(
        num_heads=num_heads,
        hidden_size=hidden_size,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        qk_nope_head_dim=qk_nope_head_dim,
        v_head_dim=v_head_dim,
    )

    # 强制使用 eager 实现
    cfg._attn_implementation = "eager"

    attn = DeepseekV3Attention(cfg, layer_idx=0).to(device=device, dtype=dtype)
    attn.eval()

    # 将共享权重加载到 HF 模型的各个 Linear 层
    with torch.no_grad():
        # q_proj（cfg.q_lora_rank is None，直接使用 q_proj）
        attn.q_proj.weight.copy_(weights.q_proj_weight)
        # kv_a_proj_with_mqa
        attn.kv_a_proj_with_mqa.weight.copy_(weights.kv_a_proj_weight)
        # kv_a_layernorm
        attn.kv_a_layernorm.weight.copy_(weights.kv_a_layernorm_weight)
        # kv_b_proj
        attn.kv_b_proj.weight.copy_(weights.kv_b_proj_weight)
        # o_proj
        attn.o_proj.weight.copy_(weights.o_proj_weight)

    return attn


def _build_identity_rope(
    seq_len: int,
    qk_rope_head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """构造 identity RoPE（cos=1, sin=0），使 RoPE 不改变向量方向。

    CPUMLAImpl 在测试中不施加 RoPE，因此 HF 侧也需要使用 identity RoPE
    以保证两者计算等价。

    :return: (cos, sin)，shape 均为 [1, seq_len, qk_rope_head_dim]
    """
    cos = torch.ones(1, seq_len, qk_rope_head_dim, dtype=dtype, device=device)
    sin = torch.zeros(1, seq_len, qk_rope_head_dim, dtype=dtype, device=device)
    return cos, sin


def compute_hf_attn_output(
    batch_spec: BatchSpec,
    hidden_states_list: list[torch.Tensor],
    hf_attn: nn.Module,
    qk_rope_head_dim: int,
    v_head_dim: int,
    num_heads: int,
    device: torch.device,
) -> torch.Tensor:
    """逐序列调用 DeepseekV3Attention.forward，只取 query token 的输出。

    HF 实现以 [batch=1, seq_len, hidden_size] 为输入，对每条序列单独计算，
    然后只取最后 query_len 个 token 的输出（注意力部分，不含 o_proj 之后的残差）。

    注意：这里取的是 o_proj 之后的输出（即 attn_output），与 CPUMLAImpl
    的输出（o_proj 之前）不同。为了公平对比，我们在 CPUMLAImpl 侧也补上 o_proj。

    :return: output，shape = [total_query_tokens, hidden_size]
    """
    outputs = []
    for i in range(batch_spec.batch_size):
        s_len = batch_spec.seq_lens[i]
        q_len = batch_spec.query_lens[i]

        hs = hidden_states_list[i].unsqueeze(0)  # [1, s_len, hidden_size]

        # identity RoPE：不改变向量方向，与 CPUMLAImpl 测试中不做 RoPE 等价
        cos, sin = _build_identity_rope(s_len, qk_rope_head_dim, hs.dtype, device)
        position_embeddings = (cos, sin)

        # 构造因果 attention_mask（HF 使用加法掩码，-inf 填充上三角）
        causal_mask = _build_causal_mask(s_len, device, hs.dtype)

        with torch.no_grad():
            attn_out, __ = hf_attn(
                hidden_states=hs,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                past_key_values=None,
                cache_position=None,
            )
        # attn_out: [1, s_len, hidden_size]，只取最后 q_len 个 token
        outputs.append(attn_out[0, s_len - q_len:, :])

    return torch.cat(outputs, dim=0)  # [total_query_tokens, hidden_size]


def _build_causal_mask(
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """构造 HF 风格的加法因果掩码（-inf 填充上三角）。

    :return: mask，shape = [1, 1, seq_len, seq_len]
    """
    mask = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)
    mask = mask.masked_fill(
        torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        ),
        float("-inf"),
    )
    return mask.unsqueeze(0).unsqueeze(0)


# ──────────────────────────────────────────────────────────────────────────────
# CPUMLAImpl 运行辅助
# ──────────────────────────────────────────────────────────────────────────────

class _MockKvBProj(nn.Module):
    """轻量级 kv_b_proj mock，接口与 ColumnParallelLinear 一致。"""

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return (F.linear(x, self.weight), None)


class _MockMLALayer(AttentionLayerBase):
    """用于测试的 Mock MLA 注意力层（复用 test_cpu_mla.py 中的逻辑）。"""

    def __init__(
        self,
        impl,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        kv_lora_rank: int,
        device: torch.device,
        kv_b_proj: _MockKvBProj,
        o_proj_weight: torch.Tensor,
    ):
        self.impl = impl
        self.num_heads = num_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.kv_lora_rank = kv_lora_rank

        # 从 kv_b_proj 中提取 W_UK_T 和 W_UV
        w = kv_b_proj.weight.T  # [kv_lora_rank, num_heads * (P + V)]
        w = w.view(kv_lora_rank, num_heads, qk_nope_head_dim + v_head_dim)
        w_uk, w_uv = w.split([qk_nope_head_dim, v_head_dim], dim=-1)
        self.W_UK_T = w_uk.permute(1, 2, 0).contiguous()   # [N, P, L]
        self.W_UV = w_uv.transpose(0, 1).contiguous()       # [N, L, V]

        # o_proj 权重，用于将注意力输出投影回 hidden_size
        self.o_proj_weight = o_proj_weight  # [hidden_size, num_heads * v_head_dim]

        self._q_scale = torch.tensor(1.0, device=device)
        self._k_scale = torch.tensor(1.0, device=device)
        self._v_scale = torch.tensor(1.0, device=device)
        self._prob_scale = torch.tensor(1.0, device=device)
        self._q_scale_float = 1.0
        self._k_scale_float = 1.0
        self._v_scale_float = 1.0
        self._decode_concat_quant_fp8_op = _DecodeConcatQuantFP8(
            static=True,
            group_shape=GroupShape.PER_TENSOR,
            compile_native=True,
        )

    def get_attn_backend(self):
        raise NotImplementedError

    def get_kv_cache_spec(self, vllm_config):
        raise NotImplementedError

    @staticmethod
    def _concat_and_cache_mla_cpu(
        kv_c: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """将 kv_c 和 k_pe 拼接后写入 kv_cache。"""
        kv_combined = torch.cat([kv_c, k_pe], dim=-1)
        block_size = kv_cache.shape[1]
        for i in range(slot_mapping.shape[0]):
            slot = slot_mapping[i].item()
            if slot < 0:
                continue
            kv_cache[slot // block_size, slot % block_size] = kv_combined[i]

    def forward_impl(
        self,
        q: torch.Tensor,
        kv_c: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """复现 MLAAttention.forward_impl 的核心逻辑，并在最后补上 o_proj。"""
        if kv_cache.numel() > 0:
            self._concat_and_cache_mla_cpu(
                kv_c,
                k_pe.squeeze(1),
                kv_cache,
                attn_metadata.slot_mapping.flatten(),
            )

        num_decode_tokens = attn_metadata.num_decode_tokens or 0
        has_decode = (attn_metadata.num_decodes or 0) > 0
        has_prefill = (attn_metadata.num_prefills or 0) > 0

        if has_prefill:
            self.impl.forward_mha(
                q[num_decode_tokens:],
                kv_c[num_decode_tokens:],
                k_pe[num_decode_tokens:],
                kv_cache,
                attn_metadata,
                self._k_scale,
                output=output[num_decode_tokens:],
            )

        if has_decode:
            decode_q = q[:num_decode_tokens]
            mqa_q_nope, mqa_q_pe = decode_q.split(
                [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
            )
            mqa_ql_nope = torch.bmm(
                mqa_q_nope.transpose(0, 1), self.W_UK_T
            ).transpose(0, 1)

            attn_out, _ = self.impl.forward_mqa(
                (mqa_ql_nope, mqa_q_pe), kv_cache, attn_metadata, self
            )
            decode_output = torch.bmm(
                attn_out.transpose(0, 1), self.W_UV
            ).transpose(0, 1)
            output[:num_decode_tokens] = decode_output.reshape(
                num_decode_tokens, self.num_heads * self.v_head_dim
            )

        # 补上 o_proj，与 HF 实现对齐
        output = F.linear(output, self.o_proj_weight)
        return output


def _create_kv_cache(
    kv_c_contexts: list[torch.Tensor],
    k_pe_contexts: list[torch.Tensor],
    block_size: int,
    head_size: int,
    dtype: torch.dtype,
    device: torch.device,
    common_attn_metadata: CommonAttentionMetadata,
    num_extra_blocks: int = 100,
) -> torch.Tensor:
    """创建并填充分页 KV Cache。"""
    batch_size = len(kv_c_contexts)
    seq_lens = common_attn_metadata.seq_lens.cpu()
    query_lens = (
        common_attn_metadata.query_start_loc_cpu[1:]
        - common_attn_metadata.query_start_loc_cpu[:-1]
    )
    context_lens = seq_lens - query_lens

    total_blocks = sum(cdiv(int(seq_lens[i]), block_size) for i in range(batch_size))
    num_blocks = total_blocks + 1 + num_extra_blocks

    kv_cache = torch.zeros(num_blocks, block_size, head_size, dtype=dtype, device=device)
    kv_cache_flat = kv_cache.view(-1, head_size)
    block_table = common_attn_metadata.block_table_tensor
    slot_mapping = common_attn_metadata.slot_mapping

    start_block_idx = 1
    for i in range(batch_size):
        kv_c_ctx = kv_c_contexts[i]
        k_pe_ctx = k_pe_contexts[i]
        ctx_len = kv_c_ctx.shape[0]
        num_blocks_for_seq = cdiv(int(seq_lens[i]), block_size)

        if ctx_len > 0:
            kv_ctx = torch.cat([kv_c_ctx, k_pe_ctx.squeeze(1)], dim=-1)
            start_flat = start_block_idx * block_size
            kv_cache_flat[start_flat:start_flat + ctx_len] = kv_ctx

        for b in range(num_blocks_for_seq):
            block_table[i, b] = start_block_idx + b
        block_table[i, num_blocks_for_seq:] = 0

        q_start = int(common_attn_metadata.query_start_loc_cpu[i])
        q_end = int(common_attn_metadata.query_start_loc_cpu[i + 1])
        for t_idx, _ in enumerate(range(q_start, q_end)):
            token_pos = int(context_lens[i]) + t_idx
            block_idx = token_pos // block_size
            block_offset = token_pos % block_size
            slot_mapping[q_start + t_idx] = (
                (start_block_idx + block_idx) * block_size + block_offset
            )

        start_block_idx += num_blocks_for_seq

    return kv_cache


def compute_cpu_mla_output(
    batch_spec: BatchSpec,
    q_list: list[torch.Tensor],
    kv_c_new_list: list[torch.Tensor],
    k_pe_new_list: list[torch.Tensor],
    kv_c_ctx_list: list[torch.Tensor],
    k_pe_ctx_list: list[torch.Tensor],
    weights: MLAWeights,
    num_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    qk_nope_head_dim: int,
    v_head_dim: int,
    scaling: float,
    vllm_config,
    device: torch.device,
) -> torch.Tensor:
    """运行 CPUMLAImpl，返回注意力输出（含 o_proj，与 HF 实现对齐）。

    :return: output，shape = [total_query_tokens, hidden_size]
    """
    from vllm.v1.attention.backends.mla.cpu_mla import (
        CPUMLABackend,
        CPUMLAMetadataBuilder,
    )

    kv_b_proj = _MockKvBProj(weights.kv_b_proj_weight)

    common_attn_metadata = create_common_attn_metadata(batch_spec, BLOCK_SIZE, device)
    kv_cache = _create_kv_cache(
        kv_c_ctx_list,
        k_pe_ctx_list,
        BLOCK_SIZE,
        HEAD_SIZE,
        q_list[0].dtype,
        device,
        common_attn_metadata,
    )

    kv_cache_spec = MLAAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=vllm_config.model_config.get_num_kv_heads(
            vllm_config.parallel_config
        ),
        head_size=HEAD_SIZE,
        dtype=vllm_config.model_config.dtype,
        sliding_window=None,
        cache_dtype_str="auto",
    )

    with set_current_vllm_config(vllm_config):
        impl = CPUMLABackend.get_impl_cls()(
            num_heads=num_heads,
            head_size=HEAD_SIZE,
            scale=scaling,
            num_kv_heads=num_heads,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            logits_soft_cap=None,
            attn_type="decoder",
            kv_sharing_target_layer_name=None,
            q_lora_rank=None,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            qk_head_dim=qk_nope_head_dim + qk_rope_head_dim,
            v_head_dim=v_head_dim,
            kv_b_proj=kv_b_proj,
        )
        impl.process_weights_after_loading(q_list[0].dtype)
        if impl.dcp_world_size == -1:
            impl.dcp_world_size = 1

        mock_layer = _MockMLALayer(
            impl=impl,
            num_heads=num_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            kv_lora_rank=kv_lora_rank,
            device=device,
            kv_b_proj=kv_b_proj,
            o_proj_weight=weights.o_proj_weight,
        )

        layer_name = "test_layer_vs_hf"
        vllm_config.compilation_config.static_forward_context[layer_name] = mock_layer

        builder = CPUMLAMetadataBuilder(
            kv_cache_spec, [layer_name], vllm_config, device
        )
        attn_metadata = builder.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
        )

        query = torch.cat(q_list, dim=0)
        kv_c_new = torch.cat(kv_c_new_list, dim=0)
        k_pe_new = torch.cat(k_pe_new_list, dim=0)
        num_tokens = query.shape[0]
        output = torch.zeros(
            num_tokens,
            num_heads * v_head_dim,
            dtype=query.dtype,
            device=device,
        )
        output = mock_layer.forward_impl(
            query, kv_c_new, k_pe_new, kv_cache, attn_metadata, output
        )

    return output


# ──────────────────────────────────────────────────────────────────────────────
# 数据准备
# ──────────────────────────────────────────────────────────────────────────────

def _apply_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """RMSNorm 前向计算。"""
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (weight * x).to(input_dtype)


def _prepare_inputs(
    batch_spec: BatchSpec,
    weights: MLAWeights,
    num_heads: int,
    hidden_size: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    qk_nope_head_dim: int,
    v_head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int = 0,
):
    """准备两个实现共享的输入数据。

    从同一套 hidden_states 出发，用相同权重推导出：
    - HF 实现所需的完整 hidden_states（含 context + query）
    - CPUMLAImpl 所需的 q / kv_c_normed / k_pe

    :return: (hidden_states_list, q_list, kv_c_new_list, k_pe_new_list,
              kv_c_ctx_list, k_pe_ctx_list)
    """
    torch.manual_seed(seed)
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

    hidden_states_list = []
    q_list = []
    kv_c_new_list = []
    k_pe_new_list = []
    kv_c_ctx_list = []
    k_pe_ctx_list = []

    for i in range(batch_spec.batch_size):
        s_len = batch_spec.seq_lens[i]
        q_len = batch_spec.query_lens[i]
        context_len = s_len - q_len

        # 完整序列的 hidden_states（HF 实现输入）
        hs = torch.randn(s_len, hidden_size, dtype=dtype, device=device)
        hidden_states_list.append(hs)

        with torch.no_grad():
            # Q 投影（只取 query token 部分）
            q_full = F.linear(hs[context_len:], weights.q_proj_weight)
            q_full = q_full.view(q_len, num_heads, qk_head_dim)
            q_list.append(q_full)

            # KV 压缩投影（完整序列）
            compressed_kv = F.linear(hs, weights.kv_a_proj_weight)
            kv_c_full, k_pe_full = torch.split(
                compressed_kv, [kv_lora_rank, qk_rope_head_dim], dim=-1
            )
            # kv_a_layernorm（与 HF 实现中的 kv_a_layernorm 等价）
            kv_c_normed_full = _apply_rms_norm(kv_c_full, weights.kv_a_layernorm_weight)

            kv_c_ctx_list.append(kv_c_normed_full[:context_len])
            k_pe_ctx_list.append(k_pe_full[:context_len].unsqueeze(1))
            kv_c_new_list.append(kv_c_normed_full[context_len:])
            k_pe_new_list.append(k_pe_full[context_len:].unsqueeze(1))

    return (
        hidden_states_list,
        q_list,
        kv_c_new_list,
        k_pe_new_list,
        kv_c_ctx_list,
        k_pe_ctx_list,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 余弦相似度计算
# ──────────────────────────────────────────────────────────────────────────────

def cosine_similarity_stats(
    a: torch.Tensor,
    b: torch.Tensor,
) -> dict:
    """计算两个输出张量逐 token 的余弦相似度统计信息。

    :param a: shape = [num_tokens, hidden_size]
    :param b: shape = [num_tokens, hidden_size]
    :return: 包含 min / mean / max 余弦相似度的字典
    """
    a_f = a.float()
    b_f = b.float()
    cos_sim = F.cosine_similarity(a_f, b_f, dim=-1)  # [num_tokens]
    return {
        "min": cos_sim.min().item(),
        "mean": cos_sim.mean().item(),
        "max": cos_sim.max().item(),
        "all_values": cos_sim,
    }


# ──────────────────────────────────────────────────────────────────────────────
# pytest 测试用例
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def cpu_device():
    return torch.device("cpu")


@pytest.fixture(scope="module")
def vllm_config_for_hf_test():
    """创建用于对比测试的 VllmConfig。"""
    cfg = create_vllm_config(
        model_name="deepseek-ai/DeepSeek-R1",
        tensor_parallel_size=1,
        max_model_len=4096,
        num_gpu_blocks=1000,
        block_size=BLOCK_SIZE,
    )
    cfg.cache_config.num_gpu_blocks = 1000
    cfg.cache_config.num_cpu_blocks = 0
    return cfg


@pytest.mark.parametrize("case_name", list(TEST_CASES.keys()))
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cosine_similarity_vs_hf(
    case_name: str,
    dtype: torch.dtype,
    vllm_config_for_hf_test,
    cpu_device,
):
    """测试 CPUMLAImpl 与 transformers.DeepseekV3Attention（eager）的余弦相似度。

    两侧均包含 o_proj，对比的是完整注意力层的输出（hidden_size 维度）。
    identity RoPE（cos=1, sin=0）确保两侧 RoPE 行为一致。
    """
    device = cpu_device
    batch_spec = TEST_CASES[case_name]
    num_heads = NUM_HEADS
    scaling = QK_HEAD_DIM ** -0.5
    threshold = COSINE_SIM_THRESHOLD[dtype]

    weights = _make_mla_weights(
        num_heads=num_heads,
        hidden_size=HIDDEN_SIZE,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        v_head_dim=V_HEAD_DIM,
        dtype=dtype,
        device=device,
        seed=42,
    )

    (
        hidden_states_list,
        q_list,
        kv_c_new_list,
        k_pe_new_list,
        kv_c_ctx_list,
        k_pe_ctx_list,
    ) = _prepare_inputs(
        batch_spec=batch_spec,
        weights=weights,
        num_heads=num_heads,
        hidden_size=HIDDEN_SIZE,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        v_head_dim=V_HEAD_DIM,
        dtype=dtype,
        device=device,
        seed=0,
    )

    # HF 参考输出（使用 transformers.DeepseekV3Attention，含 o_proj）
    hf_attn = _build_hf_attention(
        weights=weights,
        num_heads=num_heads,
        hidden_size=HIDDEN_SIZE,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        v_head_dim=V_HEAD_DIM,
        dtype=dtype,
        device=device,
    )
    hf_output = compute_hf_attn_output(
        batch_spec=batch_spec,
        hidden_states_list=hidden_states_list,
        hf_attn=hf_attn,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        v_head_dim=V_HEAD_DIM,
        num_heads=num_heads,
        device=device,
    )

    # CPUMLAImpl 输出（含 o_proj）
    cpu_mla_output = compute_cpu_mla_output(
        batch_spec=batch_spec,
        q_list=q_list,
        kv_c_new_list=kv_c_new_list,
        k_pe_new_list=k_pe_new_list,
        kv_c_ctx_list=kv_c_ctx_list,
        k_pe_ctx_list=k_pe_ctx_list,
        weights=weights,
        num_heads=num_heads,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        v_head_dim=V_HEAD_DIM,
        scaling=scaling,
        vllm_config=vllm_config_for_hf_test,
        device=device,
    )

    assert hf_output.shape == cpu_mla_output.shape, (
        f"shape mismatch: hf={hf_output.shape}, cpu_mla={cpu_mla_output.shape}"
    )

    stats = cosine_similarity_stats(hf_output, cpu_mla_output)
    min_cos = stats["min"]

    assert min_cos >= threshold, (
        f"[{case_name}][{dtype}] 余弦相似度过低：min={min_cos:.6f}，"
        f"mean={stats['mean']:.6f}，阈值={threshold}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 独立运行：输出详细报告
# ──────────────────────────────────────────────────────────────────────────────

def _run_report(args: argparse.Namespace) -> int:
    """独立运行时输出详细的余弦相似度报告。"""
    device = torch.device("cpu")
    dtypes = [torch.float32, torch.bfloat16] if not args.fp32_only else [torch.float32]
    cases = list(TEST_CASES.keys()) if not args.case else [args.case]

    vllm_config = create_vllm_config(
        model_name="deepseek-ai/DeepSeek-R1",
        tensor_parallel_size=1,
        max_model_len=4096,
        num_gpu_blocks=1000,
        block_size=BLOCK_SIZE,
    )
    vllm_config.cache_config.num_gpu_blocks = 1000
    vllm_config.cache_config.num_cpu_blocks = 0

    print("\n" + "=" * 88)
    print("CPUMLAImpl vs transformers.DeepseekV3Attention（eager 模式）余弦相似度报告")
    print("=" * 88)
    print(
        f"{'场景':<30} {'dtype':<12} {'min cos':<12} {'mean cos':<12} {'max cos':<12} {'状态'}"
    )
    print("-" * 88)

    all_pass = True
    for case_name in cases:
        batch_spec = TEST_CASES[case_name]
        for dtype in dtypes:
            threshold = COSINE_SIM_THRESHOLD[dtype]
            scaling = QK_HEAD_DIM ** -0.5

            weights = _make_mla_weights(
                num_heads=NUM_HEADS,
                hidden_size=HIDDEN_SIZE,
                kv_lora_rank=KV_LORA_RANK,
                qk_rope_head_dim=QK_ROPE_HEAD_DIM,
                qk_nope_head_dim=QK_NOPE_HEAD_DIM,
                v_head_dim=V_HEAD_DIM,
                dtype=dtype,
                device=device,
                seed=42,
            )

            (
                hidden_states_list,
                q_list,
                kv_c_new_list,
                k_pe_new_list,
                kv_c_ctx_list,
                k_pe_ctx_list,
            ) = _prepare_inputs(
                batch_spec=batch_spec,
                weights=weights,
                num_heads=NUM_HEADS,
                hidden_size=HIDDEN_SIZE,
                kv_lora_rank=KV_LORA_RANK,
                qk_rope_head_dim=QK_ROPE_HEAD_DIM,
                qk_nope_head_dim=QK_NOPE_HEAD_DIM,
                v_head_dim=V_HEAD_DIM,
                dtype=dtype,
                device=device,
                seed=0,
            )

            hf_attn = _build_hf_attention(
                weights=weights,
                num_heads=NUM_HEADS,
                hidden_size=HIDDEN_SIZE,
                kv_lora_rank=KV_LORA_RANK,
                qk_rope_head_dim=QK_ROPE_HEAD_DIM,
                qk_nope_head_dim=QK_NOPE_HEAD_DIM,
                v_head_dim=V_HEAD_DIM,
                dtype=dtype,
                device=device,
            )
            hf_output = compute_hf_attn_output(
                batch_spec=batch_spec,
                hidden_states_list=hidden_states_list,
                hf_attn=hf_attn,
                qk_rope_head_dim=QK_ROPE_HEAD_DIM,
                v_head_dim=V_HEAD_DIM,
                num_heads=NUM_HEADS,
                device=device,
            )

            cpu_mla_output = compute_cpu_mla_output(
                batch_spec=batch_spec,
                q_list=q_list,
                kv_c_new_list=kv_c_new_list,
                k_pe_new_list=k_pe_new_list,
                kv_c_ctx_list=kv_c_ctx_list,
                k_pe_ctx_list=k_pe_ctx_list,
                weights=weights,
                num_heads=NUM_HEADS,
                kv_lora_rank=KV_LORA_RANK,
                qk_rope_head_dim=QK_ROPE_HEAD_DIM,
                qk_nope_head_dim=QK_NOPE_HEAD_DIM,
                v_head_dim=V_HEAD_DIM,
                scaling=scaling,
                vllm_config=vllm_config,
                device=device,
            )

            stats = cosine_similarity_stats(hf_output, cpu_mla_output)
            passed = stats["min"] >= threshold
            status = "✅ PASS" if passed else f"❌ FAIL (threshold={threshold})"
            if not passed:
                all_pass = False

            dtype_str = str(dtype).replace("torch.", "")
            print(
                f"{case_name:<30} {dtype_str:<12} "
                f"{stats['min']:<12.6f} {stats['mean']:<12.6f} "
                f"{stats['max']:<12.6f} {status}"
            )

    print("=" * 88)
    print("总体结果：" + ("✅ 全部通过" if all_pass else "❌ 存在失败项"))
    print("=" * 88 + "\n")
    return 0 if all_pass else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="对比 CPUMLAImpl 与 transformers.DeepseekV3Attention 的余弦相似度"
    )
    parser.add_argument(
        "--case",
        type=str,
        default=None,
        choices=list(TEST_CASES.keys()),
        help="只运行指定场景（默认运行全部）",
    )
    parser.add_argument(
        "--fp32-only",
        action="store_true",
        help="只测试 float32（跳过 bfloat16）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="权重随机种子（默认 42）",
    )
    args = parser.parse_args()
    sys.exit(_run_report(args))
