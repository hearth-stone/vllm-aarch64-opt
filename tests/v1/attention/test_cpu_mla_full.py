"""完整 MLA 端到端对比测试：CPUMLAImpl vs transformers.DeepseekV3Attention。

测试目标：
    从相同的 hidden_states 出发，走完整的 MLA 计算流程（含 q_proj、kv_a_proj、
    kv_a_layernorm、kv_b_proj、attention、RoPE、o_proj），验证 CPUMLAImpl 的输出
    与 transformers.DeepseekV3Attention（eager 模式）的余弦相似度。

与 test_cpu_mla_vs_hf.py 的区别：
    - 本测试使用真实 RoPE（而非 identity RoPE），验证完整的端到端等价性。
    - 两侧均从相同的 hidden_states 出发，不手动拆分中间变量。

测试场景：
    - prefill_only：纯 prefill，无历史 context
    - prefill_with_context：prefill + 历史 context
    - decode_only：纯 decode（query_len == 1）
    - mixed_batch：prefill + decode 混合批次

运行方式：
    pytest tests/v1/attention/test_cpu_mla_full.py -v
    # 或直接运行（输出详细的余弦相似度报告）：
    python tests/v1/attention/test_cpu_mla_full.py
"""

import argparse
import importlib.util
import sys
from dataclasses import dataclass

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
from vllm.v1.attention.backends.mla.cpu_mla import CPUMLABackend
from vllm.v1.kv_cache_interface import MLAAttentionSpec

# ──────────────────────────────────────────────────────────────────────────────
# 超参数配置（与 DeepSeek-V3/R1 实际配置结构一致，数值缩小以加速测试）
# ──────────────────────────────────────────────────────────────────────────────

# 可在文件顶部集中修改所有测试参数
CONFIG = {
    "num_heads": 8,
    "kv_lora_rank": 512,
    "qk_rope_head_dim": 64,
    "qk_nope_head_dim": 128,
    "v_head_dim": 128,
    "block_size": 16,
    # hidden_size = num_heads * v_head_dim
    "hidden_size": 8 * 128,
    # 余弦相似度阈值
    "cosine_threshold": {
        torch.float32: 0.9999,
        torch.bfloat16: 0.999,
    },
    # 随机种子
    "weight_seed": 42,
    "input_seed": 0,
}

# 派生常量
NUM_HEADS: int = CONFIG["num_heads"]
KV_LORA_RANK: int = CONFIG["kv_lora_rank"]
QK_ROPE_HEAD_DIM: int = CONFIG["qk_rope_head_dim"]
QK_NOPE_HEAD_DIM: int = CONFIG["qk_nope_head_dim"]
V_HEAD_DIM: int = CONFIG["v_head_dim"]
BLOCK_SIZE: int = CONFIG["block_size"]
HIDDEN_SIZE: int = CONFIG["hidden_size"]
QK_HEAD_DIM: int = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM
HEAD_SIZE: int = KV_LORA_RANK + QK_ROPE_HEAD_DIM
COSINE_THRESHOLD: dict = CONFIG["cosine_threshold"]

# 测试场景
TEST_CASES = {
    "prefill_only_single": BatchSpec(seq_lens=[16], query_lens=[16]),
    "prefill_only_batch": BatchSpec(seq_lens=[32, 48], query_lens=[32, 48]),
    "prefill_with_context": BatchSpec(seq_lens=[32, 48], query_lens=[8, 8]),
    "decode_only_single": BatchSpec(seq_lens=[32], query_lens=[1]),
    "decode_only_batch": BatchSpec(seq_lens=[32, 48, 64], query_lens=[1, 1, 1]),
    "mixed_batch": BatchSpec(
        seq_lens=[32, 48, 64, 80],
        query_lens=[1, 1, 8, 8],
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# 权重容器
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class MLAWeights:
    """完整 MLA 层的所有权重，供两个实现共享。"""

    # Q 投影（无 LoRA）：[num_heads * qk_head_dim, hidden_size]
    q_proj_weight: torch.Tensor
    # KV 压缩投影：[kv_lora_rank + qk_rope_head_dim, hidden_size]
    kv_a_proj_weight: torch.Tensor
    # kv_a_layernorm 权重：[kv_lora_rank]
    kv_a_layernorm_weight: torch.Tensor
    # KV 上投影：[num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]
    kv_b_proj_weight: torch.Tensor
    # 输出投影：[hidden_size, num_heads * v_head_dim]
    o_proj_weight: torch.Tensor


def make_mla_weights(
    dtype: torch.dtype,
    device: torch.device,
    seed: int = 42,
) -> MLAWeights:
    """随机初始化 MLA 权重，两个实现共享同一套权重。

    :param seed: 随机种子，保证可复现性
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    def _randn(*shape):
        return torch.randn(*shape, dtype=dtype, device=device, generator=gen)

    scale_kv = KV_LORA_RANK**-0.5
    scale_h = HIDDEN_SIZE**-0.5
    scale_v = (NUM_HEADS * V_HEAD_DIM) ** -0.5

    return MLAWeights(
        q_proj_weight=_randn(NUM_HEADS * QK_HEAD_DIM, HIDDEN_SIZE) * scale_h,
        kv_a_proj_weight=_randn(KV_LORA_RANK + QK_ROPE_HEAD_DIM, HIDDEN_SIZE) * scale_h,
        kv_a_layernorm_weight=torch.ones(KV_LORA_RANK, dtype=dtype, device=device),
        kv_b_proj_weight=_randn(
            NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM), KV_LORA_RANK
        )
        * scale_kv,
        o_proj_weight=_randn(HIDDEN_SIZE, NUM_HEADS * V_HEAD_DIM) * scale_v,
    )


# ──────────────────────────────────────────────────────────────────────────────
# RoPE 工具函数
# ──────────────────────────────────────────────────────────────────────────────


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """将向量后半部分取反后与前半部分拼接（RoPE 标准操作）。"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q_rot: torch.Tensor,
    k_rot: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """对 q_rot 和 k_rot 施加 RoPE。

    :param q_rot: [batch, num_heads, seq_len, qk_rope_head_dim]
    :param k_rot: [batch, 1, seq_len, qk_rope_head_dim]
    :param cos: [1, seq_len, qk_rope_head_dim]（由 DeepseekV3RotaryEmbedding 生成）
    :param sin: [1, seq_len, qk_rope_head_dim]
    :return: (q_rot_out, k_rot_out)，shape 与输入相同
    """
    # cos/sin: [1, seq_len, dim] -> unsqueeze(1) -> [1, 1, seq_len, dim]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_out = q_rot * cos + _rotate_half(q_rot) * sin
    k_out = k_rot * cos + _rotate_half(k_rot) * sin
    return q_out, k_out


def build_rope_embeddings(
    seq_len: int,
    rope_theta: float,
    dtype: torch.dtype,
    device: torch.device,
    position_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算标准 RoPE 的 cos/sin 嵌入。

    :param seq_len: 序列长度
    :param rope_theta: RoPE base（默认 10000）
    :param position_offset: 位置偏移（用于 decode 阶段）
    :return: (cos, sin)，shape = [1, seq_len, qk_rope_head_dim]
    """
    dim = QK_ROPE_HEAD_DIM
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
    )
    positions = torch.arange(
        position_offset,
        position_offset + seq_len,
        dtype=torch.float32,
        device=device,
    )
    # freqs: [seq_len, dim // 2]
    freqs = torch.outer(positions, inv_freq)
    # emb: [seq_len, dim]
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos().to(dtype).unsqueeze(0)  # [1, seq_len, dim]
    sin = emb.sin().to(dtype).unsqueeze(0)
    return cos, sin


# ──────────────────────────────────────────────────────────────────────────────
# RMSNorm 工具函数
# ──────────────────────────────────────────────────────────────────────────────


def apply_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """RMSNorm 前向计算。"""
    input_dtype = x.dtype
    x_f32 = x.to(torch.float32)
    variance = x_f32.pow(2).mean(-1, keepdim=True)
    x_f32 = x_f32 * torch.rsqrt(variance + eps)
    return (weight * x_f32).to(input_dtype)


# ──────────────────────────────────────────────────────────────────────────────
# HF 参考实现
# ──────────────────────────────────────────────────────────────────────────────


def build_hf_attention(
    weights: MLAWeights,
    dtype: torch.dtype,
    device: torch.device,
) -> nn.Module:
    """实例化 transformers.DeepseekV3Attention 并加载共享权重。

    :return: 已加载权重的 DeepseekV3Attention 实例（eval 模式，eager 实现）
    """
    try:
        from transformers.models.deepseek_v3.configuration_deepseek_v3 import (
            DeepseekV3Config,
        )
        from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
            DeepseekV3Attention,
        )
    except ImportError as exc:
        pytest.skip(f"transformers DeepseekV3Attention 不可用：{exc}")

    cfg = DeepseekV3Config(
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_HEADS,
        q_lora_rank=None,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        v_head_dim=V_HEAD_DIM,
        attention_bias=False,
        attention_dropout=0.0,
        rope_interleave=False,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
        },
        intermediate_size=HIDDEN_SIZE * 4,
        num_hidden_layers=1,
        vocab_size=1000,
        max_position_embeddings=4096,
        rms_norm_eps=1e-6,
        n_routed_experts=1,
        n_shared_experts=1,
        num_experts_per_tok=1,
        n_group=1,
        topk_group=1,
        moe_intermediate_size=HIDDEN_SIZE,
        first_k_dense_replace=1,
        norm_topk_prob=False,
        routed_scaling_factor=1.0,
    )
    cfg._attn_implementation = "eager"

    attn = DeepseekV3Attention(cfg, layer_idx=0).to(device=device, dtype=dtype)
    attn.eval()

    with torch.no_grad():
        attn.q_proj.weight.copy_(weights.q_proj_weight)
        attn.kv_a_proj_with_mqa.weight.copy_(weights.kv_a_proj_weight)
        attn.kv_a_layernorm.weight.copy_(weights.kv_a_layernorm_weight)
        attn.kv_b_proj.weight.copy_(weights.kv_b_proj_weight)
        attn.o_proj.weight.copy_(weights.o_proj_weight)

    return attn


def compute_hf_output(
    batch_spec: BatchSpec,
    hidden_states_list: list[torch.Tensor],
    hf_attn: nn.Module,
    device: torch.device,
    rope_theta: float = 10000.0,
) -> torch.Tensor:
    """逐序列调用 DeepseekV3Attention.forward，只取 query token 的输出。

    使用真实 RoPE（与 CPUMLAImpl 侧保持一致的 rope_theta）。

    :return: output，shape = [total_query_tokens, hidden_size]
    """
    if (
        importlib.util.find_spec("transformers.models.deepseek_v3.modeling_deepseek_v3")
        is None
    ):
        pytest.skip("transformers DeepseekV3RotaryEmbedding 不可用")

    outputs = []
    for i in range(batch_spec.batch_size):
        s_len = batch_spec.seq_lens[i]
        q_len = batch_spec.query_lens[i]

        hs = hidden_states_list[i].unsqueeze(0)  # [1, s_len, hidden_size]

        # 使用 HF 的 RotaryEmbedding 生成真实 RoPE（与 CPUMLAImpl 侧一致）
        cos, sin = build_rope_embeddings(
            seq_len=s_len,
            rope_theta=rope_theta,
            dtype=hs.dtype,
            device=device,
        )
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
        outputs.append(attn_out[0, s_len - q_len :, :])

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
    """轻量级 kv_b_proj mock，接口与 ColumnParallelLinear 一致。

    除了 forward 外，还提供 `cpu_linear` 属性，与 CPU 路径下
    dispatch_cpu_unquantized_gemm 注入的 lambda 签名保持一致，
    供 CPUMLAImpl._linear 直接调用（prefill / mixed 路径）。
    """

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        # 模拟产品代码里注入的 cpu_linear(x, weight, bias) lambda
        self.cpu_linear = lambda x, weight, bias=None: F.linear(x, weight, bias)
        # 与 LinearBase 对齐，避免 _linear 里 getattr(..., 'skip_bias_add') 误判
        self.skip_bias_add = False

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return (F.linear(x, self.weight), None)


class _MockMLALayer(AttentionLayerBase):
    """用于测试的 Mock MLA 注意力层。"""

    def __init__(
        self,
        impl,
        kv_b_proj: _MockKvBProj,
        o_proj_weight: torch.Tensor,
        device: torch.device,
    ):
        self.impl = impl
        self.num_heads = NUM_HEADS
        self.qk_nope_head_dim = QK_NOPE_HEAD_DIM
        self.qk_rope_head_dim = QK_ROPE_HEAD_DIM
        self.v_head_dim = V_HEAD_DIM
        self.kv_lora_rank = KV_LORA_RANK

        # 从 kv_b_proj 中提取 W_UK_T 和 W_UV（用于 decode absorption）
        w = kv_b_proj.weight.T  # [kv_lora_rank, num_heads * (P + V)]
        w = w.view(KV_LORA_RANK, NUM_HEADS, QK_NOPE_HEAD_DIM + V_HEAD_DIM)
        w_uk, w_uv = w.split([QK_NOPE_HEAD_DIM, V_HEAD_DIM], dim=-1)
        self.W_UK_T = w_uk.permute(1, 2, 0).contiguous()  # [N, P, L]
        self.W_UV = w_uv.transpose(0, 1).contiguous()  # [N, L, V]

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
    def _write_kv_cache(
        kv_c: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """将 kv_c 和 k_pe 拼接后写入分页 KV cache。"""
        kv_combined = torch.cat([kv_c, k_pe], dim=-1)
        block_size = kv_cache.shape[1]
        for i in range(slot_mapping.shape[0]):
            slot = int(slot_mapping[i].item())
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
        """执行完整的 MLA 前向计算（含 o_proj）。"""
        if kv_cache.numel() > 0:
            self._write_kv_cache(
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
                [QK_NOPE_HEAD_DIM, QK_ROPE_HEAD_DIM], dim=-1
            )
            # absorption：q_nope 投影到潜在空间
            mqa_ql_nope = torch.bmm(mqa_q_nope.transpose(0, 1), self.W_UK_T).transpose(
                0, 1
            )

            attn_out, _ = self.impl.forward_mqa(
                (mqa_ql_nope, mqa_q_pe), kv_cache, attn_metadata, self
            )
            # 上投影回 v_head_dim
            decode_output = torch.bmm(attn_out.transpose(0, 1), self.W_UV).transpose(
                0, 1
            )
            output[:num_decode_tokens] = decode_output.reshape(
                num_decode_tokens, NUM_HEADS * V_HEAD_DIM
            )

        # o_proj：与 HF 实现对齐
        output = F.linear(output, self.o_proj_weight)
        return output


def _create_kv_cache(
    kv_c_ctx_list: list[torch.Tensor],
    k_pe_ctx_list: list[torch.Tensor],
    dtype: torch.dtype,
    device: torch.device,
    common_attn_metadata: CommonAttentionMetadata,
    num_extra_blocks: int = 100,
) -> torch.Tensor:
    """创建并填充分页 KV Cache（写入历史 context token）。"""
    batch_size = len(kv_c_ctx_list)
    seq_lens = common_attn_metadata.seq_lens.cpu()
    query_lens = (
        common_attn_metadata.query_start_loc_cpu[1:]
        - common_attn_metadata.query_start_loc_cpu[:-1]
    )
    context_lens = seq_lens - query_lens

    total_blocks = sum(cdiv(int(seq_lens[i]), BLOCK_SIZE) for i in range(batch_size))
    num_blocks = total_blocks + 1 + num_extra_blocks

    kv_cache = torch.zeros(
        num_blocks, BLOCK_SIZE, HEAD_SIZE, dtype=dtype, device=device
    )
    kv_cache_flat = kv_cache.view(-1, HEAD_SIZE)
    block_table = common_attn_metadata.block_table_tensor
    slot_mapping = common_attn_metadata.slot_mapping

    start_block_idx = 1
    for i in range(batch_size):
        kv_c_ctx = kv_c_ctx_list[i]
        k_pe_ctx = k_pe_ctx_list[i]
        ctx_len = kv_c_ctx.shape[0]
        num_blocks_for_seq = cdiv(int(seq_lens[i]), BLOCK_SIZE)

        if ctx_len > 0:
            kv_ctx = torch.cat([kv_c_ctx, k_pe_ctx.squeeze(1)], dim=-1)
            start_flat = start_block_idx * BLOCK_SIZE
            kv_cache_flat[start_flat : start_flat + ctx_len] = kv_ctx

        for b in range(num_blocks_for_seq):
            block_table[i, b] = start_block_idx + b
        block_table[i, num_blocks_for_seq:] = 0

        q_start = int(common_attn_metadata.query_start_loc_cpu[i])
        q_end = int(common_attn_metadata.query_start_loc_cpu[i + 1])
        for t_idx in range(q_end - q_start):
            token_pos = int(context_lens[i]) + t_idx
            block_idx = token_pos // BLOCK_SIZE
            block_offset = token_pos % BLOCK_SIZE
            slot_mapping[q_start + t_idx] = (
                start_block_idx + block_idx
            ) * BLOCK_SIZE + block_offset

        start_block_idx += num_blocks_for_seq

    return kv_cache


def compute_cpu_mla_output(
    batch_spec: BatchSpec,
    hidden_states_list: list[torch.Tensor],
    weights: MLAWeights,
    vllm_config,
    device: torch.device,
    rope_theta: float = 10000.0,
) -> torch.Tensor:
    """从 hidden_states 出发，运行完整的 CPUMLAImpl 前向计算。

    计算流程：
        1. q_proj：hidden_states -> q（含 nope + rope 两部分）
        2. kv_a_proj：hidden_states -> kv_c + k_pe
        3. kv_a_layernorm：对 kv_c 做 RMSNorm
        4. 对 q_rot 和 k_pe 施加真实 RoPE
        5. CPUMLAImpl.forward_mha / forward_mqa
        6. o_proj

    :return: output，shape = [total_query_tokens, hidden_size]
    """
    from vllm.v1.attention.backends.mla.cpu_mla import CPUMLAMetadataBuilder

    kv_b_proj = _MockKvBProj(weights.kv_b_proj_weight)

    # 准备每条序列的 q / kv_c_normed / k_pe（含 RoPE）
    q_list = []
    kv_c_new_list = []
    k_pe_new_list = []
    kv_c_ctx_list = []
    k_pe_ctx_list = []

    for i in range(batch_spec.batch_size):
        s_len = batch_spec.seq_lens[i]
        q_len = batch_spec.query_lens[i]
        context_len = s_len - q_len
        hs = hidden_states_list[i]  # [s_len, hidden_size]

        with torch.no_grad():
            # Q 投影（只取 query token 部分）
            q_full = F.linear(hs[context_len:], weights.q_proj_weight)
            q_full = q_full.view(q_len, NUM_HEADS, QK_HEAD_DIM)
            # 拆分 nope 和 rope 部分
            q_nope = q_full[..., :QK_NOPE_HEAD_DIM]  # [q_len, N, P]
            q_rot = q_full[..., QK_NOPE_HEAD_DIM:]  # [q_len, N, R]

            # KV 压缩投影（完整序列）
            compressed_kv = F.linear(hs, weights.kv_a_proj_weight)
            kv_c_full, k_pe_full = torch.split(
                compressed_kv, [KV_LORA_RANK, QK_ROPE_HEAD_DIM], dim=-1
            )
            # kv_a_layernorm
            kv_c_normed_full = apply_rms_norm(kv_c_full, weights.kv_a_layernorm_weight)

            # 对 query token 的 q_rot 施加 RoPE（位置从 context_len 开始）
            cos_q, sin_q = build_rope_embeddings(
                seq_len=q_len,
                rope_theta=rope_theta,
                dtype=hs.dtype,
                device=device,
                position_offset=context_len,
            )
            # q_rot: [q_len, N, R] -> [1, N, q_len, R]（HF 格式）
            q_rot_4d = q_rot.permute(1, 0, 2).unsqueeze(0)
            # k_pe 只取 query token 部分，shape: [q_len, R] -> [1, 1, q_len, R]
            k_pe_q = k_pe_full[context_len:].unsqueeze(0).unsqueeze(0)
            q_rot_4d, k_pe_q = apply_rope(q_rot_4d, k_pe_q, cos_q, sin_q)
            # 还原形状
            q_rot = q_rot_4d.squeeze(0).permute(1, 0, 2)  # [q_len, N, R]
            k_pe_q = k_pe_q.squeeze(0).squeeze(0)  # [q_len, R]

            # 对 context token 的 k_pe 施加 RoPE（位置从 0 开始）
            if context_len > 0:
                cos_ctx, sin_ctx = build_rope_embeddings(
                    seq_len=context_len,
                    rope_theta=rope_theta,
                    dtype=hs.dtype,
                    device=device,
                    position_offset=0,
                )
                # k_pe_ctx: [ctx_len, R] -> [1, 1, ctx_len, R]
                k_pe_ctx_4d = k_pe_full[:context_len].unsqueeze(0).unsqueeze(0)
                # 用零 q 占位（只需要 k_pe 的 RoPE 结果）
                q_dummy = torch.zeros_like(k_pe_ctx_4d)
                __, k_pe_ctx_4d = apply_rope(q_dummy, k_pe_ctx_4d, cos_ctx, sin_ctx)
                k_pe_ctx = k_pe_ctx_4d.squeeze(0).squeeze(0)  # [ctx_len, R]
            else:
                k_pe_ctx = k_pe_full[:0]  # 空张量

            # 拼接 q_nope 和 q_rot（CPUMLAImpl 期望的格式）
            q_combined = torch.cat([q_nope, q_rot], dim=-1)  # [q_len, N, QK_HEAD_DIM]

            q_list.append(q_combined)
            kv_c_new_list.append(kv_c_normed_full[context_len:])
            k_pe_new_list.append(k_pe_q.unsqueeze(1))  # [q_len, 1, R]
            kv_c_ctx_list.append(kv_c_normed_full[:context_len])
            k_pe_ctx_list.append(k_pe_ctx.unsqueeze(1))  # [ctx_len, 1, R]

    # 构造 vllm 元数据
    common_attn_metadata = create_common_attn_metadata(batch_spec, BLOCK_SIZE, device)
    kv_cache = _create_kv_cache(
        kv_c_ctx_list,
        k_pe_ctx_list,
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
            num_heads=NUM_HEADS,
            head_size=HEAD_SIZE,
            scale=QK_HEAD_DIM**-0.5,
            num_kv_heads=NUM_HEADS,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            logits_soft_cap=None,
            attn_type="decoder",
            kv_sharing_target_layer_name=None,
            q_lora_rank=None,
            kv_lora_rank=KV_LORA_RANK,
            qk_nope_head_dim=QK_NOPE_HEAD_DIM,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            qk_head_dim=QK_HEAD_DIM,
            v_head_dim=V_HEAD_DIM,
            kv_b_proj=kv_b_proj,
        )
        impl.process_weights_after_loading(q_list[0].dtype)
        if impl.dcp_world_size == -1:
            impl.dcp_world_size = 1

        mock_layer = _MockMLALayer(
            impl=impl,
            kv_b_proj=kv_b_proj,
            o_proj_weight=weights.o_proj_weight,
            device=device,
        )

        layer_name = "test_layer_full_mla"
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
            NUM_HEADS * V_HEAD_DIM,
            dtype=query.dtype,
            device=device,
        )
        output = mock_layer.forward_impl(
            query, kv_c_new, k_pe_new, kv_cache, attn_metadata, output
        )

    return output


# ──────────────────────────────────────────────────────────────────────────────
# 余弦相似度统计
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
# pytest fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def cpu_device():
    return torch.device("cpu")


@pytest.fixture(scope="module")
def vllm_config_fixture():
    """创建用于完整 MLA 测试的 VllmConfig。"""
    import vllm.platforms as _platforms

    if (
        not hasattr(_platforms.current_platform, "device_type")
        or _platforms.current_platform.device_type != "cpu"
    ):
        from vllm.platforms.cpu import CpuPlatform

        _platforms.current_platform = CpuPlatform()

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


# ──────────────────────────────────────────────────────────────────────────────
# pytest 测试用例
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("case_name", list(TEST_CASES.keys()))
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_full_mla_cosine_similarity(
    case_name: str,
    dtype: torch.dtype,
    vllm_config_fixture,
    cpu_device,
):
    """验证完整 MLA（hidden_states -> output）的余弦相似度。

    两侧均从相同的 hidden_states 出发，走完整的 MLA 计算流程（含真实 RoPE）。
    对比的是 o_proj 之后的最终输出（hidden_size 维度）。
    """
    device = cpu_device
    batch_spec = TEST_CASES[case_name]
    threshold = COSINE_THRESHOLD[dtype]

    # 共享权重
    weights = make_mla_weights(
        dtype=dtype,
        device=device,
        seed=CONFIG["weight_seed"],
    )

    # 共享输入 hidden_states
    torch.manual_seed(CONFIG["input_seed"])
    hidden_states_list = [
        torch.randn(s_len, HIDDEN_SIZE, dtype=dtype, device=device)
        for s_len in batch_spec.seq_lens
    ]

    # HF 参考输出
    hf_attn = build_hf_attention(weights=weights, dtype=dtype, device=device)
    hf_output = compute_hf_output(
        batch_spec=batch_spec,
        hidden_states_list=hidden_states_list,
        hf_attn=hf_attn,
        device=device,
    )

    # CPUMLAImpl 输出
    cpu_output = compute_cpu_mla_output(
        batch_spec=batch_spec,
        hidden_states_list=hidden_states_list,
        weights=weights,
        vllm_config=vllm_config_fixture,
        device=device,
    )

    assert hf_output.shape == cpu_output.shape, (
        f"shape mismatch: hf={hf_output.shape}, cpu_mla={cpu_output.shape}"
    )

    stats = cosine_similarity_stats(hf_output, cpu_output)
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
    import vllm.platforms as _platforms

    if (
        not hasattr(_platforms.current_platform, "device_type")
        or _platforms.current_platform.device_type != "cpu"
    ):
        from vllm.platforms.cpu import CpuPlatform

        _platforms.current_platform = CpuPlatform()

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

    print("\n" + "=" * 92)
    print(
        "完整 MLA 端到端对比：CPUMLAImpl vs "
        "transformers.DeepseekV3Attention（含真实 RoPE）"
    )
    print("=" * 92)
    print(
        f"{'场景':<30} {'dtype':<12} {'min cos':<12} "
        f"{'mean cos':<12} {'max cos':<12} {'状态'}"
    )
    print("-" * 92)

    all_pass = True
    for case_name in cases:
        batch_spec = TEST_CASES[case_name]
        for dtype in dtypes:
            threshold = COSINE_THRESHOLD[dtype]

            weights = make_mla_weights(
                dtype=dtype,
                device=device,
                seed=args.seed,
            )

            torch.manual_seed(CONFIG["input_seed"])
            hidden_states_list = [
                torch.randn(s_len, HIDDEN_SIZE, dtype=dtype, device=device)
                for s_len in batch_spec.seq_lens
            ]

            hf_attn = build_hf_attention(weights=weights, dtype=dtype, device=device)
            hf_output = compute_hf_output(
                batch_spec=batch_spec,
                hidden_states_list=hidden_states_list,
                hf_attn=hf_attn,
                device=device,
            )

            cpu_output = compute_cpu_mla_output(
                batch_spec=batch_spec,
                hidden_states_list=hidden_states_list,
                weights=weights,
                vllm_config=vllm_config,
                device=device,
            )

            stats = cosine_similarity_stats(hf_output, cpu_output)
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

    print("=" * 92)
    print("总体结果：" + ("✅ 全部通过" if all_pass else "❌ 存在失败项"))
    print("=" * 92 + "\n")
    return 0 if all_pass else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "完整 MLA 端到端对比：CPUMLAImpl vs transformers.DeepseekV3Attention"
        )
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
        default=CONFIG["weight_seed"],
        help=f"权重随机种子（默认 {CONFIG['weight_seed']}）",
    )
    parsed_args = parser.parse_args()
    sys.exit(_run_report(parsed_args))
