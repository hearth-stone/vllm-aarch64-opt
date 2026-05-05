# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU MLA (Multi-head Latent Attention) 后端的单元测试。

测试策略：
1. 构造随机的 Q / KV_C / K_PE 张量，并用 PyTorch SDPA 计算参考输出。
2. 将上下文部分写入分页 KV Cache，再调用 CPUMLAImpl 的 forward_mha /
   forward_mqa 路径，比较两者输出是否一致。

覆盖场景：
- Prefill-only（纯 prefill）
- Decode-only（纯 decode）
- Mixed（prefill + decode 混合）
"""

import pytest
import torch
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

# DeepSeek V3 / R1 的 MLA 超参数
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_NOPE_HEAD_DIM = 128
V_HEAD_DIM = 128
HEAD_SIZE = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
BLOCK_SIZE = 16

# 测试用的批次规格
BATCH_SPECS = {
    "single_decode": BatchSpec(seq_lens=[32], query_lens=[1]),
    "single_prefill": BatchSpec(seq_lens=[32], query_lens=[16]),
    "small_decode": BatchSpec(seq_lens=[32, 48], query_lens=[1, 1]),
    "small_prefill": BatchSpec(seq_lens=[32, 48], query_lens=[8, 8]),
    "mixed_small": BatchSpec(seq_lens=[32, 48, 64, 80], query_lens=[1, 1, 6, 6]),
    "medium_decode": BatchSpec(
        seq_lens=[128, 256, 512, 1024],
        query_lens=[1, 1, 1, 1],
    ),
    "medium_prefill": BatchSpec(
        seq_lens=[256, 512],
        query_lens=[16, 16],
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────────────────────────────────────


class _MockKvBProj(torch.nn.Module):
    """轻量级 kv_b_proj mock，替代 ColumnParallelLinear。

    避免 ColumnParallelLinear.__init__ 调用 get_tensor_model_parallel_rank()
    导致的分布式环境依赖。接口与 ColumnParallelLinear 保持一致：
    - self.weight: [output_size, input_size]（转置存储，与 nn.Linear 相同）
    - __call__(x) -> (output, None)
    - cpu_linear(x, weight, bias) -> Tensor
        与 CPU 路径下 process_weights_after_loading 注入的 `cpu_linear`
        lambda 保持签名一致，供 CPUMLAImpl._linear 直接调用。
    """

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        # weight shape: [output_size, input_size]，与 ColumnParallelLinear 一致
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        # 模拟产品代码中 dispatch_cpu_unquantized_gemm 注入的 cpu_linear。
        # 真实对象上 layer.cpu_linear(x, layer.weight, bias) 会执行一次线性变换。
        self.cpu_linear = lambda x, weight, bias=None: torch.nn.functional.linear(
            x, weight, bias
        )
        # 保持与 LinearBase 一致，便于 CPUMLAImpl._linear 里的 getattr 判断。
        self.skip_bias_add = False

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        # x: [..., input_size]，output: [..., output_size]
        return (torch.nn.functional.linear(x, self.weight), None)


def _build_kv_b_proj(
    kv_lora_rank: int,
    num_heads: int,
    qk_nope_head_dim: int,
    v_head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    W_UK: torch.Tensor,
    W_UV: torch.Tensor,
) -> _MockKvBProj:
    """根据 W_UK / W_UV 构造 kv_b_proj mock。

    weight 的形状为 [output_size, input_size]，其中
    output_size = num_heads * (qk_nope_head_dim + v_head_dim)，
    input_size  = kv_lora_rank。
    """
    # W_UK: [kv_lora_rank, num_heads, qk_nope_head_dim]
    # W_UV: [kv_lora_rank, num_heads, v_head_dim]
    # 拼接后 reshape 为 [kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim)]
    # 再转置为 [output_size, kv_lora_rank]，与 nn.Linear weight 布局一致
    kv_b_proj_weight = (
        torch.cat([W_UK, W_UV], dim=-1)
        .view(kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim))
        .T.contiguous()
        .to(device=device, dtype=dtype)
    )

    return _MockKvBProj(kv_b_proj_weight)


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
    """创建并填充分页 KV Cache。

    KV Cache 形状：[num_blocks, block_size, head_size]
    其中 head_size = kv_lora_rank + qk_rope_head_dim。

    block 0 保留为 null block，实际数据从 block 1 开始顺序写入。
    """
    batch_size = len(kv_c_contexts)
    seq_lens = common_attn_metadata.seq_lens.cpu()
    query_lens = (
        common_attn_metadata.query_start_loc_cpu[1:]
        - common_attn_metadata.query_start_loc_cpu[:-1]
    )
    context_lens = seq_lens - query_lens

    total_blocks = sum(cdiv(int(seq_lens[i]), block_size) for i in range(batch_size))
    num_blocks = total_blocks + 1 + num_extra_blocks

    kv_cache = torch.zeros(
        num_blocks, block_size, head_size, dtype=dtype, device=device
    )
    kv_cache_flat = kv_cache.view(-1, head_size)

    block_table = common_attn_metadata.block_table_tensor
    slot_mapping = common_attn_metadata.slot_mapping

    # 顺序写入上下文 token（从 block 1 开始）
    start_block_idx = 1
    for i in range(batch_size):
        kv_c_ctx = kv_c_contexts[i]
        k_pe_ctx = k_pe_contexts[i]
        ctx_len = kv_c_ctx.shape[0]

        num_blocks_for_seq = cdiv(int(seq_lens[i]), block_size)

        if ctx_len > 0:
            kv_ctx = torch.cat([kv_c_ctx, k_pe_ctx.squeeze(1)], dim=-1)
            start_flat = start_block_idx * block_size
            kv_cache_flat[start_flat : start_flat + ctx_len] = kv_ctx

        # 更新 block_table：该序列占用的 block 索引
        for b in range(num_blocks_for_seq):
            block_table[i, b] = start_block_idx + b
        block_table[i, num_blocks_for_seq:] = 0

        # 更新 slot_mapping：新 token 的 slot 位置
        q_start = int(common_attn_metadata.query_start_loc_cpu[i])
        q_end = int(common_attn_metadata.query_start_loc_cpu[i + 1])
        for t_idx, t in enumerate(range(q_start, q_end)):
            token_pos = int(context_lens[i]) + t_idx
            block_idx = token_pos // block_size
            block_offset = token_pos % block_size
            slot_mapping[t] = (start_block_idx + block_idx) * block_size + block_offset

        start_block_idx += num_blocks_for_seq

    return kv_cache


class MockMLALayer(AttentionLayerBase):
    """用于测试的 Mock MLA 注意力层。

    复现 MLAAttention.forward_impl 的核心逻辑，允许在不依赖完整模型
    基础设施的情况下测试 CPUMLAImpl。
    """

    def __init__(
        self,
        impl,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        kv_lora_rank: int,
        device: torch.device,
        kv_b_proj,
    ):
        self.impl = impl
        self.num_heads = num_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.kv_lora_rank = kv_lora_rank

        # 从 kv_b_proj 中提取 W_UK_T 和 W_UV（与 MLAAttention 一致）
        kv_b_proj_weight = kv_b_proj.weight.T  # [kv_lora_rank, num_heads * (P + V)]
        kv_b_proj_weight = kv_b_proj_weight.view(
            kv_lora_rank, num_heads, qk_nope_head_dim + v_head_dim
        )
        W_UK, W_UV = kv_b_proj_weight.split([qk_nope_head_dim, v_head_dim], dim=-1)
        # W_UK: [L, N, P] -> W_UK_T: [N, P, L]
        self.W_UK_T = W_UK.permute(1, 2, 0).contiguous()
        # W_UV: [L, N, V] -> [N, L, V]
        self.W_UV = W_UV.transpose(0, 1).contiguous()

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
        """纯 PyTorch 实现 concat_and_cache_mla，替代 CUDA 自定义算子。

        将 kv_c 和 k_pe 拼接后，按 slot_mapping 写入 kv_cache。

        Args:
            kv_c: shape = [num_tokens, kv_lora_rank]
            k_pe: shape = [num_tokens, qk_rope_head_dim]
            kv_cache: shape = [num_blocks, block_size, kv_lora_rank + qk_rope_head_dim]
            slot_mapping: shape = [num_tokens]，每个 token 对应的 cache slot 索引
        """
        # 拼接 kv_c 和 k_pe：[num_tokens, kv_lora_rank + qk_rope_head_dim]
        kv_combined = torch.cat([kv_c, k_pe], dim=-1)
        block_size = kv_cache.shape[1]
        num_tokens = slot_mapping.shape[0]
        for i in range(num_tokens):
            slot = slot_mapping[i].item()
            if slot < 0:
                # 负数 slot 表示 padding，跳过
                continue
            block_idx = slot // block_size
            block_offset = slot % block_size
            kv_cache[block_idx, block_offset] = kv_combined[i]

    def forward_impl(
        self,
        q: torch.Tensor,
        kv_c: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """复现 MLAAttention.forward_impl 的核心逻辑。"""
        # 写入 KV Cache（使用纯 PyTorch 实现，避免 CUDA 自定义算子依赖）
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

        # Prefill 路径：forward_mha
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

        # Decode 路径：forward_mqa（absorption 技巧）
        if has_decode:
            decode_q = q[:num_decode_tokens]
            mqa_q_nope, mqa_q_pe = decode_q.split(
                [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
            )
            # (B, N, P) -> (N, B, P) -> bmm -> (N, B, L) -> (B, N, L)
            mqa_ql_nope = torch.bmm(mqa_q_nope.transpose(0, 1), self.W_UK_T).transpose(
                0, 1
            )

            attn_out, _ = self.impl.forward_mqa(
                (mqa_ql_nope, mqa_q_pe), kv_cache, attn_metadata, self
            )

            # v_up 投影：(B, N, L) x (N, L, V) -> (B, N, V) -> flatten
            decode_output = torch.bmm(attn_out.transpose(0, 1), self.W_UV).transpose(
                0, 1
            )
            output[:num_decode_tokens] = decode_output.reshape(
                num_decode_tokens, self.num_heads * self.v_head_dim
            )

        return output


def _compute_sdpa_reference(
    batch_spec: BatchSpec,
    q_list: list[torch.Tensor],
    kv_c_full_list: list[torch.Tensor],
    k_pe_full_list: list[torch.Tensor],
    W_UK: torch.Tensor,
    W_UV: torch.Tensor,
    kv_b_proj_weight: torch.Tensor,
    num_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    kv_lora_rank: int,
    scale: float,
    use_decode_path: bool,
) -> torch.Tensor:
    """用 PyTorch SDPA 计算参考输出。

    Args:
        use_decode_path: True 时使用 MQA（decode）路径，
            False 时使用 MHA（prefill）路径。

    Returns:
        参考输出，shape = [total_query_tokens, num_heads * v_head_dim]
    """
    outputs = []
    for i in range(batch_spec.batch_size):
        s_len = batch_spec.seq_lens[i]
        q_len = batch_spec.query_lens[i]
        context_len = s_len - q_len

        q_i = q_list[i]  # [q_len, num_heads, qk_nope_head_dim + qk_rope_head_dim]
        kv_c_full = kv_c_full_list[i]  # [s_len, kv_lora_rank]
        k_pe_full = k_pe_full_list[i]  # [s_len, 1, qk_rope_head_dim]

        q_nope, q_pe = q_i.split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)

        # 因果掩码：query 可以看到所有 context token，以及 query 中自身之前的 token
        attn_mask = torch.ones(q_len, s_len, dtype=torch.bool, device=q_i.device)
        causal_mask = torch.tril(torch.ones(q_len, q_len, device=q_i.device))
        attn_mask[:, context_len:] = causal_mask

        if use_decode_path:
            # MQA 路径（decode absorption）
            ql_nope = torch.einsum("qnh,lnh->qnl", q_nope, W_UK)
            q_mqa = torch.cat([ql_nope, q_pe], dim=-1)
            k_mqa = (
                torch.cat([kv_c_full, k_pe_full.squeeze(1)], dim=-1)
                .unsqueeze(1)
                .expand(-1, num_heads, -1)
            )
            v_mqa = kv_c_full.unsqueeze(1).expand(-1, num_heads, -1)

            # SDPA: (1, N, q_len, D)
            sdpa_out = F.scaled_dot_product_attention(
                q_mqa.unsqueeze(0).transpose(1, 2),
                k_mqa.unsqueeze(0).transpose(1, 2),
                v_mqa.unsqueeze(0).transpose(1, 2),
                attn_mask=attn_mask.unsqueeze(0).unsqueeze(0),
                scale=scale,
            )
            # [q_len, num_heads, kv_lora_rank]
            sdpa_out = sdpa_out.transpose(1, 2).squeeze(0)
            # v_up 投影
            sdpa_out = torch.einsum("qnl,lnv->qnv", sdpa_out, W_UV)
        else:
            # MHA 路径（prefill）
            kv_nope_full = torch.einsum("sl,lnh->snh", kv_c_full, kv_b_proj_weight)
            k_nope_full, v_full = kv_nope_full.split(
                [qk_nope_head_dim, v_head_dim], dim=-1
            )
            q_mha = torch.cat([q_nope, q_pe], dim=-1)
            k_full = torch.cat(
                [k_nope_full, k_pe_full.expand(-1, num_heads, -1)], dim=-1
            )

            sdpa_out = F.scaled_dot_product_attention(
                q_mha.unsqueeze(0).transpose(1, 2),
                k_full.unsqueeze(0).transpose(1, 2),
                v_full.unsqueeze(0).transpose(1, 2),
                attn_mask=attn_mask.unsqueeze(0).unsqueeze(0),
                scale=scale,
            )
            # [q_len, num_heads, v_head_dim]
            sdpa_out = sdpa_out.transpose(1, 2).squeeze(0)

        outputs.append(sdpa_out.flatten(start_dim=-2))

    return torch.cat(outputs, dim=0)


def _run_cpu_mla(
    batch_spec: BatchSpec,
    q_list: list[torch.Tensor],
    kv_c_new_list: list[torch.Tensor],
    k_pe_new_list: list[torch.Tensor],
    kv_c_ctx_list: list[torch.Tensor],
    k_pe_ctx_list: list[torch.Tensor],
    W_UK: torch.Tensor,
    W_UV: torch.Tensor,
    kv_b_proj,
    num_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    kv_lora_rank: int,
    scale: float,
    vllm_config,
    device: torch.device,
) -> torch.Tensor:
    """运行 CPUMLAImpl 并返回输出。"""
    from vllm.v1.attention.backends.mla.cpu_mla import (
        CPUMLABackend,
        CPUMLAMetadataBuilder,
    )

    common_attn_metadata = create_common_attn_metadata(batch_spec, BLOCK_SIZE, device)

    # 创建并填充 KV Cache
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
        impl_cls = CPUMLABackend.get_impl_cls()
        impl = impl_cls(
            num_heads=num_heads,
            head_size=HEAD_SIZE,
            scale=scale,
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

        # 初始化 DCP 属性（正常由 MLAAttention.forward 设置）
        if impl.dcp_world_size == -1:
            impl.dcp_world_size = 1

        mock_layer = MockMLALayer(
            impl=impl,
            num_heads=num_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            kv_lora_rank=kv_lora_rank,
            device=device,
            kv_b_proj=kv_b_proj,
        )

        layer_name = "test_layer"
        vllm_config.compilation_config.static_forward_context[layer_name] = mock_layer

        builder = CPUMLAMetadataBuilder(
            kv_cache_spec, [layer_name], vllm_config, device
        )
        attn_metadata = builder.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
        )

        # 拼接所有序列的新 token
        query = torch.cat(q_list, dim=0)
        kv_c_new = torch.cat(kv_c_new_list, dim=0)
        k_pe_new = torch.cat(k_pe_new_list, dim=0)

        num_tokens = query.shape[0]
        output = torch.zeros(
            num_tokens, num_heads * v_head_dim, dtype=query.dtype, device=device
        )

        output = mock_layer.forward_impl(
            query, kv_c_new, k_pe_new, kv_cache, attn_metadata, output
        )

    return output


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def cpu_device():
    return torch.device("cpu")


@pytest.fixture(scope="module")
def vllm_config_cpu():
    """创建用于 CPU MLA 测试的 VllmConfig。"""
    # 确保平台检测为 CPU，避免在无 GPU 环境下 DeviceConfig() 失败
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
    # CPU 测试不需要 GPU blocks
    cfg.cache_config.num_gpu_blocks = 1000
    cfg.cache_config.num_cpu_blocks = 0
    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# 测试用例
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "batch_spec_name",
    [
        "single_decode",
        "small_decode",
        "medium_decode",
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_decode_correctness(
    batch_spec_name: str,
    dtype: torch.dtype,
    vllm_config_cpu,
    cpu_device,
):
    """测试 Decode 路径（forward_mqa）的数值正确性。

    所有序列的 query_len == 1，走 decode（MQA absorption）路径。
    """
    torch.manual_seed(42)
    device = cpu_device
    batch_spec = BATCH_SPECS[batch_spec_name]
    batch_size = batch_spec.batch_size
    num_heads = vllm_config_cpu.model_config.get_num_attention_heads(
        vllm_config_cpu.parallel_config
    )
    scale = 1.0 / (HEAD_SIZE**0.5)
    weight_scale = 1.0 / (KV_LORA_RANK**0.5)

    # 共享权重矩阵
    W_UK = (
        torch.randn(
            KV_LORA_RANK, num_heads, QK_NOPE_HEAD_DIM, dtype=dtype, device=device
        )
        * weight_scale
    )
    W_UV = (
        torch.randn(KV_LORA_RANK, num_heads, V_HEAD_DIM, dtype=dtype, device=device)
        * weight_scale
    )
    kv_b_proj = _build_kv_b_proj(
        KV_LORA_RANK, num_heads, QK_NOPE_HEAD_DIM, V_HEAD_DIM, dtype, device, W_UK, W_UV
    )
    # kv_b_proj_weight: [kv_lora_rank, num_heads, qk_nope_head_dim + v_head_dim]
    kv_b_proj_weight = torch.cat([W_UK, W_UV], dim=-1)

    q_list, kv_c_new_list, k_pe_new_list = [], [], []
    kv_c_ctx_list, k_pe_ctx_list = [], []
    kv_c_full_list, k_pe_full_list = [], []

    for i in range(batch_size):
        s_len = batch_spec.seq_lens[i]
        q_len = batch_spec.query_lens[i]
        context_len = s_len - q_len

        q_i = torch.randn(
            q_len,
            num_heads,
            QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM,
            dtype=dtype,
            device=device,
        )
        kv_c_full = torch.randn(s_len, KV_LORA_RANK, dtype=dtype, device=device)
        k_pe_full = torch.randn(s_len, 1, QK_ROPE_HEAD_DIM, dtype=dtype, device=device)

        q_list.append(q_i)
        kv_c_new_list.append(kv_c_full[context_len:])
        k_pe_new_list.append(k_pe_full[context_len:])
        kv_c_ctx_list.append(kv_c_full[:context_len])
        k_pe_ctx_list.append(k_pe_full[:context_len])
        kv_c_full_list.append(kv_c_full)
        k_pe_full_list.append(k_pe_full)

    # 参考输出（SDPA decode 路径）
    expected = _compute_sdpa_reference(
        batch_spec,
        q_list,
        kv_c_full_list,
        k_pe_full_list,
        W_UK,
        W_UV,
        kv_b_proj_weight,
        num_heads,
        QK_NOPE_HEAD_DIM,
        QK_ROPE_HEAD_DIM,
        V_HEAD_DIM,
        KV_LORA_RANK,
        scale,
        use_decode_path=True,
    )

    # CPU MLA 输出
    actual = _run_cpu_mla(
        batch_spec,
        q_list,
        kv_c_new_list,
        k_pe_new_list,
        kv_c_ctx_list,
        k_pe_ctx_list,
        W_UK,
        W_UV,
        kv_b_proj,
        num_heads,
        QK_NOPE_HEAD_DIM,
        QK_ROPE_HEAD_DIM,
        V_HEAD_DIM,
        KV_LORA_RANK,
        scale,
        vllm_config_cpu,
        device,
    )

    assert actual.shape == expected.shape, (
        f"shape mismatch: actual={actual.shape}, expected={expected.shape}"
    )
    assert torch.isfinite(actual).all(), "CPU MLA decode 输出包含非有限值"

    rtol, atol = (1e-2, 5e-2) if dtype == torch.bfloat16 else (1e-4, 1e-4)
    max_diff = (actual - expected).abs().max().item()
    assert torch.allclose(actual, expected, rtol=rtol, atol=atol), (
        f"decode 路径数值误差过大：max_diff={max_diff:.6f}，dtype={dtype}"
    )


@pytest.mark.parametrize(
    "batch_spec_name",
    [
        "single_prefill",
        "small_prefill",
        "medium_prefill",
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_prefill_correctness(
    batch_spec_name: str,
    dtype: torch.dtype,
    vllm_config_cpu,
    cpu_device,
):
    """测试 Prefill 路径（forward_mha）的数值正确性。

    所有序列的 query_len > 1，走 prefill（MHA）路径。
    """
    torch.manual_seed(42)
    device = cpu_device
    batch_spec = BATCH_SPECS[batch_spec_name]
    batch_size = batch_spec.batch_size
    num_heads = vllm_config_cpu.model_config.get_num_attention_heads(
        vllm_config_cpu.parallel_config
    )
    scale = 1.0 / (HEAD_SIZE**0.5)
    weight_scale = 1.0 / (KV_LORA_RANK**0.5)

    W_UK = (
        torch.randn(
            KV_LORA_RANK, num_heads, QK_NOPE_HEAD_DIM, dtype=dtype, device=device
        )
        * weight_scale
    )
    W_UV = (
        torch.randn(KV_LORA_RANK, num_heads, V_HEAD_DIM, dtype=dtype, device=device)
        * weight_scale
    )
    kv_b_proj = _build_kv_b_proj(
        KV_LORA_RANK, num_heads, QK_NOPE_HEAD_DIM, V_HEAD_DIM, dtype, device, W_UK, W_UV
    )
    kv_b_proj_weight = torch.cat([W_UK, W_UV], dim=-1)

    q_list, kv_c_new_list, k_pe_new_list = [], [], []
    kv_c_ctx_list, k_pe_ctx_list = [], []
    kv_c_full_list, k_pe_full_list = [], []

    for i in range(batch_size):
        s_len = batch_spec.seq_lens[i]
        q_len = batch_spec.query_lens[i]
        context_len = s_len - q_len

        q_i = torch.randn(
            q_len,
            num_heads,
            QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM,
            dtype=dtype,
            device=device,
        )
        kv_c_full = torch.randn(s_len, KV_LORA_RANK, dtype=dtype, device=device)
        k_pe_full = torch.randn(s_len, 1, QK_ROPE_HEAD_DIM, dtype=dtype, device=device)

        q_list.append(q_i)
        kv_c_new_list.append(kv_c_full[context_len:])
        k_pe_new_list.append(k_pe_full[context_len:])
        kv_c_ctx_list.append(kv_c_full[:context_len])
        k_pe_ctx_list.append(k_pe_full[:context_len])
        kv_c_full_list.append(kv_c_full)
        k_pe_full_list.append(k_pe_full)

    # 参考输出（SDPA prefill 路径）
    expected = _compute_sdpa_reference(
        batch_spec,
        q_list,
        kv_c_full_list,
        k_pe_full_list,
        W_UK,
        W_UV,
        kv_b_proj_weight,
        num_heads,
        QK_NOPE_HEAD_DIM,
        QK_ROPE_HEAD_DIM,
        V_HEAD_DIM,
        KV_LORA_RANK,
        scale,
        use_decode_path=False,
    )

    actual = _run_cpu_mla(
        batch_spec,
        q_list,
        kv_c_new_list,
        k_pe_new_list,
        kv_c_ctx_list,
        k_pe_ctx_list,
        W_UK,
        W_UV,
        kv_b_proj,
        num_heads,
        QK_NOPE_HEAD_DIM,
        QK_ROPE_HEAD_DIM,
        V_HEAD_DIM,
        KV_LORA_RANK,
        scale,
        vllm_config_cpu,
        device,
    )

    assert actual.shape == expected.shape, (
        f"shape mismatch: actual={actual.shape}, expected={expected.shape}"
    )
    assert torch.isfinite(actual).all(), "CPU MLA prefill 输出包含非有限值"

    rtol, atol = (1e-2, 5e-2) if dtype == torch.bfloat16 else (1e-4, 1e-4)
    max_diff = (actual - expected).abs().max().item()
    assert torch.allclose(actual, expected, rtol=rtol, atol=atol), (
        f"prefill 路径数值误差过大：max_diff={max_diff:.6f}，dtype={dtype}"
    )


@pytest.mark.parametrize(
    "batch_spec_name",
    [
        "mixed_small",
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mixed_correctness(
    batch_spec_name: str,
    dtype: torch.dtype,
    vllm_config_cpu,
    cpu_device,
):
    """测试 Mixed 场景（prefill + decode 混合批次）的数值正确性。

    批次中同时包含 query_len == 1（decode）和 query_len > 1（prefill）的序列。
    decode 序列排在前面，prefill 序列排在后面（与 vLLM 的 reorder 逻辑一致）。
    """
    torch.manual_seed(42)
    device = cpu_device
    batch_spec = BATCH_SPECS[batch_spec_name]
    batch_size = batch_spec.batch_size
    num_heads = vllm_config_cpu.model_config.get_num_attention_heads(
        vllm_config_cpu.parallel_config
    )
    scale = 1.0 / (HEAD_SIZE**0.5)
    weight_scale = 1.0 / (KV_LORA_RANK**0.5)

    W_UK = (
        torch.randn(
            KV_LORA_RANK, num_heads, QK_NOPE_HEAD_DIM, dtype=dtype, device=device
        )
        * weight_scale
    )
    W_UV = (
        torch.randn(KV_LORA_RANK, num_heads, V_HEAD_DIM, dtype=dtype, device=device)
        * weight_scale
    )
    kv_b_proj = _build_kv_b_proj(
        KV_LORA_RANK, num_heads, QK_NOPE_HEAD_DIM, V_HEAD_DIM, dtype, device, W_UK, W_UV
    )
    kv_b_proj_weight = torch.cat([W_UK, W_UV], dim=-1)

    q_list, kv_c_new_list, k_pe_new_list = [], [], []
    kv_c_ctx_list, k_pe_ctx_list = [], []
    kv_c_full_list, k_pe_full_list = [], []
    # 记录每个序列是否走 decode 路径
    is_decode_list = []

    for i in range(batch_size):
        s_len = batch_spec.seq_lens[i]
        q_len = batch_spec.query_lens[i]
        context_len = s_len - q_len
        is_decode_list.append(q_len == 1)

        q_i = torch.randn(
            q_len,
            num_heads,
            QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM,
            dtype=dtype,
            device=device,
        )
        kv_c_full = torch.randn(s_len, KV_LORA_RANK, dtype=dtype, device=device)
        k_pe_full = torch.randn(s_len, 1, QK_ROPE_HEAD_DIM, dtype=dtype, device=device)

        q_list.append(q_i)
        kv_c_new_list.append(kv_c_full[context_len:])
        k_pe_new_list.append(k_pe_full[context_len:])
        kv_c_ctx_list.append(kv_c_full[:context_len])
        k_pe_ctx_list.append(k_pe_full[:context_len])
        kv_c_full_list.append(kv_c_full)
        k_pe_full_list.append(k_pe_full)

    # 分别计算 decode 和 prefill 序列的参考输出，再按顺序拼接
    expected_parts = []
    for i in range(batch_size):
        single_spec = BatchSpec(
            seq_lens=[batch_spec.seq_lens[i]],
            query_lens=[batch_spec.query_lens[i]],
        )
        ref = _compute_sdpa_reference(
            single_spec,
            [q_list[i]],
            [kv_c_full_list[i]],
            [k_pe_full_list[i]],
            W_UK,
            W_UV,
            kv_b_proj_weight,
            num_heads,
            QK_NOPE_HEAD_DIM,
            QK_ROPE_HEAD_DIM,
            V_HEAD_DIM,
            KV_LORA_RANK,
            scale,
            use_decode_path=is_decode_list[i],
        )
        expected_parts.append(ref)
    expected = torch.cat(expected_parts, dim=0)

    actual = _run_cpu_mla(
        batch_spec,
        q_list,
        kv_c_new_list,
        k_pe_new_list,
        kv_c_ctx_list,
        k_pe_ctx_list,
        W_UK,
        W_UV,
        kv_b_proj,
        num_heads,
        QK_NOPE_HEAD_DIM,
        QK_ROPE_HEAD_DIM,
        V_HEAD_DIM,
        KV_LORA_RANK,
        scale,
        vllm_config_cpu,
        device,
    )

    assert actual.shape == expected.shape, (
        f"shape mismatch: actual={actual.shape}, expected={expected.shape}"
    )
    assert torch.isfinite(actual).all(), "CPU MLA mixed 输出包含非有限值"

    rtol, atol = (1e-2, 5e-2) if dtype == torch.bfloat16 else (1e-4, 1e-4)
    max_diff = (actual - expected).abs().max().item()
    assert torch.allclose(actual, expected, rtol=rtol, atol=atol), (
        f"mixed 路径数值误差过大：max_diff={max_diff:.6f}，dtype={dtype}"
    )


def test_backend_registration():
    """测试 CPUMLABackend 是否正确注册到 AttentionBackendEnum。"""
    backend_cls = AttentionBackendEnum.CPU_MLA.get_class()
    assert backend_cls.get_name() == "CPU_MLA"
    assert backend_cls.get_impl_cls().__name__ == "CPUMLAImpl"
    assert backend_cls.get_builder_cls().__name__ == "CPUMLAMetadataBuilder"


def test_unsupported_features():
    """测试 CPUMLAImpl 对不支持特性的拒绝行为。"""
    from vllm.v1.attention.backends.mla.cpu_mla import CPUMLAImpl

    common_kwargs = dict(
        num_heads=4,
        head_size=HEAD_SIZE,
        scale=1.0 / HEAD_SIZE**0.5,
        num_kv_heads=4,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type="decoder",
        kv_sharing_target_layer_name=None,
        q_lora_rank=None,
        kv_lora_rank=KV_LORA_RANK,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        qk_head_dim=QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM,
        v_head_dim=V_HEAD_DIM,
        kv_b_proj=None,
    )

    with pytest.raises(NotImplementedError, match="alibi_slopes"):
        CPUMLAImpl(alibi_slopes=[0.1], **common_kwargs)

    with pytest.raises(NotImplementedError, match="sliding_window"):
        kwargs = dict(common_kwargs)
        kwargs["sliding_window"] = 512
        CPUMLAImpl(alibi_slopes=None, **kwargs)

    with pytest.raises(NotImplementedError, match="logits_soft_cap"):
        kwargs = dict(common_kwargs)
        kwargs["logits_soft_cap"] = 30.0
        CPUMLAImpl(alibi_slopes=None, **kwargs)

    with pytest.raises(NotImplementedError, match="DECODER"):
        CPUMLAImpl(alibi_slopes=None, **{**common_kwargs, "attn_type": "encoder"})


def test_gather_kv_cache():
    """测试 _gather_kv_cache 方法的正确性。"""
    from vllm.v1.attention.backends.mla.cpu_mla import CPUMLAImpl

    torch.manual_seed(0)
    block_size = 4
    head_size = 8
    num_blocks = 10
    seq_len = 7  # 跨越两个 block（4 + 3）

    kv_cache = torch.randn(num_blocks, block_size, head_size)
    # block_table: 序列使用 block 2 和 block 5
    block_table = torch.tensor([2, 5, 0, 0], dtype=torch.int32)

    # 手动构造期望输出
    expected = torch.cat(
        [
            kv_cache[2].reshape(-1, head_size),  # block 2 的全部 4 个 token
            kv_cache[5, :3],  # block 5 的前 3 个 token
        ],
        dim=0,
    )

    # 创建一个最小化的 CPUMLAImpl 实例（仅用于调用 _gather_kv_cache）
    # 使用 object.__new__ 绕过 __init__
    impl = object.__new__(CPUMLAImpl)
    gathered = impl._gather_kv_cache(kv_cache, block_table, seq_len, block_size)

    assert gathered.shape == (seq_len, head_size)
    assert torch.allclose(gathered, expected), (
        f"_gather_kv_cache 输出不正确：\n{gathered}\n期望：\n{expected}"
    )
