# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any, Union

import torch
from safetensors.torch import _TYPES as _SAFETENSORS_TO_TORCH_DTYPE
from transformers import PretrainedConfig

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.distributed import get_tensor_model_parallel_rank, get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE,
    FusedMoEConfig,
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.parameter import GroupQuantScaleParameter, PackedvLLMParameter
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.transformers_utils.config import get_safetensors_params_metadata

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization import QuantizationMethods
    from vllm.model_executor.models.utils import WeightsMapper

logger = init_logger(__name__)

# ---- fused_cpp 分派辅助 ----
# 延迟解析 `fused_cpp.w4a8_linear`。首次调用时才尝试 import，
# 成功 → 缓存引用；失败 → 打印一次告警并永久回落到默认 AWQ 路径。
_FUSED_CPP_W4A8_LINEAR: Any = None
_FUSED_CPP_LOAD_FAILED: bool = False

def _get_fused_cpp_w4a8_linear() -> Any:
    """懒加载 fused_cpp.w4a8_linear，返回可调用对象或 None。"""
    global _FUSED_CPP_W4A8_LINEAR, _FUSED_CPP_LOAD_FAILED
    if _FUSED_CPP_W4A8_LINEAR is not None:
        return _FUSED_CPP_W4A8_LINEAR
    if _FUSED_CPP_LOAD_FAILED:
        return None
    try:
        from fused_cpp import w4a8_linear
    except ImportError as exc:
        _FUSED_CPP_LOAD_FAILED = True
        logger.warning_once(
            "VLLM_CPU_AWQ_USE_FUSED_CPP=1 但 fused_cpp 导入失败 (%s)，"
            "回落到默认 AWQ CPU 路径。",
            exc,
        )
        return None
    _FUSED_CPP_W4A8_LINEAR = w4a8_linear
    return w4a8_linear

def _should_use_fused_cpp_awq() -> bool:
    """仅在 CPU 平台 + 环境变量开启 + fused_cpp 可导入时返回 True。"""
    if not envs.VLLM_CPU_AWQ_USE_FUSED_CPP:
        return False
    if not current_platform.is_cpu():
        return False
    return _get_fused_cpp_w4a8_linear() is not None

class AWQConfig(QuantizationConfig):
    """Config class for AWQ.

    Reference: https://arxiv.org/abs/2306.00978
    """

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        zero_point: bool,
        modules_to_not_convert: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.zero_point = zero_point
        self.modules_to_not_convert = modules_to_not_convert or []

        if self.weight_bits != 4:
            raise ValueError(
                "Currently, only 4-bit weight quantization is supported for "
                f"AWQ, but got {self.weight_bits} bits."
            )
        self.pack_factor = 32 // self.weight_bits

    def __repr__(self) -> str:
        return (
            f"AWQConfig(weight_bits={self.weight_bits}, "
            f"group_size={self.group_size}, "
            f"zero_point={self.zero_point}, "
            f"modules_to_not_convert={self.modules_to_not_convert})"
        )

    def get_name(self) -> "QuantizationMethods":
        return "awq"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        # CPU 后端（AArch64 鲲鹏）下的 AWQ 通过 fused_cpp.w4a8_linear 与
        # CPUAWQFusedMoEMethod 两条路径原生支持 bf16；GPU 原生 AWQ kernel
        # 仅支持 fp16，故此处按平台放行 bf16。
        if current_platform.is_cpu():
            return [torch.half, torch.bfloat16]
        return [torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        # The AWQ kernel only supports Turing or newer GPUs.
        return 75

    @staticmethod
    def get_config_filenames() -> list[str]:
        return [
            "quant_config.json",  # E.g., casperhansen/vicuna-7b-v1.5-awq
            # E.g., abhinavkulkarni/mosaicml-mpt-7b-instruct-w4-g128-awq
            "quantize_config.json",
        ]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AWQConfig":
        weight_bits = cls.get_from_keys(config, ["w_bit", "bits"])
        group_size = cls.get_from_keys(config, ["q_group_size", "group_size"])
        zero_point = cls.get_from_keys(config, ["zero_point"])
        modules_to_not_convert = cls.get_from_keys_or(
            config, ["modules_to_not_convert"], None
        )
        return cls(weight_bits, group_size, zero_point, modules_to_not_convert)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Union["LinearMethodBase", "QuantizeMethodBase"] | None:
        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                prefix,
                self.modules_to_not_convert,
                self.packed_modules_mapping,
                skip_with_substr=True,
            ):
                return UnquantizedLinearMethod()
            return AWQLinearMethod(self)
        elif isinstance(layer, FusedMoE):
            # Lazy import to avoid circular import.
            from .awq_marlin import AWQMarlinConfig
            from .moe_wna16 import MoeWNA16Config
            from .utils.marlin_utils import check_moe_marlin_supports_layer

            # CPU（AArch64 鲲鹏）+ VLLM_CPU_AWQ_USE_FUSED_CPP=1 + fused_cpp 可用：
            # 完全绕开 GPU 侧 AWQMarlin / MoeWNA16 路径，返回 CPU 原生 AWQ MoE 方法。
            # 该方法内部通过 fused_cpp.w4a8_linear 逐 expert 执行 W4A8 GEMM，
            # 与我们已接通的 Linear 侧 fused_cpp 路径同源。
            if _should_use_fused_cpp_awq():
                return CPUAWQFusedMoEMethod(self, layer.moe_config)

            if not check_moe_marlin_supports_layer(layer, self.group_size):
                logger.warning_once(
                    f"Layer '{prefix}' is not supported by AWQMoeMarlin. "
                    "Falling back to Moe WNA16 kernels."
                )
                config = {
                    "quant_method": "awq",
                    "bits": self.weight_bits,
                    "group_size": self.group_size,
                    "zero_point": self.zero_point,
                    "lm_head": False,
                    "modules_to_not_convert": self.modules_to_not_convert,
                }
                return MoeWNA16Config.from_config(config).get_quant_method(
                    layer, prefix
                )
            marlin_compatible_config_dict = {
                "quant_method": "awq",
                "bits": self.weight_bits,
                "group_size": self.group_size,
                "zero_point": self.zero_point,
                "lm_head": False,
                "modules_to_not_convert": self.modules_to_not_convert,
            }
            awq_marlin_config = AWQMarlinConfig.from_config(
                marlin_compatible_config_dict
            )
            return awq_marlin_config.get_quant_method(layer, prefix)
        return None

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        if self.modules_to_not_convert:
            self.modules_to_not_convert = hf_to_vllm_mapper.apply_list(
                self.modules_to_not_convert
            )

    def maybe_update_config(
        self,
        model_name: str,
        hf_config: PretrainedConfig | None = None,
        revision: str | None = None,
    ):
        if self.modules_to_not_convert:
            return

        unquant_dtypes = [torch.float16, torch.bfloat16, torch.float32]
        metadata = get_safetensors_params_metadata(model_name, revision=revision)
        layers = {param_name.rsplit(".", 1)[0] for param_name in metadata}
        quant_layers: set[str] = {
            param_name.rsplit(".", 1)[0]
            for param_name, info in metadata.items()
            if (dtype := info.get("dtype", None))
            and _SAFETENSORS_TO_TORCH_DTYPE[dtype] not in unquant_dtypes
        }
        self.modules_to_not_convert = list(layers - quant_layers)


class AWQLinearMethod(LinearMethodBase):
    """Linear method for AWQ.

    Args:
        quant_config: The AWQ quantization config.
    """

    def __init__(self, quant_config: AWQConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # Normalize group_size
        if self.quant_config.group_size != -1:
            group_size = self.quant_config.group_size
        else:
            group_size = input_size

        if input_size_per_partition % group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size."
            )

        output_size_per_partition = sum(output_partition_sizes)
        if output_size_per_partition % self.quant_config.pack_factor != 0:
            raise ValueError(
                "The output size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size."
            )

        weight_loader = extra_weight_attrs.get("weight_loader")
        qweight = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader,
        )

        num_groups = input_size_per_partition // group_size

        qzeros = PackedvLLMParameter(
            data=torch.empty(
                num_groups,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader,
        )

        scales = GroupQuantScaleParameter(
            data=torch.empty(
                num_groups,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=0,
            output_dim=1,
            weight_loader=weight_loader,
        )

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("qzeros", qzeros)
        layer.register_parameter("scales", scales)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.qweight = torch.nn.Parameter(layer.qweight.data, requires_grad=False)
        layer.qzeros = torch.nn.Parameter(layer.qzeros.data, requires_grad=False)
        layer.scales = torch.nn.Parameter(layer.scales.data, requires_grad=False)

        # ─────────────────────────────────────────────────────────────────
        # MLA 兼容补丁：CPU + fused_cpp 路径下，若该 Linear 被 MLAAttention
        # 标记为 ``_preserve_original_weight=True``（目前仅 ``kv_b_proj``），
        # 则需要把 AWQ 权重反量化为 bf16 的稠密权重 ``layer.weight``，并调用
        # ``dispatch_cpu_unquantized_gemm`` 构建 ``layer.cpu_linear``，以配合
        # ``cpu_mla._linear`` 的调用契约 ``layer.cpu_linear(x, layer.weight, b)``。
        #
        # 注意：该分支**只影响 kv_b_proj 一个 Linear**；其他 AWQ Linear
        # （qkv_proj / o_proj / gate_up_proj / down_proj）依旧走 ``_apply_fused_cpp``
        # 的 w4a8 快路径，不挂 ``layer.weight``，以避免显存浪费。
        # ─────────────────────────────────────────────────────────────────
        if not getattr(layer, "_preserve_original_weight", False):
            return
        if not _should_use_fused_cpp_awq():
            return
        self._materialize_dense_weight_for_mla(layer)

    @staticmethod
    def _materialize_dense_weight_for_mla(layer: torch.nn.Module) -> None:
        """将 AWQ 量化权重反量化为 bf16 稠密 ``[N, K]`` 权重并挂到 ``layer``。

        借用 ``fused_cpp.moe.dequant_awq_to_bf16``（语义与
        ``fused_cpp.w4a8_linear`` 完全一致）完成反量化，再通过
        ``dispatch_cpu_unquantized_gemm(layer, remove_weight=False)`` 构建
        ``layer.cpu_linear``。保留 ``layer.weight`` 供 ``MLAAttention`` 切分
        ``W_UK_T`` / ``W_UV`` 使用。
        """
        # 延迟导入，避免非 CPU 平台加载 fused_cpp 依赖。
        from fused_cpp.moe import dequant_awq_to_bf16
        from vllm.model_executor.layers.utils import dispatch_cpu_unquantized_gemm

        # ``dequant_awq_to_bf16`` 返回 [K, N] bf16；nn.Linear.weight 的约定
        # 是 [out_features=N, in_features=K]，因此需要 transpose + contiguous。
        w_kn = dequant_awq_to_bf16(layer.qweight, layer.qzeros, layer.scales)
        weight_nk = w_kn.t().contiguous()
        layer.weight = torch.nn.Parameter(weight_nk, requires_grad=False)

        # 构建 cpu_linear。``remove_weight=False`` 是无条件的：内部还会结合
        # ``_preserve_original_weight`` 进一步保留原始 weight。
        dispatch_cpu_unquantized_gemm(layer, remove_weight=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # CPU + fused_cpp 路径（由 VLLM_CPU_AWQ_USE_FUSED_CPP 控制）。
        # 仅在 CPU 平台生效，其它平台完全不触发。
        if _should_use_fused_cpp_awq():
            return self._apply_fused_cpp(layer, x, bias)

        qweight = layer.qweight
        scales = layer.scales
        qzeros = layer.qzeros
        pack_factor = self.quant_config.pack_factor
        out_shape = x.shape[:-1] + (qweight.shape[-1] * pack_factor,)
        reshaped_x = x.reshape(-1, x.shape[-1])

        # num_tokens >= threshold
        FP16_MATMUL_HEURISTIC_CONDITION = x.shape[:-1].numel() >= 256
        # Batch invariant mode requires torch.matmul path
        # for Triton override
        if FP16_MATMUL_HEURISTIC_CONDITION or envs.VLLM_BATCH_INVARIANT:
            out = ops.awq_dequantize(qweight, scales, qzeros, 0, 0, 0)
            out = torch.matmul(reshaped_x, out)
        else:
            out = ops.awq_gemm(reshaped_x, qweight, scales, qzeros, pack_factor)
        if bias is not None:
            out.add_(bias)
        return out.reshape(out_shape)

    def _apply_fused_cpp(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """通过 fused_cpp.w4a8_linear 执行 per-token int8 激活 + int8 GEMM。

        约定：``--dtype=bfloat16`` 时 ``x`` / ``scales`` 均为 bf16；
        ``fused_cpp.w4a8_linear`` 已放宽为同时支持 fp16/bf16 的 scales。
        若宿主以 fp16 运行，这里统一将激活桥接为 bf16 并在输出侧转回原 dtype。
        """
        w4a8_linear = _get_fused_cpp_w4a8_linear()
        # 理论上在 _should_use_fused_cpp_awq 已校验，这里作为防御兜底：
        if w4a8_linear is None:  # pragma: no cover
            raise RuntimeError("fused_cpp.w4a8_linear 不可用")

        orig_dtype = x.dtype
        x_bf16 = x if x.dtype == torch.bfloat16 else x.to(torch.bfloat16)
        bias_bf16 = (
            None if bias is None
            else (bias if bias.dtype == torch.bfloat16 else bias.to(torch.bfloat16))
        )
        out_bf16 = w4a8_linear(
            x_bf16,
            layer.qweight,
            layer.qzeros,
            layer.scales,
            bias=bias_bf16,
        )
        return out_bf16 if orig_dtype == torch.bfloat16 else out_bf16.to(orig_dtype)


# ── CPU-only AWQ FusedMoE path（VLLM_CPU_AWQ_USE_FUSED_CPP=1 才启用）────────
#
# 该类只在 CPU（AArch64 鲲鹏）+ fused_cpp 可用时由 ``AWQConfig.get_quant_method``
# 返回。设计要点：
#
# 1. **存储格式保持 AWQ 原生**（int32 + AWQ interleave 顺序），gate/up 沿 N 维
#    简单拼接为 ``w13`` 的前后半，down 独立存为 ``w2``。这样 fused_cpp.w4a8_linear
#    可以零成本直接消费 per-expert view，无需任何格式转换。
#
# 2. **自定义 weight_loader**。vLLM 默认 ``FusedMoE.weight_loader`` 按
#    ``SHARD_ID_TO_SHARDED_DIM = {w1:0, w2:1, w3:0}`` 切分，假设参数布局是
#    ``[E, N(output), K//pack]``。但 AWQ 原生布局是 ``[K(input), N//pack]``，
#    与之相反。因此本实现绕开默认 loader，自己按 weight_name 后缀 +
#    shard_id 精确派发。
#
# 3. **shared expert 不经本类**。DeepSeek-V2 的 shared experts 是独立
#    ``DeepseekV2MLP``（3 个 Linear），自动由 ``AWQLinearMethod`` 分派，
#    进而被 ``_should_use_fused_cpp_awq()`` 短路到 ``fused_cpp.w4a8_linear``，
#    与本类无交互。
#
# 4. **EP 暂未接入**。当前路径假设 ``ep_size == 1``（TP-only）。EP 路径需要
#    expert_map、all2all 等额外配合，留到后续补齐；若在 EP>1 时被误启用，
#    ``AWQFusedMoEImpl.__init__`` 会显式抛 ``NotImplementedError``。
#
class CPUAWQFusedMoEMethod(FusedMoEMethodBase):
    """CPU-only AWQ Fused MoE method，底层复用 ``fused_cpp.w4a8_linear``。

    与 ``AWQLinearMethod`` 对等：后者驱动 shared expert 的 3 个 Linear，
    本类驱动 routed experts 的 per-expert FFN（topk 之后的部分）。
    """

    def __init__(self, quant_config: AWQConfig, moe: FusedMoEConfig) -> None:
        super().__init__(moe)
        self.quant_config = quant_config
        self.moe = moe
        # 延迟到 process_weights_after_loading 构造
        self._impl: Any | None = None

    # ── 权重注册 ────────────────────────────────────────────────────────────
    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        """注册 ``w13_*`` / ``w2_*`` 六组 Parameter，保持 AWQ 原生布局。

        布局（E = num_experts, H = hidden_size, F = intermediate_size_per_partition）::

            w13_qweight [E, H, 2F // 8]     int32   w1 前 F//8, w3 后 F//8
            w13_qzeros  [E, H // g, 2F // 8] int32
            w13_scales  [E, H // g, 2F]     params_dtype
            w2_qweight  [E, F, H // 8]      int32
            w2_qzeros   [E, F // g, H // 8] int32
            w2_scales   [E, F // g, H]      params_dtype
        """
        group_size = self.quant_config.group_size
        if group_size == -1:
            # CPU AWQ path 不支持 per-channel（每层一个 group），R1-AWQ 都是 64
            raise ValueError(
                "CPUAWQFusedMoEMethod 需要 AWQ group_size > 0，当前为 -1。"
            )
        if hidden_size % group_size != 0:
            raise ValueError(
                f"hidden_size={hidden_size} 必须被 group_size={group_size} 整除"
            )
        if intermediate_size_per_partition % group_size != 0:
            raise ValueError(
                f"intermediate_size_per_partition={intermediate_size_per_partition} "
                f"必须被 group_size={group_size} 整除"
            )
        pack = self.quant_config.pack_factor      # 8 for 4bit
        if intermediate_size_per_partition % pack != 0 or hidden_size % pack != 0:
            raise ValueError(
                f"hidden_size / intermediate 必须被 pack_factor={pack} 整除"
            )

        layer.num_experts = num_experts
        layer.hidden_size = hidden_size
        layer.intermediate_size_per_partition = intermediate_size_per_partition
        layer.group_size = group_size

        h_groups = hidden_size // group_size
        f_groups = intermediate_size_per_partition // group_size
        fused_n_packed = 2 * intermediate_size_per_partition // pack
        fused_n = 2 * intermediate_size_per_partition
        h_packed = hidden_size // pack
        f_packed = intermediate_size_per_partition // pack

        # 先构造自定义 weight_loader，用于覆盖 ``extra_weight_attrs`` 里由
        # ``FusedMoE.__init__`` 传入的默认 ``self.weight_loader``（bound method）。
        # 这一步至关重要：``DeepseekV2ForCausalLM.load_weights`` 调用的是
        # ``param.weight_loader(...)``（而非 ``layer.weight_loader``），因此
        # 必须把我们的 closure 以 Parameter 属性的形式注入，才能在真实加载
        # 流程中生效；否则会错误走进 vLLM 默认的 ``FusedMoE._load_w2`` 按
        # ``shard_dim=1`` 切分的路径，出现 dim 0 尺寸不匹配的错误。
        custom_loader = self._make_weight_loader(layer)

        # extra_weight_attrs 由 FusedMoE 传入，默认携带 ``weight_loader``
        # 指向 ``FusedMoE.weight_loader``（vLLM 默认实现）。这里替换为我们
        # 自己的 closure，同时保留其它属性（如 ``quant_method`` 等元信息）。
        param_attrs = dict(extra_weight_attrs)
        param_attrs["weight_loader"] = custom_loader

        # 六个 Parameter 都走默认 nn.Parameter（不走 PackedvLLMParameter），
        # 因为本类自定义 weight_loader，不依赖 packed_dim 元信息。
        def _register(name: str, shape: tuple[int, ...], dtype: torch.dtype) -> None:
            p = torch.nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
            layer.register_parameter(name, p)
            # 把自定义 loader 等属性挂到 Parameter 上，供 load_weights 消费。
            set_weight_attrs(p, param_attrs)

        _register("w13_qweight", (num_experts, hidden_size, fused_n_packed), torch.int32)
        _register("w13_qzeros",  (num_experts, h_groups,     fused_n_packed), torch.int32)
        _register("w13_scales",  (num_experts, h_groups,     fused_n),        params_dtype)
        _register("w2_qweight",  (num_experts, intermediate_size_per_partition, h_packed), torch.int32)
        _register("w2_qzeros",   (num_experts, f_groups,     h_packed),       torch.int32)
        _register("w2_scales",   (num_experts, f_groups,     hidden_size),    params_dtype)

        # 同时把 loader 绑到 layer 上，便于测试或其它路径通过 ``layer.weight_loader``
        # 手动触发（与挂在 Parameter 上的是同一个 closure，行为一致）。
        layer.weight_loader = custom_loader

    # ── 自定义 weight_loader ────────────────────────────────────────────────
    @staticmethod
    def _make_weight_loader(layer: torch.nn.Module) -> Any:
        """返回一个与 vLLM default ``FusedMoE.weight_loader`` 签名一致的 closure。

        只处理两类后缀：
          * qweight / qzeros / scales —— 其它后缀（``g_idx`` 等）直接跳过。
        """
        num_experts = layer.num_experts
        intermediate_size_per_partition = layer.intermediate_size_per_partition
        group_size = layer.group_size
        # TP rank/size：与 MoeWNA16 保持一致的获取方式
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = max(1, get_tp_group().world_size)

        pack = 8  # 4bit pack into int32

        def _narrow_n_for_w13(
            loaded: torch.Tensor, is_packed_n: bool,
        ) -> torch.Tensor:
            """gate/up proj：沿 N 维做 TP narrow。

            :param is_packed_n: True 则按 F//pack 维切；False 则按 F 维切（scales）。
            """
            n_dim = 1        # AWQ checkpoint gate/up: [H or H//g, F or F//pack]
            shard_n = (
                intermediate_size_per_partition // pack if is_packed_n
                else intermediate_size_per_partition
            )
            return loaded.narrow(n_dim, shard_n * tp_rank, shard_n)

        def _narrow_k_for_w2(loaded: torch.Tensor, is_grouped_k: bool) -> torch.Tensor:
            """down proj：沿 K=F 维做 TP narrow。

            :param is_grouped_k: True 则按 F/g 维切（qzeros/scales），
                                 False 则按 F 维切（qweight）。
            """
            k_dim = 0        # AWQ checkpoint down: [F or F//g, H or H//pack]
            shard_k = (
                intermediate_size_per_partition // group_size if is_grouped_k
                else intermediate_size_per_partition
            )
            return loaded.narrow(k_dim, shard_k * tp_rank, shard_k)

        def loader(
            param: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            weight_name: str,
            shard_id: str,
            expert_id: int,
            return_success: bool = False,
        ) -> bool | None:
            # expert_id 越界时 vLLM 会传 -1（EP 路径）；我们当前不支持 EP
            if expert_id < 0 or expert_id >= num_experts:
                return False if return_success else None

            # 识别参数后缀
            if weight_name.endswith("qweight"):
                suffix = "qweight"
                packed_n = True
                grouped_k = False
            elif weight_name.endswith("qzeros"):
                suffix = "qzeros"
                packed_n = True
                grouped_k = True
            elif weight_name.endswith("scales"):
                suffix = "scales"
                packed_n = False
                grouped_k = True
            else:
                # g_idx 等无关后缀静默跳过
                return False if return_success else None

            if shard_id in ("w1", "w3"):
                # gate_proj / up_proj 按 N 维切；写入 w13 的前半 / 后半
                src = _narrow_n_for_w13(loaded_weight, is_packed_n=packed_n) \
                    if tp_size > 1 else loaded_weight
                expert_data = param.data[expert_id]      # [H(/g), 2*shard_n]
                half = expert_data.shape[1] // 2
                if shard_id == "w1":
                    expert_data.narrow(1, 0, half).copy_(src)
                else:  # "w3"
                    expert_data.narrow(1, half, half).copy_(src)
                return True if return_success else None

            if shard_id == "w2":
                src = _narrow_k_for_w2(loaded_weight, is_grouped_k=grouped_k) \
                    if tp_size > 1 else loaded_weight
                param.data[expert_id].copy_(src)
                return True if return_success else None

            # 未知 shard_id
            return False if return_success else None

        # 兼容 FusedMoE.__init__ 里对 weight_loader.supports_moe_loading 的探测
        loader.supports_moe_loading = True  # type: ignore[attr-defined]
        return loader

    # ── 权重加载完成后构造 AWQFusedMoEImpl ─────────────────────────────────
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """把 ``w13_*`` 按 N 维拆成 ``gate_*`` / ``up_*``，构造 ``AWQFusedMoEImpl``。"""
        from fused_cpp import AWQFusedMoEImpl

        num_experts = layer.num_experts
        h = layer.hidden_size
        f_dim = layer.intermediate_size_per_partition
        pack = 8

        # w13_* 的 N 维顺序：w1 (gate) 前 F//pack, w3 (up) 后 F//pack
        gate_qweight, up_qweight = torch.split(
            layer.w13_qweight.data, f_dim // pack, dim=2,
        )
        gate_qzeros, up_qzeros = torch.split(
            layer.w13_qzeros.data, f_dim // pack, dim=2,
        )
        gate_scales, up_scales = torch.split(
            layer.w13_scales.data, f_dim, dim=2,
        )

        # torch.split 得到的张量是 view（非 contiguous）。AWQFusedMoEImpl 内部
        # 会 per-expert 切片再传入 fused_cpp.w4a8_linear；保持非 contiguous
        # 通常可被正确处理（w4a8_linear 会按需 contiguous），但为了避免在
        # 每次 forward 触发潜在的 .contiguous() 复制，这里一次性物化。
        gate_qweight = gate_qweight.contiguous()
        up_qweight = up_qweight.contiguous()
        gate_qzeros = gate_qzeros.contiguous()
        up_qzeros = up_qzeros.contiguous()
        gate_scales = gate_scales.contiguous()
        up_scales = up_scales.contiguous()

        self._impl = AWQFusedMoEImpl(
            num_experts=num_experts,
            hidden_size=h,
            ffn_hidden_size=f_dim,
            gate_qweight=gate_qweight,
            gate_qzeros=gate_qzeros,
            gate_scales=gate_scales,
            up_qweight=up_qweight,
            up_qzeros=up_qzeros,
            up_scales=up_scales,
            down_qweight=layer.w2_qweight.data,
            down_qzeros=layer.w2_qzeros.data,
            down_scales=layer.w2_scales.data,
        )

    # ── apply：只转发 (x, topk_weights, topk_ids) 到 AWQFusedMoEImpl ────────
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self._impl is None:
            raise RuntimeError(
                "CPUAWQFusedMoEMethod.apply 在 process_weights_after_loading "
                "之前被调用"
            )
        # shared experts 通过独立的 AWQ Linear 路径完成，不在本类处理；
        # 因此忽略 shared_experts_input，返回单一 Tensor。
        # x 形状可能是 3D [B, S, H] 或 2D [T, H]；统一压成 2D 后还原。
        orig_shape = x.shape
        if x.dim() == 3:
            x2d = x.reshape(-1, orig_shape[-1])
        else:
            x2d = x

        # AWQFusedMoEImpl.forward 需要 bf16 输入
        in_dtype = x2d.dtype
        x_bf16 = x2d if in_dtype == torch.bfloat16 else x2d.to(torch.bfloat16)
        out_bf16 = self._impl.forward(x_bf16, topk_weights, topk_ids)
        out = out_bf16 if in_dtype == torch.bfloat16 else out_bf16.to(in_dtype)
        return out.reshape(orig_shape) if x.dim() == 3 else out

    # vLLM FusedMoE runner 询问 quant_config 时返回 None —— 我们既不是 GPU 量化
    # 路径，也不需要 FusedMoEQuantConfig 基础设施。
    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module,
    ) -> Any | None:
        return None
