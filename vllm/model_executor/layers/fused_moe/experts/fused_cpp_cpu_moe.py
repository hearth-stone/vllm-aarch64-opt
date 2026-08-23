# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Arm fused_cpp Plan V2 experts for vLLM's modular MoE pipeline."""

import os
import threading
from pathlib import Path
from typing import Any, ClassVar

import torch

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config import get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kInt8DynamicTokenSym,
    kInt8StaticChannelSym,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import CpuArchEnum, current_platform
from vllm.utils.cpu_resource_utils import parse_id_list
from vllm.v1.utils import record_function_or_nullcontext

logger = init_logger(__name__)

_BACKEND = "arm_sve_bf16"
_PROFILE_ENV = "FUSED_CPP_MOE_PLANNER_PROFILE"
_PREPACKED_DIR_ENV = "VLLM_CPU_MOE_PREPACKED_DIR"
_PREPACK_EXPORT_DIR_ENV = "VLLM_CPU_MOE_PREPACK_EXPORT_DIR"
_PREPACK_PREFAULT_ENV = "VLLM_CPU_MOE_PREPACK_PREFAULT"
_PREPACK_SCHEMA = 3
_BACKEND_ABI = "fused_cpp-482e03e-plan-v2"

_RUNTIME_LOCK = threading.Lock()
_RUNTIME: Any | None = None
_RUNTIME_SIGNATURE: tuple[Any, ...] | None = None


def _load_fused_cpp_moe() -> Any | None:
    try:
        from fused_cpp import moe
    except (ImportError, AttributeError):
        return None
    required = (
        "prepare_routed_shared_moe_bf16_tiled_weights",
        "fused_moe_bf16_tiled_with_shared",
        "MoePlannerRuntime",
        "get_default_moe_planner_runtime",
        "set_default_moe_planner_runtime",
    )
    if _BACKEND not in moe.available_fused_moe_bf16_tiled_backends():
        return None
    if any(not hasattr(moe, name) for name in required):
        return None
    return moe


def _profile_path(tp_rank: int) -> str | None:
    template = os.environ.get(_PROFILE_ENV)
    if not template:
        return None
    try:
        path = template.format(
            rank=int(os.environ.get("RANK", tp_rank)),
            local_rank=int(os.environ.get("LOCAL_RANK", tp_rank)),
            tp_rank=tp_rank,
        )
    except (KeyError, IndexError, ValueError):
        return None
    return path if os.path.isfile(path) else None


def _rank_cpu_ids(tp_rank: int) -> tuple[int, ...] | None:
    binding = envs.VLLM_CPU_OMP_THREADS_BIND
    if binding in ("auto", "nobind"):
        return None
    bindings = binding.split("|")
    if tp_rank >= len(bindings):
        return None
    cpu_ids = tuple(parse_id_list(bindings[tp_rank]))
    return cpu_ids or None


def _runtime_for(layer: torch.nn.Module, moe: Any) -> Any:
    global _RUNTIME, _RUNTIME_SIGNATURE
    tp_rank = get_tensor_model_parallel_rank()
    tp_size = get_tensor_model_parallel_world_size()
    profile = _profile_path(tp_rank)
    cpu_ids = _rank_cpu_ids(tp_rank)
    if profile is None:
        raise RuntimeError(f"{_PROFILE_ENV} must name a rank-local Plan V2 profile")
    if cpu_ids is None:
        raise RuntimeError(
            "Plan V2 requires an explicit VLLM_CPU_OMP_THREADS_BIND per rank"
        )
    if torch.get_num_threads() != len(cpu_ids):
        raise RuntimeError(
            "Plan V2 thread count does not match rank affinity: "
            f"torch={torch.get_num_threads()}, affinity={len(cpu_ids)}"
        )
    hidden = int(layer.w13_weight.shape[2])
    intermediate = int(layer.w2_weight.shape[2])
    local_experts = int(layer.w13_weight.shape[0])
    global_experts = int(layer.global_num_experts)
    signature = (
        profile,
        hidden,
        intermediate,
        local_experts,
        global_experts,
        tp_size,
        cpu_ids,
    )
    with _RUNTIME_LOCK:
        if _RUNTIME is not None:
            if signature != _RUNTIME_SIGNATURE:
                raise RuntimeError("Plan V2 runtime signature changed within one rank")
            return _RUNTIME
        _RUNTIME = moe.MoePlannerRuntime(
            profile,
            hidden_size=hidden,
            intermediate_size=intermediate,
            global_experts=global_experts,
            local_experts=local_experts,
            mode="tp" if tp_size > 1 else "standalone",
            degree=tp_size,
            concurrent_ranks=1,
            cpu_ids=cpu_ids,
            shared_experts=1,
        )
        moe.set_default_moe_planner_runtime(_RUNTIME)
        if moe.get_default_moe_planner_runtime() is not _RUNTIME:
            raise RuntimeError("failed to install fused_cpp Plan V2 runtime")
        _RUNTIME_SIGNATURE = signature
        return _RUNTIME


def _layer_id(layer: torch.nn.Module) -> int:
    from vllm.model_executor.models.utils import extract_layer_index

    return extract_layer_index(layer.layer_name)


def _cache_path(root: str, layer: torch.nn.Module) -> Path:
    return Path(root) / (
        f"layer_{_layer_id(layer):03d}_tp{get_tensor_model_parallel_world_size()}_"
        f"rank{get_tensor_model_parallel_rank()}.pt"
    )


def _cache_key(layer: torch.nn.Module, quant_mode: str = "bf16") -> dict[str, Any]:
    config = get_current_vllm_config()
    shared = layer._dsv4_shared_experts
    key = {
        "schema": _PREPACK_SCHEMA,
        "backend_abi": _BACKEND_ABI,
        "model": str(config.model_config.model),
        "tp_size": get_tensor_model_parallel_world_size(),
        "tp_rank": get_tensor_model_parallel_rank(),
        "layer_id": _layer_id(layer),
        "routed_w13_shape": tuple(layer.w13_weight.shape),
        "routed_w2_shape": tuple(layer.w2_weight.shape),
        "shared_w13_shape": tuple(shared.gate_up_proj.weight.shape),
        "shared_w2_shape": tuple(shared.down_proj.weight.shape),
        "dtype": str(layer.w13_weight.dtype),
        "quant_mode": quant_mode,
    }
    if quant_mode != "bf16":
        key.update(
            {
                "routed_w13_scale_shape": tuple(layer.w13_weight_scale.shape),
                "routed_w2_scale_shape": tuple(layer.w2_weight_scale.shape),
                "shared_w13_scale_shape": tuple(shared.gate_up_proj.weight_scale.shape),
                "shared_w2_scale_shape": tuple(shared.down_proj.weight_scale.shape),
            }
        )
    return key


def _prepared_tensors(prepared: Any) -> tuple[torch.Tensor, torch.Tensor]:
    return prepared.packed.w13[0], prepared.packed.w2[0]


def _load_prepacked(
    root: str, layer: torch.nn.Module, moe: Any, quant_mode: str = "bf16"
) -> Any:
    path = _cache_path(root, layer)
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    expected = _cache_key(layer, quant_mode)
    if payload.get("key") != expected:
        raise RuntimeError(
            f"CPU MoE prepack cache key mismatch for {path}: "
            f"expected={expected}, actual={payload.get('key')}"
        )
    w13: tuple[Any, ...]
    w2: tuple[Any, ...]
    if quant_mode == "bf16":
        packed_cls = moe.PreparedBF16TiledFusedMoEWeights
        combined_cls = moe.PreparedBF16TiledRoutedSharedMoEWeights
        w13 = (payload["w13"], int(payload["w13_k"]), int(payload["w13_n"]))
        w2 = (payload["w2"], int(payload["w2_k"]), int(payload["w2_n"]))
    elif quant_mode == "W8A16":
        packed_cls = moe.PreparedW8A16TiledFusedMoEWeights
        combined_cls = moe.PreparedW8A16TiledRoutedSharedMoEWeights
        w13 = (
            payload["w13"],
            int(payload["w13_k"]),
            int(payload["w13_n"]),
            payload["w13_scale"],
        )
        w2 = (
            payload["w2"],
            int(payload["w2_k"]),
            int(payload["w2_n"]),
            payload["w2_scale"],
        )
    elif quant_mode == "W8A8":
        packed_cls = moe.PreparedW8A8TiledFusedMoEWeights
        combined_cls = moe.PreparedW8A8TiledRoutedSharedMoEWeights
        w13 = (
            payload["w13"],
            int(payload["w13_k"]),
            int(payload["w13_n"]),
            payload["w13_scale"],
        )
        w2 = (
            payload["w2"],
            int(payload["w2_k"]),
            int(payload["w2_n"]),
            payload["w2_scale"],
        )
    else:
        raise RuntimeError(f"unsupported cached CPU MoE quant mode {quant_mode}")
    packed = packed_cls(
        w13=w13,
        w2=w2,
        fused_silu=True,
        gemm_backend=int(payload["gemm_backend"]),
        backend_n_tile=int(payload["backend_n_tile"]),
        backend_name=str(payload["backend_name"]),
    )
    combined = combined_cls(
        packed=packed,
        routed_experts=int(payload["routed_experts"]),
        shared_expert_id=int(payload["shared_expert_id"]),
    )
    if os.environ.get(_PREPACK_PREFAULT_ENV, "0") == "1":
        for tensor in _prepared_tensors(combined):
            # Touch one byte per page and the tail without allocating a copy.
            byte_view = tensor.view(torch.uint8).reshape(-1)
            if byte_view.numel():
                _ = int(byte_view[::4096].sum(dtype=torch.int64).item())
                _ = int(byte_view[-1].item())
    logger.info("Loaded DeepSeek V4 CPU MoE prepack cache: %s", path)
    return combined


def _export_prepacked(
    root: str,
    layer: torch.nn.Module,
    prepared: Any,
    quant_mode: str = "bf16",
) -> None:
    path = _cache_path(root, layer)
    path.parent.mkdir(parents=True, exist_ok=True)
    packed = prepared.packed
    payload = {
        "key": _cache_key(layer, quant_mode),
        "w13": packed.w13[0],
        "w13_k": packed.w13[1],
        "w13_n": packed.w13[2],
        "w2": packed.w2[0],
        "w2_k": packed.w2[1],
        "w2_n": packed.w2[2],
        "gemm_backend": packed.gemm_backend,
        "backend_n_tile": packed.backend_n_tile,
        "backend_name": packed.backend_name,
        "routed_experts": prepared.routed_experts,
        "shared_expert_id": prepared.shared_expert_id,
    }
    if len(packed.w13) == 4:
        payload["w13_scale"] = packed.w13[3]
        payload["w2_scale"] = packed.w2[3]
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)
    logger.info("Exported DeepSeek V4 CPU MoE prepack cache: %s", path)


class FusedCppArmExperts(mk.FusedMoEExpertsModular):
    """Checkpoint-BF16 routed+shared fused_cpp Plan V2 experts."""

    fuses_shared_experts: ClassVar[bool] = True

    def __init__(
        self, moe_config: FusedMoEConfig, quant_config: FusedMoEQuantConfig
    ) -> None:
        super().__init__(moe_config, quant_config)
        self.prepared: Any | None = None
        self.runtime: Any | None = None
        self.execute: Any | None = None
        self.num_threads = 0
        self.num_experts = moe_config.num_local_experts

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return (
            current_platform.is_cpu()
            and current_platform.get_cpu_architecture() == CpuArchEnum.ARM
            and _load_fused_cpp_moe() is not None
            and _profile_path(get_tensor_model_parallel_rank()) is not None
            and _rank_cpu_ids(get_tensor_model_parallel_rank()) is not None
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None, activation_key: QuantKey | None
    ) -> bool:
        return weight_key is None and activation_key is None

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(config: FusedMoEParallelConfig) -> bool:
        return (
            not config.use_ep
            and config.dp_size == 1
            and config.pcp_size == 1
            and not config.is_sequence_parallel
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        shared = getattr(layer, "_dsv4_shared_experts", None)
        if shared is None:
            raise RuntimeError("fused_cpp DeepSeek V4 MoE requires one shared expert")
        if layer.swiglu_limit != 10.0:
            raise RuntimeError("fused_cpp DeepSeek V4 MoE requires swiglu_limit=10.0")
        if layer.expert_map is not None:
            raise RuntimeError("fused_cpp Plan V2 does not support expert parallelism")
        for name, weight in (
            ("routed w13", layer.w13_weight),
            ("routed w2", layer.w2_weight),
            ("shared w13", shared.gate_up_proj.weight),
            ("shared w2", shared.down_proj.weight),
        ):
            if weight.device.type != "cpu" or weight.dtype != torch.bfloat16:
                raise RuntimeError(f"{name} must be CPU BF16")
        moe = _load_fused_cpp_moe()
        if moe is None:
            raise RuntimeError("fused_cpp arm_sve_bf16 API is unavailable")
        self.runtime = _runtime_for(layer, moe)
        prepacked_dir = os.environ.get(_PREPACKED_DIR_ENV)
        export_dir = os.environ.get(_PREPACK_EXPORT_DIR_ENV)
        if prepacked_dir:
            self.prepared = _load_prepacked(prepacked_dir, layer, moe)
        elif export_dir and _cache_path(export_dir, layer).is_file():
            # Resume an interrupted export without repacking or rewriting
            # already complete layer/rank files.
            self.prepared = _load_prepacked(export_dir, layer, moe)
        else:
            self.prepared = moe.prepare_routed_shared_moe_bf16_tiled_weights(
                layer.w13_weight.contiguous(),
                layer.w2_weight.contiguous(),
                shared.gate_up_proj.weight.contiguous(),
                shared.down_proj.weight.contiguous(),
                backend=_BACKEND,
            )
            if export_dir:
                _export_prepacked(export_dir, layer, self.prepared)
        self.execute = moe.fused_moe_bf16_tiled_with_shared
        self.num_threads = int(self.runtime.num_cores)
        self.num_experts = int(layer.w13_weight.shape[0])

        replace_parameter(
            layer,
            "w13_weight",
            torch.empty(
                self.num_experts,
                0,
                0,
                dtype=layer.w13_weight.dtype,
                device=layer.w13_weight.device,
            ),
        )
        replace_parameter(
            layer,
            "w2_weight",
            torch.empty(
                self.num_experts,
                0,
                0,
                dtype=layer.w2_weight.dtype,
                device=layer.w2_weight.device,
            ),
        )
        for projection in (shared.gate_up_proj, shared.down_proj):
            replace_parameter(
                projection,
                "weight",
                torch.empty(
                    0,
                    dtype=projection.weight.dtype,
                    device=projection.weight.device,
                ),
            )
        logger.info_once(
            "Using fused_cpp BF16 Plan V2 routed+shared MoE (threads=%d, TP=%d).",
            self.num_threads,
            get_tensor_model_parallel_world_size(),
        )

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        del w1, w2
        return (
            self.num_experts,
            a1.shape[0],
            2 * self.moe_config.intermediate_size_per_partition,
            self.moe_config.hidden_dim,
            topk_ids.shape[1],
        )

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: Any | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        del N, topk, global_num_experts, local_num_experts, expert_tokens_meta
        del activation
        return (0,), (0,), (M, K)

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: Any | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        del w1, w2, global_num_experts, a1q_scale, a2_scale
        del workspace13, workspace2, expert_tokens_meta
        if self.prepared is None or self.execute is None or self.runtime is None:
            raise RuntimeError("fused_cpp MoE weights were not prepared")
        if expert_map is not None or apply_router_weight_on_input:
            raise RuntimeError("fused_cpp Plan V2 requires TP-local routing")
        moe = _load_fused_cpp_moe()
        if moe is None or self.runtime is not moe.get_default_moe_planner_runtime():
            raise RuntimeError("process-default fused_cpp planner runtime was replaced")
        quant_mode = getattr(self, "quant_mode", "bf16")
        with record_function_or_nullcontext(
            f"vllm::fused_cpp_moe_{quant_mode}_plan_v2"
        ):
            result = self.execute(
                hidden_states,
                self.prepared,
                topk_weights,
                topk_ids,
                num_threads=self.num_threads,
                routed_scaling_factor=1.0,
                activation=activation.value,
                swiglu_limit=self.moe_config.swiglu_limit,
                out=output,
            )
        if result is not output:
            raise RuntimeError("fused_cpp did not reuse vLLM's output buffer")


class _FusedCppArmInt8Experts(FusedCppArmExperts):
    quant_mode: ClassVar[str]
    prepare_name: ClassVar[str]
    execute_name: ClassVar[str]

    @staticmethod
    def _supports_current_device() -> bool:
        moe = _load_fused_cpp_moe()
        return (
            current_platform.is_cpu()
            and current_platform.get_cpu_architecture() == CpuArchEnum.ARM
            and moe is not None
            and _profile_path(get_tensor_model_parallel_rank()) is not None
            and _rank_cpu_ids(get_tensor_model_parallel_rank()) is not None
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        shared = getattr(layer, "_dsv4_shared_experts", None)
        if shared is None:
            raise RuntimeError("fused_cpp DeepSeek V4 INT8 MoE requires shared expert")
        if layer.swiglu_limit != 10.0:
            raise RuntimeError("fused_cpp DeepSeek V4 INT8 MoE requires clamp=10")
        if layer.expert_map is not None:
            raise RuntimeError("fused_cpp INT8 Plan V2 does not support EP")
        projections = (shared.gate_up_proj, shared.down_proj)
        values = (
            layer.w13_weight,
            layer.w13_weight_scale,
            layer.w2_weight,
            layer.w2_weight_scale,
            projections[0].weight,
            projections[0].weight_scale,
            projections[1].weight,
            projections[1].weight_scale,
        )
        for index, value in enumerate(values):
            expected = torch.int8 if index % 2 == 0 else torch.float32
            if value.device.type != "cpu" or value.dtype != expected:
                raise RuntimeError(
                    f"fused_cpp {self.quant_mode} source {index} must be "
                    f"CPU {expected}, got {value.device}/{value.dtype}"
                )
        moe = _load_fused_cpp_moe()
        if (
            moe is None
            or not hasattr(moe, self.prepare_name)
            or not hasattr(moe, self.execute_name)
        ):
            raise RuntimeError(f"fused_cpp {self.quant_mode} API is unavailable")
        self.runtime = _runtime_for(layer, moe)
        prepacked_dir = os.environ.get(_PREPACKED_DIR_ENV)
        export_dir = os.environ.get(_PREPACK_EXPORT_DIR_ENV)
        if prepacked_dir:
            self.prepared = _load_prepacked(prepacked_dir, layer, moe, self.quant_mode)
        elif export_dir and _cache_path(export_dir, layer).is_file():
            self.prepared = _load_prepacked(export_dir, layer, moe, self.quant_mode)
        else:
            prepare = getattr(moe, self.prepare_name)
            self.prepared = prepare(*values)
            if export_dir:
                _export_prepacked(export_dir, layer, self.prepared, self.quant_mode)
        self.execute = getattr(moe, self.execute_name)
        self.num_threads = int(self.runtime.num_cores)
        self.num_experts = int(layer.w13_weight.shape[0])
        for name in (
            "w13_weight",
            "w13_weight_scale",
            "w2_weight",
            "w2_weight_scale",
        ):
            value = getattr(layer, name)
            shape = (self.num_experts, 0, 0) if "weight_scale" not in name else (0,)
            replace_parameter(
                layer,
                name,
                torch.empty(shape, dtype=value.dtype, device=value.device),
            )
        for projection in projections:
            for name in ("weight", "weight_scale"):
                value = getattr(projection, name)
                replace_parameter(
                    projection,
                    name,
                    torch.empty(0, dtype=value.dtype, device=value.device),
                )
        logger.info_once(
            "Using fused_cpp %s Plan V2 routed+shared MoE (threads=%d).",
            self.quant_mode,
            self.num_threads,
        )


class FusedCppArmW8A16Experts(_FusedCppArmInt8Experts):
    quant_mode = "W8A16"
    prepare_name = "prepare_routed_shared_moe_w8a16_tiled_quantized_weights"
    execute_name = "fused_moe_w8a16_tiled_with_shared"

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None, activation_key: QuantKey | None
    ) -> bool:
        return (weight_key, activation_key) == (kInt8StaticChannelSym, None)


class FusedCppArmW8A8Experts(_FusedCppArmInt8Experts):
    quant_mode = "W8A8"
    prepare_name = "prepare_routed_shared_moe_w8a8_tiled_quantized_weights"
    execute_name = "fused_moe_w8a8_tiled_with_shared"

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None, activation_key: QuantKey | None
    ) -> bool:
        return (weight_key, activation_key) == (
            kInt8StaticChannelSym,
            kInt8DynamicTokenSym,
        )


__all__ = [
    "FusedCppArmExperts",
    "FusedCppArmW8A16Experts",
    "FusedCppArmW8A8Experts",
]
